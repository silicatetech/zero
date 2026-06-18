// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal PCI / PCIe enumeration over the legacy 0xCF8/0xCFC I/O
//! ports.
//!
//! Scope: the kernel is currently pre-MCFG — memory-mapped
//! configuration space (PCI Express Enhanced Configuration Mechanism)
//! is unavailable. We use the legacy mechanism #1, which gives us
//! 256 bytes per function (sufficient for vendor/device
//! identification, class codes, and BAR enumeration).
//!
//! The scanner enumerates **all 256 PCI bus numbers** (0..=255). This
//! is the simple brute-force approach to multi-root-complex
//! topologies common on AMD EPYC (multiple IOMS / IOMMU groups each
//! hosting its own root complex) and on systems with nested PCIe
//! switches. Bridge-based recursive enumeration is more elegant but
//! the brute-force scan is O(256 × 32 × 8) = 65 536 probes, each a
//! single outl + inl pair (~µs), so the whole scan completes well
//! under 100 ms even on real hardware. Empty buses fall through
//! cheaply because every unpopulated function returns vendor=0xFFFF
//! in a single dword read.
//!
//! Historical note: the original scanner only touched bus 0. On the
//! Cherry Server (AMD EPYC 9354P), the Intel 82545EM NIC sits on a
//! non-zero bus (downstream of one of the IOMS root complexes) and
//! was invisible to bus-0-only enumeration. The all-bus scan fixes
//! that without requiring MCFG.
//!
//! What this module produces:
//!   * a list of (bus, dev, fn) → (vendor, device, class, subclass)
//!   * a vendor filter that flags NVIDIA devices (vendor 0x10DE),
//!     used by Stage 12 / HAL for "is there a GPU at all" gating
//!   * a serial-console print of the scan result
//!
//! What this module does NOT do (deferred):
//!   * MMIO config space (256–4096 byte range)
//!   * PCI-to-PCI bridge recursion (we brute-force all buses instead)
//!   * BAR sizing / mapping
//!   * MSI/MSI-X capability parsing
//!   * Driver attachment
//!
//! V3 Pillar 1 (Performance): single scan at boot, results cached in
//! a fixed-size array — no allocator dependency.

use core::fmt::Write;

use crate::arch::serial::Serial;

const CONFIG_ADDRESS: u16 = 0x0CF8;
const CONFIG_DATA: u16 = 0x0CFC;

/// NVIDIA's PCI SIG vendor ID. Used for GPU detection.
pub const VENDOR_NVIDIA: u16 = 0x10DE;

/// Vendor ID returned by the host when a (bus, dev, fn) is absent.
const VENDOR_INVALID: u16 = 0xFFFF;

/// PCI Class Code 0x03 = Display Controller (subclass 0x02 = 3D).
const CLASS_DISPLAY: u8 = 0x03;

/// Largest number of devices we record. Real datacenter boxes —
/// especially multi-IOMS AMD EPYC platforms — expose well over 64
/// PCI functions once every root port, IOMMU, NTB, NIC, NVMe slot,
/// and chipset bridge is counted. 256 is comfortable headroom for
/// up to dual-socket EPYC at ~24 bytes per slot in BSS.
const MAX_DEVICES: usize = 256;

/// One captured PCI function. Pack tightly — this lives in BSS.
#[derive(Copy, Clone, Debug)]
pub struct PciDevice {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: u8,
    pub subclass: u8,
    /// Programming interface byte (PCI config offset 0x09).
    /// Recorded for future driver dispatch; not surfaced in the
    /// initial bus-0 report.
    #[allow(dead_code)]
    pub prog_if: u8,
    pub header_type: u8,
}

impl PciDevice {
    pub fn is_nvidia(&self) -> bool {
        self.vendor_id == VENDOR_NVIDIA
    }

    pub fn is_display_controller(&self) -> bool {
        self.class_code == CLASS_DISPLAY
    }
}

/// Result of a PCI scan.
pub struct PciScan {
    devices: [Option<PciDevice>; MAX_DEVICES],
    count: usize,
    /// Highest bus number on which at least one function responded.
    /// Useful for diagnostics: lets the bring-up log show that the
    /// scan actually reached beyond bus 0 on multi-root-complex
    /// platforms (e.g. AMD EPYC).
    max_bus_seen: u16,
}

impl PciScan {
    pub const fn empty() -> Self {
        Self {
            devices: [None; MAX_DEVICES],
            count: 0,
            max_bus_seen: 0,
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn iter(&self) -> impl Iterator<Item = &PciDevice> {
        self.devices.iter().filter_map(|d| d.as_ref())
    }

    pub fn nvidia_count(&self) -> usize {
        self.iter().filter(|d| d.is_nvidia()).count()
    }

    /// Highest bus number on which a device responded. `0` if only
    /// bus 0 had devices (or the scan was empty).
    pub fn max_bus_seen(&self) -> u16 {
        self.max_bus_seen
    }
}

/// Read a 32-bit register from configuration space at
/// (bus, device, function, offset). Offset must be aligned to 4
/// bytes; the low two bits are ignored by the hardware.
pub unsafe fn config_read32(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
    let address = (1u32 << 31)
        | ((bus as u32) << 16)
        | (((device as u32) & 0x1F) << 11)
        | (((function as u32) & 0x07) << 8)
        | ((offset as u32) & 0xFC);
    outl(CONFIG_ADDRESS, address);
    inl(CONFIG_DATA)
}

/// Write a 32-bit register in configuration space.
pub unsafe fn config_write32(bus: u8, device: u8, function: u8, offset: u8, value: u32) {
    let address = (1u32 << 31)
        | ((bus as u32) << 16)
        | (((device as u32) & 0x1F) << 11)
        | (((function as u32) & 0x07) << 8)
        | ((offset as u32) & 0xFC);
    outl(CONFIG_ADDRESS, address);
    outl(CONFIG_DATA, value);
}

/// Read a 16-bit field from configuration space (e.g. command/status).
pub unsafe fn config_read16(bus: u8, device: u8, function: u8, offset: u8) -> u16 {
    let dw = config_read32(bus, device, function, offset & 0xFC);
    let shift = (offset & 0x02) * 8;
    ((dw >> shift) & 0xFFFF) as u16
}

/// Write a 16-bit field into configuration space without disturbing
/// the adjacent half of the dword (read-modify-write).
pub unsafe fn config_write16(bus: u8, device: u8, function: u8, offset: u8, value: u16) {
    let aligned = offset & 0xFC;
    let shift = (offset & 0x02) * 8;
    let mut dw = config_read32(bus, device, function, aligned);
    dw &= !(0xFFFFu32 << shift);
    dw |= (value as u32) << shift;
    config_write32(bus, device, function, aligned, dw);
}

/// Read a memory BAR (one of BAR0..BAR5) and return the physical base
/// address as a 64-bit value.
///
/// Handles both 32-bit and 64-bit memory BAR types (PCI Local Bus
/// spec 3.0 §6.2.5.1). A 64-bit BAR consumes two consecutive BAR
/// slots: the low slot encodes type+address[31:4], the high slot
/// holds address[63:32]. Required for any modern NIC that places
/// its MMIO above 4 GiB — e.g. the Intel X710 (i40e) on platforms
/// like the Cherry Server, whose BAR0 lives at 0x300_8180_0000 and
/// would silently truncate to zero under a 32-bit-only reader.
///
/// Returns `None` if the BAR is unpopulated, points to I/O space,
/// has a reserved type encoding, or the BAR index is out of range
/// (including the case where a 64-bit BAR is requested at slot 5,
/// which has no upper half).
pub fn read_bar(dev: &PciDevice, bar: u8) -> Option<u64> {
    if bar > 5 {
        return None;
    }
    let offset = 0x10 + bar * 4;
    let raw = unsafe { config_read32(dev.bus, dev.device, dev.function, offset) };
    if raw == 0 || raw & 0x1 != 0 {
        return None;
    }
    // bits[2:1] encode the memory BAR type:
    //   0b00 — 32-bit anywhere in the 32-bit address space
    //   0b10 — 64-bit, address spans this BAR slot and the next
    // (0b01 is the obsolete "below 1 MB" form; treat as unsupported.)
    let bar_type = (raw >> 1) & 0x3;
    // bit 3 = prefetchable; not relevant for address decoding but
    // worth being aware of — the X710's BAR0 is prefetchable.
    let low = (raw & 0xFFFF_FFF0) as u64;
    // Per PCI Local Bus Spec 3.0 §6.2.5.1, an unimplemented BAR
    // returns all zeros in its address portion. A zeroed address —
    // even with the memory-type bits non-zero, as can happen on
    // partially-decoded multifunction headers — must not be treated
    // as a valid MMIO base: mapping it would collide with the AP
    // trampoline at phys 0x8000 and other reserved low memory.
    let addr = match bar_type {
        0b00 => low,
        0b10 => {
            if bar >= 5 {
                // 64-bit BAR cannot start at slot 5 — no neighbour
                // slot to hold the upper 32 bits.
                return None;
            }
            let raw_hi = unsafe { config_read32(dev.bus, dev.device, dev.function, offset + 4) };
            low | ((raw_hi as u64) << 32)
        }
        _ => return None,
    };
    if addr == 0 {
        use core::fmt::Write;
        let _ = writeln!(
            Serial,
            "PCIe: BAR{} at {:02x}:{:02x}.{} is zero, skipping",
            bar, dev.bus, dev.device, dev.function,
        );
        return None;
    }
    Some(addr)
}

/// Find the first device matching the given vendor ID in the bus-0
/// scan. Used by drivers to bind without re-running the scan.
#[allow(dead_code)]
pub fn find_by_vendor(scan: &PciScan, vendor_id: u16) -> Option<PciDevice> {
    scan.iter().copied().find(|d| d.vendor_id == vendor_id)
}

#[inline(always)]
unsafe fn outl(port: u16, val: u32) {
    core::arch::asm!("out dx, eax", in("dx") port, in("eax") val, options(nomem, nostack, preserves_flags));
}

#[inline(always)]
unsafe fn inl(port: u16) -> u32 {
    let val: u32;
    core::arch::asm!("in eax, dx", in("dx") port, out("eax") val, options(nomem, nostack, preserves_flags));
    val
}

/// Read one function's identification header. Returns `None` if the
/// slot is unpopulated (vendor=0xFFFF).
unsafe fn probe_function(bus: u8, device: u8, function: u8) -> Option<PciDevice> {
    let id = config_read32(bus, device, function, 0x00);
    let vendor_id = (id & 0xFFFF) as u16;
    if vendor_id == VENDOR_INVALID {
        return None;
    }
    let device_id = ((id >> 16) & 0xFFFF) as u16;

    // Offset 0x08: revision (8) | prog_if (8) | subclass (8) | class (8)
    let class_reg = config_read32(bus, device, function, 0x08);
    let prog_if = ((class_reg >> 8) & 0xFF) as u8;
    let subclass = ((class_reg >> 16) & 0xFF) as u8;
    let class_code = ((class_reg >> 24) & 0xFF) as u8;

    // Offset 0x0C: BIST (8) | header_type (8) | latency (8) | cache_line (8)
    let header_reg = config_read32(bus, device, function, 0x0C);
    let header_type = ((header_reg >> 16) & 0xFF) as u8;

    Some(PciDevice {
        bus,
        device,
        function,
        vendor_id,
        device_id,
        class_code,
        subclass,
        prog_if,
        header_type,
    })
}

/// Enumerate a single bus. For each device that responds at
/// function 0, also probe functions 1..7 if the multi-function bit
/// (0x80 in header_type) is set. Each found device is appended to
/// `scan` (bounded by [`MAX_DEVICES`]).
fn scan_single_bus(scan: &mut PciScan, bus: u8) {
    for device in 0u8..32 {
        let primary = unsafe { probe_function(bus, device, 0) };
        let Some(primary) = primary else {
            continue;
        };
        push(scan, primary);

        // Multi-function device: header_type bit 7 set.
        if primary.header_type & 0x80 != 0 {
            for function in 1u8..8 {
                if let Some(d) = unsafe { probe_function(bus, device, function) } {
                    push(scan, d);
                }
            }
        }
    }
}

/// Enumerate bus 0 only — legacy entry point preserved for callers
/// that explicitly want bus-0 scope (e.g. early bring-up smoke tests).
/// Prefer [`scan_all_buses`] for production code: AMD EPYC platforms
/// place root-complex devices on non-zero buses.
#[allow(dead_code)]
pub fn scan_bus0() -> PciScan {
    let mut scan = PciScan::empty();
    scan_single_bus(&mut scan, 0);
    scan
}

/// Enumerate **all 256 PCI buses**.
///
/// Brute-force iteration over bus 0..=255. Unpopulated buses cost
/// one probe each (vendor=0xFFFF early-exits), so the total wall
/// clock on real EPYC silicon is in the low tens of milliseconds.
/// This is the right default for any platform that uses more than
/// one PCI root complex — notably AMD EPYC (multiple IOMS each with
/// its own bus number range) and dual-socket Intel Xeon Scalable.
pub fn scan_all_buses() -> PciScan {
    let mut scan = PciScan::empty();
    for bus in 0u16..=255 {
        let bus_u8 = bus as u8;
        let before = scan.count;
        scan_single_bus(&mut scan, bus_u8);
        if scan.count > before {
            scan.max_bus_seen = bus;
        }
        if scan.count >= MAX_DEVICES {
            // BSS table full — stop scanning. Diagnostic-only paths
            // (NVIDIA / e1000 detection) only need the first hits.
            break;
        }
    }
    scan
}

fn push(scan: &mut PciScan, dev: PciDevice) {
    if scan.count < MAX_DEVICES {
        scan.devices[scan.count] = Some(dev);
        scan.count += 1;
    }
}

/// Print a one-line summary plus per-device rows over the serial
/// console. Designed to be called once from `kernel_main` after
/// [`scan_all_buses`].
pub fn report(scan: &PciScan) {
    let _ = writeln!(
        Serial,
        "PCIe: all-bus scan complete, {} function(s) responded (highest populated bus = 0x{:02x})",
        scan.count(),
        scan.max_bus_seen()
    );
    let nvidia = scan.nvidia_count();
    if nvidia > 0 {
        let _ = writeln!(
            Serial,
            "PCIe: NVIDIA devices found = {} (vendor 0x10DE)",
            nvidia
        );
    } else {
        let _ = writeln!(
            Serial,
            "PCIe: no NVIDIA device found — Stage 12 HAL stays mock-only"
        );
    }

    for dev in scan.iter() {
        let kind = if dev.is_nvidia() && dev.is_display_controller() {
            "NVIDIA-GPU"
        } else if dev.is_nvidia() {
            "NVIDIA-OTHER"
        } else if dev.is_display_controller() {
            "DISPLAY"
        } else {
            "device"
        };
        let _ = writeln!(
            Serial,
            "  {:02x}:{:02x}.{}  vendor=0x{:04x} device=0x{:04x} class=0x{:02x} subclass=0x{:02x} ({})",
            dev.bus, dev.device, dev.function,
            dev.vendor_id, dev.device_id,
            dev.class_code, dev.subclass,
            kind
        );
    }
}
