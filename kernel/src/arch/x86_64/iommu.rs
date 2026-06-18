// SPDX-License-Identifier: AGPL-3.0-or-later
//! AMD-Vi IOMMU bypass — ACPI IVRS-driven disable for the
//! Cherry Server's AMD EPYC 9354P.
//!
//! ## Why this exists
//!
//! On AMD platforms, firmware leaves AMD-Vi (the AMD IOMMU) enabled
//! by default after POST. With translation enabled but no driver to
//! install device-table / page-table mappings, any device-initiated
//! DMA is silently mis-translated — the IOMMU's "default" device-table
//! entry blocks the access. This shows up as a host driver bring-up
//! failure with no obvious culprit (no fault interrupts, no console
//! noise; just timeouts).
//!
//! The X710 (i40e) on the Cherry Server triggers exactly this failure:
//! HMC SD programming asks the device to read the host-resident Page
//! Descriptor Table via bus-master DMA, the read is blocked by the
//! IOMMU, PMSDWR never clears, and the data path never comes up.
//!
//! Until we ship a real AMD-Vi driver (device-table + identity-paging
//! + page-fault handling), the pragmatic fix is to disable IOMMU
//! translation altogether so DMA goes direct to physical RAM. This is
//! safe on a bring-up kernel: there are no guests, no untrusted code,
//! and one device at a time touches DMA.
//!
//! ## What we do
//!
//! 1. Walk RSDP → XSDT/RSDT looking for the IVRS table (sig "IVRS").
//! 2. Walk IVRS entries; for each IVHD (type 0x10 / 0x11 / 0x40) read
//!    the 64-bit "IOMMU Base Address" field at offset 8 of the IVHD.
//! 3. Map a single 4 KiB page of that base via `memory::map_mmio` and
//!    clear bit 0 (IommuEn) of the Control Register at MMIO+0x18.
//!
//! Multi-IOMS EPYC systems (the 9354P is single-socket but multi-IOMS)
//! expose one IOMMU per IOMS, so we disable every IVHD we find.
//!
//! ## What we do NOT do
//!
//! * Parse device-entry sub-records inside the IVHD. We don't care
//!   which BDFs each IOMMU covers — we disable them all.
//! * Verify the disable took effect via the IOMMU Status Register.
//!   Many platforms reflect the IommuEn write immediately; a few use
//!   a completion handshake we'd need to implement separately. For
//!   bring-up purposes the read-back of the Control Register after
//!   the write is sufficient diagnostic.
//! * Touch Intel VT-d (DMAR). Intel platforms don't enable VT-d by
//!   default in firmware on the SKUs we target, so this module is
//!   AMD-only.
//!
//! ## References
//!
//! * AMD I/O Virtualization Technology (IOMMU) Specification, Rev 3.07,
//!   §3.1 (MMIO register layout), §3.1.2 (IOMMU Control Register),
//!   §5.2 (IVRS / IVHD ACPI structures).
//! * ACPI 6.4 §5.2.5 (RSDP), §5.2.7 (RSDT), §5.2.8 (XSDT).

#![allow(dead_code)]

use core::fmt::Write;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{compiler_fence, Ordering};

use crate::arch::serial::Serial;
use crate::memory;

/// Maximum number of IVHD entries we'll disable. Real platforms ship
/// at most one IVHD per IOMS — 8 is comfortable headroom for current
/// AMD parts.
const MAX_IOMMUS: usize = 8;

/// Offset of the IOMMU Control Register inside the IOMMU MMIO region
/// (AMD I/O Virtualization Technology Specification §3.1).
const IOMMU_CTRL_REG_OFFSET: usize = 0x18;

/// Bit 0 of the IOMMU Control Register — IommuEn. 1 = translation
/// enabled, 0 = IOMMU bypasses (all DMA goes direct).
const IOMMU_CTRL_IOMMU_EN: u64 = 1 << 0;

/// Errors that prevent IOMMU disable. These are all soft errors —
/// callers should log and continue (the i40e bring-up may still work
/// on a platform that never had AMD-Vi enabled in the first place).
#[derive(Copy, Clone, Debug)]
pub enum IommuError {
    /// No RSDP found in EBDA / BIOS area.
    RsdpNotFound,
    /// XSDT/RSDT checksum bad.
    SdtChecksumBad,
    /// No IVRS table — the platform is not an AMD-Vi-aware system.
    IvrsNotFound,
    /// IVRS table malformed (length truncates mid-entry).
    IvrsTruncated,
    /// `memory::init` hasn't run, so we can't `map_mmio` the IOMMU.
    PhysOffsetMissing,
    /// `memory::map_mmio` failed.
    MmioMapFailed,
}

/// Summary of one disabled IOMMU.
#[derive(Copy, Clone, Debug, Default)]
pub struct IommuDisabled {
    /// Physical base of the IOMMU MMIO region (from IVHD).
    pub base_phys: u64,
    /// Control Register value before our write.
    pub ctrl_before: u64,
    /// Control Register value after our write.
    pub ctrl_after: u64,
}

/// Result of [`disable_amd_vi`].
#[derive(Copy, Clone, Debug)]
pub struct DisableReport {
    /// Number of IOMMUs touched.
    pub count: usize,
    /// Per-IOMMU details. Entries beyond `count` are zero.
    pub iommus: [IommuDisabled; MAX_IOMMUS],
}

impl DisableReport {
    const fn empty() -> Self {
        Self {
            count: 0,
            iommus: [IommuDisabled {
                base_phys: 0,
                ctrl_before: 0,
                ctrl_after: 0,
            }; MAX_IOMMUS],
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// ACPI RSDP / SDT minimal walker
// ─────────────────────────────────────────────────────────────────
//
// Duplicated (intentionally) from `acpi.rs`. Keeping this module
// self-contained means the IOMMU disable path has no dependency on
// MADT parsing and can move (one day) to a shared `acpi_tables`
// module without renaming.

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct Rsdp {
    signature: [u8; 8],
    checksum: u8,
    oem_id: [u8; 6],
    revision: u8,
    rsdt_address: u32,
    length: u32,
    xsdt_address: u64,
    extended_checksum: u8,
    _reserved: [u8; 3],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct SdtHeader {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem_id: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    creator_id: u32,
    creator_revision: u32,
}

unsafe fn checksum_ok(addr: u64, len: usize) -> bool {
    let mut sum: u8 = 0;
    for i in 0..len {
        let b = *((addr + i as u64) as *const u8);
        sum = sum.wrapping_add(b);
    }
    sum == 0
}

unsafe fn scan_rsdp_range(phys_offset: u64, base: u64, len: usize) -> Option<Rsdp> {
    const SIGNATURE: [u8; 8] = *b"RSD PTR ";
    let mut off = 0usize;
    while off + core::mem::size_of::<Rsdp>() <= len {
        let p = phys_offset + base + off as u64;
        let sig: [u8; 8] = core::ptr::read_unaligned(p as *const [u8; 8]);
        if sig == SIGNATURE {
            let rsdp: Rsdp = core::ptr::read_unaligned(p as *const Rsdp);
            if checksum_ok(p, 20) {
                if rsdp.revision >= 2 && rsdp.length as usize >= core::mem::size_of::<Rsdp>() {
                    if !checksum_ok(p, rsdp.length as usize) {
                        off += 16;
                        continue;
                    }
                }
                return Some(rsdp);
            }
        }
        off += 16;
    }
    None
}

unsafe fn find_rsdp(phys_offset: u64) -> Option<Rsdp> {
    let ebda_seg = core::ptr::read_unaligned((phys_offset + 0x40E) as *const u16);
    let ebda_base = (ebda_seg as u64) << 4;
    if ebda_base != 0 && ebda_base < 0x10_0000 {
        if let Some(r) = scan_rsdp_range(phys_offset, ebda_base, 1024) {
            return Some(r);
        }
    }
    scan_rsdp_range(phys_offset, 0xE0000, 0x20000)
}

unsafe fn read_sdt_header(phys_offset: u64, phys: u64) -> Option<SdtHeader> {
    let p = phys_offset + phys;
    let hdr: SdtHeader = core::ptr::read_unaligned(p as *const SdtHeader);
    if !checksum_ok(p, hdr.length as usize) {
        return None;
    }
    Some(hdr)
}

/// Scan XSDT/RSDT for the IVRS table. Returns the IVRS physical base,
/// or `None` if no IVRS is present.
unsafe fn find_ivrs(phys_offset: u64) -> Result<u64, IommuError> {
    let rsdp = find_rsdp(phys_offset).ok_or(IommuError::RsdpNotFound)?;
    let rev = rsdp.revision;
    let xsdt_or_rsdt_phys: u64 = if rev >= 2 && rsdp.xsdt_address != 0 {
        rsdp.xsdt_address
    } else {
        rsdp.rsdt_address as u64
    };
    let sdt = read_sdt_header(phys_offset, xsdt_or_rsdt_phys).ok_or(IommuError::SdtChecksumBad)?;
    let entry_size = if rev >= 2 && rsdp.xsdt_address != 0 {
        8
    } else {
        4
    };
    let entries_off = core::mem::size_of::<SdtHeader>();
    let entries_bytes = sdt.length as usize - entries_off;
    let n_entries = entries_bytes / entry_size;

    for i in 0..n_entries {
        let entry_phys_ptr =
            phys_offset + xsdt_or_rsdt_phys + entries_off as u64 + (i * entry_size) as u64;
        let candidate: u64 = if entry_size == 8 {
            core::ptr::read_unaligned(entry_phys_ptr as *const u64)
        } else {
            core::ptr::read_unaligned(entry_phys_ptr as *const u32) as u64
        };
        if let Some(h) = read_sdt_header(phys_offset, candidate) {
            if &h.signature == b"IVRS" {
                return Ok(candidate);
            }
        }
    }
    Err(IommuError::IvrsNotFound)
}

// ─────────────────────────────────────────────────────────────────
// IVRS / IVHD parsing
// ─────────────────────────────────────────────────────────────────
//
// IVRS layout per AMD IOMMU Spec §5.2.1:
//   bytes  0..36   — standard SDT header
//   bytes 36..40   — IVinfo (32-bit)
//   bytes 40..48   — reserved
//   bytes 48..     — variable-length entries
//
// Each IVHD entry begins with:
//   offset 0   1 byte   type        (0x10 / 0x11 / 0x40)
//   offset 1   1 byte   flags
//   offset 2   2 bytes  length      (total entry length)
//   offset 4   2 bytes  device_id   (IOMMU's BDF)
//   offset 6   2 bytes  capability_offset
//   offset 8   8 bytes  iommu_base_address   ★
//   ... (type-specific fields follow) ...
//
// Bytes [0..16) of every IVHD type carry the same prefix, which is
// the only part we read.

const IVRS_FIXED_HEADER_LEN: usize = 48;

const IVHD_TYPE_10: u8 = 0x10;
const IVHD_TYPE_11: u8 = 0x11;
const IVHD_TYPE_40: u8 = 0x40;

/// Walk the IVRS at `ivrs_phys` and call `f(iommu_base_phys)` for
/// each IVHD entry. Returns the number of IVHDs found.
unsafe fn walk_ivhds(
    phys_offset: u64,
    ivrs_phys: u64,
    mut f: impl FnMut(u64),
) -> Result<usize, IommuError> {
    let base = phys_offset + ivrs_phys;
    let hdr: SdtHeader = core::ptr::read_unaligned(base as *const SdtHeader);
    let total_len = hdr.length as usize;
    let mut off = IVRS_FIXED_HEADER_LEN;
    let mut count = 0usize;

    while off + 4 <= total_len {
        let entry_ptr = base + off as u64;
        let entry_type: u8 = core::ptr::read_unaligned(entry_ptr as *const u8);
        let entry_len: u16 = core::ptr::read_unaligned((entry_ptr + 2) as *const u16);
        if entry_len == 0 || off + entry_len as usize > total_len {
            return Err(IommuError::IvrsTruncated);
        }

        if matches!(entry_type, IVHD_TYPE_10 | IVHD_TYPE_11 | IVHD_TYPE_40) {
            // IOMMU Base Address at offset 8, 64 bits little-endian.
            let iommu_base: u64 = core::ptr::read_unaligned((entry_ptr + 8) as *const u64);
            if iommu_base != 0 {
                f(iommu_base);
                count += 1;
            }
        }
        // Non-IVHD entries (IVMD type 0x20/0x21/0x22) are skipped by
        // length — they describe DMA-exclusion regions which we don't
        // care about when the IOMMU is being disabled.

        off += entry_len as usize;
    }

    Ok(count)
}

// ─────────────────────────────────────────────────────────────────
// MMIO disable
// ─────────────────────────────────────────────────────────────────

/// Map the IOMMU's MMIO base and clear bit 0 (IommuEn) of the Control
/// Register. Returns the (before, after) Control Register values.
unsafe fn disable_one(iommu_base_phys: u64) -> Result<(u64, u64), IommuError> {
    // The Control Register sits at +0x18; a single 4 KiB page covers
    // all the registers we touch (the full MMIO region is 16 KiB but
    // only the first page matters for this operation).
    let mmio = memory::map_mmio(iommu_base_phys, 0x1000).map_err(|_| IommuError::MmioMapFailed)?;

    let ctrl_ptr = mmio.add(IOMMU_CTRL_REG_OFFSET) as *mut u64;
    let before = read_volatile(ctrl_ptr);
    let after_val = before & !IOMMU_CTRL_IOMMU_EN;
    write_volatile(ctrl_ptr, after_val);
    compiler_fence(Ordering::Release);
    // Read back to flush the posted write and observe the new value.
    let after = read_volatile(ctrl_ptr);
    Ok((before, after))
}

/// Disable AMD-Vi translation on every IOMMU advertised by the
/// firmware's IVRS table. Idempotent — if IOMMUs are already disabled
/// the read-modify-write is a no-op.
///
/// Failure is non-fatal: log + continue. Platforms without AMD-Vi
/// (Intel, BMC-only VMs, etc.) return `IvrsNotFound` quickly.
///
/// # Safety
///
/// Caller must ensure `memory::init` has run (so `map_mmio` works)
/// and that no device driver has already started DMA. Currently
/// invoked from `kernel_main` between PCI enumeration and any
/// driver bind.
pub unsafe fn disable_amd_vi(phys_offset: u64) -> Result<DisableReport, IommuError> {
    let ivrs_phys = find_ivrs(phys_offset)?;
    let _ = writeln!(
        Serial,
        "iommu: IVRS table found at phys 0x{:016x}",
        ivrs_phys
    );

    let mut report = DisableReport::empty();
    walk_ivhds(phys_offset, ivrs_phys, |iommu_base| {
        if report.count >= MAX_IOMMUS {
            let _ = writeln!(
                Serial,
                "iommu: IVHD limit reached ({}) — skipping IOMMU @0x{:016x}",
                MAX_IOMMUS, iommu_base
            );
            return;
        }
        match disable_one(iommu_base) {
            Ok((before, after)) => {
                let entry = &mut report.iommus[report.count];
                entry.base_phys = iommu_base;
                entry.ctrl_before = before;
                entry.ctrl_after = after;
                report.count += 1;
                let was_on = (before & IOMMU_CTRL_IOMMU_EN) != 0;
                let now_off = (after & IOMMU_CTRL_IOMMU_EN) == 0;
                let _ = writeln!(
                    Serial,
                    "iommu: AMD-Vi @0x{:016x} — IommuEn was {}, now {} (CTRL 0x{:016x} → 0x{:016x})",
                    iommu_base,
                    if was_on { "ON" } else { "off" },
                    if now_off { "off" } else { "ON (write rejected)" },
                    before, after
                );
            }
            Err(e) => {
                let _ = writeln!(
                    Serial,
                    "iommu: disable failed for IOMMU @0x{:016x} ({:?})",
                    iommu_base, e
                );
            }
        }
    })?;

    if report.count == 0 {
        let _ = writeln!(
            Serial,
            "iommu: IVRS present but no IVHD entries — nothing to disable"
        );
    }
    Ok(report)
}
