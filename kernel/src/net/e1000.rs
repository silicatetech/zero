// SPDX-License-Identifier: AGPL-3.0-or-later
//! Intel 8254x (e1000 / 82540EM) ethernet driver.
//!
//! Polling-only, IRQ-free. Sufficient for QEMU's `-device e1000`
//! emulation (vendor 0x8086, device 0x100E) and real-hardware
//! 82540/82541/82545 family parts. Targets:
//!   * RX path: 16 descriptors × 2 KiB buffers, BAM (broadcast accept).
//!     UPE/MPE not enabled — we don't need promiscuous mode for the
//!     shell server.
//!   * TX path: 16 descriptors × 2 KiB buffers, full-duplex preset.
//!
//! Bare-metal note: descriptor rings and per-descriptor buffers live
//! in kernel BSS (`#[repr(C, align(...))]` statics). Hardware needs
//! physical addresses, which we obtain by walking the page tables via
//! `memory::virt_to_phys`.

use core::fmt::Write;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{compiler_fence, Ordering};

use crate::arch::serial::Serial;
use crate::arch::x86_64::pcie::{self, PciDevice, PciScan};
use crate::memory;

/// Intel's PCI SIG vendor ID.
pub const VENDOR_INTEL: u16 = 0x8086;

/// QEMU's default e1000 device ID (82540EM).
pub const DEVICE_82540EM: u16 = 0x100E;
/// 82540EM-LOM variant occasionally seen on real boards.
pub const DEVICE_82540EM_LOM: u16 = 0x1015;
/// 82545EM (server-class card).
pub const DEVICE_82545EM: u16 = 0x100F;
/// 82574L.
pub const DEVICE_82574L: u16 = 0x10D3;

const SUPPORTED_DEVICES: &[u16] = &[
    DEVICE_82540EM,
    DEVICE_82540EM_LOM,
    DEVICE_82545EM,
    DEVICE_82574L,
];

// ── Register offsets ────────────────────────────────────────────────
const REG_CTRL: usize = 0x0000;
const REG_STATUS: usize = 0x0008;
const REG_ICR: usize = 0x00C0;
const REG_IMC: usize = 0x00D8;
const REG_RCTL: usize = 0x0100;
const REG_TCTL: usize = 0x0400;
const REG_TIPG: usize = 0x0410;
const REG_RDBAL: usize = 0x2800;
const REG_RDBAH: usize = 0x2804;
const REG_RDLEN: usize = 0x2808;
const REG_RDH: usize = 0x2810;
const REG_RDT: usize = 0x2818;
const REG_TDBAL: usize = 0x3800;
const REG_TDBAH: usize = 0x3804;
const REG_TDLEN: usize = 0x3808;
const REG_TDH: usize = 0x3810;
const REG_TDT: usize = 0x3818;
const REG_MTA_BASE: usize = 0x5200;
const REG_RAL0: usize = 0x5400;
const REG_RAH0: usize = 0x5404;

// ── CTRL bits ──
const CTRL_SLU: u32 = 1 << 6;
const CTRL_RST: u32 = 1 << 26;

// ── RCTL bits ──
const RCTL_EN: u32 = 1 << 1;
const RCTL_BAM: u32 = 1 << 15;
const RCTL_BSIZE_2048: u32 = 0 << 16;
const RCTL_SECRC: u32 = 1 << 26;

// ── TCTL bits ──
const TCTL_EN: u32 = 1 << 1;
const TCTL_PSP: u32 = 1 << 3;
const TCTL_CT_SHIFT: u32 = 4;
const TCTL_COLD_SHIFT: u32 = 12;

// ── TX cmd bits ──
const TX_CMD_EOP: u8 = 1 << 0;
const TX_CMD_IFCS: u8 = 1 << 1;
const TX_CMD_RS: u8 = 1 << 3;
// ── TX status bits ──
const TX_STA_DD: u8 = 1 << 0;
// ── RX status bits ──
const RX_STA_DD: u8 = 1 << 0;
const RX_STA_EOP: u8 = 1 << 1;

// ── Ring sizing ──
pub const RX_DESC_COUNT: usize = 16;
pub const TX_DESC_COUNT: usize = 16;
pub const BUFFER_SIZE: usize = 2048;

/// RX descriptor (legacy layout, 16 bytes).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct RxDesc {
    addr: u64,
    length: u16,
    checksum: u16,
    status: u8,
    errors: u8,
    special: u16,
}

/// TX descriptor (legacy layout, 16 bytes).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct TxDesc {
    addr: u64,
    length: u16,
    cso: u8,
    cmd: u8,
    status: u8,
    css: u8,
    special: u16,
}

/// Backing storage for descriptor rings and frame buffers. Aligned at
/// 16 bytes for the descriptors (hardware spec) and 4 KiB-natural for
/// the buffers (lives in BSS, page-aligned within the kernel image).
#[repr(C, align(4096))]
struct DriverStorage {
    rx_ring: [RxDesc; RX_DESC_COUNT],
    tx_ring: [TxDesc; TX_DESC_COUNT],
    rx_buffers: [[u8; BUFFER_SIZE]; RX_DESC_COUNT],
    tx_buffers: [[u8; BUFFER_SIZE]; TX_DESC_COUNT],
}

const EMPTY_RX: RxDesc = RxDesc {
    addr: 0,
    length: 0,
    checksum: 0,
    status: 0,
    errors: 0,
    special: 0,
};
const EMPTY_TX: TxDesc = TxDesc {
    addr: 0,
    length: 0,
    cso: 0,
    cmd: 0,
    status: 0,
    css: 0,
    special: 0,
};

static mut STORAGE: DriverStorage = DriverStorage {
    rx_ring: [EMPTY_RX; RX_DESC_COUNT],
    tx_ring: [EMPTY_TX; TX_DESC_COUNT],
    rx_buffers: [[0u8; BUFFER_SIZE]; RX_DESC_COUNT],
    tx_buffers: [[0u8; BUFFER_SIZE]; TX_DESC_COUNT],
};

/// Bound e1000 instance — holds the MMIO base and ring cursors.
pub struct E1000 {
    mmio: *mut u8,
    mac: [u8; 6],
    rx_cursor: usize,
    tx_cursor: usize,
}

unsafe impl Send for E1000 {}
unsafe impl Sync for E1000 {}

#[derive(Debug, Clone, Copy)]
pub enum NicError {
    NotFound,
    BarMissing,
    PhysTranslate,
}

impl E1000 {
    #[inline(always)]
    unsafe fn read_reg(&self, off: usize) -> u32 {
        read_volatile(self.mmio.add(off) as *const u32)
    }

    #[inline(always)]
    unsafe fn write_reg(&self, off: usize, value: u32) {
        write_volatile(self.mmio.add(off) as *mut u32, value);
    }

    pub fn mac(&self) -> [u8; 6] {
        self.mac
    }

    /// Bind to the first supported Intel NIC found by the PCI scan.
    pub fn bind(scan: &PciScan) -> Result<Self, NicError> {
        let dev = scan
            .iter()
            .copied()
            .find(|d| d.vendor_id == VENDOR_INTEL && SUPPORTED_DEVICES.contains(&d.device_id))
            .ok_or(NicError::NotFound)?;
        Self::bind_device(&dev)
    }

    fn bind_device(dev: &PciDevice) -> Result<Self, NicError> {
        // Enable bus-master + memory-space + I/O-space access.
        unsafe {
            let cmd = pcie::config_read16(dev.bus, dev.device, dev.function, 0x04);
            pcie::config_write16(dev.bus, dev.device, dev.function, 0x04, cmd | 0x0007);
        }

        let bar0 = pcie::read_bar(dev, 0).ok_or(NicError::BarMissing)?;
        let phys_off = memory::phys_offset().ok_or(NicError::PhysTranslate)?;
        let mmio = (phys_off + bar0) as *mut u8;

        let _ = writeln!(
            Serial,
            "e1000: bound device {:04x}:{:04x} at PCI {:02x}:{:02x}.{}, BAR0=0x{:016x}, MMIO=0x{:016x}",
            dev.vendor_id,
            dev.device_id,
            dev.bus,
            dev.device,
            dev.function,
            bar0,
            mmio as u64
        );

        let mut nic = E1000 {
            mmio,
            mac: [0; 6],
            rx_cursor: 0,
            tx_cursor: 0,
        };
        nic.reset_and_init()?;
        Ok(nic)
    }

    fn reset_and_init(&mut self) -> Result<(), NicError> {
        unsafe {
            // 1. Mask all interrupts.
            self.write_reg(REG_IMC, 0xFFFF_FFFF);
            self.read_reg(REG_ICR);

            // 2. Soft reset.
            let ctrl = self.read_reg(REG_CTRL);
            self.write_reg(REG_CTRL, ctrl | CTRL_RST);
            // ~1 us delay loop — the e1000 datasheet requires waiting
            // for the device to clear RST. QEMU clears it within a
            // handful of register reads.
            for _ in 0..1_000 {
                if self.read_reg(REG_CTRL) & CTRL_RST == 0 {
                    break;
                }
                core::hint::spin_loop();
            }

            // 3. Mask interrupts again post-reset, then bring the link
            //    up (SLU). ASDE may also be useful but isn't required
            //    under QEMU.
            self.write_reg(REG_IMC, 0xFFFF_FFFF);
            self.read_reg(REG_ICR);
            self.write_reg(REG_CTRL, self.read_reg(REG_CTRL) | CTRL_SLU);

            // 4. Clear the multicast table (128 × 32 bits).
            for i in 0..128 {
                self.write_reg(REG_MTA_BASE + i * 4, 0);
            }

            // 5. Read MAC from RAL0/RAH0 (loaded from EEPROM at reset).
            let ral = self.read_reg(REG_RAL0);
            let rah = self.read_reg(REG_RAH0);
            self.mac = [
                (ral & 0xFF) as u8,
                ((ral >> 8) & 0xFF) as u8,
                ((ral >> 16) & 0xFF) as u8,
                ((ral >> 24) & 0xFF) as u8,
                (rah & 0xFF) as u8,
                ((rah >> 8) & 0xFF) as u8,
            ];
            let _ = writeln!(
                Serial,
                "e1000: MAC = {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                self.mac[0], self.mac[1], self.mac[2], self.mac[3], self.mac[4], self.mac[5]
            );

            // 6. Set up RX ring.
            self.init_rx()?;
            // 7. Set up TX ring.
            self.init_tx()?;
        }
        Ok(())
    }

    unsafe fn init_rx(&mut self) -> Result<(), NicError> {
        let rx_ring_ptr = core::ptr::addr_of_mut!(STORAGE.rx_ring) as *mut RxDesc;
        let rx_ring_phys =
            memory::virt_to_phys(rx_ring_ptr as u64).ok_or(NicError::PhysTranslate)?;

        for i in 0..RX_DESC_COUNT {
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.rx_buffers[i]) as *mut u8;
            let buf_phys = memory::virt_to_phys(buf_ptr as u64).ok_or(NicError::PhysTranslate)?;
            let desc = &mut *rx_ring_ptr.add(i);
            desc.addr = buf_phys;
            desc.length = 0;
            desc.checksum = 0;
            desc.status = 0;
            desc.errors = 0;
            desc.special = 0;
        }
        compiler_fence(Ordering::Release);

        self.write_reg(REG_RDBAL, (rx_ring_phys & 0xFFFF_FFFF) as u32);
        self.write_reg(REG_RDBAH, (rx_ring_phys >> 32) as u32);
        self.write_reg(
            REG_RDLEN,
            (RX_DESC_COUNT * core::mem::size_of::<RxDesc>()) as u32,
        );
        self.write_reg(REG_RDH, 0);
        // Tail points just past the last valid descriptor. With ring
        // full of empty buffers, RDT = RX_DESC_COUNT - 1.
        self.write_reg(REG_RDT, (RX_DESC_COUNT - 1) as u32);

        let rctl = RCTL_EN | RCTL_BAM | RCTL_BSIZE_2048 | RCTL_SECRC;
        self.write_reg(REG_RCTL, rctl);
        Ok(())
    }

    unsafe fn init_tx(&mut self) -> Result<(), NicError> {
        let tx_ring_ptr = core::ptr::addr_of_mut!(STORAGE.tx_ring) as *mut TxDesc;
        let tx_ring_phys =
            memory::virt_to_phys(tx_ring_ptr as u64).ok_or(NicError::PhysTranslate)?;

        for i in 0..TX_DESC_COUNT {
            let desc = &mut *tx_ring_ptr.add(i);
            desc.addr = 0;
            desc.length = 0;
            desc.cso = 0;
            desc.cmd = 0;
            desc.status = TX_STA_DD;
            desc.css = 0;
            desc.special = 0;
        }
        compiler_fence(Ordering::Release);

        self.write_reg(REG_TDBAL, (tx_ring_phys & 0xFFFF_FFFF) as u32);
        self.write_reg(REG_TDBAH, (tx_ring_phys >> 32) as u32);
        self.write_reg(
            REG_TDLEN,
            (TX_DESC_COUNT * core::mem::size_of::<TxDesc>()) as u32,
        );
        self.write_reg(REG_TDH, 0);
        self.write_reg(REG_TDT, 0);

        // Inter-packet-gap: IEEE 802.3 standard values, lifted from the
        // e1000 datasheet section 13.4.34. IPGT=10, IPGR1=10, IPGR2=10.
        self.write_reg(REG_TIPG, 10 | (10 << 10) | (10 << 20));

        let tctl = TCTL_EN | TCTL_PSP | (0x10 << TCTL_CT_SHIFT) | (0x40 << TCTL_COLD_SHIFT);
        self.write_reg(REG_TCTL, tctl);
        Ok(())
    }

    /// Poll the RX ring for one frame. Returns the frame bytes (copied
    /// into `out`) and recycles the descriptor. Returns `Ok(None)` when
    /// no frame is currently waiting.
    pub fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        unsafe {
            let desc_ptr = core::ptr::addr_of_mut!(STORAGE.rx_ring[self.rx_cursor]);
            let status = read_volatile(&(*desc_ptr).status);
            if status & RX_STA_DD == 0 {
                return None;
            }

            let length = read_volatile(&(*desc_ptr).length) as usize;
            let n = length.min(out.len());
            if (status & RX_STA_EOP) != 0 && n > 0 {
                let buf_ptr = core::ptr::addr_of!(STORAGE.rx_buffers[self.rx_cursor]) as *const u8;
                core::ptr::copy_nonoverlapping(buf_ptr, out.as_mut_ptr(), n);
            }

            // Recycle: clear status, advance tail so hardware can reuse it.
            write_volatile(&mut (*desc_ptr).status, 0);
            let old_cursor = self.rx_cursor;
            self.rx_cursor = (self.rx_cursor + 1) % RX_DESC_COUNT;
            self.write_reg(REG_RDT, old_cursor as u32);

            if n > 0 {
                Some(n)
            } else {
                None
            }
        }
    }

    /// Transmit one frame. Blocks (busy-polls) until the descriptor is
    /// owned by software, copies the payload into the TX buffer, then
    /// hands it back to hardware.
    pub fn transmit(&mut self, frame: &[u8]) {
        let len = frame.len().min(BUFFER_SIZE);
        unsafe {
            let desc_ptr = core::ptr::addr_of_mut!(STORAGE.tx_ring[self.tx_cursor]);

            // Wait until prior owner of this slot is done (DD set).
            while read_volatile(&(*desc_ptr).status) & TX_STA_DD == 0 {
                core::hint::spin_loop();
            }

            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.tx_buffers[self.tx_cursor]) as *mut u8;
            core::ptr::copy_nonoverlapping(frame.as_ptr(), buf_ptr, len);
            let buf_phys = match memory::virt_to_phys(buf_ptr as u64) {
                Some(p) => p,
                None => return,
            };

            (*desc_ptr).addr = buf_phys;
            (*desc_ptr).length = len as u16;
            (*desc_ptr).cso = 0;
            (*desc_ptr).cmd = TX_CMD_EOP | TX_CMD_IFCS | TX_CMD_RS;
            (*desc_ptr).css = 0;
            (*desc_ptr).special = 0;
            // DD must be cleared so hardware will set it on completion.
            write_volatile(&mut (*desc_ptr).status, 0);
            compiler_fence(Ordering::Release);

            self.tx_cursor = (self.tx_cursor + 1) % TX_DESC_COUNT;
            self.write_reg(REG_TDT, self.tx_cursor as u32);
        }
    }
}
