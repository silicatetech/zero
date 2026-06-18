// SPDX-License-Identifier: AGPL-3.0-or-later
//! Intel 700-series (i40e / "Fortville") ethernet driver — minimal,
//! polling-only.
//!
//! Target: Intel Ethernet Controller X710 for 10GbE SFP+
//! (PCI 0x8086:0x1572 and related XL710 / X710 / XXV710 / X722 family
//! IDs). The Cherry Server's Supermicro AOC-ATG-i2SM card carries one
//! of these on bus 05:00.0 with MAC 7c:c2:55:ab:73:e4; BAR0 lives at
//! 0x300_8180_0000 (8 MiB, prefetchable, 64-bit).
//!
//! ## Bring-up sequence
//!
//! 1. PCI binding against the X710 family device IDs.
//! 2. 64-bit BAR0 map (via `pcie::read_bar` + `memory::map_mmio`).
//! 3. PF software reset (PFGEN_CTRL.PFSWR).
//! 4. Admin Queue (ASQ + ARQ) ring allocation and register programming.
//! 5. Get Version (AQ 0x0001) — proves the AQ ↔ firmware link is alive.
//! 6. CLEAR_PXE_MODE (AQ 0x0110) — leave PXE configuration.
//! 7. Port MAC discovery via PRTGL_SAL/SAH.
//! 8. Discover PF's queue allocation via PFLAN_QALLOC.
//! 9. HMC (Host Memory Cache) setup — install one PAGED Segment
//!    Descriptor that backs the LAN TX[0] + LAN RX[0] context block.
//!    The SD write is treated as posted unless PFHMC_ERROR* reports a
//!    real error; on X710 FW 20.9 the PMSDWR bit can remain latched even
//!    though the descriptor is accepted.
//! 10. Write LAN TX queue context + LAN RX queue context into HMC.
//! 11. Initialise TX / RX descriptor rings + buffers.
//! 12. Set VSI promiscuous (AQ 0x0254) so we receive all frames.
//! 13. Enable queues via QTX_ENA / QRX_ENA.
//!
//! Polling-only: no MSI-X, no interrupt routing. `transmit` writes a
//! single TX data descriptor with EOP|RS and bumps QTX_TAIL; `receive`
//! checks the current RX descriptor's DD bit and recycles the slot.

#![allow(dead_code)]

use core::fmt::Write;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{compiler_fence, AtomicBool, Ordering};

use crate::arch::serial::Serial;
use crate::arch::x86_64::pcie::{self, PciDevice, PciScan};
use crate::memory;

/// Intel's PCI SIG vendor ID.
pub const VENDOR_INTEL: u16 = 0x8086;

/// X710 / XL710 / XXV710 / X722 family device IDs.
pub const SUPPORTED_DEVICES: &[u16] = &[
    0x1572, // X710 10GbE SFP+
    0x1580, // XL710 KX-B 40GbE backplane
    0x1581, // XL710 KX-C 40GbE backplane
    0x1583, // XL710 QDA2 40GbE QSFP+
    0x1584, // XL710 QDA1 40GbE QSFP+
    0x1585, // X710 QSFP+
    0x1586, // X710 10GBASE-T
    0x1589, // X710-T*L
    0x158A, // XXV710 25GbE backplane
    0x158B, // XXV710 25GbE SFP28
    0x37CE, // X722 KX 10GbE backplane
    0x37CF, // X722 QSFP 10GbE
    0x37D0, // X722 SFP+ 10GbE
    0x37D1, // X722 1GBASE-T
    0x37D2, // X722 10GBASE-T
    0x37D3, // X722 SFP+ I-SGMII
];

// ── Register offsets (32-bit MMIO against BAR0) ──────────────────────

const REG_PFGEN_CTRL: usize = 0x00092400;
const PFGEN_CTRL_PFSWR: u32 = 1 << 0;

const REG_GLGEN_RSTAT: usize = 0x000B8188;

const REG_PRTGL_SAL: usize = 0x001E2120;
const REG_PRTGL_SAH: usize = 0x001E2140;

// Admin Queue (per-PF) — Send (ASQ) + Receive (ARQ).
const REG_PF_ATQBAL: usize = 0x00080000;
const REG_PF_ATQBAH: usize = 0x00080100;
const REG_PF_ATQLEN: usize = 0x00080200;
const REG_PF_ATQH: usize = 0x00080300;
const REG_PF_ATQT: usize = 0x00080400;
const REG_PF_ARQBAL: usize = 0x00080080;
const REG_PF_ARQBAH: usize = 0x00080180;
const REG_PF_ARQLEN: usize = 0x00080280;
const REG_PF_ARQH: usize = 0x00080380;
const REG_PF_ARQT: usize = 0x00080480;
const AQLEN_ENABLE: u32 = 1 << 31;

// HMC (Host Memory Cache) registers.
const REG_PFHMC_SDCMD: usize = 0x000C0000;
const REG_PFHMC_SDDATALOW: usize = 0x000C0100;
const REG_PFHMC_SDDATAHIGH: usize = 0x000C0200;
const REG_PFHMC_ERRORINFO: usize = 0x000C0400;
const REG_PFHMC_ERRORDATA: usize = 0x000C0500;
const PFHMC_SDCMD_PMSDWR: u32 = 1 << 31;
const PFHMC_SDDATALOW_VALID: u32 = 1 << 0;
const PFHMC_SDDATALOW_PAGED: u32 = 0 << 1;
const PFHMC_SDDATALOW_DIRECT: u32 = 1 << 1;

// GLHMC base/count registers for our PF's LAN TX/RX object arrays.
// 0x000C6200 + 4*pf for LANTXBASE; +0x100 stride for related groups.
const REG_GLHMC_LANTXBASE: usize = 0x000C6200;
const REG_GLHMC_LANTXCNT: usize = 0x000C6300;
const REG_GLHMC_LANRXBASE: usize = 0x000C6400;
const REG_GLHMC_LANRXCNT: usize = 0x000C6500;
const REG_GLHMC_FCOEMAX: usize = 0x000C2014;
const REG_GLHMC_FCOEDDPBASE: usize = 0x000C6600;
const REG_GLHMC_FCOEDDPCNT: usize = 0x000C6700;
const REG_GLHMC_FCOEFBASE: usize = 0x000C6800;
const REG_GLHMC_FCOEFCNT: usize = 0x000C6900;

// PF queue allocation — FIRSTQ/LASTQ globals for this PF.
const REG_PFLAN_QALLOC: usize = 0x001C0400;
const PFLAN_QALLOC_FIRSTQ_MASK: u32 = 0x0000_07FF;
const PFLAN_QALLOC_LASTQ_SHIFT: u32 = 16;

// Per-queue TX / RX registers. Indexed by GLOBAL queue id.
const REG_QTX_CTL_BASE: usize = 0x00104000;
const REG_QTX_HEAD_BASE: usize = 0x000E4000;
const REG_QTX_TAIL_BASE: usize = 0x00108000;
const REG_QTX_ENA_BASE: usize = 0x00100000;
const REG_QRX_TAIL_BASE: usize = 0x00128000;
const REG_QRX_ENA_BASE: usize = 0x00120000;
const REG_QINT_TQCTL_BASE: usize = 0x0003C000;
const REG_QINT_RQCTL_BASE: usize = 0x0003A000;
const REG_GLLAN_TXPRE_QDIS_BASE: usize = 0x000E6500;

const QENA_REQ: u32 = 1 << 0;
const QENA_STAT: u32 = 1 << 2;
const QTX_CTL_PF_QUEUE: u32 = 0x2; // PFVF_Q[1:0] = 0b10 → PF queue

// PF function index (bits[14:12] of GLPCI_CAPSUP — or simply derived
// from PCI function). For the Cherry Server the X710 sits at function 0;
// firmware also reflects this in PFGEN_PORTNUM[1:0]. We don't have a
// quick way to read the function id from MMIO, so we infer it from the
// PCI BDF at bind time.
fn pf_glhmc_reg(base: usize, pf: u8) -> usize {
    base + (pf as usize) * 4
}

// ── AQ descriptor flags ────────────────────────────────────────────
const AQ_FLAG_DD: u16 = 1 << 0;
const AQ_FLAG_CMP: u16 = 1 << 1;
const AQ_FLAG_ERR: u16 = 1 << 2;
const AQ_FLAG_RD: u16 = 1 << 10;
const AQ_FLAG_BUF: u16 = 1 << 12;
const AQ_FLAG_SI: u16 = 1 << 13;

// ── AQ opcodes ─────────────────────────────────────────────────────
const AQ_OP_GET_VERSION: u16 = 0x0001;
const AQ_OP_CLEAR_PXE_MODE: u16 = 0x0110;
const AQ_OP_GET_SWITCH_CONFIG: u16 = 0x0200;
const AQ_OP_GET_VSI_PARAMS: u16 = 0x0212;
const AQ_OP_SET_VSI_PROMISC_MODES: u16 = 0x0254;

/// Magic value the X710 expects in param0 byte 0 for CLEAR_PXE_MODE.
const CLEAR_PXE_MAGIC: u8 = 0x2;

const SWITCH_CONFIG_HEADER_LEN: usize = 16;
const SWITCH_CONFIG_ELEMENT_LEN: usize = 16;
const SWITCH_ELEMENT_TYPE_VSI: u8 = 19;
const VSI_PROPERTIES_LEN: usize = 128;
const VSI_QS_HANDLE_OFFSET: usize = 96;
const VSI_QS_HANDLE_INVALID: u16 = 0xFFFF;

const VSI_PROMISC_UNICAST: u16 = 0x0001;
const VSI_PROMISC_MULTICAST: u16 = 0x0002;
const VSI_PROMISC_BROADCAST: u16 = 0x0004;
const VSI_PROMISC_DEFAULT: u16 = 0x0008;

// ── Ring sizing ────────────────────────────────────────────────────

pub const AQ_DESC_COUNT: usize = 32;
pub const AQ_BUF_SIZE: usize = 4096;

/// TX descriptor count. Power of two; the X710 requires QLEN to be
/// in [8, 8160] and a multiple of 8.
pub const TX_DESC_COUNT: usize = 64;
/// RX descriptor count.
pub const RX_DESC_COUNT: usize = 64;
/// Per-RX-slot packet buffer size in bytes. Encoded in the LAN RX
/// context's DBUFF field as DBUFF = size / 128, so must be a multiple
/// of 128 in [128, 16384].
pub const RX_BUFFER_SIZE: usize = 2048;
/// TX buffer size matches RX so frames echoed by the upper stack
/// always fit. Hardware reads from the buffer per-descriptor; the
/// only restriction is that BSZ in cmd_type_offset_bsz is 14 bits
/// (≤16383 bytes).
pub const TX_BUFFER_SIZE: usize = 2048;

// ── AQ descriptor ──────────────────────────────────────────────────

#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct AqDesc {
    flags: u16,
    opcode: u16,
    datalen: u16,
    retval: u16,
    cookie_high: u32,
    cookie_low: u32,
    param0: u32,
    param1: u32,
    addr_high: u32,
    addr_low: u32,
}

const EMPTY_AQ_DESC: AqDesc = AqDesc {
    flags: 0,
    opcode: 0,
    datalen: 0,
    retval: 0,
    cookie_high: 0,
    cookie_low: 0,
    param0: 0,
    param1: 0,
    addr_high: 0,
    addr_low: 0,
};

// ── TX / RX descriptor layouts ─────────────────────────────────────
//
// TX data descriptor (16 bytes) — section 8.4.2.1 of the X710 spec.
//   QW0: buffer_addr (physical address of buffer)
//   QW1: cmd_type_offset_bsz
//     [3:0]   DTYPE (data = 0)
//     [15:4]  CMD  (10 bits, but encoded starting at bit 4 — i.e.
//             bit 4 = EOP, bit 5 = RS, bit 6 = ICRC, ...)
//     [29:16] OFFSET (3 sub-fields, all 0 for plain L2)
//     [47:30] L2TAG / TXBUFSZ — TX buffer size starts at bit 34 (14 bits)
//     [63:48] L2TAG1 — 0 for no VLAN insertion
//
// Hardware writes back DTYPE=0xF when the descriptor is done if RS is
// set. We treat writeback as advisory and just keep advancing.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct TxDesc {
    buffer_addr: u64,
    cmd_type_offset_bsz: u64,
}

const TX_DESC_DTYPE_MASK: u64 = 0xF;
const TX_DESC_CMD_EOP: u64 = 1 << 4;
const TX_DESC_CMD_RS: u64 = 1 << 5;
const TX_DESC_CMD_ICRC: u64 = 1 << 6;
const TX_DESC_BSZ_SHIFT: u64 = 34;
const TX_DESC_DTYPE_DONE: u64 = 0xF; // hardware writeback

const EMPTY_TX_DESC: TxDesc = TxDesc {
    buffer_addr: 0,
    cmd_type_offset_bsz: 0,
};

// RX 32-byte descriptor (advanced) — section 8.3.2.2.
// Read format (driver-written, before fill):
//   QW0: pkt_addr (buffer phys addr)
//   QW1: hdr_addr (header buffer, 0 for non-split)
//   QW2/QW3: reserved
// Writeback format (HW-written, after RX):
//   QW1 (the field we name `status_error_len`):
//     [18:0]   STATUS (bit 0 = DD, bit 1 = EOP, ...)
//     [37:19]  ERROR
//     [51:38]  PKT_LEN
//     [63:52]  PTYPE
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct RxDesc {
    qw0: u64,
    qw1: u64,
    qw2: u64,
    qw3: u64,
}

const RX_DESC_STATUS_DD: u64 = 1 << 0;
const RX_DESC_STATUS_EOP: u64 = 1 << 1;
const RX_DESC_PKT_LEN_SHIFT: u64 = 38;
const RX_DESC_PKT_LEN_MASK: u64 = (1 << 14) - 1;

const EMPTY_RX_DESC: RxDesc = RxDesc {
    qw0: 0,
    qw1: 0,
    qw2: 0,
    qw3: 0,
};

// ── HMC backing layout ─────────────────────────────────────────────
//
// We use PAGED HMC mode with a single Segment Descriptor (SD[0]). One
// 4 KiB Page Descriptor Table (PDT) plus one 4 KiB data page is enough
// because we configure only 1 TX queue (128 B context) and 1 RX queue
// (32 B context) — both fit comfortably in the first 4 KiB.
//
// Layout inside the HMC data page (matches the register defaults we
// program below: LANTXBASE = 0, LANRXBASE = 1 in 512-byte units):
//   bytes [0   .. 128) — LAN_TX[0] context (128 bytes)
//   bytes [512 .. 544) — LAN_RX[0] context (32 bytes)
//
// PDT[0] points at this data page; PDT[1..512] are zero (hardware will
// fault if it tries to access them, but with QCNT=1 it won't).
const HMC_PAGE_SIZE: usize = 4096;
const PDT_ENTRY_VALID: u64 = 1 << 0;
const LANTX_CONTEXT_OFFSET: usize = 0;
const LANRX_CONTEXT_OFFSET: usize = 512;

/// 4K-aligned wrapper for HMC backing pages. The X710 requires the
/// PDT and data page physical addresses to have bits[11:0] = 0 — the
/// PDT base lives in SDDATALOW bits[31:12], and PDT entries encode
/// data page address >> 12. Without this wrapper the page would land
/// at an interior offset of DriverStorage and only its 8-byte
/// alignment would be guaranteed.
#[repr(C, align(4096))]
struct HmcPage([u8; HMC_PAGE_SIZE]);

// ── Driver storage (BSS) ───────────────────────────────────────────

#[repr(C, align(4096))]
struct DriverStorage {
    // Admin queue rings + shared command buffer.
    asq_ring: [AqDesc; AQ_DESC_COUNT],
    arq_ring: [AqDesc; AQ_DESC_COUNT],
    asq_buf: [u8; AQ_BUF_SIZE],
    arq_buf: [u8; AQ_BUF_SIZE],
    // HMC backing: 1 PDT page + 1 data page. Wrapped in HmcPage so each
    // page is independently 4K-aligned (the X710 latches the low 12
    // bits of these addresses into format-encoded fields and silently
    // mis-programs HMC if they aren't zero).
    hmc_pdt: HmcPage,
    hmc_data: HmcPage,
    // TX / RX descriptor rings + per-slot buffers.
    tx_ring: [TxDesc; TX_DESC_COUNT],
    rx_ring: [RxDesc; RX_DESC_COUNT],
    tx_buffers: [[u8; TX_BUFFER_SIZE]; TX_DESC_COUNT],
    rx_buffers: [[u8; RX_BUFFER_SIZE]; RX_DESC_COUNT],
}

static mut STORAGE: DriverStorage = DriverStorage {
    asq_ring: [EMPTY_AQ_DESC; AQ_DESC_COUNT],
    arq_ring: [EMPTY_AQ_DESC; AQ_DESC_COUNT],
    asq_buf: [0u8; AQ_BUF_SIZE],
    arq_buf: [0u8; AQ_BUF_SIZE],
    hmc_pdt: HmcPage([0u8; HMC_PAGE_SIZE]),
    hmc_data: HmcPage([0u8; HMC_PAGE_SIZE]),
    tx_ring: [EMPTY_TX_DESC; TX_DESC_COUNT],
    rx_ring: [EMPTY_RX_DESC; RX_DESC_COUNT],
    tx_buffers: [[0u8; TX_BUFFER_SIZE]; TX_DESC_COUNT],
    rx_buffers: [[0u8; RX_BUFFER_SIZE]; RX_DESC_COUNT],
};

static DATA_PATH_WARNED: AtomicBool = AtomicBool::new(false);

/// Outcome of an AQ command.
#[derive(Debug, Clone, Copy)]
pub struct AqResult {
    pub retval: u16,
    pub param0: u32,
    pub param1: u32,
    pub addr_high: u32,
    pub addr_low: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum NicError {
    NotFound,
    BarMissing,
    PhysTranslate,
    MmioMapFailed,
    ResetTimeout,
    AqTimeout,
    AqRetval(u16),
    HmcTimeout,
    QueueEnableTimeout,
}

/// Bound i40e instance.
pub struct I40e {
    mmio: *mut u8,
    mac: [u8; 6],
    /// PCI function number — used to index per-PF GLHMC_* registers.
    pf_id: u8,
    /// Global queue id of our TX / RX queue (= PFLAN_QALLOC.FIRSTQ).
    queue_id: u16,
    /// Default port VSI's SEID (returned by Get Switch Config). 0 if
    /// AQ discovery failed; promiscuous + VSI updates then skipped.
    vsi_seid: u16,
    /// Scheduler queue-set handle for traffic class 0, returned by
    /// Get VSI Params. TX HMC context carries this in `rdylist`.
    tx_rdylist: u16,
    /// ASQ producer cursor.
    asq_tail: usize,
    /// ARQ consumer cursor.
    arq_head: usize,
    /// TX ring producer cursor.
    tx_cursor: usize,
    /// RX ring consumer cursor.
    rx_cursor: usize,
    /// True once admin queue + queue setup have completed. While false
    /// the data-path calls fall back to drop-and-warn.
    ready: bool,
}

unsafe impl Send for I40e {}
unsafe impl Sync for I40e {}

impl I40e {
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

    /// Bind to the first supported i40e device the PCI scan turned up.
    pub fn bind(scan: &PciScan) -> Result<Self, NicError> {
        let dev = scan
            .iter()
            .copied()
            .find(|d| d.vendor_id == VENDOR_INTEL && SUPPORTED_DEVICES.contains(&d.device_id))
            .ok_or(NicError::NotFound)?;
        Self::bind_device(&dev)
    }

    fn bind_device(dev: &PciDevice) -> Result<Self, NicError> {
        // Enable memory + bus-master so the device can DMA against our
        // rings.
        unsafe {
            let cmd = pcie::config_read16(dev.bus, dev.device, dev.function, 0x04);
            pcie::config_write16(dev.bus, dev.device, dev.function, 0x04, cmd | 0x0006);
        }

        let bar0 = pcie::read_bar(dev, 0).ok_or(NicError::BarMissing)?;
        // The X710 datasheet documents registers out to ~0x001C0480
        // (PFGEN_PORTNUM) and we add PFLAN_QALLOC (0x001C0400) +
        // QTX_TAIL (0x00108000), QRX_TAIL (0x00128000), QTX_ENA
        // (0x00100000), QRX_ENA (0x00120000), so 2 MiB covers
        // everything we touch comfortably.
        const BAR0_MAP_SIZE: usize = 2 * 1024 * 1024;
        let mmio = memory::map_mmio(bar0, BAR0_MAP_SIZE).map_err(|_| NicError::MmioMapFailed)?;

        let _ = writeln!(
            Serial,
            "i40e: bound device {:04x}:{:04x} at PCI {:02x}:{:02x}.{}, BAR0=0x{:016x}, MMIO=0x{:016x}",
            dev.vendor_id,
            dev.device_id,
            dev.bus,
            dev.device,
            dev.function,
            bar0,
            mmio as u64
        );

        let mut nic = I40e {
            mmio,
            mac: [0; 6],
            pf_id: dev.function,
            queue_id: 0,
            vsi_seid: 0,
            tx_rdylist: 0,
            asq_tail: 0,
            arq_head: 0,
            tx_cursor: 0,
            rx_cursor: 0,
            ready: false,
        };
        nic.reset_pf()?;
        unsafe { nic.init_admin_queue()? };
        nic.probe_firmware()?;
        nic.clear_pxe_mode()?;
        nic.read_port_mac();
        // From here on data-path bring-up. Failures past this point are
        // logged but don't abort the bind — the AQ + MAC discovery
        // alone are useful diagnostically, and the kernel's e1000 fall-
        // back may still be reachable elsewhere.
        if let Err(e) = nic.bring_up_data_path() {
            let _ = writeln!(
                Serial,
                "i40e: data-path bring-up FAILED ({:?}) — falling back to drop-and-warn",
                e
            );
        } else {
            nic.ready = true;
            let _ = writeln!(
                Serial,
                "i40e: data path online — TX/RX queue {} enabled, VSI SEID {}",
                nic.queue_id, nic.vsi_seid
            );
        }
        Ok(nic)
    }

    /// Trigger a PF-scope software reset and poll PFGEN_CTRL.PFSWR
    /// until the device clears it.
    fn reset_pf(&mut self) -> Result<(), NicError> {
        unsafe {
            let ctrl = self.read_reg(REG_PFGEN_CTRL);
            self.write_reg(REG_PFGEN_CTRL, ctrl | PFGEN_CTRL_PFSWR);

            for _ in 0..500_000 {
                if self.read_reg(REG_PFGEN_CTRL) & PFGEN_CTRL_PFSWR == 0 {
                    let _ = writeln!(Serial, "i40e: PF reset complete");
                    return Ok(());
                }
                core::hint::spin_loop();
            }
        }
        let _ = writeln!(Serial, "i40e: PF reset TIMED OUT");
        Err(NicError::ResetTimeout)
    }

    /// Program ASQ + ARQ base / length registers and flip the enable
    /// bit.
    unsafe fn init_admin_queue(&mut self) -> Result<(), NicError> {
        let asq_ptr = core::ptr::addr_of_mut!(STORAGE.asq_ring) as *mut AqDesc;
        let arq_ptr = core::ptr::addr_of_mut!(STORAGE.arq_ring) as *mut AqDesc;
        for i in 0..AQ_DESC_COUNT {
            write_volatile(asq_ptr.add(i), EMPTY_AQ_DESC);
            write_volatile(arq_ptr.add(i), EMPTY_AQ_DESC);
        }

        let arq_buf_base = core::ptr::addr_of_mut!(STORAGE.arq_buf) as *mut u8;
        let per_slot = (AQ_BUF_SIZE / AQ_DESC_COUNT) as u32;
        for i in 0..AQ_DESC_COUNT {
            let slot_va = arq_buf_base.add(i * per_slot as usize);
            let slot_pa = memory::virt_to_phys(slot_va as u64).ok_or(NicError::PhysTranslate)?;
            let entry = &mut *arq_ptr.add(i);
            entry.flags = AQ_FLAG_BUF;
            entry.datalen = per_slot as u16;
            entry.addr_high = (slot_pa >> 32) as u32;
            entry.addr_low = (slot_pa & 0xFFFF_FFFF) as u32;
        }
        compiler_fence(Ordering::Release);

        let asq_phys = memory::virt_to_phys(asq_ptr as u64).ok_or(NicError::PhysTranslate)?;
        let arq_phys = memory::virt_to_phys(arq_ptr as u64).ok_or(NicError::PhysTranslate)?;

        self.write_reg(REG_PF_ATQLEN, 0);
        self.write_reg(REG_PF_ATQH, 0);
        self.write_reg(REG_PF_ATQT, 0);
        self.write_reg(REG_PF_ATQBAL, (asq_phys & 0xFFFF_FFFF) as u32);
        self.write_reg(REG_PF_ATQBAH, (asq_phys >> 32) as u32);
        self.write_reg(REG_PF_ATQLEN, (AQ_DESC_COUNT as u32) | AQLEN_ENABLE);

        self.write_reg(REG_PF_ARQLEN, 0);
        self.write_reg(REG_PF_ARQH, 0);
        self.write_reg(REG_PF_ARQT, 0);
        self.write_reg(REG_PF_ARQBAL, (arq_phys & 0xFFFF_FFFF) as u32);
        self.write_reg(REG_PF_ARQBAH, (arq_phys >> 32) as u32);
        self.write_reg(REG_PF_ARQLEN, (AQ_DESC_COUNT as u32) | AQLEN_ENABLE);
        self.write_reg(REG_PF_ARQT, (AQ_DESC_COUNT - 1) as u32);

        self.asq_tail = 0;
        self.arq_head = 0;

        let _ = writeln!(
            Serial,
            "i40e: admin queue online (ASQ @0x{:016x}, ARQ @0x{:016x}, depth={})",
            asq_phys, arq_phys, AQ_DESC_COUNT
        );
        Ok(())
    }

    /// Issue an AQ command with no external buffer.
    fn aq_send_simple(
        &mut self,
        opcode: u16,
        param0: u32,
        param1: u32,
        addr_high: u32,
        addr_low: u32,
    ) -> Result<AqResult, NicError> {
        self.aq_send(opcode, 0, 0, param0, param1, addr_high, addr_low)
    }

    /// Issue an AQ command, optionally with an external buffer.
    /// `flags_in` is OR'd into the descriptor flags (callers use this
    /// to set RD when the buffer holds command data instead of being
    /// only a response sink).
    fn aq_send(
        &mut self,
        opcode: u16,
        flags_in: u16,
        datalen: u16,
        param0: u32,
        param1: u32,
        addr_high: u32,
        addr_low: u32,
    ) -> Result<AqResult, NicError> {
        unsafe {
            let desc_ptr = core::ptr::addr_of_mut!(STORAGE.asq_ring[self.asq_tail]);
            let mut desc = EMPTY_AQ_DESC;
            desc.flags = flags_in | AQ_FLAG_SI;
            desc.opcode = opcode;
            desc.datalen = datalen;
            desc.param0 = param0;
            desc.param1 = param1;
            desc.addr_high = addr_high;
            desc.addr_low = addr_low;
            write_volatile(desc_ptr, desc);
            compiler_fence(Ordering::Release);

            let next = (self.asq_tail + 1) % AQ_DESC_COUNT;
            self.write_reg(REG_PF_ATQT, next as u32);

            let mut spins = 0u64;
            loop {
                let flags = read_volatile(&(*desc_ptr).flags);
                if flags & AQ_FLAG_DD != 0 {
                    let retval = read_volatile(&(*desc_ptr).retval);
                    let result = AqResult {
                        retval,
                        param0: read_volatile(&(*desc_ptr).param0),
                        param1: read_volatile(&(*desc_ptr).param1),
                        addr_high: read_volatile(&(*desc_ptr).addr_high),
                        addr_low: read_volatile(&(*desc_ptr).addr_low),
                    };
                    self.asq_tail = next;
                    if flags & AQ_FLAG_ERR != 0 || retval != 0 {
                        return Err(NicError::AqRetval(retval));
                    }
                    return Ok(result);
                }
                spins += 1;
                if spins >= 2_000_000 {
                    let _ = writeln!(
                        Serial,
                        "i40e: AQ command 0x{:04x} TIMED OUT (flags=0x{:04x})",
                        opcode, flags
                    );
                    return Err(NicError::AqTimeout);
                }
                core::hint::spin_loop();
            }
        }
    }

    fn probe_firmware(&mut self) -> Result<(), NicError> {
        let res = self.aq_send_simple(AQ_OP_GET_VERSION, 0, 0, 0, 0)?;
        let fw_major = (res.addr_high >> 16) & 0xFFFF;
        let fw_minor = res.addr_high & 0xFFFF;
        let api_major = (res.addr_low >> 16) & 0xFFFF;
        let api_minor = res.addr_low & 0xFFFF;
        let _ = writeln!(
            Serial,
            "i40e: firmware FW {}.{}, API {}.{}, ROM 0x{:08x}, build 0x{:08x}",
            fw_major, fw_minor, api_major, api_minor, res.param0, res.param1
        );
        Ok(())
    }

    fn clear_pxe_mode(&mut self) -> Result<(), NicError> {
        let param0 = CLEAR_PXE_MAGIC as u32;
        match self.aq_send_simple(AQ_OP_CLEAR_PXE_MODE, param0, 0, 0, 0) {
            Ok(_) => {
                let _ = writeln!(Serial, "i40e: PXE mode cleared");
                Ok(())
            }
            Err(NicError::AqRetval(rc)) => {
                if rc == 13 {
                    let _ = writeln!(Serial, "i40e: PXE mode was already cleared");
                    Ok(())
                } else {
                    let _ = writeln!(Serial, "i40e: CLEAR_PXE_MODE failed (rc={})", rc);
                    Err(NicError::AqRetval(rc))
                }
            }
            Err(e) => Err(e),
        }
    }

    fn read_port_mac(&mut self) {
        unsafe {
            let sal = self.read_reg(REG_PRTGL_SAL);
            let sah = self.read_reg(REG_PRTGL_SAH);
            self.mac = [
                (sal & 0xFF) as u8,
                ((sal >> 8) & 0xFF) as u8,
                ((sal >> 16) & 0xFF) as u8,
                ((sal >> 24) & 0xFF) as u8,
                (sah & 0xFF) as u8,
                ((sah >> 8) & 0xFF) as u8,
            ];
            let _ = writeln!(
                Serial,
                "i40e: port MAC = {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} (SAL=0x{:08x}, SAH=0x{:08x})",
                self.mac[0], self.mac[1], self.mac[2], self.mac[3],
                self.mac[4], self.mac[5], sal, sah
            );
        }
    }

    /// End-to-end data-path bring-up: queue allocation, HMC, contexts,
    /// rings, queue enable.
    fn bring_up_data_path(&mut self) -> Result<(), NicError> {
        self.discover_queue_range()?;
        // Look up the default VSI before we touch HMC — the AQ is
        // already alive and the SEID is needed for the promiscuous
        // mode toggle later.
        let _ = self.discover_default_vsi(); // best-effort
        let _ = self.discover_vsi_params(); // best-effort
        unsafe {
            self.setup_hmc()?;
            self.init_tx_ring()?;
            self.init_rx_ring()?;
            self.write_lan_tx_context()?;
            self.write_lan_rx_context()?;
            self.enable_queues()?;
        }
        // Promiscuous so we receive frames regardless of the dest MAC.
        // Some firmware revisions refuse this if multicast promiscuous
        // is requested without a SEID; we tolerate failure since the
        // port MAC filter alone is enough for the TCP shell path.
        if self.vsi_seid != 0 {
            let _ = self.set_vsi_promiscuous();
        }
        Ok(())
    }

    fn discover_queue_range(&mut self) -> Result<(), NicError> {
        let qalloc = unsafe { self.read_reg(REG_PFLAN_QALLOC) };
        let firstq = (qalloc & PFLAN_QALLOC_FIRSTQ_MASK) as u16;
        let lastq = ((qalloc >> PFLAN_QALLOC_LASTQ_SHIFT) & PFLAN_QALLOC_FIRSTQ_MASK) as u16;
        self.queue_id = firstq;
        let _ = writeln!(
            Serial,
            "i40e: PF queue range [{}..{}] (using queue {})",
            firstq, lastq, firstq
        );
        Ok(())
    }

    /// AQ 0x0200 Get Switch Config — returns a list of switch elements
    /// in the ARQ-style external buffer. The first element after the
    /// header that is a VSI is the default port VSI; its SEID lives in
    /// bytes [2..4] of the element. We only need to know that it's
    /// non-zero to enable promiscuous mode.
    fn discover_default_vsi(&mut self) -> Result<(), NicError> {
        unsafe {
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.asq_buf) as *mut u8;
            // Clear so a half-populated response doesn't look valid.
            core::ptr::write_bytes(buf_ptr, 0, AQ_BUF_SIZE);
            let buf_phys = memory::virt_to_phys(buf_ptr as u64).ok_or(NicError::PhysTranslate)?;
            let datalen = AQ_BUF_SIZE as u16;
            let res = self.aq_send(
                AQ_OP_GET_SWITCH_CONFIG,
                AQ_FLAG_BUF,
                datalen,
                0,
                0,
                (buf_phys >> 32) as u32,
                (buf_phys & 0xFFFF_FFFF) as u32,
            );
            match res {
                Ok(_) => {
                    // Response layout matches Intel/Linux i40e:
                    // 16-byte header (num_reported, num_total,
                    // 12 reserved bytes), then 16-byte elements:
                    //   [0]  element_type (u8)
                    //   [1]  revision (u8)
                    //   [2..4] seid          (u16, little-endian)
                    //   [4..6] uplink_seid   (u16)
                    //   [6..8] downlink_seid (u16)
                    //   [8..11] reserved
                    //   [11] connection_type
                    //   [12..14] scheduler_id
                    //   [14..16] element_info
                    //
                    // Type 19 is VSI. This SEID is required for
                    // Set VSI Promiscuous Modes; parsing at byte 8
                    // reads the reserved header tail and misses it.
                    let num_reported =
                        u16::from_le_bytes([*buf_ptr.add(0), *buf_ptr.add(1)]) as usize;
                    let mut found = 0u16;
                    for i in 0..num_reported {
                        let off = SWITCH_CONFIG_HEADER_LEN + i * SWITCH_CONFIG_ELEMENT_LEN;
                        if off + SWITCH_CONFIG_ELEMENT_LEN > AQ_BUF_SIZE {
                            break;
                        }
                        let etype = *buf_ptr.add(off);
                        let seid =
                            u16::from_le_bytes([*buf_ptr.add(off + 2), *buf_ptr.add(off + 3)]);
                        if etype == SWITCH_ELEMENT_TYPE_VSI && seid != 0 {
                            found = seid;
                            break;
                        }
                    }
                    if found != 0 {
                        self.vsi_seid = found;
                        let _ = writeln!(
                            Serial,
                            "i40e: default VSI SEID = {} (from Get Switch Config, {} elements)",
                            found, num_reported
                        );
                    } else {
                        let _ = writeln!(
                            Serial,
                            "i40e: Get Switch Config returned {} elements but no PF VSI",
                            num_reported
                        );
                    }
                }
                Err(e) => {
                    let _ = writeln!(
                        Serial,
                        "i40e: Get Switch Config failed ({:?}) — promiscuous mode skipped",
                        e
                    );
                }
            }
        }
        Ok(())
    }

    /// AQ 0x0212 Get VSI Params. The firmware writes `qs_handle[8]`
    /// into the last 32 bytes of the 128-byte VSI property buffer.
    /// Linux feeds `qs_handle[tc]` into the TX HMC `rdylist` field;
    /// doing the same keeps our single TX queue attached to TC0's
    /// scheduler queue-set instead of relying on reset-default zero.
    fn discover_vsi_params(&mut self) -> Result<(), NicError> {
        if self.vsi_seid == 0 {
            return Ok(());
        }
        unsafe {
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.asq_buf) as *mut u8;
            core::ptr::write_bytes(buf_ptr, 0, VSI_PROPERTIES_LEN);
            let buf_phys = memory::virt_to_phys(buf_ptr as u64).ok_or(NicError::PhysTranslate)?;
            let res = self.aq_send(
                AQ_OP_GET_VSI_PARAMS,
                AQ_FLAG_BUF,
                VSI_PROPERTIES_LEN as u16,
                self.vsi_seid as u32,
                0,
                (buf_phys >> 32) as u32,
                (buf_phys & 0xFFFF_FFFF) as u32,
            );
            match res {
                Ok(_) => {
                    let qs0 = u16::from_le_bytes([
                        *buf_ptr.add(VSI_QS_HANDLE_OFFSET),
                        *buf_ptr.add(VSI_QS_HANDLE_OFFSET + 1),
                    ]);
                    if qs0 != VSI_QS_HANDLE_INVALID {
                        self.tx_rdylist = qs0 & 0x03FF;
                        let _ = writeln!(
                            Serial,
                            "i40e: VSI {} TC0 queue-set handle = {}",
                            self.vsi_seid, self.tx_rdylist
                        );
                    } else {
                        let _ = writeln!(
                            Serial,
                            "i40e: VSI {} TC0 queue-set handle invalid — using rdylist 0",
                            self.vsi_seid
                        );
                    }
                }
                Err(e) => {
                    let _ = writeln!(
                        Serial,
                        "i40e: Get VSI Params failed ({:?}) — using rdylist 0",
                        e
                    );
                }
            }
        }
        Ok(())
    }

    /// AQ 0x0254 Set VSI Promiscuous Modes.
    ///
    /// Descriptor direct command layout:
    /// * param0[15:0]  = promiscuous_flags
    /// * param0[31:16] = valid_flags
    /// * param1[15:0]  = VSI SEID
    /// * param1[31:16] = VLAN tag (0 when unused)
    fn set_vsi_promisc_flags(&mut self, flags: u16, valid: u16) -> Result<(), NicError> {
        let param0 = (flags as u32) | ((valid as u32) << 16);
        let param1 = self.vsi_seid as u32;
        let res = self.aq_send_simple(AQ_OP_SET_VSI_PROMISC_MODES, param0, param1, 0, 0);
        match res {
            Ok(_) => {
                let _ = writeln!(
                    Serial,
                    "i40e: VSI {} promisc update flags=0x{:04x}, valid=0x{:04x}",
                    self.vsi_seid, flags, valid
                );
                Ok(())
            }
            Err(e) => {
                let _ = writeln!(
                    Serial,
                    "i40e: Set VSI Promiscuous failed ({:?}) — port MAC filter only",
                    e
                );
                Ok(())
            }
        }
    }

    /// Make the discovered port VSI the default receive target, then
    /// enable unicast/multicast/broadcast promiscuous receive for
    /// bring-up diagnostics.
    fn set_vsi_promiscuous(&mut self) -> Result<(), NicError> {
        let _ = self.set_vsi_promisc_flags(VSI_PROMISC_DEFAULT, VSI_PROMISC_DEFAULT);
        let _ = self.set_vsi_promisc_flags(VSI_PROMISC_UNICAST, VSI_PROMISC_UNICAST);
        let _ = self.set_vsi_promisc_flags(VSI_PROMISC_MULTICAST, VSI_PROMISC_MULTICAST);
        self.set_vsi_promisc_flags(VSI_PROMISC_BROADCAST, VSI_PROMISC_BROADCAST)
    }

    /// Install one PAGED HMC Segment Descriptor that backs the LAN
    /// TX[0] and LAN RX[0] context block.
    unsafe fn setup_hmc(&mut self) -> Result<(), NicError> {
        // Zero the backing pages so any HMC fields we don't explicitly
        // program read back as 0 (matches the X710 reset behaviour).
        let pdt_ptr = core::ptr::addr_of_mut!(STORAGE.hmc_pdt) as *mut u8;
        let data_ptr = core::ptr::addr_of_mut!(STORAGE.hmc_data) as *mut u8;
        core::ptr::write_bytes(pdt_ptr, 0, HMC_PAGE_SIZE);
        core::ptr::write_bytes(data_ptr, 0, HMC_PAGE_SIZE);

        let pdt_phys = memory::virt_to_phys(pdt_ptr as u64).ok_or(NicError::PhysTranslate)?;
        let data_phys = memory::virt_to_phys(data_ptr as u64).ok_or(NicError::PhysTranslate)?;

        // PDT[0] = data_phys | VALID. PDT entries are 8 bytes; only
        // bit 0 (VALID) is required for paged backing. The X710 fills
        // bits [12:1] with the upper bits of the page address on
        // writeback (bit 0 always stays 1).
        let pdt_entries = pdt_ptr as *mut u64;
        write_volatile(pdt_entries, data_phys | PDT_ENTRY_VALID);
        compiler_fence(Ordering::Release);

        // Tell the X710 we have exactly 1 LAN TX queue + 1 LAN RX queue
        // for this PF. The base offsets (in 512-byte units) keep their
        // reset defaults of 0 (TX) and 1 (RX = 512 bytes past TX), so
        // LAN_TX[0] context lives at byte 0 inside the HMC data page
        // and LAN_RX[0] context lives at byte 512.
        let pf = self.pf_id;
        self.write_reg(pf_glhmc_reg(REG_GLHMC_LANTXBASE, pf), 0);
        self.write_reg(pf_glhmc_reg(REG_GLHMC_LANTXCNT, pf), 1);
        // 1 TX queue context = 128 bytes → 0.25 of a 512-byte unit, so
        // the next aligned slot is +1 unit = +512 bytes.
        self.write_reg(pf_glhmc_reg(REG_GLHMC_LANRXBASE, pf), 1);
        self.write_reg(pf_glhmc_reg(REG_GLHMC_LANRXCNT, pf), 1);
        // FCoE: not used. Setting count = 0 keeps these blocks empty
        // and tells the HMC sizing logic we don't need backing for
        // them.
        self.write_reg(pf_glhmc_reg(REG_GLHMC_FCOEDDPBASE, pf), 0);
        self.write_reg(pf_glhmc_reg(REG_GLHMC_FCOEDDPCNT, pf), 0);
        self.write_reg(pf_glhmc_reg(REG_GLHMC_FCOEFBASE, pf), 0);
        self.write_reg(pf_glhmc_reg(REG_GLHMC_FCOEFCNT, pf), 0);

        // Verify the PDT physical address is 4K-aligned — the SDDATALOW
        // PDT-base field occupies bits[31:12], so any low bit set would
        // corrupt the (immediately adjacent) PMSDBPCOUNT/PAGED/VALID
        // bits. DriverStorage is `repr(C, align(4096))` and `hmc_pdt`
        // starts at offset 0 of its 4K-sized array, so this should hold
        // by construction; assert it loudly if not.
        if pdt_phys & 0xFFF != 0 {
            let _ = writeln!(
                Serial,
                "i40e: PDT phys 0x{:016x} is NOT 4K-aligned — HMC programming would corrupt SDDATALOW",
                pdt_phys
            );
            return Err(NicError::PhysTranslate);
        }
        if data_phys & 0xFFF != 0 {
            let _ = writeln!(
                Serial,
                "i40e: HMC data phys 0x{:016x} is NOT 4K-aligned",
                data_phys
            );
            return Err(NicError::PhysTranslate);
        }

        // Issue the SD-write command.
        //   SDDATAHIGH = bits[63:32] of PDT phys.
        //   SDDATALOW  = bits[31:12] of PDT phys in bits[31:12] of the
        //                register, PMSDBPCOUNT in bits[11:2], PAGED type
        //                in bit 1, VALID in bit 0.
        //
        // PMSDBPCOUNT must be the full 512 backing-page window for a
        // paged SD. The X710 firmware silently rejects smaller values
        // on this card — PMSDWR stays latched, ERRORINFO stays 0.
        // We previously used 1 here, which is why HMC SD programming
        // timed out before the current Cherry Server run.
        const PMSDBPCOUNT: u32 = 512;
        let lo = ((pdt_phys & 0xFFFF_F000) as u32)
            | (PMSDBPCOUNT << 2)
            | PFHMC_SDDATALOW_PAGED
            | PFHMC_SDDATALOW_VALID;
        let hi = (pdt_phys >> 32) as u32;
        let _ = writeln!(
            Serial,
            "i40e: HMC SD[0] programming — PDT phys=0x{:016x}, data phys=0x{:016x}",
            pdt_phys, data_phys
        );
        let _ = writeln!(
            Serial,
            "i40e: HMC SDDATAHIGH=0x{:08x} SDDATALOW=0x{:08x} (PMSDBPCOUNT={}, PAGED, VALID)",
            hi, lo, PMSDBPCOUNT
        );
        self.write_reg(REG_PFHMC_SDDATAHIGH, hi);
        self.write_reg(REG_PFHMC_SDDATALOW, lo);
        compiler_fence(Ordering::Release);
        // PMSDIDX = 0 (we're writing SD[0]); PMSDWR = 1.
        let cmd = 0u32 | PFHMC_SDCMD_PMSDWR;
        self.write_reg(REG_PFHMC_SDCMD, cmd);

        // On the X710 FW 20.9 card in the EPYC server, SDCMD.PMSDWR
        // reads back as a posted/latching command bit: it stays set
        // while ERRORINFO/ERRORDATA remain zero and subsequent queue
        // context programming is the real acceptance test. Treat a
        // non-zero PFHMC_ERROR* as failure; otherwise continue and let
        // queue enable surface any real HMC problem.
        for _ in 0..10_000 {
            core::hint::spin_loop();
        }
        let sdcmd = self.read_reg(REG_PFHMC_SDCMD);
        let err_info = self.read_reg(REG_PFHMC_ERRORINFO);
        let err_data = self.read_reg(REG_PFHMC_ERRORDATA);
        let lantx_base = self.read_reg(pf_glhmc_reg(REG_GLHMC_LANTXBASE, pf));
        let lantx_cnt = self.read_reg(pf_glhmc_reg(REG_GLHMC_LANTXCNT, pf));
        let lanrx_base = self.read_reg(pf_glhmc_reg(REG_GLHMC_LANRXBASE, pf));
        let lanrx_cnt = self.read_reg(pf_glhmc_reg(REG_GLHMC_LANRXCNT, pf));
        let _ = writeln!(
            Serial,
            "i40e: HMC SD[0] posted — PDT @0x{:016x} → data @0x{:016x} (PF {}, SDCMD=0x{:08x})",
            pdt_phys, data_phys, pf, sdcmd
        );
        let _ = writeln!(
            Serial,
            "i40e: HMC validation — ERRORINFO=0x{:08x} ERRORDATA=0x{:08x} LANTXBASE={} LANTXCNT={} LANRXBASE={} LANRXCNT={}",
            err_info, err_data, lantx_base, lantx_cnt, lanrx_base, lanrx_cnt
        );
        if err_info == 0 && err_data == 0 {
            return Ok(());
        }
        let _ = writeln!(
            Serial,
            "i40e: HMC SD programming reported error (SDCMD=0x{:08x}, ERRORINFO=0x{:08x}, ERRORDATA=0x{:08x})",
            sdcmd, err_info, err_data
        );
        Err(NicError::HmcTimeout)
    }

    unsafe fn init_tx_ring(&mut self) -> Result<(), NicError> {
        let ring_ptr = core::ptr::addr_of_mut!(STORAGE.tx_ring) as *mut TxDesc;
        for i in 0..TX_DESC_COUNT {
            write_volatile(ring_ptr.add(i), EMPTY_TX_DESC);
        }
        compiler_fence(Ordering::Release);
        self.tx_cursor = 0;
        // QTX_HEAD / QTX_TAIL are zeroed by enable_queues via the
        // disable-then-enable handshake.
        Ok(())
    }

    unsafe fn init_rx_ring(&mut self) -> Result<(), NicError> {
        let ring_ptr = core::ptr::addr_of_mut!(STORAGE.rx_ring) as *mut RxDesc;
        for i in 0..RX_DESC_COUNT {
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.rx_buffers[i]) as *mut u8;
            let buf_phys = memory::virt_to_phys(buf_ptr as u64).ok_or(NicError::PhysTranslate)?;
            let desc = RxDesc {
                qw0: buf_phys,
                qw1: 0,
                qw2: 0,
                qw3: 0,
            };
            write_volatile(ring_ptr.add(i), desc);
        }
        compiler_fence(Ordering::Release);
        self.rx_cursor = 0;
        Ok(())
    }

    /// Write the 128-byte LAN_TX[0] context into HMC backing.
    ///
    /// Bit layout is the Intel 700-series LAN TX HMC context packing,
    /// cross-checked against public driver register tables. Absolute LSBs:
    ///
    ///     head         13 bits  @ 0
    ///     new_context   1 bit   @ 30   — must be 1 on install
    ///     base         57 bits  @ 32   — ring phys >> 7 (128-B units)
    ///     fc_ena        1 bit   @ 89
    ///     timesync_ena  1 bit   @ 90
    ///     fd_ena        1 bit   @ 91
    ///     alt_vlan_ena  1 bit   @ 92
    ///     cpuid         8 bits  @ 96
    ///     thead_wb     13 bits  @ 128
    ///     head_wb_ena   1 bit   @ 160
    ///     qlen         13 bits  @ 161
    ///     tphrdesc_ena  1 bit   @ 174
    ///     tphrpacket    1 bit   @ 175
    ///     tphwdesc_ena  1 bit   @ 176
    ///     head_wb_addr 64 bits  @ 192
    ///     crc          32 bits  @ 896
    ///     rdylist      10 bits  @ 980
    ///     rdylist_act   1 bit   @ 990
    ///
    /// For our polling, single-queue, PF-only setup we set
    /// NEW_CONTEXT, BASE, QLEN, and RDYLIST. Everything else stays 0.
    unsafe fn write_lan_tx_context(&mut self) -> Result<(), NicError> {
        let ring_ptr = core::ptr::addr_of_mut!(STORAGE.tx_ring) as *mut TxDesc;
        let ring_phys = memory::virt_to_phys(ring_ptr as u64).ok_or(NicError::PhysTranslate)?;
        let base_units = ring_phys >> 7;

        // 128 bytes = 16 u64s. Index = absolute_bit / 64.
        let mut ctx = [0u64; 16];

        // NEW_CONTEXT @ bit 30 (u64[0] bit 30).
        ctx[0] |= 1u64 << 30;
        // BASE @ bit 32, 57 bits wide → straddles u64[0]/u64[1].
        //   u64[0] bits [63:32]   = base_units[31:0]
        //   u64[1] bits [24:0]    = base_units[56:32]
        ctx[0] |= (base_units & 0xFFFF_FFFF) << 32;
        ctx[1] |= (base_units >> 32) & ((1u64 << 25) - 1);
        // QLEN @ bit 161, 13 bits → u64[2] bit (161-128) = 33.
        ctx[2] |= (TX_DESC_COUNT as u64) << 33;
        // RDYLIST @ bit 980, 10 bits → u64[15] bit (980-960) = 20.
        ctx[15] |= ((self.tx_rdylist as u64) & 0x03FF) << 20;

        let data_ptr = core::ptr::addr_of_mut!(STORAGE.hmc_data) as *mut u8;
        let ctx_ptr = data_ptr.add(LANTX_CONTEXT_OFFSET) as *mut u64;
        for i in 0..16 {
            write_volatile(ctx_ptr.add(i), ctx[i]);
        }
        compiler_fence(Ordering::Release);

        let _ = writeln!(
            Serial,
            "i40e: LAN_TX[0] context written (ring @0x{:016x}, qlen={}, rdylist={})",
            ring_phys, TX_DESC_COUNT, self.tx_rdylist
        );
        Ok(())
    }

    /// Write the 32-byte LAN_RX[0] context into HMC backing.
    ///
    /// Bit layout is the Intel 700-series LAN RX HMC context packing.
    /// Absolute LSBs:
    ///
    ///     head         13 bits  @ 0
    ///     cpuid         8 bits  @ 13
    ///     base         57 bits  @ 32   — ring phys >> 7
    ///     qlen         13 bits  @ 89
    ///     dbuff         7 bits  @ 102  — buffer size / 128
    ///     hbuff         5 bits  @ 109
    ///     dtype         2 bits  @ 114
    ///     dsize         1 bit   @ 116  — 1 = 32-byte descriptors
    ///     crcstrip      1 bit   @ 117
    ///     fc_ena        1 bit   @ 118
    ///     l2tsel        1 bit   @ 119
    ///     hsplit_0      4 bits  @ 120
    ///     hsplit_1      2 bits  @ 124
    ///     showiv        1 bit   @ 127
    ///     rxmax        14 bits  @ 174
    ///     tphrdesc_ena  1 bit   @ 193
    ///     tphwdesc_ena  1 bit   @ 194
    ///     tphdata_ena   1 bit   @ 195
    ///     tphhead_ena   1 bit   @ 196
    ///     lrxqthresh    3 bits  @ 198
    ///     prefena       1 bit   @ 201
    ///
    /// For our polling, single-queue setup we set BASE, QLEN, DBUFF,
    /// DSIZE, CRCSTRIP, RXMAX, PREFENA. Everything else stays 0.
    unsafe fn write_lan_rx_context(&mut self) -> Result<(), NicError> {
        let ring_ptr = core::ptr::addr_of_mut!(STORAGE.rx_ring) as *mut RxDesc;
        let ring_phys = memory::virt_to_phys(ring_ptr as u64).ok_or(NicError::PhysTranslate)?;
        let base_units = ring_phys >> 7;
        let dbuff_units = (RX_BUFFER_SIZE / 128) as u64; // = 16 for 2 KiB
        let rxmax_bytes = RX_BUFFER_SIZE as u64;

        // 32 bytes = 4 u64s.
        let mut ctx = [0u64; 4];

        // BASE @ bit 32, 57 bits → straddles u64[0]/u64[1].
        //   u64[0] bits [63:32]  = base_units[31:0]
        //   u64[1] bits [24:0]   = base_units[56:32]
        ctx[0] |= (base_units & 0xFFFF_FFFF) << 32;
        ctx[1] |= (base_units >> 32) & ((1u64 << 25) - 1);
        // QLEN @ bit 89 = u64[1] bit 25 (13 bits).
        ctx[1] |= (RX_DESC_COUNT as u64) << 25;
        // DBUFF @ bit 102 = u64[1] bit 38 (7 bits).
        ctx[1] |= (dbuff_units & 0x7F) << 38;
        // DSIZE @ bit 116 = u64[1] bit 52.
        ctx[1] |= 1u64 << 52;
        // CRCSTRIP @ bit 117 = u64[1] bit 53.
        ctx[1] |= 1u64 << 53;
        // RXMAX @ bit 174 = u64[2] bit 46 (14 bits).
        ctx[2] |= (rxmax_bytes & ((1u64 << 14) - 1)) << 46;
        // PREFENA @ bit 201 = u64[3] bit 9. Intel/Linux notes this
        // normally must be set during RX queue init.
        ctx[3] |= 1u64 << 9;

        let data_ptr = core::ptr::addr_of_mut!(STORAGE.hmc_data) as *mut u8;
        let ctx_ptr = data_ptr.add(LANRX_CONTEXT_OFFSET) as *mut u64;
        for i in 0..4 {
            write_volatile(ctx_ptr.add(i), ctx[i]);
        }
        compiler_fence(Ordering::Release);

        let _ = writeln!(
            Serial,
            "i40e: LAN_RX[0] context written (ring @0x{:016x}, qlen={}, dbuff={}B)",
            ring_phys, RX_DESC_COUNT, RX_BUFFER_SIZE
        );
        Ok(())
    }

    /// Enable TX + RX queues. Hardware sequence:
    ///
    /// 1. Configure QTX_CTL with PF queue type + this PF's id.
    /// 2. Mask per-queue interrupts (polling-only).
    /// 3. Clear GLLAN_TXPRE_QDIS for our queue (no-op on first bring-up
    ///    after PF reset; matches the kernel's idempotent path).
    /// 4. Zero QTX_HEAD / QTX_TAIL / QRX_TAIL.
    /// 5. Write QTX_ENA.QENA_REQ + QRX_ENA.QENA_REQ.
    /// 6. Poll until QENA_STAT mirrors QENA_REQ on both queues.
    /// 7. Set QRX_TAIL = RX_DESC_COUNT - 1 to hand the full ring to HW.
    unsafe fn enable_queues(&mut self) -> Result<(), NicError> {
        let q = self.queue_id as usize;
        let pf = self.pf_id;

        // Step 1 — bind queue to this PF.
        let qtx_ctl = QTX_CTL_PF_QUEUE | ((pf as u32) << 2);
        self.write_reg(REG_QTX_CTL_BASE + q * 4, qtx_ctl);

        // Step 2 — interrupts off.
        self.write_reg(REG_QINT_TQCTL_BASE + q * 4, 0);
        self.write_reg(REG_QINT_RQCTL_BASE + q * 4, 0);

        // Step 3 — clear TXPRE_QDIS for our queue. Register layout:
        //   bits[10:0]  QINDX  (queue index inside this 128-queue chunk)
        //   bit 30      SET_QDIS    — write to disable pre-fetch
        //   bit 31      CLEAR_QDIS  — write to re-enable pre-fetch
        // The chunk is selected by GLLAN_TXPRE_QDIS(q / 128).
        let txpre_idx = q / 128;
        let txpre_off = REG_GLLAN_TXPRE_QDIS_BASE + txpre_idx * 4;
        let qindx = (q % 128) as u32;
        self.write_reg(txpre_off, qindx | (1u32 << 31));

        // Step 4 — reset head/tail.
        self.write_reg(REG_QTX_HEAD_BASE + q * 4, 0);
        self.write_reg(REG_QTX_TAIL_BASE + q * 4, 0);
        self.write_reg(REG_QRX_TAIL_BASE + q * 4, 0);

        // Step 5 — request enable on both queues.
        self.write_reg(REG_QRX_ENA_BASE + q * 4, QENA_REQ);
        self.write_reg(REG_QTX_ENA_BASE + q * 4, QENA_REQ);

        // Step 6 — wait for QENA_STAT on both.
        let mut rx_ok = false;
        let mut tx_ok = false;
        for _ in 0..1_000_000 {
            if !rx_ok && (self.read_reg(REG_QRX_ENA_BASE + q * 4) & QENA_STAT) != 0 {
                rx_ok = true;
            }
            if !tx_ok && (self.read_reg(REG_QTX_ENA_BASE + q * 4) & QENA_STAT) != 0 {
                tx_ok = true;
            }
            if rx_ok && tx_ok {
                break;
            }
            core::hint::spin_loop();
        }
        if !rx_ok || !tx_ok {
            let rx_state = self.read_reg(REG_QRX_ENA_BASE + q * 4);
            let tx_state = self.read_reg(REG_QTX_ENA_BASE + q * 4);
            let _ = writeln!(
                Serial,
                "i40e: queue enable TIMED OUT (q={}, QRX_ENA=0x{:08x}, QTX_ENA=0x{:08x})",
                q, rx_state, tx_state
            );
            return Err(NicError::QueueEnableTimeout);
        }

        // Step 7 — hand the pre-populated RX ring to hardware. Tail
        // sits one slot behind head, advertising RX_DESC_COUNT - 1
        // empty buffers ready for incoming frames.
        self.write_reg(REG_QRX_TAIL_BASE + q * 4, (RX_DESC_COUNT - 1) as u32);

        Ok(())
    }

    /// Send one Ethernet frame. Drops silently if the data path is not
    /// ready (e.g. bring-up failed) or the frame doesn't fit a slot.
    pub fn transmit(&mut self, frame: &[u8]) {
        if !self.ready {
            if !DATA_PATH_WARNED.swap(true, Ordering::Relaxed) {
                let _ = writeln!(
                    Serial,
                    "i40e: transmit() called before data path ready (frame_len={}) — dropping",
                    frame.len()
                );
            }
            return;
        }
        let len = frame.len().min(TX_BUFFER_SIZE);
        unsafe {
            let slot = self.tx_cursor;
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.tx_buffers[slot]) as *mut u8;
            core::ptr::copy_nonoverlapping(frame.as_ptr(), buf_ptr, len);
            let buf_phys = match memory::virt_to_phys(buf_ptr as u64) {
                Some(p) => p,
                None => return,
            };

            let cmd_type: u64 = 0 // DTYPE = data
                | TX_DESC_CMD_EOP
                | TX_DESC_CMD_RS
                | TX_DESC_CMD_ICRC
                | ((len as u64) << TX_DESC_BSZ_SHIFT);

            let desc_ptr = core::ptr::addr_of_mut!(STORAGE.tx_ring[slot]);
            write_volatile(
                desc_ptr,
                TxDesc {
                    buffer_addr: buf_phys,
                    cmd_type_offset_bsz: cmd_type,
                },
            );
            compiler_fence(Ordering::Release);

            self.tx_cursor = (slot + 1) % TX_DESC_COUNT;
            self.write_reg(
                REG_QTX_TAIL_BASE + (self.queue_id as usize) * 4,
                self.tx_cursor as u32,
            );
        }
    }

    /// Poll the RX ring for one frame.
    pub fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        if !self.ready {
            return None;
        }
        unsafe {
            let slot = self.rx_cursor;
            let desc_ptr = core::ptr::addr_of_mut!(STORAGE.rx_ring[slot]);
            let qw1 = read_volatile(&(*desc_ptr).qw1);
            if qw1 & RX_DESC_STATUS_DD == 0 {
                return None;
            }
            let length = ((qw1 >> RX_DESC_PKT_LEN_SHIFT) & RX_DESC_PKT_LEN_MASK) as usize;
            let n = length.min(out.len()).min(RX_BUFFER_SIZE);

            if (qw1 & RX_DESC_STATUS_EOP) != 0 && n > 0 {
                let buf_ptr = core::ptr::addr_of!(STORAGE.rx_buffers[slot]) as *const u8;
                core::ptr::copy_nonoverlapping(buf_ptr, out.as_mut_ptr(), n);
            }

            // Recycle: re-write the descriptor with the same buffer
            // physical address and DD cleared, so hardware reuses it.
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.rx_buffers[slot]) as *mut u8;
            let buf_phys = match memory::virt_to_phys(buf_ptr as u64) {
                Some(p) => p,
                None => return None,
            };
            write_volatile(
                desc_ptr,
                RxDesc {
                    qw0: buf_phys,
                    qw1: 0,
                    qw2: 0,
                    qw3: 0,
                },
            );
            compiler_fence(Ordering::Release);

            let old_slot = slot;
            self.rx_cursor = (slot + 1) % RX_DESC_COUNT;
            // Hand the just-recycled slot back to hardware.
            self.write_reg(
                REG_QRX_TAIL_BASE + (self.queue_id as usize) * 4,
                old_slot as u32,
            );

            if n > 0 {
                Some(n)
            } else {
                None
            }
        }
    }
}
