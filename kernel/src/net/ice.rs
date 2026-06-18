// SPDX-License-Identifier: AGPL-3.0-or-later
//! Intel 800-series ("ice", E810 family) ethernet driver — minimal,
//! polling-only.
//!
//! Target: Intel Ethernet Controller E810-XXV (PCI 0x8086:0x159b plus
//! the rest of the 0x1591..0x159B SKU range). The new Cherry Server
//! batch ships these cards in place of the older X710 (i40e) — same
//! PCI slot, very different programming model.
//!
//! ## What changed vs. i40e
//!
//! Cross-checked against `drivers/net/ethernet/intel/ice/` in Linux:
//!
//! * `PFGEN_CTRL` lives at **0x00091000**, not the i40e value 0x00092400.
//! * The port MAC `PRTGL_SAL`/`PRTGL_SAH` MMIO is **gone**. FW owns the
//!   MAC; the only way to learn it is **AQ opcode 0x0107 Manage MAC
//!   Read** (indirect — response buffer carries one entry per port).
//! * No HMC (`PFHMC_*`) and no `GLLAN_TXPRE_QDIS` handshake. TX queue
//!   context is installed via **AQ 0x0C30 Add TX Queues** (one
//!   indirect command per ring); RX queue context is written into the
//!   per-queue MMIO context window `QRX_CONTEXT(q, i)` at base
//!   `0x00280000`.
//! * RX descriptor format is the **32-byte flex descriptor (RXDID=2)**,
//!   selected per queue via `QRXFLXP_CNTXT(q)` at `0x00480000 + q*4`.
//! * RX enable is a `QRX_CTRL.QENA_REQ` → `QENA_STAT` poll on a
//!   per-queue register at `0x00120000`. TX has no per-queue enable —
//!   `Add TX Queues` both creates and arms the ring.
//!
//! Same admin queue MMIO base layout as i40e (`PF_FW_ATQ*` / `ARQ*` at
//! `0x00080000`) and same 32-byte AQ descriptor with identical flag
//! bit positions (`DD=0`, `CMP=1`, `ERR=2`, `RD=10`, `BUF=12`, `SI=13`).
//!
//! ## Bring-up sequence
//!
//! 1. PCI bind against the E810 family device IDs.
//! 2. 64-bit BAR0 map (via `pcie::read_bar` + `memory::map_mmio`).
//! 3. PF software reset (`PFGEN_CTRL.PFSWR`) + reset-done poll on
//!    `GLGEN_RSTAT` and `GLNVM_ULD`.
//! 4. Admin Queue (ASQ + ARQ) ring allocation and register programming.
//! 5. AQ `Get Version` (0x0001) — proves the AQ ↔ firmware link is alive.
//! 6. AQ `Clear PXE Mode` (0x0110) — best-effort; firmware-version
//!    dependent and may report "already cleared".
//! 7. AQ `Manage MAC Read` (0x0107) — port MAC discovery.
//! 8. AQ `Get Switch Config` (0x0200) — best-effort diagnostic.
//! 9. AQ `Get Link Status` (0x0607) — best-effort diagnostic.
//!
//! Steps 1..7 always succeed on a healthy card (and are how we know we
//! have the right MAC at the IP layer). The remaining data-path bring-
//! up (RX queue context + `QRX_CTRL` enable + AQ `Add TX Queues` with
//! scheduler topology) is not yet wired — `transmit` / `receive` go
//! through a drop-and-warn fallback, exactly like i40e's "data path
//! not ready" branch. The kernel still comes up on the network from
//! the operator's point of view: stack online, correct MAC reported,
//! shells reachable as soon as we add the data-path code path.

#![allow(dead_code)]

use core::fmt::Write;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{compiler_fence, AtomicBool, Ordering};

use crate::arch::serial::Serial;
use crate::arch::x86_64::pcie::{self, PciDevice, PciScan};
use crate::memory;

/// Intel's PCI SIG vendor ID.
pub const VENDOR_INTEL: u16 = 0x8086;

/// E810 / 800-series device IDs. Cross-checked against
/// `drivers/net/ethernet/intel/ice/ice_devids.h`. The Cherry Server
/// batch ships the E810-XXV SFP28 variant (0x159B).
pub const SUPPORTED_DEVICES: &[u16] = &[
    0x1591, // E810-C backplane
    0x1592, // E810-C QSFP (CQDA1/CQDA2 / E810-C-Q1/Q2)
    0x1593, // E810-C SFP
    0x1599, // E810-XXV backplane
    0x159A, // E810-XXV QSFP (XXVDA4/XXVDA4T)
    0x159B, // E810-XXV SFP (XXVDA2 / E810-XXV-DA-OCP) ← Cherry batch
];

// ── Register offsets (32-bit MMIO against BAR0) ──────────────────────

// Reset path. PFGEN_CTRL moved between i40e and ice (don't reuse the
// 0x00092400 value from kernel/src/net/i40e.rs).
const REG_PFGEN_CTRL: usize = 0x00091000;
const PFGEN_CTRL_PFSWR: u32 = 1 << 0;

const REG_GLGEN_RSTAT: usize = 0x000B8188;
const GLGEN_RSTAT_DEVSTATE_MASK: u32 = 0x3;

const REG_GLNVM_ULD: usize = 0x000B6008;
const GLNVM_ULD_PCIER_DONE: u32 = 1 << 0;
const GLNVM_ULD_CORER_DONE: u32 = 1 << 3;
const GLNVM_ULD_GLOBR_DONE: u32 = 1 << 4;
const GLNVM_ULD_POR_DONE: u32 = 1 << 5;

// Admin Queue — Send (ASQ) + Receive (ARQ). Same base layout as i40e.
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

// Per-queue RX registers.
const REG_QRX_CTRL_BASE: usize = 0x00120000;
const REG_QRX_TAIL_BASE: usize = 0x00290000;
const REG_QRX_CONTEXT_BASE: usize = 0x00280000;
const REG_QRXFLXP_CNTXT_BASE: usize = 0x00480000;
const REG_QTX_COMM_DBELL_BASE: usize = 0x002C0000;
const REG_QTX_COMM_HEAD_BASE: usize = 0x000E0000;

// PXE mode latch — clearable both via AQ Clear PXE Mode and by writing
// 0 directly to this register. Linux does both; we follow suit.
const REG_GLLAN_RCTL_0: usize = 0x002941F8;

#[inline(always)]
unsafe fn write_le32(ptr: *mut u8, value: u32) {
    ptr.add(0).write(value as u8);
    ptr.add(1).write((value >> 8) as u8);
    ptr.add(2).write((value >> 16) as u8);
    ptr.add(3).write((value >> 24) as u8);
}

const QRX_CTRL_QENA_REQ: u32 = 1 << 0;
const QRX_CTRL_QENA_STAT: u32 = 1 << 2;

// QRXFLXP_CNTXT — selects the RX descriptor builder profile for one
// queue. The driver picks RXDID=2 (32-byte flex NIC profile) at
// priority 3 (highest). PRIO is a 3-bit field at bit 8.
const QRXFLXP_CNTXT_RXDID_IDX_SHIFT: u32 = 0;
const QRXFLXP_CNTXT_RXDID_PRIO_SHIFT: u32 = 8;
const RXDID_FLEX_NIC: u32 = 2;
const RXDID_DEFAULT_PRIO: u32 = 3;

// QRX_CONTEXT MMIO window: each queue context is 32 bytes laid out as
// 8 dwords, but the dwords are NOT contiguous — the i-th dword for
// queue q lives at offset 0x00280000 + i*0x2000 + q*4. Mirrors
// `QRX_CONTEXT(i, idx)` in Linux ice_hw_autogen.h.
const RX_CONTEXT_DWORD_STRIDE: usize = 0x2000;
const RX_CONTEXT_DWORDS: usize = 8;

// QTX_COMM_HEAD reports the hardware consumer cursor in bits[12:0].
const QTX_COMM_HEAD_MASK: u32 = 0x1FFF;

// 32-byte flex RX descriptor (RXDID=2) writeback layout — what we
// need from the writeback:
//   QW0 bits[47:32] = pkt_len (low 14 bits = length, top 2 = gen_ctx)
//   QW1 bits[15:0]  = status_error0 (bit 0 = DD, bit 1 = EOP)
const RX_FLEX_DESC_PKT_LEN_SHIFT: u64 = 32;
const RX_FLEX_DESC_PKT_LEN_MASK: u64 = 0x3FFF;
const RX_FLEX_DESC_STATUS_DD: u64 = 1 << 0;
const RX_FLEX_DESC_STATUS_EOP: u64 = 1 << 1;

// ── AQ descriptor flags (same layout as i40e) ──────────────────────
const AQ_FLAG_DD: u16 = 1 << 0;
const AQ_FLAG_CMP: u16 = 1 << 1;
const AQ_FLAG_ERR: u16 = 1 << 2;
const AQ_FLAG_RD: u16 = 1 << 10;
const AQ_FLAG_BUF: u16 = 1 << 12;
const AQ_FLAG_SI: u16 = 1 << 13;

// ── AQ opcodes (ice / E810) ─────────────────────────────────────────
const AQ_OP_GET_VERSION: u16 = 0x0001;
const AQ_OP_MANAGE_MAC_READ: u16 = 0x0107;
const AQ_OP_CLEAR_PXE_MODE: u16 = 0x0110;
const AQ_OP_GET_SWITCH_CONFIG: u16 = 0x0200;
const AQ_OP_ADD_VSI: u16 = 0x0210;
const AQ_OP_GET_DEFAULT_TOPO: u16 = 0x0400;
const AQ_OP_QUERY_SCHED_RES: u16 = 0x0412;
const AQ_OP_GET_LINK_STATUS: u16 = 0x0607;
const AQ_OP_ADD_TX_QUEUES: u16 = 0x0C30;

// Scheduler topology parse limits. The default topology for a single
// E810 port is a few dozen nodes at most; 64 covers it with headroom
// and bounds the on-stack node table during parsing (no heap in the
// kernel). MAX_PARENT_CANDIDATES caps how many distinct parent TEIDs
// Add TX Queues will try before giving up — keeps boot time bounded
// while still letting the firmware tell us which parent it accepts.
const MAX_TOPO_NODES: usize = 64;
const MAX_PARENT_CANDIDATES: usize = 8;

// `struct ice_aqc_vsi_props` size — the indirect buffer Add VSI hands
// the firmware. 96 bytes of fields + an implicit alignment pad bring
// the on-the-wire size to 128 bytes.
const VSI_PROPS_SIZE: usize = 128;

// `struct ice_aqc_get_topo_elem` per-node record size (parent_teid +
// node_teid + 24-byte data block = 32 bytes total).
const TOPO_HEADER_SIZE: usize = 8;
const TOPO_ELEM_SIZE: usize = 32;
// Scheduler node element-types we recognise. These are the values the
// firmware writes into `elem_type` (offset 8 of each 32-byte topology
// element) per the Intel E810 Admin Queue spec for opcode 0x0400:
//   0 = UNDEFINED
//   1 = ROOT_PORT
//   2 = TC          (Traffic Class)
//   3 = SE_GENERIC  (intermediate scheduler node)
//   4 = ENTRY_POINT (queue group — Add TX Queues attaches a leaf here)
//   5 = LEAF        (queue level — created by Add TX Queues; normally
//                    not present in the Get Default Topology response)
//   6 = SE_PADDED   (padding / reserved)
//
// An earlier draft of this file had LEAF=6 and QGROUP=5, which never
// matched anything the firmware sent and forced TX bring-up to fail.
const SCHED_NODE_TYPE_ROOT_PORT: u8 = 1;
const SCHED_NODE_TYPE_TC: u8 = 2;
const SCHED_NODE_TYPE_SE_GENERIC: u8 = 3;
const SCHED_NODE_TYPE_ENTRY_POINT: u8 = 4;
const SCHED_NODE_TYPE_LEAF: u8 = 5;

// Add TX Queues — per-queue context buffer is exactly 22 bytes
// (`ICE_TXQ_CTX_SZ` in Linux). The qgroup wrapper prefixes a
// parent_teid + num_txqs + 3 pad bytes (= 8 bytes), and each per-queue
// record is txq_id(2) + rsvd(2) + q_teid(4) + ctx(22) + pad(2) + sched
// elem(20) = 52 bytes. The firmware accepts the simpler "ctx only"
// short form when valid_sections in the sched elem is 0 — we stick
// with the short form to keep the buffer small and the bring-up
// readable.
const TXQ_CTX_SIZE: usize = 22;
const TXQ_PERQ_SIZE: usize = 2 + 2 + 4 + TXQ_CTX_SIZE + 2 + 20; // 52
const TXQ_QGROUP_HEADER_SIZE: usize = 4 + 1 + 3; // 8

/// Magic byte the firmware expects in Clear PXE Mode's param0[0].
/// Same value as i40e (0x02 = "stop PXE").
const CLEAR_PXE_MAGIC: u8 = 0x2;

// Indirect-buffer layout for AQ_OP_MANAGE_MAC_READ. Mirrors
// `struct ice_aqc_manage_mac_read_resp` in
// `drivers/net/ethernet/intel/ice/ice_adminq_cmd.h`. The buffer is a
// packed array of these 8-byte entries, one per port the firmware
// owns. `addr_type` == 0 indicates the LAN MAC.
const MAC_ENTRY_LEN: usize = 8;
const MAC_ADDR_TYPE_LAN: u8 = 0;

// ── Ring sizing ────────────────────────────────────────────────────

pub const AQ_DESC_COUNT: usize = 32;
pub const AQ_BUF_SIZE: usize = 4096;

/// TX descriptor count. Power of two; not yet driven (data path
/// pending — see module-level docstring).
pub const TX_DESC_COUNT: usize = 64;
/// RX descriptor count.
pub const RX_DESC_COUNT: usize = 64;
/// Per-RX-slot packet buffer size in bytes. Same convention as i40e:
/// encoded as `size / 128` in the LAN RX context's DBUF field.
pub const RX_BUFFER_SIZE: usize = 2048;
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
// TX data descriptor (16 bytes). DTYPE bits[3:0], CMD bits[15:4],
// BSZ at bit 34 (14 bits), L2TAG1 at bit 48. CMD bit 4 = EOP, bit 5 =
// RS. NOTE: command-field bit 2 (overall bit 6) is i40e's ICRC bit but
// is RESERVED on E810 — it must stay 0 (see TX_DESC_CMD constants).

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct TxDesc {
    buffer_addr: u64,
    cmd_type_offset_bsz: u64,
}

const TX_DESC_CMD_EOP: u64 = 1 << 4;
const TX_DESC_CMD_RS: u64 = 1 << 5;
// Overall bit 6 (command-field bit 2) is ICRC on i40e but RESERVED on
// E810/ice — it must be left 0. The E810 MAC inserts the Ethernet FCS
// by default; there is no per-descriptor CRC-insert request. This
// driver previously copied the i40e ICRC bit and set a reserved bit on
// every transmitted frame.
const TX_DESC_BSZ_SHIFT: u64 = 34;
const TX_DESC_DTYPE_DONE: u64 = 0xF;

const EMPTY_TX_DESC: TxDesc = TxDesc {
    buffer_addr: 0,
    cmd_type_offset_bsz: 0,
};

// RX 32-byte flex descriptor (`union ice_32b_rx_flex_desc`). We don't
// fill any of the writeback fields ourselves; the read-side layout is
// just a packet buffer pointer in QW0.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct RxDesc {
    qw0: u64, // read: pkt_addr; writeback: status_error0 | ...
    qw1: u64,
    qw2: u64,
    qw3: u64,
}

const EMPTY_RX_DESC: RxDesc = RxDesc {
    qw0: 0,
    qw1: 0,
    qw2: 0,
    qw3: 0,
};

// ── Driver storage (BSS) ───────────────────────────────────────────

#[repr(C, align(4096))]
struct DriverStorage {
    asq_ring: [AqDesc; AQ_DESC_COUNT],
    arq_ring: [AqDesc; AQ_DESC_COUNT],
    asq_buf: [u8; AQ_BUF_SIZE],
    arq_buf: [u8; AQ_BUF_SIZE],
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
    MacReadFailed,
    RxEnableTimeout,
    TopoParseFailed,
    AddVsiFailed,
    AddTxQueueFailed,
    /// The post-bring-up TX self-test timed out: a test frame was
    /// queued, the doorbell rung, and the hardware never reported the
    /// descriptor done (no DTYPE=0xF writeback, head cursor stuck).
    /// Classic symptom of a PXE/rescue-boot latch the PFSWR pass did
    /// not clear — `bind_device` retries the full bring-up once on
    /// this error before giving up.
    TxVerifyTimeout,
    /// AQ Get Link Status reports `link_up = 0`. The MAC will come up
    /// fine and the AQ chain will accept Add VSI / Add TX Queues, but
    /// the PHY is down — no frames will ever leave the wire. Used by
    /// the dual-port selection logic in [`Ice::bind`] to skip a port
    /// whose SFP is unplugged and try the next one.
    LinkDown,
}

/// Bound ice instance.
pub struct Ice {
    mmio: *mut u8,
    mac: [u8; 6],
    /// PCI function — surfaces in log lines so the operator can tell a
    /// dual-port card's two PFs apart.
    pf_id: u8,
    /// Logical port number the firmware reported for our PF in the
    /// Manage MAC Read response. Plumbed into AQ Get Default Scheduler
    /// Topology + into the TLAN_CTX `port_num` field at TX bring-up.
    lport_num: u8,
    /// Switch ID this PF belongs to, harvested from AQ Get Switch
    /// Config (`ice_aqc_get_sw_cfg_resp_elem.swid` of the PHYS_PORT
    /// element). Linux assigns `ctxt->info.sw_id = port_info->sw_id`
    /// for every VSI add; on E810 this is typically 1, NOT 0. Used in
    /// Add VSI's switch section. `sw_id_known` gates whether we declare
    /// the SW section valid at all.
    sw_id: u8,
    /// True once `probe_switch_config` parsed a real `sw_id`. When
    /// false we DROP `ICE_AQ_VSI_PROP_SW_VALID` from Add VSI so the
    /// firmware uses its default switch instead of reading a guessed
    /// (and likely wrong) `sw_id` — declaring SW_VALID with `sw_id=0`
    /// is what made 1029.x firmware reject Add VSI.
    sw_id_known: bool,
    /// Per-PF queue index. We only ever drive queue 0 for now;
    /// E810 register offsets are global-indexed and queue 0 always
    /// belongs to this PF on a single-PF card.
    queue_id: u16,
    /// VSI number returned by AQ Add VSI. Goes into TLAN_CTX `src_vsi`.
    /// 0 means TX bring-up didn't complete and we run RX-only.
    src_vsi: u16,
    /// Parent scheduler-node TEID our TX queue was finally anchored
    /// under — set to whichever candidate the firmware *accepted* in
    /// `add_tx_queue`. Goes into (and is read back from) the Add TX
    /// Queues qgroup header. 0 until a parent is accepted.
    parent_teid: u32,
    /// Total Tx scheduler layers the firmware reports (AQ Query
    /// Scheduler Resources, 0x0412). Diagnostic only: lets the serial
    /// log show the expected VSI layer (`layers - 3`) and qgroup layer
    /// (`layers - 2`) so a failed bring-up is interpretable without a
    /// hardware probe. 0 = not yet queried / query failed.
    num_sched_layers: u8,
    /// ASQ producer cursor.
    asq_tail: usize,
    /// ARQ consumer cursor.
    arq_head: usize,
    /// TX ring producer cursor.
    tx_cursor: usize,
    /// RX ring consumer cursor.
    rx_cursor: usize,
    /// RX is initialised — `receive()` may poll the ring.
    rx_ready: bool,
    /// TX is initialised — `transmit()` may push frames. The two flags
    /// move independently because the TX-side AQ chain (Get Default
    /// Topology → Add VSI → Add TX Queues) is fragile across firmware
    /// revisions, while RX setup is a straight MMIO sequence. If TX
    /// bring-up fails we still keep RX online for diagnostics.
    tx_ready: bool,
}

unsafe impl Send for Ice {}
unsafe impl Sync for Ice {}

impl Ice {
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

    /// Bind to an E810 port. On a dual-port card (Cherry Server batch
    /// ships SFP28 XXVDA2 with two PFs at e.g. 81:00.0 and 81:00.1)
    /// only one of the ports usually has a cable plugged in — picking
    /// the wrong PF binds happily but transmits nothing because the
    /// PHY is down. We therefore iterate **every** matching device,
    /// bring each up to the link-status probe, and select the first
    /// port whose AQ Get Link Status reports `link_up=1`. If no port
    /// reports link-up, we return the most informative error we saw
    /// (`LinkDown` if any port reported PHY-down, otherwise whatever
    /// the last bring-up step refused on).
    pub fn bind(scan: &PciScan) -> Result<Self, NicError> {
        let mut last_err: Option<NicError> = None;
        let mut matched = 0u32;
        let mut link_down_seen = false;

        for dev in scan.iter().copied() {
            if dev.vendor_id != VENDOR_INTEL || !SUPPORTED_DEVICES.contains(&dev.device_id) {
                continue;
            }
            matched += 1;
            let _ = writeln!(
                Serial,
                "ice: candidate port {} of (so far) — PCI {:02x}:{:02x}.{} {:04x}:{:04x}",
                matched, dev.bus, dev.device, dev.function, dev.vendor_id, dev.device_id
            );
            match Self::bind_device(&dev) {
                Ok(nic) => {
                    let _ = writeln!(
                        Serial,
                        "ice: selected PCI {:02x}:{:02x}.{} (link up, data path verified)",
                        dev.bus, dev.device, dev.function
                    );
                    return Ok(nic);
                }
                Err(NicError::LinkDown) => {
                    link_down_seen = true;
                    last_err = Some(NicError::LinkDown);
                    // Continue: maybe the next PF on this card has a cable.
                }
                Err(e) => {
                    last_err = Some(e);
                    // Continue: a single-port AQ failure on PF1 shouldn't
                    // stop us from trying PF0.
                }
            }
        }

        if matched == 0 {
            Err(NicError::NotFound)
        } else {
            let final_err = last_err.unwrap_or(NicError::NotFound);
            let _ = writeln!(
                Serial,
                "ice: tried {} E810 port(s), none usable (link_down_seen={}); returning {:?}",
                matched, link_down_seen, final_err
            );
            Err(final_err)
        }
    }

    fn bind_device(dev: &PciDevice) -> Result<Self, NicError> {
        // Enable memory + bus-master so the device can DMA against our
        // rings.
        unsafe {
            let cmd = pcie::config_read16(dev.bus, dev.device, dev.function, 0x04);
            pcie::config_write16(dev.bus, dev.device, dev.function, 0x04, cmd | 0x0006);
        }

        let bar0 = pcie::read_bar(dev, 0).ok_or(NicError::BarMissing)?;
        // Same 2 MiB window as i40e covers the registers we touch
        // (highest is QRXFLXP_CNTXT at 0x00480000 + queue*4 — well
        // under the 2 MiB ceiling for single-queue use). Re-evaluate if
        // we ever drive more than the first 128 queues.
        const BAR0_MAP_SIZE: usize = 8 * 1024 * 1024;
        let mmio = memory::map_mmio(bar0, BAR0_MAP_SIZE).map_err(|_| NicError::MmioMapFailed)?;

        let _ = writeln!(
            Serial,
            "ice: bound device {:04x}:{:04x} at PCI {:02x}:{:02x}.{}, BAR0=0x{:016x}, MMIO=0x{:016x}",
            dev.vendor_id,
            dev.device_id,
            dev.bus,
            dev.device,
            dev.function,
            bar0,
            mmio as u64
        );

        // A rescue-system PXE boot can leave the LAN engine latched in
        // states a single PFSWR pass does not clear on every FW
        // revision. One full re-run of the bring-up chain (PFSWR → AQ →
        // Clear PXE (AQ+MMIO) → MAC → RX → TX) costs milliseconds and
        // recovers those boards. LinkDown is exempt: it gates the
        // sibling-PF scan in `Ice::bind` and a retry cannot plug a cable.
        match Self::bring_up_device(dev, mmio) {
            Err(e) if !matches!(e, NicError::LinkDown) => {
                let _ = writeln!(
                    Serial,
                    "ice: bring-up attempt 1/2 failed ({:?}) — retrying once after full PF reset",
                    e
                );
                Self::bring_up_device(dev, mmio)
            }
            other => other,
        }
    }

    /// One complete bring-up pass over an already BAR-mapped device.
    /// Every step here is re-runnable: `reset_pf` starts from PFSWR, and
    /// `init_admin_queue` reprograms the AQ rings from scratch.
    fn bring_up_device(dev: &PciDevice, mmio: *mut u8) -> Result<Self, NicError> {
        let mut nic = Ice {
            mmio,
            mac: [0; 6],
            pf_id: dev.function,
            lport_num: 0,
            sw_id: 0,
            sw_id_known: false,
            queue_id: 0,
            src_vsi: 0,
            parent_teid: 0,
            num_sched_layers: 0,
            asq_tail: 0,
            arq_head: 0,
            tx_cursor: 0,
            rx_cursor: 0,
            rx_ready: false,
            tx_ready: false,
        };
        nic.reset_pf()?;
        unsafe { nic.init_admin_queue()? };
        nic.probe_firmware()?;
        let _ = nic.clear_pxe_mode(); // best-effort
                                      // MMIO-side PXE latch clear. AQ Clear PXE Mode already
                                      // signalled the same thing to firmware, but Linux belt-and-
                                      // braces this on every bind and on some FW revisions only the
                                      // MMIO write actually clears bit 0 in time for queue setup.
        unsafe { nic.write_reg(REG_GLLAN_RCTL_0, 0) };
        // Readback diagnostic for the latch theory: bit 0 still set
        // here means this FW revision ignored both clear paths and TX
        // frames will silently die in the LAN engine.
        let rctl = unsafe { nic.read_reg(REG_GLLAN_RCTL_0) };
        let _ = writeln!(
            Serial,
            "ice: GLLAN_RCTL_0 after PXE-latch clear = 0x{:08x}",
            rctl
        );
        nic.read_port_mac_via_aq()?;
        // Diagnostic AQ probes — switch-config failure is logged, not fatal.
        let _ = nic.probe_switch_config();

        // Link-status gate: on a dual-port card with one cable plugged
        // in (Cherry batch: 81:00.0 wired, 81:00.1 unwired), every
        // step past this point will *succeed* on the unwired port —
        // AQ Add VSI, Add TX Queues, even QRX enable — and we'd bind
        // happily to the wrong PF. The PHY being down means no frames
        // ever leave the wire; the operator sees "ice: TX online" on
        // serial but a `nc <host> 2222` probe hangs forever.
        //
        // Returning Err(LinkDown) here lets `Ice::bind` try the next
        // E810 PF on the same card. If Get Link Status itself fails
        // (FW bug) we fall through with a warning and let the data-
        // path bring-up surface the real problem — better than
        // locking the operator out of a cabled port over an AQ glitch.
        match nic.probe_link_status() {
            Ok(true) => {
                let _ = writeln!(
                    Serial,
                    "ice: PF{} link UP — proceeding with data-path bring-up",
                    nic.pf_id
                );
            }
            Ok(false) => {
                let _ = writeln!(
                    Serial,
                    "ice: PF{} link DOWN — skipping this port (dual-port card?)",
                    nic.pf_id
                );
                // Quiesce the failed PF before letting `Ice::bind`
                // retry on the sibling PF. The static STORAGE struct
                // backs both candidates; if PF{N}'s firmware decides
                // to push an unsolicited ARQ event (link-state-change
                // is the obvious one — its PHY just flapped) it would
                // DMA into the same physical addresses we hand to the
                // next bind. Disabling AQLEN tears the queue down so
                // the controller stops touching our storage.
                unsafe {
                    nic.write_reg(REG_PF_ATQLEN, 0);
                    nic.write_reg(REG_PF_ARQLEN, 0);
                }
                return Err(NicError::LinkDown);
            }
            Err(e) => {
                let _ = writeln!(
                    Serial,
                    "ice: PF{} Get Link Status failed ({:?}) — proceeding without link gate",
                    nic.pf_id, e
                );
            }
        }

        // RX bring-up: a straight MMIO sequence (write rlan_ctx into
        // QRX_CONTEXT, select RXDID=2 via QRXFLXP_CNTXT, toggle
        // QRX_CTRL.QENA_REQ, bump tail). Independent of the scheduler
        // tree; should always succeed on a healthy card. FATAL: if RX
        // does not come up we have no data path and there is no point
        // pretending the NIC is bound — return Err so Stack::bind()
        // surfaces a real error to the operator instead of logging
        // "net: stack online" while quietly dropping every packet.
        if let Err(e) = unsafe { nic.bring_up_rx() } {
            let _ = writeln!(
                Serial,
                "ice: RX bring-up FAILED ({:?}) — refusing to bind (no data path)",
                e
            );
            return Err(e);
        }
        nic.rx_ready = true;
        let _ = writeln!(
            Serial,
            "ice: RX online — queue {} ready (RXDID={}, dbuff={}B, qlen={})",
            nic.queue_id, RXDID_FLEX_NIC, RX_BUFFER_SIZE, RX_DESC_COUNT
        );

        // TX bring-up goes through the firmware: Get Default Scheduler
        // Topology → Add VSI → Add TX Queues. FATAL: an RX-only bind
        // looks healthy to Stack::bind() ("net: stack online") but
        // every ARP reply, every TCP ACK, every UDP shell answer hits
        // `transmit()` → tx_ready=false → silent drop. The kernel
        // appears reachable on the wire (NIC link up, ARP probes from
        // the upstream switch see the port MAC) but actually answers
        // nothing — the worst possible failure mode for a remote
        // debug surface. Return Err so the operator sees a real
        // "ice: TX bring-up FAILED" instead of "net: stack online".
        if let Err(e) = nic.bring_up_tx() {
            let _ = writeln!(
                Serial,
                "ice: TX bring-up FAILED ({:?}) — refusing to bind (RX-only would lie about reachability)",
                e
            );
            return Err(e);
        }
        nic.tx_ready = true;
        let _ = writeln!(
            Serial,
            "ice: TX online — queue {} ready (src_vsi={}, parent_teid=0x{:08x})",
            nic.queue_id, nic.src_vsi, nic.parent_teid
        );

        // TX self-test: Add TX Queues returning OK only proves the
        // firmware accepted the queue context — it does NOT prove the
        // LAN engine moves frames. A PXE-latched engine accepts the
        // whole AQ chain and then drops every frame on the floor,
        // which from outside looks exactly like "kernel up, network
        // dead". Push one real frame through the ring and require the
        // hardware's descriptor-done writeback before declaring TX
        // usable.
        if let Err(e) = nic.verify_tx() {
            let _ = writeln!(
                Serial,
                "ice: TX verify FAILED ({:?}) — refusing to bind (queue accepts frames but hardware moves nothing)",
                e
            );
            return Err(e);
        }

        let _ = writeln!(
            Serial,
            "ice: data-path summary — rx_ready={}, tx_ready={}, TX self-test passed (TX+RX verified)",
            nic.rx_ready, nic.tx_ready
        );
        Ok(nic)
    }

    /// PF software reset + wait for device-active.
    fn reset_pf(&mut self) -> Result<(), NicError> {
        unsafe {
            let ctrl = self.read_reg(REG_PFGEN_CTRL);
            self.write_reg(REG_PFGEN_CTRL, ctrl | PFGEN_CTRL_PFSWR);

            // PFSWR is self-clearing — wait for it to drop.
            let mut cleared = false;
            for _ in 0..500_000 {
                if self.read_reg(REG_PFGEN_CTRL) & PFGEN_CTRL_PFSWR == 0 {
                    cleared = true;
                    break;
                }
                core::hint::spin_loop();
            }
            if !cleared {
                let _ = writeln!(Serial, "ice: PF reset PFSWR clear TIMED OUT");
                return Err(NicError::ResetTimeout);
            }

            // Wait for GLGEN_RSTAT.DEVSTATE == 0 ("Device Active").
            let mut active = false;
            for _ in 0..1_000_000 {
                if self.read_reg(REG_GLGEN_RSTAT) & GLGEN_RSTAT_DEVSTATE_MASK == 0 {
                    active = true;
                    break;
                }
                core::hint::spin_loop();
            }
            if !active {
                let _ = writeln!(Serial, "ice: GLGEN_RSTAT device-active TIMED OUT");
                return Err(NicError::ResetTimeout);
            }

            // Wait for ULD: at minimum PCIER + CORER + GLOBR + POR done.
            let want_uld = GLNVM_ULD_PCIER_DONE
                | GLNVM_ULD_CORER_DONE
                | GLNVM_ULD_GLOBR_DONE
                | GLNVM_ULD_POR_DONE;
            let mut uld_ok = false;
            for _ in 0..1_000_000 {
                if self.read_reg(REG_GLNVM_ULD) & want_uld == want_uld {
                    uld_ok = true;
                    break;
                }
                core::hint::spin_loop();
            }
            if !uld_ok {
                let uld = self.read_reg(REG_GLNVM_ULD);
                let _ = writeln!(
                    Serial,
                    "ice: GLNVM_ULD reset-done TIMED OUT (uld=0x{:08x}, want=0x{:08x})",
                    uld, want_uld
                );
                return Err(NicError::ResetTimeout);
            }

            let _ = writeln!(Serial, "ice: PF reset complete (PFSWR cleared, ULD OK)");
        }
        Ok(())
    }

    /// Program ASQ + ARQ base / length registers and flip the enable
    /// bit. Same registers and same per-slot ARQ buffer wiring as i40e.
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
            "ice: admin queue online (ASQ @0x{:016x}, ARQ @0x{:016x}, depth={})",
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

    /// Issue an AQ command, optionally with an external buffer. Caller
    /// sets `flags_in` (e.g. BUF | RD for a command that hands FW
    /// driver-prepared data).
    #[allow(clippy::too_many_arguments)]
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
                        "ice: AQ command 0x{:04x} TIMED OUT (flags=0x{:04x})",
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
        // ice's Get Version response layout matches i40e: addr_high =
        // (fw_major<<16 | fw_minor), addr_low = (api_major<<16 |
        // api_minor), param0 = rom_ver, param1 = build.
        let fw_major = (res.addr_high >> 16) & 0xFFFF;
        let fw_minor = res.addr_high & 0xFFFF;
        let api_major = (res.addr_low >> 16) & 0xFFFF;
        let api_minor = res.addr_low & 0xFFFF;
        let _ = writeln!(
            Serial,
            "ice: firmware FW {}.{}, API {}.{}, ROM 0x{:08x}, build 0x{:08x}",
            fw_major, fw_minor, api_major, api_minor, res.param0, res.param1
        );
        Ok(())
    }

    fn clear_pxe_mode(&mut self) -> Result<(), NicError> {
        let param0 = CLEAR_PXE_MAGIC as u32;
        match self.aq_send_simple(AQ_OP_CLEAR_PXE_MODE, param0, 0, 0, 0) {
            Ok(_) => {
                let _ = writeln!(Serial, "ice: PXE mode cleared");
                Ok(())
            }
            Err(NicError::AqRetval(rc)) => {
                // FW often returns "already cleared" on a warm reboot.
                let _ = writeln!(
                    Serial,
                    "ice: CLEAR_PXE_MODE returned rc={} (continuing — likely already cleared)",
                    rc
                );
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// AQ 0x0107 Manage MAC Read — indirect. FW writes one entry per
    /// port into the external buffer. We pick the first LAN entry.
    fn read_port_mac_via_aq(&mut self) -> Result<(), NicError> {
        unsafe {
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.asq_buf) as *mut u8;
            core::ptr::write_bytes(buf_ptr, 0, AQ_BUF_SIZE);
            let buf_phys = memory::virt_to_phys(buf_ptr as u64).ok_or(NicError::PhysTranslate)?;
            let datalen = AQ_BUF_SIZE as u16;
            let res = self.aq_send(
                AQ_OP_MANAGE_MAC_READ,
                AQ_FLAG_BUF,
                datalen,
                0,
                0,
                (buf_phys >> 32) as u32,
                (buf_phys & 0xFFFF_FFFF) as u32,
            )?;

            // param1's low byte carries the number of returned entries.
            let num_entries = (res.param1 & 0xFF) as usize;
            if num_entries == 0 {
                let _ = writeln!(
                    Serial,
                    "ice: Manage MAC Read returned 0 entries — FW did not report a port MAC"
                );
                return Err(NicError::MacReadFailed);
            }

            // Walk the entries, prefer the first LAN-typed one.
            let mut picked: Option<usize> = None;
            for i in 0..num_entries {
                let off = i * MAC_ENTRY_LEN;
                if off + MAC_ENTRY_LEN > AQ_BUF_SIZE {
                    break;
                }
                let lport = *buf_ptr.add(off);
                let addr_type = *buf_ptr.add(off + 1);
                let m0 = *buf_ptr.add(off + 2);
                let m1 = *buf_ptr.add(off + 3);
                let m2 = *buf_ptr.add(off + 4);
                let m3 = *buf_ptr.add(off + 5);
                let m4 = *buf_ptr.add(off + 6);
                let m5 = *buf_ptr.add(off + 7);
                let _ = writeln!(
                    Serial,
                    "ice: MAC read entry[{}]: lport={} type={} mac={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                    i, lport, addr_type, m0, m1, m2, m3, m4, m5
                );
                if picked.is_none() && addr_type == MAC_ADDR_TYPE_LAN {
                    picked = Some(i);
                }
            }

            // Fall back to entry 0 if nothing was explicitly typed as
            // LAN — single-port cards sometimes report addr_type=0xFF.
            let idx = picked.unwrap_or(0);
            let off = idx * MAC_ENTRY_LEN;
            self.lport_num = *buf_ptr.add(off);
            self.mac = [
                *buf_ptr.add(off + 2),
                *buf_ptr.add(off + 3),
                *buf_ptr.add(off + 4),
                *buf_ptr.add(off + 5),
                *buf_ptr.add(off + 6),
                *buf_ptr.add(off + 7),
            ];

            let _ = writeln!(
                Serial,
                "ice: port MAC = {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} (lport={}, entry={}/{}, addr_type=LAN)",
                self.mac[0], self.mac[1], self.mac[2], self.mac[3],
                self.mac[4], self.mac[5], self.lport_num, idx, num_entries
            );
        }
        Ok(())
    }

    /// Best-effort `Get Switch Config` probe. Useful diagnostic: the
    /// presence of a default VSI entry tells us the firmware finished
    /// post-reset switch reconfiguration and the AQ link is fully
    /// usable. Failure here doesn't abort bind — we only need the MAC
    /// for the current "MAC handshake" phase.
    fn probe_switch_config(&mut self) -> Result<(), NicError> {
        unsafe {
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.asq_buf) as *mut u8;
            core::ptr::write_bytes(buf_ptr, 0, AQ_BUF_SIZE);
            let buf_phys = memory::virt_to_phys(buf_ptr as u64).ok_or(NicError::PhysTranslate)?;
            let res = self.aq_send(
                AQ_OP_GET_SWITCH_CONFIG,
                AQ_FLAG_BUF,
                AQ_BUF_SIZE as u16,
                0,
                0,
                (buf_phys >> 32) as u32,
                (buf_phys & 0xFFFF_FFFF) as u32,
            );
            match res {
                Ok(_) => {
                    // The indirect response buffer is a packed array of
                    // `struct ice_aqc_get_sw_cfg_resp_elem` (6 bytes
                    // each) — there is NO header in the buffer, and the
                    // element count lives in the response DESCRIPTOR's
                    // `num_elems` field, not at buffer byte 8 (an older
                    // draft read the wrong location). Per
                    // ice_adminq_cmd.h:
                    //   [0..2] vsi_port_num  (bits 15:14 = TYPE,
                    //                         bits  9:0 = port/VSI num)
                    //   [2..4] swid          (__le16)
                    //   [4..6] pf_vf_num     (__le16)
                    // TYPE 0 = PHYS_PORT: that element's `swid` is the
                    // switch this PF lives on. Linux feeds exactly this
                    // into every VSI add (`info.sw_id = pi->sw_id`).
                    const RESP_ELEM_BYTES: usize = 6;
                    const TYPE_PHYS_PORT: u16 = 0;
                    let mut found: Option<u16> = None;
                    // Scan a bounded window for the first PHYS_PORT
                    // element carrying a real (nonzero) swid. The buffer
                    // is zeroed before the call, so empty trailing slots
                    // read back all-zero and are skipped. We deliberately
                    // require swid != 0: a parsed (or genuine) swid of 0
                    // is indistinguishable from "not populated", and
                    // declaring SW_VALID with sw_id=0 is exactly the
                    // combination that made firmware reject Add VSI — in
                    // that case we fall back to omitting SW_VALID
                    // (firmware default switch).
                    let max_scan = (AQ_BUF_SIZE / RESP_ELEM_BYTES).min(64);
                    let mut e = 0usize;
                    while e < max_scan {
                        let base = e * RESP_ELEM_BYTES;
                        let vsi_port_num =
                            u16::from_le_bytes([*buf_ptr.add(base), *buf_ptr.add(base + 1)]);
                        let swid =
                            u16::from_le_bytes([*buf_ptr.add(base + 2), *buf_ptr.add(base + 3)]);
                        let elem_type = (vsi_port_num >> 14) & 0x3;
                        if elem_type == TYPE_PHYS_PORT && swid != 0 {
                            found = Some(swid);
                            break;
                        }
                        e += 1;
                    }
                    if let Some(swid) = found {
                        self.sw_id = (swid & 0xFF) as u8;
                        self.sw_id_known = true;
                        let _ = writeln!(
                            Serial,
                            "ice: Get Switch Config — sw_id={} (phys-port element, raw swid=0x{:04x})",
                            self.sw_id, swid
                        );
                    } else {
                        let _ = writeln!(
                            Serial,
                            "ice: Get Switch Config — no nonzero phys-port swid; Add VSI will omit SW_VALID (firmware-default switch)"
                        );
                    }
                }
                Err(e) => {
                    let _ = writeln!(
                        Serial,
                        "ice: Get Switch Config FAILED ({:?}) — diagnostic only, continuing",
                        e
                    );
                }
            }
        }
        Ok(())
    }

    /// `Get Link Status` (AQ 0x0607). Returns `Ok(true)` if the
    /// firmware reports link-up, `Ok(false)` if the PHY is down (no
    /// cable, peer down, wrong SFP, etc.), and `Err(_)` if the AQ
    /// command itself fails — in that case the caller should treat
    /// link state as "unknown" rather than "down" so a buggy FW
    /// revision doesn't lock the operator out of the data path on a
    /// port that physically has a cable.
    fn probe_link_status(&mut self) -> Result<bool, NicError> {
        // param0 byte 0 = LSE (link-status-event reporting); 0 = no
        // event subscription, just one-shot read.
        let r = self.aq_send_simple(AQ_OP_GET_LINK_STATUS, 0, 0, 0, 0)?;
        // Per ice_adminq_cmd.h::struct ice_aqc_get_link_status:
        // param0 byte 2 = link_info (bit 0 = LINK_UP),
        // param0 byte 3 = link_speed (bitmap),
        // remaining fields cover an_info / ext_info / etc.
        let link_info = ((r.param0 >> 16) & 0xFF) as u8;
        let link_speed = ((r.param0 >> 24) & 0xFF) as u8;
        let link_up = (link_info & 0x01) != 0;
        let _ = writeln!(
            Serial,
            "ice: PF{} link {} (link_info=0x{:02x}, link_speed=0x{:02x})",
            self.pf_id,
            if link_up { "UP" } else { "DOWN" },
            link_info,
            link_speed
        );
        Ok(link_up)
    }

    // ── RX bring-up ─────────────────────────────────────────────────

    /// End-to-end RX bring-up. Sequence mirrors what Linux's ice driver
    /// does inside `ice_vsi_cfg_rxq()` + `ice_vsi_ctrl_one_rx_ring()`,
    /// stripped down for a single PF queue, polling-only, no headers-
    /// split, no RSS.
    unsafe fn bring_up_rx(&mut self) -> Result<(), NicError> {
        self.init_rx_ring()?;
        self.write_rx_context()?;
        self.select_rx_flex_profile();
        self.enable_rx_queue()?;
        // Hand the full ring to the firmware. Tail trails head by one
        // (= "all RX_DESC_COUNT slots are empty and HW-owned").
        let q = self.queue_id as usize;
        self.write_reg(REG_QRX_TAIL_BASE + q * 4, (RX_DESC_COUNT - 1) as u32);
        Ok(())
    }

    /// Populate every RX descriptor with the physical address of its
    /// backing buffer. RXDID=2 ignores `hdr_addr` (we don't header-
    /// split), so qw1/qw2/qw3 stay zero.
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

    /// Pack the LAN RX queue context (32 bytes = 256 bits) and shovel
    /// each dword into the E810's per-queue QRX_CONTEXT MMIO window.
    /// Bit positions copied from Linux `ice_rlan_ctx_info[]` — they
    /// happen to match the i40e LAN RX context layout one-to-one
    /// (modulo `prefena` moving from bit 201 → bit 201, same value).
    unsafe fn write_rx_context(&mut self) -> Result<(), NicError> {
        let ring_ptr = core::ptr::addr_of_mut!(STORAGE.rx_ring) as *mut RxDesc;
        let ring_phys = memory::virt_to_phys(ring_ptr as u64).ok_or(NicError::PhysTranslate)?;
        // Ring base is in 128-byte units; the ring is 16-byte * 64 =
        // 1024 bytes, lives inside a 4K-aligned static, so shifting by
        // 7 never loses bits.
        let base_units = ring_phys >> 7;
        let dbuf_units = (RX_BUFFER_SIZE / 128) as u64; // = 16 for 2 KiB

        // 32 bytes = 4 u64 words. Indexes computed as bit/64.
        let mut ctx = [0u64; 4];

        // BASE @ bit 32, 57 bits → straddles u64[0]/u64[1].
        ctx[0] |= (base_units & 0xFFFF_FFFF) << 32;
        ctx[1] |= (base_units >> 32) & ((1u64 << 25) - 1);
        // QLEN @ bit 89, 13 bits → u64[1] bit 25.
        ctx[1] |= (RX_DESC_COUNT as u64) << 25;
        // DBUF @ bit 102, 7 bits → u64[1] bit 38.
        ctx[1] |= (dbuf_units & 0x7F) << 38;
        // DSIZE @ bit 116 — set so HW emits 32-byte writeback.
        ctx[1] |= 1u64 << 52;
        // CRCSTRIP @ bit 117 — HW strips Ethernet FCS so the upper
        // stack doesn't have to.
        ctx[1] |= 1u64 << 53;
        // RXMAX @ bit 174, 14 bits → u64[2] bit 46. Limits the max
        // received packet length the queue will accept.
        let rxmax_bytes = RX_BUFFER_SIZE as u64;
        ctx[2] |= (rxmax_bytes & ((1u64 << 14) - 1)) << 46;
        // PREFENA @ bit 201 → u64[3] bit 9. Linux unconditionally
        // forces this on inside ice_write_rxq_ctx().
        ctx[3] |= 1u64 << 9;

        let q = self.queue_id as usize;
        // Walk the four 64-bit words as eight 32-bit dwords, writing
        // each to its non-contiguous QRX_CONTEXT slot. The dword index
        // moves the address by 0x2000 bytes — laying out 8 dwords
        // strided this way is exactly how Linux does it.
        let dwords: [u32; RX_CONTEXT_DWORDS] = [
            (ctx[0] & 0xFFFF_FFFF) as u32,
            (ctx[0] >> 32) as u32,
            (ctx[1] & 0xFFFF_FFFF) as u32,
            (ctx[1] >> 32) as u32,
            (ctx[2] & 0xFFFF_FFFF) as u32,
            (ctx[2] >> 32) as u32,
            (ctx[3] & 0xFFFF_FFFF) as u32,
            (ctx[3] >> 32) as u32,
        ];
        for (i, dw) in dwords.iter().enumerate() {
            let off = REG_QRX_CONTEXT_BASE + i * RX_CONTEXT_DWORD_STRIDE + q * 4;
            self.write_reg(off, *dw);
        }
        compiler_fence(Ordering::Release);

        let _ = writeln!(
            Serial,
            "ice: RX context written (q={}, ring @0x{:016x}, qlen={}, dbuf={}B)",
            q, ring_phys, RX_DESC_COUNT, RX_BUFFER_SIZE
        );
        Ok(())
    }

    /// Select the 32-byte flex NIC descriptor profile (RXDID=2) at
    /// priority 3. Without this, queue 0 sticks with the firmware's
    /// reset default (legacy 16-byte) and our 32-byte writeback parse
    /// reads garbage.
    unsafe fn select_rx_flex_profile(&self) {
        let q = self.queue_id as usize;
        let value = (RXDID_FLEX_NIC << QRXFLXP_CNTXT_RXDID_IDX_SHIFT)
            | (RXDID_DEFAULT_PRIO << QRXFLXP_CNTXT_RXDID_PRIO_SHIFT);
        self.write_reg(REG_QRXFLXP_CNTXT_BASE + q * 4, value);
        let _ = writeln!(
            Serial,
            "ice: QRXFLXP_CNTXT(q={}) = 0x{:08x} (RXDID={}, PRIO={})",
            q, value, RXDID_FLEX_NIC, RXDID_DEFAULT_PRIO
        );
    }

    /// Toggle QRX_CTRL.QENA_REQ and poll for QENA_STAT to mirror.
    /// On healthy firmware this returns in well under a millisecond;
    /// we give it 1M busy spins which on a 3 GHz host is ~few hundred
    /// microseconds of headroom.
    unsafe fn enable_rx_queue(&self) -> Result<(), NicError> {
        let q = self.queue_id as usize;
        let off = REG_QRX_CTRL_BASE + q * 4;
        let prev = self.read_reg(off);
        self.write_reg(off, prev | QRX_CTRL_QENA_REQ);
        for _ in 0..1_000_000 {
            if (self.read_reg(off) & QRX_CTRL_QENA_STAT) != 0 {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        let final_state = self.read_reg(off);
        let _ = writeln!(
            Serial,
            "ice: QRX_CTRL(q={}) enable TIMED OUT (final=0x{:08x})",
            q, final_state
        );
        Err(NicError::RxEnableTimeout)
    }

    // ── TX bring-up ─────────────────────────────────────────────────

    /// End-to-end TX bring-up. Failures past the first step are not
    /// fatal but they do leave the queue down — see the per-step logs
    /// for which AQ command refused the request.
    ///
    /// Ordering matters: the Add VSI step creates *our* VSI and its
    /// scheduler subtree. The Add TX Queues `parent_teid` must name a
    /// queue-group node that lives **inside that VSI subtree** — per the
    /// ice scheduler model `Root → TC → … → VSI → QGroup → Leaf(queue)`,
    /// where `ice_sched_get_free_qparent` returns a qgroup-layer node
    /// under the VSI node. An earlier revision parsed the topology
    /// *before* Add VSI ran, so it could only ever see the firmware's
    /// **default-VSI** ENTRY_POINT nodes — attaching our queue under a
    /// foreign VSI's qgroup is what made Add TX Queues firmware-revision-
    /// dependently reject the request (AddTxQueueFailed). We now (1) add
    /// the VSI first, (2) re-read the topology so any VSI-owned node is
    /// visible, and (3) let the firmware pick the parent it accepts out
    /// of an ordered candidate list.
    fn bring_up_tx(&mut self) -> Result<(), NicError> {
        unsafe { self.init_tx_ring()? };
        self.query_sched_resources(); // best-effort diagnostic
        self.add_vsi_for_tx()?;
        let mut candidates = [0u32; MAX_PARENT_CANDIDATES];
        let n = self.collect_parent_candidates(&mut candidates)?;
        self.add_tx_queue(&candidates[..n])?;
        Ok(())
    }

    /// AQ 0x0412 Query Scheduler Resources — best-effort. The response
    /// buffer's `sched_props` reports the number of Tx scheduler layers
    /// the firmware exposes (typically 9, sometimes 5). We don't act on
    /// it, but logging it makes a failed TX bring-up interpretable: the
    /// VSI layer is `layers - 3` and the qgroup (Add TX Queues parent)
    /// layer is `layers - 2`, so the operator can read the topology dump
    /// against the expected layout instead of guessing. Failure here is
    /// non-fatal — the candidate-retry path does not depend on it.
    fn query_sched_resources(&mut self) {
        unsafe {
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.asq_buf) as *mut u8;
            core::ptr::write_bytes(buf_ptr, 0, AQ_BUF_SIZE);
            let buf_phys = match memory::virt_to_phys(buf_ptr as u64) {
                Some(p) => p,
                None => return,
            };
            match self.aq_send(
                AQ_OP_QUERY_SCHED_RES,
                AQ_FLAG_BUF,
                AQ_BUF_SIZE as u16,
                0,
                0,
                (buf_phys >> 32) as u32,
                (buf_phys & 0xFFFF_FFFF) as u32,
            ) {
                Ok(_) => {
                    // struct ice_aqc_query_txsched_res_resp begins with
                    // __le16 sched_props.phys_levels then
                    // __le16 logical_levels. The "logical" level count is
                    // the usable scheduler depth Linux drives against.
                    let phys_levels = u16::from_le_bytes([*buf_ptr.add(0), *buf_ptr.add(1)]);
                    let logical_levels = u16::from_le_bytes([*buf_ptr.add(2), *buf_ptr.add(3)]);
                    let layers = if logical_levels != 0 {
                        logical_levels
                    } else {
                        phys_levels
                    };
                    self.num_sched_layers = (layers & 0xFF) as u8;
                    let vsi_layer = layers.saturating_sub(3);
                    let qgrp_layer = layers.saturating_sub(2);
                    let _ = writeln!(
                        Serial,
                        "ice: scheduler layers = {} (phys={}, logical={}) → expect VSI@layer {}, qgroup@layer {}",
                        layers, phys_levels, logical_levels, vsi_layer, qgrp_layer
                    );
                }
                Err(e) => {
                    let _ = writeln!(
                        Serial,
                        "ice: Query Scheduler Resources failed ({:?}) — diagnostic only, continuing",
                        e
                    );
                }
            }
        }
    }

    /// Empty TX ring slots so HW sees no stale descriptors after the
    /// queue is added.
    unsafe fn init_tx_ring(&mut self) -> Result<(), NicError> {
        let ring_ptr = core::ptr::addr_of_mut!(STORAGE.tx_ring) as *mut TxDesc;
        for i in 0..TX_DESC_COUNT {
            write_volatile(ring_ptr.add(i), EMPTY_TX_DESC);
        }
        compiler_fence(Ordering::Release);
        self.tx_cursor = 0;
        Ok(())
    }

    /// AQ 0x0400 Get Default Scheduler Topology → ordered list of
    /// candidate parent TEIDs for Add TX Queues.
    ///
    /// Run *after* Add VSI so the response reflects our VSI's subtree:
    /// the qgroup node we must anchor under lives below the VSI node,
    /// per `Root → TC → … → VSI → QGroup → Leaf`. We parse the full tree
    /// into a node table, reconstruct each node's depth by walking its
    /// parent links, dump every node to serial (so a failed bring-up is
    /// interpretable without a hardware probe), and emit candidates in
    /// the order the firmware is most likely to accept:
    ///
    ///   1. ENTRY_POINT (qgroup) nodes, deepest first
    ///   2. SE_GENERIC  (intermediate) nodes, deepest first
    ///   3. LEAF nodes, deepest first (last resort)
    ///
    /// `add_tx_queue` then tries each in turn and lets the firmware pick
    /// the one it accepts — converting what used to be a single blind
    /// guess into a self-finding, fully-logged probe.
    fn collect_parent_candidates(
        &mut self,
        out: &mut [u32; MAX_PARENT_CANDIDATES],
    ) -> Result<usize, NicError> {
        // Node table: (parent_teid, node_teid, elem_type). No heap in
        // the kernel — a fixed on-stack table bounded by MAX_TOPO_NODES.
        let mut parents = [0u32; MAX_TOPO_NODES];
        let mut teids = [0u32; MAX_TOPO_NODES];
        let mut types = [0u8; MAX_TOPO_NODES];
        let mut n_nodes = 0usize;
        let mut counts = [0u32; 7];

        unsafe {
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.asq_buf) as *mut u8;
            core::ptr::write_bytes(buf_ptr, 0, AQ_BUF_SIZE);
            let buf_phys = memory::virt_to_phys(buf_ptr as u64).ok_or(NicError::PhysTranslate)?;
            // param0 byte 0 = port_num — the lport_num cached from Manage
            // MAC Read.
            let res = self.aq_send(
                AQ_OP_GET_DEFAULT_TOPO,
                AQ_FLAG_BUF,
                AQ_BUF_SIZE as u16,
                self.lport_num as u32,
                0,
                (buf_phys >> 32) as u32,
                (buf_phys & 0xFFFF_FFFF) as u32,
            )?;

            // Response layout per `struct ice_aqc_get_topo`: byte 0 =
            // port_num (echoed), byte 1 = num_branches. (The byte-0-vs-1
            // mixup that bit the first Cherry boot is pinned by
            // topo_num_branches() + its unit test.)
            let num_branches = topo_num_branches(res.param0);
            let _ = writeln!(
                Serial,
                "ice: Get Default Topology (post-VSI) returned {} branch(es) (raw param0=0x{:08x})",
                num_branches, res.param0
            );

            // Each branch: 8-byte header (parent_teid + num_elems + rsvd)
            // then num_elems × 32-byte elements. Per element:
            //   [0..4] parent_teid   [4..8] node_teid   [8] elem_type
            // (the per-element parent_teid is what lets us reconstruct
            // depth — the earlier revision ignored it.)
            let mut off: usize = 0;
            for _ in 0..num_branches {
                if off + TOPO_HEADER_SIZE > AQ_BUF_SIZE {
                    break;
                }
                let num_elems =
                    u16::from_le_bytes([*buf_ptr.add(off + 4), *buf_ptr.add(off + 5)]) as usize;
                off += TOPO_HEADER_SIZE;
                for _ in 0..num_elems {
                    if off + TOPO_ELEM_SIZE > AQ_BUF_SIZE || n_nodes >= MAX_TOPO_NODES {
                        break;
                    }
                    let parent_teid = u32::from_le_bytes([
                        *buf_ptr.add(off),
                        *buf_ptr.add(off + 1),
                        *buf_ptr.add(off + 2),
                        *buf_ptr.add(off + 3),
                    ]);
                    let node_teid = u32::from_le_bytes([
                        *buf_ptr.add(off + 4),
                        *buf_ptr.add(off + 5),
                        *buf_ptr.add(off + 6),
                        *buf_ptr.add(off + 7),
                    ]);
                    let elem_type = *buf_ptr.add(off + 8);
                    parents[n_nodes] = parent_teid;
                    teids[n_nodes] = node_teid;
                    types[n_nodes] = elem_type;
                    if (elem_type as usize) < counts.len() {
                        counts[elem_type as usize] += 1;
                    }
                    n_nodes += 1;
                    off += TOPO_ELEM_SIZE;
                }
            }
        }

        let _ = writeln!(
            Serial,
            "ice: topology histogram — ROOT={} TC={} SE_GENERIC={} ENTRY_POINT={} LEAF={} ({} node(s) parsed)",
            counts[SCHED_NODE_TYPE_ROOT_PORT as usize],
            counts[SCHED_NODE_TYPE_TC as usize],
            counts[SCHED_NODE_TYPE_SE_GENERIC as usize],
            counts[SCHED_NODE_TYPE_ENTRY_POINT as usize],
            counts[SCHED_NODE_TYPE_LEAF as usize],
            n_nodes,
        );

        // Dump every node with its reconstructed depth so a failed
        // bring-up is interpretable from the serial log alone.
        for i in 0..n_nodes {
            let _ = writeln!(
                Serial,
                "ice:   node[{}] parent=0x{:08x} teid=0x{:08x} type={} depth={}",
                i,
                parents[i],
                teids[i],
                types[i],
                node_depth(&parents, &teids, n_nodes, i)
            );
        }

        // Order candidates by priority class, deepest-first within each
        // (pure logic — see order_parent_candidates + its unit tests).
        let count = order_parent_candidates(&parents, &teids, &types, n_nodes, out);

        if count == 0 {
            let _ = writeln!(
                Serial,
                "ice: topology yielded no usable scheduler node — TX bring-up cannot anchor queue"
            );
            return Err(NicError::TopoParseFailed);
        }

        let _ = writeln!(
            Serial,
            "ice: {} parent candidate(s) to try (deepest qgroup first): {:?}",
            count,
            &out[..count]
        );
        Ok(count)
    }

    /// AQ 0x0210 Add VSI. Hands the firmware an `ice_aqc_vsi_props`
    /// buffer with the minimum sections needed for a single-queue PF
    /// TX path, and reads back the new VSI's `vsi_num` from the
    /// descriptor response. The vsi_num drives the `src_vsi` field of
    /// the TX queue context.
    ///
    /// Section-valid flags MUST be declared for every section whose
    /// fields are filled in; E810 1029.x firmware rejects Add VSI with
    /// INVALID_PARAM if the queue-mapping section is populated but
    /// RXQ_MAP_VALID is absent, and returns a zero VSI num if SW_VALID
    /// is absent (resulting in a broken TLAN_CTX src_vsi=0).
    fn add_vsi_for_tx(&mut self) -> Result<(), NicError> {
        // Section-valid flags. Matches Intel E810 Admin Queue
        // `ICE_AQ_VSI_PROP_*` definitions in ice_adminq_cmd.h.
        // BIT(0) = SW_VALID, BIT(1) = SECURITY_VALID, BIT(6) = RXQ_MAP_VALID.
        const ICE_AQ_VSI_PROP_SW_VALID: u16 = 0x0001;
        const ICE_AQ_VSI_PROP_SECURITY_VALID: u16 = 0x0002;
        const ICE_AQ_VSI_PROP_RXQ_MAP_VALID: u16 = 0x0040;

        // Layout of `struct ice_aqc_vsi_props` (relevant fields only).
        // Offsets are bytes in the AQ indirect buffer:
        //   [0..2]   valid_sections (u16 LE)
        //   [2]      sw_id
        //   [3]      sw_flags
        //   [4]      sw_flags2
        //   [6]      sec_flags
        //   [16..20] ingress_table (u32 LE)
        //   [20..24] egress_table (u32 LE)
        //   [24..26] port_based_outer_vlan
        //   [28..30] mapping_flags (u16 LE)
        //   [30..62] q_mapping[16] (LE u16 per slot)
        //   [62..78] tc_mapping[8] (LE u16 per TC)
        //   [84..88] outer_up_table (u32 LE)
        // Offsets verified against struct ice_aqc_vsi_props in
        // drivers/net/ethernet/intel/ice/ice_adminq_cmd.h (Linux).
        const ICE_AQ_VSI_SW_FLAG_SRC_PRUNE: u8 = 1 << 7;
        const ICE_AQ_VSI_SW_FLAG_LAN_ENA: u8 = 1 << 4;
        const ICE_AQ_VSI_SEC_FLAG_ALLOW_DEST_OVRD: u8 = 1 << 0;

        unsafe {
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.asq_buf) as *mut u8;
            core::ptr::write_bytes(buf_ptr, 0, VSI_PROPS_SIZE);

            // Declare the sections we populate. SECURITY_VALID +
            // RXQ_MAP_VALID mirror the Linux PF-VSI init path
            // (ice_vsi_init sets SECURITY_VALID for PF; ice_vsi_setup_q_map
            // sets RXQ_MAP_VALID). RXQ_MAP_VALID is required by E810
            // 1029.x firmware whenever the queue-mapping fields are
            // populated; omitting it causes INVALID_PARAM.
            //
            // SW_VALID is added ONLY when we know the real switch id.
            // Linux always feeds `info.sw_id = port_info->sw_id` (the
            // PHYS_PORT swid from Get Switch Config — typically 1 on
            // E810, NOT 0). The previous revision declared SW_VALID but
            // hardcoded sw_id=0; 1029.x firmware reads the SW section,
            // sees a VSI pointed at a nonexistent switch 0, and rejects
            // Add VSI (→ AddVsiFailed). When the real swid is unknown we
            // omit SW_VALID so the firmware uses its default switch
            // instead of a wrong constant.
            let mut valid_sections = ICE_AQ_VSI_PROP_SECURITY_VALID | ICE_AQ_VSI_PROP_RXQ_MAP_VALID;
            if self.sw_id_known {
                valid_sections |= ICE_AQ_VSI_PROP_SW_VALID;
            }
            buf_ptr.add(0).write(valid_sections as u8);
            buf_ptr.add(1).write((valid_sections >> 8) as u8);

            // SW section: real sw_id from Get Switch Config (only read by
            // firmware when SW_VALID is set above), source pruning and PF
            // LAN mode — matching ice_set_dflt_vsi_ctx. Offsets per
            // ice_aqc_vsi_props: sw_id[2], sw_flags[3], sw_flags2[4].
            buf_ptr.add(2).write(self.sw_id);
            buf_ptr.add(3).write(ICE_AQ_VSI_SW_FLAG_SRC_PRUNE);
            buf_ptr.add(4).write(ICE_AQ_VSI_SW_FLAG_LAN_ENA);

            // Security section: allow destination MAC override so the
            // TX path doesn't get filtered when the upper stack emits
            // a frame sourced from the port MAC. sec_flags at byte 6.
            buf_ptr.add(6).write(ICE_AQ_VSI_SEC_FLAG_ALLOW_DEST_OVRD);

            // Queue mapping section (declared valid via RXQ_MAP_VALID).
            // CORRECT offsets per struct ice_aqc_vsi_props — the
            // mapping section starts at byte 28, NOT 32. An earlier
            // draft placed mapping_flags at 32 / q_mapping at 34 /
            // tc_mapping at 66; with RXQ_MAP_VALID now set the firmware
            // reads the real offsets, so misplaced writes left the
            // queue count zero and the VSI unusable. Contiguous mode:
            //   mapping_flags [28..30] = 0 (ICE_AQ_VSI_Q_MAP_CONTIG)
            //   q_mapping[0]  [30..32] = first absolute queue id
            //   q_mapping[1]  [32..34] = queue count (= 1)
            //   tc_mapping[0] [62..64] = (offset 0) | (log2(1)=0 << 11)
            //   remaining q_mapping/tc_mapping slots stay zero.
            buf_ptr.add(28).write(0); // mapping_flags low (contiguous)
            buf_ptr.add(29).write(0); // mapping_flags high
            buf_ptr.add(30).write(self.queue_id as u8); // q_mapping[0] low
            buf_ptr.add(31).write((self.queue_id >> 8) as u8); // q_mapping[0] high
            buf_ptr.add(32).write(1); // q_mapping[1] = queue count = 1
            buf_ptr.add(33).write(0);
            buf_ptr.add(62).write(0); // tc_mapping[0] = 0 (1 queue @ offset 0)
            buf_ptr.add(63).write(0);

            let _ = writeln!(
                Serial,
                "ice: Add VSI props valid=0x{:04x} (sw_id={} known={}), sw_flags=0x{:02x}, sw_flags2=0x{:02x}, q_mapping[0]={}, tc0=0",
                valid_sections,
                self.sw_id,
                self.sw_id_known,
                ICE_AQ_VSI_SW_FLAG_SRC_PRUNE,
                ICE_AQ_VSI_SW_FLAG_LAN_ENA,
                self.queue_id
            );

            let buf_phys = memory::virt_to_phys(buf_ptr as u64).ok_or(NicError::PhysTranslate)?;
            // Descriptor params layout for Add VSI (per Intel E810 AQ
            // spec for opcode 0x0210):
            //   bytes[0..2] of param0 = vsi_num   (response only)
            //   bytes[2..4] of param0 = cmd_flags
            //   bytes[0..2] of param1 = vf_id     (PF: leave zero)
            //   bytes[2..4] of param1 = vsi_flags (PF type)
            //
            // The VSI type belongs in param1[31:16]. The previous
            // version put it into cmd_flags, which E810 1029.x
            // firmware rejects as invalid Add VSI input.
            const ICE_AQ_VSI_TYPE_PF: u16 = 2;
            let vsi_flags = ICE_AQ_VSI_TYPE_PF as u32;
            let res = self.aq_send(
                AQ_OP_ADD_VSI,
                AQ_FLAG_BUF | AQ_FLAG_RD,
                VSI_PROPS_SIZE as u16,
                0,
                vsi_flags << 16,
                (buf_phys >> 32) as u32,
                (buf_phys & 0xFFFF_FFFF) as u32,
            );
            match res {
                Ok(r) => {
                    // param0 bits[9:0] = vsi_num. Upper bits hold
                    // ext_status; mask them off the way Linux does.
                    let vsi_num = (r.param0 & 0x3FF) as u16;
                    self.src_vsi = vsi_num;
                    let _ = writeln!(
                        Serial,
                        "ice: Add VSI ok — vsi_num={} (vsi_flags=0x{:02x})",
                        vsi_num, vsi_flags
                    );
                    Ok(())
                }
                Err(e) => {
                    // Decode the firmware AQ status so the KVM screen
                    // names the reason instead of a bare number.
                    let _ = writeln!(
                        Serial,
                        "ice: Add VSI FAILED ({:?} = {}) sw_id={} known={} valid=0x{:04x} — cannot anchor TX queue",
                        e, aq_reason(e), self.sw_id, self.sw_id_known, valid_sections,
                    );
                    Err(NicError::AddVsiFailed)
                }
            }
        }
    }

    /// AQ 0x0C30 Add TX Queues. Builds one qgroup with one queue,
    /// packs `ice_tlan_ctx` (22 bytes) for our queue, and hands the
    /// whole indirect buffer to the firmware. Linux reads back the
    /// per-queue `q_teid` from the buffer; we don't need it, but the
    /// firmware writes it regardless.
    ///
    /// `parents` is the ordered candidate list from
    /// `collect_parent_candidates`. The qgroup parent the firmware will
    /// accept must belong to *our* VSI's subtree, and which scheduler
    /// node that is depends on the firmware's layer count — so rather
    /// than commit to one guess, we try each candidate (deepest qgroup
    /// first) and let the firmware adjudicate. Each rejected attempt
    /// commits nothing, so retrying with the next parent is safe. The
    /// first acceptance wins and is recorded in `self.parent_teid`.
    fn add_tx_queue(&mut self, parents: &[u32]) -> Result<(), NicError> {
        // Pack the TLAN context once — it's identical across attempts
        // (only the qgroup header's parent_teid changes). Bit positions
        // per `ice_tlan_ctx_info[]` (verified against Linux): qlen sits
        // between adjust_prof_id @129 and quanta_prof_idx @148; tso_ena
        // @152; legacy_int @164 after tso_qnum @153. They are a silicon
        // contract — getting any wrong returns INVALID_PARAM. (A draft
        // that shifted qlen/tso_ena/legacy_int to 132/149/161 overlapped
        // those neighbours; see the regression test below.)
        let ring_phys = unsafe {
            let ring_ptr = core::ptr::addr_of_mut!(STORAGE.tx_ring) as *mut TxDesc;
            memory::virt_to_phys(ring_ptr as u64).ok_or(NicError::PhysTranslate)?
        };
        let base_units = ring_phys >> 7; // 128-byte units
        let mut ctx = [0u8; TXQ_CTX_SIZE];
        pack_tlan_bits(&mut ctx, base_units, 0, 57); // base
        pack_tlan_bits(&mut ctx, self.lport_num as u64, 57, 3); // port_num
        pack_tlan_bits(&mut ctx, self.pf_id as u64, 65, 3); // pf_num
        pack_tlan_bits(&mut ctx, 2, 78, 2); // vmvf_type = PF
        pack_tlan_bits(&mut ctx, self.src_vsi as u64, 80, 10); // src_vsi
        pack_tlan_bits(&mut ctx, TX_DESC_COUNT as u64, 135, 13); // qlen
                                                                 // tso_ena: set to 1 (= legacy TX mode in the AQ semantics).
                                                                 // Some E810 FW revisions reject Add TX Queues with ENOTSUPP when
                                                                 // this bit is left zero.
        pack_tlan_bits(&mut ctx, 1, 152, 1); // tso_ena
                                             // legacy_int: polling driver, no MSI-X.
        pack_tlan_bits(&mut ctx, 1, 164, 1); // legacy_int

        let buf_len = TXQ_QGROUP_HEADER_SIZE + TXQ_PERQ_SIZE;
        let mut last_err = NicError::AddTxQueueFailed;

        for (attempt, &parent_teid) in parents.iter().enumerate() {
            let res = unsafe {
                let buf_ptr = core::ptr::addr_of_mut!(STORAGE.asq_buf) as *mut u8;
                core::ptr::write_bytes(buf_ptr, 0, AQ_BUF_SIZE);

                // qgroup header: parent_teid (4) + num_txqs (1) + 3 pad.
                buf_ptr.add(0).write(parent_teid as u8);
                buf_ptr.add(1).write((parent_teid >> 8) as u8);
                buf_ptr.add(2).write((parent_teid >> 16) as u8);
                buf_ptr.add(3).write((parent_teid >> 24) as u8);
                buf_ptr.add(4).write(1); // num_txqs

                // per-queue record at offset 8:
                //   [0..2]   txq_id (= self.queue_id)
                //   [2..4]   reserved
                //   [4..8]   q_teid (firmware fills on completion)
                //   [8..30]  TLAN_CTX (22 bytes)
                //   [30..32] reserved
                //   [32..52] scheduler element data — left zero
                let perq = buf_ptr.add(TXQ_QGROUP_HEADER_SIZE);
                perq.add(0).write(self.queue_id as u8);
                perq.add(1).write((self.queue_id >> 8) as u8);
                for (i, b) in ctx.iter().enumerate() {
                    perq.add(8 + i).write(*b);
                }

                let buf_phys =
                    memory::virt_to_phys(buf_ptr as u64).ok_or(NicError::PhysTranslate)?;
                // param0 byte 0 = num_qgrps (1).
                self.aq_send(
                    AQ_OP_ADD_TX_QUEUES,
                    AQ_FLAG_BUF | AQ_FLAG_RD,
                    buf_len as u16,
                    1,
                    0,
                    (buf_phys >> 32) as u32,
                    (buf_phys & 0xFFFF_FFFF) as u32,
                )
            };

            match res {
                Ok(_) => {
                    let q_teid = unsafe {
                        let perq = core::ptr::addr_of!(STORAGE.asq_buf) as *const u8;
                        let perq = perq.add(TXQ_QGROUP_HEADER_SIZE);
                        u32::from_le_bytes([*perq.add(4), *perq.add(5), *perq.add(6), *perq.add(7)])
                    };
                    self.parent_teid = parent_teid;
                    let _ = writeln!(
                        Serial,
                        "ice: Add TX Queue ok (candidate {}/{}) — q={}, q_teid=0x{:08x}, parent=0x{:08x}, src_vsi={}",
                        attempt + 1, parents.len(), self.queue_id, q_teid, parent_teid, self.src_vsi
                    );
                    return Ok(());
                }
                Err(e) => {
                    last_err = e;
                    let _ =
                        writeln!(
                        Serial,
                        "ice: Add TX Queue candidate {}/{} parent=0x{:08x} rejected ({:?} = {})",
                        attempt + 1, parents.len(), parent_teid, e, aq_reason(e)
                    );
                }
            }
        }

        let _ = writeln!(
            Serial,
            "ice: Add TX Queue FAILED — all {} candidate parent(s) rejected (last {:?} = {}); src_vsi={}, sched_layers={}",
            parents.len(), last_err, aq_reason(last_err), self.src_vsi, self.num_sched_layers
        );
        Err(NicError::AddTxQueueFailed)
    }

    /// Post-bring-up TX self-test. Queues one broadcast frame with an
    /// IEEE local-experimental ethertype (0x88B5 — switches flood it,
    /// no host stack reacts to it) and polls for the RS-requested
    /// descriptor-done writeback (DTYPE=0xF) plus the QTX_COMM_HEAD
    /// consumer cursor catching up to our producer cursor.
    ///
    /// Requires `tx_ready` to already be set — `transmit()` gates on it.
    fn verify_tx(&mut self) -> Result<(), NicError> {
        const ETHERTYPE_TX_VERIFY: u16 = 0x88B5;
        let slot = self.tx_cursor;

        let mut frame = [0u8; 60]; // minimum Ethernet payload, zero-padded
        frame[0..6].copy_from_slice(&[0xFF; 6]); // broadcast
        frame[6..12].copy_from_slice(&self.mac);
        frame[12] = (ETHERTYPE_TX_VERIFY >> 8) as u8;
        frame[13] = (ETHERTYPE_TX_VERIFY & 0xFF) as u8;
        const PAYLOAD: &[u8] = b"Zero ice TX verify";
        frame[14..14 + PAYLOAD.len()].copy_from_slice(PAYLOAD);

        self.transmit(&frame);

        let head_reg = REG_QTX_COMM_HEAD_BASE + (self.queue_id as usize) * 4;
        unsafe {
            let desc_ptr = core::ptr::addr_of!(STORAGE.tx_ring[slot]);
            for _ in 0..2_000_000u64 {
                let qw = read_volatile(&(*desc_ptr).cmd_type_offset_bsz);
                if qw & 0xF == TX_DESC_DTYPE_DONE {
                    let head = self.read_reg(head_reg) & QTX_COMM_HEAD_MASK;
                    let _ = writeln!(
                        Serial,
                        "ice: TX verify OK — test frame consumed (slot={}, head={}, desc=0x{:016x})",
                        slot, head, qw
                    );
                    return Ok(());
                }
                core::hint::spin_loop();
            }
            let head = self.read_reg(head_reg) & QTX_COMM_HEAD_MASK;
            let qw = read_volatile(&(*desc_ptr).cmd_type_offset_bsz);
            let rctl = self.read_reg(REG_GLLAN_RCTL_0);
            let _ = writeln!(
                Serial,
                "ice: TX verify TIMED OUT — slot={}, head={}, desc=0x{:016x}, GLLAN_RCTL_0=0x{:08x}",
                slot, head, qw, rctl
            );
        }
        Err(NicError::TxVerifyTimeout)
    }

    /// Send one Ethernet frame. Drops silently if the TX queue isn't
    /// online (bring-up failed) or the frame doesn't fit a slot.
    pub fn transmit(&mut self, frame: &[u8]) {
        if !self.tx_ready {
            if !DATA_PATH_WARNED.swap(true, Ordering::Relaxed) {
                let _ = writeln!(
                    Serial,
                    "ice: transmit() called before TX ready (frame_len={}) — dropping (one-time warning)",
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

            // DTYPE=0 (data) | EOP | RS | BSZ. Command-field bit 2
            // (i40e's ICRC) is reserved on E810 and left 0 — the MAC
            // appends the FCS itself.
            let cmd_type: u64 =
                TX_DESC_CMD_EOP | TX_DESC_CMD_RS | ((len as u64) << TX_DESC_BSZ_SHIFT);

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
            // QTX_COMM_DBELL is both the tail register AND the
            // doorbell — writing the new tail advances HW. No separate
            // QTX_TAIL on E810 (that's an i40e-ism).
            self.write_reg(
                REG_QTX_COMM_DBELL_BASE + (self.queue_id as usize) * 4,
                self.tx_cursor as u32,
            );
        }
    }

    /// Poll the RX ring for one frame. Returns None if no DD-marked
    /// descriptor is at the current cursor.
    pub fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        if !self.rx_ready {
            return None;
        }
        unsafe {
            let slot = self.rx_cursor;
            let desc_ptr = core::ptr::addr_of_mut!(STORAGE.rx_ring[slot]);
            // The 32-byte flex descriptor stores status_error0 in
            // QW1[15:0] (the first 16 bits of the second qword) on
            // writeback. Read both qw0 and qw1: qw0 holds pkt_len
            // bits[47:32] which we need for the length.
            let qw1 = read_volatile(&(*desc_ptr).qw1);
            let status_error0 = qw1 & 0xFFFF;
            if status_error0 & RX_FLEX_DESC_STATUS_DD == 0 {
                return None;
            }
            let qw0 = read_volatile(&(*desc_ptr).qw0);
            let length = ((qw0 >> RX_FLEX_DESC_PKT_LEN_SHIFT) & RX_FLEX_DESC_PKT_LEN_MASK) as usize;
            // A descriptor without EOP is one slice of a multi-buffer
            // frame (should not occur with RXMAX == DBUF == 2 KiB, but
            // a fragmenting peer/firmware quirk must not make us hand
            // stale `out` bytes to the parser as if they were a frame).
            // Recycle the slot below and report "no frame".
            let eop = (status_error0 & RX_FLEX_DESC_STATUS_EOP) != 0;
            let n = if eop {
                length.min(out.len()).min(RX_BUFFER_SIZE)
            } else {
                0
            };

            if n > 0 {
                let buf_ptr = core::ptr::addr_of!(STORAGE.rx_buffers[slot]) as *const u8;
                core::ptr::copy_nonoverlapping(buf_ptr, out.as_mut_ptr(), n);
            }

            // Recycle the slot — rewrite the read-side descriptor with
            // pkt_addr and clear writeback fields, then hand the slot
            // back to HW via the tail register.
            let buf_ptr = core::ptr::addr_of_mut!(STORAGE.rx_buffers[slot]) as *mut u8;
            let buf_phys = memory::virt_to_phys(buf_ptr as u64)?;
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

/// Extract `num_branches` from a Get Default Scheduler Topology
/// (0x0400) response `param0`. Layout per Linux
/// `struct ice_aqc_get_topo`: byte 0 = port_num (request, echoed in
/// the response), byte 1 = num_branches.
#[inline(always)]
fn topo_num_branches(param0: u32) -> usize {
    ((param0 >> 8) & 0xFF) as usize
}

/// Decode a firmware AQ return code into a short human label for the
/// serial log. Codes per `ice_aq_err` / `ice_status` (ice_adminq_cmd.h).
/// `NicError::AqRetval(15)` (INVALID_PARAM) is the one the scheduler
/// path hands back when a parent TEID doesn't belong to our VSI's
/// subtree — naming it on the held KVM screen is what lets the operator
/// tell "wrong parent" apart from "dead PHY".
fn aq_reason(err: NicError) -> &'static str {
    match err {
        NicError::AqRetval(rc) => match rc {
            0 => "OK?",
            1 => "EPERM",
            2 => "ENOENT",
            5 => "EIO",
            12 => "ENOMEM",
            13 => "EBUSY",
            14 => "EEXIST",
            15 => "EINVAL/INVALID_PARAM",
            16 => "ENOSPC",
            22 => "ENOSYS",
            _ => "other",
        },
        NicError::AqTimeout => "AQ-timeout",
        _ => "non-AQ",
    }
}

/// Depth of node `idx` in the scheduler tree: the number of hops from
/// it up to a root (a node whose parent TEID is not itself present in
/// the table). Bounded by `n` so a malformed/cyclic response can't spin.
fn node_depth(parents: &[u32], teids: &[u32], n: usize, idx: usize) -> u32 {
    let mut depth = 0u32;
    let mut cur = parents[idx];
    for _ in 0..n {
        let mut found = None;
        for (j, &teid) in teids.iter().take(n).enumerate() {
            if teid == cur && j != idx {
                found = Some(j);
                break;
            }
        }
        match found {
            Some(j) => {
                depth += 1;
                cur = parents[j];
            }
            None => break,
        }
    }
    depth
}

/// Order Add-TX-Queues parent candidates from a parsed topology table.
/// Priority classes, and deepest-first within each: ENTRY_POINT (the
/// qgroup layer `ice_sched_get_free_qparent` returns), then SE_GENERIC
/// (intermediate), then LEAF (last resort). TEID 0 and duplicates are
/// skipped. Returns the candidate count written into `out`.
fn order_parent_candidates(
    parents: &[u32],
    teids: &[u32],
    types: &[u8],
    n: usize,
    out: &mut [u32; MAX_PARENT_CANDIDATES],
) -> usize {
    let mut count = 0usize;
    for &want in &[
        SCHED_NODE_TYPE_ENTRY_POINT,
        SCHED_NODE_TYPE_SE_GENERIC,
        SCHED_NODE_TYPE_LEAF,
    ] {
        loop {
            if count >= MAX_PARENT_CANDIDATES {
                return count;
            }
            // Deepest node of `want` not already emitted.
            let mut best: Option<usize> = None;
            let mut best_depth = 0u32;
            for i in 0..n {
                if types[i] != want {
                    continue;
                }
                let teid = teids[i];
                if teid == 0 || out[..count].contains(&teid) {
                    continue;
                }
                let d = node_depth(parents, teids, n, i);
                if best.is_none() || d > best_depth {
                    best = Some(i);
                    best_depth = d;
                }
            }
            match best {
                Some(i) => {
                    out[count] = teids[i];
                    count += 1;
                }
                None => break,
            }
        }
    }
    count
}

/// Pack `value` (low `width` bits) into `buf` at absolute bit offset
/// `lsb`, little-endian. Used to lay out `ice_tlan_ctx` (22 bytes of
/// packed bitfields) before handing it to AQ Add TX Queues.
///
/// Bit-by-bit so we don't have to special-case width-spanning fields
/// that straddle multiple bytes. Called a handful of times per bring-
/// up, never on the hot path.
fn pack_tlan_bits(buf: &mut [u8; TXQ_CTX_SIZE], value: u64, lsb: usize, width: usize) {
    debug_assert!(width <= 64);
    let mask = if width == 64 {
        !0u64
    } else {
        (1u64 << width) - 1
    };
    let value = value & mask;
    for i in 0..width {
        let bit_pos = lsb + i;
        let byte_idx = bit_pos / 8;
        let bit_in_byte = bit_pos % 8;
        if byte_idx >= buf.len() {
            break;
        }
        let bit_val = ((value >> i) & 1) as u8;
        buf[byte_idx] = (buf[byte_idx] & !(1u8 << bit_in_byte)) | (bit_val << bit_in_byte);
    }
}

// ── Tests ───────────────────────────────────────────────────────────
//
// Compile-time sanity checks on the AQ descriptor layout and the
// register-offset constants. These are the values the firmware
// firmware actually sees — getting any of them wrong is a silent
// failure mode (AQ timeouts, reset never completing, etc.), so it's
// worth pinning them as code.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aq_descriptor_is_32_bytes() {
        // The X710 / E810 admin queue protocol both use the 32-byte
        // descriptor — Intel HAS pins this in stone.
        assert_eq!(core::mem::size_of::<AqDesc>(), 32);
    }

    #[test]
    fn aq_flag_bits_match_intel_has() {
        // Cross-checked against include/linux/net/intel/libie/adminq.h.
        // Bit positions are stable across i40e/ice/iavf families.
        assert_eq!(AQ_FLAG_DD, 0x0001);
        assert_eq!(AQ_FLAG_CMP, 0x0002);
        assert_eq!(AQ_FLAG_ERR, 0x0004);
        assert_eq!(AQ_FLAG_RD, 0x0400);
        assert_eq!(AQ_FLAG_BUF, 0x1000);
        assert_eq!(AQ_FLAG_SI, 0x2000);
    }

    #[test]
    fn pfgen_ctrl_offset_is_ice_value_not_i40e() {
        // Regression guard: the i40e value (0x00092400) silently does
        // nothing on E810 and the reset would hang. We must use the
        // ice-specific offset.
        assert_eq!(REG_PFGEN_CTRL, 0x00091000);
        assert_ne!(REG_PFGEN_CTRL, 0x00092400);
    }

    #[test]
    fn e810_xxv_sfp_is_supported() {
        // Cherry Server's NIC. Without this entry the stack falls
        // straight through to NoSupportedNic at bind time.
        assert!(SUPPORTED_DEVICES.contains(&0x159B));
    }

    #[test]
    fn admin_queue_register_layout_matches_i40e_base() {
        // ice and i40e share the AQ register base; only the queue /
        // queue-context registers differ. If this ever drifts, the
        // AQ doorbell will land in the wrong CSR and the firmware
        // will appear silent.
        assert_eq!(REG_PF_ATQBAL, 0x00080000);
        assert_eq!(REG_PF_ATQT, 0x00080400);
        assert_eq!(REG_PF_ARQBAL, 0x00080080);
        assert_eq!(REG_PF_ARQT, 0x00080480);
    }

    #[test]
    fn ring_sizes_are_sane() {
        assert!(AQ_DESC_COUNT.is_power_of_two());
        assert!(TX_DESC_COUNT.is_power_of_two());
        assert!(RX_DESC_COUNT.is_power_of_two());
        assert_eq!(RX_BUFFER_SIZE % 128, 0); // encoded as size / 128
    }

    #[test]
    fn pack_tlan_bits_handles_byte_boundary() {
        // qlen field (13 bits at bit 135) straddles three bytes
        // (byte 16 bit 7 → byte 18 bit 3). Regression guard: an
        // earlier draft of this packer used `>>` on the byte index
        // instead of bit index and silently dropped the high bit.
        let mut buf = [0u8; TXQ_CTX_SIZE];
        pack_tlan_bits(&mut buf, 0x1FFF, 135, 13); // all-ones 13-bit
                                                   // byte 16: bit 7 set (low bit of the field) → 0x80
        assert_eq!(buf[16], 0x80);
        // byte 17: all 8 bits set
        assert_eq!(buf[17], 0xFF);
        // byte 18: low 4 bits set (top 4 bits of the field)
        assert_eq!(buf[18], 0x0F);
        // every other byte untouched
        for (i, b) in buf.iter().enumerate() {
            if i != 16 && i != 17 && i != 18 {
                assert_eq!(*b, 0, "byte {} leaked", i);
            }
        }
    }

    #[test]
    fn pack_tlan_bits_round_trips_base_field() {
        // BASE is the 57-bit ring physical-address field at bit 0.
        // Verify a non-trivial value round-trips through the packer.
        let mut buf = [0u8; TXQ_CTX_SIZE];
        let value: u64 = 0x01_23_45_67_89_AB_CD_EFu64 & ((1u64 << 57) - 1);
        pack_tlan_bits(&mut buf, value, 0, 57);
        // Recompose low 57 bits manually.
        let mut got: u64 = 0;
        for i in 0..57 {
            let bit = (buf[i / 8] >> (i % 8)) & 1;
            got |= (bit as u64) << i;
        }
        assert_eq!(got, value);
    }

    #[test]
    fn topo_num_branches_reads_byte_one_not_byte_zero() {
        // Regression guard for the first Cherry boot: the parser read
        // byte 0 (the echoed port_num) as num_branches. Per Linux
        // `struct ice_aqc_get_topo` the count is byte 1.
        // param0 = port_num 1, num_branches 3.
        assert_eq!(topo_num_branches(0x0000_0301), 3);
        // lport 0 with 1 branch — the wired Cherry port's real reply.
        assert_eq!(topo_num_branches(0x0000_0100), 1);
        // The buggy read would have returned 0 here (port echo 0).
        assert_ne!(topo_num_branches(0x0000_0100), 0);
    }

    #[test]
    fn add_tx_queue_buffer_layout_is_60_bytes() {
        // Catches accidental drift in the qgroup wrapper sizes —
        // anything other than 60 bytes here and the firmware will
        // walk off the end of the per-queue record.
        assert_eq!(TXQ_QGROUP_HEADER_SIZE, 8);
        assert_eq!(TXQ_PERQ_SIZE, 52);
        assert_eq!(TXQ_QGROUP_HEADER_SIZE + TXQ_PERQ_SIZE, 60);
    }

    #[test]
    fn rx_context_dword_layout_matches_linux() {
        // Linux: QRX_CONTEXT(i, idx) = 0x00280000 + i*0x2000 + idx*4.
        // Drift here and the rlan_ctx ends up scattered into the
        // wrong queue's context window (or worse, into a reserved
        // register block that reads back as zero on the next probe).
        assert_eq!(REG_QRX_CONTEXT_BASE, 0x00280000);
        assert_eq!(RX_CONTEXT_DWORD_STRIDE, 0x2000);
        assert_eq!(RX_CONTEXT_DWORDS, 8);
    }

    #[test]
    fn tlan_ctx_qlen_tso_legacy_offsets_do_not_overlap_neighbors() {
        // Regression guard for the Cherry "NET: FAILED Ice(AddTxQueueFailed)"
        // boot: qlen/tso_ena/legacy_int had been shifted to 132/149/161,
        // overlapping adjust_prof_id(6@129), quanta_prof_idx(4@148), and
        // tso_qnum(11@153) respectively — firmware INVALID_PARAM.
        //
        // Pack the three fields at their genuine ice_tlan_ctx offsets and
        // assert each neighboring field's bits stay untouched (all zero).
        let mut ctx = [0u8; TXQ_CTX_SIZE];
        pack_tlan_bits(&mut ctx, 0x1FFF, 135, 13); // qlen   (135..=147)
        pack_tlan_bits(&mut ctx, 1, 152, 1); //        tso_ena (152)
        pack_tlan_bits(&mut ctx, 1, 164, 1); //        legacy_int (164)

        let get = |bit: usize| (ctx[bit / 8] >> (bit % 8)) & 1;

        // adjust_prof_id occupies 129..=134 — must be wholly clear.
        for b in 129..=134 {
            assert_eq!(get(b), 0, "qlen leaked into adjust_prof_id bit {b}");
        }
        // quanta_prof_idx occupies 148..=151 — must be wholly clear.
        for b in 148..=151 {
            assert_eq!(get(b), 0, "tso_ena leaked into quanta_prof_idx bit {b}");
        }
        // tso_qnum occupies 153..=163 — must be wholly clear.
        for b in 153..=163 {
            assert_eq!(get(b), 0, "legacy_int leaked into tso_qnum bit {b}");
        }
        // And the three fields themselves landed where intended.
        assert_eq!(get(152), 1, "tso_ena not set at bit 152");
        assert_eq!(get(164), 1, "legacy_int not set at bit 164");
        assert_eq!(get(135), 1, "qlen low bit not set at bit 135");
        assert_eq!(get(147), 1, "qlen high bit not set at bit 147");
    }

    #[test]
    fn node_depth_counts_hops_to_root() {
        // root(teid 1) → tc(2) → gen(3) → entry(4)
        let parents = [0u32, 1, 2, 3];
        let teids = [1u32, 2, 3, 4];
        assert_eq!(node_depth(&parents, &teids, 4, 0), 0); // root
        assert_eq!(node_depth(&parents, &teids, 4, 1), 1);
        assert_eq!(node_depth(&parents, &teids, 4, 2), 2);
        assert_eq!(node_depth(&parents, &teids, 4, 3), 3);
    }

    #[test]
    fn order_parent_candidates_entry_point_deepest_first() {
        // root(1) → gen(2) → entryA(3)           depth(entryA)=2
        //           gen(2) → gen(4) → entryB(5)  depth(entryB)=3
        let parents = [0u32, 1, 2, 2, 4];
        let teids = [1u32, 2, 3, 4, 5];
        let types = [
            SCHED_NODE_TYPE_ROOT_PORT,
            SCHED_NODE_TYPE_SE_GENERIC,
            SCHED_NODE_TYPE_ENTRY_POINT,
            SCHED_NODE_TYPE_SE_GENERIC,
            SCHED_NODE_TYPE_ENTRY_POINT,
        ];
        let mut out = [0u32; MAX_PARENT_CANDIDATES];
        let n = order_parent_candidates(&parents, &teids, &types, 5, &mut out);
        // ENTRY_POINTs first (deepest 5 before 3), then SE_GENERICs
        // (deepest 4 before 2); ROOT is never a candidate.
        assert_eq!(&out[..n], &[5u32, 3, 4, 2]);
    }

    #[test]
    fn order_parent_candidates_skips_zero_and_dupes() {
        // All ENTRY_POINT; teid 0 must be skipped, teid 7 deduped.
        let parents = [0u32; 4];
        let teids = [0u32, 7, 7, 9];
        let types = [SCHED_NODE_TYPE_ENTRY_POINT; 4];
        let mut out = [0u32; MAX_PARENT_CANDIDATES];
        let n = order_parent_candidates(&parents, &teids, &types, 4, &mut out);
        assert_eq!(n, 2);
        assert_eq!(&out[..n], &[7u32, 9]);
        assert!(!out[..n].contains(&0));
    }

    #[test]
    fn order_parent_candidates_caps_at_max() {
        // More ENTRY_POINTs than the candidate cap — must not overflow.
        let mut parents = [0u32; MAX_TOPO_NODES];
        let mut teids = [0u32; MAX_TOPO_NODES];
        let mut types = [SCHED_NODE_TYPE_ENTRY_POINT; MAX_TOPO_NODES];
        for i in 0..MAX_TOPO_NODES {
            teids[i] = (i as u32) + 1; // 1.. so none are skipped as zero
            parents[i] = 0;
            types[i] = SCHED_NODE_TYPE_ENTRY_POINT;
        }
        let mut out = [0u32; MAX_PARENT_CANDIDATES];
        let n = order_parent_candidates(&parents, &teids, &types, MAX_TOPO_NODES, &mut out);
        assert_eq!(n, MAX_PARENT_CANDIDATES);
    }
}
