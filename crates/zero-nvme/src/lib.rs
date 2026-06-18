// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::field_reassign_with_default)]
//! NVMe wire types — register layouts, queue entries, command
//! builders, PRP planner.
//!
//! Pure `no_std` data manipulation. Doing this in a separate crate
//! (rather than alongside the hardware-driving code in the kernel)
//! lets the workspace `cargo test` exercise every encoder on a host
//! machine without needing real NVMe silicon or a kernel test
//! harness. The kernel's `drivers::nvme::hw` module imports this
//! crate and adds the MMIO / DMA / queue-bring-up layer on top.
//!
//! Section numbers below reference NVM Express Base Spec 1.4. The
//! relevant subset (register layout, SQE/CQE format, admin opcodes,
//! Read command, PRP encoding) has been stable since NVMe 1.0, so
//! the encoders work against every modern controller — from the
//! original Intel 750 to the Solidigm Gen5 D7-PS1010 drives in our
//! Cherry Server target box.

#![no_std]
#![allow(clippy::missing_safety_doc)]

// ---------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------

/// PCI class code for "Mass storage controller".
pub const PCI_CLASS_MASS_STORAGE: u8 = 0x01;

/// PCI subclass for "Non-Volatile Memory controller" (i.e. NVMe).
pub const PCI_SUBCLASS_NVM: u8 = 0x08;

/// Programming interface for NVM Express. PCI class triple is
/// (0x01, 0x08, 0x02) per NVMe spec.
pub const PCI_PROGIF_NVME: u8 = 0x02;

// BAR0 register offsets — NVMe Base Spec 1.4 §3.1.
pub const REG_CAP: u32 = 0x00; // Controller Capabilities (64-bit)
pub const REG_VS: u32 = 0x08; // Version (32-bit)
pub const REG_INTMS: u32 = 0x0C;
pub const REG_INTMC: u32 = 0x10;
pub const REG_CC: u32 = 0x14; // Controller Configuration
pub const REG_CSTS: u32 = 0x1C; // Controller Status
pub const REG_AQA: u32 = 0x24; // Admin Queue Attributes
pub const REG_ASQ: u32 = 0x28; // Admin Submission Queue Base (64-bit)
pub const REG_ACQ: u32 = 0x30; // Admin Completion Queue Base (64-bit)

/// Doorbell array starts at BAR0 + 0x1000. Stride between consecutive
/// doorbells is `4 << DSTRD` bytes, where DSTRD comes from CAP[35:32].
pub const REG_DOORBELL_BASE: u32 = 0x1000;

/// NVMe spec mandates one 4 KiB memory page as the smallest the
/// controller has to support (CAP.MPSMIN can be 0, encoding 2^12).
/// The driver pins its PRP layout to exactly that size — matches the
/// kernel's 4 KiB page size and keeps the planner platform-agnostic.
pub const NVME_PAGE_SIZE: usize = 4096;

/// Size of one Submission Queue Entry, in bytes. Spec-mandated.
pub const SQE_SIZE: usize = 64;
/// Size of one Completion Queue Entry, in bytes. Spec-mandated.
pub const CQE_SIZE: usize = 16;

/// Default queue depth for the admin queue. Small — admin commands
/// are issued at most a handful of times during bring-up.
pub const ADMIN_QUEUE_DEPTH: u16 = 64;

/// Default queue depth for the single I/O queue. 64 entries leaves
/// headroom above the in-flight count we actually use (polling mode
/// runs one command at a time today).
pub const IO_QUEUE_DEPTH: u16 = 64;

/// Queue identifier for the (only) I/O queue. Admin queue is QID 0.
pub const IO_QID: u16 = 1;

/// Admin command opcodes (NVMe Base Spec 1.4 §5).
pub const OPC_DELETE_IOSQ: u8 = 0x00;
pub const OPC_CREATE_IOSQ: u8 = 0x01;
pub const OPC_DELETE_IOCQ: u8 = 0x04;
pub const OPC_CREATE_IOCQ: u8 = 0x05;
pub const OPC_IDENTIFY: u8 = 0x06;

/// NVM command opcodes (NVMe NVM Command Set Spec).
pub const OPC_NVM_WRITE: u8 = 0x01;
pub const OPC_NVM_READ: u8 = 0x02;

/// Identify CNS (Controller or Namespace Structure) selector.
pub const IDENTIFY_CNS_NAMESPACE: u32 = 0x00;
pub const IDENTIFY_CNS_CONTROLLER: u32 = 0x01;

/// CSTS bit positions.
pub const CSTS_RDY: u32 = 1 << 0;
pub const CSTS_CFS: u32 = 1 << 1;

/// 32-bit CC value with EN cleared. Used to take the controller down
/// before reprogramming the admin queue.
pub const CC_DISABLE: u32 = 0;

// ---------------------------------------------------------------------
//  Capabilities
// ---------------------------------------------------------------------

/// Decoded view of the 64-bit CAP register.
#[derive(Debug, Copy, Clone)]
pub struct Capabilities {
    /// Maximum Queue Entries Supported, 0-based (CAP[15:0]). A value
    /// of N means the controller supports queues of up to N+1 entries.
    pub mqes: u16,
    /// Contiguous Queues Required (CAP[16]). We always allocate
    /// physically contiguous queues, so this is informational.
    pub cqr: bool,
    /// Worst-case ready timeout in 500ms units (CAP[31:24]).
    pub timeout_500ms: u8,
    /// Doorbell Stride (CAP[35:32]). Stride in bytes between
    /// consecutive doorbell registers is `4 << dstrd`.
    pub dstrd: u8,
    /// Minimum supported memory page size = 2^(12 + mpsmin) bytes
    /// (CAP[51:48]).
    pub mpsmin: u8,
    /// Maximum supported memory page size = 2^(12 + mpsmax) bytes
    /// (CAP[55:52]).
    pub mpsmax: u8,
}

impl Capabilities {
    /// Decode the raw 64-bit CAP register.
    pub const fn from_raw(raw: u64) -> Self {
        Self {
            mqes: (raw & 0xFFFF) as u16,
            cqr: (raw >> 16) & 1 != 0,
            timeout_500ms: ((raw >> 24) & 0xFF) as u8,
            dstrd: ((raw >> 32) & 0x0F) as u8,
            mpsmin: ((raw >> 48) & 0x0F) as u8,
            mpsmax: ((raw >> 52) & 0x0F) as u8,
        }
    }

    /// Stride in bytes between consecutive doorbell registers.
    pub const fn doorbell_stride(&self) -> u32 {
        4u32 << (self.dstrd as u32)
    }

    /// Minimum supported memory page size in bytes.
    pub const fn min_page_size(&self) -> usize {
        1usize << (12 + self.mpsmin as usize)
    }

    /// Worst-case time to wait for CSTS.RDY in milliseconds.
    pub const fn enable_timeout_ms(&self) -> u32 {
        self.timeout_500ms as u32 * 500
    }
}

/// Build the 32-bit value to write into the CC register for steady
/// state with the NVM command set, 4 KiB memory pages, queue entry
/// sizes matching the spec (`IOSQES=6` → 64 B SQEs, `IOCQES=4` →
/// 16 B CQEs), round-robin arbitration, no shutdown notification.
pub const fn cc_enable(mpsmin: u8) -> u32 {
    // CC.EN[0] = 1
    // CC.CSS[6:4] = 0 (NVM command set)
    // CC.MPS[10:7] = mpsmin (must match controller floor)
    // CC.AMS[13:11] = 0 (round-robin)
    // CC.SHN[15:14] = 0 (no shutdown)
    // CC.IOSQES[19:16] = 6 (2^6 = 64-byte SQE)
    // CC.IOCQES[23:20] = 4 (2^4 = 16-byte CQE)
    let mut cc: u32 = 0;
    cc |= 1; // EN
    cc |= (mpsmin as u32 & 0xF) << 7; // MPS
    cc |= 6u32 << 16; // IOSQES
    cc |= 4u32 << 20; // IOCQES
    cc
}

/// Build the 32-bit AQA value for an admin queue of `depth` entries
/// (each side). Wire encoding is `depth - 1` in both halves.
pub const fn aqa(depth: u16) -> u32 {
    let zb = (depth as u32).saturating_sub(1) & 0x0FFF;
    (zb << 16) | zb
}

// ---------------------------------------------------------------------
//  Doorbell offsets
// ---------------------------------------------------------------------

/// Byte offset (from BAR0) of the Submission Queue Tail Doorbell for
/// queue `qid`, given a doorbell stride from CAP.DSTRD.
pub const fn sq_tail_doorbell_offset(qid: u16, dstrd: u8) -> u32 {
    let stride = 4u32 << (dstrd as u32);
    REG_DOORBELL_BASE + (qid as u32) * 2 * stride
}

/// Byte offset (from BAR0) of the Completion Queue Head Doorbell for
/// queue `qid`, given a doorbell stride from CAP.DSTRD.
pub const fn cq_head_doorbell_offset(qid: u16, dstrd: u8) -> u32 {
    let stride = 4u32 << (dstrd as u32);
    REG_DOORBELL_BASE + ((qid as u32) * 2 + 1) * stride
}

// ---------------------------------------------------------------------
//  SQE / CQE
// ---------------------------------------------------------------------

/// 64-byte Submission Queue Entry. Layout matches NVMe Base Spec 1.4
/// §4.2: CDW0 (opcode + CID + control), NSID, MPTR, two PRP entries,
/// six command-specific dwords.
///
/// `repr(C)` + size assertion in tests guarantees the bit-exact
/// layout the hardware DMA engine expects.
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SubmissionEntry {
    /// CDW0: bits 0-7 opcode, 8-9 fuse, 14-15 PSDT, 16-31 CID.
    pub cdw0: u32,
    /// Namespace ID.
    pub nsid: u32,
    /// Reserved per spec — must be zero.
    pub _rsvd: u64,
    /// Metadata pointer (unused).
    pub mptr: u64,
    /// PRP1 — first PRP entry. Physical address.
    pub prp1: u64,
    /// PRP2 — either a second page address or a PRP-list pointer,
    /// depending on transfer length.
    pub prp2: u64,
    pub cdw10: u32,
    pub cdw11: u32,
    pub cdw12: u32,
    pub cdw13: u32,
    pub cdw14: u32,
    pub cdw15: u32,
}

/// 16-byte Completion Queue Entry. Layout matches NVMe Base Spec 1.4
/// §4.6.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CompletionEntry {
    /// Command-specific result.
    pub dw0: u32,
    /// Reserved.
    pub _rsvd: u32,
    /// Submission Queue head pointer the controller has consumed past.
    pub sq_head: u16,
    /// Submission Queue identifier this completion was posted for.
    pub sq_id: u16,
    /// Command identifier echoed back from the SQE's CDW0[31:16].
    pub cid: u16,
    /// Status field: bit 0 = phase tag, bits 1-15 = status code +
    /// status code type + DNR/M.
    pub status: u16,
}

impl CompletionEntry {
    /// Phase tag. Flips between 0 and 1 each time the controller
    /// wraps the CQ — the driver uses this to detect new entries
    /// without needing interrupts.
    pub const fn phase(&self) -> bool {
        (self.status & 0x1) != 0
    }

    /// Status code + status code type, with the phase tag masked off.
    /// Zero = success; any non-zero value is a failure.
    pub const fn status_no_phase(&self) -> u16 {
        self.status >> 1
    }

    /// True if the controller reported a non-success completion.
    pub const fn is_error(&self) -> bool {
        self.status_no_phase() != 0
    }
}

// ---------------------------------------------------------------------
//  Command builders
// ---------------------------------------------------------------------

/// Encode CDW0 — opcode in [7:0], CID in [31:16].
pub const fn make_cdw0(opcode: u8, cid: u16) -> u32 {
    (opcode as u32) | ((cid as u32) << 16)
}

/// Build an Identify (CNS=controller) admin command.
pub fn build_identify_controller(cid: u16, data_prp1: u64) -> SubmissionEntry {
    SubmissionEntry {
        cdw0: make_cdw0(OPC_IDENTIFY, cid),
        nsid: 0,
        prp1: data_prp1,
        cdw10: IDENTIFY_CNS_CONTROLLER,
        ..Default::default()
    }
}

/// Build an Identify (CNS=namespace) admin command.
pub fn build_identify_namespace(cid: u16, nsid: u32, data_prp1: u64) -> SubmissionEntry {
    SubmissionEntry {
        cdw0: make_cdw0(OPC_IDENTIFY, cid),
        nsid,
        prp1: data_prp1,
        cdw10: IDENTIFY_CNS_NAMESPACE,
        ..Default::default()
    }
}

/// Build a Create I/O Completion Queue admin command (opcode 0x05).
pub fn build_create_iocq(
    cid: u16,
    qid: u16,
    qsize: u16,
    cq_base: u64,
    interrupts: bool,
) -> SubmissionEntry {
    let zb = qsize.saturating_sub(1) as u32;
    let mut cdw11: u32 = 1; // PC=1 (physically contiguous)
    if interrupts {
        cdw11 |= 1 << 1; // IEN
    }
    SubmissionEntry {
        cdw0: make_cdw0(OPC_CREATE_IOCQ, cid),
        prp1: cq_base,
        cdw10: (qid as u32) | (zb << 16),
        cdw11,
        ..Default::default()
    }
}

/// Build a Create I/O Submission Queue admin command (opcode 0x01).
pub fn build_create_iosq(
    cid: u16,
    qid: u16,
    qsize: u16,
    sq_base: u64,
    cqid: u16,
) -> SubmissionEntry {
    let zb = qsize.saturating_sub(1) as u32;
    // CDW11: PC=1, QPRIO=01 (medium), CQID in upper half.
    let cdw11: u32 = 1 | (0b01 << 1) | ((cqid as u32) << 16);
    SubmissionEntry {
        cdw0: make_cdw0(OPC_CREATE_IOSQ, cid),
        prp1: sq_base,
        cdw10: (qid as u32) | (zb << 16),
        cdw11,
        ..Default::default()
    }
}

/// Build an NVM Read command (opcode 0x02). `nlb` is the number of
/// logical blocks; the wire encoding is zero-based and the builder
/// handles that translation.
pub fn build_read(
    cid: u16,
    nsid: u32,
    slba: u64,
    nlb: u16,
    prp1: u64,
    prp2: u64,
) -> SubmissionEntry {
    SubmissionEntry {
        cdw0: make_cdw0(OPC_NVM_READ, cid),
        nsid,
        prp1,
        prp2,
        cdw10: (slba & 0xFFFF_FFFF) as u32,
        cdw11: (slba >> 32) as u32,
        cdw12: (nlb.saturating_sub(1)) as u32,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------
//  PRP planner
// ---------------------------------------------------------------------

/// Resolved PRP1/PRP2 pair for a transfer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PrpPlan {
    /// PRP1 — physical address of the first DMA page.
    pub prp1: u64,
    /// PRP2:
    ///   * exactly one page  → 0 (unused)
    ///   * exactly two pages → second page physical address
    ///   * more than two     → physical address of a PRP list page,
    ///     where entries 0..N-2 hold the physical addresses of pages
    ///     1..N-1 (the first page is in PRP1).
    pub prp2: u64,
    /// Number of u64 entries the driver must write into the PRP list
    /// page (0 when prp2 is "unused" or points directly at page 2).
    pub list_entries: usize,
}

/// Plan the PRP1/PRP2 pair for a transfer of `bytes` starting at the
/// page-aligned physical address `buffer_phys`, using a PRP list at
/// `prp_list_phys` if needed.
///
/// Constraints:
///   * `buffer_phys` and `prp_list_phys` are 4 KiB-aligned.
///   * `bytes > 0`.
///   * The transfer fits in a single PRP list page: the driver issues
///     ≤ 2 MiB Read commands (512 4 KiB pages of data — 1 in PRP1 +
///     511 in the list — exactly the capacity of a 4 KiB list page,
///     8 bytes per entry).
///
/// Returns `None` if any constraint is violated.
pub fn plan_prp(buffer_phys: u64, bytes: usize, prp_list_phys: u64) -> Option<PrpPlan> {
    if bytes == 0 {
        return None;
    }
    if buffer_phys & (NVME_PAGE_SIZE as u64 - 1) != 0 {
        return None;
    }
    if prp_list_phys & (NVME_PAGE_SIZE as u64 - 1) != 0 {
        return None;
    }
    let pages = bytes.div_ceil(NVME_PAGE_SIZE);
    let max_list_entries = NVME_PAGE_SIZE / 8;
    if pages > max_list_entries + 1 {
        return None;
    }
    Some(match pages {
        1 => PrpPlan {
            prp1: buffer_phys,
            prp2: 0,
            list_entries: 0,
        },
        2 => PrpPlan {
            prp1: buffer_phys,
            prp2: buffer_phys + NVME_PAGE_SIZE as u64,
            list_entries: 0,
        },
        n => PrpPlan {
            prp1: buffer_phys,
            prp2: prp_list_phys,
            list_entries: n - 1,
        },
    })
}

/// Fill a caller-provided PRP list slice with the physical addresses
/// of pages 1..N (page 0 already lives in PRP1). Used when the buffer
/// is physically contiguous; non-contiguous buffers need per-page
/// virt→phys translation, which the host crate cannot do.
#[allow(clippy::result_unit_err)]
pub fn fill_prp_list(buffer_phys: u64, list: &mut [u64], plan: &PrpPlan) -> Result<(), ()> {
    if plan.list_entries == 0 {
        return Ok(());
    }
    if list.len() < plan.list_entries {
        return Err(());
    }
    for (i, entry) in list.iter_mut().enumerate().take(plan.list_entries) {
        *entry = buffer_phys + ((i + 1) as u64) * NVME_PAGE_SIZE as u64;
    }
    Ok(())
}

// ---------------------------------------------------------------------
//  Data-drive selection
// ---------------------------------------------------------------------

/// Magic bytes of a GGUF file ("GGUF" little-endian), used by the boot
/// path to verify the drive we picked actually holds a model. Mirrors
/// the constant in `kernel::model_loader::GGUF_MAGIC` so this crate
/// stays the single source of truth for wire-level constants the
/// kernel reads off-device.
pub const GGUF_MAGIC_LE: u32 = 0x46554747;
/// Magic bytes of a SilicatePack `.smodel` file ("SILM"
/// little-endian). Zero Server treats this as the native model
/// artifact; raw GGUF remains an import/compatibility payload.
pub const SMODEL_MAGIC_LE: u32 = 0x4D4C4953;

/// Snapshot of one probed NVMe namespace, used by `pick_largest_namespace`
/// to choose between system drives (small) and data drives (3.2 TB
/// Solidigm Gen5 on Cherry) at boot time without committing to any
/// specific PCI BDF in advance.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NamespaceProbe {
    /// PCI bus / device / function the controller sits on. Used purely
    /// for logging — selection itself is by capacity.
    pub pci_bus: u8,
    pub pci_device: u8,
    pub pci_function: u8,
    /// Total LBAs (NSZE field of the Identify Namespace response).
    pub nlba: u64,
    /// LBA size in bytes (decoded via LBAF + FLBAS).
    pub lba_size: u32,
}

impl NamespaceProbe {
    /// Total byte capacity, saturating on overflow. 3.2 TB drives sit
    /// safely inside u64 (3.2 TB ≈ 3.5e12; u64::MAX ≈ 1.8e19), so the
    /// saturation is purely defensive against pathological inputs.
    #[inline]
    pub const fn total_bytes(&self) -> u64 {
        self.nlba.saturating_mul(self.lba_size as u64)
    }
}

/// Index of the largest-capacity probe in `probes`, or `None` if the
/// slice is empty. On ties (two drives with identical capacity, e.g.
/// the dual 3.2 TB Solidigms in the Cherry box), the earliest index
/// wins — making the choice deterministic across boots.
///
/// This is the data-drive selection primitive: the driver layer probes
/// every NVMe in the PCI scan and calls into this function. Pure
/// function, fully unit-tested on the host, so the driver can lean on
/// it without re-deriving the rule.
pub fn pick_largest_namespace(probes: &[NamespaceProbe]) -> Option<usize> {
    if probes.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_bytes = probes[0].total_bytes();
    for (i, p) in probes.iter().enumerate().skip(1) {
        let b = p.total_bytes();
        if b > best_bytes {
            best_bytes = b;
            best_idx = i;
        }
    }
    Some(best_idx)
}

/// Return `true` if the first 4 bytes of `bytes` match the GGUF magic
/// (little-endian "GGUF"). Used after [`pick_largest_namespace`] picks
/// a drive to verify the drive actually contains a model — guards
/// against the "Phase B was never run" failure mode.
#[inline]
pub fn looks_like_gguf(bytes: &[u8]) -> bool {
    bytes.len() >= 4
        && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == GGUF_MAGIC_LE
}

/// Return `true` if the first 4 bytes match any model marker the
/// Zero Server boot path knows how to route.
#[inline]
pub fn looks_like_model_container(bytes: &[u8]) -> bool {
    if bytes.len() < 4 {
        return false;
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    magic == GGUF_MAGIC_LE || magic == SMODEL_MAGIC_LE
}

// ---------------------------------------------------------------------
//  Polling-loop spin budget
// ---------------------------------------------------------------------

/// Heuristic conversion factor from `timeout_ms` to a spin-loop iteration
/// budget. ~1 million iterations per requested millisecond. Calibrated
/// loose: a controller that needs more time is far more likely to be
/// genuinely wedged than legitimately slow.
pub const SPIN_PER_MS: u64 = 1_000_000;

/// Hard floor: never spin fewer than 200 million iterations even when
/// the caller passes a low timeout. Small CAP.TO values get rounded up
/// here so a fast-CPU build doesn't expire before reality.
pub const SPIN_FLOOR: u64 = 200_000_000;

/// Convert a desired `timeout_ms` into a spin-loop iteration count.
///
/// Honours [`SPIN_FLOOR`] as a minimum and saturates `u32::MAX`
/// milliseconds without overflowing the 64-bit return. The kernel-side
/// `wait_csts_set` / `wait_csts_clear` use this to honour the CAP.TO
/// derived timeout — pre-audit the parameter was silently dropped.
#[inline]
pub const fn spin_budget(timeout_ms: u32) -> u64 {
    let from_ms = (timeout_ms as u64).saturating_mul(SPIN_PER_MS);
    if from_ms > SPIN_FLOOR {
        from_ms
    } else {
        SPIN_FLOOR
    }
}

// ---------------------------------------------------------------------
//  Identify Namespace response decoder
// ---------------------------------------------------------------------

/// Decoded subset of the Identify Namespace response (NVMe Base Spec
/// 1.4 §5.15.2). The full response is 4 KiB — we only care about the
/// fields that drive Read commands.
#[derive(Debug, Copy, Clone, Default)]
pub struct IdentifyNamespace {
    /// Number of logical blocks in the namespace (NSZE).
    pub nlba: u64,
    /// Bytes per logical block (decoded from FLBAS + LBAF table).
    pub lba_size: u32,
}

/// Decode the relevant fields out of a 4 KiB Identify Namespace
/// response buffer. Returns `None` if the buffer is too small.
pub fn decode_identify_namespace(buf: &[u8]) -> Option<IdentifyNamespace> {
    // bytes 0-7  : NSZE (u64 little-endian)
    // byte 26    : FLBAS (Formatted LBA Size — bits [4:0] index into
    //              LBAF table)
    // bytes 128 +
    //   16 * idx : LBAF[idx] (16 bytes). bits [23:16] of dword 0
    //              are LBADS (LBA Data Size, log2).
    if buf.len() < 128 + 16 {
        return None;
    }
    let nsze = u64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]);
    let flbas = buf[26];
    let lbaf_idx = (flbas & 0x0F) as usize;
    let off = 128 + 16 * lbaf_idx;
    if buf.len() < off + 4 {
        return None;
    }
    let lbaf_dword = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    let lbads = (lbaf_dword >> 16) & 0xFF;
    if lbads >= 32 {
        return None;
    }
    Some(IdentifyNamespace {
        nlba: nsze,
        lba_size: 1u32 << lbads,
    })
}

// ---------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn sqe_is_64_bytes() {
        assert_eq!(size_of::<SubmissionEntry>(), SQE_SIZE);
        assert_eq!(align_of::<SubmissionEntry>(), 64);
    }

    #[test]
    fn cqe_is_16_bytes() {
        assert_eq!(size_of::<CompletionEntry>(), CQE_SIZE);
        assert_eq!(align_of::<CompletionEntry>(), 16);
    }

    #[test]
    fn cap_decode_zen4_nvme_typical() {
        // Synthesised: MQES=0x3FF (1024 entries), TO=0x14 (10s),
        // DSTRD=0, MPSMIN=0, MPSMAX=4.
        //
        // Layout: TO is CAP[31:24], so 0x14 → 0x1400_0000; MPSMAX is
        // CAP[55:52], so 4 → 0x0040_0000_0000_0000; MQES is CAP[15:0].
        let raw: u64 = 0x0040_0000_1400_03FF;
        let cap = Capabilities::from_raw(raw);
        assert_eq!(cap.mqes, 0x3FF);
        assert_eq!(cap.dstrd, 0);
        assert_eq!(cap.mpsmin, 0);
        assert_eq!(cap.mpsmax, 4);
        assert_eq!(cap.timeout_500ms, 0x14);
        assert_eq!(cap.enable_timeout_ms(), 10_000);
        assert_eq!(cap.min_page_size(), 4096);
        assert_eq!(cap.doorbell_stride(), 4);
    }

    #[test]
    fn cap_decode_dstrd_nonzero() {
        // DSTRD=2 → stride 4 << 2 = 16.
        let raw: u64 = 0x0000_0002_0000_0000;
        let cap = Capabilities::from_raw(raw);
        assert_eq!(cap.dstrd, 2);
        assert_eq!(cap.doorbell_stride(), 16);
    }

    #[test]
    fn cap_decode_mpsmin_8kib() {
        // MPSMIN=1 → 2^(12+1) = 8192 B.
        let raw: u64 = 0x0001_0000_0000_0000;
        let cap = Capabilities::from_raw(raw);
        assert_eq!(cap.mpsmin, 1);
        assert_eq!(cap.min_page_size(), 8192);
    }

    #[test]
    fn cap_decode_cqr_bit() {
        let cap = Capabilities::from_raw(1 << 16);
        assert!(cap.cqr);
        let cap = Capabilities::from_raw(0);
        assert!(!cap.cqr);
    }

    #[test]
    fn cc_enable_encodes_iosqes_iocqes() {
        let cc = cc_enable(0);
        assert_eq!(cc & 1, 1); // EN
        assert_eq!((cc >> 7) & 0xF, 0); // MPS=0 (4 KiB)
        assert_eq!((cc >> 16) & 0xF, 6); // IOSQES=6 → 64 B
        assert_eq!((cc >> 20) & 0xF, 4); // IOCQES=4 → 16 B
                                         // CSS bits [6:4] = 0 (NVM command set).
        assert_eq!((cc >> 4) & 0x7, 0);
    }

    #[test]
    fn cc_enable_propagates_mps() {
        let cc = cc_enable(1);
        assert_eq!((cc >> 7) & 0xF, 1);
    }

    #[test]
    fn cc_disable_is_zero() {
        assert_eq!(CC_DISABLE, 0);
        assert_eq!(CC_DISABLE & 1, 0);
    }

    #[test]
    fn aqa_is_zero_based_on_both_halves() {
        let a = aqa(64);
        assert_eq!(a & 0xFFF, 63);
        assert_eq!((a >> 16) & 0xFFF, 63);
    }

    #[test]
    fn aqa_depth_zero_is_clamped_to_zero() {
        // Saturating sub means depth 0 → encoded as zero in both halves.
        let a = aqa(0);
        assert_eq!(a, 0);
    }

    #[test]
    fn doorbell_offsets_dstrd_zero() {
        // Stride = 4 bytes.
        assert_eq!(sq_tail_doorbell_offset(0, 0), 0x1000);
        assert_eq!(cq_head_doorbell_offset(0, 0), 0x1004);
        assert_eq!(sq_tail_doorbell_offset(1, 0), 0x1008);
        assert_eq!(cq_head_doorbell_offset(1, 0), 0x100C);
    }

    #[test]
    fn doorbell_offsets_dstrd_two() {
        // Stride = 16 bytes.
        assert_eq!(sq_tail_doorbell_offset(0, 2), 0x1000);
        assert_eq!(cq_head_doorbell_offset(0, 2), 0x1010);
        assert_eq!(sq_tail_doorbell_offset(1, 2), 0x1020);
        assert_eq!(cq_head_doorbell_offset(1, 2), 0x1030);
    }

    #[test]
    fn cdw0_packs_opcode_and_cid() {
        let c = make_cdw0(0x02, 0xBEEF);
        assert_eq!(c & 0xFF, 0x02);
        assert_eq!((c >> 16) & 0xFFFF, 0xBEEF);
    }

    #[test]
    fn build_identify_controller_sets_cns_1() {
        let e = build_identify_controller(7, 0x1_0000);
        assert_eq!(e.cdw0 & 0xFF, OPC_IDENTIFY as u32);
        assert_eq!((e.cdw0 >> 16) & 0xFFFF, 7);
        assert_eq!(e.nsid, 0);
        assert_eq!(e.prp1, 0x1_0000);
        assert_eq!(e.cdw10, IDENTIFY_CNS_CONTROLLER);
        // PRP2 unused for a single-page Identify response.
        assert_eq!(e.prp2, 0);
    }

    #[test]
    fn build_identify_namespace_sets_cns_0_and_nsid() {
        let e = build_identify_namespace(8, 1, 0x2_0000);
        assert_eq!(e.cdw0 & 0xFF, OPC_IDENTIFY as u32);
        assert_eq!(e.nsid, 1);
        assert_eq!(e.cdw10, IDENTIFY_CNS_NAMESPACE);
    }

    #[test]
    fn build_create_iocq_polling_no_interrupts() {
        let e = build_create_iocq(1, IO_QID, IO_QUEUE_DEPTH, 0x3_0000, false);
        assert_eq!(e.cdw0 & 0xFF, OPC_CREATE_IOCQ as u32);
        assert_eq!(e.prp1, 0x3_0000);
        assert_eq!(e.cdw10 & 0xFFFF, IO_QID as u32);
        assert_eq!((e.cdw10 >> 16) & 0xFFFF, (IO_QUEUE_DEPTH - 1) as u32);
        // CDW11: PC=1, IEN=0.
        assert_eq!(e.cdw11 & 0x1, 1);
        assert_eq!((e.cdw11 >> 1) & 0x1, 0);
    }

    #[test]
    fn build_create_iocq_interrupt_enabled() {
        let e = build_create_iocq(1, IO_QID, IO_QUEUE_DEPTH, 0, true);
        assert_eq!((e.cdw11 >> 1) & 0x1, 1);
    }

    #[test]
    fn build_create_iosq_sets_cqid_and_qprio() {
        let e = build_create_iosq(2, IO_QID, IO_QUEUE_DEPTH, 0x4_0000, IO_QID);
        assert_eq!(e.cdw0 & 0xFF, OPC_CREATE_IOSQ as u32);
        assert_eq!(e.prp1, 0x4_0000);
        assert_eq!(e.cdw10 & 0xFFFF, IO_QID as u32);
        assert_eq!((e.cdw10 >> 16) & 0xFFFF, (IO_QUEUE_DEPTH - 1) as u32);
        assert_eq!(e.cdw11 & 0x1, 1);
        assert_eq!((e.cdw11 >> 1) & 0x3, 0b01);
        assert_eq!((e.cdw11 >> 16) & 0xFFFF, IO_QID as u32);
    }

    #[test]
    fn build_read_encodes_slba_and_nlb_zero_based() {
        let e = build_read(0x55, 1, 0x1234_5678_9ABC, 8, 0x5_0000, 0);
        assert_eq!(e.cdw0 & 0xFF, OPC_NVM_READ as u32);
        assert_eq!(e.nsid, 1);
        assert_eq!(e.prp1, 0x5_0000);
        assert_eq!(e.cdw10, 0x5678_9ABC);
        assert_eq!(e.cdw11, 0x0000_1234);
        // NLB encoded as zero-based.
        assert_eq!(e.cdw12 & 0xFFFF, 7);
    }

    #[test]
    fn build_read_nlb_zero_documented_as_one_lba() {
        // Wire convention: NLB is zero-based ⇒ wire value 0 means
        // "1 LBA". Passing `nlb=0` to `build_read` therefore produces
        // the same wire encoding as `nlb=1`. The kernel driver
        // (`drivers::nvme::NvmeController::read_one`) now rejects
        // `nlb=0` to avoid the silent-mis-issue trap; this test pins
        // the builder's behaviour so the driver-side guard is
        // independently testable.
        let zero = build_read(0, 1, 0, 0, 0x1000, 0);
        let one = build_read(0, 1, 0, 1, 0x1000, 0);
        assert_eq!(zero.cdw12 & 0xFFFF, 0);
        assert_eq!(one.cdw12 & 0xFFFF, 0);
        assert_eq!(zero.cdw12, one.cdw12, "nlb=0 must encode like nlb=1");
    }

    #[test]
    fn build_read_max_nlb_fits_in_low_16_bits() {
        // Regression: `cdw12 = nlb.saturating_sub(1) as u32` must not
        // overflow into the upper-16 control bits (which are reserved
        // for DSM hints, force-unit-access, etc.). With `nlb = u16::MAX`
        // the wire encoding is 0xFFFE, which fits in `[15:0]`.
        let e = build_read(0, 1, 0, u16::MAX, 0x1000, 0);
        assert_eq!(e.cdw12 & 0xFFFF, 0xFFFE);
        assert_eq!(e.cdw12 >> 16, 0, "upper bits of CDW12 must stay zero");
    }

    #[test]
    fn plan_prp_exact_page_boundary_at_4096() {
        // Regression: 4096 bytes is exactly one NVMe page → 1-page
        // path (prp1 only). 4097 bytes spills into a 2-page path.
        // Verify both transitions hold under `div_ceil`.
        let p1 = plan_prp(0x1000, 4096, 0x9000).unwrap();
        assert_eq!(p1.list_entries, 0);
        assert_eq!(p1.prp2, 0);

        let p2 = plan_prp(0x1000, 4097, 0x9000).unwrap();
        assert_eq!(p2.list_entries, 0);
        assert_eq!(p2.prp2, 0x2000); // page 1

        let p_n = plan_prp(0x1000, 8193, 0x9000).unwrap();
        assert_eq!(p_n.list_entries, 2);
        assert_eq!(p_n.prp2, 0x9000); // list pointer
    }

    #[test]
    fn build_read_with_prp2_list_pointer() {
        let e = build_read(1, 1, 0, 512, 0x10_0000, 0xAA_0000);
        assert_eq!(e.prp1, 0x10_0000);
        assert_eq!(e.prp2, 0xAA_0000);
        // 512-LBA read = 0x1FF on the wire.
        assert_eq!(e.cdw12 & 0xFFFF, 511);
    }

    #[test]
    fn plan_prp_single_page() {
        let p = plan_prp(0x1000, 4096, 0x9000).unwrap();
        assert_eq!(p.prp1, 0x1000);
        assert_eq!(p.prp2, 0);
        assert_eq!(p.list_entries, 0);
    }

    #[test]
    fn plan_prp_partial_page_still_one_page() {
        // 100 bytes still fits in one 4 KiB page.
        let p = plan_prp(0x1000, 100, 0x9000).unwrap();
        assert_eq!(p.list_entries, 0);
        assert_eq!(p.prp2, 0);
    }

    #[test]
    fn plan_prp_two_pages() {
        let p = plan_prp(0x1000, 8192, 0x9000).unwrap();
        assert_eq!(p.prp1, 0x1000);
        assert_eq!(p.prp2, 0x2000);
        assert_eq!(p.list_entries, 0);
    }

    #[test]
    fn plan_prp_two_pages_partial() {
        // 4097 bytes = 2 pages, fits in PRP1 + PRP2.
        let p = plan_prp(0x1000, 4097, 0x9000).unwrap();
        assert_eq!(p.prp2, 0x2000);
        assert_eq!(p.list_entries, 0);
    }

    #[test]
    fn plan_prp_three_pages_uses_list() {
        let p = plan_prp(0x1000, 4096 * 3, 0x9000).unwrap();
        assert_eq!(p.prp1, 0x1000);
        assert_eq!(p.prp2, 0x9000);
        assert_eq!(p.list_entries, 2);
    }

    #[test]
    fn plan_prp_max_512_pages_fit() {
        // 512 pages = 2 MiB — the operational ceiling the driver
        // actually uses (MAX_READ_BYTES). PRP1 holds page 0; the
        // remaining 511 pages live in the PRP list.
        let bytes = 512 * 4096;
        let p = plan_prp(0x1000, bytes, 0x9000).unwrap();
        assert_eq!(p.list_entries, 511);
    }

    #[test]
    fn plan_prp_max_planner_capacity_513_pages_fit() {
        // 513 pages is the absolute planner capacity: PRP1 + 512
        // list entries. Useful for any future caller that might want
        // to push past the driver's 2 MiB ceiling.
        let bytes = 513 * 4096;
        let p = plan_prp(0x1000, bytes, 0x9000).unwrap();
        assert_eq!(p.list_entries, 512);
    }

    #[test]
    fn plan_prp_rejects_unaligned_buffer() {
        assert!(plan_prp(0x1001, 4096, 0x9000).is_none());
    }

    #[test]
    fn plan_prp_rejects_unaligned_list() {
        assert!(plan_prp(0x1000, 4096, 0x9001).is_none());
    }

    #[test]
    fn plan_prp_rejects_too_large() {
        // 514 pages would need PRP1 + 513 list entries, but a 4 KiB
        // PRP list page only holds 512 entries — overflow.
        let bytes = 514 * 4096;
        assert!(plan_prp(0x1000, bytes, 0x9000).is_none());
    }

    #[test]
    fn plan_prp_rejects_zero() {
        assert!(plan_prp(0x1000, 0, 0x9000).is_none());
    }

    #[test]
    fn fill_prp_list_populates_pages_one_onward() {
        let plan = plan_prp(0x10_0000, 4 * 4096, 0x9000).unwrap();
        assert_eq!(plan.list_entries, 3);
        let mut list = [0u64; 512];
        fill_prp_list(0x10_0000, &mut list, &plan).unwrap();
        assert_eq!(list[0], 0x10_1000);
        assert_eq!(list[1], 0x10_2000);
        assert_eq!(list[2], 0x10_3000);
        // Untouched beyond list_entries.
        assert_eq!(list[3], 0);
    }

    #[test]
    fn fill_prp_list_noop_for_small_xfer() {
        let plan = plan_prp(0x10_0000, 4096, 0x9000).unwrap();
        let mut list = [0u64; 512];
        fill_prp_list(0x10_0000, &mut list, &plan).unwrap();
        assert!(list.iter().all(|&x| x == 0));
    }

    #[test]
    fn fill_prp_list_rejects_undersized_list() {
        let plan = plan_prp(0x10_0000, 4 * 4096, 0x9000).unwrap();
        let mut small = [0u64; 2];
        assert!(fill_prp_list(0x10_0000, &mut small, &plan).is_err());
    }

    #[test]
    fn completion_entry_phase_and_status_split() {
        let mut cqe = CompletionEntry {
            status: (0x42 << 1) | 1,
            ..Default::default()
        }; // phase=1, status=0x42
        assert!(cqe.phase());
        assert_eq!(cqe.status_no_phase(), 0x42);
        assert!(cqe.is_error());

        cqe.status = 1; // phase=1, status=0
        assert!(cqe.phase());
        assert_eq!(cqe.status_no_phase(), 0);
        assert!(!cqe.is_error());
    }

    #[test]
    fn completion_entry_phase_zero() {
        let cqe = CompletionEntry {
            status: 0,
            ..Default::default()
        }; // phase=0, success
        assert!(!cqe.phase());
        assert!(!cqe.is_error());
    }

    #[test]
    fn decode_identify_namespace_4k_lba() {
        // Build a 4 KiB Identify Namespace response with NSZE=1<<30
        // (≈ 4 TiB at 4 KiB LBAs) and one LBAF entry at index 0 with
        // LBADS=12 (2^12 = 4096-byte LBAs).
        let mut buf = [0u8; 4096];
        let nsze: u64 = 1 << 30;
        buf[0..8].copy_from_slice(&nsze.to_le_bytes());
        // FLBAS = index 0.
        buf[26] = 0;
        // LBAF[0].dword0 = LBADS=12 in bits [23:16].
        let lbaf_dw0: u32 = 12u32 << 16;
        buf[128..132].copy_from_slice(&lbaf_dw0.to_le_bytes());

        let info = decode_identify_namespace(&buf).unwrap();
        assert_eq!(info.nlba, 1u64 << 30);
        assert_eq!(info.lba_size, 4096);
    }

    #[test]
    fn decode_identify_namespace_512_lba_from_lbaf_index_1() {
        let mut buf = [0u8; 4096];
        let nsze: u64 = 1 << 20;
        buf[0..8].copy_from_slice(&nsze.to_le_bytes());
        // FLBAS index = 1 → use LBAF[1].
        buf[26] = 1;
        // LBAF[1] starts at offset 128 + 16.
        let lbaf_dw0: u32 = 9u32 << 16; // LBADS=9 → 512 B
        buf[128 + 16..128 + 16 + 4].copy_from_slice(&lbaf_dw0.to_le_bytes());

        let info = decode_identify_namespace(&buf).unwrap();
        assert_eq!(info.nlba, 1u64 << 20);
        assert_eq!(info.lba_size, 512);
    }

    #[test]
    fn decode_identify_namespace_rejects_truncated_buffer() {
        let buf = [0u8; 16];
        assert!(decode_identify_namespace(&buf).is_none());
    }

    // ── Data-drive selection ────────────────────────────────────────

    fn probe(bdf: (u8, u8, u8), nlba: u64, lba_size: u32) -> NamespaceProbe {
        NamespaceProbe {
            pci_bus: bdf.0,
            pci_device: bdf.1,
            pci_function: bdf.2,
            nlba,
            lba_size,
        }
    }

    #[test]
    fn pick_largest_namespace_empty_returns_none() {
        assert_eq!(pick_largest_namespace(&[]), None);
    }

    #[test]
    fn pick_largest_namespace_single_probe_returns_zero() {
        let p = [probe((0x42, 0, 0), 1024, 512)];
        assert_eq!(pick_largest_namespace(&p), Some(0));
    }

    #[test]
    fn pick_largest_namespace_picks_largest_capacity() {
        // Mirror the Cherry box: two 240 GB system drives + two 3.2 TB
        // data drives. The selector must pick a 3.2 TB drive.
        let sys_a = probe((0x01, 0, 0), 240 * 1024 * 1024 * 1024 / 512, 512);
        let sys_b = probe((0x01, 1, 0), 240 * 1024 * 1024 * 1024 / 512, 512);
        let data_a = probe((0x41, 0, 0), 3_200_000_000_000 / 512, 512);
        let data_b = probe((0x41, 1, 0), 3_200_000_000_000 / 512, 512);
        let probes = [sys_a, sys_b, data_a, data_b];
        let pick = pick_largest_namespace(&probes).unwrap();
        // Expect index 2 (first 3.2 TB drive — ties between data_a and
        // data_b resolve to the earlier index).
        assert_eq!(pick, 2);
        assert_eq!(probes[pick].total_bytes(), 3_200_000_000_000);
    }

    #[test]
    fn pick_largest_namespace_ties_pick_first() {
        // Two drives of identical capacity → deterministic choice
        // (the earlier-indexed probe wins).
        let a = probe((0x01, 0, 0), 1_000_000, 4096);
        let b = probe((0x01, 1, 0), 1_000_000, 4096);
        assert_eq!(pick_largest_namespace(&[a, b]), Some(0));
    }

    #[test]
    fn pick_largest_handles_4k_lba_drives() {
        // 4 KiB LBA drives report fewer LBAs but the same byte total.
        // Selection must compare bytes, not LBAs.
        let small_4k = probe((0x10, 0, 0), 1_000_000, 4096); // 4 GB
        let big_512 = probe((0x20, 0, 0), 2_000_000, 512); // ~1 GB
        let probes = [small_4k, big_512];
        assert_eq!(pick_largest_namespace(&probes), Some(0));
    }

    #[test]
    fn namespace_probe_total_bytes_saturates() {
        // Pathological input — verifies the saturating multiplication.
        let p = probe((0, 0, 0), u64::MAX, u32::MAX);
        assert_eq!(p.total_bytes(), u64::MAX);
    }

    #[test]
    fn looks_like_gguf_accepts_correct_magic() {
        // 'G','G','U','F' little-endian = 0x46555447 ← wait, "GGUF" =
        // 0x47 0x47 0x55 0x46 → LE u32 = 0x46554747.
        let buf = [0x47, 0x47, 0x55, 0x46, 0x00, 0x01];
        assert!(looks_like_gguf(&buf));
    }

    #[test]
    fn looks_like_gguf_rejects_wrong_magic() {
        let zeros = [0u8; 16];
        let ones = [0xFFu8; 16];
        let elf = [0x7F, b'E', b'L', b'F', 0, 0];
        assert!(!looks_like_gguf(&zeros));
        assert!(!looks_like_gguf(&ones));
        assert!(!looks_like_gguf(&elf));
    }

    #[test]
    fn looks_like_gguf_rejects_short_buffer() {
        assert!(!looks_like_gguf(&[]));
        assert!(!looks_like_gguf(&[0x47]));
        assert!(!looks_like_gguf(&[0x47, 0x47]));
        assert!(!looks_like_gguf(&[0x47, 0x47, 0x55])); // 3 bytes — still short
    }

    #[test]
    fn looks_like_model_container_accepts_silicatepack() {
        assert!(looks_like_model_container(b"SILM\x01\0\0\0"));
        assert!(looks_like_model_container(b"GGUF\x03\0\0\0"));
    }

    #[test]
    fn looks_like_model_container_rejects_unknown_or_short() {
        assert!(!looks_like_model_container(&[]));
        assert!(!looks_like_model_container(b"SIL"));
        assert!(!looks_like_model_container(b"ELF!"));
    }

    #[test]
    fn gguf_magic_le_matches_kernel_constant() {
        // Cross-check: the wire-types crate's constant must equal the
        // one the kernel's model_loader uses (0x46554747). If either
        // drifts, the validation walks past the wrong bytes.
        assert_eq!(GGUF_MAGIC_LE, 0x4655_4747);
    }

    #[test]
    fn spin_budget_floors_to_minimum() {
        // Regression: pre-audit, `wait_csts_*` silently ignored
        // `timeout_ms`. `spin_budget` now honours the request while
        // applying a floor so small CAP.TO values still get the
        // historical 200 M-iteration runway.
        assert_eq!(spin_budget(0), SPIN_FLOOR);
        assert_eq!(spin_budget(1), SPIN_FLOOR); // 1 ms × 1M = 1M < floor
        assert_eq!(spin_budget(199), SPIN_FLOOR); // 199 ms × 1M < 200M
    }

    #[test]
    fn spin_budget_scales_with_timeout() {
        // Above the floor, the conversion is linear.
        assert_eq!(spin_budget(10_000), 10_000_u64 * SPIN_PER_MS);
        assert_eq!(spin_budget(60_000), 60_000_u64 * SPIN_PER_MS);
        // Saturating: `u32::MAX` ms ≈ 49 days, must not overflow.
        let huge = spin_budget(u32::MAX);
        assert!(huge >= SPIN_FLOOR);
        assert!(huge >= 4_000_000_000_000); // ≥ 4 T iterations
    }
}
