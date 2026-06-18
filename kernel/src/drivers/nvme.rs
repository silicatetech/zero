// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal NVMe 1.x driver — read-only, polling mode.
//!
//! Scope: the kernel needs to stream a single very large file (the
//! Kimi K2.6 GGUF, ≈ 584 GiB) from a directly-attached NVMe namespace
//! into the weight arena at boot. There is no filesystem, no write
//! path, no interrupts, no namespace management, no I/O queue sharing
//! across CPUs. Polling-mode bulk sequential read into a caller-
//! provided buffer is the whole product.
//!
//! ## Layering
//!
//! Two layers cooperate here:
//!
//! 1. **Wire types** live in the [`zero_nvme`] crate — register
//!    offsets, CAP/CC/CSTS decoders, SQE/CQE struct layouts, command
//!    builders, the PRP planner, the Identify-Namespace decoder.
//!    All pure data, no MMIO, no `unsafe`. Fully unit-tested on the
//!    host via `cargo test -p zero-nvme`.
//!
//! 2. **Hardware path** lives in this file. [`NvmeController`] holds
//!    an MMIO-mapped BAR0 plus two queue pairs (admin + I/O), drives
//!    controller bring-up (CC.EN → CSTS.RDY), issues Identify /
//!    Create-I/O-Queue on the admin queue, then loops Read commands
//!    on the I/O queue to fill a destination buffer.
//!
//! ## NVMe spec reference
//!
//! Section numbers in comments are from NVM Express Base Spec 1.4 —
//! the relevant subset has been stable since 1.0 so the driver works
//! against any modern controller (we test against the Solidigm Gen5
//! drives in the Cherry Server EPYC 9575F box).

// Driver is foundational infrastructure: the boot path that
// orchestrates NVMe loading lives on a later branch, so several
// public entry points (discover/bind/bulk_read) are intentionally
// unused at the bin-crate level today. The wire-type layer that
// gets exercised right now lives in the zero-nvme crate.
#![allow(dead_code)]

use core::fmt::Write;
use core::ptr;
use core::sync::atomic::{compiler_fence, Ordering};

use zero_nvme::{
    aqa, build_create_iocq, build_create_iosq, build_identify_controller, build_identify_namespace,
    build_read, cc_enable, cq_head_doorbell_offset, decode_identify_namespace, plan_prp,
    sq_tail_doorbell_offset, Capabilities, CompletionEntry, SubmissionEntry, ADMIN_QUEUE_DEPTH,
    CSTS_CFS, CSTS_RDY, IO_QID, IO_QUEUE_DEPTH, NVME_PAGE_SIZE, REG_ACQ, REG_AQA, REG_ASQ,
    REG_CAP, REG_CC, REG_CSTS,
};

use crate::arch::serial::Serial;
use crate::arch::x86_64::pcie::{config_read16, config_write16, read_bar, PciDevice, PciScan};
use crate::memory::{map_mmio, virt_to_phys_pa, KERNEL_ARENA};
use crate::model_loader::{model_magic_from_bytes, ModelMagic};

/// Largest single Read command we issue, in bytes. Sized so the
/// resulting PRP list fits in one 4 KiB page (max 512 entries: 1 page
/// in PRP1 + 511 in the list = 512 pages = 2 MiB). Anything larger
/// gets split by the bulk-read loop.
pub const MAX_READ_BYTES: usize = 2 * 1024 * 1024;

/// Cushion (in milliseconds) added on top of CAP.TO when waiting for
/// CSTS.RDY. Catches firmware bugs without hanging boot indefinitely.
const RDY_TIMEOUT_CUSHION_MS: u32 = 1000;

/// Cap on the per-command polling window for non-bring-up commands
/// (Identify, Create-Queue, Read). 5 s is well past worst-case
/// observed latency on Gen5 NVMe; chosen to surface real hangs.
const COMMAND_TIMEOUT_MS: u32 = 5_000;

// ---------------------------------------------------------------------
//  Errors
// ---------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NvmeError {
    /// No NVMe controller was found in the PCI scan.
    ControllerNotFound,
    /// BAR0 was unpopulated or unmappable.
    Bar0Invalid,
    /// MMIO mapping failed (typically arena out-of-space).
    MmioMapFailed,
    /// CAP reported a minimum memory page size larger than 4 KiB. The
    /// kernel only supports 4 KiB pages today.
    UnsupportedPageSize,
    /// Controller failed to leave the disabled state within the
    /// timeout reported by CAP.TO.
    ControllerEnableTimeout,
    /// Controller failed to clear CSTS.RDY after CC.EN=0.
    ControllerDisableTimeout,
    /// Controller reported a Fatal Status (CSTS.CFS=1).
    ControllerFatal,
    /// An admin or I/O command completed with a non-zero status code.
    CommandFailed { status: u16 },
    /// A command did not complete within the configured polling
    /// window. Indicates a stuck queue or a hardware problem.
    CommandTimeout,
    /// Caller asked for more bytes than the namespace contains.
    OutOfRange,
    /// Arena could not satisfy a queue allocation.
    OutOfMemory,
    /// virt→phys translation failed for a buffer the driver was
    /// asked to use as a DMA target.
    BadDmaBuffer,
    /// Identify Namespace returned an unparseable response.
    BadIdentify,
}

// ---------------------------------------------------------------------
//  PCI scan filter
// ---------------------------------------------------------------------

// The PCI enumeration leaf (`is_nvme`, `ControllerList`,
// `MAX_CONTROLLERS`) is pure — it depends only on `PciDevice` and the
// `zero_nvme` spec constants, no MMIO and no kernel arena. It lives
// in `nvme_probe` so the host test harness (`crates/kernel-tests`) can
// cover it; re-exported here so `crate::drivers::nvme::<item>` keeps
// resolving for the kernel and its callers.
#[path = "nvme_probe.rs"]
mod nvme_probe;
pub use nvme_probe::{is_nvme, ControllerList, MAX_CONTROLLERS};

/// Return every NVMe controller in `scan`, in enumeration order. Pure
/// helper — does not touch hardware.
pub fn nvme_controllers(scan: &PciScan) -> impl Iterator<Item = &PciDevice> {
    scan.iter().filter(|d| is_nvme(d))
}

/// Locate every NVMe controller in `scan` and return them as a
/// bounded list. Two Gen5 drives on the Cherry box → two entries.
pub fn discover(scan: &PciScan) -> ControllerList {
    let mut out = ControllerList::new();
    for dev in scan.iter() {
        if is_nvme(dev) {
            out.push(*dev);
        }
    }
    out
}

// ---------------------------------------------------------------------
//  Volatile MMIO helpers
// ---------------------------------------------------------------------

#[inline(always)]
unsafe fn read32(base: *mut u8, off: u32) -> u32 {
    ptr::read_volatile(base.add(off as usize) as *const u32)
}

#[inline(always)]
unsafe fn write32(base: *mut u8, off: u32, v: u32) {
    ptr::write_volatile(base.add(off as usize) as *mut u32, v)
}

#[inline(always)]
unsafe fn read64(base: *mut u8, off: u32) -> u64 {
    // PCIe root complexes commonly drop 8-byte MMIO; issue two 32-bit
    // reads instead.
    let lo = read32(base, off) as u64;
    let hi = read32(base, off + 4) as u64;
    lo | (hi << 32)
}

#[inline(always)]
unsafe fn write64(base: *mut u8, off: u32, v: u64) {
    write32(base, off, v as u32);
    write32(base, off + 4, (v >> 32) as u32);
}

// ---------------------------------------------------------------------
//  Controller
// ---------------------------------------------------------------------

/// One queue pair — submission + completion ring.
struct QueuePair {
    qid: u16,
    depth: u16,
    sq_va: *mut SubmissionEntry,
    sq_phys: u64,
    cq_va: *mut CompletionEntry,
    cq_phys: u64,
    sq_tail: u16,
    cq_head: u16,
    /// CQ phase tag the controller will set on the *next* entry it
    /// writes. Starts at 1; flips each wrap.
    phase: bool,
}

impl QueuePair {
    fn alloc(qid: u16, depth: u16) -> Result<Self, NvmeError> {
        let sq = alloc_dma_page()?;
        let cq = alloc_dma_page()?;
        // Zero the CQ so the phase tag starts at 0 — controller will
        // flip it to 1 on its first completion.
        unsafe {
            ptr::write_bytes(cq.va as *mut u8, 0, NVME_PAGE_SIZE);
        }
        Ok(Self {
            qid,
            depth,
            sq_va: sq.va as *mut SubmissionEntry,
            sq_phys: sq.phys,
            cq_va: cq.va as *mut CompletionEntry,
            cq_phys: cq.phys,
            sq_tail: 0,
            cq_head: 0,
            phase: true,
        })
    }

    unsafe fn write_sqe(&mut self, e: &SubmissionEntry) {
        ptr::write_volatile(self.sq_va.add(self.sq_tail as usize), *e);
        self.sq_tail = (self.sq_tail + 1) % self.depth;
    }

    unsafe fn read_cqe(&self) -> CompletionEntry {
        ptr::read_volatile(self.cq_va.add(self.cq_head as usize))
    }

    fn advance_cq_head(&mut self) {
        self.cq_head = (self.cq_head + 1) % self.depth;
        if self.cq_head == 0 {
            self.phase = !self.phase;
        }
    }
}

/// Per-namespace info.
#[derive(Debug, Copy, Clone, Default)]
pub struct NamespaceInfo {
    pub nsid: u32,
    pub nlba: u64,
    pub lba_size: u32,
}

/// A live NVMe controller — admin and I/O queues created, namespace
/// information cached.
pub struct NvmeController {
    bar0: *mut u8,
    cap: Capabilities,
    adminq: QueuePair,
    ioq: QueuePair,
    next_cid: u16,
    ns: NamespaceInfo,
    /// PRP list scratch page (lazily allocated on first transfer that
    /// needs it).
    prp_list_va: *mut u8,
    prp_list_phys: u64,
}

// MMIO base is volatile; arena memory is never freed. Only one CPU
// drives this controller — hand it out behind an outer Mutex.
unsafe impl Send for NvmeController {}

impl NvmeController {
    /// Discover, MMIO-map, and bring up the first NVMe controller
    /// present in `scan`.
    pub fn bind_first(scan: &PciScan) -> Result<Self, NvmeError> {
        let dev = nvme_controllers(scan)
            .next()
            .ok_or(NvmeError::ControllerNotFound)?;
        Self::bind(dev)
    }

    /// Same as `bind_first`, but for an explicit device — used when
    /// the model loader parallelises across two drives.
    pub fn bind(dev: &PciDevice) -> Result<Self, NvmeError> {
        // Enable Memory Space + Bus Master in PCI command register.
        // Without bus mastering the controller cannot DMA into our
        // queues or data buffers — and many BIOSes leave it off.
        unsafe {
            let cmd = config_read16(dev.bus, dev.device, dev.function, 0x04);
            let new_cmd = cmd | 0x0006;
            if new_cmd != cmd {
                config_write16(dev.bus, dev.device, dev.function, 0x04, new_cmd);
            }
        }

        let bar0_phys = read_bar(dev, 0).ok_or(NvmeError::Bar0Invalid)?;

        // BAR0 mapping size must cover (a) the controller register
        // block at offsets 0x00..0x1000 and (b) every doorbell we will
        // ever address. Doorbell stride = `4 << CAP.DSTRD` bytes; for
        // the maximum spec-allowed DSTRD = 15 (stride 128 KiB) and two
        // queues (admin + IO), the highest doorbell sits at
        // 0x1000 + (2*qid+1) * stride = 0x1000 + 3 * 128 KiB ≈ 384 KiB.
        //
        // Real controllers we've ever seen ship DSTRD = 0 (stride 4),
        // but the spec leaves room and we don't want to silently
        // misroute doorbells if a future drive picks a larger stride.
        // 512 KiB covers all spec-legal values without breaking the
        // bank — it's still well within a single PML4-arena allocation.
        const BAR0_MMIO_BYTES: usize = 512 * 1024;
        let bar0 =
            map_mmio(bar0_phys, BAR0_MMIO_BYTES).map_err(|_| NvmeError::MmioMapFailed)? as *mut u8;

        let cap_raw = unsafe { read64(bar0, REG_CAP) };
        let cap = Capabilities::from_raw(cap_raw);
        if cap.min_page_size() > NVME_PAGE_SIZE {
            return Err(NvmeError::UnsupportedPageSize);
        }
        // Sanity: even after the generous BAR0 mapping, refuse to
        // proceed if the controller's highest doorbell would land
        // outside it. Catches future spec extensions (DSTRD > 15).
        let highest_doorbell =
            cq_head_doorbell_offset(IO_QID, cap.dstrd) as usize + core::mem::size_of::<u32>();
        if highest_doorbell > BAR0_MMIO_BYTES {
            let _ = writeln!(
                Serial,
                "NVMe: BAR0 map ({} B) too small for DSTRD={} doorbell at offset 0x{:x}",
                BAR0_MMIO_BYTES, cap.dstrd, highest_doorbell
            );
            return Err(NvmeError::UnsupportedPageSize);
        }

        let _ = writeln!(
            Serial,
            "NVMe: bind {:02x}:{:02x}.{} BAR0=0x{:x} CAP=0x{:016x} (mqes={}, dstrd={}, mpsmin={}, to={}ms)",
            dev.bus, dev.device, dev.function,
            bar0_phys, cap_raw,
            cap.mqes, cap.dstrd, cap.mpsmin, cap.enable_timeout_ms(),
        );

        // Disable controller if it was running.
        unsafe {
            let cc = read32(bar0, REG_CC);
            if cc & 1 != 0 {
                write32(bar0, REG_CC, cc & !1);
                wait_csts_clear(
                    bar0,
                    CSTS_RDY,
                    cap.enable_timeout_ms() + RDY_TIMEOUT_CUSHION_MS,
                )
                .map_err(|_| NvmeError::ControllerDisableTimeout)?;
            }
        }

        // Admin queue.
        let adminq = QueuePair::alloc(0, ADMIN_QUEUE_DEPTH)?;
        unsafe {
            write32(bar0, REG_AQA, aqa(ADMIN_QUEUE_DEPTH));
            write64(bar0, REG_ASQ, adminq.sq_phys);
            write64(bar0, REG_ACQ, adminq.cq_phys);
        }

        // Enable.
        unsafe {
            write32(bar0, REG_CC, cc_enable(cap.mpsmin));
            wait_csts_set(
                bar0,
                CSTS_RDY,
                cap.enable_timeout_ms() + RDY_TIMEOUT_CUSHION_MS,
            )
            .map_err(|_| NvmeError::ControllerEnableTimeout)?;
        }

        let mut ctrl = Self {
            bar0,
            cap,
            adminq,
            ioq: QueuePair {
                qid: IO_QID,
                depth: IO_QUEUE_DEPTH,
                sq_va: ptr::null_mut(),
                sq_phys: 0,
                cq_va: ptr::null_mut(),
                cq_phys: 0,
                sq_tail: 0,
                cq_head: 0,
                phase: true,
            },
            next_cid: 1,
            ns: NamespaceInfo::default(),
            prp_list_va: ptr::null_mut(),
            prp_list_phys: 0,
        };

        // Identify Controller (sanity check; we don't need the fields
        // — the per-namespace identify gives us the numbers).
        let ident_buf = alloc_dma_page()?;
        ctrl.admin_identify_controller(ident_buf.phys)?;

        // Create I/O CQ then I/O SQ (CQ first per spec §5.4).
        let ioq = QueuePair::alloc(IO_QID, IO_QUEUE_DEPTH)?;
        ctrl.ioq = ioq;
        ctrl.admin_create_iocq()?;
        ctrl.admin_create_iosq()?;

        // Identify Namespace 1 — that's where the Cherry deployment
        // writes the model. NSID=0 is reserved.
        ctrl.ns.nsid = 1;
        ctrl.admin_identify_namespace(ctrl.ns.nsid, ident_buf.phys)?;
        let id_buf =
            unsafe { core::slice::from_raw_parts(ident_buf.va as *const u8, NVME_PAGE_SIZE) };
        let info = decode_identify_namespace(id_buf).ok_or(NvmeError::BadIdentify)?;
        ctrl.ns.nlba = info.nlba;
        ctrl.ns.lba_size = info.lba_size;

        let _ = writeln!(
            Serial,
            "NVMe: namespace {} → {} LBAs × {} B = {} MiB",
            ctrl.ns.nsid,
            ctrl.ns.nlba,
            ctrl.ns.lba_size,
            ctrl.ns.nlba.saturating_mul(ctrl.ns.lba_size as u64) / (1024 * 1024),
        );

        Ok(ctrl)
    }

    pub fn namespace(&self) -> &NamespaceInfo {
        &self.ns
    }

    pub fn capabilities(&self) -> &Capabilities {
        &self.cap
    }

    /// Largest LBA count a single Read command can satisfy with our
    /// "PRP list fits in one page" assumption.
    pub fn max_read_lbas(&self) -> u32 {
        (MAX_READ_BYTES / self.ns.lba_size as usize) as u32
    }

    /// Read `byte_count` bytes starting at logical block `slba` into
    /// the caller-provided buffer at virtual address `dst_va`. The
    /// buffer must be 4 KiB-aligned; every page within it must be
    /// mapped read+write at the time of the call.
    ///
    /// The transfer is broken into `MAX_READ_BYTES` chunks; each
    /// chunk issues one NVMe Read and waits for completion before
    /// queuing the next. Throughput is therefore one round-trip per
    /// 2 MiB — sufficient for the 584 GiB Kimi K2.6 load at well
    /// under a minute, while keeping the polling loop trivial.
    pub fn bulk_read(
        &mut self,
        slba: u64,
        byte_count: usize,
        dst_va: u64,
    ) -> Result<(), NvmeError> {
        if byte_count == 0 {
            return Ok(());
        }
        if self.ns.lba_size == 0 {
            return Err(NvmeError::BadIdentify);
        }
        let lba_size = self.ns.lba_size as usize;
        if dst_va & (NVME_PAGE_SIZE as u64 - 1) != 0 {
            return Err(NvmeError::BadDmaBuffer);
        }
        if byte_count % lba_size != 0 {
            return Err(NvmeError::OutOfRange);
        }
        // Defensive: NVMe spec allows arbitrarily large LBAs, but the
        // bulk-read loop quantises each chunk to `lba_size` and caps
        // chunk bytes at MAX_READ_BYTES. If a namespace ever reports an
        // LBA larger than that, `chunk_lbas` would round down to 0 and
        // the loop would advance `cur_va` while leaving `cur_slba`
        // stuck — re-reading the same block forever and corrupting the
        // destination. Reject up front so the failure is observable.
        if lba_size > MAX_READ_BYTES {
            return Err(NvmeError::OutOfRange);
        }
        let total_lbas = (byte_count / lba_size) as u64;
        if slba.saturating_add(total_lbas) > self.ns.nlba {
            return Err(NvmeError::OutOfRange);
        }

        self.ensure_prp_list()?;

        let mut remaining = byte_count;
        let mut cur_slba = slba;
        let mut cur_va = dst_va;

        while remaining > 0 {
            let chunk = remaining.min(MAX_READ_BYTES);
            let chunk_lbas_usize = chunk / lba_size;
            // With `lba_size > MAX_READ_BYTES` already rejected and
            // `byte_count % lba_size == 0`, chunk is always a positive
            // multiple of lba_size — chunk_lbas_usize ≥ 1. The
            // debug_assert documents that invariant without paying for
            // a runtime check in release builds.
            debug_assert!(chunk_lbas_usize > 0);
            let chunk_lbas = chunk_lbas_usize as u16;
            self.read_one(cur_slba, chunk_lbas, cur_va)?;
            cur_slba += chunk_lbas as u64;
            cur_va += chunk as u64;
            remaining -= chunk;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    //  Internals
    // ------------------------------------------------------------------

    fn allocate_cid(&mut self) -> u16 {
        // Spec requires CID uniqueness among in-flight commands; we
        // only ever have one in flight (polling mode). Skip 0 because
        // it's a common "no command" sentinel.
        let c = self.next_cid;
        self.next_cid = self.next_cid.wrapping_add(1);
        if self.next_cid == 0 {
            self.next_cid = 1;
        }
        c
    }

    fn ensure_prp_list(&mut self) -> Result<(), NvmeError> {
        if !self.prp_list_va.is_null() {
            return Ok(());
        }
        let page = alloc_dma_page()?;
        self.prp_list_va = page.va as *mut u8;
        self.prp_list_phys = page.phys;
        Ok(())
    }

    fn submit_admin(&mut self, sqe: &SubmissionEntry) -> Result<CompletionEntry, NvmeError> {
        self.submit(true, sqe)
    }

    fn submit_io(&mut self, sqe: &SubmissionEntry) -> Result<CompletionEntry, NvmeError> {
        self.submit(false, sqe)
    }

    fn submit(&mut self, admin: bool, sqe: &SubmissionEntry) -> Result<CompletionEntry, NvmeError> {
        let dstrd = self.cap.dstrd;
        let (qid, sq_doorbell) = if admin {
            (0u16, sq_tail_doorbell_offset(0, dstrd))
        } else {
            (IO_QID, sq_tail_doorbell_offset(IO_QID, dstrd))
        };
        let cq_doorbell = cq_head_doorbell_offset(qid, dstrd);

        unsafe {
            let q = if admin {
                &mut self.adminq
            } else {
                &mut self.ioq
            };
            q.write_sqe(sqe);
            // SQE must be globally visible before the doorbell write
            // hands ownership to the device.
            compiler_fence(Ordering::SeqCst);
            write32(self.bar0, sq_doorbell, q.sq_tail as u32);
        }

        let cqe = self.poll_completion(admin, COMMAND_TIMEOUT_MS)?;
        let new_head = if admin {
            self.adminq.cq_head
        } else {
            self.ioq.cq_head
        };
        unsafe {
            write32(self.bar0, cq_doorbell, new_head as u32);
        }
        if cqe.is_error() {
            return Err(NvmeError::CommandFailed {
                status: cqe.status_no_phase(),
            });
        }
        Ok(cqe)
    }

    fn poll_completion(
        &mut self,
        admin: bool,
        timeout_ms: u32,
    ) -> Result<CompletionEntry, NvmeError> {
        let _ = timeout_ms;
        let q = if admin {
            &mut self.adminq
        } else {
            &mut self.ioq
        };
        // No calibrated millisecond timer at boot, so we bound the
        // spin count generously and rely on the CSTS.CFS check to
        // catch truly stuck hardware.
        let mut spin: u64 = 0;
        const SPIN_LIMIT: u64 = 1_000_000_000;
        loop {
            let cqe = unsafe { q.read_cqe() };
            if cqe.phase() == q.phase {
                q.advance_cq_head();
                return Ok(cqe);
            }
            if spin & 0xFFFF == 0 {
                let csts = unsafe { read32(self.bar0, REG_CSTS) };
                if csts & CSTS_CFS != 0 {
                    return Err(NvmeError::ControllerFatal);
                }
            }
            spin += 1;
            if spin >= SPIN_LIMIT {
                return Err(NvmeError::CommandTimeout);
            }
            core::hint::spin_loop();
        }
    }

    fn admin_identify_controller(&mut self, data_phys: u64) -> Result<(), NvmeError> {
        let cid = self.allocate_cid();
        let sqe = build_identify_controller(cid, data_phys);
        self.submit_admin(&sqe).map(|_| ())
    }

    fn admin_identify_namespace(&mut self, nsid: u32, data_phys: u64) -> Result<(), NvmeError> {
        let cid = self.allocate_cid();
        let sqe = build_identify_namespace(cid, nsid, data_phys);
        self.submit_admin(&sqe).map(|_| ())
    }

    fn admin_create_iocq(&mut self) -> Result<(), NvmeError> {
        let cid = self.allocate_cid();
        let sqe = build_create_iocq(cid, self.ioq.qid, self.ioq.depth, self.ioq.cq_phys, false);
        self.submit_admin(&sqe).map(|_| ())
    }

    fn admin_create_iosq(&mut self) -> Result<(), NvmeError> {
        let cid = self.allocate_cid();
        let sqe = build_create_iosq(
            cid,
            self.ioq.qid,
            self.ioq.depth,
            self.ioq.sq_phys,
            self.ioq.qid,
        );
        self.submit_admin(&sqe).map(|_| ())
    }

    fn read_one(&mut self, slba: u64, nlb: u16, dst_va: u64) -> Result<(), NvmeError> {
        // NLB on the wire is 0-based (0 = "read 1 LBA"), so the builder
        // saturating-subs 1 from `nlb`. If a caller passes `nlb == 0`
        // we'd silently issue a 1-LBA read against the wrong slba —
        // observable corruption. Reject loudly.
        if nlb == 0 {
            return Err(NvmeError::OutOfRange);
        }
        let bytes = (nlb as usize) * self.ns.lba_size as usize;
        let buf_phys = virt_to_phys_pa(dst_va).ok_or(NvmeError::BadDmaBuffer)?;
        if buf_phys & (NVME_PAGE_SIZE as u64 - 1) != 0 {
            return Err(NvmeError::BadDmaBuffer);
        }
        let plan = plan_prp(buf_phys, bytes, self.prp_list_phys).ok_or(NvmeError::BadDmaBuffer)?;

        // Fill PRP list — pages 1..N. Walking virt→phys per page lets
        // us tolerate buffers that are physically discontiguous; the
        // walk cost is negligible vs DMA itself.
        if plan.list_entries > 0 {
            let list_slot = unsafe {
                core::slice::from_raw_parts_mut(self.prp_list_va as *mut u64, NVME_PAGE_SIZE / 8)
            };
            for i in 0..plan.list_entries {
                let va = dst_va + ((i + 1) as u64) * NVME_PAGE_SIZE as u64;
                let phys = virt_to_phys_pa(va).ok_or(NvmeError::BadDmaBuffer)?;
                list_slot[i] = phys;
            }
        }

        let cid = self.allocate_cid();
        let sqe = build_read(cid, self.ns.nsid, slba, nlb, plan.prp1, plan.prp2);
        self.submit_io(&sqe).map(|_| ())
    }
}

// ---------------------------------------------------------------------
//  Multi-NVMe data-drive selection
// ---------------------------------------------------------------------

/// A controller that has finished bring-up plus the [`NamespaceProbe`]
/// the selector used to decide it was the largest. The probe carries
/// the PCI BDF so callers can log which slot they ended up using.
pub struct ProbedController {
    pub controller: NvmeController,
    pub probe: zero_nvme::NamespaceProbe,
}

/// Diagnostic snapshot of a probe failure, used so the model-loader can
/// surface "we found N controllers but couldn't bring up any of them"
/// instead of a bare `ControllerNotFound`.
#[derive(Debug, Copy, Clone)]
pub struct ProbeFailure {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
    pub err: NvmeError,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ModelDiskMagic {
    Gguf,
    Smodel,
    Other,
}

impl ModelDiskMagic {
    pub const fn is_supported(self) -> bool {
        matches!(self, Self::Gguf | Self::Smodel)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Gguf => "GGUF",
            Self::Smodel => "SILM",
            Self::Other => "unknown",
        }
    }
}

/// Walk every NVMe controller the PCI scan exposes, bring each one up
/// far enough to capture its [`NamespaceProbe`], and return the one
/// whose namespace has the largest byte capacity. On the Cherry
/// EPYC 9575F box this reliably picks one of the 3.2 TB Solidigm Gen5
/// data drives over the smaller system drives because **byte capacity
/// is the single distinguishing dimension** — the 1.5 GiB Zero
/// image leaves the system drives well under 240 GB, while the data
/// drives report ~3.2 TB.
///
/// Discarded controllers stay hardware-enabled (the arena-allocated
/// queue pages are never freed in V3); functionally that's a no-op
/// because nothing else uses them during boot. If `errors` is `Some`,
/// per-device probe failures are appended so the caller can include
/// them in a diagnostic banner.
pub fn bind_largest_data_drive(
    scan: &PciScan,
    mut errors: Option<&mut [ProbeFailure]>,
    mut errors_len: Option<&mut usize>,
) -> Result<ProbedController, NvmeError> {
    let mut best: Option<ProbedController> = None;
    let mut total_found: usize = 0;

    for dev in nvme_controllers(scan) {
        total_found += 1;
        let _ = writeln!(
            Serial,
            "NVMe probe: scanning {:02x}:{:02x}.{} vendor=0x{:04x} device=0x{:04x}",
            dev.bus, dev.device, dev.function, dev.vendor_id, dev.device_id
        );
        match NvmeController::bind(dev) {
            Ok(c) => {
                let ns = c.namespace();
                let probe = zero_nvme::NamespaceProbe {
                    pci_bus: dev.bus,
                    pci_device: dev.device,
                    pci_function: dev.function,
                    nlba: ns.nlba,
                    lba_size: ns.lba_size,
                };
                let bytes = probe.total_bytes();
                let _ = writeln!(
                    Serial,
                    "NVMe probe: {:02x}:{:02x}.{} = {} LBAs × {} B = {} MiB",
                    dev.bus,
                    dev.device,
                    dev.function,
                    ns.nlba,
                    ns.lba_size,
                    bytes / (1024 * 1024)
                );
                // Replace `best` only on a strict improvement so ties
                // resolve to the earlier-enumerated device — same rule
                // the wire-types `pick_largest_namespace` uses.
                let is_better = match &best {
                    Some(cur) => bytes > cur.probe.total_bytes(),
                    None => true,
                };
                if is_better {
                    best = Some(ProbedController {
                        controller: c,
                        probe,
                    });
                }
                // (else: `c` is dropped here; the device stays
                // initialised on the wire but we keep no kernel-side
                // handle to it.)
            }
            Err(e) => {
                let _ = writeln!(
                    Serial,
                    "NVMe probe: {:02x}:{:02x}.{} bind FAILED: {:?}",
                    dev.bus, dev.device, dev.function, e
                );
                if let (Some(buf), Some(len_ref)) =
                    (errors.as_deref_mut(), errors_len.as_deref_mut())
                {
                    if *len_ref < buf.len() {
                        buf[*len_ref] = ProbeFailure {
                            bus: dev.bus,
                            device: dev.device,
                            function: dev.function,
                            err: e,
                        };
                        *len_ref += 1;
                    }
                }
            }
        }
    }

    if total_found == 0 {
        return Err(NvmeError::ControllerNotFound);
    }
    best.ok_or(NvmeError::ControllerNotFound)
}

/// Walk every NVMe controller the PCI scan exposes, bring each one up,
/// and return the first one whose first LBA carries a supported model
/// marker (`SILM` native `.smodel`, or raw `GGUF` compatibility).
///
/// This is the selection strategy the Cherry deployment actually wants:
/// the operator writes the GGUF raw to whichever NVMe namespace they
/// choose (typically the largest data drive) via `dd`, and the boot
/// drive carries the Zero UEFI image — also at LBA 0, but obviously
/// without a GGUF magic. The blind "largest namespace" rule used to
/// race the two 3.5 TB Solidigm drives against each other and could
/// hand back the boot drive on a PCIe-order flip, killing Stage 11.
///
/// We fall back to `bind_largest_data_drive`'s answer iff *no* drive
/// has the magic, so an operator who provided no GGUF still gets a
/// useful "magic-mismatch on drive X" log line instead of a bare
/// "controller not found".
pub fn bind_model_data_drive(
    scan: &PciScan,
    mut errors: Option<&mut [ProbeFailure]>,
    mut errors_len: Option<&mut usize>,
) -> Result<ProbedController, NvmeError> {
    let mut fallback: Option<ProbedController> = None;
    let mut total_found: usize = 0;

    for dev in nvme_controllers(scan) {
        total_found += 1;
        let _ = writeln!(
            Serial,
            "NVMe probe: scanning {:02x}:{:02x}.{} vendor=0x{:04x} device=0x{:04x}",
            dev.bus, dev.device, dev.function, dev.vendor_id, dev.device_id
        );
        let mut controller = match NvmeController::bind(dev) {
            Ok(c) => c,
            Err(e) => {
                let _ = writeln!(
                    Serial,
                    "NVMe probe: {:02x}:{:02x}.{} bind FAILED: {:?}",
                    dev.bus, dev.device, dev.function, e
                );
                if let (Some(buf), Some(len_ref)) =
                    (errors.as_deref_mut(), errors_len.as_deref_mut())
                {
                    if *len_ref < buf.len() {
                        buf[*len_ref] = ProbeFailure {
                            bus: dev.bus,
                            device: dev.device,
                            function: dev.function,
                            err: e,
                        };
                        *len_ref += 1;
                    }
                }
                continue;
            }
        };

        let ns = controller.namespace();
        let probe = zero_nvme::NamespaceProbe {
            pci_bus: dev.bus,
            pci_device: dev.device,
            pci_function: dev.function,
            nlba: ns.nlba,
            lba_size: ns.lba_size,
        };
        let bytes = probe.total_bytes();
        let _ = writeln!(
            Serial,
            "NVMe probe: {:02x}:{:02x}.{} = {} LBAs × {} B = {} MiB",
            dev.bus,
            dev.device,
            dev.function,
            ns.nlba,
            ns.lba_size,
            bytes / (1024 * 1024)
        );

        match probe_model_magic(&mut controller) {
            Ok(kind) if kind.is_supported() => {
                let _ = writeln!(
                    Serial,
                    "NVMe probe: {:02x}:{:02x}.{} {} marker FOUND at LBA 0 — selecting as model drive",
                    dev.bus, dev.device, dev.function, kind.label()
                );
                return Ok(ProbedController { controller, probe });
            }
            Ok(kind) => {
                let _ = writeln!(
                    Serial,
                    "NVMe probe: {:02x}:{:02x}.{} no supported model marker at LBA 0 (saw {}) — likely boot drive, skipping",
                    dev.bus, dev.device, dev.function, kind.label()
                );
                // Remember the largest non-model drive so we can still
                // surface a probed result if NO drive has the marker.
                let take = match &fallback {
                    Some(cur) => bytes > cur.probe.total_bytes(),
                    None => true,
                };
                if take {
                    fallback = Some(ProbedController { controller, probe });
                }
            }
            Err(e) => {
                let _ = writeln!(
                    Serial,
                    "NVMe probe: {:02x}:{:02x}.{} model-marker probe read failed: {:?}",
                    dev.bus, dev.device, dev.function, e
                );
                if let (Some(buf), Some(len_ref)) =
                    (errors.as_deref_mut(), errors_len.as_deref_mut())
                {
                    if *len_ref < buf.len() {
                        buf[*len_ref] = ProbeFailure {
                            bus: dev.bus,
                            device: dev.device,
                            function: dev.function,
                            err: e,
                        };
                        *len_ref += 1;
                    }
                }
            }
        }
    }

    if total_found == 0 {
        return Err(NvmeError::ControllerNotFound);
    }
    // None of the drives carry a supported marker — fall through with the
    // largest non-model drive so the loader still emits a clear
    // MagicMismatch error rather than a confusing ControllerNotFound.
    fallback.ok_or(NvmeError::ControllerNotFound)
}

/// Compatibility wrapper for older call sites.
pub fn bind_gguf_data_drive(
    scan: &PciScan,
    errors: Option<&mut [ProbeFailure]>,
    errors_len: Option<&mut usize>,
) -> Result<ProbedController, NvmeError> {
    bind_model_data_drive(scan, errors, errors_len)
}

/// Read the first LBA from `ctrl`'s configured namespace and identify
/// whether it starts with a supported model marker. Allocates one
/// 4 KiB scratch page from the kernel arena.
///
/// Used as a sanity check after [`bind_largest_data_drive`] has
/// selected the largest drive: it verifies the operator actually
/// wrote the model to that drive (Phase B of the Cherry deploy
/// script). A negative result is a deploy bug, not a hardware fault.
pub fn probe_model_magic(ctrl: &mut NvmeController) -> Result<ModelDiskMagic, NvmeError> {
    let lba_size = ctrl.namespace().lba_size as usize;
    if lba_size == 0 {
        return Err(NvmeError::BadIdentify);
    }
    // bulk_read needs a 4 KiB-aligned destination and `byte_count`
    // that is a multiple of `lba_size`. NVMe page size (4 KiB) is a
    // multiple of every realistic LBA size (512 or 4096) and gives
    // us a freshly-aligned buffer in one shot.
    let page = alloc_dma_page()?;
    if NVME_PAGE_SIZE % lba_size != 0 {
        // Pathological LBA size (e.g. 8 KiB) — the kernel-side
        // bulk_read already rejects this earlier via the
        // `lba_size > MAX_READ_BYTES` guard, but be defensive here
        // in case probe is ever called against a drive that slipped
        // through.
        return Err(NvmeError::OutOfRange);
    }
    ctrl.bulk_read(0, NVME_PAGE_SIZE, page.va)?;
    // Volatile-read 4 bytes — the DMA buffer is uncached only if the
    // MMIO mapper marked it so; the arena allocates from normal
    // write-back RAM, which is what we want here.
    let slice = unsafe { core::slice::from_raw_parts(page.va as *const u8, 4) };
    Ok(match model_magic_from_bytes(slice) {
        ModelMagic::Gguf => ModelDiskMagic::Gguf,
        ModelMagic::Smodel => ModelDiskMagic::Smodel,
        ModelMagic::Unknown(_) => ModelDiskMagic::Other,
    })
}

/// Compatibility wrapper for older call sites that only care about raw
/// GGUF. New model-drive discovery should use [`probe_model_magic`].
pub fn probe_gguf_magic(ctrl: &mut NvmeController) -> Result<bool, NvmeError> {
    Ok(probe_model_magic(ctrl)? == ModelDiskMagic::Gguf)
}

// ---------------------------------------------------------------------
//  DMA-page allocator — one 4 KiB page from the kernel arena.
// ---------------------------------------------------------------------

struct DmaPage {
    va: u64,
    phys: u64,
}

fn alloc_dma_page() -> Result<DmaPage, NvmeError> {
    let ptr = {
        let mut guard = KERNEL_ARENA.lock();
        let arena = guard.as_mut().ok_or(NvmeError::OutOfMemory)?;
        // NVMe PRP buffers must be 4 KiB aligned. Do not rely on
        // incidental arena state or `[u64]`'s 8-byte alignment; the
        // EPYC 9654 host can shift the kernel arena cursor before the
        // second controller probe and fail here with `BadDmaBuffer`.
        arena
            .alloc_zeroed_aligned(NVME_PAGE_SIZE, NVME_PAGE_SIZE)
            .map_err(|_| NvmeError::OutOfMemory)?
            .as_mut_ptr() as *mut u8
    };
    let va = ptr as u64;
    if va & (NVME_PAGE_SIZE as u64 - 1) != 0 {
        return Err(NvmeError::BadDmaBuffer);
    }
    let phys = virt_to_phys_pa(va).ok_or(NvmeError::BadDmaBuffer)?;
    Ok(DmaPage { va, phys })
}

fn wait_csts_set(bar0: *mut u8, mask: u32, timeout_ms: u32) -> Result<(), ()> {
    let mut spin: u64 = 0;
    let spin_limit = zero_nvme::spin_budget(timeout_ms);
    loop {
        let csts = unsafe { read32(bar0, REG_CSTS) };
        if csts & mask != 0 {
            return Ok(());
        }
        if csts & CSTS_CFS != 0 {
            return Err(());
        }
        spin += 1;
        if spin >= spin_limit {
            return Err(());
        }
        core::hint::spin_loop();
    }
}

fn wait_csts_clear(bar0: *mut u8, mask: u32, timeout_ms: u32) -> Result<(), ()> {
    let mut spin: u64 = 0;
    let spin_limit = zero_nvme::spin_budget(timeout_ms);
    loop {
        let csts = unsafe { read32(bar0, REG_CSTS) };
        if csts & mask == 0 {
            return Ok(());
        }
        if csts & CSTS_CFS != 0 {
            return Err(());
        }
        spin += 1;
        if spin >= spin_limit {
            return Err(());
        }
        core::hint::spin_loop();
    }
}

// Driver-level smoke tests — exercising the PCI predicate and the
// bounded `ControllerList` against synthesised `PciDevice` values — live
// alongside their pure implementations in `nvme_probe.rs` and run on host
// via the `kernel-tests` harness (`cargo test --workspace`). Wire-type /
// planner tests live in the `zero-nvme` crate.
