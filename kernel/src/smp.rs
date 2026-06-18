// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR-029 Phase 1+2 — Platform-independent SMP (Symmetric Multi-Processing)
//! infrastructure.
//!
//! This module provides cross-platform primitives for distributing matmul
//! work across multiple cores in parallel. It is the architectural counter-
//! part to the per-architecture AP boot machinery in `arch::x86_64::apic`
//! and `arch::x86_64::trampoline`. The contract:
//!
//! * BSP (Bootstrap Processor) parses platform topology (ACPI MADT on
//!   x86_64; DTB cpus@N on aarch64 — out of scope for v0), wakes APs
//!   (Application Processors), then calls [`set_active_cores`] once all
//!   APs have signaled ready.
//! * Each AP, after platform-specific boot, calls [`ap_register`] and
//!   then enters [`ap_worker_loop`] which polls its private work slot.
//! * For each Linear (matmul) operator: BSP calls [`split_rows`] to
//!   partition the output row range; publishes one [`WorkItem`] per
//!   active core into [`WORK_SLOTS`]; sets every `WORK_READY[i]` flag;
//!   executes its own slice in-line; then spins on the shared
//!   [`SpinBarrier`] until all APs have finished.
//!
//! # V3.1 Pillar conformance
//!
//! * **Pillar 1** — Zero allocation in the hot path. All slots and the
//!   barrier counter are `'static`, atomic, lock-free. The row split is
//!   stack-only (`[RowRange; MAX_CORES]`).
//! * **Pillar 7** — NO `#[cfg(target_arch = ...)]` in this file. The
//!   atomic primitives compile identically on aarch64 and x86_64.
//!   Architecture-specific bring-up lives in `arch::*`.
//! * **Pillar 8** — Long-term direction is Quarks reimplementation
//!   (post-Stage-24). This module is the Rust reference contract.
//!
//! # Bit-exactness guarantee
//!
//! Row splitting partitions the *output* dimension. For each output
//! row `m`, the inner reduction `Σ x[k] * dequant(W[m, k])` runs on
//! exactly one core, with the same K-order as the single-threaded
//! reference. Therefore parallel and serial execution produce the
//! same F32 bit pattern. This is the foundation for the β-anchor
//! invariant (`token=25`, `logit_bits=0x414a6497`) across SMP modes.
//!
//! CITE: AMD64 APM Vol 2 §16.4 (Local APIC / INIT-SIPI startup)
//! CITE: ACPI 6.4 §5.2.12 (MADT)
//! CITE: ADR-029 Patch v4 (β-Anchor preservation under parallelism)

// Several items in this module are public-API affordances for callers
// outside the hot path (diagnostics, halt path, the WorkItem-as-enum
// alternative used by future interrupt-driven AP loops). The current
// hot-path code uses the type-erased atomic-word variant — the named
// constants and WorkItem struct remain part of the documented contract.
#![allow(dead_code)]

use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

// Pure row-partition math (`split_rows`, `row_range_for`, `RowRange`,
// `MAX_CORES`, …) lives in `smp_partition` so the host test harness
// (`crates/kernel-tests`) can exercise it without the bare-metal SMP
// runtime. Re-exported here so `crate::smp::<item>` keeps resolving for
// the kernel and `inference_avx512.rs`.
#[path = "smp_partition.rs"]
mod smp_partition;
pub use smp_partition::{
    row_range_for, row_range_for_discounted, split_rows, split_rows_aligned, RowRange,
    MATMUL_BSP_DISCOUNT_PCT_CEILING, MAX_CORES,
};

/// Sentinel: AP work slot is idle (no pending work).
const WORK_TAG_IDLE: u32 = 0;
/// Sentinel: AP work slot holds a matmul tile waiting to be executed.
const WORK_TAG_MATMUL: u32 = 1;
/// Sentinel: BSP has requested APs to halt (shutdown / reset).
const WORK_TAG_HALT: u32 = 2;

// ─────────────────────────────────────────────────────────────────
// Topology
// ─────────────────────────────────────────────────────────────────

/// Number of cores currently participating in parallel execution.
///
/// Set by [`set_active_cores`] after AP boot. Read by [`split_rows`]
/// and the hot-path dispatcher. `1` means single-core (BSP-only)
/// execution — the contract requires that the parallel dispatch be a
/// no-op in that case, so a kernel that never wakes APs still works.
static ACTIVE_CORES: AtomicU32 = AtomicU32::new(1);

/// Bring-up gate for bare-metal Cherry/EPYC debugging.
///
/// When active, the BSP has installed the AP trampoline but has not yet
/// issued INIT-SIPI-SIPI. The network shell can then inspect the machine
/// and request the AP wake-up explicitly. This does not alter the LLM
/// parallel row-splitting contract; active cores remain BSP-only until
/// the APs have actually registered.
static AP_BOOT_GATE_ACTIVE: AtomicBool = AtomicBool::new(false);
static AP_BOOT_REQUESTED: AtomicBool = AtomicBool::new(false);
static AP_BOOT_AP_LIMIT: AtomicU32 = AtomicU32::new(u32::MAX);
static AP_BOOT_MODE: AtomicU32 = AtomicU32::new(AP_BOOT_MODE_FULL);
static AP_PROBE_STAGE: AtomicU32 = AtomicU32::new(AP_PROBE_STAGE_IDLE);

pub const AP_BOOT_MODE_FULL: u32 = 0;
pub const AP_BOOT_MODE_PROBE_ENTRY: u32 = 1;
pub const AP_BOOT_MODE_PROBE_IDT: u32 = 2;
pub const AP_BOOT_MODE_PROBE_APIC: u32 = 3;
pub const AP_BOOT_MODE_PROBE_SIMD: u32 = 4;
pub const AP_BOOT_MODE_PROBE_TRAMP_REAL: u32 = 10;
pub const AP_BOOT_MODE_PROBE_TRAMP_PROT: u32 = 11;
pub const AP_BOOT_MODE_PROBE_TRAMP_PAE: u32 = 12;
pub const AP_BOOT_MODE_PROBE_TRAMP_EFER: u32 = 13;
pub const AP_BOOT_MODE_PROBE_TRAMP_PAGING: u32 = 14;
pub const AP_BOOT_MODE_PROBE_TRAMP_LONG: u32 = 15;
pub const AP_BOOT_MODE_PROBE_TRAMP_CR3: u32 = 16;
pub const AP_BOOT_MODE_PROBE_TRAMP_RUST: u32 = 17;

pub const AP_PROBE_STAGE_IDLE: u32 = 0;
pub const AP_PROBE_STAGE_ENTRY: u32 = 1;
pub const AP_PROBE_STAGE_IDT: u32 = 2;
pub const AP_PROBE_STAGE_APIC: u32 = 3;
pub const AP_PROBE_STAGE_SIMD: u32 = 4;
pub const AP_PROBE_STAGE_TRAMP_REAL: u32 = 10;
pub const AP_PROBE_STAGE_TRAMP_PROT: u32 = 11;
pub const AP_PROBE_STAGE_TRAMP_PAE: u32 = 12;
pub const AP_PROBE_STAGE_TRAMP_EFER: u32 = 13;
pub const AP_PROBE_STAGE_TRAMP_PAGING: u32 = 14;
pub const AP_PROBE_STAGE_TRAMP_LONG: u32 = 15;
pub const AP_PROBE_STAGE_TRAMP_CR3: u32 = 16;
pub const AP_PROBE_STAGE_TRAMP_RUST: u32 = 17;

/// Returns the number of cores currently active in parallel dispatch.
///
/// Always at least `1` (the BSP). At BSP boot before AP wake-up the
/// value is `1`; once AP boot has completed [`set_active_cores`] sets
/// the real count.
#[inline(always)]
pub fn active_cores() -> u32 {
    ACTIVE_CORES.load(Ordering::Acquire)
}

/// Record the final number of cores (BSP + APs) that successfully
/// reached the AP worker loop. Called once by the BSP after all APs
/// have signaled ready via [`ap_register`]. Subsequent calls overwrite
/// the previous value; no rebinding logic — SMP topology is fixed for
/// the duration of the kernel.
#[inline]
pub fn set_active_cores(n: u32) {
    let clamped = if (n as usize) > MAX_CORES {
        MAX_CORES as u32
    } else {
        n
    };
    ACTIVE_CORES.store(clamped.max(1), Ordering::Release);
    bump_tuning_version();
}

/// Arm or disarm the manual AP boot gate.
#[inline]
pub fn set_ap_boot_gate_active(active: bool) {
    if active {
        AP_BOOT_REQUESTED.store(false, Ordering::Release);
        AP_BOOT_AP_LIMIT.store(u32::MAX, Ordering::Release);
        AP_BOOT_MODE.store(AP_BOOT_MODE_FULL, Ordering::Release);
        AP_PROBE_STAGE.store(AP_PROBE_STAGE_IDLE, Ordering::Release);
    }
    AP_BOOT_GATE_ACTIVE.store(active, Ordering::Release);
}

/// Returns true while the BSP is paused before INIT-SIPI-SIPI.
#[inline(always)]
pub fn ap_boot_gate_active() -> bool {
    AP_BOOT_GATE_ACTIVE.load(Ordering::Acquire)
}

/// Request that the BSP leaves the AP boot gate and starts AP wake-up.
#[inline]
pub fn request_ap_boot() {
    AP_BOOT_AP_LIMIT.store(u32::MAX, Ordering::Release);
    AP_BOOT_REQUESTED.store(true, Ordering::Release);
}

/// Request AP wake-up with an explicit cap on the number of APs.
///
/// This keeps Cherry/EPYC debug builds debuggable: `smp start 1` can
/// prove the trampoline path before `smp start all` exposes all APs.
/// Production Cherry builds auto-start APs and do not arm this gate.
#[inline]
pub fn request_ap_boot_limited(max_aps: u32) {
    AP_BOOT_AP_LIMIT.store(max_aps.max(1), Ordering::Release);
    AP_BOOT_MODE.store(AP_BOOT_MODE_FULL, Ordering::Release);
    AP_BOOT_REQUESTED.store(true, Ordering::Release);
}

/// Request a single AP wake-up in probe/park mode.
///
/// The AP stops at the requested early bring-up stage instead of entering
/// the worker loop. This is a controlled hardware diagnostic path used
/// when full AP startup resets the platform; normal inference never uses
/// it because [`AP_BOOT_MODE_FULL`] remains the default.
#[inline]
pub fn request_ap_probe(mode: u32) {
    AP_PROBE_STAGE.store(AP_PROBE_STAGE_IDLE, Ordering::Release);
    AP_BOOT_AP_LIMIT.store(1, Ordering::Release);
    AP_BOOT_MODE.store(mode, Ordering::Release);
    AP_BOOT_REQUESTED.store(true, Ordering::Release);
}

#[inline(always)]
pub fn ap_boot_mode() -> u32 {
    AP_BOOT_MODE.load(Ordering::Acquire)
}

#[inline]
pub fn ap_probe_mark(stage: u32) {
    AP_PROBE_STAGE.store(stage, Ordering::Release);
}

#[inline(always)]
pub fn ap_probe_stage() -> u32 {
    AP_PROBE_STAGE.load(Ordering::Acquire)
}

/// Returns true once `smp start` has requested AP wake-up.
#[inline(always)]
pub fn ap_boot_requested() -> bool {
    AP_BOOT_REQUESTED.load(Ordering::Acquire)
}

/// Maximum APs requested by the TCP shell gate. `u32::MAX` means all
/// eligible APs from MADT.
#[inline(always)]
pub fn ap_boot_ap_limit() -> u32 {
    AP_BOOT_AP_LIMIT.load(Ordering::Acquire)
}

// ─────────────────────────────────────────────────────────────────
// AP registration counter (Phase 3 hand-off)
// ─────────────────────────────────────────────────────────────────

/// Counts APs that have completed their architecture-specific bring-up
/// (real → protected → long mode) and are sitting in [`ap_worker_loop`].
/// The BSP spins on this until it equals `expected_aps` before calling
/// [`set_active_cores`]. Includes the BSP itself once it has called
/// [`bsp_register`] — see the comments on that function.
static REGISTERED_CORES: AtomicU32 = AtomicU32::new(0);

/// BSP self-registration. Should be called exactly once, very early
/// (before APs are woken). It seeds [`REGISTERED_CORES`] at 1 so that
/// the post-wake count matches `1 + n_aps`.
#[inline]
pub fn bsp_register() {
    REGISTERED_CORES.store(1, Ordering::Release);
}

/// Record the BSP's APIC ID once the LAPIC/x2APIC driver is online.
///
/// [`bsp_register`] intentionally runs before APIC initialisation, so
/// diagnostics fill in core 0's APIC ID in this later step.
#[inline]
pub fn bsp_record_apic_id(apic_id: u32) {
    CORE_APIC_IDS[0].store(apic_id, Ordering::Release);
    if (apic_id as usize) < MAX_CORES {
        APIC_ID_TO_CORE_INDEX[apic_id as usize].store(0, Ordering::Release);
    }
}

/// AP self-registration. Called by `ap_entry` after the trampoline
/// hands control to long-mode Rust on the AP. Returns the value
/// post-increment, which the AP uses as a sanity check (and which
/// the BSP can correlate against its expected count).
#[inline]
pub fn ap_register() -> u32 {
    REGISTERED_CORES.fetch_add(1, Ordering::AcqRel) + 1
}

/// Read the current registered-cores count without modifying it.
#[inline(always)]
pub fn registered_cores() -> u32 {
    REGISTERED_CORES.load(Ordering::Acquire)
}

// ─────────────────────────────────────────────────────────────────
// SpinBarrier — N-way completion sync
// ─────────────────────────────────────────────────────────────────

/// Lock-free N-way completion barrier.
///
/// Constructed with the *target* count of arrivals. Each core calls
/// [`arrive`](SpinBarrier::arrive) once it has finished its work; the
/// caller of [`wait_complete`](SpinBarrier::wait_complete) (typically
/// the BSP) spins until the count reaches the target.
///
/// Reset is explicit: the BSP calls [`reset`](SpinBarrier::reset) with
/// the new target before publishing the next batch of work. This avoids
/// re-allocation across iterations and is the pattern used for every
/// matmul in the forward pass.
///
/// **Why a custom barrier, not `spin::Barrier`?** `spin::Barrier` is a
/// general-purpose two-phase reusable barrier. We don't need the
/// arriver-also-waits guarantee — APs publish a "done" signal and
/// return to their poll loop; only the BSP waits. The two atomics are
/// simpler, leaner, and don't require generic monomorphization.
#[repr(C)]
pub struct SpinBarrier {
    target: AtomicU32,
    arrived: AtomicU32,
}

/// Wall-clock budget for `wait_complete` before the BSP gives up and
/// reports failure to the caller. Measured via TSC, not iteration
/// counts: 10 M `pause` iterations are only ~10–30 ms on EPYC 9354P,
/// which an SMI can easily exceed. One full second is orders of
/// magnitude above any honest matmul tile while still guaranteeing
/// forward progress if an AP wedges permanently.
const SMP_BARRIER_TIMEOUT_SECS: u64 = 1;

/// Number of `pause` poll iterations between TSC deadline checks.
/// Keeps `rdtsc` (serialized, ~tens of cycles) off the hot polling
/// loop so coherence traffic on the barrier cacheline stays minimal.
const SMP_BARRIER_SPIN_BATCH: u32 = 4096;

/// Lazily-detected TSC frequency for barrier deadlines. `tsc_hz()`
/// issues several `cpuid`s (serializing); with ~10 k barrier
/// dispatches per token that cost must be paid once, not per wait.
static BARRIER_TSC_HZ: AtomicU64 = AtomicU64::new(0);

#[inline]
fn barrier_tsc_hz() -> u64 {
    let cached = BARRIER_TSC_HZ.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    // tsc_hz() never returns 0 (2.5 GHz fallback floor), so a single
    // store latches it; racing cores compute the same value.
    let hz = crate::arch::cycles::tsc_hz();
    BARRIER_TSC_HZ.store(hz, Ordering::Relaxed);
    hz
}

impl Default for SpinBarrier {
    fn default() -> Self {
        Self::new()
    }
}

impl SpinBarrier {
    /// Construct a barrier with `target = 0`. Caller must call
    /// [`reset`](Self::reset) before first use.
    pub const fn new() -> Self {
        Self {
            target: AtomicU32::new(0),
            arrived: AtomicU32::new(0),
        }
    }

    /// Reset the arrival counter to zero and set a new target. Used
    /// between matmul dispatches: each dispatch publishes a fresh
    /// `target` (= `n_cores`) and the per-arrival counter starts
    /// from zero again. The `Release` store on `target` pairs with
    /// the AP-side `Acquire` on `WORK_READY` to publish in-flight
    /// work-item state.
    #[inline]
    pub fn reset(&self, target: u32) {
        self.arrived.store(0, Ordering::Relaxed);
        self.target.store(target, Ordering::Release);
    }

    /// Zero both counters. Used for whole-barrier teardown (e.g. SMP
    /// shutdown) where there is no successor dispatch and we want the
    /// barrier to be in a clean idle state.
    #[inline]
    pub fn reset_zero(&self) {
        self.arrived.store(0, Ordering::Relaxed);
        self.target.store(0, Ordering::Release);
    }

    /// Signal arrival. Returns the post-increment count clamped to
    /// `target`, so a sequence of legitimate plus overshooting arrivals
    /// never reports an inflated value to the caller.
    ///
    /// Uses an unconditional `fetch_add` (lowered to `LOCK XADD` on
    /// x86) instead of a CAS-loop. On contention from N cores, `LOCK
    /// XADD` is bounded at one cache-line round-trip per call, while a
    /// `compare_exchange_weak` loop scales super-linearly with N because
    /// every retry burns another MESI invalidation. With N = 64 cores
    /// on EPYC 9354P this difference is in the 10–30 µs class per
    /// barrier (LLM inference issues ~200 barriers per token).
    ///
    /// **Clamp.** If an SMI suspends an AP long enough that the BSP
    /// re-dispatches and the AP then arrives twice for the previous
    /// batch, we still bump the underlying counter — but the post-hoc
    /// clamp prevents callers from observing `cur > target`. The next
    /// `reset(target)` call zeroes the counter and re-seeds the target,
    /// which is the canonical recovery path. The overshoot is logged so
    /// it remains observable for diagnostics.
    #[inline]
    pub fn arrive(&self) -> u32 {
        let target = self.target.load(Ordering::Acquire);
        // LOCK XADD: single round-trip regardless of contention.
        let cur = self.arrived.fetch_add(1, Ordering::AcqRel);
        let post = cur + 1;
        if post > target {
            use core::fmt::Write;
            let _ = writeln!(
                crate::arch::Serial,
                "SMP barrier overshoot: pre={}, post={}, target={} (clamped)",
                cur,
                post,
                target,
            );
            return target;
        }
        post
    }

    /// Returns true once `arrived >= target` for the current epoch.
    /// Used by [`TreeBarrier::arrive_in`] to detect the last arriver
    /// for a leaf group.
    #[inline]
    pub fn is_complete(&self) -> bool {
        let target = self.target.load(Ordering::Acquire);
        if target == 0 {
            return true;
        }
        self.arrived.load(Ordering::Acquire) >= target
    }

    /// Spin until `arrived >= target`. Uses `core::hint::spin_loop` to
    /// permit the CPU to power down execution units on long waits — on
    /// x86 this emits `pause`, which the MESI protocol uses as a hint
    /// to reduce coherence traffic on hot cache lines.
    ///
    /// Bounded by a TSC deadline of [`SMP_BARRIER_TIMEOUT_SECS`] real
    /// seconds: if the count does not reach `target` in time, we report
    /// failure to the caller. The deadline is wall-clock, not an
    /// iteration count, so an SMI-delayed AP gets a full second of
    /// headroom. The caller must not consume the matmul output after a
    /// `false` return without recomputing — some output rows may still
    /// be stale or may be written late by an AP.
    #[inline]
    pub fn wait_complete(&self) -> bool {
        let target = self.target.load(Ordering::Acquire);
        if self.arrived.load(Ordering::Acquire) >= target {
            return true;
        }
        // Slow path: arm the TSC deadline only after the first miss so
        // the common already-complete case pays no rdtsc.
        let deadline = crate::arch::cycles::rdtsc_serialized()
            .saturating_add(barrier_tsc_hz().saturating_mul(SMP_BARRIER_TIMEOUT_SECS));
        loop {
            let mut batch = 0u32;
            while batch < SMP_BARRIER_SPIN_BATCH {
                if self.arrived.load(Ordering::Acquire) >= target {
                    return true;
                }
                core::hint::spin_loop();
                batch += 1;
            }
            if crate::arch::cycles::rdtsc_serialized() >= deadline {
                use core::fmt::Write;
                let arrived = self.arrived.load(Ordering::Acquire);
                let _ = writeln!(
                    crate::arch::Serial,
                    "SMP barrier timeout after {}s: arrived={}, target={}",
                    SMP_BARRIER_TIMEOUT_SECS,
                    arrived,
                    target,
                );
                return false;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// TreeBarrier — two-level (per-CCD-group + root) completion barrier
// ─────────────────────────────────────────────────────────────────

/// Logical-core group width used by [`TreeBarrier`]. Sized for AMD EPYC
/// 9354P (8 CCDs × 4 cores × 2 SMT siblings = 8 logical cores per CCD)
/// where keeping arrival traffic CCD-local avoids cross-IOD cacheline
/// bouncing for every arriving AP. On systems with a different topology
/// this is still correct — see [`barrier_group_of`] for the mapping. The
/// pathological case is 1 core per group (degenerates to a single-leaf
/// tree, equivalent to the flat [`SpinBarrier`]).
pub const BARRIER_GROUP_SIZE: usize = 8;

/// Number of leaf barriers. `MAX_CORES / BARRIER_GROUP_SIZE` rounded up.
pub const BARRIER_GROUP_COUNT: usize = MAX_CORES.div_ceil(BARRIER_GROUP_SIZE);

/// Map a logical core index to its barrier group.
#[inline(always)]
pub fn barrier_group_of(core_idx: usize) -> usize {
    core_idx / BARRIER_GROUP_SIZE
}

/// One CCD-group leaf of the [`TreeBarrier`], padded to its own
/// cacheline.
///
/// `SpinBarrier` is 8 bytes and `AtomicBool` is 1 byte — packing
/// `[SpinBarrier; 16]` + `[AtomicBool; 16]` as plain arrays put 8
/// leaves on ONE 64-byte line. Every leaf's arrivals come from a
/// *different* CCD, so the un-padded layout had all 64 cores hammering
/// the same two lines per dispatch — exactly the cross-IOD
/// invalidation storm the tree barrier exists to avoid. With one leaf
/// (barrier + its promoted latch) per line, arrival traffic stays
/// CCD-local until the single root promotion.
#[repr(C, align(64))]
struct BarrierLeaf {
    barrier: SpinBarrier,
    /// "Promoted to root" latch. swap(true) is the single atomic that
    /// prevents multiple arrivers in the same leaf from promoting more
    /// than one arrival to the root. Shares the leaf's line on purpose:
    /// it is only touched by the same CCD's arrivers.
    promoted: AtomicBool,
}

impl BarrierLeaf {
    const fn new() -> Self {
        Self {
            barrier: SpinBarrier::new(),
            promoted: AtomicBool::new(false),
        }
    }
}

/// Cacheline-isolated root barrier. Arrivals come from one leaf-leader
/// per CCD group plus the waiting BSP — keeping it off the leaves'
/// (and any neighbouring static's) lines avoids leaf arrivals stealing
/// the line the BSP is spinning on.
#[repr(C, align(64))]
struct BarrierRoot {
    barrier: SpinBarrier,
}

/// Two-level barrier: every arriver hits a small leaf barrier shared
/// only with cores in the same CCD-aligned group; the *last* arriver
/// in each leaf promotes a single arrival to the root barrier. The
/// BSP waits on the root.
///
/// On AMD EPYC 9354P with 64 cores, this reduces the contended
/// arrival point from one 64-way cacheline-shared atomic to (a) 8
/// barriers of 8 arrivals each, served from per-CCD-L3 slices, plus
/// (b) one 8-way root. Aggregate atomic traffic drops by ~8× and the
/// expensive cross-IOD invalidations only fire for the root.
///
/// # Bit-exactness
///
/// The TreeBarrier changes *when* a given AP signals completion, not
/// *what* it computed. Row ownership is unchanged — every output row's
/// FMA reduction still runs on exactly one core in the original
/// K-order. The β-anchor / Two-Anchor invariants therefore hold without
/// modification.
#[repr(C, align(64))]
pub struct TreeBarrier {
    /// Current dispatch epoch. Arrivals from older epochs are ignored
    /// instead of corrupting the next matmul barrier generation.
    /// Alone on the first cacheline (the 64-byte-aligned leaves array
    /// starts on the next line).
    epoch: AtomicU64,
    leaves: [BarrierLeaf; BARRIER_GROUP_COUNT],
    root: BarrierRoot,
}

impl Default for TreeBarrier {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeBarrier {
    pub const fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
            leaves: [const { BarrierLeaf::new() }; BARRIER_GROUP_COUNT],
            root: BarrierRoot {
                barrier: SpinBarrier::new(),
            },
        }
    }

    /// Prepare for a new dispatch. `per_group_targets[i]` is the number
    /// of arrivers expected in leaf group `i`; only non-zero groups
    /// participate in the root barrier.
    #[inline]
    pub fn reset(&self, per_group_targets: &[u32; BARRIER_GROUP_COUNT], epoch: u64) {
        // Publish the new epoch first so any late AP from the previous
        // dispatch is rejected before it can touch freshly reset leaves.
        self.epoch.store(epoch, Ordering::Release);
        let mut n_active = 0u32;
        let mut i = 0;
        while i < BARRIER_GROUP_COUNT {
            let t = per_group_targets[i];
            // Always reset — a previously-active leaf with target=0
            // would otherwise leak its prior `arrived` count.
            self.leaves[i].barrier.reset(t);
            self.leaves[i].promoted.store(false, Ordering::Release);
            if t > 0 {
                n_active += 1;
            }
            i += 1;
        }
        self.root.barrier.reset(n_active);
    }

    /// Reset to the idle state (all counters zero, no expected arrivals).
    #[inline]
    pub fn reset_zero(&self) {
        let mut i = 0;
        while i < BARRIER_GROUP_COUNT {
            self.leaves[i].barrier.reset_zero();
            self.leaves[i].promoted.store(false, Ordering::Release);
            i += 1;
        }
        self.root.barrier.reset_zero();
        self.epoch.store(0, Ordering::Release);
    }

    /// Signal arrival in the given group. If this completes the leaf,
    /// the last arriver promotes a single arrival to the root — exactly
    /// once per leaf per epoch, enforced by the `promoted` latch.
    #[inline]
    pub fn arrive_in(&self, group_idx: usize, epoch: u64) {
        if group_idx >= BARRIER_GROUP_COUNT {
            return;
        }
        let live_epoch = self.epoch.load(Ordering::Acquire);
        if live_epoch != epoch {
            use core::fmt::Write;
            let _ = writeln!(
                crate::arch::Serial,
                "SMP barrier stale arrival: group={}, epoch={}, live={}",
                group_idx,
                epoch,
                live_epoch,
            );
            return;
        }
        let leaf = &self.leaves[group_idx];
        let _ = leaf.barrier.arrive();
        if leaf.barrier.is_complete() && !leaf.promoted.swap(true, Ordering::AcqRel) {
            self.root.barrier.arrive();
        }
    }

    /// Spin until every active leaf has promoted to the root.
    #[inline]
    pub fn wait_complete(&self) -> bool {
        self.root.barrier.wait_complete()
    }
}

// ─────────────────────────────────────────────────────────────────
// RowRange + split_rows — work partitioning
//
// `RowRange` and the `split_rows` / `row_range_for` family are defined
// in `smp_partition` (re-exported at the top of this module) so the host
// test harness can cover the partition math. The runtime context that
// consumes them (`ParallelMatmulContext`, below) stays here.
// ─────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────
// WorkItem — type-erased matmul tile descriptor
// ─────────────────────────────────────────────────────────────────

/// A unit of work an AP can execute. Type-erased via raw pointers
/// so a single slot type can carry any matmul kernel (Q4_K, Q6_K,
/// f32 dot, etc.) without generics.
///
/// **Why type-erased instead of an enum?** Each new matmul variant
/// would otherwise add an arm to a hot-path dispatch `match`. With
/// a function pointer, the dispatcher is a single indirect call,
/// inlined at zero runtime cost in `ap_worker_loop`.
///
/// **Safety.** The pointers in [`MatmulArgs`] must outlive the
/// barrier wait on the BSP. In practice this is guaranteed by the
/// caller — `x`, `w`, `out` all live in `ACTIVATION_ARENA` or
/// `LLM_ARENA`, both of which are `'static`.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct WorkItem {
    /// Discriminator. `WORK_TAG_IDLE` = nothing to do; `WORK_TAG_MATMUL`
    /// = consume `matmul_fn` + `matmul_args`; `WORK_TAG_HALT` = exit
    /// the worker loop.
    pub tag: u32,
    /// Kernel function pointer. Receives the per-AP [`MatmulArgs`] +
    /// the AP's [`RowRange`] for this batch.
    pub matmul_fn: Option<MatmulFn>,
    /// Per-call arguments shared across all participating cores.
    pub matmul_args: MatmulArgs,
}

impl WorkItem {
    /// Idle (no work) slot.
    pub const IDLE: WorkItem = WorkItem {
        tag: WORK_TAG_IDLE,
        matmul_fn: None,
        matmul_args: MatmulArgs::ZERO,
    };
}

/// Signature of a kernel that consumes a [`RowRange`] and writes to
/// `out`. Pillar-7-pure: this signature is identical on x86_64 and
/// aarch64. Per-architecture implementations live in `arch::*::math`.
///
/// **Safety contract:**
/// * `x` points to `args.in_dim` valid `f32`s.
/// * `w` points to enough quantized bytes for rows `[range.start, range.end)`.
/// * `out` points to `args.out_dim` `f32`s; only the slice
///   `out[range.start..range.end]` is written by this call.
/// * Different APs write disjoint slices, so no synchronization is
///   needed on `out` beyond the post-wait `Acquire` on the barrier.
pub type MatmulFn = unsafe fn(args: &MatmulArgs, range: RowRange);

/// Pointer-bundle passed to every parallel matmul kernel. All pointers
/// are `*const u8` / `*mut f32` so the struct is `Copy` and the slot
/// can be written word-by-word without a `Mutex`.
///
/// **Pointer semantics.** APs read these pointers after the BSP has
/// performed a `Release` write into [`WORK_READY`]; the `Acquire` load
/// in the worker loop synchronises with that write, so all fields are
/// visible without explicit fences.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct MatmulArgs {
    pub x_ptr: *const f32,
    pub w_ptr: *const u8,
    pub out_ptr: *mut f32,
    pub in_dim: usize,
    pub out_dim: usize,
}

// SAFETY: WorkItem and MatmulArgs are sent across cores via a
// `static` slot array; the BSP and APs synchronise via the
// `WORK_READY` `AtomicBool` array (Release/Acquire). The pointers
// inside `MatmulArgs` point to arena memory that outlives the
// barrier wait. This is the standard "share data through atomics"
// pattern and is sound under the C++20 / Rust memory model.
unsafe impl Send for WorkItem {}
unsafe impl Sync for WorkItem {}
unsafe impl Send for MatmulArgs {}
unsafe impl Sync for MatmulArgs {}

impl MatmulArgs {
    /// Sentinel (all-zero) for [`WorkItem::IDLE`].
    pub const ZERO: MatmulArgs = MatmulArgs {
        x_ptr: core::ptr::null(),
        w_ptr: core::ptr::null(),
        out_ptr: core::ptr::null_mut(),
        in_dim: 0,
        out_dim: 0,
    };
}

// ─────────────────────────────────────────────────────────────────
// Global SMP state — work slots, ready flags, completion barrier
// ─────────────────────────────────────────────────────────────────

/// Per-core "work range" slot — the [`RowRange`] this core executes.
/// Companion to [`SHARED_WORK`] (which carries the fn + args shared
/// by all APs in one dispatch). Only the range varies per AP.
static WORK_RANGE: [SlotCell; MAX_CORES] = [const { SlotCell::new() }; MAX_CORES];

/// Per-core ready flag. Diagnostic / halt-nudge only — the production
/// wake signal is [`WORK_EPOCH`], and nothing reads this flag on the
/// hot path. Kept because [`request_halt`] uses it as a wake nudge and
/// external diagnostics may inspect it. NOT written per dispatch: all
/// 128 bools share two cachelines, so per-dispatch stores from BSP
/// (publish) and 63 APs (clear) turned the flag array into a
/// false-sharing hotspot adjacent to the barrier wait.
static WORK_READY: [AtomicBool; MAX_CORES] = [const { AtomicBool::new(false) }; MAX_CORES];

/// One per-core dispatch epoch on its OWN cacheline.
///
/// Padding rationale (qwen-perf-v2 round 4): every idle AP spins
/// Acquire-loads on its epoch while the BSP publishes up to 63 new
/// epochs per dispatch. As a packed `[AtomicU64; 128]` (8 epochs per
/// 64-B line) each BSP store invalidated the spin line of up to 7
/// OTHER APs, re-faulting their loads across CCD boundaries ~113
/// times per token — the same false-sharing class as the round-1
/// barrier-leaf and WORK_READY fixes. 8 KiB of static padding buys
/// every AP a private spin line that only ITS publish invalidates.
#[repr(C, align(64))]
struct EpochSlot {
    epoch: AtomicU64,
}

impl EpochSlot {
    const fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
        }
    }
}

/// Per-core dispatch epoch. This is the production AP wake-up signal.
///
/// `WORK_READY` is kept for diagnostics and halt nudging, but a boolean
/// flag is not sufficient for back-to-back matmul dispatches: an AP can
/// observe an old `true` or arrive late and damage the next reusable
/// barrier generation. Epoch publication makes every work item unique.
static WORK_EPOCH: [EpochSlot; MAX_CORES] = [const { EpochSlot::new() }; MAX_CORES];

/// Monotonic epoch source for matmul dispatches. Epoch zero is reserved
/// for the idle state, so APs can treat any non-zero unseen epoch as work.
///
/// On its own cacheline: the BSP `fetch_add`s this once per dispatch.
/// As a bare static it can share a 64-B line with [`HALT_REQUESTED`]
/// (declared adjacent, laid out adjacent in .data) — and that flag is
/// Acquire-polled by EVERY spinning AP on EVERY spin iteration, so
/// each dispatch would invalidate the poll line of all 63 APs.
#[repr(C, align(64))]
struct PaddedEpochSource {
    epoch: AtomicU64,
}

static NEXT_WORK_EPOCH: PaddedEpochSource = PaddedEpochSource {
    epoch: AtomicU64::new(1),
};

/// Global halt flag — set by the BSP via [`request_halt`] to ask
/// APs to exit their worker loops (used for shutdown / panic paths).
///
/// On its own cacheline (see [`NEXT_WORK_EPOCH`]): written exactly
/// once at shutdown, but Acquire-polled by every AP in the idle spin
/// — it must live on a line nothing writes during dispatch so the
/// poll stays a local cache hit.
#[repr(C, align(64))]
struct PaddedHaltFlag {
    halt: AtomicBool,
}

static HALT_REQUESTED: PaddedHaltFlag = PaddedHaltFlag {
    halt: AtomicBool::new(false),
};

/// Shared completion barrier (legacy flat variant). Kept for callers
/// that have not migrated to [`MATMUL_TREE_BARRIER`]. Not used by the
/// production matmul dispatch path on Cherry post-tree-barrier
/// migration.
#[allow(dead_code)]
pub static MATMUL_BARRIER: SpinBarrier = SpinBarrier::new();

/// Production matmul completion barrier — two-level
/// (per-CCD-group leaf + root). BSP calls
/// `MATMUL_TREE_BARRIER.reset(per_group_targets)` before publishing
/// work; each participating core calls
/// `MATMUL_TREE_BARRIER.arrive_in(barrier_group_of(core_idx))` after
/// it has finished its row slice; BSP waits via
/// `MATMUL_TREE_BARRIER.wait_complete()`.
pub static MATMUL_TREE_BARRIER: TreeBarrier = TreeBarrier::new();

// ─────────────────────────────────────────────────────────────────
// Shared matmul slot (ADR-029 v8.4 — broadcast publish)
// ─────────────────────────────────────────────────────────────────
//
// Before v8.4, the BSP wrote (fn, args) into PER-AP slots — 9 atomic
// Release stores per AP × 63 APs = 567 stores per dispatch, with
// each Release potentially causing a cross-CCD invalidation. v8.4
// observation: `matmul_fn` and `MatmulArgs` are IDENTICAL across
// all participating APs in one dispatch; only `RowRange` varies.
//
// New design: one shared (fn, args) slot on a single cacheline,
// written ONCE per dispatch; per-AP slots retain only RowRange +
// WORK_READY. APs read the shared slot AFTER observing WORK_READY
// (the Acquire load on WORK_READY synchronises with the BSP's
// Release store, which itself follows the BSP's Release publish of
// SHARED_*).
//
// Per dispatch the BSP writes 6 shared stores + 3 per-AP stores ×
// 63 APs = 195 stores total (was 567). The shared cacheline is
// touched once by BSP per dispatch, then ALL 63 APs read it — a
// single bcast invalidation pattern that the hardware coalesces
// much better than 63 sequential per-AP writes.
//
// Cache-line aligned; placed in its own [repr(C, align(64))]
// struct to keep it off the same line as TREE barrier counters.
#[repr(C, align(64))]
struct SharedWorkSlot {
    fn_ptr: AtomicPtr<()>,
    args: SlotCell,
}

impl SharedWorkSlot {
    const fn new() -> Self {
        Self {
            fn_ptr: AtomicPtr::new(core::ptr::null_mut()),
            args: SlotCell::new(),
        }
    }
}

static SHARED_WORK: SharedWorkSlot = SharedWorkSlot::new();

/// A `UnsafeCell`-equivalent slot holding 64-bytes of payload that
/// the BSP and APs synchronise through `WORK_READY[i]`. We use raw
/// atomic words rather than a `Mutex` because the access pattern is
/// strictly Release-write-then-Acquire-read with no contention.
#[repr(C, align(64))]
struct SlotCell {
    // 64 bytes = cache line, sized for the largest payload we put
    // through this — `MatmulArgs` is 5 pointers = 40 bytes on
    // x86_64, well under the line.
    words: [AtomicU64; 8],
}

impl SlotCell {
    const fn new() -> Self {
        Self {
            words: [const { AtomicU64::new(0) }; 8],
        }
    }

    /// Write a `MatmulArgs` payload (`Release`).
    #[inline]
    fn store_matmul_args(&self, args: MatmulArgs) {
        // Layout: 3 ptrs + 2 usize = 5 u64s (assuming usize=u64 on x86_64/aarch64).
        // SAFETY: AtomicU64::store with Release ensures publish order.
        self.words[0].store(args.x_ptr as u64, Ordering::Relaxed);
        self.words[1].store(args.w_ptr as u64, Ordering::Relaxed);
        self.words[2].store(args.out_ptr as u64, Ordering::Relaxed);
        self.words[3].store(args.in_dim as u64, Ordering::Relaxed);
        // Final word with Release — publishes all prior Relaxed stores.
        self.words[4].store(args.out_dim as u64, Ordering::Release);
    }

    /// Read a `MatmulArgs` payload (`Acquire`).
    #[inline]
    fn load_matmul_args(&self) -> MatmulArgs {
        // Read final word with Acquire — synchronises with the Release store.
        let out_dim = self.words[4].load(Ordering::Acquire) as usize;
        let x_ptr = self.words[0].load(Ordering::Relaxed) as *const f32;
        let w_ptr = self.words[1].load(Ordering::Relaxed) as *const u8;
        let out_ptr = self.words[2].load(Ordering::Relaxed) as *mut f32;
        let in_dim = self.words[3].load(Ordering::Relaxed) as usize;
        MatmulArgs {
            x_ptr,
            w_ptr,
            out_ptr,
            in_dim,
            out_dim,
        }
    }

    /// Write a `RowRange` payload (`Release`).
    #[inline]
    fn store_row_range(&self, r: RowRange) {
        self.words[0].store(r.start as u64, Ordering::Relaxed);
        self.words[1].store(r.end as u64, Ordering::Release);
    }

    /// Read a `RowRange` payload (`Acquire`).
    #[inline]
    fn load_row_range(&self) -> RowRange {
        let end = self.words[1].load(Ordering::Acquire) as usize;
        let start = self.words[0].load(Ordering::Relaxed) as usize;
        RowRange { start, end }
    }
}

// ─────────────────────────────────────────────────────────────────
// Per-core indexing
// ─────────────────────────────────────────────────────────────────

/// Each core has a unique index in `[0, MAX_CORES)`. Index 0 is the
/// BSP, index 1.. are APs in the order they registered. APIC ID is
/// preserved separately via [`CORE_APIC_IDS`] for diagnostics.
///
/// AP code reads this via [`current_core_index`], which is set during
/// AP bring-up via [`set_current_core_index`]. The BSP has it pre-set
/// to 0 by static initialization.
static CORE_INDEX_NEXT: AtomicUsize = AtomicUsize::new(1); // 0 is reserved for BSP

/// Per-CPU index storage. On x86_64 this is read via GS-relative
/// addressing once `KERNEL_GS_BASE` is set up; on aarch64 via TPIDR_EL1.
/// For v0 we keep it simple: a `static AtomicU32` indexed by APIC ID
/// (set by AP code right after registration). The trampoline writes
/// the AP's logical core index into its own stack frame as the first
/// thing it does in long mode.
///
/// The hot path uses `read_current_core_index_via_cpuid` on x86 if
/// the per-CPU register hasn't been wired up; aarch64 uses MPIDR_EL1
/// affinity bits — these are architecture-specific helpers and live
/// in `arch::*::cpu_index()` (not yet implemented; placeholder uses
/// the slow APIC-id fallback below).
static APIC_ID_TO_CORE_INDEX: [AtomicU32; MAX_CORES] =
    [const { AtomicU32::new(u32::MAX) }; MAX_CORES];

/// APIC ID of each core, indexed by core index. Populated by AP
/// registration. Diagnostic only — hot-path never reads this.
static CORE_APIC_IDS: [AtomicU32; MAX_CORES] = [const { AtomicU32::new(u32::MAX) }; MAX_CORES];

/// AMD topology IDs per logical core. `u32::MAX` means the CPU did not
/// expose CPUID Fn8000_001E or the AP has not registered yet. These
/// fields are diagnostic and policy inputs only; the matmul bit-contract
/// does not depend on their exact value.
const TOPOLOGY_ID_UNKNOWN: u32 = u32::MAX;
static CORE_TOPOLOGY_NODE_IDS: [AtomicU32; MAX_CORES] =
    [const { AtomicU32::new(TOPOLOGY_ID_UNKNOWN) }; MAX_CORES];
static CORE_TOPOLOGY_UNIT_IDS: [AtomicU32; MAX_CORES] =
    [const { AtomicU32::new(TOPOLOGY_ID_UNKNOWN) }; MAX_CORES];

/// Reserve and return a fresh logical core index. Called once per AP
/// during bring-up; the AP must remember the returned index for its
/// entire lifetime.
///
/// Returns `Some(idx)` if a slot was available, `None` if the AP count
/// would exceed [`MAX_CORES`].
#[inline]
pub fn allocate_core_index(apic_id: u32) -> Option<usize> {
    allocate_core_index_with_topology(apic_id, TOPOLOGY_ID_UNKNOWN, TOPOLOGY_ID_UNKNOWN)
}

/// Reserve a logical core index and record topology metadata.
///
/// The topology fields are used only by [`ParallelMatmulContext`] when
/// a runtime tune selects one logical CPU per physical AMD compute unit.
/// They do not change row ownership or the floating-point operation
/// order within an output row.
#[inline]
pub fn allocate_core_index_with_topology(
    apic_id: u32,
    node_id: u32,
    compute_unit_id: u32,
) -> Option<usize> {
    let idx = CORE_INDEX_NEXT.fetch_add(1, Ordering::AcqRel);
    if idx >= MAX_CORES {
        // Roll back so we don't drift forever (won't be assigned, but
        // the next allocate sees a valid range).
        CORE_INDEX_NEXT.store(MAX_CORES, Ordering::Release);
        return None;
    }
    CORE_APIC_IDS[idx].store(apic_id, Ordering::Release);
    // Reverse map APIC ID → core index (small APIC IDs only; bounded
    // by MAX_CORES — for the AMD EPYC 9354P with sequential APIC IDs
    // this is fine).
    if (apic_id as usize) < MAX_CORES {
        APIC_ID_TO_CORE_INDEX[apic_id as usize].store(idx as u32, Ordering::Release);
    }
    CORE_TOPOLOGY_NODE_IDS[idx].store(node_id, Ordering::Release);
    CORE_TOPOLOGY_UNIT_IDS[idx].store(compute_unit_id, Ordering::Release);
    Some(idx)
}

/// Record topology metadata for the BSP once CPUID is available.
#[inline]
pub fn bsp_record_topology(node_id: u32, compute_unit_id: u32) {
    CORE_TOPOLOGY_NODE_IDS[0].store(node_id, Ordering::Release);
    CORE_TOPOLOGY_UNIT_IDS[0].store(compute_unit_id, Ordering::Release);
}

/// Return the logical core index for the given APIC ID, or `u32::MAX`
/// if the APIC ID is out of range / not yet registered. Slow path
/// (used by diagnostics, not the hot loop).
#[inline]
pub fn core_index_of_apic(apic_id: u32) -> u32 {
    if (apic_id as usize) >= MAX_CORES {
        return u32::MAX;
    }
    APIC_ID_TO_CORE_INDEX[apic_id as usize].load(Ordering::Acquire)
}

/// Return the APIC ID assigned to the given logical core index, or
/// `u32::MAX` if the core hasn't registered.
#[inline]
pub fn apic_id_of_core(core_idx: usize) -> u32 {
    if core_idx >= MAX_CORES {
        return u32::MAX;
    }
    CORE_APIC_IDS[core_idx].load(Ordering::Acquire)
}

/// Return `(node_id, compute_unit_id)` for a logical core index, or
/// `(u32::MAX, u32::MAX)` if no topology metadata is known.
#[inline]
pub fn topology_of_core(core_idx: usize) -> (u32, u32) {
    if core_idx >= MAX_CORES {
        return (TOPOLOGY_ID_UNKNOWN, TOPOLOGY_ID_UNKNOWN);
    }
    (
        CORE_TOPOLOGY_NODE_IDS[core_idx].load(Ordering::Acquire),
        CORE_TOPOLOGY_UNIT_IDS[core_idx].load(Ordering::Acquire),
    )
}

// ─────────────────────────────────────────────────────────────────
// AP worker loop
// ─────────────────────────────────────────────────────────────────

/// Worker loop entered by every AP after architecture-specific bring-up.
///
/// Contract:
/// 1. AP has its logical core index in `core_idx` (allocated via
///    [`allocate_core_index`] from `arch::*::ap_entry`).
/// 2. AP has called [`ap_register`] to bump the registration count.
/// 3. AP enters this loop and never returns.
///
/// The loop polls `WORK_READY[core_idx]` with a `pause` (`spin_loop`)
/// to relax MESI pressure on the shared cache line. On `true`:
/// - Read shared work fn + args from [`SHARED_WORK`] + own
///   [`WORK_RANGE`]`[i]`.
/// - Dispatch on tag: MATMUL → call fn; HALT → break.
/// - Clear the ready flag.
/// - Arrive at [`MATMUL_TREE_BARRIER`].
///
/// **Why not WFE/HLT?** APs busy-wait. On EPYC with no DVFS pressure
/// this is the right choice for sub-millisecond matmuls (the wake
/// latency of HLT > the work itself). For longer idle periods between
/// inference tokens, future work may introduce a power-aware variant.
/// For now: tight spin with `pause`.
pub fn ap_worker_loop(core_idx: usize) -> ! {
    let mut last_epoch = 0u64;
    loop {
        // Spin until our slot is published with a new dispatch epoch.
        let work_epoch = loop {
            let epoch = WORK_EPOCH[core_idx].epoch.load(Ordering::Acquire);
            if epoch != 0 && epoch != last_epoch {
                break epoch;
            }
            if HALT_REQUESTED.halt.load(Ordering::Acquire) {
                // Architecture-specific halt; the AP has no return.
                ap_park();
            }
            core::hint::spin_loop();
        };
        if HALT_REQUESTED.halt.load(Ordering::Acquire) {
            ap_park();
        }

        // Read the shared (fn, args) tuple and our private range.
        // ACQUIRE on fn_ptr synchronises with the BSP's RELEASE
        // store of fn_ptr, which is itself ordered after the BSP's
        // SHARED_WORK.args store (program order on BSP).
        let raw_fn = SHARED_WORK.fn_ptr.load(Ordering::Acquire);
        let args = SHARED_WORK.args.load_matmul_args();
        let range = WORK_RANGE[core_idx].load_row_range();

        if !raw_fn.is_null() {
            // SAFETY: BSP published a valid MatmulFn pointer; the
            // signature is fixed; pointers inside `args` are arena-
            // backed and remain valid for the duration of this call.
            let f: MatmulFn = unsafe { core::mem::transmute(raw_fn) };
            unsafe { f(&args, range) };
        }
        // Else: idle wakeup (shouldn't happen if BSP is well-behaved);
        // we still arrive at the barrier so the BSP doesn't deadlock.

        // Remember the consumed epoch before signalling completion.
        // (No WORK_READY clear: the flag is not part of the dispatch
        // protocol — see the static's doc comment.)
        last_epoch = work_epoch;
        MATMUL_TREE_BARRIER.arrive_in(barrier_group_of(core_idx), work_epoch);
    }
}

/// Architecture-specific final park. Default impl: tight loop with
/// `spin_loop`. Architectures may override this via the
/// `arch::*::park_forever()` symbol (linker resolution); the default
/// is sufficient for the current bare-metal target.
#[inline]
fn ap_park() -> ! {
    loop {
        core::hint::spin_loop();
    }
}

/// Request all APs to exit their worker loops (used in shutdown /
/// panic paths). Sets the global halt flag, then nudges every slot
/// so APs in `spin_loop` see the change on their next iteration.
#[allow(dead_code)]
pub fn request_halt() {
    HALT_REQUESTED.halt.store(true, Ordering::Release);
    // Nudge: flip every ready flag so APs in `spin_loop` reach the
    // halt check immediately.
    for slot in &WORK_READY {
        slot.store(true, Ordering::Release);
    }
}

// ─────────────────────────────────────────────────────────────────
// ParallelMatmulContext — the BSP-side dispatcher
// ─────────────────────────────────────────────────────────────────

/// BSP-side handle for publishing a parallel matmul.
///
/// Typical use:
/// ```ignore
/// let ctx = ParallelMatmulContext::for_active_cores();
/// ctx.dispatch_matmul(
///     my_matmul_fn,
///     MatmulArgs { x_ptr, w_ptr, out_ptr, in_dim, out_dim },
///     out_dim,        // total output rows
/// );
/// // After this returns, all rows of `out` have been written by some core.
/// ```
///
/// The dispatcher:
/// 1. Splits `total_rows` across `n_cores`.
/// 2. Resets [`MATMUL_BARRIER`] to `n_cores` arrivals.
/// 3. Publishes per-AP slots (cores 1..n_cores).
/// 4. Executes the BSP's own slice in-line.
/// 5. Arrives at the barrier and waits for APs.
pub struct ParallelMatmulContext {
    pub n_cores: u32,
    participants: [usize; MAX_CORES],
}

/// Default minimum amount of row work we want to hand to one core before
/// paying the AP publish + barrier cost. The row-level bit-exactness
/// contract is unchanged: each output row is still reduced by exactly
/// one core in the same K-order as the scalar path. This only chooses
/// how many APs participate in a given projection.
pub const DEFAULT_MIN_MATMUL_ROWS_PER_CORE: usize = 16;
pub const MIN_MATMUL_ROWS_PER_CORE_FLOOR: usize = 1;
pub const MIN_MATMUL_ROWS_PER_CORE_CEILING: usize = 4096;
pub const DEFAULT_MAX_MATMUL_CORES: usize = MAX_CORES;
pub const MATMUL_THREAD_POLICY_ALL: u32 = 0;
pub const MATMUL_THREAD_POLICY_UNIQUE_CORE: u32 = 1;
/// Default BSP row-share discount (percent). 0 = even split,
/// bit-identical ranges to the pre-N4 dispatch. See
/// [`row_range_for_discounted`].
pub const DEFAULT_MATMUL_BSP_DISCOUNT_PCT: usize = 0;
// `MATMUL_BSP_DISCOUNT_PCT_CEILING` is defined in `smp_partition` (used
// by `row_range_for_discounted`) and re-exported at the top of this
// module; the discount setter below clamps against it.

static MIN_MATMUL_ROWS_PER_CORE: AtomicUsize = AtomicUsize::new(DEFAULT_MIN_MATMUL_ROWS_PER_CORE);
static MAX_MATMUL_CORES_SELECTED: AtomicUsize = AtomicUsize::new(DEFAULT_MAX_MATMUL_CORES);
static MATMUL_BSP_DISCOUNT_PCT: AtomicUsize = AtomicUsize::new(DEFAULT_MATMUL_BSP_DISCOUNT_PCT);

// ─── ParallelMatmulContext cache (ADR-029 v8.4) ───────────────────
//
// The participants array depends only on the tuning atomics below:
//   active_cores, max_matmul_cores, matmul_thread_policy
// (Not on `total_rows`; that only scales how many of the cached
//  participants are USED per dispatch.)
//
// Those tuning atomics change rarely (mostly never after boot). The
// `for_active_cores_for_rows` hot path was rebuilding the 512-byte
// participants array (`[usize; MAX_CORES=64]`) on every dispatch —
// ~200 dispatches/token × 200-500 cycles each = ~0.4-1 % overhead.
//
// Fix: cache (max_n_cores, participants[]) keyed by a u64 version
// counter that the `set_*` tuning functions bump. Dispatch fast path
// reads the version, returns the cached array if the version matches.
//
// The cache lives in its own static; no allocation, no mutex (writes
// happen via Acquire-Release on the version counter, body fields are
// only read after a version comparison that establishes happens-before).

static MATMUL_TUNING_VERSION: AtomicU64 = AtomicU64::new(0);

#[repr(C, align(64))]
struct CachedParticipants {
    /// Version this cache was built against. Hot path compares
    /// against `MATMUL_TUNING_VERSION.load(Acquire)`; mismatch
    /// triggers a rebuild.
    version: AtomicU64,
    /// Maximum number of cores the current tuning would use for a
    /// matmul large enough to saturate them. Per-dispatch n_cores
    /// is `min(needed_for_rows, max_n_cores)`.
    max_n_cores: AtomicU32,
    /// Logical participants in dispatch order. Only `[0..max_n_cores]`
    /// is valid.
    participants: [AtomicUsize; MAX_CORES],
}

impl CachedParticipants {
    const fn new() -> Self {
        Self {
            version: AtomicU64::new(u64::MAX), // miss on first read
            max_n_cores: AtomicU32::new(1),
            participants: [const { AtomicUsize::new(0) }; MAX_CORES],
        }
    }
}

static MATMUL_PARTICIPANTS_CACHE: CachedParticipants = CachedParticipants::new();

/// Bump the tuning version — invalidates the participants cache.
/// Called by every `set_*` tuning function below.
#[inline]
fn bump_tuning_version() {
    MATMUL_TUNING_VERSION.fetch_add(1, Ordering::AcqRel);
}
// Default to ALL logical CPUs. The earlier UNIQUE_CORE-by-default
// experiment (commit 863fdc8) was based on the llama.cpp folklore
// that AVX-512-heavy decode prefers one thread per physical core to
// avoid SMT-sibling FMA-pipe contention. EMPIRICAL CHERRY DATA
// (2026-05-17 boot, 0.4 tok/s baseline → 0.3 tok/s with UNIQUE_CORE)
// showed the opposite for Qwen3-1.7B Q4_K_M on EPYC 9354P: the
// workload is memory-bandwidth bound, not FMA-bound, and halving
// the number of active cores halved the parallel memory-request
// pressure on the IOD — net slowdown.
//
// The UNIQUE_CORE policy remains available via control-plane runtime
// toggle (`set_matmul_thread_policy unique-core`) for compute-bound
// workloads where SMT contention dominates, but is no longer the
// default.
static MATMUL_THREAD_POLICY: AtomicU32 = AtomicU32::new(MATMUL_THREAD_POLICY_ALL);

#[inline(always)]
pub fn min_matmul_rows_per_core() -> usize {
    MIN_MATMUL_ROWS_PER_CORE.load(Ordering::Acquire)
}

#[inline]
pub fn set_min_matmul_rows_per_core(rows: usize) -> usize {
    let clamped = rows.clamp(
        MIN_MATMUL_ROWS_PER_CORE_FLOOR,
        MIN_MATMUL_ROWS_PER_CORE_CEILING,
    );
    MIN_MATMUL_ROWS_PER_CORE.store(clamped, Ordering::Release);
    bump_tuning_version();
    clamped
}

#[inline]
pub fn reset_min_matmul_rows_per_core() -> usize {
    set_min_matmul_rows_per_core(DEFAULT_MIN_MATMUL_ROWS_PER_CORE)
}

#[inline(always)]
pub fn max_matmul_cores() -> usize {
    MAX_MATMUL_CORES_SELECTED.load(Ordering::Acquire)
}

#[inline]
pub fn set_max_matmul_cores(cores: usize) -> usize {
    let clamped = cores.clamp(1, MAX_CORES);
    MAX_MATMUL_CORES_SELECTED.store(clamped, Ordering::Release);
    bump_tuning_version();
    clamped
}

#[inline]
pub fn reset_max_matmul_cores() -> usize {
    set_max_matmul_cores(DEFAULT_MAX_MATMUL_CORES)
}

#[inline(always)]
pub fn matmul_thread_policy() -> u32 {
    MATMUL_THREAD_POLICY.load(Ordering::Acquire)
}

#[inline]
pub fn set_matmul_thread_policy(policy: u32) -> u32 {
    let selected = match policy {
        MATMUL_THREAD_POLICY_UNIQUE_CORE => MATMUL_THREAD_POLICY_UNIQUE_CORE,
        _ => MATMUL_THREAD_POLICY_ALL,
    };
    MATMUL_THREAD_POLICY.store(selected, Ordering::Release);
    bump_tuning_version();
    selected
}

#[inline]
pub fn reset_matmul_thread_policy() -> u32 {
    // Reset to documented default (ALL logical CPUs) — see the static
    // initialisation rationale.
    set_matmul_thread_policy(MATMUL_THREAD_POLICY_ALL)
}

#[inline]
pub fn matmul_thread_policy_label() -> &'static str {
    match matmul_thread_policy() {
        MATMUL_THREAD_POLICY_UNIQUE_CORE => "unique-core",
        _ => "all",
    }
}

/// Current BSP row-share discount in percent. Read ONCE per dispatch
/// (a mid-dispatch change must never split BSP and AP range
/// computations — see `dispatch_matmul_aligned`).
#[inline(always)]
pub fn matmul_bsp_discount_pct() -> usize {
    MATMUL_BSP_DISCOUNT_PCT.load(Ordering::Acquire)
}

#[inline]
pub fn set_matmul_bsp_discount_pct(pct: usize) -> usize {
    let clamped = pct.min(MATMUL_BSP_DISCOUNT_PCT_CEILING);
    MATMUL_BSP_DISCOUNT_PCT.store(clamped, Ordering::Release);
    // No participants-cache invalidation needed: the discount only
    // affects per-dispatch ranges, which are recomputed every call.
    clamped
}

#[inline]
pub fn reset_matmul_bsp_discount_pct() -> usize {
    set_matmul_bsp_discount_pct(DEFAULT_MATMUL_BSP_DISCOUNT_PCT)
}

#[inline]
pub fn reset_matmul_tuning() {
    let _ = reset_min_matmul_rows_per_core();
    let _ = reset_max_matmul_cores();
    let _ = reset_matmul_thread_policy();
    let _ = reset_matmul_bsp_discount_pct();
}

#[inline]
pub fn apply_cherry_production_matmul_tuning() {
    let _ = set_min_matmul_rows_per_core(DEFAULT_MIN_MATMUL_ROWS_PER_CORE);
    let _ = set_max_matmul_cores(MAX_CORES);
    let _ = set_matmul_thread_policy(MATMUL_THREAD_POLICY_ALL);
    let _ = set_matmul_bsp_discount_pct(DEFAULT_MATMUL_BSP_DISCOUNT_PCT);
}

#[inline]
pub fn effective_cores_for_rows(total_rows: usize) -> u32 {
    ParallelMatmulContext::for_active_cores_for_rows(total_rows).n_cores
}

impl ParallelMatmulContext {
    /// Build a context for the currently active core count.
    #[inline]
    pub fn for_active_cores() -> Self {
        let requested = active_cores().min(MAX_CORES as u32).max(1) as usize;
        Self::from_requested_cores(requested)
    }

    /// Build a context sized to the output-row count.
    ///
    /// Large projections such as LM-head still use all active cores.
    /// Smaller Q/K/V/O projections avoid waking 64 cores for only a
    /// handful of rows per core, which can otherwise turn the barrier
    /// into a larger cost than the arithmetic.
    ///
    /// ADR-029 v8.4: fast path reads the cached participants array
    /// when the tuning atomics have not changed since the last call.
    /// Only `n_cores` (which depends on `total_rows`) is recomputed.
    #[inline]
    pub fn for_active_cores_for_rows(total_rows: usize) -> Self {
        if total_rows == 0 {
            return Self::single_core();
        }
        let active = active_cores().min(MAX_CORES as u32).max(1);
        if active <= 1 {
            return Self::single_core();
        }

        // Fast path: use cached participants if version unchanged.
        let live_version = MATMUL_TUNING_VERSION.load(Ordering::Acquire);
        let cached_version = MATMUL_PARTICIPANTS_CACHE.version.load(Ordering::Acquire);
        let max_n_cores = if live_version == cached_version {
            MATMUL_PARTICIPANTS_CACHE
                .max_n_cores
                .load(Ordering::Acquire)
        } else {
            Self::rebuild_cache(live_version)
        };

        let rows_per_core = min_matmul_rows_per_core();
        let mut needed = total_rows.saturating_add(rows_per_core - 1) / rows_per_core;
        if needed == 0 {
            needed = 1;
        }
        let target = (needed as u32).min(max_n_cores).max(1);
        Self::from_cached(target)
    }

    /// Build a context with up to `units` participants, IGNORING the
    /// rows-per-core granularity tuning (but still respecting
    /// `active_cores`, `max_matmul_cores` and the thread policy via
    /// the participants cache).
    ///
    /// For work units that are much heavier than matmul rows —
    /// e.g. attention heads, where one unit is a full
    /// score+softmax+weighted-sum pass over the context — the
    /// rows-per-core heuristic (default 16) would collapse 16 heads
    /// onto a single core. One core per unit is the right shape.
    #[inline]
    pub fn for_work_units(units: usize) -> Self {
        if units <= 1 {
            return Self::single_core();
        }
        let active = active_cores().min(MAX_CORES as u32).max(1);
        if active <= 1 {
            return Self::single_core();
        }
        let live_version = MATMUL_TUNING_VERSION.load(Ordering::Acquire);
        let cached_version = MATMUL_PARTICIPANTS_CACHE.version.load(Ordering::Acquire);
        let max_n_cores = if live_version == cached_version {
            MATMUL_PARTICIPANTS_CACHE
                .max_n_cores
                .load(Ordering::Acquire)
        } else {
            Self::rebuild_cache(live_version)
        };
        let target = (units as u32).min(max_n_cores).max(1);
        Self::from_cached(target)
    }

    /// Recompute the cached participants array. Returns the new
    /// `max_n_cores` value. Called only on tuning-version miss.
    #[cold]
    #[inline(never)]
    fn rebuild_cache(live_version: u64) -> u32 {
        let active = (active_cores().min(MAX_CORES as u32).max(1)) as usize;
        let cap = max_matmul_cores().min(active).max(1);
        // Select with the FULL cap; per-dispatch we slice to actual need.
        let (participants, n_cores) = select_matmul_participants(cap);
        for (i, slot) in MATMUL_PARTICIPANTS_CACHE.participants.iter().enumerate() {
            slot.store(participants[i], Ordering::Relaxed);
        }
        MATMUL_PARTICIPANTS_CACHE
            .max_n_cores
            .store(n_cores, Ordering::Release);
        // Version stored LAST so other threads see a consistent cache.
        MATMUL_PARTICIPANTS_CACHE
            .version
            .store(live_version, Ordering::Release);
        n_cores
    }

    /// Build a context from the cached participants array, taking
    /// only the first `n_cores` entries.
    #[inline]
    fn from_cached(n_cores: u32) -> Self {
        let mut participants = [0usize; MAX_CORES];
        let n = n_cores as usize;
        for (i, slot) in MATMUL_PARTICIPANTS_CACHE
            .participants
            .iter()
            .take(n)
            .enumerate()
        {
            participants[i] = slot.load(Ordering::Relaxed);
        }
        Self {
            n_cores,
            participants,
        }
    }

    #[inline]
    fn single_core() -> Self {
        let mut participants = [0usize; MAX_CORES];
        participants[0] = 0;
        Self {
            n_cores: 1,
            participants,
        }
    }

    #[inline]
    fn from_requested_cores(requested: usize) -> Self {
        let (participants, n_cores) = select_matmul_participants(requested);
        Self {
            n_cores,
            participants,
        }
    }

    /// Dispatch a matmul across all participating cores.
    ///
    /// **Single-core fast path:** if `n_cores == 1` (no APs woken),
    /// the call collapses to a direct in-line invocation with no
    /// barrier overhead. This guarantees that an SMP-disabled kernel
    /// pays zero runtime cost for the parallel layer.
    #[inline]
    pub fn dispatch_matmul(&self, matmul_fn: MatmulFn, args: MatmulArgs, total_rows: usize) {
        self.dispatch_matmul_aligned(matmul_fn, args, total_rows, 1);
    }

    /// [`dispatch_matmul`](Self::dispatch_matmul) with row-split
    /// boundaries aligned to `align` (see [`split_rows_aligned`]).
    /// `align = 1` is the plain even split.
    #[inline]
    pub fn dispatch_matmul_aligned(
        &self,
        matmul_fn: MatmulFn,
        args: MatmulArgs,
        total_rows: usize,
        align: usize,
    ) {
        if self.n_cores <= 1 {
            // SAFETY: caller upholds MatmulFn's safety contract.
            unsafe {
                matmul_fn(
                    &args,
                    RowRange {
                        start: 0,
                        end: total_rows,
                    },
                )
            };
            return;
        }

        // Per-participant ranges come from the closed-form
        // `row_range_for` instead of a materialised `[RowRange; 128]`
        // array — the array variant zero-initialises 2 KiB per dispatch
        // (~113 dispatches/token) for entries the dispatch never reads.
        // Identical ranges by construction (see row_range_for docs).
        //
        // The BSP discount (N4) is loaded ONCE per dispatch: a
        // control-plane `smp tune bsp-discount` racing the load must
        // never make BSP and APs disagree about the partition — rows
        // would be double-computed or skipped.
        let bsp_discount = matmul_bsp_discount_pct();

        // Tally per-group target counts. Each participant's group is
        // `barrier_group_of(participants[i])`; we count BSP + APs so
        // the BSP's own group is included.
        let mut per_group_targets = [0u32; BARRIER_GROUP_COUNT];
        let mut p = 0usize;
        while (p as u32) < self.n_cores {
            let g = barrier_group_of(self.participants[p]);
            if g < BARRIER_GROUP_COUNT {
                per_group_targets[g] += 1;
            }
            p += 1;
        }
        let work_epoch = NEXT_WORK_EPOCH.epoch.fetch_add(1, Ordering::AcqRel);
        let work_epoch = if work_epoch == 0 {
            NEXT_WORK_EPOCH.epoch.store(2, Ordering::Release);
            1
        } else {
            work_epoch
        };
        MATMUL_TREE_BARRIER.reset(&per_group_targets, work_epoch);

        // ── ADR-029 v8.4: shared-slot publish ───────────────────
        //
        // Write (fn, args) once into SHARED_WORK on one cacheline.
        // APs read this after observing their own WORK_READY flag
        // — the per-AP Release on WORK_READY publishes both the
        // per-AP RowRange AND the shared (fn, args) tuple (the
        // Release/Acquire pair on WORK_READY synchronises with the
        // CPU memory model on x86_64; on aarch64 the Release store
        // emits a `stlr` that orders all prior stores).
        //
        // Important: we set SHARED_WORK BEFORE any per-AP
        // WORK_READY=true.
        SHARED_WORK.args.store_matmul_args(args);
        SHARED_WORK
            .fn_ptr
            .store(matmul_fn as *mut (), Ordering::Release);

        // Publish per-AP slots: row range + ready flag only.
        // `participants[0]` is always core 0 (the BSP); remaining
        // entries are logical AP indices chosen by the current
        // tuning policy.
        let mut i = 1;
        while (i as u32) < self.n_cores {
            let core_idx = self.participants[i];
            WORK_RANGE[core_idx].store_row_range(row_range_for_discounted(
                total_rows,
                self.n_cores,
                i,
                align,
                bsp_discount,
            ));
            // Epoch publish IS the wake signal. The AP's ACQUIRE load
            // on WORK_EPOCH synchronises with this Release store and
            // therefore with the WORK_RANGE + SHARED_WORK stores above.
            // (WORK_READY is deliberately not written here — all 128
            // flags share two cachelines and the per-AP stores were a
            // pure false-sharing cost; see the static's doc comment.)
            WORK_EPOCH[core_idx]
                .epoch
                .store(work_epoch, Ordering::Release);
            i += 1;
        }

        // BSP executes its own slice (range 0) in-line.
        // SAFETY: same contract as the AP path; the BSP's range is
        // disjoint from every AP's, so concurrent writes to `out` are
        // race-free.
        unsafe {
            matmul_fn(
                &args,
                row_range_for_discounted(total_rows, self.n_cores, 0, align, bsp_discount),
            )
        };

        // BSP arrives at the barrier and waits for APs.
        MATMUL_TREE_BARRIER.arrive_in(barrier_group_of(self.participants[0]), work_epoch);
        if !MATMUL_TREE_BARRIER.wait_complete() {
            use core::fmt::Write;
            let _ = writeln!(
                crate::arch::Serial,
                "SMP WARNING: matmul barrier timed out — recomputing AP rows on BSP \
                 and degrading to single-core dispatch"
            );
            // Degraded-mode recovery instead of a permanent halt:
            //
            // 1. Recompute every AP slice on the BSP. `matmul_fn` is a
            //    pure deterministic function of (args, range), so
            //    rewriting rows an AP already finished produces
            //    identical bytes, and rows a wedged AP never wrote are
            //    filled in correctly — the output stays bit-exact. If
            //    the AP resumes mid-flight it rewrites the same bytes
            //    (stale-epoch arrivals are ignored by the TreeBarrier).
            // 2. Cap future dispatches to single-core. A permanently
            //    wedged AP would otherwise stall EVERY subsequent
            //    matmul for the full timeout budget; serial execution
            //    is slow but keeps the machine alive and observable.
            let mut i = 1;
            while (i as u32) < self.n_cores {
                // SAFETY: same contract as the parallel path; the BSP
                // re-executes the disjoint AP slices sequentially
                // (same `bsp_discount` as the original partition).
                unsafe {
                    matmul_fn(
                        &args,
                        row_range_for_discounted(total_rows, self.n_cores, i, align, bsp_discount),
                    )
                };
                i += 1;
            }
            let _ = set_max_matmul_cores(1);
        }
    }
}

fn select_matmul_participants(requested: usize) -> ([usize; MAX_CORES], u32) {
    let active = (active_cores().min(MAX_CORES as u32).max(1)) as usize;
    let cap = max_matmul_cores().min(active).max(1);
    let target = requested.max(1).min(cap);
    if target <= 1 {
        let mut participants = [0usize; MAX_CORES];
        participants[0] = 0;
        return (participants, 1);
    }

    if matmul_thread_policy() == MATMUL_THREAD_POLICY_UNIQUE_CORE {
        if let Some(selected) = select_unique_compute_units(active, target) {
            return selected;
        }
    }

    select_sequential_participants(target)
}

fn select_sequential_participants(target: usize) -> ([usize; MAX_CORES], u32) {
    let mut participants = [0usize; MAX_CORES];
    let mut i = 0;
    while i < target {
        participants[i] = i;
        i += 1;
    }
    (participants, target as u32)
}

fn select_unique_compute_units(active: usize, target: usize) -> Option<([usize; MAX_CORES], u32)> {
    let mut participants = [0usize; MAX_CORES];
    let mut seen_nodes = [TOPOLOGY_ID_UNKNOWN; MAX_CORES];
    let mut seen_units = [TOPOLOGY_ID_UNKNOWN; MAX_CORES];

    let (bsp_node, bsp_unit) = topology_of_core(0);
    if bsp_node == TOPOLOGY_ID_UNKNOWN || bsp_unit == TOPOLOGY_ID_UNKNOWN {
        return None;
    }

    participants[0] = 0;
    seen_nodes[0] = bsp_node;
    seen_units[0] = bsp_unit;
    let mut selected = 1usize;
    let mut seen = 1usize;

    let mut idx = 1usize;
    while idx < active && selected < target {
        let (node, unit) = topology_of_core(idx);
        if node != TOPOLOGY_ID_UNKNOWN
            && unit != TOPOLOGY_ID_UNKNOWN
            && !topology_seen(&seen_nodes, &seen_units, seen, node, unit)
        {
            participants[selected] = idx;
            seen_nodes[seen] = node;
            seen_units[seen] = unit;
            selected += 1;
            seen += 1;
        }
        idx += 1;
    }

    if selected <= 1 {
        return None;
    }
    Some((participants, selected as u32))
}

fn topology_seen(
    seen_nodes: &[u32; MAX_CORES],
    seen_units: &[u32; MAX_CORES],
    seen: usize,
    node: u32,
    unit: u32,
) -> bool {
    let mut i = 0usize;
    while i < seen {
        if seen_nodes[i] == node && seen_units[i] == unit {
            return true;
        }
        i += 1;
    }
    false
}

// ─────────────────────────────────────────────────────────────────
// NUMA / first-touch warmup
// ─────────────────────────────────────────────────────────────────

/// Type-erased kernel used by [`parallel_touch`]. Reads one byte from
/// every cacheline in `args.in_dim` bytes starting at `args.x_ptr`.
/// Compiles to a tight scalar loop — the point is to *fetch* the lines
/// into the calling core's L1/L2 (and thus its CCD's L3 slice), not to
/// compute anything.
unsafe fn touch_kernel(args: &MatmulArgs, range: RowRange) {
    let base = args.x_ptr as *const u8;
    let stride = 64usize;
    let start = range.start * stride;
    let end = (range.end * stride).min(args.in_dim);
    let mut acc: u8 = 0;
    let mut off = start;
    while off < end {
        // Volatile read prevents the compiler from optimising the
        // entire walk away. The accumulator + black_box-style fence
        // at the end keeps `acc` alive across the loop.
        acc = acc.wrapping_add(core::ptr::read_volatile(base.add(off)));
        off += stride;
    }
    // Force `acc` to be observed so the loop body is not eliminated.
    core::ptr::read_volatile(&acc);
}

/// Parallel "touch" of a byte region. Each participating core reads one
/// byte from every 64-byte cacheline in its row-slice. On NPS1 EPYC the
/// physical pages are channel-interleaved at cacheline granularity, so
/// this does not change *DRAM placement*; what it does change is *L3
/// residency* — each line lands in the CCD-L3 slice of the core that
/// fetched it, so subsequent reads from that same core hit L3 instead
/// of paying the DRAM round-trip.
///
/// Only meaningful for working sets that fit in aggregate L3 (≤ 256 MB
/// on the 9354P). For full Qwen3-1.7B Q4_K_M (~1.2 GB resident) the
/// model exceeds aggregate L3 and only the most-recently-touched 256 MB
/// stays hot. For smaller boot LLMs (e.g. Qwen3-0.6B at ~300 MB) the
/// model fits and the warmup is a clear win.
///
/// Pillar 7: pure SMP primitive, no architecture-specific code. Pillar
/// 1: zero allocation, ASCII-style hot-loop. Safe to invoke from the
/// Zero control plane.
pub fn parallel_touch(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let cacheline_count = bytes.len().div_ceil(64);
    let ctx = ParallelMatmulContext::for_active_cores_for_rows(cacheline_count);
    if ctx.n_cores <= 1 {
        // Single-core fast path — just do it ourselves.
        let args = MatmulArgs {
            x_ptr: bytes.as_ptr() as *const f32,
            w_ptr: core::ptr::null(),
            out_ptr: core::ptr::null_mut(),
            in_dim: bytes.len(),
            out_dim: 0,
        };
        unsafe {
            touch_kernel(
                &args,
                RowRange {
                    start: 0,
                    end: cacheline_count,
                },
            )
        };
        return;
    }
    let args = MatmulArgs {
        x_ptr: bytes.as_ptr() as *const f32,
        w_ptr: core::ptr::null(),
        out_ptr: core::ptr::null_mut(),
        in_dim: bytes.len(),
        out_dim: 0,
    };
    ctx.dispatch_matmul(touch_kernel, args, cacheline_count);
}

// ─────────────────────────────────────────────────────────────────
// Tests (host-only — no_std friendly)
//
// The pure partition tests (`split_rows*`, `row_range_for*`) live with
// their functions in `smp_partition.rs` and run under the host test
// harness (`cargo test --workspace`). The tests below exercise the SMP
// runtime (active-core count, adaptive context, spin barrier) and remain
// here as bare-metal-coupled regression guards.
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_context_keeps_large_matmuls_wide() {
        set_active_cores(64);
        let ctx = ParallelMatmulContext::for_active_cores_for_rows(151_936);
        assert_eq!(ctx.n_cores, 64);
    }

    #[test]
    fn adaptive_context_avoids_tiny_tiles() {
        set_active_cores(64);
        let ctx = ParallelMatmulContext::for_active_cores_for_rows(512);
        assert_eq!(ctx.n_cores, 32);
    }

    #[test]
    fn adaptive_context_keeps_qwen_projection_dims_wide() {
        set_active_cores(64);
        let ctx = ParallelMatmulContext::for_active_cores_for_rows(1024);
        assert_eq!(ctx.n_cores, 64);
        let ctx = ParallelMatmulContext::for_active_cores_for_rows(2048);
        assert_eq!(ctx.n_cores, 64);
    }

    #[test]
    fn adaptive_context_still_limits_very_small_tiles() {
        set_active_cores(64);
        let ctx = ParallelMatmulContext::for_active_cores_for_rows(256);
        assert_eq!(ctx.n_cores, 16);
    }

    #[test]
    fn spinbarrier_completes_after_target_arrivals() {
        let b = SpinBarrier::new();
        b.reset(3);
        b.arrive();
        b.arrive();
        b.arrive();
        assert!(b.wait_complete()); // doesn't deadlock
    }
}
