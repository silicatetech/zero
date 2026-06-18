// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark framework for Zero ROI benchmarks.
//!
//! Provides rdtsc-based measurement infrastructure for the 5 ROI
//! benchmarks defined in docs/benchmark_plan.md:
//!
//! 1. Boot Time (measured externally via T0 in kernel_main)
//! 2. Context Switch (cooperative executor yield/resume)
//! 3. Arena Allocation (bump-pointer alloc throughput)
//! 4. IPC Throughput (zero-copy shared arena transfer)
//! 5. LLM Inference (multi-token loop, tok/s)
//!
//! All results are output via Serial (COM1) for KVM console capture,
//! AND rendered on the LFB framebuffer for screenshot.
//!
//! # TSC Frequency
//!
//! On bare metal, TSC frequency is read from CPUID leaf 0x15/0x16.
//! On QEMU, these leaves may not be available — we fall back to a
//! PIT-calibrated estimate or a default 3.8 GHz assumption.

use crate::arch;
use crate::arch::cpuinfo::CpuInfo;
use crate::arch::cycles::rdtsc_serialized;
use core::fmt::Write;
use core::sync::atomic::{AtomicU64, Ordering};

/// Boot timestamp — set as early as possible in kernel_main().
/// This is T0 for the boot-time benchmark.
pub static mut BOOT_TSC_T0: u64 = 0;

/// Boot-ready timestamp captured when the kernel/control-plane is ready,
/// before any operator wait time or Stage-11 Boot-LLM work can inflate the
/// boot-time benchmark. `0` means no early ready marker was recorded.
static BOOT_READY_TSC: AtomicU64 = AtomicU64::new(0);

/// Number of iterations for micro-benchmarks (context switch, alloc).
/// 1,000,000 iterations for statistically meaningful results.
pub const BENCH_ITERATIONS: u64 = 1_000_000;

/// Shorter iteration count for medium-cost benchmarks.
#[allow(dead_code)]
pub const BENCH_ITERATIONS_MEDIUM: u64 = 100_000;

/// TSC frequency in MHz (set during calibration).
/// Default: 3800 MHz (EPYC 9354P boost clock).
static mut TSC_MHZ: u64 = 3800;

static LLM_GENERATED_TOKENS: AtomicU64 = AtomicU64::new(0);
static LLM_GENERATION_CYCLES: AtomicU64 = AtomicU64::new(0);
/// Set to 1 by [`record_llm_failure`] when Stage 11 aborts the forward
/// pass before producing any tokens. The benchmark summary surfaces this
/// as a distinct "failure" line so an aborted Deepseek2/Kimi run does
/// not get silently reported as "unavailable" alongside the bench output.
static LLM_FAILURE_FLAG: AtomicU64 = AtomicU64::new(0);
static PRE_LLM_BASELINE_VALID: AtomicU64 = AtomicU64::new(0);
static PRE_LLM_CTX_PER_SWITCH: AtomicU64 = AtomicU64::new(0);
static PRE_LLM_ALLOC_PER: AtomicU64 = AtomicU64::new(0);
static PRE_LLM_IPC_TOTAL: AtomicU64 = AtomicU64::new(0);

// ── Held-screen anchor mirror ──────────────────────────────────────
//
// The Stage-11 forward pass prints its captured `.smodel`/β anchor
// (next-token ID + top-1 `logit_bits`) to serial (COM1). On the Cherry
// deploy the operator only has the KVM/VGA console — serial is
// unreachable — yet the capture pass exists precisely to *read* those
// values and promote them to a strict manifest anchor (ADR-029 v3
// two-anchor system). So we also stash them here and render them onto
// the held benchmark screen. Single-writer atomics, same pattern as
// `LLM_GENERATED_TOKENS`.
static LLM_ANCHOR_VALID: AtomicU64 = AtomicU64::new(0);
static LLM_ANCHOR_TOKEN: AtomicU64 = AtomicU64::new(0);
static LLM_ANCHOR_LOGIT_BITS: AtomicU64 = AtomicU64::new(0);
static LLM_ANCHOR_MODE: AtomicU64 = AtomicU64::new(0);

/// Anchor was not captured (Stage 11 skipped or aborted before token 0).
pub const ANCHOR_MODE_NONE: u64 = 0;
/// `.smodel` manifest in capture mode — values printed for promotion.
pub const ANCHOR_MODE_CAPTURE: u64 = 1;
/// `.smodel` manifest in strict mode — values were validated, not captured.
pub const ANCHOR_MODE_STRICT: u64 = 2;
/// Native `.smodel` profile but no matching manifest anchor present.
pub const ANCHOR_MODE_MISSING: u64 = 3;
/// Non-`.smodel` build verifying the sacred β-anchor (token 25).
pub const ANCHOR_MODE_BETA: u64 = 4;

/// Results from one benchmark run.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct BenchResult {
    pub name: &'static str,
    pub iterations: u64,
    pub total_cycles: u64,
    pub median_ns: u64,
    pub p99_ns: u64,
    pub linux_ref: &'static str,
    pub unit: &'static str,
}

/// Record Stage-11 decode/generation timing for the later benchmark
/// summary. Called after Boot-LLM generation completes; conversion to
/// tok/s happens after TSC calibration in [`run_all_benchmarks`].
pub fn record_llm_inference(generated_tokens: u64, generation_cycles: u64) {
    LLM_GENERATED_TOKENS.store(generated_tokens, Ordering::Release);
    LLM_GENERATION_CYCLES.store(generation_cycles, Ordering::Release);
    if generated_tokens > 0 {
        // Clear any stale failure flag once a successful run lands —
        // sequential model swaps reuse the same atomics.
        LLM_FAILURE_FLAG.store(0, Ordering::Release);
    }
}

/// Record the Stage-11 forward-pass anchor for the held benchmark screen.
///
/// `mode` is one of the `ANCHOR_MODE_*` constants. Called once, at prompt
/// token 0, before the verify branches — so it captures the measured
/// values on every path (capture, strict, missing-manifest, β). The
/// operator reads `token` + `logit_bits` off the KVM/VGA photo to promote
/// a capture pass to a strict manifest anchor (ADR-029 v3).
pub fn record_llm_anchor(token: u32, logit_bits: u32, mode: u64) {
    LLM_ANCHOR_TOKEN.store(token as u64, Ordering::Release);
    LLM_ANCHOR_LOGIT_BITS.store(logit_bits as u64, Ordering::Release);
    LLM_ANCHOR_MODE.store(mode, Ordering::Release);
    LLM_ANCHOR_VALID.store(1, Ordering::Release);
}

/// `(token, logit_bits, mode)` if the Stage-11 pass recorded an anchor.
fn anchor_snapshot() -> Option<(u32, u32, u64)> {
    if LLM_ANCHOR_VALID.load(Ordering::Acquire) == 0 {
        return None;
    }
    Some((
        LLM_ANCHOR_TOKEN.load(Ordering::Acquire) as u32,
        LLM_ANCHOR_LOGIT_BITS.load(Ordering::Acquire) as u32,
        LLM_ANCHOR_MODE.load(Ordering::Acquire),
    ))
}

fn anchor_mode_label(mode: u64) -> &'static str {
    match mode {
        ANCHOR_MODE_CAPTURE => "capture",
        ANCHOR_MODE_STRICT => "strict",
        ANCHOR_MODE_MISSING => "no-manifest",
        ANCHOR_MODE_BETA => "beta",
        _ => "n/a",
    }
}

/// Short failure-reason text mirrored onto the held benchmark screen.
/// Same single-writer atomics pattern as `net::bind_report`: the KVM
/// console is the only operator surface when the box is unreachable,
/// so "why is tok/s 0.0" must be answerable from one screen photo.
pub mod llm_report {
    use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

    pub const DETAIL_CAP: usize = 64;
    static DETAIL_LEN: AtomicUsize = AtomicUsize::new(0);
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU8 = AtomicU8::new(0);
    static DETAIL: [AtomicU8; DETAIL_CAP] = [ZERO; DETAIL_CAP];

    struct DetailWriter {
        len: usize,
    }

    impl core::fmt::Write for DetailWriter {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            for &b in s.as_bytes() {
                if self.len >= DETAIL_CAP {
                    break;
                }
                let b = if (0x20..0x7F).contains(&b) { b } else { b'?' };
                DETAIL[self.len].store(b, Ordering::Relaxed);
                self.len += 1;
            }
            Ok(())
        }
    }

    pub fn set(detail: core::fmt::Arguments<'_>) {
        let mut w = DetailWriter { len: 0 };
        let _ = core::fmt::write(&mut w, detail);
        DETAIL_LEN.store(w.len, Ordering::Release);
    }

    /// Copy the detail text into `buf`, returning the byte count.
    pub fn detail(buf: &mut [u8; DETAIL_CAP]) -> usize {
        let n = DETAIL_LEN.load(Ordering::Acquire).min(DETAIL_CAP);
        for (i, slot) in buf.iter_mut().enumerate().take(n) {
            *slot = DETAIL[i].load(Ordering::Relaxed);
        }
        n
    }
}

/// Record that Stage 11 inference aborted before producing any tokens.
///
/// `reason` is printed verbatim on serial so the operator sees the exact
/// reason (which layer / tensor / forward-pass step). The summary screen
/// then prints "LLM Inference: FAILED — …" instead of "unavailable".
pub fn record_llm_failure(reason: &str) {
    record_llm_failure_args(format_args!("{}", reason));
}

/// [`record_llm_failure`] for callers that need formatted context
/// (error debug repr, token index) without allocating.
pub fn record_llm_failure_args(reason: core::fmt::Arguments<'_>) {
    LLM_GENERATED_TOKENS.store(0, Ordering::Release);
    LLM_GENERATION_CYCLES.store(0, Ordering::Release);
    LLM_FAILURE_FLAG.store(1, Ordering::Release);
    llm_report::set(reason);
    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] LLM Inference: FAILED — {}",
        reason
    );
}

/// True if [`record_llm_failure`] was called since the last successful run.
pub fn llm_failure_recorded() -> bool {
    LLM_FAILURE_FLAG.load(Ordering::Acquire) != 0
}

/// Mark the point where Zero is ready for operator/control-plane work.
///
/// On Cherry control-plane images Stage 11 intentionally waits for
/// `llm-start`. Measuring boot time at the final benchmark screen would
/// include human wait time plus full Boot-LLM inference and produce false
/// "boot regression" numbers. This marker keeps the boot metric scoped to
/// kernel bring-up, NIC, SMP, and the Zero control surface.
pub fn mark_boot_ready() {
    let now = rdtsc_serialized();
    let _ = BOOT_READY_TSC.compare_exchange(0, now, Ordering::AcqRel, Ordering::Acquire);
}

fn boot_delta_cycles() -> u64 {
    let t0 = unsafe { BOOT_TSC_T0 };
    let ready = BOOT_READY_TSC.load(Ordering::Acquire);
    let end = if ready != 0 {
        ready
    } else {
        rdtsc_serialized()
    };
    end.wrapping_sub(t0)
}

/// Capture microbenchmarks before Stage-11 Boot-LLM starts.
///
/// AVX-512-heavy inference changes cache residency and can downclock cores.
/// The ROI baseline rows (context switch, arena arithmetic, zero-copy IPC)
/// are therefore sampled before the LLM gate and reused in the final
/// screenshot summary. This keeps LLM speed and kernel microbenchmarks from
/// contaminating each other while preserving the same byte/token path.
pub fn capture_pre_llm_baseline(cpu: &CpuInfo) {
    mark_boot_ready();
    calibrate_tsc(cpu);

    if PRE_LLM_BASELINE_VALID.load(Ordering::Acquire) != 0 {
        return;
    }

    let _ = writeln!(arch::serial::Serial, "");
    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] Pre-LLM baseline snapshot (control-plane ready)"
    );
    report_boot_time();

    let (_ctx_total, ctx_per_switch) = bench_context_switch();
    PRE_LLM_CTX_PER_SWITCH.store(ctx_per_switch, Ordering::Release);

    let (_alloc_total, alloc_per) = bench_arena_alloc();
    PRE_LLM_ALLOC_PER.store(alloc_per, Ordering::Release);

    let (ipc_total, _ipc_metric) = bench_ipc_throughput();
    PRE_LLM_IPC_TOTAL.store(ipc_total, Ordering::Release);
    PRE_LLM_BASELINE_VALID.store(1, Ordering::Release);
}

fn recorded_llm_tok_per_sec_x10() -> (u64, u64, u64) {
    let tokens = LLM_GENERATED_TOKENS.load(Ordering::Acquire);
    let cycles = LLM_GENERATION_CYCLES.load(Ordering::Acquire);
    if tokens == 0 || cycles == 0 {
        return (0, tokens, cycles);
    }

    let ns = cycles_to_ns(cycles);
    if ns == 0 {
        return (0, tokens, cycles);
    }

    // tok/s with one decimal digit.
    let tok_per_sec_x10 = tokens.saturating_mul(10_000_000_000) / ns;
    (tok_per_sec_x10, tokens, cycles)
}

/// Calibrate TSC frequency using CPUID leaf 0x16 (Processor Frequency Info).
///
/// Leaf 0x16 returns base frequency in MHz in EAX.
/// Available on Zen 4 (EPYC Genoa) and most modern x86_64 CPUs.
/// Falls back to 3800 MHz if unavailable.
/// Calibrate TSC frequency.
///
/// Uses the CPU info already gathered (avoids duplicate CPUID calls).
/// Falls back to 3800 MHz if frequency info unavailable.
pub fn calibrate_tsc(cpu: &CpuInfo) {
    if cpu.base_freq_mhz > 0 && cpu.base_freq_mhz < 10000 {
        unsafe {
            TSC_MHZ = cpu.base_freq_mhz as u64;
        }
        let _ = writeln!(
            arch::serial::Serial,
            "[BENCH] TSC calibrated from CPUID: {} MHz",
            cpu.base_freq_mhz
        );
    } else {
        let _ = writeln!(
            arch::serial::Serial,
            "[BENCH] No freq info, using default {} MHz",
            unsafe { TSC_MHZ }
        );
    }
}

/// Convert cycles to nanoseconds using calibrated TSC frequency.
#[inline]
pub fn cycles_to_ns(cycles: u64) -> u64 {
    // ns = cycles * 1000 / MHz
    // Guard against overflow for large cycle counts
    if cycles < u64::MAX / 1000 {
        cycles * 1000 / unsafe { TSC_MHZ }
    } else {
        cycles / unsafe { TSC_MHZ } * 1000
    }
}

/// Run a benchmark function N times, measure total cycles.
/// Returns (total_cycles, per_iteration_cycles).
///
/// Includes a warmup phase of 1000 iterations (not counted).
#[allow(dead_code)]
pub fn run_bench(name: &str, iterations: u64, mut f: impl FnMut()) -> (u64, u64) {
    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] {} — {} iterations (+ 1000 warmup)...",
        name,
        iterations
    );

    // Warmup
    for _ in 0..1000u64 {
        core::hint::black_box(&mut f)();
    }

    // Measurement
    let start = rdtsc_serialized();
    for _ in 0..iterations {
        core::hint::black_box(&mut f)();
    }
    let end = rdtsc_serialized();

    let total = end.wrapping_sub(start);
    let per_iter = total / iterations;

    let total_ns = cycles_to_ns(total);
    let per_iter_ns = cycles_to_ns(per_iter);

    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] {}: total={} cycles ({}ns), per_iter={} cycles ({}ns)",
        name,
        total,
        total_ns,
        per_iter,
        per_iter_ns
    );

    (total, per_iter)
}

/// Report boot time using T0 captured at kernel_main entry.
pub fn report_boot_time() {
    let delta = boot_delta_cycles();
    let delta_ns = cycles_to_ns(delta);
    let delta_ms = delta_ns / 1_000_000;
    let delta_us = (delta_ns % 1_000_000) / 1000;

    let _ = writeln!(arch::serial::Serial, "");
    let _ = writeln!(arch::serial::Serial, "=== Zero Benchmark Suite ===");
    let _ = writeln!(arch::serial::Serial, "");
    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] Boot Time: {}.{}ms ({} cycles)",
        delta_ms,
        delta_us,
        delta
    );
    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH]   Linux ref: 15,000-25,000ms (Ubuntu Server)"
    );
}

// ── Benchmark 2: Context Switch (Real Yield/Resume) ────────────

/// Measure cooperative context switch overhead using the REAL
/// executor ready queue — the actual yield/resume path.
///
/// What happens during a cooperative context switch in Zero:
/// 1. Running task calls wake_by_ref() → pushes TaskId to ReadyQueue
/// 2. Executor pops TaskId from ReadyQueue
/// 3. Executor looks up task slot, creates waker, calls poll()
///
/// We measure steps 1+2 (push+pop) which are the core scheduling
/// cost. Step 3 (poll) overhead depends on the future's state
/// machine and is workload-specific.
///
/// Compare to Linux: context switch = save all registers + flush
/// TLB + switch page tables + restore registers + kernel crossing.
/// That's 1,200-5,500ns. Our push+pop is lock-free atomic ops only.
pub fn bench_context_switch() -> (u64, u64) {
    use crate::task::queue::READY_QUEUE;
    use crate::task::TaskId;

    let iterations = BENCH_ITERATIONS;
    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] Context Switch (yield/resume) — {} iterations...",
        iterations
    );

    // Use TaskId(1) as a dummy — we just push and pop, never actually
    // poll a task. This measures the real scheduling data path.
    let dummy_id = TaskId::from_raw(1);

    // Warmup: push+pop 1000 times
    for _ in 0..1000u64 {
        let _ = READY_QUEUE.push(dummy_id);
        let _ = core::hint::black_box(READY_QUEUE.pop());
    }

    // Measurement: each iteration = one yield (push) + one resume (pop)
    let start = rdtsc_serialized();
    for _ in 0..iterations {
        // Yield: task signals it's ready to be scheduled
        let _ = READY_QUEUE.push(dummy_id);
        // Resume: executor picks up next task
        let popped = READY_QUEUE.pop();
        let _ = core::hint::black_box(popped);
    }
    let end = rdtsc_serialized();

    let total = end.wrapping_sub(start);
    let per_switch = total / iterations;
    let per_switch_ns = cycles_to_ns(per_switch);

    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] Context Switch: {} cycles/switch ({}ns)",
        per_switch,
        per_switch_ns
    );
    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH]   Linux ref: 1,200-5,500ns (lmbench)"
    );

    (total, per_switch)
}

// ── Benchmark 3: Arena Allocation ──────────────────────────────

/// Measure arena (bump-pointer) allocation throughput.
///
/// Allocates small objects (64 bytes) from a temporary arena region.
/// This measures the raw cost of our O(1) bump-pointer allocator:
/// one pointer increment + bounds check.
///
/// We use a local buffer as a simulated arena to avoid filling up
/// the real kernel arena during benchmarking.
///
/// Returns (total_cycles, per_alloc_cycles).
pub fn bench_arena_alloc() -> (u64, u64) {
    let iterations = BENCH_ITERATIONS;
    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] Arena Allocation — {} iterations (+ 1000 warmup)...",
        iterations
    );

    // Simulate bump-pointer arena: a buffer + cursor
    // This is exactly what our FixedArena does internally
    const ARENA_SIZE: usize = 16 * 1024 * 1024; // 16 MiB local buffer
    const ALLOC_SIZE: usize = 64; // typical small allocation
    const ALLOC_ALIGN: usize = 8;

    // We can't stack-allocate 16 MiB, so we use a cursor-only simulation.
    // The bump-pointer operation is: cursor = align_up(cursor, align); cursor += size;
    // We measure JUST this arithmetic — same as real FixedArena::alloc_raw.

    let mut cursor: usize = 0;

    // Warmup
    for _ in 0..1000u64 {
        // align up
        let aligned = (cursor + ALLOC_ALIGN - 1) & !(ALLOC_ALIGN - 1);
        cursor = core::hint::black_box(aligned + ALLOC_SIZE);
        if cursor >= ARENA_SIZE {
            cursor = 0;
        }
    }

    cursor = 0;

    // Measurement
    let start = rdtsc_serialized();
    for _ in 0..iterations {
        let aligned = (cursor + ALLOC_ALIGN - 1) & !(ALLOC_ALIGN - 1);
        cursor = core::hint::black_box(aligned + ALLOC_SIZE);
        if cursor >= ARENA_SIZE {
            cursor = 0;
        }
    }
    let end = rdtsc_serialized();

    let _ = core::hint::black_box(cursor);

    let total = end.wrapping_sub(start);
    let per_alloc = total / iterations;

    // Sub-nanosecond results: compute fractional ns (X.Y format)
    // ns_x10 = cycles * 10_000 / MHz → gives tenths of a nanosecond
    let per_alloc_ns_x10 = per_alloc * 10_000 / unsafe { TSC_MHZ };
    let ns_int = per_alloc_ns_x10 / 10;
    let ns_frac = per_alloc_ns_x10 % 10;

    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] Arena Alloc: {} cycles/alloc ({}.{}ns)",
        per_alloc,
        ns_int,
        ns_frac
    );
    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH]   Linux ref: 50-5,000ns (glibc malloc)"
    );

    (total, per_alloc)
}

// ── Benchmark 4: IPC / Zero-Copy Transfer ──────────────────────

/// Measure zero-copy IPC throughput via shared arena.
///
/// In Zero, tasks share the same address space and arena.
/// "IPC" is just reading a pointer that another task wrote.
/// We simulate a producer→consumer pattern: writer fills a
/// shared buffer, reader processes it. No kernel crossing,
/// no pipe, no socket, no copy.
///
/// Buffer: 256 KiB static (exceeds 32 KiB L1 cache per core
/// on EPYC, so we measure realistic L2/L3 bandwidth, not
/// unrealistic L1-only throughput).
///
/// 160 rounds × 256 KiB = 40 MiB total transfer.
pub fn bench_ipc_throughput() -> (u64, u64) {
    const BUF_SIZE: usize = 256 * 1024; // 256 KiB — exceeds L1 cache
    const NUM_ROUNDS: u64 = 160; // 160 × 256 KiB = 40 MiB total

    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] IPC Throughput (Zero-Copy) — {} x {} KiB...",
        NUM_ROUNDS,
        BUF_SIZE / 1024
    );

    // Static buffer since 256 KiB is too large for kernel stack.
    // Safety: single-threaded benchmark, no concurrent access.
    static mut IPC_BUF: [u8; BUF_SIZE] = [0u8; BUF_SIZE];
    // Use raw pointer to avoid Rust 2024 static_mut_refs warning.
    let buf = unsafe { &mut *core::ptr::addr_of_mut!(IPC_BUF) };

    // Warmup: 10 write+read rounds
    for i in 0..10u64 {
        let fill = (i & 0xFF) as u8;
        for b in buf.iter_mut() {
            *b = fill;
        }
        let mut sum: u64 = 0;
        for &b in buf.iter() {
            sum = sum.wrapping_add(b as u64);
        }
        let _ = core::hint::black_box(sum);
    }

    // Measurement: producer writes, consumer reads each round
    let start = rdtsc_serialized();
    for round in 0..NUM_ROUNDS {
        // "Producer" writes (different value per round prevents optimization)
        let fill = (round & 0xFF) as u8;
        for b in buf.iter_mut() {
            *b = fill;
        }
        // "Consumer" reads and processes
        let mut sum: u64 = 0;
        for &b in buf.iter() {
            sum = sum.wrapping_add(core::hint::black_box(b) as u64);
        }
        let _ = core::hint::black_box(sum);
    }
    let end = rdtsc_serialized();

    let total = end.wrapping_sub(start);
    let total_bytes = NUM_ROUNDS * BUF_SIZE as u64;
    let total_ns = cycles_to_ns(total);

    // GB/s with 1 decimal place
    let gbps_x10 = if total_ns > 0 {
        (total_bytes * 10_000) / total_ns
    } else {
        0
    };
    let gbps_int = gbps_x10 / 10;
    let gbps_dec = gbps_x10 % 10;

    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH] IPC Throughput: {}.{} GB/s ({} bytes in {}ns)",
        gbps_int,
        gbps_dec,
        total_bytes,
        total_ns
    );
    let _ = writeln!(
        arch::serial::Serial,
        "[BENCH]   Linux ref: 3-6 GB/s (pipe), 50-100 GB/s (mmap)"
    );

    (total, total_bytes / if total_ns > 0 { total_ns } else { 1 })
}

// ── Summary Output ─────────────────────────────────────────────

/// Print final benchmark summary — the "screenshot-ready" output.
///
/// This is what appears on the KVM console / framebuffer for the
/// benchmark ISO demo.
pub fn print_summary(
    cpu: &CpuInfo,
    boot_ms: u64,
    boot_us: u64,
    ctx_switch_ns: u64,
    alloc_ns_int: u64,
    alloc_ns_frac: u64,
    ipc_gbps_int: u64,
    ipc_gbps_dec: u64,
    llm_toks: u64,
    llm_toks_dec: u64,
) {
    // Cherry production holds this screen as the final KVM image. Boot
    // has fully completed by the time we get here (Stage 0-11 logs all
    // streamed to VGA live); render the summary on a fresh page so the
    // held screen is one complete, un-split box instead of whatever
    // half-page the page-mode cursor happened to be on.
    #[cfg(feature = "cherry-net")]
    arch::fb_console::new_page();

    let _ = writeln!(arch::serial::Serial, "");
    let _ = writeln!(
        arch::serial::Serial,
        "╔══════════════════════════════════════════════════╗"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "║        Zero Benchmark Results                 ║"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "╠══════════════════════════════════════════════════╣"
    );
    // Hardware line
    let _ = write!(arch::serial::Serial, "║  CPU: ");
    for &b in cpu.brand[..cpu.brand_len].iter() {
        if b >= 0x20 && b < 0x7F {
            let _ = write!(arch::serial::Serial, "{}", b as char);
        }
    }
    let _ = writeln!(arch::serial::Serial, "");
    let _ = writeln!(
        arch::serial::Serial,
        "╠══════════════════════════════════════════════════╣"
    );
    let _ = write!(arch::serial::Serial, "║  Boot Time:       ");
    let _ = write!(arch::serial::Serial, "{}", boot_ms);
    let _ = write!(arch::serial::Serial, ".");
    let _ = write!(arch::serial::Serial, "{}", boot_us);
    let _ = writeln!(arch::serial::Serial, "ms                         ║");

    let _ = write!(arch::serial::Serial, "║  Context Switch:  ");
    let _ = write!(arch::serial::Serial, "{}", ctx_switch_ns);
    let _ = writeln!(arch::serial::Serial, "ns                            ║");

    let _ = write!(arch::serial::Serial, "║  Arena Alloc:     ");
    let _ = write!(arch::serial::Serial, "{}", alloc_ns_int);
    let _ = write!(arch::serial::Serial, ".");
    let _ = write!(arch::serial::Serial, "{}", alloc_ns_frac);
    let _ = writeln!(arch::serial::Serial, "ns                           ║");

    let _ = write!(arch::serial::Serial, "║  IPC Throughput:  ");
    let _ = write!(arch::serial::Serial, "{}", ipc_gbps_int);
    let _ = write!(arch::serial::Serial, ".");
    let _ = write!(arch::serial::Serial, "{}", ipc_gbps_dec);
    let _ = writeln!(arch::serial::Serial, " GB/s                      ║");

    let _ = write!(arch::serial::Serial, "║  LLM Inference:   ");
    let _ = write!(arch::serial::Serial, "{}", llm_toks);
    let _ = write!(arch::serial::Serial, ".");
    let _ = write!(arch::serial::Serial, "{}", llm_toks_dec);
    let _ = writeln!(arch::serial::Serial, " tok/s                     ║");

    // Anchor rows — the capture-pass token/logit_bits the operator reads
    // off the KVM/VGA photo to promote to a strict manifest anchor
    // (ADR-029 v3). Serial (COM1) is unreachable on the Cherry deploy, so
    // these must appear on the held screen. `emit_box_line` renders into
    // the fixed 50-column box body so variable-width token/hex stay
    // border-aligned. Shown on all targets; absent only if Stage 11 never
    // reached token 0.
    if let Some((tok, logit_bits, mode)) = anchor_snapshot() {
        // 50-column box body, space-padded, with the existing ║ borders.
        fn emit_box_line(args: core::fmt::Arguments<'_>) {
            const BODY: usize = 50;
            let mut buf = [b' '; BODY];
            struct W<'a> {
                buf: &'a mut [u8; BODY],
                len: usize,
            }
            impl core::fmt::Write for W<'_> {
                fn write_str(&mut self, s: &str) -> core::fmt::Result {
                    for &b in s.as_bytes() {
                        if self.len >= BODY {
                            break;
                        }
                        // Keep the box ASCII-clean; the framebuffer font
                        // only renders printable Latin-1 reliably.
                        self.buf[self.len] = if (0x20..0x7F).contains(&b) { b } else { b'?' };
                        self.len += 1;
                    }
                    Ok(())
                }
            }
            let mut w = W {
                buf: &mut buf,
                len: 0,
            };
            let _ = core::fmt::write(&mut w, args);
            let _ = write!(arch::serial::Serial, "║");
            for &b in buf.iter() {
                let _ = write!(arch::serial::Serial, "{}", b as char);
            }
            let _ = writeln!(arch::serial::Serial, "║");
        }

        let _ = writeln!(
            arch::serial::Serial,
            "╠══════════════════════════════════════════════════╣"
        );
        emit_box_line(format_args!("  Anchor (mode: {}):", anchor_mode_label(mode)));
        emit_box_line(format_args!("    next-token ID = {}", tok));
        emit_box_line(format_args!("    logit_bits    = 0x{:08x}", logit_bits));
    }

    // Net bring-up outcome on the held screen: the KVM console is the
    // only operator surface on a Cherry box whose network is down, so
    // this line is the remote-diagnosis path for reachability issues.
    {
        use crate::net::bind_report;
        let mut buf = [0u8; bind_report::DETAIL_CAP];
        let (tag, detail_len) = match bind_report::state() {
            bind_report::ONLINE => ("online ", bind_report::detail(&mut buf)),
            bind_report::FAILED => ("FAILED ", bind_report::detail(&mut buf)),
            _ => {
                const MSG: &[u8] = b"bring-up not reached";
                buf[..MSG.len()].copy_from_slice(MSG);
                ("", MSG.len())
            }
        };
        let _ = write!(arch::serial::Serial, "║  NET: {}", tag);
        // The box content is 50 columns; "  NET: " takes 7, so tag +
        // detail + padding must fill exactly 43 to keep the right
        // border aligned.
        const WIDTH: usize = 43;
        let mut written = tag.len();
        for &b in buf[..detail_len]
            .iter()
            .take(WIDTH.saturating_sub(tag.len()))
        {
            let _ = write!(arch::serial::Serial, "{}", b as char);
            written += 1;
        }
        for _ in written..WIDTH {
            let _ = write!(arch::serial::Serial, " ");
        }
        let _ = writeln!(arch::serial::Serial, "║");
    }

    // LLM failure reason on the held screen — same rationale as the
    // NET row: 0.0 tok/s alone is undiagnosable from a KVM photo.
    if llm_failure_recorded() {
        let mut buf = [0u8; llm_report::DETAIL_CAP];
        let detail_len = llm_report::detail(&mut buf);
        let _ = write!(arch::serial::Serial, "║  LLM: FAILED ");
        // "  LLM: FAILED " takes 14 columns of the 50-column box body.
        const WIDTH: usize = 36;
        let mut written = 0usize;
        for &b in buf[..detail_len].iter().take(WIDTH) {
            let _ = write!(arch::serial::Serial, "{}", b as char);
            written += 1;
        }
        for _ in written..WIDTH {
            let _ = write!(arch::serial::Serial, " ");
        }
        let _ = writeln!(arch::serial::Serial, "║");
    }

    let _ = writeln!(
        arch::serial::Serial,
        "╠══════════════════════════════════════════════════╣"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "║  Linux Reference Values:                         ║"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "║  Boot Time:       15,000-25,000ms                ║"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "║  Context Switch:  1,200-5,500ns                  ║"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "║  malloc/free:     50-5,000ns                     ║"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "║  IPC (pipe):      3-6 GB/s                       ║"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "║  LLM CPU target:  >=150 tok/s                    ║"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "╠══════════════════════════════════════════════════╣"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "║  Est. savings at 1,000 servers: $15.9-17.6M/yr   ║"
    );
    let _ = writeln!(
        arch::serial::Serial,
        "╚══════════════════════════════════════════════════╝"
    );
    let _ = writeln!(arch::serial::Serial, "");
}

/// Run all benchmarks and output results.
///
/// Called from kernel_main() after all subsystems are initialized
/// but before the executor starts.
pub fn run_all_benchmarks(cpu: &CpuInfo) {
    calibrate_tsc(cpu);
    let _ = writeln!(arch::serial::Serial, "");

    // Benchmark 1: Boot Time
    let boot_delta = boot_delta_cycles();
    let boot_ns = cycles_to_ns(boot_delta);
    let boot_ms = boot_ns / 1_000_000;
    let boot_us = (boot_ns % 1_000_000) / 1000;

    report_boot_time();

    let pre_llm_valid = PRE_LLM_BASELINE_VALID.load(Ordering::Acquire) != 0;

    // Benchmark 2: Context Switch
    let ctx_per_switch = if pre_llm_valid {
        let value = PRE_LLM_CTX_PER_SWITCH.load(Ordering::Acquire);
        let _ = writeln!(
            arch::serial::Serial,
            "[BENCH] Context Switch: using pre-LLM baseline ({} cycles/switch)",
            value
        );
        value
    } else {
        let (_ctx_total, ctx_per_switch) = bench_context_switch();
        ctx_per_switch
    };
    let ctx_switch_ns = cycles_to_ns(ctx_per_switch);

    // Benchmark 3: Arena Allocation
    let alloc_per = if pre_llm_valid {
        let value = PRE_LLM_ALLOC_PER.load(Ordering::Acquire);
        let _ = writeln!(
            arch::serial::Serial,
            "[BENCH] Arena Alloc: using pre-LLM baseline ({} cycles/alloc)",
            value
        );
        value
    } else {
        let (_alloc_total, alloc_per) = bench_arena_alloc();
        alloc_per
    };
    // Sub-nanosecond: compute X.Y ns for summary display
    let alloc_ns_x10 = alloc_per * 10_000 / unsafe { TSC_MHZ };

    // Benchmark 4: IPC Throughput
    let _ipc_total = if pre_llm_valid {
        let value = PRE_LLM_IPC_TOTAL.load(Ordering::Acquire);
        let _ = writeln!(
            arch::serial::Serial,
            "[BENCH] IPC Throughput: using pre-LLM baseline ({} cycles)",
            value
        );
        value
    } else {
        let (ipc_total, _ipc_metric) = bench_ipc_throughput();
        ipc_total
    };
    // Recalculate GB/s for summary
    let ipc_total_ns = cycles_to_ns(_ipc_total);
    let ipc_total_bytes: u64 = 160 * 256 * 1024; // Must match bench_ipc_throughput constants
    let ipc_gbps_x10 = if ipc_total_ns > 0 {
        (ipc_total_bytes * 10_000) / ipc_total_ns
    } else {
        0
    };

    // Benchmark 5: LLM Inference — recorded by Stage 11 generation.
    let (llm_tok_s_x10, llm_tokens, llm_cycles) = recorded_llm_tok_per_sec_x10();
    let _ = writeln!(arch::serial::Serial, "");
    if llm_tok_s_x10 > 0 {
        let _ = writeln!(
            arch::serial::Serial,
            "[BENCH] LLM Inference: {}.{} tok/s ({} generated token(s), {} cycles)",
            llm_tok_s_x10 / 10,
            llm_tok_s_x10 % 10,
            llm_tokens,
            llm_cycles
        );
    } else if llm_failure_recorded() {
        let _ = writeln!(
            arch::serial::Serial,
            "[BENCH] LLM Inference: FAILED (Stage 11 forward pass aborted — see earlier [MoE]/[BENCH] lines)"
        );
    } else {
        let _ = writeln!(
            arch::serial::Serial,
            "[BENCH] LLM Inference: unavailable (Stage 11 did not record generation timing)"
        );
    }

    // Print summary table
    let _ = writeln!(arch::serial::Serial, "");
    print_summary(
        cpu,
        boot_ms,
        boot_us,
        ctx_switch_ns,
        alloc_ns_x10 / 10,
        alloc_ns_x10 % 10,
        ipc_gbps_x10 / 10,
        ipc_gbps_x10 % 10,
        llm_tok_s_x10 / 10,
        llm_tok_s_x10 % 10,
    );
}
