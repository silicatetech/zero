// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cycle-counter helpers for performance measurement (Stage 10 MP6).
//!
//! Provides serialized rdtscp-based cycle counting for AOT-vs-Interpreter
//! benchmarking in the boot path. rdtscp is preferred over rdtsc because
//! it serializes against prior instruction completion, eliminating
//! out-of-order execution noise in the cycle measurement.
//!
//! # Caveats
//!
//! - **QEMU emulation:** When running under QEMU, rdtscp values are
//!   synthetic — derived from QEMU's virtual time, not the host CPU's
//!   actual TSC. Absolute cycle counts are not hardware-accurate.
//!   The *ratio* between AOT and Interpreter cycle counts remains
//!   meaningful, however, since both paths are emulated identically.
//!
//! - **No frequency scaling concern:** Stage 10 is single-core in QEMU.
//!   Multi-core TSC drift is a Phase 12+ concern (Pillar 7 platform
//!   portability).
//!
//! - **Measurement overhead:** rdtscp itself costs ~20-30 cycles. For
//!   meaningful measurement of a 40-byte AOT program, multi-iteration
//!   loops (e.g., 10,000 iterations summed) amortize this overhead.
//!
//! Per ADR-026 MP6 mandate ("measurable, not aspirational"), this
//! module establishes the baseline measurement infrastructure for
//! V3 Pillar 1 (Maximum Performance, foundational).
//!
//! # Implementation note
//!
//! We use inline `asm!` rather than `core::arch::x86_64::__rdtscp` /
//! `__cpuid` intrinsics because those intrinsics require target-feature
//! flags (e.g., `+sse`) that are not enabled on `x86_64-unknown-none`.
//! The instructions themselves are always available on x86_64 CPUs
//! (QEMU's default `-cpu qemu64` supports both cpuid and rdtsc).
//! We use `rdtsc` (not `rdtscp`) paired with `cpuid` serialization
//! to avoid requiring the rdtscp CPU feature flag.

/// Read the time-stamp counter with serialization barrier.
///
/// Issues a `cpuid` instruction before `rdtsc` to prevent prior
/// instructions from being reordered past the measurement point.
/// Returns the 64-bit TSC value at the moment all prior instructions
/// have retired.
///
/// # Safety
///
/// Uses inline x86_64 assembly (`cpuid`, `rdtsc`). Both are safe to
/// execute on x86_64 in any privilege level (Ring 0 or otherwise).
/// The `unsafe` is required only because inline assembly is inherently
/// unsafe in Rust; there is no memory-safety or undefined-behavior risk.
#[inline]
pub fn rdtsc_serialized() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        // cpuid acts as a full execution barrier — flushes the pipeline
        // and ensures prior instructions complete before rdtsc samples
        // the TSC. We use eax=0 (basic CPUID info) as a cheap barrier.
        core::arch::asm!(
            "push rbx",       // save rbx (reserved by LLVM)
            "xor eax, eax",   // eax = 0 (CPUID leaf 0)
            "cpuid",          // serialization barrier (clobbers eax/ebx/ecx/edx)
            "rdtsc",          // EDX:EAX = timestamp counter
            "pop rbx",        // restore rbx
            out("eax") lo,
            out("edx") hi,
            out("ecx") _,     // cpuid clobbers ecx
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// Number of iterations for boot-path benchmarks.
///
/// 1,000 iterations chosen to:
/// - Amortize rdtsc overhead (~20-30 cycles per call) across enough
///   work that signal exceeds noise.
/// - Stay within the 2 MiB runtime arena budget for the interpreter
///   path: 1,000 iterations × ~456 bytes/call ≈ 456 KiB.
/// - Provide stable cycle ratios across multiple boot runs.
/// - Symmetric across both paths for valid statistical comparison.
pub const BENCH_ITERATIONS: u64 = 1_000;

/// Fallback TSC frequency assumption when CPUID enumeration is
/// unavailable. Chosen as a middle-of-the-road modern x86 base clock
/// (~2.5 GHz) — wrong by ~25% on a 3.25 GHz EPYC 9354P but never
/// catastrophically wrong, and never zero (we never divide by zero).
const FALLBACK_TSC_HZ: u64 = 2_500_000_000;

/// Raw CPUID call.
///
/// Returns (eax, ebx, ecx, edx). Preserves rbx (reserved by LLVM)
/// across the inline asm window.
#[inline]
fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    let mut a: u32 = leaf;
    let b: u32;
    let mut c: u32 = subleaf;
    let d: u32;
    unsafe {
        core::arch::asm!(
            "push rbx",
            "cpuid",
            "mov {b:e}, ebx",
            "pop rbx",
            b = out(reg) b,
            inout("eax") a,
            inout("ecx") c,
            out("edx") d,
        );
    }
    (a, b, c, d)
}

/// Detect the TSC frequency in Hz.
///
/// Tries three sources, in order of accuracy:
///
///  1. **CPUID leaf 0x15** — "Time Stamp Counter / Core Crystal Clock
///     Information". When EBX != 0 and ECX != 0, TSC_Hz = ECX * EBX/EAX.
///     This is the authoritative answer when the CPU provides it.
///     (Intel always; AMD Zen3+ inconsistently. AMD EPYC tends to
///     leave ECX=0 — we fall through.)
///
///  2. **CPUID leaf 0x16** — "Processor Frequency Information". EAX
///     holds the base frequency in MHz. Available on Intel Skylake+
///     and (importantly) on AMD Zen2+, including EPYC 9004 series
///     (e.g. EPYC 9354P at 3.25 GHz base — leaf 0x16 reports 3250).
///     The TSC on modern AMD increments at the P0 (base) frequency,
///     so this is a good proxy on EPYC.
///
///  3. **Fallback** — 2.5 GHz. Used only when neither leaf reports
///     a usable value. Will be wrong by up to ~25% but is never zero.
///
/// Returns the frequency in Hz. Safe to call at boot, before
/// interrupts are enabled. Result is suitable for cycles→milliseconds
/// conversions, not for sub-microsecond timekeeping.
///
/// CITE: Intel SDM Vol 2A §3.3 CPUID — leaf 0x15 (Crystal Clock),
///       leaf 0x16 (Processor Frequency Information).
/// CITE: AMD APM Vol 3 — CPUID Fn0000_0016h (Zen2+ base/max freq).
pub fn tsc_hz() -> u64 {
    let max_leaf = cpuid(0, 0).0;

    if max_leaf >= 0x15 {
        let (eax, ebx, ecx, _edx) = cpuid(0x15, 0);
        if eax != 0 && ebx != 0 && ecx != 0 {
            // TSC = ECX * (EBX / EAX), all u32, compute in u64 to avoid overflow.
            let crystal = ecx as u64;
            let num = ebx as u64;
            let den = eax as u64;
            let hz = crystal.saturating_mul(num) / den;
            if hz != 0 {
                return hz;
            }
        }
    }

    if max_leaf >= 0x16 {
        let (eax, _ebx, _ecx, _edx) = cpuid(0x16, 0);
        let base_mhz = eax & 0xFFFF;
        if base_mhz != 0 {
            return (base_mhz as u64) * 1_000_000;
        }
    }

    FALLBACK_TSC_HZ
}
