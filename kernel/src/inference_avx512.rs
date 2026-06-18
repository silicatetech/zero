// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR-029 Phase 1+2+3 — Kernel-level AVX-512 accelerated inference dispatch.
//!
//! Mirror of `inference_neon.rs` for x86_64. Pure kernel-level wrapper
//! around sacred-crate operators. Sacred crates UNCHANGED.
//!
//! Per Pillar 7: NO AVX-512 intrinsics in this file. This module calls
//! into `arch::x86_64::math::linear` which contains the intrinsics.
//! This file uses only `cfg(feature)`, NOT `cfg(target_arch)` — gating
//! to x86_64 happens at the module-include site in `main.rs`.
//!
//! Sacred scalar ops used: rmsnorm, rope, softmax, SiLU, embed_lookup,
//! KvCache::store_kv, KvCache::get_k_slice, KvCache::get_v_slice.
//!
//! # Parallel matmul integration (Phase 3)
//!
//! When the SMP layer has more than one active core, the linear
//! projection calls below dispatch row-parallel through
//! [`crate::smp::ParallelMatmulContext`]. On a single-core boot
//! (no APs woken) this transparently degrades to a direct call
//! with no barrier overhead — see `ParallelMatmulContext::dispatch_matmul`.

use core::fmt::Write;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};

use zero_llm_inference::attention::{softmax, AttentionError};
use zero_llm_inference::forward_pass::{
    embed_lookup, embed_lookup_dispatch, ForwardPassError, N_LAYERS,
};
use zero_llm_inference::lm_head::{LmHeadError, VOCAB_SIZE_PADDED, VOCAB_SIZE_REAL};
use zero_llm_inference::ops::{rmsnorm, LinearScratch};
use zero_llm_inference::{
    FfnDownQuant, ForwardPassDispatch, KvCache, LayerWeights, RopeContext,
};

use crate::arch::x86_64::math::activation::silu_mul_avx512;
use crate::arch::x86_64::math::linear::{
    argmax_f32_avx512, dot_product_f32_avx512, linear_q4_0_avx512_range,
    linear_q4_0x4_avx512_range, linear_q4k_avx512, linear_q4k_avx512_range, linear_q6k_avx512,
    linear_q6k_avx512_range, linear_q8_0_avx512_range, linear_q8_0x4_avx512_range,
    weighted_add_f32_avx512,
};
use crate::arch::x86_64::math::trig::{rope_apply_avx512, rope_sincos_lut_avx512};
use crate::smp::{MatmulArgs, ParallelMatmulContext, RowRange};

static Q4K_DISPATCH_LOGGED: AtomicBool = AtomicBool::new(false);
static Q6K_DISPATCH_LOGGED: AtomicBool = AtomicBool::new(false);
static LM_HEAD_DISPATCH_LOGGED: AtomicBool = AtomicBool::new(false);

// ─────────────────────────────────────────────────────────────────
// Fused multi-projection dispatch toggle (qwen-perf-v2)
// ─────────────────────────────────────────────────────────────────
//
// The qwen-perf-v2 fused multi-projection dispatch (Q‖K‖V and
// gate‖up) is the leading suspect for the hardware-only
// `Attention(NumericalInstability)`: it is the one dispatch the MP2.6
// SMP self-test does NOT exercise (the self-test only validates the
// single-segment `linear_q4_0_dispatch` parallel-vs-single), and the
// validated-correct scalar/NEON paths are single-threaded so they
// never touch it either.
//
// DIAGNOSTIC DEFAULT = ON (= the deploy-4 behaviour that fails). With
// the per-step tracing below, the next boot REPRODUCES the failure and
// prints exactly which step turns finite math into NaN/Inf — proving
// or refuting the fused-dispatch hypothesis instead of guessing. Once
// the trace pins the culprit, flip this OFF (separate-dispatch path,
// numerically identical to the single-threaded paths) to fix, or apply
// a targeted fix at the identified step. See
// ANALYSIS-llm-ice-rootcause-2026-06-13.md.
static AVX512_FUSED_DISPATCH_ENABLED: AtomicBool = AtomicBool::new(true);

/// Whether the fused multi-projection dispatch is engaged. Diagnostic
/// default ON (reproduce + trace); flip OFF to route Q/K/V and gate/up
/// through the self-test-validated separate-dispatch path.
#[inline(always)]
pub fn fused_dispatch_enabled() -> bool {
    AVX512_FUSED_DISPATCH_ENABLED.load(Ordering::Acquire)
}

/// Enable/disable the fused multi-projection dispatch at runtime.
/// Returns the stored value.
#[inline]
pub fn set_fused_dispatch_enabled(on: bool) -> bool {
    AVX512_FUSED_DISPATCH_ENABLED.store(on, Ordering::Release);
    on
}

/// One-shot non-finite diagnostic latch. The forward pass scans key
/// intermediate buffers and, the FIRST time any holds a NaN/Inf, logs
/// the layer + stage + first offending (index, value) and latches so
/// later layers stay silent. Zero log spam, O(dim) scan cost only
/// until the first trip — invaluable for pinning hardware-only
/// numerical failures to an exact stage without a reboot-per-guess
/// cycle.
static FINITE_DIAG_TRIPPED: AtomicBool = AtomicBool::new(false);

/// Scan `buf` for the first non-finite element; on the first trip
/// across the whole run, log `layer`/`stage` and latch. Cheap and
/// one-shot. Returns true if a non-finite value was found this call.
#[inline]
fn finite_diag(buf: &[f32], layer: usize, stage: &str) -> bool {
    if FINITE_DIAG_TRIPPED.load(Ordering::Acquire) {
        return false;
    }
    let mut idx = 0usize;
    while idx < buf.len() {
        let v = buf[idx];
        if !v.is_finite() {
            record_first_nonfinite(
                layer,
                stage,
                if v.is_nan() { 1 } else { 0 },
                if v.is_infinite() { 1 } else { 0 },
            );
            if !FINITE_DIAG_TRIPPED.swap(true, Ordering::AcqRel) {
                let _ = writeln!(
                    crate::arch::serial::Serial,
                    "[MP3.0][FINITE-DIAG] first non-finite: layer={} stage={} idx={} bits=0x{:08x} (fused_dispatch={})",
                    layer,
                    stage,
                    idx,
                    v.to_bits(),
                    fused_dispatch_enabled(),
                );
            }
            return true;
        }
        idx += 1;
    }
    false
}

// ─────────────────────────────────────────────────────────────────
// Per-step value-range tracing (bare-metal debug, 2026-06-13)
// ─────────────────────────────────────────────────────────────────
//
// On bare metal there is no debugger; the VGA/serial stream is the
// only window. To localise the hardware-only `Attention(NumericalInstability)`
// we dump min / max / mean / #NaN / #Inf of every key intermediate of
// the FIRST token through the AVX-512 forward. The next boot then
// shows on the KVM screen the EXACT step + layer where finite math
// first turns into NaN/Inf — breaking the reboot-per-guess loop.
//
// Gated to the first token (token_offset == 0) so the steady-state
// decode loop pays nothing, and behind a runtime flag (default ON for
// this diagnostic build; control plane can disable). Logging only —
// never mutates a value, so the β-anchor and the scalar/NEON builds
// are unaffected.
static AVX512_TRACE_ENABLED: AtomicBool = AtomicBool::new(true);

/// Whether per-step value-range tracing is active (first token only).
#[inline(always)]
pub fn avx512_trace_enabled() -> bool {
    AVX512_TRACE_ENABLED.load(Ordering::Acquire)
}

/// Enable/disable per-step tracing at runtime. Returns the stored value.
#[inline]
pub fn set_avx512_trace_enabled(on: bool) -> bool {
    AVX512_TRACE_ENABLED.store(on, Ordering::Release);
    on
}

/// (min, max, mean, nan_count, inf_count) over `buf`, ignoring
/// non-finite lanes for min/max/mean so one NaN doesn't hide the
/// finite range.
#[inline]
fn vec_stats(buf: &[f32]) -> (f32, f32, f32, usize, usize) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut finite = 0usize;
    let mut nan = 0usize;
    let mut inf = 0usize;
    let mut i = 0usize;
    while i < buf.len() {
        let v = buf[i];
        if v.is_nan() {
            nan += 1;
        } else if v.is_infinite() {
            inf += 1;
        } else {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
            sum += v as f64;
            finite += 1;
        }
        i += 1;
    }
    let mean = if finite > 0 {
        (sum / finite as f64) as f32
    } else {
        0.0
    };
    if finite == 0 {
        mn = 0.0;
        mx = 0.0;
    }
    (mn, mx, mean, nan, inf)
}

/// Dump min/max/mean/#NaN/#Inf of `buf` to serial (mirrors to VGA),
/// gated to the first token and the runtime flag. Cheap O(n) scan.
/// Also latches the FIRST stage that holds a non-finite value so the
/// held bench screen can name it (see [`first_nonfinite_detail`]).
#[inline]
fn trace_stats(layer: usize, token_offset: usize, label: &str, buf: &[f32]) {
    if token_offset != 0 || !avx512_trace_enabled() {
        return;
    }
    let (mn, mx, mean, nan, inf) = vec_stats(buf);
    if nan > 0 || inf > 0 {
        record_first_nonfinite(layer, label, nan, inf);
    }
    let _ = writeln!(
        crate::arch::serial::Serial,
        "[TRACE] L{:02} {:<14} n={:<5} min={:+.3e} max={:+.3e} mean={:+.3e} nan={} inf={}",
        layer, label, buf.len(), mn, mx, mean, nan, inf,
    );
}

// First-non-finite latch, mirrored onto the held bench screen so the
// single KVM photo names the exact step (e.g. "L00 q_proj n2048")
// even though the per-step [`trace_stats`] lines scroll behind the
// benchmark page. Written once by whichever of `trace_stats` /
// `finite_diag` sees the first NaN/Inf in forward execution order.
const FIRST_NF_CAP: usize = 32;
static FIRST_NF_LATCHED: AtomicBool = AtomicBool::new(false);
static FIRST_NF_BUF: [AtomicU8; FIRST_NF_CAP] = [const { AtomicU8::new(0) }; FIRST_NF_CAP];
static FIRST_NF_LEN: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn record_first_nonfinite(layer: usize, stage: &str, nan: usize, inf: usize) {
    if FIRST_NF_LATCHED.swap(true, Ordering::AcqRel) {
        return;
    }
    struct W {
        len: usize,
    }
    impl Write for W {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            for &b in s.as_bytes() {
                if self.len >= FIRST_NF_CAP {
                    break;
                }
                FIRST_NF_BUF[self.len].store(b, Ordering::Relaxed);
                self.len += 1;
            }
            Ok(())
        }
    }
    let mut w = W { len: 0 };
    let _ = write!(w, "L{:02} {} n{} i{}", layer, stage, nan, inf);
    FIRST_NF_LEN.store(w.len, Ordering::Release);
}

/// Copy the first-non-finite localisation (e.g. `L00 q_proj n2048`)
/// into `buf`; returns the byte count (0 if none recorded). Lets the
/// bench failure reason name the exact step on the KVM screen.
pub fn first_nonfinite_detail(buf: &mut [u8; FIRST_NF_CAP]) -> usize {
    let n = FIRST_NF_LEN.load(Ordering::Acquire).min(FIRST_NF_CAP);
    let mut i = 0;
    while i < n {
        buf[i] = FIRST_NF_BUF[i].load(Ordering::Relaxed);
        i += 1;
    }
    n
}

static Q4K_CALLS: AtomicU64 = AtomicU64::new(0);
static Q4K_CYCLES: AtomicU64 = AtomicU64::new(0);
static Q6K_CALLS: AtomicU64 = AtomicU64::new(0);
static Q6K_CYCLES: AtomicU64 = AtomicU64::new(0);
static LM_HEAD_CALLS: AtomicU64 = AtomicU64::new(0);
static LM_HEAD_CYCLES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct Avx512PerfCounters {
    pub q4k_calls: u64,
    pub q4k_cycles: u64,
    pub q6k_calls: u64,
    pub q6k_cycles: u64,
    pub lm_head_calls: u64,
    pub lm_head_cycles: u64,
    pub fused_calls: u64,
    pub fused_cycles: u64,
    pub attn_par_calls: u64,
    pub attn_par_cycles: u64,
}

impl Avx512PerfCounters {
    #[inline(always)]
    pub fn delta_since(self, before: Self) -> Self {
        Self {
            q4k_calls: self.q4k_calls.saturating_sub(before.q4k_calls),
            q4k_cycles: self.q4k_cycles.saturating_sub(before.q4k_cycles),
            q6k_calls: self.q6k_calls.saturating_sub(before.q6k_calls),
            q6k_cycles: self.q6k_cycles.saturating_sub(before.q6k_cycles),
            lm_head_calls: self.lm_head_calls.saturating_sub(before.lm_head_calls),
            lm_head_cycles: self.lm_head_cycles.saturating_sub(before.lm_head_cycles),
            fused_calls: self.fused_calls.saturating_sub(before.fused_calls),
            fused_cycles: self.fused_cycles.saturating_sub(before.fused_cycles),
            attn_par_calls: self.attn_par_calls.saturating_sub(before.attn_par_calls),
            attn_par_cycles: self.attn_par_cycles.saturating_sub(before.attn_par_cycles),
        }
    }
}

#[inline]
pub fn perf_counters_snapshot() -> Avx512PerfCounters {
    Avx512PerfCounters {
        q4k_calls: Q4K_CALLS.load(Ordering::Relaxed),
        q4k_cycles: Q4K_CYCLES.load(Ordering::Relaxed),
        q6k_calls: Q6K_CALLS.load(Ordering::Relaxed),
        q6k_cycles: Q6K_CYCLES.load(Ordering::Relaxed),
        lm_head_calls: LM_HEAD_CALLS.load(Ordering::Relaxed),
        lm_head_cycles: LM_HEAD_CYCLES.load(Ordering::Relaxed),
        fused_calls: FUSED_CALLS.load(Ordering::Relaxed),
        fused_cycles: FUSED_CYCLES.load(Ordering::Relaxed),
        attn_par_calls: ATTN_PAR_CALLS.load(Ordering::Relaxed),
        attn_par_cycles: ATTN_PAR_CYCLES.load(Ordering::Relaxed),
    }
}

#[inline]
pub fn reset_perf_counters() {
    Q4K_CALLS.store(0, Ordering::Relaxed);
    Q4K_CYCLES.store(0, Ordering::Relaxed);
    Q6K_CALLS.store(0, Ordering::Relaxed);
    Q6K_CYCLES.store(0, Ordering::Relaxed);
    LM_HEAD_CALLS.store(0, Ordering::Relaxed);
    LM_HEAD_CYCLES.store(0, Ordering::Relaxed);
    FUSED_CALLS.store(0, Ordering::Relaxed);
    FUSED_CYCLES.store(0, Ordering::Relaxed);
    ATTN_PAR_CALLS.store(0, Ordering::Relaxed);
    ATTN_PAR_CYCLES.store(0, Ordering::Relaxed);
}

// ─────────────────────────────────────────────────────────────────
// Parallel matmul kernels (called by the SMP dispatcher via fn ptr)
// ─────────────────────────────────────────────────────────────────
//
// These wrappers reconstruct the slice arguments from `MatmulArgs`
// raw pointers and forward to the AVX-512 range variants. The
// pointer-to-slice conversion is sound because `MatmulArgs` was
// constructed from valid slices in the caller (`linear_q4k_dispatch`)
// and remains valid for the duration of the barrier wait.

/// Type-erased Q4_K kernel for the SMP dispatcher.
unsafe fn q4k_kernel(args: &MatmulArgs, range: RowRange) {
    let x = core::slice::from_raw_parts(args.x_ptr, args.in_dim);
    // Bytes-per-row = (in_dim / 256) * 144.
    let bytes_per_row = (args.in_dim / 256) * 144;
    // We only need to validate that the slice covers up to range.end's
    // last byte. Using args.out_dim * bytes_per_row gives the safe
    // upper bound — w_ptr is valid for the full tensor.
    let w = core::slice::from_raw_parts(args.w_ptr, args.out_dim * bytes_per_row);
    let out = core::slice::from_raw_parts_mut(args.out_ptr, args.out_dim);
    linear_q4k_avx512_range(x, w, out, args.in_dim, args.out_dim, range);
}

/// Type-erased Q6_K kernel for the SMP dispatcher.
unsafe fn q6k_kernel(args: &MatmulArgs, range: RowRange) {
    let x = core::slice::from_raw_parts(args.x_ptr, args.in_dim);
    let bytes_per_row = (args.in_dim / 256) * 210;
    let w = core::slice::from_raw_parts(args.w_ptr, args.out_dim * bytes_per_row);
    let out = core::slice::from_raw_parts_mut(args.out_ptr, args.out_dim);
    linear_q6k_avx512_range(x, w, out, args.in_dim, args.out_dim, range);
}

/// Type-erased Q4_0 kernel for the SMP dispatcher.
///
/// Q4_0 block: 32 elements / 18 bytes (fp16 d + 16 packed nibbles).
/// Each output row has `in_dim / 32` blocks of 18 bytes.
unsafe fn q4_0_kernel(args: &MatmulArgs, range: RowRange) {
    let x = core::slice::from_raw_parts(args.x_ptr, args.in_dim);
    let bytes_per_row = (args.in_dim / 32) * 18;
    let w = core::slice::from_raw_parts(args.w_ptr, args.out_dim * bytes_per_row);
    let out = core::slice::from_raw_parts_mut(args.out_ptr, args.out_dim);
    linear_q4_0_avx512_range(x, w, out, args.in_dim, args.out_dim, range);
}

/// Type-erased Q8_0 kernel for the SMP dispatcher.
///
/// Q8_0 block: 32 elements / 34 bytes (fp16 d + 32 i8 values).
/// Each output row has `in_dim / 32` blocks of 34 bytes.
unsafe fn q8_0_kernel(args: &MatmulArgs, range: RowRange) {
    let x = core::slice::from_raw_parts(args.x_ptr, args.in_dim);
    let bytes_per_row = (args.in_dim / 32) * 34;
    let w = core::slice::from_raw_parts(args.w_ptr, args.out_dim * bytes_per_row);
    let out = core::slice::from_raw_parts_mut(args.out_ptr, args.out_dim);
    linear_q8_0_avx512_range(x, w, out, args.in_dim, args.out_dim, range);
}

/// Type-erased Q4_0X4 (`.smodel`-v2 4-row-interleaved) kernel. Same
/// total bytes per tensor as Q4_0 — only the storage order differs.
unsafe fn q4_0x4_kernel(args: &MatmulArgs, range: RowRange) {
    let x = core::slice::from_raw_parts(args.x_ptr, args.in_dim);
    let bytes_per_row = (args.in_dim / 32) * 18;
    let w = core::slice::from_raw_parts(args.w_ptr, args.out_dim * bytes_per_row);
    let out = core::slice::from_raw_parts_mut(args.out_ptr, args.out_dim);
    linear_q4_0x4_avx512_range(x, w, out, args.in_dim, args.out_dim, range);
}

/// Type-erased Q8_0X4 (`.smodel`-v2 4-row-interleaved) kernel.
unsafe fn q8_0x4_kernel(args: &MatmulArgs, range: RowRange) {
    let x = core::slice::from_raw_parts(args.x_ptr, args.in_dim);
    let bytes_per_row = (args.in_dim / 32) * 34;
    let w = core::slice::from_raw_parts(args.w_ptr, args.out_dim * bytes_per_row);
    let out = core::slice::from_raw_parts_mut(args.out_ptr, args.out_dim);
    linear_q8_0x4_avx512_range(x, w, out, args.in_dim, args.out_dim, range);
}

/// Type-erased Q4_K VNNI kernel for the SMP dispatcher.
///
/// `args.x_ptr` for this kernel points to a Q8_K-packed activation
/// buffer of size `(args.in_dim / 256) * Q8K_BLOCK_BYTES` (292 bytes
/// per superblock), produced by `quantize_row_q8k`. The pointer is
/// reinterpreted from *const f32 → *const u8 (both x86_64 layout
/// addresses; the field is named for the FP32 path's convenience).
///
/// ADR-029 v8 candidate: VNNI feature gate is checked at the
/// dispatch site, not here. This kernel is unconditionally
/// compiled when `vnni-acceleration` is enabled.
#[cfg(feature = "vnni-acceleration")]
unsafe fn q4k_vnni_kernel(args: &MatmulArgs, range: RowRange) {
    use crate::arch::x86_64::math::vnni::{linear_q4k_vnni_range, Q8K_BLOCK_BYTES};
    let n_blocks = args.in_dim / 256;
    let q8k_packed =
        core::slice::from_raw_parts(args.x_ptr as *const u8, n_blocks * Q8K_BLOCK_BYTES);
    let bytes_per_row = n_blocks * 144;
    let w = core::slice::from_raw_parts(args.w_ptr, args.out_dim * bytes_per_row);
    let out = core::slice::from_raw_parts_mut(args.out_ptr, args.out_dim);
    linear_q4k_vnni_range(q8k_packed, w, out, args.in_dim, args.out_dim, range);
}

/// Dispatch a Q4_K linear projection across all active cores (or
/// in-line if `active_cores == 1`). Bit-exact equivalent to
/// `linear_q4k_avx512(x, w, out, in_dim, out_dim)`.
///
/// # Safety
/// AVX-512F must be available on every participating core. The caller
/// (this kernel boots only on AMD EPYC 9354P or similar Zen 4+
/// silicon, all of which support AVX-512F) is responsible for the
/// CPUID check at boot — see `main.rs`.
#[inline]
unsafe fn linear_q4k_dispatch(x: &[f32], w: &[u8], out: &mut [f32], in_dim: usize, out_dim: usize) {
    let args = MatmulArgs {
        x_ptr: x.as_ptr(),
        w_ptr: w.as_ptr(),
        out_ptr: out.as_mut_ptr(),
        in_dim,
        out_dim,
    };
    let ctx = ParallelMatmulContext::for_active_cores_for_rows(out_dim);
    // ADR-029 v8.4: load-first, swap-once. After the first dispatch
    // this is a single Acquire load on a hot cacheline-resident bool
    // instead of an unconditional LOCK CMPXCHG every call.
    if !Q4K_DISPATCH_LOGGED.load(Ordering::Acquire)
        && !Q4K_DISPATCH_LOGGED.swap(true, Ordering::AcqRel)
    {
        let _ = writeln!(
            crate::arch::serial::Serial,
            "[MP3.0] AVX-512 Q4_K matmul dispatch: active_cores={} effective_cores={} registered_cores={} out_dim={} in_dim={}",
            crate::smp::active_cores(),
            ctx.n_cores,
            crate::smp::registered_cores(),
            out_dim,
            in_dim,
        );
    }
    let dispatch_t0 = crate::arch::read_cycles();
    ctx.dispatch_matmul(q4k_kernel, args, out_dim);
    let dispatch_cycles = crate::arch::read_cycles().wrapping_sub(dispatch_t0);
    Q4K_CALLS.fetch_add(1, Ordering::Relaxed);
    Q4K_CYCLES.fetch_add(dispatch_cycles, Ordering::Relaxed);
}

/// Dispatch a Q4_K VNNI linear projection across active cores.
///
/// Quantises `x` to Q8_K on a stack-resident buffer once on the BSP,
/// then fans the integer-pipeline matmul out to APs. Bit-NOT-exact
/// against `linear_q4k_dispatch` — the Q8 activation quantisation
/// introduces ~0.5–2 % per-matmul relative error which may compound
/// across 28 layers. Token-ID 25 stability is the only hard
/// invariant; ADR-029 v3 Two-Anchor permits the resulting
/// `logit_bits` drift as a fourth (VNNI) feature-mode signature.
///
/// # Stack budget
///
/// The Q8_K buffer is sized for `in_dim` up to `intermediate_dim` of
/// Qwen3-1.7B (6144) → 24 blocks × 292 B = 7008 B. Total kernel
/// stack is 256 KiB; this fits comfortably.
///
/// # Safety
/// AVX-512F + AVX-512VNNI required. See [`linear_q4k_dispatch`].
#[cfg(feature = "vnni-acceleration")]
#[inline]
unsafe fn linear_q4k_vnni_dispatch(
    x: &[f32],
    w: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    use crate::arch::x86_64::math::vnni::{quantize_row_q8k, Q8K_BLOCK_BYTES};
    debug_assert!(in_dim % 256 == 0);
    debug_assert!(
        in_dim / 256 <= 24,
        "vnni dispatch stack-buf sized for in_dim<=6144"
    );
    let n_blocks = in_dim / 256;
    let q8k_bytes = n_blocks * Q8K_BLOCK_BYTES;
    let mut q8k_buf = [0u8; 24 * Q8K_BLOCK_BYTES]; // sized for intermediate_dim=6144
    quantize_row_q8k(x, &mut q8k_buf[..q8k_bytes], in_dim);

    let args = MatmulArgs {
        x_ptr: q8k_buf.as_ptr() as *const f32,
        w_ptr: w.as_ptr(),
        out_ptr: out.as_mut_ptr(),
        in_dim,
        out_dim,
    };
    let ctx = ParallelMatmulContext::for_active_cores_for_rows(out_dim);
    let dispatch_t0 = crate::arch::read_cycles();
    ctx.dispatch_matmul(q4k_vnni_kernel, args, out_dim);
    let dispatch_cycles = crate::arch::read_cycles().wrapping_sub(dispatch_t0);
    Q4K_CALLS.fetch_add(1, Ordering::Relaxed);
    Q4K_CYCLES.fetch_add(dispatch_cycles, Ordering::Relaxed);
}

/// Cfg-shim: route Q4_K projection through VNNI when the
/// `vnni-acceleration` feature is enabled, otherwise the FP32 path.
///
/// **Default OFF** — VNNI requires per-mode `logit_bits` registration
/// and empirical Token-ID 25 verification on Cherry. A build flag flips
/// the feature on per ADR-029 v8.4 hardware-gate procedure.
#[inline]
#[allow(unused_unsafe)]
unsafe fn linear_q4k_dispatch_routed(
    x: &[f32],
    w: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    #[cfg(feature = "vnni-acceleration")]
    {
        linear_q4k_vnni_dispatch(x, w, out, in_dim, out_dim);
    }
    #[cfg(not(feature = "vnni-acceleration"))]
    {
        linear_q4k_dispatch(x, w, out, in_dim, out_dim);
    }
}

/// Dispatch a Q6_K linear projection across all active cores.
/// # Safety
/// See [`linear_q4k_dispatch`].
#[inline]
unsafe fn linear_q6k_dispatch(x: &[f32], w: &[u8], out: &mut [f32], in_dim: usize, out_dim: usize) {
    let args = MatmulArgs {
        x_ptr: x.as_ptr(),
        w_ptr: w.as_ptr(),
        out_ptr: out.as_mut_ptr(),
        in_dim,
        out_dim,
    };
    let ctx = ParallelMatmulContext::for_active_cores_for_rows(out_dim);
    if !Q6K_DISPATCH_LOGGED.load(Ordering::Acquire)
        && !Q6K_DISPATCH_LOGGED.swap(true, Ordering::AcqRel)
    {
        let _ = writeln!(
            crate::arch::serial::Serial,
            "[MP3.0] AVX-512 Q6_K matmul dispatch: active_cores={} effective_cores={} registered_cores={} out_dim={} in_dim={}",
            crate::smp::active_cores(),
            ctx.n_cores,
            crate::smp::registered_cores(),
            out_dim,
            in_dim,
        );
    }
    let dispatch_t0 = crate::arch::read_cycles();
    ctx.dispatch_matmul(q6k_kernel, args, out_dim);
    let dispatch_cycles = crate::arch::read_cycles().wrapping_sub(dispatch_t0);
    Q6K_CALLS.fetch_add(1, Ordering::Relaxed);
    Q6K_CYCLES.fetch_add(dispatch_cycles, Ordering::Relaxed);
}

// ─────────────────────────────────────────────────────────────────
// linear_q4_0 / linear_q8_0 — SMP row-split AVX-512 entry points for
// the deepseek2 / Kimi K2.6 path.
//
// Mirrors the existing Q4_K / Q6_K dispatch pattern: type-erased
// kernel wrappers (`q4_0_kernel` / `q8_0_kernel` above) are fanned
// out across active cores via `ParallelMatmulContext::dispatch_matmul`.
// On a single-core boot this transparently degrades to a direct call.
// ─────────────────────────────────────────────────────────────────

static Q4_0_DISPATCH_LOGGED: AtomicBool = AtomicBool::new(false);
static Q8_0_DISPATCH_LOGGED: AtomicBool = AtomicBool::new(false);
static Q4_0_CALLS: AtomicU64 = AtomicU64::new(0);
static Q4_0_CYCLES: AtomicU64 = AtomicU64::new(0);
static Q8_0_CALLS: AtomicU64 = AtomicU64::new(0);
static Q8_0_CYCLES: AtomicU64 = AtomicU64::new(0);

/// SMP row-split Q4_0 AVX-512 matmul. Used by the deepseek2 / Kimi
/// K2.6 path for bulk weights (expert ffn_*, MLA projections,
/// embed). Bit-exact equivalent to `linear_q4_0_avx512` — the
/// row-range kernels accumulate per-row independently.
///
/// # Safety
/// AVX-512F must be available on every participating core; caller
/// verifies via CPUID at boot.
#[inline]
unsafe fn linear_q4_0_dispatch(
    x: &[f32],
    w: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    let args = MatmulArgs {
        x_ptr: x.as_ptr(),
        w_ptr: w.as_ptr(),
        out_ptr: out.as_mut_ptr(),
        in_dim,
        out_dim,
    };
    // `.smodel`-v2 layout routing: whole-tensor slices registered by
    // the model loader run the 4-row-interleaved kernel with a
    // group-aligned row split; everything else (plain v1 artifacts,
    // GGUF tensors, Kimi expert sub-slices) stays on the plain kernel.
    let interleave = crate::weight_layout::group_of(w.as_ptr() as usize) as usize;
    let ctx = ParallelMatmulContext::for_active_cores_for_rows(out_dim);
    if !Q4_0_DISPATCH_LOGGED.load(Ordering::Acquire)
        && !Q4_0_DISPATCH_LOGGED.swap(true, Ordering::AcqRel)
    {
        let _ = writeln!(
            crate::arch::serial::Serial,
            "[MP3.0] AVX-512 Q4_0 matmul dispatch: active_cores={} effective_cores={} registered_cores={} out_dim={} in_dim={} interleave={}",
            crate::smp::active_cores(),
            ctx.n_cores,
            crate::smp::registered_cores(),
            out_dim,
            in_dim,
            interleave,
        );
    }
    let t0 = crate::arch::read_cycles();
    if interleave > 1 {
        ctx.dispatch_matmul_aligned(q4_0x4_kernel, args, out_dim, interleave);
    } else {
        ctx.dispatch_matmul(q4_0_kernel, args, out_dim);
    }
    let dt = crate::arch::read_cycles().wrapping_sub(t0);
    Q4_0_CALLS.fetch_add(1, Ordering::Relaxed);
    Q4_0_CYCLES.fetch_add(dt, Ordering::Relaxed);
}

/// SMP row-split Q8_0 AVX-512 matmul. Used by the deepseek2 / Kimi
/// K2.6 path for `token_embd.weight` and `output.weight` (LM head),
/// which bartowski's Q4_0 GGUFs keep in Q8_0 for accuracy. Bit-exact
/// equivalent to `linear_q8_0_avx512`.
///
/// # Safety
/// AVX-512F must be available on every participating core.
#[inline]
unsafe fn linear_q8_0_dispatch(
    x: &[f32],
    w: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    let args = MatmulArgs {
        x_ptr: x.as_ptr(),
        w_ptr: w.as_ptr(),
        out_ptr: out.as_mut_ptr(),
        in_dim,
        out_dim,
    };
    // `.smodel`-v2 layout routing — see `linear_q4_0_dispatch`. For
    // the native Qwen artifact this covers the LM head
    // (`output.weight`, the largest single weight stream per token).
    let interleave = crate::weight_layout::group_of(w.as_ptr() as usize) as usize;
    let ctx = ParallelMatmulContext::for_active_cores_for_rows(out_dim);
    if !Q8_0_DISPATCH_LOGGED.load(Ordering::Acquire)
        && !Q8_0_DISPATCH_LOGGED.swap(true, Ordering::AcqRel)
    {
        let _ = writeln!(
            crate::arch::serial::Serial,
            "[MP3.0] AVX-512 Q8_0 matmul dispatch: active_cores={} effective_cores={} registered_cores={} out_dim={} in_dim={} interleave={}",
            crate::smp::active_cores(),
            ctx.n_cores,
            crate::smp::registered_cores(),
            out_dim,
            in_dim,
            interleave,
        );
    }
    let t0 = crate::arch::read_cycles();
    if interleave > 1 {
        ctx.dispatch_matmul_aligned(q8_0x4_kernel, args, out_dim, interleave);
    } else {
        ctx.dispatch_matmul(q8_0_kernel, args, out_dim);
    }
    let dt = crate::arch::read_cycles().wrapping_sub(t0);
    Q8_0_CALLS.fetch_add(1, Ordering::Relaxed);
    Q8_0_CYCLES.fetch_add(dt, Ordering::Relaxed);
}

// ─────────────────────────────────────────────────────────────────
// Fused multi-projection dispatch (qwen-perf-v2)
// ─────────────────────────────────────────────────────────────────
//
// Q/K/V (and gate/up) projections consume the SAME input vector and
// write disjoint output buffers. Dispatching them separately pays the
// publish + tree-barrier round-trip once per projection — 3 barriers
// per attention block and 2 per MLP, ~140 of the ~200 dispatches per
// token. A fused dispatch concatenates the projections into one
// virtual row space (rows [0, Σ out_dim)) and fans THAT out once.
//
// # Bit-exactness
//
// Unchanged. Every output row is still reduced by exactly one core
// through the same per-quant range kernel in the same K-order; only
// the row→core assignment shifts (which never affects results — the
// row kernels are pure functions of (x, w_row)). Token-ID and
// logit_bits anchors are unaffected.

/// Maximum projections one fused dispatch can carry (Q‖K‖V = 3).
const FUSED_MAX_SEGMENTS: usize = 3;

/// One projection inside a fused dispatch. Raw pointers so the table
/// can live in a `static` shared with APs (same pattern as
/// [`MatmulArgs`]).
#[derive(Copy, Clone)]
struct FusedSegment {
    w_ptr: *const u8,
    w_len: usize,
    out_ptr: *mut f32,
    out_dim: usize,
    /// First virtual row of this segment in the fused row space.
    row_base: usize,
    quant: zero_gguf_parser::GgmlType,
    /// `.smodel`-v2 row-interleave group (1 = plain row-major). Looked
    /// up once at segment construction on the BSP.
    interleave: u32,
}

const FUSED_SEGMENT_ZERO: FusedSegment = FusedSegment {
    w_ptr: core::ptr::null(),
    w_len: 0,
    out_ptr: core::ptr::null_mut(),
    out_dim: 0,
    row_base: 0,
    quant: zero_gguf_parser::GgmlType::F32,
    interleave: 1,
};

/// BSP-written, AP-read segment table for the in-flight fused
/// dispatch.
///
/// # Synchronisation
///
/// Plain (non-atomic) stores by the BSP, sequenced-before the
/// `Release` publication of `WORK_EPOCH` inside `dispatch_matmul`;
/// APs read the table only after their `Acquire` load of the new
/// epoch, which establishes happens-before. The BSP never starts the
/// next dispatch before `wait_complete` (or the degraded single-core
/// recovery, which runs on the BSP itself), so the table is never
/// rewritten while an AP may still execute work — stale-epoch APs are
/// rejected at the barrier and never run the kernel.
#[repr(C, align(64))]
struct FusedTable {
    segments: core::cell::UnsafeCell<[FusedSegment; FUSED_MAX_SEGMENTS]>,
    n_segments: core::cell::UnsafeCell<usize>,
}

// SAFETY: see the struct-level synchronisation contract above.
unsafe impl Sync for FusedTable {}

static FUSED_TABLE: FusedTable = FusedTable {
    segments: core::cell::UnsafeCell::new([FUSED_SEGMENT_ZERO; FUSED_MAX_SEGMENTS]),
    n_segments: core::cell::UnsafeCell::new(0),
};

static FUSED_DISPATCH_LOGGED: AtomicBool = AtomicBool::new(false);
static FUSED_CALLS: AtomicU64 = AtomicU64::new(0);
static FUSED_CYCLES: AtomicU64 = AtomicU64::new(0);

/// Quants the fused kernel can route. Must stay in sync with the
/// `match` in [`fused_linear_kernel`].
#[inline(always)]
fn fused_quant_supported(quant: zero_gguf_parser::GgmlType) -> bool {
    use zero_gguf_parser::GgmlType;
    matches!(
        quant,
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q6K
    )
}

/// Build a [`FusedSegment`] from caller-held slices. `row_base` is
/// assigned by [`linear_fused_dispatch`].
#[inline(always)]
fn fused_segment(w: &[u8], out: &mut [f32], quant: zero_gguf_parser::GgmlType) -> FusedSegment {
    FusedSegment {
        w_ptr: w.as_ptr(),
        w_len: w.len(),
        out_ptr: out.as_mut_ptr(),
        out_dim: out.len(),
        row_base: 0,
        quant,
        interleave: crate::weight_layout::group_of(w.as_ptr() as usize),
    }
}

/// Type-erased fused kernel for the SMP dispatcher. Maps the global
/// (virtual) row range onto each segment and forwards to the
/// segment's per-quant range kernel.
unsafe fn fused_linear_kernel(args: &MatmulArgs, range: RowRange) {
    use zero_gguf_parser::GgmlType;
    let x = core::slice::from_raw_parts(args.x_ptr, args.in_dim);
    let segments = &*FUSED_TABLE.segments.get();
    let n_segments = *FUSED_TABLE.n_segments.get();
    let mut s = 0;
    while s < n_segments {
        let seg = &segments[s];
        let seg_start = seg.row_base;
        let seg_end = seg.row_base + seg.out_dim;
        let lo = if range.start > seg_start {
            range.start
        } else {
            seg_start
        };
        let hi = if range.end < seg_end {
            range.end
        } else {
            seg_end
        };
        if lo < hi {
            let local = RowRange {
                start: lo - seg_start,
                end: hi - seg_start,
            };
            let w = core::slice::from_raw_parts(seg.w_ptr, seg.w_len);
            let out = core::slice::from_raw_parts_mut(seg.out_ptr, seg.out_dim);
            match (seg.quant, seg.interleave > 1) {
                (GgmlType::Q4_0, false) => {
                    linear_q4_0_avx512_range(x, w, out, args.in_dim, seg.out_dim, local)
                }
                (GgmlType::Q4_0, true) => {
                    linear_q4_0x4_avx512_range(x, w, out, args.in_dim, seg.out_dim, local)
                }
                (GgmlType::Q8_0, false) => {
                    linear_q8_0_avx512_range(x, w, out, args.in_dim, seg.out_dim, local)
                }
                (GgmlType::Q8_0, true) => {
                    linear_q8_0x4_avx512_range(x, w, out, args.in_dim, seg.out_dim, local)
                }
                (GgmlType::Q4K, _) => {
                    linear_q4k_avx512_range(x, w, out, args.in_dim, seg.out_dim, local)
                }
                (GgmlType::Q6K, _) => {
                    linear_q6k_avx512_range(x, w, out, args.in_dim, seg.out_dim, local)
                }
                // Unreachable: linear_fused_dispatch call sites guard
                // every segment with fused_quant_supported().
                _ => {}
            }
        }
        s += 1;
    }
}

/// Dispatch several same-input linear projections as ONE parallel
/// matmul. All segments must share `in_dim` (they consume the same
/// `x`) and carry a [`fused_quant_supported`] quant.
///
/// # Safety
/// Same contract as [`linear_q4k_dispatch`]; additionally every
/// segment's `w_len` must cover `out_dim` rows of its quant's row
/// stride at `in_dim` columns.
#[inline]
unsafe fn linear_fused_dispatch(x: &[f32], segments: &[FusedSegment], in_dim: usize) {
    debug_assert!(segments.len() <= FUSED_MAX_SEGMENTS);
    let mut table = [FUSED_SEGMENT_ZERO; FUSED_MAX_SEGMENTS];
    let mut total_rows = 0usize;
    let mut align = 1usize;
    let mut i = 0;
    while i < segments.len() {
        let mut seg = segments[i];
        seg.row_base = total_rows;
        total_rows += seg.out_dim;
        if (seg.interleave as usize) > align {
            align = seg.interleave as usize;
        }
        // Keep the virtual row space group-aligned at segment edges so
        // a group-aligned global split maps to group-aligned local
        // ranges. The x4 kernels stay CORRECT for unaligned ranges
        // (lane masking); this only protects the perf property. All
        // Qwen3 out_dims (2048/1024/6144) are multiples of 4.
        debug_assert!(
            seg.interleave <= 1 || seg.out_dim % (seg.interleave as usize) == 0,
            "fused interleaved segment with unaligned out_dim"
        );
        table[i] = seg;
        i += 1;
    }
    // Plain stores; published to APs via dispatch_matmul's
    // Release(WORK_EPOCH) — see FusedTable's synchronisation contract.
    *FUSED_TABLE.segments.get() = table;
    *FUSED_TABLE.n_segments.get() = segments.len();

    let args = MatmulArgs {
        x_ptr: x.as_ptr(),
        w_ptr: core::ptr::null(),
        out_ptr: core::ptr::null_mut(),
        in_dim,
        out_dim: total_rows,
    };
    let ctx = ParallelMatmulContext::for_active_cores_for_rows(total_rows);
    if !FUSED_DISPATCH_LOGGED.load(Ordering::Acquire)
        && !FUSED_DISPATCH_LOGGED.swap(true, Ordering::AcqRel)
    {
        let _ = writeln!(
            crate::arch::serial::Serial,
            "[MP3.0] AVX-512 fused matmul dispatch: segments={} total_rows={} in_dim={} effective_cores={}",
            segments.len(),
            total_rows,
            in_dim,
            ctx.n_cores,
        );
    }
    let t0 = crate::arch::read_cycles();
    ctx.dispatch_matmul_aligned(fused_linear_kernel, args, total_rows, align);
    let dt = crate::arch::read_cycles().wrapping_sub(t0);
    FUSED_CALLS.fetch_add(1, Ordering::Relaxed);
    FUSED_CYCLES.fetch_add(dt, Ordering::Relaxed);
}

// ─────────────────────────────────────────────────────────────────
// Fused LM-head matmul + argmax (qwen-perf-v2 round 3, N3)
// ─────────────────────────────────────────────────────────────────
//
// The LM-head dispatch writes 151,936 logits across all participating
// cores; the BSP then re-read the full ~600 KB logit buffer twice
// (vector max + first-index scan) — with every cacheline still
// resident in some OTHER core's L1/L2 right after the barrier.
// Folding the argmax into the matmul kernel lets every core scan the
// rows it just wrote (cache-hot) and propose ONE (max, first-index)
// candidate; candidates merge through a lock-free CAS slot the BSP
// reads after the barrier.
//
// # Scalar-identical semantics (bit-exactness argument)
//
// Per-core scan: [`argmax_f32_avx512`] over the core's contiguous row
// range — first-max-wins within the range; NaN is never proposed
// (ordered compares only), an all-NaN range proposes
// (range.start, -inf) which loses against any finite candidate and
// reproduces the sequential scan's "max stays -inf" outcome when ALL
// logits are NaN (the caller's finiteness check rejects it either
// way).
// Merge: candidate (v_a, i_a) beats (v_b, i_b) iff v_a > v_b, or
// v_a == v_b and i_a < i_b. Values are never NaN, so this is a total
// order and the merged winner is independent of CAS arrival order:
// it is exactly the global maximum with the SMALLEST index —
// identical to the sequential first-max-wins scan over the full
// buffer. ±0.0 ties behave identically because IEEE `==`/`>` ignore
// the sign of zero in both implementations.
// The logits buffer itself is still written in full, byte-identically
// (the inner matmul kernels are untouched) — β-anchor logit_bits
// capture and `debug-logits` top-k read it exactly as before.

/// Initial CAS slot value: (-inf, u32::MAX) — loses every comparison.
const LM_ARGMAX_INIT: u64 = ((0xff80_0000u32 as u64) << 32) | (u32::MAX as u64); // f32::NEG_INFINITY.to_bits()

#[inline(always)]
fn lm_argmax_pack(val: f32, idx: u32) -> u64 {
    ((val.to_bits() as u64) << 32) | idx as u64
}

#[inline(always)]
fn lm_argmax_unpack(packed: u64) -> (f32, u32) {
    (f32::from_bits((packed >> 32) as u32), packed as u32)
}

/// BSP-written routing + CAS merge slot for the in-flight LM-head
/// argmax dispatch.
///
/// # Synchronisation
///
/// `quant`/`interleaved` follow the [`FusedTable`] contract: plain
/// BSP stores sequenced-before the `Release` publication of
/// `WORK_EPOCH` inside `dispatch_matmul`, AP reads after the
/// `Acquire` epoch load. `best` is reset by the BSP before the
/// dispatch and merged via CAS by every participant; the BSP reads
/// it after the barrier completes (the barrier's Release/Acquire
/// chain orders the final merge before the read — the same chain
/// that publishes the `out` rows themselves).
#[repr(C, align(64))]
struct LmHeadArgmaxCtl {
    /// Packed best candidate: high 32 bits = f32 bit pattern of the
    /// value, low 32 bits = row index.
    best: AtomicU64,
    quant: core::cell::UnsafeCell<zero_gguf_parser::GgmlType>,
    interleaved: core::cell::UnsafeCell<bool>,
}

// SAFETY: see the struct-level synchronisation contract above.
unsafe impl Sync for LmHeadArgmaxCtl {}

static LM_ARGMAX_CTL: LmHeadArgmaxCtl = LmHeadArgmaxCtl {
    best: AtomicU64::new(LM_ARGMAX_INIT),
    quant: core::cell::UnsafeCell::new(zero_gguf_parser::GgmlType::Q8_0),
    interleaved: core::cell::UnsafeCell::new(false),
};

/// Lock-free merge of one per-range candidate. The comparison is a
/// total order over the proposed pairs (values never NaN), so the
/// final slot content does not depend on arrival order.
fn lm_argmax_propose(val: f32, idx: u32) {
    let mut cur = LM_ARGMAX_CTL.best.load(Ordering::Acquire);
    loop {
        let (cv, ci) = lm_argmax_unpack(cur);
        if !(val > cv || (val == cv && idx < ci)) {
            return;
        }
        match LM_ARGMAX_CTL.best.compare_exchange_weak(
            cur,
            lm_argmax_pack(val, idx),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return,
            Err(actual) => cur = actual,
        }
    }
}

/// Quants the combined LM-head kernel can route. Must stay in sync
/// with the `match` in [`lm_head_argmax_kernel`].
#[inline(always)]
fn lm_head_argmax_quant_supported(quant: zero_gguf_parser::GgmlType) -> bool {
    use zero_gguf_parser::GgmlType;
    matches!(
        quant,
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q6K
    )
}

/// Type-erased LM-head kernel: the per-quant matmul for the range
/// (the UNCHANGED inner kernels — logits bytes identical), then a
/// local argmax over the rows this core just wrote, merged into
/// [`LM_ARGMAX_CTL`].
unsafe fn lm_head_argmax_kernel(args: &MatmulArgs, range: RowRange) {
    use zero_gguf_parser::GgmlType;
    let quant = *LM_ARGMAX_CTL.quant.get();
    let interleaved = *LM_ARGMAX_CTL.interleaved.get();
    match (quant, interleaved) {
        (GgmlType::Q4_0, false) => q4_0_kernel(args, range),
        (GgmlType::Q4_0, true) => q4_0x4_kernel(args, range),
        (GgmlType::Q8_0, false) => q8_0_kernel(args, range),
        (GgmlType::Q8_0, true) => q8_0x4_kernel(args, range),
        (GgmlType::Q4K, _) => q4k_kernel(args, range),
        (GgmlType::Q6K, _) => q6k_kernel(args, range),
        // Unreachable: the dispatch site guards with
        // lm_head_argmax_quant_supported().
        _ => return,
    }
    if range.start >= range.end {
        return;
    }
    // Same whole-buffer reconstruction as the matmul kernels above;
    // this core only READS its own just-written disjoint rows.
    let out = core::slice::from_raw_parts(args.out_ptr as *const f32, args.out_dim);
    let (local_idx, local_max) = argmax_f32_avx512(&out[range.start..range.end]);
    lm_argmax_propose(local_max, (range.start + local_idx) as u32);
}

/// Combined LM-head linear projection + argmax as ONE parallel
/// dispatch. Returns `(first_max_index, max_value)` with semantics
/// identical to `argmax_f32_avx512(out)` after a plain matmul
/// dispatch (see the module-section comment for the proof).
///
/// # Safety
/// Same contract as [`linear_dispatch_avx512`].
unsafe fn lm_head_linear_argmax_dispatch(
    x: &[f32],
    w: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
    quant: zero_gguf_parser::GgmlType,
) -> (usize, f32) {
    let interleave = crate::weight_layout::group_of(w.as_ptr() as usize) as usize;
    // Reset + routing publish. Plain/Relaxed stores are fine: both
    // are published to APs via dispatch_matmul's Release(WORK_EPOCH).
    LM_ARGMAX_CTL.best.store(LM_ARGMAX_INIT, Ordering::Relaxed);
    *LM_ARGMAX_CTL.quant.get() = quant;
    *LM_ARGMAX_CTL.interleaved.get() = interleave > 1;
    let args = MatmulArgs {
        x_ptr: x.as_ptr(),
        w_ptr: w.as_ptr(),
        out_ptr: out.as_mut_ptr(),
        in_dim,
        out_dim,
    };
    let ctx = ParallelMatmulContext::for_active_cores_for_rows(out_dim);
    ctx.dispatch_matmul_aligned(lm_head_argmax_kernel, args, out_dim, interleave.max(1));
    let (val, idx) = lm_argmax_unpack(LM_ARGMAX_CTL.best.load(Ordering::Acquire));
    (idx as usize, val)
}

/// Errors returned by `linear_dispatch_avx512`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearDispatchAvx512Error {
    /// Caller asked the AVX-512 dispatcher to handle a `GgmlType` we
    /// don't have a kernel for yet. The boot path should treat this
    /// as a hard failure during weight discovery.
    UnsupportedQuant(zero_gguf_parser::GgmlType),
}

/// Cross-quant linear projection through AVX-512 kernels. Drop-in
/// replacement for `zero_llm_inference::linear_dispatch` on
/// x86_64 + avx512-acceleration builds — uses the raw AVX-512
/// kernels directly (no scalar fallback per call; caller must
/// confirm AVX-512F at boot).
///
/// Currently routes:
///   * `Q4K`  → existing `linear_q4k_dispatch` (with SMP row-split)
///   * `Q6K`  → existing `linear_q6k_dispatch` (with SMP row-split)
///   * `Q4_0` → new `linear_q4_0_dispatch` (single-thread today)
///   * `Q8_0` → new `linear_q8_0_dispatch` (single-thread today)
///
/// Anything else → `UnsupportedQuant`.
///
/// # Safety
/// Caller must guarantee AVX-512F is available and that the slice
/// dimensions match the quant's block layout.
#[allow(dead_code)]
pub unsafe fn linear_dispatch_avx512(
    x: &[f32],
    w: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
    quant: zero_gguf_parser::GgmlType,
) -> Result<(), LinearDispatchAvx512Error> {
    use zero_gguf_parser::GgmlType;
    match quant {
        GgmlType::Q4K => {
            linear_q4k_dispatch(x, w, out, in_dim, out_dim);
            Ok(())
        }
        GgmlType::Q6K => {
            linear_q6k_dispatch(x, w, out, in_dim, out_dim);
            Ok(())
        }
        GgmlType::Q4_0 => {
            linear_q4_0_dispatch(x, w, out, in_dim, out_dim);
            Ok(())
        }
        GgmlType::Q8_0 => {
            linear_q8_0_dispatch(x, w, out, in_dim, out_dim);
            Ok(())
        }
        other => {
            // Fail-fast: every call site uses `let _ = ...` so a silent
            // Err would propagate as garbage output. Panic instead so the
            // kernel panic handler prints the offending quant type and a
            // call stack pointing at the projection that triggered it.
            panic!(
                "linear_dispatch_avx512: unsupported quant type {}",
                other as u32
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// lm_head with AVX-512 (top-level: matches inference_neon contract)
// ─────────────────────────────────────────────────────────────────

/// AVX-512-accelerated LM-head: RMSNorm + Q6_K linear + argmax.
///
/// Replaces sacred `lm_head_argmax` with AVX-512 (parallel-dispatched)
/// `linear_q6k` for the 151,936 × 2048 output projection — the largest
/// single matmul in the forward pass and the prime target for SMP
/// parallelism.
///
/// # Safety
/// Calls unsafe AVX-512 intrinsics via `linear_q6k_dispatch`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn lm_head_argmax_avx512(
    final_hidden: &[f32],
    output_norm_weight: &[f32],
    output_weight: &[u8],
    output_quant: zero_gguf_parser::GgmlType,
    rms_eps: f32,
    embedding_dim: usize,
    norm_buf: &mut [f32],
    logits_buf: &mut [f32],
) -> Result<u32, LmHeadError> {
    // Step 1: Final RMSNorm (sacred scalar — small, not bottleneck).
    rmsnorm(final_hidden, output_norm_weight, norm_buf, rms_eps);

    // Steps 2+3: LM head linear projection + argmax — ONE parallel
    // dispatch (qwen-perf-v2 round 3, N3). Each core argmax-scans the
    // rows it just wrote (cache-hot) instead of the BSP re-streaming
    // the full logit buffer out of 63 remote caches; the per-range
    // candidates CAS-merge with first-max-wins-by-smallest-index
    // semantics — provably identical to the sequential scan (see the
    // LM_ARGMAX_CTL section comment). `lm_head_cycles` now includes
    // the (former post-pass) argmax.
    let lm_head_t0 = crate::arch::read_cycles();
    let (max_idx, max_val) = if lm_head_argmax_quant_supported(output_quant) {
        lm_head_linear_argmax_dispatch(
            norm_buf,
            output_weight,
            logits_buf,
            embedding_dim,
            VOCAB_SIZE_PADDED,
            output_quant,
        )
    } else {
        // Conservative fallback: plain dispatch + BSP-side scan.
        let _ = linear_dispatch_avx512(
            norm_buf,
            output_weight,
            logits_buf,
            embedding_dim,
            VOCAB_SIZE_PADDED,
            output_quant,
        );
        argmax_f32_avx512(logits_buf)
    };
    let lm_head_cycles = crate::arch::read_cycles().wrapping_sub(lm_head_t0);
    LM_HEAD_CALLS.fetch_add(1, Ordering::Relaxed);
    LM_HEAD_CYCLES.fetch_add(lm_head_cycles, Ordering::Relaxed);
    if !LM_HEAD_DISPATCH_LOGGED.load(Ordering::Acquire)
        && !LM_HEAD_DISPATCH_LOGGED.swap(true, Ordering::AcqRel)
    {
        let _ = writeln!(
            crate::arch::serial::Serial,
            "[MP3.0] AVX-512 LM-head Q6_K dispatch: active_cores={} effective_cores={} registered_cores={} out_dim={} in_dim={} cycles={}",
            crate::smp::active_cores(),
            ParallelMatmulContext::for_active_cores_for_rows(VOCAB_SIZE_PADDED).n_cores,
            crate::smp::registered_cores(),
            VOCAB_SIZE_PADDED,
            embedding_dim,
            lm_head_cycles,
        );
    }

    if !max_val.is_finite() {
        return Err(LmHeadError::NonFiniteLogits);
    }

    if max_idx >= VOCAB_SIZE_REAL {
        return Err(LmHeadError::ArgmaxInPaddingRegion(max_idx));
    }

    Ok(max_idx as u32)
}

// ─────────────────────────────────────────────────────────────────
// MLP with AVX-512
// ─────────────────────────────────────────────────────────────────

/// AVX-512-accelerated SwiGLU MLP. Three matmuls (gate, up, down) each
/// dispatched in parallel across active cores; SiLU + element-wise
/// multiply scalar (cheap, not worth dispatch overhead).
#[allow(clippy::too_many_arguments)]
unsafe fn mlp_swiglu_avx512(
    input: &[f32],
    gate_weight: &[u8],
    up_weight: &[u8],
    down_weight: &[u8],
    gate_quant: zero_gguf_parser::GgmlType,
    up_quant: zero_gguf_parser::GgmlType,
    down_quant: zero_gguf_parser::GgmlType,
    embedding_dim: usize,
    intermediate_dim: usize,
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    hidden_buf: &mut [f32],
    output: &mut [f32],
) {
    // Steps 1+2: Gate + Up projections — ONE fused parallel dispatch.
    // Both consume `input`; fusing saves one publish + tree-barrier
    // round-trip per layer (28 per token). Per-row K-order bit-exact.
    if fused_dispatch_enabled()
        && fused_quant_supported(gate_quant)
        && fused_quant_supported(up_quant)
    {
        let segments = [
            fused_segment(gate_weight, &mut gate_buf[..intermediate_dim], gate_quant),
            fused_segment(up_weight, &mut up_buf[..intermediate_dim], up_quant),
        ];
        linear_fused_dispatch(input, &segments, embedding_dim);
    } else {
        let _ = linear_dispatch_avx512(
            input,
            gate_weight,
            gate_buf,
            embedding_dim,
            intermediate_dim,
            gate_quant,
        );
        let _ = linear_dispatch_avx512(
            input,
            up_weight,
            up_buf,
            embedding_dim,
            intermediate_dim,
            up_quant,
        );
    }

    // Steps 3+4: fused SiLU(gate) × up — single AVX-512 16-lane pass
    // per 16 elements. Replaces the ~6144 scalar `libm::expf` calls
    // and the subsequent scalar multiply that dominated the MLP path
    // at v8.3. ~1 ULP feature-mode drift vs sacred scalar SiLU per
    // ADR-029 v3 Two-Anchor.
    silu_mul_avx512(gate_buf, up_buf, hidden_buf, intermediate_dim);

    // Step 5: Down projection — native quant-aware.
    let _ = linear_dispatch_avx512(
        hidden_buf,
        down_weight,
        output,
        intermediate_dim,
        embedding_dim,
        down_quant,
    );
}

// ─────────────────────────────────────────────────────────────────
// Parallel per-head attention (qwen-perf-v2 round 3, S1)
// ─────────────────────────────────────────────────────────────────
//
// Score + softmax + weighted-sum ran entirely on the BSP while 63 APs
// idle. Per-head work is fully independent — head h reads its own
// q_out stripe and the (shared, read-only) K/V cache slices and
// writes only its own attn_concat stripe — so heads fan out across
// cores like matmul rows fan out, ONE barrier round-trip per layer.
//
// # Bit-exactness
//
// Per head, the parallel kernel executes the IDENTICAL operation
// sequence as the sequential loop: the same `dot_product_f32_avx512`
// per cached token in ascending t, the same sacred scalar `softmax`,
// the same `weighted_add_f32_avx512` accumulation in ascending t.
// Only (a) which core runs a head and (b) the score/accumulation
// buffer ADDRESSES change (per-head score stripes; accumulation
// directly in the head's attn_concat stripe instead of a shared
// staging buffer + copy). FP results do not depend on either. Token
// IDs and logit_bits of all anchors are preserved exactly.
//
// # Engagement gate
//
// At short context the per-head work is microseconds and the barrier
// would dominate; below ATTN_PARALLEL_MIN_TOKENS (runtime-tunable,
// default 64) the sequential loop runs UNGATED and byte-identical to
// the previous revision. The 13-token β-anchor prefill and the
// 48-token bench never engage the parallel path; Cherry streaming
// (2048-token class) engages it from token 64 on.

/// Default context-length threshold (tokens) for the parallel
/// attention path. Original value (64): the per-head parallel path
/// engages only at >=64 tokens, so it is OFF at prompt token 0 and
/// does NOT affect the token-0 instability we are tracing. Kept at the
/// original value so the diagnostic boot reproduces deploy-4 exactly.
/// Runtime-tunable via `set_attn_parallel_min_tokens` for hardware A/B.
pub const DEFAULT_ATTN_PARALLEL_MIN_TOKENS: usize = 64;

static ATTN_PARALLEL_MIN_TOKENS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(DEFAULT_ATTN_PARALLEL_MIN_TOKENS);

#[inline(always)]
pub fn attn_parallel_min_tokens() -> usize {
    ATTN_PARALLEL_MIN_TOKENS.load(Ordering::Acquire)
}

/// Set the parallel-attention engagement threshold. `0`/`1` engage it
/// for every decoded token; `usize::MAX` disables it. Returns the
/// stored value.
#[inline]
pub fn set_attn_parallel_min_tokens(tokens: usize) -> usize {
    ATTN_PARALLEL_MIN_TOKENS.store(tokens, Ordering::Release);
    tokens
}

#[inline]
pub fn reset_attn_parallel_min_tokens() -> usize {
    set_attn_parallel_min_tokens(DEFAULT_ATTN_PARALLEL_MIN_TOKENS)
}

static ATTN_PAR_DISPATCH_LOGGED: AtomicBool = AtomicBool::new(false);
static ATTN_PAR_CALLS: AtomicU64 = AtomicU64::new(0);
static ATTN_PAR_CYCLES: AtomicU64 = AtomicU64::new(0);

/// BSP-written, AP-read descriptor of the in-flight per-head
/// attention dispatch. Same synchronisation contract as
/// [`FusedTable`]: plain BSP stores sequenced-before the
/// `Release` epoch publish; AP reads after the `Acquire` epoch load;
/// never rewritten while any AP may still execute.
#[repr(C, align(64))]
struct AttnTable {
    /// Post-QK-norm, post-RoPE Q vectors (`n_q_heads * head_dim`).
    q_ptr: core::cell::UnsafeCell<*const f32>,
    /// K/V cache slices for `total_tokens` tokens (layout
    /// `[t][kv_head][head_dim]`).
    k_ptr: core::cell::UnsafeCell<*const f32>,
    v_ptr: core::cell::UnsafeCell<*const f32>,
    /// Per-head score stripes: head h uses
    /// `score[h * score_stride .. h * score_stride + total_tokens]`.
    score_ptr: core::cell::UnsafeCell<*mut f32>,
    score_stride: core::cell::UnsafeCell<usize>,
    /// Concatenated head outputs (`n_q_heads * head_dim`).
    concat_ptr: core::cell::UnsafeCell<*mut f32>,
    total_tokens: core::cell::UnsafeCell<usize>,
    kv_dim: core::cell::UnsafeCell<usize>,
    head_dim: core::cell::UnsafeCell<usize>,
    gqa_ratio: core::cell::UnsafeCell<usize>,
    scale: core::cell::UnsafeCell<f32>,
    /// 0 = ok; 1 = a head hit softmax NumericalInstability. Written
    /// with Relaxed by APs (ordered by the barrier), read by the BSP
    /// after wait_complete.
    error: core::sync::atomic::AtomicU32,
}

// SAFETY: see the struct-level synchronisation contract above.
unsafe impl Sync for AttnTable {}

static ATTN_TABLE: AttnTable = AttnTable {
    q_ptr: core::cell::UnsafeCell::new(core::ptr::null()),
    k_ptr: core::cell::UnsafeCell::new(core::ptr::null()),
    v_ptr: core::cell::UnsafeCell::new(core::ptr::null()),
    score_ptr: core::cell::UnsafeCell::new(core::ptr::null_mut()),
    score_stride: core::cell::UnsafeCell::new(0),
    concat_ptr: core::cell::UnsafeCell::new(core::ptr::null_mut()),
    total_tokens: core::cell::UnsafeCell::new(0),
    kv_dim: core::cell::UnsafeCell::new(0),
    head_dim: core::cell::UnsafeCell::new(0),
    gqa_ratio: core::cell::UnsafeCell::new(1),
    scale: core::cell::UnsafeCell::new(1.0),
    error: core::sync::atomic::AtomicU32::new(0),
};

/// Type-erased per-head attention kernel for the SMP dispatcher.
/// `range` indexes Q-heads, not rows.
unsafe fn attn_heads_kernel(_args: &MatmulArgs, range: RowRange) {
    let q = *ATTN_TABLE.q_ptr.get();
    let k = *ATTN_TABLE.k_ptr.get();
    let v = *ATTN_TABLE.v_ptr.get();
    let score = *ATTN_TABLE.score_ptr.get();
    let score_stride = *ATTN_TABLE.score_stride.get();
    let concat = *ATTN_TABLE.concat_ptr.get();
    let total_tokens = *ATTN_TABLE.total_tokens.get();
    let kv_dim = *ATTN_TABLE.kv_dim.get();
    let head_dim = *ATTN_TABLE.head_dim.get();
    let gqa_ratio = *ATTN_TABLE.gqa_ratio.get();
    let scale = *ATTN_TABLE.scale.get();

    let mut q_h = range.start;
    while q_h < range.end {
        let kv_h = q_h / gqa_ratio;
        let q_vec = core::slice::from_raw_parts(q.add(q_h * head_dim), head_dim);
        let sb = core::slice::from_raw_parts_mut(score.add(q_h * score_stride), total_tokens);

        // Score computation: Q × Kᵀ — identical ops/order to the
        // sequential loop.
        let mut t = 0;
        while t < total_tokens {
            let k_vec = core::slice::from_raw_parts(k.add(t * kv_dim + kv_h * head_dim), head_dim);
            sb[t] = dot_product_f32_avx512(q_vec, k_vec, head_dim) * scale;
            t += 1;
        }

        // Softmax — sacred scalar, same fn as the sequential loop.
        if softmax(sb, total_tokens).is_err() {
            ATTN_TABLE.error.store(1, Ordering::Relaxed);
            q_h += 1;
            continue;
        }

        // Weighted-sum: attn × V — accumulate directly into this
        // head's (disjoint) attn_concat stripe; same FP ops/order as
        // accumulating in a staging buffer and copying.
        let out = core::slice::from_raw_parts_mut(concat.add(q_h * head_dim), head_dim);
        let mut i = 0;
        while i < head_dim {
            out[i] = 0.0;
            i += 1;
        }
        let mut t = 0;
        while t < total_tokens {
            let v_vec = core::slice::from_raw_parts(v.add(t * kv_dim + kv_h * head_dim), head_dim);
            weighted_add_f32_avx512(sb[t], v_vec, out, head_dim);
            t += 1;
        }

        q_h += 1;
    }
}

// ─────────────────────────────────────────────────────────────────
// GQA attention with AVX-512
// ─────────────────────────────────────────────────────────────────

/// AVX-512-accelerated GQA attention for single token, single layer.
/// Replicates `inference_neon::gqa_attention_single_token_neon` using
/// HAL AVX-512 for Q/K/V/O matmuls + score-computation + weighted-sum.
///
/// **Per-head dot-product / weighted-sum** stays single-core: head_dim
/// is 128 and dispatching 16 cores on a 128-element reduction would
/// be barrier-dominated. The matmuls (Q/K/V/O at 2048 / 1024 output
/// dims) get the SMP win.
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_single_token_avx512<const HALF: usize>(
    normed_input: &[f32],
    q_weight: &[u8],
    k_weight: &[u8],
    v_weight: &[u8],
    o_weight: &[u8],
    q_quant: zero_gguf_parser::GgmlType,
    k_quant: zero_gguf_parser::GgmlType,
    v_quant: zero_gguf_parser::GgmlType,
    o_quant: zero_gguf_parser::GgmlType,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    layer_idx: usize,
    token_offset: usize,
    _rope_ctx: &RopeContext<HALF>,
    rope_cos_lut: &[f32],
    rope_sin_lut: &[f32],
    rms_eps: f32,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    embedding_dim: usize,
    kv_cache: &mut KvCache,
    q_buf: &mut [f32],
    k_buf: &mut [f32],
    v_buf: &mut [f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    score_buf: &mut [f32],
    attn_head_buf: &mut [f32],
    attn_concat: &mut [f32],
    output: &mut [f32],
) -> Result<(), AttentionError> {
    let q_dim = n_q_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let gqa_ratio = n_q_heads / n_kv_heads;
    let scale = 1.0 / libm::sqrtf(head_dim as f32);
    let total_tokens = token_offset + 1;

    // ── Step 1: Q/K/V projections — ONE fused parallel dispatch ──
    //
    // All three consume `normed_input`; fusing them saves two
    // publish + tree-barrier round-trips per layer (56 per token)
    // while keeping per-row K-order bit-exact.
    if fused_dispatch_enabled()
        && fused_quant_supported(q_quant)
        && fused_quant_supported(k_quant)
        && fused_quant_supported(v_quant)
    {
        let segments = [
            fused_segment(q_weight, &mut q_buf[..q_dim], q_quant),
            fused_segment(k_weight, &mut k_buf[..kv_dim], k_quant),
            fused_segment(v_weight, &mut v_buf[..kv_dim], v_quant),
        ];
        linear_fused_dispatch(normed_input, &segments, embedding_dim);
    } else {
        let _ =
            linear_dispatch_avx512(normed_input, q_weight, q_buf, embedding_dim, q_dim, q_quant);
        let _ = linear_dispatch_avx512(
            normed_input,
            k_weight,
            k_buf,
            embedding_dim,
            kv_dim,
            k_quant,
        );
        let _ = linear_dispatch_avx512(
            normed_input,
            v_weight,
            v_buf,
            embedding_dim,
            kv_dim,
            v_quant,
        );
    }

    // Trace: raw Q/K/V projection output. If these are already
    // non-finite at token 0, the bug is in the matmul dispatch
    // (fused/SMP), not in norm/RoPE/score. `fused_dispatch_enabled`
    // is logged once at the projection so the trace is unambiguous.
    trace_stats(layer_idx, token_offset, "q_proj", &q_buf[..q_dim]);
    trace_stats(layer_idx, token_offset, "k_proj", &k_buf[..kv_dim]);
    trace_stats(layer_idx, token_offset, "v_proj", &v_buf[..kv_dim]);

    // ── Step 2: QK-Norm (per-head RMSNorm) — sacred scalar ───────
    for h in 0..n_q_heads {
        let offset = h * head_dim;
        rmsnorm(
            &q_buf[offset..offset + head_dim],
            q_norm_weight,
            &mut q_out[offset..offset + head_dim],
            rms_eps,
        );
    }
    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        rmsnorm(
            &k_buf[offset..offset + head_dim],
            k_norm_weight,
            &mut k_out[offset..offset + head_dim],
            rms_eps,
        );
    }

    trace_stats(layer_idx, token_offset, "q_qknorm", &q_out[..q_dim]);
    trace_stats(layer_idx, token_offset, "k_qknorm", &k_out[..kv_dim]);

    // ── Step 3: RoPE — AVX-512 with shared sincos LUT ────────────
    //
    // The sacred scalar `rope()` recomputes `cosf(theta)+sinf(theta)`
    // for each (head, half-index) pair. All 24 heads at one token
    // position share the SAME (cos, sin) values, so the sacred path
    // wastes 23/24 of its trig work per layer. We pre-compute the
    // LUT ONCE per token (in the caller) and apply it here in
    // AVX-512.
    //
    // Per ADR-029 v3: ≤ 1 ULP feature-mode drift vs sacred scalar
    // `cosf/sinf`. Token-ID 25 HARD GATE preserved.
    let _ = token_offset; // LUT already encodes position
    for h in 0..n_q_heads {
        let offset = h * head_dim;
        rope_apply_avx512(
            &mut q_out[offset..offset + head_dim],
            rope_cos_lut,
            rope_sin_lut,
            HALF,
        );
    }
    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        rope_apply_avx512(
            &mut k_out[offset..offset + head_dim],
            rope_cos_lut,
            rope_sin_lut,
            HALF,
        );
    }

    trace_stats(layer_idx, token_offset, "q_rope", &q_out[..q_dim]);
    trace_stats(layer_idx, token_offset, "k_rope", &k_out[..kv_dim]);

    // ── Step 4: KV-cache write — sacred public op ────────────────
    kv_cache.store_kv(layer_idx, token_offset, k_out, v_buf)?;

    // ── Step 5: Scaled Dot-Product Attention (GQA) ───────────────
    let k_full = kv_cache.get_k_slice(layer_idx, total_tokens)?;
    let v_full = kv_cache.get_v_slice(layer_idx, total_tokens)?;

    // Parallel per-head path (S1): fan the n_q_heads independent
    // head computations out across cores — see the ATTN_TABLE
    // section comment for the bit-exactness argument and the
    // engagement gate. Requires one score stripe per head; the
    // caller allocates score_buf as n_q_heads * max_tokens.
    let score_stride = score_buf.len() / n_q_heads.max(1);
    if total_tokens >= attn_parallel_min_tokens() && n_q_heads >= 2 && score_stride >= total_tokens
    {
        let ctx = ParallelMatmulContext::for_work_units(n_q_heads);
        if ctx.n_cores > 1 {
            // Publish the dispatch descriptor (plain stores; released
            // by dispatch_matmul's WORK_EPOCH publish).
            *ATTN_TABLE.q_ptr.get() = q_out.as_ptr();
            *ATTN_TABLE.k_ptr.get() = k_full.as_ptr();
            *ATTN_TABLE.v_ptr.get() = v_full.as_ptr();
            *ATTN_TABLE.score_ptr.get() = score_buf.as_mut_ptr();
            *ATTN_TABLE.score_stride.get() = score_stride;
            *ATTN_TABLE.concat_ptr.get() = attn_concat.as_mut_ptr();
            *ATTN_TABLE.total_tokens.get() = total_tokens;
            *ATTN_TABLE.kv_dim.get() = kv_dim;
            *ATTN_TABLE.head_dim.get() = head_dim;
            *ATTN_TABLE.gqa_ratio.get() = gqa_ratio;
            *ATTN_TABLE.scale.get() = scale;
            ATTN_TABLE.error.store(0, Ordering::Relaxed);

            let args = MatmulArgs {
                x_ptr: core::ptr::null(),
                w_ptr: core::ptr::null(),
                out_ptr: core::ptr::null_mut(),
                in_dim: head_dim,
                out_dim: n_q_heads,
            };
            if !ATTN_PAR_DISPATCH_LOGGED.load(Ordering::Acquire)
                && !ATTN_PAR_DISPATCH_LOGGED.swap(true, Ordering::AcqRel)
            {
                let _ = writeln!(
                    crate::arch::serial::Serial,
                    "[MP3.0] AVX-512 parallel attention dispatch: heads={} cores={} total_tokens={} min_tokens={}",
                    n_q_heads,
                    ctx.n_cores,
                    total_tokens,
                    attn_parallel_min_tokens(),
                );
            }
            let t0 = crate::arch::read_cycles();
            ctx.dispatch_matmul(attn_heads_kernel, args, n_q_heads);
            let dt = crate::arch::read_cycles().wrapping_sub(t0);
            ATTN_PAR_CALLS.fetch_add(1, Ordering::Relaxed);
            ATTN_PAR_CYCLES.fetch_add(dt, Ordering::Relaxed);

            if ATTN_TABLE.error.load(Ordering::Relaxed) != 0 {
                // Same error class the sequential loop propagates
                // from `softmax`; generation aborts identically.
                return Err(AttentionError::NumericalInstability);
            }

            // ── Step 6: O projection ─────────────────────────────
            let _ = linear_dispatch_avx512(
                attn_concat,
                o_weight,
                output,
                q_dim,
                embedding_dim,
                o_quant,
            );
            return Ok(());
        }
    }

    for q_h in 0..n_q_heads {
        let kv_h = q_h / gqa_ratio;
        let q_offset = q_h * head_dim;
        let q_vec = &q_out[q_offset..q_offset + head_dim];

        // Score computation: Q × Kᵀ — single-core AVX-512 dot.
        for t in 0..total_tokens {
            let k_offset = t * kv_dim + kv_h * head_dim;
            let k_vec = &k_full[k_offset..k_offset + head_dim];
            score_buf[t] = dot_product_f32_avx512(q_vec, k_vec, head_dim) * scale;
        }

        // Trace head-0 scores (and any head whose scores went
        // non-finite) BEFORE softmax — this is the exact value that
        // trips NumericalInstability. Cheap: first token only.
        if token_offset == 0 && avx512_trace_enabled() {
            let (mn, mx, mean, nan, inf) = vec_stats(&score_buf[..total_tokens]);
            if nan > 0 || inf > 0 {
                // stage label encodes the head so the bench screen
                // pins which head's score went non-finite.
                record_first_nonfinite(layer_idx, "score", nan, inf);
            }
            if q_h == 0 || nan > 0 || inf > 0 {
                let _ = writeln!(
                    crate::arch::serial::Serial,
                    "[TRACE] L{:02} score[h{:02}]   t={:<5} min={:+.3e} max={:+.3e} mean={:+.3e} nan={} inf={} scale={:+.3e}",
                    layer_idx, q_h, total_tokens, mn, mx, mean, nan, inf, scale,
                );
            }
        }

        // Softmax — sacred scalar. On failure, name the exact head so
        // the KVM trace pins layer+head+score-range of the instability.
        if let Err(e) = softmax(&mut score_buf[..total_tokens], total_tokens) {
            let _ = writeln!(
                crate::arch::serial::Serial,
                "[TRACE] L{:02} softmax FAILED at head {} (total_tokens={}): {:?}",
                layer_idx, q_h, total_tokens, e,
            );
            return Err(e);
        }

        // Weighted-sum: attn × V — single-core AVX-512.
        for i in 0..head_dim {
            attn_head_buf[i] = 0.0;
        }
        for t in 0..total_tokens {
            let v_offset = t * kv_dim + kv_h * head_dim;
            let v_vec = &v_full[v_offset..v_offset + head_dim];
            let weight = score_buf[t];
            weighted_add_f32_avx512(weight, v_vec, attn_head_buf, head_dim);
        }

        // Copy to concatenated output.
        let out_offset = q_h * head_dim;
        attn_concat[out_offset..out_offset + head_dim].copy_from_slice(&attn_head_buf[..head_dim]);
    }

    trace_stats(layer_idx, token_offset, "attn_concat", &attn_concat[..q_dim]);

    // ── Step 6: O projection — parallel AVX-512, quant-aware ─────
    let _ = linear_dispatch_avx512(attn_concat, o_weight, output, q_dim, embedding_dim, o_quant);

    trace_stats(layer_idx, token_offset, "o_proj", &output[..embedding_dim]);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// Full single-token forward pass (mirror of inference_neon)
// ─────────────────────────────────────────────────────────────────

/// AVX-512-accelerated single-token forward pass through all 28 layers.
/// Replicates sacred `forward_single_token` using AVX-512 for ALL
/// matmul-heavy steps (attention Q/K/V/O + score + weighted-sum + MLP
/// gate/up/down + lm_head). Sacred scalar ops preserved for rmsnorm,
/// rope, softmax, SiLU, embed_lookup, KV-cache write/read.
///
/// # Safety
/// Calls unsafe AVX-512 intrinsics transitively. AVX-512F must be
/// available; caller verifies via CPUID at boot.
#[allow(clippy::too_many_arguments)]
pub unsafe fn forward_single_token_avx512<const HALF: usize>(
    token_id: u32,
    token_offset: usize,
    layers: &[LayerWeights; N_LAYERS],
    token_embd: &[u8],
    dispatch: &ForwardPassDispatch,
    rope_ctx: &RopeContext<HALF>,
    rms_eps: f32,
    embedding_dim: usize,
    intermediate_dim: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_cache: &mut KvCache,
    // Scratch buffers (same signature as sacred).
    hidden: &mut [f32],
    norm_buf: &mut [f32],
    attn_out: &mut [f32],
    mlp_out: &mut [f32],
    q_buf: &mut [f32],
    k_buf: &mut [f32],
    v_buf: &mut [f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    score_buf: &mut [f32],
    attn_head_buf: &mut [f32],
    attn_concat: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    hidden_mlp_buf: &mut [f32],
    _scratch: &mut LinearScratch,
) -> Result<(), ForwardPassError> {
    // ── Embedding lookup — single row dequant, native quant-aware ─
    embed_lookup_dispatch(
        token_id,
        token_embd,
        embedding_dim,
        hidden,
        dispatch.token_embd_quant,
    )
    .map_err(ForwardPassError::Embedding)?;
    trace_stats(0, token_offset, "embed", &hidden[..embedding_dim]);

    // ── RoPE sincos LUT (computed ONCE per token, shared by all
    //    24 attention heads × 28 layers) ────────────────────────────
    //
    // HALF is the half-head dimension (Qwen3 head_dim=128 → HALF=64).
    // The two 256-byte LUTs fit comfortably on the stack and stay
    // hot in L1 for the entire 28-layer chain.
    //
    // Sacred `rope()` recomputes 64 (cos, sin) pairs PER HEAD per
    // layer (24 × 28 = 672 redundant recomputations per token); we
    // do it once in AVX-512.
    let mut rope_cos_lut = [0.0_f32; 64];
    let mut rope_sin_lut = [0.0_f32; 64];
    debug_assert!(HALF <= 64, "RoPE LUT sized for HALF<=64 (Qwen3=64)");
    rope_sincos_lut_avx512(
        token_offset,
        &rope_ctx.inv_freqs[..HALF],
        &mut rope_cos_lut[..HALF],
        &mut rope_sin_lut[..HALF],
        HALF,
    );

    // ── 28-layer chain ────────────────────────────────────────────
    for layer_idx in 0..N_LAYERS {
        let layer = &layers[layer_idx];

        // ─ Attention sub-block ─
        rmsnorm(hidden, layer.attn_norm, norm_buf, rms_eps);
        finite_diag(norm_buf, layer_idx, "attn_norm_in");
        trace_stats(layer_idx, token_offset, "attn_norm", &norm_buf[..embedding_dim]);

        if let Err(e) = gqa_attention_single_token_avx512::<HALF>(
            norm_buf,
            layer.attn_q,
            layer.attn_k,
            layer.attn_v,
            layer.attn_o,
            dispatch.attn_q_quant[layer_idx],
            dispatch.attn_k_quant[layer_idx],
            dispatch.attn_v_tensor_quant[layer_idx],
            dispatch.attn_o_quant[layer_idx],
            layer.attn_q_norm,
            layer.attn_k_norm,
            layer_idx,
            token_offset,
            rope_ctx,
            &rope_cos_lut[..HALF],
            &rope_sin_lut[..HALF],
            rms_eps,
            n_q_heads,
            n_kv_heads,
            head_dim,
            embedding_dim,
            kv_cache,
            q_buf,
            k_buf,
            v_buf,
            q_out,
            k_out,
            score_buf,
            attn_head_buf,
            attn_concat,
            attn_out,
        ) {
            let _ = writeln!(
                crate::arch::serial::Serial,
                "[MP3.0] Attention failed at layer {} (token_offset={}): {:?}",
                layer_idx,
                token_offset,
                e,
            );
            return Err(ForwardPassError::Attention(e));
        }

        // Residual: hidden += attn_out.
        for i in 0..embedding_dim {
            hidden[i] += attn_out[i];
        }
        finite_diag(attn_out, layer_idx, "attn_out");
        finite_diag(hidden, layer_idx, "post_attn_residual");
        trace_stats(layer_idx, token_offset, "post_attn", &hidden[..embedding_dim]);

        // ─ MLP sub-block ─
        rmsnorm(hidden, layer.ffn_norm, norm_buf, rms_eps);
        trace_stats(layer_idx, token_offset, "ffn_norm", &norm_buf[..embedding_dim]);

        mlp_swiglu_avx512(
            norm_buf,
            layer.ffn_gate,
            layer.ffn_up,
            layer.ffn_down,
            dispatch.ffn_gate_quant[layer_idx],
            dispatch.ffn_up_quant[layer_idx],
            dispatch.ffn_down_tensor_quant[layer_idx],
            embedding_dim,
            intermediate_dim,
            gate_buf,
            up_buf,
            hidden_mlp_buf,
            mlp_out,
        );
        // gate/up/silu scratch retain their values after the call —
        // trace them to localise an MLP-side instability (e.g. the
        // SiLU multiply-order overflow path).
        trace_stats(layer_idx, token_offset, "mlp_gate", &gate_buf[..intermediate_dim]);
        trace_stats(layer_idx, token_offset, "mlp_up", &up_buf[..intermediate_dim]);
        trace_stats(layer_idx, token_offset, "mlp_silu_up", &hidden_mlp_buf[..intermediate_dim]);
        trace_stats(layer_idx, token_offset, "mlp_out", &mlp_out[..embedding_dim]);

        // Residual: hidden += mlp_out.
        for i in 0..embedding_dim {
            hidden[i] += mlp_out[i];
        }
        finite_diag(mlp_out, layer_idx, "mlp_out");
        finite_diag(hidden, layer_idx, "post_mlp_residual");
        trace_stats(layer_idx, token_offset, "post_mlp", &hidden[..embedding_dim]);
    }

    Ok(())
}

/// One-shot SMP dispatch self-test (MP2.6).
///
/// Runs one real Q4_0 projection twice — once through the parallel
/// dispatcher with all active cores, once single-threaded on the BSP —
/// and compares the outputs bitwise. Row ownership is the ONLY thing
/// the dispatcher changes (per-row K-order is identical), so any
/// difference proves a dispatch-layer defect (epoch publication,
/// barrier, row-range math) — exactly the class of bug only real
/// multi-core silicon can surface. The caller caps matmuls to
/// single-core on failure: slow, but bit-exact by construction, and
/// the box stays alive and measurable instead of producing garbage.
///
/// `scratch` must hold at least `in_dim + 2 * out_dim` f32s.
///
/// # Safety
/// AVX-512F must be available. `w` must hold `out_dim` rows of Q4_0
/// bytes (plain, or x4-interleaved and registered in `weight_layout`)
/// at `in_dim` columns.
pub unsafe fn smp_dispatch_self_test(
    w: &[u8],
    in_dim: usize,
    out_dim: usize,
    scratch: &mut [f32],
) -> bool {
    let (x, rest) = scratch.split_at_mut(in_dim);
    let (out_par, rest) = rest.split_at_mut(out_dim);
    let out_single = &mut rest[..out_dim];

    // Deterministic sign-alternating ramp — keeps every accumulator in
    // a healthy fp32 range for any sane weight content.
    let mut i = 0;
    while i < in_dim {
        x[i] = ((i % 31) as f32 - 15.0) * 0.0625;
        i += 1;
    }

    // Parallel path: the exact dispatch the forward pass uses.
    linear_q4_0_dispatch(x, w, out_par, in_dim, out_dim);

    // Reference: same kernel, full range, BSP only — no dispatcher.
    let interleave = crate::weight_layout::group_of(w.as_ptr() as usize);
    let args = MatmulArgs {
        x_ptr: x.as_ptr(),
        w_ptr: w.as_ptr(),
        out_ptr: out_single.as_mut_ptr(),
        in_dim,
        out_dim,
    };
    let full = RowRange {
        start: 0,
        end: out_dim,
    };
    if interleave > 1 {
        q4_0x4_kernel(&args, full);
    } else {
        q4_0_kernel(&args, full);
    }

    let mut i = 0;
    while i < out_dim {
        if out_par[i].to_bits() != out_single[i].to_bits() {
            let _ = writeln!(
                crate::arch::serial::Serial,
                "[MP2.6] SMP self-test MISMATCH at row {}: parallel=0x{:08x} single=0x{:08x}",
                i,
                out_par[i].to_bits(),
                out_single[i].to_bits()
            );
            return false;
        }
        i += 1;
    }
    true
}

// ─────────────────────────────────────────────────────────────────
// Single-thread helpers (kept for diagnostics + non-SMP smoke tests)
// ─────────────────────────────────────────────────────────────────

/// Single-threaded Q4_K matmul. Used for verification (compare against
/// the parallel-dispatched variant) and as a fallback when SMP has not
/// yet been initialized.
///
/// # Safety
/// AVX-512F must be available.
#[inline]
#[allow(dead_code)]
pub unsafe fn linear_q4k_single(
    x: &[f32],
    w: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    linear_q4k_avx512(x, w, out, in_dim, out_dim);
}

/// Single-threaded Q6_K matmul. Used for verification.
/// # Safety
/// AVX-512F must be available.
#[inline]
#[allow(dead_code)]
pub unsafe fn linear_q6k_single(
    x: &[f32],
    w: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    linear_q6k_avx512(x, w, out, in_dim, out_dim);
}

// ─────────────────────────────────────────────────────────────────
// DeepSeek-V2/V3 / Kimi K2.6 — MoE + MLA AVX-512 dispatch (MP-MoE-3)
// ─────────────────────────────────────────────────────────────────
//
// Native AVX-512 implementations. The matmul hot path (MoE expert
// gate/up/down + MLA kv_a/kv_b/q_a/q_b/output) routes through
// `linear_dispatch_avx512`, which in turn fans rows across the active
// SMP cores via `ParallelMatmulContext::dispatch_matmul`. The router
// (gate_inp, F32 weights), softmax, RoPE, and SiLU remain scalar —
// each is either too small to amortise the SMP barrier overhead
// (router/softmax/RoPE) or already AVX-512-fused at the activation
// layer (silu_mul_avx512).
//
// **Bit-exactness vs scalar:** NOT guaranteed for DeepSeek2 / Kimi
// K2.6. There is no β-anchor for this model family; the sacred chain
// is anchored exclusively against Qwen3-1.7B (token=25,
// logit_bits=0x414a6497), which never enters this path. The drift
// here is the accumulated FMA-ordering difference between the scalar
// kahan-style accumulation in `zero_llm_inference::ops::linear_*`
// and the AVX-512 horizontal-reduce kernels.

use zero_llm_inference::{
    mha_attention_single_token,
    moe::{
        expert_q4k_bytes, expert_quant_bytes, moe_route_f32, slice_expert_weight, MoeRoutingMode,
    },
    AttnType, Deepseek2LayerWeights, Deepseek2Scratch as LibScratch, MhaKvCache, MhaWeights,
    MlaError, MlaKvCache, MlaWeights, MlpType,
};

/// AVX-512-accelerated single-expert SwiGLU MLP — gate + up + down via
/// `linear_dispatch_avx512` (SMP row-split), fused SiLU(gate) × up via
/// `silu_mul_avx512`.
///
/// Bit-exact equivalent of `zero_llm_inference::moe::expert_swiglu`
/// in algorithm shape, but not in output bits (see module docs).
///
/// # Safety
/// AVX-512F must be available on every participating core.
#[allow(clippy::too_many_arguments)]
unsafe fn expert_swiglu_avx512(
    input: &[f32],
    gate_w: &[u8],
    up_w: &[u8],
    down_w: &[u8],
    embedding_dim: usize,
    expert_intermediate: usize,
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    hidden_buf: &mut [f32],
    output: &mut [f32],
    expert_quant: zero_gguf_parser::GgmlType,
) {
    // Gate projection.
    let _ = linear_dispatch_avx512(
        input,
        gate_w,
        gate_buf,
        embedding_dim,
        expert_intermediate,
        expert_quant,
    );

    // Up projection.
    let _ = linear_dispatch_avx512(
        input,
        up_w,
        up_buf,
        embedding_dim,
        expert_intermediate,
        expert_quant,
    );

    // SiLU(gate) × up → hidden — AVX-512-fused 16-lane pass.
    silu_mul_avx512(gate_buf, up_buf, hidden_buf, expert_intermediate);

    // Down projection.
    let _ = linear_dispatch_avx512(
        hidden_buf,
        down_w,
        output,
        expert_intermediate,
        embedding_dim,
        expert_quant,
    );
}

/// AVX-512 MoE FFN (Kimi K2.6). Router + shared expert + top-K
/// experts + weighted sum. Router stays scalar (F32 weights, small
/// matmul); per-expert SwiGLU runs through the AVX-512 hot path.
///
/// # Safety
/// AVX-512F must be available.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_ffn_avx512(
    input: &[f32],
    router_weight_f32: &[f32],
    router_bias_f32: &[f32],
    gate_exps: &[u8],
    up_exps: &[u8],
    down_exps: &[u8],
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    n_experts: usize,
    top_k: usize,
    embedding_dim: usize,
    expert_intermediate: usize,
    router_score_buf: &mut [f32],
    expert_indices: &mut [u32],
    expert_weights: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    hidden_buf: &mut [f32],
    expert_out_buf: &mut [f32],
    output: &mut [f32],
    _scratch: &mut LinearScratch,
    expert_quant: zero_gguf_parser::GgmlType,
    routing_mode: MoeRoutingMode,
    expert_weight_scale: f32,
) {
    assert!(top_k > 0, "moe_ffn_avx512: top_k must be > 0");
    assert!(
        top_k <= n_experts,
        "moe_ffn_avx512: top_k must be <= n_experts"
    );

    // Step 1: Route — scalar (F32 router, top_k partial sort, per-mode normalise).
    moe_route_f32(
        input,
        router_weight_f32,
        router_bias_f32,
        n_experts,
        top_k,
        router_score_buf,
        expert_indices,
        expert_weights,
        routing_mode,
        expert_weight_scale,
    );

    // Step 2: Shared expert (always executed) — writes directly to `output`.
    expert_swiglu_avx512(
        input,
        shared_gate,
        shared_up,
        shared_down,
        embedding_dim,
        expert_intermediate,
        gate_buf,
        up_buf,
        hidden_buf,
        output,
        expert_quant,
    );

    // Step 3: Top-K experts — weighted accumulation into `output`.
    let gate_expert_bytes = expert_quant_bytes(embedding_dim, expert_intermediate, expert_quant)
        .unwrap_or_else(|| expert_q4k_bytes(embedding_dim, expert_intermediate));
    let up_expert_bytes = gate_expert_bytes;
    let down_expert_bytes = expert_quant_bytes(expert_intermediate, embedding_dim, expert_quant)
        .unwrap_or_else(|| expert_q4k_bytes(expert_intermediate, embedding_dim));

    for k in 0..top_k {
        let eidx = expert_indices[k] as usize;
        let weight = expert_weights[k];

        let gate_w = slice_expert_weight(gate_exps, eidx, gate_expert_bytes);
        let up_w = slice_expert_weight(up_exps, eidx, up_expert_bytes);
        let down_w = slice_expert_weight(down_exps, eidx, down_expert_bytes);

        expert_swiglu_avx512(
            input,
            gate_w,
            up_w,
            down_w,
            embedding_dim,
            expert_intermediate,
            gate_buf,
            up_buf,
            hidden_buf,
            expert_out_buf,
            expert_quant,
        );

        // Weighted accumulation: output += weight * expert_out_buf.
        // weighted_add_f32_avx512 handles the 16-lane FMA tail loop.
        weighted_add_f32_avx512(weight, expert_out_buf, output, embedding_dim);
    }
}

/// AVX-512 MLA attention single-token. Replaces the 5 scalar
/// `linear_dispatch` calls (kv_a, kv_b, q_a, q_b, output) with
/// `linear_dispatch_avx512`; RMSNorm / RoPE / softmax / per-head
/// score+weighted-sum stay scalar (each operates on small
/// per-head buffers where the SMP barrier would dominate).
///
/// # Safety
/// AVX-512F must be available.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mla_attention_avx512(
    hidden: &[f32],
    weights: &MlaWeights,
    layer_idx: usize,
    token_offset: usize,
    n_heads: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    q_lora_rank: usize,
    embedding_dim: usize,
    rms_eps: f32,
    rope_freq_base: f32,
    kv_cache: &mut MlaKvCache,
    c_kv_rope_buf: &mut [f32],
    c_kv_norm_buf: &mut [f32],
    kv_decompressed: &mut [f32],
    c_q_buf: &mut [f32],
    c_q_norm_buf: &mut [f32],
    q_decompressed: &mut [f32],
    k_assembled: &mut [f32],
    score_buf: &mut [f32],
    attn_head_buf: &mut [f32],
    attn_concat: &mut [f32],
    output: &mut [f32],
    _scratch: &mut LinearScratch,
    attn_quant: zero_gguf_parser::GgmlType,
) -> Result<(), MlaError> {
    let kv_a_out_dim = kv_lora_rank + qk_rope_head_dim;
    let kv_b_out_dim = n_heads * (qk_nope_head_dim + v_head_dim);
    let q_b_out_dim = n_heads * (qk_nope_head_dim + qk_rope_head_dim);
    let total_k_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let total_tokens = token_offset + 1;
    let nope_v_per_head = qk_nope_head_dim + v_head_dim;
    // Kimi K2.6 ships split attn_k_b / attn_v_b in place of attn_kv_b.
    // Mirror the scalar mla.rs dispatch: two matmuls into head-major
    // halves of `kv_decompressed`, then index per-head differently.
    let use_split_kv_b = !weights.k_b.is_empty() && !weights.v_b.is_empty();
    let k_nope_out_dim = n_heads * qk_nope_head_dim;
    let v_out_dim = n_heads * v_head_dim;
    debug_assert!(
        score_buf.len() >= n_heads * total_tokens,
        "mla_attention_avx512: score_buf must be sized n_heads × max_tokens"
    );
    let score_stride = score_buf.len() / n_heads;
    let _ = k_assembled; // unused — compressed-cache path
    let _ = attn_head_buf; // unused — Pass B writes directly into attn_concat

    // ── Step 1: KV compression — AVX-512 matmul ─────────────────────
    let _ = linear_dispatch_avx512(
        hidden,
        weights.kv_a_mqa,
        c_kv_rope_buf,
        embedding_dim,
        kv_a_out_dim,
        attn_quant,
    );

    // ── Step 2: Normalize c_kv — scalar (kv_lora_rank ≈ 512) ────────
    {
        let (c_kv_raw, _) = c_kv_rope_buf.split_at(kv_lora_rank);
        rmsnorm(c_kv_raw, weights.kv_a_norm, c_kv_norm_buf, rms_eps);
    }

    // ── Step 3: Normalize + RoPE shared k_rope in-place ─────────────
    let half_rope = qk_rope_head_dim / 2;
    {
        let k_rope_normed = &mut c_kv_rope_buf[kv_lora_rank..kv_a_out_dim];
        if weights.k_norm.len() == qk_rope_head_dim {
            let mut ssq = 0.0f32;
            for i in 0..qk_rope_head_dim {
                ssq += k_rope_normed[i] * k_rope_normed[i];
            }
            let rms = libm::sqrtf(ssq / qk_rope_head_dim as f32 + rms_eps);
            let rms_inv = 1.0 / rms;
            for i in 0..qk_rope_head_dim {
                k_rope_normed[i] = k_rope_normed[i] * rms_inv * weights.k_norm[i];
            }
        }
        for i in 0..half_rope {
            let freq = libm::powf(
                rope_freq_base,
                -2.0 * (i as f32) / (qk_rope_head_dim as f32),
            );
            let theta = token_offset as f32 * freq;
            let cos_t = libm::cosf(theta);
            let sin_t = libm::sinf(theta);
            let x0 = k_rope_normed[i];
            let x1 = k_rope_normed[i + half_rope];
            k_rope_normed[i] = x0 * cos_t - x1 * sin_t;
            k_rope_normed[i + half_rope] = x0 * sin_t + x1 * cos_t;
        }
    }

    // ── Step 4: Cache compressed latent (c_kv_normed + k_rope) ──────
    kv_cache.store_compressed(
        layer_idx,
        token_offset,
        c_kv_norm_buf,
        &c_kv_rope_buf[kv_lora_rank..kv_a_out_dim],
    )?;

    // ── Step 5: Q compression — AVX-512 matmul ──────────────────────
    let _ = linear_dispatch_avx512(
        hidden,
        weights.q_a,
        c_q_buf,
        embedding_dim,
        q_lora_rank,
        attn_quant,
    );

    // ── Step 6: Normalize Q — scalar (q_lora_rank ≈ 1536) ───────────
    rmsnorm(c_q_buf, weights.q_a_norm, c_q_norm_buf, rms_eps);

    // ── Step 7: Decompress Q — AVX-512 matmul ───────────────────────
    let _ = linear_dispatch_avx512(
        c_q_norm_buf,
        weights.q_b,
        q_decompressed,
        q_lora_rank,
        q_b_out_dim,
        attn_quant,
    );

    // ── Step 8: RoPE on q_rope per head — scalar ────────────────────
    for h in 0..n_heads {
        let q_offset = h * total_k_head_dim + qk_nope_head_dim;
        for i in 0..half_rope {
            let freq = libm::powf(
                rope_freq_base,
                -2.0 * (i as f32) / (qk_rope_head_dim as f32),
            );
            let theta = token_offset as f32 * freq;
            let cos_t = libm::cosf(theta);
            let sin_t = libm::sinf(theta);
            let x0 = q_decompressed[q_offset + i];
            let x1 = q_decompressed[q_offset + i + half_rope];
            q_decompressed[q_offset + i] = x0 * cos_t - x1 * sin_t;
            q_decompressed[q_offset + i + half_rope] = x0 * sin_t + x1 * cos_t;
        }
    }

    // ── Pass A: Score computation across all heads ──────────────────
    // Expand each cached token's c_kv once through W_kv_b (AVX-512
    // matmul, SMP row-split). Per-token reuse of `kv_decompressed`
    // amortises the kv_b matmul across n_heads.
    let scale = 1.0 / libm::sqrtf(total_k_head_dim as f32);

    for t in 0..total_tokens {
        let c_kv_t = kv_cache.get_c_kv(layer_idx, t)?;
        let k_rope_t = kv_cache.get_k_rope(layer_idx, t)?;

        if use_split_kv_b {
            let (k_part, v_part) = kv_decompressed.split_at_mut(k_nope_out_dim);
            let _ = linear_dispatch_avx512(
                c_kv_t,
                weights.k_b,
                k_part,
                kv_lora_rank,
                k_nope_out_dim,
                attn_quant,
            );
            let _ = linear_dispatch_avx512(
                c_kv_t,
                weights.v_b,
                &mut v_part[..v_out_dim],
                kv_lora_rank,
                v_out_dim,
                attn_quant,
            );
        } else {
            let _ = linear_dispatch_avx512(
                c_kv_t,
                weights.kv_b,
                kv_decompressed,
                kv_lora_rank,
                kv_b_out_dim,
                attn_quant,
            );
        }

        for h in 0..n_heads {
            let q_offset = h * total_k_head_dim;
            let q_nope = &q_decompressed[q_offset..q_offset + qk_nope_head_dim];
            let q_rope = &q_decompressed[q_offset + qk_nope_head_dim..q_offset + total_k_head_dim];
            let k_nope_h_t = if use_split_kv_b {
                let off = h * qk_nope_head_dim;
                &kv_decompressed[off..off + qk_nope_head_dim]
            } else {
                let kv_off = h * nope_v_per_head;
                &kv_decompressed[kv_off..kv_off + qk_nope_head_dim]
            };

            let dot_nope = dot_product_f32_avx512(q_nope, k_nope_h_t, qk_nope_head_dim);
            let dot_rope = dot_product_f32_avx512(q_rope, k_rope_t, qk_rope_head_dim);
            score_buf[h * score_stride + t] = (dot_nope + dot_rope) * scale;
        }
    }

    // ── Pass A.5: Per-head softmax ──────────────────────────────────
    for h in 0..n_heads {
        let off = h * score_stride;
        softmax(&mut score_buf[off..off + total_tokens], total_tokens)
            .map_err(|_| MlaError::NumericalInstability)?;
    }

    // ── Pass B: Weighted V accumulation ─────────────────────────────
    let attn_concat_len = n_heads * v_head_dim;
    for v in attn_concat[..attn_concat_len].iter_mut() {
        *v = 0.0;
    }

    for t in 0..total_tokens {
        let c_kv_t = kv_cache.get_c_kv(layer_idx, t)?;
        if use_split_kv_b {
            let (k_part, v_part) = kv_decompressed.split_at_mut(k_nope_out_dim);
            let _ = linear_dispatch_avx512(
                c_kv_t,
                weights.k_b,
                k_part,
                kv_lora_rank,
                k_nope_out_dim,
                attn_quant,
            );
            let _ = linear_dispatch_avx512(
                c_kv_t,
                weights.v_b,
                &mut v_part[..v_out_dim],
                kv_lora_rank,
                v_out_dim,
                attn_quant,
            );
        } else {
            let _ = linear_dispatch_avx512(
                c_kv_t,
                weights.kv_b,
                kv_decompressed,
                kv_lora_rank,
                kv_b_out_dim,
                attn_quant,
            );
        }

        for h in 0..n_heads {
            let v_h_t = if use_split_kv_b {
                let off = k_nope_out_dim + h * v_head_dim;
                &kv_decompressed[off..off + v_head_dim]
            } else {
                let kv_off = h * nope_v_per_head;
                &kv_decompressed[kv_off + qk_nope_head_dim..kv_off + nope_v_per_head]
            };
            let weight = score_buf[h * score_stride + t];
            let out_off = h * v_head_dim;
            weighted_add_f32_avx512(
                weight,
                v_h_t,
                &mut attn_concat[out_off..out_off + v_head_dim],
                v_head_dim,
            );
        }
    }

    // ── Output projection — AVX-512 matmul ──────────────────────────
    let o_in_dim = n_heads * v_head_dim;
    let _ = linear_dispatch_avx512(
        &attn_concat[..o_in_dim],
        weights.output,
        output,
        o_in_dim,
        embedding_dim,
        attn_quant,
    );

    Ok(())
}

/// DeepSeek2 forward-pass error (mirrors `ForwardPassError` for the MLA path).
#[derive(Debug)]
pub enum Deepseek2ForwardError {
    Mla(MlaError),
}

impl From<MlaError> for Deepseek2ForwardError {
    fn from(e: MlaError) -> Self {
        Deepseek2ForwardError::Mla(e)
    }
}

// `Deepseek2Scratch` shape is centralized in the library
// (`zero_llm_inference::Deepseek2Scratch`). The kernel's AVX-512
// variant operates on the same layout — re-exported as `LibScratch`.

/// 61-layer single-token forward pass for Kimi K2.6 / DeepSeek-V2/V3.
///
/// Per-layer flow:
///   1. attn_norm
///   2. MLA attention (single-token, populates MlaKvCache)
///   3. residual: hidden += attn_out
///   4. ffn_norm
///   5. Dense SwiGLU OR MoE FFN (per `MlpType`)
///   6. residual: hidden += mlp_out
///
/// # Safety
/// Caller asserts AVX-512F. All scratch buffers in `s` must be sized for
/// the dimensions implied by the supplied configuration. `layers.len()`
/// is the runtime layer count (no compile-time N_LAYERS dependency on
/// this path).
#[allow(clippy::too_many_arguments)]
pub unsafe fn forward_single_token_deepseek2_avx512(
    token_id: u32,
    token_offset: usize,
    layers: &[Deepseek2LayerWeights],
    token_embd: &[u8],
    embed_quant: zero_gguf_parser::GgmlType,
    rms_eps: f32,
    rope_freq_base: f32,
    embedding_dim: usize,
    intermediate_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_experts: usize,
    top_k: usize,
    expert_intermediate: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    q_lora_rank: usize,
    // SPEC: DeepSeek-V3/Kimi K2.6 → MoeRoutingMode::SigmoidNormalize.
    routing_mode: MoeRoutingMode,
    expert_weight_scale: f32,
    kv_cache: &mut MlaKvCache,
    mha_kv_cache: Option<&mut MhaKvCache>,
    s: &mut LibScratch<'_>,
) -> Result<(), Deepseek2ForwardError> {
    // Per-quant embedding lookup. Kimi K2.6 builds `token_embd.weight` at
    // Q8_0; the legacy `embed_lookup` hardcodes Q4_K layout and would
    // misread the row stride. The scalar deepseek2 path uses the
    // dispatcher; this path must too. (Qwen3 still calls `embed_lookup`
    // directly from `forward_single_token_avx512` so its β-anchor stays
    // bit-exact.)
    let _ = embed_lookup_dispatch(token_id, token_embd, embedding_dim, s.hidden, embed_quant);

    let mut mha_kv_opt = mha_kv_cache;
    for (layer_idx, layer) in layers.iter().enumerate() {
        // ── Attention sub-block ──────────────────────────────────
        rmsnorm(s.hidden, layer.attn_norm, s.norm_buf, rms_eps);

        match layer.attn_type {
            AttnType::Mla => {
                let mla_weights = MlaWeights {
                    kv_a_mqa: layer.attn_kv_a_mqa,
                    kv_a_norm: layer.attn_kv_a_norm,
                    kv_b: layer.attn_kv_b,
                    k_b: layer.attn_k_b,
                    v_b: layer.attn_v_b,
                    q_a: layer.attn_q_a,
                    q_a_norm: layer.attn_q_a_norm,
                    q_b: layer.attn_q_b,
                    output: layer.attn_output,
                    k_norm: layer.attn_k_norm,
                };

                mla_attention_avx512(
                    s.norm_buf,
                    &mla_weights,
                    layer_idx,
                    token_offset,
                    n_heads,
                    kv_lora_rank,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    v_head_dim,
                    q_lora_rank,
                    embedding_dim,
                    rms_eps,
                    rope_freq_base,
                    kv_cache,
                    s.c_kv_rope_buf,
                    s.c_kv_norm_buf,
                    s.kv_decompressed,
                    s.c_q_buf,
                    s.c_q_norm_buf,
                    s.q_decompressed,
                    s.k_assembled,
                    s.score_buf,
                    s.attn_head_buf,
                    s.attn_concat,
                    s.attn_out,
                    s.scratch,
                    layer.attn_quant,
                )?;
            }
            AttnType::Mha => {
                // MHA layers fall through to the scalar reference path —
                // this happens at most once or twice per forward pass
                // (typically layer 0 only for Kimi K2.6 builds), so an
                // AVX-512-specific MHA kernel is not worth duplicating.
                let mha_cache = mha_kv_opt
                    .as_deref_mut()
                    .ok_or(Deepseek2ForwardError::Mla(MlaError::NumericalInstability))?;
                let mha_weights = MhaWeights {
                    q: layer.attn_q_mha,
                    k: layer.attn_k_mha,
                    v: layer.attn_v_mha,
                    output: layer.attn_output,
                };
                let head_dim_qk = qk_nope_head_dim + qk_rope_head_dim;
                let q_dim = n_heads * head_dim_qk;
                let k_dim = n_kv_heads * head_dim_qk;
                let v_dim = n_kv_heads * v_head_dim;
                let (k_part, rest) = s.kv_decompressed.split_at_mut(k_dim);
                let v_part = &mut rest[..v_dim];
                mha_attention_single_token(
                    s.norm_buf,
                    &mha_weights,
                    layer.mha_layer_idx as usize,
                    token_offset,
                    n_heads,
                    n_kv_heads,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    v_head_dim,
                    embedding_dim,
                    rope_freq_base,
                    mha_cache,
                    &mut s.q_decompressed[..q_dim],
                    k_part,
                    v_part,
                    s.score_buf,
                    s.attn_concat,
                    s.attn_out,
                    s.scratch,
                    layer.attn_quant,
                )?;
            }
        }

        for i in 0..embedding_dim {
            s.hidden[i] += s.attn_out[i];
        }

        // ── FFN sub-block ────────────────────────────────────────
        rmsnorm(s.hidden, layer.ffn_norm, s.norm_buf, rms_eps);

        match layer.mlp_type {
            MlpType::Dense => {
                // Per-tensor quant dispatch for the dense SwiGLU.
                // Kimi K2.6 ships layer 0 dense with Q4_0 — the Q4_K-only
                // `mlp_swiglu_avx512` (kept for Qwen3 β-anchor) would
                // misread those bytes. Inline the three matmuls through
                // `linear_dispatch_avx512` instead so the layer's actual
                // `ffn_dense_quant` (Q4_K / Q6_K / Q4_0 / Q8_0) drives
                // the kernel selection.
                let _ = linear_dispatch_avx512(
                    s.norm_buf,
                    layer.ffn_gate,
                    s.gate_buf,
                    embedding_dim,
                    intermediate_dim,
                    layer.ffn_dense_quant,
                );
                let _ = linear_dispatch_avx512(
                    s.norm_buf,
                    layer.ffn_up,
                    s.up_buf,
                    embedding_dim,
                    intermediate_dim,
                    layer.ffn_dense_quant,
                );
                silu_mul_avx512(s.gate_buf, s.up_buf, s.hidden_mlp_buf, intermediate_dim);
                let _ = linear_dispatch_avx512(
                    s.hidden_mlp_buf,
                    layer.ffn_down,
                    s.mlp_out,
                    intermediate_dim,
                    embedding_dim,
                    layer.ffn_dense_quant,
                );
            }
            MlpType::MoE => {
                // Zero the accumulator — moe_ffn writes shared expert to `output`
                // then accumulates weighted top-K experts into it.
                for v in s.mlp_out.iter_mut() {
                    *v = 0.0;
                }
                moe_ffn_avx512(
                    s.norm_buf,
                    layer.ffn_gate_inp,
                    layer.ffn_gate_inp_bias,
                    layer.ffn_gate_exps,
                    layer.ffn_up_exps,
                    layer.ffn_down_exps,
                    layer.ffn_gate_shexp,
                    layer.ffn_up_shexp,
                    layer.ffn_down_shexp,
                    n_experts,
                    top_k,
                    embedding_dim,
                    expert_intermediate,
                    s.router_score_buf,
                    s.expert_indices,
                    s.expert_weights,
                    s.gate_buf,
                    s.up_buf,
                    s.hidden_mlp_buf,
                    s.expert_out_buf,
                    s.mlp_out,
                    s.scratch,
                    layer.expert_quant,
                    routing_mode,
                    expert_weight_scale,
                );
            }
        }

        for i in 0..embedding_dim {
            s.hidden[i] += s.mlp_out[i];
        }
    }

    Ok(())
}
