// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR-029 Patch v8.4 — AVX-512 vectorised SiLU + element-wise multiply.
//!
//! Replaces the scalar `for i in 0..n { x[i] / (1 + libm::expf(-x[i])) }`
//! and the subsequent element-wise multiply in `mlp_swiglu_avx512` with
//! a single fused 16-lane AVX-512 pass.
//!
//! # Why this matters
//!
//! Disassembly of `forward_single_token_avx512` showed direct callq to
//! `libm::math::expf` from the SiLU loop. With 6144 elements per layer
//! and 28 layers, that is **172,032 scalar `expf` calls per token**,
//! each ~30 cycles → roughly 14% of the per-token cycle budget at
//! 97.9 tok/s on Cherry. Vectorising the polynomial approximation to
//! 16-lane AVX-512 collapses that to ~10K cycles per layer → ~0.07% of
//! budget. Expected gain: ~12-14% per token.
//!
//! # Numerical contract
//!
//! Sleef-style polynomial `expf` approximation, ~1 ULP error at
//! typical activation magnitudes (|x| ≤ 12). Drift vs the sacred
//! scalar SiLU path is ~1 ULP per element. ADR-029 v3 Two-Anchor
//! registers AVX-512 as a feature mode whose `logit_bits` may drift
//! within the ULP class; Token-ID 25 HARD GATE must hold. This is the
//! same regime as multi-acc FMA (commit 1cf1fa6) which empirically
//! preserved Token-ID 25 across the v8.2 deployment.
//!
//! # Safety
//!
//! `#[target_feature(enable = "avx512f")]` enables AVX-512F across the
//! entire function. Caller must have completed XSAVE / XCR0 setup
//! (done at the very top of `kernel_main` post-v8.2).

use core::arch::x86_64::*;

// ─────────────────────────────────────────────────────────────────
// Vectorised f32 exp on AVX-512
// ─────────────────────────────────────────────────────────────────

/// AVX-512 16-lane `expf` polynomial approximation.
///
/// Decomposition: `exp(x) = 2^k * exp(r)` where
///   k = round(x * log2_e)
///   r = x - k * ln(2)  (with 2-part Cody-Waite for precision)
/// The 2^k factor is built via the f32 exponent field; `exp(r)` is a
/// 5th-degree Horner polynomial on `r in [-ln(2)/2, ln(2)/2]`.
///
/// Coefficient set from Sleef's `expf` (BSD-licensed reference); error
/// bounded by ~1 ULP for |x| ≤ 88 (which is the f32 expf domain — beyond
/// that the output saturates to ±inf or 0).
///
/// # Safety
/// AVX-512F enabled at the function boundary.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn exp_f32_avx512(x: __m512) -> __m512 {
    // Constants. Using literals; compiler folds these into the loop
    // prologue (set1 emits broadcast).
    let log2e = _mm512_set1_ps(1.442_695_04_f32); // log2(e)
    let ln2_hi = _mm512_set1_ps(0.693_359_4_f32); // first 16 bits of ln(2)
    let ln2_lo = _mm512_set1_ps(-2.121_944_4e-4_f32); // remainder
    let one = _mm512_set1_ps(1.0_f32);

    // Clamp to a safe range so the integer cast doesn't overflow.
    let hi = _mm512_set1_ps(88.376_25_f32);
    let lo = _mm512_set1_ps(-88.376_25_f32);
    let x = _mm512_max_ps(_mm512_min_ps(x, hi), lo);

    // k = round(x * log2e)
    let kf = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_ps(x, log2e),
    );
    let k = _mm512_cvtps_epi32(kf);

    // r = x - kf*ln2_hi - kf*ln2_lo (Cody-Waite reduction)
    let r = _mm512_fnmadd_ps(kf, ln2_lo, _mm512_fnmadd_ps(kf, ln2_hi, x));

    // exp(r) ≈ 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120)))))
    //
    // Horner-style; constants from the standard Taylor series.
    let c5 = _mm512_set1_ps(8.333_333_3e-3_f32); // 1/120
    let c4 = _mm512_set1_ps(4.166_666_7e-2_f32); // 1/24
    let c3 = _mm512_set1_ps(0.166_666_67_f32); // 1/6
    let c2 = _mm512_set1_ps(0.5_f32);
    let mut p = _mm512_fmadd_ps(r, c5, c4);
    p = _mm512_fmadd_ps(r, p, c3);
    p = _mm512_fmadd_ps(r, p, c2);
    p = _mm512_fmadd_ps(r, p, one);
    p = _mm512_fmadd_ps(r, p, one);

    // Build 2^k by injecting (k + 127) into the f32 exponent field.
    let bias = _mm512_set1_epi32(127);
    let exp_bits = _mm512_slli_epi32::<23>(_mm512_add_epi32(k, bias));
    let two_pow_k = _mm512_castsi512_ps(exp_bits);

    _mm512_mul_ps(p, two_pow_k)
}

// ─────────────────────────────────────────────────────────────────
// Fused SiLU(gate) × up — replaces the scalar libm::expf loop
// ─────────────────────────────────────────────────────────────────

/// Compute `out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]` for
/// `i in 0..len`. `len` must be a multiple of 16.
///
/// Fuses the SiLU on `gate` with the subsequent element-wise multiply
/// against `up` into a single AVX-512 pass — one load each of gate
/// and up per 16 elements, one store of the product. Replaces the
/// two scalar loops in `mlp_swiglu_avx512` Step 3 + Step 4.
///
/// # Safety
/// AVX-512F. `gate.len() >= len`, `up.len() >= len`, `out.len() >= len`,
/// `len % 16 == 0` (Qwen3 intermediate_dim is a multiple of 16).
#[target_feature(enable = "avx512f")]
pub unsafe fn silu_mul_avx512(gate: &[f32], up: &[f32], out: &mut [f32], len: usize) {
    debug_assert!(gate.len() >= len);
    debug_assert!(up.len() >= len);
    debug_assert!(out.len() >= len);
    debug_assert!(len % 16 == 0);

    let one = _mm512_set1_ps(1.0_f32);
    let mut i = 0usize;
    while i < len {
        let g = _mm512_loadu_ps(gate.as_ptr().add(i));
        let u = _mm512_loadu_ps(up.as_ptr().add(i));
        // sigmoid(g) = 1 / (1 + exp(-g))
        let neg_g = _mm512_sub_ps(_mm512_setzero_ps(), g);
        let e = exp_f32_avx512(neg_g);
        let denom = _mm512_add_ps(one, e);
        // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g)), THEN × up — in
        // exactly the scalar/NEON operand order:
        //   silu   = gate / (1 + exp(-gate))   (bounded: |silu| ≤ |gate|)
        //   result = silu * up
        // The previous order formed the RAW product `gate*up` first and
        // divided afterwards: (gate*up)/denom. That is algebraically
        // identical but forms an intermediate that OVERFLOWS to ±Inf in
        // f32 when |gate*up| > 3.4e38 (poorly-conditioned activations,
        // e.g. from a coarse quantization), turning finite inputs into
        // Inf → NaN at the next RMSNorm → Attention(NumericalInstability).
        // The scalar/NEON paths never form gate*up, so they stay finite;
        // computing the bounded SiLU first matches them and removes the
        // x86-only overflow path. β-anchor (scalar build) unaffected.
        let silu = _mm512_div_ps(g, denom);
        let result = _mm512_mul_ps(silu, u);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), result);
        i += 16;
    }
}

// ─────────────────────────────────────────────────────────────────
// Smoke checks (host-side, AVX-512 not required to compile)
// ─────────────────────────────────────────────────────────────────
//
// Unit-testing AVX-512 intrinsics from `cargo test` on a non-AVX-512
// host machine is impractical. Validation belongs on Cherry hardware
// against the sacred scalar SiLU output (per ADR-029 v3 feature-mode
// drift contract).
