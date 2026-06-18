// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR-029 Patch v8.4 — AVX-512 vectorised sin/cos for RoPE.
//!
//! Replaces the per-head scalar `cosf(theta) + sinf(theta)` calls
//! inside the sacred `rope()` function with a per-token shared LUT
//! computed once on AVX-512, then a vectorised RoPE rotation applied
//! to each attention head using that LUT.
//!
//! # Why this matters
//!
//! Disassembly of `forward_single_token_avx512` showed:
//!
//! * Each call to sacred `rope()` runs HALF=64 iterations, each calling
//!   `libm::cosf(theta) + libm::sinf(theta)`.
//! * Per layer, rope is called once per attention head — 16 q-heads +
//!   8 kv-heads = 24 calls, each computing 64 (cos, sin) pairs.
//! * BUT all 24 heads at one token position use the SAME 64 (cos, sin)
//!   pairs (they depend only on `position * inv_freqs[i]`, both
//!   shared across heads).
//! * → 23/24 of the trig work is redundantly recomputed per layer.
//!
//! Combined with vectorisation: from ~86K scalar libm transcendental
//! calls per token (~2.6 M cycles ≈ 7%) down to ~64 sincos pair
//! computations per token, done in ~50 cycles total of AVX-512 work.
//! Expected gain: ~7% per-token + de-duplication.
//!
//! # Numerical contract
//!
//! Cody-Waite range reduction + 5th-degree polynomial. Accuracy
//! ~1 ULP for |theta| ≤ 2048 × max(inv_freq) (the maximum theta in
//! Qwen3's 2048-token context). Drift vs `libm::cosf/sinf` is within
//! the same ULP class — ADR-029 v3 Two-Anchor permits this as
//! feature-mode logit_bits drift.
//!
//! # Safety
//!
//! `#[target_feature(enable = "avx512f")]`. Caller must have XSAVE /
//! XCR0 setup (done at the very top of `kernel_main` post-v8.2).

use core::arch::x86_64::*;

// ─────────────────────────────────────────────────────────────────
// AVX-512 sincos polynomial
// ─────────────────────────────────────────────────────────────────

/// Compute sin(x) and cos(x) for 16 lanes of x simultaneously.
///
/// Reduces `x` to `r in [-π/4, π/4]` via Cody-Waite (2-part π/2
/// constants) and a quadrant index `k = round(x * 2/π) mod 4`. Then
/// evaluates 6th-degree sin and 5th-degree cos polynomials on r,
/// permuting/negating based on quadrant.
///
/// Returns `(sin_x, cos_x)`. ~1 ULP error for |x| ≤ 2π × 2^14
/// (well above any RoPE theta we ever see in 2048-token contexts).
///
/// # Safety
/// AVX-512F.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn sincos_f32_avx512(x: __m512) -> (__m512, __m512) {
    // Range reduction: k = round(x * 2/π); r = x - k*π/2.
    // Cody-Waite split of π/2 into two f32s for extended precision.
    let two_over_pi = _mm512_set1_ps(0.636_619_77_f32); // 2/π
    let pi_over_2_hi = _mm512_set1_ps(1.570_796_25_f32); // π/2 high bits
    let pi_over_2_lo = _mm512_set1_ps(7.549_790_3e-8_f32); // π/2 low bits

    let kf = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_ps(x, two_over_pi),
    );
    let k = _mm512_cvtps_epi32(kf);

    // r = x - kf*π/2_hi - kf*π/2_lo
    let r = _mm512_fnmadd_ps(kf, pi_over_2_lo, _mm512_fnmadd_ps(kf, pi_over_2_hi, x));
    let r2 = _mm512_mul_ps(r, r);

    // sin polynomial on r ∈ [-π/4, π/4]:
    //   sin(r) ≈ r * (1 + r²·(-1/6 + r²·(1/120 + r²·(-1/5040))))
    let s5 = _mm512_set1_ps(-1.984_126_98e-4_f32); // -1/5040
    let s3 = _mm512_set1_ps(8.333_333_3e-3_f32); // 1/120
    let s1 = _mm512_set1_ps(-0.166_666_67_f32); // -1/6
    let mut sin_p = _mm512_fmadd_ps(r2, s5, s3);
    sin_p = _mm512_fmadd_ps(r2, sin_p, s1);
    sin_p = _mm512_fmadd_ps(r2, sin_p, _mm512_set1_ps(1.0));
    let sin_r = _mm512_mul_ps(r, sin_p);

    // cos polynomial on r ∈ [-π/4, π/4]:
    //   cos(r) ≈ 1 + r²·(-1/2 + r²·(1/24 + r²·(-1/720 + r²·(1/40320))))
    let c6 = _mm512_set1_ps(2.480_158_7e-5_f32); // 1/40320
    let c4 = _mm512_set1_ps(-1.388_888_9e-3_f32); // -1/720
    let c2 = _mm512_set1_ps(4.166_666_7e-2_f32); // 1/24
    let c0 = _mm512_set1_ps(-0.5_f32);
    let mut cos_p = _mm512_fmadd_ps(r2, c6, c4);
    cos_p = _mm512_fmadd_ps(r2, cos_p, c2);
    cos_p = _mm512_fmadd_ps(r2, cos_p, c0);
    let cos_r = _mm512_fmadd_ps(r2, cos_p, _mm512_set1_ps(1.0));

    // Quadrant adjustment:
    //   k mod 4 = 0  →  sin = sin_r,  cos =  cos_r
    //          = 1  →  sin = cos_r,  cos = -sin_r
    //          = 2  →  sin = -sin_r, cos = -cos_r
    //          = 3  →  sin = -cos_r, cos =  sin_r
    let k_mod_4 = _mm512_and_epi32(k, _mm512_set1_epi32(3));

    // mask bits we need:
    //   swap   = (k_mod_4 == 1) || (k_mod_4 == 3)   → bit 0 set
    //   negsin = (k_mod_4 == 2) || (k_mod_4 == 3)   → bit 1 set
    //   negcos = (k_mod_4 == 1) || (k_mod_4 == 2)   → bit 0 XOR bit 1
    let bit0 = _mm512_and_epi32(k_mod_4, _mm512_set1_epi32(1));
    let bit1 = _mm512_srli_epi32::<1>(_mm512_and_epi32(k_mod_4, _mm512_set1_epi32(2)));
    let swap_mask: __mmask16 = _mm512_cmpneq_epi32_mask(bit0, _mm512_setzero_si512());
    let negsin_mask: __mmask16 = _mm512_cmpneq_epi32_mask(bit1, _mm512_setzero_si512());
    let negcos_mask: __mmask16 =
        _mm512_cmpneq_epi32_mask(_mm512_xor_epi32(bit0, bit1), _mm512_setzero_si512());

    // Choose pre-negation values.
    let sin_pre = _mm512_mask_blend_ps(swap_mask, sin_r, cos_r);
    let cos_pre = _mm512_mask_blend_ps(swap_mask, cos_r, sin_r);

    let sign_bit = _mm512_castsi512_ps(_mm512_set1_epi32(0x8000_0000_u32 as i32));
    let sin_neg = _mm512_xor_ps(sin_pre, sign_bit);
    let cos_neg = _mm512_xor_ps(cos_pre, sign_bit);

    let sin_x = _mm512_mask_blend_ps(negsin_mask, sin_pre, sin_neg);
    let cos_x = _mm512_mask_blend_ps(negcos_mask, cos_pre, cos_neg);

    (sin_x, cos_x)
}

// ─────────────────────────────────────────────────────────────────
// RoPE sincos LUT — compute once per (token, layer) for all heads
// ─────────────────────────────────────────────────────────────────

/// Fill `cos_out` and `sin_out` with the per-pair RoPE rotation
/// constants for the given `position`. `len` must equal the HALF
/// dimension (= head_dim / 2). Both output buffers must be `len`
/// long and ideally 64-byte aligned for `vmovaps`; the
/// implementation uses unaligned stores to be safe with arbitrary
/// caller-provided buffers.
///
/// Per Qwen3 layout: theta[i] = position * inv_freqs[i].
///
/// Loops in 16-lane strides; for HALF=64 this is exactly 4 passes.
///
/// # Safety
/// AVX-512F. `inv_freqs.len() >= len`, output buffers `>= len`,
/// `len % 16 == 0`.
#[target_feature(enable = "avx512f")]
pub unsafe fn rope_sincos_lut_avx512(
    position: usize,
    inv_freqs: &[f32],
    cos_out: &mut [f32],
    sin_out: &mut [f32],
    len: usize,
) {
    debug_assert!(inv_freqs.len() >= len);
    debug_assert!(cos_out.len() >= len);
    debug_assert!(sin_out.len() >= len);
    debug_assert!(len % 16 == 0);

    let pos_v = _mm512_set1_ps(position as f32);
    let mut i = 0usize;
    while i < len {
        let inv = _mm512_loadu_ps(inv_freqs.as_ptr().add(i));
        let theta = _mm512_mul_ps(pos_v, inv);
        let (sin_t, cos_t) = sincos_f32_avx512(theta);
        _mm512_storeu_ps(cos_out.as_mut_ptr().add(i), cos_t);
        _mm512_storeu_ps(sin_out.as_mut_ptr().add(i), sin_t);
        i += 16;
    }
}

// ─────────────────────────────────────────────────────────────────
// Apply pre-computed sincos LUT to one attention head — vectorised
// ─────────────────────────────────────────────────────────────────

/// Apply RoPE rotation in-place on one attention head, using the
/// pre-computed sin/cos LUT for the current token position.
///
/// Semantically equivalent to the sacred scalar `rope()` for the
/// half-split pair layout:
///   qk[i]      = x_old * cos[i] - y_old * sin[i]
///   qk[i+HALF] = x_old * sin[i] + y_old * cos[i]
/// where (x_old, y_old) = (qk[i], qk[i+HALF]) BEFORE the writeback.
///
/// `half` is the same `HALF` constant the sacred `rope()` uses
/// (head_dim / 2). Must be a multiple of 16 (Qwen3: 64 — exact).
///
/// # Safety
/// AVX-512F. `qk.len() >= 2 * half`, `cos.len() >= half`,
/// `sin.len() >= half`, `half % 16 == 0`.
#[target_feature(enable = "avx512f")]
pub unsafe fn rope_apply_avx512(qk: &mut [f32], cos: &[f32], sin: &[f32], half: usize) {
    debug_assert!(qk.len() >= 2 * half);
    debug_assert!(cos.len() >= half);
    debug_assert!(sin.len() >= half);
    debug_assert!(half % 16 == 0);

    let qk_ptr = qk.as_mut_ptr();
    let mut i = 0usize;
    while i < half {
        let x_old = _mm512_loadu_ps(qk_ptr.add(i));
        let y_old = _mm512_loadu_ps(qk_ptr.add(i + half));
        let c = _mm512_loadu_ps(cos.as_ptr().add(i));
        let s = _mm512_loadu_ps(sin.as_ptr().add(i));

        // x_new = x_old * c - y_old * s
        let x_new = _mm512_fmsub_ps(x_old, c, _mm512_mul_ps(y_old, s));
        // y_new = x_old * s + y_old * c
        let y_new = _mm512_fmadd_ps(x_old, s, _mm512_mul_ps(y_old, c));

        _mm512_storeu_ps(qk_ptr.add(i), x_new);
        _mm512_storeu_ps(qk_ptr.add(i + half), y_new);

        i += 16;
    }
}
