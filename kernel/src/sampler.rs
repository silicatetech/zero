// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-D3.5: Top-K + Temperature sampling module.
//!
//! Per SQ3=α ratification: Top-K=40 + Temperature=1.0. Top-P deferred
//! to potential future Sub-MP-D7.
//!
//! Per SQ4=α ratification: parallel-track `lm_head_sample` function.
//! `lm_head_argmax` signature SACRED preserved (ADR-029 v2).
//!
//! Per V3.1 Pillar 7 (Z.264-269): NO `#[cfg(target_arch)]`, NO SIMD,
//! NO platform intrinsics. Pure scalar Rust.
//!
//! Per V3.1 Pillar 1 (Z.205-210): O(K log K) Top-K sort, O(K) softmax,
//! O(K) sampling. Stack-only hot-path ([f32; TOP_K], [u32; TOP_K]).
//! Zero heap allocation. Zero runtime IO.
//!
//! Per V3.1 Pillar 8 (Z.275-285): Rust temporary; will migrate to
//! Quarks in Stage 12+ when Validator + interpreter mature.

use crate::rng::Rng;

/// Top-K filter size. Per SQ3=α: K=40 fixed.
const TOP_K: usize = 40;

/// Qwen3 real vocabulary size (must match lm_head.rs VOCAB_SIZE_REAL).
/// Sampling only considers token IDs < this bound.
const VOCAB_SIZE_REAL: usize = 151_643;

/// Sample a token from logits using Top-K + Temperature strategy.
///
/// Algorithm:
/// 1. Find top K logits and their indices from logits[0..VOCAB_SIZE_REAL]
/// 2. Divide top-K logits by temperature
/// 3. Numerically-stable softmax (subtract max before exp)
/// 4. Cumulative-distribution sample via RNG
///
/// Returns token ID (u32) of sampled token.
///
/// Per V3.1 Pillar 7: identical algorithm both platforms.
/// Per V3.1 Pillar 1: O(K log K) + O(K) + O(K), stack-only.
pub fn lm_head_sample(logits: &[f32], rng: &mut Rng, temperature: f32) -> u32 {
    // Clamp vocab to real size (exclude padding region)
    let vocab_len = if logits.len() < VOCAB_SIZE_REAL {
        logits.len()
    } else {
        VOCAB_SIZE_REAL
    };

    // Step 1: Find Top-K logits via partial selection
    // Track top K indices + values on stack
    let mut top_vals = [f32::NEG_INFINITY; TOP_K];
    let mut top_idxs = [0u32; TOP_K];
    let mut min_val = f32::NEG_INFINITY;
    let mut min_pos: usize = 0;

    for i in 0..vocab_len {
        let v = logits[i];
        if v > min_val {
            // Replace the current minimum in top-K
            top_vals[min_pos] = v;
            top_idxs[min_pos] = i as u32;

            // Find new minimum in top-K
            min_val = f32::INFINITY;
            for k in 0..TOP_K {
                if top_vals[k] < min_val {
                    min_val = top_vals[k];
                    min_pos = k;
                }
            }
        }
    }

    // Step 2: Apply temperature scaling
    // (Temperature=1.0 is no-op but kept for correctness)
    let inv_temp = 1.0 / temperature;
    for k in 0..TOP_K {
        top_vals[k] *= inv_temp;
    }

    // Step 3: Numerically-stable softmax over Top-K
    // Find max for numerical stability
    let mut max_logit = f32::NEG_INFINITY;
    for k in 0..TOP_K {
        if top_vals[k] > max_logit {
            max_logit = top_vals[k];
        }
    }

    // exp(x - max) and sum
    let mut exp_sum: f32 = 0.0;
    for k in 0..TOP_K {
        let e = exp_approx(top_vals[k] - max_logit);
        top_vals[k] = e;
        exp_sum += e;
    }

    // Normalize to probabilities
    let inv_sum = 1.0 / exp_sum;
    for k in 0..TOP_K {
        top_vals[k] *= inv_sum;
    }

    // Step 4: Cumulative-distribution sample
    // Generate random float in [0, 1)
    let r = (rng.next_u32() as f32) / (u32::MAX as f32);

    let mut cumulative: f32 = 0.0;
    for k in 0..TOP_K {
        cumulative += top_vals[k];
        if r < cumulative {
            return top_idxs[k];
        }
    }

    // Fallback: return last top-K token (rounding edge case)
    top_idxs[TOP_K - 1]
}

/// Fast scalar exp approximation for softmax.
///
/// Uses the standard approach: clamp input, then compute via
/// the identity exp(x) = 2^(x / ln(2)) with polynomial approximation.
///
/// Per V3.1 Pillar 7: pure scalar, no SIMD, no platform intrinsics.
/// Accuracy sufficient for Top-K=40 probability ranking.
fn exp_approx(x: f32) -> f32 {
    // Clamp to prevent overflow/underflow
    let x = if x < -87.0 {
        -87.0
    } else if x > 88.0 {
        88.0
    } else {
        x
    };

    // Use the bit-manipulation fast exp approximation
    // Based on Schraudolph's method with improved accuracy
    //
    // exp(x) ≈ 2^(x * log2(e))
    // We use the identity: float bits = (2^23) * (x * log2(e) + 127)
    let log2e: f32 = 1.442_695_04;
    let a = x * log2e;

    // Split into integer and fractional parts
    let ai = a as i32; // floor
    let af = a - (ai as f32); // fractional part

    // Polynomial approximation for 2^af where af in [0, 1)
    // 2^af ≈ 1 + af * (ln2 + af * (ln2^2/2 + af * ln2^3/6))
    let ln2: f32 = 0.693_147_18;
    let c1 = ln2;
    let c2 = 0.240_226_5; // ln2^2 / 2
    let c3 = 0.055_504_1; // ln2^3 / 6
    let frac = 1.0 + af * (c1 + af * (c2 + af * c3));

    // Combine: 2^ai * 2^af
    // 2^ai via bit manipulation on f32 exponent
    let exp_bits: u32 = ((127 + ai) as u32) << 23;
    let pow2_int = f32::from_bits(exp_bits);

    pow2_int * frac
}
