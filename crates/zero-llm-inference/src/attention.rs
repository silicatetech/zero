// SPDX-License-Identifier: AGPL-3.0-or-later
//! GQA Scaled Dot-Product Attention with KvCache integration.
//!
//! Per ADR-029 v1 D8/D9/D10: includes QK-Norm (Qwen3-specific),
//! Q6_K V-projection, and reference-validated against Sub-MP-C1 dumps.
//!
//! Forward-pass order per layer:
//! ```text
//! normed_input
//!   → Q_proj (linear_q4k) → Q_norm (per-head rmsnorm) → RoPE_Q
//!   → K_proj (linear_q4k) → K_norm (per-head rmsnorm) → RoPE_K
//!   → V_proj (linear_q6k)
//!   → store K,V in KvCache
//!   → scaled_dot_product_attention (GQA broadcast 2:1)
//!   → O_proj (linear_q4k)
//!   → output
//! ```

use crate::kv_cache::{KvCache, KvCacheError};
use crate::ops::{
    linear_dispatch, linear_q4k, linear_q6k, rmsnorm, rope, LinearScratch, RopeContext,
};

/// Errors from attention operations.
#[derive(Debug, Clone, Copy)]
pub enum AttentionError {
    /// Softmax produced non-finite values despite Max-Trick.
    NumericalInstability,
    /// KV-Cache operation failed.
    KvCache(KvCacheError),
}

impl From<KvCacheError> for AttentionError {
    fn from(e: KvCacheError) -> Self {
        AttentionError::KvCache(e)
    }
}

/// Softmax with Max-Trick for numerical stability.
///
/// Mandatory NaN-prevention: subtracts max before exp.
/// exp(x_i - max) / sum(exp(x_j - max)) = standard softmax but
/// with max(exp) = 1.0 (stable, no overflow).
///
/// `scores[0..len]` is modified in-place to contain softmax weights.
///
/// Returns `Err` if max_score is non-finite or sum_exp is zero/non-finite.
pub fn softmax(scores: &mut [f32], len: usize) -> Result<(), AttentionError> {
    if len == 0 {
        return Ok(());
    }

    // Find max for numerical stability
    let mut max_s = f32::NEG_INFINITY;
    for i in 0..len {
        if scores[i] > max_s {
            max_s = scores[i];
        }
    }

    if !max_s.is_finite() {
        return Err(AttentionError::NumericalInstability);
    }

    // exp(x - max)
    let mut sum_exp: f32 = 0.0;
    for i in 0..len {
        let e = libm::expf(scores[i] - max_s);
        scores[i] = e;
        sum_exp += e;
    }

    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return Err(AttentionError::NumericalInstability);
    }

    // Normalize
    let inv_sum = 1.0 / sum_exp;
    for i in 0..len {
        scores[i] *= inv_sum;
    }

    Ok(())
}

/// Single-token, single-layer GQA attention forward-pass.
///
/// Processes exactly ONE new token at `token_offset` through Layer `layer_idx`.
///
/// ## Arguments
///
/// * `normed_input` — `[embedding_dim]` post-RMSNorm input for this token
/// * `q_weight`, `k_weight` — Q4_K weight bytes for Q and K projections
/// * `v_weight` — Q4_K or Q6_K weight bytes for V projection (per-layer iMatrix)
/// * `o_weight` — Q4_K weight bytes for output projection
/// * `v_is_q6k` — dispatch flag: true for Q6_K V-projection, false for Q4_K
/// * `q_norm_weight`, `k_norm_weight` — `[head_dim]` f32 QK-norm weights
/// * `layer_idx` — layer number (for KvCache addressing)
/// * `token_offset` — position of this token in the sequence (for RoPE + KvCache)
/// * `rope_ctx` — precomputed RoPE context
/// * `rms_eps` — RMSNorm epsilon (1e-6 for Qwen3)
/// * `n_q_heads`, `n_kv_heads`, `head_dim`, `embedding_dim` — model dimensions
/// * `kv_cache` — KV cache (K/V for this token will be stored)
/// * `q_buf`, `k_buf`, `v_buf` — scratch for projections (caller-allocated)
/// * `q_out`, `k_out` — scratch for post-norm/RoPE Q and K
/// * `score_buf` — scratch for attention scores `[token_offset + 1]`
/// * `attn_head_buf` — scratch for per-head attention output `[head_dim]`
/// * `attn_out` — scratch for concatenated attention output `[n_q_heads * head_dim]`
/// * `scratch` — LinearScratch for dequant streaming
/// * `output` — final output `[embedding_dim]` (caller-allocated)
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_single_token<const HALF: usize>(
    normed_input: &[f32],
    q_weight: &[u8],
    k_weight: &[u8],
    v_weight: &[u8],
    o_weight: &[u8],
    v_is_q6k: bool,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    layer_idx: usize,
    token_offset: usize,
    rope_ctx: &RopeContext<HALF>,
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
    attn_out: &mut [f32],
    scratch: &mut LinearScratch,
    output: &mut [f32],
) -> Result<(), AttentionError> {
    let q_dim = n_q_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let gqa_ratio = n_q_heads / n_kv_heads;
    let scale = 1.0 / libm::sqrtf(head_dim as f32);
    let total_tokens = token_offset + 1; // attend to all past + current

    // ── Step 1: Q/K/V projections ────────────────────────────────
    linear_q4k(normed_input, q_weight, q_buf, scratch, embedding_dim, q_dim);
    linear_q4k(
        normed_input,
        k_weight,
        k_buf,
        scratch,
        embedding_dim,
        kv_dim,
    );
    if v_is_q6k {
        linear_q6k(
            normed_input,
            v_weight,
            v_buf,
            scratch,
            embedding_dim,
            kv_dim,
        );
    } else {
        linear_q4k(
            normed_input,
            v_weight,
            v_buf,
            scratch,
            embedding_dim,
            kv_dim,
        );
    }

    // ── Step 2: QK-Norm (per-head RMSNorm via reshape + reuse) ──
    // Q: reshape [n_q_heads, head_dim], apply rmsnorm per-head
    for h in 0..n_q_heads {
        let offset = h * head_dim;
        rmsnorm(
            &q_buf[offset..offset + head_dim],
            q_norm_weight,
            &mut q_out[offset..offset + head_dim],
            rms_eps,
        );
    }

    // K: reshape [n_kv_heads, head_dim], apply rmsnorm per-head
    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        rmsnorm(
            &k_buf[offset..offset + head_dim],
            k_norm_weight,
            &mut k_out[offset..offset + head_dim],
            rms_eps,
        );
    }

    // ── Step 3: RoPE Variante B on Q and K (per-head, in-place) ─
    for h in 0..n_q_heads {
        let offset = h * head_dim;
        rope(
            &mut q_out[offset..offset + head_dim],
            rope_ctx,
            token_offset,
        );
    }

    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        rope(
            &mut k_out[offset..offset + head_dim],
            rope_ctx,
            token_offset,
        );
    }

    // ── Step 4: Store K and V in KvCache ─────────────────────────
    kv_cache.store_kv(layer_idx, token_offset, k_out, v_buf)?;

    // ── Step 5: Scaled Dot-Product Attention (GQA broadcast) ─────
    let k_full = kv_cache.get_k_slice(layer_idx, total_tokens)?;
    let v_full = kv_cache.get_v_slice(layer_idx, total_tokens)?;

    for q_h in 0..n_q_heads {
        let kv_h = q_h / gqa_ratio; // GQA broadcast

        let q_offset = q_h * head_dim;
        let q_vec = &q_out[q_offset..q_offset + head_dim];

        // Compute attention scores: q_vec · K[kv_h, t] for each cached token
        for t in 0..total_tokens {
            let k_offset = t * kv_dim + kv_h * head_dim;
            let k_vec = &k_full[k_offset..k_offset + head_dim];

            let mut dot: f32 = 0.0;
            for i in 0..head_dim {
                dot += q_vec[i] * k_vec[i];
            }
            score_buf[t] = dot * scale;
        }

        // ═══ MAX-TRICK SOFTMAX (mandatory NaN-prevention) ═══
        softmax(&mut score_buf[..total_tokens], total_tokens)?;

        // Compute weighted sum of V for this Q-head
        for i in 0..head_dim {
            attn_head_buf[i] = 0.0;
        }
        for t in 0..total_tokens {
            let v_offset = t * kv_dim + kv_h * head_dim;
            let v_vec = &v_full[v_offset..v_offset + head_dim];
            let weight = score_buf[t];
            for i in 0..head_dim {
                attn_head_buf[i] += weight * v_vec[i];
            }
        }

        // Copy to concatenated attention output
        let out_offset = q_h * head_dim;
        attn_out[out_offset..out_offset + head_dim].copy_from_slice(&attn_head_buf[..head_dim]);
    }

    // ── Step 6: O projection ─────────────────────────────────────
    linear_q4k(attn_out, o_weight, output, scratch, q_dim, embedding_dim);

    Ok(())
}

/// Single-token GQA attention with per-projection quant dispatch.
///
/// This is the native-model variant of [`gqa_attention_single_token`]. The
/// legacy function remains Q4_K/Q6_K-specific for the byte-accuracy anchors,
/// while `.smodel` artifacts can route Q4_0/Q8_0/Q*_K tensors according to the
/// native tensor directory.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_single_token_dispatch<const HALF: usize>(
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
    rope_ctx: &RopeContext<HALF>,
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
    attn_out: &mut [f32],
    scratch: &mut LinearScratch,
    output: &mut [f32],
) -> Result<(), AttentionError> {
    let q_dim = n_q_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let gqa_ratio = n_q_heads / n_kv_heads;
    let scale = 1.0 / libm::sqrtf(head_dim as f32);
    let total_tokens = token_offset + 1;

    let _ = linear_dispatch(
        normed_input,
        q_weight,
        q_buf,
        scratch,
        embedding_dim,
        q_dim,
        q_quant,
    );
    let _ = linear_dispatch(
        normed_input,
        k_weight,
        k_buf,
        scratch,
        embedding_dim,
        kv_dim,
        k_quant,
    );
    let _ = linear_dispatch(
        normed_input,
        v_weight,
        v_buf,
        scratch,
        embedding_dim,
        kv_dim,
        v_quant,
    );

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

    for h in 0..n_q_heads {
        let offset = h * head_dim;
        rope(
            &mut q_out[offset..offset + head_dim],
            rope_ctx,
            token_offset,
        );
    }

    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        rope(
            &mut k_out[offset..offset + head_dim],
            rope_ctx,
            token_offset,
        );
    }

    kv_cache.store_kv(layer_idx, token_offset, k_out, v_buf)?;

    let k_full = kv_cache.get_k_slice(layer_idx, total_tokens)?;
    let v_full = kv_cache.get_v_slice(layer_idx, total_tokens)?;

    for q_h in 0..n_q_heads {
        let kv_h = q_h / gqa_ratio;

        let q_offset = q_h * head_dim;
        let q_vec = &q_out[q_offset..q_offset + head_dim];

        for t in 0..total_tokens {
            let k_offset = t * kv_dim + kv_h * head_dim;
            let k_vec = &k_full[k_offset..k_offset + head_dim];

            let mut dot: f32 = 0.0;
            for i in 0..head_dim {
                dot += q_vec[i] * k_vec[i];
            }
            score_buf[t] = dot * scale;
        }

        softmax(&mut score_buf[..total_tokens], total_tokens)?;

        for i in 0..head_dim {
            attn_head_buf[i] = 0.0;
        }
        for t in 0..total_tokens {
            let v_offset = t * kv_dim + kv_h * head_dim;
            let v_vec = &v_full[v_offset..v_offset + head_dim];
            let weight = score_buf[t];
            for i in 0..head_dim {
                attn_head_buf[i] += weight * v_vec[i];
            }
        }

        let out_offset = q_h * head_dim;
        attn_out[out_offset..out_offset + head_dim].copy_from_slice(&attn_head_buf[..head_dim]);
    }

    let _ = linear_dispatch(
        attn_out,
        o_weight,
        output,
        scratch,
        q_dim,
        embedding_dim,
        o_quant,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn test_softmax_single_element() {
        let mut scores = [5.0f32];
        softmax(&mut scores, 1).unwrap();
        assert!(
            (scores[0] - 1.0).abs() < 1e-6,
            "single element softmax should be 1.0"
        );
    }

    #[test]
    fn test_softmax_two_equal() {
        let mut scores = [1.0f32, 1.0];
        softmax(&mut scores, 2).unwrap();
        assert!(
            (scores[0] - 0.5).abs() < 1e-6,
            "equal inputs: got {}",
            scores[0]
        );
        assert!(
            (scores[1] - 0.5).abs() < 1e-6,
            "equal inputs: got {}",
            scores[1]
        );
    }

    #[test]
    fn test_softmax_max_trick_large_values() {
        // Without Max-Trick, exp(100) = inf → NaN
        let mut scores = [100.0f32, 101.0, 99.0];
        softmax(&mut scores, 3).unwrap();
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {}", sum);
        assert!(scores[1] > scores[0], "101 should have highest weight");
        assert!(scores[0] > scores[2], "100 > 99");
    }

    #[test]
    fn test_softmax_negative_infinity_masked() {
        // Causal mask scenario: some scores are -inf
        let mut scores = [5.0f32, 3.0, f32::NEG_INFINITY];
        softmax(&mut scores, 3).unwrap();
        assert!(scores[2] < 1e-10, "-inf should produce ~0 weight");
        let sum: f32 = scores[0..2].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "non-masked should sum to ~1.0");
    }

    #[test]
    fn test_softmax_numerical_stability_error() {
        let mut scores = [f32::NAN];
        assert!(softmax(&mut scores, 1).is_err());
    }

    #[test]
    fn test_softmax_known_values() {
        // softmax([0, 1]) = [exp(0)/(exp(0)+exp(1)), exp(1)/(exp(0)+exp(1))]
        //                  = [1/(1+e), e/(1+e)] ≈ [0.2689, 0.7311]
        let mut scores = [0.0f32, 1.0];
        softmax(&mut scores, 2).unwrap();
        assert!((scores[0] - 0.2689).abs() < 1e-3, "got {}", scores[0]);
        assert!((scores[1] - 0.7311).abs() < 1e-3, "got {}", scores[1]);
    }
}
