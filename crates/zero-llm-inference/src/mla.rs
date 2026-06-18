// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-Head Latent Attention (MLA) for DeepSeek-V2/V3 / Kimi K2.6.
//!
//! MLA compresses KV representations through a low-rank bottleneck,
//! drastically reducing KV-cache memory while maintaining attention quality.
//!
//! # MLA vs GQA
//!
//! GQA (Qwen3): Q, K, V are projected directly from hidden state.
//!   KV-cache stores full K, V per layer per token.
//!
//! MLA (Kimi K2.6):
//!   1. Compress: c_kv = W_dkv × hidden  [hidden → kv_lora_rank=512]
//!   2. Decompress for attention: [k_nope; v] = W_ukv × c_kv
//!   3. Separate RoPE: k_rope from subset of c_kv (or separate projection)
//!   4. K = concat(k_nope, k_rope)
//!   5. Q similarly compressed then decompressed
//!   6. KV-cache stores only c_kv + k_rope (compressed!)
//!
//! For initial implementation: we decompress and store full K, V in cache
//! (simpler, compatible with existing KvCache, optimize later).
//!
//! # Design Constraints
//!
//! - `no_std`, zero allocation in hot path
//! - Reuses existing Q4_K matmul and RoPE operators
//! - Compatible with existing KvCache (stores decompressed K, V)

use crate::attention::softmax;
use crate::kv_cache::{KvCacheError, MhaKvCache, MlaKvCache};
use crate::ops::{linear_dispatch, rmsnorm, LinearScratch};

/// Errors from MLA attention operations.
#[derive(Debug, Clone, Copy)]
pub enum MlaError {
    /// Softmax produced non-finite values.
    NumericalInstability,
    /// KV-Cache operation failed.
    KvCache(KvCacheError),
}

impl From<KvCacheError> for MlaError {
    fn from(e: KvCacheError) -> Self {
        MlaError::KvCache(e)
    }
}

/// MLA attention weights for a single layer.
///
/// Tensor names in GGUF (DeepSeek-V2/V3):
///   blk.{L}.attn_kv_a_mqa.weight   — KV down-projection
///   blk.{L}.attn_kv_a_norm.weight   — KV compression norm
///   blk.{L}.attn_kv_b.weight        — KV up-projection
///   blk.{L}.attn_q_a.weight         — Q down-projection (absent for layer 0)
///   blk.{L}.attn_q_a_norm.weight    — Q compression norm
///   blk.{L}.attn_q_b.weight         — Q up-projection
///   blk.{L}.attn_output.weight      — Output projection
///   blk.{L}.attn_k_norm.weight      — K RoPE norm
pub struct MlaWeights<'a> {
    /// KV down-projection: [hidden_dim → kv_lora_rank + qk_rope_head_dim]
    pub kv_a_mqa: &'a [u8],
    /// KV compression norm: [kv_lora_rank]
    pub kv_a_norm: &'a [f32],
    /// KV up-projection (combined format):
    /// [kv_lora_rank → n_heads × (qk_nope_head_dim + v_head_dim)].
    /// Used when the GGUF ships a single `attn_kv_b.weight` tensor.
    /// Empty when the GGUF uses the split format below.
    pub kv_b: &'a [u8],
    /// MLA K up-projection (split format, Kimi K2.6):
    /// [kv_lora_rank → n_heads × qk_nope_head_dim].
    /// Empty when the GGUF uses the combined `kv_b` format. When non-empty,
    /// `kv_b` is ignored and `k_b` + `v_b` are dispatched as two matmuls.
    pub k_b: &'a [u8],
    /// MLA V up-projection (split format, Kimi K2.6):
    /// [kv_lora_rank → n_heads × v_head_dim]. Pairs with `k_b`.
    pub v_b: &'a [u8],
    /// Q down-projection: [hidden_dim → q_lora_rank] (compressed Q)
    pub q_a: &'a [u8],
    /// Q compression norm: [q_lora_rank]
    pub q_a_norm: &'a [f32],
    /// Q up-projection: [q_lora_rank → n_heads × (qk_nope_head_dim + qk_rope_head_dim)]
    pub q_b: &'a [u8],
    /// Output projection: [n_heads × v_head_dim → hidden_dim]
    pub output: &'a [u8],
    /// K norm for RoPE portion: [qk_rope_head_dim]
    pub k_norm: &'a [f32],
}

/// MLA single-token attention with compressed-latent KV cache.
///
/// Stores only `c_kv_normed` + post-RoPE `k_rope` per (layer, token);
/// re-expands the latent through W_kv_b at attention time to recover
/// per-head k_nope and v vectors. Numerically equivalent to the prior
/// decompressed-cache implementation (same matmul ordering on each
/// expansion, same dot-product/weighted-sum order) — bit-exact apart
/// from when the expansion fires.
///
/// # Flow
///
/// 1. KV compression: c_kv_rope = W_kv_a × hidden  [→ kv_lora_rank + rope_dim]
/// 2. Normalize c_kv: c_kv_normed = RMSNorm(c_kv)
/// 3. Normalize + RoPE the shared k_rope portion in-place
/// 4. **CACHE COMPRESSED**: store c_kv_normed + k_rope_post_rope in MlaKvCache
/// 5. Q compression: c_q = W_q_a × hidden  [→ q_lora_rank]
/// 6. Normalize Q: c_q_normed = RMSNorm(c_q)
/// 7. Q decompression: q_full = W_q_b × c_q_normed  [→ n_heads × (qk_nope+qk_rope)]
/// 8. RoPE on q_rope per head
/// 9. **Pass A** — Per cached token: expand c_kv_t through W_kv_b to get
///    per-head k_nope_h_t and v_h_t; compute score_h_t = q_h · k_h_t × scale
///    for every head; store into score_buf[h × max_tokens + t].
/// 10. Softmax per head across the first `total_tokens` scores.
/// 11. **Pass B** — Per cached token: re-expand c_kv_t; per head accumulate
///     attn_concat[h] += softmax_h_t × v_h_t.
/// 12. Output projection.
///
/// # Scratch contract (caller-allocated)
///
/// `score_buf` MUST have capacity ≥ `n_heads × max_tokens` — the
/// stride is inferred from `score_buf.len() / n_heads`. Old callers
/// sized this as `[max_tokens]` and must be updated.
#[allow(clippy::too_many_arguments)]
pub fn mla_attention_single_token(
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
    // Scratch buffers — caller-allocated
    c_kv_rope_buf: &mut [f32],   // [kv_lora_rank + qk_rope_head_dim]
    c_kv_norm_buf: &mut [f32],   // [kv_lora_rank]
    kv_decompressed: &mut [f32], // [n_heads × (qk_nope_head_dim + v_head_dim)]
    c_q_buf: &mut [f32],         // [q_lora_rank]
    c_q_norm_buf: &mut [f32],    // [q_lora_rank]
    q_decompressed: &mut [f32],  // [n_heads × (qk_nope_head_dim + qk_rope_head_dim)]
    _k_assembled: &mut [f32],    // unused — compressed cache no longer assembles K
    score_buf: &mut [f32],       // [n_heads × max_tokens] — see scratch contract
    _attn_head_buf: &mut [f32],  // unused — Pass B writes directly into attn_concat
    attn_concat: &mut [f32],     // [n_heads × v_head_dim]
    output: &mut [f32],          // [embedding_dim]
    scratch: &mut LinearScratch,
    attn_quant: zero_gguf_parser::GgmlType,
) -> Result<(), MlaError> {
    let kv_a_out_dim = kv_lora_rank + qk_rope_head_dim;
    let kv_b_out_dim = n_heads * (qk_nope_head_dim + v_head_dim);
    let q_b_out_dim = n_heads * (qk_nope_head_dim + qk_rope_head_dim);
    let total_k_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let total_tokens = token_offset + 1;
    let nope_v_per_head = qk_nope_head_dim + v_head_dim;
    // Kimi K2.6 ships attn_k_b / attn_v_b separately instead of the
    // combined attn_kv_b. When both halves are present, run two matmuls
    // into the same `kv_decompressed` buffer (interleaved per-head:
    // [k_h0; v_h0; k_h1; v_h1; ...]) to keep downstream layout
    // unchanged. The split branch falls through to the original
    // combined matmul whenever `k_b` is empty.
    let use_split_kv_b = !weights.k_b.is_empty() && !weights.v_b.is_empty();
    let k_nope_out_dim = n_heads * qk_nope_head_dim;
    let v_out_dim = n_heads * v_head_dim;
    // score_buf is striped per-head; stride is the caller's sizing.
    debug_assert!(
        score_buf.len() >= n_heads * total_tokens,
        "mla: score_buf must be sized n_heads × max_tokens"
    );
    let score_stride = score_buf.len() / n_heads;

    // ── Step 1: KV compression ──────────────────────────────────────
    let _ = linear_dispatch(
        hidden,
        weights.kv_a_mqa,
        c_kv_rope_buf,
        scratch,
        embedding_dim,
        kv_a_out_dim,
        attn_quant,
    );

    // ── Step 2: Normalize c_kv ──────────────────────────────────────
    {
        let (c_kv_raw, _) = c_kv_rope_buf.split_at(kv_lora_rank);
        rmsnorm(c_kv_raw, weights.kv_a_norm, c_kv_norm_buf, rms_eps);
    }

    // ── Step 3: Normalize + RoPE k_rope portion in-place ────────────
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

    // ── Step 5: Q compression ───────────────────────────────────────
    let _ = linear_dispatch(
        hidden,
        weights.q_a,
        c_q_buf,
        scratch,
        embedding_dim,
        q_lora_rank,
        attn_quant,
    );

    // ── Step 6: Normalize Q ─────────────────────────────────────────
    rmsnorm(c_q_buf, weights.q_a_norm, c_q_norm_buf, rms_eps);

    // ── Step 7: Decompress Q ────────────────────────────────────────
    let _ = linear_dispatch(
        c_q_norm_buf,
        weights.q_b,
        q_decompressed,
        scratch,
        q_lora_rank,
        q_b_out_dim,
        attn_quant,
    );

    // ── Step 8: RoPE on q_rope per head ─────────────────────────────
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

    // ── Pass A: Score computation ───────────────────────────────────
    // For each cached token, expand the latent through W_kv_b once and
    // compute the score for every head. Per-head scores land into
    // score_buf[h*score_stride + t]. `kv_decompressed` is reused as the
    // per-token expansion buffer.
    let scale = 1.0 / libm::sqrtf(total_k_head_dim as f32);

    for t in 0..total_tokens {
        let c_kv_t = kv_cache.get_c_kv(layer_idx, t)?;
        let k_rope_t = kv_cache.get_k_rope(layer_idx, t)?;

        if use_split_kv_b {
            let (k_part, v_part) = kv_decompressed.split_at_mut(k_nope_out_dim);
            let _ = linear_dispatch(
                c_kv_t,
                weights.k_b,
                k_part,
                scratch,
                kv_lora_rank,
                k_nope_out_dim,
                attn_quant,
            );
            let _ = linear_dispatch(
                c_kv_t,
                weights.v_b,
                &mut v_part[..v_out_dim],
                scratch,
                kv_lora_rank,
                v_out_dim,
                attn_quant,
            );
        } else {
            let _ = linear_dispatch(
                c_kv_t,
                weights.kv_b,
                kv_decompressed,
                scratch,
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

            let mut dot = 0.0f32;
            for i in 0..qk_nope_head_dim {
                dot += q_nope[i] * k_nope_h_t[i];
            }
            for i in 0..qk_rope_head_dim {
                dot += q_rope[i] * k_rope_t[i];
            }
            score_buf[h * score_stride + t] = dot * scale;
        }
    }

    // ── Pass A.5: Per-head softmax ──────────────────────────────────
    for h in 0..n_heads {
        let off = h * score_stride;
        crate::attention::softmax(&mut score_buf[off..off + total_tokens], total_tokens)
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
            let _ = linear_dispatch(
                c_kv_t,
                weights.k_b,
                k_part,
                scratch,
                kv_lora_rank,
                k_nope_out_dim,
                attn_quant,
            );
            let _ = linear_dispatch(
                c_kv_t,
                weights.v_b,
                &mut v_part[..v_out_dim],
                scratch,
                kv_lora_rank,
                v_out_dim,
                attn_quant,
            );
        } else {
            let _ = linear_dispatch(
                c_kv_t,
                weights.kv_b,
                kv_decompressed,
                scratch,
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
            for i in 0..v_head_dim {
                attn_concat[out_off + i] += weight * v_h_t[i];
            }
        }
    }

    // ── Output projection ───────────────────────────────────────────
    let o_in_dim = n_heads * v_head_dim;
    let _ = linear_dispatch(
        &attn_concat[..o_in_dim],
        weights.output,
        output,
        scratch,
        o_in_dim,
        embedding_dim,
        attn_quant,
    );

    Ok(())
}

/// Standard MHA attention weights for a DeepSeek-V2 MHA layer.
///
/// DeepSeek-V2 / Kimi K2.6 standard-attention layers project Q/K/V
/// directly from `hidden` (no LoRA compression). K head_dim equals
/// `qk_nope_head_dim + qk_rope_head_dim` (same as the MLA path); V
/// head_dim is `v_head_dim`. RoPE is applied to the last
/// `qk_rope_head_dim` dimensions of each Q-head and K-head.
pub struct MhaWeights<'a> {
    /// Q projection: [hidden → n_heads × (qk_nope_head_dim + qk_rope_head_dim)]
    pub q: &'a [u8],
    /// K projection: [hidden → n_kv_heads × (qk_nope_head_dim + qk_rope_head_dim)]
    pub k: &'a [u8],
    /// V projection: [hidden → n_kv_heads × v_head_dim]
    pub v: &'a [u8],
    /// Output projection: [n_heads × v_head_dim → hidden]
    pub output: &'a [u8],
}

/// Standard Multi-Head Attention single-token forward — DeepSeek-V2
/// MHA layer.
///
/// Algorithm:
///   1. Q/K/V projections via `linear_dispatch` (handles Q4_K/Q6_K/Q4_0/Q8_0).
///   2. RoPE on the last `qk_rope_head_dim` lanes of each Q-head and K-head.
///   3. Store (K, V) at `(mha_layer_idx, token_offset)` in the auxiliary
///      MHA cache.
///   4. Per-Q-head: dot Q · K_t for every cached token, scale by
///      `1/sqrt(qk_nope + qk_rope)`, softmax, weighted-sum V.
///   5. Output projection.
///
/// GQA is supported via `n_kv_heads ≤ n_heads` and `gqa_ratio = n_heads /
/// n_kv_heads`.
#[allow(clippy::too_many_arguments)]
pub fn mha_attention_single_token(
    hidden: &[f32],
    weights: &MhaWeights,
    mha_layer_idx: usize,
    token_offset: usize,
    n_heads: usize,
    n_kv_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    embedding_dim: usize,
    rope_freq_base: f32,
    kv_cache: &mut MhaKvCache,
    // Scratch — caller-allocated
    q_buf: &mut [f32],       // [n_heads × head_dim_qk]
    k_buf: &mut [f32],       // [n_kv_heads × head_dim_qk]
    v_buf: &mut [f32],       // [n_kv_heads × v_head_dim]
    score_buf: &mut [f32],   // [n_heads × max_tokens]
    attn_concat: &mut [f32], // [n_heads × v_head_dim]
    output: &mut [f32],      // [embedding_dim]
    scratch: &mut LinearScratch,
    attn_quant: zero_gguf_parser::GgmlType,
) -> Result<(), MlaError> {
    let head_dim_qk = qk_nope_head_dim + qk_rope_head_dim;
    let q_out_dim = n_heads * head_dim_qk;
    let k_out_dim = n_kv_heads * head_dim_qk;
    let v_out_dim = n_kv_heads * v_head_dim;
    let total_tokens = token_offset + 1;
    let half_rope = qk_rope_head_dim / 2;
    let gqa_ratio = if n_kv_heads > 0 {
        n_heads / n_kv_heads
    } else {
        1
    };
    let scale = 1.0 / libm::sqrtf(head_dim_qk as f32);

    debug_assert!(
        score_buf.len() >= n_heads * total_tokens,
        "mha: score_buf must be sized n_heads × max_tokens"
    );
    let score_stride = score_buf.len() / n_heads;

    // ── Step 1: Q/K/V projections ───────────────────────────────────
    let _ = linear_dispatch(
        hidden,
        weights.q,
        q_buf,
        scratch,
        embedding_dim,
        q_out_dim,
        attn_quant,
    );
    let _ = linear_dispatch(
        hidden,
        weights.k,
        k_buf,
        scratch,
        embedding_dim,
        k_out_dim,
        attn_quant,
    );
    let _ = linear_dispatch(
        hidden,
        weights.v,
        v_buf,
        scratch,
        embedding_dim,
        v_out_dim,
        attn_quant,
    );

    // ── Step 2: RoPE on the rope portion of every Q-head ────────────
    for h in 0..n_heads {
        let off = h * head_dim_qk + qk_nope_head_dim;
        for i in 0..half_rope {
            let freq = libm::powf(
                rope_freq_base,
                -2.0 * (i as f32) / (qk_rope_head_dim as f32),
            );
            let theta = token_offset as f32 * freq;
            let cos_t = libm::cosf(theta);
            let sin_t = libm::sinf(theta);
            let x0 = q_buf[off + i];
            let x1 = q_buf[off + i + half_rope];
            q_buf[off + i] = x0 * cos_t - x1 * sin_t;
            q_buf[off + i + half_rope] = x0 * sin_t + x1 * cos_t;
        }
    }
    // ── Step 3: RoPE on the rope portion of every K-head ────────────
    for h in 0..n_kv_heads {
        let off = h * head_dim_qk + qk_nope_head_dim;
        for i in 0..half_rope {
            let freq = libm::powf(
                rope_freq_base,
                -2.0 * (i as f32) / (qk_rope_head_dim as f32),
            );
            let theta = token_offset as f32 * freq;
            let cos_t = libm::cosf(theta);
            let sin_t = libm::sinf(theta);
            let x0 = k_buf[off + i];
            let x1 = k_buf[off + i + half_rope];
            k_buf[off + i] = x0 * cos_t - x1 * sin_t;
            k_buf[off + i + half_rope] = x0 * sin_t + x1 * cos_t;
        }
    }

    // ── Step 4: Cache K and V for this token ────────────────────────
    kv_cache.store_kv(
        mha_layer_idx,
        token_offset,
        &k_buf[..k_out_dim],
        &v_buf[..v_out_dim],
    )?;

    // ── Step 5: Scaled-dot-product attention (GQA broadcast) ────────
    let k_full = kv_cache.get_k_slice(mha_layer_idx, total_tokens)?;
    let v_full = kv_cache.get_v_slice(mha_layer_idx, total_tokens)?;
    let k_token_stride = n_kv_heads * head_dim_qk;
    let v_token_stride = n_kv_heads * v_head_dim;

    for q_h in 0..n_heads {
        let kv_h = if gqa_ratio > 0 { q_h / gqa_ratio } else { 0 };
        let q_off = q_h * head_dim_qk;
        let q_vec = &q_buf[q_off..q_off + head_dim_qk];

        // Scores
        for t in 0..total_tokens {
            let k_off = t * k_token_stride + kv_h * head_dim_qk;
            let k_vec = &k_full[k_off..k_off + head_dim_qk];
            let mut dot = 0.0f32;
            for i in 0..head_dim_qk {
                dot += q_vec[i] * k_vec[i];
            }
            score_buf[q_h * score_stride + t] = dot * scale;
        }

        // Per-head softmax
        let so = q_h * score_stride;
        softmax(&mut score_buf[so..so + total_tokens], total_tokens)
            .map_err(|_| MlaError::NumericalInstability)?;

        // Weighted V accumulation into attn_concat[q_h * v_head_dim..]
        let out_off = q_h * v_head_dim;
        for i in 0..v_head_dim {
            attn_concat[out_off + i] = 0.0;
        }
        for t in 0..total_tokens {
            let v_off = t * v_token_stride + kv_h * v_head_dim;
            let v_vec = &v_full[v_off..v_off + v_head_dim];
            let w = score_buf[q_h * score_stride + t];
            for i in 0..v_head_dim {
                attn_concat[out_off + i] += w * v_vec[i];
            }
        }
    }

    // ── Step 6: Output projection ───────────────────────────────────
    let o_in_dim = n_heads * v_head_dim;
    let _ = linear_dispatch(
        &attn_concat[..o_in_dim],
        weights.output,
        output,
        scratch,
        o_in_dim,
        embedding_dim,
        attn_quant,
    );

    Ok(())
}
