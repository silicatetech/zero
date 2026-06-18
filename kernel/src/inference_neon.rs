// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-E2/E3: Kernel-level NEON-accelerated inference dispatch.
//!
//! Per E2-Q1=(c) ratification: Pure kernel-level wrapper around
//! sacred-crate operators. Sacred crates UNCHANGED.
//!
//! E2: MLP + lm_head NEON acceleration (85 matmuls).
//! E3: Attention NEON acceleration (Q/K/V/O + score + weighted-sum,
//!     112 matmuls + score-computation + weighted-sum per token).
//!
//! Sacred scalar ops used: rmsnorm, rope, softmax, SiLU, embed_lookup,
//! KvCache::store_kv, KvCache::get_k_slice, KvCache::get_v_slice.
//!
//! Per Pillar 7: NO NEON intrinsics in this file. This module
//! calls into arch::aarch64::math::linear which contains NEON.
//! This file uses only cfg(feature), NOT cfg(target_arch).

use zero_llm_inference::attention::{softmax, AttentionError};
use zero_llm_inference::forward_pass::{embed_lookup_dispatch, ForwardPassError, N_LAYERS};
use zero_llm_inference::lm_head::{LmHeadError, VOCAB_SIZE_PADDED, VOCAB_SIZE_REAL};
use zero_llm_inference::ops::{rmsnorm, rope, LinearScratch};
use zero_llm_inference::{ForwardPassDispatch, KvCache, LayerWeights, RopeContext};

use crate::arch::aarch64::math::linear::{
    dot_product_f32_neon, linear_q4_0_neon, linear_q4k_neon, linear_q6k_neon, linear_q8_0_neon,
    weighted_add_f32_neon,
};

/// NEON-accelerated lm_head: RMSNorm + Q6_K linear + argmax.
///
/// Replaces sacred lm_head_argmax with NEON linear_q6k for the
/// 151,936 × 2048 output projection (biggest single matmul).
///
/// # Safety
/// Calls unsafe NEON intrinsics via linear_q6k_neon.
#[allow(clippy::too_many_arguments)]
pub unsafe fn lm_head_argmax_neon(
    final_hidden: &[f32],
    output_norm_weight: &[f32],
    output_weight: &[u8],
    output_quant: zero_gguf_parser::GgmlType,
    rms_eps: f32,
    embedding_dim: usize,
    norm_buf: &mut [f32],
    logits_buf: &mut [f32],
) -> Result<u32, LmHeadError> {
    // Step 1: Final RMSNorm (sacred scalar — small, not bottleneck)
    rmsnorm(final_hidden, output_norm_weight, norm_buf, rms_eps);

    // Step 2: LM head linear projection via NEON, native quant-aware.
    let _ = linear_dispatch_neon(
        norm_buf,
        output_weight,
        logits_buf,
        embedding_dim,
        VOCAB_SIZE_PADDED,
        output_quant,
    );

    // Step 3: Argmax (scalar — sequential scan, not vectorizable)
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx: usize = 0;
    for (i, &v) in logits_buf.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }

    if !max_val.is_finite() {
        return Err(LmHeadError::NonFiniteLogits);
    }

    if max_idx >= VOCAB_SIZE_REAL {
        return Err(LmHeadError::ArgmaxInPaddingRegion(max_idx));
    }

    Ok(max_idx as u32)
}

/// NEON-accelerated SwiGLU MLP.
///
/// Replicates sacred mlp_swiglu but uses NEON linear projections.
#[allow(clippy::too_many_arguments)]
unsafe fn mlp_swiglu_neon(
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
    // Step 1: Gate projection via NEON, native quant-aware.
    let _ = linear_dispatch_neon(
        input,
        gate_weight,
        gate_buf,
        embedding_dim,
        intermediate_dim,
        gate_quant,
    );

    // Step 2: Up projection via NEON, native quant-aware.
    let _ = linear_dispatch_neon(
        input,
        up_weight,
        up_buf,
        embedding_dim,
        intermediate_dim,
        up_quant,
    );

    // Step 3: SiLU on gate (scalar — element-wise, small)
    // SiLU(x) = x / (1 + exp(-x))
    for i in 0..intermediate_dim {
        gate_buf[i] = gate_buf[i] / (1.0 + libm::expf(-gate_buf[i]));
    }

    // Step 4: Element-wise multiply
    for i in 0..intermediate_dim {
        hidden_buf[i] = gate_buf[i] * up_buf[i];
    }

    // Step 5: Down projection via NEON, native quant-aware.
    let _ = linear_dispatch_neon(
        hidden_buf,
        down_weight,
        output,
        intermediate_dim,
        embedding_dim,
        down_quant,
    );
}

/// NEON-accelerated GQA attention for single token, single layer.
///
/// Sub-MP-E3: Replicates sacred gqa_attention_single_token using HAL
/// NEON for Q/K/V/O matmuls + score-computation (Q×Kᵀ) + weighted-sum
/// (attn×V). Sacred granular ops used for QK-norm (rmsnorm), RoPE,
/// softmax, KV-cache write/read.
///
/// # Safety
/// Calls unsafe NEON intrinsics via HAL math module.
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_single_token_neon<const HALF: usize>(
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
    attn_concat: &mut [f32],
    output: &mut [f32],
) -> Result<(), AttentionError> {
    let q_dim = n_q_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let gqa_ratio = n_q_heads / n_kv_heads;
    let scale = 1.0 / libm::sqrtf(head_dim as f32);
    let total_tokens = token_offset + 1;

    // ── Step 1: Q/K/V projections — HAL NEON, native quant-aware ──
    let _ = linear_dispatch_neon(normed_input, q_weight, q_buf, embedding_dim, q_dim, q_quant);
    let _ = linear_dispatch_neon(normed_input, k_weight, k_buf, embedding_dim, kv_dim, k_quant);
    let _ = linear_dispatch_neon(normed_input, v_weight, v_buf, embedding_dim, kv_dim, v_quant);

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

    // ── Step 3: RoPE — sacred scalar ─────────────────────────────
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

    // ── Step 4: KV-cache write — sacred public op ────────────────
    kv_cache.store_kv(layer_idx, token_offset, k_out, v_buf)?;

    // ── Step 5: Scaled Dot-Product Attention (GQA) ───────────────
    let k_full = kv_cache.get_k_slice(layer_idx, total_tokens)?;
    let v_full = kv_cache.get_v_slice(layer_idx, total_tokens)?;

    for q_h in 0..n_q_heads {
        let kv_h = q_h / gqa_ratio;
        let q_offset = q_h * head_dim;
        let q_vec = &q_out[q_offset..q_offset + head_dim];

        // Score computation: Q × Kᵀ — HAL NEON dot-product
        for t in 0..total_tokens {
            let k_offset = t * kv_dim + kv_h * head_dim;
            let k_vec = &k_full[k_offset..k_offset + head_dim];
            score_buf[t] = dot_product_f32_neon(q_vec, k_vec, head_dim) * scale;
        }

        // Softmax — sacred scalar
        softmax(&mut score_buf[..total_tokens], total_tokens)?;

        // Weighted-sum: attn × V — HAL NEON
        for i in 0..head_dim {
            attn_head_buf[i] = 0.0;
        }
        for t in 0..total_tokens {
            let v_offset = t * kv_dim + kv_h * head_dim;
            let v_vec = &v_full[v_offset..v_offset + head_dim];
            let weight = score_buf[t];
            weighted_add_f32_neon(weight, v_vec, attn_head_buf, head_dim);
        }

        // Copy to concatenated output
        let out_offset = q_h * head_dim;
        attn_concat[out_offset..out_offset + head_dim].copy_from_slice(&attn_head_buf[..head_dim]);
    }

    // ── Step 6: O projection — HAL NEON, native quant-aware ──────
    let _ = linear_dispatch_neon(attn_concat, o_weight, output, q_dim, embedding_dim, o_quant);

    Ok(())
}

/// NEON-accelerated single-token forward pass through all 28 layers.
///
/// Sub-MP-E2/E3: Replicates sacred forward_single_token using NEON
/// for ALL matmul-heavy steps (attention Q/K/V/O + score + weighted-sum
/// + MLP gate/up/down + lm_head). Sacred scalar ops preserved for
/// rmsnorm, rope, softmax, SiLU, embed_lookup, KV-cache write/read.
///
/// # Safety
/// Calls unsafe NEON intrinsics transitively.
#[allow(clippy::too_many_arguments)]
pub unsafe fn forward_single_token_neon<const HALF: usize>(
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
    // Scratch buffers (same signature as sacred)
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

    // ── 28-layer chain ────────────────────────────────────────────
    for layer_idx in 0..N_LAYERS {
        let layer = &layers[layer_idx];

        // ─ Attention sub-block (E3: NEON-accelerated) ─
        rmsnorm(hidden, layer.attn_norm, norm_buf, rms_eps);

        // Sub-MP-E3: Full NEON attention (replaces E2 sacred fallback)
        gqa_attention_single_token_neon::<HALF>(
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
        )
        .map_err(ForwardPassError::Attention)?;

        // Residual: hidden += attn_out
        for i in 0..embedding_dim {
            hidden[i] += attn_out[i];
        }

        // ─ MLP sub-block (E2: NEON-accelerated) ─
        rmsnorm(hidden, layer.ffn_norm, norm_buf, rms_eps);

        mlp_swiglu_neon(
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

        // Residual: hidden += mlp_out
        for i in 0..embedding_dim {
            hidden[i] += mlp_out[i];
        }

        // Sub-MP-F2 Task C: Layer progress callback (kernel-level, NOT sacred crate).
        // Cost: ~1 fill_rect per layer = sub-millisecond. 28 calls/token total.
        // Sub-MP-G1 M7: cfg-gated explicit (this file already aarch64+neon only,
        // but cfg makes LFB dependency explicit per Pillar 7 purity).
        #[cfg(target_arch = "aarch64")]
        crate::lfb::layer_progress::set_layer(layer_idx + 1);
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// DeepSeek-V2/V3 / Kimi K2.6 — MoE + MLA NEON dispatch (MP-MoE-T2)
// ─────────────────────────────────────────────────────────────────
//
// Native NEON implementations of the deepseek2 hot path. The matmul
// hot path (MoE expert gate/up/down + MLA kv_a/kv_b/q_a/q_b/output)
// routes through `linear_dispatch_neon`, which dispatches per the
// per-tensor quant. The router (gate_inp, F32 weights), RMSNorm,
// softmax, and RoPE remain scalar — small reductions where NEON
// would not amortise the load/store cost.
//
// **Bit-exactness vs scalar:** NOT guaranteed for DeepSeek2 / Kimi
// K2.6. There is no β-anchor for this model family; the sacred
// chain is anchored exclusively against Qwen3-1.7B (token=25,
// logit_bits=0x414a6497), which never enters this path.

use zero_llm_inference::{
    mha_attention_single_token,
    moe::{
        expert_q4k_bytes, expert_quant_bytes, moe_route_f32, slice_expert_weight, MoeRoutingMode,
    },
    AttnType, Deepseek2LayerWeights, Deepseek2Scratch as LibScratch, MhaKvCache, MhaWeights,
    MlaError, MlaKvCache, MlaWeights, MlpType,
};

/// DeepSeek2 forward-pass error (mirrors the AVX-512 variant).
#[derive(Debug)]
pub enum Deepseek2ForwardError {
    Mla(MlaError),
}

impl From<MlaError> for Deepseek2ForwardError {
    fn from(e: MlaError) -> Self {
        Deepseek2ForwardError::Mla(e)
    }
}

/// Errors returned by [`linear_dispatch_neon`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearDispatchNeonError {
    UnsupportedQuant(zero_gguf_parser::GgmlType),
}

/// Cross-quant linear projection through NEON kernels. Drop-in
/// replacement for `zero_llm_inference::ops::linear_dispatch` on
/// aarch64 + neon-acceleration builds.
///
/// Currently routes:
///   * `Q4K`  → `linear_q4k_neon`
///   * `Q6K`  → `linear_q6k_neon`
///   * `Q4_0` → `linear_q4_0_neon`
///   * `Q8_0` → `linear_q8_0_neon`
///
/// # Safety
/// Caller must guarantee NEON is available and the slice dimensions
/// match the quant's block layout.
#[allow(dead_code)]
pub unsafe fn linear_dispatch_neon(
    x: &[f32],
    w: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
    quant: zero_gguf_parser::GgmlType,
) -> Result<(), LinearDispatchNeonError> {
    use zero_gguf_parser::GgmlType;
    match quant {
        GgmlType::Q4K => {
            linear_q4k_neon(x, w, out, in_dim, out_dim);
            Ok(())
        }
        GgmlType::Q6K => {
            linear_q6k_neon(x, w, out, in_dim, out_dim);
            Ok(())
        }
        GgmlType::Q4_0 => {
            linear_q4_0_neon(x, w, out, in_dim, out_dim);
            Ok(())
        }
        GgmlType::Q8_0 => {
            linear_q8_0_neon(x, w, out, in_dim, out_dim);
            Ok(())
        }
        other => {
            // Fail-fast: every call site uses `let _ = ...` so a silent
            // Err would propagate as garbage output. Panic instead so the
            // kernel panic handler prints the offending quant type and a
            // call stack pointing at the projection that triggered it.
            panic!(
                "linear_dispatch_neon: unsupported quant type {}",
                other as u32
            );
        }
    }
}

/// NEON-accelerated single-expert SwiGLU MLP — gate + up + down via
/// `linear_dispatch_neon`, SiLU + element-wise multiply scalar.
///
/// # Safety
/// NEON must be enabled at runtime (CPACR_EL1.FPEN=3).
#[allow(clippy::too_many_arguments)]
unsafe fn expert_swiglu_neon(
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
    let _ = linear_dispatch_neon(
        input,
        gate_w,
        gate_buf,
        embedding_dim,
        expert_intermediate,
        expert_quant,
    );
    let _ = linear_dispatch_neon(
        input,
        up_w,
        up_buf,
        embedding_dim,
        expert_intermediate,
        expert_quant,
    );

    // SiLU(gate) * up → hidden (scalar — small per-element loop).
    for i in 0..expert_intermediate {
        let silu = gate_buf[i] / (1.0 + libm::expf(-gate_buf[i]));
        hidden_buf[i] = silu * up_buf[i];
    }

    let _ = linear_dispatch_neon(
        hidden_buf,
        down_w,
        output,
        expert_intermediate,
        embedding_dim,
        expert_quant,
    );
}

/// NEON MoE FFN — router + shared expert + top-K experts +
/// weighted-accumulation. Router stays scalar; per-expert SwiGLU
/// runs through the NEON hot path.
///
/// # Safety
/// NEON must be enabled at runtime.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_ffn_neon(
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
    expert_quant: zero_gguf_parser::GgmlType,
    routing_mode: MoeRoutingMode,
    expert_weight_scale: f32,
) {
    assert!(top_k > 0, "moe_ffn_neon: top_k must be > 0");
    assert!(
        top_k <= n_experts,
        "moe_ffn_neon: top_k must be <= n_experts"
    );

    // Step 1: Route — scalar.
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

    // Step 2: Shared expert (writes directly to accumulator).
    expert_swiglu_neon(
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

    // Step 3: Top-K experts — weighted accumulation.
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

        expert_swiglu_neon(
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

        weighted_add_f32_neon(weight, expert_out_buf, output, embedding_dim);
    }
}

/// NEON MLA attention single-token. Replaces the 5 scalar
/// `linear_dispatch` calls (kv_a, kv_b, q_a, q_b, output) with
/// `linear_dispatch_neon`; RMSNorm / RoPE / softmax stay scalar.
///
/// # Safety
/// NEON must be enabled at runtime.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mla_attention_neon(
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
    attn_quant: zero_gguf_parser::GgmlType,
) -> Result<(), MlaError> {
    let kv_a_out_dim = kv_lora_rank + qk_rope_head_dim;
    let kv_b_out_dim = n_heads * (qk_nope_head_dim + v_head_dim);
    let q_b_out_dim = n_heads * (qk_nope_head_dim + qk_rope_head_dim);
    let total_k_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let total_tokens = token_offset + 1;
    let nope_v_per_head = qk_nope_head_dim + v_head_dim;
    // Split attn_k_b / attn_v_b path mirrors the scalar mla.rs dispatch
    // (Kimi K2.6).
    let use_split_kv_b = !weights.k_b.is_empty() && !weights.v_b.is_empty();
    let k_nope_out_dim = n_heads * qk_nope_head_dim;
    let v_out_dim = n_heads * v_head_dim;
    debug_assert!(
        score_buf.len() >= n_heads * total_tokens,
        "mla_attention_neon: score_buf must be sized n_heads × max_tokens"
    );
    let score_stride = score_buf.len() / n_heads;
    let _ = k_assembled;
    let _ = attn_head_buf;

    // ── Step 1: KV compression — NEON matmul ────────────────────────
    let _ = linear_dispatch_neon(
        hidden,
        weights.kv_a_mqa,
        c_kv_rope_buf,
        embedding_dim,
        kv_a_out_dim,
        attn_quant,
    );

    // ── Step 2: Normalize c_kv — scalar ─────────────────────────────
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

    // ── Step 5: Q compression — NEON matmul ─────────────────────────
    let _ = linear_dispatch_neon(
        hidden,
        weights.q_a,
        c_q_buf,
        embedding_dim,
        q_lora_rank,
        attn_quant,
    );

    // ── Step 6: Normalize Q — scalar ────────────────────────────────
    rmsnorm(c_q_buf, weights.q_a_norm, c_q_norm_buf, rms_eps);

    // ── Step 7: Decompress Q — NEON matmul ──────────────────────────
    let _ = linear_dispatch_neon(
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
    let scale = 1.0 / libm::sqrtf(total_k_head_dim as f32);

    for t in 0..total_tokens {
        let c_kv_t = kv_cache.get_c_kv(layer_idx, t)?;
        let k_rope_t = kv_cache.get_k_rope(layer_idx, t)?;

        if use_split_kv_b {
            let (k_part, v_part) = kv_decompressed.split_at_mut(k_nope_out_dim);
            let _ = linear_dispatch_neon(
                c_kv_t,
                weights.k_b,
                k_part,
                kv_lora_rank,
                k_nope_out_dim,
                attn_quant,
            );
            let _ = linear_dispatch_neon(
                c_kv_t,
                weights.v_b,
                &mut v_part[..v_out_dim],
                kv_lora_rank,
                v_out_dim,
                attn_quant,
            );
        } else {
            let _ = linear_dispatch_neon(
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

            let dot_nope = dot_product_f32_neon(q_nope, k_nope_h_t, qk_nope_head_dim);
            let dot_rope = dot_product_f32_neon(q_rope, k_rope_t, qk_rope_head_dim);
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
            let _ = linear_dispatch_neon(
                c_kv_t,
                weights.k_b,
                k_part,
                kv_lora_rank,
                k_nope_out_dim,
                attn_quant,
            );
            let _ = linear_dispatch_neon(
                c_kv_t,
                weights.v_b,
                &mut v_part[..v_out_dim],
                kv_lora_rank,
                v_out_dim,
                attn_quant,
            );
        } else {
            let _ = linear_dispatch_neon(
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
            weighted_add_f32_neon(
                weight,
                v_h_t,
                &mut attn_concat[out_off..out_off + v_head_dim],
                v_head_dim,
            );
        }
    }

    // ── Output projection — NEON matmul ─────────────────────────────
    let o_in_dim = n_heads * v_head_dim;
    let _ = linear_dispatch_neon(
        &attn_concat[..o_in_dim],
        weights.output,
        output,
        o_in_dim,
        embedding_dim,
        attn_quant,
    );

    Ok(())
}

/// 61-layer single-token forward pass for Kimi K2.6 / DeepSeek-V2/V3
/// — NEON variant. Mirrors `forward_single_token_deepseek2_avx512`
/// in algorithm shape; bit-exactness vs scalar is not required (no
/// β-anchor for this model family).
///
/// # Safety
/// NEON must be enabled at runtime. All scratch buffers in `s` must
/// be sized for the dimensions implied by the supplied configuration.
#[allow(clippy::too_many_arguments)]
pub unsafe fn forward_single_token_deepseek2_neon(
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
    // Per-quant embedding lookup. Mirrors the AVX-512 deepseek2 fix —
    // Kimi K2.6's Q8_0 `token_embd.weight` cannot go through the legacy
    // Q4_K-hardcoded `embed_lookup`. Qwen3's NEON path still calls
    // `embed_lookup` directly (preserved upstream).
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

                mla_attention_neon(
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
                    layer.attn_quant,
                )?;
            }
            AttnType::Mha => {
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
                // Per-tensor quant dispatch. Kimi K2.6 layer-0 dense is
                // Q4_0; the Q4_K-only `mlp_swiglu_neon` (kept for the
                // Qwen3 β-anchor) would misread those bytes. Inline the
                // three matmuls through `linear_dispatch_neon` driven by
                // the layer's actual `ffn_dense_quant`.
                let _ = linear_dispatch_neon(
                    s.norm_buf,
                    layer.ffn_gate,
                    s.gate_buf,
                    embedding_dim,
                    intermediate_dim,
                    layer.ffn_dense_quant,
                );
                let _ = linear_dispatch_neon(
                    s.norm_buf,
                    layer.ffn_up,
                    s.up_buf,
                    embedding_dim,
                    intermediate_dim,
                    layer.ffn_dense_quant,
                );
                // Scalar SiLU(gate) × up — small (intermediate_dim) and
                // there is no NEON-fused helper today.
                for i in 0..intermediate_dim {
                    let g = s.gate_buf[i];
                    let silu = g / (1.0 + libm::expf(-g));
                    s.hidden_mlp_buf[i] = silu * s.up_buf[i];
                }
                let _ = linear_dispatch_neon(
                    s.hidden_mlp_buf,
                    layer.ffn_down,
                    s.mlp_out,
                    intermediate_dim,
                    embedding_dim,
                    layer.ffn_dense_quant,
                );
            }
            MlpType::MoE => {
                // Zero accumulator — moe_ffn_neon writes shared expert
                // to `output` then accumulates weighted top-K experts.
                for v in s.mlp_out.iter_mut() {
                    *v = 0.0;
                }
                moe_ffn_neon(
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
