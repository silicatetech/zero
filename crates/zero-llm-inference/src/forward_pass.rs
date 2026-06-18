// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full 28-layer single-token forward pass for Qwen3-1.7B.
//!
//! Per ADR-029 v2: forward_single_token chains 28 transformer layers
//! (attention + SwiGLU MLP + residuals) over a single input token.
//!
//! Design:
//! - GGUF-FORTRAN-shape-immune: all dimensions from ModelConfig (D13)
//! - Per-layer ffn_down static dispatch via ForwardPassDispatch (D14)
//! - Zero allocation: all scratch buffers caller-provided (Pillar 1)
//! - no_std compatible

use crate::attention::{gqa_attention_single_token_dispatch, AttentionError};
use crate::kv_cache::{KvCache, KvCacheError, MhaKvCache, MlaKvCache};
use crate::mla::{
    mha_attention_single_token, mla_attention_single_token, MhaWeights, MlaError, MlaWeights,
};
use crate::moe::{moe_ffn, MoeRoutingMode};
use crate::ops::{linear_q4k, linear_q6k, rmsnorm, LinearScratch, RopeContext};
use zero_gguf_parser::GgmlType;

/// Number of transformer layers in Qwen3-1.7B.
///
/// Retained for the existing Qwen3 dense path. The DeepSeek2 / Kimi K2.6 path
/// uses a runtime layer count (e.g. 61) — see `Deepseek2LayerWeights` and
/// `MlpType`. Callers on the deepseek2 path must NOT depend on this constant.
pub const N_LAYERS: usize = 28;

/// FFN sub-block type for per-layer dispatch on DeepSeek-V2/V3 / Kimi K2.6.
///
/// Kimi K2.6 mixes dense FFN layers (typically the first 1–3 layers) with
/// MoE layers (remainder). Built once at boot from TensorIndex (presence of
/// `ffn_gate_exps` tensor → MoE; otherwise Dense).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpType {
    /// Standard SwiGLU MLP (single expert, full intermediate dim).
    Dense,
    /// Mixture-of-Experts FFN (top-K experts + shared expert).
    MoE,
}

/// Attention sub-block type for per-layer dispatch on DeepSeek-V2/V3 / Kimi K2.6.
///
/// DeepSeek-V2/V3 / Kimi K2.6 ships with Multi-Head Latent Attention (MLA)
/// for the bulk of layers, but some builds (notably Kimi K2.6 layer 0) use
/// standard Multi-Head Attention (MHA) with `attn_q.weight` / `attn_k.weight`
/// / `attn_v.weight` instead of the MLA tensors (`attn_kv_a_mqa`,
/// `attn_q_a`, etc.). Detection is via presence of `attn_kv_a_mqa.weight`
/// → MLA; otherwise → MHA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnType {
    /// Multi-head Latent Attention (LoRA-compressed Q/KV). DeepSeek-V2/V3
    /// / Kimi K2.6 main path.
    Mla,
    /// Standard Multi-Head Attention (direct Q/K/V projections from hidden).
    /// Used by some DeepSeek2 builds for the first transformer layer.
    Mha,
}

/// Per-layer weights for DeepSeek-V2/V3 / Kimi K2.6.
///
/// Combines MLA attention weights and either Dense FFN or MoE FFN weights.
/// All slices are zero-copy references into the GGUF mmap'd region (Pillar 1).
///
/// MoE-specific fields (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`,
/// `ffn_gate_inp`, `ffn_*_shexp`) are only populated when `mlp_type == MoE`.
/// Dense-specific fields (`ffn_gate`, `ffn_up`, `ffn_down`) are only populated
/// when `mlp_type == Dense`. Unused fields hold empty slices.
pub struct Deepseek2LayerWeights<'a> {
    pub mlp_type: MlpType,
    /// Attention sub-block type for this layer. When `Mha`, the MLA-
    /// specific fields (`attn_kv_a_mqa`, `attn_kv_a_norm`, `attn_kv_b`,
    /// `attn_k_b`, `attn_v_b`, `attn_q_a`, `attn_q_a_norm`, `attn_q_b`)
    /// hold empty slices and the `attn_q_mha` / `attn_k_mha` /
    /// `attn_v_mha` tensors below carry the standard MHA projections.
    pub attn_type: AttnType,
    /// Per-MHA-layer slot in the auxiliary `MhaKvCache`. Valid only when
    /// `attn_type == Mha`; for MLA layers this is ignored. Maps
    /// transformer-layer-idx → contiguous cache slot so we don't pay an
    /// MHA-sized KV slot for every layer when only one or two layers use
    /// standard attention.
    pub mha_layer_idx: u32,

    // ── Attention norm + MLA projections ──────────────────────────
    pub attn_norm: &'a [f32],
    /// MLA KV down-projection: [hidden → kv_lora_rank + qk_rope_head_dim]
    pub attn_kv_a_mqa: &'a [u8],
    /// MLA KV compression norm: [kv_lora_rank]
    pub attn_kv_a_norm: &'a [f32],
    /// MLA KV up-projection (combined format):
    /// [kv_lora_rank → n_heads × (qk_nope + v_head)].
    /// Empty when the GGUF ships the split format below.
    pub attn_kv_b: &'a [u8],
    /// MLA K up-projection (split format, Kimi K2.6):
    /// [kv_lora_rank → n_heads × qk_nope_head_dim]. Empty when combined.
    pub attn_k_b: &'a [u8],
    /// MLA V up-projection (split format, Kimi K2.6):
    /// [kv_lora_rank → n_heads × v_head_dim]. Empty when combined.
    pub attn_v_b: &'a [u8],
    /// MLA Q down-projection: [hidden → q_lora_rank]
    pub attn_q_a: &'a [u8],
    /// MLA Q compression norm: [q_lora_rank]
    pub attn_q_a_norm: &'a [f32],
    /// MLA Q up-projection: [q_lora_rank → n_heads × (qk_nope + qk_rope)]
    pub attn_q_b: &'a [u8],
    /// Output projection: [n_heads × v_head → hidden]
    pub attn_output: &'a [u8],
    /// K-RoPE norm: [qk_rope_head_dim]
    pub attn_k_norm: &'a [f32],

    // ── Standard MHA projections (when attn_type == Mha) ──────────
    /// MHA Q projection: [hidden → n_heads × (qk_nope_head_dim + qk_rope_head_dim)].
    /// Empty when `attn_type == Mla`.
    pub attn_q_mha: &'a [u8],
    /// MHA K projection: [hidden → n_kv_heads × (qk_nope_head_dim + qk_rope_head_dim)].
    /// Empty when `attn_type == Mla`. DeepSeek-V2 / Kimi K2.6 MHA uses the
    /// same `head_dim_qk = qk_nope + qk_rope` as the MLA path.
    pub attn_k_mha: &'a [u8],
    /// MHA V projection: [hidden → n_kv_heads × v_head_dim].
    /// Empty when `attn_type == Mla`.
    pub attn_v_mha: &'a [u8],

    // ── FFN norm ──────────────────────────────────────────────────
    pub ffn_norm: &'a [f32],

    // ── Dense FFN (when mlp_type == Dense) ────────────────────────
    pub ffn_gate: &'a [u8],
    pub ffn_up: &'a [u8],
    pub ffn_down: &'a [u8],

    // ── MoE FFN (when mlp_type == MoE) ────────────────────────────
    /// Router weights (F32): [n_experts × hidden]
    pub ffn_gate_inp: &'a [f32],
    /// Per-expert router bias (F32): [n_experts], empty when the GGUF
    /// does not ship `ffn_gate_inp_bias.weight`.
    /// SPEC: Kimi K2.6 / DeepSeek-V3 ship per-expert load-balancing bias
    /// — added to router scores before top-K selection.
    pub ffn_gate_inp_bias: &'a [f32],
    /// Stacked expert gate Q4_K: [n_experts × expert_intermediate × hidden]
    pub ffn_gate_exps: &'a [u8],
    /// Stacked expert up Q4_K: [n_experts × expert_intermediate × hidden]
    pub ffn_up_exps: &'a [u8],
    /// Stacked expert down Q4_K: [n_experts × hidden × expert_intermediate]
    pub ffn_down_exps: &'a [u8],
    /// Shared-expert gate Q4_K: [expert_intermediate × hidden]
    pub ffn_gate_shexp: &'a [u8],
    /// Shared-expert up Q4_K: [expert_intermediate × hidden]
    pub ffn_up_shexp: &'a [u8],
    /// Shared-expert down Q4_K: [hidden × expert_intermediate]
    pub ffn_down_shexp: &'a [u8],

    // ── Per-tensor-group quantisation ─────────────────────────────
    //
    // Kimi K2.6 is natively int4 and ships its GGUF as Q4_0; DeepSeek-V3
    // and similar models ship as Q4_K. Per-tensor quant is read from
    // `GgufTensorInfo.tensor_type` at boot and passed in here so the
    // forward pass dispatches to the right matmul kernel.
    //
    // We assume a single quant per group: all 5 MLA projections share a
    // quant (`attn_quant`), all expert weights share a quant
    // (`expert_quant`), the dense FFN weights share a quant
    // (`ffn_dense_quant`). For dense layers, set `expert_quant` to a
    // placeholder (e.g. the same as `ffn_dense_quant`); the deepseek2
    // forward pass only reads it when `mlp_type == MoE`. Likewise
    // `ffn_dense_quant` is unused for MoE layers.
    /// Quant of `attn_kv_a_mqa`, `attn_kv_b`, `attn_q_a`, `attn_q_b`,
    /// `attn_output`. Read by the MLA path through `linear_dispatch`.
    pub attn_quant: zero_gguf_parser::GgmlType,
    /// Quant of `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`,
    /// `ffn_*_shexp`. Read by the MoE path.
    pub expert_quant: zero_gguf_parser::GgmlType,
    /// Quant of `ffn_gate`, `ffn_up`, `ffn_down` for Dense FFN layers.
    /// Ignored when `mlp_type == MoE`.
    pub ffn_dense_quant: zero_gguf_parser::GgmlType,
}

/// Per-layer quantization type for asymmetric quant tensors.
///
/// Sub-MP-C3/C4 findings: ffn_down AND attn_v use Q4_K OR Q6_K
/// (mixed, iMatrix-driven by llama.cpp sensitivity analysis).
///
/// This enum is computed ONCE at boot via TensorIndex iteration.
/// Hot loop uses it for static-typed dispatch — no runtime type checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnDownQuant {
    Q4K,
    Q6K,
}

/// Per-layer dispatch table. Built at kernel-init via TensorIndex.
pub struct ForwardPassDispatch {
    /// Quant pathway for `token_embd.weight`.
    pub token_embd_quant: GgmlType,
    /// Per-layer quant pathways for attention projections.
    pub attn_q_quant: [GgmlType; N_LAYERS],
    pub attn_k_quant: [GgmlType; N_LAYERS],
    pub attn_v_tensor_quant: [GgmlType; N_LAYERS],
    pub attn_o_quant: [GgmlType; N_LAYERS],
    /// Per-layer quant pathways for MLP projections.
    pub ffn_gate_quant: [GgmlType; N_LAYERS],
    pub ffn_up_quant: [GgmlType; N_LAYERS],
    pub ffn_down_tensor_quant: [GgmlType; N_LAYERS],
    /// Index 0..N_LAYERS: dequant pathway for ffn_down.
    pub ffn_down_quant: [FfnDownQuant; N_LAYERS],
    /// Index 0..N_LAYERS: dequant pathway for attn_v.
    pub attn_v_quant: [FfnDownQuant; N_LAYERS],
}

impl Default for ForwardPassDispatch {
    fn default() -> Self {
        Self {
            token_embd_quant: GgmlType::Q4K,
            attn_q_quant: [GgmlType::Q4K; N_LAYERS],
            attn_k_quant: [GgmlType::Q4K; N_LAYERS],
            attn_v_tensor_quant: [GgmlType::Q4K; N_LAYERS],
            attn_o_quant: [GgmlType::Q4K; N_LAYERS],
            ffn_gate_quant: [GgmlType::Q4K; N_LAYERS],
            ffn_up_quant: [GgmlType::Q4K; N_LAYERS],
            ffn_down_tensor_quant: [GgmlType::Q4K; N_LAYERS],
            ffn_down_quant: [FfnDownQuant::Q4K; N_LAYERS],
            attn_v_quant: [FfnDownQuant::Q4K; N_LAYERS],
        }
    }
}

/// Per-layer weights (zero-copy references into GGUF mmap).
pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub attn_q: &'a [u8],
    pub attn_k: &'a [u8],
    pub attn_v: &'a [u8],
    pub attn_o: &'a [u8],
    pub attn_q_norm: &'a [f32],
    pub attn_k_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub ffn_gate: &'a [u8],
    pub ffn_up: &'a [u8],
    pub ffn_down: &'a [u8],
}

/// Errors from forward-pass operations.
#[derive(Debug)]
pub enum ForwardPassError {
    Attention(AttentionError),
    KvCache(KvCacheError),
    Embedding(EmbedDispatchError),
}

impl From<AttentionError> for ForwardPassError {
    fn from(e: AttentionError) -> Self {
        ForwardPassError::Attention(e)
    }
}

impl From<KvCacheError> for ForwardPassError {
    fn from(e: KvCacheError) -> Self {
        ForwardPassError::KvCache(e)
    }
}

impl From<EmbedDispatchError> for ForwardPassError {
    fn from(e: EmbedDispatchError) -> Self {
        ForwardPassError::Embedding(e)
    }
}

/// Embedding lookup: dequantize ONE row from token_embd (Q4_K).
///
/// Pillar-1-conform: dequant only the needed row, not entire 151_936×2048 table.
///
/// Q4_K row size: embedding_dim / 256 blocks × 144 bytes/block.
pub fn embed_lookup(
    token_id: u32,
    token_embd_weight: &[u8],
    embedding_dim: usize,
    output: &mut [f32],
) {
    let blocks_per_row = embedding_dim / 256;
    let bytes_per_row = blocks_per_row * 144;
    let row_offset = (token_id as usize) * bytes_per_row;
    let row_bytes = &token_embd_weight[row_offset..row_offset + bytes_per_row];

    zero_gguf_parser::dequant::dequant_q4k_row(row_bytes, output, blocks_per_row);
}

/// Errors returned by [`embed_lookup_dispatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedDispatchError {
    /// `token_embd` is quantised with a `GgmlType` this kernel has no
    /// dequantiser for. Caller should bail at boot rather than render
    /// garbage hidden state.
    UnsupportedQuant(zero_gguf_parser::GgmlType),
    /// `embedding_dim` is not a multiple of the quant's block size.
    /// All shipped GGUF dimensions satisfy this; the check catches
    /// corruption / pathological configs at boot.
    DimMisalignment,
}

/// Single-token embedding lookup with per-quant dispatch.
///
/// Mirrors [`embed_lookup`] but switches on the `token_embd` tensor's
/// `GgmlType` at runtime. Used by the deepseek2 / Kimi K2.6 forward
/// pass; the Qwen3 path keeps calling the Q4_K-direct [`embed_lookup`]
/// so its β-anchor numerics stay bit-identical.
///
/// Currently dispatches on `Q4K`, `Q4_0`, and `Q8_0`. Block sizes
/// differ (Q4_K = 256 / 144 B; Q4_0 = 32 / 18 B; Q8_0 = 32 / 34 B),
/// so the row stride depends on the quant. **Q8_0 is critical** for
/// bartowski-style Q4_0 builds where `token_embd.weight` stays at
/// Q8_0 because the embedding lookup is too lossy at 4 bits.
pub fn embed_lookup_dispatch(
    token_id: u32,
    token_embd_weight: &[u8],
    embedding_dim: usize,
    output: &mut [f32],
    quant: zero_gguf_parser::GgmlType,
) -> Result<(), EmbedDispatchError> {
    use zero_gguf_parser::{
        dequant::{
            dequant_q4_0_row, dequant_q4k_row, dequant_q8_0_row, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE,
            Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE,
        },
        GgmlType,
    };
    match quant {
        GgmlType::Q4K => {
            if embedding_dim % Q4K_BLOCK_SIZE != 0 {
                return Err(EmbedDispatchError::DimMisalignment);
            }
            let blocks_per_row = embedding_dim / Q4K_BLOCK_SIZE;
            let bytes_per_row = blocks_per_row * Q4K_BLOCK_BYTES;
            let row_offset = (token_id as usize) * bytes_per_row;
            dequant_q4k_row(
                &token_embd_weight[row_offset..row_offset + bytes_per_row],
                output,
                blocks_per_row,
            );
            Ok(())
        }
        GgmlType::Q4_0 => {
            if embedding_dim % Q4_0_BLOCK_SIZE != 0 {
                return Err(EmbedDispatchError::DimMisalignment);
            }
            let blocks_per_row = embedding_dim / Q4_0_BLOCK_SIZE;
            let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
            let row_offset = (token_id as usize) * bytes_per_row;
            dequant_q4_0_row(
                &token_embd_weight[row_offset..row_offset + bytes_per_row],
                output,
                blocks_per_row,
            );
            Ok(())
        }
        GgmlType::Q8_0 => {
            if embedding_dim % Q8_0_BLOCK_SIZE != 0 {
                return Err(EmbedDispatchError::DimMisalignment);
            }
            let blocks_per_row = embedding_dim / Q8_0_BLOCK_SIZE;
            let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;
            let row_offset = (token_id as usize) * bytes_per_row;
            dequant_q8_0_row(
                &token_embd_weight[row_offset..row_offset + bytes_per_row],
                output,
                blocks_per_row,
            );
            Ok(())
        }
        other => Err(EmbedDispatchError::UnsupportedQuant(other)),
    }
}

/// SwiGLU MLP forward-pass for a single token.
///
/// Formula:
///   gate = gate_w @ input          // [intermediate_dim]
///   up   = up_w @ input            // [intermediate_dim]
///   silu_gate = SiLU(gate) = gate * sigmoid(gate)
///   hidden = silu_gate * up        // element-wise
///   output = down_w @ hidden       // [embedding_dim]
///
/// `down_quant_is_q6k` selects Q6_K vs Q4_K for down projection
/// (per ForwardPassDispatch, computed at boot).
#[allow(clippy::too_many_arguments)]
pub fn mlp_swiglu(
    input: &[f32],
    gate_weight: &[u8],
    up_weight: &[u8],
    down_weight: &[u8],
    down_quant_is_q6k: bool,
    embedding_dim: usize,
    intermediate_dim: usize,
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    hidden_buf: &mut [f32],
    scratch: &mut LinearScratch,
    output: &mut [f32],
) {
    // Step 1: Gate projection (Q4_K)
    linear_q4k(
        input,
        gate_weight,
        gate_buf,
        scratch,
        embedding_dim,
        intermediate_dim,
    );

    // Step 2: Up projection (Q4_K)
    linear_q4k(
        input,
        up_weight,
        up_buf,
        scratch,
        embedding_dim,
        intermediate_dim,
    );

    // Step 3: SiLU on gate (in-place into gate_buf)
    // SiLU(x) = x / (1 + exp(-x))
    for i in 0..intermediate_dim {
        gate_buf[i] = gate_buf[i] / (1.0 + libm::expf(-gate_buf[i]));
    }

    // Step 4: Element-wise multiply (silu_gate * up → hidden_buf)
    for i in 0..intermediate_dim {
        hidden_buf[i] = gate_buf[i] * up_buf[i];
    }

    // Step 5: Down projection — static dispatch on quant type
    if down_quant_is_q6k {
        linear_q6k(
            hidden_buf,
            down_weight,
            output,
            scratch,
            intermediate_dim,
            embedding_dim,
        );
    } else {
        linear_q4k(
            hidden_buf,
            down_weight,
            output,
            scratch,
            intermediate_dim,
            embedding_dim,
        );
    }
}

/// SwiGLU MLP with per-tensor quant dispatch — drop-in replacement for
/// [`mlp_swiglu`] that switches on each weight's `GgmlType` instead of
/// the legacy Q4_K/Q6_K boolean. Used by the deepseek2 forward pass.
///
/// The Qwen3 forward pass keeps calling [`mlp_swiglu`] directly so its
/// β-anchor numerics stay bit-identical — this dispatcher is purely
/// additive.
#[allow(clippy::too_many_arguments)]
pub fn mlp_swiglu_dispatch(
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
    scratch: &mut LinearScratch,
    output: &mut [f32],
) {
    let _ = crate::ops::linear_dispatch(
        input,
        gate_weight,
        gate_buf,
        scratch,
        embedding_dim,
        intermediate_dim,
        gate_quant,
    );
    let _ = crate::ops::linear_dispatch(
        input,
        up_weight,
        up_buf,
        scratch,
        embedding_dim,
        intermediate_dim,
        up_quant,
    );
    for i in 0..intermediate_dim {
        gate_buf[i] = gate_buf[i] / (1.0 + libm::expf(-gate_buf[i]));
    }
    for i in 0..intermediate_dim {
        hidden_buf[i] = gate_buf[i] * up_buf[i];
    }
    let _ = crate::ops::linear_dispatch(
        hidden_buf,
        down_weight,
        output,
        scratch,
        intermediate_dim,
        embedding_dim,
        down_quant,
    );
}

/// Single-token forward pass through all 28 layers.
///
/// Caller pre-allocates ALL scratch buffers (Pillar 1: zero allocation in hot path).
/// After return, `hidden` contains the final hidden states (pre-final-norm).
#[allow(clippy::too_many_arguments)]
pub fn forward_single_token<const HALF: usize>(
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
    // Scratch buffers
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
    scratch: &mut LinearScratch,
) -> Result<(), ForwardPassError> {
    // ── Embedding lookup ─────────────────────────────────────────
    embed_lookup_dispatch(
        token_id,
        token_embd,
        embedding_dim,
        hidden,
        dispatch.token_embd_quant,
    )
    .map_err(ForwardPassError::Embedding)?;

    // ── 28-layer chain ───────────────────────────────────────────
    for layer_idx in 0..N_LAYERS {
        let layer = &layers[layer_idx];

        // ─ Attention sub-block ─
        rmsnorm(hidden, layer.attn_norm, norm_buf, rms_eps);

        gqa_attention_single_token_dispatch::<HALF>(
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
            scratch,
            attn_out,
        )
        .map_err(ForwardPassError::Attention)?;

        // Residual: hidden += attn_out
        for i in 0..embedding_dim {
            hidden[i] += attn_out[i];
        }

        // ─ MLP sub-block ─
        rmsnorm(hidden, layer.ffn_norm, norm_buf, rms_eps);

        mlp_swiglu_dispatch(
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
            scratch,
            mlp_out,
        );

        // Residual: hidden += mlp_out
        for i in 0..embedding_dim {
            hidden[i] += mlp_out[i];
        }
    }

    // hidden now contains final hidden states (post-28-layers, pre-final-norm)
    Ok(())
}

// ──────────────────────────────────────────────────────────────────
//  DeepSeek-V2 / Kimi K2.6 forward pass
// ──────────────────────────────────────────────────────────────────

/// Errors specific to the DeepSeek2 / Kimi forward pass.
#[derive(Debug)]
pub enum Deepseek2Error {
    Mla(MlaError),
    KvCache(KvCacheError),
}

impl From<MlaError> for Deepseek2Error {
    fn from(e: MlaError) -> Self {
        Deepseek2Error::Mla(e)
    }
}

impl From<KvCacheError> for Deepseek2Error {
    fn from(e: KvCacheError) -> Self {
        Deepseek2Error::KvCache(e)
    }
}

/// Scratch buffers for one DeepSeek2 / Kimi single-token forward pass.
///
/// Caller-owned, sized at boot from `ModelConfig`. None of these allocate
/// on the hot path. See `kernel/src/inference.rs` for the wiring.
pub struct Deepseek2Scratch<'s> {
    // Hidden state + residuals
    pub hidden: &'s mut [f32],   // [embedding_dim]
    pub norm_buf: &'s mut [f32], // [embedding_dim]
    pub attn_out: &'s mut [f32], // [embedding_dim]
    pub mlp_out: &'s mut [f32],  // [embedding_dim]

    // MLA scratch
    pub c_kv_rope_buf: &'s mut [f32], // [kv_lora_rank + qk_rope_head_dim]
    pub c_kv_norm_buf: &'s mut [f32], // [kv_lora_rank]
    pub kv_decompressed: &'s mut [f32], // [n_heads × (qk_nope_head_dim + v_head_dim)]
    pub c_q_buf: &'s mut [f32],       // [q_lora_rank]
    pub c_q_norm_buf: &'s mut [f32],  // [q_lora_rank]
    pub q_decompressed: &'s mut [f32], // [n_heads × (qk_nope_head_dim + qk_rope_head_dim)]
    pub k_assembled: &'s mut [f32],   // (unused under compressed-latent MLA cache)
    pub score_buf: &'s mut [f32],     // [n_heads × max_tokens] (per-head stripe)
    pub attn_head_buf: &'s mut [f32], // (unused under compressed-latent MLA cache)
    pub attn_concat: &'s mut [f32],   // [n_heads × v_head_dim]

    // MoE / Dense FFN scratch
    pub gate_buf: &'s mut [f32], // [expert_intermediate or feed_forward_length]
    pub up_buf: &'s mut [f32],   // [expert_intermediate or feed_forward_length]
    pub hidden_mlp_buf: &'s mut [f32], // [expert_intermediate or feed_forward_length]
    pub router_score_buf: &'s mut [f32], // [n_experts]
    pub expert_indices: &'s mut [u32], // [top_k]
    pub expert_weights: &'s mut [f32], // [top_k]
    pub expert_out_buf: &'s mut [f32], // [embedding_dim]

    pub scratch: &'s mut LinearScratch,
}

/// DeepSeek-V2 / Kimi K2.6 single-token forward pass.
///
/// Dynamic layer count (`layers.len()` — typically 61 for Kimi K2.6).
/// Each layer dispatches MLA-or-MHA attention + (Dense FFN or MoE FFN)
/// based on `Deepseek2LayerWeights.attn_type` and `.mlp_type`, and the
/// matmul kernel for each tensor is selected from the per-layer
/// `attn_quant` / `expert_quant` / `ffn_dense_quant` fields. The
/// `embed_quant` argument controls the token-embedding lookup.
///
/// `mha_kv_cache` is optional — only required when any layer has
/// `attn_type == Mha`. Boot code passes `None` when every layer is MLA.
#[allow(clippy::too_many_arguments)]
pub fn forward_single_token_deepseek2(
    token_id: u32,
    token_offset: usize,
    layers: &[Deepseek2LayerWeights<'_>],
    token_embd: &[u8],
    embed_quant: zero_gguf_parser::GgmlType,
    rms_eps: f32,
    rope_freq_base: f32,
    embedding_dim: usize,
    feed_forward_length: usize,
    // MLA dims
    n_heads: usize,
    n_kv_heads: usize,
    kv_lora_rank: usize,
    q_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    // MoE dims
    n_experts: usize,
    top_k: usize,
    expert_intermediate: usize,
    // SPEC: DeepSeek-V3/Kimi K2.6 require `SigmoidNormalize`; Qwen-style
    // MoE (none today, kept as a future hook) uses `Softmax`.
    routing_mode: MoeRoutingMode,
    expert_weight_scale: f32,
    kv_cache: &mut MlaKvCache,
    mha_kv_cache: Option<&mut MhaKvCache>,
    s: &mut Deepseek2Scratch<'_>,
) -> Result<(), Deepseek2Error> {
    // Per-quant embedding lookup. Any unsupported quant becomes an
    // attention-zero hidden state (see `embed_lookup_dispatch`'s
    // contract); we drop the error here because the boot path already
    // refused the build via the same dispatcher, but tolerating it on
    // the inner loop keeps the API surface tight.
    let _ = embed_lookup_dispatch(token_id, token_embd, embedding_dim, s.hidden, embed_quant);

    let mut mha_kv_opt = mha_kv_cache;
    for (layer_idx, layer) in layers.iter().enumerate() {
        // ─ Attention sub-block ─
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

                mla_attention_single_token(
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
                let mha_cache = mha_kv_opt
                    .as_deref_mut()
                    .ok_or(Deepseek2Error::Mla(MlaError::NumericalInstability))?;
                let mha_weights = MhaWeights {
                    q: layer.attn_q_mha,
                    k: layer.attn_k_mha,
                    v: layer.attn_v_mha,
                    output: layer.attn_output,
                };
                let head_dim_qk = qk_nope_head_dim + qk_rope_head_dim;
                // MHA scratch reuses MLA buffers:
                //   q_decompressed (n_heads × head_dim_qk)  ← q_buf
                //   kv_decompressed split into K (n_kv_heads × head_dim_qk)
                //                       + V (n_kv_heads × v_head_dim)
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

        // Residual: hidden += attn_out
        for i in 0..embedding_dim {
            s.hidden[i] += s.attn_out[i];
        }

        // ─ FFN sub-block ─
        rmsnorm(s.hidden, layer.ffn_norm, s.norm_buf, rms_eps);

        match layer.mlp_type {
            MlpType::Dense => {
                // Per-tensor quant: pre-audit, `mlp_swiglu` only supported
                // Q4_K/Q6_K for down via a bool. The deepseek2 dense path
                // now picks the right kernel via `mlp_swiglu_dispatch`
                // (added below) so Kimi K2.6 Q4_0 dense layers work too.
                mlp_swiglu_dispatch(
                    s.norm_buf,
                    layer.ffn_gate,
                    layer.ffn_up,
                    layer.ffn_down,
                    layer.ffn_dense_quant,
                    layer.ffn_dense_quant,
                    layer.ffn_dense_quant,
                    embedding_dim,
                    feed_forward_length,
                    s.gate_buf,
                    s.up_buf,
                    s.hidden_mlp_buf,
                    s.scratch,
                    s.mlp_out,
                );
            }
            MlpType::MoE => {
                // Zero the output accumulator — moe_ffn writes shared expert
                // directly to it, then weighted-sums selected experts in.
                for i in 0..embedding_dim {
                    s.mlp_out[i] = 0.0;
                }
                moe_ffn(
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

        // Residual: hidden += mlp_out
        for i in 0..embedding_dim {
            s.hidden[i] += s.mlp_out[i];
        }
    }

    Ok(())
}
