// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mixture-of-Experts (MoE) inference operators.
//!
//! Implements the MoE routing and expert dispatch used by DeepSeek-V2/V3
//! and Kimi K2.6 models. Compatible with GGUF MoE tensor layout from
//! llama.cpp / Unsloth quantizations.
//!
//! # Architecture (Kimi K2.6)
//!
//! Per layer:
//!   1. Router: hidden → scores[n_experts] via gate_inp matmul
//!   2. Top-K selection: pick top_k experts (e.g. 8 of 384)
//!   3. Softmax over selected scores → weights
//!   4. Shared expert: SwiGLU MLP (always executed)
//!   5. Selected experts: each runs SwiGLU MLP independently
//!   6. Output = shared_output + Σ(weight_i × expert_i_output)
//!
//! # Design Constraints
//!
//! - `no_std`, zero allocation in hot path
//! - Expert weights are zero-copy slices into stacked GGUF tensors
//! - Scratch buffers caller-provided (Pillar 1)
//! - SMP parallelization of independent experts (in kernel dispatch)

/// MoE routing result for a single token.
///
/// Contains the indices and weights for the top_k selected experts.
/// Caller provides the output buffers.
pub struct MoeRouteResult<'a> {
    /// Selected expert indices [top_k], sorted by weight descending.
    pub indices: &'a [u32],
    /// Softmax-normalized weights for selected experts [top_k].
    pub weights: &'a [f32],
    /// Number of experts selected (= top_k).
    pub count: usize,
}

/// MoE top-K weight normalisation mode.
///
/// Different MoE families normalise the top-K router scores differently:
///   * **`Softmax`** — classic `exp(score)/Σexp(score)` over top-K.
///     Required by Mixtral, Qwen-MoE and other Llama-family MoE models.
///   * **`SigmoidNormalize`** — `s = sigmoid(dot)` per expert; the
///     per-expert load-balancing bias enters ONLY the top-K selection
///     ranking (`s + bias`), never the weights; the selected experts'
///     unbiased `s` values are then sum-normalised to 1.
///     SPEC: DeepSeek-V3 paper §3.1 / Kimi K2.6 — required for those
///     models; using softmax instead causes the router to systematically
///     pick the wrong experts, and leaking the bias into the weights
///     systematically distorts the expert mixture.
///
/// Selected per-architecture by the kernel-level dispatcher; cannot be
/// inferred from the GGUF alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeRoutingMode {
    Softmax,
    SigmoidNormalize,
}

#[inline]
fn normalised_expert_weight_scale(expert_weight_scale: f32) -> f32 {
    if expert_weight_scale.is_finite() && expert_weight_scale > 0.0 {
        expert_weight_scale
    } else {
        1.0
    }
}

#[inline]
fn apply_expert_weight_scale(weights: &mut [f32], count: usize, expert_weight_scale: f32) {
    let scale = normalised_expert_weight_scale(expert_weight_scale);
    if scale != 1.0 {
        for weight in weights.iter_mut().take(count) {
            *weight *= scale;
        }
    }
}

/// Compute MoE routing: select top_k experts from n_experts.
///
/// Steps:
///   1. Per `mode`:
///      * `Softmax`          — score[i] = dot(hidden, router_weight[i]) + bias[i]
///      * `SigmoidNormalize` — s[i] = sigmoid(dot(hidden, router_weight[i]));
///        the bias is NOT folded into `s`
///      (bias used only when `router_bias` is non-empty)
///   2. Find top_k highest selection scores. Selection key per `mode`:
///      * `Softmax`          — score[i] (bias already included)
///      * `SigmoidNormalize` — s[i] + bias[i] (DeepSeek-V3 §3.1: the
///        load-balancing bias shifts ONLY the ranking)
///   3. Per `mode`:
///      * `Softmax`          — `exp(score)/Σexp(score)` over top-K (max-subtraction
///        for numerical stability)
///      * `SigmoidNormalize` — sum-normalise the selected experts'
///        unbiased `s` values to 1
///      NaN-guard fallback to uniform `1/top_k` in both modes.
///   4. Multiply normalized expert weights by `expert_weight_scale`.
///      DeepSeek-V3/Kimi-family models use this routed scaling factor;
///      pass `1.0` for architectures without one.
///
/// `router_weight` is the gate_inp tensor: [n_experts × embedding_dim],
/// stored in F32 (typical for router weights in GGUF MoE models).
///
/// `router_bias` is `ffn_gate_inp_bias.weight` [n_experts] in F32, or
/// an empty slice when the GGUF does not ship the tensor. Kimi K2.6
/// uses per-expert bias for load balancing; DeepSeek-V3 does too.
///
/// # Preconditions
///
/// Panics (always, even in release) if:
///   * `top_k == 0` — meaningless, the function would write past zero-sized buffers
///   * `top_k > n_experts` — would read past `score_buf` during seed-fill
///   * `router_bias` is non-empty but its length is less than `n_experts`
///
/// These are programmer errors; ModelConfig should reject such combinations
/// at boot. The asserts exist so a misconfigured deployment fails loud and
/// early instead of corrupting expert indices.
///
/// # Arguments
///
/// * `hidden` — input hidden state [embedding_dim]
/// * `router_weight` — F32 router weights [n_experts × embedding_dim]
/// * `router_bias` — F32 router bias [n_experts] or empty
/// * `n_experts` — total number of experts (e.g. 384)
/// * `top_k` — number of experts to select (e.g. 8); must satisfy `0 < top_k ≤ n_experts`
/// * `score_buf` — scratch buffer [n_experts]
/// * `indices_buf` — output expert indices [top_k]
/// * `weights_buf` — output expert weights [top_k]
/// * `mode` — top-K normalisation mode (architecture-dependent)
/// * `expert_weight_scale` — post-normalisation routed expert scale
#[allow(clippy::too_many_arguments)]
pub fn moe_route_f32(
    hidden: &[f32],
    router_weight: &[f32],
    router_bias: &[f32],
    n_experts: usize,
    top_k: usize,
    score_buf: &mut [f32],
    indices_buf: &mut [u32],
    weights_buf: &mut [f32],
    mode: MoeRoutingMode,
    expert_weight_scale: f32,
) {
    // Contract checks — fail loud on misconfiguration. See doc comment.
    assert!(top_k > 0, "moe_route_f32: top_k must be > 0");
    assert!(
        top_k <= n_experts,
        "moe_route_f32: top_k must be <= n_experts"
    );
    assert!(score_buf.len() >= n_experts);
    assert!(indices_buf.len() >= top_k);
    assert!(weights_buf.len() >= top_k);
    let use_bias = !router_bias.is_empty();
    if use_bias {
        assert!(
            router_bias.len() >= n_experts,
            "moe_route_f32: router_bias must be empty or have at least n_experts entries"
        );
    }

    let embedding_dim = hidden.len();

    // Step 1: Compute per-expert gate values.
    //   Softmax          — score = dot + bias (classic Llama-family MoE).
    //   SigmoidNormalize — score = sigmoid(dot); SPEC DeepSeek-V3 §3.1 /
    //     Kimi K2.6: the load-balancing bias must NOT enter the gate
    //     value, it only shifts the top-K ranking (see Step 2).
    for e in 0..n_experts {
        let row_offset = e * embedding_dim;
        let row = &router_weight[row_offset..row_offset + embedding_dim];
        let mut dot = 0.0f32;
        for i in 0..embedding_dim {
            dot += hidden[i] * row[i];
        }
        score_buf[e] = match mode {
            MoeRoutingMode::Softmax => {
                if use_bias {
                    dot + router_bias[e]
                } else {
                    dot
                }
            }
            // sigmoid(x) = 1 / (1 + exp(-x)) — unbiased gate value s.
            MoeRoutingMode::SigmoidNormalize => 1.0 / (1.0 + libm::expf(-dot)),
        };
    }

    // Step 2: Top-K selection via partial sort over the selection key.
    // For SigmoidNormalize the key is s + bias (ranking-only bias);
    // weights_buf temporarily holds selection keys and is rewritten to
    // the unbiased gate values after selection.
    let selection_bias = |e: usize| -> f32 {
        if use_bias && mode == MoeRoutingMode::SigmoidNormalize {
            router_bias[e]
        } else {
            0.0
        }
    };

    // Initialize with first top_k selection keys
    for i in 0..top_k {
        indices_buf[i] = i as u32;
        weights_buf[i] = score_buf[i] + selection_bias(i);
    }

    // Find minimum in current top_k
    let mut min_idx = 0usize;
    let mut min_val = weights_buf[0];
    for i in 1..top_k {
        if weights_buf[i] < min_val {
            min_val = weights_buf[i];
            min_idx = i;
        }
    }

    // Scan remaining experts, replacing minimum when beaten
    for e in top_k..n_experts {
        let key = score_buf[e] + selection_bias(e);
        if key > min_val {
            indices_buf[min_idx] = e as u32;
            weights_buf[min_idx] = key;
            // Re-find minimum
            min_val = weights_buf[0];
            min_idx = 0;
            for i in 1..top_k {
                if weights_buf[i] < min_val {
                    min_val = weights_buf[i];
                    min_idx = i;
                }
            }
        }
    }

    // Discard the selection keys: from here on weights_buf carries the
    // gate values of the selected experts (raw scores for Softmax,
    // unbiased sigmoid s for SigmoidNormalize).
    for i in 0..top_k {
        weights_buf[i] = score_buf[indices_buf[i] as usize];
    }

    // Step 3: per-mode normalisation over selected scores.
    // Find max for numerical stability + finite-check (shared between
    // both modes; sigmoid also benefits from the NaN guard).
    let mut max_score = weights_buf[0];
    for i in 1..top_k {
        if weights_buf[i] > max_score {
            max_score = weights_buf[i];
        }
    }

    // NaN/non-finite guard: if max is non-finite (every score is NaN or
    // ±inf, e.g. an upstream numeric blow-up), softmax produces NaN
    // weights that propagate silently and zero the expert contribution
    // anyway. Bail to a uniform distribution so the layer still produces
    // a meaningful (if degraded) output and the caller can detect via
    // post-hoc finite checks on `hidden`.
    if !max_score.is_finite() {
        let uniform = 1.0 / top_k as f32;
        for i in 0..top_k {
            weights_buf[i] = uniform;
        }
        apply_expert_weight_scale(weights_buf, top_k, expert_weight_scale);
        return;
    }

    match mode {
        MoeRoutingMode::Softmax => {
            let mut sum_exp = 0.0f32;
            for i in 0..top_k {
                weights_buf[i] = libm::expf(weights_buf[i] - max_score);
                sum_exp += weights_buf[i];
            }
            // After max-subtraction, exp(0) = 1 for the max entry, so sum_exp ≥ 1
            // for any finite input. If sum_exp is non-finite or ≤ 0 the inputs
            // must have contained NaNs that beat the `is_finite` filter above —
            // fall back to uniform.
            if !sum_exp.is_finite() || sum_exp <= 0.0 {
                let uniform = 1.0 / top_k as f32;
                for i in 0..top_k {
                    weights_buf[i] = uniform;
                }
                apply_expert_weight_scale(weights_buf, top_k, expert_weight_scale);
                return;
            }
            let inv_sum = 1.0 / sum_exp;
            for i in 0..top_k {
                weights_buf[i] *= inv_sum;
            }
        }
        MoeRoutingMode::SigmoidNormalize => {
            // SPEC: DeepSeek-V3 §3.1 / Kimi K2.6 —
            //   s_i = sigmoid(dot_i) (already computed in Step 1);
            //   w_i = s_i / Σ s_i over the selected experts.
            // NOT softmax, and NOT sigmoid(dot + bias): the bias only
            // shifted the ranking in Step 2; the weights are the
            // unbiased gate values.
            let mut sum = 0.0f32;
            for i in 0..top_k {
                sum += weights_buf[i];
            }
            if !sum.is_finite() || sum <= 0.0 {
                let uniform = 1.0 / top_k as f32;
                for i in 0..top_k {
                    weights_buf[i] = uniform;
                }
                apply_expert_weight_scale(weights_buf, top_k, expert_weight_scale);
                return;
            }
            let inv_sum = 1.0 / sum;
            for i in 0..top_k {
                weights_buf[i] *= inv_sum;
            }
        }
    }

    apply_expert_weight_scale(weights_buf, top_k, expert_weight_scale);
}

/// Execute a single expert's SwiGLU MLP forward pass.
///
/// Expert weights are sliced from the stacked GGUF tensors:
///   gate_exps[expert_id * expert_bytes .. (expert_id+1) * expert_bytes]
///
/// Formula:
///   gate = gate_w @ input
///   up   = up_w @ input
///   hidden = SiLU(gate) * up
///   output = down_w @ hidden
///
/// All projections are Q4_K quantized.
///
/// # Arguments
///
/// * `input` — hidden state [embedding_dim]
/// * `gate_w` — expert gate weights (Q4_K) [expert_intermediate × embedding_dim]
/// * `up_w` — expert up weights (Q4_K) [expert_intermediate × embedding_dim]
/// * `down_w` — expert down weights (Q4_K) [embedding_dim × expert_intermediate]
/// * `embedding_dim` — model hidden dimension
/// * `expert_intermediate` — expert FFN intermediate dimension
/// * `gate_buf` — scratch [expert_intermediate]
/// * `up_buf` — scratch [expert_intermediate]
/// * `hidden_buf` — scratch [expert_intermediate]
/// * `output` — result [embedding_dim]
/// * `scratch` — linear scratch buffer
pub fn expert_swiglu(
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
    scratch: &mut crate::ops::LinearScratch,
    expert_quant: zero_gguf_parser::GgmlType,
) {
    // Gate projection (per-tensor quant via dispatcher)
    let _ = crate::ops::linear_dispatch(
        input,
        gate_w,
        gate_buf,
        scratch,
        embedding_dim,
        expert_intermediate,
        expert_quant,
    );

    // Up projection
    let _ = crate::ops::linear_dispatch(
        input,
        up_w,
        up_buf,
        scratch,
        embedding_dim,
        expert_intermediate,
        expert_quant,
    );

    // SiLU(gate) * up → hidden
    for i in 0..expert_intermediate {
        let silu = gate_buf[i] / (1.0 + libm::expf(-gate_buf[i]));
        hidden_buf[i] = silu * up_buf[i];
    }

    // Down projection
    let _ = crate::ops::linear_dispatch(
        hidden_buf,
        down_w,
        output,
        scratch,
        expert_intermediate,
        embedding_dim,
        expert_quant,
    );
}

/// Compute byte size of one expert's weight tensor in Q4_K format.
///
/// Q4_K: 256 elements per block, 144 bytes per block.
/// Expert weight shape: [out_dim × in_dim]
/// Size = (in_dim / 256) * 144 * out_dim bytes
///
/// Kept for back-compat with callers that don't yet thread the quant
/// type through; prefer [`expert_quant_bytes`] for new code.
#[inline]
pub fn expert_q4k_bytes(in_dim: usize, out_dim: usize) -> usize {
    let blocks_per_row = in_dim / 256;
    blocks_per_row * 144 * out_dim
}

/// Compute byte size of one expert's weight tensor for the given quant.
///
/// Expert weight shape: `[out_dim × in_dim]`, row-major. The per-row
/// stride is `(in_dim / block_elements) × block_bytes`, multiplied by
/// `out_dim` rows for the full tensor.
///
/// Returns `None` if the quant has no row-quantised dequantiser or if
/// `in_dim` isn't a multiple of the quant's block size.
///
/// Supported quants: Q4_K (256 elements / 144 B), Q4_0 (32 / 18 B),
/// Q8_0 (32 / 34 B). The kernel's per-tensor dispatcher refuses to
/// load any other quant at boot, so callers can rely on `unwrap_or`
/// of a sane fallback (typically Q4_K for legacy code paths).
#[inline]
pub fn expert_quant_bytes(
    in_dim: usize,
    out_dim: usize,
    quant: zero_gguf_parser::GgmlType,
) -> Option<usize> {
    use zero_gguf_parser::{
        dequant::{
            Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES,
            Q8_0_BLOCK_SIZE,
        },
        GgmlType,
    };
    let (block_elems, block_bytes) = match quant {
        GgmlType::Q4K => (Q4K_BLOCK_SIZE, Q4K_BLOCK_BYTES),
        GgmlType::Q4_0 => (Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES),
        GgmlType::Q8_0 => (Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES),
        _ => return None,
    };
    if in_dim % block_elems != 0 {
        return None;
    }
    Some((in_dim / block_elems) * block_bytes * out_dim)
}

/// Slice a single expert's weights from a stacked expert tensor.
///
/// GGUF MoE stacked layout: `ffn_gate_exps` = [n_experts, intermediate, hidden]
/// The tensor data is stored row-major, so expert_i starts at
/// offset = expert_id * (intermediate * hidden_bytes_per_row).
///
/// Returns a byte slice into the stacked tensor for expert_id.
#[inline]
pub fn slice_expert_weight(
    stacked_weights: &[u8],
    expert_id: usize,
    expert_weight_bytes: usize,
) -> &[u8] {
    let offset = expert_id * expert_weight_bytes;
    &stacked_weights[offset..offset + expert_weight_bytes]
}

/// Full MoE FFN: router dispatch + shared expert + top-k experts + weighted sum.
///
/// This is the scalar reference implementation. The kernel-level AVX-512
/// variant in `inference_avx512.rs` replaces the matmuls with parallel dispatch.
///
/// # Arguments
///
/// * `input` — hidden state after FFN RMSNorm [embedding_dim]
/// * `router_weight_f32` — F32 router [n_experts × embedding_dim]
/// * `gate_exps` — stacked expert gate Q4_K [n_experts × inter × emb]
/// * `up_exps` — stacked expert up Q4_K
/// * `down_exps` — stacked expert down Q4_K
/// * `shared_gate` — shared expert gate Q4_K [inter × emb]
/// * `shared_up` — shared expert up Q4_K
/// * `shared_down` — shared expert down Q4_K
/// * Config and scratch parameters...
/// * `output` — result accumulator [embedding_dim]
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn(
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
    // Scratch buffers
    router_score_buf: &mut [f32], // [n_experts]
    expert_indices: &mut [u32],   // [top_k]
    expert_weights: &mut [f32],   // [top_k]
    gate_buf: &mut [f32],         // [expert_intermediate]
    up_buf: &mut [f32],           // [expert_intermediate]
    hidden_buf: &mut [f32],       // [expert_intermediate]
    expert_out_buf: &mut [f32],   // [embedding_dim]
    output: &mut [f32],           // [embedding_dim]
    scratch: &mut crate::ops::LinearScratch,
    expert_quant: zero_gguf_parser::GgmlType,
    routing_mode: MoeRoutingMode,
    expert_weight_scale: f32,
) {
    // Precondition: defer to moe_route_f32 for the strict checks, but
    // do an early assert here so a misconfigured deployment fails before
    // we touch any expert weight bytes (cheaper diagnostic).
    assert!(top_k > 0, "moe_ffn: top_k must be > 0");
    assert!(top_k <= n_experts, "moe_ffn: top_k must be <= n_experts");

    // Step 1: Route — select top_k experts
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

    // Step 2: Shared expert (always executed)
    expert_swiglu(
        input,
        shared_gate,
        shared_up,
        shared_down,
        embedding_dim,
        expert_intermediate,
        gate_buf,
        up_buf,
        hidden_buf,
        output, // shared expert output goes directly to accumulator
        scratch,
        expert_quant,
    );

    // Step 3: Selected experts — each adds weighted contribution.
    // Per-expert byte size depends on the quant (Q4_K: 144 B per 256
    // elements; Q4_0: 18 B per 32 elements). If the quant isn't one
    // the dispatcher can size, fall through to the Q4_K layout as a
    // best-effort — caller is responsible for refusing such configs
    // at boot via `linear_dispatch`'s `UnsupportedQuant`.
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

        expert_swiglu(
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
            scratch,
            expert_quant,
        );

        // Weighted accumulation: output += weight * expert_output
        for i in 0..embedding_dim {
            output[i] += weight * expert_out_buf[i];
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use zero_gguf_parser::GgmlType;

    #[test]
    fn expert_quant_bytes_q4k_sizes_match_reference() {
        // Q4_K: 256 elements / 144 B per block. A 256 × 256 expert
        // weight is 1 block-per-row × 144 B × 256 rows = 36 864 B.
        let bytes = expert_quant_bytes(256, 256, GgmlType::Q4K).unwrap();
        assert_eq!(bytes, 256 * 144);
    }

    #[test]
    fn expert_quant_bytes_q4_0_sizes_match_reference() {
        // Q4_0: 32 elements / 18 B per block. A 256 × 256 expert
        // weight is 8 blocks-per-row × 18 B × 256 rows = 36 864 B.
        let bytes = expert_quant_bytes(256, 256, GgmlType::Q4_0).unwrap();
        assert_eq!(bytes, 8 * 18 * 256);
    }

    #[test]
    fn expert_quant_bytes_q8_0_sizes_match_reference() {
        // Q8_0: 32 elements / 34 B per block. A 256 × 256 expert
        // weight is 8 blocks-per-row × 34 B × 256 rows = 69 632 B.
        // (Realistic Kimi K2.6 builds don't ship expert tensors as
        // Q8_0 — too big — but the sizing must still be correct so
        // mixed-quant operators that swap individual experts up to
        // Q8_0 for sensitivity analysis still get correct strides.)
        let bytes = expert_quant_bytes(256, 256, GgmlType::Q8_0).unwrap();
        assert_eq!(bytes, 8 * 34 * 256);
    }

    #[test]
    fn expert_quant_bytes_rejects_unsupported_quant() {
        // F16 has no row-quantised dequantiser today — sizing
        // returns None so the caller can refuse the model at boot.
        assert!(expert_quant_bytes(256, 256, GgmlType::F16).is_none());
        assert!(expert_quant_bytes(256, 256, GgmlType::F32).is_none());
        assert!(expert_quant_bytes(256, 256, GgmlType::BF16).is_none());
    }

    #[test]
    fn expert_quant_bytes_misaligned_in_dim_returns_none() {
        // in_dim not a multiple of the block size → None (defensive;
        // caller must catch this before dispatching the matmul).
        assert!(expert_quant_bytes(100, 256, GgmlType::Q4K).is_none()); // 100 not multiple of 256
        assert!(expert_quant_bytes(33, 256, GgmlType::Q4_0).is_none()); // 33 not multiple of 32
        assert!(expert_quant_bytes(31, 256, GgmlType::Q8_0).is_none()); // 31 not multiple of 32
    }
}
