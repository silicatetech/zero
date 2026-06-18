// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::needless_range_loop)]
//! Synthetic DeepSeek-V2/V3 / Kimi K2.6 MoE+MLA end-to-end pipeline test.
//!
//! Builds a minimal but **structurally correct** fake DeepSeek2 model in
//! memory and runs a single token through the full MoE FFN + MLA attention
//! pipeline, then runs it again and asserts the two outputs are byte-for-
//! byte identical (`f32::to_bits` equality). This proves:
//!
//!   1. `moe_route_f32` selects deterministically given fixed weights.
//!   2. `moe_ffn` (router → shared expert → top-K experts → weighted sum)
//!      produces the same float bits on two consecutive runs.
//!   3. `mla_attention_single_token` (KV compression, RoPE, K/V cache,
//!      scaled dot-product attention, output projection) is deterministic.
//!   4. None of the scratch buffers carry uninitialised reads that could
//!      sneak nondeterminism in.
//!
//! # Dimensions
//!
//! The user spec asks for "small dims (4 experts, hidden=64,
//! intermediate=128, 2 layers)". Q4_K however requires every matmul's
//! input dimension to be a multiple of 256 (`linear_q4k` debug-asserts
//! `in_dim % 256 == 0`). We therefore pick the smallest legal sizes:
//!
//! * hidden = embedding_dim = 256
//! * expert_intermediate    = 256
//! * kv_lora_rank           = 256
//! * q_lora_rank            = 256
//! * qk_nope_head_dim       = 32
//! * qk_rope_head_dim       = 32
//! * v_head_dim             = 64   (so n_heads × v_head_dim = 256, valid Q4_K in_dim for W_output)
//! * n_heads                = 4
//! * n_experts              = 4
//! * top_k                  = 2
//!
//! Every weight tensor is a constant-valued Q4_K block (gate=0.05, up=0.05,
//! down=0.01, MLA projections=0.02, …) so the matmuls produce non-zero
//! deterministic values without numeric drift into NaN.

use zero_llm_inference::{
    expert_swiglu, mla_attention_single_token, moe_ffn, moe_route_f32, LinearScratch, MlaKvCache,
    MlaWeights, MoeRoutingMode,
};

// ── Dimensions (see file header for justification) ───────────────────
const HIDDEN: usize = 256;
const INTERMEDIATE: usize = 256;
const N_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const N_HEADS: usize = 4;
const KV_LORA_RANK: usize = 256;
const Q_LORA_RANK: usize = 256;
const QK_NOPE: usize = 32;
const QK_ROPE: usize = 32;
const V_HEAD: usize = 64;

const TOTAL_K_HEAD: usize = QK_NOPE + QK_ROPE; // 64
const KV_B_OUT: usize = N_HEADS * (QK_NOPE + V_HEAD); // 4 × 96 = 384
const Q_B_OUT: usize = N_HEADS * (QK_NOPE + QK_ROPE); // 4 × 64 = 256
const KV_A_OUT: usize = KV_LORA_RANK + QK_ROPE; // 288
const O_IN: usize = N_HEADS * V_HEAD; // 256

const RMS_EPS: f32 = 1e-6;
const ROPE_FREQ_BASE: f32 = 1_000_000.0;

const MAX_TOKENS: usize = 4;

// ── Q4_K block construction helpers ──────────────────────────────────

/// Convert an f32 to its IEEE 754 binary16 (fp16) bit pattern. Only
/// covers the normal-number range we use here (≈ 6e-5 .. 6.5e4); good
/// enough for synthesis of test weights.
fn f32_to_fp16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let mant32 = bits & 0x007F_FFFF;

    if exp32 == 0 {
        // Zero or denormal in f32 → fp16 zero (we never feed denormals).
        return sign << 15;
    }
    if exp32 == 0xFF {
        // Inf / NaN — encode as fp16 inf, preserving sign. NaN payload
        // is irrelevant; this helper never produces NaN intentionally.
        return (sign << 15) | (0x1F << 10);
    }
    let unbiased = exp32 - 127;
    if unbiased < -14 {
        // Underflow → flush to zero (fine for our 0.01..1.0 values).
        return sign << 15;
    }
    if unbiased > 15 {
        // Overflow → ±inf (we never feed values this large).
        return (sign << 15) | (0x1F << 10);
    }
    let exp16 = (unbiased + 15) as u16;
    let mant16 = (mant32 >> 13) as u16; // truncate, round-to-zero is fine
    (sign << 15) | (exp16 << 10) | (mant16 & 0x3FF)
}

/// Build one Q4_K block (144 bytes) whose `dequant_q4k_block` output is
/// the constant `value` across all 256 elements.
///
/// Encoding (cross-checked against `zero_gguf_parser::dequant`):
///   d        = `value` (fp16)
///   dmin     = 0
///   sub-block scales (all 8) = 1
///   sub-block mins   (all 8) = 0
///   every 4-bit quant = 1 (qs byte = 0x11)
///
/// Per dequant: `out = d * scale_j * q − dmin * min_j = value * 1 * 1 − 0 = value`.
fn q4k_uniform_block(value: f32) -> [u8; 144] {
    let mut block = [0u8; 144];
    block[0..2].copy_from_slice(&f32_to_fp16_bits(value).to_le_bytes());
    block[2..4].copy_from_slice(&0u16.to_le_bytes()); // dmin = 0
                                                      // scales[0..4] = 1  → scale for sub-blocks 0..3, plus high bits 0 for sb 4..7 scales
    block[4..8].copy_from_slice(&[1, 1, 1, 1]);
    // scales[4..8] = 0  → min for sub-blocks 0..3, plus high bits 0 for sb 4..7 mins
    block[8..12].copy_from_slice(&[0, 0, 0, 0]);
    // scales[8..12] = 0x01 → low nibble 1 (scale low for sb 4..7), high nibble 0 (min low for sb 4..7)
    block[12..16].copy_from_slice(&[0x01, 0x01, 0x01, 0x01]);
    // qs: 128 bytes, every nibble = 1 → q = 1 everywhere
    for b in block.iter_mut().take(144).skip(16) {
        *b = 0x11;
    }
    block
}

/// Build a Q4_K weight buffer of shape `[out_dim × in_dim]` where every
/// element dequantises to `value`. `in_dim` must be a multiple of 256.
fn q4k_uniform_matrix(value: f32, out_dim: usize, in_dim: usize) -> Vec<u8> {
    assert_eq!(
        in_dim % 256,
        0,
        "Q4_K weight in_dim must be a multiple of 256 (got {in_dim})"
    );
    let blocks_per_row = in_dim / 256;
    let bytes_per_row = blocks_per_row * 144;
    let mut buf = vec![0u8; out_dim * bytes_per_row];
    let block = q4k_uniform_block(value);
    for row in 0..out_dim {
        let row_off = row * bytes_per_row;
        for b in 0..blocks_per_row {
            let block_off = row_off + b * 144;
            buf[block_off..block_off + 144].copy_from_slice(&block);
        }
    }
    buf
}

// ── Fixture: minimal DeepSeek2 weights for one transformer layer ─────

struct Layer {
    // MoE FFN
    router: Vec<f32>,     // [n_experts × hidden]
    gate_exps: Vec<u8>,   // [n_experts × intermediate × hidden] Q4_K
    up_exps: Vec<u8>,     //   "
    down_exps: Vec<u8>,   // [n_experts × hidden × intermediate] Q4_K
    shared_gate: Vec<u8>, // [intermediate × hidden] Q4_K
    shared_up: Vec<u8>,   //   "
    shared_down: Vec<u8>, // [hidden × intermediate] Q4_K

    // MLA attention
    kv_a_mqa: Vec<u8>, // [hidden → kv_lora_rank+qk_rope] Q4_K  (out_dim=KV_A_OUT, in_dim=HIDDEN)
    kv_b: Vec<u8>,     // [kv_lora_rank → n_heads×(nope+v)] Q4_K
    q_a: Vec<u8>,      // [hidden → q_lora_rank] Q4_K
    q_b: Vec<u8>,      // [q_lora_rank → n_heads×(nope+rope)] Q4_K
    attn_output: Vec<u8>, // [n_heads×v_head → hidden] Q4_K
    kv_a_norm: Vec<f32>, // [kv_lora_rank]
    q_a_norm: Vec<f32>, // [q_lora_rank]
    k_norm: Vec<f32>,  // [qk_rope_head_dim]
}

fn build_layer() -> Layer {
    Layer {
        // Router weights: small mixed pattern so different experts win.
        // F32, shape [n_experts × hidden]. Experts 0,1,2,3 get increasing
        // bias against a constant input.
        router: {
            let mut r = vec![0.0f32; N_EXPERTS * HIDDEN];
            for e in 0..N_EXPERTS {
                for i in 0..HIDDEN {
                    // Asymmetric pattern: expert 0 cold, expert 3 hot.
                    r[e * HIDDEN + i] = 0.001 * (e as f32) + 0.0001 * ((i % 8) as f32);
                }
            }
            r
        },

        gate_exps: q4k_uniform_matrix(0.05, N_EXPERTS * INTERMEDIATE, HIDDEN),
        up_exps: q4k_uniform_matrix(0.04, N_EXPERTS * INTERMEDIATE, HIDDEN),
        down_exps: q4k_uniform_matrix(0.01, N_EXPERTS * HIDDEN, INTERMEDIATE),
        shared_gate: q4k_uniform_matrix(0.03, INTERMEDIATE, HIDDEN),
        shared_up: q4k_uniform_matrix(0.03, INTERMEDIATE, HIDDEN),
        shared_down: q4k_uniform_matrix(0.01, HIDDEN, INTERMEDIATE),

        kv_a_mqa: q4k_uniform_matrix(0.02, KV_A_OUT, HIDDEN),
        kv_b: q4k_uniform_matrix(0.02, KV_B_OUT, KV_LORA_RANK),
        q_a: q4k_uniform_matrix(0.02, Q_LORA_RANK, HIDDEN),
        q_b: q4k_uniform_matrix(0.02, Q_B_OUT, Q_LORA_RANK),
        attn_output: q4k_uniform_matrix(0.02, HIDDEN, O_IN),

        kv_a_norm: vec![1.0; KV_LORA_RANK],
        q_a_norm: vec![1.0; Q_LORA_RANK],
        k_norm: vec![1.0; QK_ROPE],
    }
}

// ── Pipeline driver: returns the post-pipeline hidden state ───────────

fn run_one_token(layer: &Layer, input_seed: f32) -> Vec<f32> {
    // Synthetic input hidden state: deterministic but non-uniform.
    let mut hidden: Vec<f32> = (0..HIDDEN)
        .map(|i| input_seed + 0.01 * ((i % 16) as f32))
        .collect();

    // ── MoE FFN scratch ────────────────────────────────────────────
    let mut router_score = vec![0.0f32; N_EXPERTS];
    let mut expert_indices = vec![0u32; TOP_K];
    let mut expert_weights = vec![0.0f32; TOP_K];
    let mut gate_buf = vec![0.0f32; INTERMEDIATE];
    let mut up_buf = vec![0.0f32; INTERMEDIATE];
    let mut hidden_buf = vec![0.0f32; INTERMEDIATE];
    let mut expert_out = vec![0.0f32; HIDDEN];
    let mut moe_out = vec![0.0f32; HIDDEN];
    let mut lin = LinearScratch::new();

    moe_ffn(
        &hidden,
        &layer.router,
        &[], // no per-expert bias in this scaffolded test layer
        &layer.gate_exps,
        &layer.up_exps,
        &layer.down_exps,
        &layer.shared_gate,
        &layer.shared_up,
        &layer.shared_down,
        N_EXPERTS,
        TOP_K,
        HIDDEN,
        INTERMEDIATE,
        &mut router_score,
        &mut expert_indices,
        &mut expert_weights,
        &mut gate_buf,
        &mut up_buf,
        &mut hidden_buf,
        &mut expert_out,
        &mut moe_out,
        &mut lin,
        zero_gguf_parser::GgmlType::Q4K,
        MoeRoutingMode::Softmax,
        1.0,
    );

    // Residual: hidden += moe_out
    for i in 0..HIDDEN {
        hidden[i] += moe_out[i];
    }

    // ── MLA attention scratch (compressed-latent cache) ────────────
    let mut kv_storage =
        vec![0.0f32; MlaKvCache::required_f32(MAX_TOKENS, 1, KV_LORA_RANK, QK_ROPE)];
    let mut kv_cache = MlaKvCache::new(
        kv_storage.as_mut_ptr(),
        MAX_TOKENS,
        1,
        KV_LORA_RANK,
        QK_ROPE,
    );

    let mut c_kv_rope_buf = vec![0.0f32; KV_A_OUT];
    let mut c_kv_norm_buf = vec![0.0f32; KV_LORA_RANK];
    let mut kv_decompressed = vec![0.0f32; KV_B_OUT];
    let mut c_q_buf = vec![0.0f32; Q_LORA_RANK];
    let mut c_q_norm_buf = vec![0.0f32; Q_LORA_RANK];
    let mut q_decompressed = vec![0.0f32; Q_B_OUT];
    let mut k_assembled = vec![0.0f32; N_HEADS * TOTAL_K_HEAD];
    // score_buf is now per-head striped: n_heads × max_tokens.
    let mut score_buf = vec![0.0f32; N_HEADS * MAX_TOKENS];
    let mut attn_head_buf = vec![0.0f32; V_HEAD];
    let mut attn_concat = vec![0.0f32; N_HEADS * V_HEAD];
    let mut attn_out = vec![0.0f32; HIDDEN];

    let weights = MlaWeights {
        kv_a_mqa: &layer.kv_a_mqa,
        kv_a_norm: &layer.kv_a_norm,
        kv_b: &layer.kv_b,
        k_b: &[],
        v_b: &[],
        q_a: &layer.q_a,
        q_a_norm: &layer.q_a_norm,
        q_b: &layer.q_b,
        output: &layer.attn_output,
        k_norm: &layer.k_norm,
    };

    mla_attention_single_token(
        &hidden,
        &weights,
        /* layer_idx */ 0,
        /* token_offset */ 0,
        N_HEADS,
        KV_LORA_RANK,
        QK_NOPE,
        QK_ROPE,
        V_HEAD,
        Q_LORA_RANK,
        HIDDEN,
        RMS_EPS,
        ROPE_FREQ_BASE,
        &mut kv_cache,
        &mut c_kv_rope_buf,
        &mut c_kv_norm_buf,
        &mut kv_decompressed,
        &mut c_q_buf,
        &mut c_q_norm_buf,
        &mut q_decompressed,
        &mut k_assembled,
        &mut score_buf,
        &mut attn_head_buf,
        &mut attn_concat,
        &mut attn_out,
        &mut lin,
        zero_gguf_parser::GgmlType::Q4K,
    )
    .expect("MLA attention failed on synthetic inputs");

    // Residual: hidden += attn_out
    for i in 0..HIDDEN {
        hidden[i] += attn_out[i];
    }

    hidden
}

// ── Tests ────────────────────────────────────────────────────────────

#[test]
fn q4k_uniform_block_dequants_to_constant() {
    // Independent cross-check of our helper against the sacred dequantiser:
    // every output must equal `value` exactly (modulo fp16 round-trip).
    let value = 0.0625_f32; // exactly representable in fp16
    let block = q4k_uniform_block(value);
    let mut out = [0.0f32; 256];
    zero_gguf_parser::dequant::dequant_q4k_block(&block, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - value).abs() < 1e-6,
            "q4k_uniform_block[{i}] = {v}, expected {value}"
        );
    }
}

#[test]
fn deepseek2_moe_pipeline_runs_end_to_end_without_nans() {
    let layer = build_layer();
    let out = run_one_token(&layer, /* input_seed */ 0.1);

    assert_eq!(out.len(), HIDDEN);
    let mut max_abs = 0.0_f32;
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "hidden[{i}] = {v} is non-finite");
        if v.abs() > max_abs {
            max_abs = v.abs();
        }
    }
    // Sanity: the pipeline produced a non-trivial signal — not all zeros,
    // not blown up to absurd magnitudes. Bounds are loose; the assertion
    // exists to catch accidental "everything is zero" regressions.
    assert!(
        max_abs > 1e-6,
        "post-pipeline hidden state is suspiciously small (max_abs = {max_abs}); pipeline may be zeroing"
    );
    assert!(
        max_abs < 1e6,
        "post-pipeline hidden state blew up (max_abs = {max_abs})"
    );
}

#[test]
fn deepseek2_moe_pipeline_is_bit_exact_deterministic() {
    // Run the same configuration twice and require byte-for-byte
    // identical outputs. f32 deterministic-equality is the strongest
    // guarantee we can make for a CPU pipeline; any drift here points
    // at uninitialised scratch, nondeterministic iteration order, or
    // a stray RNG.
    let layer = build_layer();
    let a = run_one_token(&layer, 0.1);
    let b = run_one_token(&layer, 0.1);

    assert_eq!(a.len(), b.len());
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "non-deterministic output at index {i}: run-1 bits = 0x{:08x}, run-2 bits = 0x{:08x}",
            va.to_bits(),
            vb.to_bits(),
        );
    }
}

#[test]
fn moe_route_picks_expected_top_k_for_skewed_router() {
    // Direct sanity check on routing: with a router weighted toward
    // larger expert indices, top-K must include the largest indices.
    let layer = build_layer();
    let hidden: Vec<f32> = (0..HIDDEN).map(|i| 0.1 + 0.01 * (i as f32)).collect();

    let mut scores = vec![0.0f32; N_EXPERTS];
    let mut idx = vec![0u32; TOP_K];
    let mut weights = vec![0.0f32; TOP_K];

    moe_route_f32(
        &hidden,
        &layer.router,
        &[],
        N_EXPERTS,
        TOP_K,
        &mut scores,
        &mut idx,
        &mut weights,
        MoeRoutingMode::Softmax,
        1.0,
    );

    // Router weights grow with expert index → highest two expert IDs
    // (2 and 3) must appear in the top-K set.
    let selected: std::collections::BTreeSet<u32> = idx.iter().copied().collect();
    assert!(
        selected.contains(&2) && selected.contains(&3),
        "top-K experts = {idx:?}, expected to contain {{2, 3}}"
    );

    // Softmax weights sum to ~1.
    let sum: f32 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax weights sum = {sum}, expected ≈ 1.0"
    );
}

// ── Bug-hunt regression tests (Phase 1 audit) ────────────────────────

#[test]
#[should_panic(expected = "top_k must be > 0")]
fn moe_route_zero_top_k_panics() {
    // Regression: prior to the audit, top_k=0 read past empty weights_buf
    // and panicked with an opaque OOB index. The current contract asserts
    // top_k > 0 with a clear diagnostic message.
    let hidden = vec![0.5_f32; HIDDEN];
    let router = vec![0.0_f32; N_EXPERTS * HIDDEN];
    let mut scores = vec![0.0_f32; N_EXPERTS];
    let mut idx: Vec<u32> = Vec::new();
    let mut wts: Vec<f32> = Vec::new();
    moe_route_f32(
        &hidden,
        &router,
        &[],
        N_EXPERTS,
        0,
        &mut scores,
        &mut idx,
        &mut wts,
        MoeRoutingMode::Softmax,
        1.0,
    );
}

#[test]
#[should_panic(expected = "top_k must be <= n_experts")]
fn moe_route_top_k_exceeds_n_experts_panics() {
    // Regression: prior to the audit, the seed-fill loop indexed score_buf
    // beyond [0, n_experts), triggering an opaque OOB index panic. Now we
    // assert the contract up front with a clear message.
    let hidden = vec![0.5_f32; HIDDEN];
    let router = vec![0.0_f32; N_EXPERTS * HIDDEN];
    let mut scores = vec![0.0_f32; N_EXPERTS];
    let mut idx = vec![0_u32; N_EXPERTS + 1];
    let mut wts = vec![0.0_f32; N_EXPERTS + 1];
    moe_route_f32(
        &hidden,
        &router,
        &[],
        N_EXPERTS,
        N_EXPERTS + 1,
        &mut scores,
        &mut idx,
        &mut wts,
        MoeRoutingMode::Softmax,
        1.0,
    );
}

#[test]
fn moe_route_nan_router_falls_back_to_uniform() {
    // Regression: a NaN in any router row used to propagate through
    // softmax and emit NaN expert weights, which then zeroed out every
    // weighted expert contribution silently. The new NaN guard detects
    // a non-finite `max_score` and falls back to a uniform 1/top_k
    // distribution so the layer still produces a finite output.
    let hidden = vec![1.0_f32; HIDDEN];
    let mut router = vec![0.0_f32; N_EXPERTS * HIDDEN];
    router[0] = f32::NAN; // poison the first expert's row
    let mut scores = vec![0.0_f32; N_EXPERTS];
    let mut idx = vec![0_u32; TOP_K];
    let mut wts = vec![0.0_f32; TOP_K];
    moe_route_f32(
        &hidden,
        &router,
        &[],
        N_EXPERTS,
        TOP_K,
        &mut scores,
        &mut idx,
        &mut wts,
        MoeRoutingMode::Softmax,
        1.0,
    );
    let sum: f32 = wts.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6 && wts.iter().all(|w| w.is_finite()),
        "NaN router should fall back to a finite uniform distribution: weights = {:?}",
        wts
    );
    for &w in wts.iter() {
        assert!(
            (w - 1.0 / TOP_K as f32).abs() < 1e-6,
            "uniform-fallback weight = {w}, expected {}",
            1.0 / TOP_K as f32
        );
    }
}

#[test]
fn moe_route_all_identical_scores_uniform_softmax() {
    // Sanity: with a degenerate (all-zero) router, every expert scores
    // exactly 0; softmax should produce a uniform 1/top_k distribution
    // (not NaN, not all-zero) via the standard max-subtraction path.
    let hidden = vec![1.0_f32; HIDDEN];
    let router = vec![0.0_f32; N_EXPERTS * HIDDEN];
    let mut scores = vec![0.0_f32; N_EXPERTS];
    let mut idx = vec![0_u32; TOP_K];
    let mut wts = vec![0.0_f32; TOP_K];
    moe_route_f32(
        &hidden,
        &router,
        &[],
        N_EXPERTS,
        TOP_K,
        &mut scores,
        &mut idx,
        &mut wts,
        MoeRoutingMode::Softmax,
        1.0,
    );
    let expected = 1.0_f32 / TOP_K as f32;
    for &w in wts.iter() {
        assert!(
            (w - expected).abs() < 1e-6,
            "degenerate router weight = {w}, expected {expected}"
        );
    }
}

#[test]
fn moe_route_sigmoid_normalize_sums_to_one() {
    // SPEC (DeepSeek-V3 / Kimi K2.6): SigmoidNormalize must produce
    // top-K weights that sum to exactly 1 (modulo float epsilon) and
    // are pairwise positive (sigmoid range is (0, 1) → normalised
    // values stay strictly positive).
    let layer = build_layer();
    let hidden: Vec<f32> = (0..HIDDEN).map(|i| 0.1 + 0.01 * (i as f32)).collect();

    let mut scores = vec![0.0f32; N_EXPERTS];
    let mut idx = vec![0u32; TOP_K];
    let mut weights = vec![0.0f32; TOP_K];

    moe_route_f32(
        &hidden,
        &layer.router,
        &[],
        N_EXPERTS,
        TOP_K,
        &mut scores,
        &mut idx,
        &mut weights,
        MoeRoutingMode::SigmoidNormalize,
        1.0,
    );
    let sum: f32 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "sigmoid+normalize weights sum = {sum}, expected ≈ 1.0"
    );
    for &w in weights.iter() {
        assert!(
            w > 0.0 && w < 1.0 && w.is_finite(),
            "weight {w} out of (0, 1)"
        );
    }
}

#[test]
fn moe_route_sigmoid_normalize_applies_expert_weight_scale() {
    // DeepSeek-V3/Kimi-family models can carry a routed expert scale in
    // config. The normalised top-K weights must be multiplied by that
    // value; silently falling back to 1.0 changes MoE output quality.
    let layer = build_layer();
    let hidden: Vec<f32> = (0..HIDDEN).map(|i| 0.1 + 0.01 * (i as f32)).collect();

    let mut scores = vec![0.0f32; N_EXPERTS];
    let mut idx = vec![0u32; TOP_K];
    let mut weights = vec![0.0f32; TOP_K];

    moe_route_f32(
        &hidden,
        &layer.router,
        &[],
        N_EXPERTS,
        TOP_K,
        &mut scores,
        &mut idx,
        &mut weights,
        MoeRoutingMode::SigmoidNormalize,
        2.5,
    );

    let sum: f32 = weights.iter().sum();
    assert!(
        (sum - 2.5).abs() < 1e-5,
        "scaled sigmoid+normalize weights sum = {sum}, expected ≈ 2.5"
    );
}

#[test]
fn moe_route_uniform_fallback_preserves_expert_weight_scale() {
    // If router math ever produces NaN, the guard falls back to uniform
    // routing. DeepSeek/Kimi still need the routed scaling factor applied;
    // losing it here would create a silent quality-only regression.
    let hidden = vec![1.0_f32; HIDDEN];
    let router = vec![f32::NAN; N_EXPERTS * HIDDEN];
    let mut scores = vec![0.0_f32; N_EXPERTS];
    let mut idx = vec![0u32; TOP_K];
    let mut weights = vec![0.0_f32; TOP_K];

    moe_route_f32(
        &hidden,
        &router,
        &[],
        N_EXPERTS,
        TOP_K,
        &mut scores,
        &mut idx,
        &mut weights,
        MoeRoutingMode::SigmoidNormalize,
        2.5,
    );

    let sum: f32 = weights.iter().sum();
    assert!(
        (sum - 2.5).abs() < 1e-5,
        "scaled fallback weights sum = {sum}, expected ≈ 2.5"
    );
    for &w in weights.iter() {
        assert!(
            (w - 1.25).abs() < 1e-5,
            "scaled fallback weight = {w}, expected 1.25"
        );
    }
}

#[test]
fn moe_route_bias_shifts_top_k_selection() {
    // SPEC (DeepSeek-V3 §3.1 / Kimi K2.6): the per-expert router bias
    // (`ffn_gate_inp_bias.weight` / `exp_probs_b.weight`) shifts ONLY
    // the top-K ranking (selection key s + bias). The final weights are
    // the unbiased sigmoid gate values of the selected experts. With an
    // all-zero router every gate value is sigmoid(0) = 0.5, so the
    // biased expert must be selected but must NOT receive a larger
    // weight than its peers — the normalised weights stay uniform.
    let hidden = vec![0.5_f32; HIDDEN];
    let router = vec![0.0_f32; N_EXPERTS * HIDDEN]; // every gate s = 0.5
    let mut bias = vec![0.0_f32; N_EXPERTS];
    bias[N_EXPERTS - 1] = 10.0; // strongly favour the last expert
    let mut scores = vec![0.0_f32; N_EXPERTS];
    let mut idx = vec![0u32; TOP_K];
    let mut wts = vec![0.0_f32; TOP_K];
    moe_route_f32(
        &hidden,
        &router,
        &bias,
        N_EXPERTS,
        TOP_K,
        &mut scores,
        &mut idx,
        &mut wts,
        MoeRoutingMode::SigmoidNormalize,
        1.0,
    );
    let selected: std::collections::BTreeSet<u32> = idx.iter().copied().collect();
    assert!(
        selected.contains(&((N_EXPERTS - 1) as u32)),
        "biased expert must appear in top-K: idx = {idx:?}"
    );
    // Bias must not leak into the weights: all selected experts share
    // the same unbiased gate value, so every weight is exactly 1/TOP_K.
    let uniform = 1.0 / TOP_K as f32;
    for (k, &w) in wts.iter().enumerate() {
        assert!(
            (w - uniform).abs() < 1e-6,
            "bias leaked into weight[{k}] = {w}, expected uniform {uniform} \
             (weights must be the unbiased sigmoid gate values)"
        );
    }
    let sum: f32 = wts.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "sigmoid+normalize weights sum = {sum}, expected ≈ 1.0"
    );
}

#[test]
fn expert_swiglu_is_bit_exact_deterministic() {
    // Per-expert SwiGLU is the deepest call inside moe_ffn. Verifying it
    // independently rules out the expert-level path as a source of any
    // future MoE nondeterminism.
    let gate = q4k_uniform_matrix(0.05, INTERMEDIATE, HIDDEN);
    let up = q4k_uniform_matrix(0.04, INTERMEDIATE, HIDDEN);
    let down = q4k_uniform_matrix(0.01, HIDDEN, INTERMEDIATE);

    let input: Vec<f32> = (0..HIDDEN).map(|i| 0.05 + 0.01 * (i as f32)).collect();

    let mut gate_buf_1 = vec![0.0f32; INTERMEDIATE];
    let mut up_buf_1 = vec![0.0f32; INTERMEDIATE];
    let mut hidden_buf_1 = vec![0.0f32; INTERMEDIATE];
    let mut out_1 = vec![0.0f32; HIDDEN];
    let mut lin_1 = LinearScratch::new();

    expert_swiglu(
        &input,
        &gate,
        &up,
        &down,
        HIDDEN,
        INTERMEDIATE,
        &mut gate_buf_1,
        &mut up_buf_1,
        &mut hidden_buf_1,
        &mut out_1,
        &mut lin_1,
        zero_gguf_parser::GgmlType::Q4K,
    );

    let mut gate_buf_2 = vec![0.0f32; INTERMEDIATE];
    let mut up_buf_2 = vec![0.0f32; INTERMEDIATE];
    let mut hidden_buf_2 = vec![0.0f32; INTERMEDIATE];
    let mut out_2 = vec![0.0f32; HIDDEN];
    let mut lin_2 = LinearScratch::new();

    expert_swiglu(
        &input,
        &gate,
        &up,
        &down,
        HIDDEN,
        INTERMEDIATE,
        &mut gate_buf_2,
        &mut up_buf_2,
        &mut hidden_buf_2,
        &mut out_2,
        &mut lin_2,
        zero_gguf_parser::GgmlType::Q4K,
    );

    for i in 0..HIDDEN {
        assert_eq!(
            out_1[i].to_bits(),
            out_2[i].to_bits(),
            "expert_swiglu non-deterministic at index {i}"
        );
    }
}

// ── Phase-2 audit regression tests — per-quant dispatch ─────────────

/// Build a token-embedding tensor for `vocab_size` tokens × `hidden`
/// elements per row, in the given quant. Every row carries the same
/// pattern so per-token lookups produce a known reference output.
fn build_embed_table(
    vocab_size: usize,
    hidden: usize,
    quant: zero_gguf_parser::GgmlType,
) -> Vec<u8> {
    use zero_gguf_parser::dequant::{
        Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE,
    };
    use zero_gguf_parser::GgmlType;
    match quant {
        GgmlType::Q4_0 => {
            // Q4_0 dequant: `(nibble - 8) * d`. For each output to be
            // +1.0 at d=1.0 we need nibble = 9 → byte = 0x99.
            let block = {
                let mut b = [0u8; Q4_0_BLOCK_BYTES];
                b[0] = 0x00;
                b[1] = 0x3C; // d = 1.0 (fp16)
                for k in 2..Q4_0_BLOCK_BYTES {
                    b[k] = 0x99; // both nibbles = 9 → (9-8)*1.0 = 1.0
                }
                b
            };
            let blocks_per_row = hidden / Q4_0_BLOCK_SIZE;
            let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
            let mut buf = vec![0u8; vocab_size * bytes_per_row];
            for r in 0..vocab_size {
                let off = r * bytes_per_row;
                for b in 0..blocks_per_row {
                    let bo = off + b * Q4_0_BLOCK_BYTES;
                    buf[bo..bo + Q4_0_BLOCK_BYTES].copy_from_slice(&block);
                }
            }
            buf
        }
        GgmlType::Q8_0 => {
            // Per-row pattern: d=1.0, every i8 = +1 → dequant = 1.0.
            let block = {
                let mut b = [0u8; Q8_0_BLOCK_BYTES];
                b[0] = 0x00;
                b[1] = 0x3C; // d = 1.0
                for k in 0..Q8_0_BLOCK_SIZE {
                    b[2 + k] = 1u8;
                }
                b
            };
            let blocks_per_row = hidden / Q8_0_BLOCK_SIZE;
            let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;
            let mut buf = vec![0u8; vocab_size * bytes_per_row];
            for r in 0..vocab_size {
                let off = r * bytes_per_row;
                for b in 0..blocks_per_row {
                    let bo = off + b * Q8_0_BLOCK_BYTES;
                    buf[bo..bo + Q8_0_BLOCK_BYTES].copy_from_slice(&block);
                }
            }
            buf
        }
        _ => panic!("build_embed_table only supports Q4_0 / Q8_0"),
    }
}

#[test]
fn embed_lookup_dispatch_routes_q4_0() {
    // Q4_0 token-embedding lookup must produce all-1.0 hidden state
    // when the table is the unit block. Critical because the Kimi K2.6
    // forward pass uses this path for every prefill / generation step.
    use zero_llm_inference::forward_pass::embed_lookup_dispatch;
    let vocab = 16;
    let hidden = 64;
    let table = build_embed_table(vocab, hidden, zero_gguf_parser::GgmlType::Q4_0);
    let mut out = vec![0.0f32; hidden];
    embed_lookup_dispatch(
        7,
        &table,
        hidden,
        &mut out,
        zero_gguf_parser::GgmlType::Q4_0,
    )
    .expect("Q4_0 embed_lookup_dispatch must succeed");
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-4,
            "Q4_0 embed[{}] of token 7 = {}, want 1.0",
            i,
            v
        );
    }
}

#[test]
fn embed_lookup_dispatch_routes_q8_0() {
    // CRITICAL: bartowski Kimi K2.6 Q4_0 GGUFs keep token_embd.weight
    // at Q8_0. Without Q8_0 dispatch in embed_lookup_dispatch, the
    // very first token of generation has no hidden state and the LLM
    // cannot boot. This test pins the routing so a future refactor
    // that drops Q8_0 fails loud.
    use zero_llm_inference::forward_pass::embed_lookup_dispatch;
    let vocab = 16;
    let hidden = 64;
    let table = build_embed_table(vocab, hidden, zero_gguf_parser::GgmlType::Q8_0);
    let mut out = vec![0.0f32; hidden];
    embed_lookup_dispatch(
        7,
        &table,
        hidden,
        &mut out,
        zero_gguf_parser::GgmlType::Q8_0,
    )
    .expect("Q8_0 embed_lookup_dispatch must succeed (Kimi K2.6 critical path)");
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-4,
            "Q8_0 embed[{}] of token 7 = {}, want 1.0",
            i,
            v
        );
    }
}

#[test]
fn embed_lookup_dispatch_rejects_unsupported_quant() {
    // F16 is a real GgmlType but we don't dequant it here; ensure the
    // dispatcher returns an explicit error so the boot path doesn't
    // silently dispatch garbage.
    use zero_llm_inference::forward_pass::embed_lookup_dispatch;
    let mut out = vec![0.0f32; 256];
    let r = embed_lookup_dispatch(
        0,
        &[0u8; 16],
        256,
        &mut out,
        zero_gguf_parser::GgmlType::F16,
    );
    assert!(r.is_err(), "embed_lookup_dispatch must reject F16");
}

#[test]
fn embed_lookup_dispatch_rejects_dim_misalignment() {
    // hidden = 33 is not a multiple of any block size (Q4_K = 256,
    // Q4_0 = 32, Q8_0 = 32). The dispatcher must catch this rather
    // than panic in the dequantiser's debug_assert.
    use zero_llm_inference::forward_pass::embed_lookup_dispatch;
    let mut out = vec![0.0f32; 33];
    let r = embed_lookup_dispatch(
        0,
        &[0u8; 256],
        33,
        &mut out,
        zero_gguf_parser::GgmlType::Q4_0,
    );
    assert!(
        r.is_err(),
        "embed_lookup_dispatch must reject misaligned hidden dim"
    );
}

// ── Phase-2 audit: EOS edge cases ────────────────────────────────────

#[test]
fn lm_head_argmax_dynamic_routes_q8_0() {
    // The LM head matmul for Kimi K2.6 Q4_0 builds is Q8_0. Verify
    // that lm_head_argmax_dynamic accepts OutputQuant::Q8_0 and
    // produces a numerically-sane argmax (here we just check it
    // doesn't panic + returns a valid token id below vocab_size).
    use zero_llm_inference::{lm_head_argmax_dynamic, LinearScratch, OutputQuant};
    // Use a small vocab to keep the test fast.
    let vocab_padded = 32;
    let vocab_real = 32;
    let hidden = 32;
    let hidden_state = [0.1f32; 32];
    let output_norm = [1.0f32; 32];
    // Q8_0 output weight: d=1.0, every i8 = +1 → row dequants to 1.0s.
    let mut output_w = vec![0u8; vocab_padded * 34];
    for row in 0..vocab_padded {
        let off = row * 34;
        output_w[off] = 0x00;
        output_w[off + 1] = 0x3C;
        for k in 0..32 {
            output_w[off + 2 + k] = 1u8;
        }
    }
    let mut norm_buf = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab_padded];
    let mut scratch = LinearScratch::new();
    let r = lm_head_argmax_dynamic(
        &hidden_state,
        &output_norm,
        &output_w,
        OutputQuant::Q8_0,
        1e-6,
        hidden,
        vocab_padded,
        vocab_real,
        &mut norm_buf,
        &mut logits,
        &mut scratch,
    );
    let tok = r.expect("Q8_0 lm_head_argmax_dynamic must succeed");
    assert!(
        (tok as usize) < vocab_real,
        "argmax = {}, vocab_real = {}",
        tok,
        vocab_real
    );
}
