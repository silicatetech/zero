// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::needless_range_loop, clippy::identity_op)]
//! Bit-exact verification of attention forward-pass against Sub-MP-C1 reference dumps.
//!
//! Tests load Layer 0 weights from GGUF and reference tensors from binary dumps,
//! then verify each intermediate step matches within tolerance.

use std::path::Path;

use zero_gguf_parser::{parse_header, GgmlType};
use zero_llm_inference::*;

const TOLERANCE: f32 = 5e-4;

const GGUF_PATH: &str = "../../kernel/programs/Qwen_Qwen3-1.7B-Q4_K_M.gguf";
const REF_DIR: &str = "tests/reference-dumps";

// Model constants (verified in Sub-MP-C1)
const EMBEDDING_DIM: usize = 2048;
const N_Q_HEADS: usize = 16;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const Q_DIM: usize = N_Q_HEADS * HEAD_DIM; // 2048
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 1024
const RMS_EPS: f32 = 1e-6;

fn load_reference(name: &str) -> Vec<f32> {
    let path = format!("{}/{}.bin", REF_DIR, name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to load {}: {}", path, e));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], name: &str, tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch: actual {} vs expected {}",
        name,
        actual.len(),
        expected.len()
    );
    let mut max_diff: f32 = 0.0;
    let mut max_idx = 0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    assert!(
        max_diff < tol,
        "{}[{}]: max_diff={:.6e} (got {}, expected {}), tolerance={}",
        name,
        max_idx,
        max_diff,
        actual[max_idx],
        expected[max_idx],
        tol,
    );
}

/// Helper to get tensor bytes from parsed GGUF metadata + raw file data.
fn get_tensor_bytes<'a>(
    file_data: &'a [u8],
    meta: &zero_gguf_parser::GgufMetadata,
    name: &str,
) -> &'a [u8] {
    for t in &meta.tensors {
        if t.name == name {
            let start = meta.tensor_data_offset + t.offset as usize;
            // Calculate size from element count and type
            let n_elements = t.element_count() as usize;
            let bytes_per_block = match t.tensor_type {
                GgmlType::F32 => return &file_data[start..start + n_elements * 4],
                GgmlType::Q4K => 144,
                GgmlType::Q6K => 210,
                _ => panic!("Unsupported type for {}: {:?}", name, t.tensor_type),
            };
            let n_blocks = n_elements / 256;
            let size = n_blocks * bytes_per_block;
            return &file_data[start..start + size];
        }
    }
    panic!("Tensor '{}' not found", name);
}

fn get_f32_tensor(file_data: &[u8], meta: &zero_gguf_parser::GgufMetadata, name: &str) -> Vec<f32> {
    let bytes = get_tensor_bytes(file_data, meta, name);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn test_attention_reference_layer_0_step_by_step() {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("GGUF not found at {}, skipping reference test", GGUF_PATH);
        return;
    }

    let file_data = std::fs::read(GGUF_PATH).expect("Failed to read GGUF");
    let meta = parse_header(&file_data).expect("Failed to parse GGUF header");

    // Load weights
    let attn_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.attn_norm.weight");
    let q_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_q.weight");
    let k_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_k.weight");
    let v_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_v.weight");
    let q_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.attn_q_norm.weight");
    let k_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.attn_k_norm.weight");

    // Deterministic ramp input (same as Sub-MP-C1)
    let mut input = vec![0.0f32; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        input[i] = 0.01 * (i as f32 + 1.0);
    }

    // Load references
    let ref_normed = load_reference("01_normed_input");
    let ref_q_proj = load_reference("02_q_projected");
    let ref_k_proj = load_reference("03_k_projected");
    let ref_v_proj = load_reference("04_v_projected");
    let ref_q_qknorm = load_reference("05_q_post_qknorm");
    let ref_k_qknorm = load_reference("06_k_post_qknorm");
    let ref_q_rope = load_reference("07_q_post_rope");
    let ref_k_rope = load_reference("08_k_post_rope");

    let mut scratch = LinearScratch::new();

    // Step 1: RMSNorm
    let mut normed = vec![0.0f32; EMBEDDING_DIM];
    rmsnorm(&input, &attn_norm_w, &mut normed, RMS_EPS);
    assert_close(&normed, &ref_normed, "01_normed_input", TOLERANCE);

    // Step 2: Q projection
    let mut q_proj = vec![0.0f32; Q_DIM];
    linear_q4k(
        &normed,
        q_weight,
        &mut q_proj,
        &mut scratch,
        EMBEDDING_DIM,
        Q_DIM,
    );
    assert_close(&q_proj, &ref_q_proj, "02_q_projected", TOLERANCE);

    // Step 3: K projection
    let mut k_proj = vec![0.0f32; KV_DIM];
    linear_q4k(
        &normed,
        k_weight,
        &mut k_proj,
        &mut scratch,
        EMBEDDING_DIM,
        KV_DIM,
    );
    assert_close(&k_proj, &ref_k_proj, "03_k_projected", TOLERANCE);

    // Step 4: V projection (Q6_K!)
    let mut v_proj = vec![0.0f32; KV_DIM];
    linear_q6k(
        &normed,
        v_weight,
        &mut v_proj,
        &mut scratch,
        EMBEDDING_DIM,
        KV_DIM,
    );
    assert_close(&v_proj, &ref_v_proj, "04_v_projected", TOLERANCE);

    // Step 5: Q-Norm (per-head RMSNorm)
    let mut q_normed = vec![0.0f32; Q_DIM];
    for h in 0..N_Q_HEADS {
        let off = h * HEAD_DIM;
        rmsnorm(
            &q_proj[off..off + HEAD_DIM],
            &q_norm_w,
            &mut q_normed[off..off + HEAD_DIM],
            RMS_EPS,
        );
    }
    assert_close(&q_normed, &ref_q_qknorm, "05_q_post_qknorm", TOLERANCE);

    // Step 6: K-Norm (per-head RMSNorm)
    let mut k_normed = vec![0.0f32; KV_DIM];
    for h in 0..N_KV_HEADS {
        let off = h * HEAD_DIM;
        rmsnorm(
            &k_proj[off..off + HEAD_DIM],
            &k_norm_w,
            &mut k_normed[off..off + HEAD_DIM],
            RMS_EPS,
        );
    }
    assert_close(&k_normed, &ref_k_qknorm, "06_k_post_qknorm", TOLERANCE);

    // Step 7: RoPE on Q (Variante B, position=0)
    let rope_ctx = RopeContext::<64>::new(1_000_000.0);
    for h in 0..N_Q_HEADS {
        let off = h * HEAD_DIM;
        rope(&mut q_normed[off..off + HEAD_DIM], &rope_ctx, 0);
    }
    assert_close(&q_normed, &ref_q_rope, "07_q_post_rope", TOLERANCE);

    // Step 8: RoPE on K
    for h in 0..N_KV_HEADS {
        let off = h * HEAD_DIM;
        rope(&mut k_normed[off..off + HEAD_DIM], &rope_ctx, 0);
    }
    assert_close(&k_normed, &ref_k_rope, "08_k_post_rope", TOLERANCE);
}

#[test]
fn test_attention_reference_single_token_full_pipeline() {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("GGUF not found, skipping");
        return;
    }

    let file_data = std::fs::read(GGUF_PATH).expect("Failed to read GGUF");
    let meta = parse_header(&file_data).expect("Failed to parse GGUF header");

    let attn_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.attn_norm.weight");
    let q_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_q.weight");
    let k_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_k.weight");
    let v_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_v.weight");
    let o_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_output.weight");
    let q_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.attn_q_norm.weight");
    let k_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.attn_k_norm.weight");

    // Ramp input → RMSNorm
    let mut input = vec![0.0f32; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        input[i] = 0.01 * (i as f32 + 1.0);
    }
    let mut normed = vec![0.0f32; EMBEDDING_DIM];
    rmsnorm(&input, &attn_norm_w, &mut normed, RMS_EPS);

    // KvCache setup
    let max_tokens = 16;
    let total = 2 * 1 * max_tokens * N_KV_HEADS * HEAD_DIM; // 1 layer
    let mut storage = vec![0.0f32; total];
    let mut kv_cache = KvCache::new(storage.as_mut_ptr(), max_tokens, 1, N_KV_HEADS, HEAD_DIM);

    // Scratch buffers
    let mut q_buf = vec![0.0f32; Q_DIM];
    let mut k_buf = vec![0.0f32; KV_DIM];
    let mut v_buf = vec![0.0f32; KV_DIM];
    let mut q_out = vec![0.0f32; Q_DIM];
    let mut k_out = vec![0.0f32; KV_DIM];
    let mut score_buf = vec![0.0f32; max_tokens];
    let mut attn_head_buf = vec![0.0f32; HEAD_DIM];
    let mut attn_out = vec![0.0f32; Q_DIM];
    let mut scratch = LinearScratch::new();
    let mut output = vec![0.0f32; EMBEDDING_DIM];

    let rope_ctx = RopeContext::<64>::new(1_000_000.0);

    // Run full single-token attention at position 0
    gqa_attention_single_token::<64>(
        &normed,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        true, // v_is_q6k: Layer 0 attn_v is Q6_K
        &q_norm_w,
        &k_norm_w,
        0, // layer_idx
        0, // token_offset = position 0 (single-token degenerate softmax)
        &rope_ctx,
        RMS_EPS,
        N_Q_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        EMBEDDING_DIM,
        &mut kv_cache,
        &mut q_buf,
        &mut k_buf,
        &mut v_buf,
        &mut q_out,
        &mut k_out,
        &mut score_buf,
        &mut attn_head_buf,
        &mut attn_out,
        &mut scratch,
        &mut output,
    )
    .expect("attention should not fail");

    // Single token: attention scores are degenerate (softmax of 1 value = 1.0)
    // So attention output = V (just the V values for each head, broadcast via GQA)
    let ref_attn_out = load_reference("11_attention_output");
    assert_close(&attn_out, &ref_attn_out, "11_attention_output", TOLERANCE);

    let ref_final = load_reference("12_final_output");
    assert_close(&output, &ref_final, "12_final_output", TOLERANCE);
}

#[test]
fn test_two_token_softmax_non_degenerate() {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("GGUF not found, skipping");
        return;
    }

    let file_data = std::fs::read(GGUF_PATH).expect("Failed to read GGUF");
    let meta = parse_header(&file_data).expect("Failed to parse GGUF header");

    let attn_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.attn_norm.weight");
    let q_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_q.weight");
    let k_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_k.weight");
    let v_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_v.weight");
    let o_weight = get_tensor_bytes(&file_data, &meta, "blk.0.attn_output.weight");
    let q_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.attn_q_norm.weight");
    let k_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.attn_k_norm.weight");

    let max_tokens = 16;
    let total_storage = 2 * 1 * max_tokens * N_KV_HEADS * HEAD_DIM;
    let mut storage = vec![0.0f32; total_storage];
    let mut kv_cache = KvCache::new(storage.as_mut_ptr(), max_tokens, 1, N_KV_HEADS, HEAD_DIM);

    let rope_ctx = RopeContext::<64>::new(1_000_000.0);
    let mut scratch = LinearScratch::new();
    let mut q_buf = vec![0.0f32; Q_DIM];
    let mut k_buf = vec![0.0f32; KV_DIM];
    let mut v_buf = vec![0.0f32; KV_DIM];
    let mut q_out = vec![0.0f32; Q_DIM];
    let mut k_out = vec![0.0f32; KV_DIM];
    let mut score_buf = vec![0.0f32; max_tokens];
    let mut attn_head_buf = vec![0.0f32; HEAD_DIM];
    let mut attn_out = vec![0.0f32; Q_DIM];
    let mut output = vec![0.0f32; EMBEDDING_DIM];

    // Token 0: ramp × 2 at position 0 (matches Sub-MP-C1 two-token test)
    let mut input0 = vec![0.0f32; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        input0[i] = 0.02 * (i as f32 + 1.0);
    }
    let mut normed0 = vec![0.0f32; EMBEDDING_DIM];
    rmsnorm(&input0, &attn_norm_w, &mut normed0, RMS_EPS);

    gqa_attention_single_token::<64>(
        &normed0,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        true, // v_is_q6k: Layer 0 attn_v is Q6_K
        &q_norm_w,
        &k_norm_w,
        0,
        0, // layer 0, position 0
        &rope_ctx,
        RMS_EPS,
        N_Q_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        EMBEDDING_DIM,
        &mut kv_cache,
        &mut q_buf,
        &mut k_buf,
        &mut v_buf,
        &mut q_out,
        &mut k_out,
        &mut score_buf,
        &mut attn_head_buf,
        &mut attn_out,
        &mut scratch,
        &mut output,
    )
    .unwrap();

    // Token 1: ramp × 1 at position 1 (matches reference)
    let mut input1 = vec![0.0f32; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        input1[i] = 0.01 * (i as f32 + 1.0);
    }
    let mut normed1 = vec![0.0f32; EMBEDDING_DIM];
    rmsnorm(&input1, &attn_norm_w, &mut normed1, RMS_EPS);

    gqa_attention_single_token::<64>(
        &normed1,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        true, // v_is_q6k: Layer 0 attn_v is Q6_K
        &q_norm_w,
        &k_norm_w,
        0,
        1, // layer 0, position 1
        &rope_ctx,
        RMS_EPS,
        N_Q_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        EMBEDDING_DIM,
        &mut kv_cache,
        &mut q_buf,
        &mut k_buf,
        &mut v_buf,
        &mut q_out,
        &mut k_out,
        &mut score_buf,
        &mut attn_head_buf,
        &mut attn_out,
        &mut scratch,
        &mut output,
    )
    .unwrap();

    // Verify two-token attention scores (raw, pre-softmax)
    // After gqa_attention_single_token for token 1:
    //   q_out holds token 1's post-RoPE Q [n_q_heads * head_dim]
    //   KV cache holds K for both tokens
    // Recompute scores: score[qh][t] = dot(q_out[qh], k_cache[kv_h, t]) * scale
    let ref_2tok_scores = load_reference("13_2tok_attention_scores");
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let k_full = kv_cache.get_k_slice(0, 2).unwrap();
    let mut computed_scores = vec![0.0f32; N_Q_HEADS * 2];
    for qh in 0..N_Q_HEADS {
        let kv_h = qh / (N_Q_HEADS / N_KV_HEADS);
        let q_vec = &q_out[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];
        for t in 0..2 {
            let k_off = t * KV_DIM + kv_h * HEAD_DIM;
            let k_vec = &k_full[k_off..k_off + HEAD_DIM];
            let mut dot: f32 = 0.0;
            for i in 0..HEAD_DIM {
                dot += q_vec[i] * k_vec[i];
            }
            computed_scores[qh * 2 + t] = dot * scale;
        }
    }
    assert_close(
        &computed_scores,
        &ref_2tok_scores,
        "13_2tok_attention_scores",
        TOLERANCE,
    );

    // Verify two-token attention output
    let ref_2tok_attn = load_reference("14_2tok_attention_output");
    assert_close(
        &attn_out,
        &ref_2tok_attn,
        "14_2tok_attention_output",
        TOLERANCE,
    );

    let ref_2tok_final = load_reference("15_2tok_final_output");
    assert_close(&output, &ref_2tok_final, "15_2tok_final_output", TOLERANCE);

    // Explicit non-degenerate softmax verification:
    // score_buf holds Q-head 15's post-softmax weights (last head processed).
    // Reference Q-head 15: w0=0.501, w1=0.499
    // Both must be >0.1 and <0.9 to prove non-degenerate.
    let w0 = score_buf[0];
    let w1 = score_buf[1];
    assert!(
        w0 > 0.1 && w0 < 0.9 && w1 > 0.1 && w1 < 0.9,
        "Softmax NOT non-degenerate! w0={}, w1={} — expected both in (0.1, 0.9)",
        w0,
        w1,
    );
    assert!(
        (w0 + w1 - 1.0).abs() < 1e-5,
        "Softmax weights don't sum to 1.0: w0={}, w1={}, sum={}",
        w0,
        w1,
        w0 + w1,
    );
    // Reference match: w0 ≈ 0.501, w1 ≈ 0.499 (Q-head 15)
    assert!(
        (w0 - 0.501).abs() < 0.05 && (w1 - 0.499).abs() < 0.05,
        "Softmax weights deviate from reference: w0={} (expected ~0.501), w1={} (expected ~0.499)",
        w0,
        w1,
    );
}
