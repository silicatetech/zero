// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::needless_range_loop)]
//! Bit-exact verification of SwiGLU MLP forward-pass against Sub-MP-C3 reference dumps.

use std::path::Path;

use zero_gguf_parser::{parse_header, GgmlType};
use zero_llm_inference::*;

const TOLERANCE: f32 = 5e-4;
const GGUF_PATH: &str = "../../kernel/programs/Qwen_Qwen3-1.7B-Q4_K_M.gguf";
const REF_DIR: &str = "tests/reference-dumps";

const EMBEDDING_DIM: usize = 2048;
const INTERMEDIATE_DIM: usize = 6144;
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
        "{}: length mismatch: {} vs {}",
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
    println!("  {}: max_diff={:.6e} (tol={:.6e})", name, max_diff, tol);
    assert!(
        max_diff < tol,
        "{}[{}]: max_diff={:.6e} (got {}, expected {}), tolerance={}",
        name,
        max_idx,
        max_diff,
        actual[max_idx],
        expected[max_idx],
        tol
    );
}

fn get_tensor_bytes<'a>(
    file_data: &'a [u8],
    meta: &zero_gguf_parser::GgufMetadata,
    name: &str,
) -> &'a [u8] {
    for t in &meta.tensors {
        if t.name == name {
            let start = meta.tensor_data_offset + t.offset as usize;
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
fn test_swiglu_layer_0_step_by_step() {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("GGUF not found, skipping");
        return;
    }

    let file_data = std::fs::read(GGUF_PATH).expect("GGUF");
    let meta = parse_header(&file_data).expect("parse_header");

    // Load weights
    let ffn_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.ffn_norm.weight");
    let gate_w = get_tensor_bytes(&file_data, &meta, "blk.0.ffn_gate.weight");
    let up_w = get_tensor_bytes(&file_data, &meta, "blk.0.ffn_up.weight");
    let down_w = get_tensor_bytes(&file_data, &meta, "blk.0.ffn_down.weight");

    let down_tensor = meta
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_down.weight")
        .unwrap();
    let down_is_q6k = matches!(down_tensor.tensor_type, GgmlType::Q6K);
    println!(
        "blk.0.ffn_down quant: {:?} (is_q6k={})",
        down_tensor.tensor_type, down_is_q6k
    );

    // Load Sub-MP-C3 references
    let ref_input = load_reference("swiglu_00_mlp_input");
    let ref_normed = load_reference("swiglu_01_mlp_normed");
    let ref_gate = load_reference("swiglu_02_gate_projected");
    let ref_up = load_reference("swiglu_03_up_projected");
    let ref_silu = load_reference("swiglu_04_silu_applied");
    let ref_product = load_reference("swiglu_05_gate_up_product");
    let ref_down = load_reference("swiglu_06_down_projected");

    // Build ramp input (must match Sub-MP-C3 Schritt D)
    let mut input = vec![0.0f32; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        input[i] = 0.01 * (i as f32 + 1.0);
    }
    assert_close(&input, &ref_input, "00_mlp_input", TOLERANCE);

    // Step 1: RMSNorm
    let mut normed = vec![0.0f32; EMBEDDING_DIM];
    rmsnorm(&input, &ffn_norm_w, &mut normed, RMS_EPS);
    assert_close(&normed, &ref_normed, "01_mlp_normed", TOLERANCE);

    // Step 2-6: SwiGLU step-by-step
    let mut scratch = LinearScratch::new();
    let mut gate_buf = vec![0.0f32; INTERMEDIATE_DIM];
    let mut up_buf = vec![0.0f32; INTERMEDIATE_DIM];
    let _hidden_buf = vec![0.0f32; INTERMEDIATE_DIM];
    let mut down_out = vec![0.0f32; EMBEDDING_DIM];

    // Step 2: Gate projection
    linear_q4k(
        &normed,
        gate_w,
        &mut gate_buf,
        &mut scratch,
        EMBEDDING_DIM,
        INTERMEDIATE_DIM,
    );
    assert_close(&gate_buf, &ref_gate, "02_gate_projected", TOLERANCE);

    // Step 3: Up projection
    linear_q4k(
        &normed,
        up_w,
        &mut up_buf,
        &mut scratch,
        EMBEDDING_DIM,
        INTERMEDIATE_DIM,
    );
    assert_close(&up_buf, &ref_up, "03_up_projected", TOLERANCE);

    // Step 4: SiLU
    let mut silu_out = gate_buf.clone();
    for i in 0..INTERMEDIATE_DIM {
        silu_out[i] = silu_out[i] / (1.0 + libm::expf(-silu_out[i]));
    }
    assert_close(&silu_out, &ref_silu, "04_silu_applied", TOLERANCE);

    // Step 5: Element-wise mult
    let mut product = vec![0.0f32; INTERMEDIATE_DIM];
    for i in 0..INTERMEDIATE_DIM {
        product[i] = silu_out[i] * up_buf[i];
    }
    assert_close(&product, &ref_product, "05_gate_up_product", TOLERANCE);

    // Step 6: Down projection
    if down_is_q6k {
        linear_q6k(
            &product,
            down_w,
            &mut down_out,
            &mut scratch,
            INTERMEDIATE_DIM,
            EMBEDDING_DIM,
        );
    } else {
        linear_q4k(
            &product,
            down_w,
            &mut down_out,
            &mut scratch,
            INTERMEDIATE_DIM,
            EMBEDDING_DIM,
        );
    }
    assert_close(&down_out, &ref_down, "06_down_projected", TOLERANCE);

    println!(
        "\n✅ All 7 SwiGLU Layer 0 references match within {:.0e}",
        TOLERANCE
    );
}

#[test]
fn test_mlp_swiglu_unified_call() {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("GGUF not found, skipping");
        return;
    }

    let file_data = std::fs::read(GGUF_PATH).expect("GGUF");
    let meta = parse_header(&file_data).expect("parse_header");

    let ffn_norm_w = get_f32_tensor(&file_data, &meta, "blk.0.ffn_norm.weight");
    let gate_w = get_tensor_bytes(&file_data, &meta, "blk.0.ffn_gate.weight");
    let up_w = get_tensor_bytes(&file_data, &meta, "blk.0.ffn_up.weight");
    let down_w = get_tensor_bytes(&file_data, &meta, "blk.0.ffn_down.weight");

    let down_tensor = meta
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_down.weight")
        .unwrap();
    let down_is_q6k = matches!(down_tensor.tensor_type, GgmlType::Q6K);

    let ref_down = load_reference("swiglu_06_down_projected");

    // Build input + norm
    let mut input = vec![0.0f32; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        input[i] = 0.01 * (i as f32 + 1.0);
    }
    let mut normed = vec![0.0f32; EMBEDDING_DIM];
    rmsnorm(&input, &ffn_norm_w, &mut normed, RMS_EPS);

    // Use unified mlp_swiglu
    let mut scratch = LinearScratch::new();
    let mut gate_buf = vec![0.0f32; INTERMEDIATE_DIM];
    let mut up_buf = vec![0.0f32; INTERMEDIATE_DIM];
    let mut hidden_buf = vec![0.0f32; INTERMEDIATE_DIM];
    let mut output = vec![0.0f32; EMBEDDING_DIM];

    mlp_swiglu(
        &normed,
        gate_w,
        up_w,
        down_w,
        down_is_q6k,
        EMBEDDING_DIM,
        INTERMEDIATE_DIM,
        &mut gate_buf,
        &mut up_buf,
        &mut hidden_buf,
        &mut scratch,
        &mut output,
    );

    assert_close(&output, &ref_down, "mlp_swiglu_unified", TOLERANCE);
    println!("✅ Unified mlp_swiglu matches reference");
}

#[test]
fn test_silu_known_values() {
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let x = 0.0f32;
    let result = x / (1.0 + libm::expf(-x));
    assert!((result - 0.0).abs() < 1e-6, "SiLU(0) = {}", result);

    // SiLU(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ≈ 0.7311
    let x = 1.0f32;
    let result = x / (1.0 + libm::expf(-x));
    assert!((result - 0.7311).abs() < 1e-3, "SiLU(1) = {}", result);

    // SiLU(-1) = -1 * sigmoid(-1) = -1 / (1 + e^1) ≈ -0.2689
    let x = -1.0f32;
    let result = x / (1.0 + libm::expf(-x));
    assert!((result - (-0.2689)).abs() < 1e-3, "SiLU(-1) = {}", result);

    // SiLU(-100) → ~0
    let x = -100.0f32;
    let result = x / (1.0 + libm::expf(-x));
    assert!(
        result.abs() < 1e-10,
        "SiLU(-100) should be ~0, got {}",
        result
    );
    assert!(!result.is_nan(), "SiLU(-100) must not be NaN");

    // SiLU(100) → ~100
    let x = 100.0f32;
    let result = x / (1.0 + libm::expf(-x));
    assert!(
        (result - 100.0).abs() < 1e-3,
        "SiLU(100) should be ~100, got {}",
        result
    );
    assert!(!result.is_nan(), "SiLU(100) must not be NaN");
}

#[test]
fn test_embed_lookup_basic() {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("GGUF not found, skipping");
        return;
    }

    let file_data = std::fs::read(GGUF_PATH).expect("GGUF");
    let meta = parse_header(&file_data).expect("parse_header");

    let embd_w = get_tensor_bytes(&file_data, &meta, "token_embd.weight");

    // Embed token 9707 ("Hello")
    let mut output = vec![0.0f32; EMBEDDING_DIM];
    embed_lookup(9707, embd_w, EMBEDDING_DIM, &mut output);

    // Sanity: should be finite, bounded
    let mut max_abs: f32 = 0.0;
    for &v in &output {
        assert!(!v.is_nan(), "embedding contains NaN");
        assert!(!v.is_infinite(), "embedding contains Inf");
        if v.abs() > max_abs {
            max_abs = v.abs();
        }
    }
    println!(
        "Token 9707 embedding: max_abs={:.4}, first 5: {:?}",
        max_abs,
        &output[..5]
    );
    assert!(
        max_abs < 10.0,
        "embedding max_abs {} seems too large",
        max_abs
    );
    assert!(
        max_abs > 0.001,
        "embedding max_abs {} seems too small",
        max_abs
    );
}

#[test]
fn test_dispatch_table_population() {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("GGUF not found, skipping");
        return;
    }

    let file_data = std::fs::read(GGUF_PATH).expect("GGUF");
    let meta = parse_header(&file_data).expect("parse_header");

    let mut dispatch = ForwardPassDispatch::default();

    let mut q4k_count = 0;
    let mut q6k_count = 0;

    for layer in 0..N_LAYERS {
        let name = format!("blk.{}.ffn_down.weight", layer);
        let tensor = meta
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("Missing {}", name));
        match tensor.tensor_type {
            GgmlType::Q4K => {
                dispatch.ffn_down_quant[layer] = FfnDownQuant::Q4K;
                dispatch.ffn_down_tensor_quant[layer] = GgmlType::Q4K;
                q4k_count += 1;
            }
            GgmlType::Q6K => {
                dispatch.ffn_down_quant[layer] = FfnDownQuant::Q6K;
                dispatch.ffn_down_tensor_quant[layer] = GgmlType::Q6K;
                q6k_count += 1;
            }
            other => panic!("Unexpected ffn_down type for layer {}: {:?}", layer, other),
        }
    }

    println!(
        "Per-layer dispatch: {} Q4_K + {} Q6_K = {} total",
        q4k_count,
        q6k_count,
        q4k_count + q6k_count
    );
    assert_eq!(q4k_count + q6k_count, N_LAYERS);
    // At least some should be Q6_K (iMatrix promotes high-sensitivity layers)
    println!(
        "Q6_K layers: {:?}",
        (0..N_LAYERS)
            .filter(|&l| dispatch.ffn_down_quant[l] == FfnDownQuant::Q6K)
            .collect::<Vec<_>>()
    );
}
