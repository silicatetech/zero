// SPDX-License-Identifier: AGPL-3.0-or-later
//! Forward-pass operators.

use libm::{cosf, sinf, sqrtf};
// NOTE: libm::powf is used ONLY in RopeContext::new() at boot time.
// No runtime libm::powf in rope() itself — Pillar 1 performance discipline.

/// Apply RMSNorm: Y = X / RMS(X) * γ where RMS = sqrt(mean(X²) + ε)
///
/// Standard Pre-LayerNorm Transformer normalization.
/// Used in Qwen3 for input/output normalization of attention + MLP blocks.
///
/// Per ADR-029 D7: epsilon must come from GGUF metadata
/// (`qwen3.attention.layer_norm_rms_epsilon`), never hardcoded.
///
/// `no_std`, no allocation. Caller provides output buffer.
/// `libm::sqrtf` is the ADR-028-approved math primitive for Ring-0.
///
/// # Arguments
///
/// * `input` — input vector x[0..n]
/// * `weight` — γ scaling vector (per-element gain), same length as input
/// * `output` — caller-allocated output buffer, same length as input
/// * `epsilon` — small positive value for numerical stability
///
/// # Panics in debug
///
/// Lengths of input, weight, output must all be equal (debug_assert).
pub fn rmsnorm(input: &[f32], weight: &[f32], output: &mut [f32], epsilon: f32) {
    debug_assert_eq!(input.len(), weight.len());
    debug_assert_eq!(input.len(), output.len());

    let n = input.len();

    // Phase 1: Sum of squares
    let mut ssq: f32 = 0.0;
    for i in 0..n {
        ssq += input[i] * input[i];
    }

    // Phase 2: Compute reciprocal RMS with epsilon
    // libm::sqrtf is the ADR-028-approved math primitive for no_std Ring-0.
    // No fallbacks, no Newton-Raphson, no precision compromises.
    let mean_sq = ssq / (n as f32);
    let rms = sqrtf(mean_sq + epsilon);
    let rms_inv = 1.0 / rms;

    // Phase 3: Apply normalization + per-element scaling
    for i in 0..n {
        output[i] = input[i] * rms_inv * weight[i];
    }
}

/// Precomputed RoPE context — frequency base lookups for runtime efficiency.
///
/// Per V3 Pillar 1 (foundational performance, ARCHITECTURE.md Z.96-103):
/// freq_base^(-2i/head_dim) is position-independent and head-independent.
/// Precomputing once eliminates 57,344 redundant libm::powf calls per token
/// (28 layers × 16 heads × 2 (Q+K) × 64 pair indices for Qwen3).
///
/// libm::powf is used ONCE at boot in RopeContext::new().
/// libm::sinf and libm::cosf remain in rope() runtime path because sin/cos
/// depend on position (cannot precompute without per-position table).
///
/// Per ADR-028 v7: Mode-B-Boundary resolved (14 MB headroom available).
/// The precomputed-inv_freqs choice is purely Pillar-1-Performance-driven.
pub struct RopeContext<const HALF: usize> {
    /// Precomputed inv_freq[i] = freq_base^(-2i / head_dim)
    pub inv_freqs: [f32; HALF],
    /// Head dimension (= HALF * 2). Stored for runtime debug-assert checks.
    pub head_dim: usize,
}

impl<const HALF: usize> RopeContext<HALF> {
    /// Create a new RopeContext by precomputing inv_freq lookups.
    /// Called ONCE at boot per (head_dim, freq_base) combination.
    ///
    /// Per ADR-029 D7: freq_base from ModelConfig.rope_freq_base, never hardcoded.
    pub fn new(freq_base: f32) -> Self {
        let head_dim = HALF * 2;
        let head_dim_f = head_dim as f32;
        let mut inv_freqs = [0.0f32; HALF];
        let mut i = 0;
        while i < HALF {
            let exponent = -2.0 * (i as f32) / head_dim_f;
            inv_freqs[i] = libm::powf(freq_base, exponent);
            i += 1;
        }
        Self {
            inv_freqs,
            head_dim,
        }
    }
}

/// Apply RoPE (Rotary Position Embedding) in-place on a single attention head.
///
/// Variante: GPT-NeoX / Llama / Qwen — half-split pairs.
/// Resolution: empirically verified via gguf-py reference (MP2.3b Phase 0).
///
/// Per V3 Pillar 1: precomputed inv_freqs eliminate runtime powf calls.
/// libm::sinf and libm::cosf remain (position-dependent).
///
/// **STRICTLY IN-PLACE** per MP2.3b discipline. No allocation, no aux buffer.
///
/// # Arguments
///
/// * `qk` — single attention head vector, length = ctx.head_dim. Modified in-place.
/// * `ctx` — precomputed RopeContext (computed once at boot).
/// * `position` — sequence position 0..N. position=0 yields identity.
pub fn rope<const HALF: usize>(qk: &mut [f32], ctx: &RopeContext<HALF>, position: usize) {
    debug_assert_eq!(qk.len(), ctx.head_dim);

    let pos_f = position as f32;

    let mut i = 0;
    while i < HALF {
        let inv_freq = ctx.inv_freqs[i];
        let theta = pos_f * inv_freq;

        let cos_t = cosf(theta);
        let sin_t = sinf(theta);

        let x_old = qk[i];
        let y_old = qk[i + HALF];

        qk[i] = x_old * cos_t - y_old * sin_t;
        qk[i + HALF] = x_old * sin_t + y_old * cos_t;

        i += 1;
    }
}

/// Reusable scratch memory for linear_q4k() operation.
///
/// Per V3 ARCHITECTURE.md Z.276-303 (Arena-Disziplin):
/// linear_q4k() must NOT use stack-buffers in its inner loop.
/// LinearScratch is the caller-allocated struct holding the per-block
/// dequant scratch buffer, analog to RopeContext-pattern.
///
/// Per V3 Pillar 1 (foundational performance):
/// block_buf is exactly 256 f32 (= 1 Q4_K block = 1 KB), L1-cache-friendly.
/// Streaming dequant: NEVER more than 1 Q4_K block live at once.
pub struct LinearScratch {
    /// Single Q4_K block dequant scratch (256 f32 = 1024 bytes).
    pub block_buf: [f32; 256],
}

impl LinearScratch {
    /// Create a new LinearScratch with zero-initialized scratch buffer.
    pub fn new() -> Self {
        Self {
            block_buf: [0.0; 256],
        }
    }
}

impl Default for LinearScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply linear projection: out = x @ Wᵀ where W is Q4_K-quantized.
///
/// Output-major matmul (row-by-row over W), block-by-block streaming dequant.
/// Exactly 1 Q4_K block live in scratch at any time (L1-cache-friendly).
///
/// Per ADR-029 D5: out, scratch are caller-allocated by-`&mut` reference.
///
/// # Arguments
///
/// * `x` — input vector, length = in_dim
/// * `w_blocks` — raw Q4_K bytes (row-major, each row = in_dim/256 blocks of 144 bytes)
/// * `out` — output vector, length = out_dim, caller-allocated
/// * `scratch` — reusable LinearScratch (block_buf overwritten per block)
/// * `in_dim` — input dimension (must be divisible by 256)
/// * `out_dim` — output dimension
pub fn linear_q4k(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    scratch: &mut LinearScratch,
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert_eq!(x.len(), in_dim, "x length mismatch");
    debug_assert_eq!(out.len(), out_dim, "out length mismatch");
    debug_assert_eq!(in_dim % 256, 0, "in_dim must be multiple of 256");

    let blocks_per_row = in_dim / 256;
    let bytes_per_row = blocks_per_row * 144;

    debug_assert_eq!(
        w_blocks.len(),
        out_dim * bytes_per_row,
        "w_blocks length mismatch"
    );

    let mut i = 0;
    while i < out_dim {
        let mut acc: f32 = 0.0;
        let row_byte_offset = i * bytes_per_row;

        let mut b = 0;
        while b < blocks_per_row {
            let block_byte_offset = row_byte_offset + b * 144;

            // Dequant ONE Q4_K block into scratch.block_buf
            zero_gguf_parser::dequant::dequant_q4k_row(
                &w_blocks[block_byte_offset..block_byte_offset + 144],
                &mut scratch.block_buf,
                1,
            );

            // Dot-product accumulate (256 multiply-adds)
            let x_offset = b * 256;
            let mut k = 0;
            while k < 256 {
                acc += x[x_offset + k] * scratch.block_buf[k];
                k += 1;
            }

            b += 1;
        }

        out[i] = acc;
        i += 1;
    }
}

/// Apply linear projection: out = x @ Wᵀ where W is Q6_K-quantized.
///
/// Per ADR-029 v1 D9: Q6_K is used for V-projection in Qwen3-1.7B
/// (asymmetric quantization — Q/K/O are Q4_K, V is Q6_K).
///
/// Same output-major, block-by-block streaming pattern as `linear_q4k`.
/// Q6_K block = 210 bytes → 256 f32 elements.
///
/// Per ADR-029 D5: out, scratch are caller-allocated by-`&mut` reference.
pub fn linear_q6k(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    scratch: &mut LinearScratch,
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert_eq!(x.len(), in_dim, "x length mismatch");
    debug_assert_eq!(out.len(), out_dim, "out length mismatch");
    debug_assert_eq!(in_dim % 256, 0, "in_dim must be multiple of 256");

    let blocks_per_row = in_dim / 256;
    let bytes_per_row = blocks_per_row * 210; // Q6_K: 210 bytes per block

    debug_assert_eq!(
        w_blocks.len(),
        out_dim * bytes_per_row,
        "w_blocks length mismatch"
    );

    let mut i = 0;
    while i < out_dim {
        let mut acc: f32 = 0.0;
        let row_byte_offset = i * bytes_per_row;

        let mut b = 0;
        while b < blocks_per_row {
            let block_byte_offset = row_byte_offset + b * 210;

            // Dequant ONE Q6_K block into scratch.block_buf
            zero_gguf_parser::dequant::dequant_q6k_row(
                &w_blocks[block_byte_offset..block_byte_offset + 210],
                &mut scratch.block_buf,
                1,
            );

            // Dot-product accumulate (256 multiply-adds)
            let x_offset = b * 256;
            let mut k = 0;
            while k < 256 {
                acc += x[x_offset + k] * scratch.block_buf[k];
                k += 1;
            }

            b += 1;
        }

        out[i] = acc;
        i += 1;
    }
}

/// Apply linear projection: out = x @ Wᵀ where W is Q4_0-quantized.
///
/// Q4_0 was the original GGML 4-bit format and is the recommended
/// quantisation for natively-int4 models like Kimi K2.6 (Q4_0 is a
/// lossless capture there, not a downstream quantisation). Block
/// layout: 32 elements / 18 bytes (1 × fp16 scale + 16 × packed
/// nibbles). See `zero_gguf_parser::dequant::dequant_q4_0_block`
/// for the unpacking rule.
///
/// Same output-major, block-by-block streaming pattern as `linear_q4k`,
/// but with a 32-element block buffer (we reuse `scratch.block_buf`'s
/// first 32 slots; the remaining 224 slots stay unused per call —
/// keeps the scratch struct's footprint stable across quant types).
///
/// # Arguments
///
/// * `x`        — input vector, length = `in_dim`
/// * `w_blocks` — raw Q4_0 bytes (row-major; each row =
///                `in_dim / 32` blocks of 18 bytes)
/// * `out`      — output vector, length = `out_dim`
/// * `scratch`  — reusable `LinearScratch` (only the first 32 f32s of
///                `block_buf` are touched here)
/// * `in_dim`   — input dimension (MUST be divisible by 32)
/// * `out_dim`  — output dimension
pub fn linear_q4_0(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    scratch: &mut LinearScratch,
    in_dim: usize,
    out_dim: usize,
) {
    use zero_gguf_parser::dequant::{Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE};
    debug_assert_eq!(x.len(), in_dim, "x length mismatch");
    debug_assert_eq!(out.len(), out_dim, "out length mismatch");
    debug_assert_eq!(
        in_dim % Q4_0_BLOCK_SIZE,
        0,
        "in_dim must be a multiple of Q4_0_BLOCK_SIZE (32)"
    );

    let blocks_per_row = in_dim / Q4_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
    debug_assert_eq!(
        w_blocks.len(),
        out_dim * bytes_per_row,
        "w_blocks length mismatch"
    );

    // Local 32-element dequant scratch borrowed from the front of
    // `block_buf` (which is sized 256 for the Q4_K/Q6_K paths).
    let mut i = 0;
    while i < out_dim {
        let mut acc: f32 = 0.0;
        let row_byte_offset = i * bytes_per_row;

        let mut b = 0;
        while b < blocks_per_row {
            let block_byte_offset = row_byte_offset + b * Q4_0_BLOCK_BYTES;
            // Borrow exactly 32 f32s from block_buf as a typed array
            // reference for the dequant entry point. The remaining
            // 224 slots are left untouched.
            let dst32: &mut [f32; Q4_0_BLOCK_SIZE] = (&mut scratch.block_buf[..Q4_0_BLOCK_SIZE])
                .try_into()
                .expect("block_buf has at least 32 slots");
            let src: &[u8; Q4_0_BLOCK_BYTES] = w_blocks
                [block_byte_offset..block_byte_offset + Q4_0_BLOCK_BYTES]
                .try_into()
                .expect("w_blocks slice exactly Q4_0_BLOCK_BYTES");
            zero_gguf_parser::dequant::dequant_q4_0_block(src, dst32);

            // Dot-product accumulate (32 MACs).
            let x_offset = b * Q4_0_BLOCK_SIZE;
            let mut k = 0;
            while k < Q4_0_BLOCK_SIZE {
                acc += x[x_offset + k] * scratch.block_buf[k];
                k += 1;
            }
            b += 1;
        }

        out[i] = acc;
        i += 1;
    }
}

/// Apply linear projection: out = x @ Wᵀ where W is Q8_0-quantized.
///
/// Q8_0 keeps every element in 8 bits (no nibble packing). In GGUFs
/// that bulk-quantise weights to Q4_0 (the typical Kimi K2.6 build),
/// `token_embd.weight` and `output.weight` are usually Q8_0 — the
/// embedding lookup and the LM-head logit matmul aren't tolerant of
/// the lossier 4-bit format. **Without Q8_0 dispatch, those GGUFs
/// can't be loaded at all.**
///
/// Block: 32 elements / 34 bytes (1 × fp16 scale + 32 × signed int8).
/// Reuses the first 32 slots of `scratch.block_buf` for the dequant
/// staging area (same as `linear_q4_0`); the remaining 224 slots stay
/// untouched per call.
///
/// # Arguments
///
/// * `x`        — input vector, length = `in_dim`
/// * `w_blocks` — raw Q8_0 bytes (row-major; each row =
///                `in_dim / 32` blocks of 34 bytes)
/// * `out`      — output vector, length = `out_dim`
/// * `scratch`  — reusable `LinearScratch`
/// * `in_dim`   — input dimension (MUST be divisible by 32)
/// * `out_dim`  — output dimension
pub fn linear_q8_0(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    scratch: &mut LinearScratch,
    in_dim: usize,
    out_dim: usize,
) {
    use zero_gguf_parser::dequant::{Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};
    debug_assert_eq!(x.len(), in_dim, "x length mismatch");
    debug_assert_eq!(out.len(), out_dim, "out length mismatch");
    debug_assert_eq!(
        in_dim % Q8_0_BLOCK_SIZE,
        0,
        "in_dim must be a multiple of Q8_0_BLOCK_SIZE (32)"
    );

    let blocks_per_row = in_dim / Q8_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;
    debug_assert_eq!(
        w_blocks.len(),
        out_dim * bytes_per_row,
        "w_blocks length mismatch"
    );

    let mut i = 0;
    while i < out_dim {
        let mut acc: f32 = 0.0;
        let row_byte_offset = i * bytes_per_row;

        let mut b = 0;
        while b < blocks_per_row {
            let block_byte_offset = row_byte_offset + b * Q8_0_BLOCK_BYTES;
            let dst32: &mut [f32; Q8_0_BLOCK_SIZE] = (&mut scratch.block_buf[..Q8_0_BLOCK_SIZE])
                .try_into()
                .expect("block_buf has at least 32 slots");
            let src: &[u8; Q8_0_BLOCK_BYTES] = w_blocks
                [block_byte_offset..block_byte_offset + Q8_0_BLOCK_BYTES]
                .try_into()
                .expect("w_blocks slice exactly Q8_0_BLOCK_BYTES");
            zero_gguf_parser::dequant::dequant_q8_0_block(src, dst32);

            let x_offset = b * Q8_0_BLOCK_SIZE;
            let mut k = 0;
            while k < Q8_0_BLOCK_SIZE {
                acc += x[x_offset + k] * scratch.block_buf[k];
                k += 1;
            }
            b += 1;
        }

        out[i] = acc;
        i += 1;
    }
}

/// Errors from the cross-quant `linear_dispatch` helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearDispatchError {
    /// Caller asked the dispatcher to handle a `GgmlType` we don't have
    /// a kernel for yet. The boot path should treat this as a hard
    /// failure during weight discovery, not a runtime crash.
    UnsupportedQuant(zero_gguf_parser::GgmlType),
}

/// Cross-quant linear projection — dispatch the right kernel based on
/// the tensor's `GgmlType`. Used by the DeepSeek2 / Kimi forward pass
/// where per-tensor quantisation matters; the Qwen3 path keeps its
/// direct `linear_q4k` / `linear_q6k` calls so β-anchor numerics stay
/// bit-identical to the existing reference run.
///
/// Currently dispatches on:
///
/// * `Q4K`  → [`linear_q4k`]  (Qwen3, Kimi K2.6 attention-style)
/// * `Q6K`  → [`linear_q6k`]  (Qwen3 V/output, mixed-quant builds)
/// * `Q4_0` → [`linear_q4_0`] (Kimi K2.6 native int4 bulk weights)
/// * `Q8_0` → [`linear_q8_0`] (Kimi K2.6 `token_embd` / `output`
///                              keep this quant in bartowski's Q4_0
///                              build, where it'd be too lossy at Q4)
///
/// Any other `GgmlType` (`F32`, `F16`, `BF16`, `Q5_K`, …) surfaces as
/// `LinearDispatchError::UnsupportedQuant` so the boot path can refuse
/// the model with a clear diagnostic rather than corrupting the output.
pub fn linear_dispatch(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    scratch: &mut LinearScratch,
    in_dim: usize,
    out_dim: usize,
    quant: zero_gguf_parser::GgmlType,
) -> Result<(), LinearDispatchError> {
    use zero_gguf_parser::GgmlType;
    match quant {
        GgmlType::Q4K => {
            linear_q4k(x, w_blocks, out, scratch, in_dim, out_dim);
            Ok(())
        }
        GgmlType::Q6K => {
            linear_q6k(x, w_blocks, out, scratch, in_dim, out_dim);
            Ok(())
        }
        GgmlType::Q4_0 => {
            linear_q4_0(x, w_blocks, out, scratch, in_dim, out_dim);
            Ok(())
        }
        GgmlType::Q8_0 => {
            linear_q8_0(x, w_blocks, out, scratch, in_dim, out_dim);
            Ok(())
        }
        other => {
            // Fail-fast: unsupported quant in forward pass is fatal.
            // Every call site uses `let _ = linear_dispatch(...)`, so
            // returning Err silently produces garbage outputs with no
            // diagnostic. Panicking here surfaces the offending quant
            // type through the kernel panic handler (serial breadcrumb +
            // call-stack pointing at the exact projection).
            panic!("linear_dispatch: unsupported quant type {}", other as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn test_rmsnorm_unit_input_unit_weight() {
        // Input = [1, 1, 1, 1], weight = [1, 1, 1, 1]
        // ssq = 4, mean_sq = 1, rms = sqrt(1 + ε) ≈ 1
        // output ≈ [1, 1, 1, 1]
        let input = [1.0f32; 4];
        let weight = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        rmsnorm(&input, &weight, &mut output, 1e-6);
        for &v in output.iter() {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {}", v);
        }
    }

    #[test]
    fn test_rmsnorm_zero_input() {
        // Input all zero, weight=1
        // ssq=0, rms=sqrt(eps), output=0*rms_inv*1 = 0
        let input = [0.0f32; 4];
        let weight = [1.0f32; 4];
        let mut output = [99.0f32; 4];
        rmsnorm(&input, &weight, &mut output, 1e-6);
        for &v in output.iter() {
            assert_eq!(v, 0.0, "expected 0.0, got {}", v);
            assert!(!v.is_nan(), "must not be NaN");
        }
    }

    #[test]
    fn test_rmsnorm_known_values() {
        // Hand-math: Input = [3.0, 4.0]
        // ssq = 9 + 16 = 25
        // mean_sq = 25/2 = 12.5
        // rms = sqrt(12.5 + 1e-6) ≈ 3.5355339
        // rms_inv ≈ 0.2828427
        // weight = [1.0, 1.0]
        // output[0] = 3.0 * 0.2828427 ≈ 0.8485281
        // output[1] = 4.0 * 0.2828427 ≈ 1.1313708
        let input = [3.0f32, 4.0f32];
        let weight = [1.0f32, 1.0f32];
        let mut output = [0.0f32; 2];
        rmsnorm(&input, &weight, &mut output, 1e-6);
        assert!(
            (output[0] - 0.8485281).abs() < 1e-4,
            "output[0] = {}",
            output[0]
        );
        assert!(
            (output[1] - 1.1313708).abs() < 1e-4,
            "output[1] = {}",
            output[1]
        );
    }

    #[test]
    fn test_rmsnorm_with_weight_scaling() {
        // Input = [1, 1], weight = [2, 3]
        // ssq=2, mean_sq=1, rms ≈ 1, rms_inv ≈ 1
        // output[0] = 1 * 1 * 2 = 2, output[1] = 1 * 1 * 3 = 3
        let input = [1.0f32, 1.0f32];
        let weight = [2.0f32, 3.0f32];
        let mut output = [0.0f32; 2];
        rmsnorm(&input, &weight, &mut output, 1e-6);
        assert!((output[0] - 2.0).abs() < 1e-3, "output[0] = {}", output[0]);
        assert!((output[1] - 3.0).abs() < 1e-3, "output[1] = {}", output[1]);
    }

    #[test]
    fn test_rmsnorm_epsilon_stabilizes_zero() {
        // Input all zero, eps=1e-6 → no NaN regardless of rms_inv magnitude
        let input = [0.0f32; 8];
        let weight = [1.0f32; 8];
        let mut output = [0.0f32; 8];
        rmsnorm(&input, &weight, &mut output, 1e-6);
        for &v in output.iter() {
            assert!(!v.is_nan() && !v.is_infinite(), "got {}", v);
        }
    }

    #[test]
    fn test_rmsnorm_qwen3_realistic_range() {
        // Input values in [-0.5, 0.5] (typical post-embedding range)
        // weight values near 1.0 (typical RMS-init for attn_norm)
        // After RMSNorm with unit weights, output RMS should be ~1.0
        let input = [0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.05f32];
        let weight = [1.0f32; 8];
        let mut output = [0.0f32; 8];
        rmsnorm(&input, &weight, &mut output, 1e-6);

        let mut out_ssq = 0.0f32;
        for &v in output.iter() {
            out_ssq += v * v;
        }
        let out_rms = sqrtf(out_ssq / output.len() as f32);
        assert!(
            (out_rms - 1.0).abs() < 1e-3,
            "expected output RMS ~1.0, got {}",
            out_rms
        );
    }

    // ── RoPE Tests ─────────────────────────────────────────────────

    #[test]
    fn test_rope_position_zero_identity() {
        let ctx = RopeContext::<4>::new(1_000_000.0);
        let mut qk = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = qk;
        rope(&mut qk, &ctx, 0);
        for i in 0..8 {
            assert!(
                (qk[i] - original[i]).abs() < 1e-6,
                "position=0 should be identity, [{}]: {} vs {}",
                i,
                qk[i],
                original[i]
            );
        }
    }

    #[test]
    fn test_rope_known_values_position_one() {
        // Hand-math: head_dim=4 (HALF=2), position=1, freq_base=1.0
        // freq_base=1.0 → all inv_freqs = 1.0 → all theta = 1
        // cos(1) ≈ 0.5403023, sin(1) ≈ 0.8414710
        // Pair 0: x=1.0, y=3.0 → [1*0.5403-3*0.8415, 1*0.8415+3*0.5403] = [-1.9842, 2.4624]
        // Pair 1: x=2.0, y=4.0 → [2*0.5403-4*0.8415, 2*0.8415+4*0.5403] = [-2.2854, 3.8442]
        let ctx = RopeContext::<2>::new(1.0);
        let mut qk = [1.0f32, 2.0, 3.0, 4.0];
        rope(&mut qk, &ctx, 1);
        assert!((qk[0] - (-1.9842)).abs() < 1e-3, "qk[0]={}", qk[0]);
        assert!((qk[1] - (-2.2854)).abs() < 1e-3, "qk[1]={}", qk[1]);
        assert!((qk[2] - 2.4624).abs() < 1e-3, "qk[2]={}", qk[2]);
        assert!((qk[3] - 3.8442).abs() < 1e-3, "qk[3]={}", qk[3]);
    }

    #[test]
    fn test_rope_in_place_no_aux() {
        let ctx = RopeContext::<4>::new(1_000_000.0);
        let mut qk = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = qk;
        rope(&mut qk, &ctx, 1);
        let mut any_changed = false;
        for i in 0..8 {
            if (qk[i] - original[i]).abs() > 1e-6 {
                any_changed = true;
            }
        }
        assert!(any_changed, "RoPE at position=1 should modify values");
    }

    #[test]
    fn test_rope_reference_values_qwen3() {
        // Reference from gguf-py Variante B dump (MP2.3b Phase 0).
        // head_dim=128, HALF=64, position=1, freq_base=1e6, ramp [0.01..1.28]
        let ctx = RopeContext::<64>::new(1_000_000.0);
        let mut qk = [0.0f32; 128];
        for i in 0..128 {
            qk[i] = 0.01 * (i as f32 + 1.0);
        }
        rope(&mut qk, &ctx, 1);

        let reference: [f32; 16] = [
            -0.5415531171,
            -0.4622832390,
            -0.3812512551,
            -0.3051765074,
            -0.2368033646,
            -0.1767538593,
            -0.1246151535,
            -0.0795384381,
            -0.0405505749,
            -0.0067053730,
            0.0228510916,
            0.0488593503,
            0.0719468978,
            0.0926380198,
            0.1113669505,
            0.1284913603,
        ];
        for i in 0..16 {
            let diff = (qk[i] - reference[i]).abs();
            assert!(
                diff < 1e-4,
                "RoPE [{}]: got {} expected {} diff {:.6e}",
                i,
                qk[i],
                reference[i],
                diff
            );
        }
    }

    // ── RopeContext Tests ──────────────────────────────────────────

    #[test]
    fn test_rope_context_new_qwen3_values() {
        let ctx = RopeContext::<64>::new(1_000_000.0);
        assert!(
            (ctx.inv_freqs[0] - 1.0).abs() < 1e-6,
            "inv_freqs[0]={}",
            ctx.inv_freqs[0]
        );
        // inv_freqs[1] = 1e6^(-2/128) ≈ 0.8058
        assert!(
            (ctx.inv_freqs[1] - 0.8058).abs() < 1e-3,
            "inv_freqs[1]={}",
            ctx.inv_freqs[1]
        );
        // inv_freqs[63] should be very small positive
        assert!(
            ctx.inv_freqs[63] < 0.001 && ctx.inv_freqs[63] > 0.0,
            "inv_freqs[63]={}",
            ctx.inv_freqs[63]
        );
        assert_eq!(ctx.head_dim, 128);
    }

    #[test]
    fn test_rope_context_freq_base_dependency() {
        let ctx_1 = RopeContext::<64>::new(1.0);
        // 1.0^anything = 1.0
        for i in 0..64 {
            assert!(
                (ctx_1.inv_freqs[i] - 1.0).abs() < 1e-6,
                "freq_base=1.0: inv_freqs[{}]={}",
                i,
                ctx_1.inv_freqs[i]
            );
        }
        let ctx_1m = RopeContext::<64>::new(1_000_000.0);
        let mut differs = false;
        for i in 1..64 {
            if (ctx_1.inv_freqs[i] - ctx_1m.inv_freqs[i]).abs() > 1e-3 {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "different freq_base should produce different inv_freqs"
        );
    }

    // ── LinearScratch Tests ───────────────────────────────────────

    #[test]
    fn test_linear_scratch_new() {
        let scratch = LinearScratch::new();
        for v in scratch.block_buf.iter() {
            assert_eq!(*v, 0.0, "block_buf should be zero-initialized");
        }
        assert_eq!(scratch.block_buf.len(), 256);
    }

    #[test]
    fn test_linear_scratch_default() {
        let scratch_a = LinearScratch::new();
        let scratch_b = LinearScratch::default();
        for k in 0..256 {
            assert_eq!(scratch_a.block_buf[k], scratch_b.block_buf[k]);
        }
    }

    // ── linear_q4k Tests ───────────────────────────────────────

    #[test]
    fn test_linear_q4k_zero_input() {
        let in_dim = 256;
        let out_dim = 2;
        let x = [0.0f32; 256];
        let w_blocks = [0u8; 2 * 144];
        let mut out = [99.0f32; 2];
        let mut scratch = LinearScratch::new();
        linear_q4k(&x, &w_blocks, &mut out, &mut scratch, in_dim, out_dim);
        for i in 0..2 {
            assert_eq!(out[i], 0.0, "x=0 should give out=0, got {}", out[i]);
        }
    }

    #[test]
    fn test_linear_q4k_dimensional_correctness() {
        let in_dim = 256;
        let out_dim = 1;
        let x = [1.0f32; 256];
        let w_blocks = [0u8; 144];
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q4k(&x, &w_blocks, &mut out, &mut scratch, in_dim, out_dim);
        // No panic = dimensional correctness OK. Zero weights → out = 0.
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn test_linear_q4k_known_values() {
        // All-zero Q4_K blocks: dequant yields 0, so out = 0.
        // This verifies basic flow: construct blocks, run linear, get result.
        let in_dim = 256;
        let out_dim = 1;
        let x = [1.0f32; 256];
        let w_blocks = [0u8; 144];
        let mut out = [99.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q4k(&x, &w_blocks, &mut out, &mut scratch, in_dim, out_dim);
        assert!(out[0].abs() < 1e-2, "all-zero block: out={}", out[0]);
    }

    #[test]
    fn test_linear_q4k_streaming_isolation() {
        // Pre-populate scratch with garbage, run linear_q4k.
        // After call, scratch should reflect last dequanted block, not garbage.
        let in_dim = 512; // 2 Q4_K blocks
        let out_dim = 1;
        let x = [1.0f32; 512];
        let w_blocks = [0u8; 2 * 144];
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        for k in 0..256 {
            scratch.block_buf[k] = 999.0;
        }
        linear_q4k(&x, &w_blocks, &mut out, &mut scratch, in_dim, out_dim);
        let mut max_abs: f32 = 0.0;
        for k in 0..256 {
            if scratch.block_buf[k].abs() > max_abs {
                max_abs = scratch.block_buf[k].abs();
            }
        }
        assert!(
            max_abs < 1e-1,
            "scratch should reflect last block, max_abs={}",
            max_abs
        );
    }

    #[test]
    fn test_linear_q4k_reference_sanity() {
        // Sanity test: linear_q4k with ramp input and zero weights
        // produces finite near-zero output. TRUE reference test against
        // Qwen3 model verified via QEMU heartbeat (Step 8), since unit
        // tests cannot load the GGUF file (no filesystem in no_std).
        let in_dim = 256;
        let out_dim = 1;
        let x: [f32; 256] = {
            let mut arr = [0.0f32; 256];
            let mut i = 0;
            while i < 256 {
                arr[i] = 0.01 * (i as f32 + 1.0);
                i += 1;
            }
            arr
        };
        let w_blocks = [0u8; 144];
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q4k(&x, &w_blocks, &mut out, &mut scratch, in_dim, out_dim);
        assert!(!out[0].is_nan(), "output must not be NaN");
        assert!(!out[0].is_infinite(), "output must not be Inf");
        assert!(out[0].abs() < 1e-1, "zero weights: out={}", out[0]);
    }

    // ── linear_q6k Tests ───────────────────────────────────────

    #[test]
    fn test_linear_q6k_zero_input() {
        let in_dim = 256;
        let out_dim = 2;
        let x = [0.0f32; 256];
        let w_blocks = [0u8; 2 * 210]; // Q6_K: 210 bytes per block
        let mut out = [99.0f32; 2];
        let mut scratch = LinearScratch::new();
        linear_q6k(&x, &w_blocks, &mut out, &mut scratch, in_dim, out_dim);
        for i in 0..2 {
            assert_eq!(out[i], 0.0, "x=0 should give out=0, got {}", out[i]);
        }
    }

    #[test]
    fn test_linear_q6k_dimensional_correctness() {
        let in_dim = 256;
        let out_dim = 1;
        let x = [1.0f32; 256];
        let w_blocks = [0u8; 210];
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q6k(&x, &w_blocks, &mut out, &mut scratch, in_dim, out_dim);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn test_linear_q6k_known_zero_weights() {
        let in_dim = 256;
        let out_dim = 1;
        let x = [1.0f32; 256];
        let w_blocks = [0u8; 210];
        let mut out = [99.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q6k(&x, &w_blocks, &mut out, &mut scratch, in_dim, out_dim);
        assert!(out[0].abs() < 1e-2, "all-zero block: out={}", out[0]);
    }

    #[test]
    fn test_linear_q6k_reference_sanity() {
        let in_dim = 256;
        let out_dim = 1;
        let x: [f32; 256] = {
            let mut arr = [0.0f32; 256];
            let mut i = 0;
            while i < 256 {
                arr[i] = 0.01 * (i as f32 + 1.0);
                i += 1;
            }
            arr
        };
        let w_blocks = [0u8; 210];
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q6k(&x, &w_blocks, &mut out, &mut scratch, in_dim, out_dim);
        assert!(!out[0].is_nan(), "output must not be NaN");
        assert!(!out[0].is_infinite(), "output must not be Inf");
    }

    // ── linear_q4_0 + linear_dispatch ───────────────────────────────

    /// Build one Q4_0 block (18 B) at fp16 scale 1.0 with all nibbles = 9
    /// (post-recenter → +1.0). Dequantises to a row of +1.0s.
    fn q4_0_block_unit() -> [u8; 18] {
        let mut block = [0u8; 18];
        // d = 1.0 (fp16 = 0x3C00, little-endian = [0x00, 0x3C])
        block[0] = 0x00;
        block[1] = 0x3C;
        // qs: each byte 0x99 → both nibbles = 9 → (9-8) * 1.0 = 1.0
        for b in &mut block[2..] {
            *b = 0x99;
        }
        block
    }

    #[test]
    fn test_linear_q4_0_unit_weight_matches_dot_product() {
        // 32-input matmul: input is 1,2,...,32; weight row is all +1.0.
        // Expected: sum_{k=1..=32} k = 32 * 33 / 2 = 528.
        let in_dim = 32;
        let out_dim = 1;
        let mut x = [0.0f32; 32];
        for k in 0..32 {
            x[k] = (k + 1) as f32;
        }
        let w = q4_0_block_unit();
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q4_0(&x, &w, &mut out, &mut scratch, in_dim, out_dim);
        assert!(
            (out[0] - 528.0).abs() < 1e-3,
            "linear_q4_0 row = {}, want 528.0",
            out[0]
        );
    }

    #[test]
    fn test_linear_q4_0_two_row_output() {
        // 32 inputs × 2 output rows; row 0 = +1.0s, row 1 = -8.0s
        // (nibbles all 0 → (0-8) * 1.0).
        let mut w = [0u8; 36];
        w[..18].copy_from_slice(&q4_0_block_unit());
        // Row 1: d = 1.0, qs all 0x00.
        w[18] = 0x00;
        w[19] = 0x3C;
        // bytes 20..36 stay zero
        let x = [1.0f32; 32];
        let mut out = [0.0f32; 2];
        let mut scratch = LinearScratch::new();
        linear_q4_0(&x, &w, &mut out, &mut scratch, 32, 2);
        // Row 0: 32 × 1 × 1 = 32.0
        assert!((out[0] - 32.0).abs() < 1e-4, "row0 = {}", out[0]);
        // Row 1: 32 × 1 × -8 = -256.0
        assert!((out[1] - (-256.0)).abs() < 1e-4, "row1 = {}", out[1]);
    }

    #[test]
    fn test_linear_q4_0_multiblock_in_dim() {
        // in_dim = 64 → two Q4_0 blocks per row. Row of +1.0s × input
        // of all 0.5 → dot = 64 × 1 × 0.5 = 32.0.
        let mut w = [0u8; 36];
        w[..18].copy_from_slice(&q4_0_block_unit());
        w[18..].copy_from_slice(&q4_0_block_unit());
        let x = [0.5f32; 64];
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q4_0(&x, &w, &mut out, &mut scratch, 64, 1);
        assert!((out[0] - 32.0).abs() < 1e-3, "{}", out[0]);
    }

    #[test]
    fn test_linear_dispatch_routes_q4_0() {
        // Same configuration as test_linear_q4_0_unit_weight_matches_dot_product,
        // but invoked through the cross-quant dispatcher.
        use zero_gguf_parser::GgmlType;
        let in_dim = 32;
        let out_dim = 1;
        let mut x = [0.0f32; 32];
        for k in 0..32 {
            x[k] = (k + 1) as f32;
        }
        let w = q4_0_block_unit();
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        let r = linear_dispatch(
            &x,
            &w,
            &mut out,
            &mut scratch,
            in_dim,
            out_dim,
            GgmlType::Q4_0,
        );
        assert!(r.is_ok());
        assert!((out[0] - 528.0).abs() < 1e-3, "{}", out[0]);
    }

    #[test]
    #[should_panic(expected = "linear_dispatch: unsupported quant type")]
    fn test_linear_dispatch_rejects_unsupported_quant() {
        use zero_gguf_parser::GgmlType;
        let x = [0.0f32; 32];
        let w = [0u8; 4];
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        // F16 is a legitimate GgmlType but we don't dequant it through
        // linear_dispatch (it would need a separate float-matmul path).
        // Fail-fast: the dispatcher panics so the kernel panic handler
        // surfaces the offending quant type rather than silently
        // returning garbage outputs (all ~60 call sites use `let _ =`).
        let _ = linear_dispatch(&x, &w, &mut out, &mut scratch, 32, 1, GgmlType::F16);
    }

    // ── linear_q8_0 + Q8_0 dispatch ──────────────────────────────────

    /// One Q8_0 block (34 B) with fp16 scale 1.0 and every i8 = +1.
    /// Dequantises to a row of +1.0 f32s.
    fn q8_0_block_unit() -> [u8; 34] {
        let mut block = [0u8; 34];
        // d = 1.0 (fp16 = 0x3C00, little-endian).
        block[0] = 0x00;
        block[1] = 0x3C;
        for k in 0..32 {
            block[2 + k] = 1u8; // i8 = +1
        }
        block
    }

    #[test]
    fn test_linear_q8_0_unit_weight_dot_product() {
        // 32 inputs × 1 output row of +1.0 → sum_{k=1..=32} k = 528.
        let in_dim = 32;
        let out_dim = 1;
        let mut x = [0.0f32; 32];
        for k in 0..32 {
            x[k] = (k + 1) as f32;
        }
        let w = q8_0_block_unit();
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q8_0(&x, &w, &mut out, &mut scratch, in_dim, out_dim);
        assert!(
            (out[0] - 528.0).abs() < 1e-3,
            "linear_q8_0 row = {}",
            out[0]
        );
    }

    #[test]
    fn test_linear_q8_0_negative_weights() {
        // Row of -1.0s (i8 = -1 = 0xFF byte) against input of +0.5.
        // Sign-extension is the critical step here: if we accidentally
        // read the i8 as unsigned (255), we'd get +127.5 instead of
        // -0.5 per element. dot = 32 × -1 × 0.5 = -16.0.
        let mut w = [0u8; 34];
        w[0] = 0x00;
        w[1] = 0x3C;
        for k in 0..32 {
            w[2 + k] = 0xFFu8; // i8 = -1
        }
        let x = [0.5f32; 32];
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q8_0(&x, &w, &mut out, &mut scratch, 32, 1);
        assert!((out[0] - (-16.0)).abs() < 1e-3, "out = {}", out[0]);
    }

    #[test]
    fn test_linear_q8_0_multiblock_in_dim() {
        // in_dim = 64 → two Q8_0 blocks per row. Row of +1.0 × input
        // of all 0.5 → dot = 64 × 1 × 0.5 = 32.0.
        let mut w = [0u8; 68];
        w[..34].copy_from_slice(&q8_0_block_unit());
        w[34..].copy_from_slice(&q8_0_block_unit());
        let x = [0.5f32; 64];
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        linear_q8_0(&x, &w, &mut out, &mut scratch, 64, 1);
        assert!((out[0] - 32.0).abs() < 1e-3, "out = {}", out[0]);
    }

    #[test]
    fn test_linear_dispatch_routes_q8_0() {
        // bartowski Q4_0 GGUFs ship `token_embd.weight` and
        // `output.weight` as Q8_0 — this routing is load-bearing for
        // Kimi K2.6 boot. Without it, the kernel would refuse the
        // model with UnsupportedQuant(Q8_0).
        use zero_gguf_parser::GgmlType;
        let in_dim = 32;
        let out_dim = 1;
        let mut x = [0.0f32; 32];
        for k in 0..32 {
            x[k] = (k + 1) as f32;
        }
        let w = q8_0_block_unit();
        let mut out = [0.0f32; 1];
        let mut scratch = LinearScratch::new();
        let r = linear_dispatch(
            &x,
            &w,
            &mut out,
            &mut scratch,
            in_dim,
            out_dim,
            GgmlType::Q8_0,
        );
        assert!(r.is_ok(), "Q8_0 dispatch must succeed: {:?}", r);
        assert!((out[0] - 528.0).abs() < 1e-3);
    }
}
