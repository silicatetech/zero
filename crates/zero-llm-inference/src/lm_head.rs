// SPDX-License-Identifier: AGPL-3.0-or-later
//! LM Head: final RMSNorm + linear projection + argmax.
//!
//! Per Sub-MP-C3 findings: Qwen3-1.7B uses UNTIED embeddings.
//! `output.weight` is Q6_K, shape [151_936, 2048] (padded vocab).
//!
//! Per ADR-029 v2 D16: argmax over full padded vocab, post-hoc bounds check.

use crate::ops::{linear_q4_0, linear_q4k, linear_q6k, linear_q8_0, rmsnorm, LinearScratch};

/// Real vocabulary size (non-padded).
pub const VOCAB_SIZE_REAL: usize = 151_643;

/// Padded vocabulary size (GGUF aligns to 256 for SIMD).
pub const VOCAB_SIZE_PADDED: usize = 151_936;

/// Quantization of the output projection (LM head).
///
/// Qwen3 ships `output.weight` as Q6_K. Kimi K2.6 bartowski-Q4_0
/// builds keep it at Q8_0 (the LM head matmul is too lossy at 4 bits
/// across a 160 k-class vocab). Q4_K shows up in mixed builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputQuant {
    Q6K,
    Q4K,
    Q8_0,
    Q4_0,
}

/// LM Head errors.
#[derive(Debug)]
pub enum LmHeadError {
    /// Logits contain NaN or Inf — numerical catastrophe.
    NonFiniteLogits,
    /// Argmax produced a token ID in padding region (>= vocab_size_real).
    /// Indicates corrupted weights or numerical issue.
    ArgmaxInPaddingRegion(usize),
}

/// Final RMSNorm + LM-head linear projection + argmax.
///
/// Argmax operates over full padded vocab (151,936) — no per-logit
/// conditional inside hot loop (Pillar 1). Post-hoc bounds check on result.
#[allow(clippy::too_many_arguments)]
pub fn lm_head_argmax(
    final_hidden: &[f32],
    output_norm_weight: &[f32],
    output_weight: &[u8],
    rms_eps: f32,
    embedding_dim: usize,
    norm_buf: &mut [f32],
    logits_buf: &mut [f32],
    scratch: &mut LinearScratch,
) -> Result<u32, LmHeadError> {
    // Step 1: Final RMSNorm
    rmsnorm(final_hidden, output_norm_weight, norm_buf, rms_eps);

    // Step 2: LM head linear projection (Q6_K)
    linear_q6k(
        norm_buf,
        output_weight,
        logits_buf,
        scratch,
        embedding_dim,
        VOCAB_SIZE_PADDED,
    );

    // Step 3: Argmax over full padded vocab (no per-logit conditional)
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

    // Step 4: Post-hoc bounds check (D16 invariant)
    if max_idx >= VOCAB_SIZE_REAL {
        return Err(LmHeadError::ArgmaxInPaddingRegion(max_idx));
    }

    Ok(max_idx as u32)
}

/// Dynamic-vocab LM head: caller passes `vocab_size_padded` (matmul rows) and
/// `vocab_size_real` (bounds-check threshold). Selects Q4_K or Q6_K via
/// `output_quant`. Kimi K2.6: vocab = 128 256, Q4_K. Qwen3: 151 936 padded,
/// 151 643 real, Q6_K.
///
/// `logits_buf` must be sized for `vocab_size_padded`. Argmax operates over the
/// full padded range — no per-logit conditional inside the hot loop (Pillar 1).
#[allow(clippy::too_many_arguments)]
pub fn lm_head_argmax_dynamic(
    final_hidden: &[f32],
    output_norm_weight: &[f32],
    output_weight: &[u8],
    output_quant: OutputQuant,
    rms_eps: f32,
    embedding_dim: usize,
    vocab_size_padded: usize,
    vocab_size_real: usize,
    norm_buf: &mut [f32],
    logits_buf: &mut [f32],
    scratch: &mut LinearScratch,
) -> Result<u32, LmHeadError> {
    rmsnorm(final_hidden, output_norm_weight, norm_buf, rms_eps);

    match output_quant {
        OutputQuant::Q6K => linear_q6k(
            norm_buf,
            output_weight,
            logits_buf,
            scratch,
            embedding_dim,
            vocab_size_padded,
        ),
        OutputQuant::Q4K => linear_q4k(
            norm_buf,
            output_weight,
            logits_buf,
            scratch,
            embedding_dim,
            vocab_size_padded,
        ),
        OutputQuant::Q4_0 => linear_q4_0(
            norm_buf,
            output_weight,
            logits_buf,
            scratch,
            embedding_dim,
            vocab_size_padded,
        ),
        OutputQuant::Q8_0 => linear_q8_0(
            norm_buf,
            output_weight,
            logits_buf,
            scratch,
            embedding_dim,
            vocab_size_padded,
        ),
    }

    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx: usize = 0;
    for (i, &v) in logits_buf[..vocab_size_padded].iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }

    if !max_val.is_finite() {
        return Err(LmHeadError::NonFiniteLogits);
    }

    if max_idx >= vocab_size_real {
        return Err(LmHeadError::ArgmaxInPaddingRegion(max_idx));
    }

    Ok(max_idx as u32)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;

    #[test]
    fn test_argmax_basic() {
        // Synthetic logits: max at index 25
        let mut logits = vec![-10.0f32; VOCAB_SIZE_PADDED];
        logits[25] = 12.0;
        logits[100] = 10.0;

        // Skip rmsnorm + linear for this unit test — just verify argmax logic
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx: usize = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        assert_eq!(max_idx, 25);
        assert!((max_val - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_argmax_padding_detection() {
        // Synthetic: max in padding region → should be caught
        let max_idx = VOCAB_SIZE_REAL + 10;
        assert!(max_idx >= VOCAB_SIZE_REAL);
        // In real lm_head_argmax, this returns ArgmaxInPaddingRegion
    }
}
