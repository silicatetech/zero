// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dequantization routines for GGUF quantized tensors.
//!
//! MP2.2a: FP16→F32 conversion + Q4_K block dequantization.
//! MP2.2b (future): Q6_K block dequantization.
//!
//! # Design Constraints (ADR-029 D5 + ADR-028 v5)
//!
//! - `no_std`, zero allocation — caller-allocated output buffers
//! - Pure Rust bit manipulation (no `half` crate, no intrinsics)
//! - Spec source: `docs/discovery/stage11-mp2-pre-discovery.md` Task C
//! - Code-size conscious: no generics over Display, no format!

/// Q4_K super-block size: 256 elements per block.
pub const Q4K_BLOCK_SIZE: usize = 256;

/// Q4_K block byte size: 144 bytes per block.
pub const Q4K_BLOCK_BYTES: usize = 144;

// ── FP16 → F32 ─────────────────────────────────────────────────────

/// Convert IEEE 754 binary16 (half-precision) bit pattern to f32.
///
/// Layout: 1 sign + 5 exponent + 10 mantissa bits.
/// Handles normals, denormals, ±zero, ±infinity, NaN.
///
/// no_std, no allocation, no external dependencies.
pub fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // ±zero
            f32::from_bits(sign << 31)
        } else {
            // Denormal: value = (-1)^sign * 2^(-14) * (mant / 1024)
            // Use f32 arithmetic: exact for these small values
            let val = (mant as f32) * (1.0f32 / 1024.0f32) * (1.0f32 / 16384.0f32); // 2^-14 = 1/16384
            if sign != 0 {
                -val
            } else {
                val
            }
        }
    } else if exp == 31 {
        if mant == 0 {
            // ±infinity
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            // NaN — preserve quiet/signaling bit pattern
            f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
        }
    } else {
        // Normal: rebias exponent from fp16 bias 15 to fp32 bias 127.
        // Keep the arithmetic non-negative for debug builds (exp is 1..=30).
        let f32_exp = exp + (127 - 15);
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

// ── Q4_K Scale Decoding ─────────────────────────────────────────────

/// Decode sub-block scale and min from the 12-byte scales array.
/// Per Discovery v0 `get_scale_min_k4` spec.
///
/// For j < 4:
///   scale[j] = scales[j] & 0x3F
///   min[j]   = scales[j + 4] & 0x3F
/// For j >= 4:
///   scale[j] = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
///   min[j]   = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
#[inline]
fn get_scale_min_k4(scales: &[u8; 12], j: usize) -> (u8, u8) {
    if j < 4 {
        let sc = scales[j] & 0x3F;
        let mn = scales[j + 4] & 0x3F;
        (sc, mn)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, mn)
    }
}

// ── Q4_K Block Dequantization ───────────────────────────────────────

/// Dequantize one Q4_K super-block (256 elements) into f32 output.
///
/// Block layout (144 bytes total):
///   bytes  0-1:    d (FP16, super-block scale)
///   bytes  2-3:    dmin (FP16, super-block min)
///   bytes  4-15:   scales[12] (8 sub-block scales + mins, 6-bit packed)
///   bytes 16-143:  qs[128] (256 × 4-bit quants, 2 per byte)
///
/// Caller-allocated output buffer per ADR-029 D5.
pub fn dequant_q4k_block(block: &[u8; Q4K_BLOCK_BYTES], output: &mut [f32; Q4K_BLOCK_SIZE]) {
    // Read super-block scale and min (FP16 at bytes 0-3)
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    // scales[12] at bytes 4-15
    let scales: &[u8; 12] = block[4..16].try_into().unwrap();

    // qs[128] at bytes 16-143
    let qs = &block[16..144];

    // 4 groups of 64 elements each
    let mut is: usize = 0; // sub-block scale index
    let mut qs_offset: usize = 0;

    for j_group in 0..4 {
        let j_base = j_group * 64;

        // Decode two sub-block scales and mins per group
        let (sc0, mn0) = get_scale_min_k4(scales, is);
        let (sc1, mn1) = get_scale_min_k4(scales, is + 1);

        let d1 = d * (sc0 as f32);
        let m1 = dmin * (mn0 as f32);
        let d2 = d * (sc1 as f32);
        let m2 = dmin * (mn1 as f32);

        // First 32 elements: low nibble
        for l in 0..32 {
            output[j_base + l] = d1 * ((qs[qs_offset + l] & 0xF) as f32) - m1;
        }

        // Next 32 elements: high nibble
        for l in 0..32 {
            output[j_base + 32 + l] = d2 * ((qs[qs_offset + l] >> 4) as f32) - m2;
        }

        qs_offset += 32;
        is += 2;
    }
}

/// Dequantize N consecutive Q4_K blocks into f32 output.
/// Caller-allocated output buffer (output.len() must equal num_blocks * 256).
pub fn dequant_q4k_row(blocks: &[u8], output: &mut [f32], num_blocks: usize) {
    debug_assert_eq!(blocks.len(), num_blocks * Q4K_BLOCK_BYTES);
    debug_assert_eq!(output.len(), num_blocks * Q4K_BLOCK_SIZE);
    for i in 0..num_blocks {
        let block_slice: &[u8; Q4K_BLOCK_BYTES] = blocks
            [i * Q4K_BLOCK_BYTES..(i + 1) * Q4K_BLOCK_BYTES]
            .try_into()
            .unwrap();
        let out_slice: &mut [f32; Q4K_BLOCK_SIZE] = (&mut output
            [i * Q4K_BLOCK_SIZE..(i + 1) * Q4K_BLOCK_SIZE])
            .try_into()
            .unwrap();
        dequant_q4k_block(block_slice, out_slice);
    }
}

// ── Q4_0 Constants & Dequantization ────────────────────────────────
//
// Q4_0 is the original GGML 4-bit quantization, predating Q4_K's
// super-block layout. Used by Kimi K2.6 (which is natively int4 — the
// Q4_0 GGUF is a 1:1 lossless capture of the model's native weights,
// not a downstream quantisation). Block layout (18 bytes total):
//
//   bytes 0-1:  d (FP16, super-block scale)
//   bytes 2-17: qs[16] (32 × 4-bit nibbles)
//
// llama.cpp packs the 32 nibbles such that the low nibble of byte j
// (0..16) gives output position j (0..16), and the high nibble gives
// output position j + 16 (16..32). Each value dequantises as:
//
//   output[k] = (nibble - 8) * d
//
// where the `- 8` recenters the unsigned-nibble range [0..15] into the
// signed range [-8..7].

/// Q4_0 block size: 32 elements per block.
pub const Q4_0_BLOCK_SIZE: usize = 32;

/// Q4_0 block byte size: 2 (d, fp16) + 16 (32 × 4-bit nibbles) = 18 bytes.
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Dequantise one Q4_0 block into `Q4_0_BLOCK_SIZE` f32s. Caller-
/// allocated output per ADR-029 D5 (no allocation, no panic on valid
/// inputs).
pub fn dequant_q4_0_block(block: &[u8; Q4_0_BLOCK_BYTES], output: &mut [f32; Q4_0_BLOCK_SIZE]) {
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    // qs[16]: byte j carries low-nibble→output[j], high-nibble→output[j+16]
    for j in 0..16 {
        let byte = block[2 + j];
        // Recenter [0..15] → [-8..7] via i8 arithmetic; the `as i8`
        // cast is safe because the nibble is 0..15 which fits in i8.
        let x0 = (byte & 0x0F) as i8 - 8;
        let x1 = ((byte >> 4) & 0x0F) as i8 - 8;
        output[j] = (x0 as f32) * d;
        output[j + 16] = (x1 as f32) * d;
    }
}

/// Dequantise N consecutive Q4_0 blocks into f32 output.
/// `output.len() == num_blocks * Q4_0_BLOCK_SIZE`.
pub fn dequant_q4_0_row(blocks: &[u8], output: &mut [f32], num_blocks: usize) {
    debug_assert_eq!(blocks.len(), num_blocks * Q4_0_BLOCK_BYTES);
    debug_assert_eq!(output.len(), num_blocks * Q4_0_BLOCK_SIZE);
    for i in 0..num_blocks {
        let block_slice: &[u8; Q4_0_BLOCK_BYTES] = blocks
            [i * Q4_0_BLOCK_BYTES..(i + 1) * Q4_0_BLOCK_BYTES]
            .try_into()
            .unwrap();
        let out_slice: &mut [f32; Q4_0_BLOCK_SIZE] = (&mut output
            [i * Q4_0_BLOCK_SIZE..(i + 1) * Q4_0_BLOCK_SIZE])
            .try_into()
            .unwrap();
        dequant_q4_0_block(block_slice, out_slice);
    }
}

// ── Q8_0 Constants & Dequantization ────────────────────────────────
//
// Q8_0 is GGML's "8-bit per element" quantisation. Critically, GGUFs
// that bulk-quantise their weights to Q4_0 (e.g. bartowski's
// `Kimi-K2-Instruct-Q4_0.gguf`) typically keep `token_embd.weight`
// and `output.weight` in Q8_0 — the lossier 4-bit format gets too
// expensive for the embedding lookup and the LM head logit
// projection. The kernel MUST dispatch Q8_0 or those models won't
// load at all.
//
// Block layout (34 bytes total):
//
//   bytes 0-1:  d (FP16, super-block scale)
//   bytes 2-33: qs[32] (32 × signed int8 values)
//
// Dequant per element:
//
//   output[k] = qs[k] * d        (k = 0..32, qs[k] is signed i8)
//
// There is no min / offset — Q8_0 is a symmetric scale-only quant.

/// Q8_0 block size: 32 elements per block.
pub const Q8_0_BLOCK_SIZE: usize = 32;

/// Q8_0 block byte size: 2 (d, fp16) + 32 (32 × i8) = 34 bytes.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Dequantise one Q8_0 block into `Q8_0_BLOCK_SIZE` f32s. Caller-
/// allocated output per ADR-029 D5.
pub fn dequant_q8_0_block(block: &[u8; Q8_0_BLOCK_BYTES], output: &mut [f32; Q8_0_BLOCK_SIZE]) {
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for k in 0..Q8_0_BLOCK_SIZE {
        // Each `qs[k]` is a signed 8-bit value stored as a raw byte.
        // Reinterpret the unsigned byte as i8 (two's-complement view)
        // before promoting to f32 — `byte as i8` is a no-op pattern
        // at the bit level on every platform Rust supports.
        let q = block[2 + k] as i8;
        output[k] = (q as f32) * d;
    }
}

/// Dequantise N consecutive Q8_0 blocks into f32 output.
/// `output.len() == num_blocks * Q8_0_BLOCK_SIZE`.
pub fn dequant_q8_0_row(blocks: &[u8], output: &mut [f32], num_blocks: usize) {
    debug_assert_eq!(blocks.len(), num_blocks * Q8_0_BLOCK_BYTES);
    debug_assert_eq!(output.len(), num_blocks * Q8_0_BLOCK_SIZE);
    for i in 0..num_blocks {
        let block_slice: &[u8; Q8_0_BLOCK_BYTES] = blocks
            [i * Q8_0_BLOCK_BYTES..(i + 1) * Q8_0_BLOCK_BYTES]
            .try_into()
            .unwrap();
        let out_slice: &mut [f32; Q8_0_BLOCK_SIZE] = (&mut output
            [i * Q8_0_BLOCK_SIZE..(i + 1) * Q8_0_BLOCK_SIZE])
            .try_into()
            .unwrap();
        dequant_q8_0_block(block_slice, out_slice);
    }
}

// ── Q6_K Constants ─────────────────────────────────────────────────

/// Q6_K super-block size: 256 elements per block.
pub const Q6K_BLOCK_SIZE: usize = 256;

/// Q6_K block byte size: 210 bytes per block.
///   ql[128] + qh[64] + scales[16] + d[2] = 210
pub const Q6K_BLOCK_BYTES: usize = 210;

// ── Q6_K Block Dequantization ──────────────────────────────────────
//
// ── MP2.2b Bit-Packing Resolution Summary ─────────────────────────
//
// Three Q6_K bit-packing pattern candidates were implemented during
// MP2.2b and tested against a gguf-py black-box reference dump of
// output.weight block 0:
//
//   Pattern A — Sequential sub-block layout
//   Pattern B — Interleaved 32-element groups
//   Pattern C — Half-half layout (elements 0-127 use ql[0..64] +
//               qh[0..32], elements 128-255 use ql[64..128] +
//               qh[32..64])
//
// Pattern C matched the reference within 1e-7 tolerance and is the
// canonical resolved pattern. dequant_q6k_block dispatches to
// Pattern C.
//
// Patterns A and B are retained as public functions for regression
// testing and educational reference. They are NOT semantically
// correct for real GGUF data and must not be used directly in
// production code paths. See test provenance notes below.
//
// Resolution provenance:
//   ~/zero-discovery/q6k-reference/dump_q6k_reference.py
//   ~/zero-discovery/q6k-reference/q6k_reference_values.txt
// (Discovery tooling lives outside the workspace per scratch/* exclude.)
//

/// Dequantize one Q6_K super-block (256 elements) into f32 output.
///
/// Block layout (210 bytes total, per Discovery v1 + ggml-common.h):
///   bytes   0-127:  ql[128]    quants, lower 4 bits
///   bytes 128-191:  qh[64]     quants, upper 2 bits
///   bytes 192-207:  scales[16] per-sub-block 8-bit signed scales
///   bytes 208-209:  d (FP16)   super-block scale
///
/// Element ordering matches ggml `dequantize_row_q6_K` exactly:
/// two 128-element halves, each with 4-way interleaved sub-blocks.
/// Validated via three-source confluence: ggml C + gguf-py native + Rust.
///
/// Caller-allocated output buffer per ADR-029 D5.
pub fn dequant_q6k_block(block: &[u8; Q6K_BLOCK_BYTES], output: &mut [f32; Q6K_BLOCK_SIZE]) {
    let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];

    // Two halves of 128 elements each, matching ggml dequantize_row_q6_K
    for half in 0..2usize {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let sc_off = half * 8;
        let y_off = half * 128;

        for l in 0..32usize {
            let is = l / 16; // 0 or 1 within this half

            let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

            output[y_off + l] = d * (sc[sc_off + is] as i8 as f32) * (q1 as f32);
            output[y_off + l + 32] = d * (sc[sc_off + is + 2] as i8 as f32) * (q2 as f32);
            output[y_off + l + 64] = d * (sc[sc_off + is + 4] as i8 as f32) * (q3 as f32);
            output[y_off + l + 96] = d * (sc[sc_off + is + 6] as i8 as f32) * (q4 as f32);
        }
    }
}

/// Pattern A: Sequential sub-block layout.
/// NOT the resolved Q6_K bit-packing pattern.
/// Retained for regression-testing and educational reference.
///
/// Assumes element `i` maps to `ql[i/2]` nibble `i%2`, `qh[i/4]` bits `(i%4)*2`.
pub fn dequant_q6k_block_pattern_a(
    block: &[u8; Q6K_BLOCK_BYTES],
    output: &mut [f32; Q6K_BLOCK_SIZE],
) {
    let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128];
    let qh = &block[128..192];

    for sb in 0..16usize {
        let sub_scale = block[192 + sb] as i8;
        let effective_scale = d * (sub_scale as f32);

        for e in 0..16usize {
            let global_idx = sb * 16 + e;
            // Sequential: direct index mapping
            let ql_byte = global_idx / 2;
            let ql_nibble = if global_idx % 2 == 0 {
                ql[ql_byte] & 0x0F
            } else {
                (ql[ql_byte] >> 4) & 0x0F
            };
            let qh_byte = global_idx / 4;
            let qh_2bits = (qh[qh_byte] >> ((global_idx % 4) * 2)) & 0x03;

            let signed_quant = ((qh_2bits as i32) << 4 | ql_nibble as i32) - 32;
            output[global_idx] = effective_scale * (signed_quant as f32);
        }
    }
}

/// Pattern B: Interleaved 32-element groups.
/// NOT the resolved Q6_K bit-packing pattern.
/// Retained for regression-testing and educational reference.
///
/// Groups of 32 elements share 16 ql bytes (low+high nibbles) and 8 qh bytes.
pub fn dequant_q6k_block_pattern_b(
    block: &[u8; Q6K_BLOCK_BYTES],
    output: &mut [f32; Q6K_BLOCK_SIZE],
) {
    let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128];
    let qh = &block[128..192];

    for group in 0..8usize {
        let sb_lo = group * 2;
        let sb_hi = group * 2 + 1;
        let scale_lo = block[192 + sb_lo] as i8;
        let scale_hi = block[192 + sb_hi] as i8;
        let d_lo = d * (scale_lo as f32);
        let d_hi = d * (scale_hi as f32);

        let ql_offset = group * 16;
        let qh_offset = group * 8;

        for e in 0..16usize {
            // Lower 16: low nibbles
            let ql_lo = ql[ql_offset + e] & 0x0F;
            let qh_byte_lo = qh_offset + e / 2;
            let qh_shift_lo = (e % 2) * 4;
            let qh_lo = (qh[qh_byte_lo] >> qh_shift_lo) & 0x03;
            let q_lo = ((qh_lo as i32) << 4 | ql_lo as i32) - 32;
            output[group * 32 + e] = d_lo * (q_lo as f32);

            // Upper 16: high nibbles
            let ql_hi = (ql[ql_offset + e] >> 4) & 0x0F;
            let qh_hi = (qh[qh_byte_lo] >> (qh_shift_lo + 2)) & 0x03;
            let q_hi = ((qh_hi as i32) << 4 | ql_hi as i32) - 32;
            output[group * 32 + 16 + e] = d_hi * (q_hi as f32);
        }
    }
}

/// Pattern C: Half-half layout (RESOLVED canonical pattern).
///
/// ql split into two halves: ql[0..64] for elements 0..127, ql[64..128] for 128..255.
/// qh split into two halves: qh[0..32] for elements 0..127, qh[32..64] for 128..255.
/// Within each half, element `i%128` maps to `ql[half*64 + i/2]` nibble, `qh[half*32 + i/4]` bits.
///
/// Validated against gguf-py black-box reference: output.weight block 0, max_diff < 1e-7.
pub fn dequant_q6k_block_pattern_c(
    block: &[u8; Q6K_BLOCK_BYTES],
    output: &mut [f32; Q6K_BLOCK_SIZE],
) {
    let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128];
    let qh = &block[128..192];

    for sb in 0..16usize {
        let sub_scale = block[192 + sb] as i8;
        let effective_scale = d * (sub_scale as f32);

        for e in 0..16usize {
            let global_idx = sb * 16 + e;
            let half = global_idx / 128; // 0 or 1
            let half_idx = global_idx % 128;

            // ql: lower 4 bits
            let ql_byte = half * 64 + half_idx / 2;
            let ql_nibble = if half_idx % 2 == 0 {
                ql[ql_byte] & 0x0F
            } else {
                (ql[ql_byte] >> 4) & 0x0F
            };

            // qh: upper 2 bits
            let qh_byte = half * 32 + half_idx / 4;
            let qh_2bits = (qh[qh_byte] >> ((half_idx % 4) * 2)) & 0x03;

            let signed_quant = ((qh_2bits as i32) << 4 | ql_nibble as i32) - 32;
            output[global_idx] = effective_scale * (signed_quant as f32);
        }
    }
}

/// Dequantize N consecutive Q6_K blocks into f32 output.
/// Caller-allocated output buffer (output.len() must equal num_blocks * 256).
pub fn dequant_q6k_row(blocks: &[u8], output: &mut [f32], num_blocks: usize) {
    debug_assert_eq!(blocks.len(), num_blocks * Q6K_BLOCK_BYTES);
    debug_assert_eq!(output.len(), num_blocks * Q6K_BLOCK_SIZE);
    for i in 0..num_blocks {
        let block_slice: &[u8; Q6K_BLOCK_BYTES] = blocks
            [i * Q6K_BLOCK_BYTES..(i + 1) * Q6K_BLOCK_BYTES]
            .try_into()
            .unwrap();
        let out_slice: &mut [f32; Q6K_BLOCK_SIZE] = (&mut output
            [i * Q6K_BLOCK_SIZE..(i + 1) * Q6K_BLOCK_SIZE])
            .try_into()
            .unwrap();
        dequant_q6k_block(block_slice, out_slice);
    }
}

// ── Sizing constants for additional quant types ────────────────────
//
// These cover quants the inference path may encounter when loading
// mixed-quant GGUFs (e.g. unsloth Kimi K2.6). We currently only need
// byte-sizing here so `get_weight_bytes` can hand a correctly-sized
// slice to the loader — full dequantisers for these formats may be
// implemented later.
//
// Block geometry sources: ggml-common.h (llama.cpp), GGUF spec.

/// Q5_K super-block: 256 elements, 176 bytes
/// (d:fp16 + dmin:fp16 + scales[12] + qh[32] + qs[128]).
pub const Q5K_BLOCK_SIZE: usize = 256;
pub const Q5K_BLOCK_BYTES: usize = 176;

/// Q3_K super-block: 256 elements, 110 bytes
/// (hmask[32] + qs[64] + scales[12] + d:fp16).
pub const Q3K_BLOCK_SIZE: usize = 256;
pub const Q3K_BLOCK_BYTES: usize = 110;

/// Q2_K super-block: 256 elements, 84 bytes
/// (scales[16] + qs[64] + d:fp16 + dmin:fp16).
pub const Q2K_BLOCK_SIZE: usize = 256;
pub const Q2K_BLOCK_BYTES: usize = 84;

/// Q8_K super-block: 256 elements, 292 bytes
/// (d:f32 + qs[256] + bsums[16:i16]).
pub const Q8K_BLOCK_SIZE: usize = 256;
pub const Q8K_BLOCK_BYTES: usize = 292;

/// Q5_0 block: 32 elements, 22 bytes (d:fp16 + qh[4] + qs[16]).
pub const Q5_0_BLOCK_SIZE: usize = 32;
pub const Q5_0_BLOCK_BYTES: usize = 22;

/// Q5_1 block: 32 elements, 24 bytes (d:fp16 + m:fp16 + qh[4] + qs[16]).
pub const Q5_1_BLOCK_SIZE: usize = 32;
pub const Q5_1_BLOCK_BYTES: usize = 24;

/// Q4_1 block: 32 elements, 20 bytes (d:fp16 + m:fp16 + qs[16]).
pub const Q4_1_BLOCK_SIZE: usize = 32;
pub const Q4_1_BLOCK_BYTES: usize = 20;

/// Q8_1 block: 32 elements, 36 bytes (d:fp16 + s:fp16 + qs[32:i8]).
pub const Q8_1_BLOCK_SIZE: usize = 32;
pub const Q8_1_BLOCK_BYTES: usize = 36;

/// IQ4_NL block: 32 elements, 18 bytes (d:fp16 + qs[16]).
pub const IQ4NL_BLOCK_SIZE: usize = 32;
pub const IQ4NL_BLOCK_BYTES: usize = 18;

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    // ── FP16→F32 Tests ──────────────────────────────────────────

    #[test]
    fn test_fp16_positive_zero() {
        assert_eq!(fp16_to_f32(0x0000), 0.0f32);
        assert!(fp16_to_f32(0x0000).is_sign_positive());
    }

    #[test]
    fn test_fp16_negative_zero() {
        let val = fp16_to_f32(0x8000);
        assert_eq!(val, -0.0f32);
        assert!(val.is_sign_negative());
    }

    #[test]
    fn test_fp16_one() {
        assert_eq!(fp16_to_f32(0x3C00), 1.0f32);
    }

    #[test]
    fn test_fp16_negative_one() {
        assert_eq!(fp16_to_f32(0xBC00), -1.0f32);
    }

    #[test]
    fn test_fp16_largest_finite() {
        assert_eq!(fp16_to_f32(0x7BFF), 65504.0f32);
    }

    #[test]
    fn test_fp16_smallest_normal() {
        // 2^-14 ≈ 6.103_515_6e-5
        let val = fp16_to_f32(0x0400);
        let expected: f32 = 6.103_515_6e-5;
        assert!(
            (val - expected).abs() < 1e-10,
            "got {}, expected {}",
            val,
            expected
        );
    }

    #[test]
    fn test_fp16_smallest_denormal() {
        // 2^-24 ≈ 5.960_464_5e-8
        let val = fp16_to_f32(0x0001);
        let expected: f32 = 5.960_464_5e-8;
        assert!(
            (val - expected).abs() < 1e-14,
            "got {}, expected {}",
            val,
            expected
        );
    }

    #[test]
    fn test_fp16_positive_infinity() {
        assert_eq!(fp16_to_f32(0x7C00), f32::INFINITY);
    }

    #[test]
    fn test_fp16_negative_infinity() {
        assert_eq!(fp16_to_f32(0xFC00), f32::NEG_INFINITY);
    }

    #[test]
    fn test_fp16_nan() {
        assert!(fp16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn test_fp16_two() {
        // 2.0 in fp16 = 0x4000
        assert_eq!(fp16_to_f32(0x4000), 2.0f32);
    }

    #[test]
    fn test_fp16_half() {
        // 0.5 in fp16 = 0x3800
        assert_eq!(fp16_to_f32(0x3800), 0.5f32);
    }

    // ── get_scale_min_k4 Tests ──────────────────────────────────

    #[test]
    fn test_scale_min_k4_low_indices() {
        // j < 4: scale = scales[j] & 0x3F, min = scales[j+4] & 0x3F
        let mut scales = [0u8; 12];
        scales[0] = 0xFF; // scale[0] should be 0x3F = 63
        scales[4] = 0xAA; // min[0] should be 0x2A = 42
        let (sc, mn) = get_scale_min_k4(&scales, 0);
        assert_eq!(sc, 63);
        assert_eq!(mn, 42);
    }

    #[test]
    fn test_scale_min_k4_high_indices() {
        // j >= 4: more complex 6-bit packing
        let mut scales = [0u8; 12];
        // For j=4: scale = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
        //          min   = (scales[8] >> 4)   | ((scales[4] >> 6) << 4)
        scales[0] = 0xC0; // bits 7:6 = 0b11 → contributes 0b11 << 4 = 48 to scale[4]
        scales[4] = 0x80; // bits 7:6 = 0b10 → contributes 0b10 << 4 = 32 to min[4]
        scales[8] = 0x53; // low nibble = 3 → scale low, high nibble = 5 → min low
        let (sc, mn) = get_scale_min_k4(&scales, 4);
        assert_eq!(sc, 3 | (3 << 4)); // 3 + 48 = 51
        assert_eq!(mn, 5 | (2 << 4)); // 5 + 32 = 37
    }

    // ── Q4_K Block Tests ────────────────────────────────────────

    /// Helper: build a Q4_K block with specified d, dmin, scales, quants.
    fn build_q4k_block(d_fp16: u16, dmin_fp16: u16, scales: [u8; 12], qs: [u8; 128]) -> [u8; 144] {
        let mut block = [0u8; 144];
        block[0..2].copy_from_slice(&d_fp16.to_le_bytes());
        block[2..4].copy_from_slice(&dmin_fp16.to_le_bytes());
        block[4..16].copy_from_slice(&scales);
        block[16..144].copy_from_slice(&qs);
        block
    }

    #[test]
    fn test_q4k_zero_d_zero_dmin() {
        // d=0, dmin=0 → all outputs should be 0
        let block = build_q4k_block(0x0000, 0x0000, [0; 12], [0; 128]);
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        for &v in output.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q4k_zero_quants_with_min() {
        // d=1.0 (0x3C00), dmin=0.5 (0x3800)
        // All scales[j]=1, all mins[j]=1 (low indices, & 0x3F)
        // All qs=0 → output = d*scale*(0) - dmin*min = -0.5 for all
        let mut scales = [0u8; 12];
        // For j=0..3: scales[j] = 1 (scale), scales[j+4] = 1 (min)
        for i in 0..4 {
            scales[i] = 1;
            scales[i + 4] = 1;
        }
        // For j=4..7: scales[j+4] low nibble = 1 (scale low), high nibble = 1 (min low)
        // scales[j-4] >> 6 = 0 (scale high), scales[j] >> 6 = 0 (min high)
        for scale in &mut scales[8..12] {
            *scale = 0x11;
        } // low=1, high=1

        let block = build_q4k_block(0x3C00, 0x3800, scales, [0; 128]);
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        // output = 1.0 * 1 * 0 - 0.5 * 1 = -0.5
        for &v in output.iter() {
            assert!((v - (-0.5)).abs() < 1e-6, "expected -0.5, got {}", v);
        }
    }

    #[test]
    fn test_q4k_max_quants() {
        // d=1.0, dmin=0, all scales=1, all quants=0xFF (both nibbles=15)
        // output = 1.0 * 1 * 15 - 0 = 15.0
        let mut scales = [0u8; 12];
        for scale in &mut scales[0..4] {
            *scale = 1;
        }
        for scale in &mut scales[8..12] {
            *scale = 0x01;
        } // scale low nibble = 1 for j>=4

        let block = build_q4k_block(0x3C00, 0x0000, scales, [0xFF; 128]);
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        for &v in output.iter() {
            assert!((v - 15.0).abs() < 1e-6, "expected 15.0, got {}", v);
        }
    }

    #[test]
    fn test_q4k_known_values_first_group() {
        // Hand-calculated: d=2.0 (0x4000), dmin=1.0 (0x3C00)
        // scales[0]=3 (scale for sub-block 0), scales[4]=2 (min for sub-block 0)
        // scales[1]=5 (scale for sub-block 1), scales[5]=4 (min for sub-block 1)
        // First qs byte = 0x73 → low nibble=3, high nibble=7
        // output[0] = d * scale[0] * 3 - dmin * min[0] = 2*3*3 - 1*2 = 16
        // output[32] = d * scale[1] * 7 - dmin * min[1] = 2*5*7 - 1*4 = 66
        let mut scales = [0u8; 12];
        scales[0] = 3;
        scales[1] = 5;
        scales[4] = 2;
        scales[5] = 4;

        let mut qs = [0u8; 128];
        qs[0] = 0x73; // low=3, high=7

        let block = build_q4k_block(0x4000, 0x3C00, scales, qs);
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);

        assert!((output[0] - 16.0).abs() < 1e-4, "output[0]={}", output[0]);
        assert!(
            (output[32] - 66.0).abs() < 1e-4,
            "output[32]={}",
            output[32]
        );
    }

    #[test]
    fn test_q4k_signed_min_dominates() {
        // d=0, dmin=2.0 (0x4000), all mins=10, all quants=0
        // output = 0 * scale * quant - 2.0 * 10 = -20.0
        let mut scales = [0u8; 12];
        for i in 0..4 {
            scales[i + 4] = 10;
        } // min for j<4
        for scale in &mut scales[8..12] {
            *scale = 0xA0;
        } // min high nibble = 10 for j>=4

        let block = build_q4k_block(0x0000, 0x4000, scales, [0; 128]);
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        for &v in output.iter() {
            assert!((v - (-20.0)).abs() < 1e-4, "expected -20.0, got {}", v);
        }
    }

    #[test]
    fn test_q4k_zero_scale_nonzero_quant() {
        // d=1.0, dmin=0, all scales=0, quants=0xFF
        // output = 1.0 * 0 * 15 - 0 = 0.0
        let block = build_q4k_block(0x3C00, 0x0000, [0; 12], [0xFF; 128]);
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        for &v in output.iter() {
            assert_eq!(v, 0.0, "expected 0.0 with zero scales");
        }
    }

    // ── Q4_K Row Tests ──────────────────────────────────────────

    #[test]
    fn test_q4k_row_two_blocks() {
        // Two identical blocks, verify output is 512 elements
        let block = build_q4k_block(0x3C00, 0x0000, [0; 12], [0; 128]);
        let mut blocks_buf = [0u8; 288];
        blocks_buf[..144].copy_from_slice(&block);
        blocks_buf[144..288].copy_from_slice(&block);

        let mut output = [0.0f32; 512];
        dequant_q4k_row(&blocks_buf, &mut output, 2);
        // Both blocks should produce all zeros (d=1, scales=0, qs=0)
        for &v in output.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q4k_row_distinct_blocks() {
        // Block 0: d=1.0, dmin=0, scale[0]=2, qs[0]=0x05 → output[0] = 1*2*5 = 10
        let mut scales0 = [0u8; 12];
        scales0[0] = 2;
        let mut qs0 = [0u8; 128];
        qs0[0] = 0x05;
        let block0 = build_q4k_block(0x3C00, 0x0000, scales0, qs0);

        // Block 1: d=0.5, dmin=0, scale[0]=4, qs[0]=0x03 → output[256] = 0.5*4*3 = 6
        let mut scales1 = [0u8; 12];
        scales1[0] = 4;
        let mut qs1 = [0u8; 128];
        qs1[0] = 0x03;
        let block1 = build_q4k_block(0x3800, 0x0000, scales1, qs1);

        let mut blocks_buf = [0u8; 288];
        blocks_buf[..144].copy_from_slice(&block0);
        blocks_buf[144..].copy_from_slice(&block1);

        let mut output = [0.0f32; 512];
        dequant_q4k_row(&blocks_buf, &mut output, 2);

        assert!((output[0] - 10.0).abs() < 1e-4, "block0[0]={}", output[0]);
        assert!(
            (output[256] - 6.0).abs() < 1e-4,
            "block1[0]={}",
            output[256]
        );
    }

    // ── Q4_0 Tests ──────────────────────────────────────────────

    /// Build a Q4_0 block (18 bytes) with the given fp16 scale and
    /// a 16-byte qs payload. Mirrors `build_q4k_block` for test ergonomics.
    fn build_q4_0_block(d_fp16: u16, qs: [u8; 16]) -> [u8; 18] {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&d_fp16.to_le_bytes());
        block[2..18].copy_from_slice(&qs);
        block
    }

    #[test]
    fn test_q4_0_block_layout_constants() {
        assert_eq!(Q4_0_BLOCK_SIZE, 32);
        assert_eq!(Q4_0_BLOCK_BYTES, 18);
    }

    #[test]
    fn test_q4_0_zero_scale_yields_all_zero() {
        // d=0 → every output is 0 regardless of nibble payload.
        let block = build_q4_0_block(0x0000, [0xFF; 16]);
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        for &v in output.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q4_0_max_nibbles_yields_seven_times_scale() {
        // d=1.0, every nibble = 15 → (15 - 8) * 1.0 = 7.0
        let block = build_q4_0_block(0x3C00, [0xFF; 16]);
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 7.0).abs() < 1e-6, "output[{i}] = {v}, want 7.0");
        }
    }

    #[test]
    fn test_q4_0_min_nibbles_yields_minus_eight_times_scale() {
        // d=1.0, every nibble = 0 → (0 - 8) * 1.0 = -8.0
        let block = build_q4_0_block(0x3C00, [0x00; 16]);
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - (-8.0)).abs() < 1e-6, "output[{i}] = {v}, want -8.0");
        }
    }

    #[test]
    fn test_q4_0_low_high_nibble_split() {
        // d=1.0, byte 0 = 0x80 → low nibble=0, high nibble=8.
        // Per llama.cpp packing: low → output[0], high → output[0+16].
        let mut qs = [0u8; 16];
        qs[0] = 0x80;
        let block = build_q4_0_block(0x3C00, qs);
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        assert!(
            (output[0] - (-8.0)).abs() < 1e-6,
            "output[0] (low nibble of byte 0) = {}, want -8.0",
            output[0]
        );
        assert!(
            (output[16] - 0.0).abs() < 1e-6,
            "output[16] (high nibble of byte 0) = {}, want 0.0",
            output[16]
        );
        // Bytes 1..15 are 0x00 → both nibbles -8.0 in their respective
        // output slots.
        for i in 1..16 {
            assert!(
                (output[i] - (-8.0)).abs() < 1e-6,
                "output[{i}] = {}",
                output[i]
            );
            assert!(
                (output[i + 16] - (-8.0)).abs() < 1e-6,
                "output[{}] = {}",
                i + 16,
                output[i + 16]
            );
        }
    }

    #[test]
    fn test_q4_0_scale_two_known_values() {
        // d=2.0 (fp16 = 0x4000), bytes hand-picked to give known outputs.
        // byte 0 = 0x12 → low=2, high=1
        //   output[0] = (2 - 8) * 2 = -12
        //   output[16] = (1 - 8) * 2 = -14
        let mut qs = [0u8; 16];
        qs[0] = 0x12;
        let block = build_q4_0_block(0x4000, qs);
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        assert!((output[0] - (-12.0)).abs() < 1e-6, "{}", output[0]);
        assert!((output[16] - (-14.0)).abs() < 1e-6, "{}", output[16]);
    }

    #[test]
    fn test_q4_0_row_multiple_blocks() {
        // Two blocks back-to-back: block0 d=1.0 max nibbles → 7.0 everywhere,
        // block1 d=0.5 min nibbles → -4.0 everywhere.
        let block0 = build_q4_0_block(0x3C00, [0xFF; 16]);
        let block1 = build_q4_0_block(0x3800, [0x00; 16]);
        let mut buf = [0u8; 36];
        buf[..18].copy_from_slice(&block0);
        buf[18..].copy_from_slice(&block1);

        let mut output = [0.0f32; 64];
        dequant_q4_0_row(&buf, &mut output, 2);

        for (i, &v) in output[..32].iter().enumerate() {
            assert!((v - 7.0).abs() < 1e-6, "block0[{i}] = {}", v);
        }
        for (i, &v) in output[32..64].iter().enumerate() {
            assert!((v - (-4.0)).abs() < 1e-6, "block1[{i}] = {}", v);
        }
    }

    #[test]
    fn test_q4_0_byte_size_is_eighteen() {
        // Wire-layout regression: any change here breaks the on-disk
        // GGUF format. NOT something we choose — set by the spec.
        let dummy = [0u8; 18];
        assert_eq!(core::mem::size_of_val(&dummy), Q4_0_BLOCK_BYTES);
    }

    // ── Q8_0 Tests ──────────────────────────────────────────────

    /// Build a Q8_0 block (34 bytes) with the given fp16 scale and
    /// a 32-byte qs payload. Mirrors `build_q4k_block` for ergonomics.
    fn build_q8_0_block(d_fp16: u16, qs: [i8; 32]) -> [u8; 34] {
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&d_fp16.to_le_bytes());
        for (i, &v) in qs.iter().enumerate() {
            // i8 → byte via `to_ne_bytes` to make the
            // two's-complement representation explicit.
            block[2 + i] = v.to_ne_bytes()[0];
        }
        block
    }

    #[test]
    fn test_q8_0_block_layout_constants() {
        assert_eq!(Q8_0_BLOCK_SIZE, 32);
        assert_eq!(Q8_0_BLOCK_BYTES, 34);
    }

    #[test]
    fn test_q8_0_zero_scale_yields_all_zero() {
        // d = 0 → every output is 0 regardless of qs payload.
        let block = build_q8_0_block(0x0000, [127; 32]);
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        for &v in output.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q8_0_max_positive_yields_127_times_scale() {
        // d = 1.0, every qs[k] = 127 (max positive i8) → output = 127.
        let block = build_q8_0_block(0x3C00, [127; 32]);
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 127.0).abs() < 1e-6, "output[{i}] = {v}, want 127.0");
        }
    }

    #[test]
    fn test_q8_0_max_negative_yields_minus_128_times_scale() {
        // d = 1.0, every qs[k] = -128 (min i8) → output = -128.
        let block = build_q8_0_block(0x3C00, [-128; 32]);
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-128.0)).abs() < 1e-6,
                "output[{i}] = {v}, want -128.0"
            );
        }
    }

    #[test]
    fn test_q8_0_sign_extension_correct() {
        // The reinterpret of the raw byte as `i8` is the critical step
        // — if we accidentally treated bytes as unsigned (0..255), the
        // negative half of the qs range would dequantise wrong by
        // 256 × d. Verify with mixed signs.
        let mut qs = [0i8; 32];
        qs[0] = -1;
        qs[1] = 1;
        qs[2] = -100;
        qs[3] = 100;
        qs[4] = 0;
        let block = build_q8_0_block(0x3C00, qs); // d = 1.0
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        assert!((output[0] - (-1.0)).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
        assert!((output[2] - (-100.0)).abs() < 1e-6);
        assert!((output[3] - 100.0).abs() < 1e-6);
        assert!((output[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_q8_0_scale_two_known_values() {
        // d = 2.0, qs[0] = 3, qs[1] = -7 → output[0] = 6.0,
        // output[1] = -14.0.
        let mut qs = [0i8; 32];
        qs[0] = 3;
        qs[1] = -7;
        let block = build_q8_0_block(0x4000, qs);
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        assert!((output[0] - 6.0).abs() < 1e-6, "{}", output[0]);
        assert!((output[1] - (-14.0)).abs() < 1e-6, "{}", output[1]);
    }

    #[test]
    fn test_q8_0_row_multiple_blocks() {
        // Two blocks: block0 d=1.0 qs=+10, block1 d=0.5 qs=-20
        //   → block0 outputs all 10.0, block1 outputs all -10.0
        let block0 = build_q8_0_block(0x3C00, [10; 32]);
        let block1 = build_q8_0_block(0x3800, [-20; 32]);
        let mut buf = [0u8; 68];
        buf[..34].copy_from_slice(&block0);
        buf[34..].copy_from_slice(&block1);

        let mut output = [0.0f32; 64];
        dequant_q8_0_row(&buf, &mut output, 2);

        for (i, &v) in output[..32].iter().enumerate() {
            assert!((v - 10.0).abs() < 1e-6, "block0[{i}] = {}", v);
        }
        for (i, &v) in output[32..64].iter().enumerate() {
            assert!((v - (-10.0)).abs() < 1e-6, "block1[{i}] = {}", v);
        }
    }

    #[test]
    fn test_q8_0_byte_size_is_thirty_four() {
        let dummy = [0u8; 34];
        assert_eq!(core::mem::size_of_val(&dummy), Q8_0_BLOCK_BYTES);
    }

    // ── Q6_K Tests ──────────────────────────────────────────────

    #[test]
    fn test_q6k_block_layout_constants() {
        assert_eq!(Q6K_BLOCK_BYTES, 210);
        assert_eq!(Q6K_BLOCK_SIZE, 256);
    }

    /// Helper: build a Q6_K block with specified ql, qh, scales, d.
    fn build_q6k_block(ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d_fp16: u16) -> [u8; 210] {
        let mut block = [0u8; 210];
        block[0..128].copy_from_slice(&ql);
        block[128..192].copy_from_slice(&qh);
        // scales as i8 → u8
        for i in 0..16 {
            block[192 + i] = scales[i] as u8;
        }
        block[208..210].copy_from_slice(&d_fp16.to_le_bytes());
        block
    }

    #[test]
    fn test_q6k_zero_d_zero_scales() {
        let block = build_q6k_block([0; 128], [0; 64], [0; 16], 0x0000);
        let mut output = [0.0f32; 256];
        dequant_q6k_block(&block, &mut output);
        for &v in output.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q6k_zero_quants_with_scale() {
        // d=1.0 (0x3C00), scales[0]=2, all ql=qh=0
        // quant = (0<<4 | 0) - 32 = -32
        // output[0..16] = 1.0 * 2 * (-32) = -64.0
        let mut scales = [0i8; 16];
        scales[0] = 2;
        let block = build_q6k_block([0; 128], [0; 64], scales, 0x3C00);
        let mut output = [0.0f32; 256];
        dequant_q6k_block(&block, &mut output);
        // sub-block 0 (elements 0..15): scale=2, quant=-32 → -64.0
        for (i, &v) in output[..16].iter().enumerate() {
            assert!((v - (-64.0)).abs() < 1e-4, "output[{}]={}", i, v);
        }
        // sub-block 1 (elements 16..31): scale=0, quant=-32 → 0.0
        for (i, &v) in output[16..32].iter().enumerate() {
            assert_eq!(v, 0.0, "output[{}]={}", i, v);
        }
    }

    #[test]
    fn test_q6k_max_quants_pattern_c() {
        // d=1.0, scales[0]=1
        // Max 6-bit quant: ql nibble=0xF, qh 2bits=0x3 → (3<<4|15) = 63 → signed = 31
        // For Pattern C (half-half): element 0 reads ql[0] low nibble, qh[0] bits 0:1
        let ql = [0xFFu8; 128]; // all nibbles = 15
        let qh = [0xFFu8; 64]; // all 2-bit fields = 3
        let scales = [1i8; 16];
        let block = build_q6k_block(ql, qh, scales, 0x3C00);
        let mut output = [0.0f32; 256];
        dequant_q6k_block_pattern_c(&block, &mut output);
        // quant = (3<<4|15) - 32 = 63 - 32 = 31
        for &v in output.iter() {
            assert!((v - 31.0).abs() < 1e-4, "expected 31.0, got {}", v);
        }
    }

    // ── PATTERN C Hand-Math Test — Provenance Note ──────────────────
    //
    // This test verifies Pattern C's SELF-CONSISTENCY: hand-constructed
    // block, manually pre-computed outputs.
    //
    // Pattern C's SEMANTIC CORRECTNESS for real GGUF model data is
    // separately established by test_q6k_pattern_c_reference_values
    // below, which loads the actual Qwen3 1.7B Q4_K_M model file,
    // dequantizes output.weight block 0, and asserts match against 16
    // gguf-py-generated reference values within 1e-4 tolerance.
    //
    // Pattern C is the canonical resolved Q6_K bit-packing pattern;
    // dequant_q6k_block dispatches to Pattern C.
    #[test]
    fn test_q6k_known_values_pattern_c() {
        // Hand-calculated for Pattern C (half-half):
        // d=2.0 (0x4000), scales[0]=3
        // Element 0 (half=0, half_idx=0): ql[0] low nibble, qh[0] bits 0:1
        // Set ql[0]=0x05 (low=5, high=0), qh[0]=0x01 (bits 0:1 = 1, rest=0)
        // quant = (1<<4 | 5) - 32 = 21 - 32 = -11
        // output[0] = 2.0 * 3 * (-11) = -66.0
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        ql[0] = 0x05; // element 0 low nibble = 5
        qh[0] = 0x01; // element 0 qh bits 0:1 = 1
        scales[0] = 3;
        let block = build_q6k_block(ql, qh, scales, 0x4000);
        let mut output = [0.0f32; 256];
        dequant_q6k_block_pattern_c(&block, &mut output);
        assert!(
            (output[0] - (-66.0)).abs() < 1e-4,
            "output[0]={}",
            output[0]
        );
    }

    // ── PATTERN A Hand-Math Test — Provenance Note ──────────────────
    //
    // This test verifies Pattern A's SELF-CONSISTENCY: it constructs a
    // Q6_K block with known scales, ql, qh values and asserts that
    // dequant_q6k_block_pattern_a produces the manually pre-computed
    // outputs.
    //
    // It does NOT prove Pattern A's semantic correctness for real GGUF
    // model data. Pattern A and Pattern C are mathematically identical
    // for elements 0-127 (both use `ql_byte = sb*8 + e/2` for the first
    // half), so any test exercising only this region cannot distinguish
    // between the two patterns.
    //
    // Pattern C is the resolved canonical Q6_K bit-packing pattern,
    // verified empirically via gguf-py black-box reference (see
    // test_q6k_pattern_c_reference_values).
    //
    // Pattern A is retained as a public function for regression testing
    // and educational reference. Its full semantic correctness for
    // elements 128-255 is intentionally NOT established by this test.
    #[test]
    fn test_q6k_known_values_pattern_a() {
        // Pattern A (sequential): element 0 → ql[0] low nibble, qh[0] bits 0:1
        // Same inputs as Pattern C test → same result for element 0
        // (Pattern A and C agree for element 0 since half=0, half_idx=0 maps identically)
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        ql[0] = 0x05;
        qh[0] = 0x01;
        scales[0] = 3;
        let block = build_q6k_block(ql, qh, scales, 0x4000);
        let mut output = [0.0f32; 256];
        dequant_q6k_block_pattern_a(&block, &mut output);
        assert!(
            (output[0] - (-66.0)).abs() < 1e-4,
            "output[0]={}",
            output[0]
        );
    }

    // ── PATTERN B Hand-Math Test — Provenance Note ──────────────────
    //
    // This test verifies Pattern B's SELF-CONSISTENCY in the same sense
    // as the Pattern A note above: hand-constructed block, manually
    // pre-computed outputs, no real GGUF data.
    //
    // Pattern B (interleaved 32-element groups) was empirically
    // rejected during MP2.2b bit-packing resolution against the
    // gguf-py reference. It is retained as a public function for
    // regression testing and educational reference.
    //
    // For canonical Q6_K dequantization, dequant_q6k_block dispatches
    // to Pattern C (verified via test_q6k_pattern_c_reference_values).
    #[test]
    fn test_q6k_known_values_pattern_b() {
        // Pattern B (32-group): element 0 → group 0, sb_lo, ql[0] low, qh[0] bit 0
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        ql[0] = 0x05;
        qh[0] = 0x01;
        scales[0] = 3;
        let block = build_q6k_block(ql, qh, scales, 0x4000);
        let mut output = [0.0f32; 256];
        dequant_q6k_block_pattern_b(&block, &mut output);
        assert!(
            (output[0] - (-66.0)).abs() < 1e-4,
            "output[0]={}",
            output[0]
        );
    }

    #[test]
    fn test_q6k_negative_scale() {
        // d=1.0, scales[0]=-4 (negative signed scale)
        // Element 0: ql=qh=0 → quant=-32
        // output[0] = 1.0 * (-4) * (-32) = 128.0
        let mut scales = [0i8; 16];
        scales[0] = -4;
        let block = build_q6k_block([0; 128], [0; 64], scales, 0x3C00);
        let mut output = [0.0f32; 256];
        dequant_q6k_block(&block, &mut output);
        assert!((output[0] - 128.0).abs() < 1e-4, "output[0]={}", output[0]);
    }

    #[test]
    fn test_q6k_row_two_blocks() {
        let block = build_q6k_block([0; 128], [0; 64], [0; 16], 0x0000);
        let mut blocks_buf = [0u8; 420];
        blocks_buf[..210].copy_from_slice(&block);
        blocks_buf[210..420].copy_from_slice(&block);
        let mut output = [0.0f32; 512];
        dequant_q6k_row(&blocks_buf, &mut output, 2);
        for &v in output.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q6k_pattern_c_reference_values() {
        // Reference values from gguf-py black-box dump of output.weight block 0.
        // This test loads the actual GGUF model file and compares against reference.
        // Skip if model file not present (CI environments).
        let model_path = "kernel/programs/Qwen_Qwen3-1.7B-Q4_K_M.gguf";
        let Ok(model_bytes) = std::fs::read(model_path) else {
            std::println!("Model file not found, skipping reference test");
            return;
        };
        // tensor_data_offset and output.weight offset from MP2.1 boot output
        let tensor_data_offset: usize = 0x5ad1a0;
        let output_weight_offset: usize = 0;
        let block_start = tensor_data_offset + output_weight_offset;
        let block: &[u8; 210] = model_bytes[block_start..block_start + 210]
            .try_into()
            .unwrap();
        let mut output = [0.0f32; 256];
        dequant_q6k_block_pattern_c(block, &mut output);

        // gguf-py reference values (output.weight block 0, first 16):
        let reference: [f32; 16] = [
            -0.011748076,
            -0.0626564,
            -0.043076277,
            -0.017622113,
            0.011748076,
            -0.035244226,
            0.029370189,
            0.0058740377,
            0.029370189,
            0.009790063,
            -0.029370189,
            0.023496151,
            -0.003916025,
            -0.04503429,
            0.0,
            -0.0156641,
        ];

        let mut max_diff: f32 = 0.0;
        for i in 0..16 {
            let diff = (output[i] - reference[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 1e-4,
                "Pattern C elem [{}]: got {} expected {} diff {:.6e}",
                i,
                output[i],
                reference[i],
                diff
            );
        }
        std::println!(
            "Pattern C reference test PASSED, max_diff = {:.6e}",
            max_diff
        );
    }

    /// Canonical Q6_K dequant test using blk.0.attn_v.weight block 0.
    ///
    /// Verifies `dequant_q6k_block` (the main entry point, ggml-canonical
    /// 4-way interleaved layout) against gguf-py native Q6_K.dequantize_blocks.
    ///
    /// This is the tensor that exposed the Pattern C bug during Sub-MP-C2:
    /// Pattern C produced correct results for output.weight block 0 (by coincidence)
    /// but diverged on attn_v.weight block 0 due to different element ordering.
    ///
    /// Per D11 discipline: reference values from canonical gguf-py, not custom code.
    #[test]
    fn test_q6k_canonical_dequant_attn_v_reference() {
        let model_path = "kernel/programs/Qwen_Qwen3-1.7B-Q4_K_M.gguf";
        let Ok(model_bytes) = std::fs::read(model_path) else {
            std::println!("Model file not found, skipping reference test");
            return;
        };
        let meta = crate::parse_header(&model_bytes).expect("parse failed");
        let t = meta
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.attn_v.weight")
            .expect("attn_v.weight not found");
        let start = meta.tensor_data_offset + t.offset as usize;
        let block: &[u8; 210] = model_bytes[start..start + 210].try_into().unwrap();

        let mut output = [0.0f32; 256];
        dequant_q6k_block(block, &mut output);

        // gguf-py native Q6_K.dequantize_blocks reference (attn_v.weight block 0, first 8):
        let reference: [f32; 8] = [
            6.533253e-02,
            -1.0537505e-02,
            1.0537505e-02,
            -4.215002e-02,
            5.2687526e-02,
            2.9505014e-02,
            -3.793502e-02,
            8.430004e-03,
        ];

        let mut max_diff: f32 = 0.0;
        for i in 0..8 {
            let diff = (output[i] - reference[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 1e-6,
                "Canonical Q6K elem [{}]: got {} expected {} diff {:.6e}",
                i,
                output[i],
                reference[i],
                diff
            );
        }
        std::println!(
            "Canonical Q6K attn_v reference test PASSED, max_diff = {:.6e}",
            max_diff
        );
    }
}
