// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-E2: NEON-accelerated linear projections.
//!
//! Fused dequant + dot-product for Q4_K and Q6_K quantized weight matrices.
//! Per E2-Q2: NEON intrinsics exclusive in arch::aarch64::math/.
//! Per Pillar 1: O(M*K) bounded, stack-only hot-path, zero allocation.
//! Per Pillar 7: ARM NEON exclusive in this module.
//!
//! CITE: ARM ARM C7.2 (NEON intrinsics)
//! CITE: zero-gguf-parser::dequant (Q4_K/Q6_K format reference)

use core::arch::aarch64::*;

/// FP16 to f32 conversion (replicated from sacred parser to avoid
/// sacred crate dependency in HAL — Pillar 7 boundary).
#[inline(always)]
fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let val = (mant as f32) * (1.0f32 / 1024.0f32) * (1.0f32 / 16384.0f32);
            if sign != 0 {
                -val
            } else {
                val
            }
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
        }
    } else {
        let f32_exp = exp - 15 + 127;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Decode Q4_K sub-block scale and min from 12-byte scales array.
/// Replicated from sacred parser for HAL boundary isolation.
#[inline(always)]
fn get_scale_min_k4(scales: &[u8], j: usize) -> (u8, u8) {
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

/// NEON-accelerated Q4_K linear projection: out[i] = dot(x, dequant(W[i]))
///
/// Fused dequant + dot-product per block: dequant 256 elements, then
/// accumulate via 4-wide NEON FMA (vfmaq_f32).
///
/// Q4_K block layout (144 bytes):
///   bytes  0-1:    d (FP16, super-block scale)
///   bytes  2-3:    dmin (FP16, super-block min)
///   bytes  4-15:   scales[12] (8 sub-block scales + mins)
///   bytes 16-143:  qs[128] (256 × 4-bit quants, 2 per byte)
///
/// # Safety
/// Uses ARM NEON intrinsics. Caller MUST ensure:
/// - target_arch = "aarch64"
/// - FPU/NEON enabled (CPACR_EL1.FPEN = 3)
/// - x.len() == in_dim, out.len() == out_dim
/// - in_dim % 256 == 0
/// - w_blocks.len() == out_dim * (in_dim / 256) * 144
pub unsafe fn linear_q4k_neon(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    let blocks_per_row = in_dim / 256;
    let bytes_per_row = blocks_per_row * 144;

    let mut i = 0;
    while i < out_dim {
        let row_byte_offset = i * bytes_per_row;

        // 4-wide accumulator
        let mut acc = vdupq_n_f32(0.0);

        let mut b = 0;
        while b < blocks_per_row {
            let block_off = row_byte_offset + b * 144;
            let block = &w_blocks[block_off..block_off + 144];

            // Parse block header
            let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales = &block[4..16];
            let qs = &block[16..144];

            let x_base = b * 256;

            // 4 groups of 64 elements
            let mut is: usize = 0;
            let mut qs_offset: usize = 0;

            let mut j_group = 0;
            while j_group < 4 {
                let j_base = j_group * 64;

                let (sc0, mn0) = get_scale_min_k4(scales, is);
                let (sc1, mn1) = get_scale_min_k4(scales, is + 1);

                let d1 = d * (sc0 as f32);
                let m1 = dmin * (mn0 as f32);
                let d2 = d * (sc1 as f32);
                let m2 = dmin * (mn1 as f32);

                let d1_v = vdupq_n_f32(d1);
                let m1_v = vdupq_n_f32(m1);
                let d2_v = vdupq_n_f32(d2);
                let m2_v = vdupq_n_f32(m2);

                // Sub-MP-E5 v3: minimal-disruption vectorized dequant.
                // PRESERVES E3's exact iteration structure (l=0,4,8,...,28 stride 4).
                // Replaces ONLY the scalar nibble extraction + stack-roundtrip
                // pattern with NEON-vectorized 4-element dequant.
                //
                // Per-iteration before (E3): 4× ucvtf scalar + 3× mov v.s[i] +
                //   1× str/ldr stack roundtrip + 1× vfmaq
                // Per-iteration after  (E5): 1× vld1_u8 + 1× vand_u8 +
                //   1× vmovl_u8 + 1× vmovl_u16 + 1× vcvtq_f32_u32 + 1× vfmaq

                // First 32 elements: low nibble, scale d1, min m1
                let mut l = 0;
                while l < 32 {
                    // Vectorized 4-nibble extract+widen+convert
                    let qs_8 = vld1_u8(qs.as_ptr().add(qs_offset + l)); // u8x8
                    let nib = vand_u8(qs_8, vdup_n_u8(0x0F)); // u8x8 low nibbles
                    let nib_u16 = vmovl_u8(nib); // u16x8
                    let nib_u32 = vmovl_u16(vget_low_u16(nib_u16)); // u32x4 (lanes 0..3)
                    let qv = vcvtq_f32_u32(nib_u32); // f32x4

                    let dq = vsubq_f32(vmulq_f32(d1_v, qv), m1_v);
                    let xv = vld1q_f32(x.as_ptr().add(x_base + j_base + l));
                    acc = vfmaq_f32(acc, xv, dq);

                    l += 4;
                }

                // Next 32 elements: high nibble, scale d2, min m2
                l = 0;
                while l < 32 {
                    let qs_8 = vld1_u8(qs.as_ptr().add(qs_offset + l)); // u8x8
                    let nib = vshr_n_u8::<4>(qs_8); // u8x8 high nibbles
                    let nib_u16 = vmovl_u8(nib); // u16x8
                    let nib_u32 = vmovl_u16(vget_low_u16(nib_u16)); // u32x4 (lanes 0..3)
                    let qv = vcvtq_f32_u32(nib_u32); // f32x4

                    let dq = vsubq_f32(vmulq_f32(d2_v, qv), m2_v);
                    let xv = vld1q_f32(x.as_ptr().add(x_base + j_base + 32 + l));
                    acc = vfmaq_f32(acc, xv, dq);

                    l += 4;
                }

                qs_offset += 32;
                is += 2;
                j_group += 1;
            }

            b += 1;
        }

        // Horizontal reduce 4-wide accumulator
        out[i] = vaddvq_f32(acc);

        i += 1;
    }
}

/// NEON-accelerated Q6_K linear projection: out[i] = dot(x, dequant(W[i]))
///
/// Q6_K block layout (210 bytes):
///   bytes   0-127:  ql[128]    quants, lower 4 bits
///   bytes 128-191:  qh[64]     quants, upper 2 bits
///   bytes 192-207:  scales[16] per-sub-block 8-bit signed scales
///   bytes 208-209:  d (FP16)   super-block scale
///
/// # Safety
/// Same requirements as linear_q4k_neon.
pub unsafe fn linear_q6k_neon(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    let blocks_per_row = in_dim / 256;
    let bytes_per_row = blocks_per_row * 210;

    let mut i = 0;
    while i < out_dim {
        let row_byte_offset = i * bytes_per_row;
        let mut acc = vdupq_n_f32(0.0);

        let mut b = 0;
        while b < blocks_per_row {
            let block_off = row_byte_offset + b * 210;
            let block = &w_blocks[block_off..block_off + 210];

            let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let ql = &block[0..128];
            let qh = &block[128..192];
            let sc = &block[192..208];

            let x_base = b * 256;

            // Two halves of 128 elements each (matches ggml pattern)
            let mut half = 0;
            while half < 2 {
                let ql_off = half * 64;
                let qh_off = half * 32;
                let sc_off = half * 8;
                let y_off = half * 128;

                // Q6_K inner loop: E3 spoon-feeding pattern PRESERVED.
                // β-anchor bit-exact verification (E5-V2): Q6_K vectorization
                // attempts (4-separate-loops, scalar accumulation) caused
                // β-anchor drift (0x415494cc, 0x42a87032) because they changed
                // FMA accumulation order vs E3's per-lane vfmaq_f32 pattern.
                // The 4 element-sets per iteration access non-contiguous x[]
                // positions (stride 32), preventing contiguous NEON loads.
                // E3's [f32;4]→stack→vld1q_f32 is the ONLY pattern that
                // preserves β-anchor bit-exact 0x414a6497. Q4_K TRUE-vec
                // provides the speedup; Q6_K stays E3-identical per Lesson 17.5.

                let mut l = 0;
                while l < 32 {
                    let is = l / 16; // 0 or 1 within half

                    // Element set 1: y_off + l (low nibble ql, qh bits 0:1)
                    let q1_val =
                        ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                    // Element set 2: y_off + l + 32 (low nibble ql+32, qh bits 2:3)
                    let q2_val = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4))
                        as i32
                        - 32;
                    // Element set 3: y_off + l + 64 (high nibble ql, qh bits 4:5)
                    let q3_val =
                        ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                    // Element set 4: y_off + l + 96 (high nibble ql+32, qh bits 6:7)
                    let q4_val = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4))
                        as i32
                        - 32;

                    let s1 = d * (sc[sc_off + is] as i8 as f32);
                    let s2 = d * (sc[sc_off + is + 2] as i8 as f32);
                    let s3 = d * (sc[sc_off + is + 4] as i8 as f32);
                    let s4 = d * (sc[sc_off + is + 6] as i8 as f32);

                    let dq1 = s1 * (q1_val as f32);
                    let dq2 = s2 * (q2_val as f32);
                    let dq3 = s3 * (q3_val as f32);
                    let dq4 = s4 * (q4_val as f32);

                    // Load 4 x-values at corresponding positions
                    let x1 = *x.as_ptr().add(x_base + y_off + l);
                    let x2 = *x.as_ptr().add(x_base + y_off + l + 32);
                    let x3 = *x.as_ptr().add(x_base + y_off + l + 64);
                    let x4 = *x.as_ptr().add(x_base + y_off + l + 96);

                    // Accumulate via NEON — gather the 4 products into a vector
                    let dq_arr: [f32; 4] = [dq1, dq2, dq3, dq4];
                    let dq_v = vld1q_f32(dq_arr.as_ptr());
                    let x_arr: [f32; 4] = [x1, x2, x3, x4];
                    let x_v = vld1q_f32(x_arr.as_ptr());
                    acc = vfmaq_f32(acc, x_v, dq_v);

                    l += 1;
                }

                half += 1;
            }

            b += 1;
        }

        out[i] = vaddvq_f32(acc);
        i += 1;
    }
}

/// NEON-accelerated Q4_0 linear projection: out[i] = dot(x, dequant(W[i])).
///
/// Q4_0 block layout (18 bytes):
///   bytes 0-1:  d (FP16, super-block scale)
///   bytes 2-17: qs[16] — 32 × 4-bit nibbles
///     byte j: low nibble → element j, high nibble → element j + 16
///     Each nibble is recentered: (nibble & 0x0F) - 8 → signed i4 ∈ [-8, 7]
///
/// Dequant per element:
///   output[k] = ((qs_nibble - 8) as f32) * d
///
/// Used by the DeepSeek-V2/V3 / Kimi K2.6 forward pass for bulk
/// expert / MLA / embed weights (Kimi K2.6 ships natively in Q4_0).
///
/// # Safety
/// Same requirements as `linear_q4k_neon`. in_dim % 32 == 0.
pub unsafe fn linear_q4_0_neon(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    let blocks_per_row = in_dim / 32;
    let bytes_per_row = blocks_per_row * 18;

    let mut i = 0;
    while i < out_dim {
        let row_byte_offset = i * bytes_per_row;
        let mut acc = vdupq_n_f32(0.0);

        let mut b = 0;
        while b < blocks_per_row {
            let block_off = row_byte_offset + b * 18;
            let block = &w_blocks[block_off..block_off + 18];

            let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qs = &block[2..18];
            let d_v = vdupq_n_f32(d);
            let bias_v = vdupq_n_s32(-8);

            let x_base = b * 32;

            // Low nibbles → first 16 elements (bytes j=0..15)
            let mut l = 0;
            while l < 16 {
                let qs_8 = vld1_u8(qs.as_ptr().add(l)); // u8x8
                let nib = vand_u8(qs_8, vdup_n_u8(0x0F)); // low nibbles 0..15
                let nib_u16 = vmovl_u8(nib); // u16x8

                // Lower 4 lanes → first NEON vector
                let nib_lo_u32 = vmovl_u16(vget_low_u16(nib_u16));
                let nib_lo_i32 = vaddq_s32(vreinterpretq_s32_u32(nib_lo_u32), bias_v);
                let qv_lo = vcvtq_f32_s32(nib_lo_i32);
                let dq_lo = vmulq_f32(d_v, qv_lo);
                let xv_lo = vld1q_f32(x.as_ptr().add(x_base + l));
                acc = vfmaq_f32(acc, xv_lo, dq_lo);

                // Upper 4 lanes → second NEON vector
                let nib_hi_u32 = vmovl_u16(vget_high_u16(nib_u16));
                let nib_hi_i32 = vaddq_s32(vreinterpretq_s32_u32(nib_hi_u32), bias_v);
                let qv_hi = vcvtq_f32_s32(nib_hi_i32);
                let dq_hi = vmulq_f32(d_v, qv_hi);
                let xv_hi = vld1q_f32(x.as_ptr().add(x_base + l + 4));
                acc = vfmaq_f32(acc, xv_hi, dq_hi);

                l += 8;
            }

            // High nibbles → second 16 elements (bytes j=0..15 → out[16..32])
            l = 0;
            while l < 16 {
                let qs_8 = vld1_u8(qs.as_ptr().add(l));
                let nib = vshr_n_u8::<4>(qs_8);
                let nib_u16 = vmovl_u8(nib);

                let nib_lo_u32 = vmovl_u16(vget_low_u16(nib_u16));
                let nib_lo_i32 = vaddq_s32(vreinterpretq_s32_u32(nib_lo_u32), bias_v);
                let qv_lo = vcvtq_f32_s32(nib_lo_i32);
                let dq_lo = vmulq_f32(d_v, qv_lo);
                let xv_lo = vld1q_f32(x.as_ptr().add(x_base + 16 + l));
                acc = vfmaq_f32(acc, xv_lo, dq_lo);

                let nib_hi_u32 = vmovl_u16(vget_high_u16(nib_u16));
                let nib_hi_i32 = vaddq_s32(vreinterpretq_s32_u32(nib_hi_u32), bias_v);
                let qv_hi = vcvtq_f32_s32(nib_hi_i32);
                let dq_hi = vmulq_f32(d_v, qv_hi);
                let xv_hi = vld1q_f32(x.as_ptr().add(x_base + 16 + l + 4));
                acc = vfmaq_f32(acc, xv_hi, dq_hi);

                l += 8;
            }

            b += 1;
        }

        out[i] = vaddvq_f32(acc);
        i += 1;
    }
}

/// NEON-accelerated Q8_0 linear projection: out[i] = dot(x, dequant(W[i])).
///
/// Q8_0 block layout (34 bytes):
///   bytes 0-1:  d (FP16, super-block scale)
///   bytes 2-33: qs[32] — 32 × signed int8 values
///
/// Dequant per element:
///   output[k] = (qs[k] as i8 as f32) * d
///
/// Used by the DeepSeek-V2/V3 / Kimi K2.6 forward pass for the
/// embed table (`token_embd.weight`) and LM head (`output.weight`)
/// — bartowski's Q4_0 GGUFs keep these in Q8_0 for accuracy.
///
/// # Safety
/// Same requirements as `linear_q4k_neon`. in_dim % 32 == 0.
pub unsafe fn linear_q8_0_neon(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    let blocks_per_row = in_dim / 32;
    let bytes_per_row = blocks_per_row * 34;

    let mut i = 0;
    while i < out_dim {
        let row_byte_offset = i * bytes_per_row;
        let mut acc = vdupq_n_f32(0.0);

        let mut b = 0;
        while b < blocks_per_row {
            let block_off = row_byte_offset + b * 34;
            let block = &w_blocks[block_off..block_off + 34];

            let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qs = &block[2..34];
            let d_v = vdupq_n_f32(d);
            let x_base = b * 32;

            // Process 32 i8 values in 4-element NEON chunks.
            // Load 16 i8s at a time via vld1q_s8, widen to s16 then s32,
            // convert to f32, multiply by d, FMA against x.
            let mut k_off = 0;
            while k_off < 32 {
                let qs_16 = vld1q_s8(qs.as_ptr().add(k_off) as *const i8); // i8x16

                let qs_lo_i16 = vmovl_s8(vget_low_s8(qs_16)); // i16x8 (lanes 0..7)
                let qs_hi_i16 = vmovl_s8(vget_high_s8(qs_16)); // i16x8 (lanes 8..15)

                // Lanes 0..3
                let q_i32_0 = vmovl_s16(vget_low_s16(qs_lo_i16));
                let qv0 = vcvtq_f32_s32(q_i32_0);
                let dq0 = vmulq_f32(d_v, qv0);
                let xv0 = vld1q_f32(x.as_ptr().add(x_base + k_off));
                acc = vfmaq_f32(acc, xv0, dq0);

                // Lanes 4..7
                let q_i32_1 = vmovl_s16(vget_high_s16(qs_lo_i16));
                let qv1 = vcvtq_f32_s32(q_i32_1);
                let dq1 = vmulq_f32(d_v, qv1);
                let xv1 = vld1q_f32(x.as_ptr().add(x_base + k_off + 4));
                acc = vfmaq_f32(acc, xv1, dq1);

                // Lanes 8..11
                let q_i32_2 = vmovl_s16(vget_low_s16(qs_hi_i16));
                let qv2 = vcvtq_f32_s32(q_i32_2);
                let dq2 = vmulq_f32(d_v, qv2);
                let xv2 = vld1q_f32(x.as_ptr().add(x_base + k_off + 8));
                acc = vfmaq_f32(acc, xv2, dq2);

                // Lanes 12..15
                let q_i32_3 = vmovl_s16(vget_high_s16(qs_hi_i16));
                let qv3 = vcvtq_f32_s32(q_i32_3);
                let dq3 = vmulq_f32(d_v, qv3);
                let xv3 = vld1q_f32(x.as_ptr().add(x_base + k_off + 12));
                acc = vfmaq_f32(acc, xv3, dq3);

                k_off += 16;
            }

            b += 1;
        }

        out[i] = vaddvq_f32(acc);
        i += 1;
    }
}

/// NEON-accelerated f32 dot-product: result = Σ a[i] * b[i].
///
/// Sub-MP-E3: Used for attention score-computation (Q × Kᵀ per head).
/// FMA accumulation pattern matches matmul-floor scope per E2-Q2.
///
/// # Safety
/// Uses ARM NEON intrinsics. a.len() == b.len() required.
/// Handles non-multiple-of-4 lengths via scalar tail.
#[inline]
pub unsafe fn dot_product_f32_neon(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = vdupq_n_f32(0.0);

    let chunks = len / 4;
    let mut i = 0;
    while i < chunks {
        let off = i * 4;
        let av = vld1q_f32(a.as_ptr().add(off));
        let bv = vld1q_f32(b.as_ptr().add(off));
        acc = vfmaq_f32(acc, av, bv);
        i += 1;
    }

    let mut result = vaddvq_f32(acc);

    // Scalar tail for remainder
    let tail_start = chunks * 4;
    let mut j = tail_start;
    while j < len {
        result += a[j] * b[j];
        j += 1;
    }

    result
}

/// NEON-accelerated weighted accumulation: output[i] += weight * vec[i].
///
/// Sub-MP-E3: Used for attention weighted-sum (attn_score × V per token).
/// FMA accumulation pattern matches matmul-floor scope per E2-Q2.
///
/// # Safety
/// Uses ARM NEON intrinsics. vec and output must have len >= head_dim.
#[inline]
pub unsafe fn weighted_add_f32_neon(weight: f32, vec: &[f32], output: &mut [f32], head_dim: usize) {
    let wv = vdupq_n_f32(weight);

    let chunks = head_dim / 4;
    let mut i = 0;
    while i < chunks {
        let off = i * 4;
        let vv = vld1q_f32(vec.as_ptr().add(off));
        let ov = vld1q_f32(output.as_ptr().add(off));
        let res = vfmaq_f32(ov, wv, vv);
        vst1q_f32(output.as_mut_ptr().add(off), res);
        i += 1;
    }

    // Scalar tail
    let tail_start = chunks * 4;
    let mut j = tail_start;
    while j < head_dim {
        output[j] += weight * vec[j];
        j += 1;
    }
}
