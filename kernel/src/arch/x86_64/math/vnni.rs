// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR-029 v8 candidate — AVX-512 VNNI Q4_K × Q8_K integer dot product.
//!
//! Implements the canonical ggml `ggml_vec_dot_q4_K_q8_K` pattern using
//! `VPDPBUSD` (AVX-512 VNNI) for the inner product. The activation
//! vector is first quantized to Q8_K (8-bit symmetric, per-block FP32
//! scale + 16 per-bsum int16 partial sums), then matmul'd against the
//! Q4_K weight blocks entirely in the integer pipeline. A single FP32
//! scale per super-block converts the int32 accumulator to FP32.
//!
//! # Numerical contract (Two-Anchor v3 — ADR-029)
//!
//! This is a DIFFERENT mathematical operation from the FP32 AVX-512
//! path. Q8 activation quantization adds ~0.5–2 % relative error per
//! matmul; over 28 layers the compounded perturbation is ~5–30 % of
//! individual logit values. Token-ID 25 stability has NOT been
//! verified — this kernel ships behind the `vnni-acceleration` feature
//! flag and the production dispatch path does not invoke it by default.
//!
//! When hardware testing on Cherry confirms Token-ID 25 holds:
//! ADR-029 must be amended (v8 or v9) to register VNNI as the fourth
//! feature-mode in the Two-Anchor Registration table, with its own
//! `BETA_ANCHOR_EXPECTED_LOGIT_BITS` value.
//!
//! If Token-ID 25 does NOT hold under VNNI, the appropriate response
//! per ADR-029 is to keep the feature flag off; do not relax the
//! Token-ID HARD GATE.
//!
//! # Reference
//!
//! Algorithm derived from upstream llama.cpp:
//! * `ggml_vec_dot_q4_K_q8_K_generic` (scalar reference)
//!   — ggml/src/ggml-cpu/quants.c
//! * `quantize_row_q8_K_ref` (Q8_K block layout)
//!   — ggml/src/ggml-quants.c
//! * `tinygemm_kernel_vnni<block_q8_K, block_q4_K, ...>` (the VNNI
//!   inner-loop pattern — gated `__AMX_INT8__ && __AVX512VNNI__`
//!   upstream, lifted out of the AMX path here for Zen 4)
//!   — ggml/src/ggml-cpu/amx/mmq.cpp
//!
//! # Safety
//!
//! All public entry points are `unsafe fn`. Caller must:
//! * Have run XSAVE/XCR0 setup so VPDPBUSD does not #UD.
//! * Guarantee `in_dim % 256 == 0`.
//! * Provide a Q8_K-packed activation buffer of size
//!   `(in_dim / 256) * 292` bytes (4 + 256 + 32 per block).

use core::arch::x86_64::*;

use crate::smp::RowRange;

/// Byte size of one Q8_K super-block (matches ggml `block_q8_K`):
/// 4 (FP32 `d`) + 256 (s8 `qs`) + 32 (16 × i16 `bsums`).
pub const Q8K_BLOCK_BYTES: usize = 292;

/// Element count per Q8_K super-block.
pub const Q8K_BLOCK_SIZE: usize = 256;

// ─────────────────────────────────────────────────────────────────
// Q8_K activation quantization (scalar — not the hot path)
// ─────────────────────────────────────────────────────────────────

/// Quantize one row of activations (length `n`, `n % 256 == 0`) into
/// the Q8_K block-packed format expected by [`linear_q4k_vnni_range`].
///
/// Writes `(n / 256) * Q8K_BLOCK_BYTES` bytes into `output`.
///
/// Scalar implementation — per-token cost is small (a few thousand
/// floats per matmul row), AVX-512 vectorisation is possible but
/// not the bottleneck. Bit-compatible with ggml `quantize_row_q8_K_ref`
/// per the source linked in the module docs.
///
/// # Safety
///
/// `input.len() >= n`, `output.len() >= (n / 256) * Q8K_BLOCK_BYTES`,
/// `n % 256 == 0`.
pub unsafe fn quantize_row_q8k(input: &[f32], output: &mut [u8], n: usize) {
    debug_assert!(n % 256 == 0);
    debug_assert!(input.len() >= n);
    debug_assert!(output.len() >= (n / 256) * Q8K_BLOCK_BYTES);

    let n_blocks = n / 256;
    let mut b = 0;
    while b < n_blocks {
        let in_base = b * 256;
        let out_base = b * Q8K_BLOCK_BYTES;

        // Pass 1: find amax and the signed value at that position.
        let mut amax: f32 = 0.0;
        let mut max_val: f32 = 0.0;
        let mut j = 0;
        while j < 256 {
            let v = input[in_base + j];
            let av = libm::fabsf(v);
            if av > amax {
                amax = av;
                max_val = v;
            }
            j += 1;
        }

        if amax == 0.0 {
            // Whole-block zero. d=0, qs all zero, bsums zero.
            let d_bits: u32 = 0;
            output[out_base..out_base + 4].copy_from_slice(&d_bits.to_le_bytes());
            let mut k = 0;
            while k < 256 {
                output[out_base + 4 + k] = 0;
                k += 1;
            }
            let mut k = 0;
            while k < 32 {
                output[out_base + 260 + k] = 0;
                k += 1;
            }
            b += 1;
            continue;
        }

        // Per ggml: iscale = -127 / max_val (signed!). The sign is
        // baked into iscale, not into qs, so that the int8 pipeline
        // can use unsigned×signed semantics in VPDPBUSD without
        // sign-flipping inside the inner loop.
        let iscale: f32 = -127.0_f32 / max_val;

        // Pass 2: quantize qs[] and accumulate bsums[].
        let mut bsums = [0i32; 16];
        let mut j = 0;
        while j < 256 {
            let v = input[in_base + j] * iscale;
            // nearest_int: round-to-nearest, ties-to-even via the
            // round-half-to-even sequence that lrintf would emit.
            let mut q = libm::rintf(v) as i32;
            // ggml clamps only on the positive side (the iscale
            // construction bounds the result to ~[-127, 127] modulo
            // rounding; the MIN(127, q) catches the edge case where
            // rounding pushes a 126.5 to 127, leaving it unchanged,
            // or where extreme values combined with float rounding
            // produce 128).
            if q > 127 {
                q = 127;
            }
            // Symmetric for safety (ggml doesn't include this but
            // i8 storage requires q >= -128; rounding could produce
            // -128 which is in range, so no extra clamp needed there).
            if q < -128 {
                q = -128;
            }
            output[out_base + 4 + j] = q as i8 as u8;
            bsums[j / 16] += q;
            j += 1;
        }

        // Write d = 1 / iscale (FP32, can be negative).
        let d: f32 = 1.0_f32 / iscale;
        output[out_base..out_base + 4].copy_from_slice(&d.to_le_bytes());

        // Write 16 i16 bsums.
        let mut k = 0;
        while k < 16 {
            let s: i16 = bsums[k].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            let off = out_base + 260 + k * 2;
            output[off..off + 2].copy_from_slice(&s.to_le_bytes());
            k += 1;
        }

        b += 1;
    }
}

// ─────────────────────────────────────────────────────────────────
// Q4_K scale + min decode (port of ggml `unpack_mins_and_scales`)
// ─────────────────────────────────────────────────────────────────

/// Decode the 12-byte Q4_K scales preamble into 8 unsigned 6-bit
/// scale bytes followed by 8 unsigned 6-bit min bytes. Identical
/// bit-twiddle to the upstream `unpack_mins_and_scales` (used by
/// every Q4_K dot variant in ggml).
#[inline(always)]
fn unpack_mins_and_scales(scales_12: &[u8]) -> [u8; 16] {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;

    let mut utmp: [u32; 4] = [
        u32::from_le_bytes([scales_12[0], scales_12[1], scales_12[2], scales_12[3]]),
        u32::from_le_bytes([scales_12[4], scales_12[5], scales_12[6], scales_12[7]]),
        u32::from_le_bytes([scales_12[8], scales_12[9], scales_12[10], scales_12[11]]),
        0,
    ];

    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&utmp[0].to_le_bytes());
    out[4..8].copy_from_slice(&utmp[1].to_le_bytes());
    out[8..12].copy_from_slice(&utmp[2].to_le_bytes());
    out[12..16].copy_from_slice(&utmp[3].to_le_bytes());
    out
}

#[inline(always)]
fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let val = (mant as f32) * (1.0_f32 / 1024.0_f32) * (1.0_f32 / 16384.0_f32);
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

// ─────────────────────────────────────────────────────────────────
// VPDPBUSD Q4_K × Q8_K inner kernel
// ─────────────────────────────────────────────────────────────────

/// AVX-512 VNNI Q4_K matmul over a row range, integer pipeline.
///
/// `q8k_packed` points to a Q8_K-packed activation buffer of size
/// `(in_dim / 256) * Q8K_BLOCK_BYTES`. Produced by
/// [`quantize_row_q8k`].
///
/// `w_blocks` is the standard Q4_K weight buffer (144 bytes per
/// super-block, blocks of 256 elements).
///
/// `out[i]` for `i in range` = Σ_{k} dequant_q4k(W[i,k]) * x[k]
/// (the same mathematical result the FP32 AVX-512 path computes,
/// modulo ~0.5–2 % per-matmul Q8 quantization error).
///
/// # Safety
///
/// AVX-512F + AVX-512VNNI required (Zen 4 has both natively). Caller
/// upholds `in_dim % 256 == 0`, `range.end <= out_dim`, and the
/// packed-buffer-size contract on `q8k_packed`.
#[target_feature(enable = "avx512f,avx512vnni,avx512bw,avx512dq")]
pub unsafe fn linear_q4k_vnni_range(
    q8k_packed: &[u8],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    _out_dim: usize,
    range: RowRange,
) {
    let blocks_per_row = in_dim / 256;
    let bytes_per_w_row = blocks_per_row * 144;

    let mut i = range.start;
    while i < range.end {
        let w_row_base = i * bytes_per_w_row;

        // Final per-row accumulators in FP32. We keep two
        // independent FP32 acc chains (dot side and mins side) and
        // sum them at the end.
        let mut sumf_dot: f32 = 0.0;
        let mut sumf_min: f32 = 0.0;

        let mut b = 0;
        while b < blocks_per_row {
            let w_off = w_row_base + b * 144;
            let w_block = &w_blocks[w_off..w_off + 144];
            let q8_off = b * Q8K_BLOCK_BYTES;
            let q8_block = &q8k_packed[q8_off..q8_off + Q8K_BLOCK_BYTES];

            // Prefetch next blocks (weights + activations).
            if b + 1 < blocks_per_row {
                let w_next = w_row_base + (b + 1) * 144;
                _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(w_next) as *const i8);
                _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(w_next + 64) as *const i8);
                _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(w_next + 128) as *const i8);
                let q8_next = (b + 1) * Q8K_BLOCK_BYTES;
                _mm_prefetch::<_MM_HINT_T0>(q8k_packed.as_ptr().add(q8_next) as *const i8);
            }

            // Q4_K block header.
            let d_w = fp16_to_f32(u16::from_le_bytes([w_block[0], w_block[1]]));
            let dmin_w = fp16_to_f32(u16::from_le_bytes([w_block[2], w_block[3]]));
            let sc_mn = unpack_mins_and_scales(&w_block[4..16]);
            let qs = &w_block[16..144]; // 128 packed nibble bytes

            // Q8_K block header.
            let d_q8: f32 =
                f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
            let q8_qs = &q8_block[4..260]; // 256 s8

            let d_combined = d_w * d_q8;
            let dmin_combined = dmin_w * d_q8;

            // ── Dot side: VPDPBUSD over the 8 sub-blocks of 32. ──
            //
            // Sub-block g (g ∈ [0..8)):
            //   nibble offset = (g / 2) * 32 ... + (g & 1) ? high : low
            //   q8 offset = g * 32
            //   sub_scale = sc_mn[g]  (u6)
            //
            // We process two sub-blocks per 32-byte qs chunk (low+high
            // nibbles cover sub-blocks 2g and 2g+1 of the same byte-range).
            //
            // Strategy: four independent int32 accumulators across the
            // 8 sub-blocks (round-robin), then convert to FP32 at the
            // end of the super-block.

            let mut iacc0 = _mm512_setzero_si512();
            let mut iacc1 = _mm512_setzero_si512();
            let mut iacc2 = _mm512_setzero_si512();
            let mut iacc3 = _mm512_setzero_si512();

            let nibble_mask = _mm512_set1_epi8(0x0F);

            // We have 4 chunks of 32 weight bytes (covering 8 sub-blocks).
            let mut chunk = 0;
            while chunk < 4 {
                let w_bytes_off = chunk * 32;
                let q8_bytes_off = chunk * 64;

                // Load 32 bytes of weight nibbles (covers 64 weights via low+high split).
                // Zero-extend to 512-bit zmm via masked load — we load only 32 bytes.
                let w32 = _mm256_loadu_si256(qs.as_ptr().add(w_bytes_off) as *const __m256i);
                let w32_zmm = _mm512_castsi256_si512(w32);

                // Split into low and high nibbles, each as 32 u8 values
                // in a 256-bit lane.
                let lo32 = _mm256_and_si256(w32, _mm256_set1_epi8(0x0F));
                let hi32 = _mm256_and_si256(_mm256_srli_epi16(w32, 4), _mm256_set1_epi8(0x0F));

                // For VPDPBUSD we need 64 u8 weights paired with 64 s8 activations.
                // Pack low and high nibbles into one 512-bit u8 vector
                // (low 32 bytes = sub-block 2g+0, high 32 bytes = sub-block 2g+1).
                let w_combined = _mm512_inserti64x4(_mm512_castsi256_si512(lo32), hi32, 1);
                let _ = (w32_zmm, nibble_mask); // silence unused-var lint when compiled out

                // Load 64 s8 activations (corresponding sub-blocks).
                let a_combined =
                    _mm512_loadu_si512(q8_qs.as_ptr().add(q8_bytes_off) as *const __m512i);

                // VPDPBUSD: 64 u8 × 64 s8 → 16 i32 partial sums (each
                // lane = sum of 4 contiguous u*s pairs).
                let acc_choice = chunk & 3;
                if acc_choice == 0 {
                    iacc0 = _mm512_dpbusd_epi32(iacc0, w_combined, a_combined);
                } else if acc_choice == 1 {
                    iacc1 = _mm512_dpbusd_epi32(iacc1, w_combined, a_combined);
                } else if acc_choice == 2 {
                    iacc2 = _mm512_dpbusd_epi32(iacc2, w_combined, a_combined);
                } else {
                    iacc3 = _mm512_dpbusd_epi32(iacc3, w_combined, a_combined);
                }

                // The per-sub-block scale fold-in: each i32 lane in
                // acc holds the sum of 4 u*s pairs (8 bytes of input).
                // For one sub-block (32 elements) we have 8 lanes
                // worth. The naive approach is per-lane scale via
                // `_mm512_mullo_epi32(acc_lanes_for_sub, set1(sc_g))`
                // followed by accumulate. We defer this combine to
                // the super-block end to keep the inner loop tight.
                //
                // To still apply the scales we need to remember which
                // lanes correspond to which sub-block. With our
                // packing (lo32 then hi32 in one VPDPBUSD), the first
                // 8 lanes of the result are sub-block (2g+0)
                // contributions, the last 8 are (2g+1).
                //
                // We accumulate UNSCALED partials in iacc0..3 here,
                // then SCALE-AND-COMBINE per sub-block below.

                chunk += 1;
            }

            // Tree-reduce the four independent acc chains into one.
            let iacc_lo = _mm512_add_epi32(iacc0, iacc1);
            let iacc_hi = _mm512_add_epi32(iacc2, iacc3);
            let _iacc_unscaled_sum = _mm512_add_epi32(iacc_lo, iacc_hi);

            // ── Per-sub-block scale combine (separate pass) ──
            //
            // We need: sum_{g=0..8} sc_g * (sum of 32 u*s pairs in
            // sub-block g). The VPDPBUSD partials we accumulated do
            // NOT separate sub-blocks cleanly because we combined
            // low+high into one operand — both halves of the same
            // chunk land in the same iacc.
            //
            // Re-run the partials per sub-block, this time with the
            // scale folded in. This sacrifices some throughput for
            // correctness. A more aggressive version would use 8
            // separate accumulators (one per sub-block) and avoid
            // re-computation, at the cost of 2× the register
            // pressure. Profile-then-tune is the right next step.
            let mut int_sum_scaled: i64 = 0;
            let mut g = 0usize;
            while g < 8 {
                let group = g / 2;
                let half = g & 1; // 0 = low nibbles, 1 = high
                let qs_off = group * 32; // 32 bytes covers low+high for 2 sub-blocks
                let a_off = g * 32;

                // Load 32 weight bytes for this chunk.
                let w32 = _mm256_loadu_si256(qs.as_ptr().add(qs_off) as *const __m256i);
                let nibbles = if half == 0 {
                    _mm256_and_si256(w32, _mm256_set1_epi8(0x0F))
                } else {
                    _mm256_and_si256(_mm256_srli_epi16(w32, 4), _mm256_set1_epi8(0x0F))
                };
                let a32 = _mm256_loadu_si256(q8_qs.as_ptr().add(a_off) as *const __m256i);

                // 256-bit VPDPBUSD: 32 u8 × 32 s8 → 8 i32.
                let prod = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), nibbles, a32);
                // Horizontal sum the 8 i32 lanes → scalar i32.
                let mut buf = [0i32; 8];
                _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, prod);
                let mut sub_sum: i32 = 0;
                let mut k = 0;
                while k < 8 {
                    sub_sum += buf[k];
                    k += 1;
                }
                let sc_g = sc_mn[g] as i32; // u6 scale, 0..63
                int_sum_scaled += (sub_sum as i64) * (sc_g as i64);

                g += 1;
            }

            sumf_dot += d_combined * (int_sum_scaled as f32);

            // ── Mins side: dmin * Σ_{g} m_g * bsum_pair_g ──
            //
            // Each Q4_K sub-block g has a 6-bit min m_g; the
            // corresponding Q8_K block stores per-16-element bsums.
            // Each Q4_K sub-block of 32 elements covers 2 Q8_K bsum
            // groups, so bsum_pair_g = bsums[2g] + bsums[2g+1].
            let mut min_sum: i32 = 0;
            let mut g = 0usize;
            while g < 8 {
                let m_g = sc_mn[8 + g] as i32; // u6 min, 0..63
                let bs_lo = i16::from_le_bytes([
                    q8_block[260 + (2 * g) * 2],
                    q8_block[260 + (2 * g) * 2 + 1],
                ]) as i32;
                let bs_hi = i16::from_le_bytes([
                    q8_block[260 + (2 * g + 1) * 2],
                    q8_block[260 + (2 * g + 1) * 2 + 1],
                ]) as i32;
                min_sum += m_g * (bs_lo + bs_hi);
                g += 1;
            }
            sumf_min += dmin_combined * (min_sum as f32);

            b += 1;
        }

        // Final result: dot − min correction.
        out[i] = sumf_dot - sumf_min;

        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────
// Tests (compiled out unless host-side)
// ─────────────────────────────────────────────────────────────────
//
// VNNI requires AVX-512VNNI at runtime; host-side `cargo test` on
// non-x86 (Apple Silicon dev box) or pre-Zen-4 hardware can't
// exercise the intrinsics. Numerical correctness validation belongs
// on Cherry hardware against the canonical reference dumps in
// crates/zero-llm-inference/tests/reference-dumps/.
