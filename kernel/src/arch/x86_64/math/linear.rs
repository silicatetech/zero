// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR-029 Phase 1+2 — AVX-512 fused dequant + linear projection.
//!
//! Counterpart to `arch::aarch64::math::linear` (NEON). The structural
//! invariants are identical:
//!
//! * **Row-by-row matmul** per ADR-029 D8.
//! * **Per-row 16-wide reduction** via four independent `__m512`
//!   accumulators. 16 lanes exactly match the 16-element-aligned f32
//!   stride inside a Q4_K / Q6_K block, so the inner loop is
//!   branch-free; four independent chains let Zen 4 hide the 4-cycle
//!   FMA latency by issuing one FMA per cycle on each chain.
//! * **`prefetchnta` for the next weight block** — weights are streamed
//!   once per token and never reused within a single decode pass.
//!   Marking the prefetch non-temporal lets the cache controller
//!   bypass L2/L3 retention for them, leaving those caches free for
//!   activations and the KV cache.
//! * **Range-limited dispatch** (`out_start`/`out_end`) so the parallel
//!   matmul dispatcher in `crate::smp` can hand each core a contiguous
//!   row slice — exactly the same K-order as the single-threaded path,
//!   which preserves the row-ownership invariant (each output row's
//!   reduction is owned by exactly one core). The four-accumulator
//!   final tree-reduce changes the FP rounding tree by ≤ 1 ULP vs
//!   the single-accumulator linear chain; ADR-029 v3 Two-Anchor
//!   registers AVX-512 as a feature-mode that may drift in
//!   `logit_bits` but must hold Token-ID 25 strictly.
//!
//! # Bit-exactness with the scalar/NEON paths
//!
//! ADR-029 v3 (Two-Anchor Registration) ratified that the *default
//! mode* β-Anchor (`token=25`, `logit_bits=0x414a6497`) is sacred and
//! enforced as a hard gate, while *feature-mode* (NEON/AVX-512) is
//! permitted to drift within 1 ULP as a documented LLVM register
//! allocation sensitivity. We preserve the Anchor 1 strict invariant
//! for the scalar path and accept the Anchor 2 relaxed invariant for
//! the AVX-512 path — the FMA accumulation order is the same as NEON
//! (per-row contiguous K-stride), so token-ID 25 must hold; the
//! `logit_bits` may differ from NEON's value by ≤ 1 ULP but must be
//! constant across SMP modes.
//!
//! # Safety
//!
//! All public entry points are `unsafe fn`. Caller must guarantee:
//! * `target_feature = "avx512f"` is enabled at the call site (we
//!   enable it inside via `#[target_feature]` attributes).
//! * Pointer aliasing is disjoint per the `crate::smp` contract — each
//!   core writes only `out[out_start..out_end]`.
//! * `in_dim % 256 == 0` (Q-block alignment).
//!
//! CITE: Intel Intrinsics Guide §AVX512F
//! CITE: ADR-029 D8 (Matmul implementation strategy), v3 (Two-Anchor)
//! CITE: zero-gguf-parser::dequant (Q4_K / Q6_K format reference)

use core::arch::x86_64::*;

use crate::smp::RowRange;

// ─────────────────────────────────────────────────────────────────
// FP16 → F32 (replicated from sacred parser, Pillar 7 boundary)
// ─────────────────────────────────────────────────────────────────

/// IEEE 754 binary16 → binary32. Same algorithm as the aarch64 mirror
/// in `arch::aarch64::math::linear::fp16_to_f32`. Replicated rather
/// than imported from the sacred parser to keep the HAL self-contained.
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

/// Decode Q4_K sub-block scale and min from the 12-byte `scales`
/// preamble. See `zero-gguf-parser` for the canonical reference.
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

// ─────────────────────────────────────────────────────────────────
// linear_q4k_avx512 — full output range
// ─────────────────────────────────────────────────────────────────

/// AVX-512-accelerated Q4_K matmul. Computes the full output row range
/// `[0, out_dim)`. Single-threaded entry point — equivalent to calling
/// [`linear_q4k_avx512_range`] with `RowRange { start: 0, end: out_dim }`.
///
/// # Safety
/// See module docs. `target_feature = "avx512f"` is enabled inside.
#[target_feature(enable = "avx512f")]
pub unsafe fn linear_q4k_avx512(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    linear_q4k_avx512_range(
        x,
        w_blocks,
        out,
        in_dim,
        out_dim,
        RowRange {
            start: 0,
            end: out_dim,
        },
    );
}

/// AVX-512-accelerated Q4_K matmul over a *row range*. Used by the
/// parallel matmul dispatcher (`crate::smp::ParallelMatmulContext`) to
/// hand each core a disjoint slice of output rows. The reduction order
/// inside each row is identical to [`linear_q4k_avx512`], so bit-exact
/// across SMP modes.
///
/// # Safety
/// See module docs. Caller must guarantee `range.end <= out_dim` and
/// pointer slices have sufficient length.
#[target_feature(enable = "avx512f")]
pub unsafe fn linear_q4k_avx512_range(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    _out_dim: usize,
    range: RowRange,
) {
    let blocks_per_row = in_dim / 256;
    let bytes_per_row = blocks_per_row * 144;

    let mut i = range.start;
    while i < range.end {
        let row_byte_offset = i * bytes_per_row;

        // Four independent 16-wide accumulators. Zen 4 FMA: 4-cycle
        // latency, 1-cycle reciprocal throughput. A single accumulator
        // chain serialises at 0.25 FMA/cyc; four chains saturate at
        // ~1 FMA/cyc. Each Q4_K block produces 8 FMAs (2 per j-group
        // × 4 j-groups), so we round-robin them across the 4 accs.
        // Final reduction is a balanced tree, which preserves
        // associativity-class invariants at ≤ 1 ULP per row.
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        let mut b = 0;
        while b < blocks_per_row {
            let block_off = row_byte_offset + b * 144;
            let block = &w_blocks[block_off..block_off + 144];

            // Software prefetch the NEXT block — feature-gated.
            // Q4_K block = 144 B ≈ 2.25 cache lines. NTA hint asks
            // the cache controller to bypass L2/L3.
            //
            // **Default OFF (no nta-prefetch feature).** On Zen 4
            // with 4 physical cores per CCD sharing a 32 MiB L3,
            // sibling cores running adjacent row slices benefit from
            // L3 reuse that NTA bypasses. The L2 streamer reportedly
            // saturates per-core BW without SW prefetch. Enable via
            // --features nta-prefetch to A/B test against the
            // hardware prefetcher's natural behaviour.
            #[cfg(feature = "nta-prefetch")]
            if b + 1 < blocks_per_row {
                let next_off = row_byte_offset + (b + 1) * 144;
                _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(next_off) as *const i8);
                _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(next_off + 64) as *const i8);
                _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(next_off + 128) as *const i8);
            }

            // Block header
            let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales = &block[4..16];
            let qs = &block[16..144];

            let x_base = b * 256;

            // 4 groups × 64 elements per Q4_K block.
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

                let d1_v = _mm512_set1_ps(d1);
                let m1_v = _mm512_set1_ps(m1);
                let d2_v = _mm512_set1_ps(d2);
                let m2_v = _mm512_set1_ps(m2);

                // First 32 elements: low nibble of qs[qs_offset+l..+l+16],
                // scale d1. Q4_K sub-block layout: element j_base+k uses
                // the LOW nibble of byte qs[qs_offset+k] for k in 0..32
                // (see crates/zero-gguf-parser/src/dequant.rs:126).
                // We process 16 elements per iteration → 16 separate
                // bytes, byte offset == element offset.
                //
                // Round-robin across acc0..acc3 (low-l=0→acc0,
                // low-l=16→acc1, high-l=0→acc2, high-l=16→acc3) to
                // keep each FMA chain independent.
                {
                    // l = 0 — accumulates into acc0
                    let q_ptr = qs.as_ptr().add(qs_offset);
                    let bytes_i64 = _mm_loadu_si128(q_ptr as *const __m128i);
                    let nibs_u32 = _mm512_cvtepu8_epi32(bytes_i64);
                    let lo = _mm512_and_epi32(nibs_u32, _mm512_set1_epi32(0x0F));
                    let qv = _mm512_cvtepi32_ps(lo);
                    let dq = _mm512_fmsub_ps(d1_v, qv, m1_v);
                    let xv = _mm512_loadu_ps(x.as_ptr().add(x_base + j_base));
                    acc0 = _mm512_fmadd_ps(xv, dq, acc0);

                    // l = 16 — accumulates into acc1
                    let q_ptr = qs.as_ptr().add(qs_offset + 16);
                    let bytes_i64 = _mm_loadu_si128(q_ptr as *const __m128i);
                    let nibs_u32 = _mm512_cvtepu8_epi32(bytes_i64);
                    let lo = _mm512_and_epi32(nibs_u32, _mm512_set1_epi32(0x0F));
                    let qv = _mm512_cvtepi32_ps(lo);
                    let dq = _mm512_fmsub_ps(d1_v, qv, m1_v);
                    let xv = _mm512_loadu_ps(x.as_ptr().add(x_base + j_base + 16));
                    acc1 = _mm512_fmadd_ps(xv, dq, acc1);
                }

                // Next 32 elements: HIGH nibble of bytes
                // qs[qs_offset+l..+l+16], scale d2. Element j_base+32+k
                // uses the high nibble of byte qs[qs_offset+k] — same
                // byte offset as the low-nibble loop, different shift.
                {
                    // l = 0 — accumulates into acc2
                    let q_ptr = qs.as_ptr().add(qs_offset);
                    let bytes_i64 = _mm_loadu_si128(q_ptr as *const __m128i);
                    let nibs_u32 = _mm512_cvtepu8_epi32(bytes_i64);
                    let hi = _mm512_srli_epi32::<4>(nibs_u32);
                    let hi_masked = _mm512_and_epi32(hi, _mm512_set1_epi32(0x0F));
                    let qv = _mm512_cvtepi32_ps(hi_masked);
                    let dq = _mm512_fmsub_ps(d2_v, qv, m2_v);
                    let xv = _mm512_loadu_ps(x.as_ptr().add(x_base + j_base + 32));
                    acc2 = _mm512_fmadd_ps(xv, dq, acc2);

                    // l = 16 — accumulates into acc3
                    let q_ptr = qs.as_ptr().add(qs_offset + 16);
                    let bytes_i64 = _mm_loadu_si128(q_ptr as *const __m128i);
                    let nibs_u32 = _mm512_cvtepu8_epi32(bytes_i64);
                    let hi = _mm512_srli_epi32::<4>(nibs_u32);
                    let hi_masked = _mm512_and_epi32(hi, _mm512_set1_epi32(0x0F));
                    let qv = _mm512_cvtepi32_ps(hi_masked);
                    let dq = _mm512_fmsub_ps(d2_v, qv, m2_v);
                    let xv = _mm512_loadu_ps(x.as_ptr().add(x_base + j_base + 32 + 16));
                    acc3 = _mm512_fmadd_ps(xv, dq, acc3);
                }

                qs_offset += 32; // 32 bytes per 64-element group
                is += 2;
                j_group += 1;
            }

            b += 1;
        }

        // Balanced tree-reduce of the four independent chains, then
        // horizontal reduce to scalar. The pair-then-pair order is
        // reproducible across runs.
        let acc01 = _mm512_add_ps(acc0, acc1);
        let acc23 = _mm512_add_ps(acc2, acc3);
        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc01, acc23));

        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────
// linear_q6k_avx512 — full output range
// ─────────────────────────────────────────────────────────────────

/// AVX-512-accelerated Q6_K matmul. Single-threaded entry point.
///
/// # Safety
/// See module docs.
#[target_feature(enable = "avx512f")]
pub unsafe fn linear_q6k_avx512(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    linear_q6k_avx512_range(
        x,
        w_blocks,
        out,
        in_dim,
        out_dim,
        RowRange {
            start: 0,
            end: out_dim,
        },
    );
}

/// AVX-512-accelerated Q6_K matmul over a *row range*. Mirrors
/// [`linear_q4k_avx512_range`] for the parallel dispatcher.
///
/// # Q6_K layout reminder
/// 210-byte super-block: 128 B ql (low 4-bit), 64 B qh (high 2-bit
/// packed 4:1), 16 B scales (signed i8), 2 B FP16 super-block scale.
///
/// Per ADR-029 v1 D11: 4-way interleaved layout (`ql[l] & 0xF`,
/// `ql[l+32] & 0xF`, `ql[l] >> 4`, `ql[l+32] >> 4` with qh shifts
/// 0, 2, 4, 6). We replicate this exactly to keep bit-exact with
/// scalar — see the NEON mirror's commentary on Lesson 17.5.
///
/// # Safety
/// See module docs.
#[target_feature(enable = "avx512f")]
pub unsafe fn linear_q6k_avx512_range(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    _out_dim: usize,
    range: RowRange,
) {
    let blocks_per_row = in_dim / 256;
    let bytes_per_row = blocks_per_row * 210;
    let nibble_mask = _mm512_set1_epi32(0x0F);
    let high_mask = _mm512_set1_epi32(0x03);
    let q6_zero = _mm512_set1_epi32(32);

    let mut i = range.start;
    while i < range.end {
        let row_byte_offset = i * bytes_per_row;
        // Four independent accumulators — see Q4_K above for Zen 4
        // FMA-latency rationale. Q6_K's inner body issues 4 FMAs per
        // stripe-iteration which map 1:1 onto acc0..acc3.
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        let mut b = 0;
        while b < blocks_per_row {
            let block_off = row_byte_offset + b * 210;
            let block = &w_blocks[block_off..block_off + 210];

            // NTA prefetch next block (210 B = ~4 lines) — see Q4_K
            // comment for the feature-gate rationale. Default OFF.
            #[cfg(feature = "nta-prefetch")]
            if b + 1 < blocks_per_row {
                let next_off = row_byte_offset + (b + 1) * 210;
                _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(next_off) as *const i8);
                _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(next_off + 64) as *const i8);
                _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(next_off + 128) as *const i8);
                _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(next_off + 192) as *const i8);
            }

            let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let ql = &block[0..128];
            let qh = &block[128..192];
            let sc = &block[192..208];

            let x_base = b * 256;

            // Two halves of 128 elements each (matches ggml reference
            // pattern and ADR-029 D11).
            let mut half = 0;
            while half < 2 {
                let ql_off = half * 64;
                let qh_off = half * 32;
                let sc_off = half * 8;
                let y_off = half * 128;

                // Q6_K stores four 32-element logical stripes per half.
                // Each 16-element stripe uses one signed scale, so AVX-512
                // can process the canonical ggml layout as 16 contiguous
                // lanes without gathers:
                //   y+l, y+l+32, y+l+64, y+l+96.
                //
                // This keeps the row split and K-order deterministic while
                // removing the old 4-lane `_mm512_set_ps` spoon-feeding loop,
                // which dominated Cherry LM-head time.
                let mut is = 0;
                while is < 2 {
                    let l = is * 16;

                    let ql_lo_b = _mm_loadu_si128(ql.as_ptr().add(ql_off + l) as *const __m128i);
                    let ql_hi_b =
                        _mm_loadu_si128(ql.as_ptr().add(ql_off + l + 32) as *const __m128i);
                    let qh_b = _mm_loadu_si128(qh.as_ptr().add(qh_off + l) as *const __m128i);

                    let ql_lo = _mm512_cvtepu8_epi32(ql_lo_b);
                    let ql_hi = _mm512_cvtepu8_epi32(ql_hi_b);
                    let qh16 = _mm512_cvtepu8_epi32(qh_b);

                    let s1 = _mm512_set1_ps(d * (sc[sc_off + is] as i8 as f32));
                    let s2 = _mm512_set1_ps(d * (sc[sc_off + is + 2] as i8 as f32));
                    let s3 = _mm512_set1_ps(d * (sc[sc_off + is + 4] as i8 as f32));
                    let s4 = _mm512_set1_ps(d * (sc[sc_off + is + 6] as i8 as f32));

                    let q1 = _mm512_sub_epi32(
                        _mm512_or_epi32(
                            _mm512_and_epi32(ql_lo, nibble_mask),
                            _mm512_slli_epi32::<4>(_mm512_and_epi32(qh16, high_mask)),
                        ),
                        q6_zero,
                    );
                    let q2 = _mm512_sub_epi32(
                        _mm512_or_epi32(
                            _mm512_and_epi32(ql_hi, nibble_mask),
                            _mm512_slli_epi32::<4>(_mm512_and_epi32(
                                _mm512_srli_epi32::<2>(qh16),
                                high_mask,
                            )),
                        ),
                        q6_zero,
                    );
                    let q3 = _mm512_sub_epi32(
                        _mm512_or_epi32(
                            _mm512_and_epi32(_mm512_srli_epi32::<4>(ql_lo), nibble_mask),
                            _mm512_slli_epi32::<4>(_mm512_and_epi32(
                                _mm512_srli_epi32::<4>(qh16),
                                high_mask,
                            )),
                        ),
                        q6_zero,
                    );
                    let q4 = _mm512_sub_epi32(
                        _mm512_or_epi32(
                            _mm512_and_epi32(_mm512_srli_epi32::<4>(ql_hi), nibble_mask),
                            _mm512_slli_epi32::<4>(_mm512_and_epi32(
                                _mm512_srli_epi32::<6>(qh16),
                                high_mask,
                            )),
                        ),
                        q6_zero,
                    );

                    let x1 = _mm512_loadu_ps(x.as_ptr().add(x_base + y_off + l));
                    let x2 = _mm512_loadu_ps(x.as_ptr().add(x_base + y_off + l + 32));
                    let x3 = _mm512_loadu_ps(x.as_ptr().add(x_base + y_off + l + 64));
                    let x4 = _mm512_loadu_ps(x.as_ptr().add(x_base + y_off + l + 96));

                    let dq1 = _mm512_mul_ps(s1, _mm512_cvtepi32_ps(q1));
                    let dq2 = _mm512_mul_ps(s2, _mm512_cvtepi32_ps(q2));
                    let dq3 = _mm512_mul_ps(s3, _mm512_cvtepi32_ps(q3));
                    let dq4 = _mm512_mul_ps(s4, _mm512_cvtepi32_ps(q4));

                    // One FMA per accumulator chain.
                    acc0 = _mm512_fmadd_ps(x1, dq1, acc0);
                    acc1 = _mm512_fmadd_ps(x2, dq2, acc1);
                    acc2 = _mm512_fmadd_ps(x3, dq3, acc2);
                    acc3 = _mm512_fmadd_ps(x4, dq4, acc3);

                    is += 1;
                }

                half += 1;
            }

            b += 1;
        }

        let acc01 = _mm512_add_ps(acc0, acc1);
        let acc23 = _mm512_add_ps(acc2, acc3);
        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc01, acc23));
        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────
// Pure f32 dot-product (used by attention score computation)
// ─────────────────────────────────────────────────────────────────

/// AVX-512 fused dot-product: result = Σ a[i] * b[i] for i in 0..len.
///
/// # Safety
/// `a.len() >= len`, `b.len() >= len`. AVX-512F enabled inside.
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product_f32_avx512(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = _mm512_setzero_ps();
    let chunks = len / 16;
    let mut i = 0;
    while i < chunks {
        let off = i * 16;
        let av = _mm512_loadu_ps(a.as_ptr().add(off));
        let bv = _mm512_loadu_ps(b.as_ptr().add(off));
        acc = _mm512_fmadd_ps(av, bv, acc);
        i += 1;
    }
    let mut result = _mm512_reduce_add_ps(acc);

    // Scalar tail (head-dim 128 / 16 = 8 chunks, no tail in practice).
    let tail_start = chunks * 16;
    let mut j = tail_start;
    while j < len {
        result += a[j] * b[j];
        j += 1;
    }
    result
}

/// AVX-512 argmax with scalar-identical semantics. Returns
/// `(first_max_index, max_value)`.
///
/// Two passes over `values`:
/// 1. NaN-suppressed vector max — a lane only updates on an *ordered*
///    greater-than (`_CMP_GT_OQ`), exactly mirroring the scalar
///    `if v > max_val` update where NaN comparisons are false.
/// 2. First index whose value compares ordered-equal to the max.
///
/// Equivalence to the scalar first-max-wins loop: pass 2 scans in
/// array order, so the returned index is the first occurrence of the
/// global max; NaN lanes can never match (`_CMP_EQ_OQ`); and the
/// ±0.0 corner behaves identically because IEEE `>` and `==` treat
/// -0.0 and +0.0 as equal in both implementations. If every element
/// is NaN the max stays -inf and index 0 is returned — same as the
/// scalar loop — and the caller's finiteness check rejects it.
///
/// # Safety
/// AVX-512F must be available.
#[target_feature(enable = "avx512f")]
pub unsafe fn argmax_f32_avx512(values: &[f32]) -> (usize, f32) {
    let len = values.len();
    let chunks = len / 16;

    // ── Pass 1: vector max (NaN-suppressed) ──────────────────────
    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut c = 0;
    while c < chunks {
        let v = _mm512_loadu_ps(values.as_ptr().add(c * 16));
        let gt = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(v, vmax);
        vmax = _mm512_mask_mov_ps(vmax, gt, v);
        c += 1;
    }
    let mut lanes = [0f32; 16];
    _mm512_storeu_ps(lanes.as_mut_ptr(), vmax);
    let mut max_val = f32::NEG_INFINITY;
    let mut l = 0;
    while l < 16 {
        if lanes[l] > max_val {
            max_val = lanes[l];
        }
        l += 1;
    }
    // Scalar tail (vocab 151,936 is 16-aligned; kept for generality).
    let mut j = chunks * 16;
    while j < len {
        if values[j] > max_val {
            max_val = values[j];
        }
        j += 1;
    }

    // ── Pass 2: first index holding the max ──────────────────────
    let target = _mm512_set1_ps(max_val);
    let mut c = 0;
    while c < chunks {
        let v = _mm512_loadu_ps(values.as_ptr().add(c * 16));
        let eq = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(v, target);
        if eq != 0 {
            return (c * 16 + eq.trailing_zeros() as usize, max_val);
        }
        c += 1;
    }
    let mut j = chunks * 16;
    while j < len {
        if values[j] == max_val {
            return (j, max_val);
        }
        j += 1;
    }
    (0, max_val)
}

/// AVX-512 weighted accumulation: output[i] += weight * vec[i].
///
/// # Safety
/// `vec.len() >= head_dim`, `output.len() >= head_dim`. AVX-512F enabled.
#[target_feature(enable = "avx512f")]
pub unsafe fn weighted_add_f32_avx512(
    weight: f32,
    vec: &[f32],
    output: &mut [f32],
    head_dim: usize,
) {
    let wv = _mm512_set1_ps(weight);
    let chunks = head_dim / 16;
    let mut i = 0;
    while i < chunks {
        let off = i * 16;
        let vv = _mm512_loadu_ps(vec.as_ptr().add(off));
        let ov = _mm512_loadu_ps(output.as_ptr().add(off));
        let res = _mm512_fmadd_ps(wv, vv, ov);
        _mm512_storeu_ps(output.as_mut_ptr().add(off), res);
        i += 1;
    }
    let tail_start = chunks * 16;
    let mut j = tail_start;
    while j < head_dim {
        output[j] += weight * vec[j];
        j += 1;
    }
}

// ─────────────────────────────────────────────────────────────────
// linear_q4_0_avx512 — AVX-512 Q4_0 matmul (Kimi K2.6 native int4)
// ─────────────────────────────────────────────────────────────────
//
// Q4_0 block layout (from zero_gguf_parser::dequant):
//
//   bytes 0-1 : d (fp16 scale, LE)
//   bytes 2-17: qs[16] (16 bytes packing 32 nibbles)
//
// Per llama.cpp packing: byte j carries low nibble → output[j]
// (0..15) and high nibble → output[j + 16] (16..31). Dequant per
// element: `(nibble - 8) * d`.
//
// Inner loop processes 16 elements per SIMD op:
//   1. Load 16 bytes from qs → __m128i.
//   2. _mm512_cvtepu8_epi32 zero-extends them to 16 × i32.
//   3. AND with 0x0F to grab low nibbles → 16 × i32 in [0..15].
//      Sub 8 → [-8..7], convert to f32 → multiply by d.
//   4. FMA with the 16 corresponding x activations into acc_lo.
//   5. Shift right by 4 + AND 0x0F for high nibbles → output[16..32]
//      lane → multiply by d → FMA into acc_hi.
//
// Two independent accumulators per block keep the Zen 4 FMA latency
// (4 cycles) hidden by issue-rate (1 cycle reciprocal). Final
// reduce: acc_lo + acc_hi → reduce_add_ps.

/// AVX-512-accelerated Q4_0 matmul. Single-threaded entry point.
///
/// # Safety
/// See module docs. `target_feature = "avx512f"` enabled inside.
#[target_feature(enable = "avx512f")]
pub unsafe fn linear_q4_0_avx512(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    linear_q4_0_avx512_range(
        x,
        w_blocks,
        out,
        in_dim,
        out_dim,
        RowRange {
            start: 0,
            end: out_dim,
        },
    );
}

/// Row-range variant for parallel matmul dispatch.
///
/// # Row pairing (qwen-perf-v2)
///
/// Rows are processed two at a time. A single Q4_0 row only sustains
/// two FMA dependency chains (acc_lo / acc_hi) — with Zen 4's 4-cycle
/// FMA latency that leaves the FMA pipes half idle while the row
/// streams its weights. Pairing rows doubles the independent chains
/// to four AND lets both rows share each `x` vector load.
///
/// **Bit-exactness:** within each row, the per-block FMA order into
/// its own accumulator pair and the final `(lo + hi) → reduce_add`
/// are byte-identical to the single-row body. Pairing only
/// interleaves *independent* rows; no anchor drift.
///
/// # Safety
/// See module docs.
#[target_feature(enable = "avx512f")]
pub unsafe fn linear_q4_0_avx512_range(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    _out_dim: usize,
    range: RowRange,
) {
    const Q4_0_BLOCK_SIZE: usize = 32;
    const Q4_0_BLOCK_BYTES: usize = 18;

    let blocks_per_row = in_dim / Q4_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q4_0_BLOCK_BYTES;
    let nibble_mask = _mm512_set1_epi32(0x0F);
    let bias_8 = _mm512_set1_ps(8.0);

    let mut i = range.start;

    // ── Row pairs: 4 independent FMA chains, shared x loads ──────
    while i + 1 < range.end {
        let row0_offset = i * bytes_per_row;
        let row1_offset = row0_offset + bytes_per_row;

        let mut acc_lo0 = _mm512_setzero_ps();
        let mut acc_hi0 = _mm512_setzero_ps();
        let mut acc_lo1 = _mm512_setzero_ps();
        let mut acc_hi1 = _mm512_setzero_ps();

        let mut b = 0;
        while b < blocks_per_row {
            // NTA software prefetch ~1.1 KiB ahead in BOTH row
            // streams — see the Q4_0X4 kernel for the rationale.
            // Gated to every 4th block (4 × 18 B = 72 B interval, two
            // lines prefetched = 128 B window → full coverage without
            // redundant per-block prefetches). Feature-gated,
            // default OFF.
            #[cfg(feature = "nta-prefetch")]
            if b & 3 == 0 {
                const PF_AHEAD: usize = 64; // 64 × 18 B = 1152 B
                let pf0 = row0_offset + (b + PF_AHEAD) * Q4_0_BLOCK_BYTES;
                if pf0 + 64 + Q4_0_BLOCK_BYTES <= w_blocks.len() {
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf0) as *const i8);
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf0 + 64) as *const i8);
                }
                let pf1 = row1_offset + (b + PF_AHEAD) * Q4_0_BLOCK_BYTES;
                if pf1 + 64 + Q4_0_BLOCK_BYTES <= w_blocks.len() {
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf1) as *const i8);
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf1 + 64) as *const i8);
                }
            }

            // Shared activations for both rows.
            let xv_lo = _mm512_loadu_ps(x.as_ptr().add(b * Q4_0_BLOCK_SIZE));
            let xv_hi = _mm512_loadu_ps(x.as_ptr().add(b * Q4_0_BLOCK_SIZE + 16));

            // Row 0 block.
            let block0_off = row0_offset + b * Q4_0_BLOCK_BYTES;
            let block0 = &w_blocks[block0_off..block0_off + Q4_0_BLOCK_BYTES];
            let d0 = fp16_to_f32(u16::from_le_bytes([block0[0], block0[1]]));
            let d0_v = _mm512_set1_ps(d0);
            let bytes0_u32 =
                _mm512_cvtepu8_epi32(_mm_loadu_si128(block0.as_ptr().add(2) as *const __m128i));

            // Row 1 block.
            let block1_off = row1_offset + b * Q4_0_BLOCK_BYTES;
            let block1 = &w_blocks[block1_off..block1_off + Q4_0_BLOCK_BYTES];
            let d1 = fp16_to_f32(u16::from_le_bytes([block1[0], block1[1]]));
            let d1_v = _mm512_set1_ps(d1);
            let bytes1_u32 =
                _mm512_cvtepu8_epi32(_mm_loadu_si128(block1.as_ptr().add(2) as *const __m128i));

            // Low nibbles → output[j] (j = 0..15), (nibble - 8) * d.
            let lo0_f = _mm512_cvtepi32_ps(_mm512_and_epi32(bytes0_u32, nibble_mask));
            let dq_lo0 = _mm512_mul_ps(d0_v, _mm512_sub_ps(lo0_f, bias_8));
            acc_lo0 = _mm512_fmadd_ps(xv_lo, dq_lo0, acc_lo0);

            let lo1_f = _mm512_cvtepi32_ps(_mm512_and_epi32(bytes1_u32, nibble_mask));
            let dq_lo1 = _mm512_mul_ps(d1_v, _mm512_sub_ps(lo1_f, bias_8));
            acc_lo1 = _mm512_fmadd_ps(xv_lo, dq_lo1, acc_lo1);

            // High nibbles → output[j + 16] (j = 0..15).
            let hi0_f = _mm512_cvtepi32_ps(_mm512_and_epi32(
                _mm512_srli_epi32::<4>(bytes0_u32),
                nibble_mask,
            ));
            let dq_hi0 = _mm512_mul_ps(d0_v, _mm512_sub_ps(hi0_f, bias_8));
            acc_hi0 = _mm512_fmadd_ps(xv_hi, dq_hi0, acc_hi0);

            let hi1_f = _mm512_cvtepi32_ps(_mm512_and_epi32(
                _mm512_srli_epi32::<4>(bytes1_u32),
                nibble_mask,
            ));
            let dq_hi1 = _mm512_mul_ps(d1_v, _mm512_sub_ps(hi1_f, bias_8));
            acc_hi1 = _mm512_fmadd_ps(xv_hi, dq_hi1, acc_hi1);

            b += 1;
        }

        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo0, acc_hi0));
        out[i + 1] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo1, acc_hi1));
        i += 2;
    }

    // ── Tail row (odd range length) — original single-row body ───
    while i < range.end {
        let row_byte_offset = i * bytes_per_row;

        // Two independent accumulators — one for the low-nibble lane
        // (output[0..16] of each block) and one for the high-nibble
        // lane (output[16..32]).
        let mut acc_lo = _mm512_setzero_ps();
        let mut acc_hi = _mm512_setzero_ps();

        let mut b = 0;
        while b < blocks_per_row {
            let block_off = row_byte_offset + b * Q4_0_BLOCK_BYTES;
            let block = &w_blocks[block_off..block_off + Q4_0_BLOCK_BYTES];

            // fp16 scale.
            let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let d_v = _mm512_set1_ps(d);

            // Load 16 bytes of qs into the low 128 bits of __m128i.
            let qs_ptr = block.as_ptr().add(2);
            let bytes_i128 = _mm_loadu_si128(qs_ptr as *const __m128i);
            // Zero-extend each byte to a 32-bit lane in __m512i.
            let bytes_u32 = _mm512_cvtepu8_epi32(bytes_i128);

            // Low nibble of byte j → output[j] (j = 0..15).
            let lo_i32 = _mm512_and_epi32(bytes_u32, nibble_mask);
            let lo_f = _mm512_cvtepi32_ps(lo_i32);
            // (nibble - 8) * d.
            let dq_lo = _mm512_mul_ps(d_v, _mm512_sub_ps(lo_f, bias_8));
            let xv_lo = _mm512_loadu_ps(x.as_ptr().add(b * Q4_0_BLOCK_SIZE));
            acc_lo = _mm512_fmadd_ps(xv_lo, dq_lo, acc_lo);

            // High nibble of byte j → output[j + 16] (j = 0..15).
            let hi_i32 = _mm512_and_epi32(_mm512_srli_epi32::<4>(bytes_u32), nibble_mask);
            let hi_f = _mm512_cvtepi32_ps(hi_i32);
            let dq_hi = _mm512_mul_ps(d_v, _mm512_sub_ps(hi_f, bias_8));
            let xv_hi = _mm512_loadu_ps(x.as_ptr().add(b * Q4_0_BLOCK_SIZE + 16));
            acc_hi = _mm512_fmadd_ps(xv_hi, dq_hi, acc_hi);

            b += 1;
        }

        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo, acc_hi));
        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────
// linear_q4_0x4_avx512 — `.smodel`-v2 row-interleaved Q4_0 matmul
// ─────────────────────────────────────────────────────────────────
//
// Q4_0X4 layout (SilicatePack v2, SIDX dtype id 100): output rows are
// stored in groups of 4. For group g and K-block b, the 4 rows' blocks
// are interleaved into one 72-byte group-block:
//
//   bytes  0..8  : d0 d1 d2 d3        (4 × fp16 scales)
//   bytes  8..24 : qs of row g*4+0    (16 bytes, 32 nibbles)
//   bytes 24..40 : qs of row g*4+1
//   bytes 40..56 : qs of row g*4+2
//   bytes 56..72 : qs of row g*4+3
//
// Group stride = blocks_per_row * 72 bytes; total tensor bytes are
// identical to plain Q4_0 (rows * blocks_per_row * 18).
//
// Why: the 2-row paired plain kernel streams TWO weight cursors that
// run `bytes_per_row` apart, and shares each x-load between 2 rows.
// The interleaved layout streams ONE strictly sequential cursor (ideal
// for the L2 streamer — no second stream to track, no row-stride
// aliasing), shares each x-load between 4 rows, and runs 8 independent
// FMA chains (vs 4), fully covering Zen 4's 4-cycle FMA latency on
// both FMA pipes.
//
// # Bit-exactness
//
// Per output row the computation is byte-identical to the plain
// kernel's single-row body: blocks are visited in ascending K-order,
// low nibbles accumulate into that row's acc_lo, high nibbles into its
// acc_hi, and the final reduction is `reduce_add(acc_lo + acc_hi)`.
// Only the *storage order of the weights* and the *instruction
// interleaving across independent rows* change — neither affects any
// row's FP operation sequence. Token-ID and logit_bits anchors are
// preserved exactly (not just within 1 ULP).

/// Row-range Q4_0X4 (4-row-interleaved) matmul for parallel dispatch.
///
/// Handles ranges that are not 4-aligned by computing the full group
/// and storing only the owned lanes — the dispatcher aligns splits to
/// the group size, so partial groups only occur in degenerate splits
/// and at fused-segment edges.
///
/// # Safety
/// See module docs. `w_blocks` must hold `out_dim` rows in Q4_0X4
/// group-interleaved layout (`out_dim % 4 == 0` is a packer
/// guarantee).
#[target_feature(enable = "avx512f")]
pub unsafe fn linear_q4_0x4_avx512_range(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    _out_dim: usize,
    range: RowRange,
) {
    const Q4_0_BLOCK_SIZE: usize = 32;
    const Q4_0_BLOCK_BYTES: usize = 18;
    const G: usize = 4;
    const GROUP_BLOCK_BYTES: usize = G * Q4_0_BLOCK_BYTES; // 72

    let blocks_per_row = in_dim / Q4_0_BLOCK_SIZE;
    let group_stride = blocks_per_row * GROUP_BLOCK_BYTES;
    let nibble_mask = _mm512_set1_epi32(0x0F);
    let bias_8 = _mm512_set1_ps(8.0);

    let mut i = range.start;
    while i < range.end {
        let g = i / G;
        let lane_lo = i - g * G;
        let rows_left = range.end - g * G;
        let lane_hi = if rows_left < G { rows_left } else { G };
        let gbase = g * group_stride;

        let mut acc_lo0 = _mm512_setzero_ps();
        let mut acc_hi0 = _mm512_setzero_ps();
        let mut acc_lo1 = _mm512_setzero_ps();
        let mut acc_hi1 = _mm512_setzero_ps();
        let mut acc_lo2 = _mm512_setzero_ps();
        let mut acc_hi2 = _mm512_setzero_ps();
        let mut acc_lo3 = _mm512_setzero_ps();
        let mut acc_hi3 = _mm512_setzero_ps();

        let mut b = 0;
        while b < blocks_per_row {
            let blk = gbase + b * GROUP_BLOCK_BYTES;
            let block = &w_blocks[blk..blk + GROUP_BLOCK_BYTES];

            // Software prefetch ~1.1 KiB ahead in the (strictly
            // sequential) group stream — feature-gated, default OFF.
            // Unlike the legacy next-block pattern, the lookahead is
            // sized so the NTA placement actually sticks: an NTA
            // prefetch only controls cache placement if it reaches
            // the line BEFORE the demand load / L2 streamer does
            // (Zen 4: NTA lines go to L2 marked for quick eviction
            // and skip L3 insertion — keeps the per-CCD L3 free for
            // KV/activations while ~330-730 MB of weights stream
            // through per token).
            #[cfg(feature = "nta-prefetch")]
            {
                const PF_AHEAD: usize = 16; // 16 × 72 B = 1152 B
                let pf = blk + PF_AHEAD * GROUP_BLOCK_BYTES;
                if pf + GROUP_BLOCK_BYTES <= w_blocks.len() {
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf + 64) as *const i8);
                }
            }

            // Shared activations for all 4 rows.
            let xv_lo = _mm512_loadu_ps(x.as_ptr().add(b * Q4_0_BLOCK_SIZE));
            let xv_hi = _mm512_loadu_ps(x.as_ptr().add(b * Q4_0_BLOCK_SIZE + 16));

            let d0_v = _mm512_set1_ps(fp16_to_f32(u16::from_le_bytes([block[0], block[1]])));
            let d1_v = _mm512_set1_ps(fp16_to_f32(u16::from_le_bytes([block[2], block[3]])));
            let d2_v = _mm512_set1_ps(fp16_to_f32(u16::from_le_bytes([block[4], block[5]])));
            let d3_v = _mm512_set1_ps(fp16_to_f32(u16::from_le_bytes([block[6], block[7]])));

            let bytes0 =
                _mm512_cvtepu8_epi32(_mm_loadu_si128(block.as_ptr().add(8) as *const __m128i));
            let bytes1 =
                _mm512_cvtepu8_epi32(_mm_loadu_si128(block.as_ptr().add(24) as *const __m128i));
            let bytes2 =
                _mm512_cvtepu8_epi32(_mm_loadu_si128(block.as_ptr().add(40) as *const __m128i));
            let bytes3 =
                _mm512_cvtepu8_epi32(_mm_loadu_si128(block.as_ptr().add(56) as *const __m128i));

            // Low nibbles → block elements 0..16, (nibble - 8) * d.
            let lo0 = _mm512_cvtepi32_ps(_mm512_and_epi32(bytes0, nibble_mask));
            acc_lo0 = _mm512_fmadd_ps(
                xv_lo,
                _mm512_mul_ps(d0_v, _mm512_sub_ps(lo0, bias_8)),
                acc_lo0,
            );
            let lo1 = _mm512_cvtepi32_ps(_mm512_and_epi32(bytes1, nibble_mask));
            acc_lo1 = _mm512_fmadd_ps(
                xv_lo,
                _mm512_mul_ps(d1_v, _mm512_sub_ps(lo1, bias_8)),
                acc_lo1,
            );
            let lo2 = _mm512_cvtepi32_ps(_mm512_and_epi32(bytes2, nibble_mask));
            acc_lo2 = _mm512_fmadd_ps(
                xv_lo,
                _mm512_mul_ps(d2_v, _mm512_sub_ps(lo2, bias_8)),
                acc_lo2,
            );
            let lo3 = _mm512_cvtepi32_ps(_mm512_and_epi32(bytes3, nibble_mask));
            acc_lo3 = _mm512_fmadd_ps(
                xv_lo,
                _mm512_mul_ps(d3_v, _mm512_sub_ps(lo3, bias_8)),
                acc_lo3,
            );

            // High nibbles → block elements 16..32.
            let hi0 = _mm512_cvtepi32_ps(_mm512_and_epi32(
                _mm512_srli_epi32::<4>(bytes0),
                nibble_mask,
            ));
            acc_hi0 = _mm512_fmadd_ps(
                xv_hi,
                _mm512_mul_ps(d0_v, _mm512_sub_ps(hi0, bias_8)),
                acc_hi0,
            );
            let hi1 = _mm512_cvtepi32_ps(_mm512_and_epi32(
                _mm512_srli_epi32::<4>(bytes1),
                nibble_mask,
            ));
            acc_hi1 = _mm512_fmadd_ps(
                xv_hi,
                _mm512_mul_ps(d1_v, _mm512_sub_ps(hi1, bias_8)),
                acc_hi1,
            );
            let hi2 = _mm512_cvtepi32_ps(_mm512_and_epi32(
                _mm512_srli_epi32::<4>(bytes2),
                nibble_mask,
            ));
            acc_hi2 = _mm512_fmadd_ps(
                xv_hi,
                _mm512_mul_ps(d2_v, _mm512_sub_ps(hi2, bias_8)),
                acc_hi2,
            );
            let hi3 = _mm512_cvtepi32_ps(_mm512_and_epi32(
                _mm512_srli_epi32::<4>(bytes3),
                nibble_mask,
            ));
            acc_hi3 = _mm512_fmadd_ps(
                xv_hi,
                _mm512_mul_ps(d3_v, _mm512_sub_ps(hi3, bias_8)),
                acc_hi3,
            );

            b += 1;
        }

        // Store only the lanes this range owns. Reduction order per
        // row matches the plain kernel exactly: (acc_lo + acc_hi) →
        // reduce_add.
        if lane_lo <= 0 && 0 < lane_hi {
            out[g * G] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo0, acc_hi0));
        }
        if lane_lo <= 1 && 1 < lane_hi {
            out[g * G + 1] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo1, acc_hi1));
        }
        if lane_lo <= 2 && 2 < lane_hi {
            out[g * G + 2] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo2, acc_hi2));
        }
        if lane_lo <= 3 && 3 < lane_hi {
            out[g * G + 3] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo3, acc_hi3));
        }

        i = g * G + lane_hi;
    }
}

// ─────────────────────────────────────────────────────────────────
// linear_q8_0x4_avx512 — `.smodel`-v2 row-interleaved Q8_0 matmul
// ─────────────────────────────────────────────────────────────────
//
// Q8_0X4 layout (SIDX dtype id 101): same grouping as Q4_0X4 with the
// Q8_0 block body. Group-block (4 rows × one 32-element K-block):
//
//   bytes   0..8   : d0 d1 d2 d3       (4 × fp16 scales)
//   bytes   8..40  : qs of row g*4+0   (32 × i8)
//   bytes  40..72  : qs of row g*4+1
//   bytes  72..104 : qs of row g*4+2
//   bytes 104..136 : qs of row g*4+3
//
// Group-block = 136 bytes; group stride = blocks_per_row * 136.
// Targets the LM head (`output.weight`, 151,936 × 2048 — the largest
// single weight stream per token, ~330 MB).

/// Row-range Q8_0X4 (4-row-interleaved) matmul for parallel dispatch.
///
/// # Safety
/// See module docs and [`linear_q4_0x4_avx512_range`].
#[target_feature(enable = "avx512f")]
pub unsafe fn linear_q8_0x4_avx512_range(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    _out_dim: usize,
    range: RowRange,
) {
    const Q8_0_BLOCK_SIZE: usize = 32;
    const G: usize = 4;
    const GROUP_BLOCK_BYTES: usize = 8 + G * 32; // 136

    let blocks_per_row = in_dim / Q8_0_BLOCK_SIZE;
    let group_stride = blocks_per_row * GROUP_BLOCK_BYTES;

    let mut i = range.start;
    while i < range.end {
        let g = i / G;
        let lane_lo = i - g * G;
        let rows_left = range.end - g * G;
        let lane_hi = if rows_left < G { rows_left } else { G };
        let gbase = g * group_stride;

        let mut acc_lo0 = _mm512_setzero_ps();
        let mut acc_hi0 = _mm512_setzero_ps();
        let mut acc_lo1 = _mm512_setzero_ps();
        let mut acc_hi1 = _mm512_setzero_ps();
        let mut acc_lo2 = _mm512_setzero_ps();
        let mut acc_hi2 = _mm512_setzero_ps();
        let mut acc_lo3 = _mm512_setzero_ps();
        let mut acc_hi3 = _mm512_setzero_ps();

        let mut b = 0;
        while b < blocks_per_row {
            let blk = gbase + b * GROUP_BLOCK_BYTES;
            let block = &w_blocks[blk..blk + GROUP_BLOCK_BYTES];

            // NTA software prefetch ~1.2 KiB ahead in the sequential
            // group stream — see the Q4_0X4 kernel for the lookahead
            // + placement rationale. Feature-gated, default OFF.
            #[cfg(feature = "nta-prefetch")]
            {
                const PF_AHEAD: usize = 9; // 9 × 136 B = 1224 B
                let pf = blk + PF_AHEAD * GROUP_BLOCK_BYTES;
                if pf + GROUP_BLOCK_BYTES <= w_blocks.len() {
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf + 64) as *const i8);
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf + 128) as *const i8);
                }
            }

            let xv_lo = _mm512_loadu_ps(x.as_ptr().add(b * Q8_0_BLOCK_SIZE));
            let xv_hi = _mm512_loadu_ps(x.as_ptr().add(b * Q8_0_BLOCK_SIZE + 16));

            let d0_v = _mm512_set1_ps(fp16_to_f32(u16::from_le_bytes([block[0], block[1]])));
            let d1_v = _mm512_set1_ps(fp16_to_f32(u16::from_le_bytes([block[2], block[3]])));
            let d2_v = _mm512_set1_ps(fp16_to_f32(u16::from_le_bytes([block[4], block[5]])));
            let d3_v = _mm512_set1_ps(fp16_to_f32(u16::from_le_bytes([block[6], block[7]])));

            // Row 0 — SIGN-extend i8 (see plain Q8_0 kernel note).
            let lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block.as_ptr().add(8) as *const __m128i,
            )));
            acc_lo0 = _mm512_fmadd_ps(xv_lo, _mm512_mul_ps(d0_v, lo0), acc_lo0);
            let hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block.as_ptr().add(8 + 16) as *const __m128i,
            )));
            acc_hi0 = _mm512_fmadd_ps(xv_hi, _mm512_mul_ps(d0_v, hi0), acc_hi0);

            // Row 1.
            let lo1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block.as_ptr().add(40) as *const __m128i,
            )));
            acc_lo1 = _mm512_fmadd_ps(xv_lo, _mm512_mul_ps(d1_v, lo1), acc_lo1);
            let hi1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block.as_ptr().add(40 + 16) as *const __m128i,
            )));
            acc_hi1 = _mm512_fmadd_ps(xv_hi, _mm512_mul_ps(d1_v, hi1), acc_hi1);

            // Row 2.
            let lo2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block.as_ptr().add(72) as *const __m128i,
            )));
            acc_lo2 = _mm512_fmadd_ps(xv_lo, _mm512_mul_ps(d2_v, lo2), acc_lo2);
            let hi2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block.as_ptr().add(72 + 16) as *const __m128i,
            )));
            acc_hi2 = _mm512_fmadd_ps(xv_hi, _mm512_mul_ps(d2_v, hi2), acc_hi2);

            // Row 3.
            let lo3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block.as_ptr().add(104) as *const __m128i,
            )));
            acc_lo3 = _mm512_fmadd_ps(xv_lo, _mm512_mul_ps(d3_v, lo3), acc_lo3);
            let hi3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block.as_ptr().add(104 + 16) as *const __m128i,
            )));
            acc_hi3 = _mm512_fmadd_ps(xv_hi, _mm512_mul_ps(d3_v, hi3), acc_hi3);

            b += 1;
        }

        if lane_lo <= 0 && 0 < lane_hi {
            out[g * G] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo0, acc_hi0));
        }
        if lane_lo <= 1 && 1 < lane_hi {
            out[g * G + 1] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo1, acc_hi1));
        }
        if lane_lo <= 2 && 2 < lane_hi {
            out[g * G + 2] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo2, acc_hi2));
        }
        if lane_lo <= 3 && 3 < lane_hi {
            out[g * G + 3] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo3, acc_hi3));
        }

        i = g * G + lane_hi;
    }
}

// ─────────────────────────────────────────────────────────────────
// linear_q8_0_avx512 — AVX-512 Q8_0 matmul (embed / output for
//                      Kimi K2.6 Q4_0 builds)
// ─────────────────────────────────────────────────────────────────
//
// Q8_0 block layout: fp16 d (2 B) + 32 × i8 (32 B) = 34 B.
// Dequant: output[k] = (i8 qs[k]) * d.
//
// Inner loop processes 16 elements per SIMD op via
// `_mm512_cvtepi8_epi32` (sign-extends i8 → i32) — this is the
// critical difference vs Q4_0's `_mm512_cvtepu8_epi32` (zero-extend).
// If we ever swap the two by mistake, the negative half of the
// signed range would inflate by 256× and produce nonsense logits.

/// AVX-512-accelerated Q8_0 matmul. Single-threaded entry point.
///
/// # Safety
/// See module docs.
#[target_feature(enable = "avx512f")]
pub unsafe fn linear_q8_0_avx512(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    linear_q8_0_avx512_range(
        x,
        w_blocks,
        out,
        in_dim,
        out_dim,
        RowRange {
            start: 0,
            end: out_dim,
        },
    );
}

/// Row-range variant for parallel matmul dispatch.
///
/// # Row pairing (qwen-perf-v2)
///
/// Same transformation as [`linear_q4_0_avx512_range`]: two rows per
/// iteration → four independent FMA chains + shared activation
/// loads. Q8_0 is the native `.smodel` LM-head kernel
/// (`output.weight`, 151,936 × 2048 = the largest single matmul per
/// token), so chain starvation here costs the most wall-clock.
///
/// **Bit-exactness:** per row identical op order vs the single-row
/// body (kept as the odd-row tail); pairing only interleaves
/// independent rows.
///
/// # Safety
/// See module docs.
#[target_feature(enable = "avx512f")]
pub unsafe fn linear_q8_0_avx512_range(
    x: &[f32],
    w_blocks: &[u8],
    out: &mut [f32],
    in_dim: usize,
    _out_dim: usize,
    range: RowRange,
) {
    const Q8_0_BLOCK_SIZE: usize = 32;
    const Q8_0_BLOCK_BYTES: usize = 34;

    let blocks_per_row = in_dim / Q8_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    let mut i = range.start;

    // ── Row pairs: 4 independent FMA chains, shared x loads ──────
    while i + 1 < range.end {
        let row0_offset = i * bytes_per_row;
        let row1_offset = row0_offset + bytes_per_row;

        let mut acc_lo0 = _mm512_setzero_ps();
        let mut acc_hi0 = _mm512_setzero_ps();
        let mut acc_lo1 = _mm512_setzero_ps();
        let mut acc_hi1 = _mm512_setzero_ps();

        let mut b = 0;
        while b < blocks_per_row {
            // NTA software prefetch ~1.2 KiB ahead in BOTH row
            // streams — see the Q4_0X4 kernel for the rationale.
            // Gated to every 2nd block (2 × 34 B = 68 B interval, two
            // lines = 128 B window → full coverage). Feature-gated,
            // default OFF.
            #[cfg(feature = "nta-prefetch")]
            if b & 1 == 0 {
                const PF_AHEAD: usize = 34; // 34 × 34 B = 1156 B
                let pf0 = row0_offset + (b + PF_AHEAD) * Q8_0_BLOCK_BYTES;
                if pf0 + 64 + Q8_0_BLOCK_BYTES <= w_blocks.len() {
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf0) as *const i8);
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf0 + 64) as *const i8);
                }
                let pf1 = row1_offset + (b + PF_AHEAD) * Q8_0_BLOCK_BYTES;
                if pf1 + 64 + Q8_0_BLOCK_BYTES <= w_blocks.len() {
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf1) as *const i8);
                    _mm_prefetch::<_MM_HINT_NTA>(w_blocks.as_ptr().add(pf1 + 64) as *const i8);
                }
            }

            // Shared activations for both rows.
            let xv_lo = _mm512_loadu_ps(x.as_ptr().add(b * Q8_0_BLOCK_SIZE));
            let xv_hi = _mm512_loadu_ps(x.as_ptr().add(b * Q8_0_BLOCK_SIZE + 16));

            // Row 0 block.
            let block0_off = row0_offset + b * Q8_0_BLOCK_BYTES;
            let block0 = &w_blocks[block0_off..block0_off + Q8_0_BLOCK_BYTES];
            let d0_v = _mm512_set1_ps(fp16_to_f32(u16::from_le_bytes([block0[0], block0[1]])));

            // Row 1 block.
            let block1_off = row1_offset + b * Q8_0_BLOCK_BYTES;
            let block1 = &w_blocks[block1_off..block1_off + Q8_0_BLOCK_BYTES];
            let d1_v = _mm512_set1_ps(fp16_to_f32(u16::from_le_bytes([block1[0], block1[1]])));

            // Low 16 bytes of qs (signed i8 → i32 → f32, SIGN-extend).
            let lo0_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block0.as_ptr().add(2) as *const __m128i,
            )));
            acc_lo0 = _mm512_fmadd_ps(xv_lo, _mm512_mul_ps(d0_v, lo0_f), acc_lo0);

            let lo1_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block1.as_ptr().add(2) as *const __m128i,
            )));
            acc_lo1 = _mm512_fmadd_ps(xv_lo, _mm512_mul_ps(d1_v, lo1_f), acc_lo1);

            // High 16 bytes of qs (signed i8 → i32 → f32, SIGN-extend).
            let hi0_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block0.as_ptr().add(2 + 16) as *const __m128i,
            )));
            acc_hi0 = _mm512_fmadd_ps(xv_hi, _mm512_mul_ps(d0_v, hi0_f), acc_hi0);

            let hi1_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                block1.as_ptr().add(2 + 16) as *const __m128i,
            )));
            acc_hi1 = _mm512_fmadd_ps(xv_hi, _mm512_mul_ps(d1_v, hi1_f), acc_hi1);

            b += 1;
        }

        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo0, acc_hi0));
        out[i + 1] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo1, acc_hi1));
        i += 2;
    }

    // ── Tail row (odd range length) — original single-row body ───
    while i < range.end {
        let row_byte_offset = i * bytes_per_row;

        let mut acc_lo = _mm512_setzero_ps();
        let mut acc_hi = _mm512_setzero_ps();

        let mut b = 0;
        while b < blocks_per_row {
            let block_off = row_byte_offset + b * Q8_0_BLOCK_BYTES;
            let block = &w_blocks[block_off..block_off + Q8_0_BLOCK_BYTES];

            let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let d_v = _mm512_set1_ps(d);

            // Low 16 bytes of qs (signed i8 → i32 → f32).
            let qs_lo_ptr = block.as_ptr().add(2);
            let qs_lo_i128 = _mm_loadu_si128(qs_lo_ptr as *const __m128i);
            let lo_i32 = _mm512_cvtepi8_epi32(qs_lo_i128); // SIGN-extend
            let lo_f = _mm512_cvtepi32_ps(lo_i32);
            let dq_lo = _mm512_mul_ps(d_v, lo_f);
            let xv_lo = _mm512_loadu_ps(x.as_ptr().add(b * Q8_0_BLOCK_SIZE));
            acc_lo = _mm512_fmadd_ps(xv_lo, dq_lo, acc_lo);

            // High 16 bytes of qs (signed i8 → i32 → f32).
            let qs_hi_ptr = block.as_ptr().add(2 + 16);
            let qs_hi_i128 = _mm_loadu_si128(qs_hi_ptr as *const __m128i);
            let hi_i32 = _mm512_cvtepi8_epi32(qs_hi_i128); // SIGN-extend
            let hi_f = _mm512_cvtepi32_ps(hi_i32);
            let dq_hi = _mm512_mul_ps(d_v, hi_f);
            let xv_hi = _mm512_loadu_ps(x.as_ptr().add(b * Q8_0_BLOCK_SIZE + 16));
            acc_hi = _mm512_fmadd_ps(xv_hi, dq_hi, acc_hi);

            b += 1;
        }

        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo, acc_hi));
        i += 1;
    }
}
