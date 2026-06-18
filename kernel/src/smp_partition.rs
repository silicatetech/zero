// SPDX-License-Identifier: AGPL-3.0-or-later
//! Row-partition math for the parallel matmul dispatch — extracted from
//! `smp.rs` so it can be exercised by the host test harness
//! (`crates/kernel-tests`).
//!
//! This module is **pure**: it depends only on `core`, touches no
//! hardware, no statics, and no arch-specific code, so it compiles and
//! runs identically on the bare-metal kernel target and on the dev host.
//! The rest of `smp.rs` (AP boot, APIC, the SMP runtime) cannot — it is
//! `x86_64`-only `global_asm!` and MMIO — which is why only the
//! partition leaf functions live here. `smp.rs` re-exports every public
//! item below (`pub use smp_partition::*`), so `crate::smp::split_rows`,
//! `crate::smp::RowRange`, `crate::smp::MAX_CORES`, etc. continue to
//! resolve unchanged for the kernel and its dependents
//! (`inference_avx512.rs`).
//!
//! # Bit-exactness
//!
//! These functions decide only *which core owns which output rows*. The
//! per-row reduction K-order is untouched, so any split (even/aligned/
//! discounted) is bit-identical at the row level — see the per-function
//! notes carried over from `smp.rs`.

/// Maximum number of cores the SMP dispatch will ever fan a matmul out
/// to. The work-slot arrays and every `[_; MAX_CORES]` partition buffer
/// are sized from this constant; raising it costs `MAX_CORES *
/// sizeof(slot)` bytes of `.bss` and nothing else.
pub const MAX_CORES: usize = 128;

/// Upper bound on the BSP row-share discount (percent). The BSP's slice
/// can be shrunk by at most this much before the APs would have to
/// absorb an unreasonable imbalance. See [`row_range_for_discounted`].
pub const MATMUL_BSP_DISCOUNT_PCT_CEILING: usize = 90;

/// A half-open range of output-matrix rows `[start, end)`. Used as the
/// unit of work distribution for parallel matmul.
///
/// **Invariant:** `start <= end <= out_features`. A `RowRange` with
/// `start == end` is a no-op slice (assigned to cores that fall off the
/// end when rows < cores) and the worker must handle it gracefully.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct RowRange {
    pub start: usize,
    pub end: usize,
}

impl RowRange {
    /// Number of rows in this range. Returns `0` for empty/sentinel ranges.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// True iff this range carries no work.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Split `total_rows` across `n_cores` into contiguous, near-even ranges.
///
/// The first `total_rows % n` cores get `base + 1` rows, the rest `base`;
/// ranges are contiguous and cover `[0, total_rows)`. Contiguous (rather
/// than strided) ownership keeps the dispatcher simple and the K-order
/// invariant intact — each output row is still reduced by exactly one
/// core in the same per-block order as the scalar path.
#[inline]
pub fn split_rows(total_rows: usize, n_cores: u32) -> [RowRange; MAX_CORES] {
    let mut out = [RowRange { start: 0, end: 0 }; MAX_CORES];
    let n = (n_cores as usize).clamp(1, MAX_CORES);

    if total_rows == 0 {
        return out;
    }

    let base = total_rows / n;
    let extra = total_rows % n;

    let mut cursor = 0usize;
    let mut i = 0usize;
    while i < n {
        let take = base + if i < extra { 1 } else { 0 };
        let start = cursor;
        let end = cursor + take;
        out[i] = RowRange { start, end };
        cursor = end;
        i += 1;
    }
    // Sanity (debug only): cursor should equal total_rows.
    debug_assert!(cursor == total_rows);
    out
}

/// Split `total_rows` across `n_cores` with every range boundary (except
/// the final `total_rows`) aligned to a multiple of `align`.
///
/// Used by the `.smodel`-v2 interleaved kernels: a 4-row group is one
/// streaming unit, so a split boundary inside a group would force two
/// cores to re-stream the same 72/136-byte group-blocks. Alignment
/// changes ONLY row→core ownership, never the per-row reduction order —
/// bit-exactness is unaffected (same argument as [`split_rows`]).
///
/// Implementation: distribute `ceil(total_rows / align)` *groups* with
/// the standard even split, then scale back to rows and clamp the tail.
#[inline]
pub fn split_rows_aligned(total_rows: usize, n_cores: u32, align: usize) -> [RowRange; MAX_CORES] {
    if align <= 1 {
        return split_rows(total_rows, n_cores);
    }
    let groups = total_rows.div_ceil(align);
    let mut out = split_rows(groups, n_cores);
    let mut i = 0usize;
    while i < MAX_CORES {
        let start = out[i].start * align;
        let end = out[i].end * align;
        out[i] = RowRange {
            start: if start > total_rows {
                total_rows
            } else {
                start
            },
            end: if end > total_rows { total_rows } else { end },
        };
        i += 1;
    }
    out
}

/// Closed-form variant of [`split_rows_aligned`]: the row range of
/// participant `i` of `n_cores`, without materialising the full
/// `[RowRange; MAX_CORES]` array (2 KiB of zero-init per call — pure
/// dispatch overhead at ~113 dispatches/token, see PERF report N6).
///
/// Equivalence with [`split_rows`]: that function hands the first
/// `units % n` participants `base + 1` units and the rest `base`, with
/// contiguous cursors — so participant `i` starts at
/// `i * base + min(i, extra)` and takes `base + (i < extra)`. The
/// debug assertion in [`split_rows`]'s tests plus
/// `closed_form_range_matches_split_rows` below pin this down.
#[inline(always)]
pub fn row_range_for(total_rows: usize, n_cores: u32, i: usize, align: usize) -> RowRange {
    let n = (n_cores as usize).clamp(1, MAX_CORES);
    if total_rows == 0 || i >= n {
        return RowRange { start: 0, end: 0 };
    }
    let unit = align.max(1);
    let units = if unit == 1 {
        total_rows
    } else {
        total_rows.div_ceil(unit)
    };
    let base = units / n;
    let extra = units % n;
    let start_u = i * base + if i < extra { i } else { extra };
    let end_u = start_u + base + if i < extra { 1 } else { 0 };
    let start = (start_u * unit).min(total_rows);
    let end = (end_u * unit).min(total_rows);
    RowRange { start, end }
}

/// [`row_range_for`] with the BSP's share shrunk by
/// `bsp_discount_pct` percent (N4 from the qwen-perf-v2 PERF report).
///
/// Rationale: participant 0 is the BSP — it pays the publish cost
/// before computing its slice and the barrier wait after, so with an
/// even split it finishes last and every AP idles on the barrier for
/// the BSP's overhead. Shrinking the BSP slice rebalances wall-clock.
/// The right discount is empirical (Cherry A/B via `smp tune
/// bsp-discount N`); the default 0 keeps ranges bit-identical to
/// [`row_range_for`].
///
/// Ownership-only change: per-row reduction order is untouched, so
/// bit-exactness is unaffected for ANY discount (same argument as
/// [`split_rows`]).
///
/// Construction: in unit space (`align`-sized groups), the BSP takes
/// `floor(units * (100 - d) / (100 * n))` units — its even share
/// scaled by (100-d)% — and the remaining units distribute over the
/// `n - 1` APs with the standard even split.
#[inline(always)]
pub fn row_range_for_discounted(
    total_rows: usize,
    n_cores: u32,
    i: usize,
    align: usize,
    bsp_discount_pct: usize,
) -> RowRange {
    let n = (n_cores as usize).clamp(1, MAX_CORES);
    if bsp_discount_pct == 0 || n <= 1 {
        return row_range_for(total_rows, n_cores, i, align);
    }
    if total_rows == 0 || i >= n {
        return RowRange { start: 0, end: 0 };
    }
    let d = bsp_discount_pct.min(MATMUL_BSP_DISCOUNT_PCT_CEILING);
    let unit = align.max(1);
    let units = if unit == 1 {
        total_rows
    } else {
        total_rows.div_ceil(unit)
    };
    let bsp_units = units * (100 - d) / (100 * n);
    let rem = units - bsp_units;
    let m = n - 1;
    let (start_u, end_u) = if i == 0 {
        (0, bsp_units)
    } else {
        let j = i - 1;
        let base = rem / m;
        let extra = rem % m;
        let s = bsp_units + j * base + if j < extra { j } else { extra };
        (s, s + base + if j < extra { 1 } else { 0 })
    };
    RowRange {
        start: (start_u * unit).min(total_rows),
        end: (end_u * unit).min(total_rows),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_rows_distributes_evenly() {
        let r = split_rows(100, 8);
        let mut total = 0usize;
        for item in r.iter().take(8) {
            total += item.len();
        }
        assert_eq!(total, 100);
        // Imbalance ≤ 1.
        let max_len = r.iter().take(8).map(|item| item.len()).max().unwrap();
        let min_len = r.iter().take(8).map(|item| item.len()).min().unwrap();
        assert!(max_len - min_len <= 1);
    }

    #[test]
    fn split_rows_handles_more_cores_than_rows() {
        let r = split_rows(3, 8);
        // First 3 cores get 1 row each, rest get 0.
        for item in r.iter().take(3) {
            assert_eq!(item.len(), 1);
        }
        for item in r[3..8].iter() {
            assert_eq!(item.len(), 0);
        }
    }

    #[test]
    fn split_rows_zero_rows() {
        let r = split_rows(0, 8);
        for item in r.iter().take(8) {
            assert!(item.is_empty());
        }
    }

    #[test]
    fn split_rows_aligned_keeps_boundaries_on_group_multiples() {
        // 151,936 LM-head rows across 64 cores, group 4.
        let r = split_rows_aligned(151_936, 64, 4);
        let mut total = 0usize;
        for (i, item) in r.iter().take(64).enumerate() {
            assert_eq!(item.start % 4, 0);
            assert_eq!(item.end % 4, 0);
            if i > 0 {
                assert_eq!(item.start, r[i - 1].end);
            }
            total += item.len();
        }
        assert_eq!(total, 151_936);
    }

    #[test]
    fn split_rows_aligned_handles_non_multiple_totals() {
        // 103 rows, group 4 → 26 groups; final boundary clamps to 103.
        let r = split_rows_aligned(103, 8, 4);
        let mut total = 0usize;
        let mut cursor = 0usize;
        for item in r.iter().take(8) {
            assert_eq!(item.start, cursor);
            if item.end != 103 {
                assert_eq!(item.end % 4, 0);
            }
            cursor = item.end.max(cursor);
            total += item.len();
        }
        assert_eq!(total, 103);
    }

    #[test]
    fn closed_form_range_matches_split_rows() {
        for &(total, n, align) in &[
            (151_936usize, 64u32, 1usize),
            (151_936, 64, 4),
            (2048, 64, 4),
            (1024, 17, 4),
            (6144, 64, 4),
            (100, 8, 1),
            (103, 8, 4),
            (3, 8, 1),
            (3, 8, 4),
            (0, 8, 4),
        ] {
            let arr = split_rows_aligned(total, n, align);
            for (i, item) in arr.iter().take(n as usize).enumerate() {
                let cf = row_range_for(total, n, i, align);
                assert_eq!(
                    (cf.start, cf.end),
                    (item.start, item.end),
                    "mismatch at total={total} n={n} align={align} i={i}"
                );
            }
        }
    }

    #[test]
    fn discounted_zero_matches_closed_form() {
        for &(total, n, align) in &[
            (151_936usize, 64u32, 1usize),
            (151_936, 64, 4),
            (2048, 64, 4),
            (1024, 17, 4),
            (103, 8, 4),
            (3, 8, 1),
            (0, 8, 4),
        ] {
            for i in 0..(n as usize) {
                let plain = row_range_for(total, n, i, align);
                let disc = row_range_for_discounted(total, n, i, align, 0);
                assert_eq!(
                    (disc.start, disc.end),
                    (plain.start, plain.end),
                    "discount=0 must be bit-identical: total={total} n={n} align={align} i={i}"
                );
            }
        }
    }

    #[test]
    fn discounted_partition_is_contiguous_and_complete() {
        for &(total, n, align) in &[
            (151_936usize, 64u32, 1usize),
            (151_936, 64, 4),
            (2048, 64, 4),
            (6144, 64, 4),
            (1024, 17, 4),
            (103, 8, 4),
            (16, 64, 4),
            (3, 8, 1),
            (0, 8, 4),
        ] {
            for &d in &[5usize, 10, 25, 50, 90, 100 /* clamped to 90 */] {
                let mut cursor = 0usize;
                for i in 0..(n as usize) {
                    let r = row_range_for_discounted(total, n, i, align, d);
                    assert!(
                        r.start <= r.end,
                        "inverted range: total={total} n={n} align={align} d={d} i={i}"
                    );
                    assert_eq!(
                        r.start, cursor,
                        "gap/overlap: total={total} n={n} align={align} d={d} i={i}"
                    );
                    if r.end != total && align > 1 {
                        assert_eq!(r.end % align, 0, "unaligned boundary");
                    }
                    cursor = r.end;
                }
                assert_eq!(
                    cursor, total,
                    "partition incomplete: total={total} n={n} align={align} d={d}"
                );
            }
        }
    }

    #[test]
    fn discounted_bsp_share_shrinks() {
        // 64 cores on the LM head, 25 % discount: the BSP range must be
        // strictly smaller than the even share, and the AP ranges
        // absorb the difference.
        let even = row_range_for(151_936, 64, 0, 4);
        let disc = row_range_for_discounted(151_936, 64, 0, 4, 25);
        assert!(disc.len() < even.len());
        let ap_even = row_range_for(151_936, 64, 1, 4);
        let ap_disc = row_range_for_discounted(151_936, 64, 1, 4, 25);
        assert!(ap_disc.len() >= ap_even.len());
    }

    #[test]
    fn split_rows_aligned_align_one_matches_plain_split() {
        let a = split_rows_aligned(100, 8, 1);
        let b = split_rows(100, 8);
        for (ai, bi) in a.iter().take(8).zip(b.iter().take(8)) {
            assert_eq!(ai.start, bi.start);
            assert_eq!(ai.end, bi.end);
        }
    }
}
