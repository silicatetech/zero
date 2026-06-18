// SPDX-License-Identifier: AGPL-3.0-or-later
//! `.smodel`-v2 row-interleaved weight-layout registry (qwen-perf-v2).
//!
//! SilicatePack v2 can emit Q4_0/Q8_0 matrix tensors in a row-interleaved
//! layout (`Q4_0X4` / `Q8_0X4`, SIDX dtype ids 100/101): groups of 4
//! consecutive output rows are stored block-interleaved so the matmul
//! kernel streams ONE contiguous byte sequence while computing 4 output
//! rows per pass (8 independent FMA chains, shared activation loads).
//!
//! The sacred dispatch types (`ForwardPassDispatch`,
//! `zero_gguf_parser::GgmlType`) cannot carry a layout flag — they are
//! frozen crates. Instead the model loader registers the *payload start
//! address* of every interleaved tensor here, and the BSP-side dispatch
//! wrappers in `inference_avx512.rs` look the weight pointer up before
//! choosing a kernel. Lookup is an exact match on the tensor's first
//! byte: the Qwen forward pass always hands whole-tensor slices to the
//! dispatchers, while sub-tensor slices (e.g. Kimi expert slicing into a
//! 3-D fused tensor) intentionally miss and stay on the plain kernels —
//! SilicatePack never interleaves rank-3 tensors.
//!
//! # Synchronisation
//!
//! Registration happens on the BSP during model load, strictly before
//! any SMP matmul dispatch; `COUNT` is published with `Release` and read
//! with `Acquire`. Lookups happen only on the BSP (kernel routing is
//! decided before fan-out), so there is no AP-side access at all.
//!
//! # Bit-exactness
//!
//! Layout routing never changes *what* is computed — the interleaved
//! kernels reduce each output row in the identical per-block FMA order
//! as the plain kernels (see `linear_q4_0x4_avx512_range`). This module
//! only decides *which* kernel walks the bytes.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Row-group width of the v2 interleaved layout.
pub const INTERLEAVE_GROUP: u32 = 4;

/// Upper bound on interleaved tensors per model. Qwen3-1.7B has
/// 28 × 7 + 1 = 197 candidate matmul tensors; 512 leaves headroom for
/// deeper models without growing `.bss` meaningfully (512 × 12 B).
const MAX_ENTRIES: usize = 512;

static COUNT: AtomicUsize = AtomicUsize::new(0);
/// Tensor payload start addresses, sorted ascending. Only
/// `[0..COUNT)` is valid.
static ADDRS: [AtomicUsize; MAX_ENTRIES] = [const { AtomicUsize::new(0) }; MAX_ENTRIES];
/// Interleave group per entry (currently always 4).
static GROUPS: [AtomicU32; MAX_ENTRIES] = [const { AtomicU32::new(0) }; MAX_ENTRIES];

/// Drop all registrations. Called at the start of every model-index
/// build (native `.smodel` AND legacy GGUF) so a model swap can never
/// leave stale addresses behind.
pub fn clear() {
    COUNT.store(0, Ordering::Release);
}

/// Shift every registered tensor address by `delta` bytes.
///
/// The interleave registry is keyed by each tensor's *virtual* payload
/// start address, captured when the `.smodel` directory is parsed. On
/// x86_64 the model payload is subsequently re-exposed at a different
/// virtual address by the hugepage promotion (`contiguous_phys_view`):
/// the same physical bytes, mapped through the phys-linear window for
/// TLB relief. After that remap the matmul fetches weights through the
/// *new* VA, so `group_of(new_ptr)` would miss the registry and the
/// plain (non-interleaved) kernel would silently misread row-interleaved
/// bytes — producing NaNs from the very first matmul. Rebasing by the
/// uniform `new_va - old_va` delta keeps every lookup hitting.
///
/// The shift is uniform (one contiguous remap of the whole payload), so
/// the ascending sort order is preserved and no re-sort is needed.
pub fn rebase(delta: i64) {
    if delta == 0 {
        return;
    }
    let n = COUNT.load(Ordering::Acquire);
    let mut i = 0;
    while i < n {
        let a = ADDRS[i].load(Ordering::Relaxed) as i64;
        ADDRS[i].store(a.wrapping_add(delta) as usize, Ordering::Relaxed);
        i += 1;
    }
}

/// Register one interleaved tensor by payload start address. Keeps the
/// table sorted (insertion sort — boot-time only, ≤ a few hundred
/// entries). Returns `false` when the table is full; the caller must
/// treat that as a load error, not a silent fallback, because a missed
/// registration would make the plain kernel misread interleaved bytes.
pub fn register(addr: usize, group: u32) -> bool {
    let n = COUNT.load(Ordering::Acquire);
    if n >= MAX_ENTRIES {
        return false;
    }
    // Find insertion point.
    let mut pos = n;
    let mut i = 0;
    while i < n {
        if ADDRS[i].load(Ordering::Relaxed) > addr {
            pos = i;
            break;
        }
        i += 1;
    }
    // Shift tail up by one.
    let mut j = n;
    while j > pos {
        ADDRS[j].store(ADDRS[j - 1].load(Ordering::Relaxed), Ordering::Relaxed);
        GROUPS[j].store(GROUPS[j - 1].load(Ordering::Relaxed), Ordering::Relaxed);
        j -= 1;
    }
    ADDRS[pos].store(addr, Ordering::Relaxed);
    GROUPS[pos].store(group, Ordering::Relaxed);
    COUNT.store(n + 1, Ordering::Release);
    true
}

/// Number of registered interleaved tensors.
#[inline]
pub fn count() -> usize {
    COUNT.load(Ordering::Acquire)
}

/// Interleave group for the tensor whose payload starts at `addr`.
/// Returns 1 (plain row-major) when the address is not registered.
/// Binary search over the sorted table — ~8 compares for a Qwen-sized
/// model, BSP-only, ~200 calls per token.
#[inline]
pub fn group_of(addr: usize) -> u32 {
    let n = COUNT.load(Ordering::Acquire);
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let a = ADDRS[mid].load(Ordering::Relaxed);
        if a == addr {
            return GROUPS[mid].load(Ordering::Relaxed);
        } else if a < addr {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_hits_registered_and_misses_unregistered() {
        clear();
        assert!(register(0x5000, 4));
        assert!(register(0x1000, 4));
        assert!(register(0x3000, 4));
        assert_eq!(count(), 3);
        assert_eq!(group_of(0x1000), 4);
        assert_eq!(group_of(0x3000), 4);
        assert_eq!(group_of(0x5000), 4);
        // Interior pointer (expert-slice pattern) must miss.
        assert_eq!(group_of(0x3010), 1);
        assert_eq!(group_of(0x0), 1);
        clear();
        assert_eq!(group_of(0x1000), 1);
    }
}
