// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chunked bump arena with doubling growth.
//!
//! Used for compile contexts and runtime allocations where workload
//! size is not known at construction. Each new chunk is twice as
//! large as the previous, up to a maximum total capacity.
//!
//! Pointers into previously-allocated values remain valid as new
//! chunks are added; no relocation occurs.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::{align_of, size_of, MaybeUninit};
use core::ptr::NonNull;
use core::slice;

use crate::error::ArenaError;
use crate::fixed::FixedArena;

/// Source of new chunks for a [`ChunkedArena`].
pub trait ChunkSource {
    /// Request a new chunk of at least `min_size` bytes.
    /// The returned slice's lifetime must be `'static`.
    fn provide_chunk(&mut self, min_size: usize) -> Result<&'static mut [u8], ArenaError>;
}

/// Chunked bump arena with growth via linked chunks.
pub struct ChunkedArena {
    /// All chunks in order of creation. Newest is last.
    chunks: Vec<FixedArena>,
    /// Index of the active allocation chunk. Points to the last chunk.
    active_chunk: usize,
    /// Maximum total bytes across all chunks.
    max_capacity: usize,
    /// Source for new chunks when growth is needed.
    chunk_source: Box<dyn ChunkSource>,
    /// Size of the next chunk to request (doubles each growth).
    next_chunk_size: usize,
}

impl ChunkedArena {
    /// Construct a new chunked arena.
    pub fn new(
        initial_chunk: &'static mut [u8],
        max_capacity: usize,
        chunk_source: Box<dyn ChunkSource>,
    ) -> Self {
        let initial_size = initial_chunk.len();
        let initial = FixedArena::new(initial_chunk);
        let chunks = vec![initial];
        ChunkedArena {
            chunks,
            active_chunk: 0,
            max_capacity,
            chunk_source,
            next_chunk_size: initial_size.saturating_mul(2),
        }
    }

    /// Total bytes used across all chunks.
    pub fn used(&self) -> usize {
        self.chunks.iter().map(|c| c.used()).sum()
    }

    /// Total currently-allocated capacity (sum of chunk capacities).
    pub fn current_capacity(&self) -> usize {
        self.chunks.iter().map(|c| c.capacity()).sum()
    }

    /// Maximum total capacity including future growth.
    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// Number of chunks currently allocated.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Reset all chunks. Allocations are invalidated. Destructors
    /// are NOT called. Existing chunks are kept (memory is not
    /// returned to the chunk source).
    pub fn reset(&mut self) {
        for chunk in &mut self.chunks {
            chunk.reset();
        }
        self.active_chunk = 0;
    }

    /// Allocate a value.
    pub fn alloc<T>(&mut self, value: T) -> Result<&mut T, ArenaError> {
        let uninit = self.alloc_uninit::<T>()?;
        // SAFETY: `uninit` points to space sized and aligned for T;
        // we initialize before returning.
        unsafe {
            let ptr = uninit.as_mut_ptr();
            ptr.write(value);
            Ok(&mut *ptr)
        }
    }

    /// Allocate uninitialized space.
    pub fn alloc_uninit<T>(&mut self) -> Result<&mut MaybeUninit<T>, ArenaError> {
        let size = size_of::<T>();
        let align = align_of::<T>();
        let ptr = self.alloc_raw_chunked(size, align)?;
        // SAFETY: `alloc_raw_chunked` returns properly sized and
        // aligned memory. MaybeUninit<T> has the same layout as T.
        unsafe { Ok(&mut *(ptr.as_ptr() as *mut MaybeUninit<T>)) }
    }

    /// Allocate a slice copy.
    pub fn alloc_slice_copy<T: Copy>(&mut self, slice: &[T]) -> Result<&mut [T], ArenaError> {
        let len = slice.len();
        if len == 0 {
            return Ok(&mut []);
        }
        let size = size_of::<T>()
            .checked_mul(len)
            .ok_or(ArenaError::SizeOverflow)?;
        let align = align_of::<T>();
        let ptr = self.alloc_raw_chunked(size, align)?;
        // SAFETY: `alloc_raw_chunked` returns properly sized and
        // aligned memory; we copy `len` T values into it.
        unsafe {
            let dst = ptr.as_ptr() as *mut T;
            core::ptr::copy_nonoverlapping(slice.as_ptr(), dst, len);
            Ok(slice::from_raw_parts_mut(dst, len))
        }
    }

    /// Allocate `len` bytes in the arena, zero-initialised in place.
    ///
    /// Pre-A.2 (F-H8): replaces the
    /// `alloc_slice_copy(&vec![0u8; len])` Ring-0 footgun. The temp
    /// `Vec<u8>` would allocate (and immediately free) `len` bytes on
    /// the global allocator just to copy zeros into the arena. With
    /// `alloc_zeroed`, the zeros are written straight into the arena
    /// via `core::ptr::write_bytes`, so the only allocation is the
    /// arena bump itself.
    pub fn alloc_zeroed(&mut self, len: usize) -> Result<&mut [u8], ArenaError> {
        if len == 0 {
            return Ok(&mut []);
        }
        let ptr = self.alloc_raw_chunked(len, 1)?;
        // SAFETY: `alloc_raw_chunked` returns `len` valid bytes;
        // `write_bytes` zero-initialises every byte in place.
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0u8, len);
            Ok(slice::from_raw_parts_mut(ptr.as_ptr(), len))
        }
    }

    /// Allocate a string.
    pub fn alloc_str(&mut self, s: &str) -> Result<&mut str, ArenaError> {
        let bytes = self.alloc_slice_copy(s.as_bytes())?;
        // SAFETY: bytes copied from valid str, valid UTF-8 preserved.
        unsafe { Ok(core::str::from_utf8_unchecked_mut(bytes)) }
    }

    /// Internal: try to allocate from active chunk, grow if needed,
    /// retry once.
    ///
    /// This is the central allocation primitive. All public alloc-*
    /// methods delegate to this.
    fn alloc_raw_chunked(&mut self, size: usize, align: usize) -> Result<NonNull<u8>, ArenaError> {
        // First attempt: active chunk.
        match self.chunks[self.active_chunk].alloc_raw(size, align) {
            Ok(ptr) => return Ok(ptr),
            Err(ArenaError::OutOfMemory { .. }) => {
                // Fall through to growth path.
            }
            Err(other) => return Err(other),
        }
        // Growth: request new chunk large enough to fit the requested
        // allocation (size + alignment slack as worst-case overhead).
        let needed = size.checked_add(align).ok_or(ArenaError::SizeOverflow)?;
        self.grow(needed)?;
        // Retry from new active chunk. Must succeed because the new
        // chunk was sized to fit `needed` bytes.
        self.chunks[self.active_chunk].alloc_raw(size, align)
    }

    /// Internal: grow by adding a new chunk.
    fn grow(&mut self, min_size: usize) -> Result<(), ArenaError> {
        // Determine size for new chunk: max(next_chunk_size, min_size).
        let chunk_size = core::cmp::max(self.next_chunk_size, min_size);
        // Check that adding this chunk does not exceed max_capacity.
        let projected_capacity = self
            .current_capacity()
            .checked_add(chunk_size)
            .ok_or(ArenaError::SizeOverflow)?;
        if projected_capacity > self.max_capacity {
            return Err(ArenaError::OutOfMemory {
                requested: chunk_size,
                available: self.max_capacity.saturating_sub(self.current_capacity()),
            });
        }
        let new_backing = self.chunk_source.provide_chunk(chunk_size)?;
        let new_arena = FixedArena::new(new_backing);
        self.chunks.push(new_arena);
        self.active_chunk = self.chunks.len() - 1;
        self.next_chunk_size = self.next_chunk_size.saturating_mul(2);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Owning test `ChunkSource` that holds its allocations and
    /// frees them on Drop.
    ///
    /// Bug 6 / QUARKS_REVIEW.md §2.1 fix. The previous
    /// `LeakingChunkSource` called `Box::leak` on every
    /// `provide_chunk`, which means every test run permanently
    /// leaked memory equal to the arena footprint. The owning
    /// source keeps the `Box<[u8]>` chunks in its `chunks` field
    /// and reclaims them when the source is dropped (which
    /// happens when the enclosing `ChunkedArena` drops, because
    /// the arena holds the source as a `Box<dyn ChunkSource>` and
    /// the fields drop in declaration order: `chunks` first,
    /// `chunk_source` second).
    struct OwningTestChunkSource {
        chunks: Vec<alloc::boxed::Box<[u8]>>,
    }

    impl OwningTestChunkSource {
        fn new() -> Self {
            Self { chunks: Vec::new() }
        }

        fn allocate(&mut self, size: usize) -> &'static mut [u8] {
            let mut boxed: alloc::boxed::Box<[u8]> = vec![0u8; size].into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            let len = boxed.len();
            self.chunks.push(boxed);
            // SAFETY: the box now lives in `self.chunks`, which is
            // owned by the arena via its `Box<dyn ChunkSource>`
            // field. The arena drops `chunks: Vec<FixedArena>`
            // before its `chunk_source` field, so the FixedArenas
            // (which alias these bytes via NonNull) drop before
            // this storage. The 'static lifetime is a contract
            // limitation, not a literal claim.
            unsafe { core::slice::from_raw_parts_mut(ptr, len) }
        }
    }

    impl ChunkSource for OwningTestChunkSource {
        fn provide_chunk(&mut self, min_size: usize) -> Result<&'static mut [u8], ArenaError> {
            Ok(self.allocate(min_size))
        }
    }

    /// Helper constructing an arena with the owning chunk source.
    /// Replaces the previous `(make_initial(N), Box::new(LeakingChunkSource))`
    /// pattern so test runs no longer accumulate leaked storage.
    fn make_arena(initial: usize, max: usize) -> ChunkedArena {
        let mut source = OwningTestChunkSource::new();
        let initial_chunk = source.allocate(initial);
        ChunkedArena::new(initial_chunk, max, Box::new(source))
    }

    #[test]
    fn empty_chunked_arena_has_initial_capacity() {
        let arena = make_arena(1024, 4096);
        assert_eq!(arena.current_capacity(), 1024);
        assert_eq!(arena.max_capacity(), 4096);
        assert_eq!(arena.chunk_count(), 1);
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn alloc_within_initial_chunk() {
        let mut arena = make_arena(1024, 4096);
        let r = arena.alloc(42u64).unwrap();
        assert_eq!(*r, 42);
        assert_eq!(arena.chunk_count(), 1);
    }

    #[test]
    fn growth_when_initial_chunk_full() {
        let mut arena = make_arena(16, 1024);
        arena.alloc(1u64).unwrap();
        arena.alloc(2u64).unwrap();
        assert_eq!(arena.chunk_count(), 1);
        arena.alloc(3u64).unwrap();
        assert_eq!(arena.chunk_count(), 2);
    }

    #[test]
    fn pointer_stability_across_growth() {
        let mut arena = make_arena(16, 1024);
        let r1: *const u64 = arena.alloc(1u64).unwrap();
        let r2: *const u64 = arena.alloc(2u64).unwrap();
        let _r3 = arena.alloc(3u64).unwrap();
        // SAFETY: pointers into chunked arena remain valid until reset/drop.
        unsafe {
            assert_eq!(*r1, 1);
            assert_eq!(*r2, 2);
        }
    }

    #[test]
    fn growth_doubles_chunk_size() {
        let mut arena = make_arena(16, 1024);
        arena.alloc(1u64).unwrap();
        arena.alloc(2u64).unwrap();
        arena.alloc(3u64).unwrap();
        // Initial: 16, second chunk: 32 (doubled). Total: 48.
        assert_eq!(arena.current_capacity(), 48);
    }

    #[test]
    fn oom_when_max_capacity_exceeded() {
        let mut arena = make_arena(16, 32);
        arena.alloc(1u64).unwrap();
        arena.alloc(2u64).unwrap();
        // Growth would request 32 bytes; total would be 48 > 32 max.
        let result = arena.alloc(3u64);
        assert!(matches!(result, Err(ArenaError::OutOfMemory { .. })));
    }

    #[test]
    fn reset_clears_all_chunks() {
        let mut arena = make_arena(16, 1024);
        arena.alloc(1u64).unwrap();
        arena.alloc(2u64).unwrap();
        arena.alloc(3u64).unwrap();
        assert_eq!(arena.chunk_count(), 2);
        let cap_before = arena.current_capacity();
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.chunk_count(), 2);
        assert_eq!(arena.current_capacity(), cap_before);
    }

    #[test]
    fn alloc_after_reset_reuses_chunks() {
        let mut arena = make_arena(16, 1024);
        arena.alloc(1u64).unwrap();
        arena.alloc(2u64).unwrap();
        arena.alloc(3u64).unwrap();
        arena.reset();
        let r = arena.alloc(99u64).unwrap();
        assert_eq!(*r, 99);
    }

    #[test]
    fn alloc_slice_copy_across_chunks() {
        let mut arena = make_arena(16, 1024);
        arena.alloc(1u64).unwrap();
        let src = [1u64, 2, 3];
        let dst = arena.alloc_slice_copy(&src).unwrap();
        assert_eq!(dst, &[1, 2, 3]);
        assert_eq!(arena.chunk_count(), 2);
    }

    #[test]
    fn alloc_str_basic() {
        let mut arena = make_arena(1024, 4096);
        let s = arena.alloc_str("hello world").unwrap();
        assert_eq!(s, "hello world");
    }

    #[test]
    fn large_allocation_triggers_appropriately_sized_chunk() {
        let mut arena = make_arena(16, 4096);
        let src = vec![1u8; 100];
        let dst = arena.alloc_slice_copy(&src).unwrap();
        assert_eq!(dst.len(), 100);
        // Chunk must be at least 100 bytes.
        assert!(arena.current_capacity() >= 116);
    }

    #[test]
    fn alloc_uninit_works_through_chunked_path() {
        let mut arena = make_arena(1024, 4096);
        let slot = arena.alloc_uninit::<u32>().unwrap();
        slot.write(42);
        // SAFETY: just wrote.
        let val = unsafe { slot.assume_init_ref() };
        assert_eq!(*val, 42);
    }

    // === Pre-A.2 (F-H8): alloc_zeroed across chunked path ===

    #[test]
    fn alloc_zeroed_initialises_bytes_to_zero_chunked() {
        let mut arena = make_arena(1024, 4096);
        let bytes = arena.alloc_zeroed(64).unwrap();
        assert_eq!(bytes.len(), 64);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn alloc_zeroed_zero_length_is_noop_chunked() {
        let mut arena = make_arena(64, 256);
        let used_before = arena.used();
        let bytes = arena.alloc_zeroed(0).unwrap();
        assert_eq!(bytes.len(), 0);
        assert_eq!(arena.used(), used_before);
    }

    #[test]
    fn alloc_zeroed_triggers_growth_when_initial_chunk_full() {
        // Initial chunk = 16 bytes. Asking for 32 bytes forces a
        // grow; the new chunk must still come back zero-initialised.
        let mut arena = make_arena(16, 256);
        let _ = arena.alloc(1u64).unwrap(); // 8 bytes consumed
        let zeros = arena.alloc_zeroed(32).unwrap();
        assert_eq!(zeros.len(), 32);
        assert!(zeros.iter().all(|&b| b == 0));
        assert!(arena.chunk_count() >= 2, "growth must have occurred");
    }

    #[test]
    fn alloc_zeroed_oom_when_max_capacity_exceeded() {
        let mut arena = make_arena(16, 32);
        // 16 bytes available in initial; ask for 64 which exceeds
        // the 32-byte cap.
        let res = arena.alloc_zeroed(64);
        assert!(matches!(res, Err(ArenaError::OutOfMemory { .. })));
    }

    // === Bug 6 (QUARKS_REVIEW.md §2.1) regression ===
    //
    // The owning chunk source must actually own its storage —
    // dropping the arena must trigger the source's drop, which
    // releases every chunk back to the global allocator. The leak
    // is not observable from inside the test (no heap counter is
    // exposed in `no_std + alloc`); instead we exercise the drop
    // path explicitly and confirm no UAF / no panic. Miri runs of
    // this test (if/when enabled) catch the leak as `leaked
    // allocation`.

    #[test]
    fn arena_drops_cleanly_after_growth() {
        let mut arena = make_arena(16, 1024);
        // Force at least one growth so the source's storage Vec
        // has more than just the initial entry.
        arena.alloc(1u64).unwrap();
        arena.alloc(2u64).unwrap();
        arena.alloc(3u64).unwrap();
        assert!(arena.chunk_count() >= 2);
        // Explicit drop. If `OwningTestChunkSource` did not own
        // its storage, this would still succeed structurally;
        // the value lands when we run this test under Miri.
        drop(arena);
    }

    #[test]
    fn arena_drops_cleanly_when_used_in_loop() {
        // Repeated construction + drop. Under the old
        // `LeakingChunkSource` this leaked ~256 KiB per iteration
        // and accumulated unboundedly. With the owning source the
        // top-of-heap stays flat.
        for _ in 0..32 {
            let mut arena = make_arena(64, 256);
            for _ in 0..8 {
                let _ = arena.alloc(0u64);
            }
            // Implicit drop at end of scope.
        }
    }
}
