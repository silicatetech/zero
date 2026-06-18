// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(feature = "chunked")]
//! Integration tests for ChunkedArena.

use quarks_arena::{ArenaError, ChunkSource, ChunkedArena};

struct TestChunkSource;

impl ChunkSource for TestChunkSource {
    fn provide_chunk(&mut self, min_size: usize) -> Result<&'static mut [u8], ArenaError> {
        let v = vec![0u8; min_size].into_boxed_slice();
        Ok(Box::leak(v))
    }
}

fn make_initial(size: usize) -> &'static mut [u8] {
    Box::leak(vec![0u8; size].into_boxed_slice())
}

#[test]
fn integration_growth_workflow() {
    let mut arena = ChunkedArena::new(make_initial(1024), 16 * 1024, Box::new(TestChunkSource));
    for i in 0..200u64 {
        arena.alloc(i).unwrap();
    }
    assert!(arena.chunk_count() >= 2);
}

#[test]
fn integration_reset_does_not_drop_chunks() {
    let mut arena = ChunkedArena::new(make_initial(64), 4096, Box::new(TestChunkSource));
    for i in 0..20u64 {
        arena.alloc(i).unwrap();
    }
    let chunks_before_reset = arena.chunk_count();
    let cap_before_reset = arena.current_capacity();
    arena.reset();
    assert_eq!(arena.used(), 0);
    assert_eq!(arena.chunk_count(), chunks_before_reset);
    assert_eq!(arena.current_capacity(), cap_before_reset);
}

#[test]
fn integration_max_capacity_enforced() {
    let mut arena = ChunkedArena::new(make_initial(16), 24, Box::new(TestChunkSource));
    arena.alloc(1u64).unwrap();
    arena.alloc(2u64).unwrap();
    let result = arena.alloc(3u64);
    assert!(matches!(result, Err(ArenaError::OutOfMemory { .. })));
}

#[test]
fn integration_pointer_stability() {
    let mut arena = ChunkedArena::new(make_initial(16), 4096, Box::new(TestChunkSource));
    let p1: *const u64 = arena.alloc(11u64).unwrap();
    let p2: *const u64 = arena.alloc(22u64).unwrap();
    for i in 0..50u64 {
        arena.alloc(i).unwrap();
    }
    // SAFETY: pointers into chunked arena remain valid until reset/drop.
    unsafe {
        assert_eq!(*p1, 11);
        assert_eq!(*p2, 22);
    }
}
