// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for FixedArena.

use quarks_arena::{ArenaError, FixedArena};

fn make_backing(size: usize) -> &'static mut [u8] {
    Box::leak(vec![0u8; size].into_boxed_slice())
}

#[test]
fn integration_alloc_and_use() {
    let mut arena = FixedArena::new(make_backing(4096));
    let x = *arena.alloc(100u64).unwrap();
    let y = *arena.alloc(200u64).unwrap();
    assert_eq!(x + y, 300);
}

#[test]
fn integration_oom_handled_gracefully() {
    let mut arena = FixedArena::new(make_backing(8));
    arena.alloc(1u64).unwrap();
    let result = arena.alloc(2u64);
    assert!(matches!(result, Err(ArenaError::OutOfMemory { .. })));
}

#[test]
fn integration_reset_and_reuse_pattern() {
    let mut arena = FixedArena::new(make_backing(64));
    for round in 0..5 {
        for i in 0..4u64 {
            arena.alloc(round * 10 + i).unwrap();
        }
        arena.reset();
    }
    let final_alloc = arena.alloc(999u64).unwrap();
    assert_eq!(*final_alloc, 999);
}

#[test]
fn integration_strings_and_slices() {
    let mut arena = FixedArena::new(make_backing(1024));
    let s = arena.alloc_str("Zero").unwrap();
    assert_eq!(s, "Zero");
    let nums = arena.alloc_slice_copy(&[1u32, 2, 3]).unwrap();
    assert_eq!(nums, &[1, 2, 3]);
}
