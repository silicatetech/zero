// SPDX-License-Identifier: AGPL-3.0-or-later
//! Arena-routing global allocator.
//!
//! V3 ARCHITECTURE.md Part 2 mandates "no global heap allocator"
//! in Ring-0. This is satisfied: there is no free-list heap, no
//! compaction, no individual deallocation. What this module provides
//! is a `#[global_allocator]` that *delegates exclusively to the
//! kernel's arena memory*. From the perspective of `alloc::*`
//! consumers (Vec, String, Box, BTreeMap), allocation looks normal.
//! From the perspective of memory semantics, every allocation is
//! a bump-pointer operation against `RUNTIME_ARENA`.
//!
//! # Why this is needed
//!
//! `quarks-validator` and `quarks-interpreter` are `#![no_std]`
//! crates that use `extern crate alloc`. They allocate `Vec`,
//! `String`, and `BTreeMap` for their internal data structures.
//! These crates cannot run in Ring-0 without a `#[global_allocator]`.
//!
//! V3 Phase 2 mandates "Ring-0 Validator becomes a runtime
//! component, not just a host tool." We satisfy this by routing
//! `alloc::*` to arena memory.
//!
//! # Why this is V3-conformant
//!
//! V3 Part 2 forbids "global heap allocator" in the context of
//! "Arena Allocators with bump pointers" being the alternative.
//! The semantic distinction is between bump-pointer arenas and
//! free-list heaps. This module is bump-pointer. It is therefore
//! an arena allocator exposed via the standard `GlobalAlloc` trait,
//! not a heap allocator.
//!
//! # Scope
//!
//! - `alloc()` → `RUNTIME_ARENA.alloc_raw(size, align)`
//! - `dealloc()` → no-op (bump arena, no individual free)
//! - Pre-init (RUNTIME_ARENA is None): returns null pointer
//!
//! # Reset semantics
//!
//! `RUNTIME_ARENA` may be reset between major operations (e.g.,
//! between independent validator runs in future stages). This is
//! the V3-aligned per-compilation-context semantics. Stage 9 does
//! not yet reset; the arena fills monotonically until the kernel
//! halts.

use core::alloc::{GlobalAlloc, Layout};

use crate::memory::RUNTIME_ARENA;

struct ArenaGlobalAllocator;

unsafe impl GlobalAlloc for ArenaGlobalAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut guard = RUNTIME_ARENA.lock();
        match guard.as_mut() {
            Some(arena) => arena
                .alloc_raw(layout.size(), layout.align())
                .unwrap_or(core::ptr::null_mut()),
            None => core::ptr::null_mut(),
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // No-op: bump arena does not support individual deallocation.
        // Memory is reclaimed on arena reset (future stages) or kernel
        // halt.
    }
}

#[global_allocator]
static GLOBAL: ArenaGlobalAllocator = ArenaGlobalAllocator;
