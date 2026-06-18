// SPDX-License-Identifier: AGPL-3.0-or-later
//! # quarks-arena
//!
//! Bump-allocator arenas for Zero Ring-0 memory management.
//!
//! ## Overview
//!
//! Two arena types serve different memory-management patterns:
//!
//! - [`FixedArena`]: Single fixed-size memory block, no growth.
//!   Returns [`ArenaError::OutOfMemory`] when the backing buffer is
//!   exhausted. Used for kernel-internal allocations where size is
//!   known at boot time. The kernel may translate this error into a
//!   panic at the call site (V3 architecture: kernel-internal arena
//!   exhaustion is fatal), but the arena itself never panics.
//!
//! - [`ChunkedArena`]: Linked chunks with doubling growth. Used for
//!   compile contexts and runtime allocations where workload size is
//!   not known statically. Returns [`ArenaError::OutOfMemory`] only
//!   when total capacity (across all chunks) would exceed the
//!   configured maximum.
//!
//! ## Allocation pattern
//!
//! Both arenas are bump-allocators: allocations are O(1) pointer
//! increments. No deallocation of individual values; the entire arena
//! is reset or dropped as a unit.
//!
//! ## Drop semantics
//!
//! **Important:** Arena allocations do NOT call destructors on
//! `reset()` or `drop()`. Only allocate types that are trivially
//! droppable (`Copy` types, primitives, validated AST nodes), or
//! manually invoke `Drop::drop` on values before resetting the arena.
//! Allocating a `Vec` or `String` into an arena leaks the inner heap
//! allocation when the arena is reset.
//!
//! ## Thread safety
//!
//! Arenas are single-threaded. The `&mut self` requirement on
//! allocation methods enforces this at the type-system level. For
//! multi-threaded use, wrap the arena in a `Mutex` at the call site.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(any(feature = "chunked", test))]
extern crate alloc;

#[cfg(feature = "chunked")]
mod chunked;
mod config;
mod error;
mod fixed;
#[cfg(any(feature = "chunked", test))]
pub mod handle;

#[cfg(feature = "chunked")]
pub use chunked::{ChunkSource, ChunkedArena};
pub use config::{
    COMPILE_ARENA_INITIAL, COMPILE_ARENA_MAX, KERNEL_ARENA_SIZE, RUNTIME_ARENA_INITIAL,
    RUNTIME_ARENA_MAX,
};
pub use error::ArenaError;
pub use fixed::FixedArena;
#[cfg(any(feature = "chunked", test))]
pub use handle::{HandleEntry, HandleError, HandleTable, HandleType};
