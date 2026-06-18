// SPDX-License-Identifier: AGPL-3.0-or-later
//! Default arena size configurations matching ARCHITECTURE.md V3.

/// Kernel-internal arena size: 4 MiB.
///
/// Used for kernel-internal allocations (interpreter state, kernel
/// data structures). Fixed size. The arena returns
/// [`ArenaError::OutOfMemory`] on exhaustion; the kernel translates
/// this into a panic at the call site.
pub const KERNEL_ARENA_SIZE: usize = 4 * 1024 * 1024;

/// Compile-context arena initial chunk size: 1 MiB.
pub const COMPILE_ARENA_INITIAL: usize = 1024 * 1024;

/// Compile-context arena maximum total capacity: 16 MiB.
pub const COMPILE_ARENA_MAX: usize = 16 * 1024 * 1024;

/// Runtime arena initial chunk size: 2 MiB.
pub const RUNTIME_ARENA_INITIAL: usize = 2 * 1024 * 1024;

/// Runtime arena maximum total capacity: 64 MiB.
pub const RUNTIME_ARENA_MAX: usize = 64 * 1024 * 1024;
