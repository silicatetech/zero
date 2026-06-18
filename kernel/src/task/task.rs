// SPDX-License-Identifier: AGPL-3.0-or-later
//! `Task` + `TaskId` — Stage 3 scaffolding (V3 arena-based).
//!
//! A `Task` owns a pinned, arena-allocated future and the identity we
//! use to refer to it inside the executor's data structures. The
//! future is allocated into `KERNEL_ARENA` via `arena_static_alloc`,
//! eliminating all heap usage (V3 Part 2 mandate).

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::{Context, Poll};

use quarks_arena::ArenaError;

use crate::memory::arena_static_alloc;

/// Monotonic identity for a task.
///
/// `0` is reserved as the "empty slot" sentinel used later by the
/// ready-queue, so fresh IDs start at 1. The inner counter never
/// resets, so IDs are globally unique over the lifetime of the
/// kernel (2^64 tasks before wraparound — not a practical concern).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

impl TaskId {
    /// Allocate a fresh ID. Thread- and ISR-safe via `AtomicU64`.
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        // `Relaxed` is sufficient: we only need atomicity of the
        // increment, not any ordering relative to other memory ops.
        TaskId(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// Raw numeric value. Used by the ring queue to store IDs in
    /// `AtomicU64` slots (with `0` meaning "empty").
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Reconstruct a `TaskId` from a raw `u64` read out of the
    /// ready queue.  Only valid for values ≥ 1 (0 is the empty-slot
    /// sentinel and must never be wrapped in a `TaskId`).
    ///
    /// `pub(crate)` — only the queue module needs this.
    #[inline]
    pub(crate) fn from_raw(val: u64) -> Self {
        debug_assert!(val != 0, "TaskId::from_raw called with sentinel 0");
        TaskId(val)
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

/// A unit of async work owned by the executor.
///
/// The future is allocated in `KERNEL_ARENA` and accessed via a
/// `'static` mutable reference. This eliminates the need for a heap
/// allocator (`Box::pin`) — V3 Part 2 mandates arena-only allocation
/// in Ring-0.
///
/// # Memory behavior
///
/// Each `Task::new` call consumes arena memory equal to the size of
/// the future plus alignment overhead. When a task completes
/// (`*slot = None` in the executor), the arena memory is NOT freed —
/// it accumulates until the kernel halts. This is acceptable for the
/// current Stage 9 workload (max 32 tasks, typical future sizes
/// 50-200 bytes → max ~6 KiB accumulation, 0.15% of 4 MiB arena).
pub struct Task {
    pub id: TaskId,
    future: Pin<&'static mut (dyn Future<Output = ()> + Send)>,
}

impl Task {
    /// Create a new task. The future is allocated in `KERNEL_ARENA`.
    ///
    /// # Errors
    ///
    /// Returns `ArenaError::OutOfMemory` if the arena cannot fit the
    /// future.
    pub fn new<F>(future: F) -> Result<Self, ArenaError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        // Allocate the concrete future into the arena.
        let future_ref: &'static mut F = arena_static_alloc(future)?;

        // SAFETY: Pin::new_unchecked is sound here because:
        //
        // 1. STABLE ADDRESS: future_ref points into KERNEL_ARENA
        //    backing memory, which is page-mapped at init and never
        //    unmapped, moved, or reallocated. The address is stable
        //    for the kernel's entire lifetime.
        //
        // 2. NO MOVE AFTER PIN: The arena never relocates allocations
        //    (bump allocator, no compaction). Once pinned, the future
        //    stays at its arena address until the kernel halts.
        //
        // 3. UNIQUE REFERENCE: arena_static_alloc returns a unique
        //    &'static mut — no aliasing. We immediately consume it
        //    into Pin, which prevents further moves.
        //
        // The coercion &'static mut F -> &'static mut dyn Future
        // performs type erasure (vtable construction). This is the
        // same pattern Box::pin uses internally; the only difference
        // is arena backing instead of heap backing.
        let dyn_ref: &'static mut (dyn Future<Output = ()> + Send) = future_ref;
        let pinned = unsafe { Pin::new_unchecked(dyn_ref) };

        Ok(Task {
            id: TaskId::new(),
            future: pinned,
        })
    }

    /// Poll the task's future once.
    ///
    /// `Poll::Pending` means the future parked itself on a waker and
    /// is not currently runnable; `Poll::Ready(())` means the future
    /// is done and the task can be dropped by the caller.
    pub fn poll(&mut self, context: &mut Context) -> Poll<()> {
        self.future.as_mut().poll(context)
    }
}
