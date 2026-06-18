// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lock-free ring-buffer ready queue for the async executor.
//!
//! Signals which tasks are runnable.  The PIT ISR (producer) pushes
//! `TaskId`s when a waker fires; the executor loop (consumer) pops
//! them to decide what to poll next.
//!
//! **Concurrency model:** single-producer, single-consumer (SPSC).
//! See per-method docs for MPMC-safety notes.
//!
//! Slot value `0` is the "empty" sentinel, consistent with `TaskId`
//! starting at 1 (see `task.rs`).

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::TaskId;

/// Number of slots in the ring buffer.
///
/// 64 gives room for 32 tasks plus reserve, avoiding spurious
/// `QueueFull` errors during bursty wake-ups.
const CAPACITY: usize = 64;

/// Error returned when the ring buffer has no free slots.
#[derive(Debug)]
pub struct QueueFull;

/// A fixed-size, lock-free ring buffer of `TaskId` values.
///
/// No `Mutex`, no heap allocation — only atomic operations.  Safe to
/// call `push` from an ISR and `pop` from the main executor loop
/// without disabling interrupts.
pub struct ReadyQueue {
    buffer: [AtomicU64; CAPACITY],
    /// Index of the next slot to *read* (consumer side).
    head: AtomicUsize,
    /// Index of the next slot to *write* (producer side).
    tail: AtomicUsize,
}

impl ReadyQueue {
    /// Create an empty queue.
    ///
    /// Const-evaluable so the queue can live in a `static`.
    pub const fn new() -> Self {
        // `AtomicU64::new` is const; repeating a const item in an
        // array initializer produces one independent atomic per slot.
        const EMPTY_SLOT: AtomicU64 = AtomicU64::new(0);
        ReadyQueue {
            buffer: [EMPTY_SLOT; CAPACITY],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Enqueue a task ID.
    ///
    /// # MPMC safety
    ///
    /// **Not MPMC-safe.**  Two concurrent producers could load the
    /// same `tail` value, both write to the same slot, and both
    /// advance `tail` — losing one entry.  For MPMC the tail
    /// advancement would need a compare-and-swap loop.  Current
    /// usage (single PIT ISR as sole producer) is SPSC and correct.
    ///
    /// # Memory ordering
    ///
    /// - `tail` loaded with `Relaxed` — sole producer, no competing
    ///   writer.
    /// - `head` loaded with `Acquire` — observes the consumer's
    ///   latest `Release` store so we don't think the queue is full
    ///   when a slot was just freed.
    /// - Buffer slot written with `Relaxed` — the subsequent
    ///   `Release` store to `tail` establishes the happens-before
    ///   edge that makes the slot value visible to the consumer.
    /// - `tail` stored with `Release` — the consumer's `Acquire`
    ///   load of `tail` then sees both the new index *and* the
    ///   preceding slot write.
    pub fn push(&self, id: TaskId) -> Result<(), QueueFull> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail.wrapping_sub(head) >= CAPACITY {
            return Err(QueueFull);
        }

        self.buffer[tail % CAPACITY].store(id.as_u64(), Ordering::Relaxed);
        self.tail.store(tail.wrapping_add(1), Ordering::Release);

        Ok(())
    }

    /// Dequeue the next runnable task ID, if any.
    ///
    /// Returns `None` when the queue is empty (`head == tail`).
    ///
    /// # MPMC safety
    ///
    /// **Not MPMC-safe.**  Two concurrent consumers could load the
    /// same `head`, both read the same slot, and both advance `head`
    /// — delivering one ID twice and skipping the next.  For MPMC
    /// the head advancement would need a compare-and-swap loop.
    /// Current usage (single executor loop as sole consumer) is SPSC
    /// and correct.
    ///
    /// # Memory ordering
    ///
    /// - `head` loaded with `Relaxed` — sole consumer, no competing
    ///   writer.
    /// - `tail` loaded with `Acquire` — observes the producer's
    ///   latest `Release` store so we see newly pushed IDs.
    /// - Buffer slot loaded with `Relaxed` — the preceding `Acquire`
    ///   on `tail` already establishes the happens-before edge from
    ///   the producer's slot write.
    /// - Slot cleared to `0` (sentinel) with `Relaxed` — the
    ///   subsequent `Release` on `head` publishes the clearing.
    /// - `head` stored with `Release` — the producer's `Acquire`
    ///   load of `head` then sees the freed slot.
    pub fn pop(&self) -> Option<TaskId> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head == tail {
            return None;
        }

        let slot = head % CAPACITY;
        let val = self.buffer[slot].load(Ordering::Relaxed);

        // Defensive: in correct SPSC usage `val` is never 0 here,
        // because `head` only advances past slots that `push` has
        // filled.  The assert documents the invariant.
        debug_assert!(
            val != 0,
            "ready queue: popped empty sentinel from an occupied slot"
        );

        self.buffer[slot].store(0, Ordering::Relaxed);
        self.head.store(head.wrapping_add(1), Ordering::Release);

        Some(TaskId::from_raw(val))
    }

    /// Snapshot check: is the queue currently empty?
    ///
    /// # Correctness under SPSC
    ///
    /// `head == tail` is the emptiness invariant.  Because `head` is
    /// only advanced by `pop` (consumer) and `tail` only by `push`
    /// (producer), no concurrent operation can move *both* indices
    /// between our two loads.  The worst case is a stale read:
    ///
    /// - A `push` races between our loads of `head` and `tail`: we
    ///   may return `true` (empty) even though an item was just
    ///   pushed.  Benign — the executor enters its `hlt` loop one
    ///   iteration early and catches the item on the next interrupt.
    ///
    /// - A `pop` races similarly: we may return `false` (non-empty)
    ///   when the last item was just popped.  Also benign — the
    ///   executor calls `pop`, gets `None`, and loops back.
    ///
    /// In neither case is data lost or duplicated.
    #[allow(dead_code)] // public API — used by future stages
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head == tail
    }
}

/// Global ready queue — the single place where wakers enqueue task
/// IDs and the executor dequeues them.
///
/// **Design choice (Stage 3):** module-global `static` rather than
/// executor-owned with a pointer indirection.  Simpler, sufficient
/// for a single-executor kernel.  If multi-executor support is needed
/// (Stage 4+), refactor to executor-owned with a `static` pointer
/// set at init time.
pub static READY_QUEUE: ReadyQueue = ReadyQueue::new();
