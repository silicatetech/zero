// SPDX-License-Identifier: AGPL-3.0-or-later
//! `RawWakerVTable` implementation for the Stage-3 executor.
//!
//! Bridges `core::task::Waker` and our `ReadyQueue`: when a future
//! calls `waker.wake()`, the task's `TaskId` is pushed into the
//! global ready queue so the executor knows to poll it again.
//!
//! The data pointer inside each `RawWaker` is **not** a real pointer
//! — it is a `TaskId` (a `u64`) cast to `*const ()`.  This is safe
//! because the VTable functions never dereference it; they only cast
//! it back to `u64` to recover the `TaskId`.
//!
//! **Platform assumption:** the cast `u64 as usize as *const ()`
//! is lossless on x86_64 (where `usize` is 64 bits).  On a 32-bit
//! target the `u64 → usize` step would truncate and lose high bits.
//! Zero targets only x86_64, so this is safe here.  If the kernel
//! is ever ported to a 32-bit platform, the data-pointer encoding
//! must be revisited (e.g. heap-allocate a `TaskId` and use a real
//! pointer, or limit IDs to 32 bits).
//!
//! # Risk class
//!
//! Waker semantics are the Prediction-1 risk class from BUGS.md
//! ("Halluziniert-Spec").  Every VTable function below carries a
//! doc-comment quoting or paraphrasing the **exact** contract from
//! `core::task::RawWakerVTable` (Rust 1.95 docs, fetched 2026-04-22)
//! so the implementation can be audited against the spec verbatim.

use core::task::{RawWaker, RawWakerVTable, Waker};

use super::queue::READY_QUEUE;
use super::TaskId;

// ── VTable ──────────────────────────────────────────────────────

/// The single VTable shared by all wakers in this kernel.
///
/// Rust core docs (RawWakerVTable, "Thread safety" section):
///
/// > "If the RawWaker will be used to construct a Waker then these
/// > functions must all be thread-safe (even though RawWaker is
/// > !Send + !Sync).  This is because Waker is Send + Sync, and
/// > it may be moved to arbitrary threads or invoked by & reference.
/// > For example, this means that if the clone and drop functions
/// > manage a reference count, they must do so atomically."
///
/// Our functions are trivially thread-safe: `TaskId` is `Copy` with
/// no reference count, and `READY_QUEUE.push` is lock-free atomic.
static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_waker, wake, wake_by_ref, drop_waker);

// ── VTable functions ────────────────────────────────────────────

/// ### `clone`
///
/// Rust core docs (RawWakerVTable, "clone" section):
///
/// > "This function will be called when the RawWaker gets cloned,
/// > e.g. when the Waker in which the RawWaker is stored gets
/// > cloned."
///
/// > "The implementation of this function must retain all resources
/// > that are required for this additional instance of a RawWaker
/// > and associated task.  Calling wake on the resulting RawWaker
/// > should result in a wakeup of the same task that would have
/// > been awoken by the original RawWaker."
///
/// Our `data` is a `TaskId` encoded as `*const ()`.  `TaskId` is
/// `Copy` — no heap allocation, no reference count.  Cloning is
/// just constructing a new `RawWaker` with the same data and
/// VTable pointer.
unsafe fn clone_waker(data: *const ()) -> RawWaker {
    // TaskId is Copy — no resources to retain beyond the value
    // itself, which is duplicated by constructing a new RawWaker
    // with the same data pointer.
    RawWaker::new(data, &VTABLE)
}

/// ### `wake`
///
/// Rust core docs (RawWakerVTable, "wake" section):
///
/// > "This function will be called when wake is called on the
/// > Waker.  It must wake up the task associated with this
/// > RawWaker."
///
/// > "The implementation of this function must make sure to release
/// > any resources that are associated with this instance of a
/// > RawWaker and associated task."
///
/// "Releasing resources" is a no-op here: `TaskId` is `Copy` and
/// holds no heap allocation or reference count.  The `Waker` that
/// owned this `RawWaker` is consumed by `Waker::wake(self)`, so
/// the VTable's `drop` will **not** be called afterwards — but
/// since our `drop` is also a no-op, this is moot.
unsafe fn wake(data: *const ()) {
    wake_by_ref(data);
    // No resources to release — TaskId is Copy.
}

/// ### `wake_by_ref`
///
/// Rust core docs (RawWakerVTable, "wake_by_ref" section):
///
/// > "This function will be called when wake_by_ref is called on
/// > the Waker.  It must wake up the task associated with this
/// > RawWaker."
///
/// > "This function is similar to wake, but must not consume the
/// > provided data pointer."
///
/// We extract the `TaskId` from `data` and push it into the global
/// `READY_QUEUE`.  The data pointer is not consumed — it remains
/// valid for future `clone` / `wake` / `drop` calls.
unsafe fn wake_by_ref(data: *const ()) {
    let raw = data as usize as u64;
    let id = TaskId::from_raw(raw);
    // Best-effort push: if the queue is full the wakeup is lost.
    // In Stage 3 with ≤ 32 tasks and a 64-slot queue this cannot
    // happen in practice.  A future stage could log or retry.
    let _ = READY_QUEUE.push(id);
}

/// ### `drop`
///
/// Rust core docs (RawWakerVTable, "drop" section):
///
/// > "This function will be called when a Waker gets dropped."
///
/// > "The implementation of this function must make sure to release
/// > any resources that are associated with this instance of a
/// > RawWaker and associated task."
///
/// No-op.  `TaskId` is `Copy` — there are no heap allocations,
/// file handles, or reference counts to release.
unsafe fn drop_waker(_data: *const ()) {
    // Nothing to free.  TaskId is a plain u64 — no destructor.
}

// ── Public helper ───────────────────────────────────────────────

/// Create a `core::task::Waker` for the given task.
///
/// The returned `Waker` can be wrapped in a `Context` and passed
/// to `Future::poll`.  When the future calls `waker.wake()`, the
/// task's ID is pushed into `READY_QUEUE`.
pub fn create_waker(id: TaskId) -> Waker {
    let data = id.as_u64() as usize as *const ();
    let raw_waker = RawWaker::new(data, &VTABLE);

    // SAFETY:  The contract from `Waker::from_raw` (Rust core docs):
    //
    // > "The behavior of the returned Waker is undefined if the
    // > contract defined in RawWaker's and RawWakerVTable's
    // > documentation is not upheld."
    //
    // Our VTable upholds the contract:
    //
    // 1. `clone` returns a new RawWaker with the same data and
    //    VTable.  TaskId is Copy, so the clone retains all
    //    resources (there are none beyond the value).
    //
    // 2. `wake` enqueues the TaskId, then does nothing further.
    //    TaskId is Copy, so there are no resources to release.
    //
    // 3. `wake_by_ref` does the same without consuming the data
    //    pointer.
    //
    // 4. `drop` is a no-op — nothing to release.
    //
    // 5. Thread safety: all four functions operate on a Copy u64
    //    and a lock-free atomic queue.  No shared mutable state
    //    beyond the queue's own atomic operations.
    //
    // 6. The data pointer is not a real pointer — it is a TaskId
    //    (u64) cast to *const ().  It is never dereferenced; only
    //    cast back to u64 to recover the TaskId.  This is a
    //    well-established pattern (see e.g. tokio's waker impl).
    unsafe { Waker::from_raw(raw_waker) }
}
