// SPDX-License-Identifier: AGPL-3.0-or-later
//! Async executor — Stage 3 integration module.
//!
//! Brings together `Task`, `ReadyQueue`, and the waker VTable into
//! a working run loop.  This is the fourth and final micro-step of
//! the Stage-3 task infrastructure.
//!
//! # Architecture
//!
//! - **Task storage:** `static TASKS: Mutex<[Option<Task>; 32]>`.
//!   Fixed-size array, locked during spawn and poll.
//! - **Ready queue:** global `READY_QUEUE` (lock-free ring buffer).
//!   Wakers push task IDs here; the run loop pops them.
//! - **Run loop (`Executor::run`):** `cli` → pop → `sti` → poll,
//!   or `sti; hlt` when idle.  The `sti; hlt` pair is in a single
//!   `asm!` block to avoid the classic sleep/wake race.
//!
//! # Known limitations (Stage 3)
//!
//! - The `TASKS` mutex is held for the entire duration of a `poll`.
//!   This means a long-running poll blocks `spawn` and other polls.
//!   Acceptable for Stage 3 single-executor; refine in Stage 4+.
//! - TaskId → slot lookup is a linear scan, O(n) with n ≤ 32.
//!   Acceptable for Stage 3; a `BTreeMap` or hash map can replace
//!   this if the slot count grows.
//! - No task preemption; a task that never yields starves others.

use core::fmt;
use core::future::Future;
use core::task::{Context, Poll};

use quarks_arena::ArenaError;
use spin::Mutex;

use super::queue::READY_QUEUE;
use super::waker::create_waker;
use super::{Task, TaskId};
use crate::arch;

/// Maximum number of concurrent tasks.
const MAX_TASKS: usize = 32;

/// Global task storage.
///
/// Protected by a spinlock `Mutex`.  The executor locks it to poll,
/// `spawn` locks it to insert.  Because both run on the same CPU
/// (single-core Stage 3), contention only occurs if an ISR tries
/// to lock while the executor holds it — which our design avoids
/// (ISRs only touch the lock-free `READY_QUEUE`, never `TASKS`).
static TASKS: Mutex<[Option<Task>; MAX_TASKS]> = Mutex::new([const { None }; MAX_TASKS]);

/// Errors that `spawn` can produce.
#[derive(Debug)]
pub enum SpawnError {
    /// All 32 task slots are occupied.
    TaskArrayFull,
    /// The ready queue's 64-slot ring buffer is full.
    QueueFull,
    /// The kernel arena could not allocate space for the future.
    ArenaAlloc(ArenaError),
}

impl From<ArenaError> for SpawnError {
    fn from(e: ArenaError) -> Self {
        SpawnError::ArenaAlloc(e)
    }
}

impl fmt::Display for SpawnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpawnError::TaskArrayFull => write!(f, "all {} task slots occupied", MAX_TASKS),
            SpawnError::QueueFull => write!(f, "ready queue full"),
            SpawnError::ArenaAlloc(e) => write!(f, "arena allocation failed: {:?}", e),
        }
    }
}

/// The Stage-3 async executor.
///
/// Stateless struct — all mutable state lives in the module-level
/// statics (`TASKS`, `READY_QUEUE`).  The struct wrapper gives a
/// clean API surface (`executor.spawn(…)`, `executor.run()`).
/// For Stage 3 there is exactly one `Executor` instance.
pub struct Executor {
    _private: (),
}

impl Executor {
    /// Create a new executor.  Logs to serial on init.
    pub fn new() -> Self {
        arch::serial::println("Stage 3: executor initialized, 32 task slots");
        Executor { _private: () }
    }

    /// Spawn a future as a new task.
    ///
    /// 1. Allocates `future` into `KERNEL_ARENA` and wraps it in a
    ///    `Task` (arena allocation, no heap).
    /// 2. Finds the first free slot in `TASKS`.
    /// 3. Pushes the task's ID into `READY_QUEUE` for immediate poll.
    ///
    /// Returns the `TaskId` on success.
    pub fn spawn(
        &self,
        future: impl Future<Output = ()> + Send + 'static,
    ) -> Result<TaskId, SpawnError> {
        let task = Task::new(future)?;
        let id = task.id;

        // Lock the task array to find a free slot.
        let mut tasks = TASKS.lock();
        let slot = tasks
            .iter_mut()
            .find(|s| s.is_none())
            .ok_or(SpawnError::TaskArrayFull)?;
        *slot = Some(task);
        drop(tasks); // release lock before touching the queue

        // Enqueue for immediate first poll.
        READY_QUEUE.push(id).map_err(|_| SpawnError::QueueFull)?;

        Ok(id)
    }

    /// Run the executor forever.  Does not return.
    ///
    /// # The `sti; hlt` race and why they share one `asm!` block
    ///
    /// When the ready queue is empty the CPU should sleep until the
    /// next interrupt (which may push a new ID into the queue via a
    /// waker).  The naive sequence is:
    ///
    /// ```text
    /// sti          // enable interrupts
    /// hlt          // sleep until interrupt
    /// ```
    ///
    /// If these are two separate instructions with an interrupt
    /// window between them, the following race is possible:
    ///
    /// 1. Executor sees empty queue, executes `sti`.
    /// 2. Timer IRQ fires **between** `sti` and `hlt`.
    /// 3. The IRQ handler wakes a task (pushes to queue).
    /// 4. IRQ handler returns.  CPU is back in the executor.
    /// 5. Executor executes `hlt` — goes to sleep even though
    ///    work is waiting.
    ///
    /// The Intel SDM (Vol. 2, STI instruction reference) guarantees:
    ///
    /// > "After the IF flag is set, the processor begins responding
    /// > to external, maskable interrupts after the **next**
    /// > instruction is executed."
    ///
    /// (Paraphrased from Intel SDM Vol. 2, STI — verified 2026-04-22)
    ///
    /// This means `sti; hlt` in the *same* `asm!` block is atomic
    /// with respect to interrupt delivery: the interrupt that arrives
    /// after `sti` will be held pending until `hlt` executes, and
    /// `hlt` immediately wakes on that pending interrupt.  No work
    /// can be lost.
    ///
    /// This is the mitigation for **Prediction 3** (BUGS.md:
    /// "Race condition between Timer-ISR and Executor Ready-Queue
    /// access").
    pub fn run(&self) -> ! {
        loop {
            // ── Critical section: disable interrupts while we
            //    check the queue, so no IRQ can push between our
            //    pop and our decision to hlt. ──
            arch::interrupts_disable();

            match READY_QUEUE.pop() {
                Some(id) => {
                    // Re-enable interrupts before polling — a poll
                    // may take arbitrary time and we don't want to
                    // miss timer ticks.
                    arch::interrupts_enable();
                    self.poll_task(id);
                }
                None => {
                    // Queue empty.  Sleep until the next interrupt.
                    //
                    // `sti; hlt` in ONE asm! block — see the doc
                    // comment above for why this must not be split
                    // into two separate asm! invocations.
                    arch::enable_and_hlt();
                }
            }
        }
    }

    /// Poll one task identified by `id`.
    ///
    /// # Known limitation
    ///
    /// The `TASKS` mutex is held for the entire `poll` call.  This
    /// prevents concurrent `spawn` or another poll from proceeding.
    /// In Stage 3 (single-core, cooperative multitasking) this is
    /// acceptable — only one thing runs at a time anyway.  In a
    /// future stage with preemptive scheduling or SMP, the task
    /// should be *taken* out of the array, polled without the lock,
    /// then returned.
    fn poll_task(&self, id: TaskId) {
        let mut tasks = TASKS.lock();

        // Linear scan: find the slot whose task has `id`.
        // O(n) with n = MAX_TASKS = 32 — acceptable for Stage 3.
        let slot = tasks
            .iter_mut()
            .find(|s| s.as_ref().map_or(false, |t| t.id == id));

        let slot = match slot {
            Some(s) => s,
            None => {
                // Stale wakeup: the task was already completed and
                // its slot cleared.  This is normal — a waker can
                // fire after the future has returned Ready.
                return;
            }
        };

        // We have `&mut Option<Task>` — unwrap to `&mut Task`.
        let task = slot.as_mut().unwrap();

        let waker = create_waker(id);
        let mut cx = Context::from_waker(&waker);

        match task.poll(&mut cx) {
            Poll::Ready(()) => {
                // Task finished — free the slot.
                *slot = None;
            }
            Poll::Pending => {
                // Task parked itself on a waker.  When something
                // calls waker.wake(), the ID will be re-pushed into
                // READY_QUEUE and we'll poll again.
            }
        }
    }
}
