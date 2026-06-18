// SPDX-License-Identifier: AGPL-3.0-or-later
//! Task infrastructure for the Stage-3 kernel async executor.
//!
//! Built in four micro-steps to keep each commit focused and the
//! failure-mode predictions (see BUGS.md "Pre-Stage-3 Predictions")
//! testable at each step:
//!
//!   1. [`task`] — `Task` + `TaskId` scaffolding.
//!   2. [`waker`] — `RawWakerVTable` with documented semantics.
//!   3. [`queue`] — lock-free ring-buffer ready queue.
//!   4. [`executor`] — spawn + run with sti/hlt sleep-wake.
//!
//! Nothing here wires up hardware or touches the IDT; this module
//! is pure Rust over the heap that Stage 2 set up.

pub mod executor;
pub mod futures;
pub mod oneshot;
pub mod queue;
pub mod task;
pub mod waker;

// Re-exports: public API surface for the task subsystem.
// Not all are consumed by main.rs yet — some are for future stages.
#[allow(unused_imports)]
pub use executor::{Executor, SpawnError};
#[allow(unused_imports)]
pub use queue::{QueueFull, ReadyQueue, READY_QUEUE};
#[allow(unused_imports)]
pub use task::{Task, TaskId};
#[allow(unused_imports)]
pub use waker::create_waker;
