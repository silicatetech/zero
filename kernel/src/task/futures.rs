// SPDX-License-Identifier: AGPL-3.0-or-later
//! Primitive async building blocks for Stage-3 test tasks.
//!
//! - [`TimerFuture`] — waits until the PIT tick counter reaches a
//!   target value.  Uses busy-wake (`wake_by_ref` on every
//!   `Pending` poll).  **x86_64 only** — uses PIT hardware.
//! - [`YieldFuture`] — yields once to the executor, then completes.
//!   Platform-agnostic.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

#[cfg(target_arch = "x86_64")]
use crate::arch::{cycles, pit};

/// A future that completes when `pit::ticks() >= target_tick`
/// **or** when the TSC reaches a fallback deadline derived from the
/// requested tick distance.
///
/// # PIT-dead fallback
///
/// On platforms where the legacy 8259 PIC is masked (e.g. AMD EPYC
/// under UEFI/IOAPIC), IRQ 0 never reaches the kernel and
/// `pit::ticks()` stays pinned at zero. Without a fallback the
/// executor would busy-spin forever inside this future.
///
/// `TimerFuture::new` snapshots both the current PIT tick and the
/// current TSC, then computes a TSC deadline equal to
/// `(target_tick - now_tick) / pit::HZ` seconds in TSC cycles.
/// `poll` returns `Ready` as soon as **either** counter has
/// advanced past its deadline, so timers complete with bounded
/// latency regardless of whether the PIT is alive.
///
/// # Busy-wake caveat
///
/// On each `Pending` poll, `wake_by_ref` is called immediately to
/// re-enqueue the task.  This means the executor will poll this
/// future on every loop iteration until the target tick is reached
/// — effectively busy-waiting with cooperative yields.
///
/// This is acceptable for Stage 3 because:
/// - It exercises the `wake_by_ref` VTable path on every poll.
/// - Only one timer task runs at a time, and the hlt in the idle
///   path still saves power when *no* task is pending.
///
/// **Revisit in Stage 4:** replace with a proper timer-wheel that
/// lets the ISR push task IDs only when their deadline expires.
/// See `docs/DEFERRED_DECISIONS.md`.
#[cfg(target_arch = "x86_64")]
pub struct TimerFuture {
    target_tick: u64,
    tsc_deadline: u64,
}

#[cfg(target_arch = "x86_64")]
impl TimerFuture {
    /// Wait until the global tick counter reaches `target_tick`,
    /// with a TSC-derived fallback deadline so the future still
    /// completes when the PIT is dead.
    pub fn new(target_tick: u64) -> Self {
        let now_tick = pit::ticks();
        let ticks_remaining = target_tick.saturating_sub(now_tick);
        let tsc_hz = cycles::tsc_hz();
        // Convert remaining PIT ticks into TSC cycles:
        //   cycles = ticks * tsc_hz / pit::HZ
        // saturating_mul guards against pathological target values.
        let cycles_to_wait = ticks_remaining.saturating_mul(tsc_hz) / pit::HZ;
        let tsc_deadline = cycles::rdtsc_serialized().saturating_add(cycles_to_wait);
        TimerFuture {
            target_tick,
            tsc_deadline,
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Future for TimerFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if pit::ticks() >= self.target_tick || cycles::rdtsc_serialized() >= self.tsc_deadline {
            Poll::Ready(())
        } else {
            // Busy-wake: re-enqueue so we get polled again soon.
            // Exercises VTable `wake_by_ref`.
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// A future that returns `Pending` exactly once, then `Ready`.
///
/// Inserting `YieldFuture::new().await` in a loop gives the
/// executor a chance to run other tasks between iterations —
/// cooperative yielding.
///
/// On the first poll:
/// - Calls `wake_by_ref` to re-enqueue the task.
/// - Returns `Pending`.
///
/// On the second poll:
/// - Returns `Ready(())`.
pub struct YieldFuture {
    yielded: bool,
}

impl YieldFuture {
    pub fn new() -> Self {
        YieldFuture { yielded: false }
    }
}

impl Future for YieldFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            // SAFETY: YieldFuture contains only a bool — Unpin,
            // so get_mut is fine through Pin.
            self.get_mut().yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}
