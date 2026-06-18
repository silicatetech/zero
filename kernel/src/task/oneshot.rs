// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal oneshot channel for inter-task communication.
//!
//! One sender, one receiver.  The sender sets a value; the receiver
//! is a `Future` that resolves when the value arrives.  This is the
//! first real exercise of the waker mechanism:
//!
//! - `RecvFuture::poll` calls `cx.waker().clone()` → VTable `clone`.
//! - `Oneshot::send` calls `waker.wake()` → VTable `wake`.
//! - The cloned `Waker` is dropped at scope exit → VTable `drop`.
//!
//! All four VTable functions (clone, wake, wake_by_ref, drop) are
//! exercised by the combination of oneshot + timer/yield futures.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

use spin::Mutex;

/// A single-use channel: one value from sender to receiver.
///
/// Shared via `&'static` (arena-allocated via `memory::arena_static_alloc`).  Both `value`
/// and `receiver_waker` are behind `Mutex` for interior mutability.
///
/// Not reusable — once a value is sent and received, the channel
/// is spent.
pub struct Oneshot<T> {
    value: Mutex<Option<T>>,
    receiver_waker: Mutex<Option<Waker>>,
}

impl<T> Oneshot<T> {
    /// Create an empty channel.  Const-evaluable.
    pub const fn new() -> Self {
        Oneshot {
            value: Mutex::new(None),
            receiver_waker: Mutex::new(None),
        }
    }
}

impl<T: Send> Oneshot<T> {
    /// Send a value, waking the receiver if it has registered a
    /// waker via `poll`.
    ///
    /// Locking order: `value` first, then `receiver_waker`.  Same
    /// order as `RecvFuture::poll` — no deadlock risk (and on
    /// single-CPU cooperative scheduling, deadlock from two tasks
    /// is structurally impossible anyway).
    pub fn send(&self, val: T) {
        {
            let mut slot = self.value.lock();
            *slot = Some(val);
        }
        // Release value lock before waking — the wake pushes the
        // receiver's TaskId into the ReadyQueue, which may cause
        // the receiver to be polled (and lock `value`) soon.
        let waker = {
            let mut w = self.receiver_waker.lock();
            w.take()
        };
        if let Some(w) = waker {
            w.wake(); // exercises VTable `wake` (consumes Waker)
        }
    }

    /// Return a future that resolves to the sent value.
    pub fn recv(&'static self) -> RecvFuture<T> {
        RecvFuture { channel: self }
    }
}

/// Future returned by [`Oneshot::recv`].
///
/// On each poll:
/// - If the value has arrived: take it and return `Ready`.
/// - Otherwise: store `cx.waker().clone()` so the sender can wake
///   us, and return `Pending`.
pub struct RecvFuture<T: Send + 'static> {
    channel: &'static Oneshot<T>,
}

impl<T: Send> Future for RecvFuture<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        // Check if a value has been sent.
        {
            let mut value = self.channel.value.lock();
            if let Some(val) = value.take() {
                return Poll::Ready(val);
            }
        }
        // No value yet — register our waker so the sender can
        // notify us.  `cx.waker().clone()` exercises VTable `clone`.
        let mut waker = self.channel.receiver_waker.lock();
        *waker = Some(cx.waker().clone());
        Poll::Pending
    }
}
