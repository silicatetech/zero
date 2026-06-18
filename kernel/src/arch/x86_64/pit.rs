// SPDX-License-Identifier: AGPL-3.0-or-later
//! 8253/8254 Programmable Interval Timer driver — from scratch.
//!
//! Channel 0 of the PIT is wired to IRQ 0 on the master PIC. We run
//! it in Mode 3 (square-wave generator) with a 16-bit divisor.
//!
//! The PIT's base clock is 1_193_182 Hz (chosen to divide the NTSC
//! color-burst frequency evenly — a 1980s detail that x86 has kept
//! for four decades of backward compatibility).
//!
//! Dividing by 11_932 gives 1_193_182 / 11_932 ≈ 99.995 Hz, close
//! enough to 100 Hz to call the scheduler tick "10 ms."

use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering};

// -------- I/O port addresses -----------------------------------------
const CHANNEL_0_DATA: u16 = 0x40;
const COMMAND_REG: u16 = 0x43;

// -------- timing constants -------------------------------------------
const DIVISOR: u16 = 11_932; // → ≈ 99.995 Hz
/// Nominal tick frequency.
pub const HZ: u64 = 100;

/// Count of timer ticks since PIT init. The timer ISR is the only
/// writer; every reader uses [`ticks`]. `AtomicU64` makes the tick
/// value tear-free on every target we care about (and lets the ISR
/// increment without a lock).
pub static TICKS: AtomicU64 = AtomicU64::new(0);

/// Program Channel 0 for Mode 3 at approximately 100 Hz.
///
/// Order of writes matters:
/// 1. Command byte to port 0x43 — selects channel, access mode,
///    counter mode, and BCD/binary.
/// 2. Divisor low byte to port 0x40.
/// 3. Divisor high byte to port 0x40.
///
/// Reversing steps 2 and 3 ("high byte first") is the classic PIT
/// bug. The counter still runs, but at a frequency that is neither
/// the intended one nor obviously wrong, which makes it tedious to
/// diagnose.
pub fn init() {
    unsafe {
        // Command byte: 0b0011_0110
        //   bits 7-6 = 00 : Channel 0
        //   bits 5-4 = 11 : access mode "lobyte/hibyte"
        //   bits 3-1 = 011: Mode 3 (square wave generator)
        //   bit  0   = 0  : binary counting (not BCD)
        outb(COMMAND_REG, 0b0011_0110);

        // Divisor: LOW byte first, then HIGH byte.
        outb(CHANNEL_0_DATA, (DIVISOR & 0xFF) as u8);
        outb(CHANNEL_0_DATA, (DIVISOR >> 8) as u8);
    }
}

/// Increment the tick counter. Called by the timer ISR.
///
/// `Ordering::Relaxed` is sufficient: we only need atomicity, not
/// any ordering relative to other memory operations. Readers use
/// the same ordering.
#[inline]
pub fn tick() {
    TICKS.fetch_add(1, Ordering::Relaxed);
}

/// Current tick count since [`init`] was called.
#[inline]
pub fn ticks() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

#[inline]
unsafe fn outb(port: u16, val: u8) {
    asm!(
        "out dx, al",
        in("dx") port,
        in("al") val,
        options(nomem, nostack, preserves_flags),
    );
}
