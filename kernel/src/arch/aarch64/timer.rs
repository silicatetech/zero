// SPDX-License-Identifier: AGPL-3.0-or-later
//! ARM Generic Timer driver (Virtual Timer, EL1) for aarch64.
//!
//! Per ARM ARM D7.5 "The Generic Timer in AArch64":
//! Virtual Timer (CNTV) at EL1, INTID 27 PPI.
//!
//! Programming sequence:
//! 1. Read CNTFRQ_EL0 → counter frequency
//! 2. Compute ticks per heartbeat period (100 Hz)
//! 3. Write CNTV_TVAL_EL0 = ticks (downcount until fire)
//! 4. Write CNTV_CTL_EL0 = ENABLE (bit 0), clear IMASK (bit 1)
//! 5. Enable INTID 27 at GICD
//!
//! CITE: ARM ARM D7.5 — Generic Timer
//! CITE: ARM ARM D13.8 — CNTV_* registers

use core::fmt::Write;
use core::sync::atomic::{AtomicU64, Ordering};

/// Virtual Timer INTID (PPI). Per QEMU virt + ARM SBSA.
pub const TIMER_INTID: u32 = 27;

/// Timer frequency: 100 Hz (10 ms period).
const TIMER_HZ: u64 = 100;

/// Heartbeat report interval: every 100 ticks (~1 second at 100 Hz).
const HEARTBEAT_REPORT_EVERY: u64 = 100;

/// Tick counter, incremented by timer IRQ handler.
static TICK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Cached ticks per period.
static TICKS_PER_PERIOD: AtomicU64 = AtomicU64::new(0);

// ---- Register access ----

unsafe fn read_cntfrq() -> u64 {
    let val: u64;
    core::arch::asm!("mrs {}, cntfrq_el0", out(reg) val,
        options(nomem, nostack, preserves_flags));
    val
}

unsafe fn write_cntv_tval(ticks: u64) {
    core::arch::asm!("msr cntv_tval_el0, {}", in(reg) ticks,
        options(nomem, nostack, preserves_flags));
}

unsafe fn write_cntv_ctl(value: u64) {
    core::arch::asm!("msr cntv_ctl_el0, {}", in(reg) value,
        options(nomem, nostack, preserves_flags));
}

// ---- Initialization ----

/// Initialize Virtual Timer for 100 Hz heartbeat.
///
/// # Safety
///
/// Must be called after gic::init(). DAIF should mask interrupts.
pub unsafe fn init() {
    use crate::arch::aarch64::gic;
    let serial = &mut crate::arch::aarch64::serial::Serial;

    let frq = read_cntfrq();
    let ticks = frq / TIMER_HZ;
    TICKS_PER_PERIOD.store(ticks, Ordering::Release);

    let _ = writeln!(
        serial,
        "Stage 9: Generic Timer init (Virtual, INTID {})",
        TIMER_INTID
    );
    let _ = writeln!(
        serial,
        "  CNTFRQ={} Hz ({} MHz), {} Hz, ticks/period={}",
        frq,
        frq / 1_000_000,
        TIMER_HZ,
        ticks
    );

    // Program timer
    write_cntv_tval(ticks);
    write_cntv_ctl(1);

    // Enable INTID 27 at Distributor
    gic::enable_interrupt(TIMER_INTID);

    let _ = writeln!(
        serial,
        "  CNTV enabled, INTID {} unmasked at GICD",
        TIMER_INTID
    );
}

// ---- IRQ handler ----

/// Handle Virtual Timer IRQ.
///
/// # Safety
///
/// Must be called from IRQ handler context.
pub unsafe fn handle_tick() {
    let tick = TICK_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let ticks_per_period = TICKS_PER_PERIOD.load(Ordering::Relaxed);

    // Rearm timer
    write_cntv_tval(ticks_per_period);

    // Heartbeat report every N ticks
    if tick % HEARTBEAT_REPORT_EVERY == 0 {
        let serial = &mut crate::arch::aarch64::serial::Serial;
        let _ = writeln!(
            serial,
            "Heartbeat: tick={} (~{}s uptime)",
            tick,
            tick / TIMER_HZ
        );
    }
}
