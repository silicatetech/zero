// SPDX-License-Identifier: AGPL-3.0-or-later
//! aarch64 platform module — Sub-MP-D2b implementation.
//!
//! Architecture-Native Boot for QEMU virt machine.
//! DTB pointer in x0 (Linux arm64 boot protocol), PL011 UART,
//! GICv2, Generic Timer, 4-level translation tables.

/// Entry point + DTB capture (QEMU -kernel → x0 = DTB pointer).
pub mod boot;
/// Device Tree Blob parser.
pub mod dtb;
/// Exception vector table (VBAR_EL1) — Stage 2.
pub mod exceptions;
/// GICv2 interrupt controller (Stage 8).
pub mod gic;
/// Sub-MP-F1: Linear Frame Buffer via ramfb (QEMU fw-cfg).
pub mod lfb;
/// Sub-MP-E2: NEON math acceleration (feature-gated).
#[cfg(feature = "neon-acceleration")]
pub mod math;
/// MMU setup — translation tables + SCTLR_EL1 enable (Stages 5-6).
pub mod mmu;
/// PL011 UART driver (MMIO at 0x0900_0000 on QEMU virt).
pub mod serial;
/// ARM Generic Timer — Virtual Timer EL1 (Stage 9).
pub mod timer;

// ---- HAL-parity re-exports (match x86_64/mod.rs interface) ----

pub use serial::Serial;

/// Read cycle counter (CNTVCT_EL0).
#[inline(always)]
#[allow(dead_code)]
pub fn read_cycles() -> u64 {
    let val: u64;
    unsafe { core::arch::asm!("mrs {}, cntvct_el0", out(reg) val) };
    val
}

/// Critical-section abstraction — mask DAIF interrupts for `f`.
#[inline(always)]
#[allow(dead_code)]
pub fn without_interrupts<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    // Save DAIF, mask all, run f, restore
    let daif: u64;
    unsafe {
        core::arch::asm!("mrs {}, daif", out(reg) daif);
        core::arch::asm!("msr daifset, #0xf");
    }
    let result = f();
    unsafe {
        core::arch::asm!("msr daif, {}", in(reg) daif);
    }
    result
}

/// Halt-with-interrupts-enabled — `wfi` instruction.
#[inline(always)]
#[allow(dead_code)]
pub fn enable_and_hlt() {
    unsafe {
        core::arch::asm!("msr daifclr, #0xf"); // enable IRQs
        core::arch::asm!("wfi"); // wait for interrupt
    }
}

/// Disable IRQs (msr daifset, #2).
#[inline(always)]
#[allow(dead_code)]
pub fn interrupts_disable() {
    unsafe { core::arch::asm!("msr daifset, #0xf") };
}

/// Enable IRQs (msr daifclr, #2).
#[inline(always)]
#[allow(dead_code)]
pub fn interrupts_enable() {
    unsafe { core::arch::asm!("msr daifclr, #0xf") };
}

/// Halt CPU (wfi).
#[inline(always)]
#[allow(dead_code)]
pub fn hlt() {
    unsafe { core::arch::asm!("wfi") };
}
