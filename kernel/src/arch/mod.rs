// SPDX-License-Identifier: AGPL-3.0-or-later
//! Architecture-Native Boot dispatch (Sub-MP-D0 Entscheidung 2: cfg-driven).
//!
//! Per Pillar 1: zero runtime dispatch — platform resolved at compile time.
//! Per Pillar 7: native idioms per platform.
//!
//! HAL contract: both `x86_64` and `aarch64` modules export:
//! - `Serial` — write-only serial output type
//! - `without_interrupts(f)` — critical-section wrapper
//! - `enable_and_hlt()` — atomic enable-interrupts + halt
//! - `interrupts_disable()` / `interrupts_enable()` — raw control
//! - `hlt()` — halt CPU
//! - `read_cycles()` — performance counter read

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// Re-export current platform's HAL items at arch:: level.
// Callers use `crate::arch::Serial`, `crate::arch::without_interrupts`, etc.

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86_64::{
    cpuinfo, cycles, enable_and_hlt, fb_console, gdt, hlt, interrupts, interrupts_disable,
    interrupts_enable, pcie, pic, pit, read_cycles, serial, without_interrupts, Serial,
};

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use aarch64::{
    enable_and_hlt, hlt, interrupts_disable, interrupts_enable, read_cycles, serial,
    without_interrupts, Serial,
};
