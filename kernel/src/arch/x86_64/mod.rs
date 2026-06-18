// SPDX-License-Identifier: AGPL-3.0-or-later
//! x86_64 platform module.
//!
//! Re-exports x86_64-specific kernel components. All 7 sub-modules
//! were originally top-level in `kernel/src/` and moved here during
//! Sub-MP-D2a HAL extraction.
//!
//! The HAL-orchestration functions (`init`, `without_interrupts`,
//! `enable_and_hlt`, `interrupts_disable`, `interrupts_enable`)
//! provide platform-agnostic entry points that `main.rs` and
//! `task/*` modules call without knowing which architecture runs.

pub mod cpuinfo;
pub mod cycles;
pub mod fb_console;
pub mod gdt;
pub mod interrupts;
pub mod iommu;
pub mod pcie;
pub mod pic;
pub mod pit;
pub mod serial;

// ADR-029 Phase 1+2+3: SMP / multi-core boot. Feature-gated on
// avx512-acceleration because the x86_64 SMP layer's first consumer
// is the parallel AVX-512 matmul dispatcher. The AP boot machinery
// has no other consumer in v0, so compiling it out when SMP isn't
// wanted keeps the kernel size down.
#[cfg(feature = "avx512-acceleration")]
pub mod acpi;
#[cfg(feature = "avx512-acceleration")]
pub mod apic;
#[cfg(feature = "avx512-acceleration")]
pub mod trampoline;

// ADR-029 Phase 1+2: AVX-512 acceleration hooks. The `math` sub-module
// declaration itself is unconditional (mirror of aarch64); only the
// inner `math::linear` module is feature-gated, so the namespace
// stays in sync with the NEON side.
pub mod math;

// Re-exports for kernel-level platform-agnostic access.
pub use cycles::rdtsc_serialized as read_cycles;
pub use serial::Serial;

/// Critical-section abstraction — disable interrupts for the
/// duration of `f`, then restore previous interrupt state.
/// Used by `task/queue.rs` and `task/waker.rs`.
#[inline(always)]
#[allow(dead_code)]
pub fn without_interrupts<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    x86_64::instructions::interrupts::without_interrupts(f)
}

/// Halt-with-interrupts-enabled — atomically enable interrupts
/// and halt until the next interrupt arrives. The `sti; hlt`
/// pair in a single instruction window avoids the sleep/wake
/// race (see `task/executor.rs` doc comment for analysis).
/// Used by `task/executor.rs` idle path.
#[inline(always)]
pub fn enable_and_hlt() {
    x86_64::instructions::interrupts::enable_and_hlt()
}

/// Disable maskable interrupts (cli). Used for critical sections
/// where raw control is needed (executor run loop).
#[inline(always)]
pub fn interrupts_disable() {
    x86_64::instructions::interrupts::disable()
}

/// Enable maskable interrupts (sti). Used after critical section
/// exit in executor run loop.
#[inline(always)]
pub fn interrupts_enable() {
    x86_64::instructions::interrupts::enable()
}

/// Halt the CPU (hlt). Used for busy-wait loops.
#[inline(always)]
pub fn hlt() {
    x86_64::instructions::hlt()
}
