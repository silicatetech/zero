// SPDX-License-Identifier: AGPL-3.0-or-later
//! GICv2 (Generic Interrupt Controller v2) driver for aarch64.
//!
//! Per ARM GIC Architecture Specification (IHI 0048B-b):
//! - Distributor (GICD): centralized interrupt management
//! - CPU Interface (GICC): per-CPU interrupt delivery
//!
//! On QEMU virt machine:
//! - GICD base: 0x0800_0000 (from DTB, Stage 3)
//! - GICC base: 0x0801_0000 (from DTB, Stage 3)
//! - Memory mapped Device-nGnRE (Stage 5 TTBR0 L2 entry)
//!
//! Stage 8 scope: Initialize Distributor + CPU Interface.
//! Specific interrupt enables (Generic Timer etc.) come in Stage 9+.
//!
//! CITE: ARM GIC Architecture Specification IHI 0048B-b

use core::fmt::Write;
use core::sync::atomic::{AtomicUsize, Ordering};

// ---- Distributor register offsets (per GIC spec section 4.3) ----

const GICD_CTLR: usize = 0x000;
const GICD_TYPER: usize = 0x004;
const GICD_ICENABLER: usize = 0x180; // Clear-enable (32 regs × 32 bits)
const GICD_ISENABLER: usize = 0x100; // Set-enable

// ---- CPU Interface register offsets (per GIC spec section 4.4) ----

const GICC_CTLR: usize = 0x000;
const GICC_PMR: usize = 0x004;
const GICC_BPR: usize = 0x008;
const GICC_IAR: usize = 0x00C;
const GICC_EOIR: usize = 0x010;

// ---- GIC base addresses (set during init from DTB) ----

static GICD_BASE: AtomicUsize = AtomicUsize::new(0);
static GICC_BASE: AtomicUsize = AtomicUsize::new(0);

/// Spurious interrupt ID (no interrupt pending).
pub const GICC_SPURIOUS: u32 = 1023;

/// GICC_IAR INTID mask (lower 10 bits).
const IAR_INTID_MASK: u32 = 0x3FF;

// ---- Register access helpers ----

unsafe fn read_dist(offset: usize) -> u32 {
    let base = GICD_BASE.load(Ordering::Relaxed);
    core::ptr::read_volatile((base + offset) as *const u32)
}

unsafe fn write_dist(offset: usize, value: u32) {
    let base = GICD_BASE.load(Ordering::Relaxed);
    core::ptr::write_volatile((base + offset) as *mut u32, value);
}

unsafe fn read_cpu(offset: usize) -> u32 {
    let base = GICC_BASE.load(Ordering::Relaxed);
    core::ptr::read_volatile((base + offset) as *const u32)
}

unsafe fn write_cpu(offset: usize, value: u32) {
    let base = GICC_BASE.load(Ordering::Relaxed);
    core::ptr::write_volatile((base + offset) as *mut u32, value);
}

// ---- Initialization ----

/// Initialize GICv2 Distributor + CPU Interface.
///
/// Per GIC spec sections 4.3.1 (GICD_CTLR) and 4.4.1 (GICC_CTLR):
/// 1. Disable Distributor
/// 2. Disable all SPI interrupts (GICD_ICENABLER)
/// 3. Set priority mask (GICC_PMR = 0xFF, all priorities allowed)
/// 4. Set binary point (GICC_BPR = 0)
/// 5. Enable Distributor (GICD_CTLR.Enable = 1)
/// 6. Enable CPU Interface (GICC_CTLR.Enable = 1)
///
/// # Safety
///
/// Must be called once during boot, after MMU enable + DTB parse.
/// Bases must point to GICv2-compatible MMIO regions.
pub unsafe fn init(dist_base: usize, cpu_base: usize) {
    let serial = &mut crate::arch::aarch64::serial::Serial;

    GICD_BASE.store(dist_base, Ordering::Release);
    GICC_BASE.store(cpu_base, Ordering::Release);

    let _ = writeln!(
        serial,
        "Stage 8: GICv2 init dist={:#x} cpu={:#x}",
        dist_base, cpu_base
    );

    // Read GICD_TYPER: ITLinesNumber (bits [4:0]) → (N+1)*32 = max INTID+1
    let typer = read_dist(GICD_TYPER);
    let it_lines = (typer & 0x1F) as usize;
    let max_intid = (it_lines + 1) * 32;
    let _ = writeln!(serial, "  GICD_TYPER={:#x}, max INTID={}", typer, max_intid);

    // Step 1: Disable Distributor
    write_dist(GICD_CTLR, 0);

    // Step 2: Disable all SPI interrupts (INTID 32+)
    // GICD_ICENABLER[0] covers INTIDs 0-31 (SGIs/PPIs, skip)
    for n in 1..((max_intid + 31) / 32) {
        write_dist(GICD_ICENABLER + n * 4, 0xFFFF_FFFF);
    }

    // Step 3: Priority mask — allow all priorities
    write_cpu(GICC_PMR, 0xFF);

    // Step 4: Binary point — no preemption grouping
    write_cpu(GICC_BPR, 0);

    // Step 5+6: Enable Distributor + CPU Interface
    write_dist(GICD_CTLR, 1);
    write_cpu(GICC_CTLR, 1);

    let _ = writeln!(serial, "  GICD + GICC enabled");
}

// ---- Interrupt handling ----

/// Acknowledge pending interrupt. Returns INTID.
///
/// Reads GICC_IAR: returns INTID of highest-priority pending interrupt,
/// marks it Active. Returns GICC_SPURIOUS (1023) if no interrupt pending.
///
/// # Safety
///
/// Must be called from IRQ handler context.
pub unsafe fn acknowledge_irq() -> u32 {
    read_cpu(GICC_IAR) & IAR_INTID_MASK
}

/// Signal end-of-interrupt for given INTID.
///
/// Writes GICC_EOIR: marks interrupt Inactive, allowing re-trigger.
/// Must be called AFTER processing the interrupt.
///
/// # Safety
///
/// INTID must match value from acknowledge_irq().
pub unsafe fn end_of_interrupt(intid: u32) {
    write_cpu(GICC_EOIR, intid);
}

/// Enable specific INTID at the Distributor.
///
/// Sets GICD_ISENABLER bit for given INTID. For SPIs (32+) and PPIs (16-31).
///
/// # Safety
///
/// Must be called after GIC init.
#[allow(dead_code)]
pub unsafe fn enable_interrupt(intid: u32) {
    let reg_idx = (intid / 32) as usize;
    let bit_idx = intid % 32;
    write_dist(GICD_ISENABLER + reg_idx * 4, 1 << bit_idx);
}
