// SPDX-License-Identifier: AGPL-3.0-or-later
//! PL011 UART driver for aarch64 (QEMU virt machine).
//!
//! Per ARM PrimeCell UART (PL011) Technical Reference Manual (DDI 0183G):
//! - DR  (Data Register):    offset 0x000
//! - FR  (Flag Register):    offset 0x018, TXFF bit 5
//!
//! Two-phase initialization (Sub-MP-D2b Stage 4):
//! - Early: hardcoded base 0x09000000 (works pre-DTB-parse)
//! - Late:  DTB-discovered base via `set_base()` (post-Stage-3)
//!
//! On QEMU virt both bases are identical (0x9000000). On real
//! hardware they may differ — foundation for Stage 21.

use core::fmt;
use core::sync::atomic::{AtomicUsize, Ordering};

/// PL011 register offsets (per ARM PL011 TRM DDI 0183G Table 3-1).
const UARTDR: usize = 0x000; // Data Register
const UARTFR: usize = 0x018; // Flag Register
const UARTFR_TXFF: u32 = 1 << 5; // TX FIFO Full flag

/// Hardcoded PL011 base for QEMU virt machine.
/// Used during early boot before DTB parse completes.
const EARLY_UART_BASE: usize = 0x0900_0000;

/// Active UART base address.
/// Initially EARLY_UART_BASE; switched to DTB-discovered base
/// during Stage 4 via `set_base()`.
static UART_BASE: AtomicUsize = AtomicUsize::new(EARLY_UART_BASE);

/// Write single byte to PL011, blocking until TX FIFO has space.
fn write_byte(byte: u8) {
    let base = UART_BASE.load(Ordering::Relaxed);
    unsafe {
        // Wait for TX FIFO not full (FR.TXFF clear)
        while (core::ptr::read_volatile((base + UARTFR) as *const u32) & UARTFR_TXFF) != 0 {
            core::hint::spin_loop();
        }
        // Write byte to data register
        core::ptr::write_volatile((base + UARTDR) as *mut u32, byte as u32);
    }
}

/// Write a string to PL011 (with CRLF conversion).
fn write_str_serial(s: &str) {
    for byte in s.bytes() {
        if byte == b'\n' {
            write_byte(b'\r'); // CRLF for serial terminals
        }
        write_byte(byte);
    }
}

/// Serial output singleton for `write!` / `writeln!` macros.
/// Matches x86_64's `serial::Serial` HAL contract.
pub struct Serial;

impl fmt::Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        write_str_serial(s);
        Ok(())
    }
}

/// PL011 driver instance (for use when an owned struct is needed).
#[allow(dead_code)]
pub struct Pl011 {
    base: usize,
}

#[allow(dead_code)]
impl Pl011 {
    /// Create a PL011 driver at the given MMIO base address.
    pub const fn new(base: usize) -> Self {
        Self { base }
    }

    /// Write a single byte. Blocks until TX FIFO has space.
    pub fn write_byte(&self, byte: u8) {
        unsafe {
            while (core::ptr::read_volatile((self.base + UARTFR) as *const u32) & UARTFR_TXFF) != 0
            {
            }
            core::ptr::write_volatile((self.base + UARTDR) as *mut u32, byte as u32);
        }
    }
}

impl fmt::Write for Pl011 {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            if byte == b'\n' {
                self.write_byte(b'\r');
            }
            self.write_byte(byte);
        }
        Ok(())
    }
}

/// Initialize early PL011 UART at hardcoded address.
/// Called before DTB parse — QEMU virt always places PL011
/// at 0x0900_0000, no configuration needed (already initialized
/// by QEMU firmware).
pub fn init_early() {
    // QEMU virt PL011 is pre-configured — no register setup needed.
    // UART_BASE already set to EARLY_UART_BASE via AtomicUsize::new.
}

/// Initialize PL011 UART (alias for init_early).
#[allow(dead_code)]
pub fn init() {
    init_early();
}

/// Switch UART base to DTB-discovered address.
///
/// Called during Stage 4 after DTB parse provides actual UART base.
/// On QEMU virt, this is 0x09000000 (matches EARLY_UART_BASE).
/// On real hardware, may differ — Stage 21 territory.
///
/// # Safety
///
/// The provided base must point to a PL011-compatible MMIO region.
pub unsafe fn set_base(new_base: usize) {
    UART_BASE.store(new_base, Ordering::Release);
}

/// Get current UART base (for diagnostics).
pub fn current_base() -> usize {
    UART_BASE.load(Ordering::Acquire)
}

/// Print a string followed by newline.
pub fn println(s: &str) {
    write_str_serial(s);
    write_byte(b'\r');
    write_byte(b'\n');
}
