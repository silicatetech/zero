// SPDX-License-Identifier: AGPL-3.0-or-later
//! aarch64 boot entry point.
//!
//! Per Linux arm64 booting.rst:
//!   - x0 = DTB physical address
//!   - x1-x3 = reserved (must be 0)
//!   - EL1 (or EL2 with drop needed)
//!   - MMU disabled, caches disabled
//!
//! CITE: Linux Documentation/arm64/booting.rst
//! CITE: ARM ARM C1.1 — SP must be 16-byte aligned (AAPCS64)
//!
//! DTB pointer is preserved in callee-saved x19 across BSS zero,
//! then passed as first argument (x0) to kernel_main_aarch64 per
//! AAPCS64. No static mut global — eliminates BSS-zero-globals
//! coupling (Lessons-Learned #5 candidate).

use core::arch::global_asm;

/// Early stack — 64 KiB, aligned to 16 bytes (AAPCS64 requirement).
#[repr(align(16))]
#[allow(dead_code)]
pub struct EarlyStack([u8; 65536]);

#[no_mangle]
pub static mut EARLY_STACK: EarlyStack = EarlyStack([0; 65536]);

// BSS bounds — provided by linker script.
extern "C" {
    static __bss_start: u8;
    static __bss_end: u8;
}

// Assembly boot stub — must be in .text.boot section to appear
// at the ELF entry point (first in the linker script).
//
// CRITICAL: QEMU -kernel with ELF may enter at the ELF entry point
// with x0=DTB. We must capture x0 FIRST before any other operation.
//
// Register discipline:
//   x0  = DTB pointer (from QEMU firmware, Linux boot protocol)
//   x19 = callee-saved register used to preserve DTB across BSS zero
//   x1, x2 = scratch
//   sp  = set to top of EARLY_STACK
global_asm!(
    ".section .text.boot",
    ".global _start",
    ".type _start, @function",
    "_start:",
    // FIRST INSTRUCTION: preserve DTB pointer from x0
    // x19 is callee-saved per AAPCS64, safe across function calls
    "mov x19, x0",
    // Zero BSS section
    "adrp x1, __bss_start",
    "add  x1, x1, :lo12:__bss_start",
    "adrp x2, __bss_end",
    "add  x2, x2, :lo12:__bss_end",
    "1:",
    "cmp  x1, x2",
    "b.hs 2f",
    "str  xzr, [x1], #8",
    "b    1b",
    "2:",
    // Set up early stack (stack grows downward, point to top)
    "adrp x1, EARLY_STACK",
    "add  x1, x1, :lo12:EARLY_STACK",
    "add  x1, x1, #65536", // top of 64 KiB stack
    "mov  sp, x1",
    // Enable FP/SIMD access at EL1 via CPACR_EL1.FPEN (bits [21:20]).
    // Per ARM ARM D13.2.19: FPEN = 0b11 → no trapping of FP/SIMD at EL0/EL1.
    // Required because LLVM may emit NEON instructions for write!() formatting,
    // memcpy, etc. Without this, FP access triggers EC=0x07 Synchronous Exception.
    "mov  x1, #(3 << 20)",
    "msr  cpacr_el1, x1",
    "isb",
    // Restore DTB pointer to x0 (AAPCS64 first arg)
    "mov  x0, x19",
    // Branch to Rust kernel main: kernel_main_aarch64(dtb_ptr: usize)
    "b    kernel_main_aarch64",
);
