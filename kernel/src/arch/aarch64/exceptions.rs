// SPDX-License-Identifier: AGPL-3.0-or-later
//! AArch64 exception handling.
//!
//! Per ARM Architecture Reference Manual D1.10.2 "Vector tables":
//! Vector table base address held in VBAR_ELx, must be 2 KiB aligned.
//! 16 entries × 128 bytes = 2048 bytes total.
//!
//! Stage 2: Sync/FIQ/SError handlers print diagnostic + halt.
//! Stage 8: IRQ handlers dispatch via GICv2 (acknowledge + EOI).
//! Stage 9: IRQ handlers use full context save/restore + ERET for
//!          proper return-from-interrupt (required for timer ticks).
//!
//! CITE: ARM ARM D1.10.2 — Vector tables
//! CITE: ARM ARM D13.2.37 — ESR_EL1 (Exception Syndrome Register)
//! CITE: ARM ARM D13.2.36 — ELR_EL1 (Exception Link Register)
//! CITE: ARM ARM D13.2.38 — FAR_EL1 (Fault Address Register)
//! CITE: ARM ARM D13.2.149 — SPSR_EL1 (Saved Process State Register)

use crate::arch::aarch64::serial::Serial;
use core::fmt::Write;

// ---- Vector table assembly ----
//
// Per ARM ARM D1.10.2: 16 entries, each 128 bytes (32 instructions max).
//
// Sync/FIQ/SError entries: branch to Rust -> ! handlers (report + halt).
// IRQ entries: full context save, bl to Rust handler, context restore, eret.
//
// IRQ context save/restore (Stage 9):
//   Save x0-x30 + ELR_EL1 + SPSR_EL1 on stack (34 × 8 = 272 bytes,
//   rounded to 288 for 16-byte SP alignment per AAPCS64).
//   Call Rust handler via bl (handler returns normally).
//   Restore all registers, then eret to interrupted context.
//
// Layout:
//   0x000..0x1FF  Current EL with SP_EL0  (Sync, IRQ, FIQ, SError)
//   0x200..0x3FF  Current EL with SP_ELx  (Sync, IRQ, FIQ, SError)
//   0x400..0x5FF  Lower EL using AArch64  (Sync, IRQ, FIQ, SError)
//   0x600..0x7FF  Lower EL using AArch32  (Sync, IRQ, FIQ, SError)

core::arch::global_asm!(
    "
.section .text.exception_vectors
.balign 2048
.global exception_vector_table
exception_vector_table:

    // ---- Current EL with SP_EL0 ----
    .balign 128
    b vec_curr_el_sp0_sync
    .balign 128
    b irq_entry_curr_el_sp0
    .balign 128
    b vec_curr_el_sp0_fiq
    .balign 128
    b vec_curr_el_sp0_serror

    // ---- Current EL with SP_ELx ----
    .balign 128
    b vec_curr_el_spx_sync
    .balign 128
    b irq_entry_curr_el_spx
    .balign 128
    b vec_curr_el_spx_fiq
    .balign 128
    b vec_curr_el_spx_serror

    // ---- Lower EL using AArch64 ----
    .balign 128
    b vec_lower_el_aarch64_sync
    .balign 128
    b irq_entry_lower_el_aarch64
    .balign 128
    b vec_lower_el_aarch64_fiq
    .balign 128
    b vec_lower_el_aarch64_serror

    // ---- Lower EL using AArch32 ----
    .balign 128
    b vec_lower_el_aarch32_sync
    .balign 128
    b irq_entry_lower_el_aarch32
    .balign 128
    b vec_lower_el_aarch32_fiq
    .balign 128
    b vec_lower_el_aarch32_serror

// ---- IRQ entry stubs: context save, call Rust, restore, eret ----
//
// Per ARM ARM D1.10: on exception entry, hardware saves PSTATE to
// SPSR_EL1 and return address to ELR_EL1. Software must save GP regs.
//
// Stack frame (800 bytes, 16-byte aligned per AAPCS64):
//   [sp, #0]...[sp, #240]   x0-x30 (31 regs × 8 = 248 bytes)
//   [sp, #248]              ELR_EL1 (return address)
//   [sp, #256]              SPSR_EL1 (saved PSTATE)
//   [sp, #264]              padding (alignment to 288)
//   [sp, #288]...[sp, #799] q0-q31 NEON (32 × 16 = 512 bytes)
//
// Lesson 10: NEON save/restore required when IRQ interrupts code
// that uses q0-q31 (inference matrix math, LLVM-vectorized ops).
// Stack is post-MMU Normal Memory — STP safe (Lesson 11 N/A).

irq_entry_curr_el_sp0:
    sub sp, sp, #800
    stp x0,  x1,  [sp, #16*0]
    stp x2,  x3,  [sp, #16*1]
    stp x4,  x5,  [sp, #16*2]
    stp x6,  x7,  [sp, #16*3]
    stp x8,  x9,  [sp, #16*4]
    stp x10, x11, [sp, #16*5]
    stp x12, x13, [sp, #16*6]
    stp x14, x15, [sp, #16*7]
    stp x16, x17, [sp, #16*8]
    stp x18, x19, [sp, #16*9]
    stp x20, x21, [sp, #16*10]
    stp x22, x23, [sp, #16*11]
    stp x24, x25, [sp, #16*12]
    stp x26, x27, [sp, #16*13]
    stp x28, x29, [sp, #16*14]
    str x30,      [sp, #16*15]
    mrs x0, elr_el1
    mrs x1, spsr_el1
    stp x0, x1, [sp, #248]
    stp q0,  q1,  [sp, #288]
    stp q2,  q3,  [sp, #320]
    stp q4,  q5,  [sp, #352]
    stp q6,  q7,  [sp, #384]
    stp q8,  q9,  [sp, #416]
    stp q10, q11, [sp, #448]
    stp q12, q13, [sp, #480]
    stp q14, q15, [sp, #512]
    stp q16, q17, [sp, #544]
    stp q18, q19, [sp, #576]
    stp q20, q21, [sp, #608]
    stp q22, q23, [sp, #640]
    stp q24, q25, [sp, #672]
    stp q26, q27, [sp, #704]
    stp q28, q29, [sp, #736]
    stp q30, q31, [sp, #768]
    bl handle_irq_rust
    ldp q0,  q1,  [sp, #288]
    ldp q2,  q3,  [sp, #320]
    ldp q4,  q5,  [sp, #352]
    ldp q6,  q7,  [sp, #384]
    ldp q8,  q9,  [sp, #416]
    ldp q10, q11, [sp, #448]
    ldp q12, q13, [sp, #480]
    ldp q14, q15, [sp, #512]
    ldp q16, q17, [sp, #544]
    ldp q18, q19, [sp, #576]
    ldp q20, q21, [sp, #608]
    ldp q22, q23, [sp, #640]
    ldp q24, q25, [sp, #672]
    ldp q26, q27, [sp, #704]
    ldp q28, q29, [sp, #736]
    ldp q30, q31, [sp, #768]
    ldp x0, x1, [sp, #248]
    msr elr_el1, x0
    msr spsr_el1, x1
    ldp x0,  x1,  [sp, #16*0]
    ldp x2,  x3,  [sp, #16*1]
    ldp x4,  x5,  [sp, #16*2]
    ldp x6,  x7,  [sp, #16*3]
    ldp x8,  x9,  [sp, #16*4]
    ldp x10, x11, [sp, #16*5]
    ldp x12, x13, [sp, #16*6]
    ldp x14, x15, [sp, #16*7]
    ldp x16, x17, [sp, #16*8]
    ldp x18, x19, [sp, #16*9]
    ldp x20, x21, [sp, #16*10]
    ldp x22, x23, [sp, #16*11]
    ldp x24, x25, [sp, #16*12]
    ldp x26, x27, [sp, #16*13]
    ldp x28, x29, [sp, #16*14]
    ldr x30,      [sp, #16*15]
    add sp, sp, #800
    eret

irq_entry_curr_el_spx:
    sub sp, sp, #800
    stp x0,  x1,  [sp, #16*0]
    stp x2,  x3,  [sp, #16*1]
    stp x4,  x5,  [sp, #16*2]
    stp x6,  x7,  [sp, #16*3]
    stp x8,  x9,  [sp, #16*4]
    stp x10, x11, [sp, #16*5]
    stp x12, x13, [sp, #16*6]
    stp x14, x15, [sp, #16*7]
    stp x16, x17, [sp, #16*8]
    stp x18, x19, [sp, #16*9]
    stp x20, x21, [sp, #16*10]
    stp x22, x23, [sp, #16*11]
    stp x24, x25, [sp, #16*12]
    stp x26, x27, [sp, #16*13]
    stp x28, x29, [sp, #16*14]
    str x30,      [sp, #16*15]
    mrs x0, elr_el1
    mrs x1, spsr_el1
    stp x0, x1, [sp, #248]
    stp q0,  q1,  [sp, #288]
    stp q2,  q3,  [sp, #320]
    stp q4,  q5,  [sp, #352]
    stp q6,  q7,  [sp, #384]
    stp q8,  q9,  [sp, #416]
    stp q10, q11, [sp, #448]
    stp q12, q13, [sp, #480]
    stp q14, q15, [sp, #512]
    stp q16, q17, [sp, #544]
    stp q18, q19, [sp, #576]
    stp q20, q21, [sp, #608]
    stp q22, q23, [sp, #640]
    stp q24, q25, [sp, #672]
    stp q26, q27, [sp, #704]
    stp q28, q29, [sp, #736]
    stp q30, q31, [sp, #768]
    bl handle_irq_rust
    ldp q0,  q1,  [sp, #288]
    ldp q2,  q3,  [sp, #320]
    ldp q4,  q5,  [sp, #352]
    ldp q6,  q7,  [sp, #384]
    ldp q8,  q9,  [sp, #416]
    ldp q10, q11, [sp, #448]
    ldp q12, q13, [sp, #480]
    ldp q14, q15, [sp, #512]
    ldp q16, q17, [sp, #544]
    ldp q18, q19, [sp, #576]
    ldp q20, q21, [sp, #608]
    ldp q22, q23, [sp, #640]
    ldp q24, q25, [sp, #672]
    ldp q26, q27, [sp, #704]
    ldp q28, q29, [sp, #736]
    ldp q30, q31, [sp, #768]
    ldp x0, x1, [sp, #248]
    msr elr_el1, x0
    msr spsr_el1, x1
    ldp x0,  x1,  [sp, #16*0]
    ldp x2,  x3,  [sp, #16*1]
    ldp x4,  x5,  [sp, #16*2]
    ldp x6,  x7,  [sp, #16*3]
    ldp x8,  x9,  [sp, #16*4]
    ldp x10, x11, [sp, #16*5]
    ldp x12, x13, [sp, #16*6]
    ldp x14, x15, [sp, #16*7]
    ldp x16, x17, [sp, #16*8]
    ldp x18, x19, [sp, #16*9]
    ldp x20, x21, [sp, #16*10]
    ldp x22, x23, [sp, #16*11]
    ldp x24, x25, [sp, #16*12]
    ldp x26, x27, [sp, #16*13]
    ldp x28, x29, [sp, #16*14]
    ldr x30,      [sp, #16*15]
    add sp, sp, #800
    eret

irq_entry_lower_el_aarch64:
    sub sp, sp, #800
    stp x0,  x1,  [sp, #16*0]
    stp x2,  x3,  [sp, #16*1]
    stp x4,  x5,  [sp, #16*2]
    stp x6,  x7,  [sp, #16*3]
    stp x8,  x9,  [sp, #16*4]
    stp x10, x11, [sp, #16*5]
    stp x12, x13, [sp, #16*6]
    stp x14, x15, [sp, #16*7]
    stp x16, x17, [sp, #16*8]
    stp x18, x19, [sp, #16*9]
    stp x20, x21, [sp, #16*10]
    stp x22, x23, [sp, #16*11]
    stp x24, x25, [sp, #16*12]
    stp x26, x27, [sp, #16*13]
    stp x28, x29, [sp, #16*14]
    str x30,      [sp, #16*15]
    mrs x0, elr_el1
    mrs x1, spsr_el1
    stp x0, x1, [sp, #248]
    stp q0,  q1,  [sp, #288]
    stp q2,  q3,  [sp, #320]
    stp q4,  q5,  [sp, #352]
    stp q6,  q7,  [sp, #384]
    stp q8,  q9,  [sp, #416]
    stp q10, q11, [sp, #448]
    stp q12, q13, [sp, #480]
    stp q14, q15, [sp, #512]
    stp q16, q17, [sp, #544]
    stp q18, q19, [sp, #576]
    stp q20, q21, [sp, #608]
    stp q22, q23, [sp, #640]
    stp q24, q25, [sp, #672]
    stp q26, q27, [sp, #704]
    stp q28, q29, [sp, #736]
    stp q30, q31, [sp, #768]
    bl handle_irq_rust
    ldp q0,  q1,  [sp, #288]
    ldp q2,  q3,  [sp, #320]
    ldp q4,  q5,  [sp, #352]
    ldp q6,  q7,  [sp, #384]
    ldp q8,  q9,  [sp, #416]
    ldp q10, q11, [sp, #448]
    ldp q12, q13, [sp, #480]
    ldp q14, q15, [sp, #512]
    ldp q16, q17, [sp, #544]
    ldp q18, q19, [sp, #576]
    ldp q20, q21, [sp, #608]
    ldp q22, q23, [sp, #640]
    ldp q24, q25, [sp, #672]
    ldp q26, q27, [sp, #704]
    ldp q28, q29, [sp, #736]
    ldp q30, q31, [sp, #768]
    ldp x0, x1, [sp, #248]
    msr elr_el1, x0
    msr spsr_el1, x1
    ldp x0,  x1,  [sp, #16*0]
    ldp x2,  x3,  [sp, #16*1]
    ldp x4,  x5,  [sp, #16*2]
    ldp x6,  x7,  [sp, #16*3]
    ldp x8,  x9,  [sp, #16*4]
    ldp x10, x11, [sp, #16*5]
    ldp x12, x13, [sp, #16*6]
    ldp x14, x15, [sp, #16*7]
    ldp x16, x17, [sp, #16*8]
    ldp x18, x19, [sp, #16*9]
    ldp x20, x21, [sp, #16*10]
    ldp x22, x23, [sp, #16*11]
    ldp x24, x25, [sp, #16*12]
    ldp x26, x27, [sp, #16*13]
    ldp x28, x29, [sp, #16*14]
    ldr x30,      [sp, #16*15]
    add sp, sp, #800
    eret

irq_entry_lower_el_aarch32:
    sub sp, sp, #800
    stp x0,  x1,  [sp, #16*0]
    stp x2,  x3,  [sp, #16*1]
    stp x4,  x5,  [sp, #16*2]
    stp x6,  x7,  [sp, #16*3]
    stp x8,  x9,  [sp, #16*4]
    stp x10, x11, [sp, #16*5]
    stp x12, x13, [sp, #16*6]
    stp x14, x15, [sp, #16*7]
    stp x16, x17, [sp, #16*8]
    stp x18, x19, [sp, #16*9]
    stp x20, x21, [sp, #16*10]
    stp x22, x23, [sp, #16*11]
    stp x24, x25, [sp, #16*12]
    stp x26, x27, [sp, #16*13]
    stp x28, x29, [sp, #16*14]
    str x30,      [sp, #16*15]
    mrs x0, elr_el1
    mrs x1, spsr_el1
    stp x0, x1, [sp, #248]
    stp q0,  q1,  [sp, #288]
    stp q2,  q3,  [sp, #320]
    stp q4,  q5,  [sp, #352]
    stp q6,  q7,  [sp, #384]
    stp q8,  q9,  [sp, #416]
    stp q10, q11, [sp, #448]
    stp q12, q13, [sp, #480]
    stp q14, q15, [sp, #512]
    stp q16, q17, [sp, #544]
    stp q18, q19, [sp, #576]
    stp q20, q21, [sp, #608]
    stp q22, q23, [sp, #640]
    stp q24, q25, [sp, #672]
    stp q26, q27, [sp, #704]
    stp q28, q29, [sp, #736]
    stp q30, q31, [sp, #768]
    bl handle_irq_rust
    ldp q0,  q1,  [sp, #288]
    ldp q2,  q3,  [sp, #320]
    ldp q4,  q5,  [sp, #352]
    ldp q6,  q7,  [sp, #384]
    ldp q8,  q9,  [sp, #416]
    ldp q10, q11, [sp, #448]
    ldp q12, q13, [sp, #480]
    ldp q14, q15, [sp, #512]
    ldp q16, q17, [sp, #544]
    ldp q18, q19, [sp, #576]
    ldp q20, q21, [sp, #608]
    ldp q22, q23, [sp, #640]
    ldp q24, q25, [sp, #672]
    ldp q26, q27, [sp, #704]
    ldp q28, q29, [sp, #736]
    ldp q30, q31, [sp, #768]
    ldp x0, x1, [sp, #248]
    msr elr_el1, x0
    msr spsr_el1, x1
    ldp x0,  x1,  [sp, #16*0]
    ldp x2,  x3,  [sp, #16*1]
    ldp x4,  x5,  [sp, #16*2]
    ldp x6,  x7,  [sp, #16*3]
    ldp x8,  x9,  [sp, #16*4]
    ldp x10, x11, [sp, #16*5]
    ldp x12, x13, [sp, #16*6]
    ldp x14, x15, [sp, #16*7]
    ldp x16, x17, [sp, #16*8]
    ldp x18, x19, [sp, #16*9]
    ldp x20, x21, [sp, #16*10]
    ldp x22, x23, [sp, #16*11]
    ldp x24, x25, [sp, #16*12]
    ldp x26, x27, [sp, #16*13]
    ldp x28, x29, [sp, #16*14]
    ldr x30,      [sp, #16*15]
    add sp, sp, #800
    eret
"
);

// ---- Diagnostic helpers ----

/// Read exception state registers.
///
/// Per ARM ARM:
/// - ESR_EL1 (D13.2.37): Exception Syndrome — contains EC (class) + ISS (syndrome)
/// - ELR_EL1 (D13.2.36): Exception Link — return address
/// - FAR_EL1 (D13.2.38): Fault Address — address that caused abort
/// - SPSR_EL1 (D13.2.149): Saved Process State — PSTATE at exception time
#[inline(always)]
unsafe fn read_exception_state() -> (u64, u64, u64, u64) {
    let esr: u64;
    let elr: u64;
    let far: u64;
    let spsr: u64;

    core::arch::asm!(
        "mrs {esr}, esr_el1",
        "mrs {elr}, elr_el1",
        "mrs {far}, far_el1",
        "mrs {spsr}, spsr_el1",
        esr = out(reg) esr,
        elr = out(reg) elr,
        far = out(reg) far,
        spsr = out(reg) spsr,
        options(nomem, nostack, preserves_flags),
    );

    (esr, elr, far, spsr)
}

/// Decode ESR_EL1.EC (Exception Class, bits [31:26]).
/// Per ARM ARM D13.2.37 Table D13-1.
fn decode_exception_class(ec: u64) -> &'static str {
    match ec {
        0x00 => "Unknown reason",
        0x01 => "Trapped WFI/WFE",
        0x03 => "Trapped MCR/MRC (CP15)",
        0x04 => "Trapped MCRR/MRRC (CP15)",
        0x05 => "Trapped MCR/MRC (CP14)",
        0x06 => "Trapped LDC/STC (CP14)",
        0x07 => "Trapped SIMD/FP access",
        0x0C => "Trapped MRRC (CP14)",
        0x0E => "Illegal Execution state",
        0x11 => "SVC instruction (AArch32)",
        0x15 => "SVC instruction (AArch64)",
        0x18 => "Trapped MSR/MRS/System instruction",
        0x19 => "Trapped SVE access",
        0x20 => "Instruction Abort, lower EL",
        0x21 => "Instruction Abort, same EL",
        0x22 => "PC alignment fault",
        0x24 => "Data Abort, lower EL",
        0x25 => "Data Abort, same EL",
        0x26 => "SP alignment fault",
        0x28 => "Trapped FP exception (AArch32)",
        0x2C => "Trapped FP exception (AArch64)",
        0x2F => "SError interrupt",
        0x30 => "Breakpoint, lower EL",
        0x31 => "Breakpoint, same EL",
        0x32 => "Software Step, lower EL",
        0x33 => "Software Step, same EL",
        0x34 => "Watchpoint, lower EL",
        0x35 => "Watchpoint, same EL",
        0x38 => "BKPT instruction (AArch32)",
        0x3C => "BRK instruction (AArch64)",
        _ => "Reserved/Unknown",
    }
}

/// Print exception diagnostic and halt via wfi loop.
///
/// Reads ESR_EL1/ELR_EL1/FAR_EL1/SPSR_EL1, decodes exception class,
/// prints via PL011 Serial, then enters infinite wfi loop.
unsafe fn report_and_halt(label: &str) -> ! {
    let (esr, elr, far, spsr) = read_exception_state();

    // ESR.EC = bits [31:26]
    let ec = (esr >> 26) & 0x3F;

    let _ = writeln!(Serial, "");
    let _ = writeln!(Serial, "===============================================");
    let _ = writeln!(Serial, "AARCH64 EXCEPTION: {}", label);
    let _ = writeln!(Serial, "  ESR_EL1  = {:#018x}", esr);
    let _ = writeln!(Serial, "  ELR_EL1  = {:#018x}", elr);
    let _ = writeln!(Serial, "  FAR_EL1  = {:#018x}", far);
    let _ = writeln!(Serial, "  SPSR_EL1 = {:#018x}", spsr);
    let _ = writeln!(
        Serial,
        "  EC       = {:#04x} ({})",
        ec,
        decode_exception_class(ec)
    );
    let _ = writeln!(Serial, "===============================================");

    loop {
        core::arch::asm!("wfi", options(nomem, nostack, preserves_flags));
    }
}

// ---- Sync/FIQ/SError handler functions (report + halt) ----
//
// Each handler is #[no_mangle] extern "C" to be callable from the
// global_asm! vector table via `b` (branch). These never return.

// Current EL with SP_EL0

#[no_mangle]
unsafe extern "C" fn vec_curr_el_sp0_sync() -> ! {
    report_and_halt("Synchronous (Current EL, SP_EL0)");
}

#[no_mangle]
unsafe extern "C" fn vec_curr_el_sp0_fiq() -> ! {
    report_and_halt("FIQ (Current EL, SP_EL0)");
}

#[no_mangle]
unsafe extern "C" fn vec_curr_el_sp0_serror() -> ! {
    report_and_halt("SError (Current EL, SP_EL0)");
}

// Current EL with SP_ELx

#[no_mangle]
unsafe extern "C" fn vec_curr_el_spx_sync() -> ! {
    report_and_halt("Synchronous (Current EL, SP_ELx)");
}

#[no_mangle]
unsafe extern "C" fn vec_curr_el_spx_fiq() -> ! {
    report_and_halt("FIQ (Current EL, SP_ELx)");
}

#[no_mangle]
unsafe extern "C" fn vec_curr_el_spx_serror() -> ! {
    report_and_halt("SError (Current EL, SP_ELx)");
}

// Lower EL using AArch64

#[no_mangle]
unsafe extern "C" fn vec_lower_el_aarch64_sync() -> ! {
    report_and_halt("Synchronous (Lower EL, AArch64)");
}

#[no_mangle]
unsafe extern "C" fn vec_lower_el_aarch64_fiq() -> ! {
    report_and_halt("FIQ (Lower EL, AArch64)");
}

#[no_mangle]
unsafe extern "C" fn vec_lower_el_aarch64_serror() -> ! {
    report_and_halt("SError (Lower EL, AArch64)");
}

// Lower EL using AArch32

#[no_mangle]
unsafe extern "C" fn vec_lower_el_aarch32_sync() -> ! {
    report_and_halt("Synchronous (Lower EL, AArch32)");
}

#[no_mangle]
unsafe extern "C" fn vec_lower_el_aarch32_fiq() -> ! {
    report_and_halt("FIQ (Lower EL, AArch32)");
}

#[no_mangle]
unsafe extern "C" fn vec_lower_el_aarch32_serror() -> ! {
    report_and_halt("SError (Lower EL, AArch32)");
}

// ---- IRQ dispatch (Stage 9: returns via eret in assembly stub) ----
//
// All 4 IRQ vector entries call the same Rust handler via assembly
// stubs that save/restore context + eret. The Rust function returns
// normally; the assembly stub handles eret.

/// Unified IRQ handler — called from assembly IRQ entry stubs.
///
/// Dispatches by INTID:
/// - INTID 27 (Virtual Timer PPI): timer::handle_tick()
/// - INTID 1023 (spurious): ignore
/// - Other: log + EOI
#[no_mangle]
unsafe extern "C" fn handle_irq_rust() {
    use crate::arch::aarch64::{gic, timer};

    let intid = gic::acknowledge_irq();

    if intid == gic::GICC_SPURIOUS {
        return;
    }

    if intid == timer::TIMER_INTID {
        timer::handle_tick();
    } else {
        let _ = writeln!(Serial, "IRQ: unhandled INTID={}", intid);
    }

    gic::end_of_interrupt(intid);
}

// ---- Public initialization ----

extern "C" {
    static exception_vector_table: u8;
}

/// Install VBAR_EL1 exception vector table.
///
/// Per ARM ARM D1.10.2: vector table base must be 2 KiB aligned.
/// Section `.text.exception_vectors` enforces alignment via `.balign 2048`.
///
/// After MSR + ISB, all exceptions route through our table:
/// - Sync/FIQ/SError: diagnostic via PL011 Serial → halt via wfi loop
/// - IRQ: context save → GIC dispatch → context restore → eret
///
/// MUST be called BEFORE any non-trivial code that might trigger
/// exceptions (DTB parsing, memory access, etc.).
///
/// # Safety
///
/// Must be called at EL1. The vector table must be in accessible memory.
pub unsafe fn init() {
    let vbar = &exception_vector_table as *const u8 as u64;

    core::arch::asm!(
        "msr vbar_el1, {vbar}",
        "isb",
        vbar = in(reg) vbar,
        options(nomem, nostack, preserves_flags),
    );
}
