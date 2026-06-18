// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interrupt Descriptor Table + CPU-exception handlers.
//!
//! Stage 1. Wires every documented CPU exception (0–21, minus the
//! reserved/deprecated slots) to a handler that prints the frame
//! on the serial port and halts. Breakpoint is special — it prints
//! and returns, so `int3` is usable as a sanity probe.
//!
//! Hardware interrupts (IRQ 32+) are *not* wired here. That is
//! Stage 2+ territory (timer, keyboard, etc.).

use core::fmt::Write;
use lazy_static::lazy_static;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};

use super::gdt::DOUBLE_FAULT_IST_INDEX;
use super::{pic, pit, serial};

lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();

        // No error code
        idt.divide_error.set_handler_fn(divide_error);
        idt.debug.set_handler_fn(debug);
        idt.non_maskable_interrupt.set_handler_fn(nmi);
        idt.breakpoint.set_handler_fn(breakpoint);
        idt.overflow.set_handler_fn(overflow);
        idt.bound_range_exceeded.set_handler_fn(bound_range);
        idt.invalid_opcode.set_handler_fn(invalid_opcode);
        idt.device_not_available.set_handler_fn(device_not_available);
        idt.x87_floating_point.set_handler_fn(x87_floating_point);
        idt.simd_floating_point.set_handler_fn(simd_floating_point);
        idt.virtualization.set_handler_fn(virtualization);

        // With error code
        idt.invalid_tss.set_handler_fn(invalid_tss);
        idt.segment_not_present.set_handler_fn(segment_not_present);
        idt.stack_segment_fault.set_handler_fn(stack_segment_fault);
        idt.general_protection_fault.set_handler_fn(general_protection_fault);
        idt.alignment_check.set_handler_fn(alignment_check);

        // Page fault — its own error-code type carrying protection flags.
        idt.page_fault.set_handler_fn(page_fault);

        // Double fault on IST stack 0 — review-checklist item #1.
        // Without `set_stack_index` a stack-overflow = triple fault.
        unsafe {
            idt.double_fault
                .set_handler_fn(double_fault)
                .set_stack_index(DOUBLE_FAULT_IST_INDEX);
        }

        // Machine check is unrecoverable (diverging handler).
        idt.machine_check.set_handler_fn(machine_check);

        // Hardware interrupts, remapped to 32.. by `pic::init()`.
        // Timer = IRQ 0 = vector `MASTER_OFFSET` after the PIC remap.
        idt[pic::MASTER_OFFSET].set_handler_fn(timer);

        // Every other vector in the PIC range (33–47) must have a
        // present IDT entry too — otherwise a stray or spurious IRQ
        // (notably the master's spurious IRQ 7 = vector 39, seen on
        // real AMD EPYC silicon right after `sti`) raises #NP because
        // the gate descriptor is not-present.
        //
        // Master lines 1–6 (vectors 33–38) ack the master PIC; slave
        // lines 8–14 (vectors 40–46) ack both. Spurious IRQ 7 / 15
        // (vectors 39 / 47) intentionally skip the EOI: the master
        // never latched the request, so EOIing would clear a real
        // pending IRQ on a lower line.
        idt[pic::MASTER_OFFSET + 1].set_handler_fn(irq_master_stub);  // IRQ 1
        idt[pic::MASTER_OFFSET + 2].set_handler_fn(irq_master_stub);  // IRQ 2 (cascade)
        idt[pic::MASTER_OFFSET + 3].set_handler_fn(irq_master_stub);  // IRQ 3
        idt[pic::MASTER_OFFSET + 4].set_handler_fn(irq_master_stub);  // IRQ 4
        idt[pic::MASTER_OFFSET + 5].set_handler_fn(irq_master_stub);  // IRQ 5
        idt[pic::MASTER_OFFSET + 6].set_handler_fn(irq_master_stub);  // IRQ 6
        idt[pic::MASTER_OFFSET + 7].set_handler_fn(spurious_master);  // IRQ 7 (spurious)
        idt[pic::SLAVE_OFFSET].set_handler_fn(irq_slave_stub);        // IRQ 8
        idt[pic::SLAVE_OFFSET + 1].set_handler_fn(irq_slave_stub);    // IRQ 9
        idt[pic::SLAVE_OFFSET + 2].set_handler_fn(irq_slave_stub);    // IRQ 10
        idt[pic::SLAVE_OFFSET + 3].set_handler_fn(irq_slave_stub);    // IRQ 11
        idt[pic::SLAVE_OFFSET + 4].set_handler_fn(irq_slave_stub);    // IRQ 12
        idt[pic::SLAVE_OFFSET + 5].set_handler_fn(irq_slave_stub);    // IRQ 13
        idt[pic::SLAVE_OFFSET + 6].set_handler_fn(irq_slave_stub);    // IRQ 14
        idt[pic::SLAVE_OFFSET + 7].set_handler_fn(spurious_slave);    // IRQ 15 (spurious)

        idt
    };
}

/// Load the IDT. Must be called after `gdt::init`, because the IDT's
/// double-fault entry references the TSS installed by the GDT.
pub fn init() {
    IDT.load();
}

/// Load the shared IDT on an AP during SMP bring-up.
///
/// APs currently run on the trampoline GDT and do not have per-core
/// TSS/IST state yet. Loading the IDT here still gives useful diagnostics
/// for ordinary early AP exceptions (#UD, #NM, #GP, #PF) instead of a
/// silent triple fault. Double-fault hardening still requires per-core
/// GDT/TSS/IST state.
pub fn init_ap_exception_table() {
    IDT.load();
}

// ---- Formatting helpers --------------------------------------------

fn report(label: &str, frame: &InterruptStackFrame) {
    let _ = writeln!(
        serial::Serial,
        "\r\nEXCEPTION: {label}\r\n  RIP    = {:#018x}\r\n  CS     = {:?}\r\n  RFLAGS = {:?}\r\n  RSP    = {:#018x}\r\n  SS     = {:?}",
        frame.instruction_pointer.as_u64(),
        frame.code_segment,
        frame.cpu_flags,
        frame.stack_pointer.as_u64(),
        frame.stack_segment,
    );
}

fn report_with_code(label: &str, frame: &InterruptStackFrame, code: u64) {
    report(label, frame);
    let _ = writeln!(serial::Serial, "  ERROR  = {:#x}", code);
}

fn halt() -> ! {
    loop {
        unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
    }
}

// ---- Handlers — no error code --------------------------------------

extern "x86-interrupt" fn divide_error(frame: InterruptStackFrame) {
    report("DIVIDE ERROR (#DE)", &frame);
    halt();
}

extern "x86-interrupt" fn debug(frame: InterruptStackFrame) {
    report("DEBUG (#DB)", &frame);
    halt();
}

extern "x86-interrupt" fn nmi(frame: InterruptStackFrame) {
    // Emit a single short line FIRST so the operator sees the NMI even
    // if the full `report` formatter wedges on a follow-up fault. NMIs
    // on real hardware are usually a hang signal (watchdog, IPMI, MCE
    // precursor), so the RIP at NMI time is the most valuable single
    // datum we can surface before halting.
    let _ = writeln!(
        serial::Serial,
        "\r\nNMI received — RIP = {:#018x}",
        frame.instruction_pointer.as_u64(),
    );
    report("NMI", &frame);
    halt();
}

/// Special: returns. Lets `int3` serve as a "did the IDT path work
/// end-to-end" probe from `kernel_main`.
extern "x86-interrupt" fn breakpoint(frame: InterruptStackFrame) {
    let _ = writeln!(
        serial::Serial,
        "\r\n[breakpoint handler reached — IDT live, returning to caller]\r\n  RIP    = {:#018x}",
        frame.instruction_pointer.as_u64(),
    );
}

extern "x86-interrupt" fn overflow(frame: InterruptStackFrame) {
    report("OVERFLOW (#OF)", &frame);
    halt();
}

extern "x86-interrupt" fn bound_range(frame: InterruptStackFrame) {
    report("BOUND RANGE EXCEEDED (#BR)", &frame);
    halt();
}

extern "x86-interrupt" fn invalid_opcode(frame: InterruptStackFrame) {
    report("INVALID OPCODE (#UD)", &frame);
    halt();
}

extern "x86-interrupt" fn device_not_available(frame: InterruptStackFrame) {
    report("DEVICE NOT AVAILABLE (#NM)", &frame);
    halt();
}

extern "x86-interrupt" fn x87_floating_point(frame: InterruptStackFrame) {
    report("x87 FLOATING POINT (#MF)", &frame);
    halt();
}

extern "x86-interrupt" fn simd_floating_point(frame: InterruptStackFrame) {
    report("SIMD FLOATING POINT (#XM)", &frame);
    halt();
}

extern "x86-interrupt" fn virtualization(frame: InterruptStackFrame) {
    report("VIRTUALIZATION (#VE)", &frame);
    halt();
}

// ---- Handlers — with error code ------------------------------------

extern "x86-interrupt" fn invalid_tss(frame: InterruptStackFrame, code: u64) {
    report_with_code("INVALID TSS (#TS)", &frame, code);
    halt();
}

extern "x86-interrupt" fn segment_not_present(frame: InterruptStackFrame, code: u64) {
    report_with_code("SEGMENT NOT PRESENT (#NP)", &frame, code);
    halt();
}

extern "x86-interrupt" fn stack_segment_fault(frame: InterruptStackFrame, code: u64) {
    report_with_code("STACK SEGMENT FAULT (#SS)", &frame, code);
    halt();
}

extern "x86-interrupt" fn general_protection_fault(frame: InterruptStackFrame, code: u64) {
    report_with_code("GENERAL PROTECTION FAULT (#GP)", &frame, code);
    halt();
}

extern "x86-interrupt" fn alignment_check(frame: InterruptStackFrame, code: u64) {
    report_with_code("ALIGNMENT CHECK (#AC)", &frame, code);
    halt();
}

// ---- Page fault — reads CR2 ----------------------------------------

extern "x86-interrupt" fn page_fault(frame: InterruptStackFrame, code: PageFaultErrorCode) {
    // CR2 holds the faulting virtual address. Read directly via asm;
    // the x86_64 crate's `Cr2::read` signature has drifted across
    // versions, and asm is version-stable.
    let cr2: u64;
    unsafe {
        core::arch::asm!(
            "mov {}, cr2",
            out(reg) cr2,
            options(nomem, nostack, preserves_flags),
        );
    }

    report("PAGE FAULT (#PF)", &frame);
    let _ = writeln!(
        serial::Serial,
        "  CR2    = {:#018x}\r\n  ERROR  = {:?}",
        cr2,
        code,
    );
    halt();
}

// ---- Diverging handlers --------------------------------------------

extern "x86-interrupt" fn double_fault(frame: InterruptStackFrame, code: u64) -> ! {
    let _ = writeln!(serial::Serial, "\r\nEXCEPTION: DOUBLE FAULT (#DF)");
    let _ = writeln!(
        serial::Serial,
        "  RIP    = {:#018x}\r\n  ERROR  = {:#x}",
        frame.instruction_pointer.as_u64(),
        code,
    );
    halt();
}

extern "x86-interrupt" fn machine_check(frame: InterruptStackFrame) -> ! {
    let _ = writeln!(serial::Serial, "\r\nEXCEPTION: MACHINE CHECK (#MC)");
    let _ = writeln!(
        serial::Serial,
        "  RIP = {:#018x}",
        frame.instruction_pointer.as_u64(),
    );
    halt();
}

// ---- Hardware IRQ handlers -----------------------------------------

/// Timer tick (IRQ 0, vector 32 after PIC remap).
///
/// Must be short: every extra microsecond here is latency for every
/// other IRQ in the system. Increment the counter, acknowledge the
/// master PIC, return.
extern "x86-interrupt" fn timer(_frame: InterruptStackFrame) {
    pit::tick();
    // Drain RX / drive TCP retransmits from the ISR when a stack has
    // been registered via `net::register_irq_poll`. Keeps the shell
    // surfaces reachable while the BSP is blocked inside the
    // Stage-11 forward-pass (single-core boot path).
    crate::net::irq_poll_tick();
    pic::send_eoi_master();
}

/// Catch-all for master-PIC IRQs we don't drive yet (vectors 33–38).
/// All other lines are masked at boot, but the gate must still be
/// present so a stray edge doesn't raise #NP. EOI the master so the
/// PIC doesn't wedge if the line ever does fire.
extern "x86-interrupt" fn irq_master_stub(_frame: InterruptStackFrame) {
    pic::send_eoi_master();
}

/// Catch-all for slave-PIC IRQs we don't drive yet (vectors 40–46).
/// EOI must go to both chips because the master saw the cascade.
extern "x86-interrupt" fn irq_slave_stub(_frame: InterruptStackFrame) {
    pic::send_eoi_slave();
}

/// Spurious master IRQ 7 (vector 39).
///
/// Triggered when the master PIC samples an INTR line that has gone
/// low again before the CPU acknowledges — observed on AMD EPYC right
/// after `sti`. The chip did *not* latch this in its ISR, so EOIing
/// would clear a real pending IRQ on a lower line. Do nothing.
extern "x86-interrupt" fn spurious_master(_frame: InterruptStackFrame) {}

/// Spurious slave IRQ 15 (vector 47).
///
/// The slave's ISR bit 7 is *not* set, but the master *did* latch the
/// cascade IRQ 2 and is waiting on an EOI. EOI the master only.
extern "x86-interrupt" fn spurious_slave(_frame: InterruptStackFrame) {
    pic::send_eoi_master();
}
