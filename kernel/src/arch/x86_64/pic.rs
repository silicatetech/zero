// SPDX-License-Identifier: AGPL-3.0-or-later
//! 8259 Programmable Interrupt Controller driver — from scratch.
//!
//! Two cascaded 8259A controllers: the master handles IRQ 0-7 and
//! the slave IRQ 8-15 (the slave signals through the master's IRQ 2
//! line).
//!
//!   Master: port 0x20 (command), 0x21 (data)
//!   Slave:  port 0xA0 (command), 0xA1 (data)
//!
//! The real-mode BIOS leaves the PICs mapped to interrupt vectors
//! 0x08-0x0F (master) and 0x70-0x77 (slave). On x86_64 those vectors
//! collide with CPU-exception slots (0..31 are reserved). Before
//! hardware interrupts can fire into the CPU we remap both PICs:
//!   master → vectors 0x20..0x27 (32..39)
//!   slave  → vectors 0x28..0x2F (40..47)
//!
//! Initialization is a fixed four-word sequence per PIC: ICW1
//! (command port), then ICW2, ICW3, ICW4 on the data port. Each word
//! means something specific; see comments inline.

use core::arch::asm;

// -------- I/O port addresses -----------------------------------------
const MASTER_CMD: u16 = 0x20;
const MASTER_DATA: u16 = 0x21;
const SLAVE_CMD: u16 = 0xA0;
const SLAVE_DATA: u16 = 0xA1;

// -------- remap targets ----------------------------------------------
/// First CPU vector delivered by the master PIC (IRQ 0 → vector 32).
pub const MASTER_OFFSET: u8 = 0x20;
/// First CPU vector delivered by the slave PIC (IRQ 8 → vector 40).
pub const SLAVE_OFFSET: u8 = 0x28;

// -------- OCW2: end-of-interrupt -------------------------------------
const EOI: u8 = 0x20;

/// Initialize both PICs, remap them to vectors 32-47, and mask all
/// IRQs except IRQ 0 (timer).
///
/// Writes are issued in the strict order the chip expects.
pub fn init() {
    unsafe {
        // ------ ICW1: start initialization ---------------------------
        //   bit 4 = 1  : "this is ICW1"
        //   bit 3 = 0  : edge-triggered (not level)
        //   bit 1 = 0  : cascade mode (two chained 8259s)
        //   bit 0 = 1  : ICW4 will follow (required on x86)
        // → 0b0001_0001 = 0x11
        outb(MASTER_CMD, 0x11);
        outb(SLAVE_CMD, 0x11);

        // ------ ICW2: vector base offset -----------------------------
        // Written to the data port. Tells each PIC which CPU-vector
        // its IRQ 0 (or IRQ 8) maps to. Upper 5 bits only — the chip
        // ORs in the 3-bit IRQ line.
        outb(MASTER_DATA, MASTER_OFFSET); // 0x20 → IRQ 0 = vector 32
        outb(SLAVE_DATA, SLAVE_OFFSET); // 0x28 → IRQ 8 = vector 40

        // ------ ICW3: cascade wiring ---------------------------------
        // Master: bitmask of IRQ lines that have a slave attached.
        //         The slave is wired to IRQ 2, so bit 2 = 0x04.
        // Slave:  its own slave-ID on the master. The slave responds
        //         as the master's IRQ 2, so ID = 2.
        outb(MASTER_DATA, 0x04);
        outb(SLAVE_DATA, 0x02);

        // ------ ICW4: operational mode -------------------------------
        //   bit 0 = 1 : 8086/88 mode (not MCS-80/85)
        //   bit 1 = 0 : manual EOI (we send the EOI ourselves)
        //   bits 2-3  : no buffered master/slave
        //   bit 4 = 0 : not "special fully nested"
        // → 0x01
        outb(MASTER_DATA, 0x01);
        outb(SLAVE_DATA, 0x01);

        // ------ OCW1: interrupt mask ---------------------------------
        // A set bit *masks* (disables) the IRQ. We want only IRQ 0
        // (timer) delivered for now.
        //   Master: 0b1111_1110 = 0xFE → only IRQ 0 unmasked
        //   Slave : 0b1111_1111 = 0xFF → everything masked
        outb(MASTER_DATA, 0xFE);
        outb(SLAVE_DATA, 0xFF);
    }
}

/// Signal end-of-interrupt for an IRQ that originated on the master
/// PIC (IRQ 0-7, CPU vectors 32-39). Must be called from inside the
/// handler before `iretq`.
#[inline]
pub fn send_eoi_master() {
    unsafe {
        outb(MASTER_CMD, EOI);
    }
}

/// Signal end-of-interrupt for an IRQ that originated on the slave
/// PIC (IRQ 8-15, CPU vectors 40-47). Slave IRQs require EOI to
/// *both* chips — slave first, then master (which saw the cascade).
#[inline]
pub fn send_eoi_slave() {
    unsafe {
        outb(SLAVE_CMD, EOI);
        outb(MASTER_CMD, EOI);
    }
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
