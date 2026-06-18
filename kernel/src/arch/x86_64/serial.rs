// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal COM1 serial-port writer.
//!
//! Writes go to `-serial stdio` in QEMU and to Serial-over-LAN on
//! Supermicro/Cherry Servers' BMC. Configured for 115200 8N1 — the
//! BMC default; matches what `ipmitool sol activate` expects without
//! firmware re-config.
//!
//! Every byte that flows through `Serial::write_str` is also rendered
//! to the bootloader-provided GOP/VBE framebuffer via `fb_console`,
//! so the BMC's HTML5 KVM console gets the same log stream as the
//! SoL endpoint.

use core::fmt::{self, Write};

use super::fb_console;

const COM1: u16 = 0x3F8;

pub struct Serial;

impl Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for &byte in s.as_bytes() {
            unsafe {
                while (inb(COM1 + 5) & 0x20) == 0 {}
                outb(COM1, byte);
            }
        }
        fb_console::write_str(s);
        Ok(())
    }
}

pub fn init() {
    unsafe {
        outb(COM1 + 1, 0x00); // disable interrupts
        outb(COM1 + 3, 0x80); // enable DLAB
        outb(COM1 + 0, 0x01); // divisor low — 115200 baud (1.8432 MHz / 1)
        outb(COM1 + 1, 0x00); // divisor high
        outb(COM1 + 3, 0x03); // 8N1, DLAB off
        outb(COM1 + 2, 0xC7); // enable FIFO, clear, 14-byte threshold
        outb(COM1 + 4, 0x0B); // RTS/DSR set, OUT2 (for IRQ routing)
    }
}

pub fn println(s: &str) {
    let mut ser = Serial;
    let _ = ser.write_str(s);
    let _ = ser.write_str("\r\n");
}

#[inline]
unsafe fn outb(port: u16, val: u8) {
    core::arch::asm!(
        "out dx, al",
        in("dx") port,
        in("al") val,
        options(nomem, nostack, preserves_flags),
    );
}

#[inline]
unsafe fn inb(port: u16) -> u8 {
    let val: u8;
    core::arch::asm!(
        "in al, dx",
        out("al") val,
        in("dx") port,
        options(nomem, nostack, preserves_flags),
    );
    val
}
