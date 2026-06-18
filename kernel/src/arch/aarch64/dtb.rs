// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device Tree Blob parser for aarch64.
//!
//! Uses `flat_device_tree` crate (v3.1, MPL-2.0, no_std, zero-alloc).
//! Compiled at opt-level=1 via per-package Cargo.toml override to prevent
//! LLVM from vectorizing byte-level reads into aligned LDR instructions
//! that fault on Device-nGnRnE memory (pre-MMU).
//!
//! TOTALSIZE FIX: QEMU pads DTB totalsize to 1 MiB. We patch the header
//! in-place to actual content size before calling Fdt::from_ptr.
//!
//! CITE: Device Tree Specification v0.4 (devicetree.org)
//! CITE: ARM Architecture Reference Manual — Device-nGnRnE memory semantics

use core::fmt::Write;
use flat_device_tree::Fdt;

/// Parsed DTB info — all addresses the kernel needs for boot.
pub struct DtbInfo {
    /// RAM base physical address (from /memory reg).
    pub ram_base: usize,
    /// RAM size in bytes (from /memory reg).
    pub ram_size: usize,
    /// Initrd start physical address (from /chosen).
    pub initrd_start: Option<usize>,
    /// Initrd end physical address (from /chosen).
    pub initrd_end: Option<usize>,
    /// PL011 UART base address (from DTB, typically 0x0900_0000).
    pub uart_base: usize,
    /// GICv2 Distributor base address (typically 0x0800_0000).
    pub gic_dist_base: usize,
    /// GICv2 CPU Interface base address (typically 0x0801_0000).
    pub gic_cpu_base: usize,
}

/// Parse the DTB at the given physical pointer.
///
/// Patches DTB totalsize header field in-place before calling
/// `Fdt::from_ptr`. QEMU pads totalsize to 1 MiB; actual content
/// is typically ~8 KiB.
///
/// # Safety
///
/// `dtb_ptr` must point to a valid FDT blob in writable physical memory.
/// Called with MMU disabled, so physical = virtual.
pub unsafe fn parse(dtb_ptr: usize) -> DtbInfo {
    let header_ptr = dtb_ptr as *mut u8;

    // Byte-wise header read — safe on Device-nGnRnE memory (no alignment issue)
    let read_be_u32 = |offset: usize| -> u32 {
        u32::from_be_bytes([
            core::ptr::read_volatile(header_ptr.add(offset) as *const u8),
            core::ptr::read_volatile(header_ptr.add(offset + 1) as *const u8),
            core::ptr::read_volatile(header_ptr.add(offset + 2) as *const u8),
            core::ptr::read_volatile(header_ptr.add(offset + 3) as *const u8),
        ])
    };

    // Verify DTB magic (0xd00dfeed per devicetree.org spec)
    let magic = read_be_u32(0x00);
    if magic != 0xd00dfeed {
        let _ = write!(
            crate::arch::serial::Serial,
            "  DTB: invalid magic {:#x}",
            magic
        );
        crate::arch::serial::println("");
        panic!("DTB: invalid magic");
    }

    // Calculate actual content size from header fields
    let off_strings = read_be_u32(0x0C) as usize;
    let size_strings = read_be_u32(0x20) as usize;
    let actual_size = off_strings + size_strings;

    let _ = write!(
        crate::arch::serial::Serial,
        "  DTB: totalsize={:#x}, actual={:#x} ({} bytes)",
        read_be_u32(0x04),
        actual_size,
        actual_size
    );
    crate::arch::serial::println("");

    // Patch totalsize in-place to actual content size.
    // Fdt::from_ptr reads totalsize to create the data slice;
    // patching avoids creating a 1 MiB slice from QEMU padding.
    let actual_size_be = (actual_size as u32).to_be_bytes();
    core::ptr::write_volatile(header_ptr.add(0x04), actual_size_be[0]);
    core::ptr::write_volatile(header_ptr.add(0x05), actual_size_be[1]);
    core::ptr::write_volatile(header_ptr.add(0x06), actual_size_be[2]);
    core::ptr::write_volatile(header_ptr.add(0x07), actual_size_be[3]);

    // Parse DTB via flat_device_tree (compiled at opt-level=1)
    let fdt = Fdt::from_ptr(header_ptr as *const u8).expect("DTB: from_ptr failed");

    crate::arch::serial::println("  DTB: parsed successfully");

    // /memory — RAM layout
    let memory = fdt.memory().expect("DTB: no /memory node");
    let region = memory.regions().next().expect("DTB: no /memory region");
    let ram_base = region.starting_address as usize;
    let ram_size = region.size.expect("DTB: no /memory size");

    // /chosen — initrd location (devicetree.org spec property names)
    let (initrd_start, initrd_end) = if let Some(chosen) = fdt.find_node("/chosen") {
        let start = chosen
            .properties()
            .find(|p| p.name == "linux,initrd-start")
            .and_then(|p| parse_dtb_u64_prop(p.value));
        let end = chosen
            .properties()
            .find(|p| p.name == "linux,initrd-end")
            .and_then(|p| parse_dtb_u64_prop(p.value));
        (start, end)
    } else {
        (None, None)
    };

    // /pl011 — UART address (find by compatible string)
    let uart_base = fdt
        .find_compatible(&["arm,pl011"])
        .and_then(|n| n.reg().next())
        .map(|r| r.starting_address as usize)
        .unwrap_or(0x0900_0000);

    // /intc — GIC addresses (find by compatible string)
    let (gic_dist_base, gic_cpu_base) = fdt
        .find_compatible(&["arm,cortex-a15-gic"])
        .and_then(|n| {
            let mut regs = n.reg();
            let dist = regs.next()?.starting_address as usize;
            let cpu = regs.next()?.starting_address as usize;
            Some((dist, cpu))
        })
        .unwrap_or((0x0800_0000, 0x0801_0000));

    DtbInfo {
        ram_base,
        ram_size,
        initrd_start,
        initrd_end,
        uart_base,
        gic_dist_base,
        gic_cpu_base,
    }
}

/// Parse a DTB property value as a u64/u32 address.
/// DTB properties are big-endian; may be 4 or 8 bytes.
fn parse_dtb_u64_prop(bytes: &[u8]) -> Option<usize> {
    if bytes.len() == 8 {
        Some(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as usize)
    } else if bytes.len() == 4 {
        Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize)
    } else {
        None
    }
}
