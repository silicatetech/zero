// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-F1: aarch64 ramfb initialization via QEMU fw-cfg.
//!
//! Per Pillar 7: aarch64-specific. Generic LFB consumes via FrameBufferInfo.
//!
//! ramfb protocol: write RAMFBCfg struct to fw-cfg "etc/ramfb" file
//! using DMA interface at fw-cfg MMIO base (0x9020000 on QEMU virt).
//!
//! CITE: QEMU docs/specs/fw_cfg.txt
//! CITE: QEMU hw/display/ramfb.c

use crate::lfb::abi::{set_framebuffer, FrameBufferInfo};
use core::fmt::Write;

// fw-cfg MMIO register offsets (per QEMU fw_cfg spec)
const FWCFG_BASE: usize = 0x0902_0000;
const FWCFG_DATA: usize = FWCFG_BASE; // 8 bytes, data register
const FWCFG_SEL: usize = FWCFG_BASE + 0x08; // 2 bytes, selector
const FWCFG_DMA: usize = FWCFG_BASE + 0x10; // 8 bytes, DMA address

// fw-cfg DMA control bits
const FWCFG_DMA_CTL_ERROR: u32 = 1 << 0;
#[allow(dead_code)]
const FWCFG_DMA_CTL_READ: u32 = 1 << 1;
#[allow(dead_code)]
const FWCFG_DMA_CTL_SKIP: u32 = 1 << 2;
const FWCFG_DMA_CTL_SELECT: u32 = 1 << 3;
const FWCFG_DMA_CTL_WRITE: u32 = 1 << 4;

// fw-cfg well-known selectors
const FWCFG_FILE_DIR: u16 = 0x0019;

// DRM fourcc for XRGB8888
const DRM_FORMAT_XRGB8888: u32 = 0x34325258; // 'XR24'

// Framebuffer dimensions (1024x768 is reasonable for QEMU display)
const FB_WIDTH: usize = 1024;
const FB_HEIGHT: usize = 768;
const FB_BPP: usize = 32;
const FB_STRIDE: usize = FB_WIDTH * (FB_BPP / 8);
const FB_SIZE: usize = FB_STRIDE * FB_HEIGHT;

// Physical address for framebuffer — placed after GGUF model ends.
// Model ends at ~0x9470_79A0, so 0x9500_0000 is safe (16MB aligned).
// This region is in the L1[2] 1GiB Normal-cacheable block (0x8000_0000..0xC000_0000).
const FB_PHYS_ADDR: usize = 0x9500_0000;

/// RAMFBCfg structure (per QEMU hw/display/ramfb.c)
/// All fields big-endian.
#[repr(C, packed)]
struct RamfbCfg {
    addr: u64,   // physical address of framebuffer
    fourcc: u32, // DRM pixel format
    flags: u32,  // reserved, must be 0
    width: u32,  // pixels
    height: u32, // pixels
    stride: u32, // bytes per row
}

/// FWCfgDmaAccess structure (per QEMU docs/specs/fw_cfg.txt)
#[repr(C, packed)]
struct FwCfgDmaAccess {
    control: u32, // big-endian
    length: u32,  // big-endian
    address: u64, // big-endian
}

/// Initialize ramfb framebuffer via QEMU fw-cfg.
///
/// 1. Scan fw-cfg file directory for "etc/ramfb" selector
/// 2. Write RAMFBCfg struct via DMA to configure display
/// 3. Set global FrameBufferInfo
///
/// # Safety
/// Must be called after MMU enabled (fw-cfg MMIO at 0x9020000 must be mapped).
/// Must be called during single-threaded boot.
pub unsafe fn init_ramfb() -> bool {
    let serial = &mut crate::arch::aarch64::serial::Serial;
    let _ = writeln!(serial, "[LFB] Scanning fw-cfg for ramfb...");

    // Step 1: Find "etc/ramfb" file selector in fw-cfg directory
    let ramfb_selector = match find_fwcfg_file(b"etc/ramfb") {
        Some(sel) => {
            let _ = writeln!(serial, "[LFB] Found etc/ramfb at selector 0x{:04x}", sel);
            sel
        }
        None => {
            let _ = writeln!(serial, "[LFB] etc/ramfb NOT found — no ramfb device?");
            return false;
        }
    };

    // Step 2: Zero framebuffer memory region (bulk write_bytes)
    // Sub-MP-F3.6 M1: was per-u32 volatile loop (~1s), now write_bytes (~10-20ms).
    // ramfb is RAM-backed in QEMU — non-volatile writes are safe + allow
    // compiler to emit optimal memset/dc zva instructions.
    let _ = writeln!(
        serial,
        "[LFB] Clearing FB at phys 0x{:x} ({} bytes)",
        FB_PHYS_ADDR, FB_SIZE
    );
    core::ptr::write_bytes(FB_PHYS_ADDR as *mut u8, 0, FB_SIZE);

    // Step 3: Build RAMFBCfg (all fields big-endian)
    let cfg = RamfbCfg {
        addr: (FB_PHYS_ADDR as u64).to_be(),
        fourcc: DRM_FORMAT_XRGB8888.to_be(),
        flags: 0u32.to_be(),
        width: (FB_WIDTH as u32).to_be(),
        height: (FB_HEIGHT as u32).to_be(),
        stride: (FB_STRIDE as u32).to_be(),
    };

    // Step 4: Write config via DMA
    let cfg_ptr = &cfg as *const RamfbCfg as *const u8;
    let cfg_size = core::mem::size_of::<RamfbCfg>();

    let ok = fwcfg_dma_write(ramfb_selector, cfg_ptr, cfg_size);
    if !ok {
        let _ = writeln!(serial, "[LFB] fw-cfg DMA write FAILED");
        return false;
    }

    let _ = writeln!(
        serial,
        "[LFB] ramfb configured: {}x{} XRGB8888 at 0x{:x}",
        FB_WIDTH, FB_HEIGHT, FB_PHYS_ADDR
    );

    // Step 5: Set global framebuffer info
    set_framebuffer(FrameBufferInfo {
        base_addr: FB_PHYS_ADDR,
        size: FB_SIZE,
        width: FB_WIDTH,
        height: FB_HEIGHT,
        stride: FB_STRIDE,
        bpp: FB_BPP,
    });

    true
}

/// Scan fw-cfg file directory for a file by name.
/// Returns the selector index if found.
///
/// # Safety
/// FWCFG MMIO base (0x0902_0000) must be mapped and accessible.
/// Must be called after DeviceTree fw-cfg discovery confirms presence.
unsafe fn find_fwcfg_file(name: &[u8]) -> Option<u16> {
    // Select file directory
    core::ptr::write_volatile(FWCFG_SEL as *mut u16, FWCFG_FILE_DIR.to_be());

    // Read file count (4 bytes big-endian)
    let count_bytes = read_fwcfg_bytes(4);
    let count = u32::from_be_bytes([
        count_bytes[0],
        count_bytes[1],
        count_bytes[2],
        count_bytes[3],
    ]);

    // Each directory entry: 64 bytes
    //   u32 size, u16 select, u16 reserved, char name[56]
    let mut i = 0u32;
    while i < count {
        let entry = read_fwcfg_bytes(64);
        let sel = u16::from_be_bytes([entry[4], entry[5]]);
        let entry_name = &entry[8..64];

        // Compare name (null-terminated)
        let mut matches = true;
        let mut j = 0;
        while j < name.len() {
            if j >= 56 || entry_name[j] != name[j] {
                matches = false;
                break;
            }
            j += 1;
        }
        // Ensure name is properly terminated
        if matches && (name.len() >= 56 || entry_name[name.len()] == 0) {
            return Some(sel);
        }

        i += 1;
    }
    None
}

/// Read N bytes from fw-cfg data port (byte at a time, sequential).
///
/// # Safety
/// FWCFG data register must be selected via prior write to FWCFG_SEL.
/// Caller must ensure `n <= 64` (buffer size).
unsafe fn read_fwcfg_bytes(n: usize) -> [u8; 64] {
    let mut buf = [0u8; 64];
    let data_ptr = FWCFG_DATA as *const u8;
    let mut i = 0;
    while i < n && i < 64 {
        buf[i] = core::ptr::read_volatile(data_ptr);
        i += 1;
    }
    buf
}

/// Write data to a fw-cfg file via DMA interface.
///
/// # Safety
/// `data` must point to valid, readable memory of at least `len` bytes.
/// FWCFG DMA register must be mapped. DMA struct must remain valid until
/// fw-cfg reports completion (control field becomes 0).
unsafe fn fwcfg_dma_write(selector: u16, data: *const u8, len: usize) -> bool {
    // Build DMA access structure
    let control = FWCFG_DMA_CTL_SELECT | FWCFG_DMA_CTL_WRITE | ((selector as u32) << 16);

    let dma = FwCfgDmaAccess {
        control: control.to_be(),
        length: (len as u32).to_be(),
        address: (data as u64).to_be(),
    };

    let dma_addr = &dma as *const FwCfgDmaAccess as u64;

    // Write DMA address to fw-cfg DMA register (big-endian u64)
    core::ptr::write_volatile(FWCFG_DMA as *mut u64, dma_addr.to_be());

    // Poll for completion: control field becomes 0 when done.
    // Use addr_of! to avoid unaligned reference to packed struct field.
    let ctl_ptr = core::ptr::addr_of!(dma.control);
    let mut attempts = 0u32;
    loop {
        let ctl = u32::from_be(core::ptr::read_volatile(ctl_ptr));
        if ctl == 0 {
            return true;
        }
        if ctl & FWCFG_DMA_CTL_ERROR != 0 {
            return false;
        }
        attempts += 1;
        if attempts > 1_000_000 {
            return false;
        }
    }
}
