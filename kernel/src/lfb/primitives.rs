// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-F1: LFB rendering primitives.
//!
//! Per Pillar 7: NO #[cfg(target_arch)] in this file.
//! Per Pillar 1: bounded loops, no allocation, deterministic.
//! Per Pillar 8: Stage 14 Quarks migration target — keep simple.
//!
//! All pixel writes use write_volatile for correct framebuffer semantics.

use crate::lfb::abi::framebuffer;

/// Set a single pixel at (x, y) to the given RGB color.
///
/// Sub-MP-F3.6 M4: unused after M3 draw_char cache-hoist.
/// Retained as low-level API for future use.
///
/// # Safety
/// Framebuffer base_addr must be a valid mapped address.
#[allow(dead_code)]
pub unsafe fn draw_pixel(x: usize, y: usize, r: u8, g: u8, b: u8) {
    let Some(fb) = framebuffer() else { return };
    let Some(offset) = fb.pixel_byte_offset(x, y) else {
        return;
    };

    // Pack as XRGB8888 (most common for ramfb/virtio-gpu)
    let pixel: u32 = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
    let addr = (fb.base_addr + offset) as *mut u32;
    core::ptr::write_volatile(addr, pixel);
}

/// Fill a rectangle at (x, y) with dimensions (w, h).
///
/// Sub-MP-F3.6 M2: Non-volatile row-wise store. Allows LLVM to vectorize
/// (vst1q_u32 on aarch64). ramfb is RAM-backed in QEMU — safe.
///
/// # Safety
/// Framebuffer base_addr must be a valid mapped address.
pub unsafe fn fill_rect(x: usize, y: usize, w: usize, h: usize, r: u8, g: u8, b: u8) {
    let Some(fb) = framebuffer() else { return };

    let x_end = (x + w).min(fb.width);
    let y_end = (y + h).min(fb.height);
    if x >= x_end || y >= y_end {
        return;
    }
    let pixel: u32 = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
    let row_pixels = x_end - x;

    let mut py = y;
    while py < y_end {
        let row_start = (fb.base_addr + py * fb.stride + x * (fb.bpp / 8)) as *mut u32;
        // Non-volatile: allows LLVM to vectorize via vst1q_u32
        let row_slice = core::slice::from_raw_parts_mut(row_start, row_pixels);
        for p in row_slice.iter_mut() {
            *p = pixel;
        }
        py += 1;
    }
}

/// Clear entire framebuffer to the given color.
///
/// # Safety
/// Framebuffer base_addr must be a valid mapped address.
pub unsafe fn clear_screen(r: u8, g: u8, b: u8) {
    let Some(fb) = framebuffer() else { return };
    fill_rect(0, 0, fb.width, fb.height, r, g, b);
}
