// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-F1/F3.6: 8x16 bitmap font rendering.
//!
//! Per Pillar 7: NO #[cfg(target_arch)] in this file.
//! Per Pillar 8: Stage 14 Quarks migration target.
//!
//! Sub-MP-F3.6 M3: FrameBufferInfo hoisted out of pixel loop.
//! Pre-M3: 128× framebuffer() call per char (via draw_pixel).
//! Post-M3: 1× framebuffer() call per char + inline pixel writes.

use crate::lfb::abi::framebuffer;
use crate::lfb::font_data::{CHAR_HEIGHT, CHAR_WIDTH, FONT_8X16};

/// Draw a single character at pixel position (x, y).
///
/// Sub-MP-F3.6 M3: FB info hoisted out of pixel loop. Non-volatile writes
/// (consistent with M2 ramfb-RAM-semantics). ~3.5× per-pixel improvement.
///
/// # Safety
/// Framebuffer must be initialized.
pub unsafe fn draw_char(
    x: usize,
    y: usize,
    ch: u8,
    fg_r: u8,
    fg_g: u8,
    fg_b: u8,
    bg_r: u8,
    bg_g: u8,
    bg_b: u8,
) {
    // M3: hoist framebuffer fetch ONCE (was 128× per char via draw_pixel)
    let Some(fb) = framebuffer() else { return };
    if x + CHAR_WIDTH > fb.width || y + CHAR_HEIGHT > fb.height {
        return;
    }

    let idx = if ch >= 0x20 && ch <= 0x7E {
        (ch - 0x20) as usize
    } else {
        0 // space for unprintable
    };
    let glyph_offset = idx * CHAR_HEIGHT;
    let fg_pixel: u32 = ((fg_r as u32) << 16) | ((fg_g as u32) << 8) | (fg_b as u32);
    let bg_pixel: u32 = ((bg_r as u32) << 16) | ((bg_g as u32) << 8) | (bg_b as u32);

    let mut row = 0;
    while row < CHAR_HEIGHT {
        let byte = FONT_8X16[glyph_offset + row];
        let row_addr = fb.base_addr + (y + row) * fb.stride + x * (fb.bpp / 8);
        let row_ptr = row_addr as *mut u32;
        let mut col = 0;
        while col < CHAR_WIDTH {
            let lit = (byte >> (7 - col)) & 1 == 1;
            let pixel = if lit { fg_pixel } else { bg_pixel };
            // Non-volatile write (consistent with M2 ramfb-RAM-semantics)
            *row_ptr.add(col) = pixel;
            col += 1;
        }
        row += 1;
    }
}

/// Draw a string at pixel position (x, y). Single-line only for F1.
///
/// # Safety
/// Framebuffer must be initialized.
pub unsafe fn draw_string(
    x: usize,
    y: usize,
    s: &str,
    fg_r: u8,
    fg_g: u8,
    fg_b: u8,
    bg_r: u8,
    bg_g: u8,
    bg_b: u8,
) {
    let mut cur_x = x;
    for &byte in s.as_bytes() {
        if byte == b'\n' {
            return;
        }
        draw_char(cur_x, y, byte, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b);
        cur_x += CHAR_WIDTH;
    }
}
