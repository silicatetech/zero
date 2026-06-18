// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bootloader-framebuffer text console.
//!
//! Renders kernel log text into the GOP/VBE linear framebuffer that
//! bootloader 0.11 sets up and hands over via `BootInfo::framebuffer`.
//! On a Supermicro/Cherry Server, the BMC's HTML5 KVM console shows
//! whatever the GPU outputs — which is this framebuffer.
//!
//! Glyphs: the 95-char public-domain VGA 8x16 ROM font already present
//! at `kernel/src/lfb/font_data.rs`. Foreground white, background black.
//!
//! Scrolling: local/dev builds use pixel scrolling. Cherry bare-metal
//! builds use page mode instead: when the cursor passes the last row,
//! clear the screen and continue from the top. Supermicro BMC GOP
//! framebuffers make full-screen pixel scrolls expensive enough to skew
//! boot-time measurements; serial still carries the full linear log.
//!
//! Synchronisation: no locking, single static cursor — matches the
//! lock-free pattern in `serial.rs`. Interleaved writes from two
//! contexts garble bytes rather than fault.

use bootloader_api::info::PixelFormat;
use bootloader_api::BootInfo;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::lfb::font_data::{CHAR_HEIGHT, CHAR_WIDTH, FONT_8X16};

#[derive(Clone, Copy)]
struct FbInfo {
    base: *mut u8,
    width: usize,
    height: usize,
    stride_bytes: usize,
    bpp_bytes: usize,
    format: PixelFormat,
}

unsafe impl Send for FbInfo {}
unsafe impl Sync for FbInfo {}

static mut FB: Option<FbInfo> = None;
static READY: AtomicBool = AtomicBool::new(false);
static COL: AtomicUsize = AtomicUsize::new(0);
static ROW: AtomicUsize = AtomicUsize::new(0);

/// Initialise the console from BootInfo. Must run before any write_str
/// call that should appear on screen. Safe to skip — write_str becomes
/// a no-op if the bootloader didn't provide a framebuffer.
pub fn init(boot_info: &mut BootInfo) {
    let Some(fb) = boot_info.framebuffer.as_mut() else {
        return;
    };
    let info = fb.info();
    let buf = fb.buffer_mut();
    let fbi = FbInfo {
        base: buf.as_mut_ptr(),
        width: info.width,
        height: info.height,
        stride_bytes: info.stride * info.bytes_per_pixel,
        bpp_bytes: info.bytes_per_pixel,
        format: info.pixel_format,
    };
    unsafe { FB = Some(fbi) };
    clear();
    READY.store(true, Ordering::Release);
}

/// Start a fresh framebuffer page: clear the screen and home the
/// cursor. Serial output is unaffected — this only resets the mirror.
///
/// Used by the Cherry production path right before the final
/// benchmark/status screen: page-mode scrolling would otherwise split
/// the summary box across a page boundary, leaving the held KVM screen
/// with half a table and no NET status line. The boot-stage log still
/// streams to the framebuffer live during boot; this runs only after
/// the last boot stage has completed.
pub fn new_page() {
    if !READY.load(Ordering::Acquire) {
        return;
    }
    clear();
    COL.store(0, Ordering::Relaxed);
    ROW.store(0, Ordering::Relaxed);
}

pub fn write_str(s: &str) {
    if !READY.load(Ordering::Acquire) {
        return;
    }
    let Some(fbi) = (unsafe { FB }) else { return };
    let cols = fbi.width / CHAR_WIDTH;
    let rows = fbi.height / CHAR_HEIGHT;
    let mut col = COL.load(Ordering::Relaxed);
    let mut row = ROW.load(Ordering::Relaxed);
    for &b in s.as_bytes() {
        match b {
            b'\n' => {
                col = 0;
                row += 1;
            }
            b'\r' => {
                col = 0;
            }
            b'\t' => {
                let next = (col + 8) & !7;
                col = next.min(cols);
            }
            b => {
                if col >= cols {
                    col = 0;
                    row += 1;
                }
                if row >= rows {
                    row = advance_past_bottom(&fbi, rows);
                }
                draw_glyph(&fbi, col * CHAR_WIDTH, row * CHAR_HEIGHT, b);
                col += 1;
            }
        }
        if row >= rows {
            row = advance_past_bottom(&fbi, rows);
        }
    }
    COL.store(col, Ordering::Relaxed);
    ROW.store(row, Ordering::Relaxed);
}

fn clear() {
    let Some(fbi) = (unsafe { FB }) else { return };
    unsafe {
        core::ptr::write_bytes(fbi.base, 0x00, fbi.stride_bytes * fbi.height);
    }
}

fn clear_frame(fbi: &FbInfo) {
    unsafe {
        core::ptr::write_bytes(fbi.base, 0x00, fbi.stride_bytes * fbi.height);
    }
}

#[cfg(feature = "cherry-net")]
fn advance_past_bottom(fbi: &FbInfo, _rows: usize) -> usize {
    // Cherry's BMC-backed GOP aperture is very slow for repeated
    // full-frame memmoves. Page mode keeps KVM useful while avoiding
    // serial-log volume turning into synthetic boot-time latency.
    clear_frame(fbi);
    0
}

#[cfg(not(feature = "cherry-net"))]
fn advance_past_bottom(fbi: &FbInfo, rows: usize) -> usize {
    scroll(fbi);
    rows.saturating_sub(1)
}

#[cfg(not(feature = "cherry-net"))]
fn scroll(fbi: &FbInfo) {
    let row_bytes = fbi.stride_bytes;
    let shift = CHAR_HEIGHT * row_bytes;
    let total = fbi.height * row_bytes;
    if shift >= total {
        return;
    }
    unsafe {
        // Move rows [CHAR_HEIGHT .. height] up by one text row.
        core::ptr::copy(fbi.base.add(shift), fbi.base, total - shift);
        // Clear the new bottom text row.
        core::ptr::write_bytes(fbi.base.add(total - shift), 0x00, shift);
    }
}

#[inline]
fn draw_glyph(fbi: &FbInfo, x: usize, y: usize, ch: u8) {
    let idx = if (0x20..=0x7E).contains(&ch) {
        (ch - 0x20) as usize
    } else {
        0
    };
    let glyph = &FONT_8X16[idx * CHAR_HEIGHT..idx * CHAR_HEIGHT + CHAR_HEIGHT];
    for (row_idx, &bits) in glyph.iter().enumerate() {
        let row_addr = unsafe {
            fbi.base
                .add((y + row_idx) * fbi.stride_bytes + x * fbi.bpp_bytes)
        };
        for col_idx in 0..CHAR_WIDTH {
            let lit = (bits >> (7 - col_idx)) & 1 == 1;
            let px = unsafe { row_addr.add(col_idx * fbi.bpp_bytes) };
            write_pixel(px, lit, fbi.format);
        }
    }
}

#[inline]
fn write_pixel(px: *mut u8, lit: bool, fmt: PixelFormat) {
    let v: u8 = if lit { 0xFF } else { 0x00 };
    match fmt {
        PixelFormat::Rgb | PixelFormat::Bgr => unsafe {
            *px = v;
            *px.add(1) = v;
            *px.add(2) = v;
        },
        PixelFormat::U8 => unsafe { *px = v },
        _ => unsafe {
            *px = v;
            *px.add(1) = v;
            *px.add(2) = v;
        },
    }
}
