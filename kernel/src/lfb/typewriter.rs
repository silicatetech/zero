// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-F2 Task B / F4 Task A: Live typewriter rendering with scrolling.
//!
//! Per Pillar 7: NO #[cfg(target_arch)] in this module.
//! Per Pillar 1 (reframed): MUST NOT regress inference-loop wallclock.
//! Per Pillar 8: Stage 14 Quarks migration target.
//!
//! Hook callsite: kernel/src/inference.rs token-generation-loop, after
//! each token argmax + detokenize. NOT in sacred crate.
//!
//! Sub-MP-F4 Task A: When text reaches panel bottom, scroll content up
//! one CHAR_HEIGHT row via per-row memcpy. Non-volatile writes consistent
//! with F3.6 M2 ramfb-RAM-semantics.

use crate::lfb::abi::framebuffer;
use crate::lfb::font_data::{CHAR_HEIGHT, CHAR_WIDTH};

/// Typewriter render state machine.
///
/// Maintains cursor position + text panel boundaries.
/// F4: scrolling enabled when cursor_y reaches panel bottom.
pub struct TypewriterState {
    pub origin_x: usize,
    pub origin_y: usize,
    pub max_width: usize,
    pub max_height: usize,
    pub cursor_x: usize,
    pub cursor_y: usize,
    pub fg_r: u8,
    pub fg_g: u8,
    pub fg_b: u8,
    pub bg_r: u8,
    pub bg_g: u8,
    pub bg_b: u8,
}

/// Global typewriter state. Initialized via init() during boot.
static mut TYPEWRITER: Option<TypewriterState> = None;

/// Initialize typewriter panel. Clears panel area.
///
/// # Safety
/// Must be called during single-threaded boot after LFB init.
#[allow(static_mut_refs)]
pub unsafe fn init(panel_x: usize, panel_y: usize, panel_w: usize, panel_h: usize) {
    TYPEWRITER = Some(TypewriterState {
        origin_x: panel_x,
        origin_y: panel_y,
        max_width: panel_w,
        max_height: panel_h,
        cursor_x: panel_x,
        cursor_y: panel_y,
        fg_r: 0,
        fg_g: 255,
        fg_b: 0, // Matrix green
        bg_r: 0,
        bg_g: 0,
        bg_b: 0,
    });

    // Clear panel area
    crate::lfb::primitives::fill_rect(panel_x, panel_y, panel_w, panel_h, 0, 0, 0);
}

/// Push text to typewriter (live render).
///
/// Per F2 hard-gate: must be FAST (called per-token, ~2.25s/token).
/// LFB write per char is ~16 × 8 = 128 pixel-writes.
/// Per token (~3-4 chars): ~400-500 pixel-writes = sub-millisecond.
///
/// F4 Task A: when cursor_y + CHAR_HEIGHT exceeds panel bottom,
/// scroll_up_one_line() shifts all content up by CHAR_HEIGHT pixels
/// and clears the bottom row for new content.
///
/// # Safety
/// Framebuffer must be initialized.
#[allow(static_mut_refs)]
pub unsafe fn push_text(text: &[u8]) {
    if framebuffer().is_none() {
        return;
    }
    let tw = match TYPEWRITER.as_mut() {
        Some(t) => t,
        None => return,
    };

    for &byte in text {
        if byte == b'\n' {
            tw.cursor_x = tw.origin_x;
            tw.cursor_y += CHAR_HEIGHT;
            // F4 Task A: scroll when bottom reached
            scroll_if_needed(tw);
            continue;
        }

        // Line wrap
        if tw.cursor_x + CHAR_WIDTH > tw.origin_x + tw.max_width {
            tw.cursor_x = tw.origin_x;
            tw.cursor_y += CHAR_HEIGHT;
            // F4 Task A: scroll when bottom reached
            scroll_if_needed(tw);
        }

        crate::lfb::font::draw_char(
            tw.cursor_x,
            tw.cursor_y,
            byte,
            tw.fg_r,
            tw.fg_g,
            tw.fg_b,
            tw.bg_r,
            tw.bg_g,
            tw.bg_b,
        );
        tw.cursor_x += CHAR_WIDTH;
    }
}

/// F4 Task A: Scroll panel up one CHAR_HEIGHT row if cursor is past bottom.
///
/// Scrolling is per-row memcpy (non-volatile, consistent with M2 pattern).
/// Cost: ~1.4 MiB copy for 680×524 panel, triggered at most once per ~33
/// text lines. Negligible inference-loop impact for 32-token Phase 2.
///
/// # Safety
/// Framebuffer must be initialized (caller verified).
unsafe fn scroll_if_needed(tw: &mut TypewriterState) {
    let panel_bottom = tw.origin_y + tw.max_height;
    if tw.cursor_y + CHAR_HEIGHT <= panel_bottom {
        return; // Still fits, no scroll needed
    }

    let Some(fb) = framebuffer() else { return };

    // Pixels to copy: panel height minus one text row
    let copy_height = tw.max_height.saturating_sub(CHAR_HEIGHT);
    // Panel width in u32 pixels (4 bytes per pixel, XRGB8888)
    let panel_width_px = tw.max_width;

    // Per-row memcpy: source is one CHAR_HEIGHT below dest
    let mut row = 0;
    while row < copy_height {
        let src_y = tw.origin_y + CHAR_HEIGHT + row;
        let dst_y = tw.origin_y + row;
        let src = (fb.base_addr + src_y * fb.stride + tw.origin_x * 4) as *const u32;
        let dst = (fb.base_addr + dst_y * fb.stride + tw.origin_x * 4) as *mut u32;
        // Non-volatile: consistent with M2 ramfb-RAM-semantics
        core::ptr::copy_nonoverlapping(src, dst, panel_width_px);
        row += 1;
    }

    // Clear bottom row (black background)
    crate::lfb::primitives::fill_rect(
        tw.origin_x,
        tw.origin_y + copy_height,
        tw.max_width,
        CHAR_HEIGHT,
        tw.bg_r,
        tw.bg_g,
        tw.bg_b,
    );

    // Position cursor at start of new bottom row
    tw.cursor_y = tw.origin_y + copy_height;
}
