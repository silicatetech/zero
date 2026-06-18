// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-F2 Task C / F3.6 M5: Layer progress visualization for forward-pass.
//!
//! Per Pillar 7: NO #[cfg(target_arch)] in this module.
//! Per V3.1 Z.544: 28-layer architecture canonical (Qwen3-1.7B-Q4_K_M).
//! Per Pillar 1 (reframed): hook callback in inference_neon.rs layer loop.
//!
//! Sub-MP-F3.6 M5: Delta-fill pattern. Only fills NEW pixels per layer
//! (not entire bar from origin). ~10× speedup per-token bar rendering.

use crate::lfb::abi::framebuffer;

/// Total layers in Qwen3-1.7B (canonical per V3.1).
pub const TOTAL_LAYERS: usize = 28;

/// Layer progress bar state.
pub struct LayerProgressState {
    pub origin_x: usize,
    pub origin_y: usize,
    pub bar_width: usize,
    pub bar_height: usize,
}

/// Global layer progress state.
static mut LAYER_PROGRESS: Option<LayerProgressState> = None;

/// M5: Track last rendered layer for delta-fill.
static mut LAST_LAYER: usize = 0;

/// Initialize layer progress bar. Draws border frame.
///
/// # Safety
/// Must be called during single-threaded boot after LFB init.
#[allow(static_mut_refs)]
pub unsafe fn init(x: usize, y: usize, w: usize, h: usize) {
    LAYER_PROGRESS = Some(LayerProgressState {
        origin_x: x,
        origin_y: y,
        bar_width: w,
        bar_height: h,
    });
    LAST_LAYER = 0;

    if framebuffer().is_some() {
        // Draw label above bar
        crate::lfb::font::draw_string(
            x,
            y.saturating_sub(20),
            "Layer Progress [0/28]",
            180,
            180,
            255, // light blue
            0,
            0,
            0,
        );
        // Draw initial empty bar (dark green)
        crate::lfb::primitives::fill_rect(x, y, w, h, 0, 40, 0);
    }
}

/// Update layer progress (called per layer completion).
///
/// Sub-MP-F3.6 M5: Delta-fill — only fills NEW pixels since last call.
/// Pre-M5: full-bar redraw from origin per layer (~28 × full-bar = ~14k px/token).
/// Post-M5: delta-fill per layer (~500 px/token total for 28 layers).
///
/// # Safety
/// Framebuffer must be initialized.
#[allow(static_mut_refs)]
pub unsafe fn set_layer(layer: usize) {
    if framebuffer().is_none() {
        return;
    }
    let lp = match LAYER_PROGRESS.as_ref() {
        Some(p) => p,
        None => return,
    };

    if layer > TOTAL_LAYERS {
        return;
    }

    let prev_filled_width = (LAST_LAYER * lp.bar_width) / TOTAL_LAYERS;
    let curr_filled_width = (layer * lp.bar_width) / TOTAL_LAYERS;
    LAST_LAYER = layer;

    // M5: Only fill the NEW pixels (delta region)
    if curr_filled_width > prev_filled_width {
        crate::lfb::primitives::fill_rect(
            lp.origin_x + prev_filled_width,
            lp.origin_y,
            curr_filled_width - prev_filled_width,
            lp.bar_height,
            0,
            200,
            0,
        );
    } else if curr_filled_width < prev_filled_width {
        // Reset case (layer went backwards — defensive)
        crate::lfb::primitives::fill_rect(
            lp.origin_x,
            lp.origin_y,
            lp.bar_width,
            lp.bar_height,
            0,
            40,
            0,
        );
        if curr_filled_width > 0 {
            crate::lfb::primitives::fill_rect(
                lp.origin_x,
                lp.origin_y,
                curr_filled_width,
                lp.bar_height,
                0,
                200,
                0,
            );
        }
    }

    // Update label with current layer number
    let mut label_buf = [0u8; 24];
    let label_len = write_layer_label(&mut label_buf, layer);
    crate::lfb::font::draw_string(
        lp.origin_x,
        lp.origin_y.saturating_sub(20),
        core::str::from_utf8(&label_buf[..label_len]).unwrap_or(""),
        180,
        180,
        255,
        0,
        0,
        0,
    );
}

/// Reset progress bar for new token.
///
/// # Safety
/// Framebuffer must be initialized.
#[allow(static_mut_refs)]
pub unsafe fn reset() {
    if framebuffer().is_none() {
        return;
    }
    let lp = match LAYER_PROGRESS.as_ref() {
        Some(p) => p,
        None => return,
    };
    // Clear bar to dark green
    crate::lfb::primitives::fill_rect(
        lp.origin_x,
        lp.origin_y,
        lp.bar_width,
        lp.bar_height,
        0,
        40,
        0,
    );
    // M5: Reset delta-fill state
    LAST_LAYER = 0;
}

/// Write "Layer Progress [XX/28]" into buffer. Returns length.
fn write_layer_label(buf: &mut [u8], layer: usize) -> usize {
    let prefix = b"Layer Progress [";
    let suffix = b"/28]";
    let mut pos = 0;

    // Copy prefix
    for &b in prefix {
        if pos < buf.len() {
            buf[pos] = b;
            pos += 1;
        }
    }

    // Write layer number (0-28, up to 2 digits)
    if layer >= 10 {
        if pos < buf.len() {
            buf[pos] = b'0' + (layer / 10) as u8;
            pos += 1;
        }
    }
    if pos < buf.len() {
        buf[pos] = b'0' + (layer % 10) as u8;
        pos += 1;
    }

    // Copy suffix
    for &b in suffix {
        if pos < buf.len() {
            buf[pos] = b;
            pos += 1;
        }
    }

    // Pad with spaces to overwrite previous longer text
    while pos < 22 && pos < buf.len() {
        buf[pos] = b' ';
        pos += 1;
    }

    pos
}
