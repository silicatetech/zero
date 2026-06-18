// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-F3 Task B: Telemetry panel rendering on framebuffer.
//!
//! Per Pillar 7: NO #[cfg(target_arch)] in this module.
//! Per Pillar 1 (reframed): renders ONCE at boot completion, NOT per-token.
//!                          Static telemetry, zero inference-loop overhead.
//! Per Pillar 8: Stage 14 Quarks migration target, keep simple.
//! Per Lesson 36: throughput displayed at 4-decimal precision.

use crate::lfb::abi::framebuffer;
use crate::lfb::font_data::CHAR_HEIGHT;
use crate::lfb::font_data::CHAR_WIDTH;
use crate::lfb::telemetry_data::TelemetryData;

/// Telemetry panel position and dimensions.
pub struct TelemetryPanel {
    pub origin_x: usize,
    pub origin_y: usize,
    pub width: usize,
    pub height: usize,
}

/// Global telemetry panel state.
static mut TELEMETRY_PANEL: Option<TelemetryPanel> = None;

/// Initialize telemetry panel position.
///
/// # Safety
/// Must be called during single-threaded boot after LFB init.
#[allow(static_mut_refs)]
pub unsafe fn init(x: usize, y: usize, w: usize, h: usize) {
    TELEMETRY_PANEL = Some(TelemetryPanel {
        origin_x: x,
        origin_y: y,
        width: w,
        height: h,
    });
}

/// Render telemetry panel ONCE at boot (after model loaded, before inference).
///
/// Per F3 hard-gate discipline (Lesson 36): rendering happens BEFORE
/// Phase 2 inference-loop, so zero impact on token throughput rate.
///
/// # Safety
/// Framebuffer must be initialized.
#[allow(static_mut_refs)]
pub unsafe fn render_static(td: &TelemetryData) {
    let panel = match TELEMETRY_PANEL.as_ref() {
        Some(p) => p,
        None => return,
    };
    if framebuffer().is_none() {
        return;
    }

    let x = panel.origin_x;
    let y = panel.origin_y;
    let w = panel.width;
    let h = panel.height;

    // Clear panel area (dark background)
    crate::lfb::primitives::fill_rect(x, y, w, h, 5, 5, 15);

    // Draw border (1-pixel cyan outline)
    draw_border(x, y, w, h, 0, 180, 220);

    let ix = x + 8; // inner x (padding)
    let mut iy = y + 8; // inner y (current row)

    // ── Title ──
    crate::lfb::font::draw_string(ix, iy, "[ TELEMETRY ]", 0, 255, 255, 5, 5, 15);
    iy += CHAR_HEIGHT + 4;

    // Separator line
    crate::lfb::primitives::fill_rect(ix, iy, w - 16, 1, 0, 100, 100);
    iy += 6;

    // ── Speedup (N7: computed from actual baseline ratio) ──
    draw_label(ix, iy, "Speed:", 180, 180, 255);
    let speedup_x100 = td.speedup_x100(); // 157 for NEON, 100 for scalar
    let mut spd_buf = [0u8; 16];
    let spd_str = format_speedup(speedup_x100, td.mode_label, &mut spd_buf);
    draw_value(ix + LABEL_COL, iy, spd_str, 0, 255, 80);
    iy += ROW_H;

    // ── Mode ──
    draw_label(ix, iy, "Mode:", 180, 180, 255);
    draw_value(ix + LABEL_COL, iy, td.mode_label, 100, 255, 100);
    iy += ROW_H;

    // ── Model ──
    // Prefer the GGUF-derived override installed by Stage 11; fall back
    // to the build-mode default so the panel still renders when the
    // model parse hasn't completed (e.g. first render pre-Stage-11).
    draw_label(ix, iy, "Model:", 180, 180, 255);
    let model_label = crate::lfb::telemetry_data::effective_model_label(td);
    draw_value(ix + LABEL_COL, iy, model_label, 220, 200, 100);
    iy += ROW_H;

    // ── Layers ──
    draw_label(ix, iy, "Layers:", 180, 180, 255);
    let mut buf = [0u8; 16];
    let s = format_u32(td.total_layers, &mut buf);
    draw_value(ix + LABEL_COL, iy, s, 100, 255, 100);
    iy += ROW_H;

    // ── Inference wallclock ──
    draw_label(ix, iy, "Infer:", 180, 180, 255);
    let mut buf2 = [0u8; 16];
    let s2 = format_u32_suffix(td.current_baseline_wallclock_s, "s", &mut buf2);
    draw_value(ix + LABEL_COL, iy, s2, 100, 255, 100);
    iy += ROW_H;

    // ── Throughput (4-decimal precision per Lesson 36) ──
    draw_label(ix, iy, "Tok/s:", 180, 180, 255);
    let mut buf3 = [0u8; 16];
    let s3 = format_throughput(td.throughput_x10000, &mut buf3);
    draw_value(ix + LABEL_COL, iy, s3, 0, 255, 80);
    iy += ROW_H + 4;

    // ── Memory section (M6: dynamic via setter, NOT hardcoded) ──
    if let Some(mb) = crate::lfb::telemetry_data::memory_boundaries() {
        // Separator
        crate::lfb::primitives::fill_rect(ix, iy, w - 16, 1, 0, 100, 100);
        iy += 6;

        crate::lfb::font::draw_string(ix, iy, "[ MEMORY ]", 0, 255, 255, 5, 5, 15);
        iy += CHAR_HEIGHT + 4;

        // KV Arena
        draw_label(ix, iy, "KV:", 180, 180, 255);
        let mut buf4 = [0u8; 16];
        let s4 = format_u32_suffix(mb.kv_cache_arena_size_mb, " MiB", &mut buf4);
        draw_value(ix + LABEL_COL, iy, s4, 220, 200, 100);
        iy += ROW_H;

        // Framebuffer
        draw_label(ix, iy, "FB:", 180, 180, 255);
        let mut buf5 = [0u8; 16];
        let s5 = format_u32_suffix(mb.framebuffer_size_kb, " KiB", &mut buf5);
        draw_value(ix + LABEL_COL, iy, s5, 220, 200, 100);
    }
}

// ── Layout constants ──

/// Label column width in pixels (7 chars × 8px).
const LABEL_COL: usize = 7 * CHAR_WIDTH;

/// Row height (char height + 2px spacing).
const ROW_H: usize = CHAR_HEIGHT + 2;

// ── Helper functions ──

/// Draw label text.
///
/// # Safety
/// Framebuffer must be initialized via `lfb::abi::set_framebuffer()`.
unsafe fn draw_label(x: usize, y: usize, text: &str, r: u8, g: u8, b: u8) {
    crate::lfb::font::draw_string(x, y, text, r, g, b, 5, 5, 15);
}

/// Draw value text.
///
/// # Safety
/// Framebuffer must be initialized via `lfb::abi::set_framebuffer()`.
unsafe fn draw_value(x: usize, y: usize, text: &str, r: u8, g: u8, b: u8) {
    crate::lfb::font::draw_string(x, y, text, r, g, b, 5, 5, 15);
}

/// Draw rectangle border (1-pixel-wide outline using fill_rect).
///
/// # Safety
/// Framebuffer must be initialized; `(x, y, w, h)` must be within FB bounds.
unsafe fn draw_border(x: usize, y: usize, w: usize, h: usize, r: u8, g: u8, b: u8) {
    crate::lfb::primitives::fill_rect(x, y, w, 1, r, g, b); // top
    crate::lfb::primitives::fill_rect(x, y + h - 1, w, 1, r, g, b); // bottom
    crate::lfb::primitives::fill_rect(x, y, 1, h, r, g, b); // left
    crate::lfb::primitives::fill_rect(x + w - 1, y, 1, h, r, g, b); // right
}

// ── Formatting helpers (no_std, no alloc) ──

/// core::fmt::Write adapter for fixed-size byte buffer.
struct BufWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl core::fmt::Write for BufWriter<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for byte in s.bytes() {
            if self.pos >= self.buf.len() {
                return Err(core::fmt::Error);
            }
            self.buf[self.pos] = byte;
            self.pos += 1;
        }
        Ok(())
    }
}

/// Format u32 as decimal string into buffer. Returns &str.
fn format_u32<'a>(value: u32, buf: &'a mut [u8; 16]) -> &'a str {
    use core::fmt::Write;
    let mut w = BufWriter {
        buf: &mut buf[..],
        pos: 0,
    };
    let _ = write!(w, "{}", value);
    let len = w.pos;
    core::str::from_utf8(&buf[..len]).unwrap_or("?")
}

/// Format u32 with suffix (e.g., "75s", "512 MiB"). Returns &str.
fn format_u32_suffix<'a>(value: u32, suffix: &str, buf: &'a mut [u8; 16]) -> &'a str {
    use core::fmt::Write;
    let mut w = BufWriter {
        buf: &mut buf[..],
        pos: 0,
    };
    let _ = write!(w, "{}{}", value, suffix);
    let len = w.pos;
    core::str::from_utf8(&buf[..len]).unwrap_or("?")
}

/// Format throughput x10000 as "0.4267" decimal string. Returns &str.
/// Per Lesson 36: 4-decimal precision mandatory.
fn format_throughput<'a>(value_x10000: u32, buf: &'a mut [u8; 16]) -> &'a str {
    use core::fmt::Write;
    let int_part = value_x10000 / 10000;
    let frac_part = value_x10000 % 10000;
    let mut w = BufWriter {
        buf: &mut buf[..],
        pos: 0,
    };
    let _ = write!(w, "{}.{:04}", int_part, frac_part);
    let len = w.pos;
    core::str::from_utf8(&buf[..len]).unwrap_or("?")
}

/// Format speedup from x100 value + mode label.
/// N7: 157 → "1.57x", 100 → "1.00x"
/// Per honest-metrics anchor: mathematical truth from actual baselines.
fn format_speedup<'a>(speedup_x100: u32, _mode: &str, buf: &'a mut [u8; 16]) -> &'a str {
    use core::fmt::Write;
    let int_part = speedup_x100 / 100;
    let frac_part = speedup_x100 % 100;
    let mut w = BufWriter {
        buf: &mut buf[..],
        pos: 0,
    };
    let _ = write!(w, "{}.{:02}x", int_part, frac_part);
    let len = w.pos;
    core::str::from_utf8(&buf[..len]).unwrap_or("?")
}
