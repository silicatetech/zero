// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-F1/F2/F3/F4: Linear Frame Buffer — Stage 11.5 Neural IMGUI.
//!
//! Per Pillar 7: NO #[cfg(target_arch)] in this module.
//! Architecture-specific init lives in arch::<arch>::lfb.
//!
//! Module structure:
//!   abi.rs             — FrameBufferInfo struct + global state
//!   primitives.rs      — draw_pixel, fill_rect, clear_screen
//!   font.rs            — 8x16 bitmap char/string rendering
//!   font_data.rs       — VGA 8x16 font bitmap data (public domain)
//!   typewriter.rs      — F2/F4: live typewriter render + scrolling
//!   layer_progress.rs  — F2: 28-layer progress bar (M5 delta-fill)
//!   telemetry_data.rs  — F3: static telemetry + memory boundary data
//!   telemetry.rs       — F3: telemetry panel renderer
//!   dirty_rect.rs      — F4: dirty-rectangle tracking primitives

// x86_64 does not use LFB yet (Stage 14+). Suppress dead_code warnings.
#[allow(dead_code)]
pub mod abi;
#[allow(dead_code)]
pub mod dirty_rect;
#[allow(dead_code)]
pub mod font;
#[allow(dead_code)]
pub mod font_data;
#[allow(dead_code)]
pub mod layer_progress;
#[allow(dead_code)]
pub mod primitives;
#[allow(dead_code)]
pub mod telemetry;
#[allow(dead_code)]
pub mod telemetry_data;
#[allow(dead_code)]
pub mod typewriter;
