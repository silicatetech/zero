// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-F1: Framebuffer ABI — architecture-agnostic types.
//!
//! Per Pillar 7: NO #[cfg(target_arch)] in this file.
//! Per Pillar 8: Stage 14 Quarks migration target — keep simple.

/// Framebuffer configuration. Populated by platform-specific init,
/// consumed by generic rendering primitives.
#[derive(Debug, Clone, Copy)]
pub struct FrameBufferInfo {
    /// Base address for pixel writes (virtual, post-MMU).
    pub base_addr: usize,
    /// Total framebuffer size in bytes.
    pub size: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Bytes per row (may differ from width * bpp/8 due to padding).
    pub stride: usize,
    /// Bits per pixel (16 or 32).
    pub bpp: usize,
}

impl FrameBufferInfo {
    /// Byte offset for pixel at (x, y). Returns None if out of bounds.
    #[inline]
    pub fn pixel_byte_offset(&self, x: usize, y: usize) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(y * self.stride + x * (self.bpp / 8))
    }
}

/// Global LFB state. Set once by arch-specific init during boot.
/// Consumed by rendering primitives via `framebuffer()`.
///
/// Safety: written once during single-threaded boot, read-only after.
static mut FRAMEBUFFER: Option<FrameBufferInfo> = None;

/// Get the current framebuffer info, if initialized.
pub fn framebuffer() -> Option<FrameBufferInfo> {
    unsafe { FRAMEBUFFER }
}

/// Set the global framebuffer. Called once during boot.
///
/// # Safety
/// Must be called during single-threaded boot only.
pub unsafe fn set_framebuffer(fb: FrameBufferInfo) {
    FRAMEBUFFER = Some(fb);
}
