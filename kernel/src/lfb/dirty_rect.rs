// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-F4 Task B: Dirty-rectangle tracking for render-skip optimization.
//!
//! Per Pillar 7: NO #[cfg(target_arch)] in this module.
//! Per Pillar 1 (reframed): MUST NOT regress inference-loop wallclock.
//! Per Pillar 8: Stage 14 Quarks migration target — keep simple.
//!
//! Establishes tracking infrastructure. Consumers (typewriter, layer_progress)
//! mark dirty regions via mark_dirty(). Future code can use take() to get
//! bounding rect of changed region for render-skip optimization.
//!
//! Current F4 scope: tracking only. Render-skip optimization deferred to
//! Stage 14+ Quarks migration where multi-rect tracking is warranted.

/// Axis-aligned rectangle (origin + size).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    /// Construct a new rectangle.
    pub const fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Check if this rectangle has zero area.
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Check if this rectangle overlaps with another.
    #[allow(dead_code)]
    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
}

/// Simple dirty-rectangle tracker.
///
/// Accumulates dirty regions via mark_dirty() into a single bounding rect.
/// take() returns + clears the accumulated dirty region.
///
/// Limited to ONE tracked bounding-rect per instance (sufficient for
/// typewriter, layer_progress, telemetry). Multi-rect tracking with
/// per-tile granularity deferred to Stage 14+ Quarks migration.
pub struct DirtyTracker {
    current: Option<Rect>,
}

impl DirtyTracker {
    /// Create a new empty tracker.
    pub const fn new() -> Self {
        Self { current: None }
    }

    /// Mark a rectangular region as dirty (will be redrawn).
    ///
    /// Cost: 4 comparisons + assignment (~10 ns). Per token: ~4 marks.
    /// Per Phase 2: 128 marks ≈ ~1.3 µs total. Negligible.
    pub fn mark_dirty(&mut self, rect: Rect) {
        if rect.is_empty() {
            return;
        }
        self.current = match self.current {
            Some(existing) => Some(union(existing, rect)),
            None => Some(rect),
        };
    }

    /// Get current dirty region and clear tracker.
    /// Returns None if nothing has been marked dirty since last take().
    #[allow(dead_code)]
    pub fn take(&mut self) -> Option<Rect> {
        self.current.take()
    }

    /// Peek at current dirty region without clearing.
    #[allow(dead_code)]
    pub fn peek(&self) -> Option<Rect> {
        self.current
    }
}

/// Compute axis-aligned bounding rect of two rectangles.
fn union(a: Rect, b: Rect) -> Rect {
    let x = a.x.min(b.x);
    let y = a.y.min(b.y);
    let x_end = (a.x + a.width).max(b.x + b.width);
    let y_end = (a.y + a.height).max(b.y + b.height);
    Rect::new(x, y, x_end - x, y_end - y)
}
