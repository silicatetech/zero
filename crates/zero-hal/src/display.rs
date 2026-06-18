// SPDX-License-Identifier: AGPL-3.0-or-later
//! Display HAL — surface + telemetry stub.

use crate::error::HalError;
use crate::telemetry::TelemetryResolution;

/// Identifier for a display surface (sandbox-window per ARCHITECTURE
/// `Display Agent` notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DisplaySurfaceId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DisplaySurface {
    pub id: DisplaySurfaceId,
    pub width_px: u32,
    pub height_px: u32,
    pub refresh_hz: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DisplayTelemetrySnapshot {
    pub sample: u64,
    pub interval_us: u32,
    pub composited_frames: u64,
    pub dropped_frames: u64,
    pub backlight_brightness_percent: u8,
    pub resolution: TelemetryResolution,
}

impl DisplayTelemetrySnapshot {
    pub const fn zero() -> Self {
        Self {
            sample: 0,
            interval_us: 0,
            composited_frames: 0,
            dropped_frames: 0,
            backlight_brightness_percent: 0,
            resolution: TelemetryResolution::Synthetic,
        }
    }
}

pub trait DisplayTelemetry {
    fn poll_surface(
        &mut self,
        surface: DisplaySurfaceId,
        interval_us: u32,
    ) -> Result<DisplayTelemetrySnapshot, HalError>;
}

/// Deterministic synthetic display telemetry.
#[derive(Debug, Default)]
pub struct MockDisplayTelemetry {
    sample_count: u64,
}

impl MockDisplayTelemetry {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DisplayTelemetry for MockDisplayTelemetry {
    fn poll_surface(
        &mut self,
        _surface: DisplaySurfaceId,
        interval_us: u32,
    ) -> Result<DisplayTelemetrySnapshot, HalError> {
        self.sample_count = self.sample_count.saturating_add(1);
        let s = self.sample_count;
        Ok(DisplayTelemetrySnapshot {
            sample: s,
            interval_us,
            composited_frames: s.wrapping_mul(60),
            dropped_frames: s.wrapping_mul(1) % 17,
            backlight_brightness_percent: ((s.wrapping_mul(3) % 101) as u8).min(100),
            resolution: TelemetryResolution::Synthetic,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_telemetry_monotonic() {
        let mut t = MockDisplayTelemetry::new();
        let s0 = t.poll_surface(DisplaySurfaceId(0), 1000).unwrap();
        let s1 = t.poll_surface(DisplaySurfaceId(0), 1000).unwrap();
        assert_eq!(s1.sample, s0.sample + 1);
    }

    #[test]
    fn display_snapshot_zero_is_synthetic() {
        let z = DisplayTelemetrySnapshot::zero();
        assert_eq!(z.sample, 0);
        assert_eq!(z.resolution, TelemetryResolution::Synthetic);
    }

    #[test]
    fn backlight_clamped_below_100() {
        let mut t = MockDisplayTelemetry::new();
        for _ in 0..200 {
            let s = t.poll_surface(DisplaySurfaceId(0), 1000).unwrap();
            assert!(s.backlight_brightness_percent <= 100);
        }
    }
}
