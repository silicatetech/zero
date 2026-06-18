// SPDX-License-Identifier: AGPL-3.0-or-later
//! Storage HAL — telemetry stub.
//!
//! Full storage provider surface (arena-bounded region minting,
//! quota enforcement) is Stage 13+ scope. Stage 12 ships the telemetry
//! trait so the SandboxManager can route memory-bandwidth queries
//! through it without committing to a specific backing technology.

use crate::error::HalError;
use crate::telemetry::{Bandwidth, TelemetryResolution};

/// Storage region identifier (arena-bounded region per ADR-019 §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StorageRegionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StorageTelemetrySnapshot {
    pub sample: u64,
    pub interval_us: u32,
    pub read_bandwidth: Bandwidth,
    pub write_bandwidth: Bandwidth,
    /// Aggregate memory-bandwidth proxy — what `(query memory
    /// bandwidth)` returns. On x86_64 with Intel PCM this is Uncore
    /// counter sum; on Server-ARM with MPAM it is the MPAM monitor
    /// reading.
    pub memory_bandwidth: Bandwidth,
    pub resolution: TelemetryResolution,
}

impl StorageTelemetrySnapshot {
    pub const fn zero() -> Self {
        Self {
            sample: 0,
            interval_us: 0,
            read_bandwidth: Bandwidth::zero(),
            write_bandwidth: Bandwidth::zero(),
            memory_bandwidth: Bandwidth::zero(),
            resolution: TelemetryResolution::Synthetic,
        }
    }
}

pub trait StorageTelemetry {
    fn poll_region(
        &mut self,
        region: StorageRegionId,
        interval_us: u32,
    ) -> Result<StorageTelemetrySnapshot, HalError>;

    /// Whole-system memory bandwidth (used by `(query memory
    /// bandwidth)`). Distinct from `poll_region` because the
    /// system-wide reading does not require any region to be live.
    fn poll_system(&mut self, interval_us: u32) -> Result<StorageTelemetrySnapshot, HalError>;
}

/// Deterministic synthetic storage telemetry.
#[derive(Debug, Default)]
pub struct MockStorageTelemetry {
    sample_count: u64,
}

impl MockStorageTelemetry {
    pub fn new() -> Self {
        Self::default()
    }
}

impl StorageTelemetry for MockStorageTelemetry {
    fn poll_region(
        &mut self,
        _region: StorageRegionId,
        interval_us: u32,
    ) -> Result<StorageTelemetrySnapshot, HalError> {
        self.sample_count = self.sample_count.saturating_add(1);
        let s = self.sample_count;
        Ok(StorageTelemetrySnapshot {
            sample: s,
            interval_us,
            read_bandwidth: Bandwidth::from_mib_per_s((s.wrapping_mul(128) % 16_384) as u32),
            write_bandwidth: Bandwidth::from_mib_per_s((s.wrapping_mul(96) % 16_384) as u32),
            memory_bandwidth: Bandwidth::from_mib_per_s((s.wrapping_mul(256) % 32_768) as u32),
            resolution: TelemetryResolution::Synthetic,
        })
    }

    fn poll_system(&mut self, interval_us: u32) -> Result<StorageTelemetrySnapshot, HalError> {
        self.poll_region(StorageRegionId(0), interval_us)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_storage_telemetry_monotonic_sample() {
        let mut t = MockStorageTelemetry::new();
        let s0 = t.poll_region(StorageRegionId(0), 1000).unwrap();
        let s1 = t.poll_region(StorageRegionId(0), 1000).unwrap();
        assert_eq!(s0.sample, 1);
        assert_eq!(s1.sample, 2);
    }

    #[test]
    fn poll_system_advances_counter() {
        let mut t = MockStorageTelemetry::new();
        let s0 = t.poll_system(1000).unwrap();
        let s1 = t.poll_system(1000).unwrap();
        assert_eq!(s1.sample, s0.sample + 1);
    }

    #[test]
    fn snapshot_zero_starts_at_zero() {
        let z = StorageTelemetrySnapshot::zero();
        assert_eq!(z.sample, 0);
        assert_eq!(z.read_bandwidth.mib_per_s, 0);
        assert_eq!(z.write_bandwidth.mib_per_s, 0);
        assert_eq!(z.memory_bandwidth.mib_per_s, 0);
    }
}
