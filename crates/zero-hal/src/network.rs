// SPDX-License-Identifier: AGPL-3.0-or-later
//! Network HAL — telemetry stub.
//!
//! Full network provider surface (endpoint minting, bandwidth-cap
//! enforcement) is Stage 13+ scope per
//! `stage-12-completion-plan.md` §D.5 plan. Stage 12 ships the
//! telemetry trait so the SandboxManager can route
//! `(query network …)` forms through a uniform abstraction.

use crate::error::HalError;
use crate::telemetry::{Bandwidth, TelemetryResolution};

/// Network endpoint identifier (virtualized endpoint per ADR-019 §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NetworkEndpointId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NetworkTelemetrySnapshot {
    pub sample: u64,
    pub interval_us: u32,
    pub rx_bandwidth: Bandwidth,
    pub tx_bandwidth: Bandwidth,
    pub resolution: TelemetryResolution,
}

impl NetworkTelemetrySnapshot {
    pub const fn zero() -> Self {
        Self {
            sample: 0,
            interval_us: 0,
            rx_bandwidth: Bandwidth::zero(),
            tx_bandwidth: Bandwidth::zero(),
            resolution: TelemetryResolution::Synthetic,
        }
    }
}

pub trait NetworkTelemetry {
    fn poll_endpoint(
        &mut self,
        endpoint: NetworkEndpointId,
        interval_us: u32,
    ) -> Result<NetworkTelemetrySnapshot, HalError>;
}

/// Deterministic synthetic network telemetry. Mirrors
/// [`crate::mock::MockGpuProvider`]: a monotonic sample counter feeds
/// a ramp-style synthesis. Used when no native network provider is
/// registered.
#[derive(Debug, Default)]
pub struct MockNetworkTelemetry {
    sample_count: u64,
}

impl MockNetworkTelemetry {
    pub fn new() -> Self {
        Self::default()
    }
}

impl NetworkTelemetry for MockNetworkTelemetry {
    fn poll_endpoint(
        &mut self,
        _endpoint: NetworkEndpointId,
        interval_us: u32,
    ) -> Result<NetworkTelemetrySnapshot, HalError> {
        self.sample_count = self.sample_count.saturating_add(1);
        let s = self.sample_count;
        Ok(NetworkTelemetrySnapshot {
            sample: s,
            interval_us,
            rx_bandwidth: Bandwidth::from_mib_per_s((s.wrapping_mul(64) % 10_000) as u32),
            tx_bandwidth: Bandwidth::from_mib_per_s((s.wrapping_mul(48) % 10_000) as u32),
            resolution: TelemetryResolution::Synthetic,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_network_telemetry_monotonic_sample() {
        let mut t = MockNetworkTelemetry::new();
        let s0 = t.poll_endpoint(NetworkEndpointId(0), 1000).unwrap();
        let s1 = t.poll_endpoint(NetworkEndpointId(0), 1000).unwrap();
        assert_eq!(s0.sample, 1);
        assert_eq!(s1.sample, 2);
    }

    #[test]
    fn mock_network_resolution_is_synthetic() {
        let mut t = MockNetworkTelemetry::new();
        let s = t.poll_endpoint(NetworkEndpointId(0), 1000).unwrap();
        assert_eq!(s.resolution, TelemetryResolution::Synthetic);
    }

    #[test]
    fn snapshot_zero_starts_at_zero() {
        let z = NetworkTelemetrySnapshot::zero();
        assert_eq!(z.sample, 0);
        assert_eq!(z.rx_bandwidth.mib_per_s, 0);
        assert_eq!(z.tx_bandwidth.mib_per_s, 0);
    }
}
