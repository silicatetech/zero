// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic telemetry value types.
//!
//! Per `docs/discovery/hardware-abstraction-constraints.md` §1.3:
//! integer fixed-point only, no floating point. The values here are
//! the canonical Quarks-facing wire format — the Kernel-LLM
//! consumes them through `(query gpu utilization)` / `(query thermal
//! state)` / `(query memory bandwidth)`, so the encoding must be
//! stable and replay-bit-exact.

use crate::error::HalError;

/// GPU utilization as a percentage 0..=100.
///
/// Integer fixed-point: 1 percent point = 1 unit. The Apple
/// powermetrics "GPU active residency" and NVIDIA NVML "GPU
/// utilization" both map to this same domain at the provider
/// boundary, with platform-specific resolution recorded on the
/// `GpuTelemetrySnapshot` (see [`GpuTelemetrySnapshot::resolution`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Utilization {
    percent: u8,
}

impl Utilization {
    /// Construct from a raw percent value; clamps to `[0, 100]`.
    pub const fn new(pct: u8) -> Self {
        Self {
            percent: if pct > 100 { 100 } else { pct },
        }
    }

    pub const fn percent(self) -> u8 {
        self.percent
    }

    pub const fn zero() -> Self {
        Self { percent: 0 }
    }
}

/// Thermal pressure ladder.
///
/// Cross-platform taxonomy: Apple powermetrics emits this directly
/// (`Nominal/Fair/Serious/Critical`). NVIDIA NVML / hwmon emit
/// per-sensor `mDeg C` values that providers map to this ladder via
/// platform-specific thresholds. The integer discriminants 0..=3 are
/// stable wire constants for the Quarks query surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ThermalLevel {
    Nominal = THERMAL_NOMINAL,
    Fair = THERMAL_FAIR,
    Serious = THERMAL_SERIOUS,
    Critical = THERMAL_CRITICAL,
}

/// Stable wire constants for `ThermalLevel`. Equal to the matching
/// stub used by `(query thermal state)` (`STUB_QUERY_THERMAL_STATE = 0`
/// in `zero-sandbox`).
pub const THERMAL_NOMINAL: u8 = 0;
pub const THERMAL_FAIR: u8 = 1;
pub const THERMAL_SERIOUS: u8 = 2;
pub const THERMAL_CRITICAL: u8 = 3;

impl ThermalLevel {
    pub const fn from_code(code: u8) -> Result<Self, HalError> {
        match code {
            THERMAL_NOMINAL => Ok(ThermalLevel::Nominal),
            THERMAL_FAIR => Ok(ThermalLevel::Fair),
            THERMAL_SERIOUS => Ok(ThermalLevel::Serious),
            THERMAL_CRITICAL => Ok(ThermalLevel::Critical),
            _ => Err(HalError::InvalidSpec("thermal_level_code")),
        }
    }

    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Power draw in milliwatts.
///
/// Integer-only; the production NVML / RAPL / powermetrics paths
/// already emit milliwatts. Cumulative integrals
/// (`power × time`) live in the provider in `u64` to avoid wrap on
/// realistic durations (≥ centuries at 4 GW continuous).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Power {
    pub milliwatts: u32,
}

impl Power {
    pub const fn from_mw(mw: u32) -> Self {
        Self { milliwatts: mw }
    }

    pub const fn zero() -> Self {
        Self { milliwatts: 0 }
    }
}

/// Memory / interconnect bandwidth in MiB/s.
///
/// Provider-side cumulative byte counters are `u64` and converted to
/// `mib_per_s` at sample time. Sample windows are explicit (the host
/// passes `interval_us`); no implicit wall-clock dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bandwidth {
    pub mib_per_s: u32,
}

impl Bandwidth {
    pub const fn from_mib_per_s(value: u32) -> Self {
        Self { mib_per_s: value }
    }

    pub const fn zero() -> Self {
        Self { mib_per_s: 0 }
    }
}

/// Asymmetry-preserving telemetry resolution tag.
///
/// `hardware-abstraction-constraints.md` §4.4 — "Apple's closedness
/// vs. AMD/Intel openness: no glossing over". The HAL must surface
/// the *source* of a telemetry value so a downstream policy agent can
/// qualify its decision rather than treat coarse Apple `IOReport`
/// power-proxy and fine NVIDIA `DCGM` tensor-pipe-% as
/// interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TelemetryResolution {
    /// Direct hardware counter, ≤ ms resolution (NVML, DCGM, MPAM,
    /// MIG-per-instance counters).
    Hardware,
    /// Driver-side software counter (Mali Streamline, Intel GuC).
    DriverSoftware,
    /// Coarse power-proxy or activity-residency (`powermetrics`,
    /// IOReport — Apple).
    PowerProxy,
    /// Simulated/mock provider; deterministic but synthetic.
    Synthetic,
}

/// Deterministic GPU telemetry snapshot.
///
/// `sample` is a monotonically-increasing per-provider sequence number.
/// `interval_us` is the host-provided sampling interval in
/// microseconds — passed in via `poll_*` so the provider does *not*
/// read a clock itself. `resolution` records the source-fidelity tag
/// per §4.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuTelemetrySnapshot {
    pub sample: u64,
    pub interval_us: u32,
    pub utilization: Utilization,
    pub memory_bandwidth: Bandwidth,
    pub thermal: ThermalLevel,
    pub power: Power,
    pub resolution: TelemetryResolution,
}

impl GpuTelemetrySnapshot {
    pub const fn zero() -> Self {
        Self {
            sample: 0,
            interval_us: 0,
            utilization: Utilization::zero(),
            memory_bandwidth: Bandwidth::zero(),
            thermal: ThermalLevel::Nominal,
            power: Power::zero(),
            resolution: TelemetryResolution::Synthetic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utilization_clamps_above_100() {
        assert_eq!(Utilization::new(150).percent(), 100);
        assert_eq!(Utilization::new(100).percent(), 100);
        assert_eq!(Utilization::new(0).percent(), 0);
        assert_eq!(Utilization::new(50).percent(), 50);
    }

    #[test]
    fn thermal_codes_round_trip() {
        for code in 0..=3 {
            let lvl = ThermalLevel::from_code(code).expect("valid code");
            assert_eq!(lvl.code(), code);
        }
    }

    #[test]
    fn thermal_invalid_codes_rejected() {
        for code in 4u8..=255 {
            assert!(matches!(
                ThermalLevel::from_code(code),
                Err(HalError::InvalidSpec(_))
            ));
        }
    }

    #[test]
    fn snapshot_zero_is_nominal_and_unsynthetic() {
        let z = GpuTelemetrySnapshot::zero();
        assert_eq!(z.sample, 0);
        assert_eq!(z.utilization.percent(), 0);
        assert_eq!(z.thermal, ThermalLevel::Nominal);
        assert_eq!(z.resolution, TelemetryResolution::Synthetic);
    }

    #[test]
    fn power_and_bandwidth_are_value_types() {
        let p = Power::from_mw(250);
        assert_eq!(p.milliwatts, 250);
        let b = Bandwidth::from_mib_per_s(8192);
        assert_eq!(b.mib_per_s, 8192);
    }
}
