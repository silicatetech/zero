// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU HAL traits.
//!
//! Per `hardware-abstraction-constraints.md` §4.1 the trait surface is
//! split into:
//! - **Slice-minting** ([`GpuProvider::create_slice`] /
//!   [`GpuProvider::destroy_slice`]). MIG, MxGPU and software
//!   time-slicing all express here.
//! - **Telemetry-read** ([`GpuProvider::poll_slice_telemetry`] /
//!   [`GpuProvider::poll_device_telemetry`]). PLATYPUS lesson (§4.3):
//!   telemetry is a distinct authority from slice-use.
//!
//! Capability gating is *not* the HAL's responsibility — it lives on
//! the `SandboxManager` boundary in `crates/zero-sandbox`. The HAL
//! assumes its caller has already cleared `HardwareGpu` (for
//! mint/destroy) or `HardwareTelemetryRead` (for poll).

use alloc::vec::Vec;

use crate::error::HalError;
use crate::telemetry::GpuTelemetrySnapshot;

/// GPU device identifier inside a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuDeviceId(pub u32);

impl GpuDeviceId {
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Service-side identifier for a live GPU slice.
///
/// 64-bit so the service can mint without exhaustion concerns. The
/// concrete provider chooses the encoding (NVML / MIG-instance index;
/// software time-slice mints a synthetic counter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuSliceId(pub u64);

impl GpuSliceId {
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Vendor tag — useful to a Policy Agent that wants to qualify
/// telemetry resolution at the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Mali,
    Adreno,
    Imagination,
    Mock,
    /// Not yet platform-bound (boot-time stubs).
    Unknown,
}

/// Slicing mechanism a provider uses for a given device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlicingMode {
    /// NVIDIA MIG — hardware-isolated GPU instances with dedicated
    /// SMs / L2 / memory. Up to 7 instances on A100 (and equivalents
    /// on H100/B100); see [`crate::mig::MIG_MAX_INSTANCES`].
    HardwareMig,
    /// SR-IOV-based partitioning (AMD MxGPU).
    HardwareSrIov,
    /// Driver-side time-slicing fallback. Used by [`crate::TimeSliceProvider`]
    /// for consumer GPUs without HW partitioning (RTX/GTX, Apple AGX,
    /// Mali/Adreno).
    SoftwareTimeSlice,
    /// No partitioning — single tenant gets the whole device.
    Exclusive,
}

/// Requested properties of a fresh slice.
///
/// All values are integer; the provider rejects out-of-range / over-
/// committing requests via [`HalError::InvalidSpec`] or
/// [`HalError::OutOfSlices`]. Aggregate-budget enforcement (multi-
/// tenant: sum of slice compute_percent ≤ 100) is the provider's job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SliceSpec {
    /// Compute fraction in percent points, `1..=100`.
    pub compute_percent: u8,
    /// Slice memory budget in MiB.
    pub memory_mib: u32,
    /// For software time-slicing: caller-requested contiguous time
    /// window in µs (provider may clamp). For hardware modes this is
    /// informational; HW partitions slot the request into the
    /// nearest supported size.
    pub time_window_us: u32,
}

impl SliceSpec {
    pub const fn new(compute_percent: u8, memory_mib: u32, time_window_us: u32) -> Self {
        Self {
            compute_percent,
            memory_mib,
            time_window_us,
        }
    }

    pub(crate) fn validate(self) -> Result<(), HalError> {
        if self.compute_percent == 0 || self.compute_percent > 100 {
            return Err(HalError::InvalidSpec("compute_percent_out_of_range"));
        }
        if self.memory_mib == 0 {
            return Err(HalError::InvalidSpec("memory_mib_zero"));
        }
        Ok(())
    }
}

/// Live slice handle.
///
/// Owner-binding and capability rights live on the
/// [`crate::HardwareCapabilityService`] facade, *not* here. A bare
/// [`GpuSlice`] is just the data the provider tracks; the service
/// layer wraps it with owner / cap-id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuSlice {
    pub id: GpuSliceId,
    pub device: GpuDeviceId,
    pub spec: SliceSpec,
    pub mode: SlicingMode,
}

/// Core GPU HAL trait. Every provider (NVML, MIG, time-slice, mock)
/// implements it.
///
/// `&mut self`: providers mint ids and advance sampling counters.
/// Concurrent access goes through the manager-level mutex, not into
/// the provider itself. This matches the
/// `zero-sandbox::SandboxManager` discipline.
pub trait GpuProvider {
    /// Vendor tag for the underlying hardware.
    fn vendor(&self) -> GpuVendor;

    /// Enumerate known devices. Order is deterministic — providers
    /// must keep a stable enumeration for replay-bit-exact behaviour.
    fn devices(&self) -> Vec<GpuDeviceId>;

    /// Slicing mode used on `device`.
    fn slicing_mode(&self, device: GpuDeviceId) -> Result<SlicingMode, HalError>;

    /// Mint a new slice on `device` satisfying `spec`. Provider rejects
    /// the request if budget is exhausted ([`HalError::OutOfSlices`])
    /// or the spec violates an invariant
    /// ([`HalError::InvalidSpec`]).
    fn create_slice(&mut self, device: GpuDeviceId, spec: SliceSpec) -> Result<GpuSlice, HalError>;

    /// Tear a slice down. Subsequent telemetry/dereference on the same
    /// `GpuSliceId` returns [`HalError::NoSuchSlice`].
    fn destroy_slice(&mut self, slice: GpuSliceId) -> Result<(), HalError>;

    /// Snapshot of currently-live slices on `device`. Order matches the
    /// minting order (deterministic).
    fn live_slices(&self, device: GpuDeviceId) -> Result<Vec<GpuSlice>, HalError>;

    /// Sample telemetry on one live slice. `interval_us` is the
    /// caller-provided window since the last poll (host clock); the
    /// provider does *not* read its own clock per the §1.3
    /// determinism contract.
    fn poll_slice_telemetry(
        &mut self,
        slice: GpuSliceId,
        interval_us: u32,
    ) -> Result<GpuTelemetrySnapshot, HalError>;

    /// Whole-device telemetry, ignoring slice partitioning. Used by
    /// the GPU Policy Agent for "is the box hot?" decisions before
    /// any slice is minted.
    fn poll_device_telemetry(
        &mut self,
        device: GpuDeviceId,
        interval_us: u32,
    ) -> Result<GpuTelemetrySnapshot, HalError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_spec_validates_compute_percent() {
        let bad = SliceSpec::new(0, 1024, 0);
        assert!(matches!(bad.validate(), Err(HalError::InvalidSpec(_))));
        let bad2 = SliceSpec::new(101, 1024, 0);
        assert!(matches!(bad2.validate(), Err(HalError::InvalidSpec(_))));
        let good = SliceSpec::new(50, 1024, 0);
        assert!(good.validate().is_ok());
    }

    #[test]
    fn slice_spec_validates_memory() {
        let bad = SliceSpec::new(50, 0, 0);
        assert!(matches!(bad.validate(), Err(HalError::InvalidSpec(_))));
    }

    #[test]
    fn slicing_modes_are_distinct() {
        let modes = [
            SlicingMode::HardwareMig,
            SlicingMode::HardwareSrIov,
            SlicingMode::SoftwareTimeSlice,
            SlicingMode::Exclusive,
        ];
        for (i, a) in modes.iter().enumerate() {
            for (j, b) in modes.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
