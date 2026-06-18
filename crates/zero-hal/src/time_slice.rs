// SPDX-License-Identifier: AGPL-3.0-or-later
//! D.4 — Software time-slicing fallback for GPUs without hardware
//! partitioning (consumer NVIDIA, AMD non-Instinct, Apple AGX,
//! Mali/Adreno).
//!
//! Per `hardware-abstraction-constraints.md` §4.2 ("Slice minting:
//! not platform-uniform"): if a platform has no native MIG/MxGPU,
//! the HCS must still hand out *some* slice surface. The substitute is
//! a capability-bound submission queue with a deterministic time
//! window (Sandbox A gets X µs out of every Y µs).
//!
//! Implementation strategy:
//!
//! - Maintain a fixed-quantum scheduler (`window_us`) per device.
//! - Each minted slice reserves a fraction of the window
//!   (`time_window_us`).
//! - Sum of reserved windows ≤ device window — overcommit rejected
//!   with [`HalError::InvalidSpec`].
//! - Watchdog enforcement is the *kernel*'s job (cooperative scheduling
//!   in `SandboxManager`); the HAL only ratifies the allocation
//!   contract.
//!
//! Same [`GpuProvider`] trait as MIG / NVML, so the HCS can fall back
//! transparently. Auto-detection in [`crate::service::HardwareCapabilityService`]
//! picks `TimeSliceProvider` when the underlying [`MockGpuProfile::Rtx4090`]
//! profile (or its eventual native equivalent) reports
//! [`SlicingMode::SoftwareTimeSlice`].

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::error::HalError;
use crate::gpu::{
    GpuDeviceId, GpuProvider, GpuSlice, GpuSliceId, GpuVendor, SliceSpec, SlicingMode,
};
use crate::mock::{MockGpuProfile, MockGpuProvider};
use crate::telemetry::GpuTelemetrySnapshot;

/// Default scheduling window for software time-slicing (16 ms ≈ 60 Hz
/// frame budget on consumer GPUs).
pub const DEFAULT_TIME_SLICE_WINDOW_US: u32 = 16_000;

/// Software time-slicing provider.
pub struct TimeSliceProvider {
    inner: MockGpuProvider,
    /// Per-device scheduling window (µs).
    windows: BTreeMap<GpuDeviceId, u32>,
    /// Per-device sum of reserved time-window slices (µs).
    reserved: BTreeMap<GpuDeviceId, u32>,
}

impl TimeSliceProvider {
    /// Construct with a default consumer-card profile (RTX 4090) and
    /// the 16 ms default window.
    pub fn new() -> Self {
        Self::with_profile_and_window(MockGpuProfile::Rtx4090, DEFAULT_TIME_SLICE_WINDOW_US)
    }

    /// Construct with an explicit profile + window. Window of `0` is
    /// rejected at the first `create_slice` call rather than at
    /// construction time so test harnesses can build invalid configs
    /// for negative-path coverage.
    pub fn with_profile_and_window(profile: MockGpuProfile, window_us: u32) -> Self {
        let inner = MockGpuProvider::new(profile);
        let mut windows = BTreeMap::new();
        let mut reserved = BTreeMap::new();
        for d in inner.devices() {
            windows.insert(d, window_us);
            reserved.insert(d, 0);
        }
        Self {
            inner,
            windows,
            reserved,
        }
    }

    /// Configured scheduling window (µs) for a device.
    pub fn window_us(&self, device: GpuDeviceId) -> Result<u32, HalError> {
        self.windows
            .get(&device)
            .copied()
            .ok_or(HalError::NoSuchDevice)
    }

    /// Reserved-time sum across live slices (µs).
    pub fn reserved_us(&self, device: GpuDeviceId) -> Result<u32, HalError> {
        self.reserved
            .get(&device)
            .copied()
            .ok_or(HalError::NoSuchDevice)
    }

    fn release_reservation(&mut self, device: GpuDeviceId, amount: u32) {
        if let Some(r) = self.reserved.get_mut(&device) {
            *r = r.saturating_sub(amount);
        }
    }
}

impl Default for TimeSliceProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProvider for TimeSliceProvider {
    fn vendor(&self) -> GpuVendor {
        self.inner.vendor()
    }

    fn devices(&self) -> Vec<GpuDeviceId> {
        self.inner.devices()
    }

    fn slicing_mode(&self, _device: GpuDeviceId) -> Result<SlicingMode, HalError> {
        // Time-slicing provider always reports software time-slicing,
        // even if the wrapped mock profile would natively support MIG —
        // the caller chose this provider because they need the fallback.
        Ok(SlicingMode::SoftwareTimeSlice)
    }

    fn create_slice(&mut self, device: GpuDeviceId, spec: SliceSpec) -> Result<GpuSlice, HalError> {
        spec.validate()?;
        let window = *self.windows.get(&device).ok_or(HalError::NoSuchDevice)?;
        if window == 0 {
            return Err(HalError::InvalidSpec("time_slice_window_is_zero"));
        }
        if spec.time_window_us == 0 {
            return Err(HalError::InvalidSpec("spec_time_window_us_zero"));
        }
        if spec.time_window_us > window {
            return Err(HalError::InvalidSpec("spec_time_window_exceeds_device"));
        }
        let reserved = *self.reserved.get(&device).unwrap_or(&0);
        if reserved.saturating_add(spec.time_window_us) > window {
            return Err(HalError::OutOfSlices);
        }
        // Materialise on the inner mock so the slice id is unique and
        // telemetry passes through. Clamp inner compute_percent so it
        // never trips the device-wide 100% sum check (the budget we
        // actually care about for time-slicing is the time-window).
        let inner_spec = SliceSpec::new(
            spec.compute_percent.min(100),
            spec.memory_mib,
            spec.time_window_us,
        );
        let slice = self.inner.create_slice(device, inner_spec)?;
        let r = self.reserved.entry(device).or_insert(0);
        *r = r.saturating_add(spec.time_window_us);
        // Re-tag the slice mode as time-slicing — the inner mock would
        // report whatever its profile says.
        Ok(GpuSlice {
            id: slice.id,
            device: slice.device,
            spec,
            mode: SlicingMode::SoftwareTimeSlice,
        })
    }

    fn destroy_slice(&mut self, slice: GpuSliceId) -> Result<(), HalError> {
        // Look up the slice's reserved time before deleting, so we can
        // release the reservation atomically.
        let mut found: Option<(GpuDeviceId, u32)> = None;
        for d in self.inner.devices() {
            if let Ok(list) = self.inner.live_slices(d) {
                if let Some(s) = list.iter().find(|s| s.id == slice) {
                    found = Some((d, s.spec.time_window_us));
                    break;
                }
            }
        }
        let (device, amount) = found.ok_or(HalError::NoSuchSlice)?;
        self.inner.destroy_slice(slice)?;
        self.release_reservation(device, amount);
        Ok(())
    }

    fn live_slices(&self, device: GpuDeviceId) -> Result<Vec<GpuSlice>, HalError> {
        let inner = self.inner.live_slices(device)?;
        // Re-tag mode on enumeration, mirroring `create_slice`.
        Ok(inner
            .into_iter()
            .map(|s| GpuSlice {
                mode: SlicingMode::SoftwareTimeSlice,
                ..s
            })
            .collect())
    }

    fn poll_slice_telemetry(
        &mut self,
        slice: GpuSliceId,
        interval_us: u32,
    ) -> Result<GpuTelemetrySnapshot, HalError> {
        self.inner.poll_slice_telemetry(slice, interval_us)
    }

    fn poll_device_telemetry(
        &mut self,
        device: GpuDeviceId,
        interval_us: u32,
    ) -> Result<GpuTelemetrySnapshot, HalError> {
        self.inner.poll_device_telemetry(device, interval_us)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_16ms_window() {
        let p = TimeSliceProvider::new();
        assert_eq!(
            p.window_us(GpuDeviceId(0)).unwrap(),
            DEFAULT_TIME_SLICE_WINDOW_US
        );
    }

    #[test]
    fn reports_software_time_slicing_mode() {
        let p = TimeSliceProvider::new();
        assert_eq!(
            p.slicing_mode(GpuDeviceId(0)).unwrap(),
            SlicingMode::SoftwareTimeSlice
        );
    }

    #[test]
    fn reserves_time_window_on_mint() {
        let mut p = TimeSliceProvider::new();
        assert_eq!(p.reserved_us(GpuDeviceId(0)).unwrap(), 0);
        let _s = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(50, 1024, 4_000))
            .unwrap();
        assert_eq!(p.reserved_us(GpuDeviceId(0)).unwrap(), 4_000);
    }

    #[test]
    fn destroying_slice_releases_reservation() {
        let mut p = TimeSliceProvider::new();
        let s = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(50, 1024, 4_000))
            .unwrap();
        p.destroy_slice(s.id).unwrap();
        assert_eq!(p.reserved_us(GpuDeviceId(0)).unwrap(), 0);
    }

    #[test]
    fn overcommit_rejected() {
        let mut p = TimeSliceProvider::new();
        p.create_slice(GpuDeviceId(0), SliceSpec::new(40, 1024, 10_000))
            .unwrap();
        let err = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(40, 1024, 7_000))
            .unwrap_err();
        assert_eq!(err, HalError::OutOfSlices);
    }

    #[test]
    fn spec_time_window_must_be_nonzero() {
        let mut p = TimeSliceProvider::new();
        let err = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(50, 1024, 0))
            .unwrap_err();
        assert!(matches!(err, HalError::InvalidSpec(_)));
    }

    #[test]
    fn spec_time_window_must_fit_device_window() {
        let mut p = TimeSliceProvider::with_profile_and_window(MockGpuProfile::Rtx4090, 5_000);
        let err = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(50, 1024, 6_000))
            .unwrap_err();
        assert!(matches!(err, HalError::InvalidSpec(_)));
    }

    #[test]
    fn empty_devices_for_none_profile() {
        let p = TimeSliceProvider::with_profile_and_window(
            MockGpuProfile::None,
            DEFAULT_TIME_SLICE_WINDOW_US,
        );
        assert!(p.devices().is_empty());
    }

    #[test]
    fn slice_is_tagged_as_time_slicing() {
        let mut p = TimeSliceProvider::new();
        let s = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(50, 1024, 4_000))
            .unwrap();
        assert_eq!(s.mode, SlicingMode::SoftwareTimeSlice);
        let live = p.live_slices(GpuDeviceId(0)).unwrap();
        assert!(live
            .iter()
            .all(|s| s.mode == SlicingMode::SoftwareTimeSlice));
    }

    #[test]
    fn telemetry_passes_through() {
        let mut p = TimeSliceProvider::new();
        let _ = p
            .poll_device_telemetry(GpuDeviceId(0), 1000)
            .expect("telemetry");
    }
}
