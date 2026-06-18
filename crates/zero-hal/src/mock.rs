// SPDX-License-Identifier: AGPL-3.0-or-later
//! D.5 — Mock GPU provider for tests.
//!
//! Implements the full [`GpuProvider`] surface with deterministic
//! state. Three pre-canned profiles simulate the three platforms the
//! HAL has to remain coherent across:
//!
//! - [`MockGpuProfile::A100`] — datacenter NVIDIA with hardware MIG
//!   (7-instance ceiling). The MIG hierarchy is enforced through the
//!   provider's slice budget; deeper MIG semantics (GPU-Instance vs.
//!   Compute-Instance) live in [`crate::mig::MigProvider`] on top.
//! - [`MockGpuProfile::Rtx4090`] — consumer NVIDIA without MIG.
//!   Slicing mode is [`SlicingMode::SoftwareTimeSlice`]; budget is
//!   the time-slicing fallback's quantum tracking.
//! - [`MockGpuProfile::None`] — no GPU. Useful for "this platform has
//!   no GPU" path coverage.
//!
//! Telemetry values follow a deterministic ramp seeded from
//! `sample_count` so tests can assert against exact numbers without
//! pinning a hardware reading.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::error::HalError;
use crate::gpu::{
    GpuDeviceId, GpuProvider, GpuSlice, GpuSliceId, GpuVendor, SliceSpec, SlicingMode,
};
use crate::telemetry::{
    Bandwidth, GpuTelemetrySnapshot, Power, TelemetryResolution, ThermalLevel, Utilization,
};

/// Which simulated platform the mock represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockGpuProfile {
    /// NVIDIA A100 with hardware MIG. 7 instances max, datacenter
    /// telemetry resolution.
    A100,
    /// NVIDIA RTX 4090, no MIG. Software time-slicing only.
    Rtx4090,
    /// No GPU present. `devices()` returns empty; `create_slice`
    /// returns `NoSuchDevice`.
    None,
}

impl MockGpuProfile {
    fn vendor(self) -> GpuVendor {
        match self {
            MockGpuProfile::A100 | MockGpuProfile::Rtx4090 => GpuVendor::Nvidia,
            MockGpuProfile::None => GpuVendor::Unknown,
        }
    }

    fn slice_budget(self) -> u8 {
        match self {
            MockGpuProfile::A100 => 7,
            MockGpuProfile::Rtx4090 => 4,
            MockGpuProfile::None => 0,
        }
    }

    fn slicing_mode(self) -> SlicingMode {
        match self {
            MockGpuProfile::A100 => SlicingMode::HardwareMig,
            MockGpuProfile::Rtx4090 => SlicingMode::SoftwareTimeSlice,
            MockGpuProfile::None => SlicingMode::Exclusive,
        }
    }

    fn telemetry_resolution(self) -> TelemetryResolution {
        match self {
            MockGpuProfile::A100 => TelemetryResolution::Hardware,
            MockGpuProfile::Rtx4090 => TelemetryResolution::DriverSoftware,
            MockGpuProfile::None => TelemetryResolution::Synthetic,
        }
    }

    fn devices(self) -> Vec<GpuDeviceId> {
        match self {
            MockGpuProfile::None => Vec::new(),
            _ => vec![GpuDeviceId(0)],
        }
    }
}

#[derive(Debug)]
struct DeviceState {
    // Profile is captured at construction so a future
    // multi-device-per-provider extension can vary telemetry per
    // device without re-reading the parent provider's profile.
    #[allow(dead_code)]
    profile: MockGpuProfile,
    next_slice_id: u64,
    slices: BTreeMap<GpuSliceId, GpuSlice>,
    sample_count: u64,
}

/// Mock GPU provider — full deterministic [`GpuProvider`] impl.
#[derive(Debug)]
pub struct MockGpuProvider {
    profile: MockGpuProfile,
    devices: BTreeMap<GpuDeviceId, DeviceState>,
}

impl MockGpuProvider {
    pub fn new(profile: MockGpuProfile) -> Self {
        let mut devices = BTreeMap::new();
        for d in profile.devices() {
            devices.insert(
                d,
                DeviceState {
                    profile,
                    next_slice_id: 1,
                    slices: BTreeMap::new(),
                    sample_count: 0,
                },
            );
        }
        Self { profile, devices }
    }

    pub fn profile(&self) -> MockGpuProfile {
        self.profile
    }

    /// Override the telemetry seed for a device. Test-only helper that
    /// lets a test reproduce a specific snapshot value without burning
    /// through `sample_count` increments.
    pub fn set_sample_seed(&mut self, device: GpuDeviceId, seed: u64) -> Result<(), HalError> {
        let st = self
            .devices
            .get_mut(&device)
            .ok_or(HalError::NoSuchDevice)?;
        st.sample_count = seed;
        Ok(())
    }

    fn synth_snapshot(
        &mut self,
        device: GpuDeviceId,
        interval_us: u32,
    ) -> Result<GpuTelemetrySnapshot, HalError> {
        let st = self
            .devices
            .get_mut(&device)
            .ok_or(HalError::NoSuchDevice)?;
        st.sample_count = st.sample_count.saturating_add(1);
        // Deterministic ramp keyed off sample_count. Modular so values
        // stay in their respective domains without floating-point math.
        let s = st.sample_count;
        let util = ((s.wrapping_mul(7) % 41) as u8).min(100);
        let bw = (s.wrapping_mul(1024) % 32_768) as u32;
        let mw = (s.wrapping_mul(50) % 350_000) as u32;
        // Thermal ladder cycles 0..3 every 4 samples — predictable.
        let thermal = ThermalLevel::from_code((s % 4) as u8)?;
        Ok(GpuTelemetrySnapshot {
            sample: s,
            interval_us,
            utilization: Utilization::new(util),
            memory_bandwidth: Bandwidth::from_mib_per_s(bw),
            thermal,
            power: Power::from_mw(mw),
            resolution: self.profile.telemetry_resolution(),
        })
    }

    fn assert_device(&self, device: GpuDeviceId) -> Result<(), HalError> {
        if !self.devices.contains_key(&device) {
            return Err(HalError::NoSuchDevice);
        }
        Ok(())
    }
}

impl GpuProvider for MockGpuProvider {
    fn vendor(&self) -> GpuVendor {
        self.profile.vendor()
    }

    fn devices(&self) -> Vec<GpuDeviceId> {
        self.devices.keys().copied().collect()
    }

    fn slicing_mode(&self, device: GpuDeviceId) -> Result<SlicingMode, HalError> {
        self.assert_device(device)?;
        Ok(self.profile.slicing_mode())
    }

    fn create_slice(&mut self, device: GpuDeviceId, spec: SliceSpec) -> Result<GpuSlice, HalError> {
        spec.validate()?;
        if self.profile == MockGpuProfile::None {
            return Err(HalError::NoSuchDevice);
        }
        let budget = self.profile.slice_budget() as usize;
        let mode = self.profile.slicing_mode();
        let st = self
            .devices
            .get_mut(&device)
            .ok_or(HalError::NoSuchDevice)?;
        if st.slices.len() >= budget {
            return Err(HalError::OutOfSlices);
        }
        // Sum of compute percent across live slices must not exceed
        // 100 — this is the deterministic budget the provider enforces.
        let used: u32 = st
            .slices
            .values()
            .map(|s| s.spec.compute_percent as u32)
            .sum();
        if used + spec.compute_percent as u32 > 100 {
            return Err(HalError::InvalidSpec("compute_percent_overcommit"));
        }
        let id = GpuSliceId(st.next_slice_id);
        st.next_slice_id = st.next_slice_id.saturating_add(1);
        let slice = GpuSlice {
            id,
            device,
            spec,
            mode,
        };
        st.slices.insert(id, slice);
        Ok(slice)
    }

    fn destroy_slice(&mut self, slice: GpuSliceId) -> Result<(), HalError> {
        for st in self.devices.values_mut() {
            if st.slices.remove(&slice).is_some() {
                return Ok(());
            }
        }
        Err(HalError::NoSuchSlice)
    }

    fn live_slices(&self, device: GpuDeviceId) -> Result<Vec<GpuSlice>, HalError> {
        let st = self.devices.get(&device).ok_or(HalError::NoSuchDevice)?;
        Ok(st.slices.values().copied().collect())
    }

    fn poll_slice_telemetry(
        &mut self,
        slice: GpuSliceId,
        interval_us: u32,
    ) -> Result<GpuTelemetrySnapshot, HalError> {
        // Find which device hosts the slice without holding a mut borrow
        // across `synth_snapshot`.
        let mut host: Option<GpuDeviceId> = None;
        for (dev, st) in &self.devices {
            if st.slices.contains_key(&slice) {
                host = Some(*dev);
                break;
            }
        }
        let device = host.ok_or(HalError::NoSuchSlice)?;
        self.synth_snapshot(device, interval_us)
    }

    fn poll_device_telemetry(
        &mut self,
        device: GpuDeviceId,
        interval_us: u32,
    ) -> Result<GpuTelemetrySnapshot, HalError> {
        self.synth_snapshot(device, interval_us)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a100_enumerates_one_device_and_is_mig() {
        let p = MockGpuProvider::new(MockGpuProfile::A100);
        let devs = p.devices();
        assert_eq!(devs, alloc::vec![GpuDeviceId(0)]);
        assert_eq!(
            p.slicing_mode(GpuDeviceId(0)).unwrap(),
            SlicingMode::HardwareMig
        );
        assert_eq!(p.vendor(), GpuVendor::Nvidia);
    }

    #[test]
    fn rtx4090_uses_time_slicing() {
        let p = MockGpuProvider::new(MockGpuProfile::Rtx4090);
        assert_eq!(
            p.slicing_mode(GpuDeviceId(0)).unwrap(),
            SlicingMode::SoftwareTimeSlice
        );
    }

    #[test]
    fn none_profile_has_no_devices_or_slices() {
        let mut p = MockGpuProvider::new(MockGpuProfile::None);
        assert!(p.devices().is_empty());
        let err = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(50, 1024, 0))
            .unwrap_err();
        assert_eq!(err, HalError::NoSuchDevice);
    }

    #[test]
    fn a100_caps_at_seven_slices() {
        let mut p = MockGpuProvider::new(MockGpuProfile::A100);
        for _ in 0..7 {
            // Use 10% each so the compute-percent budget doesn't fire
            // before the slice-count budget does.
            p.create_slice(GpuDeviceId(0), SliceSpec::new(10, 1024, 0))
                .expect("under budget");
        }
        let err = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(10, 1024, 0))
            .unwrap_err();
        assert_eq!(err, HalError::OutOfSlices);
    }

    #[test]
    fn compute_percent_overcommit_rejected() {
        let mut p = MockGpuProvider::new(MockGpuProfile::A100);
        p.create_slice(GpuDeviceId(0), SliceSpec::new(60, 1024, 0))
            .unwrap();
        let err = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(50, 1024, 0))
            .unwrap_err();
        assert!(matches!(err, HalError::InvalidSpec(_)));
    }

    #[test]
    fn destroy_slice_frees_budget() {
        let mut p = MockGpuProvider::new(MockGpuProfile::A100);
        let s = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(50, 1024, 0))
            .unwrap();
        assert_eq!(p.live_slices(GpuDeviceId(0)).unwrap().len(), 1);
        p.destroy_slice(s.id).unwrap();
        assert!(p.live_slices(GpuDeviceId(0)).unwrap().is_empty());
        // After destroy, telemetry on the slice returns NoSuchSlice.
        assert_eq!(
            p.poll_slice_telemetry(s.id, 1000).unwrap_err(),
            HalError::NoSuchSlice
        );
    }

    #[test]
    fn destroy_unknown_slice_errors() {
        let mut p = MockGpuProvider::new(MockGpuProfile::A100);
        assert_eq!(
            p.destroy_slice(GpuSliceId(999)).unwrap_err(),
            HalError::NoSuchSlice
        );
    }

    #[test]
    fn telemetry_is_deterministic_per_sample() {
        let mut a = MockGpuProvider::new(MockGpuProfile::A100);
        let mut b = MockGpuProvider::new(MockGpuProfile::A100);
        let snap_a = a.poll_device_telemetry(GpuDeviceId(0), 1000).unwrap();
        let snap_b = b.poll_device_telemetry(GpuDeviceId(0), 1000).unwrap();
        assert_eq!(snap_a, snap_b);
    }

    #[test]
    fn telemetry_sample_is_monotonic() {
        let mut p = MockGpuProvider::new(MockGpuProfile::A100);
        let s0 = p.poll_device_telemetry(GpuDeviceId(0), 1000).unwrap();
        let s1 = p.poll_device_telemetry(GpuDeviceId(0), 1000).unwrap();
        let s2 = p.poll_device_telemetry(GpuDeviceId(0), 1000).unwrap();
        assert_eq!(s0.sample, 1);
        assert_eq!(s1.sample, 2);
        assert_eq!(s2.sample, 3);
    }

    #[test]
    fn telemetry_resolution_reflects_profile() {
        let mut p_a100 = MockGpuProvider::new(MockGpuProfile::A100);
        let mut p_rtx = MockGpuProvider::new(MockGpuProfile::Rtx4090);
        assert_eq!(
            p_a100
                .poll_device_telemetry(GpuDeviceId(0), 1000)
                .unwrap()
                .resolution,
            TelemetryResolution::Hardware
        );
        assert_eq!(
            p_rtx
                .poll_device_telemetry(GpuDeviceId(0), 1000)
                .unwrap()
                .resolution,
            TelemetryResolution::DriverSoftware
        );
    }

    #[test]
    fn slice_spec_validation_runs() {
        let mut p = MockGpuProvider::new(MockGpuProfile::A100);
        let bad = p.create_slice(GpuDeviceId(0), SliceSpec::new(0, 1024, 0));
        assert!(matches!(bad, Err(HalError::InvalidSpec(_))));
    }

    #[test]
    fn set_sample_seed_is_isolated_per_device() {
        let mut p = MockGpuProvider::new(MockGpuProfile::A100);
        p.set_sample_seed(GpuDeviceId(0), 99).unwrap();
        let s = p.poll_device_telemetry(GpuDeviceId(0), 0).unwrap();
        assert_eq!(s.sample, 100);
    }

    #[test]
    fn set_sample_seed_unknown_device_errors() {
        let mut p = MockGpuProvider::new(MockGpuProfile::A100);
        assert_eq!(
            p.set_sample_seed(GpuDeviceId(42), 0).unwrap_err(),
            HalError::NoSuchDevice
        );
    }
}
