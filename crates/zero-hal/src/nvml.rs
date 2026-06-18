// SPDX-License-Identifier: AGPL-3.0-or-later
//! D.1 — NVML / DCGM-shaped GPU provider stub.
//!
//! Per the constraints doc §2.2: NVML (`nvml.h`) is the canonical
//! NVIDIA Telemetrie-Surface — Utilization, Power, Temperature,
//! Memory-Controller, MIG-Konfiguration. DCGM extends that with richer
//! profiling counters (SM-Occupancy, Tensor-Pipe-Active, FP64/FP32/FP16
//! pipe-active).
//!
//! This module ships the **API shape** as a concrete provider built on
//! top of [`MockGpuProvider`] — no real FFI bindings yet. The shape is
//! the load-bearing part: when D.1 is hardened against real hardware
//! (NVML FFI binding), the trait surface stays unchanged and only the
//! provider's internal data path swaps.
//!
//! Production [`NvmlProvider`] equivalents will live in a separate
//! crate (`zero-hal-nvidia` or similar) gated behind a cargo feature
//! so the no_std + alloc HAL stays platform-agnostic. The trait surface
//! defined here is what that future crate plugs into.

use alloc::vec::Vec;

use crate::error::HalError;
use crate::gpu::{
    GpuDeviceId, GpuProvider, GpuSlice, GpuSliceId, GpuVendor, SliceSpec, SlicingMode,
};
use crate::mock::{MockGpuProfile, MockGpuProvider};
use crate::telemetry::GpuTelemetrySnapshot;

/// NVML/DCGM-shaped provider stub.
///
/// Wraps a [`MockGpuProvider`] tagged with the requested profile. The
/// public API mirrors what a future native NVML/DCGM-backed provider
/// will expose:
///
/// - [`Self::with_mig_capable`] — datacenter mode (A100/H100/B100 →
///   MIG slicing).
/// - [`Self::with_consumer_card`] — consumer mode (RTX/GTX → software
///   time-slicing fallback).
/// - [`Self::utilization_rates`] / [`Self::power_draw`] — the
///   `nvmlDeviceGetUtilizationRates` / `nvmlDeviceGetPowerUsage`
///   counterparts, but routed through the deterministic snapshot
///   pipeline.
///
/// **Why route through the mock:** during D.1 the goal is API-shape
/// validation against the rest of the kernel (validator, sandbox,
/// policy log). Real NVML reads will land in a Linux-userspace shim
/// later; until then the deterministic mock gives the kernel-side
/// callers something to bind against.
pub struct NvmlProvider {
    inner: MockGpuProvider,
}

impl NvmlProvider {
    /// Datacenter-capable provider (MIG-eligible).
    pub fn with_mig_capable() -> Self {
        Self {
            inner: MockGpuProvider::new(MockGpuProfile::A100),
        }
    }

    /// Consumer card without MIG (software time-slicing fallback).
    pub fn with_consumer_card() -> Self {
        Self {
            inner: MockGpuProvider::new(MockGpuProfile::Rtx4090),
        }
    }

    /// `nvmlDeviceGetUtilizationRates`-counterpart. Returns the
    /// integer percent reading for the device.
    pub fn utilization_rates(
        &mut self,
        device: GpuDeviceId,
        interval_us: u32,
    ) -> Result<u8, HalError> {
        let snap = self.inner.poll_device_telemetry(device, interval_us)?;
        Ok(snap.utilization.percent())
    }

    /// `nvmlDeviceGetPowerUsage`-counterpart. Returns milliwatts.
    pub fn power_draw(&mut self, device: GpuDeviceId, interval_us: u32) -> Result<u32, HalError> {
        let snap = self.inner.poll_device_telemetry(device, interval_us)?;
        Ok(snap.power.milliwatts)
    }

    /// `nvmlDeviceGetMemoryBandwidth`-counterpart (DCGM-only on real
    /// hardware). MiB/s.
    pub fn memory_bandwidth(
        &mut self,
        device: GpuDeviceId,
        interval_us: u32,
    ) -> Result<u32, HalError> {
        let snap = self.inner.poll_device_telemetry(device, interval_us)?;
        Ok(snap.memory_bandwidth.mib_per_s)
    }
}

impl GpuProvider for NvmlProvider {
    fn vendor(&self) -> GpuVendor {
        self.inner.vendor()
    }

    fn devices(&self) -> Vec<GpuDeviceId> {
        self.inner.devices()
    }

    fn slicing_mode(&self, device: GpuDeviceId) -> Result<SlicingMode, HalError> {
        self.inner.slicing_mode(device)
    }

    fn create_slice(&mut self, device: GpuDeviceId, spec: SliceSpec) -> Result<GpuSlice, HalError> {
        self.inner.create_slice(device, spec)
    }

    fn destroy_slice(&mut self, slice: GpuSliceId) -> Result<(), HalError> {
        self.inner.destroy_slice(slice)
    }

    fn live_slices(&self, device: GpuDeviceId) -> Result<Vec<GpuSlice>, HalError> {
        self.inner.live_slices(device)
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
    fn mig_capable_uses_mig_mode() {
        let p = NvmlProvider::with_mig_capable();
        assert_eq!(
            p.slicing_mode(GpuDeviceId(0)).unwrap(),
            SlicingMode::HardwareMig
        );
    }

    #[test]
    fn consumer_uses_time_slicing() {
        let p = NvmlProvider::with_consumer_card();
        assert_eq!(
            p.slicing_mode(GpuDeviceId(0)).unwrap(),
            SlicingMode::SoftwareTimeSlice
        );
    }

    #[test]
    fn nvml_utilization_rates_returns_percent() {
        let mut p = NvmlProvider::with_mig_capable();
        let u = p.utilization_rates(GpuDeviceId(0), 1000).unwrap();
        assert!(u <= 100);
    }

    #[test]
    fn nvml_power_draw_returns_milliwatts() {
        let mut p = NvmlProvider::with_mig_capable();
        let _ = p.power_draw(GpuDeviceId(0), 1000).unwrap();
    }

    #[test]
    fn nvml_memory_bandwidth_returns_mib_per_s() {
        let mut p = NvmlProvider::with_mig_capable();
        let _ = p.memory_bandwidth(GpuDeviceId(0), 1000).unwrap();
    }

    #[test]
    fn nvml_create_destroy_slice_round_trip() {
        let mut p = NvmlProvider::with_mig_capable();
        let s = p
            .create_slice(GpuDeviceId(0), SliceSpec::new(25, 2048, 0))
            .unwrap();
        assert_eq!(p.live_slices(GpuDeviceId(0)).unwrap().len(), 1);
        p.destroy_slice(s.id).unwrap();
        assert!(p.live_slices(GpuDeviceId(0)).unwrap().is_empty());
    }

    #[test]
    fn nvml_unknown_device_errors() {
        let p = NvmlProvider::with_mig_capable();
        assert_eq!(
            p.slicing_mode(GpuDeviceId(42)).unwrap_err(),
            HalError::NoSuchDevice
        );
    }
}
