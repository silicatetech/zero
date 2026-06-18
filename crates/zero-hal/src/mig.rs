// SPDX-License-Identifier: AGPL-3.0-or-later
//! D.2 — MIG slicing API (GPU Instance → Compute Instance hierarchy).
//!
//! NVIDIA MIG (Multi-Instance GPU, A100/H100/B100) partitions a GPU
//! into up to **7** isolated *GPU Instances*, each with dedicated SMs,
//! L2 slices, memory controllers and HBM. A GPU Instance may itself be
//! further subdivided into one or more *Compute Instances*, which
//! share the GPU Instance's memory but isolate the SM scheduler. This
//! module exposes that two-level hierarchy through deterministic
//! create / destroy / enumerate operations.
//!
//! Real-NVIDIA-API mapping:
//!
//! - `nvmlDeviceCreateGpuInstance` ↔ [`MigProvider::create_gpu_instance`]
//! - `nvmlGpuInstanceCreateComputeInstance` ↔ [`MigProvider::create_compute_instance`]
//! - `nvmlComputeInstanceDestroy` ↔ [`MigProvider::destroy_compute_instance`]
//! - `nvmlGpuInstanceDestroy` ↔ [`MigProvider::destroy_gpu_instance`]
//!
//! The provider is built on top of [`NvmlProvider::with_mig_capable`]
//! to keep the trait-shape testable without real hardware. A GPU
//! Instance corresponds 1:1 to a backing NVML slice (so the 7-instance
//! ceiling is naturally enforced by the mock's slice budget on A100);
//! Compute Instances are *sub-partitions inside the GI's compute
//! budget* and do not consume additional device-level slice budget.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::error::HalError;
use crate::gpu::{
    GpuDeviceId, GpuProvider, GpuSlice, GpuSliceId, GpuVendor, SliceSpec, SlicingMode,
};
use crate::nvml::NvmlProvider;
use crate::telemetry::GpuTelemetrySnapshot;

/// MIG ceiling per device per NVIDIA spec: 7 GPU Instances on A100
/// (and the equivalent ceiling on H100/B100).
pub const MIG_MAX_INSTANCES: usize = 7;

/// Identifier for a MIG GPU Instance (outer partition).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MigGpuInstance(pub u32);

/// Identifier for a MIG Compute Instance (inner partition; lives
/// inside a GPU Instance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MigComputeInstance(pub u32);

#[derive(Debug)]
struct ComputeInstanceState {
    compute_percent: u8,
    // Carried for future Compute-Instance-scoped telemetry / quota
    // enforcement; the GI's backing slice owns the actual memory
    // budget today.
    #[allow(dead_code)]
    memory_mib: u32,
}

#[derive(Debug)]
struct GpuInstanceState {
    spec: SliceSpec,
    backing_slice: GpuSliceId,
    compute_instances: BTreeMap<MigComputeInstance, ComputeInstanceState>,
    next_ci_id: u32,
}

impl GpuInstanceState {
    fn ci_compute_used(&self) -> u32 {
        self.compute_instances
            .values()
            .map(|c| c.compute_percent as u32)
            .sum()
    }
}

/// MIG-aware GPU provider. Wraps [`NvmlProvider::with_mig_capable`]
/// and adds the GPU-Instance / Compute-Instance hierarchy on top of
/// the bare slice surface.
pub struct MigProvider {
    inner: NvmlProvider,
    instances: BTreeMap<MigGpuInstance, GpuInstanceState>,
    next_gi_id: u32,
}

impl MigProvider {
    pub fn new() -> Self {
        Self {
            inner: NvmlProvider::with_mig_capable(),
            instances: BTreeMap::new(),
            next_gi_id: 0,
        }
    }

    /// Number of live GPU Instances on the device.
    pub fn live_gpu_instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Total compute instances across all live GPU instances.
    pub fn live_compute_instance_count(&self) -> usize {
        self.instances
            .values()
            .map(|s| s.compute_instances.len())
            .sum()
    }

    /// Enumerate live GPU Instances.
    pub fn list_gpu_instances(&self) -> Vec<MigGpuInstance> {
        self.instances.keys().copied().collect()
    }

    /// Enumerate Compute Instances inside a GPU Instance.
    pub fn list_compute_instances(
        &self,
        gi: MigGpuInstance,
    ) -> Result<Vec<MigComputeInstance>, HalError> {
        let st = self.instances.get(&gi).ok_or(HalError::NoSuchSlice)?;
        Ok(st.compute_instances.keys().copied().collect())
    }

    /// Compute fraction (percent points) the GI was minted with.
    pub fn gpu_instance_compute_percent(&self, gi: MigGpuInstance) -> Result<u8, HalError> {
        Ok(self
            .instances
            .get(&gi)
            .ok_or(HalError::NoSuchSlice)?
            .spec
            .compute_percent)
    }

    /// Backing slice id of a GPU Instance (the NVML-level slice handle
    /// that backs the partition).
    pub fn gpu_instance_backing_slice(&self, gi: MigGpuInstance) -> Result<GpuSliceId, HalError> {
        Ok(self
            .instances
            .get(&gi)
            .ok_or(HalError::NoSuchSlice)?
            .backing_slice)
    }

    /// Create a new GPU Instance backed by an NVML slice satisfying
    /// `spec`. Enforces the 7-instance ceiling.
    pub fn create_gpu_instance(
        &mut self,
        device: GpuDeviceId,
        spec: SliceSpec,
    ) -> Result<MigGpuInstance, HalError> {
        if self.instances.len() >= MIG_MAX_INSTANCES {
            return Err(HalError::OutOfSlices);
        }
        let slice = self.inner.create_slice(device, spec)?;
        let id = MigGpuInstance(self.next_gi_id);
        self.next_gi_id = self.next_gi_id.saturating_add(1);
        self.instances.insert(
            id,
            GpuInstanceState {
                spec,
                backing_slice: slice.id,
                compute_instances: BTreeMap::new(),
                next_ci_id: 0,
            },
        );
        Ok(id)
    }

    /// Subdivide a GPU Instance into a Compute Instance. The CI's
    /// compute fraction is taken from the parent GPU Instance's
    /// budget; the sum of CI percentages within a GI must not exceed
    /// the GI's own `compute_percent`.
    ///
    /// Compute Instances are pure sub-partitions of the GI's compute
    /// budget. They do not consume additional device-level slice
    /// budget — only the parent GI does.
    pub fn create_compute_instance(
        &mut self,
        gi: MigGpuInstance,
        compute_percent: u8,
        memory_mib: u32,
    ) -> Result<MigComputeInstance, HalError> {
        if compute_percent == 0 || compute_percent > 100 {
            return Err(HalError::InvalidSpec("ci_compute_percent_out_of_range"));
        }
        let gi_state = self.instances.get_mut(&gi).ok_or(HalError::NoSuchSlice)?;
        let parent_budget = gi_state.spec.compute_percent as u32;
        let used = gi_state.ci_compute_used();
        if used + compute_percent as u32 > parent_budget {
            return Err(HalError::InvalidSpec("ci_overcommit_parent"));
        }
        let id = MigComputeInstance(gi_state.next_ci_id);
        gi_state.next_ci_id = gi_state.next_ci_id.saturating_add(1);
        gi_state.compute_instances.insert(
            id,
            ComputeInstanceState {
                compute_percent,
                memory_mib,
            },
        );
        Ok(id)
    }

    pub fn destroy_compute_instance(
        &mut self,
        gi: MigGpuInstance,
        ci: MigComputeInstance,
    ) -> Result<(), HalError> {
        let gi_state = self.instances.get_mut(&gi).ok_or(HalError::NoSuchSlice)?;
        gi_state
            .compute_instances
            .remove(&ci)
            .ok_or(HalError::NoSuchSlice)?;
        Ok(())
    }

    /// Destroy a GPU Instance. Real NVML requires Compute Instances to
    /// be destroyed first; the HAL mirrors that contract by failing
    /// with [`HalError::InvalidSpec`] if any CIs are still live.
    pub fn destroy_gpu_instance(&mut self, gi: MigGpuInstance) -> Result<(), HalError> {
        let gi_state = self.instances.get(&gi).ok_or(HalError::NoSuchSlice)?;
        if !gi_state.compute_instances.is_empty() {
            return Err(HalError::InvalidSpec("destroy_gi_with_live_cis"));
        }
        let backing = gi_state.backing_slice;
        self.instances.remove(&gi);
        self.inner.destroy_slice(backing)
    }

    /// Underlying NVML provider, for [`GpuProvider`] delegation.
    pub fn inner(&self) -> &NvmlProvider {
        &self.inner
    }
}

impl Default for MigProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProvider for MigProvider {
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
    fn mig_provider_reports_mig_mode() {
        let p = MigProvider::new();
        assert_eq!(
            p.slicing_mode(GpuDeviceId(0)).unwrap(),
            SlicingMode::HardwareMig
        );
    }

    #[test]
    fn mig_create_gpu_instance_increments_counter() {
        let mut p = MigProvider::new();
        let gi = p
            .create_gpu_instance(GpuDeviceId(0), SliceSpec::new(14, 8192, 0))
            .unwrap();
        assert_eq!(gi, MigGpuInstance(0));
        assert_eq!(p.live_gpu_instance_count(), 1);
    }

    #[test]
    fn mig_caps_at_seven_gpu_instances() {
        let mut p = MigProvider::new();
        // 7 × 14% = 98% — fits inside compute budget; reaches MIG ceiling.
        for _ in 0..MIG_MAX_INSTANCES {
            p.create_gpu_instance(GpuDeviceId(0), SliceSpec::new(14, 1024, 0))
                .unwrap();
        }
        assert_eq!(p.live_gpu_instance_count(), MIG_MAX_INSTANCES);
        let err = p
            .create_gpu_instance(GpuDeviceId(0), SliceSpec::new(1, 1024, 0))
            .unwrap_err();
        assert_eq!(err, HalError::OutOfSlices);
    }

    #[test]
    fn mig_compute_instance_inherits_from_gpu_instance() {
        let mut p = MigProvider::new();
        let gi = p
            .create_gpu_instance(GpuDeviceId(0), SliceSpec::new(50, 8192, 0))
            .unwrap();
        let ci0 = p.create_compute_instance(gi, 20, 2048).unwrap();
        let ci1 = p.create_compute_instance(gi, 20, 2048).unwrap();
        assert_eq!(ci0, MigComputeInstance(0));
        assert_eq!(ci1, MigComputeInstance(1));
        let cis = p.list_compute_instances(gi).unwrap();
        assert_eq!(cis.len(), 2);
        assert_eq!(p.live_compute_instance_count(), 2);
    }

    #[test]
    fn mig_compute_instance_rejects_overcommit() {
        let mut p = MigProvider::new();
        let gi = p
            .create_gpu_instance(GpuDeviceId(0), SliceSpec::new(30, 4096, 0))
            .unwrap();
        p.create_compute_instance(gi, 20, 1024).unwrap();
        let err = p.create_compute_instance(gi, 15, 1024).unwrap_err();
        assert!(matches!(err, HalError::InvalidSpec(_)));
    }

    #[test]
    fn mig_destroy_gpu_instance_with_live_cis_rejected() {
        let mut p = MigProvider::new();
        let gi = p
            .create_gpu_instance(GpuDeviceId(0), SliceSpec::new(50, 4096, 0))
            .unwrap();
        let _ci = p.create_compute_instance(gi, 20, 1024).unwrap();
        let err = p.destroy_gpu_instance(gi).unwrap_err();
        assert!(matches!(err, HalError::InvalidSpec(_)));
    }

    #[test]
    fn mig_destroy_ci_then_gi() {
        let mut p = MigProvider::new();
        let gi = p
            .create_gpu_instance(GpuDeviceId(0), SliceSpec::new(40, 4096, 0))
            .unwrap();
        let ci = p.create_compute_instance(gi, 20, 1024).unwrap();
        p.destroy_compute_instance(gi, ci).unwrap();
        assert!(p.list_compute_instances(gi).unwrap().is_empty());
        p.destroy_gpu_instance(gi).unwrap();
        assert_eq!(p.live_gpu_instance_count(), 0);
    }

    #[test]
    fn mig_unknown_gi_rejected() {
        let mut p = MigProvider::new();
        assert_eq!(
            p.destroy_gpu_instance(MigGpuInstance(99)).unwrap_err(),
            HalError::NoSuchSlice
        );
        assert_eq!(
            p.destroy_compute_instance(MigGpuInstance(99), MigComputeInstance(0))
                .unwrap_err(),
            HalError::NoSuchSlice
        );
    }

    #[test]
    fn mig_unknown_ci_rejected() {
        let mut p = MigProvider::new();
        let gi = p
            .create_gpu_instance(GpuDeviceId(0), SliceSpec::new(30, 1024, 0))
            .unwrap();
        assert_eq!(
            p.destroy_compute_instance(gi, MigComputeInstance(42))
                .unwrap_err(),
            HalError::NoSuchSlice
        );
    }

    #[test]
    fn mig_provider_passes_through_telemetry() {
        let mut p = MigProvider::new();
        let _ = p
            .poll_device_telemetry(GpuDeviceId(0), 1000)
            .expect("telemetry available");
    }

    #[test]
    fn mig_ci_compute_percent_zero_rejected() {
        let mut p = MigProvider::new();
        let gi = p
            .create_gpu_instance(GpuDeviceId(0), SliceSpec::new(30, 1024, 0))
            .unwrap();
        let err = p.create_compute_instance(gi, 0, 1024).unwrap_err();
        assert!(matches!(err, HalError::InvalidSpec(_)));
    }

    #[test]
    fn mig_gi_backing_slice_is_queryable() {
        let mut p = MigProvider::new();
        let gi = p
            .create_gpu_instance(GpuDeviceId(0), SliceSpec::new(30, 1024, 0))
            .unwrap();
        let backing = p.gpu_instance_backing_slice(gi).unwrap();
        let live = p.inner().live_slices(GpuDeviceId(0)).unwrap();
        assert!(live.iter().any(|s| s.id == backing));
    }
}
