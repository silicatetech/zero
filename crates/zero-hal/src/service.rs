// SPDX-License-Identifier: AGPL-3.0-or-later
//! [`HardwareCapabilityService`] — top-level HCS facade.
//!
//! Per ADR-019 §4: the HCS is the deterministic kernel service that
//! mints handles on virtualized hardware slices. The SandboxManager
//! holds one [`HardwareCapabilityService`] and consults it for
//! `(policy gpu …)` slice operations and `(query gpu/thermal/memory
//! …)` telemetry reads.
//!
//! This struct is the canonical entry point. It does not enforce
//! capabilities itself (that is the SandboxManager's job) — it
//! enforces *slice budgets*, *telemetry sources*, and the
//! deterministic audit trail.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::audit::{AuditEvent, AuditTrail};
use crate::display::{
    DisplaySurfaceId, DisplayTelemetry, DisplayTelemetrySnapshot, MockDisplayTelemetry,
};
use crate::error::HalError;
use crate::gpu::{
    GpuDeviceId, GpuProvider, GpuSlice, GpuSliceId, GpuVendor, SliceSpec, SlicingMode,
};
use crate::mock::{MockGpuProfile, MockGpuProvider};
use crate::network::{
    MockNetworkTelemetry, NetworkEndpointId, NetworkTelemetry, NetworkTelemetrySnapshot,
};
use crate::nvml::NvmlProvider;
use crate::storage::{
    MockStorageTelemetry, StorageRegionId, StorageTelemetry, StorageTelemetrySnapshot,
};
use crate::telemetry::{Bandwidth, GpuTelemetrySnapshot, ThermalLevel};
use crate::time_slice::TimeSliceProvider;

/// Owner-binding for a minted slice. Equivalent to the
/// `Capability::owner` field on the SandboxManager side; tracked here
/// so the HCS can attribute audit entries even when the SandboxManager
/// has already torn down the capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceOwner {
    /// Sandbox id from `crates/zero-sandbox` (`SandboxId::raw()`).
    /// `None` for kernel-boot mints (Display Agent surface etc.).
    pub sandbox: Option<u64>,
}

impl SliceOwner {
    pub const fn kernel() -> Self {
        Self { sandbox: None }
    }

    pub const fn sandbox(id: u64) -> Self {
        Self { sandbox: Some(id) }
    }
}

#[derive(Debug, Clone, Copy)]
struct SliceRecord {
    owner: SliceOwner,
}

/// Top-level Hardware Capability Service.
pub struct HardwareCapabilityService {
    gpu: Box<dyn GpuProvider + Send + Sync>,
    network: Box<dyn NetworkTelemetry + Send + Sync>,
    storage: Box<dyn StorageTelemetry + Send + Sync>,
    display: Box<dyn DisplayTelemetry + Send + Sync>,
    /// Owner-binding table — keyed by minted GPU slice id.
    slice_owners: BTreeMap<GpuSliceId, SliceRecord>,
    audit: AuditTrail,
}

impl HardwareCapabilityService {
    /// Construct with a fully-mocked provider stack. Suitable for the
    /// boot path on hardware that has no native HAL backend yet (Stage
    /// 12 default).
    pub fn with_mock(profile: MockGpuProfile) -> Self {
        Self::with_providers(
            Box::new(MockGpuProvider::new(profile)),
            Box::new(MockNetworkTelemetry::new()),
            Box::new(MockStorageTelemetry::new()),
            Box::new(MockDisplayTelemetry::new()),
        )
    }

    /// Auto-detect the GPU slicing strategy based on the configured
    /// profile and build the HCS with the matching provider:
    ///
    /// - [`MockGpuProfile::A100`] → [`NvmlProvider::with_mig_capable`]
    ///   (datacenter MIG).
    /// - [`MockGpuProfile::Rtx4090`] → [`TimeSliceProvider`] (software
    ///   time-slicing fallback).
    /// - [`MockGpuProfile::None`] → [`MockGpuProvider`] with the
    ///   `None` profile (no GPU; `create_*` returns `NoSuchDevice`).
    pub fn auto_detect(profile: MockGpuProfile) -> Self {
        let gpu: Box<dyn GpuProvider + Send + Sync> = match profile {
            MockGpuProfile::A100 => Box::new(NvmlProvider::with_mig_capable()),
            MockGpuProfile::Rtx4090 => Box::new(TimeSliceProvider::new()),
            MockGpuProfile::None => Box::new(MockGpuProvider::new(MockGpuProfile::None)),
        };
        Self::with_providers(
            gpu,
            Box::new(MockNetworkTelemetry::new()),
            Box::new(MockStorageTelemetry::new()),
            Box::new(MockDisplayTelemetry::new()),
        )
    }

    pub fn with_providers(
        gpu: Box<dyn GpuProvider + Send + Sync>,
        network: Box<dyn NetworkTelemetry + Send + Sync>,
        storage: Box<dyn StorageTelemetry + Send + Sync>,
        display: Box<dyn DisplayTelemetry + Send + Sync>,
    ) -> Self {
        Self {
            gpu,
            network,
            storage,
            display,
            slice_owners: BTreeMap::new(),
            audit: AuditTrail::new(),
        }
    }

    // ---- GPU surface ------------------------------------------------

    pub fn gpu_vendor(&self) -> GpuVendor {
        self.gpu.vendor()
    }

    pub fn gpu_devices(&self) -> Vec<GpuDeviceId> {
        self.gpu.devices()
    }

    pub fn gpu_slicing_mode(&self, device: GpuDeviceId) -> Result<SlicingMode, HalError> {
        self.gpu.slicing_mode(device)
    }

    pub fn live_gpu_slices(&self, device: GpuDeviceId) -> Result<Vec<GpuSlice>, HalError> {
        self.gpu.live_slices(device)
    }

    /// Mint a new GPU slice on `device`, attributed to `owner`. Records
    /// the mint (or rejection) in the audit trail.
    pub fn request_gpu_slice(
        &mut self,
        owner: SliceOwner,
        device: GpuDeviceId,
        spec: SliceSpec,
    ) -> Result<GpuSlice, HalError> {
        match self.gpu.create_slice(device, spec) {
            Ok(slice) => {
                self.slice_owners.insert(slice.id, SliceRecord { owner });
                self.audit.record(
                    owner.sandbox,
                    AuditEvent::GpuSliceCreated {
                        device,
                        slice: slice.id,
                        spec: slice.spec,
                        mode: slice.mode,
                    },
                );
                Ok(slice)
            }
            Err(e) => {
                let reason = audit_reason_for_error(&e);
                self.audit.record(
                    owner.sandbox,
                    AuditEvent::GpuSliceRequestRejected {
                        device,
                        spec,
                        reason,
                    },
                );
                Err(e)
            }
        }
    }

    /// Release a GPU slice. The `owner` argument must match the owner
    /// recorded at mint time; mismatch returns `NoSuchSlice` (same
    /// surface as a fabricated id) to keep the audit signal terse.
    pub fn release_gpu_slice(
        &mut self,
        owner: SliceOwner,
        slice: GpuSliceId,
    ) -> Result<(), HalError> {
        let recorded = self.slice_owners.get(&slice).copied();
        match recorded {
            Some(rec) if rec.owner == owner => {
                self.gpu.destroy_slice(slice)?;
                self.slice_owners.remove(&slice);
                self.audit
                    .record(owner.sandbox, AuditEvent::GpuSliceDestroyed { slice });
                Ok(())
            }
            Some(_) => Err(HalError::NoSuchSlice),
            None => Err(HalError::NoSuchSlice),
        }
    }

    pub fn slice_owner(&self, slice: GpuSliceId) -> Option<SliceOwner> {
        self.slice_owners.get(&slice).map(|r| r.owner)
    }

    /// Sample telemetry on a live GPU slice.
    pub fn poll_gpu_slice_telemetry(
        &mut self,
        owner: SliceOwner,
        slice: GpuSliceId,
        interval_us: u32,
    ) -> Result<GpuTelemetrySnapshot, HalError> {
        let snap = self.gpu.poll_slice_telemetry(slice, interval_us)?;
        self.audit.record(
            owner.sandbox,
            AuditEvent::GpuSliceTelemetryPolled {
                slice,
                sample: snap.sample,
            },
        );
        Ok(snap)
    }

    /// Sample device-wide GPU telemetry.
    pub fn poll_gpu_device_telemetry(
        &mut self,
        owner: SliceOwner,
        device: GpuDeviceId,
        interval_us: u32,
    ) -> Result<GpuTelemetrySnapshot, HalError> {
        let snap = self.gpu.poll_device_telemetry(device, interval_us)?;
        self.audit.record(
            owner.sandbox,
            AuditEvent::GpuDeviceTelemetryPolled {
                device,
                sample: snap.sample,
            },
        );
        Ok(snap)
    }

    /// `(query gpu utilization)` route. Returns the device-wide
    /// utilization percent of the default device (`GpuDeviceId(0)`).
    /// On platforms without a GPU, returns `Unsupported` so the
    /// SandboxManager can fall back to the legacy stub constant.
    pub fn query_gpu_utilization(
        &mut self,
        owner: SliceOwner,
        interval_us: u32,
    ) -> Result<u8, HalError> {
        let device = self.default_gpu_device()?;
        let snap = self.poll_gpu_device_telemetry(owner, device, interval_us)?;
        Ok(snap.utilization.percent())
    }

    /// `(query thermal state)` route — ladder value 0..=3.
    pub fn query_thermal_state(
        &mut self,
        owner: SliceOwner,
        interval_us: u32,
    ) -> Result<ThermalLevel, HalError> {
        let device = self.default_gpu_device()?;
        let snap = self.poll_gpu_device_telemetry(owner, device, interval_us)?;
        Ok(snap.thermal)
    }

    /// `(query memory bandwidth)` route — system-wide MiB/s. Pulls
    /// from the storage subsystem's `poll_system` reading (where
    /// MPAM/PCM-style counters live), not the GPU's memory bandwidth
    /// (which is per-slice).
    pub fn query_memory_bandwidth(
        &mut self,
        _owner: SliceOwner,
        interval_us: u32,
    ) -> Result<Bandwidth, HalError> {
        let snap = self.storage.poll_system(interval_us)?;
        Ok(snap.memory_bandwidth)
    }

    fn default_gpu_device(&self) -> Result<GpuDeviceId, HalError> {
        self.gpu
            .devices()
            .into_iter()
            .next()
            .ok_or(HalError::Unsupported)
    }

    // ---- Network / Storage / Display telemetry passthroughs --------

    pub fn poll_network(
        &mut self,
        endpoint: NetworkEndpointId,
        interval_us: u32,
    ) -> Result<NetworkTelemetrySnapshot, HalError> {
        self.network.poll_endpoint(endpoint, interval_us)
    }

    pub fn poll_storage_region(
        &mut self,
        region: StorageRegionId,
        interval_us: u32,
    ) -> Result<StorageTelemetrySnapshot, HalError> {
        self.storage.poll_region(region, interval_us)
    }

    pub fn poll_display(
        &mut self,
        surface: DisplaySurfaceId,
        interval_us: u32,
    ) -> Result<DisplayTelemetrySnapshot, HalError> {
        self.display.poll_surface(surface, interval_us)
    }

    // ---- Audit -----------------------------------------------------

    pub fn audit(&self) -> &AuditTrail {
        &self.audit
    }
}

fn audit_reason_for_error(e: &HalError) -> &'static str {
    match e {
        HalError::Unsupported => "unsupported",
        HalError::NoSuchDevice => "no_such_device",
        HalError::NoSuchSlice => "no_such_slice",
        HalError::OutOfSlices => "out_of_slices",
        HalError::InvalidSpec(s) => s,
        HalError::TelemetryUnavailable => "telemetry_unavailable",
        HalError::AlreadyConfigured => "already_configured",
        HalError::InternalInvariant(s) => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mig_hcs() -> HardwareCapabilityService {
        HardwareCapabilityService::auto_detect(MockGpuProfile::A100)
    }

    fn rtx_hcs() -> HardwareCapabilityService {
        HardwareCapabilityService::auto_detect(MockGpuProfile::Rtx4090)
    }

    fn none_hcs() -> HardwareCapabilityService {
        HardwareCapabilityService::auto_detect(MockGpuProfile::None)
    }

    #[test]
    fn auto_detect_picks_mig_for_a100() {
        let hcs = mig_hcs();
        assert_eq!(hcs.gpu_vendor(), GpuVendor::Nvidia);
        assert_eq!(
            hcs.gpu_slicing_mode(GpuDeviceId(0)).unwrap(),
            SlicingMode::HardwareMig
        );
    }

    #[test]
    fn auto_detect_picks_time_slice_for_consumer() {
        let hcs = rtx_hcs();
        assert_eq!(
            hcs.gpu_slicing_mode(GpuDeviceId(0)).unwrap(),
            SlicingMode::SoftwareTimeSlice
        );
    }

    #[test]
    fn auto_detect_none_has_no_devices() {
        let hcs = none_hcs();
        assert!(hcs.gpu_devices().is_empty());
    }

    #[test]
    fn request_gpu_slice_records_mint() {
        let mut hcs = mig_hcs();
        let owner = SliceOwner::sandbox(7);
        let s = hcs
            .request_gpu_slice(owner, GpuDeviceId(0), SliceSpec::new(20, 1024, 0))
            .unwrap();
        assert_eq!(hcs.slice_owner(s.id), Some(owner));
        assert_eq!(hcs.audit().len(), 1);
        match hcs.audit().entries()[0].event {
            AuditEvent::GpuSliceCreated { device, slice, .. } => {
                assert_eq!(device, GpuDeviceId(0));
                assert_eq!(slice, s.id);
            }
            other => panic!("unexpected audit event: {:?}", other),
        }
    }

    #[test]
    fn request_gpu_slice_records_rejection() {
        let mut hcs = none_hcs();
        let owner = SliceOwner::sandbox(7);
        let err = hcs
            .request_gpu_slice(owner, GpuDeviceId(0), SliceSpec::new(20, 1024, 0))
            .unwrap_err();
        assert_eq!(err, HalError::NoSuchDevice);
        assert_eq!(hcs.audit().len(), 1);
        match &hcs.audit().entries()[0].event {
            AuditEvent::GpuSliceRequestRejected { reason, .. } => {
                assert_eq!(*reason, "no_such_device");
            }
            other => panic!("unexpected audit event: {:?}", other),
        }
    }

    #[test]
    fn release_gpu_slice_matches_owner() {
        let mut hcs = mig_hcs();
        let owner_a = SliceOwner::sandbox(7);
        let owner_b = SliceOwner::sandbox(99);
        let s = hcs
            .request_gpu_slice(owner_a, GpuDeviceId(0), SliceSpec::new(20, 1024, 0))
            .unwrap();
        // Wrong owner should be rejected.
        let err = hcs.release_gpu_slice(owner_b, s.id).unwrap_err();
        assert_eq!(err, HalError::NoSuchSlice);
        assert!(hcs.slice_owner(s.id).is_some());
        // Correct owner releases.
        hcs.release_gpu_slice(owner_a, s.id).unwrap();
        assert!(hcs.slice_owner(s.id).is_none());
    }

    #[test]
    fn poll_records_telemetry_event() {
        let mut hcs = mig_hcs();
        let owner = SliceOwner::sandbox(5);
        let _ = hcs
            .poll_gpu_device_telemetry(owner, GpuDeviceId(0), 1000)
            .unwrap();
        assert!(matches!(
            hcs.audit().entries().last().unwrap().event,
            AuditEvent::GpuDeviceTelemetryPolled { .. }
        ));
    }

    #[test]
    fn query_gpu_utilization_returns_percent() {
        let mut hcs = mig_hcs();
        let u = hcs
            .query_gpu_utilization(SliceOwner::kernel(), 1000)
            .unwrap();
        assert!(u <= 100);
    }

    #[test]
    fn query_gpu_utilization_no_device_returns_unsupported() {
        let mut hcs = none_hcs();
        assert_eq!(
            hcs.query_gpu_utilization(SliceOwner::kernel(), 1000)
                .unwrap_err(),
            HalError::Unsupported
        );
    }

    #[test]
    fn query_thermal_state_returns_ladder() {
        let mut hcs = mig_hcs();
        let t = hcs.query_thermal_state(SliceOwner::kernel(), 1000).unwrap();
        assert!(matches!(
            t,
            ThermalLevel::Nominal
                | ThermalLevel::Fair
                | ThermalLevel::Serious
                | ThermalLevel::Critical
        ));
    }

    #[test]
    fn query_memory_bandwidth_pulls_from_storage() {
        let mut hcs = mig_hcs();
        let b = hcs
            .query_memory_bandwidth(SliceOwner::kernel(), 1000)
            .unwrap();
        // First sample is non-zero per the synthetic ramp.
        assert!(b.mib_per_s < 100_000);
    }

    #[test]
    fn network_storage_display_passthroughs_work() {
        let mut hcs = mig_hcs();
        let _ = hcs.poll_network(NetworkEndpointId(0), 1000).unwrap();
        let _ = hcs.poll_storage_region(StorageRegionId(0), 1000).unwrap();
        let _ = hcs.poll_display(DisplaySurfaceId(0), 1000).unwrap();
    }

    #[test]
    fn determinism_two_hcss_produce_identical_audit_and_telemetry() {
        let mut a = mig_hcs();
        let mut b = mig_hcs();
        let owner = SliceOwner::sandbox(1);
        let spec = SliceSpec::new(10, 1024, 0);
        let sa = a.request_gpu_slice(owner, GpuDeviceId(0), spec).unwrap();
        let sb = b.request_gpu_slice(owner, GpuDeviceId(0), spec).unwrap();
        assert_eq!(sa.id, sb.id);
        let ta = a.poll_gpu_slice_telemetry(owner, sa.id, 1000).unwrap();
        let tb = b.poll_gpu_slice_telemetry(owner, sb.id, 1000).unwrap();
        assert_eq!(ta, tb);
        assert_eq!(a.audit().entries(), b.audit().entries());
    }
}
