// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg_attr(not(feature = "std"), no_std)]
//! Stage 12 Paket D — Hardware Capability Service HAL.
//!
//! Implements the platform-agnostic trait surface described in
//! `docs/discovery/hardware-abstraction-constraints.md` §4.
//! The HAL is the data-plane side of the Hardware Capability
//! Service: it owns the slice-minting + telemetry-read mechanics for
//! GPU/Network/Storage/Display. Capability gating and owner-binding
//! live on the `SandboxManager` boundary (`crates/zero-sandbox`)
//! and are not duplicated here — the HAL trusts its callers to have
//! cleared the capability check before invoking it.
//!
//! # Determinism contract
//!
//! Per `hardware-abstraction-constraints.md` §1.3 and §4.2:
//!
//! - **No floating point** anywhere in HAL aggregation. Telemetry
//!   values are integer fixed-point (`u8` percent, `u32` mW / MiB·s⁻¹,
//!   `u64` cumulative byte counters).
//! - **No `Instant::now()`**, `rand`, or wallclock reads. Sampling is
//!   advanced by an explicit `tick()` counter or by a host-supplied
//!   monotonic clock value passed in.
//! - **`BTreeMap` only** for any collection that needs lookup. Never
//!   `HashMap` (non-deterministic iteration order).
//! - **No panics.** Every fallible operation returns
//!   [`HalError`]. The HAL is innermost-Ring-0; a panic here is a
//!   system freeze.
//!
//! # Crate layout
//!
//! - [`error`] — [`HalError`] taxonomy.
//! - [`telemetry`] — value types ([`Utilization`], [`ThermalLevel`],
//!   [`Power`], [`Bandwidth`], [`GpuTelemetrySnapshot`]).
//! - [`gpu`] — [`GpuProvider`] trait + [`GpuSlice`] handle.
//! - [`nvml`] — D.1 NVML/DCGM-shaped stub provider.
//! - [`mig`] — D.2 MIG slicing API (GPU Instance → Compute Instance).
//! - [`time_slice`] — D.4 software time-slicing fallback for consumer
//!   GPUs without hardware partitioning.
//! - [`mock`] — D.5 fully deterministic mock provider for tests.
//! - [`network`] / [`storage`] / [`display`] — telemetry stubs for the
//!   non-GPU subsystems (full provider surface deferred per D.1 plan).
//! - [`service`] — [`HardwareCapabilityService`], the top-level façade
//!   that the `SandboxManager` registers.
//! - [`audit`] — append-only [`AuditTrail`] of HCS operations.

extern crate alloc;

pub mod audit;
pub mod display;
pub mod error;
pub mod gpu;
pub mod mig;
pub mod mock;
pub mod network;
pub mod nvml;
pub mod service;
pub mod storage;
pub mod telemetry;
pub mod time_slice;

pub use audit::{AuditEntry, AuditEvent, AuditTrail, AUDIT_TRAIL_CAPACITY};
pub use display::{DisplaySurface, DisplaySurfaceId, DisplayTelemetry, DisplayTelemetrySnapshot};
pub use error::HalError;
pub use gpu::{GpuDeviceId, GpuProvider, GpuSlice, GpuSliceId, GpuVendor, SliceSpec, SlicingMode};
pub use mig::{MigComputeInstance, MigGpuInstance, MigProvider, MIG_MAX_INSTANCES};
pub use mock::{MockGpuProfile, MockGpuProvider};
pub use network::{NetworkEndpointId, NetworkTelemetry, NetworkTelemetrySnapshot};
pub use nvml::NvmlProvider;
pub use service::{HardwareCapabilityService, SliceOwner};
pub use storage::{StorageRegionId, StorageTelemetry, StorageTelemetrySnapshot};
pub use telemetry::{
    Bandwidth, GpuTelemetrySnapshot, Power, ThermalLevel, Utilization, THERMAL_CRITICAL,
    THERMAL_FAIR, THERMAL_NOMINAL, THERMAL_SERIOUS,
};
pub use time_slice::TimeSliceProvider;
