// SPDX-License-Identifier: AGPL-3.0-or-later
//! HAL error taxonomy.
//!
//! Per the determinism contract (`lib.rs` doc-comment), every fallible
//! HAL operation must return one of these variants — never panic, never
//! `unreachable!()`. The `&'static str` payloads identify the call
//! site for forensic readability and intentionally do not allocate.

/// Errors surfaced by the Hardware Capability Service HAL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HalError {
    /// The provider does not implement this capability on this
    /// platform. Per §4.1 of the constraints doc, `Option<…>`-shaped
    /// returns encode "degraded mode" explicitly; methods that must
    /// return a concrete value use this variant instead of a sentinel.
    Unsupported,
    /// The requested device id does not exist in this provider.
    NoSuchDevice,
    /// The requested slice id is not live (never minted, or already
    /// destroyed).
    NoSuchSlice,
    /// The provider has no remaining slice budget on the named device
    /// (MIG ceiling, software time-slice queue full, etc.).
    OutOfSlices,
    /// The [`crate::gpu::SliceSpec`] is malformed or violates a
    /// provider invariant. The payload identifies which constraint.
    InvalidSpec(&'static str),
    /// Telemetry sampling failed (provider stub returns "unavailable",
    /// underlying counter has not been initialised, etc.).
    TelemetryUnavailable,
    /// `HardwareCapabilityService::register_*` called twice for the
    /// same provider slot.
    AlreadyConfigured,
    /// An internal invariant did not hold. The payload identifies
    /// the call site for the same reason as
    /// `SandboxError::Internal` — a refactor regression should surface
    /// a structured variant, not a panic.
    InternalInvariant(&'static str),
}
