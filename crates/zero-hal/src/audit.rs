// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only audit trail of HCS operations.
//!
//! Per `docs/discovery/stage-12-completion-plan.md` §D point 6: every
//! slice-minting and -release is recorded for forensic
//! nachvollziehbarkeit. Mirrors the
//! `zero-sandbox::PolicyInvocation`-log pattern: bounded-capacity
//! drop-oldest ring, never panics, no heap re-allocation in the hot
//! path once warm.

use alloc::vec::Vec;

use crate::gpu::{GpuDeviceId, GpuSliceId, SliceSpec, SlicingMode};

/// Capacity of the in-memory audit ring.
///
/// Mirrors `POLICY_LOG_CAPACITY` from `zero-sandbox`. Bounded growth
/// is the load-bearing property; persistent forensic snapshots are
/// out-of-scope for Stage 12 (tracked alongside ADR-019 §"Future
/// Work").
pub const AUDIT_TRAIL_CAPACITY: usize = 64;

/// What kind of HCS operation produced this audit entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditEvent {
    /// A new GPU slice was minted on the named device.
    GpuSliceCreated {
        device: GpuDeviceId,
        slice: GpuSliceId,
        spec: SliceSpec,
        mode: SlicingMode,
    },
    /// A GPU slice was destroyed.
    GpuSliceDestroyed { slice: GpuSliceId },
    /// A slice-minting request was rejected before the slice id was
    /// allocated (out-of-budget, malformed spec, etc.).
    GpuSliceRequestRejected {
        device: GpuDeviceId,
        spec: SliceSpec,
        reason: &'static str,
    },
    /// Telemetry was polled (device-scope).
    GpuDeviceTelemetryPolled { device: GpuDeviceId, sample: u64 },
    /// Telemetry was polled (slice-scope).
    GpuSliceTelemetryPolled { slice: GpuSliceId, sample: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEntry {
    /// Monotonic sequence number assigned by the trail at append time.
    /// Zero-based; deterministic.
    pub sequence: u64,
    /// Sandbox that originated the request, when known. `None` for
    /// kernel-boot-time mints (Display Agent surface, etc.).
    pub owner: Option<u64>,
    pub event: AuditEvent,
}

/// Append-only ring of [`AuditEntry`].
#[derive(Debug, Clone, Default)]
pub struct AuditTrail {
    next_sequence: u64,
    entries: Vec<AuditEntry>,
}

impl AuditTrail {
    pub fn new() -> Self {
        Self {
            next_sequence: 0,
            entries: Vec::with_capacity(AUDIT_TRAIL_CAPACITY),
        }
    }

    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Append an event. Drop-oldest once the capacity is reached.
    pub fn record(&mut self, owner: Option<u64>, event: AuditEvent) -> u64 {
        let sequence = self.next_sequence;
        // Saturate on the (astronomical) sequence overflow rather than
        // wrap silently. The forensic story prefers a stuck counter to
        // duplicate ids.
        self.next_sequence = self.next_sequence.saturating_add(1);
        if self.entries.len() == AUDIT_TRAIL_CAPACITY {
            self.entries.remove(0);
        }
        self.entries.push(AuditEntry {
            sequence,
            owner,
            event,
        });
        sequence
    }

    /// Test-only helper: clear the ring. Production code should rely
    /// on the bounded growth guarantee.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_event() -> AuditEvent {
        AuditEvent::GpuSliceCreated {
            device: GpuDeviceId(0),
            slice: GpuSliceId(1),
            spec: SliceSpec::new(50, 1024, 0),
            mode: SlicingMode::HardwareMig,
        }
    }

    #[test]
    fn records_are_sequenced_monotonically() {
        let mut t = AuditTrail::new();
        let s0 = t.record(Some(7), mk_event());
        let s1 = t.record(Some(7), mk_event());
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn ring_drops_oldest_when_capacity_reached() {
        let mut t = AuditTrail::new();
        for _ in 0..(AUDIT_TRAIL_CAPACITY + 5) {
            t.record(None, mk_event());
        }
        assert_eq!(t.len(), AUDIT_TRAIL_CAPACITY);
        // Oldest sequence number should now be 5 (we dropped 0..5).
        assert_eq!(t.entries().first().unwrap().sequence, 5);
    }

    #[test]
    fn owner_is_recorded() {
        let mut t = AuditTrail::new();
        t.record(Some(42), mk_event());
        t.record(None, mk_event());
        assert_eq!(t.entries()[0].owner, Some(42));
        assert_eq!(t.entries()[1].owner, None);
    }
}
