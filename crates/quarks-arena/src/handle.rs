// SPDX-License-Identifier: AGPL-3.0-or-later
//! Handle table — typed reference into arena memory.
//!
//! Per ADR-027, a Handle is a 64-bit unsigned identifier serving as
//! an index into a Handle Table. The table maps each valid Handle to
//! a typed reference into arena memory.
//!
//! Architectural properties (per ADR-027):
//! - Handle 0 is reserved as null sentinel; allocation starts at 1
//! - O(1) allocation (push to Vec)
//! - O(1) atomic revoke (vec[i] = None)
//! - O(1) dereference (vec[i].as_ref())
//! - Type information stored alongside data (V3 "typed reference")
//! - Append-only in Stage 10 (slot recycling deferred to Stage 11+)
//!
//! See ADR-027 for full architectural rationale and V3-Z.256
//! atomic-revoke mandate.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

/// Type tag carried by a HandleEntry, mirroring `ValueType` from
/// `quarks-validator`. Duplicated locally to avoid circular
/// dependency (validator already depends on arena via type system).
///
/// MP5 supports only Bytes as the underlying handle data type
/// (matches the `register` instruction signature). I64 and Handle
/// types as handle-targets are deferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleType {
    Bytes,
}

/// A single entry in the Handle Table.
///
/// Stage 10 stores the offset and length into the arena's data region.
/// Future stages may add capability tokens, sandbox-id, or promotion
/// markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandleEntry {
    pub value_type: HandleType,
    pub offset: usize,
    pub length: usize,
}

/// Errors returned by handle table operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandleError {
    /// Handle 0 was passed; null sentinel is invalid (per ADR-027 +
    /// Phase 2 type-checker).
    NullHandle,
    /// Handle ID exceeds table length.
    OutOfBounds,
    /// Slot at this handle ID has been revoked.
    Revoked,
    /// Handle table capacity exhausted (not realistic in Stage 10).
    Exhausted,
}

impl fmt::Display for HandleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HandleError::NullHandle => write!(f, "handle 0 is null sentinel"),
            HandleError::OutOfBounds => write!(f, "handle exceeds table size"),
            HandleError::Revoked => write!(f, "handle has been revoked"),
            HandleError::Exhausted => write!(f, "handle table exhausted"),
        }
    }
}

/// Handle Table: append-only `Vec<Option<HandleEntry>>`.
///
/// Slot 0 is permanently `None` (null sentinel reservation per ADR-027).
/// Allocation starts at index 1.
pub struct HandleTable {
    entries: Vec<Option<HandleEntry>>,
}

impl HandleTable {
    /// Create a new HandleTable with index 0 pre-occupied as null sentinel.
    pub fn new() -> Self {
        // Reserve index 0 as null sentinel — never returned by allocate().
        let entries = vec![None];
        HandleTable { entries }
    }

    /// Allocate a new Handle for the given entry. Returns the Handle ID (>= 1).
    pub fn allocate(&mut self, entry: HandleEntry) -> Result<u64, HandleError> {
        let idx = self.entries.len();
        if idx >= u64::MAX as usize {
            return Err(HandleError::Exhausted);
        }
        self.entries.push(Some(entry));
        Ok(idx as u64)
    }

    /// Look up a Handle. Returns the entry or an error.
    pub fn deref(&self, handle: u64) -> Result<&HandleEntry, HandleError> {
        if handle == 0 {
            return Err(HandleError::NullHandle);
        }
        let idx = handle as usize;
        if idx >= self.entries.len() {
            return Err(HandleError::OutOfBounds);
        }
        match &self.entries[idx] {
            Some(entry) => Ok(entry),
            None => Err(HandleError::Revoked),
        }
    }

    /// Atomically revoke a Handle. Returns Ok if revoked, error otherwise.
    pub fn revoke(&mut self, handle: u64) -> Result<(), HandleError> {
        if handle == 0 {
            return Err(HandleError::NullHandle);
        }
        let idx = handle as usize;
        if idx >= self.entries.len() {
            return Err(HandleError::OutOfBounds);
        }
        match self.entries[idx] {
            Some(_) => {
                self.entries[idx] = None;
                Ok(())
            }
            None => Err(HandleError::Revoked),
        }
    }

    /// Total table length (including reserved index 0 and revoked slots).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the table contains no allocated entries
    /// (only the null sentinel at index 0).
    pub fn is_empty(&self) -> bool {
        self.entries.len() <= 1
    }

    /// Reset the table — clears all entries, restores index 0 sentinel.
    pub fn reset(&mut self) {
        self.entries.clear();
        self.entries.push(None);
    }
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_bytes(offset: usize, length: usize) -> HandleEntry {
        HandleEntry {
            value_type: HandleType::Bytes,
            offset,
            length,
        }
    }

    #[test]
    fn fresh_table_has_null_sentinel_only() {
        let table = HandleTable::new();
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn allocate_returns_handle_one_first() {
        let mut table = HandleTable::new();
        let h = table.allocate(entry_bytes(0, 10)).unwrap();
        assert_eq!(h, 1);
    }

    #[test]
    fn allocate_monotonic() {
        let mut table = HandleTable::new();
        let h1 = table.allocate(entry_bytes(0, 10)).unwrap();
        let h2 = table.allocate(entry_bytes(10, 20)).unwrap();
        let h3 = table.allocate(entry_bytes(30, 5)).unwrap();
        assert_eq!(h1, 1);
        assert_eq!(h2, 2);
        assert_eq!(h3, 3);
    }

    #[test]
    fn deref_zero_is_null_handle() {
        let table = HandleTable::new();
        assert_eq!(table.deref(0), Err(HandleError::NullHandle));
    }

    #[test]
    fn deref_out_of_bounds() {
        let table = HandleTable::new();
        assert_eq!(table.deref(99), Err(HandleError::OutOfBounds));
    }

    #[test]
    fn deref_valid_handle_returns_entry() {
        let mut table = HandleTable::new();
        let h = table.allocate(entry_bytes(42, 13)).unwrap();
        let got = table.deref(h).unwrap();
        assert_eq!(got.value_type, HandleType::Bytes);
        assert_eq!(got.offset, 42);
        assert_eq!(got.length, 13);
    }

    #[test]
    fn revoke_zero_is_null_handle() {
        let mut table = HandleTable::new();
        assert_eq!(table.revoke(0), Err(HandleError::NullHandle));
    }

    #[test]
    fn revoke_out_of_bounds() {
        let mut table = HandleTable::new();
        assert_eq!(table.revoke(99), Err(HandleError::OutOfBounds));
    }

    #[test]
    fn revoke_then_deref_returns_revoked() {
        let mut table = HandleTable::new();
        let h = table.allocate(entry_bytes(0, 10)).unwrap();
        table.revoke(h).unwrap();
        assert_eq!(table.deref(h), Err(HandleError::Revoked));
    }

    #[test]
    fn double_revoke_returns_revoked() {
        let mut table = HandleTable::new();
        let h = table.allocate(entry_bytes(0, 10)).unwrap();
        table.revoke(h).unwrap();
        assert_eq!(table.revoke(h), Err(HandleError::Revoked));
    }

    #[test]
    fn revoke_does_not_recycle_in_stage_10() {
        let mut table = HandleTable::new();
        let h1 = table.allocate(entry_bytes(0, 10)).unwrap();
        table.revoke(h1).unwrap();
        let h2 = table.allocate(entry_bytes(10, 20)).unwrap();
        // Stage 10 is append-only — h2 is NOT recycled from h1's slot
        assert_ne!(h1, h2);
        assert_eq!(h2, 2);
    }

    #[test]
    fn reset_restores_null_sentinel() {
        let mut table = HandleTable::new();
        table.allocate(entry_bytes(0, 10)).unwrap();
        table.allocate(entry_bytes(10, 20)).unwrap();
        table.reset();
        assert_eq!(table.len(), 1);
        let h = table.allocate(entry_bytes(0, 5)).unwrap();
        assert_eq!(h, 1);
    }
}
