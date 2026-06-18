// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error type for arena operations.

use core::fmt;

/// Errors returned by arena allocation operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    /// Allocation request exceeded the arena's capacity (or, for
    /// chunked arenas, the maximum total capacity).
    OutOfMemory {
        /// Bytes requested.
        requested: usize,
        /// Bytes available at time of request.
        available: usize,
    },
    /// Alignment requirement was not a power of two.
    AlignmentInvalid {
        /// The invalid alignment value.
        alignment: usize,
    },
    /// Internal arithmetic overflow in offset/size calculation.
    /// Indicates pathological input.
    SizeOverflow,
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArenaError::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "arena out of memory: requested {} bytes, {} available",
                    requested, available
                )
            }
            ArenaError::AlignmentInvalid { alignment } => {
                write!(
                    f,
                    "arena alignment invalid: {} is not a power of two",
                    alignment
                )
            }
            ArenaError::SizeOverflow => {
                write!(f, "arena size calculation overflowed")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ArenaError {}
