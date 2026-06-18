// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-E2: aarch64 HAL math acceleration module.
//!
//! NEON intrinsics for performance-critical inference operations.
//! Per E2-Q2 ratification: NEON exclusive in this module hierarchy.
//! Per Pillar 7 (V3.1 Z.264-269): generic code stays scalar; only
//! HAL provides platform-specific accelerated implementations.
//!
//! Per V3.1 Pillar 8 (Z.275-285): Rust temporary; NEON intrinsics
//! may stay Rust through Stage 14 Platform Portability.
//!
//! Modules:
//! - linear: NEON-accelerated linear projections (Q4_K + Q6_K fused dequant+dot)

#[cfg(feature = "neon-acceleration")]
pub mod linear;
