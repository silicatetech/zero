// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR-029 Phase 1+2 — x86_64 HAL math acceleration module.
//!
//! AVX-512 intrinsics for performance-critical inference operations.
//! Mirror of `arch::aarch64::math` (NEON variant). Per Pillar 7
//! (V3.1 Z.264-269): generic code stays scalar; only HAL provides
//! platform-specific accelerated implementations.
//!
//! Per V3.1 Pillar 8 (Z.275-285): Rust temporary; AVX-512 intrinsics
//! may stay Rust through Stage 14 Platform Portability.
//!
//! Modules:
//! - `linear`: AVX-512-accelerated linear projections (Q4_K + Q6_K fused
//!   dequant + dot-product).

#[cfg(feature = "avx512-acceleration")]
pub mod linear;

/// ADR-029 Patch v8.4 — AVX-512 vectorised SiLU + element-wise multiply
/// for the SwiGLU MLP. Replaces the scalar `libm::expf` inner loop.
/// ~1 ULP feature-mode drift vs sacred scalar SiLU; Token-ID 25
/// preserved per ADR-029 v3 Two-Anchor.
#[cfg(feature = "avx512-acceleration")]
pub mod activation;

/// ADR-029 Patch v8.4 — AVX-512 vectorised sin/cos polynomial + RoPE
/// LUT. Computes 64 (cos, sin) pairs once per token (vs sacred path
/// that recomputes them per head × 24 heads × per layer), then
/// applies the rotation in 16-lane AVX-512.
#[cfg(feature = "avx512-acceleration")]
pub mod trig;

/// ADR-029 v8 candidate — AVX-512 VNNI (VPDPBUSD) Q4_K × Q8_K integer
/// dot product. Default OFF; numerical drift vs FP32 path must be
/// empirically verified against the β-anchor on Cherry hardware before
/// any production wiring.
#[cfg(feature = "vnni-acceleration")]
pub mod vnni;
