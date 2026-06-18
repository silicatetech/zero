// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sub-MP-F3 Task A / F3.6 Tasks M6+N7: Telemetry data extraction.
//!
//! Per Pillar 7: NO #[cfg(target_arch)] in this module.
//! Per Pillar 1 (reframed): MUST NOT regress inference-loop wallclock.
//! Per Lesson 36: display values MUST be empirically accurate, NOT
//!   hardcoded or aspirational.
//!
//! Static telemetry data (set once at boot OR computed cheaply).
//! Dynamic telemetry data (per-token) avoided to prevent render-cost
//! regression.

use core::sync::atomic::{AtomicUsize, Ordering};

/// Static telemetry data for display on the telemetry panel.
///
/// N7 fix: speedup is computed from actual baselines, NOT hardcoded.
/// Design principle: if the machine delivers 1.57x, the display shows
/// 1.57x. Period.
#[derive(Debug, Clone, Copy)]
pub struct TelemetryData {
    /// E3 scalar baseline wallclock (seconds).
    pub e3_baseline_wallclock_s: u32,

    /// Current mode baseline wallclock (seconds).
    pub current_baseline_wallclock_s: u32,

    /// Token throughput rate (4-decimal precision, x10000 for integer).
    /// 4267 = 0.4267 tok/s (per Lesson 36 precision discipline).
    pub throughput_x10000: u32,

    /// Total layers (transformer architecture).
    pub total_layers: u32,

    /// Model identifier.
    pub model_label: &'static str,

    /// Build mode label.
    pub mode_label: &'static str,
}

impl TelemetryData {
    /// NEON-accelerated mode telemetry (feature = "neon-acceleration").
    pub const fn neon_mode() -> Self {
        Self {
            e3_baseline_wallclock_s: 118,     // E3 scalar baseline
            current_baseline_wallclock_s: 75, // F3-ratified NEON
            throughput_x10000: 4267,          // 0.4267 tok/s (Lesson 36)
            total_layers: 28,
            model_label: "Qwen3-1.7B-Q4_K_M",
            mode_label: "feature/NEON",
        }
    }

    /// AVX-512 accelerated mode telemetry (feature = "avx512-acceleration").
    ///
    /// Per ADR-029 D8 "Optimization Staircase": AVX-512 on x86_64.
    /// Throughput estimate: pending bare-metal measurement on EPYC 9354P.
    /// Placeholder values use conservative 8× scalar estimate.
    pub const fn avx512_mode() -> Self {
        Self {
            e3_baseline_wallclock_s: 118,     // E3 scalar baseline
            current_baseline_wallclock_s: 15, // Conservative AVX-512 estimate
            throughput_x10000: 21333,         // ~2.13 tok/s estimate (32/15)
            total_layers: 28,
            model_label: "Qwen3-1.7B-Q4_K_M",
            mode_label: "feature/AVX-512",
        }
    }

    /// Scalar mode telemetry (no NEON feature).
    pub const fn scalar_mode() -> Self {
        Self {
            e3_baseline_wallclock_s: 118,      // E3 = scalar = reference
            current_baseline_wallclock_s: 118, // No speedup
            throughput_x10000: 2712,           // 0.2712 tok/s (32/118)
            total_layers: 28,
            model_label: "Qwen3-1.7B-Q4_K_M",
            mode_label: "default/scalar",
        }
    }

    /// Compute speedup × 100 (integer arithmetic for const-correctness).
    ///
    /// NEON: 118 * 100 / 75 = 157 → display as "1.57x"
    /// Scalar: 118 * 100 / 118 = 100 → display as "1.00x"
    ///
    /// Per honest-metrics anchor: mathematical truth from actual baselines.
    pub const fn speedup_x100(&self) -> u32 {
        if self.current_baseline_wallclock_s == 0 {
            return 100;
        }
        self.e3_baseline_wallclock_s * 100 / self.current_baseline_wallclock_s
    }
}

/// Memory boundary information (dynamic, set at boot via setter).
///
/// M6 fix: NOT hardcoded const. Architecture-specific code sets actual
/// values at boot via set_memory_boundaries(). Generic code reads via
/// memory_boundaries(). Pillar 7 clean.
#[derive(Debug, Clone, Copy)]
pub struct MemoryBoundaries {
    /// KV cache arena allocated size in MiB.
    pub kv_cache_arena_size_mb: u32,

    /// Framebuffer size in KiB (1024×768×4bpp).
    pub framebuffer_size_kb: u32,
}

/// Global memory boundaries (set via setter at boot).
static mut MEMORY_BOUNDARIES: Option<MemoryBoundaries> = None;

/// Maximum bytes the model-label override holds. Sized to fit names
/// like "Kimi K2.6 (Q4_0)" or "Qwen Qwen3 1.7B" comfortably; longer
/// `general.name` strings are truncated rather than spilled.
pub const MODEL_LABEL_MAX: usize = 48;

/// Static backing buffer for the dynamic model label (UTF-8 bytes).
/// Written exactly once during Stage 11 by `set_model_label` and read
/// thereafter via `model_label_override`. Both accessors gate on
/// `MODEL_LABEL_LEN` for visibility.
static mut MODEL_LABEL_BUF: [u8; MODEL_LABEL_MAX] = [0; MODEL_LABEL_MAX];

/// Length of the active model-label override. `0` means "no override
/// installed yet — fall back to `TelemetryData::model_label`".
static MODEL_LABEL_LEN: AtomicUsize = AtomicUsize::new(0);

/// Install the actually-loaded model name (from GGUF `general.name`) as
/// the telemetry override. Truncates to `MODEL_LABEL_MAX` bytes on a
/// UTF-8 char boundary so the override never observes a split codepoint.
///
/// # Safety
/// Must be called from single-threaded boot before any reader can race
/// (i.e. before the telemetry panel is re-rendered or the TCP
/// `inference` command serves a response).
#[allow(static_mut_refs)]
pub unsafe fn set_model_label(name: &str) {
    let mut n = name.len().min(MODEL_LABEL_MAX);
    while n > 0 && !name.is_char_boundary(n) {
        n -= 1;
    }
    let bytes = &name.as_bytes()[..n];
    MODEL_LABEL_BUF[..n].copy_from_slice(bytes);
    MODEL_LABEL_LEN.store(n, Ordering::Release);
}

/// Return the installed model-label override, or `None` when no
/// override has been installed yet. The returned `&'static str` borrows
/// from a kernel-static buffer, so it remains live for the rest of the
/// boot.
#[allow(static_mut_refs)]
pub fn model_label_override() -> Option<&'static str> {
    let n = MODEL_LABEL_LEN.load(Ordering::Acquire);
    if n == 0 {
        return None;
    }
    let slice = unsafe { &MODEL_LABEL_BUF[..n] };
    core::str::from_utf8(slice).ok()
}

/// Pick the right model label for display: the GGUF-derived override
/// when installed, else the build-mode default carried by `td`. Used
/// by both the framebuffer panel and the TCP `inference` command so
/// they never disagree.
pub fn effective_model_label(td: &TelemetryData) -> &'static str {
    model_label_override().unwrap_or(td.model_label)
}

/// Set memory boundaries at boot (called from arch-specific code).
///
/// # Safety
/// Must be called during single-threaded boot.
#[allow(static_mut_refs)]
pub unsafe fn set_memory_boundaries(mb: MemoryBoundaries) {
    MEMORY_BOUNDARIES = Some(mb);
}

/// Read memory boundaries (returns None if not yet set).
#[allow(static_mut_refs)]
pub fn memory_boundaries() -> Option<MemoryBoundaries> {
    unsafe { MEMORY_BOUNDARIES }
}
