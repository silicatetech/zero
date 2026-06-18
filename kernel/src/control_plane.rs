// SPDX-License-Identifier: AGPL-3.0-or-later
//! Zero control-plane state.
//!
//! The network rescue console can run before and during Stage 11 on
//! bare-metal hardware. This module gives that console a tiny lock-free
//! control plane: it can hold the BSP before Boot-LLM, request the start,
//! and report coarse lifecycle state without taking any inference-owned
//! locks. This is Zero-native control-plane state, not POSIX/SSH
//! session state.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

pub const STATUS_BOOTING: u32 = 0;
pub const STATUS_WAITING: u32 = 1;
pub const STATUS_REQUESTED: u32 = 2;
pub const STATUS_LOADING: u32 = 3;
pub const STATUS_RUNNING: u32 = 4;
pub const STATUS_COMPLETED: u32 = 5;
pub const STATUS_UNAVAILABLE: u32 = 6;

static BOOT_LLM_GATE_ACTIVE: AtomicBool = AtomicBool::new(false);
static BOOT_LLM_START_REQUESTED: AtomicBool = AtomicBool::new(false);
static BOOT_LLM_STATUS: AtomicU32 = AtomicU32::new(STATUS_BOOTING);

static LLM_PROFILE_VALID: AtomicBool = AtomicBool::new(false);
static LLM_PROFILE_RUN_ID: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_GENERATED_TOKENS: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_PREFILL_WALL_CYCLES: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_PREFILL_FORWARD_CYCLES: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_PREFILL_LM_HEAD_CYCLES: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_PREFILL_LOGIT_SCAN_CYCLES: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_GENERATION_WALL_CYCLES: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_GENERATION_COMPUTE_CYCLES: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_GENERATION_FORWARD_CYCLES: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_GENERATION_LM_HEAD_CYCLES: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_GENERATION_SAMPLE_CYCLES: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_GENERATION_LOGIT_SCAN_CYCLES: AtomicU64 = AtomicU64::new(0);
static LLM_PROFILE_GENERATION_RENDER_CYCLES: AtomicU64 = AtomicU64::new(0);
static MODEL_VIRT_ADDR: AtomicU64 = AtomicU64::new(0);
static MODEL_LEN_BYTES: AtomicU64 = AtomicU64::new(0);
static MODEL_SOURCE_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct LlmProfile {
    pub valid: bool,
    pub run_id: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub prefill_wall_cycles: u64,
    pub prefill_forward_cycles: u64,
    pub prefill_lm_head_cycles: u64,
    pub prefill_logit_scan_cycles: u64,
    pub generation_wall_cycles: u64,
    pub generation_compute_cycles: u64,
    pub generation_forward_cycles: u64,
    pub generation_lm_head_cycles: u64,
    pub generation_sample_cycles: u64,
    pub generation_logit_scan_cycles: u64,
    pub generation_render_cycles: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ModelRegion {
    pub present: bool,
    pub virt_addr: u64,
    pub len_bytes: u64,
    pub source_id: u32,
}

#[inline]
pub fn arm_boot_llm_gate() {
    BOOT_LLM_START_REQUESTED.store(false, Ordering::Release);
    BOOT_LLM_STATUS.store(STATUS_WAITING, Ordering::Release);
    BOOT_LLM_GATE_ACTIVE.store(true, Ordering::Release);
}

#[inline]
pub fn request_boot_llm_start() {
    BOOT_LLM_START_REQUESTED.store(true, Ordering::Release);
    if BOOT_LLM_GATE_ACTIVE.load(Ordering::Acquire) {
        BOOT_LLM_STATUS.store(STATUS_REQUESTED, Ordering::Release);
    }
}

#[inline(always)]
pub fn boot_llm_start_requested() -> bool {
    BOOT_LLM_START_REQUESTED.load(Ordering::Acquire)
}

#[inline(always)]
pub fn boot_llm_gate_active() -> bool {
    BOOT_LLM_GATE_ACTIVE.load(Ordering::Acquire)
}

#[inline]
pub fn mark_loading() {
    BOOT_LLM_GATE_ACTIVE.store(false, Ordering::Release);
    BOOT_LLM_STATUS.store(STATUS_LOADING, Ordering::Release);
}

#[inline]
pub fn mark_running() {
    BOOT_LLM_STATUS.store(STATUS_RUNNING, Ordering::Release);
}

#[inline]
pub fn mark_completed() {
    BOOT_LLM_GATE_ACTIVE.store(false, Ordering::Release);
    BOOT_LLM_STATUS.store(STATUS_COMPLETED, Ordering::Release);
}

#[inline]
pub fn mark_unavailable() {
    BOOT_LLM_GATE_ACTIVE.store(false, Ordering::Release);
    BOOT_LLM_STATUS.store(STATUS_UNAVAILABLE, Ordering::Release);
}

#[inline(always)]
pub fn status() -> u32 {
    BOOT_LLM_STATUS.load(Ordering::Acquire)
}

pub fn status_label(status: u32) -> &'static str {
    match status {
        STATUS_BOOTING => "booting",
        STATUS_WAITING => "waiting-for-start",
        STATUS_REQUESTED => "start-requested",
        STATUS_LOADING => "loading",
        STATUS_RUNNING => "running",
        STATUS_COMPLETED => "completed",
        STATUS_UNAVAILABLE => "unavailable",
        _ => "unknown",
    }
}

#[inline]
pub fn record_model_region(virt_addr: u64, len_bytes: u64, source: &str) {
    let source_id = match source {
        "ramdisk" => 1,
        "direct-memory" => 2,
        "nvme" => 3,
        _ => 4,
    };
    MODEL_VIRT_ADDR.store(virt_addr, Ordering::Release);
    MODEL_LEN_BYTES.store(len_bytes, Ordering::Release);
    MODEL_SOURCE_ID.store(source_id, Ordering::Release);
}

#[inline]
pub fn model_source_label(source_id: u32) -> &'static str {
    match source_id {
        1 => "ramdisk",
        2 => "direct-memory",
        3 => "nvme",
        4 => "other",
        _ => "absent",
    }
}

#[inline]
pub fn model_region_snapshot() -> ModelRegion {
    let virt_addr = MODEL_VIRT_ADDR.load(Ordering::Acquire);
    let len_bytes = MODEL_LEN_BYTES.load(Ordering::Acquire);
    let source_id = MODEL_SOURCE_ID.load(Ordering::Acquire);
    ModelRegion {
        present: virt_addr != 0 && len_bytes != 0,
        virt_addr,
        len_bytes,
        source_id,
    }
}

#[inline]
pub fn reset_llm_profile() {
    LLM_PROFILE_VALID.store(false, Ordering::Release);
    LLM_PROFILE_RUN_ID.fetch_add(1, Ordering::AcqRel);
    LLM_PROFILE_PROMPT_TOKENS.store(0, Ordering::Release);
    LLM_PROFILE_GENERATED_TOKENS.store(0, Ordering::Release);
    LLM_PROFILE_PREFILL_WALL_CYCLES.store(0, Ordering::Release);
    LLM_PROFILE_PREFILL_FORWARD_CYCLES.store(0, Ordering::Release);
    LLM_PROFILE_PREFILL_LM_HEAD_CYCLES.store(0, Ordering::Release);
    LLM_PROFILE_PREFILL_LOGIT_SCAN_CYCLES.store(0, Ordering::Release);
    LLM_PROFILE_GENERATION_WALL_CYCLES.store(0, Ordering::Release);
    LLM_PROFILE_GENERATION_COMPUTE_CYCLES.store(0, Ordering::Release);
    LLM_PROFILE_GENERATION_FORWARD_CYCLES.store(0, Ordering::Release);
    LLM_PROFILE_GENERATION_LM_HEAD_CYCLES.store(0, Ordering::Release);
    LLM_PROFILE_GENERATION_SAMPLE_CYCLES.store(0, Ordering::Release);
    LLM_PROFILE_GENERATION_LOGIT_SCAN_CYCLES.store(0, Ordering::Release);
    LLM_PROFILE_GENERATION_RENDER_CYCLES.store(0, Ordering::Release);
}

#[inline]
pub fn record_llm_profile(profile: LlmProfile) {
    LLM_PROFILE_PROMPT_TOKENS.store(profile.prompt_tokens, Ordering::Release);
    LLM_PROFILE_GENERATED_TOKENS.store(profile.generated_tokens, Ordering::Release);
    LLM_PROFILE_PREFILL_WALL_CYCLES.store(profile.prefill_wall_cycles, Ordering::Release);
    LLM_PROFILE_PREFILL_FORWARD_CYCLES.store(profile.prefill_forward_cycles, Ordering::Release);
    LLM_PROFILE_PREFILL_LM_HEAD_CYCLES.store(profile.prefill_lm_head_cycles, Ordering::Release);
    LLM_PROFILE_PREFILL_LOGIT_SCAN_CYCLES
        .store(profile.prefill_logit_scan_cycles, Ordering::Release);
    LLM_PROFILE_GENERATION_WALL_CYCLES.store(profile.generation_wall_cycles, Ordering::Release);
    LLM_PROFILE_GENERATION_COMPUTE_CYCLES
        .store(profile.generation_compute_cycles, Ordering::Release);
    LLM_PROFILE_GENERATION_FORWARD_CYCLES
        .store(profile.generation_forward_cycles, Ordering::Release);
    LLM_PROFILE_GENERATION_LM_HEAD_CYCLES
        .store(profile.generation_lm_head_cycles, Ordering::Release);
    LLM_PROFILE_GENERATION_SAMPLE_CYCLES.store(profile.generation_sample_cycles, Ordering::Release);
    LLM_PROFILE_GENERATION_LOGIT_SCAN_CYCLES
        .store(profile.generation_logit_scan_cycles, Ordering::Release);
    LLM_PROFILE_GENERATION_RENDER_CYCLES.store(profile.generation_render_cycles, Ordering::Release);
    LLM_PROFILE_VALID.store(true, Ordering::Release);
}

#[inline]
pub fn llm_profile_snapshot() -> LlmProfile {
    LlmProfile {
        valid: LLM_PROFILE_VALID.load(Ordering::Acquire),
        run_id: LLM_PROFILE_RUN_ID.load(Ordering::Acquire),
        prompt_tokens: LLM_PROFILE_PROMPT_TOKENS.load(Ordering::Acquire),
        generated_tokens: LLM_PROFILE_GENERATED_TOKENS.load(Ordering::Acquire),
        prefill_wall_cycles: LLM_PROFILE_PREFILL_WALL_CYCLES.load(Ordering::Acquire),
        prefill_forward_cycles: LLM_PROFILE_PREFILL_FORWARD_CYCLES.load(Ordering::Acquire),
        prefill_lm_head_cycles: LLM_PROFILE_PREFILL_LM_HEAD_CYCLES.load(Ordering::Acquire),
        prefill_logit_scan_cycles: LLM_PROFILE_PREFILL_LOGIT_SCAN_CYCLES.load(Ordering::Acquire),
        generation_wall_cycles: LLM_PROFILE_GENERATION_WALL_CYCLES.load(Ordering::Acquire),
        generation_compute_cycles: LLM_PROFILE_GENERATION_COMPUTE_CYCLES.load(Ordering::Acquire),
        generation_forward_cycles: LLM_PROFILE_GENERATION_FORWARD_CYCLES.load(Ordering::Acquire),
        generation_lm_head_cycles: LLM_PROFILE_GENERATION_LM_HEAD_CYCLES.load(Ordering::Acquire),
        generation_sample_cycles: LLM_PROFILE_GENERATION_SAMPLE_CYCLES.load(Ordering::Acquire),
        generation_logit_scan_cycles: LLM_PROFILE_GENERATION_LOGIT_SCAN_CYCLES
            .load(Ordering::Acquire),
        generation_render_cycles: LLM_PROFILE_GENERATION_RENDER_CYCLES.load(Ordering::Acquire),
    }
}
