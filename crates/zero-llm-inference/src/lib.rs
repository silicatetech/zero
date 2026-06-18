// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::excessive_precision,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::doc_overindented_list_items,
    clippy::doc_lazy_continuation,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::manual_checked_ops
)]
//! Forward-pass operators for transformer inference.
//!
//! MP2.3a: RMSNorm.
//! MP2.3b: RoPE + RopeContext.
//! MP2.3c: Linear Q4_K (LinearScratch, output-major streaming).
//! MP2.4: Linear Q6_K + GQA Attention + KvCache integration.
//! MP2.5: SwiGLU MLP + full 28-layer forward-pass + LM head + first-token generation.
//! MP2.6 (future): ollama Cross-Validation Harness (per ADR-029 D9).
//!
//! # Design Constraints (ADR-029 D5/D7 + ADR-028 v5/v6 + ADR-030)
//!
//! - `no_std`, zero allocation — caller-allocated output buffers
//! - Pure scalar implementation in v0 (SIMD deferred)
//! - Hyperparameters read from ModelConfig at boot, never hardcoded
//! - `libm` is the ONLY external math dependency, ADR-028-approved
//! - KV-Cache: layer-major layout, caller-driven state, NO zero-init (ADR-030)
//! - Separate linear_q4k / linear_q6k operators (no runtime dispatch, Pillar 1)
//! - GGUF-FORTRAN-shape-immune: all dimensions from ModelConfig (ADR-029 D13)

#![no_std]

pub mod attention;
pub mod forward_pass;
pub mod kv_cache;
pub mod lm_head;
pub mod mla;
pub mod moe;
pub mod ops;

pub use attention::{
    gqa_attention_single_token, gqa_attention_single_token_dispatch, softmax, AttentionError,
};
pub use forward_pass::{
    embed_lookup, forward_single_token, forward_single_token_deepseek2, mlp_swiglu, AttnType,
    Deepseek2Error, Deepseek2LayerWeights, Deepseek2Scratch, FfnDownQuant, ForwardPassDispatch,
    ForwardPassError, LayerWeights, MlpType, N_LAYERS,
};
pub use kv_cache::{KvCache, KvCacheError, MhaKvCache, MlaKvCache};
pub use lm_head::{
    lm_head_argmax, lm_head_argmax_dynamic, LmHeadError, OutputQuant, VOCAB_SIZE_PADDED,
    VOCAB_SIZE_REAL,
};
pub use mla::{
    mha_attention_single_token, mla_attention_single_token, MhaWeights, MlaError, MlaWeights,
};
pub use moe::{expert_swiglu, moe_ffn, moe_route_f32, MoeRouteResult, MoeRoutingMode};
pub use ops::{
    linear_dispatch, linear_q4_0, linear_q4k, linear_q6k, linear_q8_0, rmsnorm, rope,
    LinearDispatchError, LinearScratch, RopeContext,
};
