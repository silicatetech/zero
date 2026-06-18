// SPDX-License-Identifier: AGPL-3.0-or-later
//! KV-Cache for transformer inference per ADR-030.
//!
//! Layer-major + token-sparse storage layout.
//! Caller-driven state management (no internal token counter).
//! Raw-pointer + stride for model-agnostic implementation.
//! Single base_ptr API: caller passes ONE pointer; KvCache computes
//! k_base + v_base internally.
//! Infallible `new()`: caller handles arena OOM via `ArenaError`
//! BEFORE KvCache construction.

/// KV-Cache wrapper for a single inference session.
///
/// Per ADR-030: holds buffer pointers + dimensional metadata only.
/// Caller (forward-pass loop) tracks token count explicitly.
///
/// ## Layout (layer-major, single contiguous arena region)
///
/// ```text
/// [Layer 0 K block][Layer 0 V block][Layer 1 K block][Layer 1 V block]...
/// ```
///
/// Each block size = `max_tokens × num_kv_heads × head_dim` f32 elements.
///
/// ## Stride math
///
/// - `block_size = max_tokens × num_kv_heads × head_dim`
/// - `token_stride = num_kv_heads × head_dim`
/// - `layer_stride = 2 × block_size` (skips BOTH K and V per layer)
/// - `k_base = arena_start`
/// - `v_base = arena_start + block_size`
pub struct KvCache {
    /// Base pointer for K storage. K\[layer\]\[token\] starts at
    /// `k_base + (layer × layer_stride) + (token × token_stride)`.
    k_base: *mut f32,
    /// Base pointer for V storage. V\[layer\]\[token\] starts at
    /// `v_base + (layer × layer_stride) + (token × token_stride)`.
    /// Computed internally as `k_base + block_size`.
    v_base: *mut f32,
    /// Stride between consecutive layer blocks (in f32 elements).
    /// `= 2 × max_tokens × token_stride` (skips both K block AND V block).
    layer_stride: usize,
    /// Stride between consecutive tokens within a layer (in f32 elements).
    /// `= num_kv_heads × head_dim`
    token_stride: usize,
    /// Maximum number of tokens this cache can hold.
    pub max_tokens: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV attention heads.
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
}

/// SAFETY: KvCache contains raw pointers but exclusively owns the
/// underlying arena region (no aliasing). Access is gated by &mut self
/// for writes and &self for reads. Transferring between threads is safe
/// under single-threaded inference (Stage 11).
unsafe impl Send for KvCache {}

/// Errors returned by KvCache operations.
///
/// Per ADR-030: KvCache::new() is infallible. These errors cover
/// misuse of the store/get API, NOT allocation failure (which is
/// handled at the ArenaError level before KvCache construction).
#[derive(Debug, Clone, Copy)]
pub enum KvCacheError {
    /// `store_kv()` called with `token_idx >= max_tokens`.
    TokenIndexOutOfRange { token_idx: usize, max_tokens: usize },
    /// `layer >= num_layers`.
    LayerOutOfRange { layer: usize, num_layers: usize },
    /// `k.len()` or `v.len() != num_kv_heads × head_dim`.
    InvalidSliceLength { expected: usize, actual: usize },
    /// `get_*_slice()` called with `token_count > max_tokens`.
    TokenCountExceedsMax {
        token_count: usize,
        max_tokens: usize,
    },
}

impl KvCache {
    /// Initialize KvCache from a single `base_ptr` to pre-allocated arena memory.
    ///
    /// **Infallible:** caller has already obtained `base_ptr` from
    /// `KvCacheArenaInner::alloc_f32_slice(total_count)` with appropriate
    /// OOM handling. Once the slice exists, construction is pure pointer
    /// arithmetic.
    ///
    /// `total_count` expected = `2 × num_layers × max_tokens × num_kv_heads × head_dim`.
    ///
    /// For Qwen3-1.7B (28 layers, 8 kv_heads, 128 head_dim, 2340 max_tokens):
    ///   total_count = 2 × 28 × 2340 × 8 × 128 = 134,184,960 f32 elements
    ///   total_bytes = 536,739,840 bytes ≈ 511.87 MiB
    ///
    /// # Safety
    ///
    /// `base_ptr` must point to valid arena memory of at least the required
    /// total size. Caller obtained this from `KvCacheArenaInner`.
    pub fn new(
        base_ptr: *mut f32,
        max_tokens: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let token_stride = num_kv_heads * head_dim;
        let block_size = max_tokens * token_stride; // size of ONE K or V block
        let layer_stride = 2 * block_size; // skip both K and V per layer

        let k_base = base_ptr;
        // SAFETY: base_ptr points to arena memory of size
        // 2 × num_layers × block_size. block_size offset stays within bounds.
        let v_base = unsafe { base_ptr.add(block_size) };

        Self {
            k_base,
            v_base,
            layer_stride,
            token_stride,
            max_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
        }
    }

    /// Store K and V vectors for a specific layer at a specific token position.
    ///
    /// `k.len()` and `v.len()` must equal `num_kv_heads × head_dim`.
    /// Caller manages `token_idx` (no internal counter increment).
    ///
    /// Offset calculation:
    /// - `K[layer][token] = k_base + (layer × layer_stride) + (token × token_stride)`
    /// - `V[layer][token] = v_base + (layer × layer_stride) + (token × token_stride)`
    ///
    /// Note: `layer_stride = 2 × block_size`, so consecutive layers are
    /// correctly separated (L1_K starts after L0_V ends, not after L0_K).
    pub fn store_kv(
        &mut self,
        layer: usize,
        token_idx: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), KvCacheError> {
        if layer >= self.num_layers {
            return Err(KvCacheError::LayerOutOfRange {
                layer,
                num_layers: self.num_layers,
            });
        }
        if token_idx >= self.max_tokens {
            return Err(KvCacheError::TokenIndexOutOfRange {
                token_idx,
                max_tokens: self.max_tokens,
            });
        }
        let expected = self.token_stride;
        if k.len() != expected {
            return Err(KvCacheError::InvalidSliceLength {
                expected,
                actual: k.len(),
            });
        }
        if v.len() != expected {
            return Err(KvCacheError::InvalidSliceLength {
                expected,
                actual: v.len(),
            });
        }

        let offset = layer * self.layer_stride + token_idx * self.token_stride;
        unsafe {
            let k_dst = self.k_base.add(offset);
            let v_dst = self.v_base.add(offset);
            core::ptr::copy_nonoverlapping(k.as_ptr(), k_dst, expected);
            core::ptr::copy_nonoverlapping(v.as_ptr(), v_dst, expected);
        }
        Ok(())
    }

    /// Get contiguous K slice for layer covering tokens `[0..token_count)`.
    ///
    /// L1-cache-friendly: returns single contiguous slice for attention compute.
    /// Caller passes `token_count` explicitly (no internal counter).
    pub fn get_k_slice(&self, layer: usize, token_count: usize) -> Result<&[f32], KvCacheError> {
        if layer >= self.num_layers {
            return Err(KvCacheError::LayerOutOfRange {
                layer,
                num_layers: self.num_layers,
            });
        }
        if token_count > self.max_tokens {
            return Err(KvCacheError::TokenCountExceedsMax {
                token_count,
                max_tokens: self.max_tokens,
            });
        }
        let offset = layer * self.layer_stride;
        let len = token_count * self.token_stride;
        unsafe { Ok(core::slice::from_raw_parts(self.k_base.add(offset), len)) }
    }

    /// Get contiguous V slice for layer covering tokens `[0..token_count)`.
    pub fn get_v_slice(&self, layer: usize, token_count: usize) -> Result<&[f32], KvCacheError> {
        if layer >= self.num_layers {
            return Err(KvCacheError::LayerOutOfRange {
                layer,
                num_layers: self.num_layers,
            });
        }
        if token_count > self.max_tokens {
            return Err(KvCacheError::TokenCountExceedsMax {
                token_count,
                max_tokens: self.max_tokens,
            });
        }
        let offset = layer * self.layer_stride;
        let len = token_count * self.token_stride;
        unsafe { Ok(core::slice::from_raw_parts(self.v_base.add(offset), len)) }
    }
}

// ──────────────────────────────────────────────────────────────────
//  MlaKvCache — compressed-latent variant for DeepSeek-V2/V3 / Kimi
// ──────────────────────────────────────────────────────────────────
//
//  Stores only the *compressed* MLA state per (layer, token):
//
//      c_kv_normed[kv_lora_rank]     — RMSNorm-of-c_kv (kv_lora_rank
//                                      = 512 for Kimi K2.6)
//      k_rope_post_rope[qk_rope_head_dim]
//                                    — per-token shared k_rope after
//                                      its RMSNorm + RoPE rotation
//                                      (qk_rope_head_dim = 64 for
//                                      Kimi K2.6)
//
//  At attention time the caller re-expands c_kv through W_kv_b to
//  recover per-head k_nope and v vectors — see
//  `mla::mla_attention_single_token`. The trade-off is more compute
//  per attention step in exchange for ~32× less KV-cache memory:
//
//      Decompressed (old): n_heads × (k_head + v_head) f32 per token
//                          = 64 × (192 + 128) = 20 480 f32 per token
//      Compressed   (new): kv_lora_rank + qk_rope_head_dim f32 per
//                          token = 512 + 64 = 576 f32 per token
//      Ratio: ~36× compression for Kimi K2.6 (~80 GiB → ~1.2 GiB at
//      8K context × 61 layers).
//
//  Storage layout (layer-major, single contiguous arena region,
//  per-token interleaved c_kv|k_rope for cache locality):
//
//      [L0 t0 c_kv|k_rope][L0 t1 c_kv|k_rope]...[L1 t0 ...]...
//
//  Token-stride within a layer = `kv_lora_rank + qk_rope_head_dim`.
//  Layer-stride = max_tokens × token_stride.
pub struct MlaKvCache {
    base: *mut f32,
    layer_stride: usize,
    token_stride: usize,
    pub max_tokens: usize,
    pub num_layers: usize,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
}

unsafe impl Send for MlaKvCache {}

impl MlaKvCache {
    /// Required arena size, in f32 elements:
    ///   num_layers × max_tokens × (kv_lora_rank + qk_rope_head_dim)
    pub const fn required_f32(
        max_tokens: usize,
        num_layers: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
    ) -> usize {
        num_layers * max_tokens * (kv_lora_rank + qk_rope_head_dim)
    }

    /// Build an MlaKvCache over caller-owned arena memory.
    ///
    /// # Safety
    /// `base_ptr` must point to at least `required_f32(..)` writable f32 elements.
    pub fn new(
        base_ptr: *mut f32,
        max_tokens: usize,
        num_layers: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
    ) -> Self {
        let token_stride = kv_lora_rank + qk_rope_head_dim;
        let layer_stride = max_tokens * token_stride;
        Self {
            base: base_ptr,
            layer_stride,
            token_stride,
            max_tokens,
            num_layers,
            kv_lora_rank,
            qk_rope_head_dim,
        }
    }

    /// Store the compressed latent (c_kv_normed) and post-RoPE k_rope
    /// for a single token at the given layer.
    ///
    /// `c_kv.len() == kv_lora_rank`, `k_rope.len() == qk_rope_head_dim`.
    pub fn store_compressed(
        &mut self,
        layer: usize,
        token_idx: usize,
        c_kv: &[f32],
        k_rope: &[f32],
    ) -> Result<(), KvCacheError> {
        if layer >= self.num_layers {
            return Err(KvCacheError::LayerOutOfRange {
                layer,
                num_layers: self.num_layers,
            });
        }
        if token_idx >= self.max_tokens {
            return Err(KvCacheError::TokenIndexOutOfRange {
                token_idx,
                max_tokens: self.max_tokens,
            });
        }
        if c_kv.len() != self.kv_lora_rank {
            return Err(KvCacheError::InvalidSliceLength {
                expected: self.kv_lora_rank,
                actual: c_kv.len(),
            });
        }
        if k_rope.len() != self.qk_rope_head_dim {
            return Err(KvCacheError::InvalidSliceLength {
                expected: self.qk_rope_head_dim,
                actual: k_rope.len(),
            });
        }
        let token_off = layer * self.layer_stride + token_idx * self.token_stride;
        unsafe {
            core::ptr::copy_nonoverlapping(
                c_kv.as_ptr(),
                self.base.add(token_off),
                self.kv_lora_rank,
            );
            core::ptr::copy_nonoverlapping(
                k_rope.as_ptr(),
                self.base.add(token_off + self.kv_lora_rank),
                self.qk_rope_head_dim,
            );
        }
        Ok(())
    }

    /// Read the compressed latent c_kv_normed for a single (layer, token).
    pub fn get_c_kv(&self, layer: usize, token_idx: usize) -> Result<&[f32], KvCacheError> {
        if layer >= self.num_layers {
            return Err(KvCacheError::LayerOutOfRange {
                layer,
                num_layers: self.num_layers,
            });
        }
        if token_idx >= self.max_tokens {
            return Err(KvCacheError::TokenIndexOutOfRange {
                token_idx,
                max_tokens: self.max_tokens,
            });
        }
        let token_off = layer * self.layer_stride + token_idx * self.token_stride;
        unsafe {
            Ok(core::slice::from_raw_parts(
                self.base.add(token_off),
                self.kv_lora_rank,
            ))
        }
    }

    /// Read the post-RoPE k_rope for a single (layer, token).
    pub fn get_k_rope(&self, layer: usize, token_idx: usize) -> Result<&[f32], KvCacheError> {
        if layer >= self.num_layers {
            return Err(KvCacheError::LayerOutOfRange {
                layer,
                num_layers: self.num_layers,
            });
        }
        if token_idx >= self.max_tokens {
            return Err(KvCacheError::TokenIndexOutOfRange {
                token_idx,
                max_tokens: self.max_tokens,
            });
        }
        let token_off =
            layer * self.layer_stride + token_idx * self.token_stride + self.kv_lora_rank;
        unsafe {
            Ok(core::slice::from_raw_parts(
                self.base.add(token_off),
                self.qk_rope_head_dim,
            ))
        }
    }

    #[inline]
    pub fn token_stride(&self) -> usize {
        self.token_stride
    }
}

// ──────────────────────────────────────────────────────────────────
//  MhaKvCache — standard K/V cache for DeepSeek-V2 MHA layers
// ──────────────────────────────────────────────────────────────────
//
// Some DeepSeek-V2 / Kimi K2.6 GGUFs ship a small number of layers
// (often only layer 0) using standard Multi-Head Attention with
// `attn_q.weight` / `attn_k.weight` / `attn_v.weight` instead of the
// MLA tensors. Those layers need a full per-token K/V cache, but only
// for the few MHA layers — paying for an MHA-sized slot per layer
// would balloon arena use given Kimi K2.6's 61 transformer blocks.
//
// `MhaKvCache` is therefore indexed by a compact `mha_layer_idx` (the
// transformer-block layer carries the mapping). K and V have potentially
// different per-head dimensions in DeepSeek-V2 architecture:
//
//   head_dim_qk = qk_nope_head_dim + qk_rope_head_dim   (192 for Kimi K2.6)
//   head_dim_v  = v_head_dim                            (128 for Kimi K2.6)
//
// Storage layout (single contiguous arena region):
//
//   [L0 K block][L0 V block][L1 K block][L1 V block]...
//
// where K block size = max_tokens × n_kv_heads × head_dim_qk and
// V block size = max_tokens × n_kv_heads × head_dim_v.
pub struct MhaKvCache {
    base: *mut f32,
    k_block_size: usize,
    layer_stride: usize, // = k_block_size + v_block_size
    k_token_stride: usize,
    v_token_stride: usize,
    pub max_tokens: usize,
    pub num_mha_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim_qk: usize,
    pub head_dim_v: usize,
}

unsafe impl Send for MhaKvCache {}

impl MhaKvCache {
    /// Required arena size, in f32 elements:
    ///   num_mha_layers × max_tokens × n_kv_heads × (head_dim_qk + head_dim_v)
    pub const fn required_f32(
        max_tokens: usize,
        num_mha_layers: usize,
        n_kv_heads: usize,
        head_dim_qk: usize,
        head_dim_v: usize,
    ) -> usize {
        num_mha_layers * max_tokens * n_kv_heads * (head_dim_qk + head_dim_v)
    }

    /// Build an MhaKvCache over caller-owned arena memory.
    ///
    /// # Safety
    /// `base_ptr` must point to at least `required_f32(..)` writable f32 elements.
    pub fn new(
        base_ptr: *mut f32,
        max_tokens: usize,
        num_mha_layers: usize,
        n_kv_heads: usize,
        head_dim_qk: usize,
        head_dim_v: usize,
    ) -> Self {
        let k_token_stride = n_kv_heads * head_dim_qk;
        let v_token_stride = n_kv_heads * head_dim_v;
        let k_block_size = max_tokens * k_token_stride;
        let v_block_size = max_tokens * v_token_stride;
        let layer_stride = k_block_size + v_block_size;
        Self {
            base: base_ptr,
            k_block_size,
            layer_stride,
            k_token_stride,
            v_token_stride,
            max_tokens,
            num_mha_layers,
            n_kv_heads,
            head_dim_qk,
            head_dim_v,
        }
    }

    /// Store K and V vectors for a specific MHA-layer slot at a given token.
    ///
    /// `k.len()` must equal `n_kv_heads × head_dim_qk`,
    /// `v.len()` must equal `n_kv_heads × head_dim_v`.
    pub fn store_kv(
        &mut self,
        mha_layer_idx: usize,
        token_idx: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), KvCacheError> {
        if mha_layer_idx >= self.num_mha_layers {
            return Err(KvCacheError::LayerOutOfRange {
                layer: mha_layer_idx,
                num_layers: self.num_mha_layers,
            });
        }
        if token_idx >= self.max_tokens {
            return Err(KvCacheError::TokenIndexOutOfRange {
                token_idx,
                max_tokens: self.max_tokens,
            });
        }
        if k.len() != self.k_token_stride {
            return Err(KvCacheError::InvalidSliceLength {
                expected: self.k_token_stride,
                actual: k.len(),
            });
        }
        if v.len() != self.v_token_stride {
            return Err(KvCacheError::InvalidSliceLength {
                expected: self.v_token_stride,
                actual: v.len(),
            });
        }

        let layer_off = mha_layer_idx * self.layer_stride;
        let k_off = layer_off + token_idx * self.k_token_stride;
        let v_off = layer_off + self.k_block_size + token_idx * self.v_token_stride;
        unsafe {
            core::ptr::copy_nonoverlapping(k.as_ptr(), self.base.add(k_off), self.k_token_stride);
            core::ptr::copy_nonoverlapping(v.as_ptr(), self.base.add(v_off), self.v_token_stride);
        }
        Ok(())
    }

    /// Contiguous K slice for MHA-layer slot covering tokens `[0..token_count)`.
    pub fn get_k_slice(
        &self,
        mha_layer_idx: usize,
        token_count: usize,
    ) -> Result<&[f32], KvCacheError> {
        if mha_layer_idx >= self.num_mha_layers {
            return Err(KvCacheError::LayerOutOfRange {
                layer: mha_layer_idx,
                num_layers: self.num_mha_layers,
            });
        }
        if token_count > self.max_tokens {
            return Err(KvCacheError::TokenCountExceedsMax {
                token_count,
                max_tokens: self.max_tokens,
            });
        }
        let layer_off = mha_layer_idx * self.layer_stride;
        let len = token_count * self.k_token_stride;
        unsafe { Ok(core::slice::from_raw_parts(self.base.add(layer_off), len)) }
    }

    /// Contiguous V slice for MHA-layer slot covering tokens `[0..token_count)`.
    pub fn get_v_slice(
        &self,
        mha_layer_idx: usize,
        token_count: usize,
    ) -> Result<&[f32], KvCacheError> {
        if mha_layer_idx >= self.num_mha_layers {
            return Err(KvCacheError::LayerOutOfRange {
                layer: mha_layer_idx,
                num_layers: self.num_mha_layers,
            });
        }
        if token_count > self.max_tokens {
            return Err(KvCacheError::TokenCountExceedsMax {
                token_count,
                max_tokens: self.max_tokens,
            });
        }
        let layer_off = mha_layer_idx * self.layer_stride + self.k_block_size;
        let len = token_count * self.v_token_stride;
        unsafe { Ok(core::slice::from_raw_parts(self.base.add(layer_off), len)) }
    }

    #[inline]
    pub fn k_token_stride(&self) -> usize {
        self.k_token_stride
    }

    #[inline]
    pub fn v_token_stride(&self) -> usize {
        self.v_token_stride
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    /// Helper: allocate test storage of the correct size (2 × all layer-blocks)
    /// and construct a KvCache backed by it.
    ///
    /// Rust safety: storage MUST be `mut` and we use `as_mut_ptr()`.
    /// Casting `as_ptr() as *mut` would be Undefined Behavior because the
    /// caller would mutate immutable memory through that pointer.
    fn make_test_cache(
        max_tokens: usize,
        num_layers: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> (std::vec::Vec<f32>, KvCache) {
        // Total = 2 (K+V) × num_layers × max_tokens × kv_heads × head_dim
        let total = 2 * num_layers * max_tokens * kv_heads * head_dim;
        let mut storage = std::vec![0.0f32; total];
        let base_ptr = storage.as_mut_ptr();
        let cache = KvCache::new(base_ptr, max_tokens, num_layers, kv_heads, head_dim);
        (storage, cache)
    }

    #[test]
    fn test_kv_cache_new_strides() {
        // 4 tokens, 2 layers, 2 heads × 4 head_dim = 8 elements per token
        let (_storage, cache) = make_test_cache(4, 2, 2, 4);
        assert_eq!(cache.token_stride, 2 * 4); // = 8
        assert_eq!(cache.layer_stride, 2 * 4 * (2 * 4)); // = 64 (= 2 × block_size)
        assert_eq!(cache.max_tokens, 4);
        assert_eq!(cache.num_layers, 2);
    }

    #[test]
    fn test_store_kv_basic() {
        let (_storage, mut cache) = make_test_cache(4, 2, 2, 4);
        let k = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let result = cache.store_kv(0, 0, &k, &v);
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_k_slice_returns_contiguous() {
        let (_storage, mut cache) = make_test_cache(4, 2, 2, 4);
        let k_t0 = [1.0; 8];
        let k_t1 = [2.0; 8];
        cache.store_kv(0, 0, &k_t0, &k_t0).unwrap();
        cache.store_kv(0, 1, &k_t1, &k_t1).unwrap();

        let slice = cache.get_k_slice(0, 2).unwrap();
        assert_eq!(slice.len(), 16); // 2 tokens × 8 elements
        assert_eq!(slice[0], 1.0); // token 0
        assert_eq!(slice[8], 2.0); // token 1
    }

    #[test]
    fn test_layer_isolation_no_corruption() {
        // CRITICAL: this test verifies the layer_stride = 2 × block_size invariant.
        // If layer_stride were just block_size (BUG), writing to L1_K would
        // overwrite L0_V because L1_K's offset would land where L0_V begins.
        let (_storage, mut cache) = make_test_cache(4, 2, 2, 4);

        let l0_k = [100.0; 8];
        let l0_v = [200.0; 8];
        let l1_k = [300.0; 8];
        let l1_v = [400.0; 8];

        cache.store_kv(0, 0, &l0_k, &l0_v).unwrap();
        cache.store_kv(1, 0, &l1_k, &l1_v).unwrap();

        // Verify L0_K is NOT corrupted by L1_K write
        let l0_k_slice = cache.get_k_slice(0, 1).unwrap();
        assert_eq!(
            l0_k_slice[0], 100.0,
            "L0_K corrupted by L1_K write — stride bug!"
        );

        // Verify L0_V is NOT corrupted by L1_K write
        let l0_v_slice = cache.get_v_slice(0, 1).unwrap();
        assert_eq!(
            l0_v_slice[0], 200.0,
            "L0_V corrupted by L1_K write — stride bug!"
        );

        // Verify L1_K and L1_V are correct
        let l1_k_slice = cache.get_k_slice(1, 1).unwrap();
        let l1_v_slice = cache.get_v_slice(1, 1).unwrap();
        assert_eq!(l1_k_slice[0], 300.0);
        assert_eq!(l1_v_slice[0], 400.0);
    }

    #[test]
    fn test_layer_out_of_range() {
        let (_storage, mut cache) = make_test_cache(4, 2, 2, 4);
        let k = [0.0; 8];
        let result = cache.store_kv(2, 0, &k, &k); // layer 2 doesn't exist
        assert!(matches!(result, Err(KvCacheError::LayerOutOfRange { .. })));
    }

    #[test]
    fn test_token_index_out_of_range() {
        let (_storage, mut cache) = make_test_cache(4, 2, 2, 4);
        let k = [0.0; 8];
        let result = cache.store_kv(0, 4, &k, &k); // token 4 == max_tokens
        assert!(matches!(
            result,
            Err(KvCacheError::TokenIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn test_invalid_slice_length() {
        let (_storage, mut cache) = make_test_cache(4, 2, 2, 4);
        let k_wrong = [0.0; 7]; // expected 8
        let v_correct = [0.0; 8];
        let result = cache.store_kv(0, 0, &k_wrong, &v_correct);
        assert!(matches!(
            result,
            Err(KvCacheError::InvalidSliceLength { .. })
        ));
    }

    #[test]
    fn test_token_count_exceeds_max() {
        let (_storage, cache) = make_test_cache(4, 2, 2, 4);
        let result = cache.get_k_slice(0, 5); // token_count > max_tokens (4)
        assert!(matches!(
            result,
            Err(KvCacheError::TokenCountExceedsMax { .. })
        ));
    }
}
