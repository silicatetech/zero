// SPDX-License-Identifier: AGPL-3.0-or-later
//! GGUF metadata parser for Zero Boot-LLM.
//!
//! Implements the GGUF v3 binary format header parser per the
//! [GGUF specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).
//!
//! # Design Constraints (ADR-028 / ADR-002)
//!
//! - `no_std` + `alloc` only — runs in Ring-0 unikernel
//! - Zero external dependencies (pure Rust)
//! - Parses header + tensor metadata; does NOT dequantize tensor data (MP2)
//!
//! # GGUF v3 Binary Layout
//!
//! ```text
//! [magic: u32 LE "GGUF"]
//! [version: u32 LE]
//! [tensor_count: u64 LE]
//! [metadata_kv_count: u64 LE]
//! [metadata_kv pairs...]
//! [tensor_info entries...]
//! [alignment padding]
//! [tensor data...]
//! ```

#![no_std]

extern crate alloc;

pub mod dequant;

use alloc::string::String;
use alloc::vec::Vec;

/// GGUF magic bytes: "GGUF" in little-endian = 0x46554747.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Supported GGUF format version.
pub const GGUF_VERSION_3: u32 = 3;

/// Default alignment for tensor data (GGUF spec default).
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// ── Errors ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufError {
    InvalidMagic,
    UnsupportedVersion(u32),
    UnexpectedEof,
    InvalidUtf8,
    UnknownValueType(u32),
    UnknownTensorType(u32),
    MissingMetadata,
}

// ── Value types ─────────────────────────────────────────────────────

/// GGUF metadata value.
///
/// Array contents are skipped by default (only length + element type
/// recorded) to keep `parse_selective` allocation-bounded. The one
/// exception is the `tokenizer.ggml.tokens` key, which the kernel needs
/// to render generated token-ids back to text; for that key
/// `parse_selective` materialises the strings into `StringArray`.
#[derive(Debug, Clone)]
pub enum GgufValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
    /// Array — element type tag + length, with elements skipped.
    Array {
        element_type: u32,
        length: u64,
    },
}

// NB: `parse_selective` never materialises string arrays — tokens come
// out of `extract_tokenizer_from_bytes` as a dedicated `TokenizerData`.

// ── Tensor types (GGML quantization) ────────────────────────────────

/// GGML tensor data types per ggml.h / GGUF spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4, 5 deprecated
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::IQ2XXS),
            17 => Ok(Self::IQ2XS),
            18 => Ok(Self::IQ3XXS),
            19 => Ok(Self::IQ1S),
            20 => Ok(Self::IQ4NL),
            21 => Ok(Self::IQ3S),
            22 => Ok(Self::IQ2S),
            23 => Ok(Self::IQ4XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1M),
            30 => Ok(Self::BF16),
            _ => Err(GgufError::UnknownTensorType(v)),
        }
    }
}

// ── Tensor info ─────────────────────────────────────────────────────

/// Parsed GGUF tensor metadata (name, shape, type, data offset).
#[derive(Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dimensions: u32,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgmlType,
    /// Byte offset of this tensor's data from the start of the tensor
    /// data section (NOT from the start of the file).
    pub offset: u64,
}

// ── Top-level result ────────────────────────────────────────────────

/// Parsed GGUF file metadata: header, metadata KV pairs, and tensor
/// info entries. Does not include the tensor data itself.
#[derive(Debug)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset where tensor data begins (from file start).
    /// This is the header + metadata + tensor_info size, aligned.
    pub tensor_data_offset: usize,
}

// ── Cursor-based reader ─────────────────────────────────────────────

/// A simple cursor over a byte slice for sequential reading.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[allow(dead_code)]
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        if self.pos + n > self.data.len() {
            return Err(GgufError::UnexpectedEof);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a GGUF string: u64 length + UTF-8 bytes (no null terminator).
    fn read_gguf_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        core::str::from_utf8(bytes)
            .map(String::from)
            .map_err(|_| GgufError::InvalidUtf8)
    }
}

// ── Value parsing ───────────────────────────────────────────────────

/// Size of a single element of the given GGUF value type in bytes.
/// Used to skip array elements.
fn value_type_element_size(type_id: u32) -> Option<usize> {
    match type_id {
        0 => Some(1),  // uint8
        1 => Some(1),  // int8
        2 => Some(2),  // uint16
        3 => Some(2),  // int16
        4 => Some(4),  // uint32
        5 => Some(4),  // int32
        6 => Some(4),  // float32
        7 => Some(1),  // bool
        8 => None,     // string — variable length
        9 => None,     // array — nested, variable length
        10 => Some(8), // uint64
        11 => Some(8), // int64
        12 => Some(8), // float64
        _ => None,
    }
}

/// Read a single GGUF metadata value.
fn read_value(cursor: &mut Cursor<'_>) -> Result<GgufValue, GgufError> {
    let value_type = cursor.read_u32()?;
    match value_type {
        0 => Ok(GgufValue::UInt8(cursor.read_u8()?)),
        1 => Ok(GgufValue::Int8(cursor.read_i8()?)),
        2 => Ok(GgufValue::UInt16(cursor.read_u16()?)),
        3 => Ok(GgufValue::Int16(cursor.read_i16()?)),
        4 => Ok(GgufValue::UInt32(cursor.read_u32()?)),
        5 => Ok(GgufValue::Int32(cursor.read_i32()?)),
        6 => Ok(GgufValue::Float32(cursor.read_f32()?)),
        7 => Ok(GgufValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(GgufValue::String(cursor.read_gguf_string()?)),
        9 => {
            // Array: element_type (u32) + length (u64) + elements
            let element_type = cursor.read_u32()?;
            let length = cursor.read_u64()?;
            // Skip element data — MP1 only needs metadata keys, not
            // full array content deserialization.
            if let Some(elem_size) = value_type_element_size(element_type) {
                let skip = length as usize * elem_size;
                cursor.read_bytes(skip)?;
            } else if element_type == 8 {
                // Array of strings — must read each to skip
                for _ in 0..length {
                    cursor.read_gguf_string()?;
                }
            } else {
                // Nested arrays or unknown — skip is not trivially
                // computable. For MP1 we return what we have.
                // In practice, GGUF files rarely have nested arrays.
                for _ in 0..length {
                    read_value(cursor)?;
                }
            }
            Ok(GgufValue::Array {
                element_type,
                length,
            })
        }
        10 => Ok(GgufValue::UInt64(cursor.read_u64()?)),
        11 => Ok(GgufValue::Int64(cursor.read_i64()?)),
        12 => Ok(GgufValue::Float64(cursor.read_f64()?)),
        _ => Err(GgufError::UnknownValueType(value_type)),
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// Parse GGUF header, metadata KV pairs, and tensor info from a byte
/// slice. Returns structured metadata without reading tensor data.
///
/// # Arguments
///
/// * `bytes` — Complete GGUF file contents (or at minimum the header
///   through tensor info section). Tensor data can be truncated.
///
/// # Errors
///
/// Returns `GgufError` if the file has invalid magic, unsupported
/// version, or is truncated before all metadata/tensor info is read.
pub fn parse_header(bytes: &[u8]) -> Result<GgufMetadata, GgufError> {
    let mut cursor = Cursor::new(bytes);

    // ── Header (24 bytes) ───────────────────────────────────────
    let magic = cursor.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }

    let version = cursor.read_u32()?;
    if version != GGUF_VERSION_3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = cursor.read_u64()?;
    let metadata_kv_count = cursor.read_u64()?;

    // ── Metadata KV pairs ───────────────────────────────────────
    let mut metadata = Vec::with_capacity(metadata_kv_count as usize);
    for _ in 0..metadata_kv_count {
        let key = cursor.read_gguf_string()?;
        let value = read_value(&mut cursor)?;
        metadata.push((key, value));
    }

    // ── Tensor info entries ─────────────────────────────────────
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = cursor.read_gguf_string()?;
        let n_dimensions = cursor.read_u32()?;
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(cursor.read_u64()?);
        }
        let tensor_type_raw = cursor.read_u32()?;
        let tensor_type = GgmlType::from_u32(tensor_type_raw)?;
        let offset = cursor.read_u64()?;
        tensors.push(GgufTensorInfo {
            name,
            n_dimensions,
            dimensions,
            tensor_type,
            offset,
        });
    }

    // ── Compute tensor data offset (aligned) ────────────────────
    // Check if metadata contains an alignment override.
    let alignment = metadata
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| match v {
            GgufValue::UInt32(a) => Some(*a as usize),
            _ => None,
        })
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

    let header_end = cursor.pos;
    let tensor_data_offset = header_end.div_ceil(alignment) * alignment;

    Ok(GgufMetadata {
        version,
        tensor_count,
        metadata_kv_count,
        metadata,
        tensors,
        tensor_data_offset,
    })
}

// ── Skip value (zero-alloc) ─────────────────────────────────────────

/// Skip a GGUF value without allocating. Advances the cursor past the
/// value bytes. Used by `parse_selective` to skip tokenizer arrays.
fn skip_value_at(cursor: &mut Cursor<'_>, value_type: u32) -> Result<(), GgufError> {
    match value_type {
        0 | 1 | 7 => {
            cursor.read_bytes(1)?;
        }
        2 | 3 => {
            cursor.read_bytes(2)?;
        }
        4..=6 => {
            cursor.read_bytes(4)?;
        }
        10..=12 => {
            cursor.read_bytes(8)?;
        }
        8 => {
            let len = cursor.read_u64()? as usize;
            cursor.read_bytes(len)?;
        }
        9 => {
            let elem_type = cursor.read_u32()?;
            let length = cursor.read_u64()?;
            if let Some(elem_size) = value_type_element_size(elem_type) {
                cursor.read_bytes(length as usize * elem_size)?;
            } else {
                for _ in 0..length {
                    skip_value_at(cursor, elem_type)?;
                }
            }
        }
        _ => return Err(GgufError::UnknownValueType(value_type)),
    }
    Ok(())
}

// ── ModelConfig ─────────────────────────────────────────────────────

/// Model architecture parameters extracted from GGUF metadata.
/// All values read from metadata at boot — never hardcoded.
/// Per ADR-029 D7 Hyperparameter Discipline.
///
/// Supports: Qwen3 (dense), DeepSeek-V2/V3/Kimi-K2.6 (MoE + MLA).
/// MoE/MLA fields are Optional — absent for dense models like Qwen3.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub block_count: u32,
    pub context_length: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub rope_freq_base: f32,
    pub layer_norm_rms_epsilon: f32,

    // ── MoE fields (DeepSeek-V2/V3, Kimi-K2.6, Mixtral) ──────────
    /// Total number of experts per MoE layer (e.g. 384 for Kimi K2.6).
    pub expert_count: Option<u32>,
    /// Number of experts activated per token (e.g. 8 for Kimi K2.6).
    pub expert_used_count: Option<u32>,
    /// Number of shared (always-on) experts (e.g. 1 for Kimi K2.6).
    pub expert_shared_count: Option<u32>,
    /// FFN intermediate dimension per expert (e.g. 2048 for Kimi K2.6).
    /// Distinct from feed_forward_length which is the dense FFN dim.
    pub expert_feed_forward_length: Option<u32>,
    /// Expert routing weight scale factor.
    pub expert_weights_scale: Option<f32>,

    // ── MLA fields (Multi-Head Latent Attention, DeepSeek-V2/V3) ──
    /// Low-rank dimension for KV compression (e.g. 512 for Kimi K2.6).
    pub kv_lora_rank: Option<u32>,
    /// Low-rank dimension for Q compression (e.g. 1536 for Kimi K2.6).
    pub q_lora_rank: Option<u32>,
    /// Dimension of non-RoPE part of each Q/K head (e.g. 128).
    pub qk_nope_head_dim: Option<u32>,
    /// Dimension of RoPE part of each Q/K head (e.g. 64).
    pub qk_rope_head_dim: Option<u32>,
    /// Dimension of each V head (e.g. 128).
    pub v_head_dim: Option<u32>,
    /// MLA effective key length per head emitted by newer GGUF converters
    /// (Kimi K2.6). Equals `qk_nope_head_dim + qk_rope_head_dim` (e.g. 192).
    /// Lets the parser recover `qk_nope_head_dim` as
    /// `key_length_mla - qk_rope_head_dim` without inspecting any tensor.
    pub key_length_mla: Option<u32>,
    /// MLA effective value length per head emitted by newer GGUF converters
    /// (Kimi K2.6). Equals `v_head_dim` (e.g. 128).
    pub value_length_mla: Option<u32>,

    // ── Vocabulary ────────────────────────────────────────────────
    /// Vocabulary size from GGUF metadata. None falls back to Qwen3 default.
    pub vocab_size: Option<u32>,

    // ── Tokenizer EOS handling ────────────────────────────────────
    /// End-of-sequence token id from `tokenizer.ggml.eos_token_id`.
    /// `None` when the GGUF doesn't carry it — the kernel falls back to
    /// architecture-specific defaults (Qwen3: 151643, Llama-3 family /
    /// DeepSeek-V2/Kimi: 128001). Read for both architectures so the
    /// generation loop can stop deterministically without hard-coded
    /// per-arch constants.
    pub eos_token_id: Option<u32>,
}

impl ModelConfig {
    /// Extract model architecture config from parsed metadata KV pairs.
    pub fn from_metadata(kvs: &[(String, GgufValue)]) -> Result<Self, GgufError> {
        let arch = kvs
            .iter()
            .find(|(k, _)| k == "general.architecture")
            .and_then(|(_, v)| {
                if let GgufValue::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .ok_or(GgufError::MissingMetadata)?;

        let get_u32 = |key: &str| -> Result<u32, GgufError> {
            kvs.iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| {
                    if let GgufValue::UInt32(n) = v {
                        Some(*n)
                    } else {
                        None
                    }
                })
                .ok_or(GgufError::MissingMetadata)
        };

        let get_f32 = |key: &str| -> Result<f32, GgufError> {
            kvs.iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| {
                    if let GgufValue::Float32(f) = v {
                        Some(*f)
                    } else {
                        None
                    }
                })
                .ok_or(GgufError::MissingMetadata)
        };

        // Build arch-prefixed keys without alloc::format! (Mode B avoidance per ADR-028 v5).
        fn arch_key(arch: &str, suffix: &str) -> String {
            let mut s = String::with_capacity(arch.len() + 1 + suffix.len());
            s.push_str(arch);
            s.push('.');
            s.push_str(suffix);
            s
        }

        // Optional-variant helpers for MoE/MLA fields.
        // Coerce any integer type to u32 — GGUF producers (gguf-py,
        // llama-quantize, bartowski builds) are inconsistent about
        // whether MoE/MLA scalars land as UInt32, UInt64, Int32, etc.
        let get_u32_opt = |key: &str| -> Option<u32> {
            kvs.iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| match v {
                    GgufValue::UInt32(n) => Some(*n),
                    GgufValue::UInt64(n) => Some(*n as u32),
                    GgufValue::Int32(n) => Some(*n as u32),
                    GgufValue::Int64(n) => Some(*n as u32),
                    GgufValue::UInt16(n) => Some(*n as u32),
                    GgufValue::Int16(n) => Some(*n as u32),
                    GgufValue::UInt8(n) => Some(*n as u32),
                    GgufValue::Int8(n) => Some(*n as u32),
                    _ => None,
                })
        };

        let get_f32_opt = |key: &str| -> Option<f32> {
            kvs.iter().find(|(k, _)| k == key).and_then(|(_, v)| {
                if let GgufValue::Float32(f) = v {
                    Some(*f)
                } else {
                    None
                }
            })
        };

        // Required dims, read once so MLA derivations below can reuse them
        // without a second metadata scan.
        let key_length = get_u32(&arch_key(&arch, "attention.key_length"))?;
        let value_length = get_u32(&arch_key(&arch, "attention.value_length"))?;

        // MLA presence is signalled by `attention.kv_lora_rank`; non-MLA
        // architectures (Qwen3 etc.) leave it unset.
        let kv_lora_rank = get_u32_opt(&arch_key(&arch, "attention.kv_lora_rank"));
        let q_lora_rank = get_u32_opt(&arch_key(&arch, "attention.q_lora_rank"));

        // MLA per-head dims — attempt explicit GGUF keys first.
        // The *correct* derivation from tensor shapes happens in
        // parse_selective() AFTER from_metadata() returns, because
        // from_metadata only sees KV pairs, not tensor entries.
        // Here we just pick up any explicit keys (rare but future-proof)
        // and set initial values for rope_dim from rope.dimension_count.
        // qk_nope_head_dim and v_head_dim are intentionally left None
        // here for MLA — they are fixed up from blk.0.attn_kv_b.weight
        // shape in parse_selective().
        let qk_rope_head_dim =
            get_u32_opt(&arch_key(&arch, "attention.qk_rope_head_dim")).or_else(|| {
                if kv_lora_rank.is_some() {
                    get_u32_opt(&arch_key(&arch, "rope.dimension_count"))
                } else {
                    None
                }
            });
        let qk_nope_head_dim = get_u32_opt(&arch_key(&arch, "attention.qk_nope_head_dim"));
        let v_head_dim = get_u32_opt(&arch_key(&arch, "attention.v_head_dim"));
        // Newer GGUF converters (Kimi K2.6) emit the effective per-head
        // MLA lengths directly. `key_length_mla = qk_nope + qk_rope`,
        // `value_length_mla = v_head`. Captured here so the fixup below
        // can derive per-head dims without consulting any tensor shape.
        let key_length_mla = get_u32_opt(&arch_key(&arch, "attention.key_length_mla"));
        let value_length_mla = get_u32_opt(&arch_key(&arch, "attention.value_length_mla"));

        Ok(Self {
            block_count: get_u32(&arch_key(&arch, "block_count"))?,
            context_length: get_u32(&arch_key(&arch, "context_length"))?,
            embedding_length: get_u32(&arch_key(&arch, "embedding_length"))?,
            feed_forward_length: get_u32(&arch_key(&arch, "feed_forward_length"))?,
            head_count: get_u32(&arch_key(&arch, "attention.head_count"))?,
            head_count_kv: get_u32(&arch_key(&arch, "attention.head_count_kv"))?,
            key_length,
            value_length,
            rope_freq_base: get_f32(&arch_key(&arch, "rope.freq_base"))?,
            layer_norm_rms_epsilon: get_f32(&arch_key(&arch, "attention.layer_norm_rms_epsilon"))?,
            // MoE fields
            expert_count: get_u32_opt(&arch_key(&arch, "expert_count")),
            expert_used_count: get_u32_opt(&arch_key(&arch, "expert_used_count")),
            expert_shared_count: get_u32_opt(&arch_key(&arch, "expert_shared_count")),
            expert_feed_forward_length: get_u32_opt(&arch_key(&arch, "expert_feed_forward_length")),
            expert_weights_scale: get_f32_opt(&arch_key(&arch, "expert_weights_scale")),
            // MLA fields
            kv_lora_rank,
            q_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            key_length_mla,
            value_length_mla,
            // Vocabulary
            vocab_size: get_u32_opt(&arch_key(&arch, "vocab_size")),
            // Tokenizer EOS — note: this key is NOT prefixed with the
            // architecture name (it lives in the `tokenizer.ggml.*`
            // namespace shared across all models). Read as Option so
            // the kernel can fall back to per-arch defaults if absent.
            eos_token_id: get_u32_opt("tokenizer.ggml.eos_token_id"),
            architecture: arch,
        })
    }

    /// Returns true if this model uses Mixture-of-Experts.
    pub fn is_moe(&self) -> bool {
        self.expert_count.is_some_and(|c| c > 0)
    }

    /// Returns true if this model uses Multi-Head Latent Attention.
    pub fn is_mla(&self) -> bool {
        self.kv_lora_rank.is_some()
    }

    /// Returns true if this is a DeepSeek-V2/V3 architecture (MoE + MLA).
    pub fn is_deepseek2(&self) -> bool {
        self.architecture == "deepseek2"
    }
}

// ── TensorIndex ─────────────────────────────────────────────────────

/// Pre-built tensor index with O(N) linear-scan lookup.
/// Vec<TensorInfo> chosen over BTreeMap to avoid monomorphization bloat
/// (Mode B avoidance per ADR-028 v5). ADR-029 D2 explicitly permits
/// O(311) linear scan — sub-microsecond on modern hardware.
/// Optional tokenizer payload extracted from a GGUF file. Populated when
/// the source contains `tokenizer.ggml.tokens` (the string-array entry
/// listing every token's UTF-8 bytes). When absent, the kernel uses the
/// compile-time embedded Qwen3 vocab as a fallback — see
/// `kernel/src/detokenizer.rs`.
///
/// Storage layout: tokens are flattened into a single `bytes` buffer
/// with an `offsets` table (`offsets.len() == tokens + 1`, last entry
/// is `bytes.len()`). This avoids 100 000+ small allocations during
/// parse — Qwen3 ships ~152 k tokens and Kimi K2.6 ships ~164 k.
/// Per-token slicing happens via [`Self::token`].
#[derive(Debug, Clone)]
pub struct TokenizerData {
    /// Concatenated UTF-8 bytes of every token in vocab order.
    pub bytes: Vec<u8>,
    /// `offsets.len() == vocab_size + 1`. Token `i` is
    /// `bytes[offsets[i]..offsets[i+1]]`.
    pub offsets: Vec<u32>,
    /// `tokenizer.ggml.model` (e.g. "gpt2", "llama"). Information-only
    /// today; future work could dispatch BPE-merge logic on this.
    pub model: Option<String>,
    /// `tokenizer.ggml.eos_token_id` if the file carried one.
    pub eos_token_id: Option<u32>,
    /// `tokenizer.ggml.bos_token_id` if the file carried one.
    pub bos_token_id: Option<u32>,
}

impl TokenizerData {
    /// Number of tokens in the vocabulary (0 if empty).
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// `true` if no tokens are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Slice of token `idx`'s UTF-8 bytes, or `None` if out of range.
    /// Defensive against corrupt offset tables.
    pub fn token(&self, idx: usize) -> Option<&[u8]> {
        let start = *self.offsets.get(idx)? as usize;
        let end = *self.offsets.get(idx + 1)? as usize;
        if end > self.bytes.len() || start > end {
            return None;
        }
        Some(&self.bytes[start..end])
    }
}

#[derive(Debug)]
pub struct TensorIndex {
    pub tensors: Vec<GgufTensorInfo>,
    pub model_config: ModelConfig,
    pub tensor_data_offset: usize,
    /// Tokenizer extracted from `tokenizer.ggml.tokens` (and friends)
    /// when present. `None` for legacy GGUFs that don't ship a vocab,
    /// for the Qwen3 test fixtures, or when the metadata KV scan
    /// rejected the tokens array (large allocation budget gating).
    pub tokenizer: Option<TokenizerData>,
    /// `general.name` metadata when present — what the producer named
    /// the file (e.g. "Qwen Qwen3 1.7B", "Kimi K2.6"). `None` for
    /// GGUFs that omit the key. Drives the telemetry panel's Model row
    /// so the on-screen label tracks the actually-loaded weights
    /// instead of a hardcoded Qwen3 placeholder.
    pub model_name: Option<String>,
}

impl TensorIndex {
    /// Linear-scan lookup by name over 311 tensors.
    pub fn get(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Iterate all tensors of a given type.
    pub fn iter_by_type(&self, t: GgmlType) -> impl Iterator<Item = &GgufTensorInfo> {
        self.tensors
            .iter()
            .filter(move |info| info.tensor_type == t)
    }

    /// Total number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns `true` if there are no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl GgufTensorInfo {
    /// Compute total element count from dimensions.
    pub fn element_count(&self) -> u64 {
        self.dimensions.iter().product()
    }
}

// ── Selective Parser ────────────────────────────────────────────────

/// Parse GGUF selectively: skip tokenizer arrays, build TensorIndex.
/// Per ADR-029 D2 — reads architecture metadata + tensor info only.
/// Tokenizer arrays (5.93 MB) are skipped without allocation.
pub fn parse_selective(bytes: &[u8]) -> Result<TensorIndex, GgufError> {
    let mut cursor = Cursor::new(bytes);

    // Header
    let magic = cursor.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }
    let version = cursor.read_u32()?;
    if version != GGUF_VERSION_3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = cursor.read_u64()?;
    let metadata_kv_count = cursor.read_u64()?;

    // Metadata KVs — keep:
    //   * every non-tokenizer key (architecture params, etc.)
    //   * small tokenizer scalars (eos/bos/unk/padding token ids)
    //   * tokenizer.ggml.model (small string, ~5 B)
    //
    // Skip (allocation-heavy; kernel runtime arena is only 2 MiB):
    //   * tokenizer.ggml.tokens   — materialise on demand via
    //                                `extract_tokenizer_from_bytes`
    //   * tokenizer.ggml.merges   — BPE merge table, not yet consumed
    //   * tokenizer.ggml.token_type, scores  — not consumed
    let mut metadata = Vec::new();
    for _ in 0..metadata_kv_count {
        let key = cursor.read_gguf_string()?;
        let value_type = cursor.read_u32()?;
        let keep = !key.starts_with("tokenizer.")
            || matches!(
                key.as_str(),
                "tokenizer.ggml.eos_token_id"
                    | "tokenizer.ggml.bos_token_id"
                    | "tokenizer.ggml.unknown_token_id"
                    | "tokenizer.ggml.padding_token_id"
                    | "tokenizer.ggml.model"
            );
        if !keep {
            skip_value_at(&mut cursor, value_type)?;
        } else {
            let value = read_value_body(&mut cursor, value_type)?;
            metadata.push((key, value));
        }
    }

    // Tensor info
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = cursor.read_gguf_string()?;
        let n_dimensions = cursor.read_u32()?;
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(cursor.read_u64()?);
        }
        let tensor_type = GgmlType::from_u32(cursor.read_u32()?)?;
        let offset = cursor.read_u64()?;
        tensors.push(GgufTensorInfo {
            name,
            n_dimensions,
            dimensions,
            tensor_type,
            offset,
        });
    }

    // Alignment
    let alignment = metadata
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| {
            if let GgufValue::UInt32(a) = v {
                Some(*a as usize)
            } else {
                None
            }
        })
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    let tensor_data_offset = cursor.pos.div_ceil(alignment) * alignment;

    let mut model_config = ModelConfig::from_metadata(&metadata)?;

    // ── MLA dim fixup from tensor shapes ──────────────────────────────
    //
    // For DeepSeek-V2 / MLA models, the GGUF keys `attention.key_length`
    // and `attention.value_length` represent the *compressed-latent*
    // dimensions, not the actual per-head dims:
    //
    //   key_length   = kv_lora_rank + qk_rope_head_dim  (e.g. 512+64=576)
    //   value_length = kv_lora_rank                      (e.g. 512)
    //
    // The actual per-head dims (qk_nope_head_dim, v_head_dim) are NOT
    // stored as explicit GGUF keys.  Two GGUF tensor layouts exist:
    //
    //   (a) Combined: `blk.N.attn_kv_b.weight`
    //       shape = [n_heads * (qk_nope + v_head), kv_lora_rank]
    //
    //   (b) Split (Kimi K2.6 / newer converters): separate tensors
    //       `blk.N.attn_k_b.weight`  shape = [n_heads * qk_nope, kv_lora_rank]
    //       `blk.N.attn_v_b.weight`  shape = [n_heads * v_head,  kv_lora_rank]
    //
    // This fixup only fires when kv_lora_rank is set (MLA models).
    if model_config.kv_lora_rank.is_some() {
        // Priority 1: explicit GGUF keys (rare; future-proof) — already
        // populated by from_metadata above. Nothing to do here.
        //
        // Priority 2: MLA effective-length keys (Kimi K2.6+ converters).
        //   key_length_mla   = qk_nope_head_dim + qk_rope_head_dim
        //   value_length_mla = v_head_dim
        // qk_rope_head_dim came from rope.dimension_count earlier, so we
        // recover qk_nope as `key_length_mla - qk_rope_head_dim`.
        if model_config.qk_nope_head_dim.is_none() {
            if let (Some(kl_mla), Some(qk_rope)) =
                (model_config.key_length_mla, model_config.qk_rope_head_dim)
            {
                if kl_mla > qk_rope {
                    model_config.qk_nope_head_dim = Some(kl_mla - qk_rope);
                }
            }
        }
        if model_config.v_head_dim.is_none() {
            if let Some(vl_mla) = model_config.value_length_mla {
                model_config.v_head_dim = Some(vl_mla);
            }
        }

        // Priority 3+4: tensor-shape derivation when explicit/MLA-length
        // keys are absent. Try combined `attn_kv_b` first (older GGUF),
        // then split `attn_k_b` / `attn_v_b` (Kimi K2.6).
        if model_config.qk_nope_head_dim.is_none() || model_config.v_head_dim.is_none() {
            // Strategy 3: combined attn_kv_b tensor (older GGUF layout)
            if let Some(kv_b) = tensors
                .iter()
                .find(|t| t.name.ends_with(".attn_kv_b.weight"))
            {
                if !kv_b.dimensions.is_empty() {
                    let out_dim = kv_b.dimensions[0] as u32;
                    if let Some(per_head_total) = out_dim.checked_div(model_config.head_count) {
                        let half = per_head_total / 2;
                        if model_config.qk_nope_head_dim.is_none() {
                            model_config.qk_nope_head_dim = Some(half);
                        }
                        if model_config.v_head_dim.is_none() {
                            model_config.v_head_dim = Some(half);
                        }
                    }
                }
            }
            // Strategy 4: split attn_k_b / attn_v_b tensors (Kimi K2.6+)
            //
            // Two GGUF layouts exist for split tensors:
            //   (a) 2D: [n_heads * per_head_dim, kv_lora_rank]
            //   (b) 3D: [per_head_dim, kv_lora_rank, n_heads]  (unsloth converter)
            //
            // For 2D, dimensions[0] = n_heads * per_head_dim → divide by n_heads.
            // For 3D, derive per_head_dim from total element count:
            //   per_head_dim = element_count / n_heads / kv_lora_rank
            // This is layout-agnostic and works regardless of dim ordering.
            else {
                if let Some(k_b) = tensors
                    .iter()
                    .find(|t| t.name.ends_with(".attn_k_b.weight"))
                {
                    if model_config.qk_nope_head_dim.is_none() {
                        if k_b.dimensions.len() == 2 && k_b.dimensions[0] > 0 {
                            let out_dim = k_b.dimensions[0] as u32;
                            // 2D: attn_k_b output = n_heads * qk_nope_head_dim
                            if let Some(per_head) = out_dim.checked_div(model_config.head_count) {
                                model_config.qk_nope_head_dim = Some(per_head);
                            }
                        } else if k_b.dimensions.len() == 3 {
                            // 3D: derive from total elements
                            if let Some(kv_lr) = model_config.kv_lora_rank {
                                let total = k_b.element_count();
                                let n = model_config.head_count as u64;
                                if let Some(per_head) = total
                                    .checked_div(n)
                                    .and_then(|v| v.checked_div(kv_lr as u64))
                                {
                                    if per_head > 0 {
                                        model_config.qk_nope_head_dim = Some(per_head as u32);
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(v_b) = tensors
                    .iter()
                    .find(|t| t.name.ends_with(".attn_v_b.weight"))
                {
                    if model_config.v_head_dim.is_none() {
                        if v_b.dimensions.len() == 2 && v_b.dimensions[0] > 0 {
                            let out_dim = v_b.dimensions[0] as u32;
                            // 2D: attn_v_b output = n_heads * v_head_dim
                            if let Some(per_head) = out_dim.checked_div(model_config.head_count) {
                                model_config.v_head_dim = Some(per_head);
                            }
                        } else if v_b.dimensions.len() == 3 {
                            // 3D: derive from total elements
                            if let Some(kv_lr) = model_config.kv_lora_rank {
                                let total = v_b.element_count();
                                let n = model_config.head_count as u64;
                                if let Some(per_head) = total
                                    .checked_div(n)
                                    .and_then(|v| v.checked_div(kv_lr as u64))
                                {
                                    if per_head > 0 {
                                        model_config.v_head_dim = Some(per_head as u32);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Strategy 5: DeepSeek2 architectural constants as final fallback.
        // SPEC: DeepSeek-V2 MLA architectural constant (qk_nope=128, v_head=128),
        // verified against deepseek-ai/DeepSeek-V2 config.json. The whole
        // DeepSeek-V2 family (V2, V2-Lite, V3, Kimi K2/K2.5/K2.6) shares
        // these dims by spec, not per-model configuration.
        if model_config.qk_nope_head_dim.is_none() {
            model_config.qk_nope_head_dim = Some(128);
        }
        if model_config.v_head_dim.is_none() {
            model_config.v_head_dim = Some(128);
        }
        // qk_rope_head_dim: prefer rope.dimension_count (already set),
        // or derive from key_length − kv_lora_rank as cross-check
        if model_config.qk_rope_head_dim.is_none() {
            if let Some(kv_lr) = model_config.kv_lora_rank {
                let derived = model_config.key_length.saturating_sub(kv_lr);
                if derived > 0 {
                    model_config.qk_rope_head_dim = Some(derived);
                }
            }
        }
    }

    let model_name =
        metadata
            .iter()
            .find(|(k, _)| k == "general.name")
            .and_then(|(_, v)| match v {
                GgufValue::String(s) => Some(s.clone()),
                _ => None,
            });
    // Tokenizer is NOT extracted by parse_selective — the GGUF's
    // `tokenizer.ggml.tokens` array can be multi-MB and the kernel's
    // 2-MiB runtime arena chokes if we try to load it unconditionally.
    // The kernel deepseek2 path calls `extract_tokenizer_from_bytes`
    // explicitly after Phase-A architecture detection.
    let _ = metadata;

    Ok(TensorIndex {
        tensors,
        model_config,
        tensor_data_offset,
        tokenizer: None,
        model_name,
    })
}

/// Stand-alone tokenizer extractor — re-walks a GGUF blob looking only
/// for `tokenizer.ggml.tokens` and the small scalar metadata around it
/// (`model`, `eos_token_id`, `bos_token_id`). Materialises tokens into
/// flat `(bytes, offsets)` storage to keep peak allocation bounded.
///
/// Why a second walk: the kernel boot path runs `parse_selective` first
/// to learn the architecture; only when that's `deepseek2` (or any
/// non-Qwen3 architecture that ships its own tokenizer) does it call
/// this function. Qwen3 boots never pay the cost — the compile-time
/// embedded vocab is sacred for the β-anchor.
///
/// Allocation budget: two `Vec`s (bytes + offsets). The kernel calls
/// this against `KERNEL_ARENA` via its global allocator; the operator
/// must ensure that arena is sized to hold `total_token_bytes +
/// (vocab_size + 1) × 4`.
pub fn extract_tokenizer_from_bytes(bytes: &[u8]) -> Result<Option<TokenizerData>, GgufError> {
    let mut cursor = Cursor::new(bytes);
    let magic = cursor.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }
    let version = cursor.read_u32()?;
    if version != GGUF_VERSION_3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let _tensor_count = cursor.read_u64()?;
    let metadata_kv_count = cursor.read_u64()?;

    let mut flat: Option<(Vec<u8>, Vec<u32>)> = None;
    let mut model: Option<String> = None;
    let mut eos: Option<u32> = None;
    let mut bos: Option<u32> = None;

    for _ in 0..metadata_kv_count {
        let key = cursor.read_gguf_string()?;
        let value_type = cursor.read_u32()?;

        if key == "tokenizer.ggml.tokens" && value_type == 9 {
            // Materialise into flat buffers — same layout TokenizerData
            // exposes, no extra copy required downstream.
            let element_type = cursor.read_u32()?;
            let length = cursor.read_u64()? as usize;
            if element_type != 8 {
                return Err(GgufError::UnknownValueType(element_type));
            }
            let mut offsets: Vec<u32> = Vec::with_capacity(length + 1);
            let mut bytes_buf: Vec<u8> = Vec::with_capacity(length.saturating_mul(8));
            for _ in 0..length {
                offsets.push(bytes_buf.len() as u32);
                let str_len = cursor.read_u64()? as usize;
                let str_bytes = cursor.read_bytes(str_len)?;
                bytes_buf.extend_from_slice(str_bytes);
            }
            offsets.push(bytes_buf.len() as u32);
            flat = Some((bytes_buf, offsets));
            continue;
        }

        if key == "tokenizer.ggml.model" && value_type == 8 {
            model = Some(cursor.read_gguf_string()?);
            continue;
        }
        if key == "tokenizer.ggml.eos_token_id" && value_type == 4 {
            eos = Some(cursor.read_u32()?);
            continue;
        }
        if key == "tokenizer.ggml.bos_token_id" && value_type == 4 {
            bos = Some(cursor.read_u32()?);
            continue;
        }

        // Anything else — skip without materialising.
        skip_value_at(&mut cursor, value_type)?;
    }

    Ok(flat.map(|(bytes, offsets)| TokenizerData {
        bytes,
        offsets,
        model,
        eos_token_id: eos,
        bos_token_id: bos,
    }))
}

// (extract_tokenizer was here — replaced by the public
// `extract_tokenizer_from_bytes` which performs its own GGUF walk.)

/// Read value body AFTER the type tag has already been consumed.
fn read_value_body(cursor: &mut Cursor<'_>, value_type: u32) -> Result<GgufValue, GgufError> {
    match value_type {
        0 => Ok(GgufValue::UInt8(cursor.read_u8()?)),
        1 => Ok(GgufValue::Int8(cursor.read_i8()?)),
        2 => Ok(GgufValue::UInt16(cursor.read_u16()?)),
        3 => Ok(GgufValue::Int16(cursor.read_i16()?)),
        4 => Ok(GgufValue::UInt32(cursor.read_u32()?)),
        5 => Ok(GgufValue::Int32(cursor.read_i32()?)),
        6 => Ok(GgufValue::Float32(cursor.read_f32()?)),
        7 => Ok(GgufValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(GgufValue::String(cursor.read_gguf_string()?)),
        9 => {
            let element_type = cursor.read_u32()?;
            let length = cursor.read_u64()?;
            if let Some(elem_size) = value_type_element_size(element_type) {
                cursor.read_bytes(length as usize * elem_size)?;
            } else if element_type == 8 {
                for _ in 0..length {
                    cursor.read_gguf_string()?;
                }
            } else {
                for _ in 0..length {
                    read_value(&mut *cursor)?;
                }
            }
            Ok(GgufValue::Array {
                element_type,
                length,
            })
        }
        10 => Ok(GgufValue::UInt64(cursor.read_u64()?)),
        11 => Ok(GgufValue::Int64(cursor.read_i64()?)),
        12 => Ok(GgufValue::Float64(cursor.read_f64()?)),
        _ => Err(GgufError::UnknownValueType(value_type)),
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a minimal valid GGUF v3 header with 0 tensors and 0 metadata.
    fn minimal_header() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // magic
        buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
        buf
    }

    #[test]
    fn test_invalid_magic_rejected() {
        let mut header = minimal_header();
        header[0] = 0xFF; // corrupt magic
        assert_eq!(parse_header(&header).unwrap_err(), GgufError::InvalidMagic);
    }

    #[test]
    fn test_truncated_header_rejected() {
        let header = &[0x47, 0x47, 0x55, 0x46]; // just magic, no version
        assert_eq!(parse_header(header).unwrap_err(), GgufError::UnexpectedEof);
    }

    #[test]
    fn test_version_mismatch_rejected() {
        let mut header = minimal_header();
        // Set version to 99
        header[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(
            parse_header(&header).unwrap_err(),
            GgufError::UnsupportedVersion(99)
        );
    }

    #[test]
    fn test_minimal_valid_header_parses() {
        let header = minimal_header();
        let result = parse_header(&header).unwrap();
        assert_eq!(result.version, 3);
        assert_eq!(result.tensor_count, 0);
        assert_eq!(result.metadata_kv_count, 0);
        assert!(result.metadata.is_empty());
        assert!(result.tensors.is_empty());
    }

    #[test]
    fn test_metadata_kv_string_value() {
        let mut buf = minimal_header();
        // Patch metadata_kv_count = 1
        buf[16..24].copy_from_slice(&1u64.to_le_bytes());
        // KV pair: key = "arch", value = string "gemma4"
        // Key string: len=4 + "arch"
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(b"arch");
        // Value type = 8 (string)
        buf.extend_from_slice(&8u32.to_le_bytes());
        // Value string: len=6 + "gemma4"
        buf.extend_from_slice(&6u64.to_le_bytes());
        buf.extend_from_slice(b"gemma4");

        let result = parse_header(&buf).unwrap();
        assert_eq!(result.metadata_kv_count, 1);
        assert_eq!(result.metadata[0].0, "arch");
        match &result.metadata[0].1 {
            GgufValue::String(s) => assert_eq!(s, "gemma4"),
            _ => panic!("expected string value"),
        }
    }

    #[test]
    fn test_tensor_info_extraction() {
        let mut buf = minimal_header();
        // Patch tensor_count = 1
        buf[8..16].copy_from_slice(&1u64.to_le_bytes());
        // Tensor info: name = "w1", 2 dims [4096, 4096], type F16, offset 0
        // Name string: len=2 + "w1"
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(b"w1");
        // n_dimensions = 2
        buf.extend_from_slice(&2u32.to_le_bytes());
        // dim[0] = 4096, dim[1] = 4096
        buf.extend_from_slice(&4096u64.to_le_bytes());
        buf.extend_from_slice(&4096u64.to_le_bytes());
        // tensor_type = 1 (F16)
        buf.extend_from_slice(&1u32.to_le_bytes());
        // offset = 0
        buf.extend_from_slice(&0u64.to_le_bytes());

        let result = parse_header(&buf).unwrap();
        assert_eq!(result.tensor_count, 1);
        assert_eq!(result.tensors.len(), 1);
        assert_eq!(result.tensors[0].name, "w1");
        assert_eq!(result.tensors[0].n_dimensions, 2);
        assert_eq!(result.tensors[0].dimensions, vec![4096, 4096]);
        assert_eq!(result.tensors[0].tensor_type, GgmlType::F16);
        assert_eq!(result.tensors[0].offset, 0);
    }

    #[test]
    fn test_metadata_uint32_value() {
        let mut buf = minimal_header();
        buf[16..24].copy_from_slice(&1u64.to_le_bytes());
        // Key = "n_layers"
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(b"n_layers");
        // Value type = 4 (uint32)
        buf.extend_from_slice(&4u32.to_le_bytes());
        // Value = 42
        buf.extend_from_slice(&42u32.to_le_bytes());

        let result = parse_header(&buf).unwrap();
        match &result.metadata[0].1 {
            GgufValue::UInt32(v) => assert_eq!(*v, 42),
            _ => panic!("expected uint32"),
        }
    }

    #[test]
    fn test_unknown_tensor_type_rejected() {
        let mut buf = minimal_header();
        buf[8..16].copy_from_slice(&1u64.to_le_bytes());
        // Tensor: name = "t", 0 dims, type = 255 (unknown), offset 0
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(b"t");
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 dims
        buf.extend_from_slice(&255u32.to_le_bytes()); // bad type
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        assert_eq!(
            parse_header(&buf).unwrap_err(),
            GgufError::UnknownTensorType(255)
        );
    }

    // ── MP2.1 Tests ─────────────────────────────────────────────

    /// Build a GGUF with architecture metadata + tokenizer array + tensors.
    fn build_selective_test_gguf(
        arch_kvs: Vec<(&str, GgufValue)>,
        tokenizer_kvs: Vec<(&str, u32, Vec<u8>)>, // (key, value_type, raw_value_bytes)
        tensor_infos: Vec<(&str, u32, Vec<u64>)>, // (name, type_id, dims)
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        let total_kvs = arch_kvs.len() + tokenizer_kvs.len();
        let total_tensors = tensor_infos.len();

        // Header
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
        buf.extend_from_slice(&(total_tensors as u64).to_le_bytes());
        buf.extend_from_slice(&(total_kvs as u64).to_le_bytes());

        // Architecture KVs
        for (key, val) in &arch_kvs {
            // Key string
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            // Value
            match val {
                GgufValue::String(s) => {
                    buf.extend_from_slice(&8u32.to_le_bytes());
                    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    buf.extend_from_slice(s.as_bytes());
                }
                GgufValue::UInt32(n) => {
                    buf.extend_from_slice(&4u32.to_le_bytes());
                    buf.extend_from_slice(&n.to_le_bytes());
                }
                GgufValue::Float32(f) => {
                    buf.extend_from_slice(&6u32.to_le_bytes());
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                _ => panic!("unsupported test value type"),
            }
        }

        // Tokenizer KVs (raw bytes — caller controls exact binary layout)
        for (key, vtype, raw_bytes) in &tokenizer_kvs {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&vtype.to_le_bytes());
            buf.extend_from_slice(raw_bytes);
        }

        // Tensor infos
        for (name, type_id, dims) in &tensor_infos {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&type_id.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        }

        buf
    }

    /// Helper: build the standard Qwen3-like architecture KVs.
    fn qwen3_arch_kvs() -> Vec<(&'static str, GgufValue)> {
        vec![
            (
                "general.architecture",
                GgufValue::String(String::from("qwen3")),
            ),
            ("qwen3.block_count", GgufValue::UInt32(28)),
            ("qwen3.context_length", GgufValue::UInt32(32768)),
            ("qwen3.embedding_length", GgufValue::UInt32(2048)),
            ("qwen3.feed_forward_length", GgufValue::UInt32(6144)),
            ("qwen3.attention.head_count", GgufValue::UInt32(16)),
            ("qwen3.attention.head_count_kv", GgufValue::UInt32(8)),
            ("qwen3.attention.key_length", GgufValue::UInt32(128)),
            ("qwen3.attention.value_length", GgufValue::UInt32(128)),
            ("qwen3.rope.freq_base", GgufValue::Float32(1000000.0)),
            (
                "qwen3.attention.layer_norm_rms_epsilon",
                GgufValue::Float32(0.000001),
            ),
        ]
    }

    /// Helper: build a tokenizer.ggml.tokens-like u32 array KV (raw bytes).
    fn tokenizer_u32_array_kv(key: &str, count: u64) -> (&str, u32, Vec<u8>) {
        // Value type 9 (array), element type 4 (u32), length, then count * 4 bytes
        let mut raw = Vec::new();
        raw.extend_from_slice(&4u32.to_le_bytes()); // elem type = u32
        raw.extend_from_slice(&count.to_le_bytes()); // length
        for i in 0..count {
            raw.extend_from_slice(&(i as u32).to_le_bytes());
        }
        (key, 9, raw)
    }

    /// Helper: build a tokenizer string array KV (raw bytes).
    fn tokenizer_string_array_kv(key: &str, strings: &[&str]) -> (String, u32, Vec<u8>) {
        let mut raw = Vec::new();
        raw.extend_from_slice(&8u32.to_le_bytes()); // elem type = string
        raw.extend_from_slice(&(strings.len() as u64).to_le_bytes());
        for s in strings {
            raw.extend_from_slice(&(s.len() as u64).to_le_bytes());
            raw.extend_from_slice(s.as_bytes());
        }
        (String::from(key), 9, raw)
    }

    #[test]
    fn test_selective_parse_skips_tokenizer_array() {
        let tok_kv = tokenizer_u32_array_kv("tokenizer.ggml.token_type", 100);
        let buf = build_selective_test_gguf(
            qwen3_arch_kvs(),
            vec![tok_kv],
            vec![("w1", 12, vec![256, 256])], // Q4_K tensor
        );
        let idx = parse_selective(&buf).unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.model_config.block_count, 28);
    }

    #[test]
    fn test_selective_parse_skips_string_array() {
        let (key, vtype, raw) =
            tokenizer_string_array_kv("tokenizer.ggml.tokens", &["hello", "world", "test"]);
        let buf = build_selective_test_gguf(qwen3_arch_kvs(), vec![(&key, vtype, raw)], vec![]);
        let idx = parse_selective(&buf).unwrap();
        assert_eq!(idx.model_config.architecture, "qwen3");
    }

    #[test]
    fn test_tensor_index_vec_lookup() {
        let buf = build_selective_test_gguf(
            qwen3_arch_kvs(),
            vec![],
            vec![
                ("output.weight", 14, vec![2048, 151936]),     // Q6_K
                ("blk.0.attn_q.weight", 12, vec![2048, 2048]), // Q4_K
            ],
        );
        let idx = parse_selective(&buf).unwrap();
        let t = idx.get("output.weight").unwrap();
        assert_eq!(t.tensor_type, GgmlType::Q6K);
        assert_eq!(t.dimensions, vec![2048, 151936]);
        assert_eq!(t.element_count(), 2048 * 151936);
        assert!(idx.get("nonexistent").is_none());
    }

    #[test]
    fn test_tensor_index_vec_file_order() {
        let buf = build_selective_test_gguf(
            qwen3_arch_kvs(),
            vec![],
            vec![
                ("z_last", 0, vec![1]),
                ("a_first", 0, vec![1]),
                ("m_middle", 0, vec![1]),
            ],
        );
        let idx = parse_selective(&buf).unwrap();
        // Vec preserves GGUF file order (not sorted)
        let names: Vec<&str> = idx.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["z_last", "a_first", "m_middle"]);
    }

    #[test]
    fn test_model_config_extracts_qwen3_hyperparams() {
        let buf = build_selective_test_gguf(qwen3_arch_kvs(), vec![], vec![]);
        let idx = parse_selective(&buf).unwrap();
        let cfg = &idx.model_config;
        assert_eq!(cfg.architecture, "qwen3");
        assert_eq!(cfg.block_count, 28);
        assert_eq!(cfg.context_length, 32768);
        assert_eq!(cfg.embedding_length, 2048);
        assert_eq!(cfg.feed_forward_length, 6144);
        assert_eq!(cfg.head_count, 16);
        assert_eq!(cfg.head_count_kv, 8);
        assert_eq!(cfg.key_length, 128);
        assert_eq!(cfg.value_length, 128);
        assert!((cfg.rope_freq_base - 1000000.0).abs() < 1.0);
        assert!((cfg.layer_norm_rms_epsilon - 0.000001).abs() < 0.0000001);
        // Non-MLA arch: MLA head-dim fields must stay None even if the
        // GGUF happens to carry rope.dimension_count later.
        assert!(cfg.kv_lora_rank.is_none());
        assert!(cfg.qk_rope_head_dim.is_none());
        assert!(cfg.qk_nope_head_dim.is_none());
        assert!(cfg.v_head_dim.is_none());
    }

    /// Helper: build a Kimi-K2.6-shaped deepseek2 KV block, omitting
    /// the synthetic `attention.qk_{rope,nope}_head_dim` / `attention.v_head_dim`
    /// keys (which real GGUFs don't carry) so the parser must derive
    /// them from `rope.dimension_count` / `key_length` / `value_length`.
    fn deepseek2_kimi_arch_kvs() -> Vec<(&'static str, GgufValue)> {
        vec![
            (
                "general.architecture",
                GgufValue::String(String::from("deepseek2")),
            ),
            ("deepseek2.block_count", GgufValue::UInt32(61)),
            ("deepseek2.context_length", GgufValue::UInt32(131072)),
            ("deepseek2.embedding_length", GgufValue::UInt32(7168)),
            ("deepseek2.feed_forward_length", GgufValue::UInt32(18432)),
            ("deepseek2.attention.head_count", GgufValue::UInt32(128)),
            ("deepseek2.attention.head_count_kv", GgufValue::UInt32(128)),
            ("deepseek2.attention.key_length", GgufValue::UInt32(576)),
            ("deepseek2.attention.value_length", GgufValue::UInt32(512)),
            ("deepseek2.rope.freq_base", GgufValue::Float32(10000.0)),
            (
                "deepseek2.attention.layer_norm_rms_epsilon",
                GgufValue::Float32(0.000001),
            ),
            ("deepseek2.rope.dimension_count", GgufValue::UInt32(64)),
            ("deepseek2.attention.kv_lora_rank", GgufValue::UInt32(512)),
            ("deepseek2.attention.q_lora_rank", GgufValue::UInt32(1536)),
        ]
    }

    #[test]
    fn test_model_config_derives_mla_head_dims_from_spec_keys() {
        // Regression: real deepseek2 GGUFs (DeepSeek-V2/V3, Kimi K2.6)
        // do not carry attention.qk_{rope,nope}_head_dim / v_head_dim
        // as explicit metadata. The parser derives them:
        //   qk_rope_head_dim  = rope.dimension_count (64)
        //   qk_nope_head_dim  = attn_kv_b.shape[0] / n_heads / 2 (128)
        //   v_head_dim        = attn_kv_b.shape[0] / n_heads / 2 (128)
        // Note: key_length (576) = kv_lora_rank (512) + qk_rope (64)
        //       — NOT qk_nope + qk_rope. The actual per-head dims come
        //       from the tensor shape of blk.0.attn_kv_b.weight.
        let tensors = vec![
            // shape: [n_heads*(qk_nope+v_head), kv_lora_rank] = [128*(128+128), 512] = [32768, 512]
            // Use blk.1 — layer 0 is often dense (no MLA tensors) in Kimi K2.6
            ("blk.1.attn_kv_b.weight", 0u32, vec![32768u64, 512u64]),
        ];
        let buf = build_selective_test_gguf(deepseek2_kimi_arch_kvs(), vec![], tensors);
        let idx = parse_selective(&buf).unwrap();
        let cfg = &idx.model_config;
        assert_eq!(cfg.architecture, "deepseek2");
        assert_eq!(cfg.kv_lora_rank, Some(512));
        assert_eq!(cfg.q_lora_rank, Some(1536));
        // qk_rope = rope.dimension_count
        assert_eq!(cfg.qk_rope_head_dim, Some(64));
        // qk_nope = attn_kv_b.shape[0] / n_heads / 2 = 32768 / 128 / 2 = 128
        assert_eq!(cfg.qk_nope_head_dim, Some(128));
        // v_head_dim = same derivation = 128
        assert_eq!(cfg.v_head_dim, Some(128));
        assert!(cfg.is_mla());
        assert!(cfg.is_deepseek2());
    }

    #[test]
    fn test_model_config_derives_mla_dims_from_split_k_v_tensors() {
        // Kimi K2.6 (and newer GGUF converters) split the combined
        // attn_kv_b into separate attn_k_b and attn_v_b tensors:
        //   attn_k_b.weight shape = [n_heads * qk_nope_head_dim, kv_lora_rank]
        //   attn_v_b.weight shape = [n_heads * v_head_dim, kv_lora_rank]
        let tensors = vec![
            // attn_k_b: [128*128, 512] = [16384, 512]  (n_heads=128 from kvs)
            ("blk.1.attn_k_b.weight", 0u32, vec![16384u64, 512u64]),
            // attn_v_b: [128*128, 512] = [16384, 512]
            ("blk.1.attn_v_b.weight", 0u32, vec![16384u64, 512u64]),
        ];
        let buf = build_selective_test_gguf(deepseek2_kimi_arch_kvs(), vec![], tensors);
        let idx = parse_selective(&buf).unwrap();
        let cfg = &idx.model_config;
        assert_eq!(cfg.architecture, "deepseek2");
        assert_eq!(cfg.kv_lora_rank, Some(512));
        // qk_nope = attn_k_b.shape[0] / n_heads = 16384 / 128 = 128
        assert_eq!(cfg.qk_nope_head_dim, Some(128));
        // v_head = attn_v_b.shape[0] / n_heads = 16384 / 128 = 128
        assert_eq!(cfg.v_head_dim, Some(128));
        assert_eq!(cfg.qk_rope_head_dim, Some(64));
        assert!(cfg.is_mla());
    }

    #[test]
    fn test_model_config_derives_mla_dims_from_3d_split_tensors() {
        // Regression: unsloth's Kimi K2.6 GGUF converter stores attn_k_b
        // and attn_v_b as 3D tensors (e.g. [128, 512, 64]) instead of 2D
        // ([n_heads * per_head_dim, kv_lora_rank]). The parser must derive
        // correct per-head dims from the element count, not by dividing
        // dimensions[0] by n_heads (which gives a wrong value for 3D).
        //
        // Real Kimi K2.6 shapes observed on-device:
        //   attn_k_b: shape=[128, 512, 64]  = qk_nope × kv_lora_rank × n_heads
        //   attn_v_b: shape=[512, 128, 64]  = kv_lora_rank × v_head × n_heads
        //
        // Use n_heads=64 (matching the real unsloth GGUF).
        let mut kvs = deepseek2_kimi_arch_kvs();
        // Override head_count to 64 (real Kimi K2.6 unsloth GGUF value)
        kvs.retain(|(k, _)| {
            *k != "deepseek2.attention.head_count" && *k != "deepseek2.attention.head_count_kv"
        });
        kvs.push(("deepseek2.attention.head_count", GgufValue::UInt32(64)));
        kvs.push(("deepseek2.attention.head_count_kv", GgufValue::UInt32(1)));

        let tensors = vec![
            // 3D attn_k_b: [128, 512, 64] = 4,194,304 elements
            // qk_nope = 4194304 / 64 / 512 = 128
            ("blk.1.attn_k_b.weight", 0u32, vec![128u64, 512u64, 64u64]),
            // 3D attn_v_b: [512, 128, 64] = 4,194,304 elements
            // v_head = 4194304 / 64 / 512 = 128
            ("blk.1.attn_v_b.weight", 0u32, vec![512u64, 128u64, 64u64]),
        ];
        let buf = build_selective_test_gguf(kvs, vec![], tensors);
        let idx = parse_selective(&buf).unwrap();
        let cfg = &idx.model_config;
        assert_eq!(cfg.architecture, "deepseek2");
        assert_eq!(cfg.head_count, 64);
        assert_eq!(cfg.kv_lora_rank, Some(512));
        // Correct derivation from 3D element count, not dimensions[0]/n_heads
        assert_eq!(cfg.qk_nope_head_dim, Some(128));
        assert_eq!(cfg.v_head_dim, Some(128));
        assert_eq!(cfg.qk_rope_head_dim, Some(64));
        assert!(cfg.is_mla());
    }

    #[test]
    fn test_model_config_deepseek2_fallback_without_tensors() {
        // When no attn_kv_b / attn_k_b / attn_v_b tensors exist at all,
        // the DeepSeek2 hardcoded fallback (128/128) kicks in.
        let buf = build_selective_test_gguf(deepseek2_kimi_arch_kvs(), vec![], vec![]);
        let idx = parse_selective(&buf).unwrap();
        let cfg = &idx.model_config;
        assert_eq!(cfg.architecture, "deepseek2");
        assert_eq!(cfg.kv_lora_rank, Some(512));
        // Fallback: DeepSeek2 architectural constants
        assert_eq!(cfg.qk_nope_head_dim, Some(128));
        assert_eq!(cfg.v_head_dim, Some(128));
        assert_eq!(cfg.qk_rope_head_dim, Some(64));
        assert!(cfg.is_mla());
    }

    #[test]
    fn test_model_config_prefers_explicit_mla_keys_over_derivation() {
        // If a future producer ever emits the explicit
        // attention.qk_{rope,nope}_head_dim / v_head_dim keys, those
        // must take precedence over the spec-based derivation.
        let mut kvs = deepseek2_kimi_arch_kvs();
        kvs.push((
            "deepseek2.attention.qk_rope_head_dim",
            GgufValue::UInt32(99),
        ));
        kvs.push((
            "deepseek2.attention.qk_nope_head_dim",
            GgufValue::UInt32(77),
        ));
        kvs.push(("deepseek2.attention.v_head_dim", GgufValue::UInt32(55)));
        let buf = build_selective_test_gguf(kvs, vec![], vec![]);
        let idx = parse_selective(&buf).unwrap();
        let cfg = &idx.model_config;
        assert_eq!(cfg.qk_rope_head_dim, Some(99));
        assert_eq!(cfg.qk_nope_head_dim, Some(77));
        assert_eq!(cfg.v_head_dim, Some(55));
    }

    #[test]
    fn test_model_config_derives_mla_dims_from_mla_length_keys() {
        // Kimi K2.6 ships `attention.key_length_mla` (= qk_nope + qk_rope)
        // and `attention.value_length_mla` (= v_head) directly. The parser
        // must derive per-head dims from those WITHOUT inspecting any
        // tensor — even if no attn_kv_b / attn_k_b / attn_v_b tensors
        // exist in this fixture.
        let mut kvs = deepseek2_kimi_arch_kvs();
        // 192 = qk_nope(128) + qk_rope(64); 128 = v_head
        kvs.push(("deepseek2.attention.key_length_mla", GgufValue::UInt32(192)));
        kvs.push((
            "deepseek2.attention.value_length_mla",
            GgufValue::UInt32(128),
        ));
        let buf = build_selective_test_gguf(kvs, vec![], vec![]);
        let idx = parse_selective(&buf).unwrap();
        let cfg = &idx.model_config;
        assert_eq!(cfg.key_length_mla, Some(192));
        assert_eq!(cfg.value_length_mla, Some(128));
        // qk_rope = rope.dimension_count = 64
        assert_eq!(cfg.qk_rope_head_dim, Some(64));
        // qk_nope = key_length_mla - qk_rope = 192 - 64 = 128
        assert_eq!(cfg.qk_nope_head_dim, Some(128));
        // v_head = value_length_mla = 128
        assert_eq!(cfg.v_head_dim, Some(128));
        assert!(cfg.is_mla());
    }

    #[test]
    fn test_mla_length_keys_take_precedence_over_tensor_shape() {
        // If both MLA length keys and tensor shapes are present, the
        // length keys win — they are spec-authoritative and avoid
        // assumptions about tensor layout (combined vs split).
        let mut kvs = deepseek2_kimi_arch_kvs();
        kvs.push(("deepseek2.attention.key_length_mla", GgufValue::UInt32(192)));
        kvs.push((
            "deepseek2.attention.value_length_mla",
            GgufValue::UInt32(128),
        ));
        // Tensor shape would imply 64/64 — should be ignored.
        let tensors = vec![("blk.1.attn_kv_b.weight", 0u32, vec![16384u64, 512u64])];
        let buf = build_selective_test_gguf(kvs, vec![], tensors);
        let idx = parse_selective(&buf).unwrap();
        let cfg = &idx.model_config;
        assert_eq!(cfg.qk_nope_head_dim, Some(128));
        assert_eq!(cfg.v_head_dim, Some(128));
    }

    #[test]
    fn test_model_config_qwen3_keeps_mla_fields_none_even_with_rope_dim() {
        // Defensive: even if a Qwen3 GGUF carries `qwen3.rope.dimension_count`,
        // the fallback derivation is gated on `kv_lora_rank.is_some()` so
        // non-MLA architectures retain None.
        let mut kvs = qwen3_arch_kvs();
        kvs.push(("qwen3.rope.dimension_count", GgufValue::UInt32(128)));
        let buf = build_selective_test_gguf(kvs, vec![], vec![]);
        let idx = parse_selective(&buf).unwrap();
        let cfg = &idx.model_config;
        assert!(cfg.kv_lora_rank.is_none());
        assert!(cfg.qk_rope_head_dim.is_none());
        assert!(cfg.qk_nope_head_dim.is_none());
        assert!(cfg.v_head_dim.is_none());
        assert!(!cfg.is_mla());
    }

    #[test]
    fn test_parse_selective_does_not_materialise_tokens() {
        // Regression: parse_selective is the hot path the kernel takes
        // at boot to learn the architecture. It MUST NOT materialise
        // tokenizer.ggml.tokens — Qwen3 ships ~152 k tokens and the
        // kernel runtime arena is 2 MiB. Materialising would OOM the
        // boot. The extractor below is the deepseek2-only path.
        fn tokens_kv(key: &'static str, strings: &[&str]) -> (&'static str, u32, Vec<u8>) {
            let mut raw = Vec::new();
            raw.extend_from_slice(&8u32.to_le_bytes());
            raw.extend_from_slice(&(strings.len() as u64).to_le_bytes());
            for s in strings {
                raw.extend_from_slice(&(s.len() as u64).to_le_bytes());
                raw.extend_from_slice(s.as_bytes());
            }
            (key, 9, raw)
        }
        let tok_array = tokens_kv("tokenizer.ggml.tokens", &["a", "b", "c"]);
        let buf = build_selective_test_gguf(qwen3_arch_kvs(), vec![tok_array], vec![]);
        let idx = parse_selective(&buf).unwrap();
        assert!(idx.tokenizer.is_none());
    }

    #[test]
    fn test_extract_tokenizer_from_bytes_materialises_tokens() {
        // The standalone extractor IS the path that materialises the
        // tokens. Kernel deepseek2 boot calls it after architecture
        // detection rules out qwen3.
        fn tokens_kv(key: &'static str, strings: &[&str]) -> (&'static str, u32, Vec<u8>) {
            let mut raw = Vec::new();
            raw.extend_from_slice(&8u32.to_le_bytes());
            raw.extend_from_slice(&(strings.len() as u64).to_le_bytes());
            for s in strings {
                raw.extend_from_slice(&(s.len() as u64).to_le_bytes());
                raw.extend_from_slice(s.as_bytes());
            }
            (key, 9, raw)
        }
        let tok_array = tokens_kv("tokenizer.ggml.tokens", &["<bos>", "hello", "world"]);
        let mut kvs = qwen3_arch_kvs();
        kvs.push(("tokenizer.ggml.eos_token_id", GgufValue::UInt32(2)));
        kvs.push(("tokenizer.ggml.bos_token_id", GgufValue::UInt32(0)));
        kvs.push((
            "tokenizer.ggml.model",
            GgufValue::String(String::from("gpt2")),
        ));
        let buf = build_selective_test_gguf(kvs, vec![tok_array], vec![]);
        let tok = extract_tokenizer_from_bytes(&buf)
            .expect("extract_tokenizer_from_bytes parse succeeded")
            .expect("tokens present in the GGUF");
        assert_eq!(tok.len(), 3);
        assert_eq!(tok.token(0), Some(b"<bos>" as &[u8]));
        assert_eq!(tok.token(1), Some(b"hello" as &[u8]));
        assert_eq!(tok.token(2), Some(b"world" as &[u8]));
        assert!(tok.token(3).is_none());
        assert_eq!(tok.model.as_deref(), Some("gpt2"));
        assert_eq!(tok.eos_token_id, Some(2));
        assert_eq!(tok.bos_token_id, Some(0));
        assert_eq!(tok.bytes, b"<bos>helloworld");
        assert_eq!(tok.offsets, vec![0, 5, 10, 15]);
    }

    #[test]
    fn test_extract_tokenizer_returns_none_when_tokens_absent() {
        let buf = build_selective_test_gguf(qwen3_arch_kvs(), vec![], vec![]);
        let tok = extract_tokenizer_from_bytes(&buf).unwrap();
        assert!(tok.is_none());
    }

    #[test]
    fn test_model_config_extracts_eos_token_id_when_present() {
        // GGUF carries `tokenizer.ggml.eos_token_id` at the top level
        // (NOT prefixed with the architecture). Verify the parser picks
        // it up regardless of architecture name.
        let mut kvs = qwen3_arch_kvs();
        kvs.push(("tokenizer.ggml.eos_token_id", GgufValue::UInt32(151_643)));
        let buf = build_selective_test_gguf(kvs, vec![], vec![]);
        let idx = parse_selective(&buf).unwrap();
        assert_eq!(idx.model_config.eos_token_id, Some(151_643));
    }

    #[test]
    fn test_model_config_eos_token_id_absent_is_none() {
        // No `tokenizer.ggml.eos_token_id` in metadata → field stays
        // None so the kernel can fall back to per-arch defaults.
        let buf = build_selective_test_gguf(qwen3_arch_kvs(), vec![], vec![]);
        let idx = parse_selective(&buf).unwrap();
        assert_eq!(idx.model_config.eos_token_id, None);
    }

    #[test]
    fn test_q6k_filter_count() {
        let buf = build_selective_test_gguf(
            qwen3_arch_kvs(),
            vec![],
            vec![
                ("output.weight", 14, vec![2048, 151936]),
                ("blk.0.attn_v.weight", 14, vec![2048, 1024]),
                ("blk.0.attn_q.weight", 12, vec![2048, 2048]),
                ("blk.0.ffn_down.weight", 14, vec![6144, 2048]),
                ("norm.weight", 0, vec![2048]),
            ],
        );
        let idx = parse_selective(&buf).unwrap();
        let q6k_count = idx.iter_by_type(GgmlType::Q6K).count();
        assert_eq!(q6k_count, 3);
        let q4k_count = idx.iter_by_type(GgmlType::Q4K).count();
        assert_eq!(q4k_count, 1);
    }

    #[test]
    fn test_skip_value_array_advances_correctly() {
        // Build GGUF with tokenizer array between two arch KVs
        // If skip is off-by-N, the second KV will parse incorrectly
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        buf.extend_from_slice(&3u64.to_le_bytes()); // 3 KVs

        // KV 1: general.architecture = "test"
        buf.extend_from_slice(&24u64.to_le_bytes());
        buf.extend_from_slice(b"general.architecture    ");
        // Fix: use exact length
        let key1 = b"general.architecture";
        buf.truncate(buf.len() - 24 - 8); // undo
        buf.extend_from_slice(&(key1.len() as u64).to_le_bytes());
        buf.extend_from_slice(key1);
        buf.extend_from_slice(&8u32.to_le_bytes()); // string
        let val1 = b"test";
        buf.extend_from_slice(&(val1.len() as u64).to_le_bytes());
        buf.extend_from_slice(val1);

        // KV 2: tokenizer.ggml.token_type = array of 50 u32s
        let key2 = b"tokenizer.ggml.token_type";
        buf.extend_from_slice(&(key2.len() as u64).to_le_bytes());
        buf.extend_from_slice(key2);
        buf.extend_from_slice(&9u32.to_le_bytes()); // array
        buf.extend_from_slice(&4u32.to_le_bytes()); // u32 elements
        buf.extend_from_slice(&50u64.to_le_bytes()); // 50 elements
        for i in 0..50u32 {
            buf.extend_from_slice(&i.to_le_bytes());
        }

        // KV 3: test.block_count = 7
        let key3 = b"test.block_count";
        buf.extend_from_slice(&(key3.len() as u64).to_le_bytes());
        buf.extend_from_slice(key3);
        buf.extend_from_slice(&4u32.to_le_bytes()); // u32
        buf.extend_from_slice(&7u32.to_le_bytes());

        // parse_selective will skip KV2, so we need full model config.
        // But ModelConfig requires many keys — let's just test skip
        // correctness by using parse_header which reads all.
        let result = parse_header(&buf).unwrap();
        assert_eq!(result.metadata_kv_count, 3);
        // Verify KV3 parsed correctly after the array skip
        match &result.metadata[2].1 {
            GgufValue::UInt32(v) => assert_eq!(*v, 7),
            _ => panic!("KV3 value wrong after array skip"),
        }
    }

    #[test]
    fn test_missing_metadata_error() {
        // No architecture KV → MissingMetadata
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(
            parse_selective(&buf).unwrap_err(),
            GgufError::MissingMetadata
        );
    }
}
