// SPDX-License-Identifier: AGPL-3.0-or-later
//! Boot-LLM Model Loader abstraction.
//!
//! Per V3 Pillar 7 (Platform Independence) and Pillar 4 (LLM Strategy):
//! model loading mechanism must abstract over hardware/source.
//!
//! Two MP1 loaders are provided:
//!
//! * [`DirectMemoryLoader`] — QEMU `-device loader` places the model file
//!   at a fixed guest physical address (typically 0x100000000) before
//!   kernel entry. Useful in QEMU dev workflows but unavailable on real
//!   hardware.
//! * [`RamdiskLoader`] — bootloader 0.11 `set_ramdisk()` embeds the model
//!   into the boot image as a "ramdisk module". The bootloader maps it
//!   into the kernel address space and reports the virtual address +
//!   length in `BootInfo::ramdisk_addr` / `ramdisk_len`. This is the
//!   production path for bare-metal x86_64.
//!
//! Both loaders implement the [`ModelLoader`] trait; the kernel boot
//! path tries ramdisk first, then DirectMemory, and finally falls
//! through with a "model-absent → Stage 11 skipped" notice so the rest
//! of the kernel can come up without an LLM.
//!
//! Per ADR-028 MP6: trait formalization deferred until MP6, but the
//! stub already gives V3 Pillar 7 its abstraction boundary.

#[allow(unused_imports)]
use core::fmt::Write;
use alloc::string::String;
use alloc::vec::Vec;

/// GGUF magic (little-endian "GGUF") — required for plausibility checks
/// on any model byte region the kernel chooses to use.
pub const GGUF_MAGIC: u32 = 0x46554747;

/// SilicatePack model magic (little-endian "SILM"). This is the native
/// Zero Server container marker. Native `.smodel` artifacts carry their
/// own tensor directory, quant metadata, tokenizer/config sections, and
/// aligned tensor payload; they are not GGUF wrappers. Payload kind 1 is
/// retained only for legacy GGUF compatibility benchmarks.
pub const SMODEL_MAGIC: u32 = 0x4D4C4953;
pub const SMODEL_VERSION_1: u32 = 1;
/// `.smodel`-v2: identical container layout; version 2 signals that the
/// artifact MAY carry row-interleaved tensor dtypes (see
/// [`SMODEL_DTYPE_Q4_0X4`]). Old kernels reject v2 outright instead of
/// misreading interleaved bytes through plain-layout kernels.
pub const SMODEL_VERSION_2: u32 = 2;
pub const SMODEL_HEADER_SIZE: usize = 128;
pub const SMODEL_PAYLOAD_KIND_GGUF_COMPAT: u32 = 1;
pub const SMODEL_PAYLOAD_KIND_NATIVE: u32 = 2;
pub const SMODEL_PAYLOAD_ALIGNMENT: usize = 2 * 1024 * 1024;
pub const SMODEL_NATIVE_INDEX_MAGIC: u32 = 0x5844_4953; // "SIDX"
pub const SMODEL_NATIVE_INDEX_VERSION_1: u32 = 1;
pub const SMODEL_NATIVE_INDEX_HEADER_SIZE: usize = 128;
pub const SMODEL_NATIVE_TENSOR_ENTRY_SIZE: usize = 104;
/// Sanity cap for the SIDX tensor count. A malformed/corrupted SIDX
/// header with an absurd count would otherwise drive
/// `Vec::with_capacity(tensor_count)` into the 8 MiB kernel arena and
/// panic via `handle_alloc_error` at boot. Kimi K2.6 has ~1036
/// tensors; 10 000 leaves an order of magnitude of headroom while
/// keeping the worst-case index allocation bounded.
pub const SMODEL_MAX_TENSOR_COUNT: usize = 10_000;
const SMODEL_NONE_U32: u32 = 0xFFFF_FFFF;

/// SIDX dtype id: Q4_0 payload stored 4-row-interleaved (`.smodel`-v2,
/// SilicatePack `--interleave 4`). Same bytes-per-tensor as Q4_0; groups
/// of 4 output rows interleave per K-block as `d0 d1 d2 d3 qs0 qs1 qs2
/// qs3` (72-byte group-blocks). Outside the GGML id space (≤ 39) on
/// purpose — a GGUF can never carry it.
pub const SMODEL_DTYPE_Q4_0X4: u32 = 100;
/// SIDX dtype id: Q8_0 payload stored 4-row-interleaved (136-byte
/// group-blocks). See [`SMODEL_DTYPE_Q4_0X4`].
pub const SMODEL_DTYPE_Q8_0X4: u32 = 101;

/// Decode a SIDX dtype id into `(GgmlType, interleave_group)`.
///
/// The interleaved ids map onto their plain `GgmlType` for everything
/// downstream that only cares about block geometry (bytes-per-row,
/// dequant math) — byte totals are identical. The layout difference is
/// carried separately via `crate::weight_layout`, because the sacred
/// `GgmlType` enum cannot grow variants.
fn smodel_decode_dtype(dtype: u32) -> Result<(zero_gguf_parser::GgmlType, u32), LoadError> {
    match dtype {
        SMODEL_DTYPE_Q4_0X4 => Ok((zero_gguf_parser::GgmlType::Q4_0, 4)),
        SMODEL_DTYPE_Q8_0X4 => Ok((zero_gguf_parser::GgmlType::Q8_0, 4)),
        other => zero_gguf_parser::GgmlType::from_u32(other)
            .map(|t| (t, 1))
            .map_err(|_| LoadError::UnsupportedFormat),
    }
}

/// True when this build's forward-pass kernels can execute
/// row-interleaved weights. Only the x86_64 AVX-512 path has the
/// `linear_q4_0x4` / `linear_q8_0x4` kernels; the sacred scalar and
/// NEON paths would misread the bytes, so loads must fail hard there.
const fn interleaved_layout_supported() -> bool {
    cfg!(all(target_arch = "x86_64", feature = "avx512-acceleration"))
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ModelMagic {
    Gguf,
    Smodel,
    Unknown(u32),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    SmodelGgufCompat,
    SmodelNative,
}

#[derive(Debug, Copy, Clone)]
pub struct ModelView {
    pub bytes: &'static [u8],
    pub size: usize,
    pub format: ModelFormat,
    pub payload_offset: usize,
}

#[derive(Debug, Copy, Clone)]
pub struct SmodelInfo {
    pub header_len: usize,
    pub manifest_offset: usize,
    pub manifest_len: usize,
    pub payload_offset: usize,
    pub payload_len: usize,
    pub payload_kind: u32,
    pub flags: u32,
}

#[derive(Debug, Copy, Clone)]
pub struct SmodelValidationAnchor {
    pub strict: bool,
    pub matched_runtime: bool,
    pub expected_next_token: Option<u32>,
    pub expected_logit_bits: Option<u32>,
}

#[derive(Debug, Copy, Clone)]
pub struct SmodelNativeSummary {
    pub tensor_count: usize,
    pub entry_size: usize,
    pub names_len: usize,
    pub data_base: u64,
}

/// Errors during model loading.
#[derive(Debug)]
#[allow(dead_code)]
pub enum LoadError {
    InvalidAddress,
    MagicMismatch,
    SizeOutOfRange,
    UnsupportedFormat,
    InvalidContainer,
    /// No model was discoverable in BootInfo (no ramdisk, no DirectMemory).
    Absent,
}

#[inline]
fn read_u32_le(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    Some(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

#[inline]
fn read_u64_le(bytes: &[u8], offset: usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    if end > bytes.len() {
        return None;
    }
    Some(u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ]))
}

#[inline]
fn read_f32_le(bytes: &[u8], offset: usize) -> Option<f32> {
    read_u32_le(bytes, offset).map(f32::from_bits)
}

#[inline]
fn smodel_u32_opt(value: u32) -> Option<u32> {
    if value == SMODEL_NONE_U32 {
        None
    } else {
        Some(value)
    }
}

pub fn model_magic_from_bytes(bytes: &[u8]) -> ModelMagic {
    match read_u32_le(bytes, 0) {
        Some(GGUF_MAGIC) => ModelMagic::Gguf,
        Some(SMODEL_MAGIC) => ModelMagic::Smodel,
        Some(other) => ModelMagic::Unknown(other),
        None => ModelMagic::Unknown(0),
    }
}

#[inline]
pub fn looks_like_supported_model(bytes: &[u8]) -> bool {
    matches!(
        model_magic_from_bytes(bytes),
        ModelMagic::Gguf | ModelMagic::Smodel
    )
}

/// Return the GGUF payload the current compatibility runtime understands.
/// Raw GGUF is accepted only as a legacy path. Native `.smodel` artifacts
/// deliberately do not unwrap here: they must enter the native SilicatePack
/// graph loader, not the GGUF parser hot path.
pub fn gguf_payload_view(bytes: &'static [u8]) -> Result<ModelView, LoadError> {
    match model_magic_from_bytes(bytes) {
        ModelMagic::Gguf => Ok(ModelView {
            bytes,
            size: bytes.len(),
            format: ModelFormat::Gguf,
            payload_offset: 0,
        }),
        ModelMagic::Smodel => {
            let info = smodel_info(bytes)?;
            if info.payload_kind == SMODEL_PAYLOAD_KIND_NATIVE {
                return Err(LoadError::UnsupportedFormat);
            }
            smodel_gguf_compat_payload_view(bytes, info)
        }
        ModelMagic::Unknown(_) => Err(LoadError::MagicMismatch),
    }
}

pub fn smodel_payload_kind_label(payload_kind: u32) -> &'static str {
    match payload_kind {
        SMODEL_PAYLOAD_KIND_NATIVE => "native-smodel",
        SMODEL_PAYLOAD_KIND_GGUF_COMPAT => "gguf-compat",
        _ => "unknown",
    }
}

pub fn smodel_native_summary(bytes: &[u8]) -> Result<SmodelNativeSummary, LoadError> {
    let info = smodel_info(bytes)?;
    if info.payload_kind != SMODEL_PAYLOAD_KIND_NATIVE {
        return Err(LoadError::UnsupportedFormat);
    }

    let payload_start = info.payload_offset;
    let payload_end = info
        .payload_offset
        .checked_add(info.payload_len)
        .ok_or(LoadError::InvalidContainer)?;
    if payload_end > bytes.len()
        || payload_start
            .checked_add(SMODEL_NATIVE_INDEX_HEADER_SIZE)
            .ok_or(LoadError::InvalidContainer)?
            > payload_end
    {
        return Err(LoadError::InvalidContainer);
    }

    let payload = &bytes[payload_start..payload_end];
    if read_u32_le(payload, 0) != Some(SMODEL_NATIVE_INDEX_MAGIC)
        || read_u32_le(payload, 4) != Some(SMODEL_NATIVE_INDEX_VERSION_1)
        || read_u32_le(payload, 8) != Some(SMODEL_NATIVE_INDEX_HEADER_SIZE as u32)
    {
        return Err(LoadError::UnsupportedFormat);
    }

    let tensor_count = read_u32_le(payload, 12).ok_or(LoadError::InvalidContainer)? as usize;
    if tensor_count > SMODEL_MAX_TENSOR_COUNT {
        let _ = writeln!(
            crate::arch::serial::Serial,
            "SIDX rejected: tensor_count {} exceeds sanity cap {} — malformed index",
            tensor_count,
            SMODEL_MAX_TENSOR_COUNT,
        );
        return Err(LoadError::InvalidContainer);
    }
    let entry_size = read_u32_le(payload, 16).ok_or(LoadError::InvalidContainer)? as usize;
    let names_offset = read_u32_le(payload, 20).ok_or(LoadError::InvalidContainer)? as usize;
    let names_len = read_u32_le(payload, 24).ok_or(LoadError::InvalidContainer)? as usize;
    let data_base = read_u64_le(payload, 28).ok_or(LoadError::InvalidContainer)?;
    if entry_size != SMODEL_NATIVE_TENSOR_ENTRY_SIZE {
        return Err(LoadError::UnsupportedFormat);
    }

    let payload_len = payload_end - payload_start;
    let entries_start = SMODEL_NATIVE_INDEX_HEADER_SIZE;
    let entries_len = tensor_count
        .checked_mul(entry_size)
        .ok_or(LoadError::InvalidContainer)?;
    let entries_end = entries_start
        .checked_add(entries_len)
        .ok_or(LoadError::InvalidContainer)?;
    let names_end = names_offset
        .checked_add(names_len)
        .ok_or(LoadError::InvalidContainer)?;
    if entries_end > payload_len || names_offset < entries_end || names_end > payload_len {
        return Err(LoadError::InvalidContainer);
    }
    let names = &payload[names_offset..names_end];

    let mut i = 0usize;
    while i < tensor_count {
        let off = entries_start + i * entry_size;
        let name_off = read_u32_le(payload, off).ok_or(LoadError::InvalidContainer)? as usize;
        let name_len = read_u32_le(payload, off + 4).ok_or(LoadError::InvalidContainer)? as usize;
        let dtype = read_u32_le(payload, off + 8).ok_or(LoadError::InvalidContainer)?;
        let rank = read_u32_le(payload, off + 12).ok_or(LoadError::InvalidContainer)? as usize;
        if rank > 8 {
            return Err(LoadError::InvalidContainer);
        }
        let name_end = name_off
            .checked_add(name_len)
            .ok_or(LoadError::InvalidContainer)?;
        if name_end > names.len() || core::str::from_utf8(&names[name_off..name_end]).is_err() {
            return Err(LoadError::InvalidContainer);
        }
        let (_, interleave) = smodel_decode_dtype(dtype)?;
        if interleave > 1 && !interleaved_layout_supported() {
            let _ = writeln!(
                crate::arch::serial::Serial,
                "SIDX rejected: row-interleaved dtype {} requires the AVX-512 build — \
                 this runtime path has no interleaved kernels",
                dtype,
            );
            return Err(LoadError::UnsupportedFormat);
        }

        let tensor_payload_offset =
            read_u64_le(payload, off + 80).ok_or(LoadError::InvalidContainer)? as usize;
        let byte_len = read_u64_le(payload, off + 88).ok_or(LoadError::InvalidContainer)? as usize;
        let tensor_end = tensor_payload_offset
            .checked_add(byte_len)
            .ok_or(LoadError::InvalidContainer)?;
        if tensor_end > payload_len {
            return Err(LoadError::InvalidContainer);
        }
        i += 1;
    }

    Ok(SmodelNativeSummary {
        tensor_count,
        entry_size,
        names_len,
        data_base,
    })
}

pub fn smodel_native_tensor_index(
    bytes: &'static [u8],
) -> Result<zero_gguf_parser::TensorIndex, LoadError> {
    let info = smodel_info(bytes)?;
    if info.payload_kind != SMODEL_PAYLOAD_KIND_NATIVE {
        return Err(LoadError::UnsupportedFormat);
    }

    let payload_start = info.payload_offset;
    let payload_end = info
        .payload_offset
        .checked_add(info.payload_len)
        .ok_or(LoadError::InvalidContainer)?;
    if payload_end > bytes.len()
        || payload_start
            .checked_add(SMODEL_NATIVE_INDEX_HEADER_SIZE)
            .ok_or(LoadError::InvalidContainer)?
            > payload_end
    {
        return Err(LoadError::InvalidContainer);
    }

    let header = &bytes[payload_start..payload_end];
    if read_u32_le(header, 0) != Some(SMODEL_NATIVE_INDEX_MAGIC)
        || read_u32_le(header, 4) != Some(SMODEL_NATIVE_INDEX_VERSION_1)
        || read_u32_le(header, 8) != Some(SMODEL_NATIVE_INDEX_HEADER_SIZE as u32)
    {
        return Err(LoadError::UnsupportedFormat);
    }

    let tensor_count = read_u32_le(header, 12).ok_or(LoadError::InvalidContainer)? as usize;
    if tensor_count > SMODEL_MAX_TENSOR_COUNT {
        let _ = writeln!(
            crate::arch::serial::Serial,
            "SIDX rejected: tensor_count {} exceeds sanity cap {} — malformed index",
            tensor_count,
            SMODEL_MAX_TENSOR_COUNT,
        );
        return Err(LoadError::InvalidContainer);
    }
    let entry_size = read_u32_le(header, 16).ok_or(LoadError::InvalidContainer)? as usize;
    let names_offset = read_u32_le(header, 20).ok_or(LoadError::InvalidContainer)? as usize;
    let names_len = read_u32_le(header, 24).ok_or(LoadError::InvalidContainer)? as usize;
    let _data_base = read_u64_le(header, 28).ok_or(LoadError::InvalidContainer)? as usize;
    if entry_size != SMODEL_NATIVE_TENSOR_ENTRY_SIZE {
        return Err(LoadError::UnsupportedFormat);
    }

    let entries_start = SMODEL_NATIVE_INDEX_HEADER_SIZE;
    let entries_len = tensor_count
        .checked_mul(entry_size)
        .ok_or(LoadError::InvalidContainer)?;
    let entries_end = entries_start
        .checked_add(entries_len)
        .ok_or(LoadError::InvalidContainer)?;
    let names_end = names_offset
        .checked_add(names_len)
        .ok_or(LoadError::InvalidContainer)?;
    if entries_end > payload_end - payload_start
        || names_offset < entries_end
        || names_end > payload_end - payload_start
    {
        return Err(LoadError::InvalidContainer);
    }
    let names = &header[names_offset..names_end];

    let arch_id = read_u32_le(header, 36).ok_or(LoadError::InvalidContainer)?;
    let architecture = match arch_id {
        1 => String::from("qwen3"),
        2 => String::from("deepseek2"),
        _ => String::from("unknown"),
    };

    let expert_weights_scale_raw = read_f32_le(header, 124).ok_or(LoadError::InvalidContainer)?;
    let expert_weights_scale = if expert_weights_scale_raw.is_finite()
        && expert_weights_scale_raw > 0.0
    {
        Some(expert_weights_scale_raw)
    } else {
        None
    };

    let model_config = zero_gguf_parser::ModelConfig {
        architecture,
        block_count: read_u32_le(header, 40).ok_or(LoadError::InvalidContainer)?,
        context_length: read_u32_le(header, 44).ok_or(LoadError::InvalidContainer)?,
        embedding_length: read_u32_le(header, 48).ok_or(LoadError::InvalidContainer)?,
        feed_forward_length: read_u32_le(header, 52).ok_or(LoadError::InvalidContainer)?,
        head_count: read_u32_le(header, 56).ok_or(LoadError::InvalidContainer)?,
        head_count_kv: read_u32_le(header, 60).ok_or(LoadError::InvalidContainer)?,
        key_length: read_u32_le(header, 64).ok_or(LoadError::InvalidContainer)?,
        value_length: read_u32_le(header, 68).ok_or(LoadError::InvalidContainer)?,
        rope_freq_base: read_f32_le(header, 72).ok_or(LoadError::InvalidContainer)?,
        layer_norm_rms_epsilon: read_f32_le(header, 76).ok_or(LoadError::InvalidContainer)?,
        vocab_size: smodel_u32_opt(read_u32_le(header, 80).ok_or(LoadError::InvalidContainer)?),
        eos_token_id: smodel_u32_opt(read_u32_le(header, 84).ok_or(LoadError::InvalidContainer)?),
        expert_count: smodel_u32_opt(read_u32_le(header, 88).ok_or(LoadError::InvalidContainer)?),
        expert_used_count: smodel_u32_opt(
            read_u32_le(header, 92).ok_or(LoadError::InvalidContainer)?,
        ),
        expert_shared_count: smodel_u32_opt(
            read_u32_le(header, 96).ok_or(LoadError::InvalidContainer)?,
        ),
        expert_feed_forward_length: smodel_u32_opt(
            read_u32_le(header, 100).ok_or(LoadError::InvalidContainer)?,
        ),
        expert_weights_scale,
        kv_lora_rank: smodel_u32_opt(read_u32_le(header, 104).ok_or(LoadError::InvalidContainer)?),
        q_lora_rank: smodel_u32_opt(read_u32_le(header, 108).ok_or(LoadError::InvalidContainer)?),
        qk_nope_head_dim: smodel_u32_opt(
            read_u32_le(header, 112).ok_or(LoadError::InvalidContainer)?,
        ),
        qk_rope_head_dim: smodel_u32_opt(
            read_u32_le(header, 116).ok_or(LoadError::InvalidContainer)?,
        ),
        v_head_dim: smodel_u32_opt(read_u32_le(header, 120).ok_or(LoadError::InvalidContainer)?),
        key_length_mla: None,
        value_length_mla: None,
    };

    // Rebuild the row-interleave registry from scratch for this model —
    // a stale entry from a previously indexed artifact must never alias
    // a new tensor address.
    crate::weight_layout::clear();

    let mut tensors = Vec::with_capacity(tensor_count);
    let mut i = 0usize;
    while i < tensor_count {
        let off = entries_start + i * entry_size;
        let name_off = read_u32_le(header, off).ok_or(LoadError::InvalidContainer)? as usize;
        let name_len = read_u32_le(header, off + 4).ok_or(LoadError::InvalidContainer)? as usize;
        let dtype = read_u32_le(header, off + 8).ok_or(LoadError::InvalidContainer)?;
        let rank = read_u32_le(header, off + 12).ok_or(LoadError::InvalidContainer)? as usize;
        if rank > 8 {
            return Err(LoadError::InvalidContainer);
        }
        let name_end = name_off
            .checked_add(name_len)
            .ok_or(LoadError::InvalidContainer)?;
        if name_end > names.len() {
            return Err(LoadError::InvalidContainer);
        }
        let name = core::str::from_utf8(&names[name_off..name_end])
            .map_err(|_| LoadError::InvalidContainer)?;

        let mut dims = Vec::with_capacity(rank);
        let mut d = 0usize;
        while d < rank {
            dims.push(read_u64_le(header, off + 16 + d * 8).ok_or(LoadError::InvalidContainer)?);
            d += 1;
        }

        let tensor_payload_offset =
            read_u64_le(header, off + 80).ok_or(LoadError::InvalidContainer)?;
        let byte_len = read_u64_le(header, off + 88).ok_or(LoadError::InvalidContainer)? as usize;
        let tensor_abs = info
            .payload_offset
            .checked_add(tensor_payload_offset as usize)
            .ok_or(LoadError::InvalidContainer)?;
        let tensor_end = tensor_abs
            .checked_add(byte_len)
            .ok_or(LoadError::InvalidContainer)?;
        if tensor_end > bytes.len() || tensor_end > payload_end {
            return Err(LoadError::InvalidContainer);
        }

        let (tensor_type, interleave) = smodel_decode_dtype(dtype)?;
        if interleave > 1 {
            if !interleaved_layout_supported() {
                let _ = writeln!(
                    crate::arch::serial::Serial,
                    "SIDX rejected: tensor '{}' has row-interleaved dtype {} but this \
                     runtime path has no interleaved kernels",
                    name,
                    dtype,
                );
                return Err(LoadError::UnsupportedFormat);
            }
            // Interleaving is only defined for rank-2 matmul weights
            // whose output-row count is a multiple of the group size —
            // SilicatePack enforces this at pack time; the kernel
            // re-checks so a hand-edited artifact fails here instead of
            // inside a matmul. SIDX keeps the source (HF) dim order:
            // dims[0] = output rows, dims[1] = row width.
            let rows_ok = rank == 2 && dims[0] % (interleave as u64) == 0;
            if !rows_ok {
                let _ = writeln!(
                    crate::arch::serial::Serial,
                    "SIDX rejected: interleaved tensor '{}' has invalid geometry \
                     (rank={}, dims[0] must be a multiple of {})",
                    name,
                    rank,
                    interleave,
                );
                return Err(LoadError::InvalidContainer);
            }
            let addr = bytes.as_ptr() as usize + tensor_abs;
            if !crate::weight_layout::register(addr, interleave) {
                let _ = writeln!(
                    crate::arch::serial::Serial,
                    "SIDX rejected: weight-layout registry full at tensor '{}'",
                    name,
                );
                return Err(LoadError::UnsupportedFormat);
            }
        }

        tensors.push(zero_gguf_parser::GgufTensorInfo {
            name: String::from(name),
            n_dimensions: rank as u32,
            dimensions: dims,
            tensor_type,
            offset: tensor_payload_offset,
        });
        i += 1;
    }

    if crate::weight_layout::count() > 0 {
        let _ = writeln!(
            crate::arch::serial::Serial,
            "[SMODEL] v2 row-interleaved layout active: {} tensors (group=4)",
            crate::weight_layout::count(),
        );
    }

    Ok(zero_gguf_parser::TensorIndex {
        tensors,
        model_config,
        tensor_data_offset: info.payload_offset,
        tokenizer: None,
        model_name: Some(String::from("SilicatePack native .smodel")),
    })
}

pub fn smodel_info(bytes: &[u8]) -> Result<SmodelInfo, LoadError> {
    if bytes.len() < SMODEL_HEADER_SIZE {
        return Err(LoadError::InvalidContainer);
    }
    if read_u32_le(bytes, 0) != Some(SMODEL_MAGIC) {
        return Err(LoadError::MagicMismatch);
    }
    let version = read_u32_le(bytes, 4).ok_or(LoadError::InvalidContainer)?;
    if version != SMODEL_VERSION_1 && version != SMODEL_VERSION_2 {
        return Err(LoadError::UnsupportedFormat);
    }

    let header_len = read_u32_le(bytes, 8).ok_or(LoadError::InvalidContainer)? as usize;
    let manifest_offset = read_u64_le(bytes, 12).ok_or(LoadError::InvalidContainer)? as usize;
    let manifest_len = read_u64_le(bytes, 20).ok_or(LoadError::InvalidContainer)? as usize;
    let payload_offset = read_u64_le(bytes, 28).ok_or(LoadError::InvalidContainer)? as usize;
    let payload_len = read_u64_le(bytes, 36).ok_or(LoadError::InvalidContainer)? as usize;
    let payload_kind = read_u32_le(bytes, 44).ok_or(LoadError::InvalidContainer)?;
    let flags = read_u32_le(bytes, 48).ok_or(LoadError::InvalidContainer)?;

    if header_len != SMODEL_HEADER_SIZE {
        return Err(LoadError::UnsupportedFormat);
    }
    if payload_kind != SMODEL_PAYLOAD_KIND_GGUF_COMPAT && payload_kind != SMODEL_PAYLOAD_KIND_NATIVE
    {
        return Err(LoadError::UnsupportedFormat);
    }
    if manifest_len == 0 {
        return Err(LoadError::InvalidContainer);
    }

    let manifest_end = manifest_offset
        .checked_add(manifest_len)
        .ok_or(LoadError::InvalidContainer)?;
    let payload_end = payload_offset
        .checked_add(payload_len)
        .ok_or(LoadError::InvalidContainer)?;
    if manifest_offset < header_len
        || manifest_end > payload_offset
        || payload_offset < header_len
        || payload_end > bytes.len()
        || payload_len < 4
    {
        return Err(LoadError::InvalidContainer);
    }
    if flags & 1 != 0 && payload_offset % SMODEL_PAYLOAD_ALIGNMENT != 0 {
        return Err(LoadError::InvalidContainer);
    }
    if payload_kind == SMODEL_PAYLOAD_KIND_NATIVE && payload_len < SMODEL_NATIVE_INDEX_HEADER_SIZE {
        return Err(LoadError::InvalidContainer);
    }

    Ok(SmodelInfo {
        header_len,
        manifest_offset,
        manifest_len,
        payload_offset,
        payload_len,
        payload_kind,
        flags,
    })
}

pub fn smodel_expected_anchor_profile() -> &'static str {
    #[cfg(all(target_arch = "x86_64", feature = "avx512-acceleration"))]
    {
        return "cpu-avx512";
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon-acceleration"))]
    {
        return "cpu-neon";
    }
    #[cfg(target_arch = "x86_64")]
    {
        return "cpu-x86_64";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "cpu-aarch64";
    }
    #[allow(unreachable_code)]
    "unknown"
}

pub fn smodel_expected_anchor_target_arch() -> &'static str {
    #[cfg(all(target_arch = "x86_64", feature = "avx512-acceleration"))]
    {
        return "x86_64-zen4";
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon-acceleration"))]
    {
        return "aarch64-neon";
    }
    #[cfg(target_arch = "x86_64")]
    {
        return "x86_64";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "aarch64";
    }
    #[allow(unreachable_code)]
    "unknown"
}

pub fn smodel_validation_anchor(bytes: &[u8]) -> Result<Option<SmodelValidationAnchor>, LoadError> {
    let info = smodel_info(bytes)?;
    if info.payload_kind != SMODEL_PAYLOAD_KIND_NATIVE {
        return Ok(None);
    }
    let manifest_end = info
        .manifest_offset
        .checked_add(info.manifest_len)
        .ok_or(LoadError::InvalidContainer)?;
    let manifest = core::str::from_utf8(&bytes[info.manifest_offset..manifest_end])
        .map_err(|_| LoadError::InvalidContainer)?;
    let Some(anchor_pos) = manifest.find("\"validation_anchors\"") else {
        return Ok(None);
    };
    let anchor_section = &manifest[anchor_pos..];
    let strict = json_string_after_key(anchor_section, "mode")
        .map(|value| value == "strict")
        .unwrap_or(false);
    let expected_profile = smodel_expected_anchor_profile();
    let expected_target_arch = smodel_expected_anchor_target_arch();
    let matching_anchor =
        find_matching_anchor_object(anchor_section, expected_profile, expected_target_arch);
    let Some(anchor_object) = matching_anchor else {
        return Ok(Some(SmodelValidationAnchor {
            strict,
            matched_runtime: false,
            expected_next_token: None,
            expected_logit_bits: None,
        }));
    };
    let expected_next_token = json_u32_after_key(anchor_object, "expected_next_token");
    let expected_logit_bits = json_u32_after_key(anchor_object, "expected_logit_bits");
    if strict && (expected_next_token.is_none() || expected_logit_bits.is_none()) {
        return Err(LoadError::InvalidContainer);
    }
    Ok(Some(SmodelValidationAnchor {
        strict,
        matched_runtime: true,
        expected_next_token,
        expected_logit_bits,
    }))
}

fn find_matching_anchor_object<'a>(
    anchor_section: &'a str,
    expected_profile: &str,
    expected_target_arch: &str,
) -> Option<&'a str> {
    let anchors_pos = anchor_section.find("\"anchors\"")?;
    let text = &anchor_section[anchors_pos..];
    let bytes = text.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        let Some(open_rel) = text[idx..].find('{') else {
            break;
        };
        let open = idx.checked_add(open_rel)?;
        let close = find_json_object_end(text, open)?;
        let object = &text[open..=close];
        if json_string_after_key(object, "profile") == Some(expected_profile)
            && json_string_after_key(object, "target_arch") == Some(expected_target_arch)
        {
            return Some(object);
        }
        idx = close.checked_add(1)?;
    }
    None
}

fn find_json_object_end(text: &str, open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(open) != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut idx = open;
    while idx < bytes.len() {
        let ch = bytes[idx];
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == b'\\' {
                escaped = true;
            } else if ch == b'"' {
                in_string = false;
            }
        } else if ch == b'"' {
            in_string = true;
        } else if ch == b'{' {
            depth = depth.checked_add(1)?;
        } else if ch == b'}' {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(idx);
            }
        }
        idx += 1;
    }
    None
}

fn json_value_start_after_key(text: &str, key: &str) -> Option<usize> {
    let mut pattern = String::from("\"");
    pattern.push_str(key);
    pattern.push('"');
    let key_pos = text.find(&pattern)?;
    let after_key = key_pos.checked_add(pattern.len())?;
    let colon_rel = text[after_key..].find(':')?;
    let mut idx = after_key.checked_add(colon_rel)?.checked_add(1)?;
    let bytes = text.as_bytes();
    while idx < bytes.len() && matches!(bytes[idx], b' ' | b'\n' | b'\r' | b'\t') {
        idx += 1;
    }
    Some(idx)
}

fn json_string_after_key<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let start = json_value_start_after_key(text, key)?;
    let bytes = text.as_bytes();
    if bytes.get(start) != Some(&b'"') {
        return None;
    }
    let mut end = start + 1;
    while end < bytes.len() {
        if bytes[end] == b'"' && bytes[end.saturating_sub(1)] != b'\\' {
            return Some(&text[start + 1..end]);
        }
        end += 1;
    }
    None
}

fn json_u32_after_key(text: &str, key: &str) -> Option<u32> {
    let start = json_value_start_after_key(text, key)?;
    let bytes = text.as_bytes();
    if bytes.get(start) == Some(&b'"') {
        let value = json_string_after_key(text, key)?;
        return parse_u32_literal(value);
    }
    let mut end = start;
    while end < bytes.len()
        && !matches!(bytes[end], b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t')
    {
        end += 1;
    }
    parse_u32_literal(&text[start..end])
}

fn parse_u32_literal(value: &str) -> Option<u32> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        let mut out = 0u32;
        for ch in hex.bytes() {
            let digit = match ch {
                b'0'..=b'9' => ch - b'0',
                b'a'..=b'f' => ch - b'a' + 10,
                b'A'..=b'F' => ch - b'A' + 10,
                _ => return None,
            } as u32;
            out = out.checked_mul(16)?.checked_add(digit)?;
        }
        return Some(out);
    }
    let mut out = 0u32;
    for ch in trimmed.bytes() {
        if !ch.is_ascii_digit() {
            return None;
        }
        out = out.checked_mul(10)?.checked_add((ch - b'0') as u32)?;
    }
    Some(out)
}

fn smodel_gguf_compat_payload_view(
    bytes: &'static [u8],
    info: SmodelInfo,
) -> Result<ModelView, LoadError> {
    if info.payload_kind != SMODEL_PAYLOAD_KIND_GGUF_COMPAT {
        return Err(LoadError::UnsupportedFormat);
    }

    let payload_end = info
        .payload_offset
        .checked_add(info.payload_len)
        .ok_or(LoadError::InvalidContainer)?;
    if payload_end > bytes.len() || info.payload_len < 4 {
        return Err(LoadError::InvalidContainer);
    }

    let payload_offset = info.payload_offset;
    let payload_len = info.payload_len;
    let payload = &bytes[payload_offset..payload_end];
    if model_magic_from_bytes(payload) != ModelMagic::Gguf {
        return Err(LoadError::MagicMismatch);
    }

    Ok(ModelView {
        bytes: payload,
        size: payload_len,
        format: ModelFormat::SmodelGgufCompat,
        payload_offset,
    })
}

/// Trait for boot-LLM model loaders.
///
/// MP1 minimal API. Will be formalized in MP6 (LLM-Swap-ABI).
pub trait ModelLoader {
    /// Returns a slice over the loaded model bytes.
    ///
    /// # Safety
    ///
    /// The returned slice is valid for the lifetime of the kernel
    /// (model is loaded once at boot, never relocated).
    unsafe fn model_bytes(&self) -> Result<&'static [u8], LoadError>;
}

/// DirectMemoryLoader: model is pre-loaded by QEMU `-device loader`
/// at a known physical address. Kernel reads it via the bootloader's
/// physical_memory_offset mapping.
///
/// MP1 implementation. Stage 17+ may add DiskLoader, NetworkLoader, etc.
pub struct DirectMemoryLoader {
    /// Virtual address: phys_addr + physical_memory_offset
    virt_addr: u64,
    /// Model size in bytes
    size: usize,
}

impl DirectMemoryLoader {
    /// Create a loader given the physical address QEMU was instructed to
    /// load to, the bootloader's physical_memory_offset, and the file size.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - QEMU was started with `-device loader,file=...,addr=phys_addr,force-raw=on`
    /// - `physical_memory_offset` is the bootloader's full physical memory
    ///   mapping offset (boot_info.physical_memory_offset)
    /// - The memory region [phys_addr, phys_addr + size) contains the model
    pub unsafe fn new(
        phys_addr: u64,
        physical_memory_offset: u64,
        size: usize,
    ) -> Result<Self, LoadError> {
        if phys_addr == 0 {
            return Err(LoadError::InvalidAddress);
        }
        if size == 0 || size > 16 * 1024 * 1024 * 1024 {
            return Err(LoadError::SizeOutOfRange);
        }

        Ok(Self {
            virt_addr: phys_addr.wrapping_add(physical_memory_offset),
            size,
        })
    }
}

impl ModelLoader for DirectMemoryLoader {
    unsafe fn model_bytes(&self) -> Result<&'static [u8], LoadError> {
        if self.virt_addr == 0 {
            return Err(LoadError::InvalidAddress);
        }
        let ptr = self.virt_addr as *const u8;
        Ok(core::slice::from_raw_parts(ptr, self.size))
    }
}

/// RamdiskLoader: model embedded into the boot image via bootloader 0.11
/// `BiosBoot::set_ramdisk()` / `UefiBoot::set_ramdisk()`. The bootloader
/// allocates frames, copies the file in, and maps it into the kernel's
/// virtual address space. `BootInfo::ramdisk_addr` is the *virtual*
/// address of byte 0 of the ramdisk.
///
/// This is the production path on bare metal — the model rides with the
/// kernel in a single boot image, deliverable via USB stick or IPMI
/// virtual media without depending on host emulator features.
pub struct RamdiskLoader {
    virt_addr: u64,
    size: usize,
}

impl RamdiskLoader {
    /// Build a loader from `BootInfo::ramdisk_addr` (virtual address) and
    /// `BootInfo::ramdisk_len`. Returns `Err(Absent)` if no ramdisk was
    /// supplied (the common case in dev/QEMU runs).
    ///
    /// # Safety
    ///
    /// Caller must ensure `virt_addr` is the virtual address reported by
    /// bootloader 0.11 for the loaded ramdisk and that `size` matches
    /// `BootInfo::ramdisk_len`.
    pub unsafe fn from_boot_info(virt_addr: Option<u64>, size: u64) -> Result<Self, LoadError> {
        let virt_addr = virt_addr.ok_or(LoadError::Absent)?;
        if virt_addr == 0 {
            return Err(LoadError::InvalidAddress);
        }
        let size = size as usize;
        if size == 0 || size > 16 * 1024 * 1024 * 1024 {
            return Err(LoadError::SizeOutOfRange);
        }
        Ok(Self { virt_addr, size })
    }

    /// Cheap sanity test: the first four bytes spell either raw "GGUF" or
    /// native "SILM". Run before trusting the ramdisk as a Boot-LLM model.
    pub fn looks_like_supported_model(&self) -> bool {
        if self.size < 4 {
            return false;
        }
        let ptr = self.virt_addr as *const u8;
        let magic = unsafe {
            u32::from_le_bytes([
                ptr.read(),
                ptr.add(1).read(),
                ptr.add(2).read(),
                ptr.add(3).read(),
            ])
        };
        magic == GGUF_MAGIC || magic == SMODEL_MAGIC
    }

    pub fn looks_like_gguf(&self) -> bool {
        self.looks_like_supported_model()
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl ModelLoader for RamdiskLoader {
    unsafe fn model_bytes(&self) -> Result<&'static [u8], LoadError> {
        if self.virt_addr == 0 {
            return Err(LoadError::InvalidAddress);
        }
        let ptr = self.virt_addr as *const u8;
        Ok(core::slice::from_raw_parts(ptr, self.size))
    }
}

/// NvmeModelLoader — bulk loader for very large models from NVMe.
///
/// Kimi K2.6 Q4_K weights occupy ≈ 602 GiB and cannot be embedded in
/// the boot image (≤4 GiB ramdisk limit, and pushing 600 GiB through
/// the BMC virtual-media path takes hours). This loader is the
/// production deployment path: the GGUF file is written raw to a
/// known logical-block offset on a directly-attached NVMe drive
/// (typically via `dd` from the Cherry rescue system), and the kernel
/// streams it into a caller-provided destination buffer at boot.
///
/// The destination range *must* be pre-mapped 4 KiB-aligned virtual
/// memory backed by physically-resident frames — there is no demand
/// paging in Ring-0. The model loader does not own the destination
/// allocation; it is the boot path's job (a future weight-arena
/// stage) to reserve enough physically-contiguous frames before
/// calling [`Self::make_resident`].
pub struct NvmeModelLoader {
    /// NVMe namespace id (NSID) the GGUF lives on. The Cherry
    /// deployment writes to NSID=1 on the first NVMe.
    #[allow(dead_code)]
    nsid: u32,
    /// Logical block offset of byte 0 of the GGUF file on the namespace.
    lba_offset: u64,
    /// Total model size in bytes.
    size: usize,
    /// Virtual address of the resident weight buffer once
    /// [`make_resident`] has pulled the file off NVMe. `None` until
    /// residency is achieved.
    resident_virt_addr: Option<u64>,
}

impl NvmeModelLoader {
    /// Construct a loader pointing at a model file on an NVMe namespace.
    ///
    /// # Safety
    ///
    /// Caller asserts that `(nsid, lba_offset, size)` describes a valid
    /// model container on a namespace owned by the kernel.
    pub unsafe fn new(nsid: u32, lba_offset: u64, size: usize) -> Result<Self, LoadError> {
        if size == 0 {
            return Err(LoadError::SizeOutOfRange);
        }
        // Upper bound: 4 TiB. Kimi K2.6 is ≈ 602 GiB so this leaves headroom.
        if size > 4 * 1024usize * 1024 * 1024 * 1024 {
            return Err(LoadError::SizeOutOfRange);
        }
        Ok(Self {
            nsid,
            lba_offset,
            size,
            resident_virt_addr: None,
        })
    }

    /// Total bytes the model occupies on disk.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Logical-block offset of the model container on the namespace.
    pub fn lba_offset(&self) -> u64 {
        self.lba_offset
    }

    /// Pull the entire model container off NVMe into the caller-provided
    /// destination buffer and stamp the loader as resident.
    ///
    /// `pci_scan` is the result of the kernel's all-bus PCI sweep.
    /// The loader **selects the largest-capacity NVMe namespace**
    /// (Cherry deployment convention: the 3.2 TB Solidigm Gen5 data
    /// drives dwarf the system drives, so byte-capacity selection
    /// reliably picks the data drive). The selected drive is then
    /// validated by reading LBA 0 and checking for a supported marker
    /// before the bulk-read begins — a missing marker surfaces as
    /// `LoadError::MagicMismatch` so a missed Phase B step in the
    /// deploy script fails loud and fast.
    ///
    /// `dst_va` is the virtual address of a 4 KiB-aligned destination
    /// buffer of at least `self.size()` bytes, every page of which
    /// must be mapped read+write at the time of the call.
    ///
    /// On success the loader records `dst_va` so [`model_bytes`]
    /// can return the bytes as a `'static` slice, and re-verifies the
    /// model marker on the way out (cheap post-condition).
    #[cfg(target_arch = "x86_64")]
    pub fn make_resident(
        &mut self,
        pci_scan: &crate::arch::x86_64::pcie::PciScan,
        dst_va: u64,
    ) -> Result<(), LoadError> {
        use crate::drivers::nvme::{bind_model_data_drive, probe_model_magic, ProbeFailure};

        if dst_va == 0 {
            return Err(LoadError::InvalidAddress);
        }
        if dst_va & 0xFFF != 0 {
            // Driver requires 4 KiB-aligned DMA buffer.
            return Err(LoadError::InvalidAddress);
        }

        // Capture per-device probe failures so the diagnostic banner
        // can list which controllers we tried before giving up. The
        // bounded array sits on the stack — Ring-0, no alloc.
        let mut errors_buf: [ProbeFailure; 8] = [ProbeFailure {
            bus: 0,
            device: 0,
            function: 0,
            err: crate::drivers::nvme::NvmeError::ControllerNotFound,
        }; 8];
        let mut errors_len: usize = 0;

        let mut probed =
            match bind_model_data_drive(pci_scan, Some(&mut errors_buf[..]), Some(&mut errors_len))
            {
                Ok(p) => p,
                Err(e) => {
                    let _ = writeln!(
                        crate::arch::serial::Serial,
                        "NVMe model loader: no usable NVMe controller (saw {} probe failure(s))",
                        errors_len
                    );
                    for i in 0..errors_len {
                        let f = &errors_buf[i];
                        let _ = writeln!(
                            crate::arch::serial::Serial,
                            "  - {:02x}:{:02x}.{}: {:?}",
                            f.bus,
                            f.device,
                            f.function,
                            f.err
                        );
                    }
                    return Err(map_nvme_err(e));
                }
            };

        let _ = writeln!(
            crate::arch::serial::Serial,
            "NVMe model loader: selected {:02x}:{:02x}.{} as data drive ({} MiB; nsid={})",
            probed.probe.pci_bus,
            probed.probe.pci_device,
            probed.probe.pci_function,
            probed.probe.total_bytes() / (1024 * 1024),
            probed.controller.namespace().nsid,
        );

        // Validate the selected drive actually carries a supported model.
        // The probe
        // does a single 4 KiB read at LBA 0 — cheap.
        match probe_model_magic(&mut probed.controller) {
            Ok(kind) if kind.is_supported() => {
                let _ = writeln!(
                    crate::arch::serial::Serial,
                    "NVMe model loader: {} marker at LBA 0 verified",
                    kind.label()
                );
            }
            Ok(kind) => {
                let _ = writeln!(
                    crate::arch::serial::Serial,
                    "NVMe model loader: NO supported model marker at LBA 0 of {:02x}:{:02x}.{} (saw {})",
                    probed.probe.pci_bus,
                    probed.probe.pci_device,
                    probed.probe.pci_function,
                    kind.label()
                );
                return Err(LoadError::MagicMismatch);
            }
            Err(e) => {
                let _ = writeln!(
                    crate::arch::serial::Serial,
                    "NVMe model loader: magic-probe read failed: {:?}",
                    e
                );
                return Err(map_nvme_err(e));
            }
        }

        let lba_size = probed.controller.namespace().lba_size as usize;
        if lba_size == 0 {
            return Err(LoadError::SizeOutOfRange);
        }
        // Round up to the next LBA boundary so the NVMe controller
        // reads whole sectors. The extra tail bytes (< 512) sit in
        // the already-mapped arena and are never exposed via
        // `model_bytes()`, which returns exactly `self.size` bytes.
        let aligned_size = (self.size + lba_size - 1) & !(lba_size - 1);
        let _ = writeln!(
            crate::arch::serial::Serial,
            "NVMe model loader: model {} B, LBA-aligned read {} B (pad {} B)",
            self.size,
            aligned_size,
            aligned_size - self.size
        );
        // Capacity sanity: refuse to issue a read that runs off the
        // end of the namespace. Without this the controller would
        // surface an LBA-out-of-range completion that we'd report as
        // a generic NvmeError, masking the deploy-time misconfig.
        let drive_bytes = probed.probe.total_bytes();
        if (aligned_size as u64).saturating_add(self.lba_offset.saturating_mul(lba_size as u64))
            > drive_bytes
        {
            let _ = writeln!(
                crate::arch::serial::Serial,
                "NVMe model loader: requested {} B at LBA {} exceeds drive capacity {} B",
                aligned_size,
                self.lba_offset,
                drive_bytes
            );
            return Err(LoadError::SizeOutOfRange);
        }

        probed
            .controller
            .bulk_read(self.lba_offset, aligned_size, dst_va)
            .map_err(map_nvme_err)?;

        // Post-read magic re-check on the destination buffer (cheap;
        // also catches the unlikely case where the controller silently
        // returned a partial read).
        let ptr = dst_va as *const u8;
        let head = unsafe { core::slice::from_raw_parts(ptr, 4) };
        if !looks_like_supported_model(head) {
            return Err(LoadError::MagicMismatch);
        }

        self.resident_virt_addr = Some(dst_va);
        Ok(())
    }
}

/// Translate the driver's error taxonomy into the loader's. The
/// distinction matters at the boot path level: `Absent` means "no
/// controller", everything else means "controller present but
/// something else went wrong" and is surfaced as `InvalidAddress` so
/// the caller can decide whether to fall through to the ramdisk path.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
fn map_nvme_err(e: crate::drivers::nvme::NvmeError) -> LoadError {
    use crate::drivers::nvme::NvmeError;
    match e {
        NvmeError::ControllerNotFound => LoadError::Absent,
        NvmeError::OutOfRange => LoadError::SizeOutOfRange,
        _ => LoadError::InvalidAddress,
    }
}

impl ModelLoader for NvmeModelLoader {
    unsafe fn model_bytes(&self) -> Result<&'static [u8], LoadError> {
        match self.resident_virt_addr {
            Some(addr) if addr != 0 => {
                let ptr = addr as *const u8;
                Ok(core::slice::from_raw_parts(ptr, self.size))
            }
            _ => Err(LoadError::Absent),
        }
    }
}
