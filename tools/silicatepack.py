#!/usr/bin/env python3
"""SilicatePack model artifact tooling for Zero Server.

SilicatePack is the native Zero Server model packaging format. The
production path is intentionally independent of GGUF:

  safetensors + config.json + tokenizer.json -> .smodel

The `.smodel` file carries a compact Silicate header, a JSON manifest,
and a 2 MiB-aligned native tensor payload. GGUF is kept only as an
explicit legacy import path so old benchmark artifacts remain usable
while the kernel moves to the native `.smodel` graph loader.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


MAGIC = b"SILM"
VERSION = 1
# `.smodel`-v2: identical container layout; version 2 signals that the
# artifact may carry row-interleaved tensor dtypes (Q4_0X4 / Q8_0X4).
# Kernels without the interleaved AVX-512 kernels reject v2 at load
# instead of misreading interleaved bytes through plain-layout kernels.
VERSION_INTERLEAVED = 2
HEADER_SIZE = 128

PAYLOAD_KIND_GGUF_COMPAT = 1
PAYLOAD_KIND_NATIVE = 2

DEFAULT_PAYLOAD_ALIGNMENT = 2 * 1024 * 1024
DEFAULT_TENSOR_ALIGNMENT = 64
COPY_CHUNK = 16 * 1024 * 1024
NONE_U32 = 0xFFFF_FFFF

NATIVE_INDEX_MAGIC = b"SIDX"
NATIVE_INDEX_VERSION = 1
NATIVE_INDEX_HEADER_SIZE = 128
NATIVE_TENSOR_ENTRY_STRUCT = struct.Struct("<IIII8Q3Q")
NATIVE_TENSOR_ENTRY_SIZE = NATIVE_TENSOR_ENTRY_STRUCT.size

NATIVE_DTYPE_IDS = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "Q8_K": 15,
    "IQ4_NL": 20,
    "I8": 24,
    "I16": 25,
    "I32": 26,
    "I64": 27,
    "F64": 28,
    "BF16": 30,
    # `.smodel`-v2 row-interleaved layouts. Deliberately outside the GGML
    # id space (<= 39) so a GGUF import can never alias them. Groups of
    # 4 output rows interleave per K-block: d0 d1 d2 d3 then qs0..qs3
    # (72-byte group-blocks for Q4_0X4, 136-byte for Q8_0X4). Total
    # tensor bytes are identical to the plain dtype.
    "Q4_0X4": 100,
    "Q8_0X4": 101,
}

# Tensors that the kernel reads row-wise outside the matmul kernels
# (embedding lookup) — they must never be row-interleaved.
INTERLEAVE_EXCLUDED_NAMES = ("token_embd.weight", "model.embed_tokens.weight")
INTERLEAVE_GROUP = 4
INTERLEAVE_BASE_DTYPE = {"Q4_0X4": "Q4_0", "Q8_0X4": "Q8_0"}

FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
FLOAT_DTYPE_WIDTH = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
Q4_0_BLOCK_SIZE = 32
Q4_0_BLOCK_BYTES = 18
Q8_0_BLOCK_SIZE = 32
Q8_0_BLOCK_BYTES = 34
QUANT_CHOICES = ("none", "auto", "q8_0", "q4_0")
EXPERT_PARTS = {
    "gate_proj": "ffn_gate_exps",
    "up_proj": "ffn_up_exps",
    "down_proj": "ffn_down_exps",
}
EXPERT_PART_ORDER = {
    "gate_proj": 0,
    "up_proj": 1,
    "down_proj": 2,
}
EXPERT_SOURCE_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)

SAFETENSORS_DTYPES = {
    "BOOL",
    "U8",
    "I8",
    "U16",
    "I16",
    "U32",
    "I32",
    "U64",
    "I64",
    "F16",
    "BF16",
    "F32",
    "F64",
}


@dataclass(frozen=True)
class TensorSource:
    name: str
    dtype: str
    shape: tuple[int, ...]
    file: Path
    data_begin: int
    data_end: int
    file_data_offset: int

    @property
    def byte_len(self) -> int:
        return self.data_end - self.data_begin

    @property
    def elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    @property
    def absolute_begin(self) -> int:
        return self.file_data_offset + self.data_begin


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1) != 0:
        raise ValueError("alignment must be a power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(COPY_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_range(path: Path, offset: int, length: int) -> str:
    h = hashlib.sha256()
    remaining = length
    with path.open("rb") as f:
        f.seek(offset)
        while remaining:
            chunk = f.read(min(COPY_CHUNK, remaining))
            if not chunk:
                raise ValueError(f"short read while hashing {path}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def copy_range(
    src: BinaryIO,
    dst: BinaryIO,
    *,
    offset: int,
    length: int,
    payload_hash: Any | None = None,
) -> bytes:
    h = hashlib.sha256()
    remaining = length
    src.seek(offset)
    while remaining:
        chunk = src.read(min(COPY_CHUNK, remaining))
        if not chunk:
            raise ValueError("short read while copying source range")
        dst.write(chunk)
        h.update(chunk)
        if payload_hash is not None:
            payload_hash.update(chunk)
        remaining -= len(chunk)
    return h.digest()


def copy_stream(src: BinaryIO, dst: BinaryIO, payload_hash: Any | None = None) -> bytes:
    h = hashlib.sha256()
    while True:
        chunk = src.read(COPY_CHUNK)
        if not chunk:
            break
        dst.write(chunk)
        h.update(chunk)
        if payload_hash is not None:
            payload_hash.update(chunk)
    return h.digest()


def read_exact_header(path: Path) -> bytes:
    with path.open("rb") as f:
        data = f.read(HEADER_SIZE)
    if len(data) < HEADER_SIZE:
        raise ValueError(f"{path} is smaller than a SilicatePack header")
    return data


def read_magic(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read(4)


def payload_kind_label(kind: int) -> str:
    if kind == PAYLOAD_KIND_NATIVE:
        return "native-smodel"
    if kind == PAYLOAD_KIND_GGUF_COMPAT:
        return "gguf-compat"
    return f"unknown-{kind}"


def parse_u32_literal(value: str) -> int:
    parsed = int(value, 0)
    if parsed < 0 or parsed > 0xFFFF_FFFF:
        raise argparse.ArgumentTypeError(f"{value!r} is outside u32 range")
    return parsed


def parse_u32_csv(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if value.strip() == "":
        return []
    return [parse_u32_literal(part.strip()) for part in value.split(",")]


def format_u32_hex(value: int) -> str:
    return f"0x{value & 0xFFFF_FFFF:08x}"


def build_header(
    *,
    payload_kind: int,
    manifest_offset: int,
    manifest_len: int,
    payload_offset: int,
    payload_len: int,
    payload_sha256: str,
    payload_aligned: bool,
    version: int = VERSION,
) -> bytes:
    flags = 1 if payload_aligned else 0
    header = struct.pack(
        "<4sIIQQQQII32s",
        MAGIC,
        version,
        HEADER_SIZE,
        manifest_offset,
        manifest_len,
        payload_offset,
        payload_len,
        payload_kind,
        flags,
        bytes.fromhex(payload_sha256),
    )
    if len(header) > HEADER_SIZE:
        raise AssertionError("SilicatePack header layout exceeds 128 bytes")
    return header + b"\0" * (HEADER_SIZE - len(header))


def parse_header(data: bytes) -> dict[str, object]:
    (
        magic,
        version,
        header_len,
        manifest_offset,
        manifest_len,
        payload_offset,
        payload_len,
        payload_kind,
        flags,
        payload_sha,
    ) = struct.unpack("<4sIIQQQQII32s", data[:84])
    if magic != MAGIC:
        raise ValueError(f"bad SilicatePack magic: {magic!r}")
    return {
        "magic": magic.decode("ascii"),
        "version": version,
        "header_len": header_len,
        "manifest_offset": manifest_offset,
        "manifest_len": manifest_len,
        "payload_offset": payload_offset,
        "payload_len": payload_len,
        "payload_kind": payload_kind,
        "payload_kind_label": payload_kind_label(payload_kind),
        "flags": flags,
        "payload_2m_aligned": bool(flags & 1),
        "payload_sha256": payload_sha.hex(),
    }


def load_json_file(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_validation_anchors_from_args(args: argparse.Namespace) -> dict[str, object]:
    expected_next = getattr(args, "anchor_next_token", None)
    expected_bits = getattr(args, "anchor_logit_bits", None)
    generated_tokens = parse_u32_csv(getattr(args, "anchor_generated_tokens", None))
    prompt_tokens = parse_u32_csv(getattr(args, "anchor_prompt_tokens", None))
    profile = getattr(args, "profile", "unknown")
    target_arch = getattr(args, "target_arch", "unknown")

    anchor: dict[str, object] = {
        "name": getattr(args, "anchor_name", "zero-server-smoke-v1"),
        "profile": profile,
        "target_arch": target_arch,
        "prompt": getattr(args, "anchor_prompt", "Hello"),
    }
    if prompt_tokens is not None:
        anchor["prompt_tokens"] = prompt_tokens
        anchor["prompt_token_count"] = len(prompt_tokens)
    if expected_next is not None:
        anchor["expected_next_token"] = expected_next
    if expected_bits is not None:
        anchor["expected_logit_bits"] = format_u32_hex(expected_bits)
    if generated_tokens is not None:
        anchor["generated_tokens"] = generated_tokens
        anchor["generated_token_count"] = len(generated_tokens)

    capture = bool(getattr(args, "capture", False))
    strict = (expected_next is not None or expected_bits is not None) and not capture
    # Capture mode keeps the anchor OBJECT (name/profile/prompt) so the
    # kernel's runtime-profile match succeeds and it logs the measured
    # (token, logit_bits) pair — without expected values there is
    # nothing to hard-fail against. Used for first-run anchor
    # collection, e.g. after a quantizer change invalidated the
    # previous baseline's values.
    return {
        "schema": "zero-server-validation-anchors-v1",
        "mode": "strict" if strict else "capture",
        "anchors": [anchor] if (strict or capture) else [],
        "note": (
            "Kernel must hard-fail native .smodel anchor drift when mode=strict. "
            "Capture mode is for first-run anchor collection only."
        ),
    }


def parse_safetensors_file(path: Path) -> tuple[dict[str, object], list[TensorSource]]:
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is too small for a SafeTensors header")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len == 0 or header_len > 1024 * 1024 * 1024:
            raise ValueError(f"{path} has invalid SafeTensors header length {header_len}")
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path} ended before SafeTensors header completed")
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"{path} SafeTensors header is not a JSON object")

    data_offset = 8 + header_len
    file_size = path.stat().st_size
    tensors: list[TensorSource] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise ValueError(f"{path}:{name} tensor metadata is not an object")
        dtype = str(meta.get("dtype"))
        if dtype not in SAFETENSORS_DTYPES:
            raise ValueError(f"{path}:{name} unsupported SafeTensors dtype {dtype!r}")
        shape_obj = meta.get("shape")
        offsets_obj = meta.get("data_offsets")
        if not isinstance(shape_obj, list) or not all(isinstance(v, int) for v in shape_obj):
            raise ValueError(f"{path}:{name} invalid shape")
        if (
            not isinstance(offsets_obj, list)
            or len(offsets_obj) != 2
            or not all(isinstance(v, int) for v in offsets_obj)
        ):
            raise ValueError(f"{path}:{name} invalid data_offsets")
        begin, end = int(offsets_obj[0]), int(offsets_obj[1])
        if begin < 0 or end < begin or data_offset + end > file_size:
            raise ValueError(f"{path}:{name} data_offsets out of range")
        tensors.append(
            TensorSource(
                name=name,
                dtype=dtype,
                shape=tuple(int(v) for v in shape_obj),
                file=path,
                data_begin=begin,
                data_end=end,
                file_data_offset=data_offset,
            )
        )
    return header, tensors


def tensor_sort_key(tensor: TensorSource) -> tuple[int, int, str]:
    name = tensor.name
    if name.startswith("token_embd") or name.startswith("model.embed_tokens"):
        return (0, 0, name)
    if name.startswith("blk.") or name.startswith("model.layers."):
        parts = name.split(".")
        layer = 999999
        for part in parts:
            if part.isdigit():
                layer = int(part)
                break
        order_map = (
            ("attn_norm", 10),
            ("input_layernorm", 10),
            ("attn_q", 20),
            ("self_attn.q_proj", 20),
            ("attn_k", 30),
            ("self_attn.k_proj", 30),
            ("attn_v", 40),
            ("self_attn.v_proj", 40),
            ("attn_output", 50),
            ("self_attn.o_proj", 50),
            ("ffn_gate", 60),
            ("mlp.gate_proj", 60),
            ("ffn_up", 70),
            ("mlp.up_proj", 70),
            ("ffn_down", 80),
            ("mlp.down_proj", 80),
            ("ffn_norm", 90),
            ("post_attention_layernorm", 90),
        )
        op = 500
        for needle, value in order_map:
            if needle in name:
                op = value
                break
        return (100 + layer, op, name)
    if "output_norm" in name or "norm.weight" in name:
        return (900000, 0, name)
    if name == "output.weight" or name.endswith("lm_head.weight"):
        return (900001, 0, name)
    return (500000, 0, name)


def find_safetensors_files(input_dir: Path, explicit: list[str] | None) -> list[Path]:
    if explicit:
        files = [Path(p).expanduser().resolve() for p in explicit]
    else:
        files = sorted(input_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files found in {input_dir}")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def maybe_file(path: str | None, default: Path) -> Path | None:
    if path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
    if default.is_file():
        return default.resolve()
    return None


def file_section_entry(path: Path, *, payload_offset: int, length: int) -> dict[str, object]:
    return {
        "name": path.name,
        "payload_offset": payload_offset,
        "byte_len": length,
        "sha256": sha256_file(path),
    }


def get_cfg_int(config: dict[str, object] | None, keys: tuple[str, ...], default: int = 0) -> int:
    if not isinstance(config, dict):
        return default
    for key in keys:
        value = config.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return int(value)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, int):
                return int(first)
    return default


def get_cfg_float(config: dict[str, object] | None, keys: tuple[str, ...], default: float = 0.0) -> float:
    if not isinstance(config, dict):
        return default
    for key in keys:
        value = config.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return default


def none_if_zero(value: int) -> int:
    return NONE_U32 if value == 0 else value


def architecture_name(config: dict[str, object] | None) -> str:
    if not isinstance(config, dict):
        return "unknown"
    raw = config.get("model_type") or config.get("architectures") or config.get("general.architecture")
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, str):
        return "unknown"
    lower = raw.lower()
    if "qwen3" in lower or lower == "qwen":
        return "qwen3"
    if "deepseek" in lower or "kimi" in lower:
        return "deepseek2"
    return lower


def architecture_id(name: str) -> int:
    if name == "qwen3":
        return 1
    if name == "deepseek2":
        return 2
    return 0


def native_model_config_fields(config: object | None) -> dict[str, int | float | str]:
    cfg = config if isinstance(config, dict) else None
    arch = architecture_name(cfg)
    hidden = get_cfg_int(cfg, ("hidden_size", "n_embd", "dim"))
    heads = get_cfg_int(cfg, ("num_attention_heads", "n_head", "n_heads"), 1)
    kv_heads = get_cfg_int(cfg, ("num_key_value_heads", "n_head_kv", "n_kv_heads"), heads)
    head_dim = get_cfg_int(cfg, ("head_dim",), hidden // heads if heads else 0)
    key_len = get_cfg_int(cfg, ("attention_key_length", "key_length"), head_dim)
    value_len = get_cfg_int(cfg, ("attention_value_length", "value_length"), head_dim)
    qk_nope = get_cfg_int(cfg, ("qk_nope_head_dim",), 0)
    qk_rope = get_cfg_int(cfg, ("qk_rope_head_dim",), 0)
    v_head = get_cfg_int(cfg, ("v_head_dim",), 0)
    if qk_nope or qk_rope:
        key_len = qk_nope + qk_rope
    if v_head:
        value_len = v_head

    return {
        "architecture": arch,
        "architecture_id": architecture_id(arch),
        "block_count": get_cfg_int(cfg, ("num_hidden_layers", "n_layers", "num_layers")),
        "context_length": get_cfg_int(
            cfg,
            ("max_position_embeddings", "seq_length", "model_max_length", "n_ctx"),
        ),
        "embedding_length": hidden,
        "feed_forward_length": get_cfg_int(
            cfg,
            ("intermediate_size", "ffn_hidden_size", "ffn_dim"),
        ),
        "head_count": heads,
        "head_count_kv": kv_heads,
        "key_length": key_len,
        "value_length": value_len,
        "rope_freq_base": get_cfg_float(cfg, ("rope_theta", "rope_freq_base"), 10000.0),
        "layer_norm_rms_epsilon": get_cfg_float(
            cfg,
            ("rms_norm_eps", "layer_norm_epsilon", "norm_eps"),
            0.000001,
        ),
        "vocab_size": none_if_zero(get_cfg_int(cfg, ("vocab_size", "n_vocab"))),
        "eos_token_id": none_if_zero(get_cfg_int(cfg, ("eos_token_id",))),
        "expert_count": none_if_zero(get_cfg_int(cfg, ("n_routed_experts", "num_experts", "moe_num_experts"))),
        "expert_used_count": none_if_zero(get_cfg_int(cfg, ("num_experts_per_tok", "topk_group", "top_k"))),
        "expert_shared_count": none_if_zero(get_cfg_int(cfg, ("n_shared_experts", "num_shared_experts"))),
        "expert_feed_forward_length": none_if_zero(get_cfg_int(cfg, ("moe_intermediate_size", "expert_intermediate_size"))),
        "expert_weights_scale": get_cfg_float(
            cfg,
            (
                "expert_weights_scale",
                "routed_scaling_factor",
                "moe_router_scaling_factor",
                "router_scale",
            ),
            0.0,
        ),
        "kv_lora_rank": none_if_zero(get_cfg_int(cfg, ("kv_lora_rank",))),
        "q_lora_rank": none_if_zero(get_cfg_int(cfg, ("q_lora_rank",))),
        "qk_nope_head_dim": none_if_zero(qk_nope),
        "qk_rope_head_dim": none_if_zero(qk_rope),
        "v_head_dim": none_if_zero(v_head),
    }


def native_dtype_id(dtype: str) -> int:
    try:
        return NATIVE_DTYPE_IDS[dtype]
    except KeyError as exc:
        raise ValueError(f"SafeTensors dtype {dtype!r} has no native SilicatePack dtype id yet") from exc


def native_tensor_name(name: str, arch: str) -> str:
    """Map common Hugging Face tensor names to Zero Server runtime names."""
    del arch  # Current mappings are architecture-neutral where possible.
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name in ("model.norm.weight", "model.final_layernorm.weight"):
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"

    m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
    if not m:
        return name

    layer = int(m.group(1))
    suffix = m.group(2)
    mapped = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        # DeepSeek2 / Kimi MLA naming used by common HF checkpoints.
        "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
        "self_attn.kv_a_layernorm.weight": "attn_kv_a_norm.weight",
        "self_attn.kv_b_proj.weight": "attn_kv_b.weight",
        "self_attn.q_a_proj.weight": "attn_q_a.weight",
        "self_attn.q_a_layernorm.weight": "attn_q_a_norm.weight",
        "self_attn.q_b_proj.weight": "attn_q_b.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "mlp.gate.weight": "ffn_gate_inp.weight",
        "mlp.gate.e_score_correction_bias": "ffn_gate_inp_bias.weight",
        "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
        "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
        "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
    }.get(suffix)
    if mapped:
        return f"blk.{layer}.{mapped}"

    expert = re.fullmatch(r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix)
    if expert:
        # Fallback for non-normalized/debug layouts. The production native
        # plan fuses these into packed expert-major tensors because the
        # Zero Server MoE kernel consumes one contiguous group per part.
        expert_id = int(expert.group(1))
        part = EXPERT_PARTS[expert.group(2)]
        return f"blk.{layer}.{part}.{expert_id}.weight"

    return name


def parse_expert_source_name(name: str) -> tuple[int, int, str] | None:
    m = EXPERT_SOURCE_RE.fullmatch(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3)


def packed_expert_tensor_name(layer: int, source_part: str) -> str:
    return f"blk.{layer}.{EXPERT_PARTS[source_part]}.weight"


def tensor_source_manifest(src: TensorSource, source_sha: str | None = None) -> dict[str, object]:
    return {
        "name": src.name,
        "file": src.file.name,
        "dtype": src.dtype,
        "byte_len": src.byte_len,
        "data_begin": src.data_begin,
        "data_end": src.data_end,
        "absolute_begin": src.absolute_begin,
        "sha256": source_sha if source_sha is not None else sha256_range(src.file, src.absolute_begin, src.byte_len),
    }


def collect_expert_groups(
    tensor_sources: list[TensorSource],
) -> tuple[dict[tuple[int, str], list[tuple[int, TensorSource]]], set[str]]:
    groups: dict[tuple[int, str], list[tuple[int, TensorSource]]] = {}
    names: set[str] = set()
    for src in tensor_sources:
        parsed = parse_expert_source_name(src.name)
        if parsed is None:
            continue
        layer, expert_id, source_part = parsed
        key = (layer, source_part)
        groups.setdefault(key, []).append((expert_id, src))
        names.add(src.name)
    return groups, names


def validate_expert_groups(groups: dict[tuple[int, str], list[tuple[int, TensorSource]]]) -> None:
    by_layer: dict[int, dict[str, set[int]]] = {}
    for (layer, source_part), members in groups.items():
        ids = {expert_id for expert_id, _src in members}
        if len(ids) != len(members):
            raise ValueError(f"layer {layer} {source_part}: duplicate expert tensor id")
        by_layer.setdefault(layer, {})[source_part] = ids

    required = set(EXPERT_PARTS)
    for layer, parts in sorted(by_layer.items()):
        missing = required - set(parts)
        if missing:
            raise ValueError(
                f"layer {layer}: incomplete MoE expert set; missing {sorted(missing)}"
            )
        reference = parts["gate_proj"]
        for source_part in ("up_proj", "down_proj"):
            if parts[source_part] != reference:
                raise ValueError(
                    f"layer {layer}: expert ids differ between gate_proj and {source_part}"
                )


def is_norm_or_scalar_tensor(name: str, shape: tuple[int, ...]) -> bool:
    if len(shape) <= 1:
        return True
    lowered = name.lower()
    return (
        lowered.endswith(".bias")
        or "norm.weight" in lowered
        or "layernorm.weight" in lowered
        or "rope" in lowered
    )


# Tensors whose values feed the MoE routing decision. Routing logits are
# precision-critical: a 4-bit router systematically skews expert selection
# for the whole model (llama.cpp keeps routers F32 for the same reason).
# These names must never be quantized, regardless of quant policy.
ROUTING_PRECISION_CRITICAL_SUFFIXES = (
    "ffn_gate_inp.weight",
    "ffn_gate_inp_bias.weight",
)


def is_routing_precision_critical(final_name: str) -> bool:
    return final_name.endswith(ROUTING_PRECISION_CRITICAL_SUFFIXES)


def tensor_row_width(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return int(shape[-1])


def quantized_byte_len(elements: int, row_width: int, quant: str) -> int:
    if row_width <= 0 or elements % row_width != 0:
        raise ValueError("invalid tensor row geometry")
    if row_width % 32 != 0:
        raise ValueError(f"row width {row_width} is not divisible by 32")
    blocks_per_row = row_width // 32
    rows = elements // row_width
    if quant in ("Q8_0", "Q8_0X4"):
        return rows * blocks_per_row * Q8_0_BLOCK_BYTES
    if quant in ("Q4_0", "Q4_0X4"):
        return rows * blocks_per_row * Q4_0_BLOCK_BYTES
    raise ValueError(f"unsupported native quant {quant}")


def select_output_dtype(src: TensorSource, final_name: str, quant_policy: str) -> tuple[str, str]:
    """Return (output_dtype, transform)."""
    if src.dtype in FLOAT_DTYPES and is_norm_or_scalar_tensor(final_name, src.shape):
        return ("F32", "f32")
    if src.dtype in FLOAT_DTYPES and is_routing_precision_critical(final_name):
        return ("F32", "f32")
    if quant_policy == "none" or src.dtype not in FLOAT_DTYPES or len(src.shape) < 2:
        return (src.dtype, "raw")

    row_width = tensor_row_width(src.shape)
    if row_width % 32 != 0:
        raise ValueError(
            f"{final_name}: float matrix row width {row_width} is not divisible by 32; "
            "native Performance-v1 tensors must be packable into Q4_0/Q8_0 blocks"
        )

    if quant_policy == "q8_0":
        return ("Q8_0", "q8_0")
    if quant_policy == "q4_0":
        return ("Q4_0", "q4_0")
    if quant_policy == "auto":
        if final_name in ("token_embd.weight", "output.weight") or final_name.endswith("lm_head.weight"):
            return ("Q8_0", "q8_0")
        return ("Q4_0", "q4_0")
    raise ValueError(f"unsupported quant policy {quant_policy!r}")


def planned_tensor_byte_len(src: TensorSource, output_dtype: str, transform: str) -> int:
    if transform == "raw":
        return src.byte_len
    if transform == "f32":
        return src.elements * 4
    if transform in ("q8_0", "q4_0", "q8_0x4", "q4_0x4"):
        return quantized_byte_len(src.elements, tensor_row_width(src.shape), output_dtype)
    raise ValueError(f"unsupported tensor transform {transform}")


def interleave_eligible(final_name: str, shape: tuple[int, ...], output_dtype: str) -> bool:
    """True when a planned tensor may be emitted row-interleaved.

    Only rank-2 matmul weights qualify: expert-fused tensors gain a
    third dim (the kernel slices into them per expert), and embedding
    tables are read row-wise by the lookup path. The output-row count
    (HF shape[0]) must divide the interleave group so every group-block
    is complete.
    """
    if output_dtype not in ("Q4_0", "Q8_0"):
        return False
    if len(shape) != 2:
        return False
    if final_name in INTERLEAVE_EXCLUDED_NAMES:
        return False
    return shape[0] % INTERLEAVE_GROUP == 0


def float_to_fp16_bytes(value: float) -> bytes:
    if not math.isfinite(value):
        value = 0.0
    if value > 65504.0:
        value = 65504.0
    elif value < -65504.0:
        value = -65504.0
    return struct.pack("<e", float(value))


def decode_float_values(data: bytes, dtype: str) -> tuple[float, ...]:
    if dtype == "F32":
        return struct.unpack(f"<{len(data) // 4}f", data)
    if dtype == "F64":
        return tuple(float(v) for v in struct.unpack(f"<{len(data) // 8}d", data))
    if dtype == "F16":
        return tuple(float(v) for v in struct.unpack(f"<{len(data) // 2}e", data))
    if dtype == "BF16":
        out: list[float] = []
        for (bits,) in struct.iter_unpack("<H", data):
            out.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
        return tuple(out)
    raise ValueError(f"cannot decode non-float dtype {dtype}")


def encode_f32_values(values: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _reject_non_finite(block: list[float]) -> None:
    # NaN/Inf must abort the pack, not slip through: a NaN that is not
    # the first block element escapes max()-based guards (NaN comparisons
    # are false), then crashes int() mid-pack — hours into a Kimi-sized
    # run — or, as ±Inf, silently zeroes the whole 32-element block.
    for idx, v in enumerate(block):
        if not math.isfinite(v):
            raise ValueError(f"non-finite value {v!r} at block element {idx}")


def quantize_q8_0_block(values: tuple[float, ...]) -> bytes:
    block = list(values[:Q8_0_BLOCK_SIZE])
    if len(block) < Q8_0_BLOCK_SIZE:
        block.extend([0.0] * (Q8_0_BLOCK_SIZE - len(block)))
    _reject_non_finite(block)
    max_abs = max(abs(v) for v in block)
    if max_abs == 0.0:
        return b"\0\0" + b"\0" * 32
    d = max_abs / 127.0
    inv_d = 1.0 / d
    qs = []
    for value in block:
        q = int(round(value * inv_d))
        qs.append(max(-127, min(127, q)))
    return float_to_fp16_bytes(d) + struct.pack("<32b", *qs)


def quantize_q4_0_block(values: tuple[float, ...]) -> bytes:
    # GGML reference encoding (ggml-quants.c quantize_row_q4_0_ref):
    # the scale is derived from the SIGNED value with the largest
    # magnitude, d = max / -8, so the full asymmetric level range
    # [-8, 7] is used (level -8 encodes the max-magnitude value).
    # A symmetric max_abs/7 scale never emits level -8 and costs
    # ~14 % extra quantization error on every block.
    block = list(values[:Q4_0_BLOCK_SIZE])
    if len(block) < Q4_0_BLOCK_SIZE:
        block.extend([0.0] * (Q4_0_BLOCK_SIZE - len(block)))
    _reject_non_finite(block)
    max_abs = 0.0
    max_signed = 0.0
    for v in block:
        if abs(v) > max_abs:
            max_abs = abs(v)
            max_signed = v
    if max_abs == 0.0:
        return b"\0\0" + b"\0" * 16
    d = max_signed / -8.0
    inv_d = 1.0 / d
    nibbles: list[int] = []
    for value in block:
        # GGML: xi = MIN(15, (int8_t)(x*id + 8.5f)) — the +8.5 with
        # truncation is round-to-nearest of (x*id + 8).
        q = int(value * inv_d + 8.5)
        nibbles.append(max(0, min(15, q)))
    packed = bytearray(16)
    for j in range(16):
        packed[j] = nibbles[j] | (nibbles[j + 16] << 4)
    return float_to_fp16_bytes(d) + bytes(packed)


def write_transformed_tensor(
    *,
    src: TensorSource,
    dst: BinaryIO,
    payload_hash: Any,
    transform: str,
) -> str:
    h = hashlib.sha256()
    with src.file.open("rb") as f:
        f.seek(src.absolute_begin)
        if transform == "raw":
            digest = copy_range(
                f,
                dst,
                offset=src.absolute_begin,
                length=src.byte_len,
                payload_hash=payload_hash,
            )
            return digest.hex()

        if src.dtype not in FLOAT_DTYPES:
            raise ValueError(f"{src.name}: transform {transform} requires float source, got {src.dtype}")

        elem_width = FLOAT_DTYPE_WIDTH[src.dtype]
        if transform == "f32":
            chunk_elems = 8192
            remaining = src.elements
            while remaining:
                elems = min(chunk_elems, remaining)
                data = f.read(elems * elem_width)
                if len(data) != elems * elem_width:
                    raise ValueError(f"{src.name}: short read during f32 conversion")
                out = encode_f32_values(decode_float_values(data, src.dtype))
                dst.write(out)
                h.update(out)
                payload_hash.update(out)
                remaining -= elems
            return h.hexdigest()

        if transform in ("q8_0x4", "q4_0x4"):
            # Row-interleaved emission: quantize INTERLEAVE_GROUP rows
            # with the exact same per-block encoder as the plain
            # transform, then store each K-block position as one
            # group-block: the 4 fp16 scales first, then the 4 quant
            # bodies. Every (row, block) byte sequence is identical to
            # the plain layout — only the storage order changes, so
            # dequantized values (and therefore all logit anchors) are
            # untouched.
            block_fn = quantize_q8_0_block if transform == "q8_0x4" else quantize_q4_0_block
            row_width = tensor_row_width(src.shape)
            if row_width % 32 != 0:
                raise ValueError(f"{src.name}: row width {row_width} is not divisible by 32")
            rows = src.elements // row_width
            if rows % INTERLEAVE_GROUP != 0:
                raise ValueError(
                    f"{src.name}: {rows} rows not a multiple of interleave group "
                    f"{INTERLEAVE_GROUP}"
                )
            row_bytes = row_width * elem_width
            blocks_per_row = row_width // 32
            for group_start in range(0, rows, INTERLEAVE_GROUP):
                group_blocks: list[list[bytes]] = []
                for lane in range(INTERLEAVE_GROUP):
                    row_idx = group_start + lane
                    row = f.read(row_bytes)
                    if len(row) != row_bytes:
                        raise ValueError(f"{src.name}: short read during {transform} conversion")
                    values = decode_float_values(row, src.dtype)
                    row_blocks: list[bytes] = []
                    for start in range(0, row_width, 32):
                        try:
                            row_blocks.append(block_fn(values[start : start + 32]))
                        except ValueError as exc:
                            block_index = row_idx * blocks_per_row + start // 32
                            raise ValueError(
                                f"{src.name}: {exc} (tensor block {block_index}, "
                                f"row {row_idx}, col {start}) — aborting pack; "
                                "the source checkpoint contains NaN/Inf weights"
                            ) from exc
                    group_blocks.append(row_blocks)
                for b in range(blocks_per_row):
                    out = b"".join(group_blocks[lane][b][:2] for lane in range(INTERLEAVE_GROUP))
                    out += b"".join(group_blocks[lane][b][2:] for lane in range(INTERLEAVE_GROUP))
                    dst.write(out)
                    h.update(out)
                    payload_hash.update(out)
            return h.hexdigest()

        if transform not in ("q8_0", "q4_0"):
            raise ValueError(f"{src.name}: unsupported transform {transform}")

        row_width = tensor_row_width(src.shape)
        if row_width % 32 != 0:
            raise ValueError(f"{src.name}: row width {row_width} is not divisible by 32")
        rows = src.elements // row_width
        row_bytes = row_width * elem_width
        block_fn = quantize_q8_0_block if transform == "q8_0" else quantize_q4_0_block
        blocks_per_row = row_width // 32
        for row_idx in range(rows):
            row = f.read(row_bytes)
            if len(row) != row_bytes:
                raise ValueError(f"{src.name}: short read during {transform} conversion")
            values = decode_float_values(row, src.dtype)
            for start in range(0, row_width, 32):
                try:
                    out = block_fn(values[start : start + 32])
                except ValueError as exc:
                    block_index = row_idx * blocks_per_row + start // 32
                    raise ValueError(
                        f"{src.name}: {exc} (tensor block {block_index}, "
                        f"row {row_idx}, col {start}) — aborting pack; "
                        "the source checkpoint contains NaN/Inf weights"
                    ) from exc
                dst.write(out)
                h.update(out)
                payload_hash.update(out)
        return h.hexdigest()


class HashFanout:
    def __init__(self, *hashes: Any) -> None:
        self.hashes = hashes

    def update(self, data: bytes) -> None:
        for h in self.hashes:
            h.update(data)


def write_transformed_tensor_group(
    *,
    sources: list[TensorSource],
    dst: BinaryIO,
    payload_hash: Any,
    transform: str,
) -> str:
    group_hash = hashlib.sha256()
    fanout = HashFanout(payload_hash, group_hash)
    for src in sources:
        write_transformed_tensor(src=src, dst=dst, payload_hash=fanout, transform=transform)
    return group_hash.hexdigest()


def native_names_blob(tensor_entries: list[dict[str, object]]) -> tuple[bytes, dict[str, tuple[int, int]]]:
    blob = bytearray()
    offsets: dict[str, tuple[int, int]] = {}
    for entry in tensor_entries:
        name = str(entry["name"])
        encoded = name.encode("utf-8")
        offsets[name] = (len(blob), len(encoded))
        blob.extend(encoded)
        blob.append(0)
    return bytes(blob), offsets


def build_native_index(
    *,
    config_json: object | None,
    tensor_entries: list[dict[str, object]],
    names_blob: bytes,
    name_offsets: dict[str, tuple[int, int]],
    names_offset: int,
    data_base: int,
) -> bytes:
    fields = native_model_config_fields(config_json)
    header = bytearray(NATIVE_INDEX_HEADER_SIZE)
    struct.pack_into("<4s", header, 0, NATIVE_INDEX_MAGIC)
    struct.pack_into("<I", header, 4, NATIVE_INDEX_VERSION)
    struct.pack_into("<I", header, 8, NATIVE_INDEX_HEADER_SIZE)
    struct.pack_into("<I", header, 12, len(tensor_entries))
    struct.pack_into("<I", header, 16, NATIVE_TENSOR_ENTRY_SIZE)
    struct.pack_into("<I", header, 20, names_offset)
    struct.pack_into("<I", header, 24, len(names_blob))
    struct.pack_into("<Q", header, 28, data_base)
    struct.pack_into("<I", header, 36, int(fields["architecture_id"]))
    struct.pack_into("<I", header, 40, int(fields["block_count"]))
    struct.pack_into("<I", header, 44, int(fields["context_length"]))
    struct.pack_into("<I", header, 48, int(fields["embedding_length"]))
    struct.pack_into("<I", header, 52, int(fields["feed_forward_length"]))
    struct.pack_into("<I", header, 56, int(fields["head_count"]))
    struct.pack_into("<I", header, 60, int(fields["head_count_kv"]))
    struct.pack_into("<I", header, 64, int(fields["key_length"]))
    struct.pack_into("<I", header, 68, int(fields["value_length"]))
    struct.pack_into("<f", header, 72, float(fields["rope_freq_base"]))
    struct.pack_into("<f", header, 76, float(fields["layer_norm_rms_epsilon"]))
    struct.pack_into("<I", header, 80, int(fields["vocab_size"]))
    struct.pack_into("<I", header, 84, int(fields["eos_token_id"]))
    struct.pack_into("<I", header, 88, int(fields["expert_count"]))
    struct.pack_into("<I", header, 92, int(fields["expert_used_count"]))
    struct.pack_into("<I", header, 96, int(fields["expert_shared_count"]))
    struct.pack_into("<I", header, 100, int(fields["expert_feed_forward_length"]))
    struct.pack_into("<I", header, 104, int(fields["kv_lora_rank"]))
    struct.pack_into("<I", header, 108, int(fields["q_lora_rank"]))
    struct.pack_into("<I", header, 112, int(fields["qk_nope_head_dim"]))
    struct.pack_into("<I", header, 116, int(fields["qk_rope_head_dim"]))
    struct.pack_into("<I", header, 120, int(fields["v_head_dim"]))
    struct.pack_into("<f", header, 124, float(fields["expert_weights_scale"]))

    entries = bytearray()
    for entry in tensor_entries:
        name = str(entry["name"])
        name_offset, name_len = name_offsets[name]
        shape = list(entry["shape"])
        if len(shape) > 8:
            raise ValueError(f"{name}: rank {len(shape)} exceeds native index limit 8")
        padded_shape = [int(v) for v in shape] + [1] * (8 - len(shape))
        entries.extend(
            NATIVE_TENSOR_ENTRY_STRUCT.pack(
                name_offset,
                name_len,
                native_dtype_id(str(entry["dtype"])),
                len(shape),
                *padded_shape,
                int(entry["payload_offset"]),
                int(entry["byte_len"]),
                int(entry["elements"]),
            )
        )

    return bytes(header) + bytes(entries) + names_blob


def make_native_manifest(
    *,
    input_dir: Path,
    safetensors_files: list[Path],
    source_file_hashes: dict[Path, str],
    config_path: Path | None,
    tokenizer_path: Path | None,
    config_json: object | None,
    payload_offset: int,
    payload_len: int,
    sections: dict[str, dict[str, object]],
    tensors: list[dict[str, object]],
    target_product: str,
    profile: str,
    target_arch: str,
    source_repo: str | None,
    source_revision: str | None,
    license_name: str | None,
    tensor_alignment: int,
    payload_alignment: int,
    quantization: str,
    normalize_names: bool,
    validation_anchors: dict[str, object] | None,
    artifact_version: int = VERSION,
    row_interleave: int = 0,
) -> dict[str, object]:
    return {
        "format": "SilicatePack",
        "container": "native",
        "magic": "SILM",
        "version": artifact_version,
        "artifact_extension": ".smodel",
        "ready_label": "Ready for Zero Server",
        "target_product": target_product,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source": {
            "format": "safetensors",
            "input_dir": str(input_dir),
            "repo": source_repo,
            "revision": source_revision,
            "license": license_name,
            "files": [
                {
                    "name": p.name,
                    "path": str(p),
                    "size": p.stat().st_size,
                    # Hashes are precomputed once by the caller — this
                    # function runs in the manifest-size convergence
                    # loop, and re-hashing a multi-hundred-GiB
                    # checkpoint per iteration costs hours.
                    "sha256": source_file_hashes[p],
                }
                for p in safetensors_files
            ],
        },
        "model_config": config_json if isinstance(config_json, dict) else {},
        "sections": sections,
        "layout": {
            "payload_offset": payload_offset,
            "payload_len": payload_len,
            "payload_alignment": payload_alignment,
            "tensor_alignment": tensor_alignment,
            "tensor_order": "zero-server-forward-order-v1",
            "gguf_payload": False,
            "native_index": sections.get("native_index", {}),
            "name_normalization": "hf-to-zero-runtime-v1" if normalize_names else "source",
            "quantization": quantization,
            # 0 = plain row-major (v1). 4 = `.smodel`-v2: eligible rank-2
            # Q4_0/Q8_0 matmul tensors are stored 4-row-interleaved
            # (dtype ids Q4_0X4/Q8_0X4); per-tensor "row_interleave"
            # markers in `tensors` identify them.
            "row_interleave": row_interleave,
        },
        "runtime_profiles": {
            "cpu": {
                "profile": profile,
                "target_arch": target_arch,
                "status": "performance-v1",
                "native_quantization": quantization,
                "constraints": [
                    "no GGUF parsing in kernel hot path",
                    "tensor payload is native SilicatePack layout",
                    "byte-exact tensor checksums recorded in manifest",
                ],
            },
            "gpu": {
                "status": "declared-forward-compatible",
                "note": "GPU layout profiles are manifest-declared and can be added without changing the container magic.",
            },
        },
        "tensor_count": len(tensors),
        "tensors": tensors,
        "validation_anchors": validation_anchors
        if validation_anchors is not None
        else {
            "schema": "zero-server-validation-anchors-v1",
            "mode": "capture",
            "anchors": [],
        },
        "compatibility": {
            "gguf_runtime_payload": False,
            "gguf_import_supported_by_tool": True,
        },
        "files": {
            "config": str(config_path) if config_path else None,
            "tokenizer": str(tokenizer_path) if tokenizer_path else None,
        },
    }


def plan_native_layout(
    *,
    config_json: object | None,
    config_path: Path | None,
    tokenizer_path: Path | None,
    tensor_sources: list[TensorSource],
    tensor_alignment: int,
    quantization: str,
    normalize_names: bool,
    interleave: int = 0,
) -> tuple[int, dict[str, dict[str, object]], list[dict[str, object]]]:
    sections: dict[str, dict[str, object]] = {}

    sorted_sources = sorted(tensor_sources, key=tensor_sort_key)
    names_seen: set[str] = set()
    preliminary_entries: list[dict[str, object]] = []
    arch = architecture_name(config_json if isinstance(config_json, dict) else None)
    expert_groups, expert_source_names = (
        collect_expert_groups(sorted_sources) if normalize_names else ({}, set())
    )
    validate_expert_groups(expert_groups)

    for src in sorted_sources:
        if src.name in expert_source_names:
            continue
        final_name = native_tensor_name(src.name, arch) if normalize_names else src.name
        if final_name in names_seen:
            raise ValueError(
                f"duplicate tensor name after Zero Server normalization: {final_name} "
                f"(source {src.name})"
            )
        names_seen.add(final_name)
        output_dtype, transform = select_output_dtype(src, final_name, quantization)
        if interleave == INTERLEAVE_GROUP and interleave_eligible(
            final_name, src.shape, output_dtype
        ):
            output_dtype = f"{output_dtype}X{INTERLEAVE_GROUP}"
            transform = f"{transform}x{INTERLEAVE_GROUP}"
        byte_len = planned_tensor_byte_len(src, output_dtype, transform)
        source_sha = sha256_range(src.file, src.absolute_begin, src.byte_len)
        entry: dict[str, object] = {
            "name": final_name,
            "dtype": output_dtype,
            "shape": list(src.shape),
            "elements": src.elements,
            "byte_len": byte_len,
            "payload_offset": 0,
            "transform": transform,
            "source": tensor_source_manifest(src, source_sha),
            # Real payload checksum is filled after writing. Keep a fixed
            # 64-char placeholder so manifest length remains stable.
            "sha256": source_sha if transform == "raw" else "0" * 64,
        }
        if output_dtype in INTERLEAVE_BASE_DTYPE:
            entry["row_interleave"] = INTERLEAVE_GROUP
        preliminary_entries.append(entry)

    for (layer, source_part), members in sorted(
        expert_groups.items(), key=lambda item: (item[0][0], EXPERT_PART_ORDER[item[0][1]])
    ):
        final_name = packed_expert_tensor_name(layer, source_part)
        if final_name in names_seen:
            raise ValueError(
                f"duplicate packed expert tensor after Zero Server normalization: {final_name}"
            )
        sorted_members = sorted(members, key=lambda item: item[0])
        expert_ids = [expert_id for expert_id, _src in sorted_members]
        expected_ids = list(range(len(expert_ids)))
        if expert_ids != expected_ids:
            raise ValueError(
                f"{final_name}: expert ids must be contiguous from 0; got {expert_ids[:8]}"
            )
        first = sorted_members[0][1]
        for expert_id, src in sorted_members:
            if src.dtype != first.dtype or src.shape != first.shape or src.elements != first.elements:
                raise ValueError(
                    f"{final_name}: expert {expert_id} shape/dtype mismatch "
                    f"(got {src.dtype} {src.shape}, expected {first.dtype} {first.shape})"
                )
        output_dtype, transform = select_output_dtype(first, final_name, quantization)
        per_expert_byte_len = planned_tensor_byte_len(first, output_dtype, transform)
        source_group: list[dict[str, object]] = []
        for expert_id, src in sorted_members:
            source_entry = tensor_source_manifest(src)
            source_entry["expert_id"] = expert_id
            source_group.append(source_entry)
        preliminary_entries.append(
            {
                "name": final_name,
                "dtype": output_dtype,
                "shape": list(first.shape) + [len(sorted_members)],
                "elements": first.elements * len(sorted_members),
                "byte_len": per_expert_byte_len * len(sorted_members),
                "payload_offset": 0,
                "transform": transform,
                "expert_group": {
                    "layer": layer,
                    "source_part": source_part,
                    "expert_count": len(sorted_members),
                    "per_expert_elements": first.elements,
                    "per_expert_byte_len": per_expert_byte_len,
                    "layout": "expert-major-contiguous-v1",
                },
                "source_group": source_group,
                "sha256": "0" * 64,
            }
        )
        names_seen.add(final_name)

    names_blob, _name_offsets = native_names_blob(preliminary_entries)
    raw_index_len = NATIVE_INDEX_HEADER_SIZE + len(preliminary_entries) * NATIVE_TENSOR_ENTRY_SIZE + len(names_blob)
    cursor = align_up(raw_index_len, tensor_alignment)
    sections["native_index"] = {
        "payload_offset": 0,
        "byte_len": cursor,
        "raw_byte_len": raw_index_len,
        "magic": NATIVE_INDEX_MAGIC.decode("ascii"),
        "version": NATIVE_INDEX_VERSION,
        "tensor_count": len(preliminary_entries),
        "entry_size": NATIVE_TENSOR_ENTRY_SIZE,
    }

    for key, path in (("config_json", config_path), ("tokenizer_json", tokenizer_path)):
        if not path:
            continue
        cursor = align_up(cursor, tensor_alignment)
        size = path.stat().st_size
        sections[key] = file_section_entry(path, payload_offset=cursor, length=size)
        cursor += size

    tensor_entries: list[dict[str, object]] = []
    for entry in preliminary_entries:
        cursor = align_up(cursor, tensor_alignment)
        entry = dict(entry)
        entry["payload_offset"] = cursor
        tensor_entries.append(entry)
        cursor += int(entry["byte_len"])
    return cursor, sections, tensor_entries


def pack_hf(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to overwrite")

    safetensors_files = find_safetensors_files(input_dir, args.safetensors)
    config_path = maybe_file(args.config, input_dir / "config.json")
    tokenizer_path = maybe_file(args.tokenizer, input_dir / "tokenizer.json")
    config_json = load_json_file(config_path) if config_path else None

    tensor_sources: list[TensorSource] = []
    for path in safetensors_files:
        _header, tensors = parse_safetensors_file(path)
        tensor_sources.extend(tensors)
    if not tensor_sources:
        raise ValueError("no tensors found in SafeTensors inputs")

    interleave = getattr(args, "interleave", 0)
    payload_len, sections, tensor_entries = plan_native_layout(
        config_json=config_json,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        tensor_sources=tensor_sources,
        tensor_alignment=args.tensor_alignment,
        quantization=args.quant,
        normalize_names=args.normalize_names,
        interleave=interleave,
    )
    # The artifact is v2 exactly when at least one tensor actually got
    # the interleaved layout — `--interleave 4` on a model with no
    # eligible tensor still emits a fully v1-compatible artifact.
    has_interleaved = any(str(e["dtype"]) in INTERLEAVE_BASE_DTYPE for e in tensor_entries)
    artifact_version = VERSION_INTERLEAVED if has_interleaved else VERSION
    effective_row_interleave = INTERLEAVE_GROUP if has_interleaved else 0

    # Hash every SafeTensors source exactly once. The manifest builder
    # below runs at least twice (seed pass + convergence loop); without
    # the cache each pass re-reads the full checkpoint (Kimi: 584 GiB)
    # for SHA-256 — days of duplicated I/O over a pack run.
    source_file_hashes = {p: sha256_file(p) for p in safetensors_files}

    def build_manifest_with_payload_offset(payload_offset: int) -> bytes:
        manifest = make_native_manifest(
            input_dir=input_dir,
            safetensors_files=safetensors_files,
            source_file_hashes=source_file_hashes,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            config_json=config_json,
            payload_offset=payload_offset,
            payload_len=payload_len,
            sections=sections,
            tensors=tensor_entries,
            target_product=args.target_product,
            profile=args.profile,
            target_arch=args.target_arch,
            source_repo=args.source_repo,
            source_revision=args.source_revision,
            license_name=args.license,
            tensor_alignment=args.tensor_alignment,
            payload_alignment=args.payload_alignment,
            quantization=args.quant,
            normalize_names=args.normalize_names,
            validation_anchors=build_validation_anchors_from_args(args),
            artifact_version=artifact_version,
            row_interleave=effective_row_interleave,
        )
        return json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"

    manifest_bytes = build_manifest_with_payload_offset(0)
    payload_offset = align_up(HEADER_SIZE + len(manifest_bytes), args.payload_alignment)
    for _ in range(8):
        manifest_bytes = build_manifest_with_payload_offset(payload_offset)
        next_payload_offset = align_up(HEADER_SIZE + len(manifest_bytes), args.payload_alignment)
        if next_payload_offset == payload_offset:
            break
        payload_offset = next_payload_offset
    else:
        raise RuntimeError("native manifest size did not converge")

    output.parent.mkdir(parents=True, exist_ok=True)
    zero_sha = "00" * 32
    with output.open("wb") as out:
        out.write(
            build_header(
                payload_kind=PAYLOAD_KIND_NATIVE,
                manifest_offset=HEADER_SIZE,
                manifest_len=len(manifest_bytes),
                payload_offset=payload_offset,
                payload_len=payload_len,
                payload_sha256=zero_sha,
                # Header flag bit 0 is read by the kernel as payload_2m_aligned;
                # it must reflect the 2 MiB constant, not whatever
                # --payload-alignment was passed.
                payload_aligned=(payload_offset % DEFAULT_PAYLOAD_ALIGNMENT == 0),
                version=artifact_version,
            )
        )
        out.write(manifest_bytes)
        out.write(b"\0" * (payload_offset - out.tell()))

        payload_hash = hashlib.sha256()

        def write_padding(target_payload_offset: int) -> None:
            current_payload_offset = out.tell() - payload_offset
            if current_payload_offset > target_payload_offset:
                raise AssertionError("payload cursor passed planned offset")
            remaining = target_payload_offset - current_payload_offset
            zero_chunk = b"\0" * min(COPY_CHUNK, max(remaining, 1))
            while remaining:
                chunk = zero_chunk[: min(len(zero_chunk), remaining)]
                out.write(chunk)
                payload_hash.update(chunk)
                remaining -= len(chunk)

        native_index_section = sections["native_index"]
        names_blob, name_offsets = native_names_blob(tensor_entries)
        native_index = build_native_index(
            config_json=config_json,
            tensor_entries=tensor_entries,
            names_blob=names_blob,
            name_offsets=name_offsets,
            names_offset=NATIVE_INDEX_HEADER_SIZE + len(tensor_entries) * NATIVE_TENSOR_ENTRY_SIZE,
            data_base=0,
        )
        if len(native_index) != int(native_index_section["raw_byte_len"]):
            raise AssertionError("native index length changed after planning")
        write_padding(int(native_index_section["payload_offset"]))
        out.write(native_index)
        payload_hash.update(native_index)

        for key in ("config_json", "tokenizer_json"):
            section = sections.get(key)
            path = config_path if key == "config_json" else tokenizer_path
            if not section or not path:
                continue
            write_padding(int(section["payload_offset"]))
            with path.open("rb") as src:
                digest = copy_stream(src, out, payload_hash)
            if digest.hex() != section["sha256"]:
                raise AssertionError(f"{key} checksum changed during write")

        sources_by_name = {t.name: t for t in tensor_sources}
        written_hashes: dict[str, str] = {}
        for entry in tensor_entries:
            source = entry.get("source")
            source_group = entry.get("source_group")
            if isinstance(source_group, list):
                sources: list[TensorSource] = []
                for group_entry in source_group:
                    if not isinstance(group_entry, dict):
                        raise AssertionError("tensor source_group entry is not metadata")
                    source_name = str(group_entry["name"])
                    sources.append(sources_by_name[source_name])
                write_padding(int(entry["payload_offset"]))
                digest_hex = write_transformed_tensor_group(
                    sources=sources,
                    dst=out,
                    payload_hash=payload_hash,
                    transform=str(entry["transform"]),
                )
                written_hashes[str(entry["name"])] = digest_hex
                continue

            if not isinstance(source, dict):
                raise AssertionError("tensor entry source metadata missing")
            source_name = str(source["name"])
            src = sources_by_name[source_name]
            write_padding(int(entry["payload_offset"]))
            digest_hex = write_transformed_tensor(
                src=src,
                dst=out,
                payload_hash=payload_hash,
                transform=str(entry["transform"]),
            )
            if str(entry["transform"]) == "raw" and digest_hex != entry["sha256"]:
                raise AssertionError(f"{src.name} checksum changed during write")
            written_hashes[str(entry["name"])] = digest_hex

        write_padding(payload_len)
        payload_sha = payload_hash.hexdigest()
        for entry in tensor_entries:
            entry["sha256"] = written_hashes[str(entry["name"])]
        final_manifest_bytes = build_manifest_with_payload_offset(payload_offset)
        if len(final_manifest_bytes) != len(manifest_bytes):
            raise RuntimeError("native manifest length changed after tensor checksum finalization")
        out.seek(HEADER_SIZE)
        out.write(final_manifest_bytes)
        out.seek(0)
        out.write(
            build_header(
                payload_kind=PAYLOAD_KIND_NATIVE,
                manifest_offset=HEADER_SIZE,
                manifest_len=len(final_manifest_bytes),
                payload_offset=payload_offset,
                payload_len=payload_len,
                payload_sha256=payload_sha,
                # Header flag bit 0 is read by the kernel as payload_2m_aligned;
                # it must reflect the 2 MiB constant, not whatever
                # --payload-alignment was passed.
                payload_aligned=(payload_offset % DEFAULT_PAYLOAD_ALIGNMENT == 0),
                version=artifact_version,
            )
        )

    if args.verify:
        verify_path(output, strict=False, hash_payload=False, json_output=False)

    print(f"wrote {output}")
    print(
        f"payload_kind=native-smodel tensors={len(tensor_entries)} "
        f"quant={args.quant} payload_len={payload_len}"
    )
    return 0


def make_gguf_compat_manifest(
    *,
    src: Path,
    source_size: int,
    source_sha256: str,
    payload_offset: int,
    payload_alignment: int,
    target_product: str,
    profile: str,
    target_arch: str,
    note: str | None,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "format": "SilicatePack",
        "container": "gguf-compat",
        "magic": "SILM",
        "version": VERSION,
        "artifact_extension": ".smodel",
        "ready_label": "Ready for Zero Server (compatibility import)",
        "target_product": target_product,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source": {
            "format": "GGUF",
            "path": str(src),
            "name": src.name,
            "size": source_size,
            "sha256": source_sha256,
        },
        "payload": {
            "kind": "gguf-compat",
            "offset": payload_offset,
            "length": source_size,
            "alignment": payload_alignment,
            "sha256": source_sha256,
        },
        "compatibility": {
            "native_zero_server_format": False,
            "reason": "Legacy benchmark and migration path only. Production artifacts should be packed with pack-hf.",
        },
        "runtime_profiles": {
            "cpu": {
                "profile": profile,
                "target_arch": target_arch,
                "status": "compatibility-only",
            },
            "gpu": {
                "status": "not-applicable",
            },
        },
    }
    if note:
        manifest["note"] = note
    return manifest


def import_gguf_compat(args: argparse.Namespace) -> int:
    src = Path(args.input).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    if read_magic(src) != b"GGUF":
        raise ValueError(f"{src} is not a raw GGUF file")
    if dst.exists() and not args.force:
        raise FileExistsError(f"{dst} exists; pass --force to overwrite")

    source_size = src.stat().st_size
    source_sha = sha256_file(src)

    manifest = make_gguf_compat_manifest(
        src=src,
        source_size=source_size,
        source_sha256=source_sha,
        payload_offset=0,
        payload_alignment=args.payload_alignment,
        target_product=args.target_product,
        profile=args.profile,
        target_arch=args.target_arch,
        note=args.note,
    )
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    payload_offset = align_up(HEADER_SIZE + len(manifest_bytes), args.payload_alignment)
    manifest = make_gguf_compat_manifest(
        src=src,
        source_size=source_size,
        source_sha256=source_sha,
        payload_offset=payload_offset,
        payload_alignment=args.payload_alignment,
        target_product=args.target_product,
        profile=args.profile,
        target_arch=args.target_arch,
        note=args.note,
    )
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    payload_offset = align_up(HEADER_SIZE + len(manifest_bytes), args.payload_alignment)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as out:
        out.write(
            build_header(
                payload_kind=PAYLOAD_KIND_GGUF_COMPAT,
                manifest_offset=HEADER_SIZE,
                manifest_len=len(manifest_bytes),
                payload_offset=payload_offset,
                payload_len=source_size,
                payload_sha256=source_sha,
                payload_aligned=(payload_offset % DEFAULT_PAYLOAD_ALIGNMENT == 0),
            )
        )
        out.write(manifest_bytes)
        out.write(b"\0" * (payload_offset - out.tell()))
        with src.open("rb") as inp:
            copy_stream(inp, out)

    if args.verify:
        verify_path(dst, strict=False, hash_payload=False, json_output=False)

    print(f"wrote {dst}")
    print(f"payload_kind=gguf-compat payload_offset={payload_offset} payload_len={source_size}")
    return 0


def inspect_path(path: Path, *, json_output: bool) -> dict[str, object]:
    header = parse_header(read_exact_header(path))
    with path.open("rb") as f:
        f.seek(int(header["manifest_offset"]))
        manifest_bytes = f.read(int(header["manifest_len"]))
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    result = {"header": header, "manifest": manifest, "file": str(path)}
    if json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{path}")
        print(
            f"  magic={header['magic']} version={header['version']} "
            f"payload={header['payload_kind_label']}"
        )
        print(f"  payload_offset={header['payload_offset']} payload_len={header['payload_len']}")
        print(
            f"  payload_2m_aligned={header['payload_2m_aligned']} "
            f"sha256={header['payload_sha256']}"
        )
        print(f"  target_product={manifest.get('target_product')} ready_label={manifest.get('ready_label')}")
        print(f"  tensor_count={manifest.get('tensor_count', 'n/a')}")
        anchors = manifest.get("validation_anchors")
        if isinstance(anchors, dict):
            anchor_list = anchors.get("anchors")
            anchor_count = len(anchor_list) if isinstance(anchor_list, list) else 0
            print(f"  validation_anchors={anchors.get('mode', 'unknown')} count={anchor_count}")
    return result


def _issue(issues: list[str], message: str) -> None:
    issues.append(message)


def _read_at(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(length)
    if len(data) != length:
        raise ValueError(f"short read at {offset:#x}: wanted {length}, got {len(data)}")
    return data


def _manifest_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001 - verifier reports malformed fields via caller.
        return default


def _native_index_summary(path: Path, payload_offset: int, payload_len: int) -> dict[str, object]:
    if payload_len < NATIVE_INDEX_HEADER_SIZE:
        raise ValueError("native payload is smaller than SIDX header")
    header = _read_at(path, payload_offset, NATIVE_INDEX_HEADER_SIZE)
    magic, version, header_len, tensor_count, entry_size, names_offset, names_len, data_base = struct.unpack_from(
        "<4sIIIIIIQ", header, 0
    )
    expert_weights_scale = struct.unpack_from("<f", header, 124)[0]
    if magic != NATIVE_INDEX_MAGIC:
        raise ValueError(f"bad native index magic: {magic!r}")
    if version != NATIVE_INDEX_VERSION:
        raise ValueError(f"unsupported native index version: {version}")
    if header_len != NATIVE_INDEX_HEADER_SIZE:
        raise ValueError(f"unsupported native index header size: {header_len}")
    if entry_size != NATIVE_TENSOR_ENTRY_SIZE:
        raise ValueError(f"unsupported native tensor entry size: {entry_size}")

    entries_start = NATIVE_INDEX_HEADER_SIZE
    entries_end = entries_start + tensor_count * entry_size
    names_end = names_offset + names_len
    if entries_end > payload_len:
        raise ValueError("native tensor entries exceed payload")
    if names_offset < entries_end:
        raise ValueError("native names blob overlaps tensor entries")
    if names_end > payload_len:
        raise ValueError("native names blob exceeds payload")

    index_bytes = _read_at(path, payload_offset, names_end)
    names = index_bytes[names_offset:names_end]
    dtype_by_id = {v: k for k, v in NATIVE_DTYPE_IDS.items()}
    seen_names: set[str] = set()
    max_tensor_end = 0
    interleaved_count = 0

    for i in range(tensor_count):
        off = entries_start + i * entry_size
        (
            name_off,
            name_len,
            dtype_id,
            rank,
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            tensor_payload_offset,
            byte_len,
            _flags,
        ) = NATIVE_TENSOR_ENTRY_STRUCT.unpack_from(index_bytes, off)
        if dtype_id not in dtype_by_id:
            raise ValueError(f"tensor[{i}] has unsupported dtype id {dtype_id}")
        if rank > 8:
            raise ValueError(f"tensor[{i}] rank {rank} exceeds 8")
        name_end = name_off + name_len
        if name_end > len(names):
            raise ValueError(f"tensor[{i}] name exceeds names blob")
        name = names[name_off:name_end].decode("utf-8")
        if not name:
            raise ValueError(f"tensor[{i}] has empty name")
        if name in seen_names:
            raise ValueError(f"duplicate tensor name {name!r}")
        seen_names.add(name)
        dims = (d0, d1, d2, d3, d4, d5, d6, d7)[:rank]
        if any(dim == 0 for dim in dims):
            raise ValueError(f"tensor[{i}] {name!r} has zero dimension")
        if dtype_by_id[dtype_id] in INTERLEAVE_BASE_DTYPE:
            # Interleaving is only defined for rank-2 matmul weights with
            # complete groups; embeddings must stay row-major (the kernel
            # reads them row-wise outside the matmul kernels).
            if rank != 2:
                raise ValueError(
                    f"tensor[{i}] {name!r} is row-interleaved but has rank {rank} (must be 2)"
                )
            if d0 % INTERLEAVE_GROUP != 0:
                raise ValueError(
                    f"tensor[{i}] {name!r} is row-interleaved but its row count {d0} "
                    f"is not a multiple of {INTERLEAVE_GROUP}"
                )
            if name in INTERLEAVE_EXCLUDED_NAMES:
                raise ValueError(f"tensor[{i}] {name!r} must never be row-interleaved")
            interleaved_count += 1
        tensor_end = tensor_payload_offset + byte_len
        if tensor_end > payload_len:
            raise ValueError(f"tensor[{i}] {name!r} exceeds payload")
        if tensor_payload_offset % DEFAULT_TENSOR_ALIGNMENT != 0:
            raise ValueError(f"tensor[{i}] {name!r} is not {DEFAULT_TENSOR_ALIGNMENT}-byte aligned")
        max_tensor_end = max(max_tensor_end, tensor_end)

    return {
        "tensor_count": tensor_count,
        "entry_size": entry_size,
        "names_len": names_len,
        "data_base": data_base,
        "max_tensor_end": max_tensor_end,
        "expert_weights_scale": expert_weights_scale,
        "interleaved_tensor_count": interleaved_count,
    }


def verify_path(
    path: Path,
    *,
    strict: bool,
    hash_payload: bool,
    json_output: bool,
) -> dict[str, object]:
    path = path.expanduser().resolve()
    size = path.stat().st_size
    issues: list[str] = []
    header = parse_header(read_exact_header(path))

    version = int(header["version"])
    header_len = int(header["header_len"])
    manifest_offset = int(header["manifest_offset"])
    manifest_len = int(header["manifest_len"])
    payload_offset = int(header["payload_offset"])
    payload_len = int(header["payload_len"])
    payload_kind = int(header["payload_kind"])
    flags = int(header["flags"])
    payload_end = payload_offset + payload_len
    manifest_end = manifest_offset + manifest_len

    if version not in (VERSION, VERSION_INTERLEAVED):
        _issue(issues, f"unsupported header version {version}")
    if header_len != HEADER_SIZE:
        _issue(issues, f"unsupported header size {header_len}")
    if payload_kind not in (PAYLOAD_KIND_GGUF_COMPAT, PAYLOAD_KIND_NATIVE):
        _issue(issues, f"unsupported payload kind {payload_kind}")
    if manifest_len <= 0:
        _issue(issues, "manifest is empty")
    if manifest_offset < HEADER_SIZE:
        _issue(issues, "manifest starts before header end")
    if manifest_end > payload_offset:
        _issue(issues, "manifest overlaps payload")
    if payload_offset < HEADER_SIZE:
        _issue(issues, "payload starts before header end")
    if payload_end > size:
        _issue(issues, "payload exceeds file size")
    if flags & 1 and payload_offset % DEFAULT_PAYLOAD_ALIGNMENT != 0:
        _issue(issues, "payload_2m_aligned flag set but payload is not 2 MiB aligned")

    manifest: dict[str, object] = {}
    if not issues:
        manifest_bytes = _read_at(path, manifest_offset, manifest_len)
        try:
            manifest_obj = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - verifier should report the concrete parser failure.
            _issue(issues, f"manifest JSON parse failed: {exc}")
            manifest_obj = {}
        if not isinstance(manifest_obj, dict):
            _issue(issues, "manifest root is not an object")
        else:
            manifest = manifest_obj

    native_summary: dict[str, object] | None = None
    if payload_kind == PAYLOAD_KIND_NATIVE and not issues:
        try:
            native_summary = _native_index_summary(path, payload_offset, payload_len)
            manifest_tensor_count = manifest.get("tensor_count")
            if isinstance(manifest_tensor_count, int) and manifest_tensor_count != native_summary["tensor_count"]:
                _issue(
                    issues,
                    f"manifest tensor_count={manifest_tensor_count} does not match SIDX tensor_count={native_summary['tensor_count']}",
                )
            layout = manifest.get("layout")
            if isinstance(layout, dict):
                if int(layout.get("payload_offset", payload_offset)) != payload_offset:
                    _issue(issues, "manifest layout.payload_offset does not match header")
                if int(layout.get("payload_len", payload_len)) != payload_len:
                    _issue(issues, "manifest layout.payload_len does not match header")
        except Exception as exc:  # noqa: BLE001
            _issue(issues, f"native SIDX verification failed: {exc}")
    elif payload_kind == PAYLOAD_KIND_GGUF_COMPAT and not issues:
        if _read_at(path, payload_offset, 4) != b"GGUF":
            _issue(issues, "GGUF compatibility payload does not start with GGUF")

    if strict:
        if payload_kind != PAYLOAD_KIND_NATIVE:
            _issue(issues, "strict verification requires native .smodel payload")
        if manifest.get("target_product") != "Zero Server":
            _issue(issues, "strict verification requires target_product=Zero Server")
        if manifest.get("ready_label") != "Ready for Zero Server":
            _issue(issues, "strict verification requires ready_label=Ready for Zero Server")
        if manifest.get("container") != "native":
            _issue(issues, "strict verification requires container=native")
        if payload_offset % DEFAULT_PAYLOAD_ALIGNMENT != 0:
            _issue(issues, "strict verification requires 2 MiB payload alignment")

        layout = manifest.get("layout")
        if not isinstance(layout, dict):
            _issue(issues, "strict verification requires layout object")
        else:
            if layout.get("gguf_payload") is not False:
                _issue(issues, "strict verification requires layout.gguf_payload=false")
            if _manifest_int(layout.get("payload_offset"), -1) != payload_offset:
                _issue(issues, "layout.payload_offset does not match header")
            if _manifest_int(layout.get("payload_len"), -1) != payload_len:
                _issue(issues, "layout.payload_len does not match header")
            if _manifest_int(layout.get("payload_alignment"), 0) < DEFAULT_PAYLOAD_ALIGNMENT:
                _issue(issues, "layout.payload_alignment must be at least 2 MiB")
            native_index = layout.get("native_index")
            if not isinstance(native_index, dict):
                _issue(issues, "strict verification requires layout.native_index object")
            else:
                if native_index.get("magic") != NATIVE_INDEX_MAGIC.decode("ascii"):
                    _issue(issues, "layout.native_index.magic must be SIDX")
                if _manifest_int(native_index.get("version"), 0) != NATIVE_INDEX_VERSION:
                    _issue(issues, "layout.native_index.version mismatch")

        compatibility = manifest.get("compatibility")
        if not isinstance(compatibility, dict):
            _issue(issues, "strict verification requires compatibility object")
        elif compatibility.get("gguf_runtime_payload") is not False:
            _issue(issues, "strict verification requires compatibility.gguf_runtime_payload=false")

        runtime_profiles = manifest.get("runtime_profiles")
        if not isinstance(runtime_profiles, dict):
            _issue(issues, "strict verification requires runtime_profiles object")
        else:
            cpu_profile = runtime_profiles.get("cpu")
            if not isinstance(cpu_profile, dict):
                _issue(issues, "strict verification requires runtime_profiles.cpu object")
            elif cpu_profile.get("status") != "performance-v1":
                _issue(issues, "strict verification requires runtime_profiles.cpu.status=performance-v1")

        sections = manifest.get("sections")
        if not isinstance(sections, dict):
            _issue(issues, "strict verification requires sections object")
        else:
            native_section = sections.get("native_index")
            if not isinstance(native_section, dict):
                _issue(issues, "strict verification requires native_index section")
            else:
                if native_section.get("magic") != NATIVE_INDEX_MAGIC.decode("ascii"):
                    _issue(issues, "native_index section magic must be SIDX")
                if _manifest_int(native_section.get("tensor_count"), -1) != manifest.get("tensor_count"):
                    _issue(issues, "native_index section tensor_count does not match manifest tensor_count")
            for section_name in ("config_json", "tokenizer_json"):
                section = sections.get(section_name)
                if not isinstance(section, dict):
                    _issue(issues, f"strict verification requires {section_name} section")
                    continue
                try:
                    section_offset = int(section["payload_offset"])
                    section_len = int(section["byte_len"])
                except Exception:  # noqa: BLE001 - verifier reports malformed manifest fields.
                    _issue(issues, f"{section_name} section misses payload_offset/byte_len")
                    continue
                if section_len <= 0:
                    _issue(issues, f"{section_name} section is empty")
                if section_offset < 0 or section_offset + section_len > payload_len:
                    _issue(issues, f"{section_name} section exceeds native payload")
        tensors = manifest.get("tensors")
        if isinstance(tensors, list):
            for idx, tensor in enumerate(tensors):
                if not isinstance(tensor, dict):
                    _issue(issues, f"tensor[{idx}] manifest entry is not an object")
                    continue
                sha = tensor.get("sha256")
                if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
                    _issue(issues, f"tensor[{idx}] misses valid native sha256")
                elif sha == "0" * 64:
                    _issue(issues, f"tensor[{idx}] native sha256 was not finalized")
        if native_summary is not None and isinstance(manifest.get("model_config"), dict):
            expected_fields = native_model_config_fields(manifest["model_config"])
            expected_scale = float(expected_fields["expert_weights_scale"])
            actual_scale = float(native_summary.get("expert_weights_scale", 0.0))
            if not math.isfinite(actual_scale):
                _issue(issues, "SIDX expert_weights_scale is not finite")
            elif abs(actual_scale - expected_scale) > 1e-6:
                _issue(
                    issues,
                    "SIDX expert_weights_scale does not match manifest model_config "
                    f"({actual_scale} != {expected_scale})",
                )
        # `.smodel`-v2 consistency: the SIDX interleaved-tensor count,
        # the header version, and the manifest layout.row_interleave
        # field must agree — an artifact that carries interleaved
        # tensors under a v1 header would be silently misread by old
        # kernels through plain-layout kernels.
        if native_summary is not None:
            sidx_interleaved = _manifest_int(native_summary.get("interleaved_tensor_count"), 0)
            manifest_interleave = 0
            layout_obj = manifest.get("layout")
            if isinstance(layout_obj, dict):
                manifest_interleave = _manifest_int(layout_obj.get("row_interleave"), 0)
            if sidx_interleaved > 0 and version != VERSION_INTERLEAVED:
                _issue(
                    issues,
                    f"SIDX carries {sidx_interleaved} row-interleaved tensors but header "
                    f"version is {version} (must be {VERSION_INTERLEAVED})",
                )
            if sidx_interleaved == 0 and version == VERSION_INTERLEAVED:
                _issue(issues, "header version 2 but SIDX has no row-interleaved tensors")
            if sidx_interleaved > 0 and manifest_interleave != INTERLEAVE_GROUP:
                _issue(
                    issues,
                    "SIDX carries row-interleaved tensors but manifest "
                    f"layout.row_interleave is {manifest_interleave}",
                )
            if sidx_interleaved == 0 and manifest_interleave not in (0, None):
                _issue(issues, "manifest layout.row_interleave set but SIDX has no interleaved tensors")

        anchors = manifest.get("validation_anchors")
        if not isinstance(anchors, dict):
            _issue(issues, "strict verification requires validation_anchors object")
        else:
            if anchors.get("mode") != "strict":
                _issue(issues, "strict verification requires validation_anchors.mode=strict")
            anchor_list = anchors.get("anchors")
            if not isinstance(anchor_list, list) or not anchor_list:
                _issue(issues, "strict verification requires at least one validation anchor")
            else:
                first = anchor_list[0]
                if not isinstance(first, dict):
                    _issue(issues, "first validation anchor is not an object")
                else:
                    if "expected_next_token" not in first:
                        _issue(issues, "first validation anchor misses expected_next_token")
                    if "expected_logit_bits" not in first:
                        _issue(issues, "first validation anchor misses expected_logit_bits")

    payload_sha256 = None
    if hash_payload and not issues:
        payload_sha256 = sha256_range(path, payload_offset, payload_len)
        if payload_sha256 != header["payload_sha256"]:
            _issue(issues, "payload sha256 does not match header")

    result = {
        "file": str(path),
        "ok": not issues,
        "issues": issues,
        "size": size,
        "header": header,
        "manifest": {
            "target_product": manifest.get("target_product"),
            "container": manifest.get("container"),
            "tensor_count": manifest.get("tensor_count"),
            "ready_label": manifest.get("ready_label"),
        },
        "native_index": native_summary,
        "payload_sha256": payload_sha256,
    }
    if json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif issues:
        print(f"{path}: FAIL")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"{path}: OK")
        print(
            f"  payload={header['payload_kind_label']} offset={payload_offset} len={payload_len} "
            f"aligned={bool(flags & 1)}"
        )
        if native_summary:
            print(
                f"  native tensors={native_summary['tensor_count']} names={native_summary['names_len']} "
                f"data_base={native_summary['data_base']:#x}"
            )
            interleaved = native_summary.get("interleaved_tensor_count", 0)
            if interleaved:
                print(f"  row_interleave=4 tensors={interleaved} (.smodel-v2)")
        anchors = manifest.get("validation_anchors")
        if isinstance(anchors, dict):
            anchor_list = anchors.get("anchors")
            anchor_count = len(anchor_list) if isinstance(anchor_list, list) else 0
            print(f"  validation_anchors={anchors.get('mode', 'unknown')} count={anchor_count}")
    if issues:
        raise ValueError("SilicatePack verification failed")
    return result


def inspect(args: argparse.Namespace) -> int:
    inspect_path(Path(args.path).expanduser().resolve(), json_output=args.json)
    return 0


def verify(args: argparse.Namespace) -> int:
    verify_path(
        Path(args.path).expanduser().resolve(),
        strict=args.strict,
        hash_payload=args.hash_payload,
        json_output=args.json,
    )
    return 0


def set_anchors(args: argparse.Namespace) -> int:
    if not getattr(args, "capture", False) and (
        args.anchor_next_token is None or args.anchor_logit_bits is None
    ):
        raise ValueError(
            "set-anchors requires --anchor-next-token AND --anchor-logit-bits "
            "(strict mode), or --capture for first-run anchor collection"
        )
    path = Path(args.path).expanduser().resolve()
    header = parse_header(read_exact_header(path))
    with path.open("rb") as f:
        f.seek(int(header["manifest_offset"]))
        manifest_bytes = f.read(int(header["manifest_len"]))
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest root is not an object")
    if header["payload_kind"] != PAYLOAD_KIND_NATIVE:
        raise ValueError("strict validation anchors are only supported for native .smodel payloads")

    manifest["validation_anchors"] = build_validation_anchors_from_args(args)
    new_manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    manifest_offset = int(header["manifest_offset"])
    payload_offset = int(header["payload_offset"])
    if manifest_offset != HEADER_SIZE:
        raise ValueError("unexpected manifest offset; refusing in-place rewrite")
    max_manifest_len = payload_offset - manifest_offset
    if len(new_manifest_bytes) > max_manifest_len:
        raise ValueError(
            f"updated manifest is {len(new_manifest_bytes)} bytes, only {max_manifest_len} bytes available before payload"
        )

    with path.open("r+b") as f:
        f.seek(manifest_offset)
        f.write(new_manifest_bytes)
        f.write(b"\0" * (max_manifest_len - len(new_manifest_bytes)))
        f.seek(0)
        f.write(
            build_header(
                payload_kind=int(header["payload_kind"]),
                manifest_offset=manifest_offset,
                manifest_len=len(new_manifest_bytes),
                payload_offset=payload_offset,
                payload_len=int(header["payload_len"]),
                payload_sha256=str(header["payload_sha256"]),
                payload_aligned=bool(header["flags"] & 1),
                # Preserve the artifact version — a v2 (row-interleaved)
                # artifact must not be silently downgraded to v1 by an
                # in-place anchor update.
                version=int(header["version"]),
            )
        )

    print(f"updated anchors in {path}")
    if getattr(args, "capture", False):
        print(f"anchor={args.anchor_name} mode=capture (kernel logs measured values, no hard gate)")
    else:
        print(
            f"anchor={args.anchor_name} expected_next_token={args.anchor_next_token} "
            f"expected_logit_bits={format_u32_hex(args.anchor_logit_bits)}"
        )
    return 0


def add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, help="destination .smodel file")
    parser.add_argument("--target-product", default="Zero Server")
    parser.add_argument("--profile", default="cpu-avx512")
    parser.add_argument("--target-arch", default="x86_64-zen4")
    parser.add_argument("--payload-alignment", type=int, default=DEFAULT_PAYLOAD_ALIGNMENT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--force", action="store_true")


def add_anchor_args(parser: argparse.ArgumentParser, *, require_expected: bool) -> None:
    parser.add_argument("--anchor-name", default="zero-server-smoke-v1")
    parser.add_argument("--anchor-prompt", default="Hello")
    parser.add_argument(
        "--anchor-prompt-tokens",
        help="comma-separated u32 token ids used for the anchor prompt",
    )
    parser.add_argument(
        "--anchor-next-token",
        type=parse_u32_literal,
        required=require_expected,
        help="expected first anchor argmax token id; accepts decimal or 0x-prefixed hex",
    )
    parser.add_argument(
        "--anchor-logit-bits",
        type=parse_u32_literal,
        required=require_expected,
        help="expected top-1 logit f32 bits; accepts decimal or 0x-prefixed hex",
    )
    parser.add_argument(
        "--anchor-generated-tokens",
        help="optional comma-separated generated u32 token ids from the reference run",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="silicatepack",
        description="Create and inspect native Zero Server .smodel artifacts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pack_p = sub.add_parser(
        "pack-hf",
        help="pack Hugging Face SafeTensors + config/tokenizer into a native .smodel",
    )
    pack_p.add_argument("--input-dir", required=True, help="directory containing safetensors/config/tokenizer")
    pack_p.add_argument("--safetensors", nargs="*", help="explicit safetensors files; defaults to *.safetensors")
    pack_p.add_argument("--config", help="config.json path; defaults to input-dir/config.json")
    pack_p.add_argument("--tokenizer", help="tokenizer.json path; defaults to input-dir/tokenizer.json")
    pack_p.add_argument("--source-repo")
    pack_p.add_argument("--source-revision")
    pack_p.add_argument("--license")
    pack_p.add_argument(
        "--quant",
        choices=QUANT_CHOICES,
        default="auto",
        help="native tensor emission policy: auto uses Q8_0 for embeddings/LM-head and Q4_0 for matrix weights",
    )
    pack_p.add_argument(
        "--interleave",
        type=int,
        choices=(0, INTERLEAVE_GROUP),
        default=0,
        help=(
            ".smodel-v2 row-interleaved layout: store eligible rank-2 Q4_0/Q8_0 matmul "
            "tensors in 4-row group-blocks (dtype ids Q4_0X4/Q8_0X4). Dequantized values "
            "are identical to the plain layout, so existing token/logit anchors stay "
            "valid; requires a Zero Server build with the interleaved AVX-512 kernels."
        ),
    )
    pack_p.add_argument(
        "--no-normalize-names",
        dest="normalize_names",
        action="store_false",
        help="preserve Hugging Face tensor names instead of mapping them to Zero Server runtime names",
    )
    pack_p.add_argument("--tensor-alignment", type=int, default=DEFAULT_TENSOR_ALIGNMENT)
    add_common_output_args(pack_p)
    add_anchor_args(pack_p, require_expected=False)
    pack_p.set_defaults(normalize_names=True)
    pack_p.set_defaults(func=pack_hf)

    gguf_p = sub.add_parser(
        "import-gguf-compat",
        help="legacy-only: wrap a raw GGUF for compatibility testing",
    )
    gguf_p.add_argument("--input", required=True, help="source GGUF file")
    gguf_p.add_argument("--note", default=None)
    add_common_output_args(gguf_p)
    gguf_p.set_defaults(func=import_gguf_compat)

    convert_p = sub.add_parser("convert", help=argparse.SUPPRESS)
    convert_p.add_argument("--input", required=True, help="source GGUF file")
    convert_p.add_argument("--note", default=None)
    add_common_output_args(convert_p)
    convert_p.set_defaults(func=import_gguf_compat)

    inspect_p = sub.add_parser("inspect", help="print .smodel header and manifest")
    inspect_p.add_argument("path")
    inspect_p.add_argument("--json", action="store_true")
    inspect_p.set_defaults(func=inspect)

    verify_p = sub.add_parser("verify", help="validate .smodel structure and optional strict anchors")
    verify_p.add_argument("path")
    verify_p.add_argument("--strict", action="store_true", help="require native payload and strict validation anchors")
    verify_p.add_argument(
        "--hash-payload",
        action="store_true",
        help="also hash the complete payload and compare it to the header; expensive for large models",
    )
    verify_p.add_argument("--json", action="store_true")
    verify_p.set_defaults(func=verify)

    anchor_p = sub.add_parser(
        "set-anchors",
        help="write strict validation anchors into an existing native .smodel manifest",
    )
    anchor_p.add_argument("path")
    anchor_p.add_argument("--profile", default="cpu-avx512")
    anchor_p.add_argument("--target-arch", default="x86_64-zen4")
    anchor_p.add_argument(
        "--capture",
        action="store_true",
        help=(
            "write the anchor in CAPTURE mode (no expected values): the kernel "
            "logs the measured token/logit_bits at the first run and continues "
            "instead of hard-failing. Use after a quantizer change invalidated "
            "the previous baseline values; promote to strict with the captured "
            "values afterwards."
        ),
    )
    add_anchor_args(anchor_p, require_expected=False)
    anchor_p.set_defaults(func=set_anchors)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI should report concise failures.
        print(f"silicatepack: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
