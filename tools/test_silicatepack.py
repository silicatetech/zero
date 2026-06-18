#!/usr/bin/env python3
"""Focused regression tests for the native SilicatePack contract."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SILICATEPACK_PATH = REPO_ROOT / "tools" / "silicatepack.py"


def load_silicatepack():
    spec = importlib.util.spec_from_file_location("silicatepack", SILICATEPACK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load tools/silicatepack.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sp = load_silicatepack()


def deepseek_config(scale: float = 2.5) -> dict[str, object]:
    return {
        "model_type": "deepseek_v2",
        "num_hidden_layers": 1,
        "max_position_embeddings": 128,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "eos_token_id": 2,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 8,
        "kv_lora_rank": 4,
        "q_lora_rank": 8,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "routed_scaling_factor": scale,
    }


def strict_anchors() -> dict[str, object]:
    return {
        "mode": "strict",
        "anchors": [
            {
                "name": "unit",
                "expected_next_token": 1,
                "expected_logit_bits": "0x3f800000",
            }
        ],
    }


def write_smodel(
    path: Path,
    *,
    config: dict[str, object],
    sections: dict[str, dict[str, object]],
    payload: bytes,
) -> None:
    sections = {name: dict(value) for name, value in sections.items()}
    native_section = sections.setdefault("native_index", {})
    native_section.setdefault("magic", sp.NATIVE_INDEX_MAGIC.decode("ascii"))
    native_section.setdefault("version", sp.NATIVE_INDEX_VERSION)
    native_section.setdefault("tensor_count", 0)
    manifest = {
        "format": "SilicatePack",
        "container": "native",
        "target_product": "Zero Server",
        "ready_label": "Ready for Zero Server",
        "tensor_count": 0,
        "model_config": config,
        "sections": sections,
        "validation_anchors": strict_anchors(),
        "layout": {
            "payload_offset": 0,
            "payload_len": len(payload),
            "payload_alignment": sp.DEFAULT_PAYLOAD_ALIGNMENT,
            "gguf_payload": False,
            "native_index": {
                "magic": sp.NATIVE_INDEX_MAGIC.decode("ascii"),
                "version": sp.NATIVE_INDEX_VERSION,
                "tensor_count": 0,
            },
        },
        "compatibility": {
            "gguf_runtime_payload": False,
        },
        "runtime_profiles": {
            "cpu": {
                "profile": "cpu-avx512",
                "target_arch": "x86_64-zen4",
                "status": "performance-v1",
            },
            "gpu": {
                "status": "not-applicable",
            },
        },
    }
    manifest_bytes = b""
    payload_offset = sp.DEFAULT_PAYLOAD_ALIGNMENT
    for _ in range(8):
        manifest["layout"]["payload_offset"] = payload_offset
        manifest["layout"]["payload_len"] = len(payload)
        manifest_bytes = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
        next_payload_offset = sp.align_up(sp.HEADER_SIZE + len(manifest_bytes), sp.DEFAULT_PAYLOAD_ALIGNMENT)
        if next_payload_offset == payload_offset:
            break
        payload_offset = next_payload_offset
    else:
        raise RuntimeError("test manifest size did not converge")
    header = sp.build_header(
        payload_kind=sp.PAYLOAD_KIND_NATIVE,
        manifest_offset=sp.HEADER_SIZE,
        manifest_len=len(manifest_bytes),
        payload_offset=payload_offset,
        payload_len=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_aligned=True,
    )
    padding = b"\0" * (payload_offset - sp.HEADER_SIZE - len(manifest_bytes))
    path.write_bytes(header + manifest_bytes + padding + payload)


class SilicatePackNativeContractTests(unittest.TestCase):
    def test_sidx_preserves_expert_weights_scale(self) -> None:
        config = deepseek_config(scale=2.5)
        index = sp.build_native_index(
            config_json=config,
            tensor_entries=[],
            names_blob=b"",
            name_offsets={},
            names_offset=sp.NATIVE_INDEX_HEADER_SIZE,
            data_base=0,
        )

        self.assertAlmostEqual(struct.unpack_from("<f", index, 124)[0], 2.5)

    def test_strict_verify_requires_native_config_and_tokenizer_sections(self) -> None:
        config = deepseek_config(scale=2.5)
        index = sp.build_native_index(
            config_json=config,
            tensor_entries=[],
            names_blob=b"",
            name_offsets={},
            names_offset=sp.NATIVE_INDEX_HEADER_SIZE,
            data_base=0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-tokenizer.smodel"
            write_smodel(
                path,
                config=config,
                sections={
                    "native_index": {
                        "payload_offset": 0,
                        "byte_len": len(index),
                    },
                },
                payload=index,
            )

            with self.assertRaises(ValueError):
                with contextlib.redirect_stdout(io.StringIO()):
                    sp.verify_path(path, strict=True, hash_payload=True, json_output=False)

    def test_strict_verify_accepts_native_sidecars_and_expert_scale(self) -> None:
        config = deepseek_config(scale=2.5)
        index = sp.build_native_index(
            config_json=config,
            tensor_entries=[],
            names_blob=b"",
            name_offsets={},
            names_offset=sp.NATIVE_INDEX_HEADER_SIZE,
            data_base=0,
        )
        config_json = json.dumps(config, sort_keys=True).encode("utf-8")
        tokenizer_json = b'{"type":"unit-tokenizer"}'
        config_offset = len(index)
        tokenizer_offset = config_offset + len(config_json)
        payload = index + config_json + tokenizer_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native-ok.smodel"
            write_smodel(
                path,
                config=config,
                sections={
                    "native_index": {
                        "payload_offset": 0,
                        "byte_len": len(index),
                    },
                    "config_json": {
                        "payload_offset": config_offset,
                        "byte_len": len(config_json),
                    },
                    "tokenizer_json": {
                        "payload_offset": tokenizer_offset,
                        "byte_len": len(tokenizer_json),
                    },
                },
                payload=payload,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                result = sp.verify_path(path, strict=True, hash_payload=True, json_output=False)

        self.assertTrue(result["ok"])
        self.assertEqual(result["native_index"]["tensor_count"], 0)
        self.assertAlmostEqual(result["native_index"]["expert_weights_scale"], 2.5)

    def test_router_tensors_stay_f32_under_quant_policies(self) -> None:
        # Routing logits are precision-critical: the MoE router weight and
        # bias must never be quantized, regardless of quant policy.
        def source(name: str, shape: tuple[int, ...]) -> object:
            elements = 1
            for dim in shape:
                elements *= dim
            return sp.TensorSource(
                name=name,
                dtype="BF16",
                shape=shape,
                file=Path("unused.safetensors"),
                data_begin=0,
                data_end=2 * elements,
                file_data_offset=0,
            )

        router = source("blk.0.ffn_gate_inp.weight", (4, 32))
        bias = source("blk.0.ffn_gate_inp_bias.weight", (4,))
        expert = source("blk.0.ffn_gate_exps.weight", (4, 8, 32))

        for policy in ("auto", "q4_0", "q8_0"):
            self.assertEqual(
                sp.select_output_dtype(router, "blk.0.ffn_gate_inp.weight", policy),
                ("F32", "f32"),
                f"router weight must stay F32 under policy {policy}",
            )
            self.assertEqual(
                sp.select_output_dtype(bias, "blk.0.ffn_gate_inp_bias.weight", policy),
                ("F32", "f32"),
                f"router bias must stay F32 under policy {policy}",
            )

        # Sanity: expert tensors are still quantized under auto.
        self.assertEqual(
            sp.select_output_dtype(expert, "blk.0.ffn_gate_exps.weight", "auto"),
            ("Q4_0", "q4_0"),
        )

    def test_q4_0_block_matches_ggml_reference_encoding(self) -> None:
        # GGML reference: d = max_signed / -8; the max-magnitude value
        # maps to level -8 (nibble 0). Decode is (nibble - 8) * d.
        def decode_block(block: bytes) -> list[float]:
            (d,) = struct.unpack("<e", block[:2])
            out = [0.0] * 32
            for j in range(16):
                byte = block[2 + j]
                out[j] = ((byte & 0x0F) - 8) * d
                out[j + 16] = (((byte >> 4) & 0x0F) - 8) * d
            return out

        # Negative max-magnitude value: d = -8 / -8 = 1.0, exact levels.
        values = [0.0] * 32
        values[0] = -8.0
        values[1] = 4.0
        values[2] = 1.0
        block = sp.quantize_q4_0_block(tuple(values))
        decoded = decode_block(block)
        self.assertEqual(decoded[0], -8.0, "max-magnitude value must hit level -8")
        self.assertEqual(decoded[1], 4.0)
        self.assertEqual(decoded[2], 1.0)
        self.assertEqual(block[2] & 0x0F, 0, "level -8 (nibble 0) must be produced")

        # Positive max-magnitude value: d = 8 / -8 = -1.0; the max value
        # still lands on level -8 and decodes exactly.
        values = [0.0] * 32
        values[0] = 8.0
        values[1] = -4.0
        block = sp.quantize_q4_0_block(tuple(values))
        decoded = decode_block(block)
        self.assertEqual(decoded[0], 8.0)
        self.assertEqual(decoded[1], -4.0)

        # All-zero / non-finite-free guard block stays all zero.
        block = sp.quantize_q4_0_block(tuple([0.0] * 32))
        self.assertEqual(block, b"\0" * 18)

    def test_interleave_eligibility_rules(self) -> None:
        # Rank-2 Q4_0/Q8_0 matmul weights with complete groups qualify;
        # embeddings, rank-3 expert tensors, odd row counts and float
        # dtypes never do.
        self.assertTrue(sp.interleave_eligible("blk.0.attn_q.weight", (64, 64), "Q4_0"))
        self.assertTrue(sp.interleave_eligible("output.weight", (32, 64), "Q8_0"))
        self.assertFalse(sp.interleave_eligible("token_embd.weight", (32, 64), "Q8_0"))
        self.assertFalse(sp.interleave_eligible("model.embed_tokens.weight", (32, 64), "Q8_0"))
        self.assertFalse(sp.interleave_eligible("blk.0.ffn_gate_exps.weight", (4, 8, 32), "Q4_0"))
        self.assertFalse(sp.interleave_eligible("blk.0.attn_q.weight", (63, 64), "Q4_0"))
        self.assertFalse(sp.interleave_eligible("blk.0.attn_norm.weight", (64,), "F32"))
        self.assertFalse(sp.interleave_eligible("blk.0.attn_q.weight", (64, 64), "F32"))

    def _write_f32_source(self, tmp: Path, rows: int, cols: int) -> object:
        # Deterministic, non-degenerate values (varied magnitudes/signs).
        values = [
            ((r * 31 + c * 7) % 17 - 8) * 0.25 + 0.0625 * ((r + c) % 5)
            for r in range(rows)
            for c in range(cols)
        ]
        data = struct.pack(f"<{rows * cols}f", *values)
        src_path = tmp / "weights.bin"
        src_path.write_bytes(data)
        return sp.TensorSource(
            name="blk.0.attn_q.weight",
            dtype="F32",
            shape=(rows, cols),
            file=src_path,
            data_begin=0,
            data_end=len(data),
            file_data_offset=0,
        )

    def test_interleaved_writer_is_exact_block_permutation(self) -> None:
        # The x4 writer must emit, per (row, K-block), byte-identical
        # blocks to the plain writer — only the storage order changes:
        # group-block = d0 d1 d2 d3 then qs0 qs1 qs2 qs3. This is the
        # property that keeps all token/logit anchors valid for v2.
        rows, cols = 8, 64
        blocks_per_row = cols // 32
        for plain_t, x4_t, block_bytes in (
            ("q4_0", "q4_0x4", sp.Q4_0_BLOCK_BYTES),
            ("q8_0", "q8_0x4", sp.Q8_0_BLOCK_BYTES),
        ):
            with tempfile.TemporaryDirectory() as tmp_str:
                tmp = Path(tmp_str)
                src = self._write_f32_source(tmp, rows, cols)
                plain_out = io.BytesIO()
                sp.write_transformed_tensor(
                    src=src, dst=plain_out, payload_hash=hashlib.sha256(), transform=plain_t
                )
                x4_out = io.BytesIO()
                sp.write_transformed_tensor(
                    src=src, dst=x4_out, payload_hash=hashlib.sha256(), transform=x4_t
                )
                plain = plain_out.getvalue()
                x4 = x4_out.getvalue()
                self.assertEqual(len(plain), len(x4), f"{x4_t}: byte length must not change")

                def plain_block(row: int, b: int) -> bytes:
                    off = (row * blocks_per_row + b) * block_bytes
                    return plain[off : off + block_bytes]

                expected = bytearray()
                for group in range(0, rows, 4):
                    for b in range(blocks_per_row):
                        for lane in range(4):
                            expected += plain_block(group + lane, b)[:2]
                        for lane in range(4):
                            expected += plain_block(group + lane, b)[2:]
                self.assertEqual(bytes(expected), x4, f"{x4_t}: not the expected permutation")

    def test_interleaved_writer_rejects_partial_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = self._write_f32_source(tmp, 6, 64)
            with self.assertRaises(ValueError):
                sp.write_transformed_tensor(
                    src=src, dst=io.BytesIO(), payload_hash=hashlib.sha256(), transform="q4_0x4"
                )

    def test_pack_hf_interleave_emits_v2_artifact(self) -> None:
        # End-to-end: pack a tiny synthetic checkpoint with --interleave 4
        # and check (a) header version 2, (b) SIDX dtype ids 100/101 on
        # eligible tensors only, (c) verify --strict passes including the
        # v2 consistency checks, (d) --interleave 0 still emits v1.
        def write_safetensors(path: Path, tensors: dict[str, tuple[tuple[int, ...], bytes]]) -> None:
            header: dict[str, object] = {}
            cursor = 0
            blob = b""
            for name, (shape, data) in tensors.items():
                header[name] = {
                    "dtype": "F32",
                    "shape": list(shape),
                    "data_offsets": [cursor, cursor + len(data)],
                }
                cursor += len(data)
                blob += data
            hbytes = json.dumps(header).encode("utf-8")
            path.write_bytes(struct.pack("<Q", len(hbytes)) + hbytes + blob)

        def f32(rows: int, cols: int = 0) -> bytes:
            count = rows * cols if cols else rows
            return struct.pack(
                f"<{count}f", *[((i * 13) % 23 - 11) * 0.125 for i in range(count)]
            )

        qwen_config = {
            "model_type": "qwen3",
            "num_hidden_layers": 1,
            "max_position_embeddings": 128,
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "eos_token_id": 2,
        }

        for interleave, expected_version in ((4, sp.VERSION_INTERLEAVED), (0, sp.VERSION)):
            with tempfile.TemporaryDirectory() as tmp_str:
                tmp = Path(tmp_str)
                write_safetensors(
                    tmp / "model.safetensors",
                    {
                        "model.embed_tokens.weight": ((32, 64), f32(32, 64)),
                        "model.layers.0.self_attn.q_proj.weight": ((64, 64), f32(64, 64)),
                        "model.layers.0.input_layernorm.weight": ((64,), f32(64)),
                        "lm_head.weight": ((32, 64), f32(32, 64)),
                    },
                )
                (tmp / "config.json").write_text(json.dumps(qwen_config))
                (tmp / "tokenizer.json").write_text(json.dumps({"version": "test"}))
                out_path = tmp / "model.smodel"
                rc = sp.main(
                    [
                        "pack-hf",
                        "--input-dir",
                        str(tmp),
                        "--output",
                        str(out_path),
                        "--quant",
                        "auto",
                        "--interleave",
                        str(interleave),
                        "--anchor-name",
                        "unit",
                        "--anchor-prompt",
                        "Hello",
                        "--anchor-prompt-tokens",
                        "9707",
                        "--anchor-next-token",
                        "1",
                        "--anchor-logit-bits",
                        "0x3f800000",
                        "--force",
                    ]
                )
                self.assertEqual(rc, 0, "pack-hf failed")

                header = sp.parse_header(sp.read_exact_header(out_path))
                self.assertEqual(int(header["version"]), expected_version)

                summary = sp._native_index_summary(
                    out_path, int(header["payload_offset"]), int(header["payload_len"])
                )
                expected_interleaved = 2 if interleave == 4 else 0
                self.assertEqual(
                    int(summary["interleaved_tensor_count"]),
                    expected_interleaved,
                    "q_proj + lm_head must interleave; embedding and norm must not",
                )

                result = sp.verify_path(
                    out_path, strict=True, hash_payload=True, json_output=False
                )
                self.assertTrue(result["ok"])

                # Manifest layout flag mirrors the SIDX state.
                manifest = sp.inspect_path(out_path, json_output=False)["manifest"]
                self.assertEqual(
                    manifest["layout"]["row_interleave"], 4 if interleave == 4 else 0
                )
                self.assertEqual(manifest["version"], expected_version)

    def test_quantizer_rejects_nan_and_inf(self) -> None:
        # NaN/Inf in the source must abort the pack with a clear error,
        # not crash later or silently zero whole blocks. Crucially a NaN
        # that is NOT the first block element must also be caught (max()
        # with NaN is position-dependent in Python).
        for bad in (float("nan"), float("inf"), float("-inf")):
            for position in (0, 17, 31):
                values = [1.0] * 32
                values[position] = bad
                with self.assertRaises(ValueError):
                    sp.quantize_q4_0_block(tuple(values))
                with self.assertRaises(ValueError):
                    sp.quantize_q8_0_block(tuple(values))


if __name__ == "__main__":
    unittest.main()
