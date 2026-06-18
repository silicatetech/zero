# SilicatePack User Guide

**Status:** active operator guide for Silicate Zero Server releases  
**Product:** Silicate Zero Server, shortened to Zero Server in commands and logs  
**Tool:** `tools/silicatepack.py`  
**Artifact:** `.smodel`  
**Related:** `docs/plans/performance-v1-zero-server.md`

SilicatePack converts model source files into native Zero Server model
artifacts. The production target is a native `.smodel` file, not a GGUF
wrapper. GGUF remains a legacy compatibility/import path only.

SilicatePack and Silicate Zero Server are released together:

- SilicatePack is the host-side packaging and verification tool.
- `.smodel` is the native model artifact.
- Zero Server is the bare-metal runtime that consumes `.smodel`.
- **Ready for Zero Server** means the artifact passed strict verification,
  boot acceptance, and benchmark evidence capture.

The release contract is simple:

```text
Hugging Face SafeTensors + config.json + tokenizer sidecars
    -> silicatepack pack-hf
    -> native .smodel
    -> silicatepack verify --strict
    -> Zero Server boot / benchmark
```

## Why SilicatePack Exists

Zero Server runs LLM inference directly in the kernel. Generic model formats
are useful for ecosystem compatibility, but they force the kernel to parse,
guess, and normalize model details during boot. SilicatePack moves that work to
the build/deploy host.

SilicatePack provides:

- a native Zero Server tensor directory (`SIDX`);
- cache-line aligned tensor payloads;
- 2 MiB payload alignment for hugepage-friendly model mapping;
- normalized tensor names for the Zero Server runtime ABI;
- model config and runtime profile metadata;
- source and payload checksums;
- strict validation anchors for token/logit regression gates;
- a distribution artifact suitable for a **Ready for Zero Server** release.

## Requirements

Run SilicatePack on a normal host system, not inside Zero Server:

- Python 3.11+ recommended;
- local checkout of this repository;
- a Hugging Face model directory containing:
  - one or more `*.safetensors` files;
  - `config.json`;
  - `tokenizer.json` or tokenizer sidecars.

Optional but recommended:

- enough local disk space for source weights plus the emitted `.smodel`;
- model source revision and license metadata for reproducible release notes.

## Quick Start

From the repository root:

```bash
cd zero-kernel
```

Pack a Hugging Face model directory:

```bash
python3 tools/silicatepack.py pack-hf \
  --input-dir /models/Qwen3-1.7B \
  --output target/models/qwen3-1.7b-zero-server.smodel \
  --target-product "Zero Server" \
  --profile cpu-avx512 \
  --target-arch x86_64-zen4 \
  --source-repo Qwen/Qwen3-1.7B \
  --source-revision <commit-or-tag> \
  --license Apache-2.0 \
  --quant auto \
  --verify \
  --force
```

Inspect the artifact:

```bash
python3 tools/silicatepack.py inspect target/models/qwen3-1.7b-zero-server.smodel
```

Verify the artifact:

```bash
python3 tools/silicatepack.py verify target/models/qwen3-1.7b-zero-server.smodel
```

Strict verification for release artifacts:

```bash
python3 tools/silicatepack.py verify --strict target/models/qwen3-1.7b-zero-server.smodel
```

For very large models, avoid `--hash-payload` during fast iteration. Use it for
release candidates:

```bash
python3 tools/silicatepack.py verify --strict --hash-payload target/models/qwen3-1.7b-zero-server.smodel
```

## Commands

### `pack-hf`

Creates a native `.smodel` from Hugging Face SafeTensors input.

```bash
python3 tools/silicatepack.py pack-hf \
  --input-dir <hf-model-dir> \
  --output <model.smodel> \
  [--config <config.json>] \
  [--tokenizer <tokenizer.json>] \
  [--safetensors <file1.safetensors> <file2.safetensors> ...] \
  [--source-repo <repo>] \
  [--source-revision <revision>] \
  [--license <license>] \
  [--quant none|auto|q8_0|q4_0] \
  [--target-product "Zero Server"] \
  [--profile cpu-avx512] \
  [--target-arch x86_64-zen4] \
  [--verify] \
  [--force]
```

Default behavior:

- `--quant auto` emits embeddings and LM head as `Q8_0`;
- `--quant auto` emits matrix weights as `Q4_0`;
- scalar and normalization tensors are emitted as `F32`;
- Hugging Face names are normalized into Zero Server runtime tensor names;
- tensor payloads are 64-byte aligned;
- the native payload is 2 MiB aligned.

Use `--no-normalize-names` only for debugging. Production artifacts should use
the normalized Zero Server ABI.

### `verify`

Validates `.smodel` structure.

```bash
python3 tools/silicatepack.py verify <model.smodel>
```

Release-grade validation:

```bash
python3 tools/silicatepack.py verify --strict <model.smodel>
```

`--strict` requires:

- native `.smodel` payload;
- valid `SILM` container;
- `container=native`;
- `target_product=Zero Server`;
- `ready_label=Ready for Zero Server`;
- 2 MiB-aligned native payload;
- `layout.gguf_payload=false`;
- `compatibility.gguf_runtime_payload=false`;
- `runtime_profiles.cpu.status=performance-v1`;
- valid `SIDX` tensor directory;
- native `config_json` and `tokenizer_json` sidecar sections;
- valid manifest and payload bounds;
- valid dtype ids;
- unique tensor names;
- valid tensor byte ranges;
- `SIDX` model fields that match the manifest, including MoE
  `expert_weights_scale`;
- strict validation anchors.

### `inspect`

Prints the `.smodel` header and manifest.

```bash
python3 tools/silicatepack.py inspect <model.smodel>
python3 tools/silicatepack.py inspect --json <model.smodel>
```

Use this before deploying a model to verify:

- source repository and revision;
- target product and profile;
- architecture;
- tensor count;
- payload kind;
- validation anchor state.

### `set-anchors`

Writes strict validation anchors into an existing native `.smodel`.

```bash
python3 tools/silicatepack.py set-anchors <model.smodel> \
  --anchor-name zero-server-qwen3-smoke-v1 \
  --anchor-prompt "Hello" \
  --anchor-prompt-tokens 9707 \
  --anchor-next-token 25 \
  --anchor-logit-bits 0x414a6497 \
  --anchor-generated-tokens 25
```

The anchor values must come from a known-good reference run. Do not invent or
approximate anchors. If the expected token or logit bits are not known, capture
them first and then lock them.

### `import-gguf-compat`

Legacy-only path for old benchmark artifacts.

```bash
python3 tools/silicatepack.py import-gguf-compat \
  --input kernel/programs/Qwen_Qwen3-1.7B-Q4_K_M.gguf \
  --output target/models/qwen3-gguf-compat.smodel \
  --note "legacy benchmark compatibility artifact" \
  --verify \
  --force
```

Do not use this for new production models. It exists only to migrate or compare
old GGUF-based benchmark artifacts.

## Qwen3 Release Benchmark Flow

Qwen3-1.7B is the Zero Server CPU-only regression anchor. A release candidate
must prove that the native `.smodel` path keeps the known Qwen performance and
correctness properties.

Pack:

```bash
python3 tools/silicatepack.py pack-hf \
  --input-dir /models/Qwen3-1.7B \
  --output target/models/qwen3-1.7b-zero-server.smodel \
  --target-product "Zero Server" \
  --profile cpu-avx512 \
  --target-arch x86_64-zen4 \
  --source-repo Qwen/Qwen3-1.7B \
  --source-revision <commit-or-tag> \
  --license Apache-2.0 \
  --quant auto \
  --verify \
  --force
```

Set or verify anchors:

```bash
python3 tools/silicatepack.py verify --strict target/models/qwen3-1.7b-zero-server.smodel
```

Run local acceptance:

```bash
python3 tools/zero_acceptance.py \
  --smodel target/models/qwen3-1.7b-zero-server.smodel \
  --strict-smodel \
  --target x86_64
```

Build Zero Server with the `.smodel`:

```bash
MODEL_PATH=target/models/qwen3-1.7b-zero-server.smodel \
  tools/build-x86_64-baremetal.sh --cherry
```

After boot, query the control plane:

```bash
printf 'model\nllm status\nllm profile\nsmp status\nmem\nexit\n' | nc -w 8 <zero-server-ip> 2222
```

Expected release result:

- model format reports native `.smodel`;
- Qwen validation anchor passes;
- SMP reports all expected cores active;
- LLM benchmark records generated tokens;
- throughput remains in the known Qwen CPU-only class.

Current Qwen baseline:

- hardware: AMD EPYC 9354P;
- model: Qwen3-1.7B Q4_K_M class;
- runtime: Zero Server, CPU-only, AVX-512;
- known result: about `149.3 tok/s`, with later runs crossing the `150 tok/s`
  target.

Do not publish a higher `.smodel` number until it is captured with the full
artifact metadata: commit, model source revision, `.smodel` hash, hardware,
build command, and screenshot/log output.

## Release Checklist

For a **Ready for Zero Server** model release, capture:

- model source repo;
- model source revision;
- model license;
- SilicatePack commit;
- `.smodel` format version;
- `.smodel` SHA-256;
- target product: `Zero Server`;
- target profile: for example `cpu-avx512`;
- target architecture: for example `x86_64-zen4`;
- source tensor checksum manifest;
- native payload checksum manifest;
- strict anchor name and expected values;
- benchmark hardware;
- benchmark command;
- benchmark result, median of at least 3 runs for public claims.

Minimum commands:

```bash
python3 -m unittest tools/test_silicatepack.py
python3 tools/silicatepack.py inspect --json <model.smodel> > <model>.inspect.json
python3 tools/silicatepack.py verify --strict --hash-payload <model.smodel>
python3 tools/zero_acceptance.py --smodel <model.smodel> --strict-smodel --target x86_64
```

Do not order or deploy fresh bare-metal hardware for a Release-v1 CPU run until
the strict verifier and `zero_acceptance.py` pass locally against the exact
`.smodel` artifact that will be written to NVMe or bundled into the image.

## AI-Agent Operating Rules

### End-to-End Workflow

An AI agent deploying a model to Zero Server follows this sequence. Every step
must succeed before advancing to the next.

```
Step 1: Acquire source model
  → Hugging Face directory with *.safetensors, config.json, tokenizer.json

Step 2: Pack per target profile
  → python3 tools/silicatepack.py pack-hf \
      --input-dir <hf-dir> --quant auto --interleave <0|4> \
      --profile <cpu-avx512|cpu-neon> --target-arch <arch> \
      --source-repo <repo> --license <license> \
      --output <model.smodel> --verify

Step 3: Inspect the artifact
  → python3 tools/silicatepack.py inspect <model.smodel>
  → Verify: tensor dtypes, layer count, head count, profile, target-arch

Step 4: Strict verification
  → python3 tools/silicatepack.py verify --strict <model.smodel>
  → Must pass. If it fails, fix the packing parameters and return to Step 2.

Step 5: Embed in boot image and boot
  → MODEL_PATH=<model.smodel> tools/build-x86_64-baremetal.sh --cherry
  → Boot on QEMU or target hardware

Step 6: Capture anchor (first boot of a new model)
  → set-anchors --capture, boot, read kernel log for measured values
  → Promote to strict: set-anchors --anchor-next-token <N> --anchor-logit-bits <hex>
  → Re-run verify --strict to confirm anchor is embedded

Step 7: Verify output coherence
  → Read the generated text on serial console or framebuffer
  → A passing anchor proves reproducibility, NOT quality
  → If the output is garbage, the model or the packing is wrong

Step 8: Release checklist (see Release Checklist section)
  → Capture all metadata: hash, commit, hardware, benchmark median
```

An agent must not skip steps or reorder them. In particular, Step 7 (output
coherence) is not optional — a capture-mode anchor records whatever the model
produces, including nonsense, and then "passes" because the model reproduces
its own output.

### Rules

1. Prefer `pack-hf` from SafeTensors input for production artifacts.
2. Use `import-gguf-compat` only for legacy benchmarks or migration.
3. Never call a `.smodel` release-ready until `verify --strict` passes.
4. Never fabricate missing tensors or substitute zero tensors.
5. Never treat a skipped model test as green.
6. Preserve strict token anchors for Qwen.
7. Preserve source attribution and license metadata.
8. Do not publish benchmark claims without model hash, commit, hardware, and
   command line.
9. Do not regress the known Qwen CPU-only baseline while integrating larger
   models.
10. Do not add model-specific kernel guesses when a model fails to load.
    Fix the SilicatePack conversion contract and regenerate the `.smodel`.
11. Pack once per target profile. An AVX-512 interleaved artifact must not be
    deployed to a NEON/scalar build. Profile mismatch causes silent anchor
    bypass or hard rejection at boot.
12. Always read the generated text after first boot of any new model or
    repacked artifact. An anchor only proves bit-exact reproducibility of the
    forward pass — it does not prove the output is meaningful.
13. After any SilicatePack update that touches the quantizer, repack all
    affected artifacts and re-capture anchors. The encoder is versioned by
    behavior, not by a flag.
14. For MoE models (DeepSeek-V2, Qwen-MoE): expert fusion is handled
    automatically by `pack-hf`. Verify that the SIDX manifest contains
    `expert_weights_scale` and that `verify --strict` checks it. If the
    kernel does not yet support the model's MoE variant, the artifact is
    packable but not runnable (see Packable vs Runnable).

## Packable vs Runnable

SilicatePack is responsible for normalizing source model repositories into
native `.smodel` artifacts. Zero Server is responsible for executing only
architectures whose graph kernels are implemented and tested.

This distinction is deliberate:

- a model can be syntactically packable before it is runnable;
- `verify --strict` must fail artifacts that miss required runtime metadata;
- unsupported architectures must fail loudly before benchmark or release;
- the kernel must not infer missing graph fields from tensor names at boot.

Release-ready artifacts therefore need both sides:

- a complete native `.smodel` with tensor directory, config, tokenizer,
  checksums, layout metadata, and byte/logit anchors;
- a Zero Server graph profile that consumes that model without GGUF parsing
  or model-specific fallbacks in the hot path.

## Troubleshooting

### `strict verification requires native .smodel payload`

The artifact is probably GGUF compatibility mode. Repack from SafeTensors with
`pack-hf`.

### `strict validation anchors` missing

The artifact is structurally valid but not release-ready. Capture or set the
anchor values with `set-anchors`, then run `verify --strict` again.

### Zero Server reports `0.0 tok/s`

The benchmark screen only shows a real token rate after Stage 11 records a
completed generation. Query the control plane:

```bash
printf 'model\nllm status\nllm profile\nexit\n' | nc -w 8 <zero-server-ip> 2222
```

If `llm profile` says no completed generation profile exists, the model did not
finish a generation loop. Check the serial log before the benchmark screen.

### Large model boots but crashes before token output

Do not patch the kernel by adding model-specific guesses. Inspect the `.smodel`
manifest and tensor directory first:

```bash
python3 tools/silicatepack.py inspect <model.smodel>
python3 tools/silicatepack.py verify <model.smodel>
```

If tensors are missing or named unexpectedly, fix the SilicatePack conversion
rules and regenerate the `.smodel`.

### GGUF shard problems

Do not solve this in the kernel. GGUF shards are legacy import input. Merge or
convert them offline, then emit one native `.smodel` artifact.

## Supported Model Architectures

SilicatePack can pack any Hugging Face SafeTensors model into `.smodel`. Whether
the kernel can *run* the model depends on the inference engine's graph support.

| Architecture | Pack | Run | Notes |
|---|---|---|---|
| **Qwen3** (1.7B, 4B, 8B, etc.) | Yes | Yes | Primary anchor model, production-validated |
| **Llama** (3.x family) | Yes | Yes | GQA attention |
| **Gemma** (2B, 7B) | Yes | Yes | GQA attention |
| **DeepSeek-V2/V3** | Yes | Hardening | MLA attention + MoE, expert fusion at pack time |
| **Qwen-MoE** | Yes | Hardening | MoE with shared experts |
| **Other GQA Transformers** | Yes | Likely | Any model using standard GQA + RoPE + SiLU FFN |

"Hardening" means the packing path works and the kernel has initial support,
but the inference path has not been fully validated with anchors on production
hardware.

Models that use non-standard attention (e.g., sliding window, cross-attention,
encoder-decoder) or non-standard activations are packable but will fail at
runtime until the kernel adds support.

## Current Limits

SilicatePack v1 is focused on the Zero Server inference path:

- native SafeTensors input is the production path;
- Qwen3 is the primary correctness and performance anchor;
- native quant emission currently covers `F32`, `Q8_0`, and `Q4_0`;
- K-quant dtype ids are reserved in the ABI, but Q4_K/Q5_K/Q6_K native
  writers and kernel acceptance tests are not release-ready yet;
- native tokenizer sidecars are required by `verify --strict`; any GGUF
  tokenizer fallback is legacy-only and must not define a Zero Server
  release artifact;
- Kimi/DeepSeek2 support is still being hardened;
- GPU layout metadata can be carried, but GPU execution is a future runtime
  path;
- GGUF compatibility is intentionally not the production path.
