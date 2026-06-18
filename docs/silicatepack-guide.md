# SilicatePack Guide — Packing Your Own LLM for Silicate Zero Server

**Audience:** end users who want to run their own Hugging Face LLM on Silicate
Zero Server (bare-metal, Ring-0, CPU-only inference — no OS, no external
inference runtime).

**Tool:** `tools/silicatepack.py` (pure Python, stdlib only).
**Format:** `.smodel` — the native Zero Server model container, the
ONLY production model format. There is no other.

> **TL;DR**
> ```bash
> # AVX-512 server (EPYC / Zen 4), recommended interleaved layout:
> python3 tools/silicatepack.py pack-hf \
>   --input-dir ~/models/hf/MyModel \
>   --quant auto --interleave 4 \
>   --profile cpu-avx512 --target-arch x86_64-zen4 \
>   --output ~/models/smodel/mymodel-avx512-x4.smodel --verify
> ```
> Then capture an anchor (§6), verify coherent output, and embed the `.smodel`
> in the boot image. **Always check the model actually produces sensible text,
> not just that the anchor passes** (§7, Troubleshooting).

---

## 1. The `.smodel` format

Zero Server runs in Ring-0 with no filesystem and no external inference
runtime, so it uses its own native container, `.smodel`:

- **SILM** header (128 bytes) + **SIDX** tensor directory (JSON manifest) +
  64-byte-aligned, 2-MiB-payload-aligned tensor payload.
- Host-side expert fusion, runtime-profile tagging, and strict
  token/logit **anchors** baked into the manifest.

`.smodel` is the only production format. The supported path is `.smodel`
produced from Hugging
Face SafeTensors with `pack-hf`.

---

## 2. Quantization types

SilicatePack emits these native dtypes (SIDX dtype ids in parentheses):

| dtype     | id  | bits/weight | block            | Use                                              |
|-----------|-----|-------------|------------------|--------------------------------------------------|
| `F32`     | 0   | 32          | —                | norms, RoPE freqs, anything kept full-precision  |
| `Q4_0`    | 2   | 4.5         | 32 vals / 18 B   | matrix weights (attn/FFN) — the default for 2-D  |
| `Q8_0`    | 8   | 8.5         | 32 vals / 34 B   | embeddings + LM head (accuracy-critical)         |
| `Q4_0X4`  | 100 | 4.5         | 4 rows × 18 B    | Q4_0, **4-row interleaved** (AVX-512 only)        |
| `Q8_0X4`  | 101 | 8.5         | 4 rows × 34 B    | Q8_0, **4-row interleaved** (AVX-512 only)        |

`--quant` choices: **`none` | `auto` | `q8_0` | `q4_0`**.
- `auto` (default & recommended): **Q8_0** for embeddings/LM-head, **Q4_0** for
  every other 2-D matrix weight; norms stay F32.
- `q4_0` / `q8_0`: force that dtype for all matrix weights.
- `none`: keep source precision (F32/F16) — huge, mainly for debugging.

**Q4_K / Q6_K are NOT packable, and you don't need them.** `pack-hf` emits
Q4_0/Q8_0 only. The native **Q4_0 encoder uses the standard symmetric 4-bit
scale (`d = max/-8`, full [-8,7] level range)**, the same encoding any correct
Q4_0 implementation produces. (Models packed *before* commit `d632197` used a
coarser `max_abs/7` scale that never emits the −8 level — repack them.)

### Interleave (`--interleave 0|4`) — the AVX-512 target layout
`--interleave 4` stores eligible rank-2 Q4_0/Q8_0 tensors in 4-row group-blocks
(`Q4_0X4`/`Q8_0X4`). Dequantized values are **bit-identical** to the plain
layout (so existing anchors stay valid); it only changes the *storage order*
to feed the AVX-512 streaming kernels (shared activation loads, 8 independent
FMA chains). Eligibility: rank-2, `out_rows % 4 == 0`.
- **Only AVX-512 builds have the x4 kernels.** A NEON or scalar build **rejects**
  a v2 (interleaved) `.smodel` at load instead of misreading it.
- `--interleave 0` (default) = plain row-major, runs everywhere.

---

## 3. Which quant/layout for which hardware

| Target build            | Feature        | Reads                          | Recommended pack                                  |
|-------------------------|----------------|--------------------------------|---------------------------------------------------|
| **x86_64 Zen 4 server** | `avx512-acceleration` | Q4_0, Q8_0, **Q4_0X4, Q8_0X4** | `--quant auto --interleave 4`, `--profile cpu-avx512 --target-arch x86_64-zen4` |
| **aarch64 (Apple/ARM)** | `neon-acceleration`   | Q4_0, Q8_0 (plain only)        | `--quant auto --interleave 0`, `--profile cpu-neon --target-arch aarch64-neon` |
| **Scalar (sacred)**     | none (no features)    | Q4_0, Q8_0 (plain only)        | `--quant auto --interleave 0` (any profile)       |

Rules of thumb:
- **Interleave only for AVX-512.** NEON/scalar builds **hard-reject** an
  interleaved (`Q4_0X4`/`Q8_0X4`) model at load — this is the one quant/layout
  gate enforced unconditionally.
- **Profile/target-arch should match the runtime build.** The kernel derives its
  own profile (`cpu-avx512` / `cpu-neon` / scalar) and uses it to *locate the
  matching anchor*: a mismatched `--profile`/`--target-arch` means the strict
  anchor is never found and not validated (and on the MoE/DeepSeek2 path it is a
  hard `anchor profile mismatch` failure). Plain weights still load, but you lose
  the anchor gate — so always pack with the runtime's profile.
- One source model → pack it **once per target** (e.g. an `-avx512-x4` and an
  `-aarch64-neon` artifact), as the existing Qwen3 artifacts do.

> **Note for developers:** the AVX-512 code paths cannot be executed on an ARM
> dev machine — QEMU TCG has no AVX-512 and `hvf` is ARM. Local β-anchor
> verification therefore runs the **scalar/NEON** build only. AVX-512 numerics
> (and the 64-core SMP dispatch) are validated **on the target server**. Budget
> for at least one on-hardware boot when bringing up a new model.

---

## 3b. MoE models (DeepSeek-V2, Qwen-MoE)

Mixture-of-Experts models require **expert fusion** at pack time. `pack-hf`
handles this automatically: it detects `num_experts` / `num_experts_per_tok` in
`config.json`, reads the per-expert weight shards, and fuses them into the SIDX
tensor directory with the correct `expert_weights_scale` metadata.

What to know:
- `--quant auto` applies the same Q4_0/Q8_0 split to expert weights as to
  dense weights.
- `--interleave 4` interleaves eligible expert weight matrices the same way as
  dense matrices. The kernel's MoE dispatch selects experts *before*
  dequantization, so the interleave layout is transparent to the routing logic.
- `verify --strict` checks that `expert_weights_scale` in the SIDX matches the
  manifest. A missing or mismatched scale fails strict verification.
- **DeepSeek-V2/V3 MLA attention** uses compressed KV projections (low-rank
  `kv_a_proj` + `kv_b_proj`). SilicatePack packs these as regular tensors; the
  kernel's MLA codepath handles the decomposition at inference time.

MoE support is still being hardened. After packing, always boot and read the
generated text — coherence is the real acceptance test for MoE models, not just
the anchor.

---

## 4. Packing a model step by step

### 4.0 Prerequisites
- A Hugging Face model directory with `*.safetensors`, `config.json`,
  `tokenizer.json` (standard HF layout).
- Python 3 (stdlib only — no heavyweight ML deps needed for packing).
- Disk: ~0.6× the F16 size per Q4_0 artifact.

### 4.1 Pack (AVX-512 server, recommended)
```bash
python3 tools/silicatepack.py pack-hf \
  --input-dir   ~/models/hf/Qwen3-1.7B \
  --quant       auto \
  --interleave  4 \
  --profile     cpu-avx512 \
  --target-arch x86_64-zen4 \
  --source-repo Qwen/Qwen3-1.7B \
  --output      ~/models/smodel/qwen3-1.7b-avx512-x4.smodel \
  --verify
```
Useful extras: `--safetensors a.safetensors b.safetensors` (explicit shards),
`--config`/`--tokenizer` (non-default paths), `--no-normalize-names` (keep HF
tensor names instead of Zero Server runtime names), `--force` (overwrite),
`--tensor-alignment 64` / `--payload-alignment 2097152` (defaults shown).

### 4.2 Pack (aarch64 / NEON)
```bash
python3 tools/silicatepack.py pack-hf \
  --input-dir ~/models/hf/Qwen3-1.7B --quant auto --interleave 0 \
  --profile cpu-neon --target-arch aarch64-neon \
  --output ~/models/smodel/qwen3-1.7b-aarch64-neon.smodel --verify
```

### 4.3 Inspect what you packed
```bash
python3 tools/silicatepack.py inspect ~/models/smodel/...smodel          # human
python3 tools/silicatepack.py inspect ~/models/smodel/...smodel --json   # machine
```
Check: every `blk.N.*` weight has the dtype you expect (`Q4_0`/`Q4_0X4`/`Q8_0`),
the `model_config` matches the source (layers, heads, `rope_theta`, `rms_norm_eps`),
and `profile`/`target_arch` are right.

### 4.4 Verify structure (+ optional strict anchors / full hash)
```bash
python3 tools/silicatepack.py verify ~/models/smodel/...smodel --strict
python3 tools/silicatepack.py verify ~/models/smodel/...smodel --hash-payload  # slow
```

### 4.5 Embed in the boot image
```bash
MODEL_PATH=~/models/smodel/qwen3-1.7b-avx512-x4.smodel \
  tools/build-x86_64-baremetal.sh --cherry
# → target/zero-images/zero-uefi.img  (and zero-bios.img)
```
For local aarch64 QEMU testing the model is passed as a ramdisk:
`make run-aarch64` (uses `MODEL_PATH` — point it at your `.smodel`).

---

## 5. Realistic performance per profile

| Hardware / profile                         | Model           | tok/s (decode)   | Notes                                  |
|--------------------------------------------|-----------------|------------------|----------------------------------------|
| EPYC 9354P (32C/64T), AVX-512, `cpu-avx512`| Qwen3-1.7B 4-bit| **~170 tok/s**   | memory-bandwidth bound |
| same, with `--interleave 4` x4 kernels     | Qwen3-1.7B      | 1.2–1.5× matmul wall-clock vs plain | weight-stream bound; biggest single lever |
| aarch64 NEON (QEMU hvf, dev box)           | Qwen3-1.7B      | single-thread, dev/verify only | not a perf target — correctness gate    |
| Scalar (sacred, no features)               | Qwen3-1.7B      | slow, reference only | bit-exact β-anchor source              |

Reality check: decode is **memory-bandwidth bound** — tok/s scales with weight
bytes streamed per token, so a smaller quant (Q4_0) and more active cores help
more than raw FLOPs. The x4 interleave exists to make the weight stream
sequential for the L2 streamer. Numbers above are for a 1.7B model; larger
models scale roughly inversely with parameter bytes.

---

## 6. Anchors (`--capture` and strict)

An **anchor** is a reproducibility gate baked into the manifest: a fixed prompt
plus the expected first argmax **token id** and top-1 **logit bits**. At boot the
kernel runs the anchor prompt and compares — a hard gate against silent
weight/dequant corruption.

Anchor fields (on `pack-hf` and `set-anchors`): `--anchor-name`,
`--anchor-prompt` (or `--anchor-prompt-tokens 9707,4337,...`),
`--anchor-next-token`, `--anchor-logit-bits`, `--anchor-generated-tokens`.

### Two-step capture → promote (use after any quantizer change)
A quantizer or layout change moves the logits, so the old baseline values no
longer match. Don't guess them — **capture** them:

```bash
# 1) Pack with a CAPTURE-mode anchor (no expected values; kernel logs measured)
python3 tools/silicatepack.py set-anchors mymodel.smodel \
  --capture --profile cpu-avx512 --target-arch x86_64-zen4 \
  --anchor-prompt-tokens 9707,4337,358,2776,279,1156,444,10994,4303,389,60792,5251,21323

# 2) Boot once. The kernel logs e.g.:
#    [MP3.0] .smodel anchor captured: token=25 logit_bits=0x414a6497
# 3) Promote to a strict anchor with those measured values:
python3 tools/silicatepack.py set-anchors mymodel.smodel \
  --anchor-next-token 25 --anchor-logit-bits 0x414a6497 \
  --profile cpu-avx512 --target-arch x86_64-zen4
```

> ⚠️ **A passing anchor only proves reproducibility, NOT output quality.** A
> capture-mode anchor records whatever the model produces — including garbage —
> and then "passes" because the model reproduces its own output. Always *read
> the generated text* (§7) in addition to the anchor.

> ⚠️ **Anchor profile/target-arch must match the runtime.** A strict anchor
> captured under `cpu-avx512` will not be found by a NEON build. The verifier
> does not cross-check this — re-capture per profile. Note also that the AVX-512
> feature build may legitimately drift `logit_bits` by ≤1 ULP vs the scalar
> sacred value (documented "two-anchor" regime): the **token id** is
> the hard gate; `logit_bits` is diagnostic for feature builds.

---

## 7. Best practices & troubleshooting

### Best practices
- **Pack once per target profile**, never ship an AVX-512-only (interleaved)
  model to a NEON/scalar build.
- Keep embeddings + LM-head at **Q8_0** (`--quant auto`); 4-bit there visibly
  hurts argmax quality.
- Run **`verify --strict`** before every deploy; run **`--hash-payload`** at
  least once per artifact to catch a torn write.
- **Capture anchors per profile**, then promote to strict.
- **Read the actual generated tokens** on first boot — coherence is the real
  acceptance test, not the anchor.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Boot: `SIDX rejected: row-interleaved dtype … requires the AVX-512 build` | Interleaved (`Q4_0X4`) model on a NEON/scalar build | Re-pack with `--interleave 0`, or use the AVX-512 build |
| Boot: anchor profile mismatch / anchor never found | Anchor captured under a different `--profile`/`--target-arch` | Re-`set-anchors` for the runtime's profile |
| Output is repetitive garbage / random high-vocab tokens | Either (a) packed with the old pre-`d632197` Q4_0 encoder → **repack**; or (b) the kernel corrupted the model in RAM — check the boot log for `[INTEGRITY] CORRUPT output.weight` (an arena/model memory overlap; see the KV-arena fix). A passing capture anchor does NOT prove the output is good — read the generated text. |
| `Attention(NumericalInstability)` / `0.0 tok/s` on an AVX-512 **server only** (NEON/scalar fine) | A bug in the AVX-512 **SMP parallel-dispatch** layer, not the quant — it is the one path local verification can't exercise | Boot with the per-step tracer (logs the exact step that goes NaN); as a fallback, disable the fused multi-projection dispatch (`set_fused_dispatch_enabled(false)`) to use the self-test-validated separate-dispatch path |
| Boot panic on malformed SIDX / huge tensor_count | Hand-edited or truncated `.smodel` | Re-pack from source; never edit the manifest by hand |
| Packing is very slow on a big model | Multiple full SHA-256 passes over the checkpoint | Expected for large models; pack on a fast disk, once |

### Honest limits
- **You cannot validate AVX-512 numerics on an ARM dev box.** Local gates
  (β-anchor on QEMU hvf) cover scalar/NEON only. Plan an on-server boot, and
  read the generated text on that boot — coherence is the real acceptance test.
- **Repack after any SilicatePack update that touches the quantizer.** The
  encoder is versioned by behaviour, not by a flag: a `.smodel` carries whatever
  the packer produced at pack time. The current Q4_0 path is the standard 4-bit encoding; an old
  artifact does not. When in doubt, repack and re-capture the anchor.

---

*Maintained alongside `tools/silicatepack.py`.
See `docs/SILICATEPACK.md` for format internals.*
