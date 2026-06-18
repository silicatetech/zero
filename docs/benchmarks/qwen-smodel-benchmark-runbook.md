# Qwen `.smodel` Benchmark Runbook

**Status:** required release runbook for Silicate Zero Server + SilicatePack  
**Model anchor:** Qwen3-1.7B  
**Purpose:** prove that the native `.smodel` path keeps the known Zero Server
CPU-only baseline.

This runbook is the first public release gate for the combined SilicatePack
and Silicate Zero Server launch. It proves that a model packed outside the
kernel can boot and benchmark through the native `.smodel` path.

## Baseline

Known validated class:

- hardware: AMD EPYC 9354P;
- runtime: Zero Server, bare metal, CPU-only;
- model: Qwen3-1.7B Q4 class;
- established result: `169.8 tok/s` on the native `.smodel` path.

A Qwen `.smodel` release candidate must not be treated as accepted until it
has matched this class or the regression is explained and documented.

## 1. Build The `.smodel`

```bash
cd zero-kernel

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

## 2. Verify The Artifact

```bash
python3 tools/silicatepack.py inspect target/models/qwen3-1.7b-zero-server.smodel
python3 tools/silicatepack.py verify --strict target/models/qwen3-1.7b-zero-server.smodel
```

For final release candidates:

```bash
python3 tools/silicatepack.py verify --strict --hash-payload \
  target/models/qwen3-1.7b-zero-server.smodel
```

## 3. Run Local Acceptance

```bash
python3 tools/zero_acceptance.py \
  --smodel target/models/qwen3-1.7b-zero-server.smodel \
  --strict-smodel \
  --target x86_64
```

Optional ARM token gate:

```bash
python3 tools/zero_arm_token_accuracy.py \
  --hf-dir /models/Qwen3-1.7B \
  --smodel-out target/zero-arm-token-accuracy/qwen3-1.7b.smodel
```

## 4. Build Zero Server

```bash
MODEL_PATH=target/models/qwen3-1.7b-zero-server.smodel \
  tools/build-x86_64-baremetal.sh --cherry
```

Record:

- git commit;
- build command;
- `.smodel` SHA-256;
- kernel image SHA-256;
- target hardware.

## 5. Boot And Query Runtime State

After Zero Server boots:

```bash
printf 'model\nllm status\nllm profile\nsmp status\nmem\nexit\n' | nc -w 8 <zero-server-ip> 2222
```

Minimum expected output:

- `model` reports native `.smodel`;
- `llm status` reports completed control state after generation;
- `llm profile` reports generated tokens and decode tok/s;
- `smp status` reports expected active/registered cores;
- `mem` shows model memory as write-back cached.

## 6. Capture Benchmark

Capture at least 3 full runs:

| Run | tok/s | generated tokens | cycles | commit | model hash | notes |
|-----|-------|------------------|--------|--------|------------|-------|
| 1   | 169.8 | 32               | 215287312 | 9baf658 | 68e0d3b3ed614bc77117be9c39bd96c0dc5f3c9ab65412da0ba50dfd9e5838f8 | EPYC 9354P, CPU-only, native `.smodel`, screenshot captured 2026-06-02 |
| 2   |       |                  |        |        |            |       |
| 3   |       |                  |        |        |            |       |

Use the median for public claims. Keep min/max for engineering notes.

### 2026-06-02 Release Snapshot

This snapshot records the first validated Zero Server Qwen `.smodel`
run above the public CPU-only target.

- Product: Silicate Zero Server;
- Model artifact: `qwen3-1.7b-x86_64-zen4-avx512.smodel`;
- Model SHA-256:
  `68e0d3b3ed614bc77117be9c39bd96c0dc5f3c9ab65412da0ba50dfd9e5838f8`;
- Kernel/image commit: `9baf658` (`fix smodel matmul barrier epochs`);
- UEFI image SHA-256:
  `1ec887a82c9d12bf0bb415ede7a34cd799831f24ef6c59c606e6d1be857505eb`;
- Hardware: AMD EPYC 9354P 32-Core Processor;
- Cores active: 64 logical CPUs;
- Boot time: `3635.326 ms`;
- Context switch: `1 ns`;
- Arena allocation: `0.5 ns`;
- IPC throughput: `1315.3 GB/s`;
- LLM inference: `169.8 tok/s`;
- Generated tokens: `32`;
- Inference cycles: `215287312`.

The benchmark is CPU-only. No GPU path was active.

## 7. Pass / Fail Criteria

Pass:

- `silicatepack verify --strict` passes;
- local `zero_acceptance.py` passes;
- Zero Server boots the `.smodel`;
- Stage 11 records completed generation timing;
- benchmark reports generated tokens;
- median throughput is in the known Qwen CPU-only baseline class;
- token/logit anchor is preserved or explicitly updated with a documented
  reason.

Fail:

- missing strict anchors;
- skipped model test counted as green;
- model loads through GGUF compatibility instead of native `.smodel`;
- `0.0 tok/s` or `unavailable`;
- silent tensor substitution;
- scalar fallback on AVX-512 hardware;
- performance regression without a written explanation.

## 8. Release Evidence Bundle

Store these files with the release:

```bash
python3 tools/silicatepack.py inspect --json \
  target/models/qwen3-1.7b-zero-server.smodel \
  > target/models/qwen3-1.7b-zero-server.inspect.json

sha256sum target/models/qwen3-1.7b-zero-server.smodel \
  > target/models/qwen3-1.7b-zero-server.sha256
```

Also keep:

- KVM/BMC screenshot of benchmark result;
- serial log;
- TCP control-plane output;
- build log;
- exact hardware SKU and memory configuration.
