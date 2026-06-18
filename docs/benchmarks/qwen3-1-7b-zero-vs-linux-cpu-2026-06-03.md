# Zero Server vs Linux CPU Baseline: Qwen3 1.7B Q4_K_M

Date: 2026-06-03
Status: release evidence draft
Hardware: Cherry Servers bare metal, AMD EPYC 9354P / EPYC 9354, 32 physical cores / 64 logical CPUs
Mode: CPU-only, no GPU
Model family: Qwen3 1.7B
Benchmark focus: decode/token-generation throughput (`tg`), not prefill (`pp`)

## Summary

Zero Server reached `169.8 tok/s` CPU-only on Qwen3 1.7B using the native
`.smodel` path.

On the same hardware under Linux rescue mode, `llama.cpp` reached
`152.8 tok/s` on Qwen3 1.7B Q4_K_M GGUF with 32 threads. The 64-thread SMT
run was slower at `144.8 tok/s`.

Current headline, with the caveats below:

```text
Zero Server CPU-only:      169.8 tok/s
Linux llama.cpp CPU-only:  152.8 tok/s
Delta:                     +11.1% Zero Server
```

This is the strongest claim we can defend today: same machine, same model
family, CPU-only, measured by us. It is not a universal claim against every
GPU, every runtime, or every prompt length.

## Zero Server Result

Source: Zero Server bare-metal benchmark screen from the Qwen `.smodel` run.

```text
[BENCH] LLM Inference: 169.8 tok/s (32 generated token(s), 215287312 cycles)

CPU: AMD EPYC 9354P 32-Core Processor
Boot Time:       3635.326ms
Context Switch: 1ns
Arena Alloc:    0.5ns
IPC Throughput: 1315.3 GB/s
LLM Inference:  169.8 tok/s
```

Runtime notes:

- Product path: Silicate Zero Server.
- Model path: native `.smodel`.
- Execution: bare metal, CPU-only.
- Active hardware: 64 logical CPUs registered.
- Strict anchors remain part of the release gate; token/logit anchor drift
  must not be hidden behind benchmark output.

## Linux Reference Setup

Linux rescue environment:

```text
Linux 5.15.0-177-generic x86_64
CPU: AMD EPYC 9354 32-Core Processor
CPU(s): 64
Thread(s) per core: 2
Core(s) per socket: 32
NUMA nodes: 1
RAM: 188 GiB
AVX-512: present
llama.cpp devices: (none)
```

llama.cpp:

```text
Repository: https://github.com/ggml-org/llama.cpp
Commit: 63e66fdd23eda3a2659a7af9ff6ef15d71efbff1
Backend: CPU
Build: Release, GGML_NATIVE=ON
GPU layers: 0
```

Model:

```text
Repository: https://huggingface.co/enacimie/Qwen3-1.7B-Q4_K_M-GGUF
Repository commit: 912f298a8f85df08f3bed3328b3d0935106bf593
File: qwen3-1.7b-q4_k_m.gguf
SHA256: 54e0d3dbd2388f3c414bf31fb3e22e4954c8edcf4ab83e315d44995bea764eb9
Size: 1.2 GiB
llama.cpp detected type: qwen3 1.7B Q4_K - Medium
Parameters: 2,031,739,904
```

Evidence on the Linux rescue system:

```text
/root/zero-bench/models/qwen3-1.7b-q4_k_m.gguf
/root/zero-bench/results/llama-bench-qwen3-1.7b-q4km.json
/root/zero-bench/results/llama-bench-qwen3-1.7b-q4km-pinned.json
/root/zero-bench/results/llama-bench-qwen3-thread-sweep.json
```

## Linux Commands

Build:

```bash
apt-get update
apt-get install -y build-essential cmake pkg-config ca-certificates python3

mkdir -p /root/zero-bench/models /root/zero-bench/results
cd /root/zero-bench
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git rev-parse HEAD
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release -j "$(nproc)" --target llama-bench
./build/bin/llama-bench --list-devices
```

Model download:

```bash
cd /root/zero-bench/models
curl -L --fail --retry 3 \
  -o qwen3-1.7b-q4_k_m.gguf \
  https://huggingface.co/enacimie/Qwen3-1.7B-Q4_K_M-GGUF/resolve/main/qwen3-1.7b-q4_k_m.gguf
sha256sum qwen3-1.7b-q4_k_m.gguf
```

Primary benchmark:

```bash
cd /root/zero-bench/llama.cpp
./build/bin/llama-bench \
  -m /root/zero-bench/models/qwen3-1.7b-q4_k_m.gguf \
  -ngl 0 \
  -t 32,64 \
  -p 512 \
  -n 128 \
  -r 3 \
  -o json \
  2>/root/zero-bench/results/llama-bench-qwen3-1.7b-q4km.err \
  | tee /root/zero-bench/results/llama-bench-qwen3-1.7b-q4km.json
```

Pinned physical-core check:

```bash
taskset -c 0-31 ./build/bin/llama-bench \
  -m /root/zero-bench/models/qwen3-1.7b-q4_k_m.gguf \
  -ngl 0 \
  -t 32 \
  -p 512 \
  -n 128 \
  -r 5 \
  --prio 3 \
  -o json \
  2>/root/zero-bench/results/llama-bench-qwen3-1.7b-q4km-pinned.err \
  | tee /root/zero-bench/results/llama-bench-qwen3-1.7b-q4km-pinned.json
```

Decode-only thread sweep:

```bash
./build/bin/llama-bench \
  -m /root/zero-bench/models/qwen3-1.7b-q4_k_m.gguf \
  -ngl 0 \
  -t 8,16,24,32,40,48,56,64 \
  -p 0 \
  -n 128 \
  -r 3 \
  --prio 3 \
  -o json \
  2>/root/zero-bench/results/llama-bench-qwen3-thread-sweep.err \
  | tee /root/zero-bench/results/llama-bench-qwen3-thread-sweep.json
```

## Linux Results

Primary run:

| Runtime | Threads | Prompt / Prefill | Decode / Generation | Notes |
|---|---:|---:|---:|---|
| llama.cpp CPU | 32 | `1017.9 tok/s` | `152.8 tok/s` | best measured full benchmark |
| llama.cpp CPU | 64 | `950.3 tok/s` | `144.8 tok/s` | SMT slower for decode |

Pinned physical-core run:

| Runtime | CPU pinning | Threads | Prompt / Prefill | Decode / Generation | Notes |
|---|---|---:|---:|---:|---|
| llama.cpp CPU | `taskset -c 0-31` | 32 | `974.6 tok/s` | `146.1 tok/s` | lower than unpinned in this run |

Decode-only thread sweep:

| Threads | Decode tok/s |
|---:|---:|
| 8 | `93.8` |
| 16 | `91.3` |
| 24 | `107.6` |
| 32 | `123.9` |
| 40 | `127.1` |
| 48 | `134.2` |
| 56 | `138.3` |
| 64 | `140.1` |

The thread sweep used `-p 0 -n 128`; the primary run used `-p 512 -n 128`.
The primary run is the cleaner public reference because it includes the
standard `llama-bench` prompt/decode pair.

## Comparison

| System | Artifact | Runtime | CPU threads | Decode tok/s | Relative |
|---|---|---|---:|---:|---:|
| Zero Server | `.smodel` | bare metal | 64 logical CPUs registered | `169.8` | `1.111x` vs best Linux run |
| Linux | GGUF Q4_K_M | llama.cpp CPU | 32 | `152.8` | baseline |
| Linux | GGUF Q4_K_M | llama.cpp CPU | 64 | `144.8` | `0.947x` vs Linux 32-thread run |
| Linux | GGUF Q4_K_M | llama.cpp CPU | pinned 0-31 | 32 | `146.1` | `0.956x` vs Linux 32-thread run |

Interpretation:

- Zero Server is `+11.1%` over the best measured Linux/llama.cpp decode run.
- Zero Server is `+17.3%` over the Linux/llama.cpp 64-thread SMT run.
- SMT did not help Linux decode on this EPYC 9354P class machine.
- The result is CPU-only. No CUDA, ROCm, Metal, Vulkan, or other GPU path was
  active in the Linux reference or the Zero Server run.

## Methodology Caveats

These caveats must stay attached to the result:

- Zero Server used native `.smodel`; Linux used GGUF Q4_K_M. They represent the
  same model family and quantization class, but the runtime artifact is not
  byte-identical.
- Zero Server generated 32 tokens in the captured run; llama.cpp decode rows
  generated 128 tokens. This is acceptable for an initial evidence baseline,
  but the next public benchmark should run matched prompt and generation sizes
  on both systems.
- `llama-bench` reports prompt processing and text generation separately. The
  public comparison uses `tg`/decode, not `pp`/prefill.
- Rescue Linux scheduling, CPU frequency policy, and thermal state can move
  results. Public tables should use median of at least 3 cold boots or 5
  controlled runs.
- Any `.smodel` anchor mismatch invalidates a performance claim until the
  anchor is regenerated or the drift is explained in a release note.

## External GPU Context

External GPU numbers are context only. They are not a direct Qwen3 1.7B
comparison.

NVIDIA reports internal llama.cpp measurements where RTX 4090 reaches roughly
`150 tokens/s` on Llama 3 8B int4 with 100 input tokens and 100 generated
tokens. That is useful market context, but it differs by model size, model
family, quantization details, hardware, prompt length, and runtime backend.

Do not present external GPU numbers as proof that Zero Server beats a specific
GPU setup until we run the same model and prompt configuration ourselves.

## Next Controlled Benchmark

Run this when we want a stricter public table:

```bash
cd /root/zero-bench/llama.cpp
./build/bin/llama-bench \
  -m /root/zero-bench/models/qwen3-1.7b-q4_k_m.gguf \
  -ngl 0 \
  -t 32,64 \
  -p 13 \
  -n 32 \
  -r 5 \
  --prio 3 \
  -o json \
  2>/root/zero-bench/results/llama-bench-qwen3-1.7b-q4km-matched-13p-32g.err \
  | tee /root/zero-bench/results/llama-bench-qwen3-1.7b-q4km-matched-13p-32g.json
```

Pair it with a fresh Zero Server `.smodel` run on the same prompt and generated
token budget. Publish median, min, max, commit hash, model hash, and raw logs.

## Sources

- llama.cpp `llama-bench` README: https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- Qwen3 1.7B Q4_K_M GGUF used for Linux test: https://huggingface.co/enacimie/Qwen3-1.7B-Q4_K_M-GGUF
- NVIDIA RTX llama.cpp GPU context: https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/
