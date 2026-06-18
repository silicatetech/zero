# Silicate Zero — Architecture

A bare-metal Rust kernel for LLM inference. No OS. No runtime. No GPU required.

This document describes the architecture of Silicate Zero as shipped. For
performance data see [`PERFORMANCE.md`](PERFORMANCE.md). For the native model
format see [`SILICATEPACK.md`](SILICATEPACK.md). For the network stack see
[`net/network-stack.md`](net/network-stack.md). For codebase orientation see
[`codebase-guide.md`](codebase-guide.md).

---

## Overview

Silicate Zero is a from-scratch operating system kernel written in Rust that
runs LLM inference directly in Ring-0 on bare metal. The kernel *is* the
inference engine. A single bootable image contains the OS, drivers, and model
weights. Boot, infer.

The kernel eliminates every layer between the CPU and the model: no Linux, no
containers, no Python, no CUDA. This removes scheduling jitter, context-switch
overhead, and the speculative-execution mitigations (KPTI, Spectre retpolines,
IBRS/IBPB/STIBP) that mainstream kernels must carry. There is no user-space to
protect — all code runs at Ring-0 in a single address space.

### Supported Platforms

| Architecture | Acceleration | Status |
|---|---|---|
| x86_64 | AVX-512 | Production (EPYC Zen 4) |
| x86_64 | AVX2 | Functional |
| aarch64 | NEON | Functional (Apple Silicon via HVF, ARM servers) |
| Scalar | None | Reference / verification |

A single canonical inference function (`kernel/src/inference.rs`) executes
identically on all platforms. Platform-specific code is confined to the HAL
layer (`kernel/src/arch/{x86_64,aarch64}/`).

---

## Kernel Architecture

### Boot

Zero boots via UEFI (x86_64) or direct kernel load (aarch64 QEMU). The boot
sequence:

1. **Hardware init** — GDT, IDT, page tables, serial console.
2. **Memory layout** — physical memory mapped into a flat virtual address space.
   Model weights, KV cache, and kernel arenas each occupy dedicated physical
   regions.
3. **Model load** — the `.smodel` artifact is mapped from physical memory.
   The SIDX tensor directory is parsed, tensor pointers are resolved, and the
   model config / tokenizer are extracted. No filesystem involved — the model
   is embedded in the boot image or provided as a QEMU ramdisk.
4. **Inference start** — prompt tokens are fed through the forward pass.
   Output streams to serial console and (on x86_64) to a framebuffer.

The kernel boots in release mode only. Debug builds collide with the
bootloader's identity mapping.

### Memory Management

All dynamic memory is managed via **arena allocators** with bump pointers.
There is no free-list heap allocator in Ring-0.

| Arena | Size | Purpose |
|---|---|---|
| `KERNEL_ARENA` | 4 MiB | Kernel-internal `'static` allocations, async tasks |
| `RUNTIME_ARENA` | 2 MiB | Validator/interpreter allocations (`alloc::*` routing) |
| `KV_CACHE_ARENA` | 512 MiB | KV cache for autoregressive generation |
| Model region | Model-dependent | Zero-copy mapped `.smodel` weights |

Allocations are O(1) and lock-free. `dealloc()` is a no-op — memory is
reclaimed only by arena reset. This eliminates fragmentation, use-after-free,
and allocator contention.

### Network Stack

A polling Layer 2–4 network stack provides:

- **NIC drivers**: e1000 (QEMU dev), i40e (Intel X710), ice (Intel E810)
- **Layer 2**: Ethernet frame parsing and construction
- **Layer 3**: IPv4 with static configuration
- **Layer 4**: TCP (connection-oriented) and UDP
- **Application**: Telnet-style control plane on port 2222

The control plane accepts commands for model status, inference benchmarks, SMP
status, and memory diagnostics. See [`net/network-stack.md`](net/network-stack.md)
for protocol details and [`net/ice-e810-driver.md`](net/ice-e810-driver.md) for
the E810 driver.

---

## Inference Engine

The inference engine is a pure-Rust, `no_std`, zero-dependency implementation
running entirely in Ring-0. No external inference runtime (llama.cpp, vLLM,
ONNX) is involved.

### Forward Pass

The forward pass implements a standard Transformer decoder architecture:

1. **Token embedding** — lookup from the embedding matrix (Q8_0 quantized).
2. **Per-layer processing** (repeated N times):
   - RMSNorm on the residual stream
   - Grouped Query Attention (GQA) with RoPE positional encoding
   - KV cache read/write for autoregressive generation
   - RMSNorm on the attention output
   - Feed-forward network (gate/up projection, SiLU activation, down projection)
   - Residual connection
3. **Final RMSNorm** on the output residual.
4. **LM head** — project to vocabulary logits, argmax for next token.

The implementation supports configurable architecture parameters (layer count,
head count, head dimension, intermediate size, vocabulary size, RoPE theta,
RMS norm epsilon) loaded from the `.smodel` manifest at boot.

### Attention Variants

| Variant | Description | Status |
|---|---|---|
| **GQA** (Grouped Query Attention) | Multiple query heads share fewer KV heads. Standard for Qwen3, Llama, Gemma. | Production |
| **MLA** (Multi-head Latent Attention) | Compressed KV via low-rank projection. Used by DeepSeek-V2/V3. | Supported |
| **MoE** (Mixture of Experts) | Sparse expert routing with top-K gating. Used by DeepSeek-V2, Qwen-MoE. | Supported |

MoE models use a shared-expert + routed-expert architecture. Expert weights
are fused during packing (SilicatePack `pack-hf` handles expert fusion
automatically). The kernel dispatches to the selected experts per-token based
on the gating network output.

### Quantization

All matrix math operates on dequantized values. Weights are stored quantized
and dequantized on-the-fly during matmul:

| Format | Bits/weight | Block size | Use |
|---|---|---|---|
| F32 | 32 | — | Norms, RoPE frequencies, scalars |
| Q8_0 | 8.5 | 32 values / 34 bytes | Embeddings, LM head |
| Q4_0 | 4.5 | 32 values / 18 bytes | Attention/FFN weight matrices |
| Q4_0X4 | 4.5 | 4 rows interleaved | AVX-512 optimized layout |
| Q8_0X4 | 8.5 | 4 rows interleaved | AVX-512 optimized layout |

The interleaved X4 layouts (`Q4_0X4`, `Q8_0X4`) store 4 consecutive rows in
group-blocks, enabling the AVX-512 kernels to share activation loads across
8 independent FMA chains. Dequantized values are bit-identical to the plain
layout — only the storage order changes. NEON and scalar builds reject
interleaved models at load time.

### KV Cache

The KV cache stores key and value projections for all layers at f32 precision:

- Dedicated `KV_CACHE_ARENA` (512 MiB) at a fixed physical address
- Layer-strided layout: `layer_stride = 2 * block_size` (keys + values)
- Context length capped at the model's configured maximum
- No zero-initialization — arena pages are demand-allocated

### SMP (Symmetric Multiprocessing)

On multi-core x86_64 systems, the matmul workload is distributed across all
available cores via a lock-free work-stealing dispatcher. Each core processes
a row range of the output matrix independently. Core discovery uses ACPI/MADT
enumeration; cores are woken via INIT-SIPI-SIPI.

---

## Hardware Abstraction Layer

Platform-specific code lives in `kernel/src/arch/{x86_64,aarch64}/`. The HAL
provides:

- **Math kernels**: vectorized matmul (AVX-512 / NEON / scalar), RMSNorm,
  softmax, SiLU, RoPE
- **Dequantization**: Q4_0, Q8_0, Q4_0X4, Q8_0X4 block decode
- **Boot**: platform-specific init (GDT/IDT on x86_64, exception vectors on
  aarch64)
- **Console**: serial output (UART on both platforms), framebuffer on x86_64

The HAL boundary ensures that `kernel/src/inference.rs` contains no
platform-conditional code. Adding a new platform requires implementing the
math trait and the boot stub — the inference pipeline, model loading, and
control plane are shared.

### AVX-512 Acceleration

The x86_64 AVX-512 path implements:

- Fused Q4_0 dequantize + dot product in 512-bit registers
- 4-row interleaved layout (Q4_0X4/Q8_0X4) for sequential weight streaming
- 8 independent FMA chains per matmul tile
- Multi-core parallel dispatch across row ranges

This is the production performance path. The ~194.5 tok/s baseline on EPYC
9354P is achieved through this path.

### NEON Acceleration

The aarch64 NEON path implements:

- 128-bit SIMD dequantize + multiply-accumulate
- Intrinsic-level implementation (not auto-vectorized)

The NEON path is functional and verified for correctness but is not the
primary performance target.

---

## .smodel Format

`.smodel` (SilicatePack) is the native model container for Silicate Zero. It
replaces GGUF as the production format.

Structure:
- **SILM header** (128 bytes) — magic, version, offsets
- **SIDX tensor directory** — JSON manifest with tensor names, dtypes, byte
  ranges, model config, tokenizer, and validation anchors
- **Tensor payload** — 64-byte aligned tensors, 2 MiB payload alignment for
  hugepage-friendly mapping

SilicatePack (`tools/silicatepack.py`) converts Hugging Face SafeTensors
models to `.smodel`. See [`SILICATEPACK.md`](SILICATEPACK.md) for the format
specification and [`silicatepack-guide.md`](silicatepack-guide.md) for the
step-by-step packing guide.

---

## Beta-Anchor Verification

Silicate Zero enforces **bit-exact determinism** of the forward pass via
beta-anchor verification at boot. This is a hard gate, not a soft check.

### How It Works

Every `.smodel` artifact carries validation anchors in its manifest:

1. **Anchor prompt** — a fixed token sequence (e.g., the token for "Hello").
2. **Expected next token** — the argmax token ID the model must produce.
3. **Expected logit bits** — the IEEE 754 bit pattern of the top-1 logit.

At boot, the kernel runs the anchor prompt through the full forward pass and
compares the result:

- **Token ID mismatch** → kernel halts (infinite spin loop). The model is
  corrupt or the inference engine has a regression.
- **Logit bits mismatch** → warning on serial console (feature builds may
  exhibit <= 1 ULP drift due to different FP instruction scheduling).
- **Both match** → model integrity verified, inference proceeds.

### Why Bit-Exactness

A single f32 LSB drift propagates exponentially across 28+ layers, 16+ query
heads, and 64+ generation steps. Epsilon tolerance at the output is equivalent
to arbitrary behavior at intermediate layers. Bit-exactness is the only
reproducibility contract that composes.

### Two-Anchor Regime

| Mode | Token ID | Logit Bits | 32-Token Sequence |
|---|---|---|---|
| **Sacred path** (no SIMD features) | Hard gate | Hard gate | Hard gate |
| **Feature build** (NEON / AVX-512) | Hard gate | Diagnostic only | Hard gate |

Feature builds may produce logit bits that differ by <= 1 ULP from the sacred
path due to LLVM instruction reordering within FMA chains. The token ID (argmax)
and the full 32-token autoregressive sequence remain identical.

### Per-Profile Anchors

Anchors are profile-specific. An anchor captured under `cpu-avx512` is not
validated by a NEON build. SilicatePack supports per-profile anchor storage:
pack once per target, capture the anchor on the target hardware, then promote
to strict. See [`silicatepack-guide.md`](silicatepack-guide.md) §6 for the
capture-then-promote workflow.

---

## Quarks Language

Quarks is Zero's native kernel language. It uses S-Expression syntax and is
validated by a Ring-0 type checker before execution.

Current components:

- **quarks-frontend** — lexer, parser, type checker (135 tests)
- **quarks-validator** — Ring-0 bytecode validator (270 tests)
- **quarks-interpreter** — Ring-0 S-Expression interpreter with fn/call
  semantics, stack frames, recursion limits
- **quarks-lsp** — Language Server Protocol implementation
- **vscode-quarks** — VS Code extension

Quarks programs are expressed as S-Expressions:

```lisp
(program
  (fn main () i64
    (add 1 2))
  (call main))
```

The validator ensures type safety, stack balance, and capability correctness
at load time. Once validated, code executes in the shared Ring-0 address space
without hardware isolation overhead.

---

## Test Suite

The workspace contains 741+ tests across nine crates, covering:

- **Unit tests** — per-function validation of lexer tokens, validator rules,
  bytecode instructions, arena allocations, dequantization
- **Integration tests** — component combinations: interpreter with validator
  IR, forward pass with weights and KV cache
- **Property tests** (proptest) — algebraic invariants: determinism across
  execution paths, pause/resume equivalence, panic-freedom on random input

```bash
cargo test --workspace --release
```

---

## Project Layout

```
kernel/
  src/
    main.rs              # boot entry point
    inference.rs         # forward pass (platform-independent)
    inference_neon.rs    # NEON-specific inference hooks
    arch/
      x86_64/            # x86_64 HAL, boot, AVX-512 math
      aarch64/           # aarch64 HAL, boot, NEON math
crates/
  zero-llm-inference/    # inference operator library
  zero-gguf-parser/      # model format parsing
  quarks-frontend/       # language frontend
  quarks-validator/      # Ring-0 validator
  quarks-interpreter/    # Ring-0 interpreter
  quarks-arena/          # arena allocator
  quarks-lsp/            # LSP server
tools/
  silicatepack.py        # model packing tool
  build-x86_64-baremetal.sh
docs/
  ARCHITECTURE.md        # this file
  PERFORMANCE.md         # benchmark data
  SILICATEPACK.md        # .smodel format reference
  silicatepack-guide.md  # model packing guide
  net/                   # network stack docs
```

---

## License

AGPL-3.0-or-later. See [LICENSE](../LICENSE).
