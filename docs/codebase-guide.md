# Zero Codebase Guide

> A map for new contributors: how the code is organized, how to build and
> test it, where the hard constraints are, and how to make a change that
> survives review.

This is the *orientation* guide — how to understand the code and work on
it. For the operational deploy path (building an image, flashing NVMe,
serial console), see [`developer-guide.md`](developer-guide.md). For the
why behind the design, see [`ARCHITECTURE.md`](ARCHITECTURE.md) and the
[ADRs](adr/).

---

## 1. The shape of the project

Zero is a **Unikernel**: one Ring-0 address space, no user/kernel
split, no host OS underneath. It boots on bare metal (x86_64 and
aarch64), brings up a 1.7B-parameter Qwen3 **Boot-LLM** in-kernel, and
runs a custom language, **Quarks**, behind a compile-time validator.

The repository splits cleanly into two build worlds:

| World | What | How it builds | Target |
|---|---|---|---|
| **Kernel** | `kernel/`, `boot/` | `make` (not `cargo` at root) | `x86_64-unknown-none`, `aarch64-unknown-none` (`no_std`) |
| **Host crates** | `crates/*` | `cargo build/test --workspace` | host triple (std) |

`kernel/` and `boot/` are **excluded** from the Cargo workspace
(`Cargo.toml`) because they are `no_std` and target bare metal — you
build them through the `Makefile`, never with a bare `cargo build` at the
root.

```
zero-kernel/
├── kernel/src/            no_std Ring-0 kernel (boots on bare metal)
├── boot/                  host-side disk-image builder
├── crates/                std host crates (compiler, inference, sandbox…)
├── tools/                 SilicatePack (.smodel producer), VS Code ext, scripts
├── docs/                  architecture, ADRs, runbooks, this guide
└── Makefile               build orchestration for kernel + images
```

---

## 2. The host crates (`crates/`)

These are normal Rust crates with `std`, tested via `cargo test
--workspace`. They are where most logic that *can* be tested off-target
lives — the kernel re-uses or mirrors them.

| Crate | Purpose |
|---|---|
| `quarks-frontend` | Lexer, parser, source-level type checker, codegen |
| `quarks-validator` | Ring-0 IR validator — type safety, stack analysis, capability checks |
| `quarks-codegen` | Quarks → S-Expression IR codegen + perf harness |
| `quarks-interpreter` | Reference interpreter for IR / per-sandbox execution context |
| `quarks-arena` | Arena allocator primitives (V3 Arena-Disziplin) |
| `quarks-lsp` | Language Server Protocol — editor diagnostics |
| `zero-llm-inference` | Reference LLM inference engine (ops, attention, KV cache, LM head) |
| `zero-sandbox` | Stage-12 sandbox model (lifecycle, capabilities, isolation) |
| `zero-hal` | Hardware-abstraction service traits (network, time-slice, …) |
| `zero-nvme` | NVMe host-side helpers |
| `zero-translator` | Format/translation helpers |
| `zero-gguf-parser` | Legacy GGUF-compat parser, retained only for comparison benchmarks |

> The **native** model format is `.smodel` (SilicatePack) — see §6. The
> `zero-gguf-parser` crate is legacy and only used for compatibility
> benchmarks; new work targets `.smodel`.

---

## 3. The kernel (`kernel/src/`)

The kernel is one `no_std` binary that boots on both architectures from
`main.rs`. The modules group by concern:

**Boot & architecture**
- `main.rs` — kernel entry; the staged boot sequence for x86_64 +
  aarch64.
- `arch/x86_64/` — GDT, IDT, PIC/APIC, serial, PCIe, interrupts.
- `arch/aarch64/` — VBAR_EL1, MMU, GICv2, generic timer, PL011 UART.
- `smp.rs` — application-processor (AP) bring-up, trampoline, per-core
  matmul row distribution.

**Memory (handle with care — see §5)**
- `memory.rs` — **SACRED.** Cross-platform arena allocator, physical
  offset, hugepage promotion, MMIO mapping. Do not modify without an ADR.
- `arena_allocator.rs`, `llm_arena.rs` — bump-pointer arenas for runtime
  and model/KV data.

**Boot-LLM (the inference engine)**
- `model_loader.rs` — `.smodel` container + SIDX tensor-directory parse,
  validation anchors.
- `weight_layout.rs` — row-interleave registry (`group_of`, `register`,
  `rebase`) for `.smodel`-v2 Q4_0X4/Q8_0X4 tensors.
- `inference.rs` — Boot-LLM orchestration: prompt prefill, the 28-layer
  loop, the boot-time weight-integrity scan, the β-anchor gate.
- `inference_avx512.rs` — x86_64 AVX-512 forward-pass kernels.
- `inference_neon.rs` — aarch64 NEON feature path.
- `sampler.rs`, `rng.rs` — feature-gated stochastic sampling.
- `detokenizer.rs`, `vocab_*.bin`, `kimi_vocab_*` — tokenizer assets.
- `bench.rs` — boot-path benchmark + held benchmark screen / anchor
  recording.

**Subsystems**
- `net/` — the network stack ([`net/network-stack.md`](net/network-stack.md)).
- `drivers/` — device drivers (NVMe, etc.).
- `lfb/` — linear framebuffer UI.
- `task/` — cooperative async executor.
- `sandbox.rs`, `control_plane.rs` — Stage-12 sandboxing + control plane.
- `aot.rs` — AOT-compiled validator path.

---

## 4. Building and testing

### Host crates (fast, do this constantly)

```bash
cargo test --workspace --release          # the workspace gate
cargo build -p quarks-lsp              # the LSP server binary
```

### Kernel images (via Makefile, never bare cargo)

```bash
make image            # x86_64 BIOS + UEFI images
make run              # boot x86_64 in QEMU with the Boot-LLM
make build-aarch64    # aarch64 kernel binary
make run-aarch64      # boot on Apple HVF (Apple Silicon)
make run-aarch64-tcg  # boot on QEMU TCG (any host)

make kernel-cherry    # Cherry Server build (cherry-net + AVX-512 features)
make image-cherry     # bootable Cherry image
```

> **Release-mode only.** The kernel *must* build in release. Debug builds
> collide with the bootloader's identity mapping (`PageAlreadyMapped`).
> The Makefile enforces this.

### The verification gate

A change to kernel code is "green" when **all three** pass:

1. `cargo test --workspace` — the host test suite (currently **1,930
   tests** passing on the latest network-fix gate; the README's
   historical figure was 741 host-side tests at the Stage-12 G5 landing).
2. `make kernel-cherry` — the x86_64 Cherry/AVX-512 build *Finished*.
3. `make build-aarch64` — the aarch64 build *Finished*.

For any inference-path change, the **β-anchor** must still hold
(Token-ID 25 — see §5). The commit messages for the recent network fixes
(`8349fbd`, `cd12ec1`) are good templates for what a passing gate report
looks like.

> **Dev-machine caveat.** The reference dev box is an Apple Silicon Mac
> (ARM). AVX-512 code paths therefore *cannot* be exercised locally — they
> are validated on the Cherry Server. Plan x86-specific changes around a
> Cherry boot for real verification.

---

## 5. Sacred boundaries (read before you touch anything)

Zero has a small set of invariants that are **load-bearing**.
Breaking one silently is the most expensive mistake you can make here.

### `memory.rs` is SACRED

Do not modify `kernel/src/memory.rs` without an ADR. The arena layout,
physical offsets, and hugepage-promotion logic are co-designed with the
inference engine's address assumptions. A past bug
(`6c1fa45`) where the aarch64 KV arena was placed *inside* the 1.4 GiB
`.smodel` payload silently corrupted the LM-head and produced garbage
logits — fixed by moving the KV arena above the model, not by touching
the model. The boot-time integrity scan (§6) exists because of this
class of bug.

### The β-anchor (Token-ID 25)

The deterministic Boot-LLM path is **bit-exact** and gated. After
processing the first prompt token (ID 9707, "Hello"), the argmax output
must be **Token-ID 25** with `logit_bits = 0x414a6497` on the Sacred
Scalar path. Every optimization (AVX-512, interleave-4, NUMA) preserves
the K-order and per-output-row FMA sequence so this anchor never moves.
If your change shifts the anchor, the change is wrong — not the anchor.
The anchor is recorded by `bench.rs::record_llm_anchor` and surfaced on
the held benchmark screen for KVM photo capture. See the
[two-anchor verification](discovery/sub-mp-g2/g2-two-anchor-preservation-proof.md).

### "No foreign patterns" in native drivers

The bare-metal drivers (`net/`, `drivers/`) are independent native
implementations. Reference structures (e.g. Intel's `ice_tlan_ctx_info[]`)
are used only as *field-layout cross-checks*, never copied wholesale. The
ICRC bug ([ice doc §6.2](net/ice-e810-driver.md)) was exactly a foreign
i40e idiom leaking into the E810 path — that is the anti-pattern to avoid.

### Streaming mode is separate

`streaming-mode` is a feature-gated demo/prototype path (the "Eternal
Dream Stream"). It is **not** part of the deterministic default Boot-LLM
and is not bound by the β-anchor. Keep the two mentally separate.

---

## 6. The `.smodel` model format (orientation)

`.smodel` (SilicatePack) is the native Zero-Server model container —
mapped directly from physical memory, zero-copy after page-table setup.
Full spec in [`SILICATEPACK.md`](SILICATEPACK.md);
the loader lives in `model_loader.rs`. Orientation-level facts:

- **Magic** `"SILM"` (`0x4D4C4953`), header 128 bytes, payload aligned to
  2 MiB. Version 1 = plain layouts; **version 2** signals tensors *may*
  be row-interleaved.
- Native payload starts with a **SIDX** tensor directory (`"SIDX"`,
  `0x5844_4953`); 104-byte fixed tensor entries followed by a name blob,
  config/tokenizer sidecars, then the aligned tensor payload.
- **Interleaved dtypes** (v2): `Q4_0X4` (id 100, 72-byte 4-row groups)
  and `Q8_0X4` (id 101, 136-byte 4-row groups). Produced by SilicatePack
  with `--interleave 4`; these dtype ids sit *outside* any legacy id
  space on purpose so the format can never be mistaken for another.
- `weight_layout.rs` maps a byte address back to its interleave group
  (`group_of`, binary search) so the integrity scan and the kernels read
  interleaved bytes correctly. Interleaving **never changes the
  computed result** — the same per-block FMA order is preserved.
- The boot-time integrity scan (`inference.rs::check_weight_integrity`)
  probes block scales (a quantized scale > 64 proves the bytes are
  garbage) to catch arena/model overlap *before* it shows up as bad
  logits.

The producer toolchain is `tools/silicatepack.py`; the end-user guide is
[`silicatepack-guide.md`](silicatepack-guide.md).

---

## 7. Quarks & the compiler

Quarks is Zero's own statically-typed, intent-first language. The
pipeline:

```
.qk source → Lexer → Parser → Type Checker → Codegen → S-Expression IR → Ring-0 Validator
            └──────────── quarks-frontend ──────────┘   └ quarks-validator ┘
```

Safety is enforced at **compile time** by the validator (type safety,
stack-depth analysis, capability checks) — this is what replaces hardware
process isolation in the Unikernel model. The `quarks-lsp` server
gives editors live diagnostics; `tools/vscode-quarks/` is the VS Code
client.

---

## 8. How decisions are recorded

- **`docs/lessons-learned.md`** — empirical lessons; 10+ are now
  canonical architecture.
- **`docs/DEFERRED_DECISIONS.md`** — tracked decisions with explicit
  revisit triggers.
- **Discovery reports** (`docs/discovery/`) — investigation write-ups
  (SMP debugging, performance audits, anchor captures).
- **Runbooks** (`docs/deploy/`, root `DEPLOY-RUNBOOK-*.md`) — operational
  deploy procedures.

Every commit is a small, verified step with a gate report in the message.
The project's identity is that it is built *by* agents under human
oversight, so the git history is the audit trail — keep commits atomic
and their messages honest about what was and wasn't verified.

---

## 9. A first-change checklist

1. **Locate the world.** Host crate (`cargo`) or kernel (`make`)?
2. **Check the boundaries.** Does it touch `memory.rs`, the inference
   path, or a native driver's hardware contract? If yes, read the
   relevant ADR first and expect to write/extend one.
3. **Make it testable off-target where possible.** Pure logic (bit
   packers, parsers, candidate ordering) gets a unit test — see the
   `pack_tlan_bits_*` and `order_parent_candidates_*` tests in
   `net/ice.rs` for the pattern.
4. **Run the gate.** `cargo test --workspace`, `make kernel-cherry`,
   `make build-aarch64`. For inference changes, confirm the β-anchor.
5. **Write the commit like the recent ones.** State scope, root cause,
   the fix, and the exact gate results — including what you could *not*
   verify locally (e.g. AVX-512 on an ARM dev box).

---

## 10. See also

- [`developer-guide.md`](developer-guide.md) — end-to-end build → deploy
  → serial-console operational guide.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the V3.4 north-star vision.
- [`net/network-stack.md`](net/network-stack.md) /
  [`net/ice-e810-driver.md`](net/ice-e810-driver.md) — the network
  subsystem.
- [`PERFORMANCE.md`](PERFORMANCE.md) and
  [`../PERF-REPORT-qwen-smodel-2026-06-12.md`](../PERF-REPORT-qwen-smodel-2026-06-12.md)
  — the inference performance baseline.
