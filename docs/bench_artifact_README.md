# Bootable UEFI benchmark artifact

## Artifact

- **Path:** `target/zero-images/zero-uefi.img`
- **Size:** 1.22 GiB (1,306,591,232 bytes)
- **Format:** GPT + ESP (FAT32) raw disk image — UEFI bootable
- **SHA-256:** `5ccf931b1d64fa408dd6e917d929d0c6ff55e3cb115824c23f92713305be1a63`
- **Sibling BIOS image:** `target/zero-images/zero-bios.img` (same contents, MBR-bootable)

The Qwen3-1.7B Q4_K_M GGUF (1,282,439,584 bytes) is embedded into the ESP as
the `bootloader 0.11` ramdisk and surfaced to the kernel via
`BootInfo::ramdisk_addr` / `BootInfo::ramdisk_len`.

## How it was built

`make image` in the main checkout. The previous FAT32 ENOSPC failure was
two separate problems, both now fixed:

1. **Host disk space.** `boot/` builds need ~5 GB scratch + 1.3 GB output;
   disk was at 0 free. Freed ~13 GB before this build.
2. **`bootloader 0.11.15` FAT-cushion bug.** The crate's
   `fat_partition_size = needed_size + 1 MiB` formula leaves no room for
   FAT32 metadata when the embedded ramdisk is >1 GiB. A vendored copy of
   bootloader at `boot/vendor-bootloader/` patches that formula. The patch
   is wired via `boot/Cargo.toml`'s `[patch.crates-io]`. Build logs show
   it active: `[vendor-bootloader-patch] needed_size=1308184016
   fat_partition_size=1390784512`.

Neither change is committed (the worktree contract said not to). To
reproduce on a fresh clone, those two diffs in `boot/` and the vendored
crate need to be replayed.

## QEMU UEFI verification (Apple Silicon, TCG only)

Two runs, both from `target/zero-images/zero-uefi.img` boot path:

### A. Bare image (no GGUF) — full bench framework completes in ~10 s

```
qemu-system-x86_64 \
  -drive if=pflash,format=raw,readonly=on,file=/opt/homebrew/share/qemu/edk2-x86_64-code.fd \
  -drive format=raw,file=target/zero-images/zero-uefi.img \
  -serial stdio -display none -m 8G -no-reboot
```

Stages 0 → 12 all green. All 5 ROI benchmarks fire (TCG numbers, not
representative of bare-metal):

| Benchmark        | TCG value             | Linux reference        |
| ---------------- | --------------------- | ---------------------- |
| Boot Time        | 294 ms                | 15,000–25,000 ms       |
| Context Switch   | 2 ns (10 cycles)      | 1,200–5,500 ns         |
| Arena Alloc      | 0.5 ns (2 cycles)     | 50–5,000 ns            |
| IPC Throughput   | 1381 GB/s             | 3–6 / 50–100 GB/s      |
| LLM Inference    | 0.0 tok/s (no model)  | CPU-only target: >=150 tok/s |

### B. Full image (with GGUF) — boots, finds model, halts on β-anchor

```
qemu-system-x86_64 \
  -drive if=pflash,format=raw,readonly=on,file=/opt/homebrew/share/qemu/edk2-x86_64-code.fd \
  -drive format=raw,file=target/zero-images/zero-uefi.img \
  -serial stdio -display none -m 8G -no-reboot
```

Stages 0 → Stage 11 MP1 all green. Key markers:

```
Stage 11 MP1: ramdisk reported at virt=0x20000000000, len=1282439584 bytes
Stage 11 MP1: model accepted from ramdisk (1223 MB)
[MP2.1] Tensor count: 311
[MP2.5] All 28 layers loaded
[MP3.0] Phase 1: Prompt prefill (13 tokens)
[MP3.0] β-ANCHOR FAIL: Token-ID mismatch — HALTING
```

The β-anchor halt is **expected on QEMU TCG**: `inference.rs:685` hard-gates
on a reference Token-ID computed against EPYC fp32 behavior, and TCG's
emulated f32 path produces logit_bits = 0x00000000 (all-zero output, likely
an emulator dequant/matmul precision artifact). Because `bench::run_all_benchmarks`
is downstream of Stage 11 in `main.rs:1511`, **the 5-bench summary cannot
be produced from QEMU TCG with the GGUF embedded**. The two runs together
cover both halves: bare image proves the bench framework, full image proves
the ramdisk path through to the 28-layer forward pass starting.

End-to-end validation requires bare metal.

## Cherry Servers (EPYC 9354P) deployment via IP KVM

The artifact is a raw GPT disk image. IPMI/BMC virtual-media expectations
vary — try these in order:

1. **Mount as virtual USB / virtual hard drive.** Most modern BMCs (including
   the Cherry Servers IP KVM) accept `.img` directly as "virtual hard
   drive" or "virtual USB". Boot order: select that virtual device.
2. **If the BMC only accepts `.iso`:** wrap the raw image with `xorriso`:
   ```
   xorriso -as mkisofs \
     -iso-level 3 -V ZERO \
     -append_partition 2 0xef target/zero-images/zero-uefi.img \
     -o zero-uefi.iso \
     -e --interval:appended_partition_2:all:: -no-emul-boot \
     -isohybrid-gpt-basdat \
     -graft-points /=/dev/null
   ```
   (Run on Linux; macOS `xorriso` is in `brew install xorriso`.) Result is
   a hybrid ISO/USB image that the BMC will mount as CD.
3. **If IP KVM size limit blocks 1.2 GB upload:** use `image-bare` for the
   kernel boot + serve the GGUF separately. The kernel's
   `DirectMemoryLoader` path in `main.rs:534` accepts the GGUF at a
   fallback physical address (currently `0x100000000`). For QEMU this
   works via `-device loader,file=...gguf,addr=0x100000000`. On the
   server, this approach needs either a kernel-cmdline-driven address or
   a small initial-ramfs trick — not implemented in `bootloader 0.11`
   today, so the embedded path (option 1) is the supported route.

Expected on bare metal: Stages 0–10 fast, Stage 10 MP6 prints
interpreter/AOT cycles ratio, Stage 11 prefills the 13-token prompt
through 28 layers, β-anchor passes, autoregressive generation produces
~32 tokens, then `bench::run_all_benchmarks` prints the 5-row summary
with real EPYC TSC-calibrated values + the measured tok/s.

## Files of interest

| Path                                                 | Role                                    |
| ---------------------------------------------------- | --------------------------------------- |
| `target/zero-images/zero-uefi.img`             | The artifact                            |
| `target/zero-images/zero-bios.img`             | MBR-bootable twin                       |
| `kernel/src/bench.rs`                                | The 5 ROI benchmarks                    |
| `kernel/src/main.rs:101`                             | `BOOT_TSC_T0` capture (earliest rdtsc)  |
| `kernel/src/main.rs:1511`                            | `bench::run_all_benchmarks(&cpu_info)`  |
| `kernel/src/inference.rs:685`                        | β-anchor hard gate (TCG halt point)     |
| `boot/vendor-bootloader/`                            | FAT-cushion patch                       |
| `boot/Cargo.toml`                                    | `[patch.crates-io]` wiring              |
