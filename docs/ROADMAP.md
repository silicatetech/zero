# Hardware Roadmap

This document is about *where* Zero runs. The *what* (kernel stages 0–12) lives in [ARCHITECTURE.md](ARCHITECTURE.md). The two advance in parallel but are independent — you can add kernel features without changing hardware target, and you can port to new hardware without changing kernel features.

**Phases are capability-triggered, not calendar-dated.** Time estimates for this kind of project are unreliable in general and especially so for a solo developer working with AI assistance. Each phase lists what makes it ready to start and what makes it "done enough" to consider the next phase.

## Phase 1 — QEMU on x86_64

**Status:** current.

**Readiness to start:** zero. This is where the project lives now.

**Why this phase:** fastest iteration loop. No hardware debugging surface. No boot-variance between runs. Every Rust-kernel hobby project starts here.

**Tooling:** `bootloader` crate 0.11 (BIOS and UEFI images produced), `qemu-system-x86_64`, serial port for diagnostics, framebuffer for display.

**Exit criteria (what should be true before moving on):**
- Kernel has memory management (frame allocator, paging, heap).
- Kernel has interrupt handling (GDT, IDT, CPU exception handlers).
- Kernel has a basic scheduler.
- Intent IPC primitive exists.
- At least one non-trivial agent runs inside the kernel.

These correspond to roughly ARCHITECTURE.md kernel stages 0 through 6.

## Phase 2 — x86_64 real hardware

**Status:** not started.

**Readiness to start:** Phase 1 exit criteria met.

**Why this phase:** proves the kernel works on silicon, not just emulation. Catches assumptions that QEMU quietly fixes but real hardware does not (timing, memory layout, broken ACPI tables, BIOS quirks).

**Candidates:**
- An old laptop or desktop with UEFI and x86_64. Boot from USB stick.
- A dedicated x86_64 dev board (Intel NUC, Minisforum, etc.).

**What changes from Phase 1:**
- UEFI boot path becomes the primary (bootloader 0.11 supports both).
- Real-hardware drivers start to matter (NVMe or SATA for storage, at minimum).
- Thermal, power, and timing behaviors differ from QEMU.

**Exit criteria:**
- Kernel boots from USB on at least two different physical x86_64 machines.
- Serial or framebuffer output works on real hardware.
- Memory map from real UEFI is consumed correctly.
- Agents that worked in QEMU also work on bare metal.

## Phase 3 — ARM64 dev board

**Status:** not started.

**Readiness to start:** Phase 2 stable, Quarks or Rust has ARM64 code generation sorted out.

**Why this phase:** the architecture that matters for the long-term target is ARM64 — specifically Apple Silicon, but also the future direction of servers and consumer devices. x86_64-only is a dead end. An ARM64 port also exercises the kernel's architecture-abstraction seams.

**Candidates (in rough order of ease):**
- **Raspberry Pi 5** — most-documented ARM64 SBC, active hobby-OS community, straightforward boot path (UEFI firmware available), affordable.
- **Rock 5B** — RK3588 SoC, better GPU, slightly more peripheral variance. Good if RPi5 ecosystem feels limiting.
- **Pine64 Quartz64** — alternative, smaller community.
- **Standard UEFI ARM64 server boards** (if available) — closest match to what will eventually run on Apple Silicon.

Raspberry Pi 5 is the default candidate unless specific reasons arise to prefer another.

**What changes from Phase 2:**
- Second code-generation target in Quarks (or Rust, depending on dependency policy phase).
- ARM64-specific boot path, interrupts (GIC), MMU semantics.
- New driver set: storage (SD/eMMC/NVMe), display (HDMI or DSI), network (USB or Ethernet), GPIO if useful.

**Exit criteria:**
- Kernel boots on a Raspberry Pi 5 (or chosen equivalent) from SD card or USB.
- Serial output works.
- Basic display output works (framebuffer via HDMI).
- Same agents run on ARM64 and x86_64 without divergence.

## Phase 4 — Apple Silicon (optional)

**Status:** not started. Explicitly marked optional.

**Readiness to start:** Phase 3 stable; clear path to Apple-Silicon reverse engineering, which in practice means either (a) collaborating with or building directly on [Asahi Linux](https://asahilinux.org)'s hardware abstractions, or (b) a massive independent reverse-engineering effort we are unlikely to have resources for.

**Why this phase is difficult — the Asahi benchmark:**

Asahi Linux is the reference for what it takes to run a non-Apple operating system on Apple Silicon.

- Started: 2020-2021.
- Core team: ~5-10 contributors, broader community of occasional participants.
- By 2026: M1, M2, M3 supported; M4 still in progress.
- Workload: reverse-engineering a closed hardware platform with no vendor documentation. Each new SoC generation requires significant additional work.
- Visible results include a custom GPU driver (by Alyssa Rosenzweig) and the m1n1 bootloader.

Replicating their effort from scratch is not realistic for this project. Building on it (using m1n1, borrowing hardware abstractions) is the only feasible path.

**Hostile components on Apple Silicon:**
- Boot chain is closed (iBoot, secure boot).
- GPU (AGX) — no public documentation.
- NVMe controller — proprietary variant.
- DART (Apple's IOMMU) — undocumented.
- Display (HDMI / internal panel) — undocumented pipeline.
- Wi-Fi, Thunderbolt, audio codec, secure enclave — all undocumented.

**What changes from Phase 3:**
- Almost everything. Apple Silicon is a different world. Phase 3 ARM64 knowledge transfers in theory; in practice, every driver is a new problem.

**Exit criteria:**
- Bootable image for at least one M-series Mac.
- Serial or framebuffer output works.
- Basic agents run.

**Reality check:** this phase may never happen. Zero is not less valuable if it never runs on Apple Silicon. The original framing that "Mac Mini M4 is the target" was aspirational — the *real* target is an agent-native OS, running on whatever hardware the project can reach. Apple Silicon is a bonus, not a requirement.

## Non-phases

The following are explicitly out of scope for this roadmap:

- **iOS / iPadOS** — closed; no path.
- **Game consoles** (PlayStation, Xbox, Switch) — closed; no path.
- **Android devices** — partial but variance is too high to usefully target; not planned.
- **RISC-V** — interesting direction but not a committed phase. If a compelling RISC-V platform emerges, a new phase can be added.
