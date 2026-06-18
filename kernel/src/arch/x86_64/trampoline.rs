// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR-029 Phase 3 — AP trampoline: real-mode → protected-mode → long-mode.
//!
//! When the BSP fires an INIT-SIPI-SIPI at an AP via the Local APIC,
//! the AP wakes in 16-bit real mode with `CS:IP = (vector << 8):0000`,
//! where `vector` is the second byte of the SIPI's ICR low half. The
//! AP has none of the BSP's state: no GDT, no IDT, no paging, none of
//! the long-mode infrastructure that lets Rust run.
//!
//! This module is responsible for assembling the bridge code that
//! takes an AP from that minimal state all the way into 64-bit Rust
//! ([`ap_long_mode_entry`]). The bridge is:
//!
//! 1. **16-bit real mode**: load a temporary GDT, set `CR0.PE`, and
//!    far-jump to a 32-bit protected-mode segment.
//! 2. **32-bit protected mode**: set data segments, set `CR4.PAE`, load
//!    a low bootstrap `CR3`, enable `EFER.LME|NXE`, set `CR0.PG`, and
//!    far-jump into the 64-bit code segment.
//! 3. **64-bit long mode**: set up a per-AP stack from the
//!    [`AP_STACKS`] arena, mirror the BSP's real CR0/CR4, switch to
//!    the BSP's real CR3, then jump into [`ap_long_mode_entry`] (Rust
//!    function).
//!
//! # Why hand-rolled byte arrays?
//!
//! The trampoline must live at a *physical* address ≤ 0xFFFFF (real-
//! mode addressable range, capped because SIPI vector is 8-bit and
//! the start address is `vector << 12`). The kernel binary itself
//! lives in the higher half (virtual `0xFFFF_8000_*`), so we can't
//! just `&function as _`. Instead we encode the trampoline as raw
//! bytes in `.rodata`, then [`install_trampoline`] copies them to
//! the chosen low-memory page (0x8000) at runtime.
//!
//! Inline `core::arch::asm!` would let us write source-form assembly,
//! but Rust would still link the bytes into the higher-half binary
//! and we'd need a special section to relocate them. The raw-bytes
//! approach is simpler, fully reviewable (`xxd` the byte array), and
//! avoids any toolchain magic.
//!
//! # Layout at physical 0x8000 after [`install_trampoline`]
//!
//! ```text
//!   0x8000  trampoline_start (16-bit code)
//!   0x8180  trampoline GDT (4 entries × 8 B = 32 B)
//!   0x81A0  GDTR pointer (6 B)
//!   0x81B0  64-bit jump address (8 B, points to ap_long_mode_entry)
//!   0x81B8  bootstrap CR3 value (8 B, low PML4 physical address)
//!   0x81C0  AP stack pointer slot (8 B, written by BSP per AP)
//!   0x81C8  real CR3 value (8 B, BSP's PML4 physical address)
//!   0x81D0  real CR4 value (8 B, BSP page-table feature bits)
//!   0x81D8  real CR0 value (8 B, BSP core-control baseline)
//!   0x81E0  raw trampoline probe mode byte
//!   0x81E1  raw trampoline probe stage byte
//!   0x8200  end
//! ```
//!
//! # The CR3 / page-tables question
//!
//! We use a tiny low bootstrap PML4 for the real-mode/compat-mode
//! transition, then switch to the BSP's real PML4 once the AP is in
//! 64-bit mode. This avoids truncating high physical CR3 values on
//! large EPYC systems while still ending in the shared Zero
//! address space. This requires:
//! * The bootloader has identity-mapped (or linear-mapped) the first
//!   1 MiB so the trampoline can execute from physical 0x8000 once
//!   paging is on. `bootloader_api` does this — see
//!   `kernel_main::BOOTLOADER_CONFIG::mappings.physical_memory =
//!   FixedAddress(0xFFFF_8000_0000_0000)`. The linear physical map
//!   covers all of RAM, but for AP boot we need an *identity* mapping
//!   for the 0x8000 page so the CPU's RIP works pre-far-jump.
//!
//!   On x86_64 with `bootloader_api`, the kernel sees a higher-half
//!   linear map of physical memory, but the boot stub's identity
//!   map of low memory is torn down after CR3 switch. We must install
//!   an explicit 4-KiB identity-mapped page for 0x8000 before issuing
//!   INIT-SIPI-SIPI; see [`ensure_trampoline_identity_mapped`].
//!
//! # Pillar conformance
//!
//! * **Pillar 1** — Trampoline runs once per AP; no heap; per-AP stack
//!   allocation is from a fixed `static` arena.
//! * **Pillar 7** — Architecture-specific by construction. Lives in
//!   `arch::x86_64`.
//! * **Pillar 8** — Long-term: replace with an Quarks-emitted
//!   trampoline. Not in scope before Stage 24.
//!
//! CITE: AMD64 APM Vol 2 §14.6 (Long Mode Activation)

// TRAMPOLINE_MAX_BYTES is a documented size ceiling used by callers
// who pre-reserve the 4 KiB page; it has no current reader in-tree
// (the page is reserved by the bootloader's identity map).
#![allow(dead_code)]

use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use super::{apic, cpuinfo, interrupts};
use crate::smp::{self, MAX_CORES};

// ─────────────────────────────────────────────────────────────────
// Trampoline physical placement
// ─────────────────────────────────────────────────────────────────

/// Physical address where the trampoline lives. The SIPI vector is
/// `TRAMPOLINE_PHYS >> 12 = 0x08`. Must be ≤ 0xFF000 (real-mode
/// 20-bit addressable range), aligned to 4 KiB, and below the boot
/// stub's working memory.
///
/// `0x8000` is a conventional SIPI trampoline placement: below
/// 1 MiB, page-aligned, and away from the 0x7C00 boot sector window.
pub const TRAMPOLINE_PHYS: u64 = 0x8000;

/// SIPI vector byte (`TRAMPOLINE_PHYS >> 12`).
pub const TRAMPOLINE_SIPI_VECTOR: u8 = (TRAMPOLINE_PHYS >> 12) as u8;

/// Maximum trampoline image size (one 4 KiB page). The actual image
/// is ~250 bytes; this is the install ceiling.
pub const TRAMPOLINE_MAX_BYTES: usize = 4096;

// ─────────────────────────────────────────────────────────────────
// Per-AP stack arena
// ─────────────────────────────────────────────────────────────────

/// Per-AP stack size. 64 KiB. The AP's kernel-side work is the
/// matmul kernel itself (`linear_q4k_avx512_range`) plus the SMP
/// poll loop. Both are leaf functions with small stack frames —
/// the AVX-512 kernel uses 32 stack-spill bytes at most. 64 KiB is
/// 1000× headroom and matches the BSP's stack class.
pub const AP_STACK_SIZE: usize = 64 * 1024;

/// Static per-AP stack arena. One stack per possible core (BSP +
/// up to MAX_CORES-1 APs). Allocated in `.bss`, zero-initialized,
/// page-aligned for clean MMU permissions.
///
/// **Why not arena-allocated dynamically?** The arena allocator is
/// available at AP boot, but using it for stacks introduces a
/// dependency from `arch::x86_64::trampoline` to `crate::memory`
/// that complicates the boot order. A static array is simpler,
/// has the right size, and costs `MAX_CORES * 64 KiB = 4 MiB` of
/// `.bss` — well within the kernel's memory budget.
#[repr(C, align(16))]
struct ApStack {
    bytes: [u8; AP_STACK_SIZE],
}

static mut AP_STACKS: [ApStack; MAX_CORES] = [const {
    ApStack {
        bytes: [0u8; AP_STACK_SIZE],
    }
}; MAX_CORES];

/// Return the top-of-stack virtual address for the AP at `core_idx`.
/// On x86_64 the stack grows down, so `top = base + AP_STACK_SIZE`.
fn ap_stack_top_for(core_idx: usize) -> u64 {
    debug_assert!(core_idx < MAX_CORES);
    // SAFETY: We're computing an address only; no read or write.
    unsafe {
        let base = &raw const AP_STACKS[core_idx].bytes as *const u8;
        (base as u64) + AP_STACK_SIZE as u64
    }
}

// ─────────────────────────────────────────────────────────────────
// Trampoline image bytes
// ─────────────────────────────────────────────────────────────────

/// Hand-assembled trampoline bytes.
///
/// The exact assembly source is documented inline. We encode the
/// instructions as raw bytes because Rust at `target_arch = x86_64`
/// only emits 64-bit code; 16-bit and 32-bit prologue can't be
/// expressed as a Rust function, so we hand-encode and use `core::
/// arch::asm!` blocks only for sanity-checking individual
/// instructions during development.
///
/// **Patch sites** (filled in by [`install_trampoline`]):
/// * `[PATCH_BOOTSTRAP_CR3]` — 8 bytes, low bootstrap CR3 value.
/// * `[PATCH_LME_ENTRY]` — 8 bytes, virtual address of
///   [`ap_long_mode_entry`].
/// * `[PATCH_STACK_SLOT]` — 8 bytes, atomic stack-handoff slot
///   address.
/// * `[PATCH_REAL_CR3]` — 8 bytes, BSP's real CR3 value.
/// * `[PATCH_REAL_CR4]` — 8 bytes, BSP's real CR4 value.
/// * `[PATCH_REAL_CR0]` — 8 bytes, BSP's real CR0 value.
///
/// We keep these patch offsets as constants instead of computing them
/// from the byte sequence at runtime — it makes the structure obvious
/// and lets us assert at boot that the trampoline image is well-formed.
///
/// **NB.** The byte sequence below follows the AMD64 long-mode
/// activation sequence: PAE page tables, EFER.LME|NXE, CR0.PG, and a
/// far jump into a 64-bit code segment.
#[repr(C, align(4096))]
struct TrampolineImage {
    bytes: [u8; TRAMPOLINE_LEN],
}

/// Length of the assembled trampoline image. 512 bytes covers the
/// 16-bit + 32-bit + 64-bit segments plus the GDT, GDTR pointer, and
/// patch slots; see the offset table in the module docs.
const TRAMPOLINE_LEN: usize = 512;

/// Offset of the 8-byte low bootstrap CR3 patch slot from `TRAMPOLINE_PHYS`.
const PATCH_BOOTSTRAP_CR3_OFFSET: usize = 0x1B8;
/// Offset of the 8-byte long-mode-entry patch slot.
const PATCH_LME_ENTRY_OFFSET: usize = 0x1B0;
/// Offset of the 8-byte stack-handoff slot.
const PATCH_STACK_SLOT_OFFSET: usize = 0x1C0;
/// Offset of the 8-byte real BSP CR3 patch slot.
const PATCH_REAL_CR3_OFFSET: usize = 0x1C8;
/// Offset of the 8-byte real BSP CR4 patch slot.
const PATCH_REAL_CR4_OFFSET: usize = 0x1D0;
/// Offset of the 8-byte real BSP CR0 patch slot.
const PATCH_REAL_CR0_OFFSET: usize = 0x1D8;
/// Offset of the GDTR pseudo-descriptor (6 bytes: limit u16 + base u32 or u64).
const TRAMPOLINE_GDTR_OFFSET: usize = 0x1A0;
/// Offset of the inline GDT (4 entries × 8 bytes).
const TRAMPOLINE_GDT_OFFSET: usize = 0x180;
const TRAMPOLINE_PROBE_MODE_OFFSET: usize = 0x1E0;
const TRAMPOLINE_PROBE_STAGE_OFFSET: usize = 0x1E1;

/// Source-form pseudo-assembly for the trampoline image. Reproduced
/// here so future maintainers can re-derive the byte sequence if the
/// patch slot offsets ever change.
///
/// ```text
/// ; -- 16-bit real mode at TRAMPOLINE_PHYS --
/// .code16
///     cli
///     cld
///     xor   ax, ax
///     mov   ds, ax
///     mov   es, ax
///     mov   ss, ax
///     probe_stage(real16) ; optional spin-loop if probe_mode == real16
///
///     ; Load trampoline GDT.
///     ; Selectors mirror the kernel GDT where it matters:
///     ;   0x08 = 64-bit kernel code (matches BSP-built IDT gates)
///     ;   0x10 = writable data
///     ;   0x18 = temporary 32-bit protected/compat code for transition
///     lgdt  [TRAMPOLINE_PHYS + TRAMPOLINE_GDTR_OFFSET]
///
///     ; CR0: enter protected mode first. Real-mode direct jumps into
///     ; IA-32e are brittle across firmware/CPU combinations; the AMD64
///     ; activation path is protected-mode first, then paging/LME.
///     mov   eax, cr0
///     or    eax, 0x00000001
///     mov   cr0, eax
///
///     ; Far jump to 32-bit protected-mode code segment.
///     jmp   0x18:protected_entry_32
///
/// .code32
/// protected_entry_32:
///     mov   ax, 0x10  ; data selector (not strictly needed in 64-bit, but tidy)
///     mov   ds, ax
///     mov   es, ax
///     mov   ss, ax
///     probe_stage(protected32)
///
///     ; Enable PAE (CR4 bit 5).
///     mov   eax, cr4
///     or    eax, (1 << 5)
///     mov   cr4, eax
///     probe_stage(pae)
///
///     ; Load CR3 with a low bootstrap PML4 (patched by install_trampoline).
///     mov   eax, [TRAMPOLINE_PHYS + PATCH_BOOTSTRAP_CR3_OFFSET]
///     mov   cr3, eax
///
///     ; Enable EFER.LME (long mode enable) and NXE via MSR 0xC0000080.
///     ; The BSP page tables may contain NX bits; APs must enable NXE
///     ; before loading the real CR3 or the first high-half fetch can
///     ; raise a reserved-bit page fault before an AP IDT exists.
///     mov   ecx, 0xC0000080
///     rdmsr
///     or    eax, (1 << 8) | (1 << 11)
///     wrmsr
///     probe_stage(efer)
///
///     ; CR0: paging on. With CR4.PAE and EFER.LME set, this activates
///     ; IA-32e mode; the far jump below loads the 64-bit CS descriptor.
///     probe_stage(before_paging)
///     mov   eax, cr0
///     or    eax, 0x80000000
///     mov   cr0, eax
///
///     jmp   0x08:long_mode_entry_64
///
/// .code64
/// long_mode_entry_64:
///     probe_stage(long64)
///     ; Load AP stack from handoff slot, mirror the BSP's control-
///     ; register baseline, switch to the real BSP CR3, then jump to Rust.
///     mov   rsp, [TRAMPOLINE_PHYS + PATCH_STACK_SLOT_OFFSET]
///     sub   rsp, 8   ; make the jumped-to Rust ABI look call-entered
///     mov   rax, [TRAMPOLINE_PHYS + PATCH_REAL_CR0_OFFSET]
///     mov   cr0, rax
///     mov   rax, [TRAMPOLINE_PHYS + PATCH_REAL_CR4_OFFSET]
///     mov   cr4, rax
///     probe_stage(before_real_cr3)
///     mov   rax, [TRAMPOLINE_PHYS + PATCH_REAL_CR3_OFFSET]
///     mov   cr3, rax
///     probe_stage(before_rust)
///     mov   rax, [TRAMPOLINE_PHYS + PATCH_LME_ENTRY_OFFSET]
///     jmp   rax
///
/// ; -- GDT and pseudo-descriptor at fixed offsets --
/// gdt_start:
///     dq 0x0000000000000000     ; 0x00: null
///     dq 0x00AF9A000000FFFF     ; 0x08: 64-bit code, L=1
///     dq 0x00CF92000000FFFF     ; 0x10: data, base=0, limit=4G
///     dq 0x00CF9A000000FFFF     ; 0x18: 32-bit compat code
/// gdtr_pseudo:
///     dw <limit = 31>
///     dq <base = TRAMPOLINE_PHYS + TRAMPOLINE_GDT_OFFSET>
/// patch_lme_entry: dq 0
/// patch_boot_cr3:   dq 0
/// patch_stack_slot: dq 0
/// patch_real_cr3:   dq 0
/// patch_real_cr4:   dq 0
/// patch_real_cr0:   dq 0
/// raw_probe_mode:   db 0
/// raw_probe_stage:  db 0
/// ```
///
/// The byte sequence below is the assembled output. Each row is
/// annotated with the corresponding assembly mnemonic. If you change
/// the assembly, **re-run** the assembler (e.g. `gas --32 ...`) and
/// update both the bytes and the patch offset constants.
const TRAMPOLINE_BYTES: TrampolineImage = TrampolineImage {
    bytes: {
        let mut bytes = [0u8; TRAMPOLINE_LEN];

        // ── 16-bit prologue at offset 0x00 ──
        // Total length: 48 bytes (0x30). The far jump at the tail
        // transfers control to byte 0x50 of the image, where the
        // 32-bit protected-mode entry lives. The probe marker writes
        // a one-byte raw stage to 0x81E1 and parks if 0x81E0 matches.
        let prologue: [u8; 48] = [
            // cli; cld
            0xFA, 0xFC, // xor ax,ax ; mov ds,ax ; mov es,ax ; mov ss,ax
            0x31, 0xC0, 0x8E, 0xD8, 0x8E, 0xC0, 0x8E, 0xD0,
            // raw-probe(real16): stage=10; if mode==10 then spin-loop
            0xC6, 0x06, 0xE1, 0x81, 0x0A, 0x80, 0x3E, 0xE0, 0x81, 0x0A, 0x75, 0x03, 0x90, 0xEB,
            0xFD, // lgdt fword ptr [0x81A0]  →  0x0F 0x01 0x16 0xA0 0x81
            0x0F, 0x01, 0x16, 0xA0, 0x81, // mov eax, cr0 ; or eax, 1 ; mov cr0, eax
            0x0F, 0x20, 0xC0, 0x66, 0x83, 0xC8, 0x01, 0x0F, 0x22, 0xC0,
            // Far jump to 0x18:0x8050.
            // Encoding: 0x66 0xEA <offset:u32 LE> <selector:u16 LE>.
            0x66, 0xEA, 0x50, 0x80, 0x00, 0x00, // offset = 0x8050
            0x18, 0x00, // selector = 0x18
        ];
        // Copy prologue.
        let mut i = 0;
        while i < prologue.len() {
            bytes[i] = prologue[i];
            i += 1;
        }

        // ── 32-bit segment entry at byte 0x50 (protected mode) ──
        // mov ax, 0x10 ; mov ds,ax ; mov es,ax ; mov ss,ax
        // raw probes: protected32, after PAE, after EFER, before paging.
        // mov eax,cr4 ; or eax,0x20 ; mov cr4,eax
        // mov eax,[0x81B8] ; mov cr3,eax
        // mov ecx,0xC0000080 ; rdmsr ; or eax,0x900 ; wrmsr
        // mov eax,cr0 ; or eax,0x80000000 ; mov cr0,eax
        // jmp 0x08:0x80E0    (transition to 64-bit code segment)
        let entry32: [u8; 135] = [
            0x66, 0xB8, 0x10, 0x00, // mov ax, 0x10
            0x8E, 0xD8, // mov ds, ax
            0x8E, 0xC0, // mov es, ax
            0x8E, 0xD0, // mov ss, ax
            // raw-probe(prot32): stage=11; if mode==11 then spin-loop
            0xC6, 0x05, 0xE1, 0x81, 0x00, 0x00, 0x0B, 0x80, 0x3D, 0xE0, 0x81, 0x00, 0x00, 0x0B,
            0x75, 0x03, 0x90, 0xEB, 0xFD, 0x0F, 0x20, 0xE0, // mov eax, cr4
            0x83, 0xC8, 0x20, // or eax, 0x20
            0x0F, 0x22, 0xE0, // mov cr4, eax
            // raw-probe(pae): stage=12; if mode==12 then spin-loop
            0xC6, 0x05, 0xE1, 0x81, 0x00, 0x00, 0x0C, 0x80, 0x3D, 0xE0, 0x81, 0x00, 0x00, 0x0C,
            0x75, 0x03, 0x90, 0xEB, 0xFD, 0xA1, 0xB8, 0x81, 0x00, 0x00, // mov eax, [0x81B8]
            0x0F, 0x22, 0xD8, // mov cr3, eax
            0xB9, 0x80, 0x00, 0x00, 0xC0, // mov ecx, 0xC0000080
            0x0F, 0x32, // rdmsr
            0x0D, 0x00, 0x09, 0x00, 0x00, // or eax, 0x900 (LME | NXE)
            0x0F, 0x30, // wrmsr
            // raw-probe(efer): stage=13; if mode==13 then spin-loop
            0xC6, 0x05, 0xE1, 0x81, 0x00, 0x00, 0x0D, 0x80, 0x3D, 0xE0, 0x81, 0x00, 0x00, 0x0D,
            0x75, 0x03, 0x90, 0xEB, 0xFD,
            // raw-probe(paging): stage=14; if mode==14 then spin-loop
            0xC6, 0x05, 0xE1, 0x81, 0x00, 0x00, 0x0E, 0x80, 0x3D, 0xE0, 0x81, 0x00, 0x00, 0x0E,
            0x75, 0x03, 0x90, 0xEB, 0xFD, 0x0F, 0x20, 0xC0, // mov eax, cr0
            0x0D, 0x00, 0x00, 0x00, 0x80, // or eax, 0x80000000
            0x0F, 0x22, 0xC0, // mov cr0, eax
            // Far jmp 0x08:0x80E0  →  encoding 0xEA imm32 imm16
            0xEA, 0xE0, 0x80, 0x00, 0x00, // offset = 0x80E0
            0x08, 0x00, // selector = 0x08
        ];
        let mut j = 0;
        while j < entry32.len() {
            bytes[0x50 + j] = entry32[j];
            j += 1;
        }

        // ── 64-bit entry at byte offset 0xE0 ──
        // raw probes: long64, before real CR3 switch, before Rust jump.
        // mov rsp, [0x81C0]         (qword pointer to per-AP stack top)
        // sub rsp, 8                (SysV/Rust call-entry stack shape)
        // mov rax, [0x81D8]         (qword real BSP CR0)
        // mov cr0, rax             (mirror BSP control baseline)
        // mov rax, [0x81D0]         (qword real BSP CR4)
        // mov cr4, rax             (mirror BSP page-table feature bits)
        // mov rax, [0x81C8]         (qword real BSP CR3)
        // mov cr3, rax             (switch to shared kernel tables)
        // mov rax, [0x81B0]         (qword pointer to ap_long_mode_entry)
        // jmp rax
        let entry64: [u8; 121] = [
            // raw-probe(long64): stage=15; if mode==15 then spin-loop
            0xC6, 0x04, 0x25, 0xE1, 0x81, 0x00, 0x00, 0x0F, 0x80, 0x3C, 0x25, 0xE0, 0x81, 0x00,
            0x00, 0x0F, 0x75, 0x03, 0x90, 0xEB, 0xFD, 0x48, 0x8B, 0x24, 0x25, 0xC0, 0x81, 0x00,
            0x00, 0x48, 0x83, 0xEC, 0x08, 0x48, 0x8B, 0x04, 0x25, 0xD8, 0x81, 0x00, 0x00, 0x48,
            0x0F, 0x22, 0xC0, 0x48, 0x8B, 0x04, 0x25, 0xD0, 0x81, 0x00, 0x00, 0x48, 0x0F, 0x22,
            0xE0, // raw-probe(cr3): stage=16; if mode==16 then spin-loop
            0xC6, 0x04, 0x25, 0xE1, 0x81, 0x00, 0x00, 0x10, 0x80, 0x3C, 0x25, 0xE0, 0x81, 0x00,
            0x00, 0x10, 0x75, 0x03, 0x90, 0xEB, 0xFD, 0x48, 0x8B, 0x04, 0x25, 0xC8, 0x81, 0x00,
            0x00, 0x48, 0x0F, 0x22, 0xD8,
            // raw-probe(rust): stage=17; if mode==17 then spin-loop
            0xC6, 0x04, 0x25, 0xE1, 0x81, 0x00, 0x00, 0x11, 0x80, 0x3C, 0x25, 0xE0, 0x81, 0x00,
            0x00, 0x11, 0x75, 0x03, 0x90, 0xEB, 0xFD, 0x48, 0x8B, 0x04, 0x25, 0xB0, 0x81, 0x00,
            0x00, 0xFF, 0xE0,
        ];
        let mut k = 0;
        while k < entry64.len() {
            bytes[0xE0 + k] = entry64[k];
            k += 1;
        }

        // ── GDT at TRAMPOLINE_GDT_OFFSET (0xB0) ──
        // 4 entries × 8 bytes = 32 bytes (we declare 0x18 limit which
        // covers selectors 0x00..0x1F).
        // Entries:
        //   0x00 null:           0x00 00 00 00 00 00 00 00
        //   0x08 64-bit code:    0xFF FF 00 00 00 9A AF 00   (L=1 in flags)
        //   0x10 data:           0xFF FF 00 00 00 92 CF 00
        //   0x18 32-bit code:    0xFF FF 00 00 00 9A CF 00
        let gdt: [u8; 32] = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x9A,
            0xAF, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x92, 0xCF, 0x00, 0xFF, 0xFF, 0x00, 0x00,
            0x00, 0x9A, 0xCF, 0x00,
        ];
        let mut g = 0;
        while g < gdt.len() {
            bytes[TRAMPOLINE_GDT_OFFSET + g] = gdt[g];
            g += 1;
        }

        // ── GDTR pseudo-descriptor at 0x1A0 ──
        // 6 bytes: limit u16 (= sizeof(gdt) - 1 = 0x1F), base u32 = 0x8180.
        bytes[TRAMPOLINE_GDTR_OFFSET + 0] = 0x1F;
        bytes[TRAMPOLINE_GDTR_OFFSET + 1] = 0x00;
        bytes[TRAMPOLINE_GDTR_OFFSET + 2] =
            (TRAMPOLINE_PHYS as u32 + TRAMPOLINE_GDT_OFFSET as u32) as u8;
        bytes[TRAMPOLINE_GDTR_OFFSET + 3] =
            ((TRAMPOLINE_PHYS as u32 + TRAMPOLINE_GDT_OFFSET as u32) >> 8) as u8;
        bytes[TRAMPOLINE_GDTR_OFFSET + 4] =
            ((TRAMPOLINE_PHYS as u32 + TRAMPOLINE_GDT_OFFSET as u32) >> 16) as u8;
        bytes[TRAMPOLINE_GDTR_OFFSET + 5] =
            ((TRAMPOLINE_PHYS as u32 + TRAMPOLINE_GDT_OFFSET as u32) >> 24) as u8;

        // The PATCH slots at 0x1B0 (entry), 0x1B8 (bootstrap CR3),
        // 0x1C0 (stack), 0x1C8 (real CR3), 0x1D0 (real CR4), and
        // 0x1D8 (real CR0) remain zero until `install_trampoline`
        // overwrites them.

        bytes
    },
};

// ─────────────────────────────────────────────────────────────────
// Trampoline installation
// ─────────────────────────────────────────────────────────────────

/// Stack-handoff slot for the current AP about to wake. The BSP
/// writes the per-AP stack top here, fires the SIPI, waits for the
/// AP to register, then writes the next AP's stack top. Sequential
/// AP boots are serialized at the BSP level.
static AP_STACK_HANDOFF: AtomicU64 = AtomicU64::new(0);

/// Track which logical-core slot will receive the next AP. Used by
/// [`prepare_next_ap_stack`] to walk through `AP_STACKS[1..]` in
/// registration order.
static NEXT_AP_SLOT: AtomicUsize = AtomicUsize::new(1);

static TRAMPOLINE_INSTALLED: AtomicBool = AtomicBool::new(false);
static LAST_BOOTSTRAP_CR3: AtomicU64 = AtomicU64::new(0);
static LAST_REAL_CR3: AtomicU64 = AtomicU64::new(0);
static LAST_REAL_CR4: AtomicU64 = AtomicU64::new(0);
static LAST_REAL_CR0: AtomicU64 = AtomicU64::new(0);

/// Read-only trampoline diagnostics exposed through the TCP shell.
#[derive(Copy, Clone, Debug, Default)]
pub struct TrampolineStatus {
    pub installed: bool,
    pub phys: u64,
    pub sipi_vector: u8,
    pub bootstrap_cr3: u64,
    pub real_cr3: u64,
    pub real_cr4: u64,
    pub real_cr0: u64,
    pub raw_probe_mode: u8,
    pub raw_probe_stage: u8,
}

pub fn status() -> TrampolineStatus {
    let installed = TRAMPOLINE_INSTALLED.load(Ordering::Acquire);
    TrampolineStatus {
        installed,
        phys: TRAMPOLINE_PHYS,
        sipi_vector: TRAMPOLINE_SIPI_VECTOR,
        bootstrap_cr3: LAST_BOOTSTRAP_CR3.load(Ordering::Acquire),
        real_cr3: LAST_REAL_CR3.load(Ordering::Acquire),
        real_cr4: LAST_REAL_CR4.load(Ordering::Acquire),
        real_cr0: LAST_REAL_CR0.load(Ordering::Acquire),
        raw_probe_mode: if installed {
            unsafe { read_probe_byte(TRAMPOLINE_PROBE_MODE_OFFSET) }
        } else {
            0
        },
        raw_probe_stage: if installed {
            unsafe { read_probe_byte(TRAMPOLINE_PROBE_STAGE_OFFSET) }
        } else {
            0
        },
    }
}

#[inline]
unsafe fn read_probe_byte(offset: usize) -> u8 {
    core::ptr::read_volatile((TRAMPOLINE_PHYS + offset as u64) as *const u8)
}

#[inline]
unsafe fn write_probe_byte(offset: usize, value: u8) {
    core::ptr::write_volatile((TRAMPOLINE_PHYS + offset as u64) as *mut u8, value);
}

unsafe fn sync_raw_probe_control() {
    let mode = smp::ap_boot_mode().min(u8::MAX as u32) as u8;
    write_probe_byte(TRAMPOLINE_PROBE_MODE_OFFSET, mode);
    write_probe_byte(
        TRAMPOLINE_PROBE_STAGE_OFFSET,
        smp::AP_PROBE_STAGE_IDLE as u8,
    );
    core::sync::atomic::fence(Ordering::SeqCst);
}

const PAGE_TABLE_PRESENT: u64 = 1 << 0;
const PAGE_TABLE_WRITABLE: u64 = 1 << 1;
const PAGE_TABLE_HUGE: u64 = 1 << 7;
const LOW_BOOTSTRAP_LIMIT: u64 = 0x1_0000_0000;

/// Build the minimal low CR3 root used only for the AP's transition
/// into long mode.
///
/// The AP starts in real mode and can only load the initial CR3 through
/// the 32-bit transition path. On a large EPYC system the BSP's real
/// PML4 can live above 4 GiB, so the trampoline first uses this tiny
/// identity map:
///
/// * PML4[0] -> PDPT[0]
/// * PDPT[0] -> PD[0]
/// * PD[0] -> 0..2 MiB as a 2 MiB page
///
/// Once the AP reaches 64-bit mode it loads the full BSP CR3 and jumps
/// into the shared higher-half kernel.
fn ensure_low_identity_page(phys: u64) -> Result<(), &'static str> {
    if (phys & 0xfff) != 0 {
        return Err("low identity page is not page-aligned");
    }
    crate::memory::ensure_identity_mapped_4k(phys)
        .map(|_| ())
        .map_err(|_| "low identity page map failed")
}

unsafe fn build_low_bootstrap_page_tables() -> Result<u64, &'static str> {
    let [pml4_phys, pdpt_phys, pd_phys] =
        crate::memory::low_bootstrap_frames().ok_or("low bootstrap frames unavailable")?;

    let combined = pml4_phys | pdpt_phys | pd_phys;
    if (combined & 0xfff) != 0 {
        return Err("low bootstrap frame is not page-aligned");
    }
    if pml4_phys >= LOW_BOOTSTRAP_LIMIT
        || pdpt_phys >= LOW_BOOTSTRAP_LIMIT
        || pd_phys >= LOW_BOOTSTRAP_LIMIT
    {
        return Err("low bootstrap frame above 4 GiB");
    }

    // Touch the low bootstrap tables through explicit VA==PA mappings.
    // The AP's first CR3 is a low physical PML4; using the same identity
    // view on the BSP avoids relying on the bootloader's higher-half
    // physical map for these early SMP pages.
    ensure_low_identity_page(pml4_phys)?;
    ensure_low_identity_page(pdpt_phys)?;
    ensure_low_identity_page(pd_phys)?;

    let pml4 = pml4_phys as *mut u64;
    let pdpt = pdpt_phys as *mut u64;
    let pd = pd_phys as *mut u64;

    let mut i = 0usize;
    while i < 512 {
        core::ptr::write_volatile(pml4.add(i), 0);
        core::ptr::write_volatile(pdpt.add(i), 0);
        core::ptr::write_volatile(pd.add(i), 0);
        i += 1;
    }

    let flags = PAGE_TABLE_PRESENT | PAGE_TABLE_WRITABLE;
    core::ptr::write_volatile(pml4.add(0), pdpt_phys | flags);
    core::ptr::write_volatile(pdpt.add(0), pd_phys | flags);
    core::ptr::write_volatile(pd.add(0), flags | PAGE_TABLE_HUGE);
    core::sync::atomic::compiler_fence(Ordering::SeqCst);

    Ok(pml4_phys)
}

/// Copy [`TRAMPOLINE_BYTES`] to physical [`TRAMPOLINE_PHYS`] and
/// patch in the run-time values (bootstrap CR3, entry RIP, stack slot,
/// and real BSP CR3).
///
/// Must be called once on the BSP after:
/// * memory init (so we have `PHYS_OFFSET`).
/// * the LAPIC has been initialized (so we know who we are).
///
/// And before any `Apic::boot_ap` call.
///
/// # Safety
///
/// Caller must ensure:
/// * The page at [`TRAMPOLINE_PHYS`] is identity-mapped (so AP CPUs
///   in real mode can fetch instructions from it). On `bootloader_api`
///   the first 1 MiB is identity-mapped through the linear physical
///   mapping; we just need to ensure the page isn't write-protected.
///   See [`ensure_trampoline_identity_mapped`].
/// * The active page tables may install VA==PA mappings for low SMP
///   bootstrap pages.
pub unsafe fn install_trampoline(_phys_offset: u64) -> Result<(), &'static str> {
    ensure_low_identity_page(TRAMPOLINE_PHYS)?;
    let bootstrap_cr3 = build_low_bootstrap_page_tables()?;
    let real_cr3 = read_cr3_value();
    let real_cr4 = read_cr4_value();
    let real_cr0 = read_cr0_value();
    let dst = TRAMPOLINE_PHYS;
    let src = TRAMPOLINE_BYTES.bytes.as_ptr();

    // Step 1: blit the image.
    core::ptr::copy_nonoverlapping(src, dst as *mut u8, TRAMPOLINE_LEN);

    // Step 2: patch bootstrap CR3 (low PML4 root used before the AP
    // can load a 64-bit control-register value).
    let bootstrap_cr3_slot = (dst + PATCH_BOOTSTRAP_CR3_OFFSET as u64) as *mut u64;
    core::ptr::write_volatile(bootstrap_cr3_slot, bootstrap_cr3);

    // Step 3: patch long-mode entry (Rust function pointer).
    // `as *const ()` first is the recommended idiom for fn-item-to-int
    // casts under recent rustc (function_casts_as_integer lint).
    let entry = ap_long_mode_entry_trampoline_target as *const () as u64;
    let lme_slot = (dst + PATCH_LME_ENTRY_OFFSET as u64) as *mut u64;
    core::ptr::write_volatile(lme_slot, entry);

    // Step 4: zero the stack slot — the BSP populates it per-AP.
    let stack_slot = (dst + PATCH_STACK_SLOT_OFFSET as u64) as *mut u64;
    core::ptr::write_volatile(stack_slot, 0);

    // Step 5: patch real CR3/CR4/CR0. The AP loads these only after
    // entering 64-bit mode, so high physical PML4 roots and BSP core-
    // control/page-table feature bits are preserved.
    let real_cr3_slot = (dst + PATCH_REAL_CR3_OFFSET as u64) as *mut u64;
    core::ptr::write_volatile(real_cr3_slot, real_cr3);
    let real_cr4_slot = (dst + PATCH_REAL_CR4_OFFSET as u64) as *mut u64;
    core::ptr::write_volatile(real_cr4_slot, real_cr4);
    let real_cr0_slot = (dst + PATCH_REAL_CR0_OFFSET as u64) as *mut u64;
    core::ptr::write_volatile(real_cr0_slot, real_cr0);
    write_probe_byte(TRAMPOLINE_PROBE_MODE_OFFSET, smp::AP_BOOT_MODE_FULL as u8);
    write_probe_byte(
        TRAMPOLINE_PROBE_STAGE_OFFSET,
        smp::AP_PROBE_STAGE_IDLE as u8,
    );
    core::sync::atomic::fence(Ordering::SeqCst);
    LAST_BOOTSTRAP_CR3.store(bootstrap_cr3, Ordering::Release);
    LAST_REAL_CR3.store(real_cr3, Ordering::Release);
    LAST_REAL_CR4.store(real_cr4, Ordering::Release);
    LAST_REAL_CR0.store(real_cr0, Ordering::Release);
    TRAMPOLINE_INSTALLED.store(true, Ordering::Release);

    Ok(())
}

/// Write the stack-top for the next AP that the BSP will wake. Called
/// just before each `Apic::boot_ap` so the AP's first long-mode
/// instruction loads a valid `RSP`.
///
/// `core_idx` is the *logical* core index the AP will claim once it
/// reaches Rust. The mapping APIC-ID → logical-index is established
/// later in [`ap_long_mode_entry`] via `smp::allocate_core_index`,
/// but we pre-assign the stack slot here so each AP gets an
/// independent stack region.
///
/// # Safety
/// Caller ensures `core_idx < MAX_CORES` and the trampoline page is
/// identity-mapped in the active BSP page tables.
pub unsafe fn prepare_next_ap_stack(_phys_offset: u64, core_idx: usize) {
    let stack_top = ap_stack_top_for(core_idx);
    AP_STACK_HANDOFF.store(stack_top, Ordering::Release);
    sync_raw_probe_control();

    let slot = (TRAMPOLINE_PHYS + PATCH_STACK_SLOT_OFFSET as u64) as *mut u64;
    core::ptr::write_volatile(slot, stack_top);
    // x2APIC ICR writes are not a reliable memory-serialization point.
    // Drain the stack-handoff store before the BSP sends SIPI, otherwise
    // a fast AP can observe the old zero slot and fault before Rust.
    core::sync::atomic::fence(Ordering::SeqCst);
}

/// Atomically claim and return the next AP slot index. Used by
/// [`ap_long_mode_entry`] to know which `AP_STACKS[i]` it's running
/// on. Returns `None` if exhausted.
fn claim_next_ap_slot() -> Option<usize> {
    let idx = NEXT_AP_SLOT.fetch_add(1, Ordering::AcqRel);
    if idx >= MAX_CORES {
        None
    } else {
        Some(idx)
    }
}

// ─────────────────────────────────────────────────────────────────
// Long-mode entry (called by trampoline once in 64-bit)
// ─────────────────────────────────────────────────────────────────

/// First-touch Rust function on the AP. Executes in 64-bit mode with:
/// * BSP's page tables (CR3 already loaded by trampoline).
/// * AP's own stack (RSP loaded from handoff slot).
/// * No TLS, no per-CPU GS-base; IDT is loaded immediately below.
///
/// Bootstraps the AP into the SMP worker loop. Does not return.
///
/// **`extern "C"`** so the symbol has a stable ABI if this entry later
/// grows arguments. The current trampoline jumps to it directly.
#[no_mangle]
pub extern "C" fn ap_long_mode_entry_trampoline_target() -> ! {
    // APs return from INIT with CPU-local architectural state reset.
    // Install exception diagnostics first, then re-enable the APIC mode
    // chosen by the BSP and the SIMD/XSTATE bits required by workers.
    let boot_mode = smp::ap_boot_mode();
    if boot_mode == smp::AP_BOOT_MODE_PROBE_ENTRY {
        smp::ap_probe_mark(smp::AP_PROBE_STAGE_ENTRY);
        park_forever();
    }

    interrupts::init_ap_exception_table();
    if boot_mode == smp::AP_BOOT_MODE_PROBE_IDT {
        smp::ap_probe_mark(smp::AP_PROBE_STAGE_IDT);
        park_forever();
    }

    unsafe {
        apic::enable_current_core();
    }
    if boot_mode == smp::AP_BOOT_MODE_PROBE_APIC {
        smp::ap_probe_mark(smp::AP_PROBE_STAGE_APIC);
        park_forever();
    }

    unsafe {
        let _ = cpuinfo::enable_fpu_simd();
    }
    if boot_mode == smp::AP_BOOT_MODE_PROBE_SIMD {
        smp::ap_probe_mark(smp::AP_PROBE_STAGE_SIMD);
        park_forever();
    }

    // ── Step 1: claim a logical core index ──
    // The APIC ID we read here is per-CPU (the LAPIC routes reads to
    // the running core); the index is reserved by us, not by the BSP.
    let apic_id = apic::current_apic_id();
    let topology = cpuinfo::topology_ids();
    let core_idx = match if topology.valid {
        smp::allocate_core_index_with_topology(apic_id, topology.node_id, topology.compute_unit_id)
    } else {
        smp::allocate_core_index(apic_id)
    } {
        Some(i) => i,
        None => {
            // Out of slots — park forever. This only fires if MADT
            // reports more CPUs than MAX_CORES.
            park_forever();
        }
    };

    // ── Step 2: register with SMP layer ──
    let _ = smp::ap_register();

    // ── Step 3: enter the worker loop ──
    // Pillar 7: shared cross-platform function. No platform-specific
    // logic from here on; AP behaviour is the same as it would be on
    // aarch64 if we ever bring up SMP there.
    smp::ap_worker_loop(core_idx);
}

/// Tight halt for APs that fail to register. Used as a fail-safe
/// only; the BSP will time out on its registration wait and continue
/// in degraded (BSP-only) mode if any AP fails to come up.
#[inline]
fn park_forever() -> ! {
    loop {
        // `hlt` is the right instruction here (wait for next IRQ);
        // these APs were never expected to do anything useful.
        unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
    }
}

// ─────────────────────────────────────────────────────────────────
// CR3 helper
// ─────────────────────────────────────────────────────────────────

/// Read the current CR3 value. Used at trampoline install time to
/// hand the AP the BSP's page tables.
#[inline]
fn read_cr3_value() -> u64 {
    let cr3: u64;
    unsafe {
        core::arch::asm!(
            "mov {}, cr3",
            out(reg) cr3,
            options(nomem, nostack, preserves_flags),
        );
    }
    // Hand APs the physical page-table root, not a PCID-tagged CR3 value.
    // AP trampoline code enables the architectural minimum CR4/EFER bits
    // before loading this value; preserving low PCID/PWT/PCD bits here can
    // turn the AP's first real-CR3 load into #GP before an IDT exists.
    cr3 & !0xfff
}

#[inline]
fn read_cr0_value() -> u64 {
    let cr0: u64;
    unsafe {
        core::arch::asm!(
            "mov {}, cr0",
            out(reg) cr0,
            options(nomem, nostack, preserves_flags),
        );
    }
    cr0
}

#[inline]
fn read_cr4_value() -> u64 {
    let cr4: u64;
    unsafe {
        core::arch::asm!(
            "mov {}, cr4",
            out(reg) cr4,
            options(nomem, nostack, preserves_flags),
        );
    }
    cr4
}

// ─────────────────────────────────────────────────────────────────
// Identity-mapping helper
// ─────────────────────────────────────────────────────────────────

/// Ensure the trampoline page at [`TRAMPOLINE_PHYS`] is identity-
/// mapped in the BSP's page tables (so the AP can execute it before
/// far-jumping out of real mode).
///
/// `bootloader_api` 0.11 places the kernel in the higher half and
/// keeps the lower 1 MiB unmapped *except* via the linear physical
/// mapping at `phys_offset`. The AP, however, fetches instructions
/// using its physical address `0x8000` directly — it has no concept
/// of the higher-half mapping until it's in long mode with the
/// BSP's CR3.
///
/// This function walks the active page tables to see if VA == PA
/// already holds for the trampoline page; if so it logs "verified"
/// and probes the mapping with a single volatile byte read to confirm
/// the descriptor really resolves. If the mapping is absent it
/// installs a fresh PRESENT|WRITABLE 4 KiB identity page via the
/// kernel's persistent frame allocator and logs "installed". If a
/// conflicting mapping exists (VA mapped, but to a different PA),
/// the function returns `Err` and the caller falls back to BSP-only
/// SMP.
pub fn ensure_trampoline_identity_mapped() -> Result<(), ()> {
    use crate::arch::serial::Serial;
    use crate::memory::{ensure_identity_mapped_4k, IdentityMapOutcome};
    use core::fmt::Write;

    match ensure_identity_mapped_4k(TRAMPOLINE_PHYS) {
        Ok(IdentityMapOutcome::Verified) => {
            // SAFETY: translate_addr returned VA == PA, so the page is
            // PRESENT in the active page tables. A single-byte volatile
            // read confirms the descriptor resolves end-to-end and
            // surfaces any latent walker bug as a #PF rather than as
            // silent corruption at AP wake-up time.
            unsafe {
                let probe = core::ptr::read_volatile(TRAMPOLINE_PHYS as *const u8);
                core::hint::black_box(probe);
            }
            let _ = writeln!(
                Serial,
                "ADR-029 P3: trampoline identity map at 0x{:x}: verified",
                TRAMPOLINE_PHYS,
            );
            Ok(())
        }
        Ok(IdentityMapOutcome::Installed) => {
            // Same probe-read after install — proves the mapping we
            // just wrote actually resolves (catches stale TLB / bad
            // intermediate-table issues immediately rather than at the
            // first AP's INIT-SIPI-SIPI).
            unsafe {
                let probe = core::ptr::read_volatile(TRAMPOLINE_PHYS as *const u8);
                core::hint::black_box(probe);
            }
            let _ = writeln!(
                Serial,
                "ADR-029 P3: trampoline identity map at 0x{:x}: installed",
                TRAMPOLINE_PHYS,
            );
            Ok(())
        }
        Err(e) => {
            let _ = writeln!(
                Serial,
                "ADR-029 P3: trampoline identity map at 0x{:x}: FAILED ({:?})",
                TRAMPOLINE_PHYS, e,
            );
            Err(())
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Self-check
// ─────────────────────────────────────────────────────────────────

/// Sanity-check that the trampoline image is well-formed. Run by the
/// BSP before installing — catches off-by-one errors when the byte
/// table is edited.
pub fn trampoline_self_check() -> Result<(), &'static str> {
    // Patch slots at expected offsets should be zero before install.
    let img = &TRAMPOLINE_BYTES.bytes;
    if img[0x2e] != 0x18 || img[0x2f] != 0x00 {
        return Err("real-mode far jump must target 32-bit protected selector 0x18");
    }
    if img[0x89] != 0xa1 || img[0x8a] != 0xb8 || img[0x8b] != 0x81 {
        return Err("protected-mode entry must load bootstrap CR3 from patch slot");
    }
    if img[0xd5] != 0x08 || img[0xd6] != 0x00 {
        return Err("protected-mode far jump must target 64-bit kernel selector 0x08");
    }
    if img[0x98] != 0x0d || read_dword_le(img, 0x99) != 0x900 {
        return Err("protected-mode EFER setup must enable LME|NXE");
    }
    let cr3_slot = read_qword_le(img, PATCH_BOOTSTRAP_CR3_OFFSET);
    if cr3_slot != 0 {
        return Err("bootstrap CR3 patch slot non-zero in source image");
    }
    let lme_slot = read_qword_le(img, PATCH_LME_ENTRY_OFFSET);
    if lme_slot != 0 {
        return Err("LME-entry patch slot non-zero in source image");
    }
    let stk_slot = read_qword_le(img, PATCH_STACK_SLOT_OFFSET);
    if stk_slot != 0 {
        return Err("stack patch slot non-zero in source image");
    }
    let real_cr3_slot = read_qword_le(img, PATCH_REAL_CR3_OFFSET);
    if real_cr3_slot != 0 {
        return Err("real CR3 patch slot non-zero in source image");
    }
    let real_cr4_slot = read_qword_le(img, PATCH_REAL_CR4_OFFSET);
    if real_cr4_slot != 0 {
        return Err("real CR4 patch slot non-zero in source image");
    }
    let real_cr0_slot = read_qword_le(img, PATCH_REAL_CR0_OFFSET);
    if real_cr0_slot != 0 {
        return Err("real CR0 patch slot non-zero in source image");
    }
    if img[TRAMPOLINE_PROBE_MODE_OFFSET] != 0 || img[TRAMPOLINE_PROBE_STAGE_OFFSET] != 0 {
        return Err("raw probe control bytes non-zero in source image");
    }
    // GDTR base.
    let gdtr_lo = img[TRAMPOLINE_GDTR_OFFSET + 2] as u32
        | ((img[TRAMPOLINE_GDTR_OFFSET + 3] as u32) << 8)
        | ((img[TRAMPOLINE_GDTR_OFFSET + 4] as u32) << 16)
        | ((img[TRAMPOLINE_GDTR_OFFSET + 5] as u32) << 24);
    let expected = (TRAMPOLINE_PHYS as u32) + TRAMPOLINE_GDT_OFFSET as u32;
    if gdtr_lo != expected {
        return Err("GDTR base does not match expected GDT offset");
    }
    Ok(())
}

#[inline]
fn read_qword_le(buf: &[u8], off: usize) -> u64 {
    let mut v = 0u64;
    let mut i = 0usize;
    while i < 8 {
        v |= (buf[off + i] as u64) << (i * 8);
        i += 1;
    }
    v
}

#[inline]
fn read_dword_le(buf: &[u8], off: usize) -> u32 {
    (buf[off] as u32)
        | ((buf[off + 1] as u32) << 8)
        | ((buf[off + 2] as u32) << 16)
        | ((buf[off + 3] as u32) << 24)
}

// ─────────────────────────────────────────────────────────────────
// High-level BSP-side boot-all-APs helper
// ─────────────────────────────────────────────────────────────────

/// Drive INIT-SIPI-SIPI for every AP in the `cpus` list that is not
/// the BSP. After all APs have signaled ready, returns the final
/// active-core count.
///
/// `cpus` is the `AcpiInfo::cpus[0..cpu_count]` array. `bsp_apic_id`
/// is the BSP's own APIC ID (so we skip it). `apic` is the
/// initialized BSP-side LAPIC driver.
/// `max_aps` caps how many APs are attempted; `u32::MAX` means all
/// eligible MADT CPUs.
///
/// **Wait policy.** After each AP boot we wait up to ~100 ms for the
/// AP to call [`smp::ap_register`]. If it doesn't show up, we move on
/// — the AP is presumed wedged and is not counted as active. This
/// produces a graceful degradation rather than a kernel hang on
/// flaky hardware.
///
/// # Safety
/// `phys_offset` must be the active linear physical-memory mapping.
pub unsafe fn boot_all_aps(
    cpus: &[crate::arch::x86_64::acpi::CpuInfo],
    bsp_apic_id: u32,
    apic: &super::apic::Apic,
    phys_offset: u64,
    max_aps: u32,
) -> u32 {
    use crate::arch::serial::Serial;
    use core::fmt::Write;

    let mut booted: u32 = 0;
    let mut failed: u32 = 0;

    // Count how many APs we will actually attempt to wake (skipping
    // the BSP and any firmware-disabled entries). Logged up-front so a
    // long stretch of silence during the boot loop is observable.
    let total_eligible_aps: u32 = cpus
        .iter()
        .filter(|c| c.enabled && c.apic_id != bsp_apic_id && apic.can_address(c.apic_id))
        .count() as u32;
    let eligible_aps = total_eligible_aps.min(max_aps);
    let _ = writeln!(
        Serial,
        "ADR-029 P3: waking {} AP(s) via INIT-SIPI-SIPI ({} eligible, BSP apic_id={}, mode={})",
        eligible_aps,
        total_eligible_aps,
        bsp_apic_id,
        apic.mode_label()
    );

    let tsc_hz = super::cycles::tsc_hz();
    let budget_cycles = tsc_hz / 10; // 100 ms

    for cpu in cpus.iter() {
        if booted + failed >= eligible_aps {
            break;
        }
        if !cpu.enabled || cpu.apic_id == bsp_apic_id {
            continue;
        }
        if !apic.can_address(cpu.apic_id) {
            let _ = writeln!(
                Serial,
                "ADR-029 P3: AP apic_id={} cannot be addressed in {} mode — skipped",
                cpu.apic_id,
                apic.mode_label()
            );
            continue;
        }
        // Assign a logical slot and write the per-AP stack handoff.
        let slot = match claim_next_ap_slot() {
            Some(s) => s,
            None => break, // no more slots
        };
        prepare_next_ap_stack(phys_offset, slot);

        // Wait up to ~100 ms for this AP to register. Timed against
        // `rdtsc` (not the legacy PIT) because the 8259 is masked
        // under UEFI on EPYC and `pit::ticks()` would never advance.
        let before = smp::registered_cores();
        let expected = before.saturating_add(1);

        // Fire INIT-SIPI-SIPI.
        apic.boot_ap(cpu.apic_id, TRAMPOLINE_SIPI_VECTOR);

        let start_tsc = super::cycles::rdtsc_serialized();
        while smp::registered_cores() < expected
            && super::cycles::rdtsc_serialized().wrapping_sub(start_tsc) < budget_cycles
        {
            core::hint::spin_loop();
        }
        if smp::registered_cores() >= expected {
            booted += 1;
        } else {
            // AP didn't register within the registration window; leave
            // its stack slot allocated (zeroed) and continue. The
            // active-core count reflects what actually showed up.
            // NEXT_AP_SLOT was already incremented and we don't roll
            // back — deterministic slot-to-physical-stack mapping is
            // preserved across failed APs.
            failed += 1;
            let raw_stage = read_probe_byte(TRAMPOLINE_PROBE_STAGE_OFFSET);
            let _ = writeln!(
                Serial,
                "ADR-029 P3: AP apic_id={} (slot {}) did not register within 100 ms — skipped (raw_stage={})",
                cpu.apic_id, slot, raw_stage
            );
        }
    }

    let _ = writeln!(
        Serial,
        "ADR-029 P3: AP wake-up loop complete — booted={} failed={} of {} eligible",
        booted, failed, eligible_aps
    );

    // BSP itself counts as one core; total active = 1 + booted.
    1 + booted
}
