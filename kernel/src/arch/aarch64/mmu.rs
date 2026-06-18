// SPDX-License-Identifier: AGPL-3.0-or-later
//! AArch64 MMU setup — translation tables for TTBR0_EL1 + TTBR1_EL1.
//!
//! Per ARM ARM D5: VMSAv8-64 translation system.
//! - 4 KiB granule, 4-level translation, 48-bit VA
//! - TTBR0_EL1: low-half identity map (kernel runs at physical addresses)
//! - TTBR1_EL1: high-half (reserved for Stage 7+)
//!
//! Stage 5: Build tables, configure MAIR/TCR, set TTBR registers.
//!          MMU stays OFF (SCTLR_EL1.M = 0).
//! Stage 6: Cache/TLB invalidate, barriers, SCTLR_EL1.M = 1.
//!
//! Identity map layout (TTBR0_EL1):
//!   0x0000_0000..0x4000_0000 (1 GiB): via L2, Device MMIO + invalid
//!     GIC dist  (0x800_0000): Device-nGnRE
//!     GIC cpu   (0x801_0000): Device-nGnRE (same 2 MiB block)
//!     PL011     (0x900_0000): Device-nGnRE
//!   0x4000_0000..0x8000_0000 (1 GiB): Normal cacheable (kernel + RAM)
//!   0x8000_0000..0xC000_0000 (1 GiB): Normal cacheable (initrd + DTB)
//!
//! CITE: ARM ARM D5.2 (translation system)
//! CITE: ARM ARM D5.3 (descriptor formats)
//! CITE: ARM ARM D5.5 (memory attributes)
//! CITE: ARM ARM D7.2.83 (MAIR_EL1)
//! CITE: ARM ARM D7.2.91 (SCTLR_EL1)
//! CITE: ARM ARM D7.2.106 (TCR_EL1)
//! CITE: ARM ARM D7.2.119/120 (TTBR0/1_EL1)

use core::fmt::Write;
use core::sync::atomic::{compiler_fence, Ordering};

// ---- Descriptor bits (per ARM ARM D5.3.1, D5.3.3) ----

/// Bit 0: valid descriptor
const DESC_VALID: u64 = 1 << 0;
/// Bit 1: table descriptor (L0/L1/L2 → points to next-level table)
const DESC_TABLE: u64 = 1 << 1;
/// Block descriptor: bits [1:0] = 01 (valid + NOT table)
/// Used at L1 (1 GiB blocks) and L2 (2 MiB blocks).
const DESC_BLOCK: u64 = DESC_VALID; // bit 0 = 1, bit 1 = 0

// AttrIndx field (bits [4:2]) — indexes into MAIR_EL1
/// MAIR Attr0: Normal Memory, Inner+Outer WB Non-transient
const ATTR_IDX_NORMAL: u64 = 0 << 2;
/// MAIR Attr1: Device-nGnRnE (unused currently, available)
#[allow(dead_code)]
const ATTR_IDX_DEVICE_NGNRNE: u64 = 1 << 2;
/// MAIR Attr2: Device-nGnRE (for UART, GIC)
const ATTR_IDX_DEVICE_NGNRE: u64 = 2 << 2;

/// AP[2:1] (bits [7:6]): Read-Write at EL1 only
const AP_RW_EL1: u64 = 0b00 << 6;

/// SH (bits [9:8]): Inner Shareable (for Normal cacheable regions)
const SH_INNER: u64 = 0b11 << 8;

/// AF (bit 10): Access Flag — must be set to avoid access flag faults
const AF: u64 = 1 << 10;

// Composite descriptor templates
/// Normal Memory block: WB cacheable, RW EL1, Inner Shareable
const BLOCK_NORMAL: u64 = DESC_BLOCK | ATTR_IDX_NORMAL | AP_RW_EL1 | SH_INNER | AF;
/// Device Memory block: nGnRE, RW EL1, Non-Shareable (typical for Device)
const BLOCK_DEVICE: u64 = DESC_BLOCK | ATTR_IDX_DEVICE_NGNRE | AP_RW_EL1 | AF;

// ---- MAIR_EL1 encoding (per ARM ARM D7.2.83) ----
//
// 8 attribute slots × 8 bits each = 64-bit register.
// Attr0 = 0xFF: Normal Memory, Inner+Outer Write-Back Non-transient, R+W allocate
// Attr1 = 0x00: Device-nGnRnE (strictest ordering)
// Attr2 = 0x04: Device-nGnRE (no gathering, no reordering, early ack OK)
const MAIR_EL1_VALUE: u64 = 0xFF | (0x00 << 8) | (0x04 << 16);

// ---- TCR_EL1 encoding (per ARM ARM D7.2.106) ----
//
// T0SZ [5:0]   = 16 → VA size = 2^(64-16) = 2^48 = 256 TB
// EPD0 [7]     = 0  → TTBR0 walks enabled
// IRGN0 [9:8]  = 01 → Inner WB cacheable for table walks
// ORGN0 [11:10]= 01 → Outer WB cacheable
// SH0 [13:12]  = 11 → Inner Shareable
// TG0 [15:14]  = 00 → 4 KiB granule
//
// T1SZ [21:16] = 16
// EPD1 [23]    = 0  → TTBR1 walks enabled
// IRGN1 [25:24]= 01
// ORGN1 [27:26]= 01
// SH1 [29:28]  = 11
// TG1 [31:30]  = 10 → 4 KiB (TG1 encoding differs from TG0!)
//
// IPS [34:32]  = 010 → 40-bit PA (1 TB physical address space)
const TCR_EL1_VALUE: u64 = (16 << 0)       // T0SZ
    | (0b01 << 8)   // IRGN0
    | (0b01 << 10)  // ORGN0
    | (0b11 << 12)  // SH0
    | (0b00 << 14)  // TG0 = 4K
    | (16 << 16)    // T1SZ
    | (0b01 << 24)  // IRGN1
    | (0b01 << 26)  // ORGN1
    | (0b11 << 28)  // SH1
    | (0b10u64 << 30) // TG1 = 4K (different encoding!)
    | (0b010u64 << 32); // IPS = 40-bit

// ---- Static page table allocation in .bss ----
//
// Each table: 512 × u64 = 4096 bytes, must be 4 KiB aligned.
// Per ARM ARM D5.2: TTBR base address must be aligned to table size.

#[repr(C, align(4096))]
struct PageTable([u64; 512]);

impl PageTable {
    const fn zeroed() -> Self {
        Self([0; 512])
    }
}

// TTBR0_EL1 tables (low-half identity map)
static mut TTBR0_L0: PageTable = PageTable::zeroed();
static mut TTBR0_L1: PageTable = PageTable::zeroed();
static mut TTBR0_L2_DEVIO: PageTable = PageTable::zeroed(); // L2 for 0x0..0x4000_0000

// TTBR1_EL1 tables (high-half, empty for Stage 5-6)
static mut TTBR1_L0: PageTable = PageTable::zeroed();

// ---- TTBR1_EL1 high-half tables (Stage 7) ----
//
// High-half virtual address layout (aarch64-idiomatic):
//   0xFFFF_0000_0000_0000 - 0xFFFF_FFFF_FFFF_FFFF: TTBR1 region
//
// Stage 7 fills L0[0] → L1 table covering first 512 GiB sub-region.
// Within L1: 1 GiB block descriptors for identity-style mapping
// (physical X → virtual 0xFFFF_0000_X).
//
// Mapped regions:
//   L1[1] → physical 0x4000_0000 (1 GiB Normal cacheable)
//     Kernel image + early initrd portion
//     GGUF starts at physical 0x4800_0000 → virtual 0xFFFF_0000_4800_0000
//
//   L1[2] → physical 0x8000_0000 (1 GiB Normal cacheable)
//     Late initrd portion
//     GGUF ends at physical 0x9470_79A0 → virtual 0xFFFF_0000_9470_79A0
static mut TTBR1_L1: PageTable = PageTable::zeroed();

// ---- Mode-B-Resolution: BOOT_LLM_VIRT_BASE for aarch64 ----
//
// ADR-028 v7 (Mode-B-Resolution): "Boot-LLM accessible at known
// kernel-virtual-address at boot time."
//
// Cross-platform principle, platform-native mechanics (Pillar 7):
//   x86_64:  PML4[291] = 0xFFFF_91F8_0000_0000 (bootloader-historical)
//   aarch64: TTBR1 L1[1]+offset = 0xFFFF_0000_4800_0000 (idiomatic high-half)
//
// Both achieve: GGUF accessible at known virtual address.
// Cross-platform code uses cfg-conditional BOOT_LLM_VIRT_BASE.

/// Virtual base address for Boot-LLM GGUF on aarch64.
///
/// Physical 0x4800_0000 (initrd start from DTB) mapped to high-half
/// via identity-style TTBR1 mapping: PA X → VA 0xFFFF_0000_X.
pub const BOOT_LLM_VIRT_BASE: usize = 0xFFFF_0000_4800_0000;

// ---- Stage 5: Build tables + configure registers ----

/// Stage 5: Build identity-mapped translation tables and configure
/// MAIR_EL1, TCR_EL1, TTBR0_EL1, TTBR1_EL1.
///
/// MMU is NOT enabled here — that's Stage 6.
///
/// # Safety
///
/// Must be called once during boot, before `enable_mmu()`.
/// Addresses (uart_base, gic_*) must be valid physical addresses
/// from DTB parse.
pub unsafe fn init_tables(uart_base: usize, gic_dist_base: usize, gic_cpu_base: usize) {
    let serial = &mut crate::arch::aarch64::serial::Serial;

    let _ = writeln!(serial, "Stage 5: Building translation tables...");

    // ---- IMPORTANT: write_volatile for all pre-MMU stores ----
    //
    // Pre-MMU, all memory is Device-nGnRnE (per Lesson 7). HVF traps
    // Device memory stores at EL2 and decodes them via ESR_EL2.ISV.
    // ARM ARM: STP instructions set ISV=0 (Instruction Syndrome Invalid),
    // which HVF cannot emulate → assertion failure.
    //
    // write_volatile forces LLVM to emit individual STR instructions
    // (never merged into STP), guaranteeing ISV=1 for HVF compatibility.
    //
    // CITE: ARM ARM D13.2.37 — ESR_EL2 ISV field
    // CITE: Lesson 9 — HVF-Acceleration-Constraint-Discipline

    // ---- TTBR0 L0: single entry → L1 table ----
    let l1_addr = core::ptr::addr_of!(TTBR0_L1) as u64;
    core::ptr::write_volatile(
        core::ptr::addr_of_mut!(TTBR0_L0.0[0]),
        l1_addr | DESC_VALID | DESC_TABLE,
    );

    // ---- TTBR0 L1[0]: table → L2 (covers 0x0..0x4000_0000) ----
    let l2_addr = core::ptr::addr_of!(TTBR0_L2_DEVIO) as u64;
    core::ptr::write_volatile(
        core::ptr::addr_of_mut!(TTBR0_L1.0[0]),
        l2_addr | DESC_VALID | DESC_TABLE,
    );

    // ---- TTBR0 L1[1]: 1 GiB Normal block (0x4000_0000..0x8000_0000) ----
    // Covers: kernel image, stack, early initrd portion, page tables
    core::ptr::write_volatile(
        core::ptr::addr_of_mut!(TTBR0_L1.0[1]),
        0x4000_0000u64 | BLOCK_NORMAL,
    );

    // ---- TTBR0 L1[2]: 1 GiB Normal block (0x8000_0000..0xC000_0000) ----
    // Covers: late initrd portion, DTB region
    core::ptr::write_volatile(
        core::ptr::addr_of_mut!(TTBR0_L1.0[2]),
        0x8000_0000u64 | BLOCK_NORMAL,
    );

    // ---- TTBR0 L2 entries for Device MMIO regions ----
    // L2 index = phys_addr >> 21 (each entry covers 2 MiB)
    // Most entries remain 0 (invalid) — unmapped regions fault.

    // GIC distributor region (typically 0x0800_0000)
    let gic_dist_l2 = gic_dist_base >> 21;
    let gic_dist_block = (gic_dist_base as u64) & !0x1F_FFFF; // align to 2 MiB
    core::ptr::write_volatile(
        core::ptr::addr_of_mut!(TTBR0_L2_DEVIO.0[gic_dist_l2]),
        gic_dist_block | BLOCK_DEVICE,
    );

    // GIC CPU interface (typically 0x0801_0000) — check if same 2 MiB block
    let gic_cpu_l2 = gic_cpu_base >> 21;
    if gic_cpu_l2 != gic_dist_l2 {
        let gic_cpu_block = (gic_cpu_base as u64) & !0x1F_FFFF;
        core::ptr::write_volatile(
            core::ptr::addr_of_mut!(TTBR0_L2_DEVIO.0[gic_cpu_l2]),
            gic_cpu_block | BLOCK_DEVICE,
        );
    }

    // PL011 UART (typically 0x0900_0000)
    let uart_l2 = uart_base >> 21;
    let uart_block = (uart_base as u64) & !0x1F_FFFF;
    core::ptr::write_volatile(
        core::ptr::addr_of_mut!(TTBR0_L2_DEVIO.0[uart_l2]),
        uart_block | BLOCK_DEVICE,
    );

    // Sub-MP-F1: fw-cfg MMIO (0x0902_0000) — same 2 MiB block as UART
    // (L2 index 4 covers 0x0900_0000..0x09200000, so already mapped above).
    // No additional L2 entry needed.

    let _ = writeln!(
        serial,
        "  L2 Device: GIC dist@idx={}, cpu@idx={}, UART+fwcfg@idx={}",
        gic_dist_l2, gic_cpu_l2, uart_l2
    );

    // ---- TTBR1 L0: empty (Stage 7 fills high-half entries) ----
    // Already zeroed via PageTable::zeroed()

    let _ = writeln!(
        serial,
        "  Tables: L0={:#x} L1={:#x} L2={:#x}",
        core::ptr::addr_of!(TTBR0_L0) as usize,
        l1_addr,
        l2_addr
    );

    // ---- Configure system registers ----

    // MAIR_EL1: memory attribute encoding
    core::arch::asm!("msr mair_el1, {}", in(reg) MAIR_EL1_VALUE,
        options(nomem, nostack, preserves_flags));

    // TCR_EL1: translation control
    core::arch::asm!("msr tcr_el1, {}", in(reg) TCR_EL1_VALUE,
        options(nomem, nostack, preserves_flags));

    // TTBR0_EL1: low-half page table base
    let ttbr0 = core::ptr::addr_of!(TTBR0_L0) as u64;
    core::arch::asm!("msr ttbr0_el1, {}", in(reg) ttbr0,
        options(nomem, nostack, preserves_flags));

    // TTBR1_EL1: high-half page table base (empty for now)
    let ttbr1 = core::ptr::addr_of!(TTBR1_L0) as u64;
    core::arch::asm!("msr ttbr1_el1, {}", in(reg) ttbr1,
        options(nomem, nostack, preserves_flags));

    // ISB: ensure all register writes take effect
    core::arch::asm!("isb", options(nomem, nostack, preserves_flags));

    let _ = writeln!(
        serial,
        "Stage 5: MAIR/TCR/TTBR0/TTBR1 configured. MMU still OFF."
    );
}

// ---- Stage 6: MMU enable ----

/// Stage 6: Enable MMU via SCTLR_EL1.
///
/// Sequence per ARM ARM D5.2.5 "Enabling and disabling translation":
/// 1. IC IALLU   — invalidate all instruction caches
/// 2. TLBI VMALLE1 — invalidate all TLB entries (EL1)
/// 3. DSB SY     — ensure invalidations complete
/// 4. ISB        — synchronize context before SCTLR write
/// 5. Read SCTLR_EL1, set M (bit 0) + C (bit 2) + I (bit 12)
/// 6. Write SCTLR_EL1 + ISB
///
/// After ISB: MMU active, all accesses translated via page tables.
///
/// # Safety
///
/// `init_tables()` MUST have been called first. Page tables must
/// identity-map all currently accessed regions (kernel code, stack,
/// UART, VBAR_EL1 vectors, page tables themselves).
pub unsafe fn enable_mmu() {
    let serial = &mut crate::arch::aarch64::serial::Serial;

    let _ = writeln!(
        serial,
        "Stage 6: Invalidating caches/TLB before MMU enable..."
    );

    // Step 1: Invalidate instruction cache (all)
    core::arch::asm!("ic iallu", options(nomem, nostack, preserves_flags));

    // Step 2: Invalidate TLB (all entries, EL1)
    core::arch::asm!("tlbi vmalle1", options(nomem, nostack, preserves_flags));

    // Step 3: DSB SY — ensure all invalidation completes
    core::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags));

    // Step 4: ISB — synchronize before SCTLR write
    core::arch::asm!("isb", options(nomem, nostack, preserves_flags));

    let _ = writeln!(serial, "Stage 6: Setting SCTLR_EL1.M=1, C=1, I=1...");

    // Compiler fence: ensure UART output above is flushed before MMU enable
    compiler_fence(Ordering::SeqCst);

    // Step 5-6: Read-modify-write SCTLR_EL1
    let mut sctlr: u64;
    core::arch::asm!("mrs {}, sctlr_el1", out(reg) sctlr,
        options(nomem, nostack, preserves_flags));

    sctlr |= 1 << 0; // M: MMU enable
    sctlr |= 1 << 2; // C: Data cache enable
    sctlr |= 1 << 12; // I: Instruction cache enable

    core::arch::asm!(
        "msr sctlr_el1, {}",
        "isb",
        in(reg) sctlr,
        options(nomem, nostack, preserves_flags),
    );

    // =========================================================
    // MMU IS NOW ACTIVE. All accesses translated via page tables.
    // =========================================================

    let _ = writeln!(serial, "Stage 6: MMU ENABLED. Virtual addressing active.");
    let _ = writeln!(serial, "Stage 6: Post-MMU UART write confirmed.");
}

// ---- Stage 7: High-half TTBR1 mapping (Mode-B-Resolution) ----

/// Stage 7: Map initrd region (GGUF) to high-half via TTBR1_EL1.
///
/// Mode-B-Resolution for aarch64 platform (ADR-028 v7 cross-platform).
///
/// Maps physical 0x4000_0000 - 0xC000_0000 (2 GiB) to virtual
/// 0xFFFF_0000_4000_0000 - 0xFFFF_0000_C000_0000 via TTBR1_EL1.
///
/// GGUF physical region (0x4800_0000 - 0x9470_79A0) becomes accessible
/// at virtual BOOT_LLM_VIRT_BASE = 0xFFFF_0000_4800_0000.
///
/// # Safety
///
/// Must be called AFTER MMU enabled (Stage 6). TLB invalidation
/// after mapping is critical for post-mapping access.
///
/// CITE: ARM ARM D5.2 (TTBR1 translation regime)
/// CITE: ARM ARM D8.7 (TLB maintenance)
pub unsafe fn map_initrd_high_half() {
    let serial = &mut crate::arch::aarch64::serial::Serial;

    let _ = writeln!(
        serial,
        "Stage 7: Mapping initrd to high-half via TTBR1_EL1..."
    );

    // ---- TTBR1 L0[0]: table descriptor → L1 ----
    // L0[0] covers 0xFFFF_0000_0000_0000 + 512 GiB region
    let l1_addr = core::ptr::addr_of!(TTBR1_L1) as u64;
    core::ptr::write_volatile(
        core::ptr::addr_of_mut!(TTBR1_L0.0[0]),
        l1_addr | DESC_VALID | DESC_TABLE,
    );

    // ---- TTBR1 L1[1]: 1 GiB Normal block at physical 0x4000_0000 ----
    // Maps virtual 0xFFFF_0000_4000_0000 - 0xFFFF_0000_8000_0000
    //    to physical 0x4000_0000 - 0x8000_0000
    // GGUF starts at physical 0x4800_0000 → virtual 0xFFFF_0000_4800_0000
    core::ptr::write_volatile(
        core::ptr::addr_of_mut!(TTBR1_L1.0[1]),
        0x4000_0000u64 | BLOCK_NORMAL,
    );

    // ---- TTBR1 L1[2]: 1 GiB Normal block at physical 0x8000_0000 ----
    // Maps virtual 0xFFFF_0000_8000_0000 - 0xFFFF_0000_C000_0000
    //    to physical 0x8000_0000 - 0xC000_0000
    // GGUF ends at physical 0x9470_79A0 → virtual 0xFFFF_0000_9470_79A0
    core::ptr::write_volatile(
        core::ptr::addr_of_mut!(TTBR1_L1.0[2]),
        0x8000_0000u64 | BLOCK_NORMAL,
    );

    // ---- TLB invalidation (critical post-mapping) ----
    // Per ARM ARM D8.7: TLB must be invalidated after page table updates.
    core::arch::asm!(
        "dsb ishst",      // ensure page table writes visible before TLB invalidation
        "tlbi vmalle1is", // invalidate all TLB entries (EL1, Inner Shareable)
        "dsb ish",        // ensure invalidation completes
        "isb",            // synchronize instruction stream
        options(nomem, nostack, preserves_flags),
    );

    let _ = writeln!(
        serial,
        "  TTBR1 L0[0] -> L1, L1[1]+L1[2] = 2 GiB Normal blocks"
    );
    let _ = writeln!(
        serial,
        "  BOOT_LLM_VIRT_BASE = {:#x} (physical 0x48000000)",
        BOOT_LLM_VIRT_BASE
    );
}

/// Stage 7: Verify a supported model container magic accessible at
/// BOOT_LLM_VIRT_BASE.
///
/// Reads first 4 bytes at virtual address and confirms either legacy "GGUF"
/// or native Zero Server "SILM" magic. Empirical validation that high-half
/// mapping works before Stage 11 chooses the concrete parser.
///
/// # Safety
///
/// Must be called AFTER `map_initrd_high_half()` and TLB invalidation.
pub unsafe fn verify_model_at_virt() -> bool {
    let serial = &mut crate::arch::aarch64::serial::Serial;

    let virt_ptr = BOOT_LLM_VIRT_BASE as *const u8;

    let b0 = core::ptr::read_volatile(virt_ptr);
    let b1 = core::ptr::read_volatile(virt_ptr.add(1));
    let b2 = core::ptr::read_volatile(virt_ptr.add(2));
    let b3 = core::ptr::read_volatile(virt_ptr.add(3));

    let _ = writeln!(
        serial,
        "  Bytes at {:#x}: {:#04x} {:#04x} {:#04x} {:#04x}",
        BOOT_LLM_VIRT_BASE, b0, b1, b2, b3
    );

    // GGUF magic: 'G' 'G' 'U' 'F' = 0x47 0x47 0x55 0x46
    // .smodel magic: 'S' 'I' 'L' 'M' = native SilicatePack model container.
    let is_gguf = b0 == b'G' && b1 == b'G' && b2 == b'U' && b3 == b'F';
    let is_smodel = b0 == b'S' && b1 == b'I' && b2 == b'L' && b3 == b'M';

    if is_gguf {
        let _ = writeln!(serial, "  GGUF magic VERIFIED at virtual address");
    } else if is_smodel {
        let _ = writeln!(serial, "  .smodel magic VERIFIED at virtual address");
    } else {
        let _ = writeln!(
            serial,
            "  model magic NOT FOUND (expected GGUF or SILM)"
        );
    }

    is_gguf || is_smodel
}
