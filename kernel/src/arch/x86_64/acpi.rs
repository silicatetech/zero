// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR-029 Phase 1+2 — x86_64 ACPI parser.
//!
//! Enumerates the platform's logical processors and the Local APIC base
//! by walking the firmware ACPI tables (RSDP → XSDT/RSDT → MADT). This
//! is the only mechanism Intel/AMD provide for discovering CPU topology
//! without resorting to MP-Table (legacy, removed in modern firmware)
//! or per-CPU `CPUID`-on-IPI probes (impossible before APs are awake).
//!
//! # Scope
//!
//! v1 parses four MADT entry types:
//! * Type 0 — **Processor Local APIC** (one per logical CPU on
//!   platforms that use xAPIC). We collect the 8-bit `apic_id` and
//!   the `enabled` flag.
//! * Type 1 — **I/O APIC** (one or more, used for IRQ routing). We
//!   record the first I/O APIC's MMIO base; this is read but not yet
//!   consumed by the boot path.
//! * Type 5 — **Local APIC Address Override** — critical for some
//!   AMD platforms where the firmware relocates the LAPIC. We read
//!   the 64-bit physical address from the override entry and use it
//!   in place of the default `0xFEE0_0000` (and in place of the
//!   value reported in the MADT fixed header). Per ACPI 6.4
//!   §5.2.12.5 this entry, when present, supersedes the
//!   `local_apic_address` field of the MADT.
//! * Type 9 — **Processor Local x2APIC** (one per logical CPU on
//!   x2APIC-mode platforms, including AMD EPYC and modern Intel
//!   Xeon). We collect the 32-bit `x2apic_id` and the `enabled`
//!   flag. The Cherry Server's EPYC 9354P uses Type-9 exclusively;
//!   without this parser the MADT walker reports zero CPUs and the
//!   kernel stays BSP-only.
//!
//! Other entry types (Type 2 ISO, Type 3 NMI, Type 4 LINT, Type 0xF
//! GIC for ARM, etc.) are recognised and skipped by length.
//!
//! Type-0 and Type-9 are deduplicated: some firmware emits both for
//! the same logical CPU (legacy + modern representations). We treat
//! them as identifying the same processor and store each APIC ID
//! exactly once.
//!
//! # Pillar conformance
//!
//! * **Pillar 1** — Zero alloc. Output structure is `[u32; MAX_CPUS]`
//!   on the stack; the caller copies into [`crate::smp`] storage.
//! * **Pillar 7** — Architecture-specific by definition (ACPI is an
//!   x86/x86_64 ecosystem standard). Lives in `arch::x86_64`.
//!
//! # Safety
//!
//! All firmware reads happen through the higher-half physical-memory
//! map (`PHYS_OFFSET + addr`). The caller must ensure this offset is
//! installed before `parse_madt` is called. We assume identity-style
//! page tables for firmware regions (`0x0000_0000 .. 0x0010_0000` and
//! 32-bit ACPI table addresses), which the bootloader provides via the
//! linear physical-memory mapping pinned in `main.rs::BOOTLOADER_CONFIG`.
//!
//! CITE: ACPI 6.4 §5.2.5 (RSDP), §5.2.7 (RSDT), §5.2.8 (XSDT)
//! CITE: ACPI 6.4 §5.2.12 (MADT)
//! CITE: AMD64/x86 Local APIC architectural base = 0xFEE00000 default

#![allow(dead_code)] // Some helpers are exercised only via tests.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Maximum number of logical processors we record. Sized to match
/// [`crate::smp::MAX_CORES`] (64). The deployment target is 32 cores,
/// so 64 leaves 2× headroom for SMT-on if the firmware exposes it.
pub const MAX_CPUS: usize = 64;

/// Default Local APIC physical base for AMD64/x86 systems. Overridden by
/// the firmware via MADT field `local_apic_address` (always present)
/// or, in rare cases, by MADT entry Type 5 (LAPIC Address Override).
pub const DEFAULT_LAPIC_PHYS: u64 = 0xFEE0_0000;

// ─────────────────────────────────────────────────────────────────
// Public output: per-CPU table and aggregate info
// ─────────────────────────────────────────────────────────────────

/// A single CPU's ACPI-reported identity.
#[derive(Copy, Clone, Debug, Default)]
pub struct CpuInfo {
    /// Local APIC ID. Unique per logical processor. On the AMD EPYC 9354P
    /// deployment target, firmware may expose SMT as 64 logical CPUs.
    pub apic_id: u32,
    /// `true` iff the firmware reports this CPU as enabled (bit 0 of
    /// the Local-APIC-flags field). Disabled CPUs (bit 0 clear but
    /// bit 1 = "Online Capable" set) are recorded but skipped during
    /// AP boot; the contract is "boot only what firmware says is on".
    pub enabled: bool,
}

/// Aggregate ACPI topology info returned by [`parse_madt`].
#[derive(Copy, Clone, Debug)]
pub struct AcpiInfo {
    /// Physical address of the Local APIC MMIO region. The kernel
    /// must map this region uncacheable (strong-ordered, write-
    /// through-disabled) before issuing IPIs.
    pub lapic_phys_base: u64,
    /// Physical address of the first I/O APIC found. `0` if none.
    pub ioapic_phys_base: u64,
    /// Number of valid entries in `cpus[0..cpu_count]`.
    pub cpu_count: u32,
    /// `true` iff a MADT Type-5 (Local APIC Address Override) entry
    /// was encountered and applied to `lapic_phys_base`. Surfaced so
    /// the boot path can log "Type-5 override active" — important on
    /// AMD platforms where the firmware moves the LAPIC away from
    /// the SDM default of `0xFEE0_0000`.
    pub lapic_override_applied: bool,
    /// Per-CPU info. Entries beyond `cpu_count` are zeroed (apic_id=0,
    /// enabled=false). The caller must NOT iterate past `cpu_count`.
    pub cpus: [CpuInfo; MAX_CPUS],
}

impl AcpiInfo {
    /// Build an empty (zero-CPU) info struct. Used as the failure-mode
    /// return when parsing fails — the kernel then runs in BSP-only mode.
    pub const fn empty() -> Self {
        Self {
            lapic_phys_base: DEFAULT_LAPIC_PHYS,
            ioapic_phys_base: 0,
            cpu_count: 0,
            lapic_override_applied: false,
            cpus: [CpuInfo {
                apic_id: 0,
                enabled: false,
            }; MAX_CPUS],
        }
    }
}

/// Errors that prevent ACPI enumeration. Each variant is named after
/// the parse stage that fails, so log lines pinpoint where firmware
/// is unexpected.
#[derive(Copy, Clone, Debug)]
pub enum AcpiError {
    /// RSDP signature ("RSD PTR ") not found in EBDA or BIOS area.
    RsdpNotFound,
    /// RSDP checksum mismatch (first 20 bytes must sum to 0 mod 256
    /// per ACPI 1.0; full 36 bytes for ACPI 2.0+).
    RsdpChecksumBad,
    /// XSDT/RSDT signature mismatch.
    SdtSignatureBad,
    /// XSDT/RSDT checksum mismatch.
    SdtChecksumBad,
    /// No MADT (signature "APIC") found in the XSDT/RSDT entry array.
    MadtNotFound,
    /// MADT length field truncated mid-entry.
    MadtTruncated,
    /// Physical-memory offset hasn't been set up yet.
    PhysOffsetMissing,
}

// ─────────────────────────────────────────────────────────────────
// Cache: parsed info available to the rest of the kernel
// ─────────────────────────────────────────────────────────────────

/// Cached LAPIC physical base, populated by [`parse_madt`]. Read by
/// `crate::arch::x86_64::apic::init`. `0` means "not parsed yet".
pub static LAPIC_PHYS_BASE: AtomicU64 = AtomicU64::new(0);

/// Cached I/O APIC physical base. Read by future PIC-replacement code.
pub static IOAPIC_PHYS_BASE: AtomicU64 = AtomicU64::new(0);

static RECORDED_CPU_COUNT: AtomicU32 = AtomicU32::new(0);
static RECORDED_LAPIC_PHYS_BASE: AtomicU64 = AtomicU64::new(0);
static RECORDED_IOAPIC_PHYS_BASE: AtomicU64 = AtomicU64::new(0);
static RECORDED_LAPIC_OVERRIDE: AtomicBool = AtomicBool::new(false);
static RECORDED_CPUS: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(0) }; MAX_CPUS];

const REC_CPU_VALID: u64 = 1 << 63;
const REC_CPU_ENABLED: u64 = 1 << 32;
const REC_CPU_APIC_MASK: u64 = 0xffff_ffff;

// ─────────────────────────────────────────────────────────────────
// Top-level parse entry
// ─────────────────────────────────────────────────────────────────

/// Walk RSDP → XSDT/RSDT → MADT and return the platform's CPU topology.
///
/// `phys_offset` is the bootloader-installed higher-half mapping of
/// the physical address space, e.g. `0xFFFF_8000_0000_0000` per
/// `BOOTLOADER_CONFIG` in `main.rs`. A physical address `P` is read
/// as `(phys_offset + P) as *const T`.
///
/// On success: caches LAPIC + IOAPIC base in the atomics above and
/// returns the [`AcpiInfo`]. The caller is expected to relay this to
/// `crate::arch::x86_64::apic::init` and the SMP bring-up loop.
///
/// # Safety
///
/// The caller must guarantee:
/// * `phys_offset` is the active linear-physical-memory mapping.
/// * Physical addresses in the firmware tables point to memory that
///   the bootloader has identity-or-linear-mapped (true on UEFI x86_64
///   for ACPI tables, which UEFI reports as `Reserved` / `ACPI*` memory
///   and `bootloader_api` includes in the linear map).
pub unsafe fn parse_madt(
    phys_offset: u64,
    bootloader_rsdp: Option<u64>,
) -> Result<AcpiInfo, AcpiError> {
    let rsdp = find_rsdp(phys_offset, bootloader_rsdp).ok_or(AcpiError::RsdpNotFound)?;

    // RSDP version: 0 = ACPI 1.0 (RSDT only, 32-bit ptr), 2 = ACPI 2.0+
    // (XSDT preferred, 64-bit ptr). We trust the firmware's revision
    // field rather than feature-test, per ACPI 6.4 §5.2.5.3.
    let rev = rsdp.revision;
    let xsdt_or_rsdt_phys: u64 = if rev >= 2 && rsdp.xsdt_address != 0 {
        rsdp.xsdt_address
    } else {
        rsdp.rsdt_address as u64
    };

    let sdt = read_sdt_header(phys_offset, xsdt_or_rsdt_phys)?;
    // Walk the entries — each is a 4-byte (RSDT) or 8-byte (XSDT) pointer.
    let entry_size = if rev >= 2 && rsdp.xsdt_address != 0 {
        8
    } else {
        4
    };
    let entries_off = core::mem::size_of::<SdtHeader>();
    let entries_bytes = sdt.length as usize - entries_off;
    let n_entries = entries_bytes / entry_size;

    let mut madt_phys: u64 = 0;
    for i in 0..n_entries {
        let entry_phys_ptr =
            phys_offset + xsdt_or_rsdt_phys + entries_off as u64 + (i * entry_size) as u64;
        let candidate: u64 = if entry_size == 8 {
            core::ptr::read_unaligned(entry_phys_ptr as *const u64)
        } else {
            core::ptr::read_unaligned(entry_phys_ptr as *const u32) as u64
        };
        // Each entry points to an SDT; check signature.
        let hdr = read_sdt_header(phys_offset, candidate);
        if let Ok(h) = hdr {
            if &h.signature == b"APIC" {
                madt_phys = candidate;
                break;
            }
        }
    }

    if madt_phys == 0 {
        return Err(AcpiError::MadtNotFound);
    }

    parse_madt_table(phys_offset, madt_phys)
}

// ─────────────────────────────────────────────────────────────────
// RSDP location
// ─────────────────────────────────────────────────────────────────

/// Raw RSDP structure per ACPI 6.4 §5.2.5.3. `packed` because the spec
/// lays it out without natural alignment.
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct Rsdp {
    signature: [u8; 8],
    checksum: u8,
    oem_id: [u8; 6],
    revision: u8,
    rsdt_address: u32,
    // ACPI 2.0+ extension:
    length: u32,
    xsdt_address: u64,
    extended_checksum: u8,
    _reserved: [u8; 3],
}

/// Scan the BIOS RSDP locations and return the first valid RSDP.
///
/// Per ACPI 6.4 §5.2.5.1: on legacy BIOS the RSDP is in one of:
/// 1. The first 1 KiB of the Extended BIOS Data Area (EBDA).
/// 2. The address range `0xE0000 .. 0xFFFFF` (BIOS read-only memory).
///
/// Strict UEFI firmware does not preserve these legacy regions. UEFI
/// instead exposes the RSDP through the EFI System Table's
/// configuration table array, and `bootloader_api` 0.11 forwards that
/// address as `BootInfo::rsdp_addr`. If the legacy scan fails, we fall
/// back to the bootloader-provided physical address: read the RSDP
/// at `(phys_offset + bootloader_rsdp)` and re-validate the checksum.
///
/// In each region the signature is 16-byte aligned. We scan both
/// legacy regions before consulting the bootloader address so that
/// hybrid BIOS+UEFI boots prefer the firmware-canonical placement.
unsafe fn find_rsdp(phys_offset: u64, bootloader_rsdp: Option<u64>) -> Option<Rsdp> {
    use crate::arch::serial::Serial;
    use core::fmt::Write;

    // EBDA: 2-byte segment at physical 0x40E, shifted left by 4.
    // Some firmware reports 0 here, in which case fall straight through
    // to the BIOS range.
    let ebda_seg = core::ptr::read_unaligned((phys_offset + 0x40E) as *const u16);
    let ebda_base = (ebda_seg as u64) << 4;
    if ebda_base != 0 && ebda_base < 0x10_0000 {
        if let Some(r) = scan_rsdp_range(phys_offset, ebda_base, 1024) {
            return Some(r);
        }
    }

    // Legacy BIOS extension area.
    if let Some(r) = scan_rsdp_range(phys_offset, 0xE0000, 0x20000) {
        return Some(r);
    }

    // UEFI fallback: trust the bootloader-forwarded address. The
    // bootloader read it from the EFI System Table's configuration
    // table array (ACPI 2.0 GUID), so we only need to validate the
    // signature + checksum at the destination.
    if let Some(rsdp_phys) = bootloader_rsdp {
        let p = phys_offset + rsdp_phys;
        let sig: [u8; 8] = core::ptr::read_unaligned(p as *const [u8; 8]);
        const SIGNATURE: [u8; 8] = *b"RSD PTR ";
        if sig == SIGNATURE {
            let rsdp: Rsdp = core::ptr::read_unaligned(p as *const Rsdp);
            if rsdp_checksum_ok(p, 20) {
                if rsdp.revision >= 2 && rsdp.length as usize >= core::mem::size_of::<Rsdp>() {
                    if !rsdp_checksum_ok(p, rsdp.length as usize) {
                        return None;
                    }
                }
                let _ = writeln!(
                    Serial,
                    "ACPI: RSDP found via bootloader at 0x{:x}",
                    rsdp_phys,
                );
                return Some(rsdp);
            }
        }
    }

    None
}

/// Scan `len` bytes starting at physical `base` for the RSDP signature.
/// The signature is "RSD PTR " (with trailing space). Aligned at 16 bytes.
unsafe fn scan_rsdp_range(phys_offset: u64, base: u64, len: usize) -> Option<Rsdp> {
    const SIGNATURE: [u8; 8] = *b"RSD PTR ";
    let mut off = 0usize;
    while off + core::mem::size_of::<Rsdp>() <= len {
        let p = phys_offset + base + off as u64;
        let sig: [u8; 8] = core::ptr::read_unaligned(p as *const [u8; 8]);
        if sig == SIGNATURE {
            let rsdp: Rsdp = core::ptr::read_unaligned(p as *const Rsdp);
            // Validate ACPI 1.0 checksum (first 20 bytes sum to 0).
            if rsdp_checksum_ok(p, 20) {
                // For ACPI 2.0+ we'd also want to verify the extended
                // checksum (`length` bytes), but if revision >= 2 and
                // length is sensible we trust the firmware. Defensive
                // verify only if length looks valid.
                if rsdp.revision >= 2 && rsdp.length as usize >= core::mem::size_of::<Rsdp>() {
                    if !rsdp_checksum_ok(p, rsdp.length as usize) {
                        // Bad extended checksum; skip and keep scanning.
                        off += 16;
                        continue;
                    }
                }
                return Some(rsdp);
            }
        }
        off += 16; // ACPI spec: RSDP is 16-byte aligned.
    }
    None
}

/// Byte-sum a `len`-byte region starting at virtual `addr` and verify
/// the sum is zero mod 256 (ACPI's universal "structure integrity"
/// checksum convention).
unsafe fn rsdp_checksum_ok(addr: u64, len: usize) -> bool {
    let mut sum: u8 = 0;
    for i in 0..len {
        let b = *((addr + i as u64) as *const u8);
        sum = sum.wrapping_add(b);
    }
    sum == 0
}

// ─────────────────────────────────────────────────────────────────
// SDT (System Description Table) header
// ─────────────────────────────────────────────────────────────────

/// ACPI System Description Table header — common to RSDT, XSDT, MADT,
/// FADT, every ACPI table. Per ACPI 6.4 §5.2.6.
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct SdtHeader {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem_id: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    creator_id: u32,
    creator_revision: u32,
}

/// Read and validate an SDT header at physical `phys`. Returns the
/// header by-value (caller does not own the underlying memory).
unsafe fn read_sdt_header(phys_offset: u64, phys: u64) -> Result<SdtHeader, AcpiError> {
    let p = phys_offset + phys;
    let hdr: SdtHeader = core::ptr::read_unaligned(p as *const SdtHeader);
    if !rsdp_checksum_ok(p, hdr.length as usize) {
        return Err(AcpiError::SdtChecksumBad);
    }
    Ok(hdr)
}

// ─────────────────────────────────────────────────────────────────
// MADT parsing
// ─────────────────────────────────────────────────────────────────

/// MADT-specific header following the SDT header. Per ACPI 6.4 §5.2.12.
///
/// Layout:
/// ```text
///   offset  size  field
///   36      4     local_apic_address (physical, 32-bit even on x86_64)
///   40      4     flags (bit 0 = PCAT_COMPAT i.e. 8259 PIC present)
///   44      ..    variable-length sequence of MADT entries
/// ```
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct MadtFixed {
    sdt: SdtHeader,
    local_apic_address: u32,
    flags: u32,
}

/// MADT entry header (variable-length payload follows).
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct MadtEntry {
    entry_type: u8,
    length: u8,
}

/// MADT entry Type 0 — Processor Local APIC.
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct MadtLapic {
    hdr: MadtEntry,
    /// ACPI processor ID — used for naming, not for IPI.
    acpi_processor_id: u8,
    /// Local APIC ID — what we actually send IPIs to.
    apic_id: u8,
    /// Bit 0 = "Enabled", bit 1 = "Online Capable" (ACPI 6.3+).
    flags: u32,
}

/// MADT entry Type 1 — I/O APIC.
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct MadtIoapic {
    hdr: MadtEntry,
    ioapic_id: u8,
    _reserved: u8,
    ioapic_address: u32,
    global_system_interrupt_base: u32,
}

/// MADT entry Type 5 — Local APIC Address Override.
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct MadtLapicOverride {
    hdr: MadtEntry,
    _reserved: u16,
    /// 64-bit physical address that supersedes the MADT's `local_apic_address`.
    address: u64,
}

/// MADT entry Type 9 — Processor Local x2APIC.
///
/// Per ACPI 6.4 §5.2.12.12. Used by platforms operating in x2APIC
/// mode where the local APIC ID exceeds 8 bits or where firmware
/// chooses the modern representation regardless. AMD EPYC firmware
/// almost always emits Type-9 entries instead of (or alongside)
/// Type-0.
///
/// Layout (16 bytes total):
/// ```text
///   offset  size  field
///   0       1     type (= 9)
///   1       1     length (= 16)
///   2       2     reserved
///   4       4     x2apic_id (32-bit local APIC ID)
///   8       4     flags (bit 0 = enabled, bit 1 = online-capable)
///   12      4     acpi_processor_uid
/// ```
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct MadtX2Apic {
    hdr: MadtEntry,
    _reserved: u16,
    /// 32-bit local APIC ID. x2APIC-mode wake-up uses this full value
    /// through the x2APIC ICR MSR; legacy xAPIC fallback can only address
    /// IDs that fit in the 8-bit physical destination field.
    x2apic_id: u32,
    /// Bit 0 = "Enabled", bit 1 = "Online Capable" — same encoding
    /// as Type-0 (ACPI 6.3+).
    flags: u32,
    /// ACPI processor UID — names this CPU in the namespace; not
    /// used by our boot path.
    _acpi_processor_uid: u32,
}

/// Parse the MADT at physical `madt_phys` and return the AcpiInfo.
unsafe fn parse_madt_table(phys_offset: u64, madt_phys: u64) -> Result<AcpiInfo, AcpiError> {
    let mp = phys_offset + madt_phys;
    let fixed: MadtFixed = core::ptr::read_unaligned(mp as *const MadtFixed);

    let mut info = AcpiInfo::empty();
    info.lapic_phys_base = fixed.local_apic_address as u64;

    // Walk variable-length entries.
    let madt_total_len = fixed.sdt.length as usize;
    let mut off = core::mem::size_of::<MadtFixed>();

    while off + core::mem::size_of::<MadtEntry>() <= madt_total_len {
        let e_ptr = mp + off as u64;
        let entry: MadtEntry = core::ptr::read_unaligned(e_ptr as *const MadtEntry);
        if entry.length == 0 {
            // Avoid infinite loop on malformed firmware (rare but
            // observed on some VirtualBox builds pre-7.0).
            return Err(AcpiError::MadtTruncated);
        }
        if off + entry.length as usize > madt_total_len {
            return Err(AcpiError::MadtTruncated);
        }

        match entry.entry_type {
            0 => {
                // Processor Local APIC (xAPIC)
                let lapic: MadtLapic = core::ptr::read_unaligned(e_ptr as *const MadtLapic);
                let enabled = (lapic.flags & 1) != 0;
                record_cpu(&mut info, lapic.apic_id as u32, enabled);
            }
            1 => {
                // I/O APIC — record the first one only (v0 doesn't
                // support multiple I/O APICs; future work).
                if info.ioapic_phys_base == 0 {
                    let io: MadtIoapic = core::ptr::read_unaligned(e_ptr as *const MadtIoapic);
                    info.ioapic_phys_base = io.ioapic_address as u64;
                }
            }
            5 => {
                // Local APIC Address Override.
                //
                // Per ACPI 6.4 §5.2.12.5: this 64-bit physical address
                // overrides whatever the MADT fixed header reported in
                // `local_apic_address` (and, transitively, the SDM's
                // `0xFEE0_0000` default). On AMD platforms whose
                // firmware relocates the LAPIC, missing this branch
                // would leave the kernel writing into a non-LAPIC
                // page during AP wake-up. The override is recorded
                // on `info.lapic_override_applied` so the boot log
                // can surface that the alternative base is in use.
                let ov: MadtLapicOverride =
                    core::ptr::read_unaligned(e_ptr as *const MadtLapicOverride);
                info.lapic_phys_base = ov.address;
                info.lapic_override_applied = true;
            }
            9 => {
                // Processor Local x2APIC. AMD EPYC firmware emits these
                // instead of Type-0; without this branch the MADT walker
                // returns zero CPUs and the kernel falls back to BSP-only.
                let x2: MadtX2Apic = core::ptr::read_unaligned(e_ptr as *const MadtX2Apic);
                let enabled = (x2.flags & 1) != 0;
                record_cpu(&mut info, x2.x2apic_id, enabled);
            }
            _ => {
                // Skip — types 2..4 (ISO/NMI/LINT), 6..8, 10..0xF (NMI
                // variants, GIC for ARM repurposed MADT, etc.). Length
                // skip is sufficient; we don't need their content.
            }
        }

        off += entry.length as usize;
    }

    // Cache for downstream consumers.
    LAPIC_PHYS_BASE.store(info.lapic_phys_base, Ordering::Release);
    IOAPIC_PHYS_BASE.store(info.ioapic_phys_base, Ordering::Release);
    record_topology(&info);

    Ok(info)
}

/// Record the last parsed MADT topology for shell diagnostics.
pub fn record_topology(info: &AcpiInfo) {
    RECORDED_CPU_COUNT.store(0, Ordering::Release);
    RECORDED_LAPIC_PHYS_BASE.store(info.lapic_phys_base, Ordering::Release);
    RECORDED_IOAPIC_PHYS_BASE.store(info.ioapic_phys_base, Ordering::Release);
    RECORDED_LAPIC_OVERRIDE.store(info.lapic_override_applied, Ordering::Release);

    let mut i = 0usize;
    while i < MAX_CPUS {
        let raw = if i < info.cpu_count as usize {
            let mut v = REC_CPU_VALID | (info.cpus[i].apic_id as u64 & REC_CPU_APIC_MASK);
            if info.cpus[i].enabled {
                v |= REC_CPU_ENABLED;
            }
            v
        } else {
            0
        };
        RECORDED_CPUS[i].store(raw, Ordering::Release);
        i += 1;
    }
    RECORDED_CPU_COUNT.store(info.cpu_count, Ordering::Release);
}

/// Append a CPU to `info.cpus`, deduplicating by APIC ID.
///
/// Some firmware emits both a Type-0 (xAPIC) and a Type-9 (x2APIC)
/// entry for the same logical processor — for example to remain
/// backward-compatible with OS loaders that only understand legacy
/// MADT entries. We treat identical APIC IDs as the same CPU and
/// store it exactly once; if the duplicate carries `enabled=true`
/// while the original was `enabled=false`, we upgrade the flag.
fn record_cpu(info: &mut AcpiInfo, apic_id: u32, enabled: bool) {
    for i in 0..(info.cpu_count as usize) {
        if info.cpus[i].apic_id == apic_id {
            info.cpus[i].enabled = info.cpus[i].enabled || enabled;
            return;
        }
    }
    if (info.cpu_count as usize) < MAX_CPUS {
        info.cpus[info.cpu_count as usize] = CpuInfo { apic_id, enabled };
        info.cpu_count += 1;
    }
    // CPUs beyond MAX_CPUS are silently dropped — the deployment
    // target (32 cores) is well within bounds.
}

// ─────────────────────────────────────────────────────────────────
// Convenience accessors
// ─────────────────────────────────────────────────────────────────

/// Returns the parsed LAPIC physical base, or [`DEFAULT_LAPIC_PHYS`]
/// if `parse_madt` hasn't been called yet (the SDM default is correct
/// for all current Intel/AMD silicon).
#[inline]
pub fn lapic_phys_base() -> u64 {
    let cached = LAPIC_PHYS_BASE.load(Ordering::Acquire);
    if cached == 0 {
        DEFAULT_LAPIC_PHYS
    } else {
        cached
    }
}

/// Returns the parsed I/O APIC physical base, or `0` if none.
#[inline]
pub fn ioapic_phys_base() -> u64 {
    IOAPIC_PHYS_BASE.load(Ordering::Acquire)
}

/// Return the last parsed MADT summary for network-shell diagnostics.
pub fn recorded_summary() -> (u32, u64, u64, bool) {
    (
        RECORDED_CPU_COUNT.load(Ordering::Acquire),
        RECORDED_LAPIC_PHYS_BASE.load(Ordering::Acquire),
        RECORDED_IOAPIC_PHYS_BASE.load(Ordering::Acquire),
        RECORDED_LAPIC_OVERRIDE.load(Ordering::Acquire),
    )
}

/// Return a recorded CPU entry from the last successful MADT parse.
pub fn recorded_cpu(idx: usize) -> Option<CpuInfo> {
    if idx >= MAX_CPUS {
        return None;
    }
    let raw = RECORDED_CPUS[idx].load(Ordering::Acquire);
    if (raw & REC_CPU_VALID) == 0 {
        return None;
    }
    Some(CpuInfo {
        apic_id: (raw & REC_CPU_APIC_MASK) as u32,
        enabled: (raw & REC_CPU_ENABLED) != 0,
    })
}
