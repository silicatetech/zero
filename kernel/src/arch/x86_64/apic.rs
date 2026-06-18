// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR-029 Phase 1+2 — x86_64 Local APIC driver and AP wake-up.
//!
//! Provides the minimum surface needed to wake up Application Processors
//! (APs) via the INIT-SIPI-SIPI sequence and to identify which logical
//! processor the current code is running on.
//!
//! # Scope (v0)
//!
//! * xAPIC MMIO is preferred when the MADT topology fits in 8-bit APIC
//!   IDs. We only use x2APIC when a target APIC ID exceeds xAPIC's
//!   physical-destination range. The Cherry EPYC target reports IDs up
//!   to 119, so xAPIC is sufficient and avoids depending on x2APIC
//!   interrupt-remapping policy while AMD-Vi is deliberately bypassed.
//! * Software-enable the APIC via SVR (Spurious-Interrupt Vector
//!   Register). The default-disabled state on cold boot leaves all
//!   IPIs masked.
//! * INIT-SIPI-SIPI per the AMD64/x86 AP startup protocol. Used by the
//!   BSP to drive APs through their trampoline.
//! * Read-only access to the current core's APIC ID via the APIC ID
//!   register (offset 0x20). APs use this in `ap_entry` to claim their
//!   logical core index.
//!
//! # Not implemented (deferred)
//!
//! * Inter-processor interrupts (IPI) other than INIT/SIPI. The SMP
//!   work-publication mechanism uses shared atomics + spin-wait, not
//!   IPIs, so there's nothing to wire up here.
//! * APIC timer. The BSP retains the legacy PIT for time-keeping.
//! * EOI (End-of-Interrupt). No interrupt sources are routed through
//!   the APIC in v0; the legacy PIC still drives IRQ 0..15.
//!
//! # Pillar conformance
//!
//! * **Pillar 1** — No heap. The driver is a `pub struct Apic` carrying
//!   only the MMIO base; all reads/writes are direct.
//! * **Pillar 7** — x86_64-specific by definition (LAPIC is part of
//!   the x86 ISA). aarch64 has GICv2/v3 in `arch::aarch64::gic`.
//!
//! CITE: AMD64 APM Vol 2 §16.4 (Local APIC)
//! CITE: x86 SDM/APM INIT-SIPI-SIPI universal startup sequence

// Several constants (REG_EOI, DELIVERY_FIXED) and diagnostic methods
// (esr) are unused in the BSP-boot v0 path but kept as documented
// API surface for the upcoming I/O APIC / cross-core IPI work.
#![allow(dead_code)]

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::{cpuinfo, cycles};
use crate::memory;

// ─────────────────────────────────────────────────────────────────
// Local APIC register offsets (architectural xAPIC MMIO layout)
// ─────────────────────────────────────────────────────────────────

/// Local APIC ID register. Read-only on most CPUs.
/// Bits 31:24 = APIC ID (xAPIC). x2APIC uses full 32 bits at MSR 0x802.
const REG_ID: usize = 0x020;
/// APIC version register (read-only). Bits 7:0 = version, 23:16 = max LVT entry.
const REG_VERSION: usize = 0x030;
/// Task Priority Register — set to 0 to accept all interrupt classes.
const REG_TPR: usize = 0x080;
/// End-of-Interrupt register — write any value to acknowledge an interrupt.
const REG_EOI: usize = 0x0B0;
/// Spurious-Interrupt Vector Register. Bit 8 = APIC software enable.
const REG_SVR: usize = 0x0F0;
/// Error Status Register.
const REG_ESR: usize = 0x280;
/// Interrupt Command Register, low half (32 bits). Writing the low half
/// triggers the IPI dispatch.
const REG_ICR_LOW: usize = 0x300;
/// Interrupt Command Register, high half. Holds destination APIC ID in
/// bits 31:24 (xAPIC physical destination).
const REG_ICR_HIGH: usize = 0x310;

// ─────────────────────────────────────────────────────────────────
// x2APIC MSRs
// ─────────────────────────────────────────────────────────────────

const MSR_IA32_APIC_BASE: u32 = 0x0000_001B;
const IA32_APIC_BASE_ENABLE: u64 = 1 << 11;
const IA32_APIC_BASE_X2APIC: u64 = 1 << 10;

const MSR_X2APIC_ID: u32 = 0x0000_0802;
const MSR_X2APIC_VERSION: u32 = 0x0000_0803;
const MSR_X2APIC_TPR: u32 = 0x0000_0808;
const MSR_X2APIC_EOI: u32 = 0x0000_080B;
const MSR_X2APIC_SVR: u32 = 0x0000_080F;
const MSR_X2APIC_ESR: u32 = 0x0000_0828;
const MSR_X2APIC_ICR: u32 = 0x0000_0830;

const APIC_MODE_UNKNOWN: u32 = 0;
const APIC_MODE_XAPIC: u32 = 1;
const APIC_MODE_X2APIC: u32 = 2;

// ─────────────────────────────────────────────────────────────────
// ICR fields (architectural APIC Interrupt Command Register layout)
// ─────────────────────────────────────────────────────────────────

/// Delivery mode = INIT (5 << 8).
const DELIVERY_INIT: u32 = 0b101 << 8;
/// Delivery mode = Startup (SIPI) (6 << 8).
const DELIVERY_STARTUP: u32 = 0b110 << 8;
/// Delivery mode = Fixed (0 << 8).
const DELIVERY_FIXED: u32 = 0b000 << 8;
/// Destination mode = Physical (bit 11 = 0). We use physical for AP wake.
const DEST_MODE_PHYSICAL: u32 = 0;
/// Level = Assert (bit 14 = 1). Required for INIT IPIs.
const LEVEL_ASSERT: u32 = 1 << 14;
/// Trigger mode = Level (bit 15 = 1). Used for INIT-deassert in legacy.
const TRIGGER_LEVEL: u32 = 1 << 15;
/// Shorthand: no shorthand (bits 19:18 = 00). Default — destination
/// field in ICR_HIGH selects the target.
const SHORTHAND_NONE: u32 = 0;
/// "Delivery Status" bit 12 — read-only, indicates the previous IPI is
/// still being delivered. We poll this to serialize back-to-back IPIs.
const DELIVERY_STATUS: u32 = 1 << 12;

// ─────────────────────────────────────────────────────────────────
// SVR (Spurious-Interrupt Vector Register) fields
// ─────────────────────────────────────────────────────────────────

/// Bit 8 of SVR = APIC software enable. Setting this to 1 enables the
/// local APIC; setting to 0 disables it.
const SVR_ENABLE: u32 = 1 << 8;
/// Spurious vector — bottom byte. We use 0xFF (the canonical "unused"
/// vector), which the IDT routes to a no-op stub.
const SVR_SPURIOUS_VECTOR: u32 = 0xFF;

// ─────────────────────────────────────────────────────────────────
// The Apic driver handle
// ─────────────────────────────────────────────────────────────────

/// Local APIC handle for the BSP. APs share the same MMIO region
/// (each core sees its *own* registers when reading APIC_ID, because
/// the LAPIC is a per-core unit despite all cores using the same
/// physical address — the chipset routes reads to the local LAPIC of
/// the issuing core).
///
/// The handle stores the *virtual* address at which the LAPIC MMIO is
/// mapped. This is `phys_offset + lapic_phys_base` once the kernel
/// has the linear physical-memory mapping installed.
#[derive(Copy, Clone)]
pub struct Apic {
    mmio_base_virt: u64,
    mode: ApicMode,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum ApicMode {
    XApic,
    X2Apic,
}

/// Cached LAPIC virtual base for `current_apic_id` / `boot_ap`. Set by
/// [`Apic::init`]; `0` until then.
pub static LAPIC_VIRT_BASE: AtomicU64 = AtomicU64::new(0);
static LAPIC_MODE: AtomicU32 = AtomicU32::new(APIC_MODE_UNKNOWN);

impl Apic {
    /// Build the driver handle and software-enable the APIC.
    ///
    /// `lapic_phys` = physical address from `acpi::lapic_phys_base()`.
    /// `phys_offset` = higher-half linear mapping.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// * The MMIO range `[lapic_phys, lapic_phys + 0x400)` is mapped
    ///   as uncacheable (UC). The bootloader's linear map is
    ///   cacheable (write-back) by default, so this driver currently
    ///   relies on the firmware-installed MTRRs covering the LAPIC
    ///   page with a UC entry — the SDM-canonical default for the
    ///   `0xFEE0_0000` region, and the documented behaviour on the
    ///   AMD EPYC 9354P firmware in our deployment target.
    ///
    ///   **Assumption (firmware MTRR = UC at LAPIC base):** verified
    ///   on EPYC 9354P; not verified on every UEFI we may run under.
    ///   If a board boots with the LAPIC page in WB, MMIO writes
    ///   become silently coalesced and IPIs (INIT/SIPI) will not
    ///   reach APs.
    ///
    ///   **TODO (PAT/MTRR enforcement):** install an explicit
    ///   PAT/MTRR override for the LAPIC page so we no longer rely
    ///   on firmware. Tracked as future work; out of scope for
    ///   ADR-029 Phase 1+2.
    /// * `phys_offset` is valid.
    pub unsafe fn init(lapic_phys: u64, phys_offset: u64, max_apic_id: u32) -> Self {
        if cpuinfo::x2apic_supported() && max_apic_id > 0xFF {
            let mut base = rdmsr(MSR_IA32_APIC_BASE);
            base |= IA32_APIC_BASE_ENABLE | IA32_APIC_BASE_X2APIC;
            wrmsr(MSR_IA32_APIC_BASE, base);

            LAPIC_VIRT_BASE.store(0, Ordering::Release);
            LAPIC_MODE.store(APIC_MODE_X2APIC, Ordering::Release);

            let apic = Apic {
                mmio_base_virt: 0,
                mode: ApicMode::X2Apic,
            };
            apic.enable_software_apic();
            apic
        } else {
            let virt = match memory::map_mmio(lapic_phys, 0x1000) {
                Ok(mapped) => mapped as u64,
                Err(_) => phys_offset + lapic_phys,
            };
            LAPIC_VIRT_BASE.store(virt, Ordering::Release);
            LAPIC_MODE.store(APIC_MODE_XAPIC, Ordering::Release);

            let mut base = rdmsr(MSR_IA32_APIC_BASE);
            if (base & IA32_APIC_BASE_X2APIC) != 0 {
                // Architecturally leave x2APIC by disabling the local
                // APIC and clearing EXTD together, then re-enable in
                // xAPIC mode. Cherry's APIC IDs fit in 8 bits, so this
                // is the simpler bring-up path while AMD-Vi/IR is off.
                base &= !(IA32_APIC_BASE_ENABLE | IA32_APIC_BASE_X2APIC);
                wrmsr(MSR_IA32_APIC_BASE, base);
            }
            base |= IA32_APIC_BASE_ENABLE;
            base &= !IA32_APIC_BASE_X2APIC;
            wrmsr(MSR_IA32_APIC_BASE, base);

            let apic = Apic {
                mmio_base_virt: virt,
                mode: ApicMode::XApic,
            };
            apic.enable_software_apic();
            apic
        }
    }

    /// Reconstruct a handle from the cached virtual base. Used by APs
    /// (which don't run `init` themselves) to read their own APIC ID.
    /// Returns `None` if `init` hasn't been called yet.
    #[inline]
    pub fn current() -> Option<Self> {
        match LAPIC_MODE.load(Ordering::Acquire) {
            APIC_MODE_X2APIC => Some(Apic {
                mmio_base_virt: 0,
                mode: ApicMode::X2Apic,
            }),
            APIC_MODE_XAPIC => {
                let v = LAPIC_VIRT_BASE.load(Ordering::Acquire);
                if v == 0 {
                    None
                } else {
                    Some(Apic {
                        mmio_base_virt: v,
                        mode: ApicMode::XApic,
                    })
                }
            }
            _ => None,
        }
    }

    #[inline]
    fn enable_software_apic(&self) {
        self.write(REG_SVR, SVR_ENABLE | SVR_SPURIOUS_VECTOR);
        self.write(REG_TPR, 0);
    }

    #[inline]
    pub fn mode_label(&self) -> &'static str {
        match self.mode {
            ApicMode::XApic => "xAPIC",
            ApicMode::X2Apic => "x2APIC",
        }
    }

    #[inline]
    pub fn can_address(&self, target_apic_id: u32) -> bool {
        self.mode == ApicMode::X2Apic || target_apic_id <= 0xFF
    }

    /// Read the current core's APIC ID (xAPIC: 8 bits in bits 31:24).
    ///
    /// **Per-core register.** Even though every core sees the same
    /// physical MMIO address for `REG_ID`, hardware routes the read
    /// to that core's local APIC unit — so this call returns the
    /// running core's ID, not the BSP's.
    #[inline]
    pub fn id(&self) -> u32 {
        match self.mode {
            ApicMode::XApic => (self.read(REG_ID) >> 24) & 0xFF,
            ApicMode::X2Apic => unsafe { rdmsr(MSR_X2APIC_ID) as u32 },
        }
    }

    /// Read the APIC version register. Bits 7:0 = version, 23:16 =
    /// max LVT entry count - 1. Used for diagnostics only.
    #[inline]
    pub fn version(&self) -> u32 {
        match self.mode {
            ApicMode::XApic => self.read(REG_VERSION),
            ApicMode::X2Apic => unsafe { rdmsr(MSR_X2APIC_VERSION) as u32 },
        }
    }

    /// Read the Error Status Register, which records IPI delivery
    /// errors. Read-then-write-clear is the canonical pattern; we
    /// expose just the read for diagnostics around INIT-SIPI-SIPI.
    #[inline]
    pub fn esr(&self) -> u32 {
        match self.mode {
            ApicMode::XApic => self.read(REG_ESR),
            ApicMode::X2Apic => unsafe { rdmsr(MSR_X2APIC_ESR) as u32 },
        }
    }

    /// Clear the ESR (write any value).
    #[inline]
    pub fn clear_esr(&self) {
        self.write(REG_ESR, 0);
    }

    // ─────────────────────────────────────────────────────────────
    // IPI primitives
    // ─────────────────────────────────────────────────────────────

    /// Send a raw IPI. `target_apic_id` goes into ICR_HIGH bits 31:24;
    /// `icr_low_bits` is the full ICR_LOW value including delivery
    /// mode, vector, level, trigger, shorthand.
    ///
    /// **Spin on delivery status.** In xAPIC MMIO mode, the ICR
    /// "Delivery Status" bit must be polled until clear before issuing
    /// a new IPI. Otherwise the second IPI can overwrite the first while
    /// the fabric is still routing it.
    #[inline]
    pub fn send_ipi_raw(&self, target_apic_id: u32, icr_low_bits: u32) {
        match self.mode {
            ApicMode::XApic => {
                // Wait for previous IPI to finish dispatching.
                while (self.read(REG_ICR_LOW) & DELIVERY_STATUS) != 0 {
                    core::hint::spin_loop();
                }
                // Write high half first (destination), then low half (which
                // triggers dispatch).
                self.write(REG_ICR_HIGH, (target_apic_id & 0xFF) << 24);
                self.write(REG_ICR_LOW, icr_low_bits);
            }
            ApicMode::X2Apic => unsafe {
                let icr = ((target_apic_id as u64) << 32) | (icr_low_bits as u64);
                wrmsr(MSR_X2APIC_ICR, icr);
            },
        }
        // Don't wait for completion here — the caller knows the right
        // post-delay (10 ms after INIT, 200 µs between SIPIs).
    }

    /// Issue the INIT-assert IPI to the target AP. This starts the AP
    /// reset sequence; the matching INIT-deassert below leaves the AP
    /// in wait-for-SIPI state.
    pub fn send_init_assert(&self, target_apic_id: u32) {
        let icr_low = DELIVERY_INIT
            | DEST_MODE_PHYSICAL
            | LEVEL_ASSERT
            | TRIGGER_LEVEL // level-triggered for INIT
            | SHORTHAND_NONE;
        self.send_ipi_raw(target_apic_id, icr_low);
    }

    /// Issue the INIT-deassert IPI. This is part of the canonical
    /// x86 AP wake-up sequence (Linux names it "INIT, INIT, STARTUP"):
    /// assert INIT, deassert INIT, then send one or two SIPIs.
    pub fn send_init_deassert(&self, target_apic_id: u32) {
        let icr_low = DELIVERY_INIT | DEST_MODE_PHYSICAL | TRIGGER_LEVEL | SHORTHAND_NONE;
        self.send_ipi_raw(target_apic_id, icr_low);
    }

    /// Issue a SIPI (Startup IPI) to the target AP. `vector` is the
    /// trampoline page (physical address >> 12). Valid range: 0x00..0xFF.
    ///
    /// Per the x86 startup IPI protocol, the AP begins executing at
    /// `CS:IP = (vector << 8):0000` in real mode after receiving SIPI.
    /// For a trampoline at physical 0x8000, vector = 0x08.
    pub fn send_sipi(&self, target_apic_id: u32, vector: u8) {
        // Linux sends STARTUP with just delivery-mode + vector; INIT
        // uses level/trigger bits, but SIPI does not need them and some
        // x2APIC implementations are stricter about reserved fields.
        let icr_low = DELIVERY_STARTUP | DEST_MODE_PHYSICAL | SHORTHAND_NONE | (vector as u32);
        self.send_ipi_raw(target_apic_id, icr_low);
    }

    // ─────────────────────────────────────────────────────────────
    // High-level AP wake-up
    // ─────────────────────────────────────────────────────────────

    /// Drive a single AP through the AMD64/x86 INIT-SIPI-SIPI sequence.
    ///
    /// `target_apic_id` = the AP's APIC ID from the ACPI MADT.
    /// `trampoline_page` = physical page number of the AP trampoline
    /// (e.g. 0x08 if the trampoline is loaded at physical 0x8000).
    ///
    /// **Sequence (x86 multiple-processor initialization):**
    /// 1. Clear ESR.
    /// 2. Send INIT assert IPI.
    /// 3. Wait 10 ms.
    /// 4. Send INIT deassert IPI.
    /// 5. Send first SIPI.
    /// 6. Wait 200 µs (allow AP to start trampoline).
    /// 7. Send second SIPI.
    /// 8. Wait 200 µs.
    ///
    /// The second SIPI is a safety net — if the AP missed the first
    /// (rare, but part of the canonical x86 startup protocol), the
    /// second wakes it. APs that already accepted the first ignore
    /// duplicates.
    ///
    /// **Delay implementation.** All gaps are timed against `rdtsc`,
    /// not the legacy PIT. Under UEFI on modern x86 platforms the
    /// 8259 PIC is masked and PIT IRQs never reach the kernel — using
    /// `pit::ticks()` here would hang the BSP forever on the very first
    /// AP. TSC is monotonic and runs regardless of interrupt routing,
    /// so it is the only delay primitive safe to use during AP wake-up.
    pub fn boot_ap(&self, target_apic_id: u32, trampoline_page: u8) {
        self.clear_esr();

        // ─── Step 1: INIT assert IPI ───
        self.send_init_assert(target_apic_id);

        // ─── Step 2: wait 10 ms ───
        delay_microseconds(10_000);

        // ─── Step 3: INIT deassert IPI ───
        self.send_init_deassert(target_apic_id);

        // ─── Step 4: first SIPI ───
        self.clear_esr();
        self.send_sipi(target_apic_id, trampoline_page);

        // ─── Step 5: wait ~200 µs ───
        delay_microseconds(200);

        // ─── Step 6: second SIPI (safety net) ───
        self.clear_esr();
        self.send_sipi(target_apic_id, trampoline_page);

        // ─── Step 7: wait another ~200 µs for the AP to consume ───
        delay_microseconds(200);

        // The AP has been told to start. The BSP now spins on
        // smp::REGISTERED_CORES expecting the AP to register.
    }

    // ─────────────────────────────────────────────────────────────
    // Low-level MMIO
    // ─────────────────────────────────────────────────────────────

    /// MMIO 32-bit read at register offset `off`.
    #[inline(always)]
    fn read(&self, off: usize) -> u32 {
        match self.mode {
            ApicMode::XApic => {
                // SAFETY: caller guaranteed at `init` that the mapping is
                // valid and `off` is a defined register per the SDM.
                unsafe {
                    core::ptr::read_volatile((self.mmio_base_virt + off as u64) as *const u32)
                }
            }
            ApicMode::X2Apic => unsafe { rdmsr(x2apic_msr_for_mmio_offset(off)) as u32 },
        }
    }

    /// MMIO 32-bit write at register offset `off`.
    #[inline(always)]
    fn write(&self, off: usize, val: u32) {
        match self.mode {
            ApicMode::XApic => unsafe {
                core::ptr::write_volatile((self.mmio_base_virt + off as u64) as *mut u32, val)
            },
            ApicMode::X2Apic => unsafe {
                wrmsr(x2apic_msr_for_mmio_offset(off), val as u64);
            },
        }
    }
}

#[inline]
unsafe fn rdmsr(msr: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    core::arch::asm!(
        "rdmsr",
        in("ecx") msr,
        out("eax") lo,
        out("edx") hi,
        options(nomem, nostack, preserves_flags),
    );
    ((hi as u64) << 32) | (lo as u64)
}

#[inline]
unsafe fn wrmsr(msr: u32, value: u64) {
    core::arch::asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") value as u32,
        in("edx") (value >> 32) as u32,
        options(nomem, nostack, preserves_flags),
    );
}

#[inline]
fn x2apic_msr_for_mmio_offset(off: usize) -> u32 {
    match off {
        REG_ID => MSR_X2APIC_ID,
        REG_VERSION => MSR_X2APIC_VERSION,
        REG_TPR => MSR_X2APIC_TPR,
        REG_EOI => MSR_X2APIC_EOI,
        REG_SVR => MSR_X2APIC_SVR,
        REG_ESR => MSR_X2APIC_ESR,
        REG_ICR_LOW | REG_ICR_HIGH => MSR_X2APIC_ICR,
        _ => MSR_X2APIC_ESR,
    }
}

/// Enable the already-selected APIC mode on the currently running CPU.
///
/// The BSP chooses x2APIC vs xAPIC in [`Apic::init`]. APs wake from INIT
/// with CPU-local APIC state reset, so they must mirror that mode before
/// reading their own APIC ID or entering the shared SMP worker loop.
pub unsafe fn enable_current_core() {
    match LAPIC_MODE.load(Ordering::Acquire) {
        APIC_MODE_X2APIC => {
            let mut base = rdmsr(MSR_IA32_APIC_BASE);
            base |= IA32_APIC_BASE_ENABLE | IA32_APIC_BASE_X2APIC;
            wrmsr(MSR_IA32_APIC_BASE, base);
            wrmsr(MSR_X2APIC_SVR, (SVR_ENABLE | SVR_SPURIOUS_VECTOR) as u64);
            wrmsr(MSR_X2APIC_TPR, 0);
        }
        APIC_MODE_XAPIC => {
            if let Some(apic) = Apic::current() {
                let mut base = rdmsr(MSR_IA32_APIC_BASE);
                if (base & IA32_APIC_BASE_X2APIC) != 0 {
                    base &= !(IA32_APIC_BASE_ENABLE | IA32_APIC_BASE_X2APIC);
                    wrmsr(MSR_IA32_APIC_BASE, base);
                }
                base |= IA32_APIC_BASE_ENABLE;
                base &= !IA32_APIC_BASE_X2APIC;
                wrmsr(MSR_IA32_APIC_BASE, base);
                apic.enable_software_apic();
            }
        }
        _ => {}
    }
}

// ─────────────────────────────────────────────────────────────────
// Delay helpers
// ─────────────────────────────────────────────────────────────────

/// Cached TSC frequency for delay calibration. Initialised lazily on
/// the first call to [`tsc_cycles_per_us`]. `0` while uninitialised.
///
/// We cache because [`cycles::tsc_hz`] issues `cpuid` (a serializing
/// instruction); doing it once per AP boot is fine, but the SIPI-gap
/// path inside `boot_ap` runs three delays per AP and the cached path
/// is one MSR-free atomic load.
static TSC_CYCLES_PER_US: AtomicU64 = AtomicU64::new(0);

/// Return cycles-per-microsecond, computed once and cached.
#[inline]
fn tsc_cycles_per_us() -> u64 {
    let cached = TSC_CYCLES_PER_US.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    let hz = cycles::tsc_hz();
    // Floor of hz/1_000_000 — guaranteed >= 1 on any plausible CPU.
    let per_us = (hz / 1_000_000).max(1);
    TSC_CYCLES_PER_US.store(per_us, Ordering::Relaxed);
    per_us
}

/// Busy-wait approximately `us` microseconds via `rdtsc`.
///
/// All AP wake-up timing flows through this helper. We deliberately
/// avoid the legacy PIT: under UEFI on AMD EPYC (and any platform that
/// disables the 8259 PIC in favour of the IOAPIC), IRQ 0 is masked
/// and `pit::ticks()` never advances — a PIT-based delay would hang
/// the BSP indefinitely on the very first AP's INIT-SIPI gap.
///
/// `rdtsc` is monotonic and runs regardless of interrupt-controller
/// state, so it is the only safe primitive here. The SDM specifies
/// the gaps (10 ms after INIT, 200 µs between SIPIs) as **minimums**;
/// modest over-delay from rdtsc-loop overhead is fine.
fn delay_microseconds(us: u64) {
    let per_us = tsc_cycles_per_us();
    // Saturating mul because we never want to wrap to a tiny budget.
    let budget_cycles: u64 = per_us.saturating_mul(us);
    let start = cycles::rdtsc_serialized();
    loop {
        let now = cycles::rdtsc_serialized();
        if now.wrapping_sub(start) >= budget_cycles {
            break;
        }
        core::hint::spin_loop();
    }
}

// ─────────────────────────────────────────────────────────────────
// Convenience: pure read of own APIC ID without holding a handle
// ─────────────────────────────────────────────────────────────────

/// Read the current core's APIC ID, returning `0` if the APIC hasn't
/// been mapped yet (BSP pre-`init`). APs call this in `ap_entry` to
/// get their identity for [`crate::smp::allocate_core_index`].
#[inline]
pub fn current_apic_id() -> u32 {
    match Apic::current() {
        Some(a) => a.id(),
        None => 0,
    }
}
