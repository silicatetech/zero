// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU identification via CPUID instruction.
//!
//! Reads vendor string, brand string (model name), core/thread counts,
//! cache info, and feature flags. Outputs a human-readable summary
//! to serial for boot diagnostics and benchmark context.
//!
//! Used by the benchmark suite to display hardware info alongside
//! performance results — essential for the KVM screenshot demo.

use crate::arch;
use core::fmt::Write;

/// CPU information gathered from CPUID.
pub struct CpuInfo {
    /// Vendor string, e.g. "AuthenticAMD" or "GenuineIntel" (12 bytes)
    pub vendor: [u8; 12],
    /// Brand string, e.g. "AMD EPYC 9354P 32-Core Processor" (48 bytes)
    pub brand: [u8; 48],
    /// Brand string valid length
    pub brand_len: usize,
    /// Base frequency in MHz (from CPUID leaf 0x16, 0 if unavailable)
    pub base_freq_mhz: u32,
    /// Max frequency in MHz (from CPUID leaf 0x16, 0 if unavailable)
    pub max_freq_mhz: u32,
    /// Max standard CPUID leaf
    pub max_leaf: u32,
    /// Max extended CPUID leaf
    pub max_ext_leaf: u32,
    /// Feature flags
    pub has_sse: bool,
    pub has_sse2: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_xsave: bool,
    pub has_x2apic: bool,
    /// APIC initial count (logical processors visible to this core)
    pub logical_cores: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct TopologyIds {
    pub valid: bool,
    pub compute_unit_id: u32,
    pub node_id: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct SimdState {
    pub xcr0: u64,
    pub avx_enabled: bool,
    pub avx512_enabled: bool,
}

/// Execute CPUID with given leaf, return (eax, ebx, ecx, edx).
#[inline]
fn cpuid(leaf: u32) -> (u32, u32, u32, u32) {
    let eax: u32;
    let ebx: u32;
    let ecx: u32;
    let edx: u32;
    unsafe {
        core::arch::asm!(
            "push rbx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "pop rbx",
            inout("eax") leaf => eax,
            inout("ecx") 0u32 => ecx,
            ebx_out = out(reg) ebx,
            out("edx") edx,
        );
    }
    (eax, ebx, ecx, edx)
}

/// Execute CPUID with given leaf and sub-leaf.
#[inline]
fn cpuid_sub(leaf: u32, sub: u32) -> (u32, u32, u32, u32) {
    let eax: u32;
    let ebx: u32;
    let ecx: u32;
    let edx: u32;
    unsafe {
        core::arch::asm!(
            "push rbx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "pop rbx",
            inout("eax") leaf => eax,
            inout("ecx") sub => ecx,
            ebx_out = out(reg) ebx,
            out("edx") edx,
        );
    }
    (eax, ebx, ecx, edx)
}

/// Read CPU information from CPUID and return structured data.
pub fn detect() -> CpuInfo {
    let mut info = CpuInfo {
        vendor: [0u8; 12],
        brand: [0u8; 48],
        brand_len: 0,
        base_freq_mhz: 0,
        max_freq_mhz: 0,
        max_leaf: 0,
        max_ext_leaf: 0,
        has_sse: false,
        has_sse2: false,
        has_avx: false,
        has_avx2: false,
        has_avx512f: false,
        has_xsave: false,
        has_x2apic: false,
        logical_cores: 1,
    };

    // Leaf 0: Vendor string + max standard leaf
    let (max_leaf, ebx, ecx, edx) = cpuid(0);
    info.max_leaf = max_leaf;

    // Vendor string is in EBX:EDX:ECX (yes, that order)
    info.vendor[0..4].copy_from_slice(&ebx.to_le_bytes());
    info.vendor[4..8].copy_from_slice(&edx.to_le_bytes());
    info.vendor[8..12].copy_from_slice(&ecx.to_le_bytes());

    // Leaf 1: Feature flags + logical core count
    if max_leaf >= 1 {
        let (_, ebx1, ecx1, edx1) = cpuid(1);
        info.has_sse = (edx1 & (1 << 25)) != 0;
        info.has_sse2 = (edx1 & (1 << 26)) != 0;
        info.has_x2apic = (ecx1 & (1 << 21)) != 0;
        info.has_xsave = (ecx1 & (1 << 26)) != 0;
        info.has_avx = (ecx1 & (1 << 28)) != 0;
        // Logical processor count (bits 23:16 of EBX)
        info.logical_cores = (ebx1 >> 16) & 0xFF;
        if info.logical_cores == 0 {
            info.logical_cores = 1;
        }
    }

    // Leaf 7: Extended features (AVX2, AVX-512)
    if max_leaf >= 7 {
        let (_, ebx7, _, _) = cpuid_sub(7, 0);
        info.has_avx2 = (ebx7 & (1 << 5)) != 0;
        info.has_avx512f = (ebx7 & (1 << 16)) != 0;
    }

    // Leaf 0x16: Processor frequency info
    // EAX=base MHz, EBX=max MHz, ECX=bus MHz
    if max_leaf >= 0x16 {
        let (base_mhz, max_mhz, _, _) = cpuid(0x16);
        info.base_freq_mhz = base_mhz;
        info.max_freq_mhz = max_mhz;
    }

    // Extended leaf 0x80000000: max extended leaf
    let (max_ext, _, _, _) = cpuid(0x80000000);
    info.max_ext_leaf = max_ext;

    // Extended leaves 0x80000002-0x80000004: Brand string (48 bytes)
    if max_ext >= 0x80000004 {
        for i in 0u32..3 {
            let leaf = 0x80000002 + i;
            let (a, b, c, d) = cpuid(leaf);
            let base = (i as usize) * 16;
            info.brand[base..base + 4].copy_from_slice(&a.to_le_bytes());
            info.brand[base + 4..base + 8].copy_from_slice(&b.to_le_bytes());
            info.brand[base + 8..base + 12].copy_from_slice(&c.to_le_bytes());
            info.brand[base + 12..base + 16].copy_from_slice(&d.to_le_bytes());
        }
        // Find actual length (strip trailing nulls/spaces)
        info.brand_len = 48;
        while info.brand_len > 0
            && (info.brand[info.brand_len - 1] == 0 || info.brand[info.brand_len - 1] == b' ')
        {
            info.brand_len -= 1;
        }
    }

    info
}

/// CPUID leaf-1 x2APIC support bit. Kept as a tiny helper so the APIC
/// driver does not need to duplicate CPUID plumbing.
#[inline]
pub fn x2apic_supported() -> bool {
    let (max_leaf, _, _, _) = cpuid(0);
    if max_leaf < 1 {
        return false;
    }
    let (_, _, ecx1, _) = cpuid(1);
    (ecx1 & (1 << 21)) != 0
}

/// AMD CPUID topology identifiers for the *currently executing* CPU.
///
/// CPUID Fn8000_001E is the AMD topology-extension leaf. Zero uses
/// it only for scheduling policy: it lets the SMP matmul dispatcher
/// select one hardware thread per physical compute unit when SMT hurts
/// an AVX-512-heavy workload. The arithmetic contract stays unchanged:
/// each output row is still reduced by exactly one logical CPU.
#[inline]
pub fn topology_ids() -> TopologyIds {
    let (max_ext, _, _, _) = cpuid(0x80000000);
    if max_ext < 0x8000001e {
        return TopologyIds {
            valid: false,
            compute_unit_id: 0,
            node_id: 0,
        };
    }

    let (_, ebx, ecx, _) = cpuid(0x8000001e);
    TopologyIds {
        valid: true,
        compute_unit_id: ebx & 0xff,
        node_id: ecx & 0xff,
    }
}

#[inline]
unsafe fn read_cr0() -> u64 {
    let value: u64;
    core::arch::asm!("mov {}, cr0", out(reg) value, options(nomem, nostack, preserves_flags));
    value
}

#[inline]
unsafe fn write_cr0(value: u64) {
    core::arch::asm!("mov cr0, {}", in(reg) value, options(nomem, nostack, preserves_flags));
}

#[inline]
unsafe fn read_cr4() -> u64 {
    let value: u64;
    core::arch::asm!("mov {}, cr4", out(reg) value, options(nomem, nostack, preserves_flags));
    value
}

#[inline]
unsafe fn write_cr4(value: u64) {
    core::arch::asm!("mov cr4, {}", in(reg) value, options(nomem, nostack, preserves_flags));
}

#[inline]
unsafe fn xgetbv(index: u32) -> u64 {
    let eax: u32;
    let edx: u32;
    core::arch::asm!(
        "xgetbv",
        in("ecx") index,
        out("eax") eax,
        out("edx") edx,
        options(nomem, nostack, preserves_flags),
    );
    ((edx as u64) << 32) | eax as u64
}

#[inline]
unsafe fn xsetbv(index: u32, value: u64) {
    let eax = value as u32;
    let edx = (value >> 32) as u32;
    core::arch::asm!(
        "xsetbv",
        in("ecx") index,
        in("eax") eax,
        in("edx") edx,
        options(nomem, nostack, preserves_flags),
    );
}

/// Enable x87/SSE/AVX/AVX-512 architectural state on the current core.
///
/// Firmware and bootloaders are inconsistent about how much SIMD state
/// they leave enabled. The AVX-512 inference path and the AP worker
/// loops need an explicit OS-owned XCR0 setup on every core, otherwise
/// the first ZMM instruction can raise #UD/#NM after SMP bring-up.
///
/// # Safety
///
/// Must run at CPL0 on an x86_64 CPU. The caller should invoke it once
/// on the BSP and once on every AP before executing SIMD code.
pub unsafe fn enable_fpu_simd() -> SimdState {
    const CR0_MP: u64 = 1 << 1;
    const CR0_EM: u64 = 1 << 2;
    const CR0_TS: u64 = 1 << 3;
    const CR4_OSFXSR: u64 = 1 << 9;
    const CR4_OSXMMEXCPT: u64 = 1 << 10;
    const CR4_OSXSAVE: u64 = 1 << 18;

    const XCR0_X87: u64 = 1 << 0;
    const XCR0_SSE: u64 = 1 << 1;
    const XCR0_YMM: u64 = 1 << 2;
    const XCR0_OPMASK: u64 = 1 << 5;
    const XCR0_ZMM_HI256: u64 = 1 << 6;
    const XCR0_HI16_ZMM: u64 = 1 << 7;

    let mut cr0 = read_cr0();
    cr0 |= CR0_MP;
    cr0 &= !(CR0_EM | CR0_TS);
    write_cr0(cr0);

    let mut cr4 = read_cr4();
    cr4 |= CR4_OSFXSR | CR4_OSXMMEXCPT;

    let (max_leaf, _, _, _) = cpuid(0);
    let mut xcr0 = XCR0_X87 | XCR0_SSE;
    let mut avx_enabled = false;
    let mut avx512_enabled = false;

    if max_leaf >= 1 {
        let (_, _, ecx1, _) = cpuid(1);
        let has_xsave = (ecx1 & (1 << 26)) != 0;
        let has_avx = (ecx1 & (1 << 28)) != 0;
        let has_avx512f = if max_leaf >= 7 {
            let (_, ebx7, _, _) = cpuid_sub(7, 0);
            (ebx7 & (1 << 16)) != 0
        } else {
            false
        };

        if has_xsave {
            cr4 |= CR4_OSXSAVE;
            write_cr4(cr4);

            let supported_xcr0 = if max_leaf >= 0x0d {
                let (eax, _, _, edx) = cpuid_sub(0x0d, 0);
                ((edx as u64) << 32) | (eax as u64)
            } else {
                XCR0_X87 | XCR0_SSE
            };

            if (supported_xcr0 & (XCR0_X87 | XCR0_SSE)) == (XCR0_X87 | XCR0_SSE) {
                if has_avx && (supported_xcr0 & XCR0_YMM) != 0 {
                    xcr0 |= XCR0_YMM;
                }

                let avx512_xstate = XCR0_YMM | XCR0_OPMASK | XCR0_ZMM_HI256 | XCR0_HI16_ZMM;
                if has_avx512f && (supported_xcr0 & avx512_xstate) == avx512_xstate {
                    xcr0 |= avx512_xstate;
                }

                xsetbv(0, xcr0);
                xcr0 = xgetbv(0);

                avx_enabled = (xcr0 & XCR0_YMM) != 0;
                avx512_enabled = (xcr0 & avx512_xstate) == avx512_xstate;
            } else {
                xcr0 = xgetbv(0);
            }
        } else {
            write_cr4(cr4);
        }
    } else {
        write_cr4(cr4);
    }

    SimdState {
        xcr0,
        avx_enabled,
        avx512_enabled,
    }
}

/// Print CPU info to serial console.
pub fn print_info(info: &CpuInfo) {
    let _ = writeln!(arch::serial::Serial, "");
    let _ = writeln!(arch::serial::Serial, "=== CPU Information ===");

    // Vendor
    let _ = write!(arch::serial::Serial, "Vendor:   ");
    for &b in info.vendor.iter() {
        if b != 0 {
            let _ = write!(arch::serial::Serial, "{}", b as char);
        }
    }
    let _ = writeln!(arch::serial::Serial, "");

    // Brand string (model name)
    let _ = write!(arch::serial::Serial, "Model:    ");
    for &b in info.brand[..info.brand_len].iter() {
        if b >= 0x20 && b < 0x7F {
            let _ = write!(arch::serial::Serial, "{}", b as char);
        }
    }
    let _ = writeln!(arch::serial::Serial, "");

    // Frequencies
    if info.base_freq_mhz > 0 {
        let _ = writeln!(arch::serial::Serial, "Base:     {} MHz", info.base_freq_mhz);
    }
    if info.max_freq_mhz > 0 {
        let _ = writeln!(arch::serial::Serial, "Max:      {} MHz", info.max_freq_mhz);
    }

    // Logical cores
    let _ = writeln!(
        arch::serial::Serial,
        "Cores:    {} logical",
        info.logical_cores
    );

    // Features
    let _ = write!(arch::serial::Serial, "Features: ");
    if info.has_sse {
        let _ = write!(arch::serial::Serial, "SSE ");
    }
    if info.has_sse2 {
        let _ = write!(arch::serial::Serial, "SSE2 ");
    }
    if info.has_avx {
        let _ = write!(arch::serial::Serial, "AVX ");
    }
    if info.has_avx2 {
        let _ = write!(arch::serial::Serial, "AVX2 ");
    }
    if info.has_avx512f {
        let _ = write!(arch::serial::Serial, "AVX-512F ");
    }
    if info.has_xsave {
        let _ = write!(arch::serial::Serial, "XSAVE ");
    }
    if info.has_x2apic {
        let _ = write!(arch::serial::Serial, "x2APIC ");
    }
    let _ = writeln!(arch::serial::Serial, "");

    let _ = writeln!(
        arch::serial::Serial,
        "CPUID:    max_leaf=0x{:x}, max_ext=0x{:x}",
        info.max_leaf,
        info.max_ext_leaf
    );
    let _ = writeln!(arch::serial::Serial, "===========================");
    let _ = writeln!(arch::serial::Serial, "");
}
