// SPDX-License-Identifier: AGPL-3.0-or-later
//! Memory management — Stage 2 (V3 arena-based).
//!
//! Wired via `memory::init(boot_info)`. Three responsibilities:
//!
//! 1. Build the active [`OffsetPageTable`] over the level-4 table the
//!    bootloader left in CR3. We assume (and the bootloader config in
//!    `main.rs` enforces) that all physical memory is linearly mapped
//!    at a known higher-half offset, so that Phys → Virt is addition.
//!
//! 2. Walk the bootloader's memory map and expose every
//!    [`MemoryRegionKind::Usable`] frame through a
//!    [`BootInfoFrameAllocator`]. Bump-style, no free list.
//!    Structurally this is a drop-in slot for a real allocator later.
//!
//! 3. Map 4 MiB of virtual address space at [`KERNEL_ARENA_START`],
//!    one frame per page, with `PRESENT | WRITABLE`. Hand that range
//!    to a [`KernelArenaInner`] (newtype around [`FixedArena`])
//!    exposed as the global [`KERNEL_ARENA`].
//!    No `#[global_allocator]` — V3 architecture mandates explicit
//!    arena allocation instead of implicit heap (Box/Vec/String).
//!
//! The module is intentionally flat — no sub-modules, no trait
//! indirection beyond what the x86_64 crate asks for — because Stage 2
//! is the foundation, not the architecture.

use quarks_arena::{ArenaError, FixedArena};
#[cfg(target_arch = "x86_64")]
use bootloader_api::info::{MemoryRegion, MemoryRegionKind, MemoryRegions};
#[cfg(target_arch = "x86_64")]
use bootloader_api::BootInfo;
#[allow(unused_imports)]
use core::fmt::Write;
use core::slice;
#[cfg(target_arch = "x86_64")]
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;
#[cfg(target_arch = "x86_64")]
use x86_64::{
    structures::paging::{
        mapper::{MapToError, MappedFrame, Translate, TranslateResult},
        FrameAllocator, Mapper, OffsetPageTable, Page, PageTable, PageTableFlags, PhysFrame,
        Size2MiB, Size4KiB,
    },
    PhysAddr, VirtAddr,
};

/// Bootloader's physical-memory linear-mapping offset, stashed by
/// [`init`] so non-init code paths (DMA-driver bring-up, virt→phys
/// translation) can construct page-table walkers without re-plumbing
/// `boot_info`.
#[cfg(target_arch = "x86_64")]
pub static PHYS_OFFSET: AtomicU64 = AtomicU64::new(0);

/// Returns the cached physical-memory linear-mapping offset. Returns
/// `None` if [`init`] has not run yet.
#[cfg(target_arch = "x86_64")]
pub fn phys_offset() -> Option<u64> {
    let v = PHYS_OFFSET.load(Ordering::Acquire);
    if v == 0 {
        None
    } else {
        Some(v)
    }
}

/// Walk the active page tables to translate a kernel virtual address
/// into a physical address. Required by DMA drivers that must hand
/// physical addresses to hardware (e.g. e1000 descriptor rings).
///
/// Returns `None` if the address is not currently mapped or if
/// [`init`] has not run yet.
#[cfg(target_arch = "x86_64")]
pub fn virt_to_phys(va: u64) -> Option<u64> {
    let offset = phys_offset()?;
    let mapper = unsafe { build_mapper(VirtAddr::new(offset)) };
    mapper.translate_addr(VirtAddr::new(va)).map(|p| p.as_u64())
}

#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone, Debug)]
pub struct MappingInfo {
    pub phys_addr: u64,
    pub frame_size: u64,
    pub flags_bits: u64,
    pub pwt: bool,
    pub pcd: bool,
    pub huge: bool,
}

/// Inspect the active page-table entry for a virtual address.
///
/// This is a diagnostic-only helper for Cherry performance bring-up. It
/// lets the Zero control plane verify that model bytes and arenas are
/// mapped as write-back cacheable memory (`PWT=0`, `PCD=0`) and whether
/// the mapping uses 4 KiB, 2 MiB, or 1 GiB pages.
#[cfg(target_arch = "x86_64")]
pub fn mapping_info(va: u64) -> Option<MappingInfo> {
    let offset = phys_offset()?;
    let mapper = unsafe { build_mapper(VirtAddr::new(offset)) };
    match mapper.translate(VirtAddr::new(va)) {
        TranslateResult::Mapped {
            frame,
            offset,
            flags,
        } => Some(MappingInfo {
            phys_addr: frame.start_address().as_u64().wrapping_add(offset),
            frame_size: match frame {
                MappedFrame::Size4KiB(_) => 4 * 1024,
                MappedFrame::Size2MiB(_) => 2 * 1024 * 1024,
                MappedFrame::Size1GiB(_) => 1024 * 1024 * 1024,
            },
            flags_bits: flags.bits(),
            pwt: flags.contains(PageTableFlags::WRITE_THROUGH),
            pcd: flags.contains(PageTableFlags::NO_CACHE),
            huge: flags.contains(PageTableFlags::HUGE_PAGE),
        }),
        TranslateResult::NotMapped | TranslateResult::InvalidFrameAddress(_) => None,
    }
}

/// Translate `va` to its backing physical address, or `None` if not
/// mapped. Thin wrapper around [`mapping_info`] that drops the
/// auxiliary fields the typical caller does not need.
#[cfg(target_arch = "x86_64")]
pub fn virt_to_phys_pa(va: u64) -> Option<u64> {
    mapping_info(va).map(|m| m.phys_addr)
}

/// Outcome of a hugepage-view promotion attempt — see
/// [`contiguous_phys_view`].
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone, Debug)]
pub enum HugepageViewOutcome {
    /// Region is physically contiguous *and* the phys-mem-linear map
    /// uses a page size larger than 4 KiB (2 MiB or 1 GiB). Caller
    /// should switch to the returned VA for cache-/TLB-friendly access.
    Promoted {
        new_va: u64,
        phys_base: u64,
        page_size: u64,
    },
    /// Region is contiguous but the phys-mem-linear map at the target
    /// VA is still 4 KiB pages. No win possible without doing the
    /// re-mapping ourselves.
    ContiguousButSmallPages { phys_base: u64, page_size: u64 },
    /// Region is not physically contiguous — every chunk would need
    /// its own mapping, which defeats the hugepage promotion.
    NotContiguous { broken_at_offset: u64 },
    /// The original VA does not map (or `phys_offset` is unset).
    Unmapped,
}

/// Attempt to expose a virtually-mapped region through the bootloader's
/// phys-mem-linear map instead of its original VA. The bootloader 0.11
/// `physical_memory` mapping uses 1 GiB pages where alignment permits
/// (see [`init`] discussion); when a region (typically the ramdisk that
/// carries our GGUF model) is contiguous in physical memory, we can
/// re-expose its bytes via that already-hugepage-mapped alternate VA
/// without copying anything or installing new page tables.
///
/// For a 1.2 GB model this collapses TLB pressure from ~300 K 4 KiB
/// entries to ~600 2 MiB entries (or 2 1 GiB entries), saving the
/// dominant cost on weight-streaming matmul kernels.
///
/// Walks the source region in `2 MiB` strides to confirm physical
/// contiguity without paying the 300 K-iteration cost of a 4 KiB walk.
///
/// # Safety
///
/// `src_va`..`src_va + len` must be fully mapped, readable bytes for
/// the kernel's lifetime. The returned VA, if any, points to the same
/// physical bytes — caller may use it as a `&'static [u8]` of the same
/// length, **provided the caller treats the original VA as no longer
/// authoritative** (writes through one VA become visible to reads
/// through the other only via DRAM, with WB cache they coalesce).
#[cfg(target_arch = "x86_64")]
pub fn contiguous_phys_view(src_va: u64, len: usize) -> HugepageViewOutcome {
    let Some(phys_off) = phys_offset() else {
        return HugepageViewOutcome::Unmapped;
    };
    let Some(phys_base) = virt_to_phys_pa(src_va) else {
        return HugepageViewOutcome::Unmapped;
    };

    // Stride contiguity check. 2 MiB is large enough that 1.2 GB takes
    // ~600 iterations (microseconds), and small enough that any real
    // discontinuity is caught quickly.
    let stride: u64 = 2 * 1024 * 1024;
    let mut offset: u64 = stride;
    while offset < len as u64 {
        let expected = phys_base + offset;
        match virt_to_phys_pa(src_va + offset) {
            Some(actual) if actual == expected => {}
            _ => {
                return HugepageViewOutcome::NotContiguous {
                    broken_at_offset: offset,
                }
            }
        }
        offset += stride;
    }
    // Final-byte check catches discontinuities in the tail.
    let tail_off = (len as u64).saturating_sub(1);
    if tail_off > offset.saturating_sub(stride) {
        let expected = phys_base + tail_off;
        match virt_to_phys_pa(src_va + tail_off) {
            Some(actual) if actual == expected => {}
            _ => {
                return HugepageViewOutcome::NotContiguous {
                    broken_at_offset: tail_off,
                }
            }
        }
    }

    let new_va = phys_off.wrapping_add(phys_base);
    let page_size = match mapping_info(new_va) {
        Some(info) => info.frame_size,
        None => return HugepageViewOutcome::Unmapped,
    };

    if page_size > 4096 {
        HugepageViewOutcome::Promoted {
            new_va,
            phys_base,
            page_size,
        }
    } else {
        HugepageViewOutcome::ContiguousButSmallPages {
            phys_base,
            page_size,
        }
    }
}

/// Return the pre-reserved low bootstrap page-table frames.
///
/// The x86_64 AP trampoline first enables long mode from 16/32-bit
/// transition code. Its initial CR3 path must therefore be safely
/// reachable below 4 GiB on large AMD EPYC hosts. These frames are
/// reserved during [`init`] before arena mapping consumes the general
/// frame stream, so the trampoline never depends on fixed low memory.
#[cfg(target_arch = "x86_64")]
pub fn low_bootstrap_frames() -> Option<[u64; 3]> {
    let mut frames = [0u64; 3];
    let mut i = 0usize;
    while i < LOW_BOOTSTRAP_FRAMES.len() {
        let frame = LOW_BOOTSTRAP_FRAMES[i].load(Ordering::Acquire);
        if frame == 0 {
            return None;
        }
        frames[i] = frame;
        i += 1;
    }
    Some(frames)
}

// ── MMIO mapping (high-BAR support) ──────────────────────────────────
//
// The bootloader linearly maps RAM at `PHYS_OFFSET`, but BARs above
// the RAM ceiling are NOT in that map. Most importantly, the Intel
// X710 NIC on the Cherry Server places BAR0 at physical
// 0x300_8180_0000 (~3 TiB) which is far above any RAM mapping the
// bootloader produces. Drivers that try to access such BARs through
// `phys_offset() + bar` will page-fault.
//
// `map_mmio` allocates virtual address space inside a reserved PML4
// slot, then walks the page tables to install 4 KiB mappings with
// strong uncacheable attributes (PCD | PWT). The frame allocator
// preserved from boot is used only for intermediate page-table
// frames — the target physical pages are the BAR itself, not RAM.

/// Base virtual address of the MMIO carve-out — populated by
/// [`init`] once a safe PML4 slot has been chosen. `0` before init.
#[cfg(target_arch = "x86_64")]
pub static MMIO_REGION_BASE: AtomicU64 = AtomicU64::new(0);

/// Bump pointer into the MMIO carve-out. Equal to `MMIO_REGION_BASE`
/// at init, advanced (page-aligned) on each successful `map_mmio`.
#[cfg(target_arch = "x86_64")]
static MMIO_NEXT: AtomicU64 = AtomicU64::new(0);

/// One 512 GiB PML4 slot is plenty for every MMIO BAR a modern
/// server will ever expose. Bounded explicitly so a runaway caller
/// cannot scribble into the neighbouring slot.
#[cfg(target_arch = "x86_64")]
const MMIO_REGION_SIZE: u64 = 512 * 1024 * 1024 * 1024;

/// Persistent frame allocator state. Populated by [`init`] just
/// before it returns — the boot-time [`BootInfoFrameAllocator`]'s
/// `next` counter is handed off so subsequent `map_mmio` calls
/// continue past the frames already consumed by the arena setup.
///
/// Holds a raw pointer to the bootloader's memory-region table; the
/// referent lives at a 'static address inside the bootloader-provided
/// [`BootInfo`] (caller of [`init`] passes a `&'static mut BootInfo`),
/// but the function-level borrow we get does not advertise that, so
/// we drop down to raw pointers to keep [`init`]'s signature stable.
#[cfg(target_arch = "x86_64")]
static FRAME_ALLOCATOR: Mutex<Option<PostBootFrameAllocator>> = Mutex::new(None);

#[cfg(target_arch = "x86_64")]
static LOW_BOOTSTRAP_FRAMES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

#[cfg(target_arch = "x86_64")]
const LOW_BOOTSTRAP_MIN_PHYS: u64 = 0x0010_0000;

#[cfg(target_arch = "x86_64")]
const LOW_BOOTSTRAP_MAX_PHYS: u64 = 0x1_0000_0000;

/// Low physical frames claimed by architecture boot code before the
/// general frame allocator may use them. `0x8000` hosts the x86 AP
/// trampoline; if the bootloader reports low memory as usable, handing
/// this frame to an arena/page-table allocation would make AP startup
/// overwrite live kernel state.
#[cfg(target_arch = "x86_64")]
const RESERVED_BOOT_FRAMES: [u64; 1] = [0x8000];

#[cfg(target_arch = "x86_64")]
#[inline]
fn align_up_4k(addr: u64) -> u64 {
    (addr + 4095) & !4095
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn is_reserved_boot_frame(addr: u64) -> bool {
    let frame = addr & !0xfff;
    let mut i = 0usize;
    while i < RESERVED_BOOT_FRAMES.len() {
        if frame == RESERVED_BOOT_FRAMES[i] {
            return true;
        }
        i += 1;
    }
    false
}

/// Frame-allocator state preserved past [`init`]. Behaviour mirrors
/// [`BootInfoFrameAllocator`] (cursor + region-end fast path, cold
/// boundary-crossing scan) but stores raw pointers so it can sit in
/// a `'static` global without a lifetime parameter.
#[cfg(target_arch = "x86_64")]
struct PostBootFrameAllocator {
    regions_ptr: *const MemoryRegion,
    regions_len: usize,
    next: usize,
    cursor_addr: u64,
    cursor_region_end: u64,
    cursor_region_idx: usize,
}

#[cfg(target_arch = "x86_64")]
unsafe impl Send for PostBootFrameAllocator {}

#[cfg(target_arch = "x86_64")]
impl PostBootFrameAllocator {
    fn regions(&self) -> &[MemoryRegion] {
        // SAFETY: regions_ptr / regions_len were taken from a live
        // `&MemoryRegions` whose backing storage lives inside the
        // 'static BootInfo for the kernel's lifetime. We only ever
        // read.
        unsafe { slice::from_raw_parts(self.regions_ptr, self.regions_len) }
    }

    fn advance_to_next_region(&mut self) -> bool {
        let regions_len = self.regions_len;
        let mut idx = self.cursor_region_idx + 1;
        while idx < regions_len {
            // Re-borrow per iteration so the slice borrow does not
            // overlap the &mut self field assignments below.
            let (start, end, kind) = {
                let r = &self.regions()[idx];
                (r.start, r.end, r.kind)
            };
            if kind == MemoryRegionKind::Usable && end > start {
                self.cursor_addr = start;
                self.cursor_region_end = end;
                self.cursor_region_idx = idx;
                return true;
            }
            idx += 1;
        }
        self.cursor_region_end = 0;
        self.cursor_addr = 0;
        false
    }
}

#[cfg(target_arch = "x86_64")]
unsafe impl FrameAllocator<Size4KiB> for PostBootFrameAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        loop {
            if self.cursor_addr < self.cursor_region_end {
                if is_reserved_boot_frame(self.cursor_addr) {
                    self.cursor_addr += 4096;
                    continue;
                }
                let frame = PhysFrame::containing_address(PhysAddr::new(self.cursor_addr));
                self.cursor_addr += 4096;
                self.next += 1;
                return Some(frame);
            }
            if !self.advance_to_next_region() {
                return None;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl PostBootFrameAllocator {
    /// Reserve a single contiguous physical run of `byte_size` bytes
    /// (rounded up to 4 KiB) from a Usable region by skipping the
    /// frame-allocator cursor forward by that many bytes. Returns the
    /// reservation's first physical byte, or `None` if no Usable
    /// region has that much contiguous room left past the current
    /// cursor.
    ///
    /// This bypasses the per-frame [`allocate_frame`] loop — useful
    /// when the caller needs a multi-GiB virtually-contiguous buffer
    /// (e.g. the 584 GiB Kimi K2.6 weight arena). Each
    /// [`allocate_frame`] call after this one resumes from the
    /// post-reservation cursor, so the same frames cannot be handed
    /// out twice.
    /// Reserve one 2 MiB-aligned, 2 MiB-sized physical chunk from any
    /// Usable region the cursor can reach. Used by the scatter-gather
    /// arena mapper to harvest fragments across the EFI memory map.
    ///
    /// Returns the chunk's physical start byte, or `None` if no Usable
    /// region has a 2 MiB-aligned 2 MiB hole left past the cursor.
    /// Advances the cursor past the claimed chunk so subsequent
    /// allocations (small frames or further huge chunks) start from
    /// the next byte.
    fn reserve_huge_2mib_chunk(&mut self) -> Option<u64> {
        const HUGE_2MIB: u64 = 2 * 1024 * 1024;
        loop {
            // Skip reserved bootstrap frame (AP trampoline at 0x8000).
            if is_reserved_boot_frame(self.cursor_addr) {
                self.cursor_addr += 4096;
                continue;
            }
            // Round cursor up to next 2 MiB boundary. The gap (up to
            // 2 MiB-4 KiB) is wasted but the trade is intentional:
            // huge-page mappings require 2 MiB-aligned PA. The slack
            // is bounded by the number of Usable regions, not by the
            // arena size.
            let aligned_start = (self.cursor_addr + HUGE_2MIB - 1) & !(HUGE_2MIB - 1);
            let aligned_end = aligned_start.checked_add(HUGE_2MIB)?;
            if aligned_end <= self.cursor_region_end {
                // Defensive: confirm no reserved frame straddles the
                // chunk. RESERVED_BOOT_FRAMES is a static single-entry
                // list today; iteration is O(1).
                let mut straddles = false;
                for &rb in RESERVED_BOOT_FRAMES.iter() {
                    if rb >= aligned_start && rb < aligned_end {
                        straddles = true;
                        break;
                    }
                }
                if straddles {
                    // Skip past the chunk that overlaps the reserved
                    // frame and try the next 2 MiB slot.
                    self.cursor_addr = aligned_end;
                    continue;
                }
                self.cursor_addr = aligned_end;
                self.next = self.next.saturating_add((HUGE_2MIB / 4096) as usize);
                return Some(aligned_start);
            }
            // Current Usable region doesn't have a 2 MiB-aligned slot
            // left — advance to the next Usable region and retry.
            if !self.advance_to_next_region() {
                return None;
            }
        }
    }

    fn reserve_contiguous_bytes(&mut self, byte_size: usize) -> Option<u64> {
        let aligned = (byte_size as u64 + 4095) & !4095;
        if aligned == 0 {
            return None;
        }
        loop {
            // Skip any reserved bootstrap frame at the very start of
            // the candidate run — RESERVED_BOOT_FRAMES only contains
            // 0x8000 (low-MiB AP-trampoline frame), so this branch
            // only fires when the allocator cursor is in low memory.
            // In that case nudge past the reserved frame and retry.
            if is_reserved_boot_frame(self.cursor_addr) {
                self.cursor_addr += 4096;
                continue;
            }
            let end = self.cursor_addr.checked_add(aligned)?;
            if end <= self.cursor_region_end {
                // Defensive: make sure no reserved frame lives inside
                // the chosen run. RESERVED_BOOT_FRAMES is a static
                // single-entry list today so this is O(1).
                let mut straddles = false;
                for &rb in RESERVED_BOOT_FRAMES.iter() {
                    if rb >= self.cursor_addr && rb < end {
                        straddles = true;
                        break;
                    }
                }
                if straddles {
                    // Skip past the reserved frame and retry —
                    // re-aligning to 4 KiB. The candidate run would
                    // overlap a boot-reserved frame; pick a fresh
                    // start above it.
                    self.cursor_addr = (self.cursor_addr.max(0) | 0xFFF) + 1;
                    continue;
                }
                let phys_base = self.cursor_addr;
                self.cursor_addr = end;
                // Keep `next` roughly consistent with the number of
                // frames burned so capacity-accounting reads sane.
                self.next = self.next.saturating_add((aligned / 4096) as usize);
                return Some(phys_base);
            }
            if !self.advance_to_next_region() {
                return None;
            }
        }
    }
}

/// Reserve a virtually + physically contiguous region of `byte_size`
/// bytes (rounded up to 4 KiB) backed by the bootloader's already-
/// installed phys-linear map at `PHYS_OFFSET`. The caller can read /
/// write the region as a `&[u8]` / `&mut [u8]` at the returned
/// `virt_base` immediately — no `map_to` calls are issued because
/// every Usable physical byte is already mapped via 1 GiB / 2 MiB
/// huge pages by `init` + [`extend_phys_memory_mapping`].
///
/// Used by the NVMe model-loader path to land a multi-GiB GGUF
/// weight buffer in RAM at boot without burning 150 M page-table
/// entries on per-page mappings.
///
/// Returns `Some((phys_base, virt_base))` on success, `None` if no
/// Usable region has enough contiguous room past the frame
/// allocator's current cursor.
#[cfg(target_arch = "x86_64")]
pub fn alloc_contiguous_phys_linear(byte_size: usize) -> Option<(u64, u64)> {
    let mut guard = FRAME_ALLOCATOR.lock();
    let alloc = guard.as_mut()?;
    let phys_base = alloc.reserve_contiguous_bytes(byte_size)?;
    drop(guard);
    let phys_off = phys_offset()?;
    let virt_base = phys_off.wrapping_add(phys_base);
    Some((phys_base, virt_base))
}

/// Allocate a virtually-contiguous range backed by physically-scattered
/// 2 MiB chunks harvested across any number of Usable EFI memory
/// regions, and install the mapping with 2 MiB huge pages.
///
/// Used by the Kimi K2.6 weight loader. The EPYC EFI memory map
/// breaks high RAM into multiple Usable bands separated by ACPI
/// reclaim, BIOS-reserved (firmware encryption metadata), and the
/// occasional PCI hole; the largest single Usable region is typically
/// well below the 584 GiB Kimi K2.6 weight footprint even on a fully-
/// populated 768 GiB system, so the older [`alloc_contiguous_phys_linear`]
/// path fails before NVMe load ever begins.
///
/// This function instead:
///
/// 1. Picks `ceil(size / 512 GiB)` consecutive *unused* higher-half
///    PML4 slots — the virtual range stays contiguous to keep the
///    caller's `&[u8]` interface trivial.
/// 2. Walks the frame allocator, claiming one 2 MiB-aligned, 2 MiB
///    physical chunk per virtual huge page. Chunks may come from any
///    Usable region — fragmentation is fine.
/// 3. Installs each (virt_page → phys_chunk) 2 MiB huge-page mapping
///    with `PRESENT | WRITABLE`. Intermediate PDPT/PD frames are
///    pulled from the same frame allocator (4 KiB at a time).
///
/// Returns `(virt_base, mapped_bytes)` on success — `mapped_bytes` is
/// `byte_size` rounded up to the next 2 MiB boundary. Returns `None`
/// if no consecutive PML4 range is large enough, if the frame
/// allocator runs out of Usable 2 MiB chunks, or if `map_to` refuses
/// any single huge page (which would indicate a kernel-side mapping
/// bug, not a runtime condition).
///
/// The page-table memory footprint is ≈4 KiB per GiB of arena, i.e.
/// ~2.3 MiB for the full 584 GiB Kimi arena — negligible.
#[cfg(target_arch = "x86_64")]
pub fn alloc_scattered_virt_contiguous(byte_size: usize) -> Option<(u64, usize)> {
    const HUGE_2MIB: u64 = 2 * 1024 * 1024;

    if byte_size == 0 {
        return None;
    }
    let aligned = (byte_size as u64 + HUGE_2MIB - 1) & !(HUGE_2MIB - 1);
    let num_pages = (aligned / HUGE_2MIB) as usize;

    // PML4 slots needed for the virtual range. Each slot = 512 GiB =
    // 262144 × 2 MiB pages.
    let pages_per_slot = (PML4_ENTRY_SIZE / HUGE_2MIB) as usize;
    let needed_slots = (num_pages + pages_per_slot - 1) / pages_per_slot;

    let phys_off = phys_offset()?;
    let phys_off_va = VirtAddr::new(phys_off);

    let start_slot = pick_consecutive_pml4_slots(phys_off_va, needed_slots)?;
    let virt_base = pml4_slot_base(start_slot);

    let _ = writeln!(
        crate::arch::serial::Serial,
        "alloc_scattered_virt_contiguous: planning {} GiB across {} PML4 slot(s) starting at slot {} (virt_base=0x{:x})",
        aligned / (1024 * 1024 * 1024),
        needed_slots,
        start_slot,
        virt_base
    );

    let mut mapper = unsafe { build_mapper(phys_off_va) };

    let mut guard = FRAME_ALLOCATOR.lock();
    let fa = guard.as_mut()?;

    let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;

    let mut mapped_bytes: u64 = 0;
    let mut chunks_from_regions: usize = 0;
    let mut last_log_pct: u64 = 0;
    let mut first_phys: u64 = 0;
    let mut last_phys: u64 = 0;
    for i in 0..num_pages {
        let phys_addr = match fa.reserve_huge_2mib_chunk() {
            Some(p) => p,
            None => {
                let _ = writeln!(
                    crate::arch::serial::Serial,
                    "alloc_scattered_virt_contiguous: exhausted 2 MiB chunks after {} of {} pages ({} MiB / {} MiB) — insufficient Usable RAM",
                    i,
                    num_pages,
                    mapped_bytes / (1024 * 1024),
                    aligned / (1024 * 1024)
                );
                return None;
            }
        };
        if i == 0 {
            first_phys = phys_addr;
        }
        last_phys = phys_addr;
        chunks_from_regions += 1;

        let virt_addr = virt_base.wrapping_add((i as u64) * HUGE_2MIB);
        let page: Page<Size2MiB> = Page::containing_address(VirtAddr::new(virt_addr));
        let frame: PhysFrame<Size2MiB> = PhysFrame::containing_address(PhysAddr::new(phys_addr));
        unsafe {
            match mapper.map_to(page, frame, flags, fa) {
                Ok(t) => t.flush(),
                Err(e) => {
                    let _ = writeln!(
                        crate::arch::serial::Serial,
                        "alloc_scattered_virt_contiguous: map_to FAILED at page {} virt=0x{:x} phys=0x{:x}: {:?}",
                        i, virt_addr, phys_addr, e
                    );
                    return None;
                }
            }
        }
        mapped_bytes += HUGE_2MIB;

        // Coarse progress log every 5% — handy for a 584 GiB arena
        // where the per-page loop runs 299 008 iterations.
        let pct = (mapped_bytes * 100) / aligned;
        if pct >= last_log_pct + 5 {
            last_log_pct = pct;
            let _ = writeln!(
                crate::arch::serial::Serial,
                "alloc_scattered_virt_contiguous: {}% mapped ({} GiB / {} GiB)",
                pct,
                mapped_bytes / (1024 * 1024 * 1024),
                aligned / (1024 * 1024 * 1024)
            );
        }
    }

    let _ = writeln!(
        crate::arch::serial::Serial,
        "alloc_scattered_virt_contiguous: done — {} chunks claimed (first phys=0x{:x}, last phys=0x{:x}); flushing TLB",
        chunks_from_regions, first_phys, last_phys
    );
    // Conservative: flush the whole TLB once at the end of the loop.
    // The map_to+flush() calls above flushed each new page individually,
    // but a single global flush guarantees no stale entries from any
    // earlier walk through the freshly-installed PML4 sub-tree.
    x86_64::instructions::tlb::flush_all();

    Some((virt_base, mapped_bytes as usize))
}

/// Find `count` consecutive entirely-unused higher-half PML4 slots.
///
/// "Entirely unused" means `is_unused() == true` — the whole 512 GiB
/// sub-range is virgin, no huge pages or partial sub-tables. We
/// require *consecutive* slots so the virtual address space we hand
/// the caller stays contiguous across the PML4 boundary at
/// `slot * 512 GiB`.
///
/// Search order matches [`pick_arena_pml4_slot`]:
/// `[PREFERRED_ARENA_PML4 .. 512)` first, then `[256 .. PREFERRED_ARENA_PML4)`.
/// Does NOT wrap across the boundary because the two ranges may have
/// the bootloader's phys-linear map between them, and that map is
/// definitely not unused.
#[cfg(target_arch = "x86_64")]
fn pick_consecutive_pml4_slots(physical_memory_offset: VirtAddr, count: usize) -> Option<usize> {
    if count == 0 {
        return None;
    }
    let (l4_frame, _) = x86_64::registers::control::Cr3::read();
    let virt = physical_memory_offset + l4_frame.start_address().as_u64();
    // SAFETY: bootloader linearly maps all physical memory at
    // physical_memory_offset, so the L4 table is reachable here. Read-only.
    let l4_table: &PageTable = unsafe { &*virt.as_ptr::<PageTable>() };

    let mut scan = |lo: usize, hi: usize| -> Option<usize> {
        let mut consecutive = 0usize;
        let mut start: Option<usize> = None;
        for slot in lo..hi {
            if l4_table[slot].is_unused() {
                if consecutive == 0 {
                    start = Some(slot);
                }
                consecutive += 1;
                if consecutive >= count {
                    return start;
                }
            } else {
                consecutive = 0;
                start = None;
            }
        }
        None
    };

    if let Some(s) = scan(PREFERRED_ARENA_PML4, 512) {
        return Some(s);
    }
    scan(256, PREFERRED_ARENA_PML4)
}

/// Map a physical MMIO range into the kernel address space.
///
/// `phys_addr` and `size` may be unaligned; the call rounds the base
/// down and the size up to page granularity. Pages are installed with
/// `PRESENT | WRITABLE | NO_CACHE | WRITE_THROUGH` — the PCD+PWT
/// combination encodes "strong uncacheable" in the default x86 PAT
/// configuration. That is the right choice for device registers
/// where every load must reach the hardware and every store must
/// post in program order.
///
/// Returns a pointer into the MMIO carve-out whose offset within
/// the first page matches `phys_addr` (so reads/writes through the
/// returned pointer hit the same byte the BAR would have addressed).
///
/// # Errors
///
/// * [`MapToError::FrameAllocationFailed`] — the persisted frame
///   allocator ran out of usable RAM frames for intermediate page
///   tables. In practice this only happens if [`init`] never ran or
///   the system has pathologically little RAM.
/// * [`MapToError::ParentEntryHugePage`] — the chosen MMIO PML4 slot
///   was no longer virgin. Should never happen because [`init`]
///   reserves a fresh slot, but propagated for safety.
/// * [`MapToError::PageAlreadyMapped`] — the bump pointer wrapped
///   into an already-mapped page; signals a logic bug in the caller
///   (mapping too much MMIO) rather than a hardware condition.
#[cfg(target_arch = "x86_64")]
pub fn map_mmio(phys_addr: u64, size: usize) -> Result<*mut u8, MapToError<Size4KiB>> {
    let page_size = 4096u64;
    let phys_base = phys_addr & !(page_size - 1);
    let phys_offset_in_page = (phys_addr - phys_base) as usize;
    let total = (phys_offset_in_page + size + (page_size as usize - 1)) & !(page_size as usize - 1);
    let pages = (total as u64) / page_size;

    let region_base = MMIO_REGION_BASE.load(Ordering::Acquire);
    if region_base == 0 {
        // map_mmio called before memory::init — surface as
        // FrameAllocationFailed because we have nothing else.
        return Err(MapToError::FrameAllocationFailed);
    }

    // Reserve `pages` worth of virtual space via a CAS bump.
    let span = pages * page_size;
    let virt_base = loop {
        let cur = MMIO_NEXT.load(Ordering::Acquire);
        let new = cur
            .checked_add(span)
            .ok_or(MapToError::FrameAllocationFailed)?;
        if new > region_base + MMIO_REGION_SIZE {
            return Err(MapToError::PageAlreadyMapped(
                PhysFrame::containing_address(PhysAddr::new(phys_base)),
            ));
        }
        match MMIO_NEXT.compare_exchange(cur, new, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => break cur,
            Err(_) => continue,
        }
    };

    let phys_off_va = phys_offset().ok_or(MapToError::FrameAllocationFailed)?;
    let mut mapper = unsafe { build_mapper(VirtAddr::new(phys_off_va)) };

    let mut guard = FRAME_ALLOCATOR.lock();
    let frame_alloc = guard.as_mut().ok_or(MapToError::FrameAllocationFailed)?;

    let flags = PageTableFlags::PRESENT
        | PageTableFlags::WRITABLE
        | PageTableFlags::NO_CACHE
        | PageTableFlags::WRITE_THROUGH;

    for i in 0..pages {
        let page: Page<Size4KiB> =
            Page::containing_address(VirtAddr::new(virt_base + i * page_size));
        let frame: PhysFrame<Size4KiB> =
            PhysFrame::containing_address(PhysAddr::new(phys_base + i * page_size));
        unsafe {
            mapper.map_to(page, frame, flags, frame_alloc)?.flush();
        }
    }

    Ok((virt_base as usize + phys_offset_in_page) as *mut u8)
}

/// Result of an identity-mapping probe: did the page already exist,
/// or did we install it just now?
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone, Debug)]
pub enum IdentityMapOutcome {
    /// The page at the requested physical address was already identity-
    /// mapped (i.e. virt == phys) when we checked.
    Verified,
    /// The mapping was absent and we installed a fresh 4 KiB identity
    /// page via the persistent frame allocator.
    Installed,
}

/// Ensure that physical page `phys_addr` (4 KiB-aligned) is identity-
/// mapped in the active page tables (i.e. virtual address `phys_addr`
/// translates to physical address `phys_addr`).
///
/// Used by the AP trampoline: APs wake in real mode at a low physical
/// address, then execute through the first far-jump while paging is
/// on but `RIP` still equals the physical address. A working identity
/// mapping is a hard prerequisite — without it, the AP triple-faults
/// on its first instruction post-CR0.PG.
///
/// Returns:
/// * `Ok(Verified)` — the mapping already existed and points to itself.
/// * `Ok(Installed)` — a new 4 KiB PRESENT|WRITABLE mapping was created.
/// * `Err(MapToError)` — propagated from the page-table walk; treat as
///   "identity mapping not available" by the caller.
#[cfg(target_arch = "x86_64")]
pub fn ensure_identity_mapped_4k(
    phys_addr: u64,
) -> Result<IdentityMapOutcome, MapToError<Size4KiB>> {
    let phys_off_va = phys_offset().ok_or(MapToError::FrameAllocationFailed)?;

    // Step 1: check whether the VA == PA mapping already exists.
    {
        let mapper = unsafe { build_mapper(VirtAddr::new(phys_off_va)) };
        if let Some(translated) = mapper.translate_addr(VirtAddr::new(phys_addr)) {
            if translated.as_u64() == phys_addr {
                return Ok(IdentityMapOutcome::Verified);
            }
            // The VA is mapped, but to a different PA. This is a
            // conflict; surface it as PageAlreadyMapped so the caller
            // does not silently overwrite an unrelated mapping.
            return Err(MapToError::PageAlreadyMapped(
                PhysFrame::containing_address(PhysAddr::new(phys_addr)),
            ));
        }
    }

    // Step 2: install VA == PA. Use the persistent frame allocator for
    // intermediate page-table pages; the target frame is the one
    // identified by `phys_addr` itself.
    let mut mapper = unsafe { build_mapper(VirtAddr::new(phys_off_va)) };
    let mut guard = FRAME_ALLOCATOR.lock();
    let frame_alloc = guard.as_mut().ok_or(MapToError::FrameAllocationFailed)?;

    let page: Page<Size4KiB> = Page::containing_address(VirtAddr::new(phys_addr));
    let frame: PhysFrame<Size4KiB> = PhysFrame::containing_address(PhysAddr::new(phys_addr));
    let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
    unsafe { mapper.map_to(page, frame, flags, frame_alloc)?.flush() };
    Ok(IdentityMapOutcome::Installed)
}

#[allow(unused_imports)]
use crate::arch::serial;

/// Preferred PML4 slot for the arena block. PML4[288] is reserved
/// for arena use in the V3 layout (PML4[256] holds the bootloader's
/// 1 GiB huge-page physical-memory linear mapping).
///
/// On bare metal with >128 GiB of RAM the bootloader's
/// `Mapping::Dynamic` placements (boot_info, framebuffer, kernel
/// stack) can spill past the linear mapping and land in our preferred
/// slot — touching PML4[288] with 2 MiB huge pages and breaking
/// 4 KiB `map_to` calls with `ParentEntryHugePage`. [`init()`]
/// therefore probes PML4 in `[PREFERRED_ARENA_PML4 .. 512)` and uses
/// the first **entirely unused** slot it finds. An unused PML4 entry
/// guarantees the whole 512 GiB sub-range below it is virgin
/// (no huge pages, no partial mappings), which is the only
/// load-bearing invariant the arena mapper needs.
#[cfg(target_arch = "x86_64")]
const PREFERRED_ARENA_PML4: usize = 288;

/// Kernel-arena base address — populated by [`init()`] once a safe
/// PML4 slot has been chosen at runtime. `0` before init.
#[cfg(target_arch = "x86_64")]
pub static KERNEL_ARENA_START: AtomicU64 = AtomicU64::new(0);

/// Kernel-arena size: 4 MiB (V3 default per ARCHITECTURE.md Part 5,
/// matches `quarks_arena::KERNEL_ARENA_SIZE`).
#[cfg(target_arch = "x86_64")]
pub const KERNEL_ARENA_SIZE: usize = 4 * 1024 * 1024; // 4 MiB

/// Runtime-arena base address — populated by [`init()`]. `0` before
/// init.
#[cfg(target_arch = "x86_64")]
pub static RUNTIME_ARENA_START: AtomicU64 = AtomicU64::new(0);

/// Runtime-arena size: 8 MiB.
///
/// Raised from the V3 RUNTIME_ARENA_INITIAL of 2 MiB to support
/// multi-LLM architecture (Stage 17 swap-capability, ADR-028 §4).
/// The GGUF selective parser allocates Vec<GgufTensorInfo> + String
/// names + Vec<u64> dimensions via `#[global_allocator]` which routes
/// to this arena. Kimi K2.6 (61 layers, 384 experts) indexes ~1036
/// tensors requiring ~320 KiB of parser allocations alone. At 2 MiB
/// the arena OOM'd mid-parse, silently truncating the tensor index.
///
/// 8 MiB fits any model up to ~10 000 tensors with comfortable
/// headroom for validator, interpreter, and future Stage-12 consumers.
/// ARCHITECTURE.md V3.1 permits RUNTIME_ARENA_MAX = 64 MiB.
#[cfg(target_arch = "x86_64")]
pub const RUNTIME_ARENA_SIZE: usize = 8 * 1024 * 1024; // 8 MiB

/// Activation-arena base address — populated by [`init()`]. `0` before
/// init.
#[cfg(target_arch = "x86_64")]
pub static ACTIVATION_ARENA_START: AtomicU64 = AtomicU64::new(0);

/// Activation-arena size: 16 MiB per ADR-029 D4.
/// Per-token forward-pass working set ~700 KB; 16 MiB gives 24x margin.
///
/// Kimi K2.6 sizing note: scratch per token is dominated by router scores
/// (384 f32 = 1.5 KB), expert SwiGLU buffers (2048 × 4 ≈ 8 KB), MLA Q/K/V
/// decompressed (128 heads × 192 × 4 ≈ 96 KB) plus padded logits
/// (128 256 × 4 ≈ 502 KB). Worst-case ≈ 1.5 MiB — fits in current
/// `ACTIVATION_ARENA_SIZE` without change.
#[cfg(target_arch = "x86_64")]
pub const ACTIVATION_ARENA_SIZE: usize = 16 * 1024 * 1024; // 16 MiB

/// KV-Cache arena base address per ADR-030 — populated by [`init()`].
/// `0` before init.
#[cfg(target_arch = "x86_64")]
pub static KV_CACHE_ARENA_START: AtomicU64 = AtomicU64::new(0);

/// KV-Cache arena size.
///
/// # Default build (Qwen3 path)
///
/// 512 MiB per ADR-030. Qwen3-1.7B at 28 layers × 8 kv_heads × 128
/// head_dim × 4 B per f32 × 2 (K+V) needs ≈ 230 KiB / token; the 512
/// MiB arena therefore fits ≈ 2 340 tokens — more than the 2 182-token
/// streaming-mode capacity.
///
/// # `kimi-k26-arena` feature (1.5 GiB KV arena, compressed-latent)
///
/// Switches the KV arena to **1.5 GiB** for the production Kimi K2.6
/// deploy on Cherry EPYC 9575F. With the compressed-latent MlaKvCache
/// the per-token footprint is dominated by the low-rank latent rather
/// than the decompressed K/V tensors:
///
/// ```text
/// per_token = (kv_lora_rank + qk_rope_head_dim) × 4 bytes
///           = (512          + 64              ) × 4 = 2 304 B
/// per_layer (8 192 tokens) = 8 192 × 2 304 B  ≈ 18 MiB
/// total     (61 layers)    = 61    × 18 MiB   ≈ 1.07 GiB
/// ```
///
/// The 1.5 GiB ceiling leaves ~430 MiB headroom for arena bookkeeping
/// and alignment slop, and still fits the full 8K production context.
/// This is ~36× less than the prior decompressed-K/V layout. Per-head
/// k_nope and v are re-derived at attention time via W_kv_b expansion
/// — see `MlaKvCache::new` and `mla::mla_attention_single_token`.
///
/// # Runtime sizing
///
/// `run_forward_pass_deepseek2` in `inference.rs` computes
/// `max_tokens` from `KV_CACHE_ARENA_SIZE` divided by the per-token
/// per-layer footprint at boot, so the arena size and the realised
/// context length stay in lockstep without rebuilds. Reducing
/// `KV_CACHE_ARENA_SIZE` only reduces the context length the
/// deepseek2 path can carry; it never silently corrupts.
#[cfg(all(target_arch = "x86_64", not(feature = "kimi-k26-arena")))]
pub const KV_CACHE_ARENA_SIZE: usize = 512 * 1024 * 1024; // 512 MiB
#[cfg(all(target_arch = "x86_64", feature = "kimi-k26-arena"))]
pub const KV_CACHE_ARENA_SIZE: usize = 1536 * 1024 * 1024; // 1.5 GiB (compressed-latent)

// aarch64 reuses the same 512 MiB ceiling so cross-arch code paths
// (`inference::run_forward_pass_deepseek2`) can read this constant
// without an arch-specific dispatch. The actual aarch64 arena
// allocation happens in `init_aarch64` via local constants pinned to
// the same value — keep both in lockstep if you bump one.
#[cfg(target_arch = "aarch64")]
pub const KV_CACHE_ARENA_SIZE: usize = 512 * 1024 * 1024; // 512 MiB

/// Kimi K2.6 weight arena upper bound — informational only.
///
/// Kimi K2.6 Q4_K weights occupy ≈ 600 GiB on disk
/// (1 T params × ~0.6 B/param avg with router/shared weights). Runtime
/// requires the full weight set mmap'd or NVMe-paged. Actual allocation
/// happens via the model loader (NVMe path) and weight pinning logic at
/// boot; this constant exists so callers can sanity-check available RAM.
pub const KIMI_K26_WEIGHT_FOOTPRINT_GIB: usize = 602;

/// Kernel-arena wrapper that hides [`FixedArena::reset()`] from
/// kernel callers.
///
/// # Why this newtype exists
///
/// [`arena_static_alloc<T>`] returns `&'static mut T` references
/// that are sound only as long as `KERNEL_ARENA` is never reset.
/// If `FixedArena::reset()` were called, the bump pointer would
/// rewind to zero and subsequent allocations would overwrite memory
/// that previously-issued `'static` references still point at —
/// Undefined Behavior.
///
/// `KernelArenaInner` enforces this invariant at compile time by
/// not exposing `reset()`. Kernel code can only allocate, never
/// reset. The kernel arena lives for the kernel's lifetime; that
/// is the architectural invariant V3 implies and that this wrapper
/// enforces.
///
/// All useful methods of [`FixedArena`] (allocation, capacity
/// inspection) are forwarded.
pub struct KernelArenaInner(FixedArena);

#[allow(dead_code)]
impl KernelArenaInner {
    /// Construct from a backing slice. Not `pub` — only [`init()`]
    /// is allowed to construct one.
    fn new(backing: &'static mut [u8]) -> Self {
        Self(FixedArena::new(backing))
    }

    /// Allocate a value into the kernel arena.
    pub fn alloc<T>(&mut self, value: T) -> Result<&mut T, ArenaError> {
        self.0.alloc(value)
    }

    /// Allocate uninitialized space for `T`.
    #[allow(dead_code)] // API completeness — used by future stages
    pub fn alloc_uninit<T>(&mut self) -> Result<&mut core::mem::MaybeUninit<T>, ArenaError> {
        self.0.alloc_uninit::<T>()
    }

    /// Allocate a slice copy.
    pub fn alloc_slice_copy<T: Copy>(&mut self, slice: &[T]) -> Result<&mut [T], ArenaError> {
        self.0.alloc_slice_copy(slice)
    }

    /// Allocate a zeroed byte slice with an explicit alignment.
    ///
    /// DMA command buffers need stronger guarantees than Rust's
    /// element alignment can provide. Keep that requirement local to
    /// the arena instead of relying on incidental packing state.
    pub fn alloc_zeroed_aligned(
        &mut self,
        len: usize,
        align: usize,
    ) -> Result<&mut [u8], ArenaError> {
        let ptr = self.0.alloc_raw(len, align)?;
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0, len);
            Ok(slice::from_raw_parts_mut(ptr.as_ptr(), len))
        }
    }

    /// Allocate a string.
    #[allow(dead_code)] // API completeness — used by future stages
    pub fn alloc_str(&mut self, s: &str) -> Result<&mut str, ArenaError> {
        self.0.alloc_str(s)
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Bytes used so far.
    pub fn used(&self) -> usize {
        self.0.used()
    }

    /// Bytes available.
    #[allow(dead_code)] // API completeness — used by future stages
    pub fn available(&self) -> usize {
        self.0.available()
    }

    // NOTE: NO `reset()` method exposed. This is a load-bearing
    // invariant for `arena_static_alloc<T>` soundness. See
    // type-level documentation for rationale.
}

/// Global kernel arena. Initialized in [`init()`].
///
/// # V3 Conformance Note
///
/// V3 ARCHITECTURE.md Part 2 mandates two properties for Ring-0
/// arenas: allocations are "lock-free" and "each agent, each
/// compilation context, each request gets its own arena."
///
/// This single-Mutex global arena does not yet meet either property
/// in letter:
///
/// - **Lock-free:** The `spin::Mutex` introduces a lock. In Stage 9
///   this is uncontended in practice — there is exactly one caller
///   (the async executor's main thread). Phase 5 (Stage 12,
///   Sandboxing) introduces per-sandbox arenas, eliminating the
///   global lock. Until then, the `Mutex` is the pragmatic shim
///   that lets the executor expose the arena as a globally
///   accessible resource.
///
/// - **Per-agent ownership:** All Stage 9 callers (Stage-2 smoke
///   tests, Stage-3 task futures, the oneshot channel) share this
///   single arena. None of them are "agents" in the V3 sense —
///   they are boot-stub test scaffolding. Per-agent arenas are
///   structurally introduced when V3 agents become first-class in
///   Stage 12.
///
/// The Mutex wrapping `Option<KernelArenaInner>` enforces:
/// 1. The arena is uninitialized (`None`) before [`init()`] runs.
/// 2. Reset is impossible (see [`KernelArenaInner`]), so
///    `arena_static_alloc<T>` references remain sound for the
///    kernel's lifetime.
///
/// # Safety
///
/// Safe to access AFTER `memory::init()`. Before that, `lock()`
/// returns `None`.
pub static KERNEL_ARENA: Mutex<Option<KernelArenaInner>> = Mutex::new(None);

/// Runtime arena — backing store for `alloc::*` via the global
/// allocator.
///
/// Unlike [`KernelArenaInner`], this wrapper *does* expose `reset()`
/// because the runtime arena is per-compilation-context (V3 Part 2)
/// and may be reset between independent validator/interpreter runs.
///
/// Stage 9 does not yet reset; future stages (per-request validation,
/// repeated interpreter invocation) will.
pub struct RuntimeArenaInner(FixedArena);

#[allow(dead_code)]
impl RuntimeArenaInner {
    fn new(backing: &'static mut [u8]) -> Self {
        Self(FixedArena::new(backing))
    }

    /// Raw byte allocation — used by [`crate::arena_allocator`].
    pub fn alloc_raw(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        self.0.alloc_raw(size, align).ok().map(|ptr| ptr.as_ptr())
    }

    /// Reset the arena. Invalidates ALL previously-issued allocations.
    /// Caller must ensure no live references remain.
    #[allow(dead_code)] // Used by future stages
    pub fn reset(&mut self) {
        self.0.reset();
    }

    /// Bytes used.
    pub fn used(&self) -> usize {
        self.0.used()
    }

    /// Total capacity.
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }
}

/// Runtime arena — V3 per-compilation-context arena for
/// `alloc::*`-using code (validator, interpreter).
///
/// Initialized in [`init()`] alongside `KERNEL_ARENA`.
///
/// # V3 Conformance
///
/// V3 Part 2: "each agent, each compilation context, each request
/// gets its own arena." The runtime arena is the per-context arena
/// for validator and interpreter runs. Currently shared globally;
/// Stage 12 (Sandboxing) introduces per-sandbox runtime arenas.
pub static RUNTIME_ARENA: Mutex<Option<RuntimeArenaInner>> = Mutex::new(None);

/// Activation-arena wrapper — scratch memory for forward-pass operators.
///
/// Per ADR-029 D4: 16 MB FixedArena for per-token activation buffers.
/// Exposes `reset()` (called between forward-pass invocations) and
/// `alloc_f32_slice()` for typed allocation of f32 scratch buffers.
///
/// Unlike KERNEL_ARENA (never reset) and RUNTIME_ARENA (per-compilation),
/// ACTIVATION_ARENA is per-token: reset at the start of each token's
/// forward pass, reusing the same 16 MB for all operators.
pub struct ActivationArenaInner(FixedArena);

#[allow(dead_code)]
impl ActivationArenaInner {
    fn new(backing: &'static mut [u8]) -> Self {
        Self(FixedArena::new(backing))
    }

    /// Allocate a zeroed f32 slice from the activation arena.
    /// Caller-allocated output pattern per ADR-029 D5.
    pub fn alloc_f32_slice(&mut self, count: usize) -> Result<&'static mut [f32], ArenaError> {
        let byte_size = count * core::mem::size_of::<f32>();
        let align = core::mem::align_of::<f32>();
        let ptr = self.0.alloc_raw(byte_size, align)?;
        // Zero-initialize (forward-pass expects clean buffers)
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0, byte_size);
        }
        // SAFETY: ptr is aligned to f32, byte_size = count * 4,
        // backing memory is page-mapped and valid for kernel lifetime.
        // Lifetime extended to 'static via raw-pointer round-trip;
        // sound as long as caller respects reset() semantics.
        Ok(unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr() as *mut f32, count) })
    }

    /// Allocate a zeroed u32 slice from the activation arena. Used by
    /// MoE routing for the expert-indices buffer.
    pub fn alloc_u32_slice(&mut self, count: usize) -> Result<&'static mut [u32], ArenaError> {
        let byte_size = count * core::mem::size_of::<u32>();
        let align = core::mem::align_of::<u32>();
        let ptr = self.0.alloc_raw(byte_size, align)?;
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0, byte_size);
        }
        Ok(unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr() as *mut u32, count) })
    }

    /// Reset the arena. Invalidates ALL previously-issued allocations.
    /// Called at the start of each forward-pass token.
    pub fn reset(&mut self) {
        self.0.reset();
    }

    /// Bytes used.
    pub fn used(&self) -> usize {
        self.0.used()
    }

    /// Total capacity.
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }
}

/// Activation arena — per-token scratch memory for forward-pass operators.
///
/// Initialized in [`init()`] alongside KERNEL_ARENA and RUNTIME_ARENA.
/// Per ADR-029 D4: 16 MB capacity, bump-allocated, reset between tokens.
pub static ACTIVATION_ARENA: Mutex<Option<ActivationArenaInner>> = Mutex::new(None);

/// KV-Cache arena wrapper — per-session storage for K/V vectors.
///
/// Per ADR-030: 512 MiB FixedArena for layer-major KV storage.
/// Exposes `reset()` (called between inference sessions) and
/// `alloc_f32_slice()` for typed allocation of f32 buffers.
///
/// **Critical difference from ActivationArenaInner:** `alloc_f32_slice()`
/// does NOT zero-initialize allocated memory. This is a deliberate
/// Pillar 1 (Zero-Overhead) decision per ADR-030: 512 MiB memset
/// costs ~50 ms, and the KvCache invariant guarantees write-before-read
/// (store_kv writes data before get_*_slice reads it).
pub struct KvCacheArenaInner(FixedArena);

impl KvCacheArenaInner {
    fn new(backing: &'static mut [u8]) -> Self {
        Self(FixedArena::new(backing))
    }

    /// Allocate an f32 slice from the KV-Cache arena.
    ///
    /// Per ADR-030: **NO zero-initialization.** Saves ~50 ms at full
    /// 511.87 MiB pre-allocation. Caller MUST write before read — the
    /// KvCache wrapper enforces this via its API contract (token_idx +
    /// token_count discipline).
    ///
    /// # Safety contract
    ///
    /// Returned slice contains **uninitialized** memory. Reading before
    /// writing is undefined behavior. The KvCache wrapper's store_kv()
    /// + get_*_slice() API enforces write-before-read.
    #[allow(dead_code)] // API used in MP2.4 (GQA Attention)
    pub fn alloc_f32_slice(&mut self, count: usize) -> Result<&'static mut [f32], ArenaError> {
        let byte_size = count * core::mem::size_of::<f32>();
        let align = core::mem::align_of::<f32>();
        let ptr = self.0.alloc_raw(byte_size, align)?;
        // NO write_bytes — Pillar 1 (Zero-Overhead) over Pattern-Consistency.
        // KvCache invariant ensures write-before-read.
        Ok(unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr() as *mut f32, count) })
    }

    /// Reset the arena. Invalidates ALL previously-issued allocations.
    /// Called between inference sessions (new prompt).
    #[allow(dead_code)] // API used in MP2.4 (GQA Attention)
    pub fn reset(&mut self) {
        self.0.reset();
    }

    /// Bytes used.
    #[allow(dead_code)] // API used in MP2.4 (GQA Attention)
    pub fn used(&self) -> usize {
        self.0.used()
    }

    /// Total capacity.
    #[allow(dead_code)] // API used in MP2.4 (GQA Attention)
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }
}

/// KV-Cache arena — per-session storage for K/V attention vectors.
///
/// Initialized in [`init()`] alongside other arenas.
/// Per ADR-030: 512 MiB capacity, bump-allocated, reset between sessions.
pub static KV_CACHE_ARENA: Mutex<Option<KvCacheArenaInner>> = Mutex::new(None);

/// Bring memory management online. Call exactly once, after GDT/IDT.
///
/// Returns the number of usable frames reported by the bootloader so
/// callers can log it. On failure to map an arena page, returns the
/// underlying `MapToError`.
#[cfg(target_arch = "x86_64")]
pub fn init(boot_info: &mut BootInfo) -> Result<usize, MapToError<Size4KiB>> {
    // Step 1 — physical memory offset. The bootloader config in main.rs
    // pins this to a fixed canonical higher-half address; if it is
    // None here, something is very wrong and we cannot proceed.
    let physical_memory_offset = match boot_info.physical_memory_offset {
        bootloader_api::info::Optional::Some(o) => VirtAddr::new(o),
        bootloader_api::info::Optional::None => {
            panic!("bootloader did not provide physical_memory_offset");
        }
    };

    let _ = writeln!(
        serial::Serial,
        "Stage 2: higher-half mapping active, physical memory linearly mapped at offset {:#018x}",
        physical_memory_offset.as_u64(),
    );

    // Stash for later virt→phys translation (DMA drivers, etc).
    PHYS_OFFSET.store(physical_memory_offset.as_u64(), Ordering::Release);

    // Step 2 — mapper + frame allocator.
    let mut mapper = unsafe { build_mapper(physical_memory_offset) };
    let mut frame_allocator = unsafe { BootInfoFrameAllocator::new(&boot_info.memory_regions) };
    let usable_frames = frame_allocator.usable_frame_count();

    let _ = writeln!(
        serial::Serial,
        "Stage 2: frame allocator ready, {} usable frames",
        usable_frames,
    );

    // Reserve AP bootstrap page-table frames early and below 4 GiB.
    // This keeps SMP bring-up independent from the arena allocator's
    // later high-throughput frame stream and avoids fixed low-memory
    // assumptions on large EPYC machines.
    reserve_low_bootstrap_frames(&mut frame_allocator);

    // Step 2.25 — extend the bootloader's physical-memory linear map
    // beyond the first PML4 slot (512 GiB) if the host has more RAM.
    //
    // The bootloader's `Mapping::FixedAddress(0xFFFF_8000_0000_0000)`
    // anchors phys-mem at PML4[256]. A single PML4 entry covers 2^39
    // bytes = 512 GiB. Hosts with more RAM than that — notably the
    // Cherry EPYC 9575F (768 GiB) and dual-socket EPYC variants
    // (≥ 1.5 TiB) — need additional PML4 entries at slots 257, 258, …
    //
    // Safe to call even on small QEMU hosts: if `max_phys ≤ 512 GiB`
    // the function is a no-op because PML4[256] is already mapped.
    match unsafe {
        extend_phys_memory_mapping(
            &mut frame_allocator,
            physical_memory_offset,
            &boot_info.memory_regions,
        )
    } {
        Ok(slots_installed) => {
            let _ = writeln!(
                serial::Serial,
                "Stage 2: phys-mem linear map extended (added {} PML4 slots beyond bootloader)",
                slots_installed,
            );
        }
        Err(e) => {
            let _ = writeln!(
                serial::Serial,
                "Stage 2: phys-mem extension FAILED: {:?} — continuing with 512 GiB cap",
                e,
            );
        }
    }

    // Step 2.5 — choose a safe PML4 slot for the arena block.
    //
    // On bare metal with large RAM the bootloader's huge-page
    // mappings (1 GiB linear map + 2 MiB dynamic mappings) can
    // collide with hard-coded arena slots. Scan for the first
    // entirely unused PML4 entry — that guarantees the whole
    // 512 GiB sub-range is virgin (no huge pages, no partial
    // tables), which is the only invariant we need to safely
    // install 4 KiB arena pages.
    let arena_slot = pick_arena_pml4_slot(physical_memory_offset)
        .expect("no unused PML4 slot available for arena block");

    // A second virgin PML4 slot for the MMIO carve-out. Used by
    // [`map_mmio`] to host BARs that live above the bootloader's
    // linear RAM map (e.g. the X710 NIC's BAR0 at ~3 TiB).
    let mmio_slot = pick_mmio_pml4_slot(physical_memory_offset, arena_slot)
        .expect("no unused PML4 slot available for MMIO block");
    let mmio_base = pml4_slot_base(mmio_slot);
    MMIO_REGION_BASE.store(mmio_base, Ordering::Release);
    MMIO_NEXT.store(mmio_base, Ordering::Release);
    let _ = writeln!(
        serial::Serial,
        "Stage 2: MMIO carve-out at PML4[{}] base {:#018x} (512 GiB)",
        mmio_slot,
        mmio_base,
    );

    // All four arenas (~534 MiB total) live inside the chosen
    // 512 GiB PML4 slot. Layout is packed and 2 MiB-aligned so
    // every page-table walk stops at intermediate tables we own.
    let slot_base = pml4_slot_base(arena_slot);
    let kernel_start = slot_base;
    let runtime_start = kernel_start + KERNEL_ARENA_SIZE as u64;
    let activation_start = runtime_start + RUNTIME_ARENA_SIZE as u64;
    let kv_cache_start = activation_start + ACTIVATION_ARENA_SIZE as u64;

    KERNEL_ARENA_START.store(kernel_start, Ordering::Release);
    RUNTIME_ARENA_START.store(runtime_start, Ordering::Release);
    ACTIVATION_ARENA_START.store(activation_start, Ordering::Release);
    KV_CACHE_ARENA_START.store(kv_cache_start, Ordering::Release);

    let _ = writeln!(
        serial::Serial,
        "Stage 2: arena block at PML4[{}] base {:#018x} (preferred was [{}])",
        arena_slot,
        slot_base,
        PREFERRED_ARENA_PML4,
    );

    // Step 3 — kernel arena. Map every page in
    // [kernel_start, kernel_start + KERNEL_ARENA_SIZE) to a fresh
    // frame, WRITABLE, then construct a FixedArena over the mapped
    // region. Order matters: arena construction touches the backing
    // memory (FixedArena stores base pointer), so pages must be
    // mapped first.
    map_arena_range(
        &mut mapper,
        &mut frame_allocator,
        kernel_start,
        KERNEL_ARENA_SIZE,
    )?;

    // SAFETY: Three conditions must hold for this `from_raw_parts_mut`
    // with 'static lifetime to be sound:
    //
    // 1. EXCLUSIVE ACCESS: The region was just mapped by
    //    `map_arena_range` using freshly allocated frames inside a
    //    PML4 slot that was unused before this call. No other code
    //    in the kernel maps or accesses this region. The FixedArena
    //    becomes the sole owner via the Mutex.
    //
    // 2. LIFETIME: The mapped pages persist for the kernel's entire
    //    lifetime (we never unmap them). The kernel never returns,
    //    so 'static is satisfied.
    //
    // 3. VALIDITY: The frames came from BootInfoFrameAllocator which
    //    only yields MemoryRegionKind::Usable frames. The pages are
    //    mapped PRESENT | WRITABLE. FixedArena treats the backing as
    //    uninitialized (bump allocator writes before reading).
    let arena_backing: &'static mut [u8] =
        unsafe { slice::from_raw_parts_mut(kernel_start as *mut u8, KERNEL_ARENA_SIZE) };

    let arena = KernelArenaInner::new(arena_backing);
    *KERNEL_ARENA.lock() = Some(arena);

    let _ = writeln!(
        serial::Serial,
        "Stage 2: kernel arena online, {} KiB at {:#018x}",
        KERNEL_ARENA_SIZE / 1024,
        kernel_start,
    );

    // Step 4 — runtime arena. Packed immediately after the kernel
    // arena inside the same PML4 slot.
    map_arena_range(
        &mut mapper,
        &mut frame_allocator,
        runtime_start,
        RUNTIME_ARENA_SIZE,
    )?;

    // SAFETY: Same argument as KERNEL_ARENA backing — exclusive access,
    // page-mapped, never unmapped. Disjoint byte range inside the
    // chosen PML4 slot.
    let runtime_backing: &'static mut [u8] =
        unsafe { slice::from_raw_parts_mut(runtime_start as *mut u8, RUNTIME_ARENA_SIZE) };

    let runtime_arena = RuntimeArenaInner::new(runtime_backing);
    *RUNTIME_ARENA.lock() = Some(runtime_arena);

    let _ = writeln!(
        serial::Serial,
        "Stage 2: runtime arena online, {} KiB at {:#018x}",
        RUNTIME_ARENA_SIZE / 1024,
        runtime_start,
    );

    // Step 5 — activation arena (MP2.3a). 16 MiB for forward-pass
    // scratch buffers per ADR-029 D4.
    map_arena_range(
        &mut mapper,
        &mut frame_allocator,
        activation_start,
        ACTIVATION_ARENA_SIZE,
    )?;

    // SAFETY: Same argument as RUNTIME_ARENA — exclusive access,
    // page-mapped, never unmapped. Disjoint byte range inside the
    // chosen PML4 slot.
    let activation_backing: &'static mut [u8] =
        unsafe { slice::from_raw_parts_mut(activation_start as *mut u8, ACTIVATION_ARENA_SIZE) };

    let activation_arena = ActivationArenaInner::new(activation_backing);
    *ACTIVATION_ARENA.lock() = Some(activation_arena);

    let _ = writeln!(
        serial::Serial,
        "Stage 2: activation arena online, {} KiB at {:#018x}",
        ACTIVATION_ARENA_SIZE / 1024,
        activation_start,
    );

    // Step 6 — KV-Cache arena (ADR-030). 512 MiB for layer-major
    // K/V storage across inference sessions.
    let kv_init_start = crate::arch::cycles::rdtsc_serialized();

    let _ = writeln!(
        serial::Serial,
        "[KV_CACHE] Initializing 512 MiB at {:#018x}...",
        kv_cache_start,
    );

    map_arena_range(
        &mut mapper,
        &mut frame_allocator,
        kv_cache_start,
        KV_CACHE_ARENA_SIZE,
    )?;

    // SAFETY: Same argument as ACTIVATION_ARENA — exclusive access,
    // page-mapped, never unmapped. Disjoint byte range inside the
    // chosen PML4 slot.
    let kv_backing: &'static mut [u8] =
        unsafe { slice::from_raw_parts_mut(kv_cache_start as *mut u8, KV_CACHE_ARENA_SIZE) };

    let kv_arena = KvCacheArenaInner::new(kv_backing);
    *KV_CACHE_ARENA.lock() = Some(kv_arena);

    let kv_init_cycles = crate::arch::cycles::rdtsc_serialized() - kv_init_start;
    // Detect TSC frequency via CPUID (leaf 0x15 → leaf 0x16 → fallback).
    // Hard-coding 2.5 GHz mis-reported timings by ~25% on EPYC 9354P
    // (3.25 GHz) and on any host that isn't roughly 2.5 GHz.
    let tsc_hz = crate::arch::cycles::tsc_hz();
    let kv_millis = (kv_init_cycles * 1000) / tsc_hz;
    let tsc_mhz = tsc_hz / 1_000_000;

    let _ = write!(
        serial::Serial,
        "[KV_CACHE] Initialized: 512 MiB at {:#018x}, mapped in ",
        kv_cache_start,
    );
    let _ = write!(serial::Serial, "{}", kv_millis);
    let _ = write!(serial::Serial, " ms (TSC ~");
    let _ = write!(serial::Serial, "{}", tsc_mhz);
    let _ = writeln!(serial::Serial, " MHz)");

    let _ = write!(serial::Serial, "[KV_CACHE] Capacity: ");
    let _ = write!(
        serial::Serial,
        "{}",
        KV_CACHE_ARENA.lock().as_ref().unwrap().capacity()
    );
    let _ = writeln!(
        serial::Serial,
        " bytes (used: 0, zero-init: skipped per ADR-030)"
    );

    // Hand off the frame allocator's cursor state to the global
    // [`FRAME_ALLOCATOR`] so [`map_mmio`] can keep allocating
    // intermediate page-table frames after init returns. The raw
    // pointer is valid for 'static — `boot_info.memory_regions`'s
    // backing memory lives inside the bootloader-provided BootInfo
    // (always 'static in our entry-point signature).
    let regions: &MemoryRegions = &boot_info.memory_regions;
    let post = PostBootFrameAllocator {
        regions_ptr: regions.as_ptr(),
        regions_len: regions.len(),
        next: frame_allocator.next,
        cursor_addr: frame_allocator.cursor_addr,
        cursor_region_end: frame_allocator.cursor_region_end,
        cursor_region_idx: frame_allocator.cursor_region_idx,
    };
    *FRAME_ALLOCATOR.lock() = Some(post);

    Ok(usable_frames)
}

/// First PML4 slot used for the bootloader's physical-memory linear map.
/// Anchored by `Mapping::FixedAddress(0xFFFF_8000_0000_0000)` in main.rs;
/// any change to that constant must be mirrored here.
#[cfg(target_arch = "x86_64")]
const PHYS_MAP_BASE_PML4: usize = 256;

/// Bytes covered by a single PML4 entry: 2^39 = 512 GiB.
#[cfg(target_arch = "x86_64")]
const PML4_ENTRY_SIZE: u64 = 512 * 1024 * 1024 * 1024;

/// Maximum number of PML4 slots `extend_phys_memory_mapping` will install
/// beyond what the bootloader already provided. Each slot adds 512 GiB of
/// virtual address space, so 7 means we cover up to 4 TiB of physical
/// memory in total (PML4[256..263]). Cherry EPYC 9575F = 768 GiB needs
/// exactly one extra slot; dual-socket EPYC variants with ≥ 1.5 TiB need
/// two or three. The 4 TiB ceiling leaves room for future hardware
/// without burning PML4 entries that the arena/MMIO picker might want.
#[cfg(target_arch = "x86_64")]
const PHYS_MAP_MAX_EXTRA_SLOTS: usize = 7;

/// Extend the bootloader's physical-memory linear map to cover all RAM
/// reported by `MemoryRegions`, installing additional 512-GiB PML4
/// entries (each backed by a freshly-allocated PDPT with 512 × 1 GiB
/// huge pages) when the host has more than 512 GiB of physical memory.
///
/// # Why this exists
///
/// bootloader 0.11's `Mapping::FixedAddress` anchors phys-mem-linear at
/// a fixed virtual base but only installs one PML4 entry's worth of
/// mapping (512 GiB) before kernel entry. Zero's Cherry deployment
/// target — AMD EPYC 9575F with 768 GiB DDR5 — exceeds that bound and
/// would page-fault on any access to phys > 512 GiB through the linear
/// map. This function walks the memory map at boot and installs as many
/// PML4 entries as needed (capped at [`PHYS_MAP_MAX_EXTRA_SLOTS`]), each
/// covering 512 GiB via 1 GiB huge pages.
///
/// # Safety / invariants
///
/// * Mutates the active PML4 in place: callers must run **before** any
///   per-CPU TLB-sensitive work and **after** the frame allocator is
///   live (we need fresh frames for the PDPTs).
/// * Only installs entries where the existing PML4 slot `is_unused()`,
///   which respects whatever the bootloader already populated.
/// * Issues a global TLB flush only if at least one slot was added.
/// * 1 GiB huge pages require CPUID.80000001H:EDX.PDPE1GB = 1. All
///   Zero deployment targets (Zen 4, Sapphire Rapids+) provide this.
///
/// Returns the number of newly-installed PML4 slots, or a
/// `MapToError::FrameAllocationFailed` if the frame allocator ran out of
/// frames while building PDPTs (extremely unlikely with a healthy boot).
#[cfg(target_arch = "x86_64")]
unsafe fn extend_phys_memory_mapping(
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
    physical_memory_offset: VirtAddr,
    memory_regions: &MemoryRegions,
) -> Result<usize, MapToError<Size4KiB>> {
    // 1. Highest usable physical byte (exclusive end) reported by the
    //    bootloader. We extend coverage up to and including this byte.
    let max_phys_end: u64 = memory_regions
        .iter()
        .filter(|r| matches!(r.kind, MemoryRegionKind::Usable))
        .map(|r| r.end)
        .max()
        .unwrap_or(0);

    if max_phys_end == 0 {
        return Ok(0);
    }

    // 2. How many 512-GiB PML4 slots span [0 .. max_phys_end)?
    //    Ceil-divide, saturating to avoid pathological overflow.
    let needed_slots: u64 = (max_phys_end + PML4_ENTRY_SIZE - 1) / PML4_ENTRY_SIZE;

    // 3. Locate the active PML4 via CR3 + phys-mem-linear offset.
    let (l4_frame, _) = x86_64::registers::control::Cr3::read();
    let l4_virt = physical_memory_offset + l4_frame.start_address().as_u64();
    // SAFETY: bootloader linearly maps all physical memory at
    // physical_memory_offset, so the L4 table is reachable here.
    let l4_table: &mut PageTable = unsafe { &mut *l4_virt.as_mut_ptr::<PageTable>() };

    let mut installed: usize = 0;
    let huge_flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::HUGE_PAGE;
    let pdpt_flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;

    for offset in 0..needed_slots {
        if (installed as u64) >= PHYS_MAP_MAX_EXTRA_SLOTS as u64 {
            break;
        }
        let slot_idx = PHYS_MAP_BASE_PML4 + offset as usize;
        if slot_idx >= 512 {
            // Don't walk into PML4[512+] — impossible in 4-level paging.
            break;
        }

        // Bootloader-installed slot (PML4[256] in the common case) stays
        // untouched: it may already contain 2 MiB or 1 GiB mappings the
        // bootloader has chosen. We only add what is missing.
        if !l4_table[slot_idx].is_unused() {
            continue;
        }

        // 4. Allocate a fresh PDPT and zero it before populating.
        let pdpt_frame = frame_allocator
            .allocate_frame()
            .ok_or(MapToError::FrameAllocationFailed)?;
        let pdpt_virt = physical_memory_offset + pdpt_frame.start_address().as_u64();
        // SAFETY: the frame just left the bootloader's Usable list, no
        // one else holds a reference. Phys→virt translation is the
        // bootloader's linear map at physical_memory_offset.
        let pdpt: &mut PageTable = unsafe { &mut *pdpt_virt.as_mut_ptr::<PageTable>() };
        pdpt.zero();

        // 5. Fill all 512 PDPT entries with 1 GiB huge pages. Each entry
        //    maps virt slot_base + i*1GiB → phys offset*512GiB + i*1GiB.
        let phys_base = offset * PML4_ENTRY_SIZE;
        for i in 0..512 {
            let phys_1gib = phys_base + (i as u64) * (1024 * 1024 * 1024);
            pdpt[i].set_addr(PhysAddr::new(phys_1gib), huge_flags);
        }

        // 6. Install the PDPT into PML4. set_addr() takes a phys frame
        //    + non-huge flags (the leaf entries we just populated carry
        //    HUGE_PAGE; the PDPT itself doesn't).
        l4_table[slot_idx].set_addr(pdpt_frame.start_address(), pdpt_flags);
        installed += 1;
    }

    // 7. Flush TLB once if we actually changed anything. No-op for
    //    QEMU dev hosts with ≤ 512 GiB RAM.
    if installed > 0 {
        x86_64::instructions::tlb::flush_all();
    }

    Ok(installed)
}

/// Virtual base address of the 512 GiB region covered by PML4 slot
/// `slot` (in higher half, sign-extended to 64-bit canonical form).
#[cfg(target_arch = "x86_64")]
fn pml4_slot_base(slot: usize) -> u64 {
    // PML4 index occupies bits 47..39. For higher-half slots
    // (256..512), bit 47 is set, so the canonical form
    // sign-extends bits 48..63 to 1.
    let addr_48 = (slot as u64) << 39;
    if slot >= 256 {
        0xFFFF_0000_0000_0000 | addr_48
    } else {
        addr_48
    }
}

/// Scan the active PML4 for the first entirely-unused entry in
/// `[PREFERRED_ARENA_PML4 .. 512)`, then wrap around to
/// `[256 .. PREFERRED_ARENA_PML4)` if none found above.
///
/// An "entirely unused" PML4 entry (`is_unused() == true`) means the
/// whole 512 GiB sub-range below it is unmapped: no huge pages, no
/// partial PML3/PML2 tables, nothing the bootloader has touched. That
/// is the only invariant the 4 KiB arena mapper requires.
///
/// Returns `None` only if every higher-half PML4 entry is occupied —
/// in practice this never happens, but the caller treats it as a
/// hard failure.
#[cfg(target_arch = "x86_64")]
fn pick_arena_pml4_slot(physical_memory_offset: VirtAddr) -> Option<usize> {
    let (l4_frame, _) = x86_64::registers::control::Cr3::read();
    let virt = physical_memory_offset + l4_frame.start_address().as_u64();
    // SAFETY: bootloader linearly maps all physical memory at
    // physical_memory_offset, so the L4 table is reachable via this
    // virtual address. We only read entry presence — no mutation.
    let l4_table: &PageTable = unsafe { &*virt.as_ptr::<PageTable>() };

    // Prefer slots at or above PREFERRED_ARENA_PML4 (above the
    // bootloader's physical-memory mapping at PML4[256]). Fall back
    // to scanning the rest of higher half if needed.
    for slot in PREFERRED_ARENA_PML4..512 {
        if l4_table[slot].is_unused() {
            return Some(slot);
        }
    }
    for slot in 256..PREFERRED_ARENA_PML4 {
        if l4_table[slot].is_unused() {
            return Some(slot);
        }
    }
    None
}

/// Scan for an unused PML4 slot to host the MMIO carve-out, excluding
/// the slot already taken by the arena block. Same virginity guarantee
/// as [`pick_arena_pml4_slot`] — the whole 512 GiB sub-range must be
/// unmapped so [`map_mmio`] can install 4 KiB pages without colliding
/// with bootloader huge pages.
#[cfg(target_arch = "x86_64")]
fn pick_mmio_pml4_slot(physical_memory_offset: VirtAddr, exclude: usize) -> Option<usize> {
    let (l4_frame, _) = x86_64::registers::control::Cr3::read();
    let virt = physical_memory_offset + l4_frame.start_address().as_u64();
    // SAFETY: see `pick_arena_pml4_slot` — read-only walk of the
    // bootloader's linear physical-memory map.
    let l4_table: &PageTable = unsafe { &*virt.as_ptr::<PageTable>() };

    for slot in PREFERRED_ARENA_PML4..512 {
        if slot != exclude && l4_table[slot].is_unused() {
            return Some(slot);
        }
    }
    for slot in 256..PREFERRED_ARENA_PML4 {
        if slot != exclude && l4_table[slot].is_unused() {
            return Some(slot);
        }
    }
    None
}

/// Allocate a value into the kernel arena and return a `'static`
/// mutable reference.
///
/// This is the canonical way for kernel code to obtain stable
/// references to data structures that must outlive their allocation
/// context (e.g. futures stored in the task array, oneshot channels
/// passed across tasks).
///
/// # Lifetime soundness
///
/// The returned `&'static mut T` lifetime is sound because:
///
/// 1. `KERNEL_ARENA` is initialized once in [`init()`] and never
///    reset for the kernel's lifetime. Memory allocated from it
///    is valid until the kernel halts.
/// 2. `FixedArena::alloc()` returns a unique `&mut T` (no aliasing).
///    We extend the lifetime via raw-pointer round-trip without
///    introducing aliasing — there is exactly one returned reference
///    per allocation.
/// 3. The kernel never returns from `kernel_main()`. `'static` is
///    therefore equivalent to "kernel lifetime".
///
/// # Panics
///
/// Panics if called before `memory::init()` (`KERNEL_ARENA` is
/// `None`).
///
/// # Errors
///
/// Returns `ArenaError::OutOfMemory` if the arena cannot satisfy
/// the allocation.
#[cfg(target_arch = "x86_64")]
pub fn arena_static_alloc<T>(value: T) -> Result<&'static mut T, ArenaError> {
    let mut arena_guard = KERNEL_ARENA.lock();
    let arena = arena_guard
        .as_mut()
        .expect("kernel arena not initialized (memory::init must run first)");
    let r: &mut T = arena.alloc(value)?;
    // SAFETY: see function-level doc comment. The raw-pointer
    // round-trip extends the borrow lifetime from the Mutex guard's
    // scope to 'static. Soundness depends on three invariants:
    //   1. KERNEL_ARENA is never reset. This is enforced at compile
    //      time by KernelArenaInner not exposing reset().
    //   2. KernelArenaInner::alloc returns unique, non-overlapping
    //      refs (forwarded from FixedArena::alloc).
    //   3. Arena backing memory is page-mapped at init() and never
    //      unmapped (kernel never returns).
    Ok(unsafe { &mut *(r as *mut T) })
}

/// Construct an [`OffsetPageTable`] over the current level-4 page table.
///
/// # Safety
///
/// The caller must guarantee that the full physical memory is mapped
/// at `physical_memory_offset`. bootloader_api's config with
/// `Mapping::FixedAddress(physical_memory_offset)` makes this true.
#[cfg(target_arch = "x86_64")]
unsafe fn build_mapper(physical_memory_offset: VirtAddr) -> OffsetPageTable<'static> {
    let (l4_frame, _) = x86_64::registers::control::Cr3::read();
    let phys = l4_frame.start_address();
    let virt = physical_memory_offset + phys.as_u64();
    let l4_table: &mut PageTable = &mut *virt.as_mut_ptr();
    OffsetPageTable::new(l4_table, physical_memory_offset)
}

/// Map `[start, start + size)` page-by-page to fresh 4 KiB frames,
/// marked `PRESENT | WRITABLE`.
///
/// Any page-fault on an arena access after this step is a logic bug
/// in the arena or in user code, not in the mapping. Caller must
/// ensure the virtual range sits inside an unused PML4 slot — see
/// [`pick_arena_pml4_slot`].
#[cfg(target_arch = "x86_64")]
fn map_arena_range(
    mapper: &mut impl Mapper<Size4KiB>,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
    start: u64,
    size: usize,
) -> Result<(), MapToError<Size4KiB>> {
    let arena_start = VirtAddr::new(start);
    let arena_end = arena_start + (size - 1) as u64;
    let first_page: Page<Size4KiB> = Page::containing_address(arena_start);
    let last_page: Page<Size4KiB> = Page::containing_address(arena_end);

    let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;

    for page in Page::range_inclusive(first_page, last_page) {
        let frame = frame_allocator
            .allocate_frame()
            .ok_or(MapToError::FrameAllocationFailed)?;
        unsafe { mapper.map_to(page, frame, flags, frame_allocator)?.flush() };
    }

    Ok(())
}

// ---- Frame allocator ----------------------------------------------

/// Bump-style frame allocator that walks the bootloader's memory map.
///
/// ## Performance (Sub-MP-A refactor, post-6dbb407)
///
/// `allocate_frame()` is TRUE O(1) amortized via a caching cursor that
/// maintains a direct physical-address pointer + cached region end-addr.
/// Hot path = pure pointer arithmetic (`cursor_addr += 4096`). Cold path
/// (region boundary crossing) is O(R) where R = number of memory regions,
/// but each region is traversed at most once across the allocator's
/// entire lifetime.
///
/// Pre-refactor: `usable_frames().nth(next)` rebuilt the iterator and
/// scanned N elements on the Nth call — O(N) per call, O(N²) total.
/// Empirically measured on QEMU x86_64 8G (discovery/kv-cache-frame-perf):
///   64 MB = 0.29s, 128 MB = 0.94s, 256 MB = 3.47s, 512 MB = 14.3s.
///
/// ## Design discipline
///
/// - Only [`MemoryRegionKind::Usable`] regions are considered.
/// - No free path. Frames are never returned.
/// - NO recursion — flat iterative loops only (Ring-0 stack safety).
/// - State-safe lazy init: if `next > 0` at first `allocate_frame()`,
///   cursor fast-forwards past already-allocated frames.
/// - Pillar 1 (Zero-Overhead) compliant: hot path is pure arithmetic.
///
/// Anyone holding `&mut impl FrameAllocator<Size4KiB>` does not care
/// which concrete type is behind it, so the swap is a drop-in later.
#[cfg(target_arch = "x86_64")]
pub struct BootInfoFrameAllocator<'a> {
    memory_regions: &'a MemoryRegions,
    /// Cumulative count of frames allocated. Preserved from pre-refactor
    /// for diagnostics + serves as fast-forward target for lazy init.
    next: usize,
    /// Direct physical address of the next frame to allocate.
    /// Hot path: return frame at this addr, then += 4096.
    cursor_addr: u64,
    /// End address (exclusive) of the current usable region.
    /// Hot path check: if cursor_addr < cursor_region_end → allocate.
    cursor_region_end: u64,
    /// Index into memory_regions for finding the NEXT usable region
    /// on boundary crossing. Used only in cold path.
    cursor_region_idx: usize,
    /// Lazy init flag — cursor is initialized on first allocate_frame().
    cursor_initialized: bool,
}

#[cfg(target_arch = "x86_64")]
impl<'a> BootInfoFrameAllocator<'a> {
    /// # Safety
    ///
    /// The caller must guarantee that the memory map is accurate and
    /// that the usable regions are genuinely free for kernel use (i.e.
    /// the bootloader has handed off and nothing else is running).
    pub unsafe fn new(memory_regions: &'a MemoryRegions) -> Self {
        Self {
            memory_regions,
            next: 0,
            cursor_addr: 0,
            cursor_region_end: 0,
            cursor_region_idx: 0,
            cursor_initialized: false,
        }
    }

    /// Iterator over all usable frames. Preserved for `usable_frame_count()`.
    fn usable_frames(&self) -> impl Iterator<Item = PhysFrame<Size4KiB>> + '_ {
        self.memory_regions
            .iter()
            .filter(|r| r.kind == MemoryRegionKind::Usable)
            .flat_map(|r| (r.start..r.end).step_by(4096))
            .filter(|addr| !is_reserved_boot_frame(*addr))
            .map(|addr| PhysFrame::containing_address(PhysAddr::new(addr)))
    }

    fn usable_frame_count(&self) -> usize {
        self.usable_frames().count()
    }

    /// Initialize the caching cursor. Called lazily on first allocate_frame().
    ///
    /// STATE-SAFE: if `self.next > 0`, fast-forwards cursor past that many
    /// frames to prevent double-allocation. All region traversal is
    /// iterative (flat loops) — NO recursion.
    fn init_cursor(&mut self) {
        self.cursor_initialized = true;

        // Find first usable region with frames.
        let mut region_idx = 0usize;
        for region in self.memory_regions.iter() {
            if region.kind == MemoryRegionKind::Usable && region.end > region.start {
                self.cursor_addr = region.start;
                self.cursor_region_end = region.end;
                self.cursor_region_idx = region_idx;

                // STATE-SAFE FAST-FORWARD: skip past already-allocated frames.
                // Iterative — flat loop over regions, NO recursion.
                let mut to_skip = self.next;
                while to_skip > 0 {
                    let remaining_in_region =
                        ((self.cursor_region_end - self.cursor_addr) / 4096) as usize;
                    if remaining_in_region > to_skip {
                        // All remaining skips fit in current region.
                        self.cursor_addr += (to_skip as u64) * 4096;
                        to_skip = 0;
                    } else {
                        // Skip rest of this region, advance to next usable.
                        to_skip -= remaining_in_region;
                        if !self.advance_to_next_region() {
                            // No more usable regions — cursor exhausted.
                            return;
                        }
                    }
                }
                return;
            }
            region_idx += 1;
        }
        // No usable regions found — signals exhausted state.
    }

    /// Advance cursor to the next usable region. Returns false if none remain.
    ///
    /// COLD PATH only — called on region boundary crossings.
    /// Iterative scan from cursor_region_idx + 1 forward. NO recursion.
    fn advance_to_next_region(&mut self) -> bool {
        let mut idx = self.cursor_region_idx + 1;
        for region in self.memory_regions.iter().skip(idx) {
            if region.kind == MemoryRegionKind::Usable && region.end > region.start {
                self.cursor_addr = region.start;
                self.cursor_region_end = region.end;
                self.cursor_region_idx = idx;
                return true;
            }
            idx += 1;
        }
        // All regions exhausted.
        self.cursor_region_end = 0;
        self.cursor_addr = 0;
        false
    }

    fn allocate_frame_in_range(&mut self, min: u64, max: u64) -> Option<PhysFrame<Size4KiB>> {
        if max <= min {
            return None;
        }
        if !self.cursor_initialized {
            self.init_cursor();
        }

        loop {
            let end = self.cursor_region_end.min(max);
            let addr = align_up_4k(self.cursor_addr.max(min));
            if addr < end {
                if is_reserved_boot_frame(addr) {
                    self.cursor_addr = addr + 4096;
                    continue;
                }
                let frame = PhysFrame::containing_address(PhysAddr::new(addr));
                self.cursor_addr = addr + 4096;
                self.next += 1;
                return Some(frame);
            }
            if !self.advance_to_next_region() {
                return None;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn reserve_low_bootstrap_frames(frame_allocator: &mut BootInfoFrameAllocator<'_>) {
    let mut i = 0usize;
    while i < LOW_BOOTSTRAP_FRAMES.len() {
        if LOW_BOOTSTRAP_FRAMES[i].load(Ordering::Acquire) != 0 {
            i += 1;
            continue;
        }
        match frame_allocator
            .allocate_frame_in_range(LOW_BOOTSTRAP_MIN_PHYS, LOW_BOOTSTRAP_MAX_PHYS)
        {
            Some(frame) => {
                LOW_BOOTSTRAP_FRAMES[i].store(frame.start_address().as_u64(), Ordering::Release);
            }
            None => break,
        }
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe impl<'a> FrameAllocator<Size4KiB> for BootInfoFrameAllocator<'a> {
    /// TRUE O(1) amortized frame allocation via caching cursor.
    ///
    /// HOT PATH: `cursor_addr < cursor_region_end` → return frame,
    /// advance cursor by 4096. Pure pointer arithmetic, no iteration.
    ///
    /// COLD PATH (region boundary): `advance_to_next_region()` scans
    /// forward to next usable region. Each region traversed at most once.
    ///
    /// NO recursion. NO `.iter().nth()`. Ring-0-safe, Pillar 1 compliant.
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        if !self.cursor_initialized {
            self.init_cursor();
        }

        // Iterative loop handles region-boundary crossings without recursion.
        loop {
            // HOT PATH — pure pointer arithmetic, TRUE O(1).
            if self.cursor_addr < self.cursor_region_end {
                if is_reserved_boot_frame(self.cursor_addr) {
                    self.cursor_addr += 4096;
                    continue;
                }
                let frame = PhysFrame::containing_address(PhysAddr::new(self.cursor_addr));
                self.cursor_addr += 4096;
                self.next += 1;
                return Some(frame);
            }

            // COLD PATH — region exhausted, find next usable region.
            if !self.advance_to_next_region() {
                return None;
            }
            // Loop continues — re-check hot path for new region.
        }
    }
}

// ---- aarch64 Arena Backing Memory ----
//
// Sub-MP-D2b Stage 10: Feed cross-platform Arena infrastructure
// with hardcoded aarch64 RAM region.
//
// Memory layout (physical RAM, identity-mapped Normal cacheable via TTBR0):
// - 0x40000000 - 0x40080000: Reserved (kernel load area)
// - 0x40080000 - 0x44000000: Kernel image + stack + BSS + padding
// - 0x44000000 - 0x45600000: core arena backing (22 MiB)  ← Stage 10
// - 0x48000000 - 0x947079A0: initrd (Boot-LLM GGUF)
// - 0x95000000 - 0x95300000: ramfb framebuffer (Sub-MP-F1)
// - 0xA0000000 - 0xC0000000: KV_CACHE_ARENA (512 MiB, moved above .smodel 2026-06-13)
//
// All arena backing is identity-mapped Normal cacheable (TTBR0 L1[1]/L1[2]).
// No page table manipulation needed — physical addresses are virtual.

/// Initialize arena infrastructure for aarch64.
///
/// Feeds the same cross-platform Arena statics (KERNEL_ARENA,
/// RUNTIME_ARENA, ACTIVATION_ARENA, KV_CACHE_ARENA) with backing
/// memory from identity-mapped physical RAM.
///
/// # Safety
///
/// Must be called once during boot, after MMU enabled (Stage 6).
/// The arena regions below must be unused by kernel image, stack, BSS,
/// initrd, or framebuffer.
#[cfg(target_arch = "aarch64")]
pub unsafe fn init_aarch64() {
    use core::fmt::Write;

    // Arena layout for aarch64 streaming-era prototype:
    //   KERNEL_ARENA:     0x4400_0000..0x4440_0000  (4 MiB)
    //   RUNTIME_ARENA:    0x4440_0000..0x4460_0000  (2 MiB)
    //   ACTIVATION_ARENA: 0x4460_0000..0x4560_0000  (16 MiB)
    //   KV_CACHE_ARENA:   0xA000_0000..0xC000_0000  (512 MiB)
    //
    // CRITICAL (2026-06-13): KV_CACHE_BASE was 0x9600_0000, sized for an
    // earlier, smaller (~1.28 GiB) model artifact whose initrd ended below
    // it. The native .smodel-v2 is 1.4 GiB: the initrd spans
    // 0x4800_0000..0x9F7F_F980, so the model END (0x9F7F_F980) is ~153 MiB
    // ABOVE 0x9600_0000. The KV arena therefore overlapped the model and KV
    // writes corrupted the upper half of `output.weight` (the LM head),
    // producing garbage logits → degenerate generation on every .smodel run
    // (forward weights below the overlap stayed intact, which is why only
    // the final/generated token was wrong; β-anchor on the smaller artifact
    // never reached 0x9600_0000 and so passed, masking it). Move KV ABOVE
    // the model: the TTBR0 identity map covers 0x8000_0000..0xC000_0000
    // Normal-cacheable, and the model ends at 0x9F7F_F980, so
    // 0xA000_0000..0xC000_0000 clears it and still fits the cacheable window.
    // ramfb (0x9500_0000..0x9530_0000) is below the model end and unaffected.

    const KERNEL_BASE: usize = 0x4400_0000;
    const KERNEL_SZ: usize = 4 * 1024 * 1024; // 4 MiB
    const RUNTIME_BASE: usize = 0x4440_0000;
    const RUNTIME_SZ: usize = 2 * 1024 * 1024; // 2 MiB
    const ACTIVATION_BASE: usize = 0x4460_0000;
    const ACTIVATION_SZ: usize = 16 * 1024 * 1024; // 16 MiB
    const KV_CACHE_BASE: usize = 0xA000_0000;
    const KV_CACHE_SZ: usize = 512 * 1024 * 1024; // 512 MiB

    let serial = &mut crate::arch::aarch64::serial::Serial;
    let _ = writeln!(serial, "Stage 10: aarch64 arena init (V3-Arena-Disziplin)");
    let _ = writeln!(
        serial,
        "  Core backing: {:#x}..{:#x} (22 MiB)",
        KERNEL_BASE,
        ACTIVATION_BASE + ACTIVATION_SZ
    );
    let _ = writeln!(
        serial,
        "  KV backing:   {:#x}..{:#x} (512 MiB)",
        KV_CACHE_BASE,
        KV_CACHE_BASE + KV_CACHE_SZ
    );

    // SAFETY: Identity-mapped Normal cacheable via TTBR0 L1[1].
    // Arena ranges are disjoint from kernel, initrd, and framebuffer.
    // 'static: kernel never returns.
    let kernel_backing = slice::from_raw_parts_mut(KERNEL_BASE as *mut u8, KERNEL_SZ);
    *KERNEL_ARENA.lock() = Some(KernelArenaInner::new(kernel_backing));

    let runtime_backing = slice::from_raw_parts_mut(RUNTIME_BASE as *mut u8, RUNTIME_SZ);
    *RUNTIME_ARENA.lock() = Some(RuntimeArenaInner::new(runtime_backing));

    let activation_backing = slice::from_raw_parts_mut(ACTIVATION_BASE as *mut u8, ACTIVATION_SZ);
    *ACTIVATION_ARENA.lock() = Some(ActivationArenaInner::new(activation_backing));

    let kv_backing = slice::from_raw_parts_mut(KV_CACHE_BASE as *mut u8, KV_CACHE_SZ);
    *KV_CACHE_ARENA.lock() = Some(KvCacheArenaInner::new(kv_backing));

    let _ = writeln!(
        serial,
        "  KERNEL_ARENA:     {:#x} ({} MiB)",
        KERNEL_BASE,
        KERNEL_SZ / (1024 * 1024)
    );
    let _ = writeln!(
        serial,
        "  RUNTIME_ARENA:    {:#x} ({} MiB)",
        RUNTIME_BASE,
        RUNTIME_SZ / (1024 * 1024)
    );
    let _ = writeln!(
        serial,
        "  ACTIVATION_ARENA: {:#x} ({} MiB)",
        ACTIVATION_BASE,
        ACTIVATION_SZ / (1024 * 1024)
    );
    let _ = writeln!(
        serial,
        "  KV_CACHE_ARENA:   {:#x} ({} MiB)",
        KV_CACHE_BASE,
        KV_CACHE_SZ / (1024 * 1024)
    );
    let _ = writeln!(serial, "Stage 10: V3-Arena infrastructure ready");
}
