# Zero Virtual Address-Space Layout

A living design note — the single source of truth for what virtual
addresses are used by what. Not an ADR, because the layout will
evolve; but load-bearing enough that updates happen *before* the
code, not after.

## Why this document exists

During Stage 2 we placed the kernel heap at `0xFFFF_8000_4444_0000`
and hit `ParentEntryHugePage` from the page-mapper. The bootloader
backs the physical-memory linear map with 1 GiB huge pages, so
every PML3 entry under PML4[256] was already marked huge — a
4 KiB mapping under it is structurally impossible without tearing
the huge page down first. We moved the heap to PML4[288] and
that worked.

Without a written layout plan, every future address choice
(kernel stacks, IO regions, agent memory, user-space) would
re-discover the same class of problem. This document prevents
that.

## Canonical x86_64 addresses — reminder

A virtual address on x86_64 is 48 bits, sign-extended to 64:

- Bits 63..48 must equal bit 47. Anything else is non-canonical and
  faults when used.
- Bit 47 = 0 → **lower half**, `0x0000_0000_0000_0000 .. 0x0000_7FFF_FFFF_FFFF`
- Bit 47 = 1 → **upper half**, `0xFFFF_8000_0000_0000 .. 0xFFFF_FFFF_FFFF_FFFF`

A single PML4 entry covers 2³⁹ bytes = 512 GiB. 512 entries per
half = 256 TiB of addressable space per half.

## Current allocation

| PML4 idx | Virt start | Purpose | Owner | Backing |
|---:|---|---|---|---|
| 0 | `0x0000_0000_0000_0000` | Kernel image (ELF text, rodata, data, bss). Placed at 1 MiB by `kernel/linker.ld`. | Kernel | Identity-ish, set up by bootloader at hand-off. |
| *varies* | *Dynamic, upper half* | Kernel stack. | Bootloader | Dedicated pages. Address chosen by bootloader. |
| *varies* | *Dynamic, upper half* | `BootInfo` struct. | Bootloader | Dedicated pages. |
| *varies* | *Dynamic, upper half* | Framebuffer MMIO. | Bootloader | Hardware MMIO range. |
| **256** | `0xFFFF_8000_0000_0000` | Physical-memory linear map: `VirtAddr = PhysAddr + 0xFFFF_8000_0000_0000`. Covers the actual RAM the bootloader found (~128 MiB on default QEMU). | Bootloader | **1 GiB huge pages.** PML3 entries under this PML4 entry are huge-page leaves. **4 KiB mappings under PML4[256] are not possible** without first unmapping the huge page. |
| **288** | `0xFFFF_9000_0000_0000` | Kernel heap — `linked_list_allocator`. 1 MiB starting at `0xFFFF_9000_4444_0000`. | Kernel | Fresh frames from `BootInfoFrameAllocator`, mapped `PRESENT \| WRITABLE`. |

## Reserved / planned

These ranges are not yet used. They are reserved here so later
stages do not have to re-decide where each thing goes.

| PML4 idx | Virt start | Planned purpose | Notes |
|---:|---|---|---|
| 257..287 | `0xFFFF_8080_0000_0000` | Expansion slot for the physical-memory linear map if we ever face > 15 TiB of RAM. | Almost certainly unused. Reserved to keep the linear map contiguous if it ever grows. |
| 289..319 | `0xFFFF_9080_0000_0000` | Additional kernel heap arenas, large buffers, per-CPU kernel data once SMP arrives. | |
| 320 | `0xFFFF_A000_0000_0000` | Kernel-owned IO regions: memory-mapped device ranges retained by the kernel (not handed to agents). | |
| 321..383 | `0xFFFF_A080_0000_0000` | Kernel-subsystem expansion. | |
| 384 | `0xFFFF_C000_0000_0000` | Agent-memory arena — each agent's private heap. Isolation enforced per-agent via page-table entries. | Distance from PML4[288] (kernel heap) makes range-based protections straightforward. |
| 385..447 | `0xFFFF_C080_0000_0000` | Agent-memory expansion. | |
| 448 | `0xFFFF_E000_0000_0000` | Recursive page-table mapping, if we ever need to walk our own tables from inside the kernel. | Optional; only justified if SMP or complex page-table surgery arrives. |
| 449..510 | `0xFFFF_E080_0000_0000` | Unassigned. | |
| 511 | `0xFFFF_FF80_0000_0000` | Reserved as a sentinel slot. | |

## Lower half (user-space), placeholder

The lower half (PML4[0..255]) is reserved for user-space agents
once the intent format and address-space model
settle the details. Currently only PML4[0] is touched, by the
kernel ELF itself at its 1 MiB physical/virtual address.

When user-space arrives, a typical split might be:

- PML4[0..127] — per-agent user mappings
- PML4[128..255] — reserved / not yet used

This will be updated once the address-space model is finalized.

## Rules of the road

1. **PML4[256] is off-limits for new kernel mappings.** It is the
   bootloader's huge-page-backed linear map. Any attempt to insert
   a 4 KiB mapping under it returns `ParentEntryHugePage`.

2. **Every new address region gets its own PML4 entry** until there
   is a defensible reason to pack regions together. Address space
   is cheap; isolation is free when you plan for it.

3. **Update this table first, code second.** If you catch yourself
   choosing a virtual address in code, the table update goes in the
   same commit. A heap/stack/IO collision caught at runtime is wasted
   time.

4. **Canonical addresses only.** Bit 47 of every kernel-chosen
   higher-half address must be 1. Addresses like `0xFFFF_7000_...`
   are non-canonical and fault on use.
