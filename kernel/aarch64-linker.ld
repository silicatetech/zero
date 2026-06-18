/*
 * aarch64 Linker Script for Silicate Zero
 *
 * Sub-MP-D2b Phase 1 — Build System Setup
 *
 * QEMU virt machine loads -kernel ELFs at their specified virtual
 * addresses (ELF entry point). For flat binaries, load address is
 * typically 0x4000_0000 (1 GiB). We use an ELF with load at
 * 0x4008_0000 (1 GiB + 512 KiB) to leave room for DTB at lower
 * addresses.
 *
 * Layout:
 *   .text.boot  — _start entry point (must be first)
 *   .text       — Rust code
 *   .rodata     — Read-only data (const strings, GGUF embedded data)
 *   .data       — Initialized mutable data
 *   .bss        — Zero-initialized data (page tables, stacks)
 *
 * CITE: Linux Documentation/arm64/booting.rst — kernel image requirements
 * CITE: ARM ARM D5.2 — translation table walk (4KB granule, 48-bit VA)
 */

ENTRY(_start)

SECTIONS {
    /* QEMU virt RAM starts at 0x4000_0000.
     * Place kernel at 0x4008_0000 (512 KiB offset) to avoid
     * DTB/QEMU internal data at RAM base. */
    . = 0x40080000;

    .text.boot : {
        KEEP(*(.text.boot))
    }

    .text : {
        KEEP(*(.text.exception_vectors))  /* Stage 2: VBAR_EL1 table, 2 KiB aligned */
        *(.text .text.*)
    }

    . = ALIGN(4096);
    .rodata : {
        *(.rodata .rodata.*)
    }

    . = ALIGN(4096);
    .data : {
        *(.data .data.*)
    }

    . = ALIGN(4096);
    .bss : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    }

    . = ALIGN(4096);
    __kernel_end = .;

    /* Discard debug info and other non-essential sections */
    /DISCARD/ : {
        *(.comment)
        *(.note*)
    }
}
