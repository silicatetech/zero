/* Silicate Zero kernel linker script.
 *
 *
 * Kernel loads at virtual 2 MiB (0x200000) — above the bootloader 0.11.15
 * BIOS Stage 4 region (0x130000) and context_switch identity-map (0x13a000).
 * Provides ~14 MB headroom until the next bootloader identity-mapped frame.
 * See ADR-028 v7 for root-cause analysis of the Mode-B boundary collision.
 */

ENTRY(_start)

SECTIONS {
    . = 2M;

    .text : ALIGN(4K) {
        *(.text .text.*)
    }

    .rodata : ALIGN(4K) {
        *(.rodata .rodata.*)
    }

    .data : ALIGN(4K) {
        *(.data .data.*)
    }

    .bss : ALIGN(4K) {
        *(.bss .bss.*)
        *(COMMON)
    }

    .got : ALIGN(4K) {
        *(.got .got.*)
    }

    /DISCARD/ : {
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.note.*)
        *(.comment)
    }
}
