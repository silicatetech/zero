// SPDX-License-Identifier: AGPL-3.0-or-later
//! Global Descriptor Table + Task State Segment.
//!
//! Stage 1. We replace whatever the bootloader left us with a GDT
//! we control, plus a TSS whose IST table has a dedicated stack for
//! the double-fault handler at index 0. Without that, a kernel
//! stack-overflow would push onto an already-invalid stack and
//! cascade into a triple fault — instant reboot, no diagnostic.
//!
//! We also install a kernel data segment and reload SS/DS/ES so that
//! stale bootloader selectors (whose indices now point to *our* TSS
//! descriptor) don't cause #GP on iretq.

use core::ptr::addr_of;
use lazy_static::lazy_static;
use x86_64::instructions::segmentation::{Segment, CS, DS, ES, SS};
use x86_64::instructions::tables::load_tss;
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector};
use x86_64::structures::tss::TaskStateSegment;
use x86_64::VirtAddr;

/// IST slot for the double-fault handler's dedicated stack.
/// `interrupts::init` must pass the same index to `set_stack_index`.
pub const DOUBLE_FAULT_IST_INDEX: u16 = 0;

/// Size of the IST stack. Generous on purpose — double-fault
/// handling should not be stingy, and we have no heap yet so
/// there is no cost to static allocation.
const DOUBLE_FAULT_STACK_SIZE: usize = 4096 * 5;

#[repr(align(16))]
#[allow(dead_code)] // Bytes are accessed by the CPU, not from Rust.
struct Stack([u8; DOUBLE_FAULT_STACK_SIZE]);

/// The IST stack itself. `static mut` is acceptable because the CPU
/// owns this memory exclusively during double-fault dispatch; no
/// Rust code reads or writes it. We only ever take its address.
static mut DOUBLE_FAULT_STACK: Stack = Stack([0; DOUBLE_FAULT_STACK_SIZE]);

lazy_static! {
    static ref TSS: TaskStateSegment = {
        let mut tss = TaskStateSegment::new();
        tss.interrupt_stack_table[DOUBLE_FAULT_IST_INDEX as usize] = {
            // The resulting address is stable for the lifetime of the
            // program; `addr_of!` does not dereference, so no unsafe.
            let stack_bottom = VirtAddr::from_ptr(addr_of!(DOUBLE_FAULT_STACK));
            stack_bottom + DOUBLE_FAULT_STACK_SIZE as u64
        };
        tss
    };
}

struct Selectors {
    code_selector: SegmentSelector,
    data_selector: SegmentSelector,
    tss_selector: SegmentSelector,
}

lazy_static! {
    static ref GDT: (GlobalDescriptorTable, Selectors) = {
        let mut gdt = GlobalDescriptorTable::new();
        let code_selector = gdt.append(Descriptor::kernel_code_segment());
        let data_selector = gdt.append(Descriptor::kernel_data_segment());
        let tss_selector = gdt.append(Descriptor::tss_segment(&TSS));
        (
            gdt,
            Selectors {
                code_selector,
                data_selector,
                tss_selector,
            },
        )
    };
}

/// Load the GDT and TSS. Call before `interrupts::init`, because the
/// IDT's double-fault entry references the TSS's IST stack.
pub fn init() {
    GDT.0.load();
    unsafe {
        // Install our own segment selectors. SS/DS/ES must be reloaded
        // so that the bootloader's stale selector values (which in our
        // GDT now point at the TSS descriptor) don't linger in the
        // segment registers and fault on iretq.
        CS::set_reg(GDT.1.code_selector);
        SS::set_reg(GDT.1.data_selector);
        DS::set_reg(GDT.1.data_selector);
        ES::set_reg(GDT.1.data_selector);
        load_tss(GDT.1.tss_selector);
    }
}
