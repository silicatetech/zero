// SPDX-License-Identifier: AGPL-3.0-or-later
//! PCI-level NVMe controller discovery — extracted from `nvme.rs` so it
//! can be exercised by the host test harness (`crates/kernel-tests`).
//!
//! This module is **pure**: it depends only on `core`, the
//! `PciDevice` data type, and the spec constants from `zero_nvme`.
//! It touches no hardware, no MMIO, no statics and no kernel arena, so
//! it compiles and runs identically on the bare-metal kernel target and
//! on the dev host. The rest of `nvme.rs` (BAR0 map, admin/IO queue
//! bring-up, DMA via `KERNEL_ARENA`) cannot — which is why only the
//! enumeration leaf (`is_nvme`, `ControllerList`) lives here. `nvme.rs`
//! re-exports every public item below, so `crate::drivers::nvme::is_nvme`,
//! `ControllerList`, `MAX_CORES`-style constants keep resolving unchanged
//! for the kernel and its dependents.

use crate::arch::x86_64::pcie::PciDevice;
use zero_nvme::{PCI_CLASS_MASS_STORAGE, PCI_SUBCLASS_NVM};

/// True iff `dev` is an NVMe controller. The canonical NVMe class triple
/// is (0x01, 0x08, 0x02) per spec; we tolerate alternative prog_if
/// values (e.g. NVMe-MI, prog_if 0x03) since the BAR0 wire protocol
/// is unchanged.
pub fn is_nvme(dev: &PciDevice) -> bool {
    dev.class_code == PCI_CLASS_MASS_STORAGE && dev.subclass == PCI_SUBCLASS_NVM
}

/// Tiny no-alloc bounded list of `PciDevice` — the driver can't reach
/// for `Vec` here because we may run before the runtime arena is fully
/// populated, and a fixed cap of 8 is more than the Cherry Server (or
/// any sane single-box) will ever expose.
pub struct ControllerList {
    entries: [Option<PciDevice>; MAX_CONTROLLERS],
    len: usize,
}

pub const MAX_CONTROLLERS: usize = 8;

impl ControllerList {
    pub const fn new() -> Self {
        Self {
            entries: [None; MAX_CONTROLLERS],
            len: 0,
        }
    }

    pub fn push(&mut self, d: PciDevice) {
        if self.len < MAX_CONTROLLERS {
            self.entries[self.len] = Some(d);
            self.len += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &PciDevice> {
        self.entries.iter().filter_map(|e| e.as_ref())
    }

    pub fn get(&self, idx: usize) -> Option<&PciDevice> {
        self.entries.get(idx).and_then(|e| e.as_ref())
    }
}

impl Default for ControllerList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::x86_64::pcie::PciDevice;

    fn dev(class: u8, sub: u8) -> PciDevice {
        PciDevice {
            bus: 0,
            device: 0,
            function: 0,
            vendor_id: 0xABCD,
            device_id: 0x0001,
            class_code: class,
            subclass: sub,
            prog_if: 0x02,
            header_type: 0,
        }
    }

    #[test]
    fn is_nvme_matches_spec_triple() {
        assert!(is_nvme(&dev(0x01, 0x08)));
    }

    #[test]
    fn is_nvme_rejects_other_storage() {
        // SATA: subclass 0x06.
        assert!(!is_nvme(&dev(0x01, 0x06)));
        // SCSI: subclass 0x00.
        assert!(!is_nvme(&dev(0x01, 0x00)));
        // Network controller (class 0x02).
        assert!(!is_nvme(&dev(0x02, 0x08)));
    }

    #[test]
    fn controller_list_bounded_and_iterable() {
        let mut list = ControllerList::new();
        list.push(dev(0x01, 0x08));
        list.push(dev(0x01, 0x08));
        assert_eq!(list.len(), 2);
        assert!(list.get(0).is_some());
        assert!(list.get(2).is_none());
        // Push past cap silently drops.
        for _ in 0..MAX_CONTROLLERS {
            list.push(dev(0x01, 0x08));
        }
        assert_eq!(list.len(), MAX_CONTROLLERS);
    }
}
