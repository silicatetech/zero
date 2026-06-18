// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardware device drivers.
//!
//! Lives outside `arch/x86_64/` because protocols like NVMe are
//! platform-agnostic at the wire level — the spec describes register
//! layouts, queue formats, and command opcodes that look identical
//! whether you reach them through x86 MMIO or an aarch64 ECAM window.
//! Hardware-touching paths inside each driver are still gated on
//! `target_arch` because the surrounding kernel only has MMIO mapping
//! / virt-to-phys translation wired up on x86_64 today.

#[cfg(target_arch = "x86_64")]
pub mod nvme;
