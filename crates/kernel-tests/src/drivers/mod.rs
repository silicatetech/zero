// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mirror of the kernel's `drivers` module, holding the pure NVMe probe
//! leaf under test. Path is relative to this directory
//! (`crates/kernel-tests/src/drivers/`).

#[path = "../../../../kernel/src/drivers/nvme_probe.rs"]
pub mod nvme_probe;
