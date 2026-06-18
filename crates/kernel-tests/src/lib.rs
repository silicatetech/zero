// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::manual_clamp,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::new_without_default,
    clippy::collapsible_match,
    clippy::collapsible_if,
    clippy::result_unit_err,
    clippy::len_without_is_empty,
    clippy::identity_op,
    clippy::question_mark,
    clippy::manual_div_ceil,
    clippy::excessive_precision
)]
//! Host-buildable test harness for the Zero kernel.
//!
//! The kernel crate (`kernel/`) targets `x86_64-unknown-none` (`no_std`,
//! no host toolchain) and is excluded from the workspace, so its many
//! `#[cfg(test)]` suites never ran under `cargo test`. This crate pulls
//! the kernel's *host-compilable* modules in verbatim with `#[path]` and
//! exposes them under a module tree that mirrors `crate::…` inside the
//! kernel, so every `#[cfg(test)] mod tests` they carry now executes via
//! `cargo test --workspace`.
//!
//! # What runs here
//!
//! | kernel module            | how it is reached            | tests |
//! |--------------------------|------------------------------|-------|
//! | `smp` (+ `smp_partition`) | whole-file include + shims   | 15    |
//! | `net::ice`               | whole-file include + shims   | 16    |
//! | `net::tcp`               | whole-file include           | 8     |
//! | `drivers::nvme_probe`    | extracted pure leaf          | 3     |
//! | `weight_layout`          | whole-file include           | 1     |
//!
//! These modules are pure (or, for `smp`/`ice`, depend only on the tiny
//! bare-metal surfaces shimmed below — serial logging, the cycle
//! counter, PCI config access, MMIO/translation — none of which the
//! tests exercise). The kernel source files are **unmodified**: the only
//! kernel-side change for this harness was extracting the pure NVMe
//! probe leaf into `drivers/nvme_probe.rs`, which `nvme.rs` re-exports.
//!
//! # Shims
//!
//! The included files reference `crate::arch`, `crate::memory`, etc. In
//! the kernel those resolve to the real bare-metal implementations; here
//! they resolve to the inert stand-ins below. The shims exist only so
//! the *non-test* code in the included files type-checks on the host —
//! the test bodies never call into them. They do **no** hardware work.

// ─────────────────────────────────────────────────────────────────
// Shims: minimal stand-ins for the bare-metal surfaces the included
// kernel modules reference. None of these are touched by the tests.
// ─────────────────────────────────────────────────────────────────

pub mod arch {
    use core::fmt;

    /// Stand-in for the kernel's write-only serial port. The kernel uses
    /// it for `writeln!(Serial, …)` diagnostics; here it sinks output.
    #[derive(Copy, Clone, Default)]
    pub struct Serial;

    impl fmt::Write for Serial {
        fn write_str(&mut self, _s: &str) -> fmt::Result {
            Ok(())
        }
    }

    /// `crate::arch::serial::Serial` path used by the net/driver modules.
    pub mod serial {
        pub use super::Serial;
    }

    /// `crate::arch::cycles::*` used by the SMP spin barrier's timeout.
    pub mod cycles {
        /// Real kernel reads the invariant TSC frequency; a fixed,
        /// non-zero value is enough for the host build to type-check and
        /// keeps the deadline arithmetic finite.
        pub fn tsc_hz() -> u64 {
            1_000_000_000
        }

        /// Real kernel issues a serialising RDTSC; the host stub returns
        /// 0. The SMP barrier tests complete on arrival count, not on
        /// the timeout path, so the value is never consulted.
        pub fn rdtsc_serialized() -> u64 {
            0
        }
    }

    pub mod x86_64 {
        /// `crate::arch::x86_64::pcie` — PCI types + config accessors the
        /// ice/nvme modules reference. The real module drives port I/O;
        /// these stubs let the non-test code compile on host.
        pub mod pcie {
            /// Mirror of `kernel::arch::x86_64::pcie::PciDevice`. Field
            /// layout must match — the nvme probe tests construct it and
            /// `is_nvme` reads `class_code`/`subclass`. A drift here is a
            /// compile error in those tests, not a silent skip.
            #[derive(Copy, Clone, Debug)]
            pub struct PciDevice {
                pub bus: u8,
                pub device: u8,
                pub function: u8,
                pub vendor_id: u16,
                pub device_id: u16,
                pub class_code: u8,
                pub subclass: u8,
                pub prog_if: u8,
                pub header_type: u8,
            }

            /// Stand-in for the captured PCI scan. The driver bring-up
            /// paths take `&PciScan` and call `.iter()`; the tests never
            /// build one, so an empty backing store is sufficient.
            #[derive(Default)]
            pub struct PciScan {
                devices: [Option<PciDevice>; 0],
            }

            impl PciScan {
                pub fn iter(&self) -> impl Iterator<Item = &PciDevice> {
                    self.devices.iter().filter_map(|d| d.as_ref())
                }
            }

            /// # Safety
            /// Inert stub — performs no port I/O.
            pub unsafe fn config_read16(_bus: u8, _device: u8, _function: u8, _offset: u8) -> u16 {
                0
            }

            /// # Safety
            /// Inert stub — performs no port I/O.
            pub unsafe fn config_write16(
                _bus: u8,
                _device: u8,
                _function: u8,
                _offset: u8,
                _value: u16,
            ) {
            }

            pub fn read_bar(_dev: &PciDevice, _bar: u8) -> Option<u64> {
                None
            }
        }
    }
}

pub mod memory {
    /// Stand-in for the kernel MMIO mapper. The ice driver consumes the
    /// error via `.map_err(|_| …)`, so the concrete error type is
    /// irrelevant; the unit error keeps the host build dependency-free.
    #[allow(clippy::result_unit_err)]
    pub fn map_mmio(_phys_addr: u64, _size: usize) -> Result<*mut u8, ()> {
        Err(())
    }

    pub fn virt_to_phys(_va: u64) -> Option<u64> {
        None
    }
}

// ─────────────────────────────────────────────────────────────────
// Included kernel modules. Paths are relative to this file
// (crates/kernel-tests/src/). `#[cfg(test)]` suites inside each fire
// under `cargo test`.
// ─────────────────────────────────────────────────────────────────

#[path = "../../../kernel/src/smp.rs"]
pub mod smp; // pulls in `smp_partition` via its own `#[path] mod` decl

#[path = "../../../kernel/src/weight_layout.rs"]
pub mod weight_layout;

// `net` and `drivers` are real directory modules (`src/net/mod.rs`,
// `src/drivers/mod.rs`) rather than inline blocks: child `#[path]`
// includes resolve relative to those directories, which must exist on
// disk for the `..` traversal to the kernel tree to succeed.
pub mod drivers;
pub mod net;
