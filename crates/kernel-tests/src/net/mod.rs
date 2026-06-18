// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mirror of the kernel's `net` module tree, holding the host-compilable
//! pieces under test. `tcp` references `super::eth` / `super::ipv4`, so
//! all three must live under the same `net` parent. Paths are relative to
//! this directory (`crates/kernel-tests/src/net/`).

#[path = "../../../../kernel/src/net/eth.rs"]
pub mod eth;

#[path = "../../../../kernel/src/net/ipv4.rs"]
pub mod ipv4;

#[path = "../../../../kernel/src/net/tcp.rs"]
pub mod tcp;

#[path = "../../../../kernel/src/net/ice.rs"]
pub mod ice;
