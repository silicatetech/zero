// SPDX-License-Identifier: AGPL-3.0-or-later
#![no_std]
#![allow(
    clippy::manual_is_multiple_of,
    clippy::unnecessary_map_or,
    clippy::collapsible_if,
    clippy::useless_asref,
    clippy::assertions_on_constants
)]
extern crate alloc;

pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod type_checker;
