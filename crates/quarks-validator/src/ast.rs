// SPDX-License-Identifier: AGPL-3.0-or-later
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq)]
pub enum Atom {
    Integer(i64),
    Bytes(Vec<u8>),
    Handle(u64),
    Parameter(u32),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    Atom(Atom),
    List(Vec<SExpr>),
}
