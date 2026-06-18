// SPDX-License-Identifier: AGPL-3.0-or-later
#![no_std]
#![allow(
    clippy::ptr_arg,
    clippy::too_many_arguments,
    clippy::doc_overindented_list_items,
    clippy::doc_lazy_continuation
)]

extern crate alloc;

pub mod ast;
pub mod instructions;
pub mod parser;
pub mod type_checker;
pub mod validator;

pub use ast::{Atom, SExpr};
pub use instructions::{lookup, ArgShape, InstructionSignature, ValueType, INSTRUCTIONS};
pub use parser::{parse, ParseError, ParseErrorKind};
pub use type_checker::{
    type_check, MatchPatternKind, StructInfo, StructTable, TypeCheckError, TypeCheckErrorKind,
};
pub use validator::{
    validate_structure, CondMalformedKind, MatchMalformedKind, PolicyMalformedKind,
    ValidationError, ValidationErrorKind,
};
