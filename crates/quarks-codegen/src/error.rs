// SPDX-License-Identifier: AGPL-3.0-or-later
//! Codegen errors.

use core::fmt;

/// Classification of codegen failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodegenErrorKind {
    /// iced-x86 `CodeAssembler::new()` failed.
    AssemblerInit,
    /// iced-x86 `assemble()` failed during byte emission.
    AssemblerFinalize,
    /// An individual instruction emit failed.
    EmitFailed,
    /// Attempted to compile an empty list `()`.
    EmptyList,
    /// List head is not a symbol (e.g. `(42 1 2)`).
    NonSymbolHead,
    /// Instruction not supported in current Stage 10 scope.
    UnsupportedInstruction,
    /// Atom type not supported (e.g. Bytes, Handle, bare Symbol).
    UnsupportedAtom,
    /// Wrong number of arguments for an instruction or call.
    ArityMismatch,
    /// Register allocator pool exhausted (all slots in use).
    AllocatorExhausted,
    /// Function defined twice in the same program.
    DuplicateFunction,
    /// `(fn ...)` form has wrong structure.
    MalformedFunction,
    /// Call target not found in function table.
    FunctionNotFound,
    /// Function arity exceeds System V AMD64 register-only ABI (> 6).
    ArityExceedsAbi,
    /// `(program ...)` missing a `(call ...)` entry point.
    MissingEntryCall,
}

/// A codegen error with classification and human-readable message.
#[derive(Debug, Clone)]
pub struct CodegenError {
    pub kind: CodegenErrorKind,
    pub message: String,
}

impl CodegenError {
    pub fn new(kind: CodegenErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for CodegenError {}
