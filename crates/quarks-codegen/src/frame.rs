// SPDX-License-Identifier: AGPL-3.0-or-later
//! Function-Codegen-Context.
//!
//! When compiling a function body, parameter references (`%n`) must
//! be resolved to physical registers per System V AMD64 ABI:
//!   %0 → RDI, %1 → RSI, %2 → RDX, %3 → RCX, %4 → R8, %5 → R9
//!
//! The `FunctionContext` tracks the current function being compiled
//! and provides parameter-to-register lookup.
//!
//! Stage 10 MP2 does NOT support local variables (let-locals) — `%n`
//! with `n >= arity` returns `UnsupportedAtom`. Local variables
//! are Stage 11+ material.

use iced_x86::code_asm::{self, AsmRegister64};

use crate::error::{CodegenError, CodegenErrorKind};

/// Maximum function arity in Stage 10 (System V AMD64 register-only ABI).
pub const MAX_ARITY: usize = 6;

/// Resolve parameter index to System V AMD64 register.
///
/// `%0 → RDI, %1 → RSI, %2 → RDX, %3 → RCX, %4 → R8, %5 → R9`
fn param_reg(idx: usize) -> AsmRegister64 {
    match idx {
        0 => code_asm::rdi,
        1 => code_asm::rsi,
        2 => code_asm::rdx,
        3 => code_asm::rcx,
        4 => code_asm::r8,
        5 => code_asm::r9,
        _ => unreachable!("param_reg called with idx {} > MAX_ARITY", idx),
    }
}

/// Function-codegen context.
///
/// Holds the current function's arity for parameter-reference
/// resolution. Constructed at function-definition entry, used during
/// body compilation, discarded at exit.
#[derive(Debug)]
pub struct FunctionContext {
    pub arity: usize,
}

impl FunctionContext {
    pub fn new(arity: usize) -> Result<Self, CodegenError> {
        if arity > MAX_ARITY {
            return Err(CodegenError::new(
                CodegenErrorKind::ArityExceedsAbi,
                format!(
                    "function arity {} exceeds Stage 10 max of {} \
                     (System V AMD64 register-only ABI; stack-spilled \
                     args deferred to Stage 11+)",
                    arity, MAX_ARITY
                ),
            ));
        }
        Ok(Self { arity })
    }

    /// Resolve `%n` parameter reference to physical register.
    ///
    /// Returns the System V AMD64 parameter register for `idx`.
    /// `idx >= arity` returns UnsupportedAtom (let-locals are Stage 11+).
    pub fn resolve_parameter(&self, idx: u32) -> Result<AsmRegister64, CodegenError> {
        let idx = idx as usize;
        if idx >= self.arity {
            return Err(CodegenError::new(
                CodegenErrorKind::UnsupportedAtom,
                format!(
                    "%{} references slot beyond arity {}; \
                     let-locals are Stage 11+",
                    idx, self.arity
                ),
            ));
        }
        Ok(param_reg(idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_zero_ok() {
        let ctx = FunctionContext::new(0).unwrap();
        assert_eq!(ctx.arity, 0);
    }

    #[test]
    fn arity_six_ok() {
        let ctx = FunctionContext::new(6).unwrap();
        assert_eq!(ctx.arity, 6);
    }

    #[test]
    fn arity_seven_rejected() {
        let result = FunctionContext::new(7);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::ArityExceedsAbi);
    }

    #[test]
    fn parameter_zero_resolves_to_rdi() {
        let ctx = FunctionContext::new(2).unwrap();
        let reg = ctx.resolve_parameter(0).unwrap();
        assert_eq!(reg, code_asm::rdi);
    }

    #[test]
    fn parameter_one_resolves_to_rsi() {
        let ctx = FunctionContext::new(2).unwrap();
        let reg = ctx.resolve_parameter(1).unwrap();
        assert_eq!(reg, code_asm::rsi);
    }

    #[test]
    fn parameter_outside_arity_unsupported() {
        let ctx = FunctionContext::new(2).unwrap();
        let result = ctx.resolve_parameter(2);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::UnsupportedAtom);
    }
}
