// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::useless_conversion, clippy::manual_is_multiple_of)]
//! Quarks AOT Codegen — S-Expression IR → x86_64 Machine Code
//!
//! Compiles validated Quarks IR to native x86_64 byte sequences
//! for AOT execution.
//!
//! # Supported forms
//!
//! - **Bare expressions** (MP1): `42`, `(add 1 2)`, `(mul (add 2 3) 4)`
//! - **Programs with functions** (MP2):
//!   `(program (fn main () i64 (add 1 2)) (call main))`
//!
//! # Pipeline
//!
//! ```ignore
//! let ir = "(program (fn main () i64 (add 1 2)) (call main))";
//! let ast = quarks_validator::parse(ir)?;
//! quarks_validator::type_check(&ast)?;
//! let bytes = quarks_codegen::compile(&ast)?;
//! // bytes is Vec<u8> of x86_64 machine code
//! ```
//!
//! # Architecture
//!
//! - **Linear-Scan Allocator** ([`allocator`]): Manages a 3-register
//!   pool (RAX/RCX/RDX) with parameter-aware reservation.
//!
//! - **Assembler** ([`assembler`]): Recursive SExpr → iced-x86
//!   CodeAssembler translation with function prologue/epilogue and
//!   System V AMD64 calling convention.
//!
//! - **Frame** ([`frame`]): Function context for parameter resolution
//!   (%0 → RDI, %1 → RSI, etc.).
//!
//! See ARCHITECTURE.md for the full design rationale.

mod allocator;
mod assembler;
mod error;
mod frame;

pub use error::{CodegenError, CodegenErrorKind};

use iced_x86::code_asm::*;
use quarks_validator::{Atom, SExpr};

/// Compile an S-Expression IR tree to x86_64 machine code bytes.
///
/// Accepts both bare expressions (`(add 1 2)`) and program forms
/// (`(program (fn ...) (call ...))`). The result is a byte sequence
/// suitable for execution as `extern "C" fn() -> i64`.
///
/// # Errors
///
/// Returns [`CodegenError`] on unsupported forms, arity mismatches,
/// undefined functions, or internal assembler failures.
pub fn compile(ast: &SExpr) -> Result<Vec<u8>, CodegenError> {
    let mut asm = CodeAssembler::new(64).map_err(|e| {
        CodegenError::new(
            CodegenErrorKind::AssemblerInit,
            format!("CodeAssembler::new(64) failed: {}", e),
        )
    })?;

    let mut alloc = allocator::LinearScanAllocator::new();

    let result_reg = assembler::compile_top_level(ast, &mut asm, &mut alloc)?;

    // Bare-expression path: need to add mov rax + ret.
    // Program path: compile_program already emits its own prologue/epilogue/ret.
    if !is_program(ast) {
        if result_reg != allocator::RegSlot::Rax {
            asm.mov(rax, result_reg.to_iced()).map_err(|e| {
                CodegenError::new(
                    CodegenErrorKind::EmitFailed,
                    format!("mov rax failed: {}", e),
                )
            })?;
        }
        asm.ret().map_err(|e| {
            CodegenError::new(CodegenErrorKind::EmitFailed, format!("ret failed: {}", e))
        })?;
    }

    asm.assemble(0).map_err(|e| {
        CodegenError::new(
            CodegenErrorKind::AssemblerFinalize,
            format!("assemble failed: {}", e),
        )
    })
}

fn is_program(ast: &SExpr) -> bool {
    if let SExpr::List(items) = ast {
        if let Some(SExpr::Atom(Atom::Symbol(s))) = items.first() {
            return s == "program";
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use quarks_validator::parse;

    fn compile_str(src: &str) -> Vec<u8> {
        let ast = parse(src).expect("parse failed");
        compile(&ast).expect("compile failed")
    }

    fn hex(bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|b| format!("{:02X}", b))
            .collect::<Vec<_>>()
            .join(" ")
    }

    // ═══════════════════════════════════════════════════════
    // MP1 TESTS (preserved — bare expressions)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn iconst_42() {
        let bytes = compile_str("42");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x2A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn iconst_0() {
        let bytes = compile_str("0");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn iconst_negative() {
        let bytes = compile_str("-1");
        let expected: &[u8] = &[
            0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn add_1_2() {
        let bytes = compile_str("(add 1 2)");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB9, 0x02, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x01, 0xC8, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn sub_5_3() {
        let bytes = compile_str("(sub 5 3)");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB9, 0x03, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x29, 0xC8, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn mul_4_6() {
        let bytes = compile_str("(mul 4 6)");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB9, 0x06, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x0F, 0xAF, 0xC1, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn nested_add_mul_2_3_4() {
        let bytes = compile_str("(add (mul 2 3) 4)");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB9, 0x03, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x0F, 0xAF, 0xC1, 0x48, 0xB9, 0x04, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x01, 0xC8, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn nested_add_mul_sub_3reg() {
        let bytes = compile_str("(add (mul 2 3) (sub 5 1))");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB9, 0x03, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x0F, 0xAF, 0xC1, 0x48, 0xB9, 0x05, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xBA, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x48, 0x29, 0xD1, 0x48, 0x01, 0xC8, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    // ── MP1 error paths ─────────────────────────────────────

    #[test]
    fn empty_list_is_error() {
        let ast = parse("()").expect("parse");
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::EmptyList);
    }

    #[test]
    fn unsupported_instruction() {
        let ast = parse("(div 6 2)").expect("parse");
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind,
            CodegenErrorKind::UnsupportedInstruction
        );
    }

    #[test]
    fn fn_definition_unsupported_bare() {
        let ast = parse("(fn main () i64 (add 1 2))").expect("parse");
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind,
            CodegenErrorKind::UnsupportedInstruction
        );
    }

    #[test]
    fn arity_mismatch_add() {
        let ast = parse("(add 1)").expect("parse");
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::ArityMismatch);
    }

    #[test]
    fn non_symbol_head() {
        let ast = parse("(42 1 2)").expect("parse");
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::NonSymbolHead);
    }

    // ═══════════════════════════════════════════════════════
    // MP2 TESTS (new — fn/call/program)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn program_zero_arg_main() {
        // (program (fn main () i64 (add 1 2)) (call main))
        let bytes = compile_str("(program (fn main () i64 (add 1 2)) (call main))");
        // Verified via /tmp recon:
        // Entry:  55 48 89 E5 E8 02 00 00 00 5D C3
        // main:   55 48 89 E5 48 B8 01... 48 B9 02... 48 01 C8 5D C3
        let expected: &[u8] = &[
            // entry: push rbp; mov rbp,rsp; call main; pop rbp; ret
            0x55, 0x48, 0x89, 0xE5, 0xE8, 0x02, 0x00, 0x00, 0x00, 0x5D, 0xC3,
            // main: push rbp; mov rbp,rsp; mov rax,1; mov rcx,2; add rax,rcx; pop rbp; ret
            0x55, 0x48, 0x89, 0xE5, 0x48, 0xB8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x48, 0xB9, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x01, 0xC8, 0x5D,
            0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn program_one_param_function() {
        // (program (fn add1 (i64) i64 (add %0 1)) (call add1 5))
        let bytes = compile_str("(program (fn add1 (i64) i64 (add %0 1)) (call add1 5))");
        // Verified via /tmp recon: 46 bytes
        let expected: &[u8] = &[
            // entry: push rbp; mov rbp,rsp; mov rax,5; mov rdi,rax; call add1; pop rbp; ret
            0x55, 0x48, 0x89, 0xE5, 0x48, 0xB8, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x48, 0x89, 0xC7, 0xE8, 0x02, 0x00, 0x00, 0x00, 0x5D, 0xC3,
            // add1: push rbp; mov rbp,rsp; mov rax,rdi; mov rcx,1; add rax,rcx; pop rbp; ret
            0x55, 0x48, 0x89, 0xE5, 0x48, 0x89, 0xF8, 0x48, 0xB9, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x48, 0x01, 0xC8, 0x5D, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn program_two_param_function() {
        // (program (fn add (i64 i64) i64 (add %0 %1)) (call add 10 20))
        let bytes = compile_str("(program (fn add (i64 i64) i64 (add %0 %1)) (call add 10 20))");
        // Verified via /tmp recon: 52 bytes
        let expected: &[u8] = &[
            // entry: push rbp; mov rbp,rsp; mov rax,10; mov rcx,20;
            //        mov rdi,rax; mov rsi,rcx; call add; pop rbp; ret
            0x55, 0x48, 0x89, 0xE5, 0x48, 0xB8, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x48, 0xB9, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x89, 0xC7, 0x48,
            0x89, 0xCE, 0xE8, 0x02, 0x00, 0x00, 0x00, 0x5D, 0xC3,
            // add: push rbp; mov rbp,rsp; mov rax,rdi; mov rcx,rsi;
            //      add rax,rcx; pop rbp; ret
            0x55, 0x48, 0x89, 0xE5, 0x48, 0x89, 0xF8, 0x48, 0x89, 0xF1, 0x48, 0x01, 0xC8, 0x5D,
            0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    // ── MP2 error paths ─────────────────────────────────────

    #[test]
    fn duplicate_function_error() {
        let ast = parse("(program (fn foo () i64 1) (fn foo () i64 2) (call foo))").unwrap();
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind,
            CodegenErrorKind::DuplicateFunction
        );
    }

    #[test]
    fn function_not_found_in_call() {
        let ast = parse("(program (fn foo () i64 1) (call bar))").unwrap();
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::FunctionNotFound);
    }

    #[test]
    fn arity_mismatch_in_call() {
        let ast = parse("(program (fn foo (i64) i64 %0) (call foo 1 2))").unwrap();
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::ArityMismatch);
    }

    #[test]
    fn arity_exceeds_abi() {
        let src = "(program (fn f (i64 i64 i64 i64 i64 i64 i64) i64 0) (call f 1 2 3 4 5 6 7))";
        let ast = parse(src).unwrap();
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::ArityExceedsAbi);
    }

    #[test]
    fn missing_entry_call() {
        let ast = parse("(program (fn foo () i64 1))").unwrap();
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::MissingEntryCall);
    }

    #[test]
    fn nested_call_in_body_unsupported() {
        let src = "(program (fn helper () i64 1) (fn main () i64 (call helper)) (call main))";
        let ast = parse(src).unwrap();
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind,
            CodegenErrorKind::UnsupportedInstruction
        );
    }

    #[test]
    fn parameter_outside_function_error() {
        let ast = parse("%0").unwrap();
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::UnsupportedAtom);
    }

    // ═══════════════════════════════════════════════════════
    // MP5 TESTS (new — Handle atoms, per ADR-027)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn handle_atom_5_compiles_as_immediate() {
        // ADR-027: Atom::Handle(5) → mov rax, 5; ret
        // Identical to Atom::Integer(5) at machine-code level.
        let bytes = compile_str("@5");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn handle_atom_large_id() {
        // @1000 → mov rax, 1000; ret
        let bytes = compile_str("@1000");
        let expected: &[u8] = &[
            0x48, 0xB8, 0xE8, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn handle_as_function_parameter() {
        // Function accepting and returning a handle
        let bytes = compile_str("(program (fn echo (handle) handle %0) (call echo @7))");
        // Entry: push rbp; mov rbp,rsp; mov rax,7; mov rdi,rax; call echo; pop rbp; ret
        // echo:  push rbp; mov rbp,rsp; mov rax,rdi; pop rbp; ret
        let expected: &[u8] = &[
            // entry
            0x55, 0x48, 0x89, 0xE5, 0x48, 0xB8, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x48, 0x89, 0xC7, 0xE8, 0x02, 0x00, 0x00, 0x00, 0x5D, 0xC3, // echo
            0x55, 0x48, 0x89, 0xE5, 0x48, 0x89, 0xF8, 0x5D, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    // ═══════════════════════════════════════════════════════
    // Phase 4 Step 1 — Bitwise op codegen
    // ═══════════════════════════════════════════════════════

    #[test]
    fn bit_and_5_3() {
        // (bit-and 5 3): mov rax, 5; mov rcx, 3; and rax, rcx; ret
        let bytes = compile_str("(bit-and 5 3)");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, 5
            0x48, 0xB9, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rcx, 3
            0x48, 0x21, 0xC8, // and rax, rcx
            0xC3, // ret
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn bit_or_1_2() {
        // (bit-or 1 2): mov rax, 1; mov rcx, 2; or rax, rcx; ret
        let bytes = compile_str("(bit-or 1 2)");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB9, 0x02, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x09, 0xC8, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn bit_xor_5_3() {
        // (bit-xor 5 3): mov rax, 5; mov rcx, 3; xor rax, rcx; ret
        let bytes = compile_str("(bit-xor 5 3)");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB9, 0x03, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x31, 0xC8, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn bit_shl_1_8() {
        // (bit-shl 1 8): mov rax, 1; mov rcx, 8; shl rax, cl; ret
        // The shift count flows into rcx naturally (right_slot == RCX),
        // so no extra mov/xchg is emitted.
        let bytes = compile_str("(bit-shl 1 8)");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB9, 0x08, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xD3, 0xE0, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn bit_shr_256_4() {
        // (bit-shr 256 4): mov rax, 256; mov rcx, 4; sar rax, cl; ret
        // SAR encodes the arithmetic right shift; matches the
        // interpreter's `i64 >>` semantics.
        let bytes = compile_str("(bit-shr 256 4)");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB9, 0x04, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xD3, 0xF8, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn bit_and_with_shifted_mask() {
        // (bit-and (bit-shl 1 8) 65280): a composition.
        //   mov rax, 1
        //   mov rcx, 8
        //   shl rax, cl       ; rax = 256
        //   mov rcx, 65280
        //   and rax, rcx      ; rax = 256
        //   ret
        let bytes = compile_str("(bit-and (bit-shl 1 8) 65280)");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xB9, 0x08, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xD3, 0xE0, 0x48, 0xB9, 0x00, 0xFF, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x21, 0xC8, 0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn bit_and_inside_function() {
        // (program (fn mask (i64 i64) i64 (bit-and %0 %1)) (call mask 255 15))
        // Entry sets up the two params, then call mask; mask is
        // `and rax, rcx`. Bit-pattern parallels `program_two_param_function`
        // — only the data-processing opcode differs (0x21 = AND, vs
        // 0x01 = ADD).
        let src = "(program (fn mask (i64 i64) i64 (bit-and %0 %1)) (call mask 255 15))";
        let bytes = compile_str(src);
        let expected: &[u8] = &[
            // entry: push rbp; mov rbp,rsp; mov rax,255; mov rcx,15;
            //        mov rdi,rax; mov rsi,rcx; call mask; pop rbp; ret
            0x55, 0x48, 0x89, 0xE5, 0x48, 0xB8, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x48, 0xB9, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x89, 0xC7, 0x48,
            0x89, 0xCE, 0xE8, 0x02, 0x00, 0x00, 0x00, 0x5D, 0xC3,
            // mask: push rbp; mov rbp,rsp; mov rax,rdi; mov rcx,rsi;
            //       and rax,rcx; pop rbp; ret
            0x55, 0x48, 0x89, 0xE5, 0x48, 0x89, 0xF8, 0x48, 0x89, 0xF1, 0x48, 0x21, 0xC8, 0x5D,
            0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn bit_shl_inside_function() {
        // (program (fn sl (i64 i64) i64 (bit-shl %0 %1)) (call sl 1 10))
        let src = "(program (fn sl (i64 i64) i64 (bit-shl %0 %1)) (call sl 1 10))";
        let bytes = compile_str(src);
        let expected: &[u8] = &[
            // entry
            0x55, 0x48, 0x89, 0xE5, 0x48, 0xB8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x48, 0xB9, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x89, 0xC7, 0x48,
            0x89, 0xCE, 0xE8, 0x02, 0x00, 0x00, 0x00, 0x5D, 0xC3,
            // sl: push rbp; mov rbp,rsp; mov rax,rdi; mov rcx,rsi;
            //     shl rax,cl; pop rbp; ret
            0x55, 0x48, 0x89, 0xE5, 0x48, 0x89, 0xF8, 0x48, 0x89, 0xF1, 0x48, 0xD3, 0xE0, 0x5D,
            0xC3,
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn bit_shl_xchg_path_when_left_slot_is_rcx() {
        // Forces `shift_op` case 2 in `assembler.rs` (left_slot == RCX,
        // right_slot != RCX). The outer `(add 7 ...)` holds RAX before
        // the inner shift compiles, so the shift's left operand lands
        // in RCX and its right operand in RDX — triggering the
        // `xchg rcx, rdx` path. Without that swap the count would
        // clobber the value being shifted.
        //
        // Emission sequence:
        //   mov  rax, 7
        //   mov  rcx, 5    ; inner shift's left slot
        //   mov  rdx, 3    ; inner shift's right slot
        //   xchg rcx, rdx  ; left value → RDX, count → RCX
        //   shl  rdx, cl   ; shift the value (now in RDX) by CL
        //   add  rax, rdx
        //   ret
        let bytes = compile_str("(add 7 (bit-shl 5 3))");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, 7
            0x48, 0xB9, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rcx, 5
            0x48, 0xBA, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rdx, 3
            0x48, 0x87, 0xD1, // xchg rcx, rdx
            0x48, 0xD3, 0xE2, // shl  rdx, cl
            0x48, 0x01, 0xD0, // add  rax, rdx
            0xC3, // ret
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn bit_shr_xchg_path_when_left_slot_is_rcx() {
        // Same shape as the bit-shl variant but with `sar rdx, cl`
        // (opcode 0x48 0xD3 0xF8 with ModR/M /7 for SAR). Locks in
        // that the SAR encoding is reached via the xchg case too —
        // not just the trivial right_slot == RCX path.
        let bytes = compile_str("(add 7 (bit-shr 256 3))");
        let expected: &[u8] = &[
            0x48, 0xB8, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, 7
            0x48, 0xB9, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rcx, 256
            0x48, 0xBA, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rdx, 3
            0x48, 0x87, 0xD1, // xchg rcx, rdx
            0x48, 0xD3, 0xFA, // sar  rdx, cl
            0x48, 0x01, 0xD0, // add  rax, rdx
            0xC3, // ret
        ];
        assert_eq!(bytes, expected, "actual: {}", hex(&bytes));
    }

    #[test]
    fn arity_mismatch_bit_and() {
        let ast = parse("(bit-and 1)").expect("parse");
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::ArityMismatch);
    }

    #[test]
    fn arity_mismatch_bit_shl() {
        let ast = parse("(bit-shl 1)").expect("parse");
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::ArityMismatch);
    }

    #[test]
    fn bytes_atom_still_unsupported() {
        let ast = parse("#x48656c6c6f").unwrap();
        let result = compile(&ast);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, CodegenErrorKind::UnsupportedAtom);
    }
}
