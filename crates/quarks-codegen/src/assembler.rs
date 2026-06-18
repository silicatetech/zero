// SPDX-License-Identifier: AGPL-3.0-or-later
//! SExpr → iced-x86 Assembly translation.
//!
//! Stage 10 MP1 scope: bare expressions (iconst/add/sub/mul).
//! Stage 10 MP2 scope: + (program (fn ...) (call ...)) with System V
//! AMD64 calling convention.
//!
//! # Calling Convention (System V AMD64)
//!
//! - Parameters %0-%5 → RDI, RSI, RDX, RCX, R8, R9
//! - Return value → RAX
//! - Arity > 6 rejected (Stage 11+ may add stack-spilling)
//! - Caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11
//! - Callee-saved: RBP, RBX, R12-R15 (we use RBP for frame pointer)
//!
//! # Function Prologue / Epilogue
//!
//! ```x86asm
//! ; Prologue
//! push rbp
//! mov  rbp, rsp
//! ; ... body ...
//! ; Epilogue
//! pop  rbp
//! ret
//! ```
//!
//! # Stack Alignment
//!
//! System V AMD64 requires RSP ≡ 16 (mod 16) immediately before
//! every `call`. The `call` pushes 8 bytes (return address), and
//! the callee's `push rbp` pushes 8 more → RSP is 16-aligned in
//! the callee body.
//!
//! Before emitting `call` in the caller:
//! 1. Push caller-saved registers (N pushes × 8 bytes each)
//! 2. If N is odd: `sub rsp, 8` for alignment padding
//! 3. Move arguments to parameter registers
//! 4. `call`
//! 5. If N was odd: `add rsp, 8` to remove padding
//! 6. Pop caller-saved registers in reverse
//!
//! # Recursion-Limit Enforcement
//!
//! Stage 10 MP2 generates functions WITHOUT runtime recursion-limit
//! checks. Stage 10 boot.ir has no recursive calls, and recursion-
//! limit architecture is deferred to Stage 11+ (see ADR-026
//! V3-Tension section). The interpreter's MAX_RECURSION_DEPTH
//! (ADR-021) applies at interpretation, not compilation.

use std::collections::BTreeMap;

use iced_x86::code_asm::*;

use quarks_validator::{Atom, SExpr};

use crate::allocator::{LinearScanAllocator, RegSlot};
use crate::error::{CodegenError, CodegenErrorKind};
use crate::frame::{FunctionContext, MAX_ARITY};

// ── Top-level dispatch ──────────────────────────────────────

/// Compile a top-level expression or program.
///
/// Two patterns:
/// - `(program (fn ...) ... (call entry))` → full fn/call codegen
/// - Bare expression (e.g., `(add 1 2)`) → inline body, caller adds ret
pub fn compile_top_level(
    expr: &SExpr,
    asm: &mut CodeAssembler,
    allocator: &mut LinearScanAllocator,
) -> Result<RegSlot, CodegenError> {
    if let SExpr::List(items) = expr {
        if let Some(SExpr::Atom(Atom::Symbol(s))) = items.first() {
            if s == "program" {
                return compile_program(items, asm);
            }
        }
    }
    // Bare expression — MP1 path (caller adds mov rax + ret)
    compile_expr(expr, asm, allocator, None)
}

// ── Program compilation ─────────────────────────────────────

/// A parsed function definition.
struct FunctionDef {
    name: String,
    arity: usize,
    body: SExpr,
}

/// Compile `(program (fn ...) ... (call entry))`.
///
/// Pass 1: Collect all `(fn ...)` definitions and the entry `(call ...)`.
/// Pass 2: Emit entry wrapper (prologue + call + epilogue).
/// Pass 3: Emit each function body as labeled code.
fn compile_program(items: &[SExpr], asm: &mut CodeAssembler) -> Result<RegSlot, CodegenError> {
    let children = &items[1..]; // skip "program" head

    // Pass 1: collect fn definitions and entry call
    let mut functions: BTreeMap<String, FunctionDef> = BTreeMap::new();
    let mut entry_call: Option<&SExpr> = None;

    for child in children {
        if let SExpr::List(child_items) = child {
            if let Some(SExpr::Atom(Atom::Symbol(s))) = child_items.first() {
                match s.as_str() {
                    "fn" => {
                        let def = parse_fn_definition(child_items)?;
                        if functions.contains_key(&def.name) {
                            return Err(CodegenError::new(
                                CodegenErrorKind::DuplicateFunction,
                                format!("function '{}' defined twice", def.name),
                            ));
                        }
                        functions.insert(def.name.clone(), def);
                    }
                    "call" => {
                        entry_call = Some(child);
                    }
                    _ => {} // ignore other top-level forms
                }
            }
        }
    }

    let entry = entry_call.ok_or_else(|| {
        CodegenError::new(
            CodegenErrorKind::MissingEntryCall,
            "(program ...) must contain a (call ...) entry point",
        )
    })?;

    // Create labels for each function (forward-referenceable)
    let mut labels: BTreeMap<String, CodeLabel> = BTreeMap::new();
    for name in functions.keys() {
        labels.insert(name.clone(), asm.create_label());
    }

    // Pass 2: emit entry wrapper
    asm.push(rbp).map_err(emit_err)?;
    asm.mov(rbp, rsp).map_err(emit_err)?;

    let entry_args = match entry {
        SExpr::List(items) => &items[1..], // skip "call" head
        _ => unreachable!(),
    };

    let mut top_alloc = LinearScanAllocator::new();
    let result_slot = compile_call(entry_args, asm, &mut top_alloc, None, &labels, &functions)?;

    if result_slot != RegSlot::Rax {
        asm.mov(rax, result_slot.to_iced()).map_err(emit_err)?;
    }

    asm.pop(rbp).map_err(emit_err)?;
    asm.ret().map_err(emit_err)?;

    // Pass 3: emit each function body
    for (name, def) in &functions {
        compile_function(name, def, asm, &mut labels)?;
    }

    Ok(RegSlot::Rax)
}

fn parse_fn_definition(items: &[SExpr]) -> Result<FunctionDef, CodegenError> {
    // (fn name (params...) return-type body) — exactly 5 elements
    if items.len() != 5 {
        return Err(CodegenError::new(
            CodegenErrorKind::MalformedFunction,
            format!("fn form must have 5 elements, got {}", items.len()),
        ));
    }

    let name = match &items[1] {
        SExpr::Atom(Atom::Symbol(s)) => s.clone(),
        _ => {
            return Err(CodegenError::new(
                CodegenErrorKind::MalformedFunction,
                "fn name must be a symbol",
            ))
        }
    };

    let arity = match &items[2] {
        SExpr::List(params) => params.len(),
        _ => {
            return Err(CodegenError::new(
                CodegenErrorKind::MalformedFunction,
                "fn params must be a list",
            ))
        }
    };

    if arity > MAX_ARITY {
        return Err(CodegenError::new(
            CodegenErrorKind::ArityExceedsAbi,
            format!(
                "function '{}' arity {} exceeds Stage 10 max of {}",
                name, arity, MAX_ARITY
            ),
        ));
    }

    Ok(FunctionDef {
        name,
        arity,
        body: items[4].clone(),
    })
}

// ── Function compilation ────────────────────────────────────

/// Emit a function body with prologue and epilogue.
fn compile_function(
    name: &str,
    def: &FunctionDef,
    asm: &mut CodeAssembler,
    labels: &mut BTreeMap<String, CodeLabel>,
) -> Result<(), CodegenError> {
    let label = labels.get_mut(name).ok_or_else(|| {
        CodegenError::new(
            CodegenErrorKind::FunctionNotFound,
            format!("internal: no label for function '{}'", name),
        )
    })?;

    asm.set_label(label).map_err(emit_err)?;

    // Prologue
    asm.push(rbp).map_err(emit_err)?;
    asm.mov(rbp, rsp).map_err(emit_err)?;

    // Body — allocator is arity-aware
    let mut body_alloc = LinearScanAllocator::for_function(def.arity);
    let ctx = FunctionContext::new(def.arity)?;
    let result_slot = compile_expr(&def.body, asm, &mut body_alloc, Some(&ctx))?;

    // Move result to RAX if not already
    if result_slot != RegSlot::Rax {
        asm.mov(rax, result_slot.to_iced()).map_err(emit_err)?;
    }

    // Epilogue
    asm.pop(rbp).map_err(emit_err)?;
    asm.ret().map_err(emit_err)?;

    Ok(())
}

// ── Call compilation ────────────────────────────────────────

/// Compile a `(call name args...)` form.
///
/// Stage 10 MP2 supports calls only from the top-level entry wrapper
/// in `compile_program`. Body-internal calls (`(call ...)` inside a
/// function body) hit `UnsupportedInstruction` in `compile_list`.
fn compile_call(
    args: &[SExpr], // [name, arg0, arg1, ...]
    asm: &mut CodeAssembler,
    allocator: &mut LinearScanAllocator,
    ctx: Option<&FunctionContext>,
    labels: &BTreeMap<String, CodeLabel>,
    functions: &BTreeMap<String, FunctionDef>,
) -> Result<RegSlot, CodegenError> {
    if args.is_empty() {
        return Err(CodegenError::new(
            CodegenErrorKind::EmptyList,
            "call requires function name",
        ));
    }

    let fn_name = match &args[0] {
        SExpr::Atom(Atom::Symbol(s)) => s.as_str(),
        _ => {
            return Err(CodegenError::new(
                CodegenErrorKind::NonSymbolHead,
                "call target must be a symbol",
            ))
        }
    };

    let def = functions.get(fn_name).ok_or_else(|| {
        CodegenError::new(
            CodegenErrorKind::FunctionNotFound,
            format!("call target '{}' not defined", fn_name),
        )
    })?;

    let arg_exprs = &args[1..];
    if arg_exprs.len() != def.arity {
        return Err(CodegenError::new(
            CodegenErrorKind::ArityMismatch,
            format!(
                "call '{}' expects {} args, got {}",
                fn_name,
                def.arity,
                arg_exprs.len()
            ),
        ));
    }

    // 1. Evaluate each argument into a body-local slot
    let mut arg_slots = Vec::with_capacity(arg_exprs.len());
    for arg_expr in arg_exprs {
        let slot = compile_expr(arg_expr, asm, allocator, ctx)?;
        arg_slots.push(slot);
    }

    // 2. Save caller-saved registers (excluding arg_slots we're about to consume)
    let in_use = allocator.in_use_slots();
    let to_save: Vec<RegSlot> = in_use
        .iter()
        .filter(|s| !arg_slots.contains(s))
        .copied()
        .collect();

    for slot in &to_save {
        asm.push(slot.to_iced()).map_err(emit_err)?;
    }

    // 3. Stack alignment: N pushes × 8 bytes. If N odd → sub rsp, 8
    let padding_needed = (to_save.len() % 2) != 0;
    if padding_needed {
        asm.sub(rsp, 8i32).map_err(emit_err)?;
    }

    // 4. Move evaluated arguments to System V AMD64 parameter registers
    for (idx, &slot) in arg_slots.iter().enumerate() {
        let param_reg = match idx {
            0 => rdi,
            1 => rsi,
            2 => rdx,
            3 => rcx,
            4 => r8,
            5 => r9,
            _ => {
                return Err(CodegenError::new(
                    CodegenErrorKind::ArityExceedsAbi,
                    format!("argument {} exceeds max of {}", idx, MAX_ARITY),
                ))
            }
        };
        asm.mov(param_reg, slot.to_iced()).map_err(emit_err)?;
    }

    // 5. Release arg slots (parameter registers now hold them)
    for &slot in &arg_slots {
        allocator.release(slot);
    }

    // 6. Emit call
    let label = labels.get(fn_name).ok_or_else(|| {
        CodegenError::new(
            CodegenErrorKind::FunctionNotFound,
            format!("internal: no label for '{}'", fn_name),
        )
    })?;
    asm.call(*label).map_err(emit_err)?;

    // 7. Restore stack alignment padding
    if padding_needed {
        asm.add(rsp, 8i32).map_err(emit_err)?;
    }

    // 8. Restore caller-saved registers in reverse order
    for slot in to_save.iter().rev() {
        asm.pop(slot.to_iced()).map_err(emit_err)?;
    }

    // 9. Result is in RAX. Acquire RAX in body allocator.
    //
    // Note: for Stage 10 MP2 (entry call only), the allocator starts
    // empty before the call so to_save is empty and no RAX conflict
    // occurs. Stage 11+ nested calls (call inside function body)
    // would need to capture RAX before restoring, but nested calls
    // are unsupported in Stage 10 (UnsupportedInstruction error).
    let result = allocator.acquire()?;
    debug_assert_eq!(result, RegSlot::Rax);
    Ok(result)
}

// ── Expression compilation ──────────────────────────────────

/// Compile an expression. Returns the register slot holding the result.
///
/// `ctx` is `Some` when inside a function body (for %n resolution),
/// `None` for top-level / bare expressions.
pub fn compile_expr(
    expr: &SExpr,
    asm: &mut CodeAssembler,
    allocator: &mut LinearScanAllocator,
    ctx: Option<&FunctionContext>,
) -> Result<RegSlot, CodegenError> {
    match expr {
        SExpr::Atom(atom) => compile_atom(atom, asm, allocator, ctx),
        SExpr::List(items) => compile_list(items, asm, allocator, ctx),
    }
}

fn compile_atom(
    atom: &Atom,
    asm: &mut CodeAssembler,
    allocator: &mut LinearScanAllocator,
    ctx: Option<&FunctionContext>,
) -> Result<RegSlot, CodegenError> {
    match atom {
        Atom::Integer(n) => {
            let slot = allocator.acquire()?;
            asm.mov(slot.to_iced(), *n).map_err(|e| {
                CodegenError::new(
                    CodegenErrorKind::EmitFailed,
                    format!("mov {:?}, {} failed: {}", slot, n, e),
                )
            })?;
            Ok(slot)
        }
        Atom::Parameter(n) => {
            let ctx = ctx.ok_or_else(|| {
                CodegenError::new(
                    CodegenErrorKind::UnsupportedAtom,
                    format!("%{} parameter reference outside function body", n),
                )
            })?;
            let param_reg = ctx.resolve_parameter(*n)?;
            // Move parameter value into a body-local slot
            let slot = allocator.acquire()?;
            asm.mov(slot.to_iced(), param_reg).map_err(|e| {
                CodegenError::new(
                    CodegenErrorKind::EmitFailed,
                    format!("mov {:?}, %{} failed: {}", slot, n, e),
                )
            })?;
            Ok(slot)
        }
        Atom::Symbol(_) => Err(CodegenError::new(
            CodegenErrorKind::UnsupportedAtom,
            "bare symbol atom not evaluable (only in fn/call head positions)",
        )),
        Atom::Handle(id) => {
            // ADR-027: Handle is opaque u64 ID. Codegen emits as 64-bit
            // immediate — identical to Integer at the machine-code level.
            // The Handle ID is meaningful only when dereferenced via the
            // arena's handle table at runtime.
            let slot = allocator.acquire()?;
            asm.mov(slot.to_iced(), *id as i64).map_err(|e| {
                CodegenError::new(
                    CodegenErrorKind::EmitFailed,
                    format!("mov {:?}, handle@{} failed: {}", slot, id, e),
                )
            })?;
            Ok(slot)
        }
        Atom::Bytes(_) => Err(CodegenError::new(
            CodegenErrorKind::UnsupportedAtom,
            "Bytes atoms not supported in Stage 10 (Phase 4+ material)",
        )),
    }
}

fn compile_list(
    items: &[SExpr],
    asm: &mut CodeAssembler,
    allocator: &mut LinearScanAllocator,
    ctx: Option<&FunctionContext>,
) -> Result<RegSlot, CodegenError> {
    if items.is_empty() {
        return Err(CodegenError::new(
            CodegenErrorKind::EmptyList,
            "cannot compile empty list",
        ));
    }

    let head = match &items[0] {
        SExpr::Atom(Atom::Symbol(s)) => s.as_str(),
        _ => {
            return Err(CodegenError::new(
                CodegenErrorKind::NonSymbolHead,
                "list head must be a symbol",
            ))
        }
    };

    let args = &items[1..];

    match head {
        "add" => binary_op(args, asm, allocator, ctx, BinOp::Add),
        "sub" => binary_op(args, asm, allocator, ctx, BinOp::Sub),
        "mul" => binary_op(args, asm, allocator, ctx, BinOp::Mul),
        // Phase 4 Step 1 — bitwise AND/OR/XOR map directly onto the
        // commutative two-operand integer encodings; shifts are
        // handled separately because x86 requires the count in CL.
        // Codegen does not bounds-check the shift count — the
        // interpreter is the front line for `ShiftCountOutOfRange`,
        // and the compiler-emitted code accepts the hardware's
        // low-6-bit masking semantics for already-type-checked input.
        "bit-and" => binary_op(args, asm, allocator, ctx, BinOp::BitAnd),
        "bit-or" => binary_op(args, asm, allocator, ctx, BinOp::BitOr),
        "bit-xor" => binary_op(args, asm, allocator, ctx, BinOp::BitXor),
        "bit-shl" => shift_op(args, asm, allocator, ctx, ShiftOp::Shl),
        "bit-shr" => shift_op(args, asm, allocator, ctx, ShiftOp::Shr),
        "call" => Err(CodegenError::new(
            CodegenErrorKind::UnsupportedInstruction,
            "call inside expression not supported in Stage 10 MP2 \
             (only top-level entry call); nested calls deferred to Stage 11+",
        )),
        "fn" | "program" => Err(CodegenError::new(
            CodegenErrorKind::UnsupportedInstruction,
            format!("'{}' only valid at top-level (program ...) wrapper", head),
        )),
        _ => Err(CodegenError::new(
            CodegenErrorKind::UnsupportedInstruction,
            format!("instruction '{}' not supported in Stage 10 scope", head),
        )),
    }
}

// ── Binary operations ───────────────────────────────────────

#[derive(Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    /// Phase 4 Step 1 — `r64 &= r64`.
    BitAnd,
    /// Phase 4 Step 1 — `r64 |= r64`.
    BitOr,
    /// Phase 4 Step 1 — `r64 ^= r64`.
    BitXor,
}

fn binary_op(
    args: &[SExpr],
    asm: &mut CodeAssembler,
    allocator: &mut LinearScanAllocator,
    ctx: Option<&FunctionContext>,
    op: BinOp,
) -> Result<RegSlot, CodegenError> {
    if args.len() != 2 {
        return Err(CodegenError::new(
            CodegenErrorKind::ArityMismatch,
            format!("binary op requires exactly 2 arguments, got {}", args.len()),
        ));
    }

    let left_slot = compile_expr(&args[0], asm, allocator, ctx)?;
    let right_slot = compile_expr(&args[1], asm, allocator, ctx)?;

    let dst = left_slot.to_iced();
    let src = right_slot.to_iced();

    match op {
        BinOp::Add => asm.add(dst, src),
        BinOp::Sub => asm.sub(dst, src),
        BinOp::Mul => asm.imul_2(dst, src),
        BinOp::BitAnd => asm.and(dst, src),
        BinOp::BitOr => asm.or(dst, src),
        BinOp::BitXor => asm.xor(dst, src),
    }
    .map_err(emit_err)?;

    allocator.release(right_slot);
    Ok(left_slot)
}

// ── Shift operations ────────────────────────────────────────
//
// x86 mandates the shift count in CL (low 8 bits of RCX). The
// allocator's three-slot pool is RAX, RCX, RDX — so the count
// register is contended. Three shapes after `compile_expr` runs
// on both operands:
//
//   1. right_slot == RCX → count is already where it belongs.
//   2. left_slot == RCX  → count target conflicts with the value
//      being shifted; we `xchg` to move the value into right_slot's
//      register and the count into CL.
//   3. neither slot is RCX → with only 2 of 3 slots in use, RCX is
//      free; `mov rcx, right` and emit the shift.
//
// The returned slot is whichever register ends up holding the
// shifted result; the other allocated slot is released.
//
// TODO(Stage 11+): AOT shift bounds-check divergence.
//
// `shift_op` emits bare `shl r64, cl` / `sar r64, cl`. x86 masks
// the count to its low 6 bits (`count & 0x3F`), so a runtime count
// of 64 silently shifts by 0, a count of 65 shifts by 1, and a
// negative count shifts by its low-6-bit reinterpretation. The
// interpreter (`after_binop_rhs` in quarks-interpreter) by
// contrast surfaces any count outside `0..64` as
// `InterpretErrorKind::ShiftCountOutOfRange`.
//
// This is acceptable as long as AOT output is isolated (Stage 10:
// boot.ir is hand-curated and only well-typed shift counts in
// `0..64` reach codegen). It is **not** acceptable once the AOT
// pipeline accepts LLM-generated programs at runtime (Stage 11+).
// At that point this function must grow a prologue along the
// lines of:
//
//     cmp  rcx, 64
//     jae  .shift_out_of_range     ; unsigned cmp catches both
//                                  ; n >= 64 AND n < 0
//     shl  rdst, cl
//     ...
//   .shift_out_of_range:
//     ; emit a trap / branch to a per-program ShiftOutOfRange
//     ; handler that converts to the Quarks typed error.
//
// Tracking: aligns with `ShiftCountOutOfRange` in
// `crates/quarks-interpreter/src/error.rs`.

#[derive(Clone, Copy)]
enum ShiftOp {
    /// `shl r64, cl`.
    Shl,
    /// `sar r64, cl` — arithmetic right shift, matching the
    /// interpreter's `i64 >>` semantics.
    Shr,
}

fn shift_op(
    args: &[SExpr],
    asm: &mut CodeAssembler,
    allocator: &mut LinearScanAllocator,
    ctx: Option<&FunctionContext>,
    op: ShiftOp,
) -> Result<RegSlot, CodegenError> {
    if args.len() != 2 {
        return Err(CodegenError::new(
            CodegenErrorKind::ArityMismatch,
            format!("shift op requires exactly 2 arguments, got {}", args.len()),
        ));
    }

    let left_slot = compile_expr(&args[0], asm, allocator, ctx)?;
    let right_slot = compile_expr(&args[1], asm, allocator, ctx)?;

    let (result_slot, released_slot) = if right_slot == RegSlot::Rcx {
        // Case 1: count already in CL. Shift left_slot in place.
        (left_slot, right_slot)
    } else if left_slot == RegSlot::Rcx {
        // Case 2: value being shifted occupies RCX. Swap RCX with
        // the right_slot register so the value lives in
        // right_slot's register and the count moves into CL.
        asm.xchg(rcx, right_slot.to_iced()).map_err(emit_err)?;
        (right_slot, left_slot)
    } else {
        // Case 3: RCX is free (3-slot pool, only 2 in use). Move
        // the count into RCX and leave the value in left_slot.
        asm.mov(rcx, right_slot.to_iced()).map_err(emit_err)?;
        (left_slot, right_slot)
    };

    let dst = result_slot.to_iced();
    match op {
        ShiftOp::Shl => asm.shl(dst, cl),
        ShiftOp::Shr => asm.sar(dst, cl),
    }
    .map_err(emit_err)?;

    allocator.release(released_slot);
    Ok(result_slot)
}

// ── Helpers ─────────────────────────────────────────────────

fn emit_err(e: impl std::fmt::Display) -> CodegenError {
    CodegenError::new(CodegenErrorKind::EmitFailed, format!("emit failed: {}", e))
}
