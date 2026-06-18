// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit state-machine driver for the Quarks interpreter.
//!
//! This module replaces the original recursive tree-walker (Stage 9
//! Phase 2) with an explicit instruction stack + value stack + call
//! stack. Each [`Machine::step`] call dispatches exactly one
//! [`Instr`], so a watchdog or cooperative scheduler can preempt the
//! interpreter between any two atomic operations rather than only
//! between whole `(program ...)` runs.
//!
//! The refactor preserves **bitwise-equivalent observable semantics**
//! relative to the recursive tree-walker: identical operand
//! evaluation order, identical type-check timing, identical
//! handle-counter dispatch order, identical error variants. See the
//! Stage 12 Paket A specification in
//! `docs/discovery/stage-12-completion-plan.md` §2A and the
//! determinism constraints in
//! `docs/discovery/hardware-abstraction-constraints.md` §1.
//!
//! # Determinism contract
//!
//! - **Explicit Program Counter / Stack.** No reliance on the Rust
//!   call stack — both control flow (`instr_stack`) and intermediate
//!   values (`value_stack`) live in `alloc::vec::Vec` so they can be
//!   pre-allocated, bounded, and inspected.
//! - **No floating point.** The ISA (`add`, `sub`, `mul`, …) is
//!   integer-only; the machine never introduces FP.
//! - **`BTreeMap`-only.** Function lookups go through the existing
//!   [`crate::symbol_table::SymbolTable`], which uses
//!   [`alloc::collections::BTreeMap`] for deterministic iteration.
//!   The machine itself holds no maps.
//! - **No wallclock reads.** The only counter exposed is
//!   [`Machine::instructions_executed`], which is incremented per
//!   step and is purely an instruction count.
//! - **Per-instruction error abort.** On error, `step` returns the
//!   same [`InterpretError`] kind the recursive interpreter would
//!   have raised, and execution halts in place; the same error is
//!   reported on subsequent `step` calls (machine is poisoned).
//!
//! # Ownership (A.4)
//!
//! Stage 12 Paket A.4 made the machine **lifetime-free**: there is no
//! `&'a SExpr` reference anywhere in the machine state. Subtree
//! references are stored as [`alloc::boxed::Box<SExpr>`] for `Eval`
//! continuations and [`alloc::sync::Arc<SExpr>`] for function bodies
//! (which are shared across recursive calls). This lets a
//! [`Session`] live inside a `Sandbox` without producing a
//! self-referential struct — the precondition for instruction-level
//! watchdog preemption.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use quarks_validator::{Atom, SExpr};

use crate::error::{InterpretError, InterpretErrorKind};
use crate::frame::{CallStack, StackFrame};
use crate::policy::{PolicyContext, PolicyError};
use crate::symbol_table::{collect_signatures, SymbolTable};
use crate::value::Value;

/// Outcome of a single [`Machine::step`] / [`Session::step`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome {
    /// More work remains; call `step` again.
    Continue,
    /// The instruction stack has drained and the final value is
    /// returned. Subsequent `step` calls return `Done` with the same
    /// value popped off again, or — if the value stack is also drained
    /// — an error.
    Done(Value),
}

/// Pre-A.2 (F-H14): fixed signature-test value for `(register …)`
/// until Paket A wires the interpreter into a per-sandbox
/// [`HandleTable`] context. The prior implementation used a static
/// `AtomicU64` counter that lived across test boundaries, which (i)
/// broke deterministic-reproducibility (12h §1.5) and (ii) gave
/// false comfort that handle allocation was "real". Returning a
/// constant makes the lack of a real allocator self-evident; the
/// type signature (`Value::Handle(_)`) still flows through every
/// validator/typechecker test path.
///
/// Per ADR-027 the value `1` is the first non-null handle id.
const MOCK_REGISTER_HANDLE_ID: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinOpKind {
    Add,
    Sub,
    Mul,
    /// Pre-A.2 (F-C2): validator/interpreter convergence. `Div` uses
    /// `i64::checked_div` and surfaces division-by-zero (and the
    /// `i64::MIN / -1` overflow) as [`InterpretErrorKind::DivisionByZero`]
    /// — never a panic.
    Div,
    /// Phase 4 Step 1 — bitwise AND on i64. Total, never traps.
    BitAnd,
    /// Phase 4 Step 1 — bitwise OR on i64. Total, never traps.
    BitOr,
    /// Phase 4 Step 1 — bitwise XOR on i64. Total, never traps.
    BitXor,
    /// Phase 4 Step 1 — wrapping left shift. Runtime count outside
    /// `0..64` surfaces as [`InterpretErrorKind::ShiftCountOutOfRange`].
    Shl,
    /// Phase 4 Step 1 — arithmetic right shift (SAR). Runtime count
    /// outside `0..64` surfaces as
    /// [`InterpretErrorKind::ShiftCountOutOfRange`].
    Shr,
}

/// Paket B.1: comparison operators on `i64` that produce `Bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CmpOpKind {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

/// Paket B.2: logical binary operators on `Bool`.
/// `and`/`or` evaluate both operands (no short-circuit), matching
/// the validator's eager type-checking semantics. Short-circuit
/// is deferred to a later language MP if it proves valuable for
/// LLM-generated policy code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoolBinOpKind {
    And,
    Or,
}

/// Pre-A.1 (F-C1) — Package A state machine: maximum depth of the
/// explicit instruction stack. The state machine structurally
/// eliminates the native stack-overflow vector of the recursive
/// tree walker, because [`Instr`]s live on the heap-allocated
/// [`Vec<Instr>`](alloc::vec::Vec). This guard is the heap variant
/// of the old `MAX_EXPR_DEPTH=256` limit: it protects the arena
/// from OOM caused by pathologically deeply nested inputs
/// (`(add (add (add … 1 1 …)))`).
///
/// `1024` is generous for real LLM policies (typical depth < 100),
/// yet stays well below the point at which the recursive
/// [`SExpr::clone`] operation in [`find_program_entry`] and
/// [`Session::new`] overflows the Ring-0 native stack. The
/// heap-allocated instr stack itself is `1024 × ~16 bytes ≈ 16 KiB`
/// worst case — negligible.
///
/// Triggers [`InterpretErrorKind::ExpressionDepthExceeded`] — no
/// panic, deterministically reproducible.
pub const MAX_INSTR_STACK_DEPTH: usize = 1024;

/// One unit of pending work in the state machine.
///
/// Each variant corresponds to one atomic step. The recursive
/// tree-walker's "evaluate, type-check, evaluate, type-check, apply"
/// flow is decomposed into a sequence of these variants so the
/// type-check timing matches the recursive interpreter's *before*-
/// next-operand-eval semantics. This preserves observable side
/// effects (e.g. handle counter increments) bit-for-bit.
pub(crate) enum Instr {
    /// Evaluate this expression and push the resulting Value on the
    /// value stack.
    Eval(Box<SExpr>),
    /// Continuation after the lhs of a binary op has been evaluated
    /// and pushed on the value stack. This step type-checks the lhs
    /// as an integer (matching the recursive walker's
    /// type-check-before-next-eval order), then schedules the rhs
    /// evaluation and a final `BinOpAfterRhs` continuation.
    BinOpAfterLhs { op: BinOpKind, rhs: Box<SExpr> },
    /// Continuation after the rhs has been evaluated. Type-checks
    /// the rhs, computes the wrapping integer op, and pushes the
    /// result.
    BinOpAfterRhs { op: BinOpKind, lhs: i64 },
    /// Continuation after all `argc` arguments of a function call
    /// have been evaluated. Pops them off the value stack,
    /// constructs the call frame, pushes it onto the call stack
    /// (enforcing `MAX_RECURSION_DEPTH`), and schedules the body
    /// evaluation followed by a `PopFrame` continuation.
    InvokeCall { body: Arc<SExpr>, argc: usize },
    /// Final continuation of a function call: pops the topmost call
    /// frame. The body's return value is already on the value stack
    /// at this point and is left untouched.
    PopFrame,
    /// Continuation after a `(register x)` argument has been
    /// evaluated. Type-checks it as bytes and replaces with a freshly
    /// minted Handle.
    ApplyRegister,
    /// Continuation after all `argc` value-args of a `(policy ...)`
    /// or `(query ...)` form have been evaluated. Pops them off the
    /// value stack and dispatches into the [`PolicyContext`].
    ApplyPolicy {
        is_policy: bool,
        subsystem: String,
        operation: String,
        argc: usize,
    },
    /// Paket B.1: continuation after the lhs of a comparison op.
    /// Type-checks lhs as `Integer`, schedules rhs eval + post.
    CmpOpAfterLhs { op: CmpOpKind, rhs: Box<SExpr> },
    /// Paket B.1: continuation after the rhs of a comparison op.
    /// Computes the Bool result and pushes it.
    CmpOpAfterRhs { op: CmpOpKind, lhs: i64 },
    /// Paket B.2: continuation after the lhs of `and`/`or`.
    BoolBinOpAfterLhs { op: BoolBinOpKind, rhs: Box<SExpr> },
    /// Paket B.2: continuation after the rhs of `and`/`or`.
    BoolBinOpAfterRhs { op: BoolBinOpKind, lhs: bool },
    /// Paket B.2: continuation after the operand of `not`.
    BoolNot,
    /// Paket B.1b: continuation after `(if cond then else)` cond is
    /// evaluated; pops Bool, schedules either then or else.
    IfBranch {
        then_expr: Box<SExpr>,
        else_expr: Box<SExpr>,
    },
    /// Phase 4 Step 4: continuation after a `(cond …)` clause's
    /// predicate has been evaluated. Pops the Bool from the value
    /// stack and either schedules `body` (predicate was true) or
    /// peels the next `(predicate body)` pair from `rest` (predicate
    /// was false). If `rest` is empty when the predicate is false,
    /// schedules the captured `default` body. Clauses in `rest` are
    /// stored in REVERSE source order so that `Vec::pop()` produces
    /// the next clause in O(1).
    CondAfterPred {
        /// Body to evaluate if the just-evaluated predicate was true.
        body: Box<SExpr>,
        /// Remaining `(predicate, body)` pairs to try if false,
        /// reversed so the next-in-source-order is the last element.
        rest: Vec<(SExpr, SExpr)>,
        /// Mandatory `(default body)` fallback. Captured by the
        /// queue_cond planner and forwarded unchanged through every
        /// CondAfterPred until either some predicate matches or the
        /// rest queue drains.
        default: Box<SExpr>,
    },
    /// Paket B.3: continuation after `(let %n value body)` value is
    /// evaluated; binds value into the current frame at `idx`,
    /// schedules body Eval + LetUnbind.
    LetBind { idx: u32, body: Box<SExpr> },
    /// Paket B.3: unbind a let-local after its body has been
    /// evaluated. Pops the binding from the current frame.
    LetUnbind { idx: u32 },
    /// Paket B.1c: end-of-iteration continuation for `(loop body)`.
    /// Body produced a residual value; discard it and re-schedule
    /// another iteration.
    LoopIterEnd { body: Arc<SExpr> },
    /// Paket B.1c: `(break v)` after evaluating its value
    /// expression. Unwinds to the innermost loop frame and pushes
    /// the captured break value on the value stack.
    ApplyBreak,
    /// Paket B.1d: per-iteration cond-evaluation continuation for
    /// `(while cond body)`.
    WhileEvalCond { cond: Arc<SExpr>, body: Arc<SExpr> },
    /// Paket B.1d: post-cond continuation for `(while cond body)`.
    /// Pops Bool, dispatches body+iter or implicit (break false).
    WhileAfterCond { cond: Arc<SExpr>, body: Arc<SExpr> },
    /// Paket B.1d: post-body continuation for `(while cond body)`.
    /// Discards body residual and schedules the next cond eval.
    WhileAfterBody { cond: Arc<SExpr>, body: Arc<SExpr> },
    /// Paket B (discard wiring): post-`(discard expr)` pop. The
    /// `(seq …)` form does NOT push this — the validator
    /// guarantees every effect arg is stack-neutral (typically
    /// pre-wrapped in `(discard …)`), so `seq` just sequences
    /// the children's evaluation without an additional pop.
    ApplyDiscard,
    /// Paket B.2: post-eval continuation for `(list e1 e2 … eN)`.
    /// Pops `count` i64 values off the value stack (deepest-first
    /// becomes leftmost-element) and pushes a single `Value::List`.
    ApplyList { count: usize },
    /// Paket B.2: post-eval continuation for `(list-get l i)`.
    ApplyListGet,
    /// Paket B.2: post-eval continuation for `(list-len l)`.
    ApplyListLen,
    /// Paket B.2: post-eval continuation for `(list-append l x)`.
    ApplyListAppend,
    /// Phase 4 Step 3: post-eval continuation for
    /// `(list-slice l start len)`.
    ApplyListSlice,
    /// Paket B.3: post-eval continuation for `(map-put m k v)`.
    ApplyMapPut,
    /// Paket B.3: post-eval continuation for `(map-get m k)`.
    ApplyMapGet,
    /// Paket B.3: post-eval continuation for `(map-contains m k)`.
    ApplyMapContains,
    /// Paket B.5: post-eval continuation for `(string-from-int n)`.
    ApplyStringFromInt,
    /// Paket B.5: post-eval continuation for `(string-concat a b)`.
    ApplyStringConcat,
    /// Paket B.5: post-eval continuation for `(string-eq a b)`.
    ApplyStringEq,
    /// Paket B.6: post-eval continuation for `(read-handle h)` —
    /// pops the handle off the value stack and dispatches into the
    /// `PolicyContext::read_handle` surface.
    ApplyReadHandle,
    /// Phase 4 Step 2: post-eval continuation for `(bytes-len b)`.
    ApplyBytesLen,
    /// Phase 4 Step 2: post-eval continuation for `(bytes-get b i)`.
    ApplyBytesGet,
    /// Phase 4 Step 2: post-eval continuation for
    /// `(bytes-slice b start len)`.
    ApplyBytesSlice,
    /// Phase 4 Step 2: post-eval continuation for
    /// `(bytes-concat a b)`.
    ApplyBytesConcat,
    /// Phase 4 Step 2: post-eval continuation for `(bytes-eq a b)`.
    ApplyBytesEq,
    /// Phase 4 Step 2: post-eval continuation for
    /// `(bytes-from-int n)`. Encodes the i64 as little-endian 8
    /// bytes (mirrors Rust `i64::to_le_bytes`).
    ApplyBytesFromInt,
    /// Paket B.6: post-eval continuation for `(write-host-state key
    /// value)` — pops the value off the value stack and dispatches
    /// into `PolicyContext::write_host_state` with the captured key.
    ApplyWriteHostState { key: String },
    /// Phase 4 Step 5: post-eval continuation for `(some v)`. Pops
    /// the operand and wraps it in `Value::Maybe(Some(_))`.
    ApplySome,
    /// Phase 4 Step 5: zero-operand step for `(none)`. Pushes
    /// `Value::Maybe(None)` directly — no preceding `Eval` needed.
    ApplyNone,
    /// Phase 4 Step 5: post-eval continuation for `(is-some m)`.
    ApplyIsSome,
    /// Phase 4 Step 5: post-eval continuation for `(is-none m)`.
    ApplyIsNone,
    /// Phase 4 Step 5: post-eval continuation for `(unwrap m)`. Pops
    /// the Maybe and either pushes the inner value or surfaces
    /// `UnwrapOnNone`.
    ApplyUnwrap,
    /// Phase 4 Step 5: post-eval continuation for `(unwrap-or m d)`.
    /// Pops the default and the Maybe; pushes the inner value if
    /// Some, the default if None. Never traps.
    ApplyUnwrapOr,
    /// Paket B.4: per-iteration evaluation of `(loop-with-bound bound
    /// body)`. `remaining` counts down each iteration; when it
    /// reaches 0 without a `(break v)`, the loop emits an implicit
    /// `Value::Bool(false)` (matching `while`).
    LoopBoundedIter { remaining: i64, body: Arc<SExpr> },
    /// Paket B.4: post-body continuation for `loop-with-bound` — pops
    /// the body's residual, decrements `remaining`, and either
    /// schedules another iteration or finalises the loop.
    LoopBoundedAfterBody { remaining: i64, body: Arc<SExpr> },
    /// Paket B.4: post-source continuation for `(for %n source
    /// body)`. Captures the iteration source and dispatches the
    /// first iteration; subsequent iterations are scheduled by
    /// [`Instr::ForIter`].
    ForAfterSource { idx: u32, body: Arc<SExpr> },
    /// Paket B.4: per-element iteration of a `for` loop. Pre-binds
    /// the next element to `%idx` and schedules the body's
    /// evaluation followed by [`Instr::ForAfterBody`].
    ForIter {
        idx: u32,
        remaining: Vec<i64>,
        body: Arc<SExpr>,
    },
    /// Paket B.4: post-body continuation for a `for` iteration —
    /// drops the residual value, unbinds the loop variable, and
    /// schedules the next iteration (or finalises the loop).
    ForAfterBody {
        idx: u32,
        remaining: Vec<i64>,
        body: Arc<SExpr>,
    },
    /// Phase 4 Step 6: post-eval continuation for
    /// `(struct-new name v1 v2 … vN)`. The struct name and the
    /// ordered list of field names are captured at queue time from
    /// the symbol table; the `argc` values are popped off the value
    /// stack in source order and zipped with `fields` to build the
    /// `Value::Struct` map.
    ApplyStructNew { name: String, fields: Vec<String> },
    /// Phase 4 Step 6: post-eval continuation for
    /// `(struct-get expr field-name)`. Pops a Struct value, returns
    /// the named field's value. Captures the field name at queue
    /// time so the continuation is self-contained.
    ApplyStructGet { field: String },
    /// Phase 4 Step 6: post-eval continuation for
    /// `(struct-set expr field-name new-value)`. Pops the new value
    /// and the Struct, builds a NEW Struct with the field updated,
    /// pushes the new Struct. No in-place mutation — functional
    /// update semantics.
    ApplyStructSet { field: String },
    /// Phase 4 Step 7: continuation after the `(match scrutinee …)`
    /// scrutinee has been evaluated. The scrutinee value is on top
    /// of the value stack. The continuation tries each case in
    /// `cases` (source order) against the scrutinee, executing the
    /// first body whose pattern matches. If no case matches and the
    /// list drains (only possible on adversarial/unvalidated IR —
    /// validator requires a wildcard last case), the machine
    /// surfaces [`InterpretErrorKind::MatchNonExhaustive`].
    ///
    /// We carry `cases` in source order — Vec::remove(0) drains
    /// from the front; the list is small (typically ≤ 5) so the
    /// O(n) shift per peel is fine. A reversed layout (à la
    /// CondAfterPred) would also work; the validator already caps
    /// case count via overall AST size.
    MatchDispatch { cases: Vec<MatchCase> },
    /// Phase 4 Step 7: post-body continuation for a single match
    /// case. Unbinds every binding the pattern introduced — same
    /// scope discipline as `LetUnbind`. Bindings are unbound in
    /// reverse order so the most-recently-bound slot pops first.
    MatchUnbind { bindings: Vec<u32> },
}

/// Phase 4 Step 7 — a single match clause as carried by the
/// [`Instr::MatchDispatch`] continuation. `pattern` is the raw AST
/// (parsed by the validator's structural pass) and `body` is the
/// expression to evaluate when the pattern matches the scrutinee.
/// Carrying the AST through to runtime lets the interpreter
/// re-validate shape defensively rather than trusting a pre-baked
/// classification that an adversarial caller could forge.
#[derive(Debug, Clone)]
pub(crate) struct MatchCase {
    pattern: SExpr,
    body: SExpr,
}

/// Explicit state-machine driver — replaces the original recursive
/// tree-walker. Holds an explicit instruction stack, value stack,
/// and call stack so that execution can be paused between any two
/// atomic operations.
///
/// Typical usage is via [`Session`], which bundles the machine with
/// its [`SymbolTable`] for one program run. Direct use is reserved
/// for code that already owns the symbol table externally (e.g.
/// callers that re-use the table across multiple program entries).
/// Paket B.1c: an active-loop record on the machine's loop stack.
/// When `(break v)` fires inside the body, the machine truncates
/// the instruction and value stacks back to these snapshots,
/// then re-pushes `v` as the loop's residual value.
#[derive(Debug, Clone, Copy)]
struct LoopFrame {
    instr_stack_depth_at_entry: usize,
    value_stack_depth_at_entry: usize,
}

pub struct Machine {
    instr_stack: Vec<Instr>,
    value_stack: Vec<Value>,
    call_stack: CallStack,
    /// Paket B.1c: stack of active `(loop …)` / `(while …)`
    /// frames, used by `(break v)` to unwind.
    loop_stack: Vec<LoopFrame>,
    instructions_executed: u64,
}

impl Machine {
    /// Build a fresh machine seeded with `entry` as the top
    /// instruction. The caller is responsible for choosing the
    /// program entry point (typically [`find_program_entry`]).
    pub fn new(entry: SExpr) -> Self {
        let mut m = Self {
            instr_stack: Vec::new(),
            value_stack: Vec::new(),
            call_stack: CallStack::new(),
            loop_stack: Vec::new(),
            instructions_executed: 0,
        };
        m.instr_stack.push(Instr::Eval(Box::new(entry)));
        m
    }

    /// `true` if the instruction stack has drained. A subsequent
    /// `step` will return `Done` with the final value popped off the
    /// value stack.
    pub fn is_done(&self) -> bool {
        self.instr_stack.is_empty()
    }

    /// Monotonic count of [`Instr`]s dispatched by `step` since
    /// construction. Watchdog and scheduler code can sample this
    /// counter for instruction-level budget enforcement.
    pub fn instructions_executed(&self) -> u64 {
        self.instructions_executed
    }

    /// Current call-stack depth (number of active function frames).
    #[allow(dead_code)]
    pub(crate) fn call_depth(&self) -> usize {
        self.call_stack.depth()
    }

    /// Execute exactly one instruction. Returns `Done(v)` when the
    /// instruction stack is empty at entry — meaning the previous
    /// step's pushed value is the program's final result.
    ///
    /// On error the machine is poisoned: the instruction stack is
    /// left in whatever state it was, and callers should drop the
    /// machine rather than calling `step` again.
    pub fn step<C: PolicyContext + ?Sized>(
        &mut self,
        symbols: &SymbolTable,
        ctx: &mut C,
    ) -> Result<StepOutcome, InterpretError> {
        let instr = match self.instr_stack.pop() {
            Some(i) => i,
            None => {
                let v = self.value_stack.pop().ok_or_else(|| {
                    InterpretError::new(
                        InterpretErrorKind::EmptyList,
                        "machine drained with no value produced",
                    )
                })?;
                return Ok(StepOutcome::Done(v));
            }
        };

        // Count the instruction we are about to dispatch.
        self.instructions_executed = self.instructions_executed.saturating_add(1);

        match instr {
            Instr::Eval(expr) => self.dispatch_eval(*expr, symbols)?,
            Instr::BinOpAfterLhs { op, rhs } => self.after_binop_lhs(op, rhs)?,
            Instr::BinOpAfterRhs { op, lhs } => self.after_binop_rhs(op, lhs)?,
            Instr::InvokeCall { body, argc } => self.invoke_call(body, argc)?,
            Instr::PopFrame => {
                let _ = self.call_stack.pop();
            }
            Instr::ApplyRegister => self.apply_register()?,
            Instr::ApplyPolicy {
                is_policy,
                subsystem,
                operation,
                argc,
            } => self.apply_policy(is_policy, &subsystem, &operation, argc, ctx)?,
            Instr::CmpOpAfterLhs { op, rhs } => self.after_cmpop_lhs(op, rhs)?,
            Instr::CmpOpAfterRhs { op, lhs } => self.after_cmpop_rhs(op, lhs)?,
            Instr::BoolBinOpAfterLhs { op, rhs } => self.after_boolbinop_lhs(op, rhs)?,
            Instr::BoolBinOpAfterRhs { op, lhs } => self.after_boolbinop_rhs(op, lhs)?,
            Instr::BoolNot => self.apply_bool_not()?,
            Instr::IfBranch {
                then_expr,
                else_expr,
            } => self.dispatch_if_branch(then_expr, else_expr)?,
            Instr::CondAfterPred {
                body,
                rest,
                default,
            } => self.dispatch_cond_after_pred(body, rest, default)?,
            Instr::LetBind { idx, body } => self.dispatch_let_bind(idx, body)?,
            Instr::LetUnbind { idx } => self.dispatch_let_unbind(idx)?,
            Instr::LoopIterEnd { body } => self.dispatch_loop_iter_end(body)?,
            Instr::ApplyBreak => self.dispatch_apply_break()?,
            Instr::WhileEvalCond { cond, body } => self.dispatch_while_eval_cond(cond, body)?,
            Instr::WhileAfterCond { cond, body } => self.dispatch_while_after_cond(cond, body)?,
            Instr::WhileAfterBody { cond, body } => self.dispatch_while_after_body(cond, body)?,
            Instr::ApplyDiscard => self.dispatch_apply_discard()?,
            Instr::ApplyList { count } => self.apply_list(count)?,
            Instr::ApplyListGet => self.apply_list_get()?,
            Instr::ApplyListLen => self.apply_list_len()?,
            Instr::ApplyListAppend => self.apply_list_append()?,
            Instr::ApplyListSlice => self.apply_list_slice()?,
            Instr::ApplyMapPut => self.apply_map_put()?,
            Instr::ApplyMapGet => self.apply_map_get()?,
            Instr::ApplyMapContains => self.apply_map_contains()?,
            Instr::ApplyStringFromInt => self.apply_string_from_int()?,
            Instr::ApplyStringConcat => self.apply_string_concat()?,
            Instr::ApplyStringEq => self.apply_string_eq()?,
            Instr::ApplyReadHandle => self.apply_read_handle(ctx)?,
            Instr::ApplyBytesLen => self.apply_bytes_len()?,
            Instr::ApplyBytesGet => self.apply_bytes_get()?,
            Instr::ApplyBytesSlice => self.apply_bytes_slice()?,
            Instr::ApplyBytesConcat => self.apply_bytes_concat()?,
            Instr::ApplyBytesEq => self.apply_bytes_eq()?,
            Instr::ApplyBytesFromInt => self.apply_bytes_from_int()?,
            Instr::ApplyWriteHostState { key } => self.apply_write_host_state(&key, ctx)?,
            Instr::ApplySome => self.apply_some()?,
            Instr::ApplyNone => self.apply_none()?,
            Instr::ApplyIsSome => self.apply_is_some()?,
            Instr::ApplyIsNone => self.apply_is_none()?,
            Instr::ApplyUnwrap => self.apply_unwrap()?,
            Instr::ApplyUnwrapOr => self.apply_unwrap_or()?,
            Instr::LoopBoundedIter { remaining, body } => {
                self.dispatch_loop_bounded_iter(remaining, body)?
            }
            Instr::LoopBoundedAfterBody { remaining, body } => {
                self.dispatch_loop_bounded_after_body(remaining, body)?
            }
            Instr::ForAfterSource { idx, body } => self.dispatch_for_after_source(idx, body)?,
            Instr::ForIter {
                idx,
                remaining,
                body,
            } => self.dispatch_for_iter(idx, remaining, body)?,
            Instr::ForAfterBody {
                idx,
                remaining,
                body,
            } => self.dispatch_for_after_body(idx, remaining, body)?,
            Instr::ApplyStructNew { name, fields } => self.apply_struct_new(name, fields)?,
            Instr::ApplyStructGet { field } => self.apply_struct_get(&field)?,
            Instr::ApplyStructSet { field } => self.apply_struct_set(&field)?,
            Instr::MatchDispatch { cases } => self.dispatch_match(cases, symbols)?,
            Instr::MatchUnbind { bindings } => self.dispatch_match_unbind(bindings)?,
        }

        // Pre-A.1 (F-C1): bound the explicit instruction stack so a
        // pathologically deep AST (`(add (add (add … 1 1 …)))`) does
        // not exhaust Ring-0 heap before completion. The original
        // recursive walker had a native-stack-overflow vector at
        // `MAX_EXPR_DEPTH=256`; the state machine moves that vector
        // to heap, and this guard puts a generous ceiling on the
        // heap footprint. Triggers a deterministic
        // `ExpressionDepthExceeded` — never a panic.
        if self.instr_stack.len() > MAX_INSTR_STACK_DEPTH {
            return Err(InterpretError::new(
                InterpretErrorKind::ExpressionDepthExceeded,
                "instruction stack depth exceeded MAX_INSTR_STACK_DEPTH",
            ));
        }

        Ok(StepOutcome::Continue)
    }

    /// Convenience: drive `step` in a loop until either an error
    /// surfaces or `Done(v)` is returned, then return `v`. This is
    /// the program-level entry point used by [`crate::interpret`]
    /// and [`crate::interpret_with_context`].
    pub fn run<C: PolicyContext + ?Sized>(
        &mut self,
        symbols: &SymbolTable,
        ctx: &mut C,
    ) -> Result<Value, InterpretError> {
        loop {
            match self.step(symbols, ctx)? {
                StepOutcome::Continue => continue,
                StepOutcome::Done(v) => return Ok(v),
            }
        }
    }

    // ── Instruction handlers ──────────────────────────────────

    fn dispatch_eval(&mut self, expr: SExpr, symbols: &SymbolTable) -> Result<(), InterpretError> {
        match expr {
            SExpr::Atom(atom) => self.eval_atom(atom),
            SExpr::List(items) => self.eval_list(items, symbols),
        }
    }

    fn eval_atom(&mut self, atom: Atom) -> Result<(), InterpretError> {
        let v = match atom {
            Atom::Integer(n) => Value::Integer(n),
            Atom::Handle(id) => Value::Handle(id),
            Atom::Bytes(b) => Value::Bytes(b),
            Atom::Parameter(n) => {
                let frame = self.call_stack.current().ok_or_else(|| {
                    InterpretError::new(
                        InterpretErrorKind::ParameterOutsideFunction,
                        "%n reference outside any function call frame",
                    )
                })?;
                frame.resolve_parameter(n)?
            }
            // Paket B.2: `true` / `false` are Bool literals when
            // they appear in a value position. Every other bare
            // symbol remains a non-evaluable token (head symbols
            // of forms, type-symbols, etc.).
            Atom::Symbol(ref s) if s == "true" => Value::Bool(true),
            Atom::Symbol(ref s) if s == "false" => Value::Bool(false),
            Atom::Symbol(_) => {
                return Err(InterpretError::new(
                    InterpretErrorKind::UnsupportedAtom,
                    "bare symbol atom not evaluable (only in fn/call positions)",
                ));
            }
        };
        self.value_stack.push(v);
        Ok(())
    }

    fn eval_list(
        &mut self,
        mut items: Vec<SExpr>,
        symbols: &SymbolTable,
    ) -> Result<(), InterpretError> {
        if items.is_empty() {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "cannot interpret empty list",
            ));
        }
        // Borrow the head to identify the dispatch symbol; we then
        // drain the tail by ownership so child SExprs can be moved
        // directly into Instr variants without cloning the entire
        // subtree.
        let head_sym = match &items[0] {
            SExpr::Atom(Atom::Symbol(s)) => s.clone(),
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::NonSymbolHead,
                    "list head must be a symbol",
                ));
            }
        };
        // Drop the head; remaining elements are the args.
        items.remove(0);
        let args = items;
        match head_sym.as_str() {
            "add" => self.queue_binop(BinOpKind::Add, args),
            "sub" => self.queue_binop(BinOpKind::Sub, args),
            "mul" => self.queue_binop(BinOpKind::Mul, args),
            // Pre-A.2 (F-C2): `(div a b)` with signature `[I64, I64] -> [I64]`.
            // Division by zero and the `i64::MIN / -1` overflow are
            // reported as `DivisionByZero` in the `BinOpAfterRhs`
            // continuation (see `after_binop_rhs`).
            "div" => self.queue_binop(BinOpKind::Div, args),
            // Phase 4 Step 1 — bitwise ops. Same dispatch shape as
            // `add/sub/mul/div`; shift-count bounds are enforced in
            // `after_binop_rhs`.
            "bit-and" => self.queue_binop(BinOpKind::BitAnd, args),
            "bit-or" => self.queue_binop(BinOpKind::BitOr, args),
            "bit-xor" => self.queue_binop(BinOpKind::BitXor, args),
            "bit-shl" => self.queue_binop(BinOpKind::Shl, args),
            "bit-shr" => self.queue_binop(BinOpKind::Shr, args),
            "eq" => self.queue_cmpop(CmpOpKind::Eq, args),
            "ne" => self.queue_cmpop(CmpOpKind::Ne, args),
            "lt" => self.queue_cmpop(CmpOpKind::Lt, args),
            "gt" => self.queue_cmpop(CmpOpKind::Gt, args),
            "le" => self.queue_cmpop(CmpOpKind::Le, args),
            "ge" => self.queue_cmpop(CmpOpKind::Ge, args),
            "and" => self.queue_boolbinop(BoolBinOpKind::And, args),
            "or" => self.queue_boolbinop(BoolBinOpKind::Or, args),
            "not" => self.queue_bool_not(args),
            "if" => self.queue_if(args),
            // Phase 4 Step 4 — N-way Bool dispatch.
            "cond" => self.queue_cond(args),
            "let" => self.queue_let(args),
            "loop" => self.queue_loop(args),
            "break" => self.queue_break(args),
            "while" => self.queue_while(args),
            "seq" => self.queue_seq(args),
            "discard" => self.queue_discard(args),
            "call" => self.queue_call(args, symbols),
            "register" => self.queue_register(args),
            "policy" => self.queue_policy(true, args),
            "query" => self.queue_policy(false, args),
            "fn" => self.queue_fn_passthrough(args),
            "program" => self.queue_program_passthrough(args),
            // Stage 12 Paket B — data-structure, loop, string, I/O ops.
            "list" => self.queue_list(args),
            "list-get" => self.queue_list_get(args),
            "list-len" => self.queue_list_len(args),
            "list-append" => self.queue_list_append(args),
            "list-slice" => self.queue_list_slice(args),
            "map-new" => self.queue_map_new(args),
            "map-put" => self.queue_map_put(args),
            "map-get" => self.queue_map_get(args),
            "map-contains" => self.queue_map_contains(args),
            "string-from-int" => self.queue_string_from_int(args),
            "string-concat" => self.queue_string_concat(args),
            "string-eq" => self.queue_string_eq(args),
            "read-handle" => self.queue_read_handle(args),
            "write-host-state" => self.queue_write_host_state(args),
            // Phase 4 Step 2 — Bytes operations.
            "bytes-len" => self.queue_bytes_len(args),
            "bytes-get" => self.queue_bytes_get(args),
            "bytes-slice" => self.queue_bytes_slice(args),
            "bytes-concat" => self.queue_bytes_concat(args),
            "bytes-eq" => self.queue_bytes_eq(args),
            "bytes-from-int" => self.queue_bytes_from_int(args),
            // Phase 4 Step 5 — Maybe / Option type.
            "some" => self.queue_some(args),
            "none" => self.queue_none(args),
            "is-some" => self.queue_is_some(args),
            "is-none" => self.queue_is_none(args),
            "unwrap" => self.queue_unwrap(args),
            "unwrap-or" => self.queue_unwrap_or(args),
            "loop-with-bound" => self.queue_loop_with_bound(args),
            "for" => self.queue_for(args),
            // Phase 4 Step 6 — nominal structs.
            "struct-new" => self.queue_struct_new(args, symbols),
            "struct-get" => self.queue_struct_get(args),
            "struct-set" => self.queue_struct_set(args),
            // Phase 4 Step 7 — pattern matching.
            "match" => self.queue_match(args),
            _ => Err(InterpretError::new(
                InterpretErrorKind::UnsupportedInstruction,
                "instruction not yet supported in Stage 9 minimal interpreter",
            )),
        }
    }

    fn queue_binop(&mut self, op: BinOpKind, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "binary op requires exactly 2 arguments",
            ));
        }
        // Take owned subtrees out of `args` so they can be moved into
        // the instr stack without cloning.
        let rhs = args.remove(1);
        let lhs = args.remove(0);
        // Execution order: Eval(lhs) → BinOpAfterLhs (type-check, schedule rhs)
        //                  → Eval(rhs) → BinOpAfterRhs (type-check, apply).
        // Stack push order is reverse: deepest first.
        self.instr_stack.push(Instr::BinOpAfterLhs {
            op,
            rhs: Box::new(rhs),
        });
        self.instr_stack.push(Instr::Eval(Box::new(lhs)));
        Ok(())
    }

    fn after_binop_lhs(&mut self, op: BinOpKind, rhs: Box<SExpr>) -> Result<(), InterpretError> {
        let lhs_val = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "binop missing lhs value")
        })?;
        let lhs = expect_integer(lhs_val, "binary op requires integer arguments")?;
        self.instr_stack.push(Instr::BinOpAfterRhs { op, lhs });
        self.instr_stack.push(Instr::Eval(rhs));
        Ok(())
    }

    fn after_binop_rhs(&mut self, op: BinOpKind, lhs: i64) -> Result<(), InterpretError> {
        let rhs_val = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "binop missing rhs value")
        })?;
        let rhs = expect_integer(rhs_val, "binary op requires integer arguments")?;
        let r = match op {
            BinOpKind::Add => lhs.wrapping_add(rhs),
            BinOpKind::Sub => lhs.wrapping_sub(rhs),
            BinOpKind::Mul => lhs.wrapping_mul(rhs),
            // Pre-A.2 (F-C2): `i64::checked_div` returns `None` for
            // `b == 0` and for the `i64::MIN / -1`-overflow; both
            // surface deterministically as `DivisionByZero` — never
            // a panic.
            BinOpKind::Div => lhs.checked_div(rhs).ok_or_else(|| {
                InterpretError::new(
                    InterpretErrorKind::DivisionByZero,
                    "div: division by zero or i64::MIN / -1 overflow",
                )
            })?,
            // Phase 4 Step 1 — bitwise ops on i64. AND/OR/XOR are
            // total. The shifts perform a manual `0..64` range check
            // on `rhs`, then apply `wrapping_shl` / `wrapping_shr`.
            // A bare `wrapping_shl(n as u32)` would silently mask
            // the count to its low 6 bits — the same x86-aligned
            // behaviour the AOT codegen accepts. The interpreter
            // instead surfaces any out-of-range count as a typed
            // `ShiftCountOutOfRange` so policy code can reason about
            // the error deterministically. `wrapping_shr` on `i64`
            // is an arithmetic shift (SAR), matching the AOT path.
            BinOpKind::BitAnd => lhs & rhs,
            BinOpKind::BitOr => lhs | rhs,
            BinOpKind::BitXor => lhs ^ rhs,
            BinOpKind::Shl => {
                if !(0..64).contains(&rhs) {
                    return Err(InterpretError::new(
                        InterpretErrorKind::ShiftCountOutOfRange { count: rhs },
                        "bit-shl: shift count must be in 0..64",
                    ));
                }
                lhs.wrapping_shl(rhs as u32)
            }
            BinOpKind::Shr => {
                if !(0..64).contains(&rhs) {
                    return Err(InterpretError::new(
                        InterpretErrorKind::ShiftCountOutOfRange { count: rhs },
                        "bit-shr: shift count must be in 0..64",
                    ));
                }
                // `wrapping_shr` on `i64` performs an arithmetic
                // shift (SAR) — sign bit is replicated, matching
                // Rust's `i64 >>` and the AOT codegen's `sar`.
                lhs.wrapping_shr(rhs as u32)
            }
        };
        self.value_stack.push(Value::Integer(r));
        Ok(())
    }

    fn queue_call(
        &mut self,
        mut args: Vec<SExpr>,
        symbols: &SymbolTable,
    ) -> Result<(), InterpretError> {
        if args.is_empty() {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "call requires function name",
            ));
        }
        let fn_name = match &args[0] {
            SExpr::Atom(Atom::Symbol(s)) => s.clone(),
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::NonSymbolHead,
                    "call target must be a symbol",
                ));
            }
        };
        let def = symbols.lookup(&fn_name).ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::FunctionNotFound,
                "call target function not defined",
            )
        })?;
        // Strip the function-name head; `args` now holds only the
        // argument expressions.
        args.remove(0);
        if args.len() != def.arity {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "call argument count does not match function arity",
            ));
        }
        let argc = args.len();
        let body = Arc::clone(&def.body);
        // Execution order: Eval(arg_0), Eval(arg_1), ..., Eval(arg_{N-1}),
        //                  then InvokeCall (which pops N and runs body).
        self.instr_stack.push(Instr::InvokeCall { body, argc });
        // Push args in reverse so the deepest (arg_0) is popped first.
        for arg in args.into_iter().rev() {
            self.instr_stack.push(Instr::Eval(Box::new(arg)));
        }
        Ok(())
    }

    fn invoke_call(&mut self, body: Arc<SExpr>, argc: usize) -> Result<(), InterpretError> {
        if self.value_stack.len() < argc {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "invoke_call: value stack underflow",
            ));
        }
        // Args are on the value stack in evaluation order: arg_0 was
        // pushed first (deepest), arg_{N-1} last (topmost). Pop in
        // reverse, then reverse the buffer to restore source order.
        //
        // Pre-A.2 (F-C4): the `len() < argc` check above guarantees
        // each `pop` returns `Some`; we still surface the failure as
        // a typed error instead of `.unwrap()` so a future refactor
        // that breaks the invariant cannot panic in Ring-0.
        let mut resolved = Vec::with_capacity(argc);
        for _ in 0..argc {
            let v = self.value_stack.pop().ok_or_else(|| {
                InterpretError::new(
                    InterpretErrorKind::EmptyList,
                    "invoke_call: value stack underflow during pop",
                )
            })?;
            resolved.push(v);
        }
        resolved.reverse();
        let frame = StackFrame::new(resolved);
        self.call_stack.push(frame)?;
        // Run body, then pop the frame. Stack push order is reverse
        // of execution: PopFrame first, then Eval(body).
        //
        // `body` is an `Arc<SExpr>`. We try to unwrap if there is no
        // other live Arc-reference (fast path: the symbol table is
        // the only other holder and we cloned at the call site);
        // otherwise we fall back to a clone of the inner SExpr.
        let body_owned: SExpr = match Arc::try_unwrap(body) {
            Ok(inner) => inner,
            Err(arc) => (*arc).clone(),
        };
        self.instr_stack.push(Instr::PopFrame);
        self.instr_stack.push(Instr::Eval(Box::new(body_owned)));
        Ok(())
    }

    fn queue_register(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "register requires exactly 1 argument (bytes)",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyRegister);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_register(&mut self) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "apply_register: value stack empty",
            )
        })?;
        match v {
            Value::Bytes(_) => {}
            Value::Integer(_) => {
                return Err(InterpretError::new(
                    InterpretErrorKind::UnsupportedAtom,
                    "register expects bytes argument, got integer",
                ));
            }
            Value::Handle(_) => {
                return Err(InterpretError::new(
                    InterpretErrorKind::UnsupportedAtom,
                    "register expects bytes argument, got handle",
                ));
            }
            Value::Bool(_) => {
                return Err(InterpretError::new(
                    InterpretErrorKind::UnsupportedAtom,
                    "register expects bytes argument, got bool",
                ));
            }
            Value::List(_) | Value::Map(_) | Value::String(_) => {
                return Err(InterpretError::new(
                    InterpretErrorKind::UnsupportedAtom,
                    "register expects bytes argument, got List/Map/String",
                ));
            }
            Value::Maybe(_) => {
                return Err(InterpretError::new(
                    InterpretErrorKind::UnsupportedAtom,
                    "register expects bytes argument, got Maybe",
                ));
            }
            Value::Struct { .. } => {
                return Err(InterpretError::new(
                    InterpretErrorKind::UnsupportedAtom,
                    "register expects bytes argument, got Struct",
                ));
            }
        };
        // Pre-A.2 (F-H14): fixed handle id — see
        // `MOCK_REGISTER_HANDLE_ID` for rationale. The real per-sandbox
        // [`HandleTable`] allocation comes via Paket A.
        self.value_stack
            .push(Value::Handle(MOCK_REGISTER_HANDLE_ID));
        Ok(())
    }

    fn queue_policy(
        &mut self,
        is_policy: bool,
        mut args: Vec<SExpr>,
    ) -> Result<(), InterpretError> {
        if args.len() < 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::PolicyMalformed,
                "policy/query requires at least <subsystem> <operation>",
            ));
        }
        // Subsystem + operation symbols come out as owned Strings.
        let subsystem = match &args[0] {
            SExpr::Atom(Atom::Symbol(s)) => s.to_string(),
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::PolicyMalformed,
                    "policy/query subsystem must be a symbol literal",
                ));
            }
        };
        let operation = match &args[1] {
            SExpr::Atom(Atom::Symbol(s)) => s.to_string(),
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::PolicyMalformed,
                    "policy/query operation must be a symbol literal",
                ));
            }
        };
        // Drop the head symbols; remaining items are the value args.
        args.remove(0);
        args.remove(0);
        let argc = args.len();
        self.instr_stack.push(Instr::ApplyPolicy {
            is_policy,
            subsystem,
            operation,
            argc,
        });
        for arg in args.into_iter().rev() {
            self.instr_stack.push(Instr::Eval(Box::new(arg)));
        }
        Ok(())
    }

    fn apply_policy<C: PolicyContext + ?Sized>(
        &mut self,
        is_policy: bool,
        subsystem: &str,
        operation: &str,
        argc: usize,
        ctx: &mut C,
    ) -> Result<(), InterpretError> {
        if self.value_stack.len() < argc {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "apply_policy: value stack underflow",
            ));
        }
        // Pre-A.2 (F-C4): see `invoke_call` — typed error instead of
        // `.unwrap()` for the post-len-check pops.
        let mut evaluated = Vec::with_capacity(argc);
        for _ in 0..argc {
            let v = self.value_stack.pop().ok_or_else(|| {
                InterpretError::new(
                    InterpretErrorKind::EmptyList,
                    "apply_policy: value stack underflow during pop",
                )
            })?;
            evaluated.push(v);
        }
        evaluated.reverse();
        let result = if is_policy {
            ctx.policy(subsystem, operation, &evaluated)
        } else {
            ctx.query(subsystem, operation, &evaluated)
        };
        match result {
            Ok(status) => {
                self.value_stack.push(Value::Integer(status));
                Ok(())
            }
            Err(err) => {
                let kind = match err {
                    PolicyError::NotSupported => InterpretErrorKind::PolicyNotSupported,
                    _ => InterpretErrorKind::PolicyDispatchFailed,
                };
                let message = match err {
                    PolicyError::NotSupported => "policy/query not supported by this context",
                    PolicyError::UnknownSubsystem => "policy/query: unknown subsystem",
                    PolicyError::UnknownOperation => {
                        "policy/query: unknown operation for subsystem"
                    }
                    PolicyError::InvalidArgument => "policy/query: invalid argument",
                    PolicyError::PermissionDenied => {
                        "policy/query: capability missing or wrong kind"
                    }
                    PolicyError::OperationFailed => "policy/query: underlying operation failed",
                };
                Err(InterpretError::new(kind, message))
            }
        }
    }

    fn queue_fn_passthrough(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        // `(fn name (params) ret body)` encountered as a bare
        // expression (not at program-level) — the original
        // tree-walker forwarded transparently to the body for
        // backward compatibility with stand-alone test cases. We
        // preserve that behaviour here.
        if let Some(last) = args.pop() {
            self.instr_stack.push(Instr::Eval(Box::new(last)));
            Ok(())
        } else {
            Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "fn requires body",
            ))
        }
    }

    fn queue_program_passthrough(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if let Some(last) = args.pop() {
            self.instr_stack.push(Instr::Eval(Box::new(last)));
            Ok(())
        } else {
            Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "program requires at least one expression",
            ))
        }
    }

    // ── Paket B dispatch helpers ────────────────────────────

    /// Paket B.1: queue a comparison op `(eq|ne|lt|gt|le|ge a b)`.
    fn queue_cmpop(&mut self, op: CmpOpKind, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "comparison op requires exactly 2 arguments",
            ));
        }
        let rhs = args.remove(1);
        let lhs = args.remove(0);
        self.instr_stack.push(Instr::CmpOpAfterLhs {
            op,
            rhs: Box::new(rhs),
        });
        self.instr_stack.push(Instr::Eval(Box::new(lhs)));
        Ok(())
    }

    fn after_cmpop_lhs(&mut self, op: CmpOpKind, rhs: Box<SExpr>) -> Result<(), InterpretError> {
        let lhs_val = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "cmpop missing lhs value")
        })?;
        let lhs = expect_integer(lhs_val, "comparison op requires integer arguments")?;
        self.instr_stack.push(Instr::CmpOpAfterRhs { op, lhs });
        self.instr_stack.push(Instr::Eval(rhs));
        Ok(())
    }

    fn after_cmpop_rhs(&mut self, op: CmpOpKind, lhs: i64) -> Result<(), InterpretError> {
        let rhs_val = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "cmpop missing rhs value")
        })?;
        let rhs = expect_integer(rhs_val, "comparison op requires integer arguments")?;
        let r = match op {
            CmpOpKind::Eq => lhs == rhs,
            CmpOpKind::Ne => lhs != rhs,
            CmpOpKind::Lt => lhs < rhs,
            CmpOpKind::Gt => lhs > rhs,
            CmpOpKind::Le => lhs <= rhs,
            CmpOpKind::Ge => lhs >= rhs,
        };
        self.value_stack.push(Value::Bool(r));
        Ok(())
    }

    /// Paket B.2: queue a binary Bool op `(and|or a b)`. Both
    /// operands are evaluated (no short-circuit) to match the
    /// validator's eager type-checking semantics.
    fn queue_boolbinop(
        &mut self,
        op: BoolBinOpKind,
        mut args: Vec<SExpr>,
    ) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "and/or requires exactly 2 arguments",
            ));
        }
        let rhs = args.remove(1);
        let lhs = args.remove(0);
        self.instr_stack.push(Instr::BoolBinOpAfterLhs {
            op,
            rhs: Box::new(rhs),
        });
        self.instr_stack.push(Instr::Eval(Box::new(lhs)));
        Ok(())
    }

    fn after_boolbinop_lhs(
        &mut self,
        op: BoolBinOpKind,
        rhs: Box<SExpr>,
    ) -> Result<(), InterpretError> {
        let lhs_val = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "and/or missing lhs value")
        })?;
        let lhs = expect_bool(lhs_val, "and/or requires Bool arguments")?;
        self.instr_stack.push(Instr::BoolBinOpAfterRhs { op, lhs });
        self.instr_stack.push(Instr::Eval(rhs));
        Ok(())
    }

    fn after_boolbinop_rhs(&mut self, op: BoolBinOpKind, lhs: bool) -> Result<(), InterpretError> {
        let rhs_val = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "and/or missing rhs value")
        })?;
        let rhs = expect_bool(rhs_val, "and/or requires Bool arguments")?;
        let r = match op {
            BoolBinOpKind::And => lhs && rhs,
            BoolBinOpKind::Or => lhs || rhs,
        };
        self.value_stack.push(Value::Bool(r));
        Ok(())
    }

    /// Paket B.2: queue a unary `not` op.
    fn queue_bool_not(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "not requires exactly 1 argument",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::BoolNot);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_bool_not(&mut self) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "not missing operand")
        })?;
        let b = expect_bool(v, "not requires a Bool argument")?;
        self.value_stack.push(Value::Bool(!b));
        Ok(())
    }

    /// Paket B.1b: queue an `(if cond then else)` form.
    fn queue_if(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 3 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "if requires (if cond then else)",
            ));
        }
        let else_expr = args.remove(2);
        let then_expr = args.remove(1);
        let cond = args.remove(0);
        self.instr_stack.push(Instr::IfBranch {
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        });
        self.instr_stack.push(Instr::Eval(Box::new(cond)));
        Ok(())
    }

    fn dispatch_if_branch(
        &mut self,
        then_expr: Box<SExpr>,
        else_expr: Box<SExpr>,
    ) -> Result<(), InterpretError> {
        let cond_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "if missing cond value")
        })?;
        let cond = expect_bool(cond_v, "if condition must be Bool")?;
        let branch = if cond { then_expr } else { else_expr };
        self.instr_stack.push(Instr::Eval(branch));
        Ok(())
    }

    /// Phase 4 Step 4 — schedule a `(cond (p1 b1) … (default body))`
    /// form for evaluation. Each clause is a 2-element list whose
    /// head is either a predicate expression (non-last clauses) or
    /// the literal `default` keyword (last clause). The planner
    /// captures the default body once, peels the first predicate-
    /// body pair for immediate evaluation, and stores the remaining
    /// pairs in REVERSE source order so that `Vec::pop()` produces
    /// the next clause in O(1) inside [`dispatch_cond_after_pred`].
    ///
    /// Structural shape (clause count, default placement, clause
    /// arity) is enforced statically by the validator's
    /// `validate_cond`, but the interpreter re-checks defensively so
    /// the dispatch stays panic-free even when fed an unvalidated
    /// AST. Failures surface as typed `InterpretError`s — never
    /// panics — to preserve Ring-0 invariants.
    fn queue_cond(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.is_empty() {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "cond requires at least one (default body) clause",
            ));
        }

        // Extract the mandatory default clause (last clause). Its
        // head must be the symbol `default`; its second item is the
        // body to evaluate when no earlier predicate matches.
        let default_clause = args.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "cond has no clauses")
        })?;
        let mut default_items = match default_clause {
            SExpr::List(items) => items,
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::NonSymbolHead,
                    "cond last clause must be a (default body) list",
                ));
            }
        };
        if default_items.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "cond default clause must be (default body)",
            ));
        }
        let default_body = default_items.remove(1);
        let default_head = default_items.remove(0);
        match &default_head {
            SExpr::Atom(Atom::Symbol(s)) if s == "default" => {}
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::NonSymbolHead,
                    "cond last clause head must be the symbol `default`",
                ));
            }
        }

        // No non-default clauses → just evaluate the default body
        // unconditionally. (Equivalent to `(seq default-body)`.)
        if args.is_empty() {
            self.instr_stack.push(Instr::Eval(Box::new(default_body)));
            return Ok(());
        }

        // Decompose each non-default clause into a (predicate, body)
        // tuple. Reject malformed clauses up-front so the
        // CondAfterPred continuation never has to re-validate shape.
        let mut clauses: Vec<(SExpr, SExpr)> = Vec::with_capacity(args.len());
        for clause in args.into_iter() {
            let mut clause_items = match clause {
                SExpr::List(items) => items,
                _ => {
                    return Err(InterpretError::new(
                        InterpretErrorKind::NonSymbolHead,
                        "cond clause must be a (predicate body) list",
                    ));
                }
            };
            if clause_items.len() != 2 {
                return Err(InterpretError::new(
                    InterpretErrorKind::ArityMismatch,
                    "cond clause must be (predicate body)",
                ));
            }
            // Reject `default` as a non-last predicate head — keeps
            // the runtime semantics aligned with the validator and
            // type-checker, which both treat `default` as the
            // exclusive last-clause keyword. The last clause has
            // already been peeled off above; everything reaching
            // here MUST NOT carry the `default` head.
            if let SExpr::Atom(Atom::Symbol(s)) = &clause_items[0] {
                if s == "default" {
                    return Err(InterpretError::new(
                        InterpretErrorKind::NonSymbolHead,
                        "cond: `default` may appear only in the last clause",
                    ));
                }
            }
            let body = clause_items.remove(1);
            let pred = clause_items.remove(0);
            clauses.push((pred, body));
        }

        // Pull the first non-default clause for immediate evaluation
        // and reverse the remainder so `Vec::pop()` yields the next
        // clause in source order. Reversing here is O(n) once; each
        // subsequent peel is O(1).
        let (first_pred, first_body) = clauses.remove(0);
        clauses.reverse();

        self.instr_stack.push(Instr::CondAfterPred {
            body: Box::new(first_body),
            rest: clauses,
            default: Box::new(default_body),
        });
        self.instr_stack.push(Instr::Eval(Box::new(first_pred)));
        Ok(())
    }

    /// Phase 4 Step 4 — handle a `CondAfterPred` continuation. Pops
    /// the Bool the predicate produced and either schedules `body`
    /// (true), peels the next clause from `rest` (false with more to
    /// try), or schedules `default` (false and rest drained). The
    /// `rest` queue is consumed via `Vec::pop()` because clauses
    /// were pushed in reverse source order by [`queue_cond`].
    fn dispatch_cond_after_pred(
        &mut self,
        body: Box<SExpr>,
        mut rest: Vec<(SExpr, SExpr)>,
        default: Box<SExpr>,
    ) -> Result<(), InterpretError> {
        let pred_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "cond clause predicate produced no value",
            )
        })?;
        let pred = expect_bool(pred_v, "cond clause predicate must be Bool")?;
        if pred {
            self.instr_stack.push(Instr::Eval(body));
        } else if let Some((next_pred, next_body)) = rest.pop() {
            self.instr_stack.push(Instr::CondAfterPred {
                body: Box::new(next_body),
                rest,
                default,
            });
            self.instr_stack.push(Instr::Eval(Box::new(next_pred)));
        } else {
            self.instr_stack.push(Instr::Eval(default));
        }
        Ok(())
    }

    /// Paket B.3: queue a `(let %n value body)` form.
    fn queue_let(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 3 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "let requires (let %n value body)",
            ));
        }
        let idx = match &args[0] {
            SExpr::Atom(Atom::Parameter(n)) => *n,
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::NonSymbolHead,
                    "let binding target must be a parameter atom (%n)",
                ));
            }
        };
        let body = args.remove(2);
        let value_expr = args.remove(1);
        // Sequence: Eval(value) → LetBind(idx, body) → Eval(body) → LetUnbind(idx)
        self.instr_stack.push(Instr::LetBind {
            idx,
            body: Box::new(body),
        });
        self.instr_stack.push(Instr::Eval(Box::new(value_expr)));
        Ok(())
    }

    fn dispatch_let_bind(&mut self, idx: u32, body: Box<SExpr>) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "let value expression produced no value",
            )
        })?;
        let frame = self.call_stack.current_mut().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::ParameterOutsideFunction,
                "let outside any function body (no active frame)",
            )
        })?;
        frame.bind_local(idx, v)?;
        // Sequence: schedule body eval, then unbind.
        self.instr_stack.push(Instr::LetUnbind { idx });
        self.instr_stack.push(Instr::Eval(body));
        Ok(())
    }

    fn dispatch_let_unbind(&mut self, idx: u32) -> Result<(), InterpretError> {
        if let Some(frame) = self.call_stack.current_mut() {
            frame.unbind_local(idx);
        }
        Ok(())
    }

    /// Paket B.1c: queue `(loop body)`. Pushes a LoopFrame snapshot
    /// and schedules the body for evaluation with a LoopIterEnd
    /// continuation that re-iterates on fall-through.
    fn queue_loop(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "loop requires exactly 1 argument (body)",
            ));
        }
        let body = Arc::new(args.remove(0));
        // Push the loop frame BEFORE pushing the iter continuations
        // so that ApplyBreak truncates to the correct depths.
        self.loop_stack.push(LoopFrame {
            instr_stack_depth_at_entry: self.instr_stack.len(),
            value_stack_depth_at_entry: self.value_stack.len(),
        });
        self.instr_stack.push(Instr::LoopIterEnd {
            body: Arc::clone(&body),
        });
        let body_owned: SExpr = match Arc::try_unwrap(body) {
            Ok(inner) => inner,
            Err(arc) => (*arc).clone(),
        };
        self.instr_stack.push(Instr::Eval(Box::new(body_owned)));
        Ok(())
    }

    fn dispatch_loop_iter_end(&mut self, body: Arc<SExpr>) -> Result<(), InterpretError> {
        // Body completed by fall-through. Discard its residual
        // value (the validator guarantees the body pushed at most
        // one value; if it pushed zero, the validator already
        // rejected the program).
        if self.value_stack.pop().is_none() {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "loop body produced no residual value to discard",
            ));
        }
        // Re-schedule next iteration. The LoopFrame stays on the
        // loop_stack throughout.
        self.instr_stack.push(Instr::LoopIterEnd {
            body: Arc::clone(&body),
        });
        let body_owned: SExpr = match Arc::try_unwrap(body) {
            Ok(inner) => inner,
            Err(arc) => (*arc).clone(),
        };
        self.instr_stack.push(Instr::Eval(Box::new(body_owned)));
        Ok(())
    }

    /// Paket B.1c: queue `(break v)`. Evaluates `v`, then ApplyBreak
    /// unwinds to the innermost LoopFrame and pushes the captured
    /// value as the loop's residual.
    fn queue_break(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "break requires exactly 1 argument (the break value)",
            ));
        }
        let value_expr = args.remove(0);
        self.instr_stack.push(Instr::ApplyBreak);
        self.instr_stack.push(Instr::Eval(Box::new(value_expr)));
        Ok(())
    }

    fn dispatch_apply_break(&mut self) -> Result<(), InterpretError> {
        // The break value is on top of the value stack.
        let break_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "break value expression produced no value",
            )
        })?;
        let frame = self.loop_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::BreakOutsideLoop,
                "break has no enclosing loop frame",
            )
        })?;
        // Unwind both stacks to the loop's entry snapshot.
        self.instr_stack.truncate(frame.instr_stack_depth_at_entry);
        self.value_stack.truncate(frame.value_stack_depth_at_entry);
        // Push the broken value as the loop expression's residual.
        self.value_stack.push(break_v);
        Ok(())
    }

    /// Paket B.1d: queue `(while cond body)`. Pushes a LoopFrame and
    /// schedules a WhileEvalCond continuation. The implicit `(break
    /// false)` is realised by `WhileAfterCond` when cond evaluates
    /// to false.
    fn queue_while(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "while requires (while cond body)",
            ));
        }
        let body = Arc::new(args.remove(1));
        let cond = Arc::new(args.remove(0));
        self.loop_stack.push(LoopFrame {
            instr_stack_depth_at_entry: self.instr_stack.len(),
            value_stack_depth_at_entry: self.value_stack.len(),
        });
        self.instr_stack.push(Instr::WhileEvalCond { cond, body });
        Ok(())
    }

    fn dispatch_while_eval_cond(
        &mut self,
        cond: Arc<SExpr>,
        body: Arc<SExpr>,
    ) -> Result<(), InterpretError> {
        self.instr_stack.push(Instr::WhileAfterCond {
            cond: Arc::clone(&cond),
            body,
        });
        let cond_owned: SExpr = match Arc::try_unwrap(cond) {
            Ok(inner) => inner,
            Err(arc) => (*arc).clone(),
        };
        self.instr_stack.push(Instr::Eval(Box::new(cond_owned)));
        Ok(())
    }

    fn dispatch_while_after_cond(
        &mut self,
        cond: Arc<SExpr>,
        body: Arc<SExpr>,
    ) -> Result<(), InterpretError> {
        let cond_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "while cond expression produced no value",
            )
        })?;
        let cond_b = expect_bool(cond_v, "while condition must be Bool")?;
        if cond_b {
            // Evaluate body, then iterate again.
            self.instr_stack.push(Instr::WhileAfterBody {
                cond,
                body: Arc::clone(&body),
            });
            let body_owned: SExpr = match Arc::try_unwrap(body) {
                Ok(inner) => inner,
                Err(arc) => (*arc).clone(),
            };
            self.instr_stack.push(Instr::Eval(Box::new(body_owned)));
        } else {
            // Implicit (break false): unwind to loop frame and
            // push Bool(false) as the while expression's residual.
            let frame = self.loop_stack.pop().ok_or_else(|| {
                InterpretError::new(
                    InterpretErrorKind::BreakOutsideLoop,
                    "while: no enclosing loop frame at implicit break",
                )
            })?;
            self.instr_stack.truncate(frame.instr_stack_depth_at_entry);
            self.value_stack.truncate(frame.value_stack_depth_at_entry);
            self.value_stack.push(Value::Bool(false));
        }
        Ok(())
    }

    fn dispatch_while_after_body(
        &mut self,
        cond: Arc<SExpr>,
        body: Arc<SExpr>,
    ) -> Result<(), InterpretError> {
        // Body completed by fall-through; discard residual.
        if self.value_stack.pop().is_none() {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "while body produced no residual value to discard",
            ));
        }
        // Next iteration.
        self.instr_stack.push(Instr::WhileEvalCond { cond, body });
        Ok(())
    }

    /// Paket B: queue `(seq e1 e2 ... eN)`. The validator
    /// guarantees effects e1..eN-1 are stack-neutral (typically
    /// wrapped in `(discard …)`), so seq just sequences their
    /// evaluation — no implicit discard is needed inside seq.
    /// The last arg eN produces the seq expression's value.
    fn queue_seq(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() < 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "seq requires at least 2 arguments",
            ));
        }
        // Push in reverse so eval order is left-to-right.
        let last = args.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "seq with no last arg")
        })?;
        self.instr_stack.push(Instr::Eval(Box::new(last)));
        while let Some(effect) = args.pop() {
            self.instr_stack.push(Instr::Eval(Box::new(effect)));
        }
        Ok(())
    }

    fn dispatch_apply_discard(&mut self) -> Result<(), InterpretError> {
        if self.value_stack.pop().is_none() {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "seq/discard expected a residual value to drop",
            ));
        }
        Ok(())
    }

    /// Paket B: queue `(discard expr)` — evaluate expr, drop result.
    fn queue_discard(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "discard requires exactly 1 argument",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyDiscard);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    // ── Paket B.2 — Lists ────────────────────────────────────────

    fn queue_list(&mut self, args: Vec<SExpr>) -> Result<(), InterpretError> {
        let count = args.len();
        // Push ApplyList first so it executes after all element evals.
        self.instr_stack.push(Instr::ApplyList { count });
        // Push elements in reverse so leftmost (arg[0]) ends up
        // pushed last (i.e. evaluated first deepest-first).
        // Eval order: arg[0], arg[1], …, arg[N-1]; the stack of
        // residual i64s is then popped by ApplyList in reverse.
        for arg in args.into_iter().rev() {
            self.instr_stack.push(Instr::Eval(Box::new(arg)));
        }
        Ok(())
    }

    fn apply_list(&mut self, count: usize) -> Result<(), InterpretError> {
        if self.value_stack.len() < count {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "apply_list: value stack underflow",
            ));
        }
        let mut elems = Vec::with_capacity(count);
        for _ in 0..count {
            let v = self.value_stack.pop().ok_or_else(|| {
                InterpretError::new(
                    InterpretErrorKind::EmptyList,
                    "apply_list: value stack underflow during pop",
                )
            })?;
            let n = expect_integer(v, "list element must be i64")?;
            elems.push(n);
        }
        elems.reverse();
        self.value_stack.push(Value::List(elems));
        Ok(())
    }

    fn queue_list_get(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "list-get requires exactly 2 arguments",
            ));
        }
        let idx_expr = args.remove(1);
        let list_expr = args.remove(0);
        // Execution order: Eval(list), Eval(idx), ApplyListGet.
        self.instr_stack.push(Instr::ApplyListGet);
        self.instr_stack.push(Instr::Eval(Box::new(idx_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(list_expr)));
        Ok(())
    }

    fn apply_list_get(&mut self) -> Result<(), InterpretError> {
        let idx_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "list-get missing index")
        })?;
        let list_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "list-get missing list")
        })?;
        let idx = expect_integer(idx_v, "list-get index must be i64")?;
        let list = expect_list(list_v, "list-get target must be a list")?;
        if idx < 0 || (idx as usize) >= list.len() {
            return Err(InterpretError::new(
                InterpretErrorKind::ListIndexOutOfBounds,
                "list-get: index out of range",
            ));
        }
        self.value_stack.push(Value::Integer(list[idx as usize]));
        Ok(())
    }

    fn queue_list_len(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "list-len requires exactly 1 argument",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyListLen);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_list_len(&mut self) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "list-len missing operand")
        })?;
        let list = expect_list(v, "list-len target must be a list")?;
        self.value_stack.push(Value::Integer(list.len() as i64));
        Ok(())
    }

    fn queue_list_append(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "list-append requires exactly 2 arguments",
            ));
        }
        let elem_expr = args.remove(1);
        let list_expr = args.remove(0);
        self.instr_stack.push(Instr::ApplyListAppend);
        self.instr_stack.push(Instr::Eval(Box::new(elem_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(list_expr)));
        Ok(())
    }

    fn apply_list_append(&mut self) -> Result<(), InterpretError> {
        let elem_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "list-append missing element")
        })?;
        let list_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "list-append missing list")
        })?;
        let elem = expect_integer(elem_v, "list-append element must be i64")?;
        let mut list = expect_list(list_v, "list-append target must be a list")?;
        list.push(elem);
        self.value_stack.push(Value::List(list));
        Ok(())
    }

    // ── Phase 4 Step 3 — list-slice ──────────────────────────────

    fn queue_list_slice(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 3 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "list-slice requires exactly 3 arguments",
            ));
        }
        let len_expr = args.remove(2);
        let start_expr = args.remove(1);
        let list_expr = args.remove(0);
        // Execution order: Eval(list), Eval(start), Eval(len), ApplyListSlice.
        self.instr_stack.push(Instr::ApplyListSlice);
        self.instr_stack.push(Instr::Eval(Box::new(len_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(start_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(list_expr)));
        Ok(())
    }

    fn apply_list_slice(&mut self) -> Result<(), InterpretError> {
        let len_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "list-slice missing len")
        })?;
        let start_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "list-slice missing start")
        })?;
        let list_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "list-slice missing list")
        })?;
        let len = expect_integer(len_v, "list-slice len must be i64")?;
        let start = expect_integer(start_v, "list-slice start must be i64")?;
        let list = expect_list(list_v, "list-slice target must be a list")?;
        // Range check uses i128 to avoid overflow when adding two
        // i64s near the type's extremes.
        let list_len_i128 = list.len() as i128;
        let end_i128 = (start as i128).saturating_add(len as i128);
        if start < 0 || len < 0 || (start as i128) > list_len_i128 || end_i128 > list_len_i128 {
            return Err(InterpretError::new(
                InterpretErrorKind::ListSliceOutOfBounds {
                    start,
                    len,
                    list_length: list.len(),
                },
                "list-slice: range out of bounds",
            ));
        }
        let s = start as usize;
        let n = len as usize;
        self.value_stack.push(Value::List(list[s..s + n].to_vec()));
        Ok(())
    }

    // ── Phase 4 Step 2 — Bytes ───────────────────────────────────

    fn queue_bytes_len(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "bytes-len requires exactly 1 argument",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyBytesLen);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_bytes_len(&mut self) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "bytes-len missing operand")
        })?;
        let bytes = expect_bytes(v, "bytes-len target must be bytes")?;
        self.value_stack.push(Value::Integer(bytes.len() as i64));
        Ok(())
    }

    fn queue_bytes_get(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "bytes-get requires exactly 2 arguments",
            ));
        }
        let idx_expr = args.remove(1);
        let bytes_expr = args.remove(0);
        // Execution order: Eval(bytes), Eval(idx), ApplyBytesGet.
        self.instr_stack.push(Instr::ApplyBytesGet);
        self.instr_stack.push(Instr::Eval(Box::new(idx_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(bytes_expr)));
        Ok(())
    }

    fn apply_bytes_get(&mut self) -> Result<(), InterpretError> {
        let idx_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "bytes-get missing index")
        })?;
        let bytes_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "bytes-get missing bytes")
        })?;
        let idx = expect_integer(idx_v, "bytes-get index must be i64")?;
        let bytes = expect_bytes(bytes_v, "bytes-get target must be bytes")?;
        if idx < 0 || (idx as usize) >= bytes.len() {
            return Err(InterpretError::new(
                InterpretErrorKind::BytesIndexOutOfBounds {
                    index: idx,
                    length: bytes.len(),
                },
                "bytes-get: index out of range",
            ));
        }
        self.value_stack
            .push(Value::Integer(bytes[idx as usize] as i64));
        Ok(())
    }

    fn queue_bytes_slice(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 3 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "bytes-slice requires exactly 3 arguments",
            ));
        }
        let len_expr = args.remove(2);
        let start_expr = args.remove(1);
        let bytes_expr = args.remove(0);
        // Execution order: Eval(bytes), Eval(start), Eval(len), ApplyBytesSlice.
        self.instr_stack.push(Instr::ApplyBytesSlice);
        self.instr_stack.push(Instr::Eval(Box::new(len_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(start_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(bytes_expr)));
        Ok(())
    }

    fn apply_bytes_slice(&mut self) -> Result<(), InterpretError> {
        let len_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "bytes-slice missing len")
        })?;
        let start_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "bytes-slice missing start")
        })?;
        let bytes_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "bytes-slice missing bytes")
        })?;
        let len = expect_integer(len_v, "bytes-slice len must be i64")?;
        let start = expect_integer(start_v, "bytes-slice start must be i64")?;
        let bytes = expect_bytes(bytes_v, "bytes-slice target must be bytes")?;
        // Range check uses i128 to avoid overflow when adding two
        // i64s near the type's extremes.
        let bytes_len_i128 = bytes.len() as i128;
        let end_i128 = (start as i128).saturating_add(len as i128);
        if start < 0 || len < 0 || (start as i128) > bytes_len_i128 || end_i128 > bytes_len_i128 {
            return Err(InterpretError::new(
                InterpretErrorKind::BytesSliceOutOfBounds {
                    start,
                    len,
                    bytes_length: bytes.len(),
                },
                "bytes-slice: range out of bounds",
            ));
        }
        let s = start as usize;
        let n = len as usize;
        self.value_stack
            .push(Value::Bytes(bytes[s..s + n].to_vec()));
        Ok(())
    }

    fn queue_bytes_concat(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "bytes-concat requires exactly 2 arguments",
            ));
        }
        let b_expr = args.remove(1);
        let a_expr = args.remove(0);
        self.instr_stack.push(Instr::ApplyBytesConcat);
        self.instr_stack.push(Instr::Eval(Box::new(b_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(a_expr)));
        Ok(())
    }

    fn apply_bytes_concat(&mut self) -> Result<(), InterpretError> {
        let b_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "bytes-concat missing rhs")
        })?;
        let a_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "bytes-concat missing lhs")
        })?;
        let b = expect_bytes(b_v, "bytes-concat rhs must be bytes")?;
        let mut a = expect_bytes(a_v, "bytes-concat lhs must be bytes")?;
        a.extend_from_slice(&b);
        self.value_stack.push(Value::Bytes(a));
        Ok(())
    }

    fn queue_bytes_eq(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "bytes-eq requires exactly 2 arguments",
            ));
        }
        let b_expr = args.remove(1);
        let a_expr = args.remove(0);
        self.instr_stack.push(Instr::ApplyBytesEq);
        self.instr_stack.push(Instr::Eval(Box::new(b_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(a_expr)));
        Ok(())
    }

    fn apply_bytes_eq(&mut self) -> Result<(), InterpretError> {
        let b_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "bytes-eq missing rhs")
        })?;
        let a_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "bytes-eq missing lhs")
        })?;
        let b = expect_bytes(b_v, "bytes-eq rhs must be bytes")?;
        let a = expect_bytes(a_v, "bytes-eq lhs must be bytes")?;
        self.value_stack.push(Value::Bool(a == b));
        Ok(())
    }

    fn queue_bytes_from_int(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "bytes-from-int requires exactly 1 argument",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyBytesFromInt);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_bytes_from_int(&mut self) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "bytes-from-int missing operand",
            )
        })?;
        let n = expect_integer(v, "bytes-from-int operand must be i64")?;
        self.value_stack
            .push(Value::Bytes(n.to_le_bytes().to_vec()));
        Ok(())
    }

    // ── Phase 4 Step 5 — Maybe / Option ──────────────────────────

    fn queue_some(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "some requires exactly 1 argument",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplySome);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_some(&mut self) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "some missing operand")
        })?;
        // Phase 4 monomorphic: the validator already typed the inner
        // expression as I64. Re-check defensively so an unvalidated
        // AST cannot smuggle a non-i64 into the wrapper.
        let _ = expect_integer(v.clone(), "some operand must be i64")?;
        self.value_stack.push(Value::Maybe(Some(Box::new(v))));
        Ok(())
    }

    fn queue_none(&mut self, args: Vec<SExpr>) -> Result<(), InterpretError> {
        if !args.is_empty() {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "none takes no arguments",
            ));
        }
        self.instr_stack.push(Instr::ApplyNone);
        Ok(())
    }

    fn apply_none(&mut self) -> Result<(), InterpretError> {
        self.value_stack.push(Value::Maybe(None));
        Ok(())
    }

    fn queue_is_some(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "is-some requires exactly 1 argument",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyIsSome);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_is_some(&mut self) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "is-some missing operand")
        })?;
        let m = expect_maybe(v, "is-some operand must be Maybe")?;
        self.value_stack.push(Value::Bool(m.is_some()));
        Ok(())
    }

    fn queue_is_none(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "is-none requires exactly 1 argument",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyIsNone);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_is_none(&mut self) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "is-none missing operand")
        })?;
        let m = expect_maybe(v, "is-none operand must be Maybe")?;
        self.value_stack.push(Value::Bool(m.is_none()));
        Ok(())
    }

    fn queue_unwrap(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "unwrap requires exactly 1 argument",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyUnwrap);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_unwrap(&mut self) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "unwrap missing operand")
        })?;
        let m = expect_maybe(v, "unwrap operand must be Maybe")?;
        match m {
            Some(inner) => {
                // Phase 4 monomorphic: the validator types `unwrap`
                // output as I64. Re-verify defensively so a richer
                // inner type smuggled past the validator surfaces a
                // typed error rather than corrupting the value stack.
                let _ = expect_integer(*inner.clone(), "unwrap inner must be i64")?;
                self.value_stack.push(*inner);
                Ok(())
            }
            None => Err(InterpretError::new(
                InterpretErrorKind::UnwrapOnNone,
                "unwrap: Maybe is None",
            )),
        }
    }

    fn queue_unwrap_or(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "unwrap-or requires exactly 2 arguments",
            ));
        }
        let default_expr = args.remove(1);
        let maybe_expr = args.remove(0);
        // Eval order: Eval(maybe), Eval(default), ApplyUnwrapOr.
        self.instr_stack.push(Instr::ApplyUnwrapOr);
        self.instr_stack.push(Instr::Eval(Box::new(default_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(maybe_expr)));
        Ok(())
    }

    fn apply_unwrap_or(&mut self) -> Result<(), InterpretError> {
        let default_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "unwrap-or missing default")
        })?;
        let maybe_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "unwrap-or missing Maybe")
        })?;
        let default = expect_integer(default_v, "unwrap-or default must be i64")?;
        let m = expect_maybe(maybe_v, "unwrap-or operand must be Maybe")?;
        match m {
            Some(inner) => {
                let n = expect_integer(*inner, "unwrap-or inner must be i64")?;
                self.value_stack.push(Value::Integer(n));
            }
            None => {
                self.value_stack.push(Value::Integer(default));
            }
        }
        Ok(())
    }

    // ── Paket B.3 — Maps ─────────────────────────────────────────

    fn queue_map_new(&mut self, args: Vec<SExpr>) -> Result<(), InterpretError> {
        if !args.is_empty() {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "map-new takes no arguments",
            ));
        }
        self.value_stack.push(Value::Map(BTreeMap::new()));
        Ok(())
    }

    fn queue_map_put(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 3 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "map-put requires exactly 3 arguments",
            ));
        }
        let v_expr = args.remove(2);
        let k_expr = args.remove(1);
        let m_expr = args.remove(0);
        // Eval order: m, k, v.
        self.instr_stack.push(Instr::ApplyMapPut);
        self.instr_stack.push(Instr::Eval(Box::new(v_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(k_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(m_expr)));
        Ok(())
    }

    fn apply_map_put(&mut self) -> Result<(), InterpretError> {
        let v_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "map-put missing value")
        })?;
        let k_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "map-put missing key")
        })?;
        let m_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "map-put missing map")
        })?;
        let v = expect_integer(v_v, "map-put value must be i64")?;
        let k = expect_integer(k_v, "map-put key must be i64")?;
        let mut m = expect_map(m_v, "map-put target must be a map")?;
        m.insert(k, v);
        self.value_stack.push(Value::Map(m));
        Ok(())
    }

    fn queue_map_get(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "map-get requires exactly 2 arguments",
            ));
        }
        let k_expr = args.remove(1);
        let m_expr = args.remove(0);
        self.instr_stack.push(Instr::ApplyMapGet);
        self.instr_stack.push(Instr::Eval(Box::new(k_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(m_expr)));
        Ok(())
    }

    fn apply_map_get(&mut self) -> Result<(), InterpretError> {
        let k_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "map-get missing key")
        })?;
        let m_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "map-get missing map")
        })?;
        let k = expect_integer(k_v, "map-get key must be i64")?;
        let m = expect_map(m_v, "map-get target must be a map")?;
        let result = m.get(&k).copied().unwrap_or(0);
        self.value_stack.push(Value::Integer(result));
        Ok(())
    }

    fn queue_map_contains(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "map-contains requires exactly 2 arguments",
            ));
        }
        let k_expr = args.remove(1);
        let m_expr = args.remove(0);
        self.instr_stack.push(Instr::ApplyMapContains);
        self.instr_stack.push(Instr::Eval(Box::new(k_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(m_expr)));
        Ok(())
    }

    fn apply_map_contains(&mut self) -> Result<(), InterpretError> {
        let k_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "map-contains missing key")
        })?;
        let m_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "map-contains missing map")
        })?;
        let k = expect_integer(k_v, "map-contains key must be i64")?;
        let m = expect_map(m_v, "map-contains target must be a map")?;
        self.value_stack.push(Value::Bool(m.contains_key(&k)));
        Ok(())
    }

    // ── Paket B.5 — Strings ──────────────────────────────────────

    fn queue_string_from_int(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "string-from-int requires exactly 1 argument",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyStringFromInt);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_string_from_int(&mut self) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "string-from-int missing operand",
            )
        })?;
        let n = expect_integer(v, "string-from-int requires i64")?;
        // Deterministic int→string: rely on Rust's i64 ToString, which
        // is decimal and locale-free.
        use alloc::string::ToString;
        self.value_stack.push(Value::String(n.to_string()));
        Ok(())
    }

    fn queue_string_concat(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "string-concat requires exactly 2 arguments",
            ));
        }
        let b_expr = args.remove(1);
        let a_expr = args.remove(0);
        self.instr_stack.push(Instr::ApplyStringConcat);
        self.instr_stack.push(Instr::Eval(Box::new(b_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(a_expr)));
        Ok(())
    }

    fn apply_string_concat(&mut self) -> Result<(), InterpretError> {
        let b_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "string-concat missing rhs")
        })?;
        let a_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "string-concat missing lhs")
        })?;
        let b = expect_string(b_v, "string-concat rhs must be string")?;
        let mut a = expect_string(a_v, "string-concat lhs must be string")?;
        a.push_str(&b);
        self.value_stack.push(Value::String(a));
        Ok(())
    }

    fn queue_string_eq(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "string-eq requires exactly 2 arguments",
            ));
        }
        let b_expr = args.remove(1);
        let a_expr = args.remove(0);
        self.instr_stack.push(Instr::ApplyStringEq);
        self.instr_stack.push(Instr::Eval(Box::new(b_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(a_expr)));
        Ok(())
    }

    fn apply_string_eq(&mut self) -> Result<(), InterpretError> {
        let b_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "string-eq missing rhs")
        })?;
        let a_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "string-eq missing lhs")
        })?;
        let b = expect_string(b_v, "string-eq rhs must be string")?;
        let a = expect_string(a_v, "string-eq lhs must be string")?;
        self.value_stack.push(Value::Bool(a == b));
        Ok(())
    }

    // ── Paket B.6 — I/O ──────────────────────────────────────────

    fn queue_read_handle(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 1 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "read-handle requires exactly 1 argument (handle)",
            ));
        }
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyReadHandle);
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_read_handle<C: PolicyContext + ?Sized>(
        &mut self,
        ctx: &mut C,
    ) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "read-handle missing operand")
        })?;
        let handle = match v {
            Value::Handle(h) => h,
            other => {
                return Err(InterpretError::new(
                    InterpretErrorKind::UnsupportedAtom,
                    match other {
                        Value::Integer(_) => "read-handle expects Handle, got Integer",
                        Value::Bytes(_) => "read-handle expects Handle, got Bytes",
                        Value::Bool(_) => "read-handle expects Handle, got Bool",
                        Value::List(_) => "read-handle expects Handle, got List",
                        Value::Map(_) => "read-handle expects Handle, got Map",
                        Value::String(_) => "read-handle expects Handle, got String",
                        Value::Maybe(_) => "read-handle expects Handle, got Maybe",
                        Value::Struct { .. } => "read-handle expects Handle, got Struct",
                        Value::Handle(_) => unreachable!(),
                    },
                ));
            }
        };
        match ctx.read_handle(handle) {
            Ok(bytes) => {
                self.value_stack.push(Value::Bytes(bytes));
                Ok(())
            }
            Err(err) => Err(map_policy_error(err)),
        }
    }

    fn queue_write_host_state(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "write-host-state requires exactly 2 arguments (key value)",
            ));
        }
        let key = match &args[0] {
            SExpr::Atom(Atom::Symbol(s)) => s.clone(),
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::PolicyMalformed,
                    "write-host-state key (arg 0) must be a symbol literal",
                ));
            }
        };
        // Drop key, value remains in args[0] after the remove.
        args.remove(0);
        let value_expr = args.remove(0);
        self.instr_stack.push(Instr::ApplyWriteHostState { key });
        self.instr_stack.push(Instr::Eval(Box::new(value_expr)));
        Ok(())
    }

    fn apply_write_host_state<C: PolicyContext + ?Sized>(
        &mut self,
        key: &str,
        ctx: &mut C,
    ) -> Result<(), InterpretError> {
        let v_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "write-host-state missing value",
            )
        })?;
        let val = expect_integer(v_v, "write-host-state value must be i64")?;
        match ctx.write_host_state(key, val) {
            Ok(status) => {
                self.value_stack.push(Value::Integer(status));
                Ok(())
            }
            Err(err) => Err(map_policy_error(err)),
        }
    }

    // ── Paket B.4 — Bounded Loops ────────────────────────────────

    fn queue_loop_with_bound(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "loop-with-bound requires (loop-with-bound bound body)",
            ));
        }
        let body = Arc::new(args.remove(1));
        let bound_expr = args.remove(0);

        // Reserve the loop frame BEFORE evaluating the bound so that
        // user-level (break v) inside body unwinds to this depth. We
        // store the loop-frame snapshot at the depth that will hold
        // when bound-eval finishes.
        //
        // Sequence: Eval(bound) → LoopBoundedIter(remaining, body) …
        // We use a custom helper instr: the next iteration's start.
        // The Eval(bound) pushes one i64 onto value_stack; the
        // LoopBoundedIter pops it, snapshots the loop frame, and
        // dispatches the first iteration.
        //
        // Stage: push LoopBoundedIter with remaining=-1 (sentinel to
        // mean "consume bound from stack"); the dispatch reads from
        // value_stack instead of from `remaining`. Simpler: introduce
        // a dedicated start variant.
        //
        // We model it with a small inline helper: after Eval(bound)
        // completes, the value_stack top holds the bound i64; we then
        // dispatch a starter instr that pops it.
        //
        // To keep the Instr surface lean, we reuse LoopBoundedIter
        // with `remaining == i64::MIN` as the "needs to read from
        // stack" sentinel. That avoids adding another Instr variant.
        self.instr_stack.push(Instr::LoopBoundedIter {
            remaining: i64::MIN,
            body,
        });
        self.instr_stack.push(Instr::Eval(Box::new(bound_expr)));
        Ok(())
    }

    fn dispatch_loop_bounded_iter(
        &mut self,
        remaining: i64,
        body: Arc<SExpr>,
    ) -> Result<(), InterpretError> {
        let remaining = if remaining == i64::MIN {
            // First invocation — consume the bound from the value
            // stack and snapshot the loop frame at the current depth.
            let bound_v = self.value_stack.pop().ok_or_else(|| {
                InterpretError::new(
                    InterpretErrorKind::EmptyList,
                    "loop-with-bound: bound expr produced no value",
                )
            })?;
            let bound = expect_integer(bound_v, "loop-with-bound bound must be i64")?;
            if bound < 0 {
                return Err(InterpretError::new(
                    InterpretErrorKind::LoopBoundInvalid,
                    "loop-with-bound: negative bound",
                ));
            }
            // Push the loop frame snapshot now (post-bound-pop).
            self.loop_stack.push(LoopFrame {
                instr_stack_depth_at_entry: self.instr_stack.len(),
                value_stack_depth_at_entry: self.value_stack.len(),
            });
            bound
        } else {
            remaining
        };

        if remaining <= 0 {
            // Bound exhausted — implicit (break false).
            let frame = self.loop_stack.pop().ok_or_else(|| {
                InterpretError::new(
                    InterpretErrorKind::BreakOutsideLoop,
                    "loop-with-bound: no enclosing loop frame at implicit break",
                )
            })?;
            self.instr_stack.truncate(frame.instr_stack_depth_at_entry);
            self.value_stack.truncate(frame.value_stack_depth_at_entry);
            self.value_stack.push(Value::Bool(false));
            return Ok(());
        }

        // Schedule one iteration: Eval(body) then LoopBoundedAfterBody.
        self.instr_stack.push(Instr::LoopBoundedAfterBody {
            remaining,
            body: Arc::clone(&body),
        });
        let body_owned: SExpr = match Arc::try_unwrap(body) {
            Ok(inner) => inner,
            Err(arc) => (*arc).clone(),
        };
        self.instr_stack.push(Instr::Eval(Box::new(body_owned)));
        Ok(())
    }

    fn dispatch_loop_bounded_after_body(
        &mut self,
        remaining: i64,
        body: Arc<SExpr>,
    ) -> Result<(), InterpretError> {
        // Body completed by fall-through; discard residual.
        if self.value_stack.pop().is_none() {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "loop-with-bound: body produced no residual value to discard",
            ));
        }
        // Schedule next iteration with remaining decremented.
        self.instr_stack.push(Instr::LoopBoundedIter {
            remaining: remaining - 1,
            body,
        });
        Ok(())
    }

    fn queue_for(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 3 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "for requires (for %n source body)",
            ));
        }
        let idx = match &args[0] {
            SExpr::Atom(Atom::Parameter(n)) => *n,
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::NonSymbolHead,
                    "for binding target must be a parameter atom (%n)",
                ));
            }
        };
        let body = Arc::new(args.remove(2));
        let source_expr = args.remove(1);
        // Sequence: Eval(source) → ForAfterSource(idx, body).
        // ForAfterSource snapshots the loop frame and dispatches the
        // first iteration.
        self.instr_stack.push(Instr::ForAfterSource { idx, body });
        self.instr_stack.push(Instr::Eval(Box::new(source_expr)));
        Ok(())
    }

    fn dispatch_for_after_source(
        &mut self,
        idx: u32,
        body: Arc<SExpr>,
    ) -> Result<(), InterpretError> {
        let source_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "for: source expression produced no value",
            )
        })?;
        let source = expect_list(source_v, "for source must be a list")?;
        // Snapshot the loop frame at the post-source-pop depth.
        self.loop_stack.push(LoopFrame {
            instr_stack_depth_at_entry: self.instr_stack.len(),
            value_stack_depth_at_entry: self.value_stack.len(),
        });
        // Dispatch the first iteration.
        self.instr_stack.push(Instr::ForIter {
            idx,
            remaining: source,
            body,
        });
        Ok(())
    }

    fn dispatch_for_iter(
        &mut self,
        idx: u32,
        mut remaining: Vec<i64>,
        body: Arc<SExpr>,
    ) -> Result<(), InterpretError> {
        if remaining.is_empty() {
            // Iteration exhausted — implicit (break true).
            let frame = self.loop_stack.pop().ok_or_else(|| {
                InterpretError::new(
                    InterpretErrorKind::BreakOutsideLoop,
                    "for: no enclosing loop frame at exhaustion",
                )
            })?;
            self.instr_stack.truncate(frame.instr_stack_depth_at_entry);
            self.value_stack.truncate(frame.value_stack_depth_at_entry);
            self.value_stack.push(Value::Bool(true));
            return Ok(());
        }
        // Pop the head of the remaining list into the loop variable.
        let head = remaining.remove(0);
        // Bind into the current frame's locals. The validator
        // guaranteed `for` is inside a function body and the local
        // index does not collide.
        let frame = self.call_stack.current_mut().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::ParameterOutsideFunction,
                "for outside any function body (no active frame)",
            )
        })?;
        // The local may already be bound from the previous iteration;
        // unbind first to avoid the redefinition error.
        frame.unbind_local(idx);
        frame.bind_local(idx, Value::Integer(head))?;
        // Schedule body eval, then ForAfterBody.
        self.instr_stack.push(Instr::ForAfterBody {
            idx,
            remaining,
            body: Arc::clone(&body),
        });
        let body_owned: SExpr = match Arc::try_unwrap(body) {
            Ok(inner) => inner,
            Err(arc) => (*arc).clone(),
        };
        self.instr_stack.push(Instr::Eval(Box::new(body_owned)));
        Ok(())
    }

    fn dispatch_for_after_body(
        &mut self,
        idx: u32,
        remaining: Vec<i64>,
        body: Arc<SExpr>,
    ) -> Result<(), InterpretError> {
        // Body completed by fall-through; discard residual.
        if self.value_stack.pop().is_none() {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "for: body produced no residual value to discard",
            ));
        }
        // Schedule next iteration.
        self.instr_stack.push(Instr::ForIter {
            idx,
            remaining,
            body,
        });
        Ok(())
    }

    // ── Phase 4 Step 6 — Structs ─────────────────────────────────

    /// Queue `(struct-new name v1 v2 … vN)`. Looks up the struct
    /// declaration in the symbol table to capture the field-name
    /// order, then schedules N value evaluations followed by an
    /// ApplyStructNew continuation.
    fn queue_struct_new(
        &mut self,
        mut args: Vec<SExpr>,
        symbols: &SymbolTable,
    ) -> Result<(), InterpretError> {
        if args.is_empty() {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "struct-new requires a struct name",
            ));
        }
        let name = match &args[0] {
            SExpr::Atom(Atom::Symbol(s)) => s.clone(),
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::NonSymbolHead,
                    "struct-new target must be a struct-name symbol",
                ));
            }
        };
        args.remove(0);
        let def = symbols.lookup_struct(&name).ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::FunctionNotFound,
                "struct-new: unknown struct",
            )
        })?;
        if args.len() != def.fields.len() {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "struct-new: argument count does not match struct field count",
            ));
        }
        let fields = def.fields.clone();
        // Push ApplyStructNew first so it runs after all field evals;
        // then push field evals in reverse so the leftmost arg
        // evaluates first (its result ends up deepest on the stack
        // and matches fields[0]).
        self.instr_stack
            .push(Instr::ApplyStructNew { name, fields });
        for arg in args.into_iter().rev() {
            self.instr_stack.push(Instr::Eval(Box::new(arg)));
        }
        Ok(())
    }

    fn apply_struct_new(
        &mut self,
        name: String,
        fields: Vec<String>,
    ) -> Result<(), InterpretError> {
        let n = fields.len();
        if self.value_stack.len() < n {
            return Err(InterpretError::new(
                InterpretErrorKind::EmptyList,
                "apply_struct_new: value stack underflow",
            ));
        }
        // Pop in reverse and reverse the buffer to restore source
        // order before zipping with the field names.
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            let v = self.value_stack.pop().ok_or_else(|| {
                InterpretError::new(
                    InterpretErrorKind::EmptyList,
                    "apply_struct_new: value stack underflow during pop",
                )
            })?;
            values.push(v);
        }
        values.reverse();
        let mut map: BTreeMap<String, Value> = BTreeMap::new();
        for (fname, val) in fields.into_iter().zip(values.into_iter()) {
            map.insert(fname, val);
        }
        self.value_stack.push(Value::Struct { name, fields: map });
        Ok(())
    }

    /// Queue `(struct-get expr field-name)`. Captures the field-name
    /// at queue time so the continuation is self-contained.
    fn queue_struct_get(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "struct-get requires (struct-get expr field-name)",
            ));
        }
        let field = match &args[1] {
            SExpr::Atom(Atom::Symbol(s)) => s.clone(),
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::NonSymbolHead,
                    "struct-get field-name must be a symbol literal",
                ));
            }
        };
        args.remove(1);
        let inner = args.remove(0);
        self.instr_stack.push(Instr::ApplyStructGet { field });
        self.instr_stack.push(Instr::Eval(Box::new(inner)));
        Ok(())
    }

    fn apply_struct_get(&mut self, field: &str) -> Result<(), InterpretError> {
        let v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "struct-get missing operand")
        })?;
        match v {
            Value::Struct { fields, .. } => match fields.get(field) {
                Some(val) => {
                    self.value_stack.push(val.clone());
                    Ok(())
                }
                None => Err(InterpretError::new(
                    InterpretErrorKind::FunctionNotFound,
                    "struct-get: unknown field",
                )),
            },
            _ => Err(InterpretError::new(
                InterpretErrorKind::UnsupportedAtom,
                "struct-get target must be a Struct",
            )),
        }
    }

    /// Queue `(struct-set expr field-name new-value)`. Captures the
    /// field-name at queue time; evaluates expr then new-value.
    fn queue_struct_set(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() != 3 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "struct-set requires (struct-set expr field-name new-value)",
            ));
        }
        let field = match &args[1] {
            SExpr::Atom(Atom::Symbol(s)) => s.clone(),
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::NonSymbolHead,
                    "struct-set field-name must be a symbol literal",
                ));
            }
        };
        let new_value_expr = args.remove(2);
        args.remove(1); // field-name
        let struct_expr = args.remove(0);
        // Eval order: Eval(struct), Eval(new_value), ApplyStructSet.
        self.instr_stack.push(Instr::ApplyStructSet { field });
        self.instr_stack.push(Instr::Eval(Box::new(new_value_expr)));
        self.instr_stack.push(Instr::Eval(Box::new(struct_expr)));
        Ok(())
    }

    fn apply_struct_set(&mut self, field: &str) -> Result<(), InterpretError> {
        let new_val = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "struct-set missing new value",
            )
        })?;
        let struct_v = self.value_stack.pop().ok_or_else(|| {
            InterpretError::new(InterpretErrorKind::EmptyList, "struct-set missing struct")
        })?;
        match struct_v {
            Value::Struct { name, mut fields } => {
                if !fields.contains_key(field) {
                    return Err(InterpretError::new(
                        InterpretErrorKind::FunctionNotFound,
                        "struct-set: unknown field",
                    ));
                }
                fields.insert(String::from(field), new_val);
                self.value_stack.push(Value::Struct { name, fields });
                Ok(())
            }
            _ => Err(InterpretError::new(
                InterpretErrorKind::UnsupportedAtom,
                "struct-set target must be a Struct",
            )),
        }
    }

    // ── Phase 4 Step 7 — Pattern matching ──────────────────────

    /// Queue `(match scrutinee (case pattern body) …)`. Plans:
    ///   Eval(scrutinee) → MatchDispatch{cases}
    /// The scrutinee evaluation pushes its value to the value
    /// stack; `MatchDispatch` then peels and matches cases in
    /// source order.
    fn queue_match(&mut self, mut args: Vec<SExpr>) -> Result<(), InterpretError> {
        if args.len() < 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::ArityMismatch,
                "match requires (match scrutinee (case pattern body) …)",
            ));
        }
        // args[0] = scrutinee, args[1..] = case clauses.
        let scrutinee = args.remove(0);
        let mut cases: Vec<MatchCase> = Vec::with_capacity(args.len());
        for case in args.into_iter() {
            let mut case_items = match case {
                SExpr::List(l) => l,
                _ => {
                    return Err(InterpretError::new(
                        InterpretErrorKind::NonSymbolHead,
                        "match case must be a (case pattern body) list",
                    ));
                }
            };
            if case_items.len() != 3 {
                return Err(InterpretError::new(
                    InterpretErrorKind::ArityMismatch,
                    "match case must be (case pattern body) — exactly 3 items",
                ));
            }
            let head_is_case = matches!(
                &case_items[0],
                SExpr::Atom(Atom::Symbol(s)) if s == "case"
            );
            if !head_is_case {
                return Err(InterpretError::new(
                    InterpretErrorKind::NonSymbolHead,
                    "match case head must be the symbol `case`",
                ));
            }
            let body = case_items.remove(2);
            let pattern = case_items.remove(1);
            cases.push(MatchCase { pattern, body });
        }
        // Push the dispatch continuation FIRST, then the scrutinee
        // eval — the eval pops first and runs the scrutinee
        // expression, leaving the value on top of the value stack
        // when the dispatch is reached.
        self.instr_stack.push(Instr::MatchDispatch { cases });
        self.instr_stack.push(Instr::Eval(Box::new(scrutinee)));
        Ok(())
    }

    /// Handle one `MatchDispatch` step:
    /// 1. Read (clone) the scrutinee value from the top of the
    ///    value stack. Cloning lets us preserve it across pattern
    ///    attempts that may need to inspect the same value.
    /// 2. Try each case in source order. The first matching pattern
    ///    pops the scrutinee, binds any pattern parameters into the
    ///    current frame, pushes `MatchUnbind { bindings }`, then
    ///    pushes the body's `Eval`.
    /// 3. If no case matches, surface `MatchNonExhaustive` — the
    ///    validator requires a wildcard last case, so this path is
    ///    unreachable for validated IR. Defensive only.
    ///
    /// The `symbols` reference threads the [`SymbolTable`] through
    /// to the pattern matcher so struct patterns can resolve the
    /// declared field order from `StructDef::fields` — necessary
    /// because `Value::Struct` stores fields in a `BTreeMap` which
    /// iterates alphabetically, not in declaration order.
    fn dispatch_match(
        &mut self,
        cases: Vec<MatchCase>,
        symbols: &SymbolTable,
    ) -> Result<(), InterpretError> {
        let scrutinee = self.value_stack.last().cloned().ok_or_else(|| {
            InterpretError::new(
                InterpretErrorKind::EmptyList,
                "match scrutinee value missing from value stack",
            )
        })?;
        for case in cases.into_iter() {
            if let Some(bindings) = pattern_match(&case.pattern, &scrutinee, symbols)? {
                // Pattern matched. Pop the scrutinee from the value
                // stack — the body must run against the pre-match
                // stack state.
                let _ = self.value_stack.pop();
                // Apply bindings into the current frame.
                let bound_indices: Vec<u32> = bindings.iter().map(|(i, _)| *i).collect();
                if !bindings.is_empty() {
                    let frame = self.call_stack.current_mut().ok_or_else(|| {
                        InterpretError::new(
                            InterpretErrorKind::ParameterOutsideFunction,
                            "match pattern bindings require an active function frame",
                        )
                    })?;
                    for (idx, val) in bindings.into_iter() {
                        frame.bind_local(idx, val)?;
                    }
                }
                // Schedule the body's evaluation followed by the
                // unbind. Push order is reverse-of-execution.
                self.instr_stack.push(Instr::MatchUnbind {
                    bindings: bound_indices,
                });
                self.instr_stack.push(Instr::Eval(Box::new(case.body)));
                return Ok(());
            }
        }
        // No case matched — validator-guaranteed unreachable.
        Err(InterpretError::new(
            InterpretErrorKind::MatchNonExhaustive,
            "match drained all cases without a matching pattern",
        ))
    }

    fn dispatch_match_unbind(&mut self, bindings: Vec<u32>) -> Result<(), InterpretError> {
        if bindings.is_empty() {
            return Ok(());
        }
        // Unbind in reverse so the most-recently bound slot pops
        // first, mirroring `let`'s LIFO scope discipline.
        if let Some(frame) = self.call_stack.current_mut() {
            for idx in bindings.into_iter().rev() {
                frame.unbind_local(idx);
            }
        }
        Ok(())
    }
}

/// Phase 4 Step 7 — runtime pattern matcher. Returns:
///   - `Ok(Some(bindings))` — pattern matched; `bindings` is the
///     ordered list of `(parameter-index, captured-value)` pairs
///     ready for `frame.bind_local`. Empty for patterns that bind
///     nothing (wildcard, integer literal, `(none)`).
///   - `Ok(None)` — pattern did not match this scrutinee.
///   - `Err(_)` — pattern was structurally malformed. Validator
///     catches these statically; this branch is for adversarial /
///     unvalidated IR.
///
/// `symbols` is threaded through so struct patterns can resolve
/// the declared field order from `SymbolTable::lookup_struct`.
fn pattern_match(
    pattern: &SExpr,
    scrutinee: &Value,
    symbols: &SymbolTable,
) -> Result<Option<Vec<(u32, Value)>>, InterpretError> {
    // Wildcard — matches anything.
    if let SExpr::Atom(Atom::Symbol(s)) = pattern {
        if s == "_" {
            return Ok(Some(Vec::new()));
        }
    }
    // Integer literal — exact value match.
    if let SExpr::Atom(Atom::Integer(n)) = pattern {
        return match scrutinee {
            Value::Integer(m) if m == n => Ok(Some(Vec::new())),
            Value::Integer(_) => Ok(None),
            _ => Err(InterpretError::new(
                InterpretErrorKind::UnsupportedAtom,
                "match integer-literal pattern requires an integer scrutinee",
            )),
        };
    }
    // List patterns: (some %n), (none), (struct T %a %b …).
    let p_items = match pattern {
        SExpr::List(l) => l,
        _ => {
            return Err(InterpretError::new(
                InterpretErrorKind::UnsupportedAtom,
                "match pattern is not a recognised shape",
            ));
        }
    };
    let head = match p_items.first() {
        Some(SExpr::Atom(Atom::Symbol(s))) => s.as_str(),
        _ => {
            return Err(InterpretError::new(
                InterpretErrorKind::NonSymbolHead,
                "match pattern head must be a symbol",
            ));
        }
    };
    match head {
        "some" => {
            if p_items.len() != 2 {
                return Err(InterpretError::new(
                    InterpretErrorKind::ArityMismatch,
                    "match `(some %n)` pattern requires exactly one bind slot",
                ));
            }
            let idx = match &p_items[1] {
                SExpr::Atom(Atom::Parameter(n)) => *n,
                _ => {
                    return Err(InterpretError::new(
                        InterpretErrorKind::NonSymbolHead,
                        "match `(some …)` bind slot must be a parameter atom (%n)",
                    ));
                }
            };
            match scrutinee {
                Value::Maybe(Some(inner)) => Ok(Some(alloc::vec![(idx, (**inner).clone())])),
                Value::Maybe(None) => Ok(None),
                _ => Err(InterpretError::new(
                    InterpretErrorKind::UnsupportedAtom,
                    "match `(some …)` pattern requires a Maybe scrutinee",
                )),
            }
        }
        "none" => {
            if p_items.len() != 1 {
                return Err(InterpretError::new(
                    InterpretErrorKind::ArityMismatch,
                    "match `(none)` pattern takes no bind slots",
                ));
            }
            match scrutinee {
                Value::Maybe(None) => Ok(Some(Vec::new())),
                Value::Maybe(Some(_)) => Ok(None),
                _ => Err(InterpretError::new(
                    InterpretErrorKind::UnsupportedAtom,
                    "match `(none)` pattern requires a Maybe scrutinee",
                )),
            }
        }
        "struct" => {
            if p_items.len() < 2 {
                return Err(InterpretError::new(
                    InterpretErrorKind::ArityMismatch,
                    "match `(struct …)` pattern requires a struct name",
                ));
            }
            let struct_name = match &p_items[1] {
                SExpr::Atom(Atom::Symbol(s)) => s,
                _ => {
                    return Err(InterpretError::new(
                        InterpretErrorKind::NonSymbolHead,
                        "match `(struct …)` pattern name must be a symbol",
                    ));
                }
            };
            let bind_slots = &p_items[2..];
            // Scrutinee must be a struct with the same nominal name.
            let (s_name, s_fields) = match scrutinee {
                Value::Struct { name, fields } => (name, fields),
                _ => {
                    return Err(InterpretError::new(
                        InterpretErrorKind::UnsupportedAtom,
                        "match `(struct …)` pattern requires a Struct scrutinee",
                    ));
                }
            };
            if s_name != struct_name {
                return Ok(None);
            }
            // Resolve declared field order from the SymbolTable —
            // `Value::Struct` stores fields in a BTreeMap (keyed by
            // field-name string for determinism), so we need the
            // declaration's order to bind positional `%n` slots
            // correctly. The validator enforced shape + field count
            // statically; we re-check defensively.
            let def = symbols.lookup_struct(struct_name).ok_or_else(|| {
                InterpretError::new(
                    InterpretErrorKind::FunctionNotFound,
                    "match: struct name not in symbol table",
                )
            })?;
            if bind_slots.len() != def.fields.len() {
                return Err(InterpretError::new(
                    InterpretErrorKind::ArityMismatch,
                    "match struct pattern: bind-slot count differs from struct's field count",
                ));
            }
            if bind_slots.len() != s_fields.len() {
                return Err(InterpretError::new(
                    InterpretErrorKind::ArityMismatch,
                    "match struct pattern: scrutinee's field count differs from declaration",
                ));
            }
            let mut bindings: Vec<(u32, Value)> = Vec::with_capacity(bind_slots.len());
            for (slot, field_name) in bind_slots.iter().zip(def.fields.iter()) {
                let idx = match slot {
                    SExpr::Atom(Atom::Parameter(n)) => *n,
                    _ => {
                        return Err(InterpretError::new(
                            InterpretErrorKind::NonSymbolHead,
                            "match struct pattern field slot must be a parameter atom (%n)",
                        ));
                    }
                };
                let field_val = s_fields.get(field_name).ok_or_else(|| {
                    InterpretError::new(
                        InterpretErrorKind::FunctionNotFound,
                        "match struct pattern: declared field not present in scrutinee",
                    )
                })?;
                bindings.push((idx, field_val.clone()));
            }
            Ok(Some(bindings))
        }
        _ => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "match pattern head is not a recognised pattern shape",
        )),
    }
}

/// Paket B.6: map a `PolicyError` from `read_handle`/`write_host_state`
/// onto the interpreter's error surface. Mirrors `apply_policy`'s
/// translation (PolicyNotSupported / PolicyDispatchFailed).
fn map_policy_error(err: PolicyError) -> InterpretError {
    let kind = match err {
        PolicyError::NotSupported => InterpretErrorKind::PolicyNotSupported,
        _ => InterpretErrorKind::PolicyDispatchFailed,
    };
    let message = match err {
        PolicyError::NotSupported => "I/O surface not supported by this context",
        PolicyError::UnknownSubsystem => "I/O surface: unknown subsystem/key",
        PolicyError::UnknownOperation => "I/O surface: unknown operation/key",
        PolicyError::InvalidArgument => "I/O surface: invalid argument",
        PolicyError::PermissionDenied => "I/O surface: capability missing or wrong kind",
        PolicyError::OperationFailed => "I/O surface: underlying operation failed",
    };
    InterpretError::new(kind, message)
}

fn expect_integer(v: Value, _ctx: &'static str) -> Result<i64, InterpretError> {
    match v {
        Value::Integer(n) => Ok(n),
        Value::Handle(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "binary op requires integer arguments, got handle",
        )),
        Value::Bytes(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "binary op requires integer arguments, got bytes",
        )),
        Value::Bool(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "binary op requires integer arguments, got bool",
        )),
        Value::List(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "binary op requires integer arguments, got list",
        )),
        Value::Map(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "binary op requires integer arguments, got map",
        )),
        Value::String(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "binary op requires integer arguments, got string",
        )),
        Value::Maybe(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "binary op requires integer arguments, got Maybe",
        )),
        Value::Struct { .. } => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "binary op requires integer arguments, got Struct",
        )),
    }
}

/// Paket B.2: expect a [`Value::List`]. Surfaces a typed error on
/// every other variant so the list dispatch paths stay panic-free.
fn expect_list(v: Value, _ctx: &'static str) -> Result<Vec<i64>, InterpretError> {
    match v {
        Value::List(l) => Ok(l),
        Value::Integer(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "List expected, got Integer",
        )),
        Value::Handle(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "List expected, got Handle",
        )),
        Value::Bytes(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "List expected, got Bytes",
        )),
        Value::Bool(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "List expected, got Bool",
        )),
        Value::Map(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "List expected, got Map",
        )),
        Value::String(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "List expected, got String",
        )),
        Value::Maybe(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "List expected, got Maybe",
        )),
        Value::Struct { .. } => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "List expected, got Struct",
        )),
    }
}

/// Phase 4 Step 2: expect a [`Value::Bytes`]. Enumerates every
/// other variant explicitly so the compiler enforces extension
/// when new `Value` variants are added — mirroring `expect_list` /
/// `expect_map`.
fn expect_bytes(v: Value, _ctx: &'static str) -> Result<Vec<u8>, InterpretError> {
    match v {
        Value::Bytes(b) => Ok(b),
        Value::Integer(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bytes expected, got Integer",
        )),
        Value::Handle(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bytes expected, got Handle",
        )),
        Value::Bool(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bytes expected, got Bool",
        )),
        Value::List(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bytes expected, got List",
        )),
        Value::Map(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bytes expected, got Map",
        )),
        Value::String(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bytes expected, got String",
        )),
        Value::Maybe(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bytes expected, got Maybe",
        )),
        Value::Struct { .. } => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bytes expected, got Struct",
        )),
    }
}

/// Paket B.3: expect a [`Value::Map`].
fn expect_map(v: Value, _ctx: &'static str) -> Result<BTreeMap<i64, i64>, InterpretError> {
    match v {
        Value::Map(m) => Ok(m),
        Value::Integer(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Map expected, got Integer",
        )),
        Value::Handle(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Map expected, got Handle",
        )),
        Value::Bytes(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Map expected, got Bytes",
        )),
        Value::Bool(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Map expected, got Bool",
        )),
        Value::List(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Map expected, got List",
        )),
        Value::String(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Map expected, got String",
        )),
        Value::Maybe(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Map expected, got Maybe",
        )),
        Value::Struct { .. } => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Map expected, got Struct",
        )),
    }
}

/// Paket B.5: expect a [`Value::String`].
fn expect_string(v: Value, _ctx: &'static str) -> Result<String, InterpretError> {
    match v {
        Value::String(s) => Ok(s),
        Value::Integer(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "String expected, got Integer",
        )),
        Value::Handle(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "String expected, got Handle",
        )),
        Value::Bytes(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "String expected, got Bytes",
        )),
        Value::Bool(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "String expected, got Bool",
        )),
        Value::List(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "String expected, got List",
        )),
        Value::Map(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "String expected, got Map",
        )),
        Value::Maybe(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "String expected, got Maybe",
        )),
        Value::Struct { .. } => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "String expected, got Struct",
        )),
    }
}

/// Paket B: expect a [`Value::Bool`]. Surfaces a typed error on
/// every other variant so the comparison/logical/`if`/`while`
/// dispatch paths stay panic-free in Ring-0.
fn expect_bool(v: Value, _ctx: &'static str) -> Result<bool, InterpretError> {
    match v {
        Value::Bool(b) => Ok(b),
        Value::Integer(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bool expected, got Integer",
        )),
        Value::Handle(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bool expected, got Handle",
        )),
        Value::Bytes(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bool expected, got Bytes",
        )),
        Value::List(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bool expected, got List",
        )),
        Value::Map(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bool expected, got Map",
        )),
        Value::String(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bool expected, got String",
        )),
        Value::Maybe(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bool expected, got Maybe",
        )),
        Value::Struct { .. } => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Bool expected, got Struct",
        )),
    }
}

/// Phase 4 Step 5: expect a [`Value::Maybe`]. Surfaces a typed error
/// on every other variant so the `is-some` / `is-none` / `unwrap` /
/// `unwrap-or` dispatch paths stay panic-free in Ring-0.
fn expect_maybe(v: Value, _ctx: &'static str) -> Result<Option<Box<Value>>, InterpretError> {
    match v {
        Value::Maybe(m) => Ok(m),
        Value::Integer(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Maybe expected, got Integer",
        )),
        Value::Handle(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Maybe expected, got Handle",
        )),
        Value::Bytes(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Maybe expected, got Bytes",
        )),
        Value::Bool(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Maybe expected, got Bool",
        )),
        Value::List(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Maybe expected, got List",
        )),
        Value::Map(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Maybe expected, got Map",
        )),
        Value::String(_) => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Maybe expected, got String",
        )),
        Value::Struct { .. } => Err(InterpretError::new(
            InterpretErrorKind::UnsupportedAtom,
            "Maybe expected, got Struct",
        )),
    }
}

/// Find the entry-point expression for evaluation, returning an
/// owned clone of the appropriate subtree.
///
/// - Bare expression: a clone of the AST itself.
/// - `(program ...)` wrapper: a clone of the last child (typically a
///   `(call main)`). `(fn ...)` siblings are handled by
///   [`collect_signatures`] in Pass 1, not the machine.
pub(crate) fn find_program_entry(ast: &SExpr) -> SExpr {
    // Pre-A.1 (F-C3): unwrap-free. If the `(program …)` pattern does
    // not match (or the `last` binding is unexpectedly empty), the AST
    // itself is returned (bare-expression fallback); no path can
    // panic.
    if let SExpr::List(items) = ast {
        if let (Some(SExpr::Atom(Atom::Symbol(s))), Some(last)) = (items.first(), items.last()) {
            if s == "program" && items.len() > 1 {
                return last.clone();
            }
        }
    }
    ast.clone()
}

/// Bundle of [`SymbolTable`] + [`Machine`] for one program run. This
/// is the recommended entry point for instruction-level execution.
///
/// Stage 12 Paket A.4 made `Session` **lifetime-free**: the type no
/// longer borrows from the caller's AST. Internally the session owns
/// the function-body bindings (via [`Arc<SExpr>`](alloc::sync::Arc)
/// inside the symbol table) and the active subtree (inside
/// [`Box<SExpr>`](alloc::boxed::Box) Instr variants). This lets the
/// session be stored inside a `Sandbox` and resumed across watchdog
/// preemption without producing a self-referential struct.
///
/// # Stepping
///
/// ```ignore
/// let ast = quarks_validator::parse(src)?;
/// let mut session = quarks_interpreter::Session::new(&ast)?;
/// let mut ctx = quarks_interpreter::NullPolicyContext::new();
/// loop {
///     match session.step(&mut ctx)? {
///         quarks_interpreter::StepOutcome::Continue => continue,
///         quarks_interpreter::StepOutcome::Done(v) => break v,
///     }
/// }
/// ```
///
/// Equivalent to [`crate::interpret`] / [`crate::interpret_with_context`]
/// when stepped to completion; the per-step API additionally exposes
/// [`Session::instructions_executed`] for watchdog integration.
pub struct Session {
    symbols: SymbolTable,
    machine: Machine,
}

impl Session {
    /// Build a session over `ast`. Runs Pass 1
    /// ([`collect_signatures`]) to populate the function symbol
    /// table; the machine is seeded with the program entry point
    /// (see [`find_program_entry`]).
    ///
    /// The session owns the symbol table and the (cloned) entry
    /// subtree; the caller's `&SExpr` may go out of scope as soon as
    /// this call returns.
    pub fn new(ast: &SExpr) -> Result<Self, InterpretError> {
        let symbols = collect_signatures(ast)?;
        let entry = find_program_entry(ast);
        let machine = Machine::new(entry);
        Ok(Self { symbols, machine })
    }

    /// Execute exactly one [`Instr`]. See [`Machine::step`].
    pub fn step<C: PolicyContext + ?Sized>(
        &mut self,
        ctx: &mut C,
    ) -> Result<StepOutcome, InterpretError> {
        self.machine.step(&self.symbols, ctx)
    }

    /// Drive the machine to completion. See [`Machine::run`].
    pub fn run<C: PolicyContext + ?Sized>(&mut self, ctx: &mut C) -> Result<Value, InterpretError> {
        self.machine.run(&self.symbols, ctx)
    }

    /// Has the instruction stack drained?
    pub fn is_done(&self) -> bool {
        self.machine.is_done()
    }

    /// Monotonic count of dispatched [`Instr`]s.
    pub fn instructions_executed(&self) -> u64 {
        self.machine.instructions_executed()
    }

    /// Current function-call depth.
    pub fn call_depth(&self) -> usize {
        self.machine.call_depth()
    }
}
