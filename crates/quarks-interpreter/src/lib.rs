// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::while_let_loop,
    clippy::useless_conversion,
    clippy::assertions_on_constants
)]
//! Quarks Interpreter — explicit state-machine driver.
//!
//! Executes validated S-Expression IR in Ring-0. The interpreter
//! sits behind two entry points:
//!
//! - [`interpret`] / [`interpret_with_context`] — program-level
//!   convenience wrappers (build a [`Session`], drive it to
//!   completion, return the final [`Value`]).
//! - [`Session`] — instruction-level stepping API for cooperative
//!   schedulers and watchdogs. Each [`Session::step`] dispatches
//!   exactly one atomic operation and returns either
//!   [`StepOutcome::Continue`] or [`StepOutcome::Done`].
//!
//! Stage 12 Paket A (state-machine refactor) replaced the original
//! recursive tree-walker with the explicit machine in [`machine`];
//! the determinism contract is documented in
//! `docs/discovery/hardware-abstraction-constraints.md` §1 and the
//! refactor goals in `docs/discovery/stage-12-completion-plan.md`
//! §2A.
//!
//! # Pipeline
//!
//! ```ignore
//! let ir_text: &str = "(program (fn main () i64 (add 1 2)) (call main))";
//! let ast = quarks_validator::parse(ir_text)?;
//! quarks_validator::type_check(&ast)?;
//! let result = quarks_interpreter::interpret(&ast)?;
//! // result == Value::Integer(3)
//! ```

#![no_std]

extern crate alloc;

use quarks_validator::SExpr;

mod error;
mod frame;
mod machine;
mod policy;
mod symbol_table;
mod value;

pub use error::{InterpretError, InterpretErrorKind};
pub use machine::{Session, StepOutcome, MAX_INSTR_STACK_DEPTH};
pub use policy::{NullPolicyContext, PolicyContext, PolicyError};
pub use value::Value;

/// Interpret an S-Expression IR tree to completion.
///
/// Pipeline:
/// 1. Pass 1: `collect_signatures` walks the AST and builds a
///    `SymbolTable` of all `(fn ...)` definitions.
/// 2. Pass 2: a [`Session`] is built and driven to completion. For
///    bare expressions, the expression itself is the entry point;
///    for `(program ...)` wrappers, the entry point is the last
///    child (typically a `(call main)`).
///
/// Programs that use `(policy ...)` or `(query ...)` instructions
/// (12i) must use [`interpret_with_context`] instead — this entry
/// point uses a [`NullPolicyContext`] that rejects every
/// policy/query dispatch with
/// [`InterpretErrorKind::PolicyNotSupported`].
pub fn interpret(ast: &SExpr) -> Result<Value, InterpretError> {
    let mut ctx = NullPolicyContext::new();
    interpret_with_context(ast, &mut ctx)
}

/// 12i — interpret with a caller-supplied [`PolicyContext`] for the
/// `(policy ...)` and `(query ...)` first-class instructions. The
/// context routes capability checks (ADR-019 §5) and the
/// Hardware Capability Service hand-off (ADR-019 §4) for the calling
/// sandbox.
///
/// Programs that do not use policy/query instructions behave
/// identically to [`interpret`].
pub fn interpret_with_context<C: PolicyContext + ?Sized>(
    ast: &SExpr,
    ctx: &mut C,
) -> Result<Value, InterpretError> {
    let mut session = Session::new(ast)?;
    session.run(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quarks_validator::{parse, Atom};

    fn run(src: &str) -> Result<Value, InterpretError> {
        let ast = parse(src).expect("parse failed");
        interpret(&ast)
    }

    // === Stage 9 MP5 Tests (preserved) ===

    #[test]
    fn integer_atom() {
        assert_eq!(run("42").unwrap(), Value::Integer(42));
    }

    #[test]
    fn add() {
        assert_eq!(run("(add 1 2)").unwrap(), Value::Integer(3));
    }

    #[test]
    fn sub() {
        assert_eq!(run("(sub 5 3)").unwrap(), Value::Integer(2));
    }

    #[test]
    fn mul() {
        assert_eq!(run("(mul 4 6)").unwrap(), Value::Integer(24));
    }

    #[test]
    fn nested() {
        assert_eq!(run("(add (mul 2 3) 4)").unwrap(), Value::Integer(10));
    }

    #[test]
    fn unsupported_instruction() {
        // Pre-A.2 (F-C2): `div` is now implemented; pick a sentinel
        // that the validator-+-interpreter pair still does not know.
        // `mod` is in neither registry (Paket B will add it together
        // with the comparison/control-flow surface).
        let result = run("(mod 6 2)");
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::UnsupportedInstruction,
                ..
            })
        ));
    }

    // === Pre-A.2 (F-C2): div Validator-/Interpreter-Convergence ===

    #[test]
    fn div_basic_integer_division() {
        assert_eq!(run("(div 10 2)").unwrap(), Value::Integer(5));
        assert_eq!(run("(div 7 2)").unwrap(), Value::Integer(3)); // i64 trunc
        assert_eq!(run("(div -7 2)").unwrap(), Value::Integer(-3)); // toward zero
    }

    #[test]
    fn div_by_zero_returns_division_by_zero_error() {
        let result = run("(div 6 0)");
        assert!(
            matches!(
                result,
                Err(InterpretError {
                    kind: InterpretErrorKind::DivisionByZero,
                    ..
                })
            ),
            "expected DivisionByZero, got {:?}",
            result
        );
    }

    #[test]
    fn div_overflow_returns_division_by_zero_error() {
        // i64::MIN / -1 overflows in two's-complement arithmetic.
        // `i64::checked_div` returns None for both b == 0 and this
        // overflow; we surface both as DivisionByZero (the variant's
        // doc-comment documents the dual semantics).
        let src = "(div -9223372036854775808 -1)";
        let result = run(src);
        assert!(
            matches!(
                result,
                Err(InterpretError {
                    kind: InterpretErrorKind::DivisionByZero,
                    ..
                })
            ),
            "expected DivisionByZero on i64::MIN / -1, got {:?}",
            result
        );
    }

    #[test]
    fn div_nested_with_arithmetic() {
        // (div (add 4 6) (sub 5 3)) == 10 / 2 == 5
        assert_eq!(run("(div (add 4 6) (sub 5 3))").unwrap(), Value::Integer(5));
    }

    // === Pre-A.1 (F-C1): MAX_INSTR_STACK_DEPTH ===
    //
    // The state machine eliminates the native-stack-overflow vector
    // the original recursive walker had at depth 256. The remaining
    // hazard is unbounded heap growth in the instruction stack; the
    // `MAX_INSTR_STACK_DEPTH` guard caps it.
    //
    // Note: we construct the AST iteratively, bypassing the parser.
    // The parser is itself recursive and would stack-overflow on
    // deeply nested input — a separate hardening task (out of scope
    // here). We also drop the AST iteratively, because `Vec<SExpr>`
    // drop on a deeply linked list is recursive and would overflow.

    fn build_nested_add_ast(depth: usize) -> SExpr {
        // Build `(add 1 (add 1 (add 1 ... 1)))` from the inside out.
        let mut inner = SExpr::Atom(Atom::Integer(1));
        for _ in 0..depth {
            inner = SExpr::List(alloc::vec![
                SExpr::Atom(Atom::Symbol("add".into())),
                SExpr::Atom(Atom::Integer(1)),
                inner,
            ]);
        }
        inner
    }

    /// Iteratively drop a left-spine-like nested SExpr so the test
    /// teardown does not recurse and overflow.
    fn drop_nested_ast_iteratively(mut node: SExpr) {
        loop {
            let next_inner = if let SExpr::List(ref mut items) = node {
                if items.len() == 3 {
                    let inner = core::mem::replace(&mut items[2], SExpr::Atom(Atom::Integer(0)));
                    Some(inner)
                } else {
                    None
                }
            } else {
                None
            };
            drop(node);
            match next_inner {
                Some(n) => node = n,
                None => return,
            }
        }
    }

    #[test]
    fn shallow_nesting_succeeds() {
        // 100 levels — well below the instr-stack ceiling. The AST
        // has exactly 100 `(add 1 ...)` wrappers around a `1` leaf,
        // so the result is `1 + 100 == 101`.
        let ast = build_nested_add_ast(100);
        let result = interpret(&ast);
        drop_nested_ast_iteratively(ast);
        assert_eq!(result.unwrap(), Value::Integer(101));
    }

    #[test]
    fn deep_nesting_returns_expression_depth_exceeded() {
        // MAX_INSTR_STACK_DEPTH = 1024. With 2000 levels of nested
        // `(add 1 ...)` the instr stack will exceed the ceiling
        // before the program completes; the machine surfaces a
        // clean `ExpressionDepthExceeded` rather than running out of
        // memory or panicking. (2000 is the same depth the sandbox-
        // level test in `pre_a2_hardening.rs` uses; matching them
        // keeps the two guards observably aligned.)
        let ast = build_nested_add_ast(2000);
        let result = interpret(&ast);
        drop_nested_ast_iteratively(ast);
        assert!(
            matches!(
                result,
                Err(InterpretError {
                    kind: InterpretErrorKind::ExpressionDepthExceeded,
                    ..
                })
            ),
            "expected ExpressionDepthExceeded at depth 2000, got {:?}",
            result
        );
    }

    #[test]
    fn max_instr_stack_depth_in_sensible_range() {
        assert!(MAX_INSTR_STACK_DEPTH >= 64);
        assert!(MAX_INSTR_STACK_DEPTH <= 65_536);
    }

    // === Phase-2-Erweiterung MP1 Tests ===

    #[test]
    fn program_with_simple_main() {
        let src = "(program (fn main () i64 (add 1 2)) (call main))";
        assert_eq!(run(src).unwrap(), Value::Integer(3));
    }

    #[test]
    fn function_with_one_parameter() {
        let src = "(program (fn add1 (i64) i64 (add %0 1)) (call add1 5))";
        assert_eq!(run(src).unwrap(), Value::Integer(6));
    }

    #[test]
    fn function_with_multiple_parameters() {
        let src = "(program (fn add (i64 i64) i64 (add %0 %1)) (call add 10 20))";
        assert_eq!(run(src).unwrap(), Value::Integer(30));
    }

    #[test]
    fn nested_call_in_argument() {
        let src = "(program (fn double (i64) i64 (mul %0 2)) (call double (add 1 2)))";
        assert_eq!(run(src).unwrap(), Value::Integer(6));
    }

    #[test]
    fn mutual_recursion_simple() {
        // ping(0) returns 0
        // ping(n) calls pong(n-1), pong(n) calls ping(n-1)
        // Test: ping(0) — base case only, no actual recursion needed
        let src = "(program \
            (fn ping (i64) i64 %0) \
            (fn pong (i64) i64 (call ping %0)) \
            (call pong 42))";
        assert_eq!(run(src).unwrap(), Value::Integer(42));
    }

    #[test]
    fn function_not_found() {
        let src = "(program (fn foo () i64 0) (call bar))";
        let result = run(src);
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::FunctionNotFound,
                ..
            })
        ));
    }

    #[test]
    fn arity_mismatch_on_call() {
        let src = "(program (fn add1 (i64) i64 (add %0 1)) (call add1 1 2))";
        let result = run(src);
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::ArityMismatch,
                ..
            })
        ));
    }

    #[test]
    fn recursion_limit_enforced() {
        // infinite_recurse calls itself with the same arg → unbounded recursion
        let src = "(program \
            (fn loop_forever (i64) i64 (call loop_forever %0)) \
            (call loop_forever 0))";
        let result = run(src);
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::RecursionLimitExceeded,
                ..
            })
        ));
    }

    #[test]
    fn parameter_outside_function() {
        let src = "%0"; // bare parameter reference, no frame
        let result = run(src);
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::ParameterOutsideFunction,
                ..
            })
        ));
    }

    // === Stage 10 MP5 Handle Tests ===

    #[test]
    fn handle_atom_returns_value_handle() {
        assert_eq!(run("@5").unwrap(), Value::Handle(5));
    }

    #[test]
    fn handle_atom_large_id() {
        assert_eq!(run("@999").unwrap(), Value::Handle(999));
    }

    #[test]
    fn handle_as_function_parameter() {
        let src = "(program (fn echo (handle) handle %0) (call echo @7))";
        assert_eq!(run(src).unwrap(), Value::Handle(7));
    }

    #[test]
    fn handle_in_add_is_type_error() {
        let result = run("(add @5 1)");
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::UnsupportedAtom,
                ..
            })
        ));
    }

    // === Stage 10 MP5 fix: register + bytes ===

    #[test]
    fn bytes_atom_returns_value_bytes() {
        let result = run("#x4142").unwrap();
        assert_eq!(result, Value::Bytes(alloc::vec![0x41, 0x42]));
    }

    #[test]
    fn register_with_bytes_returns_handle() {
        // Verify register with bytes produces a Handle value.
        // We check the variant type (Handle) rather than a specific ID
        // because the static counter is shared across parallel tests.
        let result = run("(register #x4142)").unwrap();
        match result {
            Value::Handle(id) => assert!(
                id >= 1,
                "handle ID must be >= 1 (null sentinel), got {}",
                id
            ),
            other => panic!("expected Value::Handle, got {:?}", other),
        }
    }

    #[test]
    fn register_returns_fixed_mock_handle_id() {
        // Pre-A.2 (F-H14): the interpreter-side `(register …)` is a
        // signature-test mock until Paket A's per-sandbox HandleTable
        // arrives. The mock returns a fixed id (`1`) instead of an
        // AtomicU64 counter so two `interpret` calls in the same
        // process — and across parallel tests — produce the same
        // value. Determinism is the ADR-019 §1.5 contract.
        let r1 = run("(register #x41)").unwrap();
        let r2 = run("(register #x42)").unwrap();
        assert_eq!(r1, Value::Handle(1));
        assert_eq!(r2, Value::Handle(1));
    }

    #[test]
    fn register_with_integer_is_type_error() {
        let result = run("(register 42)");
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::UnsupportedAtom,
                ..
            })
        ));
    }

    #[test]
    fn register_with_handle_is_type_error() {
        let result = run("(register @5)");
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::UnsupportedAtom,
                ..
            })
        ));
    }

    #[test]
    fn register_arity_zero_is_error() {
        let result = run("(register)");
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::ArityMismatch,
                ..
            })
        ));
    }

    // === 12i: Policy / Query interpreter dispatch ===

    /// Recording context: captures every (subsystem, operation, args)
    /// triple and lets us script the return value via stub closures.
    /// Used in interpreter-side tests below to exercise the dispatch
    /// path without depending on `zero-sandbox`.
    struct RecordingPolicyCtx {
        policy_log: alloc::vec::Vec<(alloc::string::String, alloc::string::String, usize)>,
        query_log: alloc::vec::Vec<(alloc::string::String, alloc::string::String, usize)>,
        policy_return: i64,
        query_return: i64,
        policy_err: Option<PolicyError>,
        query_err: Option<PolicyError>,
    }

    impl RecordingPolicyCtx {
        fn new() -> Self {
            Self {
                policy_log: alloc::vec::Vec::new(),
                query_log: alloc::vec::Vec::new(),
                policy_return: 0,
                query_return: 0,
                policy_err: None,
                query_err: None,
            }
        }
    }

    impl PolicyContext for RecordingPolicyCtx {
        fn policy(
            &mut self,
            subsystem: &str,
            operation: &str,
            args: &[Value],
        ) -> Result<i64, PolicyError> {
            use alloc::string::ToString;
            self.policy_log
                .push((subsystem.to_string(), operation.to_string(), args.len()));
            if let Some(e) = self.policy_err.clone() {
                Err(e)
            } else {
                Ok(self.policy_return)
            }
        }

        fn query(
            &mut self,
            subsystem: &str,
            metric: &str,
            args: &[Value],
        ) -> Result<i64, PolicyError> {
            use alloc::string::ToString;
            self.query_log
                .push((subsystem.to_string(), metric.to_string(), args.len()));
            if let Some(e) = self.query_err.clone() {
                Err(e)
            } else {
                Ok(self.query_return)
            }
        }
    }

    fn run_with_ctx<C: PolicyContext>(src: &str, ctx: &mut C) -> Result<Value, InterpretError> {
        let ast = parse(src).expect("parse failed");
        interpret_with_context(&ast, ctx)
    }

    #[test]
    fn policy_dispatches_into_context_with_evaluated_args() {
        let mut ctx = RecordingPolicyCtx::new();
        ctx.policy_return = 0;
        let v = run_with_ctx("(policy gpu allocate-slice 5 50)", &mut ctx).unwrap();
        assert_eq!(v, Value::Integer(0));
        assert_eq!(ctx.policy_log.len(), 1);
        assert_eq!(ctx.policy_log[0].0, "gpu");
        assert_eq!(ctx.policy_log[0].1, "allocate-slice");
        assert_eq!(ctx.policy_log[0].2, 2); // <sandbox-id> <percentage>
    }

    #[test]
    fn query_dispatches_into_context_returning_telemetry() {
        let mut ctx = RecordingPolicyCtx::new();
        ctx.query_return = 73;
        let v = run_with_ctx("(query gpu utilization)", &mut ctx).unwrap();
        assert_eq!(v, Value::Integer(73));
        assert_eq!(ctx.query_log.len(), 1);
        assert_eq!(ctx.query_log[0].0, "gpu");
        assert_eq!(ctx.query_log[0].1, "utilization");
        assert_eq!(ctx.query_log[0].2, 0);
    }

    #[test]
    fn policy_evaluates_nested_arithmetic_args() {
        let mut ctx = RecordingPolicyCtx::new();
        // 12i: nested arithmetic must be evaluated to scalar before
        // dispatch. The context observes `args.len()`; the values are
        // discarded by the recording stub.
        let v = run_with_ctx(
            "(policy schedule set-time-slice (add 1 2) (mul 4 5))",
            &mut ctx,
        )
        .unwrap();
        assert_eq!(v, Value::Integer(0));
        assert_eq!(ctx.policy_log[0].0, "schedule");
        assert_eq!(ctx.policy_log[0].1, "set-time-slice");
        assert_eq!(ctx.policy_log[0].2, 2);
    }

    #[test]
    fn policy_with_null_context_returns_not_supported() {
        let result = run("(policy gpu allocate-slice 1 25)");
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::PolicyNotSupported,
                ..
            })
        ));
    }

    #[test]
    fn query_with_null_context_returns_not_supported() {
        let result = run("(query thermal state)");
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::PolicyNotSupported,
                ..
            })
        ));
    }

    #[test]
    fn policy_missing_args_is_malformed() {
        let mut ctx = RecordingPolicyCtx::new();
        let result = run_with_ctx("(policy gpu)", &mut ctx);
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::PolicyMalformed,
                ..
            })
        ));
        assert!(ctx.policy_log.is_empty());
    }

    #[test]
    fn policy_subsystem_must_be_symbol() {
        let mut ctx = RecordingPolicyCtx::new();
        let result = run_with_ctx("(policy 1 set-priority)", &mut ctx);
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::PolicyMalformed,
                ..
            })
        ));
    }

    #[test]
    fn policy_permission_denied_propagates_as_dispatch_failed() {
        let mut ctx = RecordingPolicyCtx::new();
        ctx.policy_err = Some(PolicyError::PermissionDenied);
        let result = run_with_ctx("(policy network set-bandwidth 1 1000)", &mut ctx);
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::PolicyDispatchFailed,
                ..
            })
        ));
    }

    #[test]
    fn query_unknown_subsystem_propagates_as_dispatch_failed() {
        let mut ctx = RecordingPolicyCtx::new();
        ctx.query_err = Some(PolicyError::UnknownSubsystem);
        let result = run_with_ctx("(query power state)", &mut ctx);
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::PolicyDispatchFailed,
                ..
            })
        ));
    }

    #[test]
    fn add_subtree_does_not_invoke_context() {
        let mut ctx = RecordingPolicyCtx::new();
        let v = run_with_ctx("(add (mul 2 3) 4)", &mut ctx).unwrap();
        assert_eq!(v, Value::Integer(10));
        assert!(ctx.policy_log.is_empty());
        assert!(ctx.query_log.is_empty());
    }

    // === Stage 12 Paket A: state-machine refactor tests ===
    //
    // These tests target the explicit state-machine surface
    // ([`Session`], [`StepOutcome`]) introduced by the Paket A
    // refactor of `quarks-interpreter`. They live next to the
    // existing tests so the equivalence guarantees travel with the
    // recursive-walker test surface.

    /// Drive a session step-by-step, asserting that we get exactly
    /// `expected_instructions` `Continue` outcomes followed by a
    /// `Done`. Returns the final value plus the actual instruction
    /// count, so call-sites can assert ranges or exact counts.
    fn step_until_done(src: &str) -> (Value, u64) {
        let ast = parse(src).expect("parse failed");
        let mut session = Session::new(&ast).expect("session::new failed");
        let mut ctx = NullPolicyContext::new();
        let mut steps = 0u64;
        loop {
            match session.step(&mut ctx).expect("step error") {
                StepOutcome::Continue => steps += 1,
                StepOutcome::Done(v) => return (v, steps),
            }
        }
    }

    #[test]
    fn session_integer_atom_produces_done() {
        let (v, _) = step_until_done("42");
        assert_eq!(v, Value::Integer(42));
    }

    #[test]
    fn session_add_steps_match_recursive() {
        let (v, steps) = step_until_done("(add 1 2)");
        assert_eq!(v, Value::Integer(3));
        // Expect: Eval((add 1 2)) → Eval(1) → BinOpAfterLhs → Eval(2)
        // → BinOpAfterRhs.  Five dispatched instructions in total.
        assert_eq!(steps, 5);
    }

    #[test]
    fn session_nested_steps_more_than_flat() {
        let (_, flat) = step_until_done("(add 1 2)");
        let (_, nested) = step_until_done("(add (mul 2 3) 4)");
        assert!(
            nested > flat,
            "nested ({}) should require more steps than flat ({})",
            nested,
            flat
        );
    }

    #[test]
    fn session_call_with_args_runs_to_done() {
        let (v, _) = step_until_done("(program (fn add1 (i64) i64 (add %0 1)) (call add1 5))");
        assert_eq!(v, Value::Integer(6));
    }

    #[test]
    fn session_is_done_only_after_final_step() {
        let ast = parse("42").expect("parse failed");
        let mut session = Session::new(&ast).expect("session::new failed");
        let mut ctx = NullPolicyContext::new();
        assert!(!session.is_done());
        // Step 1 evaluates the atom, leaving instr_stack empty.
        match session.step(&mut ctx).unwrap() {
            StepOutcome::Continue => {}
            other => panic!("expected Continue, got {:?}", other),
        }
        assert!(session.is_done(), "instr_stack should be drained");
        // Step 2 pops the final value off the value stack.
        match session.step(&mut ctx).unwrap() {
            StepOutcome::Done(v) => assert_eq!(v, Value::Integer(42)),
            other => panic!("expected Done, got {:?}", other),
        }
    }

    #[test]
    fn session_instructions_executed_monotonic() {
        let ast = parse("(add (mul 2 3) 4)").expect("parse failed");
        let mut session = Session::new(&ast).expect("session::new failed");
        let mut ctx = NullPolicyContext::new();
        let mut prev = 0u64;
        loop {
            let count_before = session.instructions_executed();
            match session.step(&mut ctx).unwrap() {
                StepOutcome::Continue => {
                    let count_after = session.instructions_executed();
                    assert!(
                        count_after > count_before,
                        "instructions_executed must increase per step ({} → {})",
                        count_before,
                        count_after
                    );
                    assert!(count_after >= prev, "monotonic invariant violated");
                    prev = count_after;
                }
                StepOutcome::Done(_) => return,
            }
        }
    }

    #[test]
    fn session_run_matches_interpret_for_arithmetic() {
        // Cross-check: Session::run(P) ≡ interpret(P) for a range of
        // arithmetic programs. This is the core bitwise-equivalence
        // property of the Paket A refactor.
        let cases = &[
            "0",
            "42",
            "-7",
            "(add 1 2)",
            "(sub 5 3)",
            "(mul 4 6)",
            "(add (mul 2 3) (sub 10 4))",
            "(mul (add 1 1) (add 2 2))",
        ];
        for src in cases {
            let recursive = run(src).unwrap();
            let (machine, _) = step_until_done(src);
            assert_eq!(recursive, machine, "mismatch for {}", src);
        }
    }

    #[test]
    fn session_run_matches_interpret_for_calls() {
        let cases: &[&str] = &[
            "(program (fn main () i64 42) (call main))",
            "(program (fn id (i64) i64 %0) (call id 7))",
            "(program (fn add (i64 i64) i64 (add %0 %1)) (call add 3 4))",
            "(program \
                (fn dbl (i64) i64 (mul %0 2)) \
                (fn quad (i64) i64 (call dbl (call dbl %0))) \
                (call quad 5))",
        ];
        for src in cases {
            let recursive = run(src).unwrap();
            let (machine, _) = step_until_done(src);
            assert_eq!(recursive, machine, "mismatch for {}", src);
        }
    }

    #[test]
    fn session_recursion_limit_via_machine() {
        // Same program as `recursion_limit_enforced`, but stepped
        // through the explicit machine. Error must surface with the
        // same kind.
        let src = "(program \
            (fn loop_forever (i64) i64 (call loop_forever %0)) \
            (call loop_forever 0))";
        let ast = parse(src).expect("parse failed");
        let mut session = Session::new(&ast).expect("session::new failed");
        let mut ctx = NullPolicyContext::new();
        let result = session.run(&mut ctx);
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::RecursionLimitExceeded,
                ..
            })
        ));
    }

    #[test]
    fn session_pause_and_resume_equivalent_to_full_run() {
        // Take an arbitrary intermediate number of steps via
        // step(), then call run() to finish — the final value
        // must equal a fresh full-run via Session::run from start.
        let src = "(program (fn add (i64 i64) i64 (add %0 %1)) (call add 10 20))";
        let ast = parse(src).expect("parse failed");

        // Reference run.
        let mut ref_session = Session::new(&ast).expect("session::new failed");
        let mut ref_ctx = NullPolicyContext::new();
        let reference = ref_session.run(&mut ref_ctx).unwrap();
        assert_eq!(reference, Value::Integer(30));

        // Pause-and-resume run: 3 steps via step(), then run() to
        // finish. Must produce the same value.
        let mut session = Session::new(&ast).expect("session::new failed");
        let mut ctx = NullPolicyContext::new();
        for _ in 0..3 {
            match session.step(&mut ctx).unwrap() {
                StepOutcome::Continue => {}
                StepOutcome::Done(_) => {
                    panic!("3 steps was enough to complete; program is too small")
                }
            }
        }
        let resumed = session.run(&mut ctx).unwrap();
        assert_eq!(resumed, reference);
    }

    #[test]
    fn session_step_by_step_equivalent_to_run_for_many_programs() {
        // Property-style: for each program, asserting that
        // Session::run(P) equals stepping all the way using
        // step() + Continue/Done pumps. Run on a diverse cohort.
        let cases: &[&str] = &[
            "0",
            "(add 1 2)",
            "(mul 2 (add 3 4))",
            "(sub 10 (mul 2 3))",
            "(program (fn main () i64 (add 1 2)) (call main))",
            "(program (fn id (i64) i64 %0) (call id (mul 7 6)))",
            "(program \
                (fn f (i64) i64 (add %0 1)) \
                (fn g (i64) i64 (mul %0 2)) \
                (call f (call g 21)))",
        ];
        for src in cases {
            let ast = parse(src).expect("parse failed");

            let mut a = Session::new(&ast).unwrap();
            let mut actx = NullPolicyContext::new();
            let by_run = a.run(&mut actx).unwrap();

            let mut b = Session::new(&ast).unwrap();
            let mut bctx = NullPolicyContext::new();
            let by_step = loop {
                match b.step(&mut bctx).unwrap() {
                    StepOutcome::Continue => continue,
                    StepOutcome::Done(v) => break v,
                }
            };

            assert_eq!(by_run, by_step, "mismatch for {}", src);
        }
    }

    #[test]
    fn session_determinism_two_runs_identical_value() {
        // Determinism contract: same program, fresh sessions, two
        // separate runs must produce the same Value. (Hardware
        // counters are explicitly *not* part of the determinism
        // contract — they are observably monotonic only.)
        let src = "(program \
            (fn add (i64 i64) i64 (add %0 %1)) \
            (call add (add 1 2) (mul 3 4)))";
        let ast = parse(src).expect("parse failed");
        let mut s1 = Session::new(&ast).unwrap();
        let mut c1 = NullPolicyContext::new();
        let v1 = s1.run(&mut c1).unwrap();
        let mut s2 = Session::new(&ast).unwrap();
        let mut c2 = NullPolicyContext::new();
        let v2 = s2.run(&mut c2).unwrap();
        assert_eq!(v1, v2);
        assert_eq!(v1, Value::Integer(15));
    }

    #[test]
    fn session_call_depth_grows_and_shrinks() {
        // Walk a small program through step() and observe the
        // call_depth invariant: it grows when InvokeCall fires and
        // shrinks back to zero before Done.
        let src = "(program (fn id (i64) i64 %0) (call id 7))";
        let ast = parse(src).expect("parse failed");
        let mut session = Session::new(&ast).unwrap();
        let mut ctx = NullPolicyContext::new();
        let mut peak_depth = 0usize;
        loop {
            match session.step(&mut ctx).unwrap() {
                StepOutcome::Continue => {
                    peak_depth = peak_depth.max(session.call_depth());
                }
                StepOutcome::Done(_) => break,
            }
        }
        assert!(peak_depth >= 1, "must enter at least one frame");
        // At Done time, frame has been popped — depth must be 0.
        assert_eq!(session.call_depth(), 0);
    }

    #[test]
    fn session_policy_dispatches_via_step() {
        // Driving (policy ...) through the explicit machine must
        // dispatch into the context exactly once, exactly as the
        // run() / interpret_with_context() path does.
        let src = "(policy gpu allocate-slice 5 50)";
        let ast = parse(src).expect("parse failed");

        let mut ctx = RecordingPolicyCtx::new();
        ctx.policy_return = 0;
        let mut session = Session::new(&ast).unwrap();
        let v = loop {
            match session.step(&mut ctx).unwrap() {
                StepOutcome::Continue => continue,
                StepOutcome::Done(v) => break v,
            }
        };
        assert_eq!(v, Value::Integer(0));
        assert_eq!(ctx.policy_log.len(), 1);
        assert_eq!(ctx.policy_log[0].0, "gpu");
        assert_eq!(ctx.policy_log[0].1, "allocate-slice");
        assert_eq!(ctx.policy_log[0].2, 2);
    }

    #[test]
    fn session_empty_list_errors_at_dispatch() {
        let ast = parse("()").expect("parse failed");
        let mut session = Session::new(&ast).unwrap();
        let mut ctx = NullPolicyContext::new();
        let result = session.run(&mut ctx);
        assert!(matches!(
            result,
            Err(InterpretError {
                kind: InterpretErrorKind::EmptyList,
                ..
            })
        ));
    }
}
