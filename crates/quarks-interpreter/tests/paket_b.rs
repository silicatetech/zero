// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stage 12 Paket B — Quarks Spracherweiterung end-to-end tests.
//!
//! These tests exercise the validator+interpreter pair for every
//! Paket B language feature so that the executed value matches the
//! validator's typed expectation. They live in `tests/` (integration
//! tier) so the same path a kernel-LLM-generated policy would take —
//! parse → validate → type-check → interpret — is covered.
//!
//! References:
//! - `docs/discovery/stage-12-completion-plan.md` §B (language scope)
//! - ADR-019 §5 (Validator-monotone typing rules)

use quarks_interpreter::{interpret, InterpretError, InterpretErrorKind, Value};
use quarks_validator::{parse, type_check, validate_structure, ValueType};

fn run(src: &str) -> Result<Value, InterpretError> {
    let ast = parse(src).expect("parse");
    validate_structure(&ast).expect("validator structural");
    type_check(&ast).expect("validator type-check");
    interpret(&ast)
}

fn type_stack(src: &str) -> Vec<ValueType> {
    let ast = parse(src).expect("parse");
    validate_structure(&ast).expect("validator structural");
    type_check(&ast).expect("type-check")
}

// ── Bool literals ──────────────────────────────────────────────

#[test]
fn b2_bool_true_literal() {
    assert_eq!(run("true").unwrap(), Value::Bool(true));
    assert_eq!(type_stack("true"), vec![ValueType::Bool]);
}

#[test]
fn b2_bool_false_literal() {
    assert_eq!(run("false").unwrap(), Value::Bool(false));
    assert_eq!(type_stack("false"), vec![ValueType::Bool]);
}

// ── Comparison operators (Paket B.1) ───────────────────────────

#[test]
fn b1_eq_true_when_equal() {
    assert_eq!(run("(eq 5 5)").unwrap(), Value::Bool(true));
}

#[test]
fn b1_eq_false_when_unequal() {
    assert_eq!(run("(eq 5 6)").unwrap(), Value::Bool(false));
}

#[test]
fn b1_ne() {
    assert_eq!(run("(ne 5 5)").unwrap(), Value::Bool(false));
    assert_eq!(run("(ne 5 6)").unwrap(), Value::Bool(true));
}

#[test]
fn b1_lt() {
    assert_eq!(run("(lt 5 6)").unwrap(), Value::Bool(true));
    assert_eq!(run("(lt 6 5)").unwrap(), Value::Bool(false));
    assert_eq!(run("(lt 5 5)").unwrap(), Value::Bool(false));
}

#[test]
fn b1_gt() {
    assert_eq!(run("(gt 6 5)").unwrap(), Value::Bool(true));
    assert_eq!(run("(gt 5 6)").unwrap(), Value::Bool(false));
}

#[test]
fn b1_le() {
    assert_eq!(run("(le 5 5)").unwrap(), Value::Bool(true));
    assert_eq!(run("(le 5 6)").unwrap(), Value::Bool(true));
    assert_eq!(run("(le 6 5)").unwrap(), Value::Bool(false));
}

#[test]
fn b1_ge() {
    assert_eq!(run("(ge 5 5)").unwrap(), Value::Bool(true));
    assert_eq!(run("(ge 6 5)").unwrap(), Value::Bool(true));
    assert_eq!(run("(ge 5 6)").unwrap(), Value::Bool(false));
}

#[test]
fn b1_comparison_with_nested_arithmetic() {
    // (lt (add 1 2) (mul 2 3)) — 3 < 6 → true
    assert_eq!(run("(lt (add 1 2) (mul 2 3))").unwrap(), Value::Bool(true));
}

#[test]
fn b1_comparison_with_negative_numbers() {
    assert_eq!(run("(lt -5 -3)").unwrap(), Value::Bool(true));
    assert_eq!(run("(gt -5 -3)").unwrap(), Value::Bool(false));
}

#[test]
fn b1_comparison_with_i64_extremes() {
    // Boundary values still compare without panic / overflow.
    let src = "(lt -9223372036854775808 9223372036854775807)";
    assert_eq!(run(src).unwrap(), Value::Bool(true));
}

// ── Logical operators (Paket B.2) ──────────────────────────────

#[test]
fn b2_and_truth_table() {
    assert_eq!(run("(and true true)").unwrap(), Value::Bool(true));
    assert_eq!(run("(and true false)").unwrap(), Value::Bool(false));
    assert_eq!(run("(and false true)").unwrap(), Value::Bool(false));
    assert_eq!(run("(and false false)").unwrap(), Value::Bool(false));
}

#[test]
fn b2_or_truth_table() {
    assert_eq!(run("(or true true)").unwrap(), Value::Bool(true));
    assert_eq!(run("(or true false)").unwrap(), Value::Bool(true));
    assert_eq!(run("(or false true)").unwrap(), Value::Bool(true));
    assert_eq!(run("(or false false)").unwrap(), Value::Bool(false));
}

#[test]
fn b2_not() {
    assert_eq!(run("(not true)").unwrap(), Value::Bool(false));
    assert_eq!(run("(not false)").unwrap(), Value::Bool(true));
}

#[test]
fn b2_logical_with_comparison() {
    // (and (lt 1 2) (gt 5 3)) → true && true → true
    assert_eq!(run("(and (lt 1 2) (gt 5 3))").unwrap(), Value::Bool(true));
    // (or (eq 1 2) (lt 1 2)) → false || true → true
    assert_eq!(run("(or (eq 1 2) (lt 1 2))").unwrap(), Value::Bool(true));
    // (not (lt 5 3)) → true
    assert_eq!(run("(not (lt 5 3))").unwrap(), Value::Bool(true));
}

// ── (if cond then else) ─ Paket B.1b ───────────────────────────

#[test]
fn b1b_if_picks_then_branch_when_cond_true() {
    assert_eq!(run("(if true 1 2)").unwrap(), Value::Integer(1));
}

#[test]
fn b1b_if_picks_else_branch_when_cond_false() {
    assert_eq!(run("(if false 1 2)").unwrap(), Value::Integer(2));
}

#[test]
fn b1b_if_with_computed_condition() {
    // (if (eq 1 1) 42 99) → 42
    assert_eq!(run("(if (eq 1 1) 42 99)").unwrap(), Value::Integer(42));
    // (if (lt 5 3) 42 99) → 99
    assert_eq!(run("(if (lt 5 3) 42 99)").unwrap(), Value::Integer(99));
}

#[test]
fn b1b_nested_if() {
    // (if (lt x y) (if (eq x 0) 100 200) 300) with x=0, y=1 → 100
    let src = "(program \
        (fn nested (i64 i64) i64 \
            (if (lt %0 %1) (if (eq %0 0) 100 200) 300)) \
        (call nested 0 1))";
    assert_eq!(run(src).unwrap(), Value::Integer(100));
    let src = "(program \
        (fn nested (i64 i64) i64 \
            (if (lt %0 %1) (if (eq %0 0) 100 200) 300)) \
        (call nested 1 5))";
    assert_eq!(run(src).unwrap(), Value::Integer(200));
}

#[test]
fn b1b_if_with_bool_branches() {
    // (if true true false) → true (Bool type)
    assert_eq!(run("(if true true false)").unwrap(), Value::Bool(true));
}

// ── (let %n value body) ─ Paket B.3 ────────────────────────────

#[test]
fn b3_let_binds_local() {
    // Function with one parameter; let binds %1 to 100 then returns %1.
    let src = "(program \
        (fn f (i64) i64 (let %1 100 %1)) \
        (call f 7))";
    assert_eq!(run(src).unwrap(), Value::Integer(100));
}

#[test]
fn b3_let_can_use_parameter() {
    // Let-binding uses parameter %0 in its value expression.
    let src = "(program \
        (fn f (i64) i64 (let %1 (add %0 1) %1)) \
        (call f 41))";
    assert_eq!(run(src).unwrap(), Value::Integer(42));
}

#[test]
fn b3_nested_let() {
    let src = "(program \
        (fn f (i64) i64 \
            (let %1 (mul %0 2) \
                (let %2 (add %1 1) \
                    (add %1 %2)))) \
        (call f 3))";
    // %0 = 3, %1 = 6, %2 = 7, result = 6 + 7 = 13
    assert_eq!(run(src).unwrap(), Value::Integer(13));
}

#[test]
fn b3_let_bool_value() {
    let src = "(program \
        (fn check (i64) bool (let %1 (gt %0 0) %1)) \
        (call check 5))";
    assert_eq!(run(src).unwrap(), Value::Bool(true));
}

// ── (loop body) + (break value) ─ Paket B.1c ───────────────────

#[test]
fn b1c_loop_with_immediate_break() {
    // (loop (break 42)) → 42
    assert_eq!(run("(loop (break 42))").unwrap(), Value::Integer(42));
}

#[test]
fn b1c_loop_with_break_bool() {
    // Loop output type tracks the break value's type.
    assert_eq!(run("(loop (break true))").unwrap(), Value::Bool(true));
}

#[test]
fn b1c_loop_with_conditional_break_in_function() {
    // Function-scope loop: countdown from n to 0 via repeated
    // self-call would be cleaner, but we test the loop+break
    // primitive directly. The simplest reachable break: always
    // break with a value derived from a parameter.
    let src = "(program \
        (fn make_zero () i64 (loop (break 0))) \
        (call make_zero))";
    assert_eq!(run(src).unwrap(), Value::Integer(0));
}

#[test]
fn b1c_break_with_computed_value() {
    // (loop (break (add 1 2))) → 3
    assert_eq!(run("(loop (break (add 1 2)))").unwrap(), Value::Integer(3));
}

#[test]
fn b1c_nested_loops_inner_break_first() {
    // Outer loop, inner loop breaks first → inner loop produces its
    // break value (Integer), outer body's residual = Integer, then
    // outer breaks with its own value.
    let src = "(loop (seq (discard (loop (break 7))) (break 99)))";
    assert_eq!(run(src).unwrap(), Value::Integer(99));
}

// ── (while cond body) ─ Paket B.1d ─────────────────────────────

#[test]
fn b1d_while_false_cond_exits_immediately_with_bool_false() {
    // cond=false → loop never enters body → produces Bool(false).
    assert_eq!(run("(while false 0)").unwrap(), Value::Bool(false));
}

#[test]
fn b1d_while_terminates_via_user_break() {
    // (while true (break false)) — body immediately breaks with
    // false. Loop produces Bool(false).
    assert_eq!(
        run("(while true (break false))").unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn b1d_while_counts_via_function_recursion_proxy() {
    // We can't mutate state in this surface yet; use a function +
    // (while (lt %0 5) (break false)) to demonstrate the cond is
    // honoured. With (lt 0 5) initially true, body breaks
    // immediately → loop output Bool(false).
    let src = "(program \
        (fn try (i64) bool (while (lt %0 5) (break false))) \
        (call try 0))";
    assert_eq!(run(src).unwrap(), Value::Bool(false));
}

#[test]
fn b1d_while_with_false_cond_in_function() {
    // (while (lt 5 0) ...) — cond starts false → exit immediately.
    let src = "(program \
        (fn try () bool (while (lt 5 0) (break false))) \
        (call try))";
    assert_eq!(run(src).unwrap(), Value::Bool(false));
}

// ── (seq …) and (discard …) ────────────────────────────────────

#[test]
fn seq_evaluates_in_order_and_takes_last() {
    // (seq (discard 1) (discard 2) 99) → 99
    assert_eq!(
        run("(seq (discard 1) (discard 2) 99)").unwrap(),
        Value::Integer(99)
    );
}

#[test]
fn discard_drops_a_value() {
    // (seq (discard (add 1 2)) 5) → 5
    assert_eq!(
        run("(seq (discard (add 1 2)) 5)").unwrap(),
        Value::Integer(5)
    );
}

#[test]
fn seq_with_arithmetic_tail() {
    // (seq (discard 1) (discard 2) (add 10 20)) → 30
    assert_eq!(
        run("(seq (discard 1) (discard 2) (add 10 20))").unwrap(),
        Value::Integer(30)
    );
}

// ── Validator-rejected programs (negative tests) ───────────────

#[test]
fn validator_rejects_if_with_i64_condition() {
    let ast = parse("(if 1 1 2)").expect("parse");
    let err = type_check(&ast).expect_err("type-check should fail (cond must be Bool)");
    // The error variant is TypeMismatch with expected=Bool, actual=I64.
    use quarks_validator::TypeCheckErrorKind;
    match err.kind {
        TypeCheckErrorKind::TypeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, ValueType::Bool);
            assert_eq!(actual, ValueType::I64);
        }
        other => panic!("expected TypeMismatch, got {:?}", other),
    }
}

#[test]
fn validator_rejects_break_outside_loop() {
    let ast = parse("(break false)").expect("parse");
    let err = type_check(&ast).expect_err("break outside loop");
    use quarks_validator::TypeCheckErrorKind;
    assert!(matches!(
        err.kind,
        TypeCheckErrorKind::BreakOutsideLoopError
    ));
}

#[test]
fn validator_rejects_break_type_disagreement() {
    let ast = parse("(loop (if true (break true) (break 0)))").expect("parse");
    let err = type_check(&ast).expect_err("break type disagreement");
    use quarks_validator::TypeCheckErrorKind;
    assert!(matches!(
        err.kind,
        TypeCheckErrorKind::BreakTypeMismatch { .. }
    ));
}

#[test]
fn validator_rejects_loop_without_reachable_break() {
    let ast = parse("(loop (add 1 2))").expect("parse");
    let err = type_check(&ast).expect_err("loop without break");
    use quarks_validator::TypeCheckErrorKind;
    assert_eq!(err.kind, TypeCheckErrorKind::LoopWithoutBreak);
}

#[test]
fn validator_rejects_and_with_i64_operand() {
    let ast = parse("(and 1 true)").expect("parse");
    let err = type_check(&ast).expect_err("and requires Bool");
    use quarks_validator::TypeCheckErrorKind;
    assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
}

// ── Determinism property: same program → same value, two runs ──

#[test]
fn determinism_two_runs_match() {
    let cases: &[&str] = &[
        "(eq 1 2)",
        "(if (lt 3 5) 1 2)",
        "(loop (break (add 1 2)))",
        "(and (or true false) (not false))",
        "(while false 0)",
    ];
    for src in cases {
        let v1 = run(src).unwrap();
        let v2 = run(src).unwrap();
        assert_eq!(v1, v2, "non-determinism in {}", src);
    }
}

#[test]
fn determinism_complex_program_repeatable() {
    let src = "(program \
        (fn predicate (i64) bool (and (gt %0 0) (lt %0 100))) \
        (if (call predicate 42) (loop (break 1)) (loop (break 0))))";
    let v1 = run(src).unwrap();
    let v2 = run(src).unwrap();
    let v3 = run(src).unwrap();
    assert_eq!(v1, v2);
    assert_eq!(v2, v3);
    assert_eq!(v1, Value::Integer(1));
}

// ── Mid-program preemption (Paket A.4 invariant carrier) ───────

#[test]
fn paket_b_works_with_session_stepping() {
    use quarks_interpreter::{NullPolicyContext, Session, StepOutcome};
    let ast = parse("(if (lt 1 2) (add 10 20) 0)").expect("parse");
    let mut sess = Session::new(&ast).expect("session");
    let mut ctx = NullPolicyContext::new();
    let result = loop {
        match sess.step(&mut ctx).expect("step") {
            StepOutcome::Continue => continue,
            StepOutcome::Done(v) => break v,
        }
    };
    assert_eq!(result, Value::Integer(30));
    // instructions_executed must be monotone positive.
    assert!(sess.instructions_executed() > 0);
}

// ── Ring-0 safety: bounded stacks under depth pressure ─────────

#[test]
fn deep_nested_if_does_not_blow_stack() {
    use quarks_validator::{Atom, SExpr};
    // Construct (if true (if true (if true ... 0) 0) 0) iteratively.
    let mut expr: SExpr = SExpr::Atom(Atom::Integer(0));
    for _ in 0..100 {
        expr = SExpr::List(alloc::vec::Vec::from([
            SExpr::Atom(Atom::Symbol("if".into())),
            SExpr::Atom(Atom::Symbol("true".into())),
            expr,
            SExpr::Atom(Atom::Integer(0)),
        ]));
    }
    // Type-checking is recursive in the validator; skip and go
    // straight to interpret to test the machine's bounded behavior.
    let result = interpret(&expr);
    // Either succeeds (depth 100 is small) or surfaces a typed
    // ExpressionDepthExceeded — never a panic.
    match result {
        Ok(Value::Integer(0)) => {}
        Err(InterpretError {
            kind: InterpretErrorKind::ExpressionDepthExceeded,
            ..
        }) => {}
        other => panic!("unexpected result: {:?}", other),
    }
}

extern crate alloc;
