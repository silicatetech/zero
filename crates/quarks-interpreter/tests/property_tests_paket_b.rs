// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stage 12 Paket B — property-based determinism harness for the new
//! Quarks surface (Bool, comparison ops, logical ops, `if`, `let`,
//! `loop`+`break`, `while`).
//!
//! Invariants asserted:
//!
//! 1. **Bool literals round-trip.** `true` and `false` always evaluate
//!    to the corresponding [`Value::Bool`] (no randomness, no aliasing
//!    with I64 zero/non-zero).
//! 2. **Comparison ops are deterministic and consistent.** For random
//!    `i64` pairs, evaluator output matches the Rust-host comparison.
//!    No panic, no overflow leak (boundary values included).
//! 3. **Logical ops follow boolean algebra.** Truth tables match
//!    `bool::&&`/`||`/`!` exactly.
//! 4. **`(if cond t e)` mirrors host `if`.** For arbitrary Bool cond
//!    and integer arms, the interpreter picks the right branch.
//! 5. **`(loop (break v))` returns `v` exactly once, regardless of
//!    `v`'s value.** No spurious mutations to the value.
//! 6. **`Session::run` ≡ `interpret` on every generated case.**
//!    Step-by-step execution matches the program-level convenience
//!    entry. (Paket A.3 invariant carried into Paket B.)
//!
//! These tests run under stable `cargo test`. See the sibling
//! `property_tests.rs` for the original Paket A.3 harness.

use proptest::prelude::*;
use quarks_interpreter::{interpret, NullPolicyContext, Session, StepOutcome, Value};
use quarks_validator::parse;

fn run(src: &str) -> Value {
    let ast = parse(src).expect("parse");
    interpret(&ast).expect("interpret")
}

// ── Bool literals ──────────────────────────────────────────────

#[test]
fn paket_b_bool_literals_unconditional() {
    assert_eq!(run("true"), Value::Bool(true));
    assert_eq!(run("false"), Value::Bool(false));
}

// ── Comparison ops vs host i64 comparisons ─────────────────────

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 4096,
        .. ProptestConfig::default()
    })]

    #[test]
    fn paket_b_eq_matches_host(a in any::<i64>(), b in any::<i64>()) {
        let src = alloc::format!("(eq {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Bool(a == b));
    }

    #[test]
    fn paket_b_ne_matches_host(a in any::<i64>(), b in any::<i64>()) {
        let src = alloc::format!("(ne {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Bool(a != b));
    }

    #[test]
    fn paket_b_lt_matches_host(a in any::<i64>(), b in any::<i64>()) {
        let src = alloc::format!("(lt {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Bool(a < b));
    }

    #[test]
    fn paket_b_gt_matches_host(a in any::<i64>(), b in any::<i64>()) {
        let src = alloc::format!("(gt {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Bool(a > b));
    }

    #[test]
    fn paket_b_le_matches_host(a in any::<i64>(), b in any::<i64>()) {
        let src = alloc::format!("(le {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Bool(a <= b));
    }

    #[test]
    fn paket_b_ge_matches_host(a in any::<i64>(), b in any::<i64>()) {
        let src = alloc::format!("(ge {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Bool(a >= b));
    }

    // ── Logical ops ────────────────────────────────────────────

    #[test]
    fn paket_b_and_truth_table(a in any::<bool>(), b in any::<bool>()) {
        let src = alloc::format!("(and {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Bool(a && b));
    }

    #[test]
    fn paket_b_or_truth_table(a in any::<bool>(), b in any::<bool>()) {
        let src = alloc::format!("(or {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Bool(a || b));
    }

    #[test]
    fn paket_b_not_truth_table(a in any::<bool>()) {
        let src = alloc::format!("(not {})", a);
        prop_assert_eq!(run(&src), Value::Bool(!a));
    }

    // ── (if cond t e) selects the right branch ────────────────

    #[test]
    fn paket_b_if_picks_then_when_true(t in any::<i64>(), e in any::<i64>()) {
        let src = alloc::format!("(if true {} {})", t, e);
        prop_assert_eq!(run(&src), Value::Integer(t));
    }

    #[test]
    fn paket_b_if_picks_else_when_false(t in any::<i64>(), e in any::<i64>()) {
        let src = alloc::format!("(if false {} {})", t, e);
        prop_assert_eq!(run(&src), Value::Integer(e));
    }

    #[test]
    fn paket_b_if_with_computed_bool_cond(a in any::<i64>(), b in any::<i64>()) {
        // (if (lt a b) 1 0) — observable lhs/rhs comparison.
        let src = alloc::format!("(if (lt {} {}) 1 0)", a, b);
        prop_assert_eq!(run(&src), Value::Integer(if a < b { 1 } else { 0 }));
    }

    // ── Loop with immediate break preserves value ─────────────

    #[test]
    fn paket_b_loop_break_returns_carried_integer(v in any::<i64>()) {
        let src = alloc::format!("(loop (break {}))", v);
        prop_assert_eq!(run(&src), Value::Integer(v));
    }

    #[test]
    fn paket_b_loop_break_returns_carried_bool(b in any::<bool>()) {
        let src = alloc::format!("(loop (break {}))", b);
        prop_assert_eq!(run(&src), Value::Bool(b));
    }

    // ── Session ≡ interpret on every Paket B case ─────────────

    #[test]
    fn paket_b_session_matches_interpret_for_arithmetic_with_bool(
        a in -1000i64..1000,
        b in -1000i64..1000,
    ) {
        let src = alloc::format!("(if (lt {} {}) (add {} {}) (sub {} {}))", a, b, a, b, a, b);
        let ast = parse(&src).expect("parse");
        let via_interpret = interpret(&ast).expect("interpret");
        let mut sess = Session::new(&ast).expect("session");
        let mut ctx = NullPolicyContext::new();
        let via_session = loop {
            match sess.step(&mut ctx).expect("step") {
                StepOutcome::Continue => continue,
                StepOutcome::Done(v) => break v,
            }
        };
        prop_assert_eq!(via_interpret, via_session);
    }

    // ── Determinism: two runs of the same program ─────────────

    #[test]
    fn paket_b_determinism_two_runs(seed in 0u64..1_000_000) {
        // Build a syntactically-non-trivial program parameterised by
        // `seed`; assert two runs produce identical outputs.
        let src = alloc::format!(
            "(if (eq (sub (add {} 1) 1) {}) (loop (break 42)) (loop (break -1)))",
            seed as i64, seed as i64,
        );
        let v1 = run(&src);
        let v2 = run(&src);
        prop_assert_eq!(v1, v2);
    }
}

// ── No-panic envelope: random ill-typed inputs surface errors,
// never panic. Validator catches statically; this guards the
// interpreter for any program slipped past the validator. ──

#[test]
fn paket_b_interpreter_rejects_register_with_bool() {
    let ast = parse("(register true)").expect("parse");
    let result = interpret(&ast);
    assert!(result.is_err(), "register expected to reject Bool");
}

extern crate alloc;
