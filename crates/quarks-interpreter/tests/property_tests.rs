// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stage 12 Paket A.3 — property-based determinism harness for the
//! explicit state-machine interpreter.
//!
//! These tests assert three invariants over a randomised cohort of
//! generated Quarks programs:
//!
//! 1. **`Session::run(P) ≡ interpret(P)`** — driving the explicit
//!    machine to completion matches the public `interpret` surface.
//! 2. **No panics, no UB.** The state machine must never panic on
//!    well-formed or malformed input; errors must surface as
//!    [`InterpretError`] variants.
//! 3. **Determinism contract.** Two independent runs of the same
//!    program yield bitwise-identical [`Value`] outputs.
//!
//! Why `proptest` rather than `cargo-fuzz`: the interpreter is
//! `no_std + alloc`, but the kernel-side production targets do not
//! expose a libFuzzer runtime. `cargo-fuzz` also requires a nightly
//! toolchain and a separate workspace (`fuzz/`). `proptest` runs
//! under stable `cargo test`, integrates with the existing CI
//! pipeline, and shrinks failing inputs to a minimal repro — which is
//! the more valuable property for a kernel-TCB interpreter than raw
//! coverage throughput.
//!
//! To run only the property harness:
//!
//! ```text
//! cargo test -p quarks-interpreter --test property_tests
//! ```
//!
//! To increase the number of generated cases (default `256`) for a
//! deeper sweep, set the [`PROPTEST_CASES`] environment variable, e.g.
//!
//! ```text
//! PROPTEST_CASES=5000 cargo test -p quarks-interpreter --test property_tests
//! ```
//!
//! Proptest persists shrink seeds for any failing inputs under
//! `proptest-regressions/` so failures are reproducible without re-
//! running the entire generator. See
//! `docs/discovery/hardware-abstraction-constraints.md` §1 for the
//! determinism contract this harness defends.

use proptest::prelude::*;
use quarks_interpreter::{interpret, NullPolicyContext, Session, StepOutcome, Value};
use quarks_validator::{parse, SExpr};

// ---- Generator surface ----

/// Generate a finite arithmetic Quarks program of bounded depth.
///
/// Restricted to `(add ...)` / `(sub ...)` / `(mul ...)` with integer
/// leaves so that:
/// - every generated program parses and type-checks deterministically;
/// - the value space is bounded (i64 wrapping arithmetic) — the
///   interpreter handles wrap-around bit-for-bit;
/// - the generator size is correlated with depth (controls test
///   wall-time).
fn arith_program() -> impl Strategy<Value = String> {
    let leaf = (any::<i32>()).prop_map(|n| format!("{}", n as i64));
    leaf.prop_recursive(
        4,  // max recursion depth
        32, // max generated nodes
        2,  // expected branching factor
        |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(a, b)| format!("(add {} {})", a, b)),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| format!("(sub {} {})", a, b)),
                (inner.clone(), inner).prop_map(|(a, b)| format!("(mul {} {})", a, b)),
            ]
        },
    )
}

/// Generate either a bare arithmetic program or a `(program (fn add
/// (i64 i64) i64 ...) (call add ...))` wrapper. This exercises the
/// `program`-passthrough and the `call`/`fn` machinery alongside the
/// raw arithmetic-only path.
fn arith_or_call_program() -> impl Strategy<Value = String> {
    let bare = arith_program();
    let call = (arith_program(), arith_program()).prop_map(|(a, b)| {
        format!(
            "(program (fn f (i64 i64) i64 (add %0 %1)) (call f {} {}))",
            a, b
        )
    });
    prop_oneof![bare, call]
}

// ---- Property 1: Step ≡ Run equivalence ----

proptest! {
    /// `Session::new(P).run() == interpret(P)` for every generated
    /// arithmetic program.
    #[test]
    fn session_run_matches_interpret(src in arith_or_call_program()) {
        let ast = match parse(&src) {
            Ok(a) => a,
            Err(_) => return Ok(()),  // generator only emits valid src; skip on parse fail
        };
        let via_recursive = interpret(&ast);
        let mut session = Session::new(&ast).expect("Session::new");
        let mut ctx = NullPolicyContext::new();
        let via_machine = session.run(&mut ctx);
        prop_assert_eq!(via_recursive, via_machine);
    }
}

// ---- Property 2: Step-by-step ≡ Run ----

proptest! {
    /// Driving `Session::step` in a loop until `Done` produces the
    /// same final value as `Session::run` does.
    #[test]
    fn step_loop_matches_run(src in arith_or_call_program()) {
        let ast = match parse(&src) {
            Ok(a) => a,
            Err(_) => return Ok(()),
        };

        let mut a = Session::new(&ast).expect("Session::new");
        let mut actx = NullPolicyContext::new();
        let by_run = a.run(&mut actx);

        let mut b = Session::new(&ast).expect("Session::new");
        let mut bctx = NullPolicyContext::new();
        let by_step = loop {
            match b.step(&mut bctx) {
                Ok(StepOutcome::Continue) => continue,
                Ok(StepOutcome::Done(v)) => break Ok(v),
                Err(e) => break Err(e),
            }
        };

        prop_assert_eq!(by_run, by_step);
    }
}

// ---- Property 3: Bitwise determinism — two independent runs agree ----

proptest! {
    /// Two fresh sessions over the same program produce
    /// bit-identical [`Value`] outputs. This is the testable
    /// formulation of the determinism contract from
    /// `hardware-abstraction-constraints.md` §1.4.
    #[test]
    fn two_independent_runs_agree(src in arith_or_call_program()) {
        let ast = match parse(&src) {
            Ok(a) => a,
            Err(_) => return Ok(()),
        };

        let v1 = interpret(&ast);

        let mut s2 = Session::new(&ast).expect("Session::new");
        let mut c2 = NullPolicyContext::new();
        let v2 = s2.run(&mut c2);

        let mut s3 = Session::new(&ast).expect("Session::new");
        let mut c3 = NullPolicyContext::new();
        let v3 = s3.run(&mut c3);

        prop_assert_eq!(v1.clone(), v2.clone());
        prop_assert_eq!(v2, v3);
        // Sanity: arithmetic-only inputs produce Integer values when
        // they succeed.
        if let Ok(value) = v1 {
            prop_assert!(matches!(value, Value::Integer(_)));
        }
    }
}

// ---- Property 4: Step-by-step pause/resume preserves the result ----

proptest! {
    /// Take a randomised number of steps and then finish via `run`;
    /// the value must match a full single-shot `run`. This validates
    /// the A.4 mid-program preemption contract: pause/resume is
    /// value-preserving.
    #[test]
    fn pause_at_arbitrary_step_then_resume_agrees(
        src in arith_or_call_program(),
        steps_first_slice in 0u32..32u32,
    ) {
        let ast = match parse(&src) {
            Ok(a) => a,
            Err(_) => return Ok(()),
        };

        // Reference value.
        let reference = interpret(&ast);

        // Stepped run: take `steps_first_slice` steps, then run() to
        // completion. Must equal the reference.
        let mut session = Session::new(&ast).expect("Session::new");
        let mut ctx = NullPolicyContext::new();
        let mut early_value: Option<Value> = None;
        for _ in 0..steps_first_slice {
            match session.step(&mut ctx) {
                Ok(StepOutcome::Continue) => {}
                Ok(StepOutcome::Done(v)) => {
                    early_value = Some(v);
                    break;
                }
                Err(e) => {
                    // If the *first slice* errors, the reference must
                    // also error.  Assert equivalence and stop.
                    prop_assert_eq!(reference, Err(e));
                    return Ok(());
                }
            }
        }
        let final_value = match early_value {
            Some(v) => Ok(v),
            None => session.run(&mut ctx),
        };

        prop_assert_eq!(reference, final_value);
    }
}

// ---- Property 5: instructions_executed is monotonic and bounded ----

proptest! {
    /// `instructions_executed` strictly increases on every successful
    /// `Continue` step and never wraps backwards. It is also bounded
    /// (well below `u64::MAX`) for the depth-4 generator, providing a
    /// crude smoke test against runaway counters.
    #[test]
    fn instructions_executed_is_monotonic(src in arith_or_call_program()) {
        let ast = match parse(&src) {
            Ok(a) => a,
            Err(_) => return Ok(()),
        };
        let mut session = Session::new(&ast).expect("Session::new");
        let mut ctx = NullPolicyContext::new();
        let mut prev = 0u64;
        loop {
            let before = session.instructions_executed();
            prop_assert_eq!(before, prev);
            match session.step(&mut ctx) {
                Ok(StepOutcome::Continue) => {
                    let after = session.instructions_executed();
                    prop_assert!(
                        after > before,
                        "instructions_executed must strictly increase across a Continue step: {} → {}",
                        before, after
                    );
                    prev = after;
                }
                Ok(StepOutcome::Done(_)) => break,
                Err(_) => break,
            }
            // Hard upper bound for the depth-4 generator. If a
            // generated input ever exceeds this, the test fails
            // loudly so we know the generator outgrew the harness.
            prop_assert!(prev < 100_000, "runaway instruction count: {}", prev);
        }
    }
}

// ---- Property 6: malformed input never panics ----

/// Free-form S-expression-like text generator. Produces arbitrary
/// bracketed strings — many will fail to parse, many more will fail
/// type-check, and most surviving inputs will produce `InterpretError`
/// variants. The test only asserts *no panic*.
fn freeform_input() -> impl Strategy<Value = String> {
    let symbol = "[a-z][a-z0-9]{0,4}".prop_map(String::from);
    let int = (any::<i32>()).prop_map(|n| format!("{}", n));
    let atom = prop_oneof![symbol, int];
    atom.prop_recursive(3, 16, 3, |inner| {
        prop::collection::vec(inner, 0..4).prop_map(|items| format!("({})", items.join(" ")))
    })
}

proptest! {
    /// Feed random S-expression-like strings into the parser +
    /// interpreter. The harness asserts *no panic* — any failure must
    /// surface as a Result error. This is the closest stable-Rust
    /// analogue of a libFuzzer panic-only run.
    #[test]
    fn never_panics_on_random_input(src in freeform_input()) {
        // Catch any panic at the harness boundary.
        let outcome = std::panic::catch_unwind(|| {
            let parsed = parse(&src);
            if let Ok(ast) = parsed {
                let _ = interpret(&ast);
                if let Ok(mut s) = Session::new(&ast) {
                    let mut ctx = NullPolicyContext::new();
                    let _ = s.run(&mut ctx);
                }
            }
        });
        prop_assert!(outcome.is_ok(), "panic on input {:?}", src);
    }
}

// ---- SExpr-level direct invariant: handle/symbol leaves never panic ----

proptest! {
    /// Feed direct SExpr leaves (not via the parser) to assert
    /// dispatch surfaces never panic on out-of-band atoms. Catches
    /// regressions where a new SExpr variant lands without an
    /// interpreter dispatch arm.
    #[test]
    fn direct_atom_inputs_never_panic(n in any::<i64>()) {
        let ast = SExpr::Atom(quarks_validator::Atom::Integer(n));
        let outcome = std::panic::catch_unwind(|| {
            let _ = interpret(&ast);
        });
        prop_assert!(outcome.is_ok());
    }
}
