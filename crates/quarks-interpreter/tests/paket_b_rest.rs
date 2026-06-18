// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stage 12 Paket B.2/B.3/B.4/B.5/B.6 — Data-structure, loop,
//! string, and I/O extensions of Quarks.
//!
//! These tests exercise the parse → validate → type-check → interpret
//! pipeline for the second wave of Paket B (lists, maps, bounded
//! loops, strings, host I/O), running side by side with the existing
//! `paket_b.rs` Bool/comparison/let/while suite.
//!
//! References:
//! - `docs/discovery/stage-12-completion-plan.md` §B.2/B.3/B.4/B.5/B.6

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::ToString;

use quarks_interpreter::{
    interpret, interpret_with_context, InterpretError, InterpretErrorKind, NullPolicyContext,
    PolicyContext, PolicyError, Value,
};
use quarks_validator::{parse, type_check, validate_structure, ValueType};

fn run(src: &str) -> Result<Value, InterpretError> {
    let ast = parse(src).expect("parse");
    validate_structure(&ast).expect("validator structural");
    type_check(&ast).expect("validator type-check");
    interpret(&ast)
}

fn run_with<C: PolicyContext>(src: &str, ctx: &mut C) -> Result<Value, InterpretError> {
    let ast = parse(src).expect("parse");
    validate_structure(&ast).expect("validator structural");
    type_check(&ast).expect("validator type-check");
    interpret_with_context(&ast, ctx)
}

fn type_stack(src: &str) -> Vec<ValueType> {
    let ast = parse(src).expect("parse");
    validate_structure(&ast).expect("validator structural");
    type_check(&ast).expect("type-check")
}

// ── B.2 Lists ──────────────────────────────────────────────────

#[test]
fn b2_list_empty() {
    assert_eq!(run("(list)").unwrap(), Value::List(alloc::vec![]));
    assert_eq!(type_stack("(list)"), alloc::vec![ValueType::List]);
}

#[test]
fn b2_list_literal_elements() {
    assert_eq!(
        run("(list 1 2 3)").unwrap(),
        Value::List(alloc::vec![1, 2, 3])
    );
}

#[test]
fn b2_list_evaluates_computed_elements() {
    // Elements may be arbitrary i64 expressions.
    assert_eq!(
        run("(list (add 1 2) (mul 2 5) -3)").unwrap(),
        Value::List(alloc::vec![3, 10, -3])
    );
}

#[test]
fn b2_list_get_returns_element() {
    assert_eq!(
        run("(list-get (list 10 20 30) 0)").unwrap(),
        Value::Integer(10)
    );
    assert_eq!(
        run("(list-get (list 10 20 30) 2)").unwrap(),
        Value::Integer(30)
    );
}

#[test]
fn b2_list_get_out_of_bounds() {
    let err = run("(list-get (list 10 20 30) 5)").unwrap_err();
    assert!(matches!(err.kind, InterpretErrorKind::ListIndexOutOfBounds));
}

#[test]
fn b2_list_get_negative_index() {
    let err = run("(list-get (list 10 20 30) -1)").unwrap_err();
    assert!(matches!(err.kind, InterpretErrorKind::ListIndexOutOfBounds));
}

#[test]
fn b2_list_len_returns_size() {
    assert_eq!(run("(list-len (list))").unwrap(), Value::Integer(0));
    assert_eq!(
        run("(list-len (list 7 8 9 10))").unwrap(),
        Value::Integer(4)
    );
}

#[test]
fn b2_list_append_adds_element() {
    assert_eq!(
        run("(list-append (list 1 2) 3)").unwrap(),
        Value::List(alloc::vec![1, 2, 3])
    );
}

#[test]
fn b2_list_append_to_empty() {
    assert_eq!(
        run("(list-append (list) 99)").unwrap(),
        Value::List(alloc::vec![99])
    );
}

#[test]
fn b2_list_chain_append_then_get() {
    assert_eq!(
        run("(list-get (list-append (list 1 2) 3) 2)").unwrap(),
        Value::Integer(3)
    );
}

#[test]
fn b2_list_type_signature() {
    // Validator must report List as the program's stack type.
    assert_eq!(type_stack("(list 1 2 3)"), alloc::vec![ValueType::List]);
    assert_eq!(
        type_stack("(list-len (list 1 2 3))"),
        alloc::vec![ValueType::I64]
    );
}

#[test]
fn b2_list_rejects_non_i64_element() {
    // The validator must reject Bool-typed elements.
    let ast = parse("(list 1 true 3)").unwrap();
    validate_structure(&ast).expect("structural pass should accept");
    let err = type_check(&ast).expect_err("type-check should reject");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::TypeMismatch { .. }
    ));
}

// ── B.3 Maps ───────────────────────────────────────────────────

#[test]
fn b3_map_new_is_empty() {
    assert_eq!(run("(map-new)").unwrap(), Value::Map(BTreeMap::new()));
    assert_eq!(type_stack("(map-new)"), alloc::vec![ValueType::Map]);
}

#[test]
fn b3_map_put_and_get() {
    assert_eq!(
        run("(map-get (map-put (map-new) 1 100) 1)").unwrap(),
        Value::Integer(100)
    );
}

#[test]
fn b3_map_get_missing_key_returns_zero() {
    // Missing key → 0 (combined with map-contains for presence test).
    assert_eq!(run("(map-get (map-new) 42)").unwrap(), Value::Integer(0));
}

#[test]
fn b3_map_contains_true_after_put() {
    assert_eq!(
        run("(map-contains (map-put (map-new) 5 99) 5)").unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn b3_map_contains_false_when_missing() {
    assert_eq!(
        run("(map-contains (map-new) 99)").unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn b3_map_put_overwrites_existing_key() {
    assert_eq!(
        run("(map-get (map-put (map-put (map-new) 1 100) 1 200) 1)").unwrap(),
        Value::Integer(200)
    );
}

#[test]
fn b3_map_determinism_btreemap_ordering() {
    // Two parallel programs producing maps with the same logical
    // content. Since BTreeMap stores by key order, the resulting
    // Value::Map's PartialEq compares as equal regardless of insert
    // order.
    let a = run("(map-put (map-put (map-new) 1 10) 2 20)").unwrap();
    let b = run("(map-put (map-put (map-new) 2 20) 1 10)").unwrap();
    assert_eq!(a, b);
}

// ── B.4 Bounded Loops ──────────────────────────────────────────

#[test]
fn b4_loop_with_bound_completes_via_explicit_break() {
    // body always breaks → loop returns the break value.
    assert_eq!(
        run("(loop-with-bound 10 (break true))").unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn b4_loop_with_bound_exhausts_to_implicit_false() {
    // body never breaks; iteration cap forces implicit (break false).
    // We use seq+discard with a computation that produces a Bool
    // residual (loop bodies must produce one value per iteration).
    assert_eq!(
        run("(loop-with-bound 3 (or true false))").unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn b4_loop_with_bound_zero_iterations() {
    // bound=0 → loop never enters body → implicit (break false).
    assert_eq!(
        run("(loop-with-bound 0 (break true))").unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn b4_loop_with_bound_rejects_negative_bound() {
    let err = run("(loop-with-bound -1 (break true))").unwrap_err();
    assert!(matches!(err.kind, InterpretErrorKind::LoopBoundInvalid));
}

#[test]
fn b4_for_iterates_list_and_completes() {
    // Iterate over a 3-element list; body always discards %1.
    let src = "(program \
        (fn run () bool (for %0 (list 1 2 3) (gt %0 0))) \
        (call run))";
    assert_eq!(run(src).unwrap(), Value::Bool(true));
}

#[test]
fn b4_for_iterates_empty_list() {
    let src = "(program \
        (fn run () bool (for %0 (list) (gt %0 0))) \
        (call run))";
    assert_eq!(run(src).unwrap(), Value::Bool(true));
}

#[test]
fn b4_for_break_exits_early() {
    // Once %0 is observed in the body, `(break false)` exits the
    // for-loop with Bool(false).
    let src = "(program \
        (fn run () bool (for %0 (list 1 2 3) (break false))) \
        (call run))";
    assert_eq!(run(src).unwrap(), Value::Bool(false));
}

#[test]
fn b4_for_rejects_non_list_source() {
    let ast = parse("(program (fn run () bool (for %0 42 (gt %0 0))) (call run))").unwrap();
    validate_structure(&ast).expect("structural pass should accept");
    let err = type_check(&ast).expect_err("type-check should reject");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::TypeMismatch { .. }
    ));
}

// ── B.5 Strings ────────────────────────────────────────────────

#[test]
fn b5_string_from_int_basic() {
    assert_eq!(
        run("(string-from-int 42)").unwrap(),
        Value::String("42".to_string())
    );
}

#[test]
fn b5_string_from_int_negative() {
    assert_eq!(
        run("(string-from-int -7)").unwrap(),
        Value::String("-7".to_string())
    );
}

#[test]
fn b5_string_concat() {
    assert_eq!(
        run("(string-concat (string-from-int 1) (string-from-int 2))").unwrap(),
        Value::String("12".to_string())
    );
}

#[test]
fn b5_string_eq_identical() {
    assert_eq!(
        run("(string-eq (string-from-int 5) (string-from-int 5))").unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn b5_string_eq_different() {
    assert_eq!(
        run("(string-eq (string-from-int 5) (string-from-int 6))").unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn b5_string_type_signature() {
    assert_eq!(
        type_stack("(string-from-int 1)"),
        alloc::vec![ValueType::String]
    );
    assert_eq!(
        type_stack("(string-eq (string-from-int 1) (string-from-int 2))"),
        alloc::vec![ValueType::Bool]
    );
}

// ── B.6 I/O ────────────────────────────────────────────────────

/// Recording PolicyContext that captures read_handle / write_host_state
/// invocations so tests can both assert dispatch and script return
/// values.
struct RecordingIo {
    reads: Vec<u64>,
    writes: Vec<(alloc::string::String, i64)>,
    read_response: Vec<u8>,
    write_response: i64,
    fail_with: Option<PolicyError>,
}

impl RecordingIo {
    fn new() -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            read_response: alloc::vec![0xAB, 0xCD],
            write_response: 0,
            fail_with: None,
        }
    }
}

impl PolicyContext for RecordingIo {
    fn policy(
        &mut self,
        _subsystem: &str,
        _operation: &str,
        _args: &[Value],
    ) -> Result<i64, PolicyError> {
        Err(PolicyError::NotSupported)
    }
    fn query(
        &mut self,
        _subsystem: &str,
        _metric: &str,
        _args: &[Value],
    ) -> Result<i64, PolicyError> {
        Err(PolicyError::NotSupported)
    }
    fn read_handle(&mut self, handle: u64) -> Result<Vec<u8>, PolicyError> {
        self.reads.push(handle);
        if let Some(e) = self.fail_with.clone() {
            return Err(e);
        }
        Ok(self.read_response.clone())
    }
    fn write_host_state(&mut self, key: &str, value: i64) -> Result<i64, PolicyError> {
        self.writes.push((key.to_string(), value));
        if let Some(e) = self.fail_with.clone() {
            return Err(e);
        }
        Ok(self.write_response)
    }
}

#[test]
fn b6_read_handle_dispatches_and_returns_bytes() {
    let mut ctx = RecordingIo::new();
    let v = run_with("(read-handle @5)", &mut ctx).unwrap();
    assert_eq!(v, Value::Bytes(alloc::vec![0xAB, 0xCD]));
    assert_eq!(ctx.reads, alloc::vec![5u64]);
}

#[test]
fn b6_read_handle_with_null_context_is_not_supported() {
    let mut ctx = NullPolicyContext::new();
    let err = run_with("(read-handle @5)", &mut ctx).unwrap_err();
    assert!(matches!(err.kind, InterpretErrorKind::PolicyNotSupported));
}

#[test]
fn b6_read_handle_propagates_permission_denied() {
    let mut ctx = RecordingIo::new();
    ctx.fail_with = Some(PolicyError::PermissionDenied);
    let err = run_with("(read-handle @5)", &mut ctx).unwrap_err();
    assert!(matches!(err.kind, InterpretErrorKind::PolicyDispatchFailed));
}

#[test]
fn b6_write_host_state_dispatches_with_key_and_value() {
    let mut ctx = RecordingIo::new();
    ctx.write_response = 0;
    let v = run_with("(write-host-state cpu-share 75)", &mut ctx).unwrap();
    assert_eq!(v, Value::Integer(0));
    assert_eq!(ctx.writes, alloc::vec![("cpu-share".to_string(), 75)]);
}

#[test]
fn b6_write_host_state_evaluates_value_expression() {
    let mut ctx = RecordingIo::new();
    let v = run_with("(write-host-state mem-pct (add 30 40))", &mut ctx).unwrap();
    assert_eq!(v, Value::Integer(0));
    assert_eq!(ctx.writes, alloc::vec![("mem-pct".to_string(), 70)]);
}

#[test]
fn b6_write_host_state_with_null_context_is_not_supported() {
    let mut ctx = NullPolicyContext::new();
    let err = run_with("(write-host-state key 1)", &mut ctx).unwrap_err();
    assert!(matches!(err.kind, InterpretErrorKind::PolicyNotSupported));
}

#[test]
fn b6_write_host_state_propagates_permission_denied() {
    let mut ctx = RecordingIo::new();
    ctx.fail_with = Some(PolicyError::PermissionDenied);
    let err = run_with("(write-host-state foo 1)", &mut ctx).unwrap_err();
    assert!(matches!(err.kind, InterpretErrorKind::PolicyDispatchFailed));
}

#[test]
fn b6_write_host_state_rejects_non_symbol_key() {
    // type-checker should reject a non-symbol key slot structurally.
    let ast = parse("(write-host-state 42 100)").unwrap();
    // Parser accepts; structural validator accepts (custom dispatch);
    // type-checker rejects.
    validate_structure(&ast).expect("structural");
    let err = type_check(&ast).expect_err("type-check should reject");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::InvalidProgramStructureError
    ));
}

// ── End-to-end integration: combine Paket B.2/B.3/B.4/B.5 ─────

#[test]
fn integration_list_in_function_sums_via_for_break() {
    // The for-loop breaks with the first element it sees, ignoring
    // the rest. This exercises list construction, function-scoped
    // for-loop, parameter binding, and break.
    let src = "(program \
        (fn first () bool (for %0 (list 7 8 9) (break (gt %0 5)))) \
        (call first))";
    assert_eq!(run(src).unwrap(), Value::Bool(true));
}

#[test]
fn integration_map_used_in_function() {
    let src = "(program \
        (fn lookup (i64) i64 (map-get (map-put (map-new) %0 999) %0)) \
        (call lookup 42))";
    assert_eq!(run(src).unwrap(), Value::Integer(999));
}

#[test]
fn integration_string_compose_then_eq() {
    // (string-eq "12" (string-concat "1" "2")) — where each "n" is
    // built via string-from-int.
    let src = "(string-eq \
        (string-from-int 12) \
        (string-concat (string-from-int 1) (string-from-int 2)))";
    assert_eq!(run(src).unwrap(), Value::Bool(true));
}
