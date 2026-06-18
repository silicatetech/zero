// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 4 Step 1 — Bitwise operations: end-to-end tests.
//!
//! Exercise the parse → validate → type-check → interpret pipeline
//! for the five new instructions `bit-and`, `bit-or`, `bit-xor`,
//! `bit-shl`, `bit-shr`. The validator-only and interpreter-only
//! aspects are covered in their crates' unit suites; this file is
//! the integration tier that the kernel-LLM-generated code follows.
//!
//! Numeric literals in Quarks IR are decimal only (the parser
//! does not accept `0xff` syntax) — the tests use the decimal
//! equivalents and document the hex value in comments.

use quarks_interpreter::{interpret, InterpretError, InterpretErrorKind, Value};
use quarks_validator::{parse, type_check, validate_structure, TypeCheckError, ValueType};

fn run(src: &str) -> Result<Value, InterpretError> {
    let ast = parse(src).expect("parse");
    validate_structure(&ast).expect("validator structural");
    type_check(&ast).expect("validator type-check");
    interpret(&ast)
}

fn run_err(src: &str) -> InterpretError {
    let ast = parse(src).expect("parse");
    validate_structure(&ast).expect("validator structural");
    type_check(&ast).expect("validator type-check");
    interpret(&ast).expect_err("interpret should fail")
}

fn type_stack(src: &str) -> Vec<ValueType> {
    let ast = parse(src).expect("parse");
    validate_structure(&ast).expect("validator structural");
    type_check(&ast).expect("type-check")
}

/// Phase 4 Step 4 helper — parse + validate structurally, then
/// expect type-check to fail. Used for `(cond …)` branch-mismatch
/// tests where the structural pass succeeds but the type pass
/// rejects mismatched branch bodies.
fn type_check_err(src: &str) -> TypeCheckError {
    let ast = parse(src).expect("parse");
    validate_structure(&ast).expect("validator structural");
    type_check(&ast).expect_err("type-check should fail")
}

// ── Basic AND/OR/XOR ──────────────────────────────────────────

#[test]
fn p4_bit_and_masks_low_nibble() {
    // 0xff & 0x0f = 0x0f
    assert_eq!(run("(bit-and 255 15)").unwrap(), Value::Integer(15));
    assert_eq!(type_stack("(bit-and 255 15)"), vec![ValueType::I64]);
}

#[test]
fn p4_bit_or_sets_bits() {
    assert_eq!(run("(bit-or 1 2)").unwrap(), Value::Integer(3));
}

#[test]
fn p4_bit_xor_toggles_bits() {
    assert_eq!(run("(bit-xor 5 3)").unwrap(), Value::Integer(6));
}

#[test]
fn p4_bit_and_zero_clears() {
    assert_eq!(run("(bit-and 12345 0)").unwrap(), Value::Integer(0));
}

#[test]
fn p4_bit_or_with_minus_one_sets_all() {
    // -1 is all-ones in two's complement.
    assert_eq!(run("(bit-or 0 -1)").unwrap(), Value::Integer(-1));
}

// ── Shifts ────────────────────────────────────────────────────

#[test]
fn p4_bit_shl_one_by_ten_is_1024() {
    assert_eq!(run("(bit-shl 1 10)").unwrap(), Value::Integer(1024));
}

#[test]
fn p4_bit_shl_one_by_eight_is_256() {
    assert_eq!(run("(bit-shl 1 8)").unwrap(), Value::Integer(256));
}

#[test]
fn p4_bit_shr_arithmetic_preserves_sign() {
    // -1 >> 1 is -1 under arithmetic right shift (SAR).
    assert_eq!(run("(bit-shr -1 1)").unwrap(), Value::Integer(-1));
}

#[test]
fn p4_bit_shr_positive_is_floor_div_two() {
    assert_eq!(run("(bit-shr 256 4)").unwrap(), Value::Integer(16));
}

#[test]
fn p4_bit_shr_one_by_sixty_three_is_zero() {
    // Positive operand shifted past its MSB → 0.
    assert_eq!(run("(bit-shr 1 63)").unwrap(), Value::Integer(0));
}

#[test]
fn p4_bit_shl_one_by_sixty_three_is_i64_min() {
    // 1 << 63 = i64::MIN under wrapping semantics (sign bit flips).
    assert_eq!(run("(bit-shl 1 63)").unwrap(), Value::Integer(i64::MIN));
}

#[test]
fn p4_bit_shr_negative_one_far_stays_negative() {
    assert_eq!(run("(bit-shr -1 63)").unwrap(), Value::Integer(-1));
}

#[test]
fn p4_bit_shl_zero_count_is_identity() {
    assert_eq!(run("(bit-shl 42 0)").unwrap(), Value::Integer(42));
}

// ── Shift error path: ShiftCountOutOfRange ────────────────────

#[test]
fn p4_bit_shl_count_64_errors() {
    let err = run_err("(bit-shl 1 64)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::ShiftCountOutOfRange { count: 64 }
    );
}

#[test]
fn p4_bit_shl_count_negative_errors() {
    let err = run_err("(bit-shl 1 -1)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::ShiftCountOutOfRange { count: -1 }
    );
}

#[test]
fn p4_bit_shr_count_64_errors() {
    let err = run_err("(bit-shr 1 64)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::ShiftCountOutOfRange { count: 64 }
    );
}

#[test]
fn p4_bit_shr_count_negative_errors() {
    let err = run_err("(bit-shr 1 -5)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::ShiftCountOutOfRange { count: -5 }
    );
}

#[test]
fn p4_bit_shl_huge_count_errors() {
    // i64::MAX as count must also surface a typed error, not
    // wrap-mask to a valid count via x86's low-6-bit semantics.
    let src = "(bit-shl 1 9223372036854775807)";
    let err = run_err(src);
    assert_eq!(
        err.kind,
        InterpretErrorKind::ShiftCountOutOfRange { count: i64::MAX }
    );
}

// ── Composition ───────────────────────────────────────────────

#[test]
fn p4_bit_and_with_shifted_mask() {
    // (1 << 8) & 0xff00 → 256 & 65280 = 256
    assert_eq!(
        run("(bit-and (bit-shl 1 8) 65280)").unwrap(),
        Value::Integer(256)
    );
}

#[test]
fn p4_chained_xor_returns_to_origin() {
    // ((x XOR y) XOR y) == x
    assert_eq!(
        run("(bit-xor (bit-xor 1234 5678) 5678)").unwrap(),
        Value::Integer(1234)
    );
}

#[test]
fn p4_or_mask_via_shift() {
    // Setting bit 4 of 1: 1 | (1 << 4) = 17
    assert_eq!(run("(bit-or 1 (bit-shl 1 4))").unwrap(), Value::Integer(17));
}

// ── Function calls (program (fn ...) ...) ────────────────────

#[test]
fn p4_bit_and_inside_function() {
    let src = "(program \
        (fn mask (i64 i64) i64 (bit-and %0 %1)) \
        (call mask 255 15))";
    assert_eq!(run(src).unwrap(), Value::Integer(15));
}

#[test]
fn p4_bit_shl_inside_function() {
    let src = "(program \
        (fn shift_left (i64 i64) i64 (bit-shl %0 %1)) \
        (call shift_left 1 10))";
    assert_eq!(run(src).unwrap(), Value::Integer(1024));
}

#[test]
fn p4_bit_shr_function_propagates_shift_error() {
    // Same `ShiftCountOutOfRange` surfaces from inside a callee.
    let src = "(program \
        (fn shift_right (i64 i64) i64 (bit-shr %0 %1)) \
        (call shift_right 1 64))";
    let err = run_err(src);
    assert_eq!(
        err.kind,
        InterpretErrorKind::ShiftCountOutOfRange { count: 64 }
    );
}

// ── Boundary value: 63 is the highest legal shift count ──────

#[test]
fn p4_bit_shl_count_63_is_legal() {
    // Highest legal shift; result wraps to i64::MIN as documented.
    assert_eq!(run("(bit-shl 1 63)").unwrap(), Value::Integer(i64::MIN));
}

#[test]
fn p4_bit_shr_count_63_is_legal() {
    assert_eq!(run("(bit-shr 1 63)").unwrap(), Value::Integer(0));
}

// ─────────────────────────────────────────────────────────────
// Phase 4 Step 2 — Bytes operations
// ─────────────────────────────────────────────────────────────

// ── bytes-len ────────────────────────────────────────────────

#[test]
fn p4_bytes_len_basic() {
    // #x48656c6c6f = "Hello" = 5 bytes
    assert_eq!(run("(bytes-len #x48656c6c6f)").unwrap(), Value::Integer(5));
    assert_eq!(type_stack("(bytes-len #x48656c6c6f)"), vec![ValueType::I64]);
}

#[test]
fn p4_bytes_len_empty() {
    assert_eq!(run("(bytes-len #x)").unwrap(), Value::Integer(0));
}

// ── bytes-get ────────────────────────────────────────────────

#[test]
fn p4_bytes_get_first_byte() {
    // #x48 = 0x48 = 72 ('H')
    assert_eq!(
        run("(bytes-get #x48656c6c6f 0)").unwrap(),
        Value::Integer(72)
    );
    assert_eq!(
        type_stack("(bytes-get #x48656c6c6f 0)"),
        vec![ValueType::I64]
    );
}

#[test]
fn p4_bytes_get_last_byte() {
    // 'o' = 0x6f = 111
    assert_eq!(
        run("(bytes-get #x48656c6c6f 4)").unwrap(),
        Value::Integer(111)
    );
}

#[test]
fn p4_bytes_get_index_out_of_bounds_high() {
    let err = run_err("(bytes-get #x48656c6c6f 5)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::BytesIndexOutOfBounds {
            index: 5,
            length: 5,
        }
    );
}

#[test]
fn p4_bytes_get_index_negative() {
    let err = run_err("(bytes-get #x48656c6c6f -1)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::BytesIndexOutOfBounds {
            index: -1,
            length: 5,
        }
    );
}

#[test]
fn p4_bytes_get_byte_value_range() {
    // 0xff = 255 must round-trip as a positive i64.
    assert_eq!(run("(bytes-get #xff 0)").unwrap(), Value::Integer(255));
    // 0x80 = 128 (high bit set) — must NOT sign-extend to a negative i64.
    assert_eq!(run("(bytes-get #x80 0)").unwrap(), Value::Integer(128));
}

// ── bytes-slice ──────────────────────────────────────────────

#[test]
fn p4_bytes_slice_basic() {
    // slice #x48656c6c6f starting at 1, length 3 → #x656c6c
    assert_eq!(
        run("(bytes-slice #x48656c6c6f 1 3)").unwrap(),
        Value::Bytes(vec![0x65, 0x6c, 0x6c])
    );
    assert_eq!(
        type_stack("(bytes-slice #x48656c6c6f 1 3)"),
        vec![ValueType::Bytes]
    );
}

#[test]
fn p4_bytes_slice_full() {
    assert_eq!(
        run("(bytes-slice #x48656c6c6f 0 5)").unwrap(),
        Value::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f])
    );
}

#[test]
fn p4_bytes_slice_empty_len() {
    assert_eq!(
        run("(bytes-slice #x48656c6c6f 2 0)").unwrap(),
        Value::Bytes(vec![])
    );
}

#[test]
fn p4_bytes_slice_at_end_zero_len() {
    // start == length, len == 0 is a legal empty slice.
    assert_eq!(
        run("(bytes-slice #x48656c6c6f 5 0)").unwrap(),
        Value::Bytes(vec![])
    );
}

#[test]
fn p4_bytes_slice_out_of_bounds_overrun() {
    let err = run_err("(bytes-slice #x48656c6c6f 3 5)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::BytesSliceOutOfBounds {
            start: 3,
            len: 5,
            bytes_length: 5,
        }
    );
}

#[test]
fn p4_bytes_slice_negative_start() {
    let err = run_err("(bytes-slice #x48656c6c6f -1 2)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::BytesSliceOutOfBounds {
            start: -1,
            len: 2,
            bytes_length: 5,
        }
    );
}

#[test]
fn p4_bytes_slice_negative_len() {
    let err = run_err("(bytes-slice #x48656c6c6f 0 -1)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::BytesSliceOutOfBounds {
            start: 0,
            len: -1,
            bytes_length: 5,
        }
    );
}

#[test]
fn p4_bytes_slice_start_beyond_end() {
    let err = run_err("(bytes-slice #x48656c6c6f 6 0)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::BytesSliceOutOfBounds {
            start: 6,
            len: 0,
            bytes_length: 5,
        }
    );
}

// ── bytes-concat ─────────────────────────────────────────────

#[test]
fn p4_bytes_concat_two_non_empty() {
    assert_eq!(
        run("(bytes-concat #x4865 #x6c6c6f)").unwrap(),
        Value::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f])
    );
    assert_eq!(
        type_stack("(bytes-concat #x4865 #x6c6c6f)"),
        vec![ValueType::Bytes]
    );
}

#[test]
fn p4_bytes_concat_with_empty_lhs() {
    assert_eq!(
        run("(bytes-concat #x #x6c6c6f)").unwrap(),
        Value::Bytes(vec![0x6c, 0x6c, 0x6f])
    );
}

#[test]
fn p4_bytes_concat_with_empty_rhs() {
    assert_eq!(
        run("(bytes-concat #x4865 #x)").unwrap(),
        Value::Bytes(vec![0x48, 0x65])
    );
}

#[test]
fn p4_bytes_concat_two_empties() {
    assert_eq!(run("(bytes-concat #x #x)").unwrap(), Value::Bytes(vec![]));
}

#[test]
fn p4_bytes_concat_length_sums() {
    // len(a ++ b) == len(a) + len(b)
    assert_eq!(
        run("(bytes-len (bytes-concat #x4865 #x6c6c6f))").unwrap(),
        Value::Integer(5)
    );
}

// ── bytes-eq ─────────────────────────────────────────────────

#[test]
fn p4_bytes_eq_equal() {
    assert_eq!(
        run("(bytes-eq #x48656c6c6f #x48656c6c6f)").unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        type_stack("(bytes-eq #x4865 #x4865)"),
        vec![ValueType::Bool]
    );
}

#[test]
fn p4_bytes_eq_not_equal_same_length() {
    assert_eq!(run("(bytes-eq #x4865 #x4866)").unwrap(), Value::Bool(false));
}

#[test]
fn p4_bytes_eq_not_equal_different_length() {
    assert_eq!(
        run("(bytes-eq #x4865 #x486500)").unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn p4_bytes_eq_empty_vs_empty() {
    assert_eq!(run("(bytes-eq #x #x)").unwrap(), Value::Bool(true));
}

#[test]
fn p4_bytes_eq_empty_vs_non_empty() {
    assert_eq!(run("(bytes-eq #x #x00)").unwrap(), Value::Bool(false));
}

// ── bytes-from-int ───────────────────────────────────────────

#[test]
fn p4_bytes_from_int_one_little_endian() {
    // 1 as little-endian i64 = 01 00 00 00 00 00 00 00
    assert_eq!(
        run("(bytes-from-int 1)").unwrap(),
        Value::Bytes(vec![0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    );
    assert_eq!(type_stack("(bytes-from-int 1)"), vec![ValueType::Bytes]);
}

#[test]
fn p4_bytes_from_int_zero() {
    assert_eq!(
        run("(bytes-from-int 0)").unwrap(),
        Value::Bytes(vec![0x00; 8])
    );
}

#[test]
fn p4_bytes_from_int_negative_one() {
    // -1 in two's complement = all-ones = 0xff x 8
    assert_eq!(
        run("(bytes-from-int -1)").unwrap(),
        Value::Bytes(vec![0xff; 8])
    );
}

#[test]
fn p4_bytes_from_int_length_is_eight() {
    assert_eq!(
        run("(bytes-len (bytes-from-int 123456))").unwrap(),
        Value::Integer(8)
    );
}

// ── Composition ──────────────────────────────────────────────

#[test]
fn p4_bytes_get_after_concat() {
    // (bytes-concat #x0102 #x0304) → #x01020304, index 2 → 3
    assert_eq!(
        run("(bytes-get (bytes-concat #x0102 #x0304) 2)").unwrap(),
        Value::Integer(3)
    );
}

#[test]
fn p4_bytes_slice_after_concat() {
    // (bytes-slice (bytes-concat #x4865 #x6c6c6f) 1 3) → #x656c6c
    assert_eq!(
        run("(bytes-slice (bytes-concat #x4865 #x6c6c6f) 1 3)").unwrap(),
        Value::Bytes(vec![0x65, 0x6c, 0x6c])
    );
}

#[test]
fn p4_bytes_from_int_roundtrip_low_byte() {
    // bytes-from-int 0x42 -> low byte is 0x42 (66 decimal)
    assert_eq!(
        run("(bytes-get (bytes-from-int 66) 0)").unwrap(),
        Value::Integer(66)
    );
}

#[test]
fn p4_bytes_eq_after_concat_split() {
    // (bytes-concat #x4865 #x6c6c6f) == #x48656c6c6f
    let src = "(bytes-eq (bytes-concat #x4865 #x6c6c6f) #x48656c6c6f)";
    assert_eq!(run(src).unwrap(), Value::Bool(true));
}

// ── Function calls ───────────────────────────────────────────

#[test]
fn p4_bytes_len_inside_function() {
    let src = "(program \
        (fn measure (bytes) i64 (bytes-len %0)) \
        (call measure #x48656c6c6f))";
    assert_eq!(run(src).unwrap(), Value::Integer(5));
}

#[test]
fn p4_bytes_get_inside_function_propagates_oob() {
    let src = "(program \
        (fn pick (bytes i64) i64 (bytes-get %0 %1)) \
        (call pick #x4865 10))";
    let err = run_err(src);
    assert_eq!(
        err.kind,
        InterpretErrorKind::BytesIndexOutOfBounds {
            index: 10,
            length: 2,
        }
    );
}

// ─────────────────────────────────────────────────────────────
// Phase 4 Step 3 — list-slice
// ─────────────────────────────────────────────────────────────

#[test]
fn p4_list_slice_basic() {
    // slice (list 10 20 30 40 50) starting at 1, length 3 → (list 20 30 40)
    assert_eq!(
        run("(list-slice (list 10 20 30 40 50) 1 3)").unwrap(),
        Value::List(vec![20, 30, 40])
    );
    assert_eq!(
        type_stack("(list-slice (list 10 20 30 40 50) 1 3)"),
        vec![ValueType::List]
    );
}

#[test]
fn p4_list_slice_full() {
    assert_eq!(
        run("(list-slice (list 10 20 30 40 50) 0 5)").unwrap(),
        Value::List(vec![10, 20, 30, 40, 50])
    );
}

#[test]
fn p4_list_slice_empty_len() {
    assert_eq!(
        run("(list-slice (list 10 20 30 40 50) 2 0)").unwrap(),
        Value::List(vec![])
    );
}

#[test]
fn p4_list_slice_at_end_zero_len() {
    // start == length, len == 0 is a legal empty slice.
    assert_eq!(
        run("(list-slice (list 10 20 30 40 50) 5 0)").unwrap(),
        Value::List(vec![])
    );
}

#[test]
fn p4_list_slice_empty_list() {
    assert_eq!(run("(list-slice (list) 0 0)").unwrap(), Value::List(vec![]));
}

#[test]
fn p4_list_slice_out_of_bounds_overrun() {
    let err = run_err("(list-slice (list 10 20 30 40 50) 3 5)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::ListSliceOutOfBounds {
            start: 3,
            len: 5,
            list_length: 5,
        }
    );
}

#[test]
fn p4_list_slice_negative_start() {
    let err = run_err("(list-slice (list 10 20 30 40 50) -1 2)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::ListSliceOutOfBounds {
            start: -1,
            len: 2,
            list_length: 5,
        }
    );
}

#[test]
fn p4_list_slice_negative_len() {
    let err = run_err("(list-slice (list 10 20 30 40 50) 0 -1)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::ListSliceOutOfBounds {
            start: 0,
            len: -1,
            list_length: 5,
        }
    );
}

#[test]
fn p4_list_slice_start_beyond_end() {
    let err = run_err("(list-slice (list 10 20 30 40 50) 6 0)");
    assert_eq!(
        err.kind,
        InterpretErrorKind::ListSliceOutOfBounds {
            start: 6,
            len: 0,
            list_length: 5,
        }
    );
}

// ── Composition ──────────────────────────────────────────────

#[test]
fn p4_list_slice_after_append() {
    // (list-slice (list-append (list 10 20) 30) 1 2) → (list 20 30)
    assert_eq!(
        run("(list-slice (list-append (list 10 20) 30) 1 2)").unwrap(),
        Value::List(vec![20, 30])
    );
}

#[test]
fn p4_list_get_after_slice() {
    // (list-get (list-slice (list 10 20 30 40 50) 1 3) 1) → 30
    assert_eq!(
        run("(list-get (list-slice (list 10 20 30 40 50) 1 3) 1)").unwrap(),
        Value::Integer(30)
    );
}

#[test]
fn p4_list_len_after_slice() {
    assert_eq!(
        run("(list-len (list-slice (list 10 20 30 40 50) 1 3))").unwrap(),
        Value::Integer(3)
    );
}

#[test]
fn p4_list_slice_inside_function() {
    let src = "(program \
        (fn middle (list) list (list-slice %0 1 2)) \
        (call middle (list 1 2 3 4)))";
    assert_eq!(run(src).unwrap(), Value::List(vec![2, 3]));
}

#[test]
fn p4_list_slice_inside_function_propagates_oob() {
    let src = "(program \
        (fn take (list i64) list (list-slice %0 0 %1)) \
        (call take (list 10 20) 10))";
    let err = run_err(src);
    assert_eq!(
        err.kind,
        InterpretErrorKind::ListSliceOutOfBounds {
            start: 0,
            len: 10,
            list_length: 2,
        }
    );
}

// ── Phase 4 Step 4 — `cond` N-way Bool dispatch ──────────────

#[test]
fn p4_cond_default_only_returns_default_body() {
    assert_eq!(run("(cond (default 42))").unwrap(), Value::Integer(42));
    assert_eq!(type_stack("(cond (default 42))"), vec![ValueType::I64]);
}

#[test]
fn p4_cond_two_clause_match_first() {
    assert_eq!(
        run("(cond ((eq 1 1) 10) (default 20))").unwrap(),
        Value::Integer(10)
    );
}

#[test]
fn p4_cond_two_clause_falls_through_to_default() {
    assert_eq!(
        run("(cond ((eq 1 2) 10) (default 20))").unwrap(),
        Value::Integer(20)
    );
}

#[test]
fn p4_cond_three_clause_match_first() {
    assert_eq!(
        run("(cond ((eq 1 1) 10) ((eq 2 2) 20) (default 30))").unwrap(),
        Value::Integer(10)
    );
}

#[test]
fn p4_cond_three_clause_match_second() {
    assert_eq!(
        run("(cond ((eq 1 2) 10) ((eq 2 2) 20) (default 30))").unwrap(),
        Value::Integer(20)
    );
}

#[test]
fn p4_cond_three_clause_match_default() {
    assert_eq!(
        run("(cond ((eq 1 2) 10) ((eq 2 3) 20) (default 30))").unwrap(),
        Value::Integer(30)
    );
}

#[test]
fn p4_cond_first_match_wins_among_multiple_true() {
    // Both predicates would match; the first one's body must run.
    assert_eq!(
        run("(cond ((eq 1 1) 10) ((eq 2 2) 20) (default 30))").unwrap(),
        Value::Integer(10)
    );
}

#[test]
fn p4_cond_returns_bool() {
    assert_eq!(
        run("(cond ((eq 1 1) true) (default false))").unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        type_stack("(cond ((eq 1 1) true) (default false))"),
        vec![ValueType::Bool]
    );
}

#[test]
fn p4_cond_returns_bytes() {
    assert_eq!(
        run("(cond ((eq 1 1) #x42) (default #x00))").unwrap(),
        Value::Bytes(vec![0x42])
    );
    assert_eq!(
        type_stack("(cond ((eq 1 1) #x42) (default #x00))"),
        vec![ValueType::Bytes]
    );
}

#[test]
fn p4_cond_inside_function_dispatches_by_arg() {
    let src = "(program \
        (fn pick (i64) i64 \
            (cond ((eq %0 1) 10) ((eq %0 2) 20) (default 99))) \
        (call pick 2))";
    assert_eq!(run(src).unwrap(), Value::Integer(20));
}

#[test]
fn p4_cond_inside_function_falls_through_to_default() {
    let src = "(program \
        (fn pick (i64) i64 \
            (cond ((eq %0 1) 10) ((eq %0 2) 20) (default 99))) \
        (call pick 7))";
    assert_eq!(run(src).unwrap(), Value::Integer(99));
}

#[test]
fn p4_cond_nested_inside_body() {
    // Outer match-1 fires; its body is itself a cond.
    let src = "(cond \
        ((eq 1 1) (cond ((eq 0 0) 100) ((eq 1 1) 200) (default 300))) \
        (default 999))";
    assert_eq!(run(src).unwrap(), Value::Integer(100));
}

#[test]
fn p4_cond_nested_inside_default() {
    // Outer falls through to default; default is itself a cond.
    let src = "(cond \
        ((eq 1 2) 999) \
        (default (cond ((eq 1 1) 42) (default 0))))";
    assert_eq!(run(src).unwrap(), Value::Integer(42));
}

// ── Short-circuit: later clauses must NOT be evaluated ───────

#[test]
fn p4_cond_short_circuit_does_not_evaluate_later_predicate() {
    // The second clause's predicate divides by zero. If cond
    // weren't short-circuiting, this would raise DivisionByZero;
    // since clause-1 matches, predicate-2 is never evaluated.
    let src = "(cond \
        ((eq 1 1) 100) \
        ((eq (div 1 0) 0) 999) \
        (default 0))";
    assert_eq!(run(src).unwrap(), Value::Integer(100));
}

#[test]
fn p4_cond_short_circuit_does_not_evaluate_later_body() {
    // Body of the second clause is `(div 1 0)` — only the matched
    // body runs. The matched body is `100`, so this returns 100.
    let src = "(cond \
        ((eq 1 1) 100) \
        ((eq 0 0) (div 1 0)) \
        (default 0))";
    assert_eq!(run(src).unwrap(), Value::Integer(100));
}

#[test]
fn p4_cond_short_circuit_does_not_evaluate_default_body() {
    // Default body would divide by zero; first clause matches.
    let src = "(cond ((eq 1 1) 100) (default (div 1 0)))";
    assert_eq!(run(src).unwrap(), Value::Integer(100));
}

#[test]
fn p4_cond_evaluates_default_when_no_match() {
    // Default body runs only when no predicate matches.
    let src = "(cond ((eq 1 2) 1) ((eq 2 3) 2) (default 999))";
    assert_eq!(run(src).unwrap(), Value::Integer(999));
}

// ── Predicate runtime error propagates ─────────────────────────

#[test]
fn p4_cond_predicate_runtime_error_propagates() {
    // First predicate's evaluation crashes — must surface as
    // DivisionByZero, not be swallowed by cond dispatch.
    let src = "(cond ((eq (div 1 0) 0) 1) (default 0))";
    let err = run_err(src);
    assert_eq!(err.kind, InterpretErrorKind::DivisionByZero);
}

// ── Branch type mismatch is a type-check error ─────────────────

#[test]
fn p4_cond_branch_type_mismatch_default_differs() {
    // First clause body is I64; default body is Bytes — type check
    // rejects with BranchStackMismatch.
    let err = type_check_err("(cond ((eq 1 1) 1) (default #x00))");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::BranchStackMismatch { .. }
    ));
}

#[test]
fn p4_cond_branch_type_mismatch_between_nondefault_clauses() {
    // Two non-default clauses with different body types — rejected
    // by the type checker.
    let err = type_check_err("(cond ((eq 1 1) 1) ((eq 2 2) #x00) (default 0))");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::BranchStackMismatch { .. }
    ));
}

// ── Cond inside loop ────────────────────────────────────────────

#[test]
fn p4_cond_breaks_out_of_loop() {
    // Loop body is a cond; one clause breaks with a value, default
    // continues iterating (no-op via store). The loop terminates
    // when the break clause fires.
    let src = "(program \
        (fn drive (i64) bool \
            (loop (cond ((eq %0 %0) (break true)) (default false)))) \
        (call drive 1))";
    assert_eq!(run(src).unwrap(), Value::Bool(true));
}

// ── Phase 4 Step 5 — Maybe / Option type ─────────────────────────

#[test]
fn p4_some_wraps_integer() {
    assert_eq!(
        run("(some 42)").unwrap(),
        Value::Maybe(Some(Box::new(Value::Integer(42))))
    );
    assert_eq!(type_stack("(some 42)"), vec![ValueType::Maybe]);
}

#[test]
fn p4_none_creates_empty_maybe() {
    assert_eq!(run("(none)").unwrap(), Value::Maybe(None));
    assert_eq!(type_stack("(none)"), vec![ValueType::Maybe]);
}

#[test]
fn p4_some_then_is_some_is_true() {
    assert_eq!(run("(is-some (some 1))").unwrap(), Value::Bool(true));
    assert_eq!(type_stack("(is-some (some 1))"), vec![ValueType::Bool]);
}

#[test]
fn p4_none_then_is_some_is_false() {
    assert_eq!(run("(is-some (none))").unwrap(), Value::Bool(false));
}

#[test]
fn p4_some_then_is_none_is_false() {
    assert_eq!(run("(is-none (some 1))").unwrap(), Value::Bool(false));
}

#[test]
fn p4_none_then_is_none_is_true() {
    assert_eq!(run("(is-none (none))").unwrap(), Value::Bool(true));
    assert_eq!(type_stack("(is-none (none))"), vec![ValueType::Bool]);
}

#[test]
fn p4_unwrap_on_some_returns_inner() {
    assert_eq!(run("(unwrap (some 42))").unwrap(), Value::Integer(42));
    assert_eq!(type_stack("(unwrap (some 42))"), vec![ValueType::I64]);
}

#[test]
fn p4_unwrap_on_none_traps_unwrap_on_none() {
    let err = run_err("(unwrap (none))");
    assert_eq!(err.kind, InterpretErrorKind::UnwrapOnNone);
}

#[test]
fn p4_unwrap_or_on_some_returns_inner() {
    assert_eq!(run("(unwrap-or (some 42) 99)").unwrap(), Value::Integer(42));
    assert_eq!(type_stack("(unwrap-or (some 42) 99)"), vec![ValueType::I64]);
}

#[test]
fn p4_unwrap_or_on_none_returns_default() {
    assert_eq!(run("(unwrap-or (none) 99)").unwrap(), Value::Integer(99));
}

#[test]
fn p4_some_with_negative_integer() {
    assert_eq!(run("(unwrap (some -7))").unwrap(), Value::Integer(-7));
}

#[test]
fn p4_unwrap_or_uses_nested_default() {
    // default itself is a computed expression
    assert_eq!(
        run("(unwrap-or (none) (add 1 2))").unwrap(),
        Value::Integer(3)
    );
}

#[test]
fn p4_some_inside_function() {
    let src = "(program \
        (fn wrap (i64) maybe (some %0)) \
        (call wrap 7))";
    assert_eq!(
        run(src).unwrap(),
        Value::Maybe(Some(Box::new(Value::Integer(7))))
    );
}

#[test]
fn p4_unwrap_or_inside_function() {
    let src = "(program \
        (fn safe-get (maybe i64) i64 (unwrap-or %0 %1)) \
        (call safe-get (none) 42))";
    assert_eq!(run(src).unwrap(), Value::Integer(42));
}

#[test]
fn p4_unwrap_or_inside_function_uses_some() {
    let src = "(program \
        (fn safe-get (maybe i64) i64 (unwrap-or %0 %1)) \
        (call safe-get (some 7) 42))";
    assert_eq!(run(src).unwrap(), Value::Integer(7));
}

#[test]
fn p4_maybe_in_cond_branches() {
    // A cond producing Maybe values: both branches must produce
    // Maybe for the type checker to accept the program.
    assert_eq!(
        run("(cond ((eq 1 1) (some 10)) (default (none)))").unwrap(),
        Value::Maybe(Some(Box::new(Value::Integer(10))))
    );
    assert_eq!(
        type_stack("(cond ((eq 1 1) (some 10)) (default (none)))"),
        vec![ValueType::Maybe]
    );
}

#[test]
fn p4_maybe_branch_mismatch_is_type_error() {
    // One branch Maybe, the other I64 — branch stack mismatch.
    let err = type_check_err("(cond ((eq 1 1) (some 1)) (default 0))");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::BranchStackMismatch { .. }
    ));
}

#[test]
fn p4_is_some_dispatches_to_unwrap_via_if() {
    // Typical safe-unwrap pattern: only unwrap after is-some.
    let src = "(if (is-some (some 5)) (unwrap (some 5)) -1)";
    assert_eq!(run(src).unwrap(), Value::Integer(5));
}

#[test]
fn p4_is_none_dispatches_to_default_via_if() {
    // Inverse safe-unwrap pattern.
    let src = "(if (is-none (none)) 99 (unwrap (none)))";
    assert_eq!(run(src).unwrap(), Value::Integer(99));
}

#[test]
fn p4_some_inner_value_is_evaluated_expression() {
    // (some (add 1 2)) wraps the result of evaluating the inner.
    assert_eq!(run("(unwrap (some (add 1 2)))").unwrap(), Value::Integer(3));
}

#[test]
fn p4_unwrap_on_none_inside_function_propagates() {
    let src = "(program \
        (fn force () i64 (unwrap (none))) \
        (call force))";
    let err = run_err(src);
    assert_eq!(err.kind, InterpretErrorKind::UnwrapOnNone);
}

// ── Type-checker rejection tests ──────────────────────────────────

#[test]
fn p4_is_some_on_integer_is_type_error() {
    let err = type_check_err("(is-some 5)");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::TypeMismatch { .. }
    ));
}

#[test]
fn p4_is_none_on_integer_is_type_error() {
    let err = type_check_err("(is-none 5)");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::TypeMismatch { .. }
    ));
}

#[test]
fn p4_unwrap_on_integer_is_type_error() {
    let err = type_check_err("(unwrap 5)");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::TypeMismatch { .. }
    ));
}

#[test]
fn p4_unwrap_or_default_non_i64_is_type_error() {
    // Default arg must be I64. Bytes here ⇒ type mismatch.
    let err = type_check_err("(unwrap-or (some 1) #x00)");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::TypeMismatch { .. }
    ));
}

#[test]
fn p4_some_inner_non_i64_is_type_error() {
    // Phase 4 monomorphic — `some` wraps i64 only.
    let err = type_check_err("(some #x00)");
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::TypeMismatch { .. }
    ));
}

// ── Phase 4 Step 6 — Nominal Structs ──────────────────────────────

#[test]
fn p4_struct_basic_create_and_access() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (fn dist () i64 \
            (let %0 (struct-new point 3 4) \
                (add (struct-get %0 x) (struct-get %0 y)))) \
        (call dist))";
    assert_eq!(run(src).unwrap(), Value::Integer(7));
}

#[test]
fn p4_struct_two_field_returns_struct() {
    // Function that returns a struct.
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (fn make-point () point (struct-new point 10 20)) \
        (fn sum () i64 \
            (let %0 (call make-point) \
                (add (struct-get %0 x) (struct-get %0 y)))) \
        (call sum))";
    assert_eq!(run(src).unwrap(), Value::Integer(30));
}

#[test]
fn p4_struct_single_field() {
    let src = "(program \
        (struct wrapper ((v i64))) \
        (fn unwrap-wrap () i64 (struct-get (struct-new wrapper 99) v)) \
        (call unwrap-wrap))";
    assert_eq!(run(src).unwrap(), Value::Integer(99));
}

#[test]
fn p4_struct_three_fields_access_each() {
    let src = "(program \
        (struct triple ((a i64) (b i64) (c i64))) \
        (fn middle () i64 \
            (let %0 (struct-new triple 1 2 3) \
                (struct-get %0 b))) \
        (call middle))";
    assert_eq!(run(src).unwrap(), Value::Integer(2));
}

#[test]
fn p4_struct_set_functional_update() {
    // struct-set returns a new struct; original is unchanged.
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (fn shifted () i64 \
            (let %0 (struct-new point 1 2) \
                (let %1 (struct-set %0 x 100) \
                    (add (struct-get %1 x) (struct-get %0 x))))) \
        (call shifted))";
    // %1.x = 100, %0.x = 1 (unchanged), sum = 101.
    assert_eq!(run(src).unwrap(), Value::Integer(101));
}

#[test]
fn p4_struct_in_cond_branches() {
    // Both branches must produce the same struct type.
    let src = "(program \
        (struct pair ((a i64) (b i64))) \
        (fn pick (i64) i64 \
            (let %1 (cond \
                ((eq %0 0) (struct-new pair 10 20)) \
                (default (struct-new pair 30 40))) \
                (struct-get %1 a))) \
        (call pick 0))";
    assert_eq!(run(src).unwrap(), Value::Integer(10));
}

#[test]
fn p4_struct_field_access_chain_via_set() {
    let src = "(program \
        (struct counter ((n i64))) \
        (fn step () i64 \
            (let %0 (struct-new counter 5) \
                (let %1 (struct-set %0 n (add (struct-get %0 n) 1)) \
                    (struct-get %1 n)))) \
        (call step))";
    assert_eq!(run(src).unwrap(), Value::Integer(6));
}

#[test]
fn p4_struct_as_fn_param() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (fn sum-pt (point) i64 \
            (add (struct-get %0 x) (struct-get %0 y))) \
        (call sum-pt (struct-new point 7 8)))";
    assert_eq!(run(src).unwrap(), Value::Integer(15));
}

// ── Type-checker rejection tests ──────────────────────────────────

#[test]
fn p4_struct_field_count_mismatch_too_few() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (struct-new point 1))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::StructFieldCountMismatch { .. }
    ));
}

#[test]
fn p4_struct_field_count_mismatch_too_many() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (struct-new point 1 2 3))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::StructFieldCountMismatch { .. }
    ));
}

#[test]
fn p4_struct_field_type_mismatch() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (struct-new point 1 #x00))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::TypeMismatch { .. }
    ));
}

#[test]
fn p4_struct_unknown_struct_name() {
    let src = "(program (struct-new ghost 1 2))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::UnknownStructError { .. }
    ));
}

#[test]
fn p4_struct_get_unknown_field() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (struct-get (struct-new point 1 2) z))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::UnknownFieldError { .. }
    ));
}

#[test]
fn p4_struct_set_unknown_field() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (struct-set (struct-new point 1 2) z 99))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::UnknownFieldError { .. }
    ));
}

#[test]
fn p4_struct_set_field_type_mismatch() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (struct-set (struct-new point 1 2) x #x00))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::TypeMismatch { .. }
    ));
}

#[test]
fn p4_struct_get_on_non_struct() {
    // struct-get target is an i64, not a Struct.
    let src = "(struct-get 5 x)";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::TypeMismatch { .. }
    ));
}

#[test]
fn p4_struct_recursive_self_reference_rejected() {
    let src = "(program (struct node ((next node))) 0)";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::RecursiveStructError { .. }
    ));
}

#[test]
fn p4_struct_duplicate_definition_rejected() {
    let src = "(program \
        (struct foo ((x i64))) \
        (struct foo ((y i64))) \
        0)";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::DuplicateStructError { .. }
    ));
}

#[test]
fn p4_struct_inline_declaration_rejected() {
    // struct declarations must appear at program top level only.
    let src = "(program (add 1 (struct foo ((x i64)))))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::InvalidProgramStructureError
    ));
}

#[test]
fn p4_struct_two_types_compared_separately() {
    // Two structs with identical fields are still distinct nominal
    // types — the validator rejects passing a `b` where `a` is
    // expected.
    let src = "(program \
        (struct a ((x i64))) \
        (struct b ((x i64))) \
        (fn take-a (a) i64 (struct-get %0 x)) \
        (call take-a (struct-new b 5)))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::FunctionTypeMismatch { .. }
    ));
}

#[test]
fn p4_struct_forward_reference_rejected() {
    // Structs must be declared in source order: b cannot reference a
    // if a comes after.
    let src = "(program \
        (struct b ((next a))) \
        (struct a ((x i64))) \
        0)";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::UnknownTypeError { .. }
    ));
}

#[test]
fn p4_struct_nested_struct_field() {
    // A struct field can reference a previously-defined struct.
    let src = "(program \
        (struct inner ((v i64))) \
        (struct outer ((i inner) (k i64))) \
        (fn build () i64 \
            (let %0 (struct-new outer (struct-new inner 42) 7) \
                (add (struct-get (struct-get %0 i) v) (struct-get %0 k)))) \
        (call build))";
    assert_eq!(run(src).unwrap(), Value::Integer(49));
}

// ── Determinism / round-trip ─────────────────────────────────────

#[test]
fn p4_struct_determinism_same_program_same_value() {
    // Running the same struct program twice yields bitwise-identical
    // Values. The BTreeMap-based field carrier makes field iteration
    // deterministic; the i64 elements provide stable equality.
    let src = "(program \
        (struct triple ((a i64) (b i64) (c i64))) \
        (fn sum () i64 \
            (let %0 (struct-new triple 100 200 300) \
                (add (add (struct-get %0 a) (struct-get %0 b)) (struct-get %0 c)))) \
        (call sum))";
    let v1 = run(src).unwrap();
    let v2 = run(src).unwrap();
    assert_eq!(v1, v2);
    assert_eq!(v1, Value::Integer(600));
}

// ──────────────────────────────────────────────────────────────
// Phase 4 Step 7 — Pattern matching (`match`)
// ──────────────────────────────────────────────────────────────

// ── Happy path: Maybe matching ───────────────────────────────────

#[test]
fn p4_match_maybe_some_binds_inner() {
    // (some 5) matches (case (some %0) (add %0 1)) → 6
    let src = "(program \
        (fn f () i64 \
            (match (some 5) \
                (case (some %0) (add %0 1)) \
                (case (none) 0) \
                (case _ -1))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(6));
}

#[test]
fn p4_match_maybe_none_takes_none_branch() {
    let src = "(program \
        (fn f () i64 \
            (match (none) \
                (case (some %0) (add %0 1)) \
                (case (none) 42) \
                (case _ -1))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(42));
}

#[test]
fn p4_match_maybe_falls_through_to_wildcard() {
    // Without an explicit (none) case, the wildcard catches None.
    let src = "(program \
        (fn f () i64 \
            (match (none) \
                (case (some %0) %0) \
                (case _ 99))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(99));
}

#[test]
fn p4_match_some_negative_inner() {
    let src = "(program \
        (fn f () i64 \
            (match (some -7) \
                (case (some %0) %0) \
                (case _ 0))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(-7));
}

// ── Happy path: struct destructuring ─────────────────────────────

#[test]
fn p4_match_struct_destructure_binds_fields() {
    // (struct-new point 3 4) → match (struct point %0 %1) binds x=3, y=4.
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (fn f () i64 \
            (match (struct-new point 3 4) \
                (case (struct point %0 %1) (add %0 %1)) \
                (case _ -1))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(7));
}

#[test]
fn p4_match_struct_three_fields() {
    let src = "(program \
        (struct triple ((a i64) (b i64) (c i64))) \
        (fn f () i64 \
            (match (struct-new triple 10 20 30) \
                (case (struct triple %0 %1 %2) (add %0 (add %1 %2))) \
                (case _ -1))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(60));
}

#[test]
fn p4_match_struct_field_order_non_alphabetical() {
    // Field declaration order is z, a — non-alphabetical. The
    // pattern binds in DECLARED order: %0=z, %1=a. This exercises
    // the SymbolTable lookup path that resolves declared field
    // order from `StructDef::fields`, which would fail if the
    // matcher iterated the BTreeMap alphabetically.
    let src = "(program \
        (struct frob ((z i64) (a i64))) \
        (fn f () i64 \
            (match (struct-new frob 100 1) \
                (case (struct frob %0 %1) (sub %0 %1)) \
                (case _ -1))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(99));
}

// ── Happy path: integer-literal matching ─────────────────────────

#[test]
fn p4_match_integer_literal_exact() {
    let src = "(program \
        (fn classify (i64) i64 \
            (match %0 \
                (case 0 1000) \
                (case 1 1001) \
                (case 2 1002) \
                (case _ 9999))) \
        (call classify 1))";
    assert_eq!(run(src).unwrap(), Value::Integer(1001));
}

#[test]
fn p4_match_integer_literal_falls_through_to_wildcard() {
    let src = "(program \
        (fn classify (i64) i64 \
            (match %0 \
                (case 0 1000) \
                (case 1 1001) \
                (case _ 9999))) \
        (call classify 42))";
    assert_eq!(run(src).unwrap(), Value::Integer(9999));
}

#[test]
fn p4_match_negative_integer_literal() {
    let src = "(program \
        (fn classify (i64) i64 \
            (match %0 \
                (case -1 100) \
                (case 0 200) \
                (case _ 300))) \
        (call classify -1))";
    assert_eq!(run(src).unwrap(), Value::Integer(100));
}

// ── Happy path: wildcard-only ────────────────────────────────────

#[test]
fn p4_match_wildcard_only() {
    let src = "(match 42 (case _ 100))";
    assert_eq!(run(src).unwrap(), Value::Integer(100));
}

// ── Type-stack output ────────────────────────────────────────────

#[test]
fn p4_match_pushes_branch_type() {
    let src = "(program \
        (fn f () i64 \
            (match (some 1) \
                (case (some %0) %0) \
                (case _ 0))) \
        (call f))";
    assert_eq!(type_stack(src), vec![ValueType::I64]);
}

// ── Validator structural rejections ──────────────────────────────

fn validate_structural_err(src: &str) -> quarks_validator::ValidationError {
    let ast = parse(src).expect("parse");
    quarks_validator::validate_structure(&ast).expect_err("structural should fail")
}

#[test]
fn p4_match_missing_wildcard_is_structural_error() {
    let err = validate_structural_err("(match (some 1) (case (some %0) %0) (case (none) 0))");
    assert!(matches!(
        err.kind,
        quarks_validator::ValidationErrorKind::MatchMalformed {
            message_kind: quarks_validator::MatchMalformedKind::MissingWildcard,
        }
    ));
}

#[test]
fn p4_match_no_cases_is_structural_error() {
    let err = validate_structural_err("(match 42)");
    assert!(matches!(
        err.kind,
        quarks_validator::ValidationErrorKind::MatchMalformed {
            message_kind: quarks_validator::MatchMalformedKind::NoCases,
        }
    ));
}

#[test]
fn p4_match_no_scrutinee_is_structural_error() {
    let err = validate_structural_err("(match)");
    assert!(matches!(
        err.kind,
        quarks_validator::ValidationErrorKind::MatchMalformed {
            message_kind: quarks_validator::MatchMalformedKind::MissingScrutinee,
        }
    ));
}

#[test]
fn p4_match_wildcard_not_last_is_structural_error() {
    let err = validate_structural_err("(match 42 (case _ 1) (case 0 2))");
    assert!(matches!(
        err.kind,
        quarks_validator::ValidationErrorKind::MatchMalformed {
            message_kind: quarks_validator::MatchMalformedKind::WildcardNotLast { .. },
        }
    ));
}

#[test]
fn p4_match_case_wrong_arity_is_structural_error() {
    let err = validate_structural_err("(match 42 (case 0) (case _ 1))");
    assert!(matches!(
        err.kind,
        quarks_validator::ValidationErrorKind::MatchMalformed {
            message_kind: quarks_validator::MatchMalformedKind::CaseWrongArity { .. },
        }
    ));
}

#[test]
fn p4_match_case_head_not_case_is_structural_error() {
    // First clause is a 3-element list whose head is `clause`,
    // not `case`. Arity matches the (case pattern body) shape, so
    // the head-keyword check fires.
    let err = validate_structural_err("(match 42 (clause 0 1) (case _ 0))");
    assert!(matches!(
        err.kind,
        quarks_validator::ValidationErrorKind::MatchMalformed {
            message_kind: quarks_validator::MatchMalformedKind::CaseHeadNotCaseKeyword { .. },
        }
    ));
}

#[test]
fn p4_match_unrecognised_pattern_is_structural_error() {
    // (case (foo %0) body) — `foo` is not a recognised pattern head.
    let err = validate_structural_err("(match 42 (case (foo %0) 0) (case _ 0))");
    assert!(matches!(
        err.kind,
        quarks_validator::ValidationErrorKind::MatchMalformed {
            message_kind: quarks_validator::MatchMalformedKind::UnrecognisedPattern { .. },
        }
    ));
}

// ── Type-checker rejections ──────────────────────────────────────

#[test]
fn p4_match_pattern_type_mismatch_some_on_integer() {
    // (some %0) against an i64 scrutinee → type error.
    let src = "(match 42 \
                (case (some %0) %0) \
                (case _ 0))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::MatchPatternTypeMismatch { .. }
    ));
}

#[test]
fn p4_match_pattern_type_mismatch_integer_on_maybe() {
    let src = "(program \
        (fn f () i64 \
            (match (some 1) \
                (case 0 100) \
                (case _ 200))) \
        (call f))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::MatchPatternTypeMismatch { .. }
    ));
}

#[test]
fn p4_match_pattern_type_mismatch_struct_on_integer() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (fn f () i64 \
            (match 42 \
                (case (struct point %0 %1) (add %0 %1)) \
                (case _ 0))) \
        (call f))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::MatchPatternTypeMismatch { .. }
    ));
}

#[test]
fn p4_match_unknown_struct_in_pattern() {
    let src = "(program \
        (fn f () i64 \
            (match 0 \
                (case (struct ghost %0) %0) \
                (case _ 0))) \
        (call f))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::MatchUnknownStructError { .. }
    ));
}

#[test]
fn p4_match_struct_field_count_mismatch() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (fn f () i64 \
            (match (struct-new point 1 2) \
                (case (struct point %0) %0) \
                (case _ 0))) \
        (call f))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::MatchStructFieldCountMismatch { .. }
    ));
}

#[test]
fn p4_match_branch_stack_mismatch() {
    // Two cases produce different residual types — branch mismatch.
    let src = "(program \
        (fn f () i64 \
            (match (some 1) \
                (case (some %0) true) \
                (case _ 0))) \
        (call f))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::BranchStackMismatch { .. }
    ));
}

#[test]
fn p4_match_some_bind_non_parameter_is_error() {
    // (case (some 42) body) — bind slot is an integer literal,
    // not a %n.
    let src = "(program \
        (fn f () i64 \
            (match (some 1) \
                (case (some 42) 100) \
                (case _ 0))) \
        (call f))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::MatchSomeBindNotParameter { .. }
    ));
}

#[test]
fn p4_match_struct_field_non_parameter_is_error() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (fn f () i64 \
            (match (struct-new point 1 2) \
                (case (struct point 42 %1) %1) \
                (case _ 0))) \
        (call f))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::MatchStructFieldNotParameter { .. }
    ));
}

#[test]
fn p4_match_binding_collides_with_parameter() {
    // fn `f` declares one i64 parameter (%0). The pattern tries
    // to rebind %0.
    let src = "(program \
        (fn f (i64) i64 \
            (match (some 5) \
                (case (some %0) %0) \
                (case _ 0))) \
        (call f 7))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::MatchBindingCollidesWithParameter { .. }
    ));
}

#[test]
fn p4_match_struct_pattern_binds_same_slot_twice() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (fn f () i64 \
            (match (struct-new point 1 2) \
                (case (struct point %0 %0) %0) \
                (case _ 0))) \
        (call f))";
    let err = type_check_err(src);
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::MatchBindingRedefinition { .. }
    ));
}

// ── Scope discipline ─────────────────────────────────────────────

#[test]
fn p4_match_binding_scope_does_not_leak() {
    // %1 bound inside the some-case must not be visible outside.
    // Same regression-guard pattern as the let scope test.
    let src = "(program \
        (fn f () i64 \
            (add \
                (match (some 5) \
                    (case (some %1) %1) \
                    (case _ 0)) \
                %1)) \
        (call f))";
    let err = type_check_err(src);
    // The second `%1` reads outside any binding → InvalidParameterIndex.
    assert!(matches!(
        err.kind,
        quarks_validator::TypeCheckErrorKind::InvalidParameterIndex { .. }
    ));
}

#[test]
fn p4_match_binding_can_be_reused_across_cases() {
    // Each case has its own scope — binding %0 in two non-overlapping
    // cases is fine (cases are siblings, not nested).
    let src = "(program \
        (fn f () i64 \
            (match (some 5) \
                (case (some %0) %0) \
                (case (none) 0) \
                (case _ -1))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(5));
}

// ── Integration: match inside let / if / cond ────────────────────

#[test]
fn p4_match_inside_let_body() {
    let src = "(program \
        (fn f () i64 \
            (let %0 (some 10) \
                (match %0 \
                    (case (some %1) (add %1 1)) \
                    (case _ 0)))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(11));
}

#[test]
fn p4_match_inside_if_branch() {
    let src = "(program \
        (fn f () i64 \
            (if true \
                (match (some 5) (case (some %0) %0) (case _ 0)) \
                (match (none)   (case (some %0) %0) (case _ 0)))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(5));
}

#[test]
fn p4_match_as_scrutinee_of_match() {
    // match-of-match: outer scrutinee is the result of an inner
    // match. Stresses scope cleanup between nested matches.
    let src = "(program \
        (fn f () i64 \
            (match (match (some 5) (case (some %0) %0) (case _ 0)) \
                (case 5 1000) \
                (case _ 9999))) \
        (call f))";
    assert_eq!(run(src).unwrap(), Value::Integer(1000));
}

// ── Match on fn parameter / fn return ────────────────────────────

#[test]
fn p4_match_on_fn_parameter() {
    let src = "(program \
        (fn at (i64) i64 \
            (match %0 \
                (case 0 100) \
                (case 1 200) \
                (case _ 300))) \
        (add (call at 0) (add (call at 1) (call at 99))))";
    assert_eq!(run(src).unwrap(), Value::Integer(100 + 200 + 300));
}

#[test]
fn p4_match_in_fn_returning_maybe() {
    let src = "(program \
        (fn split (i64) maybe \
            (match %0 \
                (case 0 (none)) \
                (case _ (some %0)))) \
        (unwrap-or (call split 7) -1))";
    assert_eq!(run(src).unwrap(), Value::Integer(7));
}

// ── Underscore symbol rules ──────────────────────────────────────

#[test]
fn p4_wildcard_underscore_is_legal_in_pattern() {
    // Confirms the reserved-symbol pre-scan was relaxed to allow
    // bare `_` while still rejecting `_foo`.
    let src = "(match 42 (case _ 1))";
    assert_eq!(run(src).unwrap(), Value::Integer(1));
}

#[test]
fn p4_multi_char_underscore_prefix_still_reserved() {
    // `_foo` in any position remains a reserved-symbol error —
    // the relaxation was narrow.
    let src = "(match _foo (case _ 1))";
    let ast = parse(src).expect("parse");
    let structural = validate_structure(&ast);
    // Structural pass doesn't catch reserved symbols (that's the
    // type-checker's job). Type-check must reject.
    let _ = structural; // structural ok or pattern-related error; we don't depend on it
    let tc_err = type_check(&ast).expect_err("type-check must reject _foo");
    assert!(matches!(
        tc_err.kind,
        quarks_validator::TypeCheckErrorKind::ReservedSymbolError { .. }
    ));
}

// ── Property-style: integer dispatch table ───────────────────────

#[test]
fn p4_match_integer_dispatch_table_property() {
    // For each input in 0..=4, the matching case body produces a
    // distinct labelled output; other inputs fall through to the
    // wildcard. This is the LLM's typical N-way integer dispatch
    // shape and should round-trip 1-for-1.
    let make = |n: i64| {
        std::format!(
            "(program \
                (fn dispatch (i64) i64 \
                    (match %0 \
                        (case 0 1000) \
                        (case 1 1001) \
                        (case 2 1002) \
                        (case 3 1003) \
                        (case _ 9999))) \
                (call dispatch {}))",
            n
        )
    };
    for n in 0..=3i64 {
        let v = run(&make(n)).unwrap();
        assert_eq!(v, Value::Integer(1000 + n));
    }
    for n in [4i64, 5, -1, 100] {
        let v = run(&make(n)).unwrap();
        assert_eq!(v, Value::Integer(9999));
    }
}

// ── Determinism ──────────────────────────────────────────────────

#[test]
fn p4_match_determinism_same_program_same_value() {
    let src = "(program \
        (struct point ((x i64) (y i64))) \
        (fn area () i64 \
            (match (struct-new point 6 7) \
                (case (struct point %0 %1) (mul %0 %1)) \
                (case _ 0))) \
        (call area))";
    let v1 = run(src).unwrap();
    let v2 = run(src).unwrap();
    assert_eq!(v1, v2);
    assert_eq!(v1, Value::Integer(42));
}
