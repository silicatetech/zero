// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 4 Step 1 — property-based tests for the bitwise op family.
//!
//! Invariants asserted:
//!
//! 1. **Host parity.** `bit-and`/`bit-or`/`bit-xor` evaluator output
//!    matches Rust's `&`/`|`/`^` on the same `i64` pair, for arbitrary
//!    inputs.
//! 2. **Shifts match Rust.** For shift counts in `0..64`, `bit-shl`
//!    matches `i64::wrapping_shl(.. as u32)` and `bit-shr` matches
//!    `i64::wrapping_shr(.. as u32)` (arithmetic semantics).
//! 3. **Shift out-of-range surfaces a typed error**, never a panic
//!    or x86 low-6-bit masking.
//! 4. **Identity laws.** `x AND -1 = x`, `x OR 0 = x`, `x XOR 0 = x`,
//!    `x AND 0 = 0`, `x OR -1 = -1`.
//! 5. **XOR self-inverse.** `(x XOR y) XOR y = x` for all `x`, `y`.
//! 6. **Commutativity.** `a OP b = b OP a` for AND, OR, XOR.
//! 7. **Determinism.** Two independent runs of the same program yield
//!    identical results — including the shift error path.
//!
//! See `property_tests_paket_b.rs` for the comparable Bool/cmp suite.

use proptest::prelude::*;
use quarks_interpreter::{interpret, InterpretErrorKind, Value};
use quarks_validator::parse;

fn run(src: &str) -> Value {
    let ast = parse(src).expect("parse");
    interpret(&ast).expect("interpret")
}

// ── Host parity on AND/OR/XOR ──────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 4096,
        .. ProptestConfig::default()
    })]

    #[test]
    fn p4_bit_and_matches_host(a in any::<i64>(), b in any::<i64>()) {
        let src = format!("(bit-and {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Integer(a & b));
    }

    #[test]
    fn p4_bit_or_matches_host(a in any::<i64>(), b in any::<i64>()) {
        let src = format!("(bit-or {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Integer(a | b));
    }

    #[test]
    fn p4_bit_xor_matches_host(a in any::<i64>(), b in any::<i64>()) {
        let src = format!("(bit-xor {} {})", a, b);
        prop_assert_eq!(run(&src), Value::Integer(a ^ b));
    }

    // ── Shifts: in-range matches host wrapping shifts ──────────

    #[test]
    fn p4_bit_shl_matches_host(a in any::<i64>(), n in 0i64..64) {
        let src = format!("(bit-shl {} {})", a, n);
        prop_assert_eq!(
            run(&src),
            Value::Integer(a.wrapping_shl(n as u32))
        );
    }

    #[test]
    fn p4_bit_shr_matches_host(a in any::<i64>(), n in 0i64..64) {
        // `i64::wrapping_shr` is arithmetic (SAR); matches the
        // interpreter's `i64 >>` semantics.
        let src = format!("(bit-shr {} {})", a, n);
        prop_assert_eq!(
            run(&src),
            Value::Integer(a.wrapping_shr(n as u32))
        );
    }

    // ── Shift counts outside 0..64 surface a typed error ───────

    #[test]
    fn p4_bit_shl_out_of_range_errors(
        a in any::<i64>(),
        n in prop_oneof![
            (i64::MIN..0),
            (64i64..=i64::MAX),
        ],
    ) {
        let src = format!("(bit-shl {} {})", a, n);
        let ast = parse(&src).expect("parse");
        let err = interpret(&ast).expect_err("must error");
        prop_assert_eq!(
            err.kind,
            InterpretErrorKind::ShiftCountOutOfRange { count: n }
        );
    }

    #[test]
    fn p4_bit_shr_out_of_range_errors(
        a in any::<i64>(),
        n in prop_oneof![
            (i64::MIN..0),
            (64i64..=i64::MAX),
        ],
    ) {
        let src = format!("(bit-shr {} {})", a, n);
        let ast = parse(&src).expect("parse");
        let err = interpret(&ast).expect_err("must error");
        prop_assert_eq!(
            err.kind,
            InterpretErrorKind::ShiftCountOutOfRange { count: n }
        );
    }

    // ── Identity laws ──────────────────────────────────────────

    #[test]
    fn p4_and_with_all_ones_is_identity(x in any::<i64>()) {
        let src = format!("(bit-and {} -1)", x);
        prop_assert_eq!(run(&src), Value::Integer(x));
    }

    #[test]
    fn p4_or_with_zero_is_identity(x in any::<i64>()) {
        let src = format!("(bit-or {} 0)", x);
        prop_assert_eq!(run(&src), Value::Integer(x));
    }

    #[test]
    fn p4_xor_with_zero_is_identity(x in any::<i64>()) {
        let src = format!("(bit-xor {} 0)", x);
        prop_assert_eq!(run(&src), Value::Integer(x));
    }

    #[test]
    fn p4_and_with_zero_is_zero(x in any::<i64>()) {
        let src = format!("(bit-and {} 0)", x);
        prop_assert_eq!(run(&src), Value::Integer(0));
    }

    #[test]
    fn p4_or_with_all_ones_is_all_ones(x in any::<i64>()) {
        let src = format!("(bit-or {} -1)", x);
        prop_assert_eq!(run(&src), Value::Integer(-1));
    }

    // ── XOR is its own inverse ─────────────────────────────────

    #[test]
    fn p4_xor_self_inverse(x in any::<i64>(), y in any::<i64>()) {
        // (x XOR y) XOR y == x
        let src = format!("(bit-xor (bit-xor {} {}) {})", x, y, y);
        prop_assert_eq!(run(&src), Value::Integer(x));
    }

    // ── Commutativity ──────────────────────────────────────────

    #[test]
    fn p4_and_commutative(a in any::<i64>(), b in any::<i64>()) {
        let left = format!("(bit-and {} {})", a, b);
        let right = format!("(bit-and {} {})", b, a);
        prop_assert_eq!(run(&left), run(&right));
    }

    #[test]
    fn p4_or_commutative(a in any::<i64>(), b in any::<i64>()) {
        let left = format!("(bit-or {} {})", a, b);
        let right = format!("(bit-or {} {})", b, a);
        prop_assert_eq!(run(&left), run(&right));
    }

    #[test]
    fn p4_xor_commutative(a in any::<i64>(), b in any::<i64>()) {
        let left = format!("(bit-xor {} {})", a, b);
        let right = format!("(bit-xor {} {})", b, a);
        prop_assert_eq!(run(&left), run(&right));
    }

    // ── Determinism: two runs of the same program ──────────────

    #[test]
    fn p4_bit_and_determinism(a in any::<i64>(), b in any::<i64>()) {
        let src = format!("(bit-and {} {})", a, b);
        prop_assert_eq!(run(&src), run(&src));
    }

    #[test]
    fn p4_bit_shl_determinism(a in any::<i64>(), n in 0i64..64) {
        let src = format!("(bit-shl {} {})", a, n);
        prop_assert_eq!(run(&src), run(&src));
    }

    #[test]
    fn p4_shift_error_determinism(a in any::<i64>(), n in 64i64..=128) {
        // The error path is also deterministic.
        let src = format!("(bit-shl {} {})", a, n);
        let ast = parse(&src).expect("parse");
        let e1 = interpret(&ast).expect_err("must error");
        let e2 = interpret(&ast).expect_err("must error");
        prop_assert_eq!(e1.kind, e2.kind);
    }

    // ─────────────────────────────────────────────────────────
    // Phase 4 Step 2 — Bytes operations
    // ─────────────────────────────────────────────────────────

    // ── Determinism: two runs produce identical results ───────

    #[test]
    fn p4_bytes_from_int_determinism(n in any::<i64>()) {
        let src = format!("(bytes-from-int {})", n);
        prop_assert_eq!(run(&src), run(&src));
    }

    #[test]
    fn p4_bytes_concat_determinism(
        a in proptest::collection::vec(any::<u8>(), 0..32),
        b in proptest::collection::vec(any::<u8>(), 0..32),
    ) {
        let src = format!(
            "(bytes-concat {} {})",
            bytes_literal(&a),
            bytes_literal(&b),
        );
        prop_assert_eq!(run(&src), run(&src));
    }

    // ── bytes-len after bytes-concat: len(a ++ b) == len(a) + len(b) ──

    #[test]
    fn p4_bytes_concat_length_sums(
        a in proptest::collection::vec(any::<u8>(), 0..64),
        b in proptest::collection::vec(any::<u8>(), 0..64),
    ) {
        let src = format!(
            "(bytes-len (bytes-concat {} {}))",
            bytes_literal(&a),
            bytes_literal(&b),
        );
        prop_assert_eq!(run(&src), Value::Integer((a.len() + b.len()) as i64));
    }

    // ── bytes-eq reflexivity: bytes-eq(x, x) == true ──────────

    #[test]
    fn p4_bytes_eq_reflexive(x in proptest::collection::vec(any::<u8>(), 0..64)) {
        let lit = bytes_literal(&x);
        let src = format!("(bytes-eq {} {})", lit, lit);
        prop_assert_eq!(run(&src), Value::Bool(true));
    }

    // ── bytes-from-int roundtrip: each byte matches LE encoding ──

    #[test]
    fn p4_bytes_from_int_le_byte0(n in any::<i64>()) {
        let expected = n.to_le_bytes()[0] as i64;
        let src = format!("(bytes-get (bytes-from-int {}) 0)", n);
        prop_assert_eq!(run(&src), Value::Integer(expected));
    }

    #[test]
    fn p4_bytes_from_int_le_byte1(n in any::<i64>()) {
        let expected = n.to_le_bytes()[1] as i64;
        let src = format!("(bytes-get (bytes-from-int {}) 1)", n);
        prop_assert_eq!(run(&src), Value::Integer(expected));
    }

    #[test]
    fn p4_bytes_from_int_le_byte2(n in any::<i64>()) {
        let expected = n.to_le_bytes()[2] as i64;
        let src = format!("(bytes-get (bytes-from-int {}) 2)", n);
        prop_assert_eq!(run(&src), Value::Integer(expected));
    }

    #[test]
    fn p4_bytes_from_int_le_byte3(n in any::<i64>()) {
        let expected = n.to_le_bytes()[3] as i64;
        let src = format!("(bytes-get (bytes-from-int {}) 3)", n);
        prop_assert_eq!(run(&src), Value::Integer(expected));
    }

    #[test]
    fn p4_bytes_from_int_le_byte4(n in any::<i64>()) {
        let expected = n.to_le_bytes()[4] as i64;
        let src = format!("(bytes-get (bytes-from-int {}) 4)", n);
        prop_assert_eq!(run(&src), Value::Integer(expected));
    }

    #[test]
    fn p4_bytes_from_int_le_byte5(n in any::<i64>()) {
        let expected = n.to_le_bytes()[5] as i64;
        let src = format!("(bytes-get (bytes-from-int {}) 5)", n);
        prop_assert_eq!(run(&src), Value::Integer(expected));
    }

    #[test]
    fn p4_bytes_from_int_le_byte6(n in any::<i64>()) {
        let expected = n.to_le_bytes()[6] as i64;
        let src = format!("(bytes-get (bytes-from-int {}) 6)", n);
        prop_assert_eq!(run(&src), Value::Integer(expected));
    }

    #[test]
    fn p4_bytes_from_int_le_byte7(n in any::<i64>()) {
        let expected = n.to_le_bytes()[7] as i64;
        let src = format!("(bytes-get (bytes-from-int {}) 7)", n);
        prop_assert_eq!(run(&src), Value::Integer(expected));
    }

    #[test]
    fn p4_bytes_from_int_length_always_eight(n in any::<i64>()) {
        let src = format!("(bytes-len (bytes-from-int {}))", n);
        prop_assert_eq!(run(&src), Value::Integer(8));
    }

    // ─────────────────────────────────────────────────────────
    // Phase 4 Step 3 — list-slice
    // ─────────────────────────────────────────────────────────

    /// Two independent runs of the same in-bounds `list-slice`
    /// program yield identical results. `start` and `len` are drawn
    /// from valid sub-ranges of the generated list so the slice
    /// always succeeds.
    #[test]
    fn p4_list_slice_determinism_in_bounds(
        elems in proptest::collection::vec(any::<i64>(), 0..32),
        start in 0usize..32,
        len in 0usize..32,
    ) {
        let n = elems.len();
        let s = if n == 0 { 0 } else { start % (n + 1) };
        let l = len % (n - s + 1);
        let src = format!(
            "(list-slice {} {} {})",
            list_literal(&elems),
            s,
            l,
        );
        prop_assert_eq!(run(&src), run(&src));
    }

    /// Determinism on the error path: an out-of-bounds slice on a
    /// fixed list produces the same typed error on both runs.
    /// `start` is forced past the list length so the slice always
    /// fails.
    #[test]
    fn p4_list_slice_determinism_oob(start in 4i64..100, len in 0i64..100) {
        let src = format!("(list-slice (list 10 20 30) {} {})", start, len);
        let ast = parse(&src).expect("parse");
        let e1 = interpret(&ast).expect_err("must error");
        let e2 = interpret(&ast).expect_err("must error");
        prop_assert_eq!(e1.kind, e2.kind);
    }
}

/// Helper: render a `[i64]` as an Quarks list literal
/// `(list e0 e1 …)`. Mirrors `bytes_literal` for the list family.
fn list_literal(elems: &[i64]) -> String {
    let mut s = String::from("(list");
    for e in elems {
        s.push(' ');
        s.push_str(&e.to_string());
    }
    s.push(')');
    s
}

/// Helper: render a `[u8]` as an Quarks bytes literal `#x….` —
/// lowercase hex pairs, as required by the parser.
fn bytes_literal(b: &[u8]) -> String {
    let mut s = String::with_capacity(2 + 2 * b.len());
    s.push_str("#x");
    for byte in b {
        // `{:02x}` always yields lowercase, matching the parser's
        // lowercase-only hex rule (see `parse_bytes` in
        // `crates/quarks-validator/src/parser.rs`).
        s.push_str(&format!("{:02x}", byte));
    }
    s
}

// ── Phase 4 Step 4 — `cond` N-way Bool dispatch properties ────
//
// Invariants asserted:
//
// 1. **First-match-wins.** For any tuple of predicate booleans
//    `(p0, p1, …, pN-1)` and bodies producing distinct i64 markers,
//    the cond returns the marker of the FIRST true predicate (or
//    the default marker if all are false). Matches a host-side
//    fold equivalent.
// 2. **Determinism.** Two independent runs of the same cond program
//    produce identical results.
// 3. **Default-fallthrough.** When all predicates are statically
//    false, cond returns the default body.

/// Build a cond program whose i-th clause has predicate `pi` and
/// body i (an i64 marker). The default body returns -1. We index
/// by 0-based clause position so that the first-true semantics is
/// directly observable from the returned value.
fn build_cond_program(preds: &[bool]) -> String {
    let mut s = String::from("(cond");
    for (i, p) in preds.iter().enumerate() {
        // Bool literals: `true` / `false` parse as Bool values.
        let pred_src = if *p { "true" } else { "false" };
        s.push_str(&format!(" ({} {})", pred_src, i as i64));
    }
    s.push_str(" (default -1))");
    s
}

/// Host-side reference: return the index of the first true element
/// or -1 if none are true. Mirrors the `(cond …)` first-match-wins
/// semantics for the program built by [`build_cond_program`].
fn host_first_true(preds: &[bool]) -> i64 {
    for (i, p) in preds.iter().enumerate() {
        if *p {
            return i as i64;
        }
    }
    -1
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 4096,
        .. ProptestConfig::default()
    })]

    /// Invariant 1: the cond's output matches the host-side
    /// first-true index. Predicates are randomly true/false; the
    /// body of each clause is its own clause-index, so we read off
    /// which clause matched directly from the returned Integer.
    #[test]
    fn p4_cond_first_match_matches_host(
        preds in proptest::collection::vec(any::<bool>(), 1..=6)
    ) {
        let src = build_cond_program(&preds);
        let expected = host_first_true(&preds);
        prop_assert_eq!(run(&src), Value::Integer(expected));
    }

    /// Invariant 2: determinism across repeated runs.
    #[test]
    fn p4_cond_is_deterministic(
        preds in proptest::collection::vec(any::<bool>(), 1..=6)
    ) {
        let src = build_cond_program(&preds);
        prop_assert_eq!(run(&src), run(&src));
    }

    /// Invariant 3: when every predicate is false, the default body
    /// runs. The marker `-1` is reachable only via the default
    /// clause in [`build_cond_program`].
    #[test]
    fn p4_cond_all_false_falls_through_to_default(n in 1usize..=6) {
        let preds = vec![false; n];
        let src = build_cond_program(&preds);
        prop_assert_eq!(run(&src), Value::Integer(-1));
    }

    /// Invariant 4: a single matching clause at any position is
    /// honoured exactly (the marker returned equals the position
    /// of the match). Stress-tests dispatch through arbitrary
    /// numbers of preceding false-predicate clauses.
    #[test]
    fn p4_cond_single_match_at_position(
        len in 1usize..=8,
        pos in 0usize..8usize,
    ) {
        let pos = pos.min(len - 1);
        let mut preds = vec![false; len];
        preds[pos] = true;
        let src = build_cond_program(&preds);
        prop_assert_eq!(run(&src), Value::Integer(pos as i64));
    }
}

// ── Phase 4 Step 5 — Maybe properties ──────────────────────────

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 4096,
        .. ProptestConfig::default()
    })]

    /// Invariant: `unwrap-or (some x) d == x` for any i64 `x`, `d`.
    /// The default must be ignored when the Maybe is Some.
    #[test]
    fn p4_unwrap_or_some_returns_inner(x in any::<i64>(), d in any::<i64>()) {
        let src = format!("(unwrap-or (some {}) {})", x, d);
        prop_assert_eq!(run(&src), Value::Integer(x));
    }

    /// Invariant: `unwrap-or (none) d == d` for any i64 `d`.
    #[test]
    fn p4_unwrap_or_none_returns_default(d in any::<i64>()) {
        let src = format!("(unwrap-or (none) {})", d);
        prop_assert_eq!(run(&src), Value::Integer(d));
    }

    /// Invariant: `unwrap (some x) == x` for any i64 `x`.
    #[test]
    fn p4_unwrap_some_roundtrips_inner(x in any::<i64>()) {
        let src = format!("(unwrap (some {}))", x);
        prop_assert_eq!(run(&src), Value::Integer(x));
    }

    /// Invariant: `is-some (some _)` is always true.
    #[test]
    fn p4_is_some_on_some_is_true(x in any::<i64>()) {
        let src = format!("(is-some (some {}))", x);
        prop_assert_eq!(run(&src), Value::Bool(true));
    }

    /// Invariant: `is-none (some _)` is always false.
    #[test]
    fn p4_is_none_on_some_is_false(x in any::<i64>()) {
        let src = format!("(is-none (some {}))", x);
        prop_assert_eq!(run(&src), Value::Bool(false));
    }

    /// Determinism: two runs of the same Maybe program agree.
    #[test]
    fn p4_unwrap_or_is_deterministic(x in any::<i64>(), d in any::<i64>()) {
        let src = format!("(unwrap-or (some {}) {})", x, d);
        prop_assert_eq!(run(&src), run(&src));
    }
}
