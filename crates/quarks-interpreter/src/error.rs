// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interpreter errors.

use alloc::string::String;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpretErrorKind {
    UnsupportedAtom,
    UnsupportedInstruction,
    EmptyList,
    NonSymbolHead,
    ArityMismatch,
    // Phase-2-Erweiterung MP1: fn/call semantics
    DuplicateFunction,
    MalformedFunction,
    FunctionNotFound,
    ParameterOutOfRange,
    ParameterOutsideFunction,
    RecursionLimitExceeded,
    /// Pre-A.1 (F-C1) — Package A state machine: the [`Instr`] stack
    /// depth has exceeded [`MAX_INSTR_STACK_DEPTH`]. The explicit
    /// state machine eliminates the native stack-overflow vector of
    /// the original recursive walker; this guard protects the arena
    /// from unbounded growth caused by pathologically deeply nested
    /// inputs (`(add (add (add … 1 1 …)))`).
    /// Deterministically reproducible, no panic.
    ///
    /// [`Instr`]: super::machine
    /// [`MAX_INSTR_STACK_DEPTH`]: super::machine::MAX_INSTR_STACK_DEPTH
    ExpressionDepthExceeded,
    /// Pre-A.2 (F-C2): Validator/interpreter convergence. `(div a b)`
    /// with `b == 0` fails deterministically instead of panicking. The
    /// validator accepts `(div a b)` with signature
    /// `[I64, I64] -> [I64]`; a runtime zero in the divisor is reported
    /// here, because `i64::checked_div` returns `None` for `b == 0` and
    /// for the `i64::MIN / -1` overflow.
    DivisionByZero,
    /// Phase 4 Step 1 — Bitshift: `(bit-shl a n)` / `(bit-shr a n)`
    /// require `0 <= n < 64`. A runtime count outside that range
    /// surfaces here as a typed error instead of an undefined
    /// `shl`/`sar` on x86-64 (which masks the count to its low 6
    /// bits). Mirrors the `DivisionByZero` pattern: the validator
    /// type-checks both operands as `I64`, the interpreter guards
    /// the value-range deterministically. The offending count is
    /// carried so debug output and tooling can surface it.
    ShiftCountOutOfRange {
        count: i64,
    },
    // 12i: Policy-Instruction-Set
    /// `(policy ...)` / `(query ...)` evaluated against a context that
    /// does not implement the dispatch — typically the
    /// `NullPolicyContext` used by the standalone `interpret` entry
    /// point. Use `interpret_with_context` and supply a real context
    /// to enable policy/query dispatch.
    PolicyNotSupported,
    /// `(policy ...)` / `(query ...)` form was malformed — missing
    /// subsystem or operation symbol, or a non-symbol in those slots.
    /// Validator-level structural validation should reject these
    /// before interpretation, but the interpreter checks defensively.
    PolicyMalformed,
    /// Policy context rejected the call. Common cases:
    /// `PolicyError::PermissionDenied` (capability missing/wrong),
    /// `PolicyError::UnknownSubsystem`,
    /// `PolicyError::UnknownOperation`,
    /// `PolicyError::InvalidArgument`,
    /// `PolicyError::NotSupported`.
    PolicyDispatchFailed,
    /// Paket B.1c: `(break v)` evaluated with no enclosing `(loop …)`
    /// frame on the loop stack. The validator catches this
    /// structurally (`BreakOutsideLoopError`); the interpreter
    /// surfaces the same situation here for defence-in-depth.
    BreakOutsideLoop,
    /// Paket B.3: `(let %n v body)` tried to bind a parameter index
    /// (`n < arity`). The validator catches this statically; the
    /// interpreter surfaces a typed error for adversarial / racy IR.
    LetCollidesWithParameter,
    /// Paket B.3: the same `%n` was bound twice in the same scope.
    /// Validator-caught structurally; interpreter rejects defensively.
    LetRedefinition,
    /// Paket B.2: `list-get` was called with an index outside the
    /// list's valid range (`i < 0` or `i >= len`). The validator
    /// cannot catch this statically because the index is runtime
    /// data; the interpreter surfaces a deterministic typed error
    /// rather than panicking.
    ListIndexOutOfBounds,
    /// Phase 4 Step 3: `list-slice` was called with `start`/`len`
    /// arguments that would slice outside the list — i.e.
    /// `start < 0`, `len < 0`, or `start + len > list_length`.
    /// Mirrors `BytesSliceOutOfBounds`; both inputs are surfaced
    /// for tooling.
    ListSliceOutOfBounds {
        start: i64,
        len: i64,
        list_length: usize,
    },
    /// Paket B.4: a `(loop-with-bound N body)` form exhausted its
    /// iteration bound without reaching a `(break v)`. The implicit
    /// `(break false)` semantics turn this into a successful exit
    /// at the value level, so this variant is reserved for cases
    /// where the bound argument was structurally malformed (e.g.
    /// negative) and the interpreter has to abort.
    LoopBoundExceeded,
    /// Paket B.4: `(loop-with-bound N body)` was given a negative
    /// or non-i64 bound at runtime. The validator type-checks `N` as
    /// I64; this variant is the runtime defensive guard against a
    /// negative bound that the type-checker would otherwise let
    /// through (i64 covers signed range).
    LoopBoundInvalid,
    /// Phase 4 Step 2: `bytes-get` was called with an index outside
    /// the byte sequence's valid range (`i < 0` or `i >= length`).
    /// Mirrors `ListIndexOutOfBounds` — the validator cannot catch
    /// this statically because the index is runtime data; the
    /// interpreter surfaces a deterministic typed error.
    BytesIndexOutOfBounds {
        index: i64,
        length: usize,
    },
    /// Phase 4 Step 2: `bytes-slice` was called with `start`/`len`
    /// arguments that would slice outside the byte sequence — i.e.
    /// `start < 0`, `len < 0`, or `start + len > length`. Both
    /// inputs are surfaced for tooling.
    BytesSliceOutOfBounds {
        start: i64,
        len: i64,
        bytes_length: usize,
    },
    /// Phase 4 Step 5: `(unwrap m)` was called on a `Maybe` that is
    /// the `None` case. Mirrors `DivisionByZero` / `ListIndexOutOfBounds`
    /// — the validator type-checks the operand as `Maybe` but cannot
    /// statically know whether the value is `Some` or `None`. The
    /// interpreter surfaces the trap deterministically; policy code
    /// that needs total-function semantics on `None` should use
    /// `unwrap-or` instead.
    UnwrapOnNone,
    /// Phase 4 Step 7: a `(match scrutinee …)` form drained every
    /// case without finding a matching pattern. The validator
    /// requires a mandatory wildcard `_` as the last case, so this
    /// path is unreachable for validated IR. The interpreter still
    /// surfaces a typed error if it encounters an unvalidated AST
    /// or a structurally-malformed match — Ring-0 stays panic-free.
    MatchNonExhaustive,
}

#[derive(Debug, Clone)]
pub struct InterpretError {
    pub kind: InterpretErrorKind,
    pub message: String,
}

impl InterpretError {
    pub fn new(kind: InterpretErrorKind, message: &str) -> Self {
        use alloc::string::ToString;
        Self {
            kind,
            message: message.to_string(),
        }
    }
}

// PartialEq manually because we compare by kind only
impl PartialEq for InterpretError {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}
