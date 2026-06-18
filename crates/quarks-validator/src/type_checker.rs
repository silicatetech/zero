// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::ast::{Atom, SExpr};
use crate::instructions::{lookup, ValueType};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::Cell;

/// Paket B.1c: per-active-loop context used by `simulate_break`
/// and `simulate_loop` (and `simulate_while`) to thread the loop's
/// entry stack and the inferred break-value type.
///
/// `break_type` is a [`Cell`] because mutating it from inside the
/// recursive simulate-tree would otherwise require a `&mut` chain
/// through every intermediate frame. The `Cell` is interior-mutable
/// no_std-friendly state — set by the first reachable `(break v)`
/// in the body, then cross-checked against every subsequent break
/// for type consistency.
///
/// Lifetime: one `LoopCtx` per `simulate_loop`/`simulate_while`
/// activation, scoped to that loop's body. Nested loops shadow the
/// outer ctx (innermost loop wins for `(break v)` resolution); the
/// outer ctx is restored on stack pop.
struct LoopCtx {
    entry_stack: Vec<ValueType>,
    break_type: Cell<Option<ValueType>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeCheckError {
    pub kind: TypeCheckErrorKind,

    /// Path from the program root to the error location, using 1-based
    /// positional indexing (position 1 = first item in list, including
    /// the instruction name itself).
    ///
    /// **Four resolution categories depending on error variant:**
    ///
    /// - **Atom-level errors** (`InvalidHandleError`, `BytesSizeError`,
    ///   `ReservedSymbolError`, `ParameterOutsideFunction`,
    ///   `InvalidParameterIndex`, `UnknownTypeError`):
    ///   `list_path` resolves directly to the offending atom. No further
    ///   lookup needed.
    ///
    /// - **Instruction-level errors with argument resolution**
    ///   (`TypeMismatch`, `FunctionTypeMismatch`,
    ///   `SeqEffectNotStackNeutral`):
    ///   `list_path` resolves to the enclosing instruction list. The
    ///   problematic argument is located via `argument_index` (0-based,
    ///   semantic) combined with `position_offset` (encoded per-variant
    ///   to reflect the list’s structural layout).
    ///   See “Self-describing offset” below.
    ///
    /// - **Instruction-level errors without argument resolution**
    ///   (`ArityError`, `StackUnderflow`, `NestedFunctionError`,
    ///   `DuplicateFunctionError`, `InvalidProgramStructureError`,
    ///   `UndefinedFunctionError`, `FunctionArityMismatch`,
    ///   `LetCollidesWithParameter`, `LetRedefinitionError`):
    ///   `list_path` resolves to the instruction list itself. No
    ///   `argument_index` field; no sub-position resolution needed.
    ///
    /// - **Branch/stack errors** (`BranchStackMismatch`,
    ///   `StackBalanceError`, `BreakOutsideLoopError`):
    ///   `list_path` resolves to the instruction list where the
    ///   imbalance surfaces.
    ///
    /// **Self-describing offset (for argument-resolution errors):**
    ///
    /// The path to the offending atom within its enclosing list is:
    ///
    /// ```text
    /// atom_path = list_path ++ [argument_index + position_offset]
    /// ```
    ///
    /// `position_offset` is carried by the error variant itself, not
    /// hardcoded by consumers. Current values:
    ///
    /// - `TypeMismatch.position_offset = 2` — regular instructions:
    ///   items[0] = instruction name, items[1] = first argument.
    ///   First argument sits at 1-based position 2.
    ///
    /// - `FunctionTypeMismatch.position_offset = 3` — `call`:
    ///   items[0] = "call", items[1] = function name, items[2] = first
    ///   argument. First argument sits at 1-based position 3.
    ///
    /// Future instruction forms add their own `position_offset` value;
    /// consumers read the offset from the error, not hardcoded.
    pub list_path: Vec<usize>,

    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeCheckErrorKind {
    StackUnderflow {
        instruction: String,
        needed: usize,
        available: usize,
    },
    TypeMismatch {
        instruction: String,
        argument_index: usize,
        expected: ValueType,
        actual: ValueType,
        position_offset: u32,
    },
    BranchStackMismatch {
        then_stack: Vec<ValueType>,
        else_stack: Vec<ValueType>,
    },
    StackBalanceError {
        expected: usize,
        actual: usize,
    },
    BreakOutsideLoopError,
    ArityError {
        instruction: String,
        expected: usize,
        actual: usize,
    },
    BytesSizeError {
        actual: usize,
        max: usize,
    },
    InvalidHandleError,
    ReservedSymbolError {
        name: String,
    },
    ParameterOutsideFunction {
        parameter_index: u32,
    },
    InvalidParameterIndex {
        parameter_index: u32,
        arity: u32,
    },
    UnknownTypeError {
        symbol: String,
    },
    NestedFunctionError,
    DuplicateFunctionError,
    InvalidProgramStructureError,
    UndefinedFunctionError {
        name: String,
    },
    FunctionArityMismatch {
        function_name: String,
        expected: u32,
        actual: u32,
    },
    FunctionTypeMismatch {
        function_name: String,
        argument_index: usize,
        expected: ValueType,
        actual: ValueType,
        position_offset: u32,
    },
    LetCollidesWithParameter {
        parameter_index: u32,
    },
    LetRedefinitionError {
        parameter_index: u32,
    },
    SeqEffectNotStackNeutral {
        argument_index: u32,
        expected_depth: usize,
        actual_depth: usize,
        position_offset: u32,
    },
    /// Paket B.1c: two `(break v)` expressions inside the same
    /// loop produced values of incompatible types. The loop's exit
    /// type is determined by the first reachable break; subsequent
    /// breaks must agree. Surfacing this explicitly (rather than
    /// via a generic `TypeMismatch`) makes the kernel-LLM repair
    /// pass faster — the error variant tells the model where to
    /// look (the second break site) and what shape the fix takes.
    BreakTypeMismatch {
        expected: ValueType,
        actual: ValueType,
    },
    /// Paket B.1c: a `(loop body)` form has no reachable `(break
    /// v)` site. The loop is statically infinite; without a break
    /// value, its result type is undefined. The kernel-LLM should
    /// add a conditional break or refactor to `(while cond body)`.
    /// Cooperative scheduling and the per-instruction watchdog
    /// still bound runtime cost, but the validator refuses to type
    /// un-typeable expressions at the language surface.
    LoopWithoutBreak,
    /// Pre-A.1 (F-C6 / F-C7): a structurally-implied internal
    /// invariant of the type-checker did not hold. Validator code is
    /// Ring-0 — every input, including adversarially-constructed
    /// ones, must surface a `TypeCheckError` rather than a panic. The
    /// carried `&'static str` identifies the call site so consumers
    /// (the LSP, the kernel-LLM forensic UI) can localise the issue;
    /// the strings are intentionally short and stable so `match` on
    /// the variant remains useful even without inspecting the tag.
    InternalInvariantViolated(&'static str),
    // ── Phase 4 Step 6 — Struct errors ─────────────────────────
    /// Two `(struct name …)` declarations share a name.
    DuplicateStructError {
        name: String,
    },
    /// A `(struct foo ((field foo)))` (or any chain reaching back to
    /// the struct being defined) was rejected: a struct field type
    /// cannot reference its own declaration.
    RecursiveStructError {
        name: String,
        field: String,
    },
    /// `(struct-new name …)` / `(struct-get expr name)` /
    /// `(struct-set expr name v)` referenced an unknown struct name.
    UnknownStructError {
        name: String,
    },
    /// `(struct-get expr name)` / `(struct-set expr name v)`
    /// referenced an unknown field of an otherwise-valid struct.
    UnknownFieldError {
        struct_name: String,
        field: String,
    },
    /// `(struct-new name v1 v2 …)` arity did not match the struct's
    /// declared field count.
    StructFieldCountMismatch {
        struct_name: String,
        expected: usize,
        actual: usize,
    },
    /// A struct field declaration was malformed (not a 2-element
    /// `(name type-symbol)` clause, or a non-symbol name/type slot).
    MalformedStructField {
        struct_name: String,
        clause_index: usize,
    },
    // ── Phase 4 Step 7 — Match errors ──────────────────────────
    /// A pattern's structural shape does not match the scrutinee's
    /// type. Example: `(case (some %0) body)` when the scrutinee
    /// type-checks as `Struct(_)`. The discriminator carries the
    /// per-shape mismatch detail so tooling can surface a precise
    /// repair suggestion.
    MatchPatternTypeMismatch {
        case_index: usize,
        scrutinee_type: ValueType,
        pattern_kind: MatchPatternKind,
    },
    /// `(case (struct T %0 %1) body)` named a struct `T` that the
    /// struct table does not contain. Distinct from
    /// `UnknownStructError` so consumers can localise to the match
    /// case rather than to a `struct-new` call.
    MatchUnknownStructError {
        case_index: usize,
        name: String,
    },
    /// `(case (struct T %0 %1 %2) body)` bound the wrong number of
    /// fields — struct `T` declares 2 but the pattern lists 3.
    MatchStructFieldCountMismatch {
        case_index: usize,
        struct_name: String,
        expected: usize,
        actual: usize,
    },
    /// A struct-destructuring binding slot is not a `%n` Parameter
    /// atom. Patterns only allow Parameter atoms in field positions
    /// in Phase 4 — no nested patterns, no wildcards inside struct
    /// destructure.
    MatchStructFieldNotParameter {
        case_index: usize,
        struct_name: String,
        field_index: usize,
    },
    /// A `(some %n)` pattern's bind slot is not a `%n` Parameter
    /// atom.
    MatchSomeBindNotParameter {
        case_index: usize,
    },
    /// A pattern's binding `%n` collides with a function parameter
    /// — the validator mirrors `let`'s scope discipline. The
    /// pattern would attempt to rebind a parameter index.
    MatchBindingCollidesWithParameter {
        case_index: usize,
        parameter_index: u32,
    },
    /// A pattern's bindings collide with each other in the same
    /// case (e.g. `(case (struct T %0 %0) body)`). Phase 4
    /// rejects duplicate bind slots within a single pattern.
    MatchBindingRedefinition {
        case_index: usize,
        parameter_index: u32,
    },
}

/// Phase 4 Step 7 — coarse pattern discriminator used by
/// `MatchPatternTypeMismatch` to identify which pattern shape
/// failed against the scrutinee type. Kept structural-only:
/// it does not carry binding indices because the validator surfaces
/// type-level errors here, not binding-level ones.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchPatternKind {
    Wildcard,
    IntegerLiteral,
    Some,
    None,
    Struct,
}

impl TypeCheckError {
    /// Serialize this error to IR-SPEC Section 7 JSON format.
    ///
    /// Used by the API/middleware layer to deliver structured
    /// error information to consuming systems (LLMs, language servers,
    /// CI pipelines). No external serde dependency — manual string
    /// construction keeps the Ring-0 surface minimal.
    pub fn to_json(&self) -> String {
        let kind = self.kind_name();
        let message = json_escape_string(&self.message);
        let path = json_usize_array(&self.list_path);
        let details = self.details_json();

        format!(
            "{{\"error\":{{\"kind\":\"{}\",\"message\":\"{}\",\"list_path\":{},\"details\":{}}}}}",
            kind, message, path, details
        )
    }

    fn kind_name(&self) -> &'static str {
        match &self.kind {
            TypeCheckErrorKind::StackUnderflow { .. } => "StackUnderflow",
            TypeCheckErrorKind::TypeMismatch { .. } => "TypeMismatch",
            TypeCheckErrorKind::BranchStackMismatch { .. } => "BranchStackMismatch",
            TypeCheckErrorKind::StackBalanceError { .. } => "StackBalanceError",
            TypeCheckErrorKind::BreakOutsideLoopError => "BreakOutsideLoopError",
            TypeCheckErrorKind::ArityError { .. } => "ArityError",
            TypeCheckErrorKind::BytesSizeError { .. } => "BytesSizeError",
            TypeCheckErrorKind::InvalidHandleError => "InvalidHandleError",
            TypeCheckErrorKind::ReservedSymbolError { .. } => "ReservedSymbolError",
            TypeCheckErrorKind::ParameterOutsideFunction { .. } => "ParameterOutsideFunction",
            TypeCheckErrorKind::InvalidParameterIndex { .. } => "InvalidParameterIndex",
            TypeCheckErrorKind::UnknownTypeError { .. } => "UnknownTypeError",
            TypeCheckErrorKind::NestedFunctionError => "NestedFunctionError",
            TypeCheckErrorKind::DuplicateFunctionError => "DuplicateFunctionError",
            TypeCheckErrorKind::InvalidProgramStructureError => "InvalidProgramStructureError",
            TypeCheckErrorKind::UndefinedFunctionError { .. } => "UndefinedFunctionError",
            TypeCheckErrorKind::FunctionArityMismatch { .. } => "FunctionArityMismatch",
            TypeCheckErrorKind::FunctionTypeMismatch { .. } => "FunctionTypeMismatch",
            TypeCheckErrorKind::LetCollidesWithParameter { .. } => "LetCollidesWithParameter",
            TypeCheckErrorKind::LetRedefinitionError { .. } => "LetRedefinitionError",
            TypeCheckErrorKind::SeqEffectNotStackNeutral { .. } => "SeqEffectNotStackNeutral",
            TypeCheckErrorKind::BreakTypeMismatch { .. } => "BreakTypeMismatch",
            TypeCheckErrorKind::LoopWithoutBreak => "LoopWithoutBreak",
            TypeCheckErrorKind::InternalInvariantViolated(_) => "InternalInvariantViolated",
            TypeCheckErrorKind::DuplicateStructError { .. } => "DuplicateStructError",
            TypeCheckErrorKind::RecursiveStructError { .. } => "RecursiveStructError",
            TypeCheckErrorKind::UnknownStructError { .. } => "UnknownStructError",
            TypeCheckErrorKind::UnknownFieldError { .. } => "UnknownFieldError",
            TypeCheckErrorKind::StructFieldCountMismatch { .. } => "StructFieldCountMismatch",
            TypeCheckErrorKind::MalformedStructField { .. } => "MalformedStructField",
            TypeCheckErrorKind::MatchPatternTypeMismatch { .. } => "MatchPatternTypeMismatch",
            TypeCheckErrorKind::MatchUnknownStructError { .. } => "MatchUnknownStructError",
            TypeCheckErrorKind::MatchStructFieldCountMismatch { .. } => {
                "MatchStructFieldCountMismatch"
            }
            TypeCheckErrorKind::MatchStructFieldNotParameter { .. } => {
                "MatchStructFieldNotParameter"
            }
            TypeCheckErrorKind::MatchSomeBindNotParameter { .. } => "MatchSomeBindNotParameter",
            TypeCheckErrorKind::MatchBindingCollidesWithParameter { .. } => {
                "MatchBindingCollidesWithParameter"
            }
            TypeCheckErrorKind::MatchBindingRedefinition { .. } => "MatchBindingRedefinition",
        }
    }

    fn details_json(&self) -> String {
        match &self.kind {
            TypeCheckErrorKind::StackUnderflow {
                instruction,
                needed,
                available,
            } => {
                format!(
                    "{{\"instruction\":\"{}\",\"needed\":{},\"available\":{}}}",
                    json_escape_string(instruction),
                    needed,
                    available
                )
            }
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                position_offset,
            } => {
                format!(
                    "{{\"instruction\":\"{}\",\"argument_index\":{},\"expected\":\"{}\",\"actual\":\"{}\",\"position_offset\":{}}}",
                    json_escape_string(instruction), argument_index,
                    value_type_name(*expected), value_type_name(*actual),
                    position_offset
                )
            }
            TypeCheckErrorKind::BranchStackMismatch {
                then_stack,
                else_stack,
            } => {
                format!(
                    "{{\"then_stack\":{},\"else_stack\":{}}}",
                    json_value_type_array(then_stack),
                    json_value_type_array(else_stack)
                )
            }
            TypeCheckErrorKind::StackBalanceError { expected, actual } => {
                format!("{{\"expected\":{},\"actual\":{}}}", expected, actual)
            }
            TypeCheckErrorKind::BreakOutsideLoopError => String::from("null"),
            TypeCheckErrorKind::ArityError {
                instruction,
                expected,
                actual,
            } => {
                format!(
                    "{{\"instruction\":\"{}\",\"expected\":{},\"actual\":{}}}",
                    json_escape_string(instruction),
                    expected,
                    actual
                )
            }
            TypeCheckErrorKind::BytesSizeError { actual, max } => {
                format!("{{\"actual\":{},\"max\":{}}}", actual, max)
            }
            TypeCheckErrorKind::InvalidHandleError => String::from("null"),
            TypeCheckErrorKind::ReservedSymbolError { name } => {
                format!("{{\"name\":\"{}\"}}", json_escape_string(name))
            }
            TypeCheckErrorKind::ParameterOutsideFunction { parameter_index } => {
                format!("{{\"parameter_index\":{}}}", parameter_index)
            }
            TypeCheckErrorKind::InvalidParameterIndex {
                parameter_index,
                arity,
            } => {
                format!(
                    "{{\"parameter_index\":{},\"arity\":{}}}",
                    parameter_index, arity
                )
            }
            TypeCheckErrorKind::UnknownTypeError { symbol } => {
                format!("{{\"symbol\":\"{}\"}}", json_escape_string(symbol))
            }
            TypeCheckErrorKind::NestedFunctionError => String::from("null"),
            TypeCheckErrorKind::DuplicateFunctionError => String::from("null"),
            TypeCheckErrorKind::InvalidProgramStructureError => String::from("null"),
            TypeCheckErrorKind::UndefinedFunctionError { name } => {
                format!("{{\"name\":\"{}\"}}", json_escape_string(name))
            }
            TypeCheckErrorKind::FunctionArityMismatch {
                function_name,
                expected,
                actual,
            } => {
                format!(
                    "{{\"function_name\":\"{}\",\"expected\":{},\"actual\":{}}}",
                    json_escape_string(function_name),
                    expected,
                    actual
                )
            }
            TypeCheckErrorKind::FunctionTypeMismatch {
                function_name,
                argument_index,
                expected,
                actual,
                position_offset,
            } => {
                format!(
                    "{{\"function_name\":\"{}\",\"argument_index\":{},\"expected\":\"{}\",\"actual\":\"{}\",\"position_offset\":{}}}",
                    json_escape_string(function_name), argument_index,
                    value_type_name(*expected), value_type_name(*actual),
                    position_offset
                )
            }
            TypeCheckErrorKind::LetCollidesWithParameter { parameter_index } => {
                format!(r#"{{"parameter_index":{}}}"#, parameter_index)
            }
            TypeCheckErrorKind::LetRedefinitionError { parameter_index } => {
                format!(r#"{{"parameter_index":{}}}"#, parameter_index)
            }
            TypeCheckErrorKind::SeqEffectNotStackNeutral {
                argument_index,
                expected_depth,
                actual_depth,
                position_offset,
            } => {
                format!(
                    r#"{{"argument_index":{},"expected_depth":{},"actual_depth":{},"position_offset":{}}}"#,
                    argument_index, expected_depth, actual_depth, position_offset
                )
            }
            TypeCheckErrorKind::BreakTypeMismatch { expected, actual } => {
                format!(
                    r#"{{"expected":"{}","actual":"{}"}}"#,
                    value_type_name(*expected),
                    value_type_name(*actual)
                )
            }
            TypeCheckErrorKind::LoopWithoutBreak => String::from("null"),
            TypeCheckErrorKind::InternalInvariantViolated(site) => {
                format!(r#"{{"site":"{}"}}"#, json_escape_string(site))
            }
            TypeCheckErrorKind::DuplicateStructError { name } => {
                format!("{{\"name\":\"{}\"}}", json_escape_string(name))
            }
            TypeCheckErrorKind::RecursiveStructError { name, field } => {
                format!(
                    "{{\"name\":\"{}\",\"field\":\"{}\"}}",
                    json_escape_string(name),
                    json_escape_string(field)
                )
            }
            TypeCheckErrorKind::UnknownStructError { name } => {
                format!("{{\"name\":\"{}\"}}", json_escape_string(name))
            }
            TypeCheckErrorKind::UnknownFieldError { struct_name, field } => {
                format!(
                    "{{\"struct_name\":\"{}\",\"field\":\"{}\"}}",
                    json_escape_string(struct_name),
                    json_escape_string(field)
                )
            }
            TypeCheckErrorKind::StructFieldCountMismatch {
                struct_name,
                expected,
                actual,
            } => {
                format!(
                    "{{\"struct_name\":\"{}\",\"expected\":{},\"actual\":{}}}",
                    json_escape_string(struct_name),
                    expected,
                    actual
                )
            }
            TypeCheckErrorKind::MalformedStructField {
                struct_name,
                clause_index,
            } => {
                format!(
                    "{{\"struct_name\":\"{}\",\"clause_index\":{}}}",
                    json_escape_string(struct_name),
                    clause_index
                )
            }
            TypeCheckErrorKind::MatchPatternTypeMismatch {
                case_index,
                scrutinee_type,
                pattern_kind,
            } => {
                format!(
                    "{{\"case_index\":{},\"scrutinee_type\":\"{}\",\"pattern_kind\":\"{}\"}}",
                    case_index,
                    value_type_name(*scrutinee_type),
                    match pattern_kind {
                        MatchPatternKind::Wildcard => "Wildcard",
                        MatchPatternKind::IntegerLiteral => "IntegerLiteral",
                        MatchPatternKind::Some => "Some",
                        MatchPatternKind::None => "None",
                        MatchPatternKind::Struct => "Struct",
                    }
                )
            }
            TypeCheckErrorKind::MatchUnknownStructError { case_index, name } => {
                format!(
                    "{{\"case_index\":{},\"name\":\"{}\"}}",
                    case_index,
                    json_escape_string(name)
                )
            }
            TypeCheckErrorKind::MatchStructFieldCountMismatch {
                case_index,
                struct_name,
                expected,
                actual,
            } => {
                format!(
                    "{{\"case_index\":{},\"struct_name\":\"{}\",\"expected\":{},\"actual\":{}}}",
                    case_index,
                    json_escape_string(struct_name),
                    expected,
                    actual
                )
            }
            TypeCheckErrorKind::MatchStructFieldNotParameter {
                case_index,
                struct_name,
                field_index,
            } => {
                format!(
                    "{{\"case_index\":{},\"struct_name\":\"{}\",\"field_index\":{}}}",
                    case_index,
                    json_escape_string(struct_name),
                    field_index
                )
            }
            TypeCheckErrorKind::MatchSomeBindNotParameter { case_index } => {
                format!("{{\"case_index\":{}}}", case_index)
            }
            TypeCheckErrorKind::MatchBindingCollidesWithParameter {
                case_index,
                parameter_index,
            } => {
                format!(
                    "{{\"case_index\":{},\"parameter_index\":{}}}",
                    case_index, parameter_index
                )
            }
            TypeCheckErrorKind::MatchBindingRedefinition {
                case_index,
                parameter_index,
            } => {
                format!(
                    "{{\"case_index\":{},\"parameter_index\":{}}}",
                    case_index, parameter_index
                )
            }
        }
    }
}

/// JSON string escaping for safe embedding in JSON output.
/// Handles: " → \", \ → \\, control chars → \uXXXX.
///
/// Expects ASCII-safe input as guaranteed by the Quarks parser
/// spec (instruction names and symbols are ASCII-only). Non-ASCII
/// bytes in byte literals are not included in error messages — only
/// their length is reported — so this function never sees non-ASCII
/// content in production. Non-ASCII chars would pass through raw if
/// supplied, which is acceptable JSON but not strictly RFC 8259
/// conformant for chars >= 0x7F without escaping.
fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                // Control characters → \u00XX
                let code = c as u32;
                out.push_str(&format!("\\u{:04x}", code));
            }
            c => out.push(c),
        }
    }
    out
}

fn json_usize_array(values: &[usize]) -> String {
    let mut out = String::from("[");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("{}", v));
    }
    out.push(']');
    out
}

fn json_value_type_array(types: &[ValueType]) -> String {
    let mut out = String::from("[");
    for (i, t) in types.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(value_type_name(*t));
        out.push('"');
    }
    out.push(']');
    out
}

fn value_type_name(vt: ValueType) -> &'static str {
    match vt {
        ValueType::I64 => "I64",
        ValueType::Bytes => "Bytes",
        ValueType::Handle => "Handle",
        ValueType::Bool => "Bool",
        ValueType::List => "List",
        ValueType::Map => "Map",
        ValueType::String => "String",
        ValueType::Maybe => "Maybe",
        // Phase 4 Step 6 — generic "Struct" without the index/name.
        // Richer error messages (concrete struct name) are emitted at
        // construction sites that have a `StructTable` in scope; this
        // function is a static fallback used by `to_json` where no
        // table is available.
        ValueType::Struct(_) => "Struct",
    }
}

// ── Struct table types (Phase 4 Step 6) ────────────────────────

/// A nominal struct definition: ordered list of (field-name,
/// field-type) pairs collected from `(struct name ((f t) …))` at
/// program top-level. Field order matches source order and is the
/// positional contract used by `(struct-new name v1 v2 …)`.
#[derive(Debug, Clone, PartialEq)]
pub struct StructInfo {
    pub name: String,
    pub fields: Vec<(String, ValueType)>,
}

/// Struct table. Built in a pre-pass before fn-signature collection
/// (so fn parameter / return types can reference struct names) and
/// queried during Pass 2 body validation.
///
/// `entries[i]` is the struct whose nominal index is `i`. The
/// `by_name` map gives an O(log n) reverse lookup when resolving a
/// struct-name symbol in `(fn …)` type slots or `(struct-new name …)`
/// heads.
#[derive(Debug, Clone, Default)]
pub struct StructTable {
    entries: Vec<StructInfo>,
    by_name: BTreeMap<String, u32>,
}

impl StructTable {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            by_name: BTreeMap::new(),
        }
    }

    /// Number of structs in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` iff no structs are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a struct under its declared name. Returns the assigned
    /// nominal index. Fails with the duplicate name on collision —
    /// program-level `(struct foo …) (struct foo …)` is rejected.
    pub fn insert(&mut self, info: StructInfo) -> Result<u32, String> {
        if self.by_name.contains_key(&info.name) {
            return Err(info.name.clone());
        }
        let idx = self.entries.len() as u32;
        self.by_name.insert(info.name.clone(), idx);
        self.entries.push(info);
        Ok(idx)
    }

    pub fn lookup_index(&self, name: &str) -> Option<u32> {
        self.by_name.get(name).copied()
    }

    pub fn lookup(&self, idx: u32) -> Option<&StructInfo> {
        self.entries.get(idx as usize)
    }
}

// ── Function table types ───────────────────────────────────────

/// A function's type signature, collected in Pass 1 and consulted during
/// Pass 2 body validation and (MP2b) call-site validation.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    pub name: String,
    pub parameters: Vec<ValueType>,
    pub return_type: ValueType,
    pub arity: u32,
}

/// Function-table, keyed by function name.
///
/// Built in Pass 1. Queried in Pass 2 (body validation — this MP) and
/// Pass 2 call-sites (MP2b).
#[derive(Debug, Clone, Default)]
pub struct FunctionTable {
    entries: BTreeMap<String, FunctionSignature>,
}

impl FunctionTable {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, sig: FunctionSignature) -> Result<(), String> {
        if self.entries.contains_key(&sig.name) {
            return Err(sig.name.clone());
        }
        self.entries.insert(sig.name.clone(), sig);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn get(&self, name: &str) -> Option<&FunctionSignature> {
        self.entries.get(name)
    }
}

/// Runtime scope for one active function body during simulation.
/// Pushed when entering an `fn`-body, popped when leaving.
/// Used by `simulate` to resolve `%n` parameter and local atoms.
#[derive(Debug, Clone)]
struct FunctionContext {
    signature: FunctionSignature,
    locals: BTreeMap<u32, ValueType>,
}

impl FunctionContext {
    fn new(signature: FunctionSignature) -> Self {
        Self {
            signature,
            locals: BTreeMap::new(),
        }
    }

    /// Resolve a parameter index to its type.
    /// Parameters (idx < arity) are checked first, then locals.
    fn lookup(&self, idx: u32) -> Option<ValueType> {
        let idx_usize = idx as usize;
        if idx_usize < self.signature.parameters.len() {
            Some(self.signature.parameters[idx_usize])
        } else {
            self.locals.get(&idx).copied()
        }
    }

    /// Bind a local variable at the given index.
    fn bind_local(&mut self, idx: u32, ty: ValueType) -> Result<(), LetBindError> {
        let idx_usize = idx as usize;
        if idx_usize < self.signature.parameters.len() {
            return Err(LetBindError::CollidesWithParameter);
        }
        if self.locals.contains_key(&idx) {
            return Err(LetBindError::CollidesWithExistingLocal);
        }
        self.locals.insert(idx, ty);
        Ok(())
    }

    /// Unbind a local variable (restores scope on let-body exit).
    fn unbind_local(&mut self, idx: u32) {
        self.locals.remove(&idx);
    }
}

enum LetBindError {
    CollidesWithParameter,
    CollidesWithExistingLocal,
}

/// Resolve a type-symbol string to a ValueType.
///
/// Phase 4 Step 6 — when a `StructTable` is in scope, unrecognised
/// type names are resolved against registered struct declarations
/// before failing. The struct table is built BEFORE fn signature
/// collection so `(fn make-point () point …)` resolves correctly.
fn resolve_type_symbol(sym: &str, structs: &StructTable) -> Result<ValueType, ()> {
    match sym {
        "i64" => Ok(ValueType::I64),
        "bytes" => Ok(ValueType::Bytes),
        "handle" => Ok(ValueType::Handle),
        "bool" => Ok(ValueType::Bool),
        // Paket B.2/B.3/B.5: composite data-type names. Use the same
        // lowercase-symbol convention as `i64`/`bytes`/`handle`/`bool`
        // for parameter / return type slots in `(fn …)` forms.
        "list" => Ok(ValueType::List),
        "map" => Ok(ValueType::Map),
        "string" => Ok(ValueType::String),
        // Phase 4 Step 5 — `maybe` is the type-symbol for the
        // `(some v)` / `(none)` constructor pair. Monomorphic in
        // Phase 4: inner type is fixed to I64.
        "maybe" => Ok(ValueType::Maybe),
        // Phase 4 Step 6 — nominal structs. Lookup the symbol against
        // the registered struct names. Unknown names fall through to
        // the generic UnknownTypeError path.
        other => match structs.lookup_index(other) {
            Some(idx) => Ok(ValueType::Struct(idx)),
            None => Err(()),
        },
    }
}

/// Check if an SExpr is a (fn ...) form.
fn is_fn_form(expr: &SExpr) -> bool {
    if let SExpr::List(items) = expr {
        if let Some(SExpr::Atom(Atom::Symbol(s))) = items.first() {
            return s == "fn";
        }
    }
    false
}

/// Phase 4 Step 6 — check if an SExpr is a `(struct …)` declaration.
fn is_struct_form(expr: &SExpr) -> bool {
    if let SExpr::List(items) = expr {
        if let Some(SExpr::Atom(Atom::Symbol(s))) = items.first() {
            return s == "struct";
        }
    }
    false
}

/// Phase 4 Step 6 — Pass 1a collector: extract a struct declaration
/// from `(struct name ((field1 type1) (field2 type2) …))` and register
/// it in the struct table.
///
/// Direct self-reference (a field of the struct being defined whose
/// type symbol matches the struct's own name) is rejected with
/// `RecursiveStructError`. Indirect cycles via other structs cannot
/// arise here because structs are processed in source order and
/// `resolve_type_symbol` only finds previously-registered names —
/// forward references fall through to `UnknownTypeError`.
fn collect_struct(
    expr: &SExpr,
    base_path: &[usize],
    structs: &mut StructTable,
) -> Result<(), TypeCheckError> {
    let items = match expr {
        SExpr::List(i) => i,
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "collect_struct called on non-list",
                ),
                list_path: base_path.to_vec(),
                message: String::from("internal invariant violated: collect_struct on non-list"),
            });
        }
    };

    // (struct name field-list) = exactly 3 items.
    if items.len() != 3 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("struct"),
                expected: 3,
                actual: items.len(),
            },
            list_path: base_path.to_vec(),
            message: String::from("struct requires (struct name ((field type) …))"),
        });
    }

    // items[1] = name symbol.
    let name = match &items[1] {
        SExpr::Atom(Atom::Symbol(s)) => s.clone(),
        _ => {
            let mut p = base_path.to_vec();
            p.push(2);
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: p,
                message: String::from("struct name must be a symbol"),
            });
        }
    };

    // items[2] = field declaration list.
    let field_items = match &items[2] {
        SExpr::List(l) => l,
        _ => {
            let mut p = base_path.to_vec();
            p.push(3);
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: p,
                message: String::from("struct field list must be a list of (name type) clauses"),
            });
        }
    };

    let mut fields: Vec<(String, ValueType)> = Vec::with_capacity(field_items.len());
    for (i, clause) in field_items.iter().enumerate() {
        let clause_items = match clause {
            SExpr::List(l) => l,
            _ => {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::MalformedStructField {
                        struct_name: name.clone(),
                        clause_index: i,
                    },
                    list_path: base_path.to_vec(),
                    message: format!("struct '{}' field {} must be a (name type) list", name, i),
                });
            }
        };
        if clause_items.len() != 2 {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::MalformedStructField {
                    struct_name: name.clone(),
                    clause_index: i,
                },
                list_path: base_path.to_vec(),
                message: format!(
                    "struct '{}' field {} must be (name type); got {} items",
                    name,
                    i,
                    clause_items.len()
                ),
            });
        }
        let field_name = match &clause_items[0] {
            SExpr::Atom(Atom::Symbol(s)) => s.clone(),
            _ => {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::MalformedStructField {
                        struct_name: name.clone(),
                        clause_index: i,
                    },
                    list_path: base_path.to_vec(),
                    message: format!("struct '{}' field {} name must be a symbol", name, i),
                });
            }
        };
        let type_sym = match &clause_items[1] {
            SExpr::Atom(Atom::Symbol(s)) => s,
            _ => {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::MalformedStructField {
                        struct_name: name.clone(),
                        clause_index: i,
                    },
                    list_path: base_path.to_vec(),
                    message: format!(
                        "struct '{}' field '{}' type must be a type symbol",
                        name, field_name
                    ),
                });
            }
        };
        // Direct self-reference is the only recursion we can detect
        // at this point — the struct is not yet in the table. Reject
        // before resolve_type_symbol falls through to UnknownTypeError
        // (which it would, since the current name is unregistered).
        if type_sym == &name {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::RecursiveStructError {
                    name: name.clone(),
                    field: field_name.clone(),
                },
                list_path: base_path.to_vec(),
                message: format!(
                    "struct '{}' field '{}' references its own type — recursive structs are not allowed",
                    name, field_name
                ),
            });
        }
        let field_ty = match resolve_type_symbol(type_sym, structs) {
            Ok(t) => t,
            Err(()) => {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::UnknownTypeError {
                        symbol: type_sym.clone(),
                    },
                    list_path: base_path.to_vec(),
                    message: format!(
                        "struct '{}' field '{}' has unknown type '{}'",
                        name, field_name, type_sym
                    ),
                });
            }
        };
        fields.push((field_name, field_ty));
    }

    let info = StructInfo {
        name: name.clone(),
        fields,
    };
    structs.insert(info).map_err(|dup_name| TypeCheckError {
        kind: TypeCheckErrorKind::DuplicateStructError {
            name: dup_name.clone(),
        },
        list_path: base_path.to_vec(),
        message: format!("struct '{}' defined more than once", dup_name),
    })?;
    Ok(())
}

/// Pass 1: extract function signature from a (fn name (params...) ret body) form.
fn collect_signature(
    expr: &SExpr,
    base_path: &[usize],
    structs: &StructTable,
) -> Result<FunctionSignature, TypeCheckError> {
    // Pre-A.1 (F-C6): callers gate on `is_fn_form` before reaching
    // here, which structurally implies `SExpr::List`. Surface a
    // typed error instead of panicking in case a future refactor
    // ever calls this on an atom — Ring-0 must stay panic-free.
    let items = match expr {
        SExpr::List(i) => i,
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "collect_signature called on non-list",
                ),
                list_path: base_path.to_vec(),
                message: String::from("internal invariant violated: collect_signature on non-list"),
            });
        }
    };

    // (fn name params-list return-type body) = exactly 5 items
    if items.len() != 5 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("fn"),
                expected: 5,
                actual: items.len(),
            },
            list_path: base_path.to_vec(),
            message: String::from("fn requires: name, params-list, return-type, body"),
        });
    }

    // items[1] = name symbol
    let name = match &items[1] {
        SExpr::Atom(Atom::Symbol(s)) => s.clone(),
        _ => {
            let mut p = base_path.to_vec();
            p.push(2);
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: p,
                message: String::from("fn name must be a symbol"),
            });
        }
    };

    // items[2] = param types list
    let params = match &items[2] {
        SExpr::List(param_items) => {
            let mut types = Vec::new();
            for (j, p) in param_items.iter().enumerate() {
                match p {
                    SExpr::Atom(Atom::Symbol(s)) => match resolve_type_symbol(s, structs) {
                        Ok(t) => types.push(t),
                        Err(()) => {
                            let mut path = base_path.to_vec();
                            path.push(3);
                            path.push(j + 1);
                            return Err(TypeCheckError {
                                kind: TypeCheckErrorKind::UnknownTypeError { symbol: s.clone() },
                                list_path: path,
                                message: format!("unknown type: '{}'", s),
                            });
                        }
                    },
                    _ => {
                        let mut path = base_path.to_vec();
                        path.push(3);
                        path.push(j + 1);
                        return Err(TypeCheckError {
                            kind: TypeCheckErrorKind::InvalidProgramStructureError,
                            list_path: path,
                            message: String::from(
                                "parameter type must be a type symbol (i64, bytes, handle)",
                            ),
                        });
                    }
                }
            }
            types
        }
        _ => {
            let mut p = base_path.to_vec();
            p.push(3);
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: p,
                message: String::from("fn parameter list must be a list of type symbols"),
            });
        }
    };

    // items[3] = return type symbol
    let return_type = match &items[3] {
        SExpr::Atom(Atom::Symbol(s)) => match resolve_type_symbol(s, structs) {
            Ok(t) => t,
            Err(()) => {
                let mut path = base_path.to_vec();
                path.push(4);
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::UnknownTypeError { symbol: s.clone() },
                    list_path: path,
                    message: format!("unknown return type: '{}'", s),
                });
            }
        },
        _ => {
            let mut path = base_path.to_vec();
            path.push(4);
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path,
                message: String::from("fn return type must be a type symbol"),
            });
        }
    };

    let arity = params.len() as u32;
    Ok(FunctionSignature {
        name,
        parameters: params,
        return_type,
        arity,
    })
}

/// Recursively detect nested fn forms inside an expression.
fn detect_nested_fn(expr: &SExpr, base_path: &[usize]) -> Result<(), TypeCheckError> {
    if is_fn_form(expr) {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::NestedFunctionError,
            list_path: base_path.to_vec(),
            message: String::from("fn definitions must be at program top level"),
        });
    }
    if let SExpr::List(items) = expr {
        for (i, item) in items.iter().enumerate() {
            let mut child_path = base_path.to_vec();
            child_path.push(i + 1);
            detect_nested_fn(item, &child_path)?;
        }
    }
    Ok(())
}

// ── Entry points ───────────────────────────────────────────────

/// Type-check a parsed S-expression program via stack simulation.
///
/// Checks 4-8 from IR-SPEC Section 6:
/// - Pre-pass (Checks 6/7/8): bytes size, invalid handle @0, reserved symbols
/// - Stack simulation (Checks 4/5): type checking, branch equality, stack balance
///
/// If the top-level expression is `(program ...)`, two-pass validation is used:
/// Pass 1 collects `fn` signatures, Pass 2 validates bodies with parameter scope.
/// Otherwise, the legacy single-expression path is used.
///
/// Returns the final simulated stack state on success.
pub fn type_check(program: &SExpr) -> Result<Vec<ValueType>, TypeCheckError> {
    // Check for (program ...) wrapper
    if let SExpr::List(items) = program {
        if let Some(SExpr::Atom(Atom::Symbol(s))) = items.first() {
            if s == "program" {
                return type_check_program(items);
            }
        }
    }
    // Legacy path: single-expression validation
    type_check_legacy(program)
}

/// Legacy single-expression type checking (pre-MP2a behavior).
fn type_check_legacy(program: &SExpr) -> Result<Vec<ValueType>, TypeCheckError> {
    pre_scan_ast(program, &mut Vec::new())?;

    let table = FunctionTable::new();
    let structs = StructTable::new();
    let mut fn_stack: Vec<FunctionContext> = Vec::new();
    let mut path = Vec::new();
    let mut stack: Vec<ValueType> = Vec::new();
    let result = simulate(
        program,
        &mut stack,
        &mut path,
        None,
        &table,
        &mut fn_stack,
        &structs,
    )?;
    let final_stack = match result {
        SimResult::Continues(s) => s,
        SimResult::Returns(s) => s,
        SimResult::Breaks(_) => {
            // Pre-A.1 (F-C6): a `Breaks` at the program root means
            // an enclosing-loop check was skipped — semantically a
            // bug in `simulate_break`. Ring-0 must surface a typed
            // error, not panic, even on adversarial input.
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "type_check_legacy: Breaks escaped to root",
                ),
                list_path: Vec::new(),
                message: String::from(
                    "internal invariant violated: Breaks escaped to program root",
                ),
            });
        }
    };
    if final_stack.len() != 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: 1,
                actual: final_stack.len(),
            },
            list_path: Vec::new(),
            message: format!(
                "program must end with stack depth 1, got {}",
                final_stack.len()
            ),
        });
    }
    Ok(final_stack)
}

/// Two-pass type checking for (program def1 def2 ... main-expr).
fn type_check_program(items: &[SExpr]) -> Result<Vec<ValueType>, TypeCheckError> {
    // items[0] = "program" symbol, items[1..] = definitions + main
    let body = &items[1..];
    if body.is_empty() {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::InvalidProgramStructureError,
            list_path: Vec::new(),
            message: String::from("program requires at least one expression"),
        });
    }

    // Pre-scan the entire program first (bytes size, @0, reserved symbols)
    pre_scan_ast(&SExpr::List(items.to_vec()), &mut Vec::new())?;

    // Phase 4 Step 6 — Pass 1a: collect struct declarations BEFORE
    // function signatures, so that fn parameter / return type slots
    // can reference struct names. Structs reference earlier structs
    // in their field types only (forward references are rejected,
    // mirroring source-order resolution for fn calls is unnecessary
    // because the struct table is fully built before any fn type
    // resolution runs).
    let mut structs = StructTable::new();
    for (i, item) in body.iter().enumerate() {
        let item_path = vec![i + 2]; // 1-based: program=1, first child=2
        if is_struct_form(item) {
            collect_struct(item, &item_path, &mut structs)?;
        }
    }

    // Pass 1b: collect fn signatures. Type resolution consults the
    // already-built struct table for any non-builtin type symbol.
    let mut table = FunctionTable::new();
    for (i, item) in body.iter().enumerate() {
        let item_path = vec![i + 2]; // 1-based: program=1, first child=2
        if is_fn_form(item) {
            let sig = collect_signature(item, &item_path, &structs)?;
            table.insert(sig).map_err(|name| TypeCheckError {
                kind: TypeCheckErrorKind::DuplicateFunctionError,
                list_path: item_path.clone(),
                message: format!("function '{}' defined more than once", name),
            })?;
        }
    }

    // Pass 2: validate bodies and main expression
    for (i, item) in body.iter().enumerate() {
        let item_path = vec![i + 2];
        if is_fn_form(item) {
            validate_fn_body(item, &table, &item_path, &structs)?;
        } else if is_struct_form(item) {
            // Already processed in Pass 1a; nothing to do.
        } else if i < body.len() - 1 {
            // Non-fn/non-struct before the last child → structural error
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: item_path,
                message: String::from("non-fn expression found before end of program; function definitions must precede main expression"),
            });
        }
    }

    // Validate main expression (last child) with empty function_stack
    let main_idx = body.len() - 1;
    let main = &body[main_idx];
    let main_path = vec![main_idx + 2];

    // If the main expression is a fn / struct form, it was already
    // processed above; the program then has no runnable main and we
    // reject it.
    if is_fn_form(main) || is_struct_form(main) {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::InvalidProgramStructureError,
            list_path: main_path,
            message: String::from(
                "program requires a main expression after function/struct definitions",
            ),
        });
    }

    let mut fn_stack: Vec<FunctionContext> = Vec::new();
    let mut path = main_path;
    let mut stack: Vec<ValueType> = Vec::new();
    let result = simulate(
        main,
        &mut stack,
        &mut path,
        None,
        &table,
        &mut fn_stack,
        &structs,
    )?;
    let final_stack = match result {
        SimResult::Continues(s) => s,
        SimResult::Returns(s) => s,
        SimResult::Breaks(_) => {
            // Pre-A.1 (F-C6): see `type_check_legacy` for the
            // Breaks-escaped-to-root rationale.
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "type_check_program: Breaks escaped to root",
                ),
                list_path: path,
                message: String::from(
                    "internal invariant violated: Breaks escaped to program root",
                ),
            });
        }
    };
    if final_stack.len() != 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: 1,
                actual: final_stack.len(),
            },
            list_path: Vec::new(),
            message: format!(
                "program must end with stack depth 1, got {}",
                final_stack.len()
            ),
        });
    }
    Ok(final_stack)
}

/// Pass 2: validate a single fn body.
fn validate_fn_body(
    expr: &SExpr,
    table: &FunctionTable,
    base_path: &[usize],
    structs: &StructTable,
) -> Result<(), TypeCheckError> {
    // Pre-A.1 (F-C6): callers gate on `is_fn_form` before reaching
    // here, but Ring-0 surfaces a typed error rather than panicking
    // if the structural invariant is ever broken.
    let items = match expr {
        SExpr::List(i) => i,
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "validate_fn_body called on non-list",
                ),
                list_path: base_path.to_vec(),
                message: String::from("internal invariant violated: validate_fn_body on non-list"),
            });
        }
    };

    let sig = collect_signature(expr, base_path, structs)?;
    let body = &items[4];
    let mut body_path = base_path.to_vec();
    body_path.push(5); // 1-based: fn=1, name=2, params=3, ret=4, body=5

    // Detect nested fn before simulating
    detect_nested_fn(body, &body_path)?;

    // Simulate body with function context
    let ctx = FunctionContext::new(sig.clone());
    let mut fn_stack = vec![ctx];
    let mut stack: Vec<ValueType> = Vec::new();
    let mut sim_path = body_path.clone();
    let result = simulate(
        body,
        &mut stack,
        &mut sim_path,
        None,
        table,
        &mut fn_stack,
        structs,
    )?;

    let body_stack = match result {
        SimResult::Continues(s) => s,
        SimResult::Returns(s) => s,
        SimResult::Breaks(_) => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::BreakOutsideLoopError,
                list_path: body_path,
                message: String::from("break used in fn body outside of loop"),
            });
        }
    };

    // Body must produce exactly one value matching return_type
    if body_stack.len() != 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: 1,
                actual: body_stack.len(),
            },
            list_path: body_path,
            message: format!(
                "function '{}' body must produce exactly 1 value, got {}",
                sig.name,
                body_stack.len()
            ),
        });
    }
    if body_stack[0] != sig.return_type {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::TypeMismatch {
                instruction: String::from("fn"),
                argument_index: 0,
                expected: sig.return_type,
                actual: body_stack[0],
                position_offset: 2,
            },
            list_path: body_path,
            message: format!(
                "function '{}' declares return type {:?} but body produces {:?}",
                sig.name,
                value_type_name(sig.return_type),
                value_type_name(body_stack[0])
            ),
        });
    }

    Ok(())
}

/// Test-only helper that invokes `simulate` directly without the root
/// stack-depth check. Used for tests that validate sub-semantics
/// (loop/break/merge behavior) in isolation, where the whole-program
/// depth-1 contract does not apply.
#[cfg(test)]
fn simulate_raw(program: &SExpr) -> Result<Vec<ValueType>, TypeCheckError> {
    pre_scan_ast(program, &mut Vec::new())?;
    let table = FunctionTable::new();
    let structs = StructTable::new();
    let mut fn_stack: Vec<FunctionContext> = Vec::new();
    let mut path = Vec::new();
    let mut stack: Vec<ValueType> = Vec::new();
    let result = simulate(
        program,
        &mut stack,
        &mut path,
        None,
        &table,
        &mut fn_stack,
        &structs,
    )?;
    match result {
        SimResult::Continues(s) => Ok(s),
        SimResult::Returns(s) => Ok(s),
        SimResult::Breaks(_) => {
            // Pre-A.1 (F-C6): see `type_check_legacy`. Even in the
            // test-only path, surface an error instead of panicking
            // so a fuzzer hitting this path gets a deterministic
            // outcome.
            Err(TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "simulate_raw: Breaks escaped to root",
                ),
                list_path: path,
                message: String::from(
                    "internal invariant violated: Breaks escaped to raw-simulate root",
                ),
            })
        }
    }
}

/// Combined AST pre-pass for Checks 6, 7, 8 (IR-SPEC Section 6).
///
/// Runs as a single recursive walk before stack simulation. All three
/// checks are pure AST scans with no stack-state dependency, so they
/// execute before simulate() to catch policy violations early.
///
/// - Check 6: Bytes literals must not exceed 16 KiB.
/// - Check 7: Handle literal @0 is globally banned. Handles are
///   kernel-assigned at runtime; a naked @0 in source is a bug or
///   attack. Stack-produced handles are not affected (no AST literal).
/// - Check 8: Symbols starting with '_' are reserved for future
///   language extensions. User-land code must not use them.
fn pre_scan_ast(sexpr: &SExpr, path: &mut Vec<usize>) -> Result<(), TypeCheckError> {
    const BYTES_MAX: usize = 16 * 1024; // 16 KiB

    match sexpr {
        SExpr::Atom(atom) => {
            match atom {
                Atom::Bytes(b) if b.len() > BYTES_MAX => {
                    return Err(TypeCheckError {
                        kind: TypeCheckErrorKind::BytesSizeError {
                            actual: b.len(),
                            max: BYTES_MAX,
                        },
                        list_path: path.clone(),
                        message: format!(
                            "bytes literal size {} exceeds maximum {}",
                            b.len(),
                            BYTES_MAX
                        ),
                    });
                }
                Atom::Handle(0) => {
                    return Err(TypeCheckError {
                        kind: TypeCheckErrorKind::InvalidHandleError,
                        list_path: path.clone(),
                        message: String::from(
                            "handle @0 is invalid; handles are kernel-assigned at runtime",
                        ),
                    });
                }
                // Phase 4 Step 7 — `_` (single underscore) is the
                // `match` wildcard pattern and is intentionally
                // exempt from the reserved-prefix rule. Multi-char
                // underscore-prefixed symbols (`_foo`, `__dunder`)
                // remain reserved for kernel-internal use.
                Atom::Symbol(name) if name.starts_with('_') && name.as_str() != "_" => {
                    return Err(TypeCheckError {
                        kind: TypeCheckErrorKind::ReservedSymbolError { name: name.clone() },
                        list_path: path.clone(),
                        message: format!("symbol '{}' starts with underscore (reserved)", name),
                    });
                }
                // Parameter atoms pass pre_scan in MP1. Scope-checking
                // (%n must be inside fn body, n < arity) comes in MP2.
                Atom::Parameter(_) => {}
                _ => {} // All other atoms pass
            }
            Ok(())
        }
        SExpr::List(items) => {
            for (i, item) in items.iter().enumerate() {
                path.push(i + 1); // 1-based, consistent with simulate_instruction
                pre_scan_ast(item, path)?;
                path.pop();
            }
            Ok(())
        }
    }
}

/// Stack simulation result. All variants carry the stack snapshot at
/// the point of resolution.
///
/// Phase A: SimResult carries stack for merge logic.
/// Phase A.5: Returns and Breaks are validated at their source
/// instructions (WASM-style), making merge-time discards sound.
#[derive(Debug, Clone, PartialEq)]
enum SimResult {
    Continues(Vec<ValueType>),
    Returns(Vec<ValueType>),
    Breaks(Vec<ValueType>),
}

fn simulate(
    sexpr: &SExpr,
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    match sexpr {
        SExpr::Atom(atom) => {
            match atom {
                Atom::Integer(_) => stack.push(ValueType::I64),
                Atom::Bytes(_) => stack.push(ValueType::Bytes),
                Atom::Handle(_) => stack.push(ValueType::Handle),
                Atom::Parameter(idx) => match fn_stack.last() {
                    None => {
                        return Err(TypeCheckError {
                            kind: TypeCheckErrorKind::ParameterOutsideFunction {
                                parameter_index: *idx,
                            },
                            list_path: path.clone(),
                            message: format!("parameter %{} used outside of a function body", idx),
                        });
                    }
                    Some(ctx) => match ctx.lookup(*idx) {
                        Some(ty) => stack.push(ty),
                        None => {
                            return Err(TypeCheckError {
                                kind: TypeCheckErrorKind::InvalidParameterIndex {
                                    parameter_index: *idx,
                                    arity: ctx.signature.arity,
                                },
                                list_path: path.clone(),
                                message: format!(
                                    "parameter %{} is not a parameter (arity {}) nor a bound local",
                                    idx, ctx.signature.arity
                                ),
                            });
                        }
                    },
                },
                // Paket B.2: `true` / `false` are Bool literals when they
                // appear in a value position. Every other bare symbol
                // remains a non-value token (head symbols of forms like
                // `program`, `fn`, type-symbols, etc.).
                Atom::Symbol(s) if s == "true" || s == "false" => stack.push(ValueType::Bool),
                Atom::Symbol(_) => {} // symbols don't push values
            }
            Ok(SimResult::Continues(stack.clone()))
        }
        SExpr::List(items) => simulate_instruction(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
    }
}

fn simulate_instruction(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // Pre-A.1 (F-C6 sub-finding): an empty list `()` reaches here
    // and previously panicked on `items[0]`. Pass it through as a
    // structurally-invalid program node — `InvalidProgramStructureError`
    // is the closest existing match, and propagating it lets the
    // parser/validator caller decide whether to reject or normalise
    // the input.
    let head = match items.first() {
        Some(h) => h,
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: alloc::string::String::from("empty list `()` cannot be simulated"),
            });
        }
    };
    let instruction_name = match head {
        SExpr::Atom(Atom::Symbol(name)) => name.as_str(),
        _ => return Ok(SimResult::Continues(stack.clone())),
    };

    match instruction_name {
        "if" => simulate_if(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        // Phase 4 Step 4 — N-way Bool dispatch.
        "cond" => simulate_cond(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "loop" => simulate_loop(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "while" => simulate_while(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "return" => simulate_return(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "break" => simulate_break(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "dup" => simulate_dup(stack, path),
        "drop" => simulate_drop(stack, path),
        "swap" => simulate_swap(stack, path),
        "call" => simulate_call(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "let" => simulate_let(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "seq" => simulate_seq(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "intent" => simulate_intent(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "policy" => simulate_policy_or_query(
            "policy",
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "query" => simulate_policy_or_query(
            "query",
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "discard" => simulate_discard(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        // Stage 12 Paket B (Spracherweiterung) — variadic forms with
        // idiosyncratic substructure.
        "list" => simulate_list(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "loop-with-bound" => simulate_loop_with_bound(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "for" => simulate_for(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "write-host-state" => simulate_write_host_state(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        // Phase 4 Step 6 — nominal structs.
        "struct-new" => simulate_struct_new(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "struct-get" => simulate_struct_get(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        "struct-set" => simulate_struct_set(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        // Phase 4 Step 7 — pattern matching.
        "match" => simulate_match(
            items,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        ),
        // `(struct …)` declarations are top-level only; reaching here
        // means a `struct` appeared inside an expression, which is a
        // structural error. We surface it as InvalidProgramStructureError.
        "struct" => Err(TypeCheckError {
            kind: TypeCheckErrorKind::InvalidProgramStructureError,
            list_path: path.clone(),
            message: String::from(
                "struct declarations must appear at program top level, not inside expressions",
            ),
        }),
        _ => {
            let signature = match lookup(instruction_name) {
                Some(sig) => sig,
                None => return Ok(SimResult::Continues(stack.clone())),
            };
            simulate_regular(
                instruction_name,
                signature.inputs,
                signature.outputs,
                &items[1..],
                stack,
                path,
                loop_context,
                fn_table,
                fn_stack,
                struct_table,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn simulate_regular(
    name: &str,
    inputs: &[ValueType],
    outputs: &[ValueType],
    args: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // Simulate all arguments (they push their results onto stack).
    // If any arg terminates, propagate immediately.
    for (i, arg) in args.iter().enumerate() {
        path.push(i + 2); // args = &items[1..], so args[i] = items[i+1], 1-based pos = i+2
        let result = simulate(
            arg,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        )?;
        path.pop();
        match result {
            SimResult::Continues(_) => {}
            other => return Ok(other),
        }
    }

    // Check stack has enough values for inputs
    if stack.len() < inputs.len() {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackUnderflow {
                instruction: String::from(name),
                needed: inputs.len(),
                available: stack.len(),
            },
            list_path: path.clone(),
            message: format!("instruction '{}' needs {} stack values", name, inputs.len()),
        });
    }

    // Check types match
    let stack_top_start = stack.len() - inputs.len();
    for (i, expected) in inputs.iter().enumerate() {
        let actual = stack[stack_top_start + i];
        if actual != *expected {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::TypeMismatch {
                    instruction: String::from(name),
                    argument_index: i,
                    expected: *expected,
                    actual,
                    position_offset: 2,
                },
                list_path: path.clone(),
                message: format!("instruction '{}' expects {:?}", name, expected),
            });
        }
    }

    // Pop inputs, push outputs
    stack.truncate(stack_top_start);
    stack.extend_from_slice(outputs);

    Ok(SimResult::Continues(stack.clone()))
}

/// Simulate a `(call name arg1 arg2 ...)` instruction.
///
/// Type-checks args against the function signature from `fn_table`.
/// Does NOT re-execute the function body — the body was already
/// validated in Pass 2. This is what makes recursion (self and mutual)
/// safe: call-sites only consult the signature, never walk the body.
fn simulate_call(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // (call name arg1 arg2 ...) → items[0]=call, items[1]=name, items[2..]=args
    if items.len() < 2 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("call"),
                expected: 2,
                actual: items.len(),
            },
            list_path: path.clone(),
            message: String::from("call requires at least a function name"),
        });
    }

    // items[1] must be a symbol (function name)
    let fn_name = match &items[1] {
        SExpr::Atom(Atom::Symbol(s)) => s.clone(),
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: String::from("call target must be a function name symbol"),
            });
        }
    };

    // Lookup function signature
    let sig = match fn_table.get(&fn_name) {
        Some(s) => s.clone(),
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::UndefinedFunctionError {
                    name: fn_name.clone(),
                },
                list_path: path.clone(),
                message: format!("undefined function '{}'", fn_name),
            });
        }
    };

    // Arity check: args = items[2..], expected = sig.arity
    let args = &items[2..];
    let actual_arity = args.len() as u32;
    if actual_arity != sig.arity {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::FunctionArityMismatch {
                function_name: fn_name.clone(),
                expected: sig.arity,
                actual: actual_arity,
            },
            list_path: path.clone(),
            message: format!(
                "function '{}' expects {} arguments, got {}",
                sig.name, sig.arity, actual_arity
            ),
        });
    }

    // Simulate all arguments (they push their results onto stack).
    // Path tracking: args[i] = items[i+2], 1-based pos = i+3.
    // Wait — items[0]=call, items[1]=name, items[2]=arg0.
    // 1-based: call=1, name=2, arg0=3, arg1=4, ...
    // So args[i] is at 1-based position i+3.
    for (i, arg) in args.iter().enumerate() {
        path.push(i + 3); // 1-based: call=1, name=2, first arg=3
        let result = simulate(
            arg,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        )?;
        path.pop();
        match result {
            SimResult::Continues(_) => {}
            other => return Ok(other),
        }
    }

    // Type-check each arg against the signature's parameter types.
    // Args were pushed in order, so they occupy the top of the stack.
    let arg_count = sig.arity as usize;
    if stack.len() < arg_count {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackUnderflow {
                instruction: format!("call {}", fn_name),
                needed: arg_count,
                available: stack.len(),
            },
            list_path: path.clone(),
            message: format!("call '{}' needs {} stack values", fn_name, arg_count),
        });
    }

    let stack_top_start = stack.len() - arg_count;
    for (i, expected) in sig.parameters.iter().enumerate() {
        let actual = stack[stack_top_start + i];
        if actual != *expected {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::FunctionTypeMismatch {
                    function_name: fn_name.clone(),
                    argument_index: i,
                    expected: *expected,
                    actual,
                    position_offset: 3,
                },
                list_path: path.clone(),
                message: format!(
                    "function '{}' argument {} expects {:?}, got {:?}",
                    fn_name, i, expected, actual
                ),
            });
        }
    }

    // Pop args, push return type
    stack.truncate(stack_top_start);
    stack.push(sig.return_type);

    Ok(SimResult::Continues(stack.clone()))
}

fn simulate_if(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    if items.len() != 4 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("if"),
                expected: 4,
                actual: items.len(),
            },
            list_path: path.clone(),
            message: format!("if expects (if cond then else), got {} items", items.len()),
        });
    }

    // Simulate condition argument (items[1] → 1-based position 2)
    path.push(2);
    let cond_result = simulate(
        &items[1],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match cond_result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }

    // Paket B.1b: condition must produce Bool (not I64). The Bool
    // type is the canonical predicate carrier; using I64 here would
    // re-open the "looks-like-zero is false" ambiguity that Paket
    // B intentionally closed.
    if stack.is_empty() {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackUnderflow {
                instruction: String::from("if"),
                needed: 1,
                available: 0,
            },
            list_path: path.clone(),
            message: String::from("if condition produced no value"),
        });
    }
    let cond_type = stack.pop().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_if: stack.pop() returned None despite is_empty() guard",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: simulate_if condition stack vanished"),
    })?;
    if cond_type != ValueType::Bool {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::TypeMismatch {
                instruction: String::from("if"),
                argument_index: 0,
                expected: ValueType::Bool,
                actual: cond_type,
                position_offset: 2,
            },
            list_path: path.clone(),
            message: String::from("if condition must be Bool"),
        });
    }

    // Save stack state before branches
    let stack_before_branches = stack.clone();

    // Simulate then-branch (items[2] → 1-based position 3)
    path.push(3);
    let then_result = simulate(
        &items[2],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();

    // Reset stack for else-branch
    *stack = stack_before_branches;

    // Simulate else-branch (items[3] → 1-based position 4)
    path.push(4);
    let else_result = simulate(
        &items[3],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();

    // Merge branch results per Phase A merge table.
    merge_branches(then_result, else_result, stack, path)
}

/// Phase 4 Step 4 — type-check `(cond (p1 b1) (p2 b2) … (default body))`.
///
/// Each non-default clause's predicate must produce `Bool`. After the
/// predicate is conceptually consumed by the implicit dispatch, the
/// clause's body is type-checked against the branch-entry stack.
/// The default clause has no predicate; its body is type-checked
/// directly. All branch results — every clause body, including the
/// default — are merged via [`merge_n_branches`], which enforces the
/// same domination rules as the 2-way `if` merge.
///
/// Structural malformation (non-list clause, wrong arity, missing
/// default) is caught earlier by [`validate_cond`] in the structural
/// pass; we still defensively re-check here so the type-checker can
/// run standalone against an unvalidated AST.
fn simulate_cond(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // items[0] = "cond". The remaining items are the clauses.
    let clauses = &items[1..];
    if clauses.is_empty() {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("cond"),
                expected: 1,
                actual: 0,
            },
            list_path: path.clone(),
            message: String::from("cond requires at least one (default body) clause"),
        });
    }

    let stack_before = stack.clone();
    let mut branch_results: Vec<SimResult> = Vec::new();

    let last_index = clauses.len() - 1;
    for (i, clause) in clauses.iter().enumerate() {
        // Each clause must structurally be a 2-element list (validated
        // up-front by validate_cond; re-checked here defensively).
        let clause_items = match clause {
            SExpr::List(items) => items,
            _ => {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::InvalidProgramStructureError,
                    list_path: path.clone(),
                    message: format!("cond clause {} must be a (predicate body) list", i),
                });
            }
        };
        if clause_items.len() != 2 {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: format!(
                    "cond clause {} must be (predicate body); got {} items",
                    i,
                    clause_items.len()
                ),
            });
        }

        let is_last = i == last_index;
        let is_default_head = matches!(
            &clause_items[0],
            SExpr::Atom(Atom::Symbol(s)) if s == "default"
        );
        if is_last {
            if !is_default_head {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::InvalidProgramStructureError,
                    list_path: path.clone(),
                    message: String::from(
                        "cond last clause must be (default body) — fallback is mandatory",
                    ),
                });
            }
        } else if is_default_head {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: format!(
                    "cond clause {} uses (default …) but default may appear only in the last clause",
                    i
                ),
            });
        }

        // Reset to the cond-entry stack before evaluating this clause.
        *stack = stack_before.clone();

        // Path layout for this clause: clauses[i] is items[i+1] in the
        // outer cond list, which is 1-based position i+2.
        path.push(i + 2);

        if !is_last {
            // Simulate predicate at position 1 inside the clause (the
            // clause is (predicate body); 1-based positions are
            // predicate=1, body=2). Predicate must produce a Bool.
            path.push(1);
            let pred_result = simulate(
                &clause_items[0],
                stack,
                path,
                loop_context,
                fn_table,
                fn_stack,
                struct_table,
            )?;
            path.pop();

            match pred_result {
                SimResult::Continues(_) => {}
                // Predicate diverged statically — propagate. Matches
                // the simulate_if treatment of a divergent condition.
                other => {
                    path.pop();
                    return Ok(other);
                }
            }

            if stack.is_empty() {
                path.pop();
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::StackUnderflow {
                        instruction: String::from("cond"),
                        needed: 1,
                        available: 0,
                    },
                    list_path: path.clone(),
                    message: format!("cond clause {} predicate produced no value", i),
                });
            }
            let pred_type = stack.pop().ok_or_else(|| TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "simulate_cond: stack.pop() returned None despite is_empty() guard",
                ),
                list_path: path.clone(),
                message: String::from(
                    "internal invariant violated: simulate_cond predicate stack vanished",
                ),
            })?;
            if pred_type != ValueType::Bool {
                path.pop();
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::TypeMismatch {
                        instruction: String::from("cond"),
                        argument_index: i,
                        expected: ValueType::Bool,
                        actual: pred_type,
                        position_offset: 2,
                    },
                    list_path: path.clone(),
                    message: format!("cond clause {} predicate must be Bool", i),
                });
            }
        }

        // Simulate body at position 2 inside the clause.
        path.push(2);
        let body_result = simulate(
            &clause_items[1],
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        )?;
        path.pop();

        path.pop();
        branch_results.push(body_result);
    }

    merge_n_branches(branch_results, stack, path, "cond")
}

/// Merge two branch SimResults according to the Phase A.75 merge table.
///
/// | Left              | Right             | Result                                    |
/// |-------------------|-------------------|-------------------------------------------|
/// | Continues(s1)     | Continues(s2)     | s1==s2 → Continues(s1), else Mismatch    |
/// | Continues(s)      | Returns/Breaks    | Continues(s)                              |
/// | Returns/Breaks    | Continues(s)      | Continues(s)                              |
/// | Returns(s1)       | Returns(s2)       | s1==s2 → Returns(s1), else Mismatch      |
/// | Breaks(s1)        | Breaks(s2)        | s1==s2 → Breaks(s1), else Mismatch       |
/// | Returns(_)        | Breaks(s)         | Breaks(s) — Breaks dominates              |
/// | Breaks(s)         | Returns(_)        | Breaks(s) — Breaks dominates              |
///
/// Breaks dominates Returns in mixed pairs because a runtime break-path
/// forces post-loop code to execute, which must be validated. If Returns
/// dominated, the validator would silently skip post-loop code when a
/// break-path exists, letting runtime stack errors through. Both variants
/// are validated at their source instruction (A.5), so dropping one at
/// merge is sound.
fn merge_branches(
    then_result: SimResult,
    else_result: SimResult,
    stack: &mut Vec<ValueType>,
    path: &[usize],
) -> Result<SimResult, TypeCheckError> {
    merge_two_branches(then_result, else_result, stack, path, "if")
}

/// Phase 4 Step 4 — N-way generalisation of [`merge_branches`] for
/// `(cond …)`. Folds [`merge_two_branches`] left-to-right across an
/// arbitrary number of branch results. The pairwise merge table is
/// associative under its rules (Continues dominates Returns/Breaks
/// in the residual stack; Breaks dominates Returns in mixed pairs)
/// so left-fold yields the same answer as any other order: the
/// invariant the validator depends on is that the final residual
/// stack is the unique stack shared by every continuing branch.
///
/// `label` is used in error messages ("cond branches produce …").
/// `path` points at the cond's outer list — branch-mismatch errors
/// surface at the cond expression rather than at any single clause.
fn merge_n_branches(
    results: Vec<SimResult>,
    stack: &mut Vec<ValueType>,
    path: &[usize],
    label: &str,
) -> Result<SimResult, TypeCheckError> {
    let mut iter = results.into_iter();
    let mut acc = match iter.next() {
        Some(r) => r,
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "merge_n_branches: empty results",
                ),
                list_path: path.to_vec(),
                message: String::from(
                    "internal invariant violated: N-way branch merge called with zero branches",
                ),
            });
        }
    };
    for next in iter {
        acc = merge_two_branches(acc, next, stack, path, label)?;
    }
    Ok(acc)
}

/// Shared 2-branch merge core used by both [`merge_branches`] (for
/// `if`) and [`merge_n_branches`] (for `cond`). The `label` parameter
/// only customises the error message; the merge semantics are
/// identical across forms.
fn merge_two_branches(
    then_result: SimResult,
    else_result: SimResult,
    stack: &mut Vec<ValueType>,
    path: &[usize],
    label: &str,
) -> Result<SimResult, TypeCheckError> {
    match (then_result, else_result) {
        // Both continue: branch equality check
        (SimResult::Continues(s1), SimResult::Continues(s2)) => {
            if s1 != s2 {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::BranchStackMismatch {
                        then_stack: s1,
                        else_stack: s2,
                    },
                    list_path: path.to_vec(),
                    message: format!("{} branches produce different stacks", label),
                });
            }
            *stack = s1.clone();
            Ok(SimResult::Continues(s1))
        }

        // One continues, one terminates: use the continuing branch
        (SimResult::Continues(s), SimResult::Returns(_))
        | (SimResult::Continues(s), SimResult::Breaks(_)) => {
            *stack = s.clone();
            Ok(SimResult::Continues(s))
        }
        (SimResult::Returns(_), SimResult::Continues(s))
        | (SimResult::Breaks(_), SimResult::Continues(s)) => {
            *stack = s.clone();
            Ok(SimResult::Continues(s))
        }

        // Both return: branch equality check
        (SimResult::Returns(s1), SimResult::Returns(s2)) => {
            if s1 != s2 {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::BranchStackMismatch {
                        then_stack: s1,
                        else_stack: s2,
                    },
                    list_path: path.to_vec(),
                    message: format!("{} branches return with different stacks", label),
                });
            }
            Ok(SimResult::Returns(s1))
        }

        // Both break: branch equality check
        (SimResult::Breaks(s1), SimResult::Breaks(s2)) => {
            if s1 != s2 {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::BranchStackMismatch {
                        then_stack: s1,
                        else_stack: s2,
                    },
                    list_path: path.to_vec(),
                    message: format!("{} branches break with different stacks", label),
                });
            }
            Ok(SimResult::Breaks(s1))
        }

        // Mixed (Returns, Breaks): Breaks dominates, no equality check.
        // Sound because Phase A.5 validates both at their source instruction.
        // Rationale: Returns-dominance would silently skip post-loop code
        // validation when the break-path is taken at runtime. Breaks-dominance
        // forces the enclosing loop to resolve to Continues(stack_in), which
        // propagates into post-loop validation.
        (SimResult::Returns(_), SimResult::Breaks(s)) => Ok(SimResult::Breaks(s)),
        (SimResult::Breaks(s), SimResult::Returns(_)) => Ok(SimResult::Breaks(s)),
    }
}

/// Paket B.1c — `(loop body)` produces the value carried by the
/// `(break v)` form reached from inside the body.
///
/// Type semantics:
/// - Body is one expression. It is evaluated repeatedly. Its own
///   residual stack contribution is discarded at the end of each
///   iteration (the loop is iteratively stack-neutral from the
///   outside).
/// - All `(break v)` forms reachable inside the body must agree on
///   the type T of `v`. T becomes the loop's output type.
/// - The loop's net stack effect is `[..]` → `[..; T]`.
/// - If no `(break v)` is reachable, the loop is statically
///   infinite; the type-checker rejects it with
///   [`TypeCheckErrorKind::LoopWithoutBreak`].
///
/// The outer `loop_context` is shadowed by a fresh per-loop
/// `LoopCtx`; nested loops bind to their own ctx so `(break v)`
/// always targets the innermost enclosing loop.
fn simulate_loop(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    _outer_loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    if items.len() != 2 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("loop"),
                expected: 2,
                actual: items.len(),
            },
            list_path: path.clone(),
            message: format!("loop expects (loop body), got {} items", items.len()),
        });
    }

    let stack_in = stack.clone();
    let loop_ctx = LoopCtx {
        entry_stack: stack_in.clone(),
        break_type: Cell::new(None),
    };

    // Body is items[1] → 1-based position 2.
    path.push(2);
    let body_result = simulate(
        &items[1],
        stack,
        path,
        Some(&loop_ctx),
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();

    finalise_loop_body(loop_ctx, stack_in, body_result, stack, path)
}

/// Paket B.1c shared finaliser used by both `simulate_loop` and
/// `simulate_while`. Combines the body's [`SimResult`] with the
/// loop's collected break-type to produce the loop expression's
/// stack effect.
#[allow(clippy::ptr_arg)]
fn finalise_loop_body(
    loop_ctx: LoopCtx,
    stack_in: Vec<ValueType>,
    body_result: SimResult,
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
) -> Result<SimResult, TypeCheckError> {
    match body_result {
        SimResult::Returns(s) => {
            // Loop propagates returns (enclosing function exits).
            Ok(SimResult::Returns(s))
        }
        SimResult::Continues(_) | SimResult::Breaks(_) => {
            let break_t = loop_ctx.break_type.get().ok_or_else(|| TypeCheckError {
                kind: TypeCheckErrorKind::LoopWithoutBreak,
                list_path: path.clone(),
                message: String::from(
                    "loop body contains no reachable (break v); loop has no statically determinable output type",
                ),
            })?;
            *stack = stack_in;
            stack.push(break_t);
            Ok(SimResult::Continues(stack.clone()))
        }
    }
}

/// Paket B.1d — `(while cond body)` is type-checked directly (no
/// AST rewrite) but is semantically equivalent to
/// `(loop (if cond body (break false)))`. The synthetic
/// `(break false)` fires when the condition becomes false, so the
/// loop's output type is always `Bool`.
///
/// Constraints:
/// - `cond` must produce `Bool`.
/// - `body` may produce any one value (discarded per iteration);
///   the body's residual type does not influence the loop's output.
/// - Any user-level `(break v)` inside the body must produce
///   `Bool`, matching the synthetic break.
fn simulate_while(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    _outer_loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    if items.len() != 3 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("while"),
                expected: 3,
                actual: items.len(),
            },
            list_path: path.clone(),
            message: format!("while expects (while cond body), got {} items", items.len()),
        });
    }

    let stack_in = stack.clone();
    // Seed the loop's break_type with Bool — the type of the
    // synthetic (break false) that fires when cond becomes false.
    // User-level (break v) inside body must agree.
    let loop_ctx = LoopCtx {
        entry_stack: stack_in.clone(),
        break_type: Cell::new(Some(ValueType::Bool)),
    };

    // Cond must end up Bool.
    path.push(2);
    let cond_result = simulate(
        &items[1],
        stack,
        path,
        Some(&loop_ctx),
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match cond_result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }
    if stack.len() != stack_in.len() + 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: stack_in.len() + 1,
                actual: stack.len(),
            },
            list_path: path.clone(),
            message: format!(
                "while cond must push exactly 1 value (stack depth went from {} to {})",
                stack_in.len(),
                stack.len()
            ),
        });
    }
    let cond_type = stack.pop().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_while: stack.pop() returned None despite length guard",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: simulate_while cond stack vanished"),
    })?;
    if cond_type != ValueType::Bool {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::TypeMismatch {
                instruction: String::from("while"),
                argument_index: 0,
                expected: ValueType::Bool,
                actual: cond_type,
                position_offset: 2,
            },
            list_path: path.clone(),
            message: String::from("while condition must be Bool"),
        });
    }

    // Simulate body. Body's residual type is irrelevant (discarded
    // per iteration). User-level breaks inside body are validated
    // against loop_ctx (already seeded with Bool).
    path.push(3);
    let body_result = simulate(
        &items[2],
        stack,
        path,
        Some(&loop_ctx),
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();

    finalise_loop_body(loop_ctx, stack_in, body_result, stack, path)
}

/// Phase A.5: WASM-style source validation for `return`.
/// v0.1 root constraint: return stack depth must be exactly 1.
fn simulate_return(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    if items.len() != 2 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("return"),
                expected: 2,
                actual: items.len(),
            },
            list_path: path.clone(),
            message: format!("return expects (return value), got {} items", items.len()),
        });
    }

    // Value is items[1] → 1-based position 2.
    path.push(2);
    let result = simulate(
        &items[1],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }

    // Phase A.5: validate return stack depth == 1 at the source instruction.
    if stack.len() != 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: 1,
                actual: stack.len(),
            },
            list_path: path.clone(),
            message: format!("return expects stack depth 1, got {}", stack.len()),
        });
    }

    Ok(SimResult::Returns(stack.clone()))
}

/// Paket B.1c — `(break v)` exits the innermost enclosing loop and
/// carries `v` as the loop's value.
///
/// Type semantics:
/// - The break's value expression is evaluated; it must produce
///   exactly one value of some type T.
/// - The enclosing `LoopCtx.break_type` is set to T on the first
///   reachable break, and is cross-checked against T on subsequent
///   breaks. Disagreement surfaces as
///   [`TypeCheckErrorKind::BreakTypeMismatch`].
/// - `(break v)` outside any loop is
///   [`TypeCheckErrorKind::BreakOutsideLoopError`].
///
/// Stack contract: at break time the stack must look like
/// `entry_stack + [T]` — i.e. the loop's invariant is preserved
/// with exactly one extra value on top, identical to the loop's
/// intended output. The state-machine interpreter (machine.rs)
/// relies on this to unwind cleanly.
fn simulate_break(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // (break value) → items[0]=break, items[1]=value. Exactly 1
    // arg (the break-value expression).
    if items.len() != 2 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("break"),
                expected: 1,
                actual: items.len() - 1,
            },
            list_path: path.clone(),
            message: format!(
                "break expects exactly 1 argument (the break value), got {}",
                items.len() - 1
            ),
        });
    }

    let ctx = match loop_context {
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::BreakOutsideLoopError,
                list_path: path.clone(),
                message: String::from("break used outside of loop"),
            });
        }
        Some(c) => c,
    };

    // Evaluate the break-value expression. It pushes exactly one
    // value of some type T.
    path.push(2); // 1-based: break=1, value=2
    let value_result = simulate(
        &items[1],
        stack,
        path,
        Some(ctx),
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match value_result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }

    // Stack must be exactly entry_stack + [T] for some T.
    let entry_len = ctx.entry_stack.len();
    if stack.len() != entry_len + 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: entry_len + 1,
                actual: stack.len(),
            },
            list_path: path.clone(),
            message: format!(
                "break stack depth {} does not match loop entry depth + 1 = {}",
                stack.len(),
                entry_len + 1
            ),
        });
    }
    if stack[..entry_len] != ctx.entry_stack[..] {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: entry_len,
                actual: stack.len(),
            },
            list_path: path.clone(),
            message: String::from(
                "break stack prefix does not match loop entry stack (the body must not corrupt the loop's outer stack)",
            ),
        });
    }
    let break_t = stack[entry_len];

    // Update or cross-check the loop's inferred break type.
    match ctx.break_type.get() {
        None => ctx.break_type.set(Some(break_t)),
        Some(expected) if expected == break_t => {}
        Some(expected) => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::BreakTypeMismatch {
                    expected,
                    actual: break_t,
                },
                list_path: path.clone(),
                message: format!(
                    "break value type {:?} does not match this loop's earlier break value type {:?}",
                    break_t, expected
                ),
            });
        }
    }

    Ok(SimResult::Breaks(stack.clone()))
}

#[allow(clippy::ptr_arg)]
fn simulate_dup(
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
) -> Result<SimResult, TypeCheckError> {
    if stack.is_empty() {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackUnderflow {
                instruction: String::from("dup"),
                needed: 1,
                available: 0,
            },
            list_path: path.clone(),
            message: String::from("dup on empty stack"),
        });
    }
    // Pre-A.1 (F-C7): structurally safe under the just-checked
    // `is_empty()` guard; Ring-0 surfaces a typed error otherwise.
    let top = *stack.last().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_dup: stack.last() returned None despite is_empty() guard",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: simulate_dup top-of-stack vanished"),
    })?;
    stack.push(top);
    Ok(SimResult::Continues(stack.clone()))
}

#[allow(clippy::ptr_arg)]
fn simulate_drop(
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
) -> Result<SimResult, TypeCheckError> {
    if stack.is_empty() {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackUnderflow {
                instruction: String::from("drop"),
                needed: 1,
                available: 0,
            },
            list_path: path.clone(),
            message: String::from("drop on empty stack"),
        });
    }
    stack.pop();
    Ok(SimResult::Continues(stack.clone()))
}

#[allow(clippy::ptr_arg)]
fn simulate_swap(
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
) -> Result<SimResult, TypeCheckError> {
    if stack.len() < 2 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackUnderflow {
                instruction: String::from("swap"),
                needed: 2,
                available: stack.len(),
            },
            list_path: path.clone(),
            message: String::from("swap requires 2 values"),
        });
    }
    let len = stack.len();
    stack.swap(len - 1, len - 2);
    Ok(SimResult::Continues(stack.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use alloc::vec;

    /// Helper: parse then type-check, return Result
    fn tc(input: &str) -> Result<Vec<ValueType>, TypeCheckError> {
        let sexpr = parse(input).expect("parse should succeed for type-check tests");
        type_check(&sexpr)
    }

    /// Helper: parse then type-check, expect Ok, return stack
    fn tc_ok(input: &str) -> Vec<ValueType> {
        tc(input).expect("type check should succeed")
    }

    /// Helper: parse then type-check, expect Err, return error
    fn tc_err(input: &str) -> TypeCheckError {
        tc(input).expect_err("type check should fail")
    }

    /// Like tc_ok but bypasses the root depth-1 check.
    fn tcr_ok(input: &str) -> Vec<ValueType> {
        let sexpr = parse(input).expect("parse should succeed");
        simulate_raw(&sexpr).expect("simulate_raw should succeed")
    }

    /// Like tc_err but bypasses the root depth-1 check.
    fn tcr_err(input: &str) -> TypeCheckError {
        let sexpr = parse(input).expect("parse should succeed");
        simulate_raw(&sexpr).expect_err("simulate_raw should fail")
    }

    // ── 1. Literal Push Tests ───────────────────────────────────

    #[test]
    fn literal_integer_pushes_i64() {
        assert_eq!(tc_ok("42"), vec![ValueType::I64]);
    }

    #[test]
    fn literal_bytes_pushes_bytes() {
        assert_eq!(tc_ok("#x48656c6c6f"), vec![ValueType::Bytes]);
    }

    #[test]
    fn literal_handle_pushes_handle() {
        assert_eq!(tc_ok("@5"), vec![ValueType::Handle]);
    }

    #[test]
    fn return_integer() {
        let stack = tc_ok("(return 42)");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn return_bytes() {
        let stack = tc_ok("(return #x00)");
        assert_eq!(stack, vec![ValueType::Bytes]);
    }

    #[test]
    fn return_handle() {
        let stack = tc_ok("(return @5)");
        assert_eq!(stack, vec![ValueType::Handle]);
    }

    // ── 2. Arithmetic Tests ─────────────────────────────────────

    #[test]
    fn add_two_integers() {
        assert_eq!(tc_ok("(add 1 2)"), vec![ValueType::I64]);
    }

    #[test]
    fn sub_two_integers() {
        assert_eq!(tc_ok("(sub 10 4)"), vec![ValueType::I64]);
    }

    #[test]
    fn mul_two_integers() {
        assert_eq!(tc_ok("(mul 2 3)"), vec![ValueType::I64]);
    }

    #[test]
    fn div_two_integers() {
        assert_eq!(tc_ok("(div 10 2)"), vec![ValueType::I64]);
    }

    // ── Phase 4 Step 1 — bitwise ops: types ─────────────────────

    #[test]
    fn bit_and_two_integers() {
        assert_eq!(tc_ok("(bit-and 255 15)"), vec![ValueType::I64]);
    }

    #[test]
    fn bit_or_two_integers() {
        assert_eq!(tc_ok("(bit-or 1 2)"), vec![ValueType::I64]);
    }

    #[test]
    fn bit_xor_two_integers() {
        assert_eq!(tc_ok("(bit-xor 5 3)"), vec![ValueType::I64]);
    }

    #[test]
    fn bit_shl_two_integers() {
        assert_eq!(tc_ok("(bit-shl 1 10)"), vec![ValueType::I64]);
    }

    #[test]
    fn bit_shr_two_integers() {
        assert_eq!(tc_ok("(bit-shr 256 4)"), vec![ValueType::I64]);
    }

    #[test]
    fn bit_and_bool_first_arg_type_mismatch() {
        // Disambiguates the bit-* family from the Bool `and`/`or`:
        // a Bool literal in the lhs slot must be rejected.
        let err = tc_err("(bit-and true 1)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "bit-and");
                assert_eq!(*argument_index, 0);
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::Bool);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn bit_or_bytes_second_arg_type_mismatch() {
        let err = tc_err("(bit-or 1 #x00)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "bit-or");
                assert_eq!(*argument_index, 1);
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::Bytes);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn bit_xor_handle_first_arg_type_mismatch() {
        let err = tc_err("(bit-xor @5 1)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "bit-xor");
                assert_eq!(*argument_index, 0);
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::Handle);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn bit_shl_string_second_arg_type_mismatch() {
        // `(string-from-int 5)` produces a String value, which must
        // not satisfy the shift-count I64 slot.
        let err = tc_err("(bit-shl 1 (string-from-int 5))");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "bit-shl");
                assert_eq!(*argument_index, 1);
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::String);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn bit_shr_list_first_arg_type_mismatch() {
        let err = tc_err("(bit-shr (list 1 2 3) 1)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "bit-shr");
                assert_eq!(*argument_index, 0);
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::List);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn bit_and_map_second_arg_type_mismatch() {
        let err = tc_err("(bit-and 1 (map-new))");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "bit-and");
                assert_eq!(*argument_index, 1);
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::Map);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn eq_two_integers() {
        // Paket B.1: eq now returns Bool.
        assert_eq!(tc_ok("(eq 0 1)"), vec![ValueType::Bool]);
    }

    #[test]
    fn add_bytes_first_arg_type_mismatch() {
        let err = tc_err("(add #x00 2)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "add");
                assert_eq!(*argument_index, 0);
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::Bytes);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn add_handle_second_arg_type_mismatch() {
        let err = tc_err("(add 1 @5)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "add");
                assert_eq!(*argument_index, 1);
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::Handle);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    // ── 3. Send/Recv Tests ──────────────────────────────────────

    #[test]
    fn send_correct_types() {
        assert_eq!(tc_ok("(send @5 #x00)"), vec![ValueType::I64]);
    }

    #[test]
    fn send_wrong_first_arg() {
        let err = tc_err("(send 1 #x00)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "send");
                assert_eq!(*argument_index, 0);
                assert_eq!(*expected, ValueType::Handle);
                assert_eq!(*actual, ValueType::I64);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn send_wrong_second_arg() {
        let err = tc_err("(send @5 42)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "send");
                assert_eq!(*argument_index, 1);
                assert_eq!(*expected, ValueType::Bytes);
                assert_eq!(*actual, ValueType::I64);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn recv_pushes_three_values() {
        // Sub-semantics: recv pushes 3 values, not a depth-1 program
        assert_eq!(
            tcr_ok("(recv)"),
            vec![ValueType::Handle, ValueType::I64, ValueType::Bytes]
        );
    }

    #[test]
    fn spawn_correct_type() {
        assert_eq!(tc_ok("(spawn #x00)"), vec![ValueType::Handle]);
    }

    #[test]
    fn spawn_wrong_type() {
        let err = tc_err("(spawn 42)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "spawn");
                assert_eq!(*expected, ValueType::Bytes);
                assert_eq!(*actual, ValueType::I64);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn register_correct_type() {
        assert_eq!(tc_ok("(register #x00)"), vec![ValueType::Handle]);
    }

    // ── 4. If Tests ─────────────────────────────────────────────

    #[test]
    fn if_matching_branches_return() {
        let stack = tc_ok("(if (eq 0 1) (return 1) (return 0))");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn if_condition_must_be_bool() {
        // Paket B.1b: if condition must be Bool.
        let err = tc_err("(if #x00 (return 1) (return 0))");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "if");
                assert_eq!(*expected, ValueType::Bool);
                assert_eq!(*actual, ValueType::Bytes);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn if_condition_handle_not_bool() {
        // Paket B.1b: if condition must be Bool.
        let err = tc_err("(if @5 (return 1) (return 0))");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "if");
                assert_eq!(*expected, ValueType::Bool);
                assert_eq!(*actual, ValueType::Handle);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    // ── 5. Branch Stack Mismatch Tests (MP3) ────────────────────

    #[test]
    fn branch_mismatch_different_types() {
        let err = tc_err("(if true (add 1 2) (spawn #x00))");
        match &err.kind {
            TypeCheckErrorKind::BranchStackMismatch {
                then_stack,
                else_stack,
            } => {
                assert_eq!(*then_stack, vec![ValueType::I64]);
                assert_eq!(*else_stack, vec![ValueType::Handle]);
            }
            other => panic!("expected BranchStackMismatch, got {:?}", other),
        }
    }

    #[test]
    fn branch_mismatch_different_depth() {
        let err = tc_err("(if true (recv) (add 1 2))");
        match &err.kind {
            TypeCheckErrorKind::BranchStackMismatch {
                then_stack,
                else_stack,
            } => {
                assert_eq!(then_stack.len(), 3);
                assert_eq!(else_stack.len(), 1);
            }
            other => panic!("expected BranchStackMismatch, got {:?}", other),
        }
    }

    #[test]
    fn branch_matching_when_both_terminate() {
        let stack = tc_ok("(if true (return 42) (return 0))");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn branch_one_terminates_uses_other() {
        // Paket B.1c: loop produces the break value's type;
        // then breaks Bool, else is stack-neutral.
        let stack = tcr_ok("(loop (if true (break false) (store 0 0)))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn branch_other_terminates_uses_first() {
        // Paket B.1c: symmetric to branch_one_terminates_uses_other.
        let stack = tcr_ok("(loop (if true (store 0 0) (break false)))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    // ── 6. Stack Operations Tests ───────────────────────────────

    #[test]
    fn dup_on_empty_stack() {
        let err = tc_err("(dup)");
        match &err.kind {
            TypeCheckErrorKind::StackUnderflow {
                instruction,
                needed,
                available,
            } => {
                assert_eq!(instruction, "dup");
                assert_eq!(*needed, 1);
                assert_eq!(*available, 0);
            }
            other => panic!("expected StackUnderflow, got {:?}", other),
        }
    }

    #[test]
    fn drop_on_empty_stack() {
        let err = tc_err("(drop)");
        match &err.kind {
            TypeCheckErrorKind::StackUnderflow {
                instruction,
                needed,
                available,
            } => {
                assert_eq!(instruction, "drop");
                assert_eq!(*needed, 1);
                assert_eq!(*available, 0);
            }
            other => panic!("expected StackUnderflow, got {:?}", other),
        }
    }

    #[test]
    fn swap_on_empty_stack() {
        let err = tc_err("(swap)");
        match &err.kind {
            TypeCheckErrorKind::StackUnderflow {
                instruction,
                needed,
                available,
            } => {
                assert_eq!(instruction, "swap");
                assert_eq!(*needed, 2);
                assert_eq!(*available, 0);
            }
            other => panic!("expected StackUnderflow, got {:?}", other),
        }
    }

    #[test]
    fn swap_with_one_element() {
        let err = tc_err("(swap)");
        match &err.kind {
            TypeCheckErrorKind::StackUnderflow { instruction, .. } => {
                assert_eq!(instruction, "swap");
            }
            other => panic!("expected StackUnderflow, got {:?}", other),
        }
    }

    // ── 7. Complex Integration Tests ────────────────────────────

    #[test]
    fn complex_nested_arithmetic() {
        let stack = tc_ok("(return (add (mul 2 3) (sub 10 4)))");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn load_correct_type() {
        assert_eq!(tc_ok("(load 0)"), vec![ValueType::I64]);
    }

    #[test]
    fn store_correct_types() {
        // Sub-semantics: store consumes 2, produces 0
        assert_eq!(tcr_ok("(store 0 42)"), vec![]);
    }

    #[test]
    fn store_wrong_address_type() {
        let err = tc_err("(store #x00 42)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "store");
                assert_eq!(*argument_index, 0);
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::Bytes);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn if_with_nested_arithmetic_matching() {
        let stack = tc_ok("(if (eq 1 2) (add 10 20) (sub 5 3))");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn loop_with_break_produces_bool() {
        // Paket B.1c: `(loop (break false))` produces the broken value.
        let stack = tcr_ok("(loop (break false))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn underflow_add_no_args_on_stack() {
        let err = tc_err("(load #x00)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "load");
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::Bytes);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    // ── 8. Phase A — SimResult Merge Tests ──────────────────────

    #[test]
    fn merge_returns_returns_same_stack() {
        // Both branches return with same type → Returns propagated
        let stack = tc_ok("(if true (return 42) (return 0))");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn merge_returns_returns_different_stack() {
        // Both branches return with different types → BranchStackMismatch
        let err = tc_err("(if true (return 42) (return #x00))");
        match &err.kind {
            TypeCheckErrorKind::BranchStackMismatch {
                then_stack,
                else_stack,
            } => {
                assert_eq!(*then_stack, vec![ValueType::I64]);
                assert_eq!(*else_stack, vec![ValueType::Bytes]);
            }
            other => panic!("expected BranchStackMismatch, got {:?}", other),
        }
    }

    #[test]
    fn merge_breaks_breaks_same_stack() {
        // Paket B.1c: both branches break with Bool; loop output Bool.
        let stack = tcr_ok("(loop (if true (break false) (break false)))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn merge_returns_breaks_mixed_breaks_dominates() {
        // Paket B.1c reframe: Breaks still dominates Returns;
        // loop output = break value type.
        let stack = tc_ok("(loop (if true (return 42) (break false)))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn merge_breaks_returns_mixed_breaks_dominates() {
        // Paket B.1c reframe: symmetric to mixed_breaks_dominates.
        let stack = tc_ok("(loop (if true (break false) (return 42)))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn soundness_breaks_dominates_forces_post_loop_validation() {
        // Paket B.1c reframe: Breaks still dominates Returns in
        // merge_branches. The behaviorally observable effect is
        // that the loop's output type tracks the break-side, not
        // the return-side; post-loop context sees the break
        // value's type rather than silently inheriting the
        // Returns side's type.
        let stack = tc_ok("(loop (if true (return 0) (break false)))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn merge_return_in_one_branch_continue_in_other() {
        // then: return (terminates), else: add (continues)
        // Result: Continues with else-branch stack
        let stack = tc_ok("(if true (return 42) (add 1 2))");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    // ── 9. Phase A — Loop Tests ─────────────────────────────────

    #[test]
    fn loop_body_continues_stack_neutral_ok() {
        // Paket B.1c: body produces a residual per iteration (here:
        // store is stack-neutral after consuming its args); loop
        // output = break value type = Bool.
        let stack = tcr_ok("(loop (if true (break false) (store 0 0)))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn loop_without_reachable_break_errors() {
        // Paket B.1c: a loop body that never breaks is statically
        // infinite; the validator refuses to type it.
        let err = tc_err("(loop (add 1 2))");
        assert_eq!(err.kind, TypeCheckErrorKind::LoopWithoutBreak);
    }

    #[test]
    fn loop_body_returns_propagates() {
        // Loop body returns → loop result is Returns
        let stack = tc_ok("(loop (return 42))");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn loop_body_breaks_becomes_continues_with_value() {
        // Paket B.1c: break → loop catches → Continues(stack_in + [T_break]).
        let stack = tcr_ok("(loop (break false))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    // ── 10. Phase A.5 — Source Validation Tests ─────────────────

    #[test]
    fn break_inside_loop_matching_stack() {
        // Paket B.1c: break carries a value; loop produces that type.
        let stack = tcr_ok("(loop (break false))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn break_inside_loop_mismatching_stack() {
        // Paket B.1c: two breaks in the same loop must agree on
        // value type; disagreement → BreakTypeMismatch. (This
        // replaces the old depth-mismatch test, which is no
        // longer reachable under the new (break v) semantics.)
        let err = tcr_err("(loop (if true (break false) (break 0)))");
        match &err.kind {
            TypeCheckErrorKind::BreakTypeMismatch { expected, actual } => {
                assert_eq!(*expected, ValueType::Bool);
                assert_eq!(*actual, ValueType::I64);
            }
            other => panic!("expected BreakTypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn break_outside_loop() {
        // Paket B.1c: (break v) outside any loop.
        let err = tc_err("(break false)");
        assert_eq!(err.kind, TypeCheckErrorKind::BreakOutsideLoopError);
    }

    #[test]
    fn return_stack_depth_1_ok() {
        let stack = tc_ok("(return 42)");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn return_stack_depth_not_1() {
        // (return (recv)): recv pushes 3 values [Handle, I64, Bytes].
        // return sees stack depth 3 ≠ 1 → StackBalanceError.
        let err = tc_err("(return (recv))");
        match &err.kind {
            TypeCheckErrorKind::StackBalanceError { expected, actual } => {
                assert_eq!(*expected, 1);
                assert_eq!(*actual, 3);
            }
            other => panic!("expected StackBalanceError from return, got {:?}", other),
        }
    }

    #[test]
    fn soundness_return_wrong_depth_despite_merge_dominance() {
        // Critical soundness test: (if cond (return (recv)) (add 1 2))
        // return (recv) has stack depth 3 → StackBalanceError at the return,
        // even though the merge would discard it (Continues dominates).
        let err = tc_err("(if true (return (recv)) (add 1 2))");
        match &err.kind {
            TypeCheckErrorKind::StackBalanceError { expected, actual } => {
                assert_eq!(*expected, 1);
                assert_eq!(*actual, 3);
            }
            other => panic!("expected StackBalanceError from return, got {:?}", other),
        }
    }

    #[test]
    fn soundness_break_wrong_stack_despite_returns_dominance() {
        // Paket B.1c reframe: simulate_break checks stack ==
        // entry + [T]. Inside `(add 1 (break false))`: lhs `1`
        // pushed I64; the break value `false` pushes Bool; stack
        // at break = [I64, Bool] depth 2. Loop entry = [] so
        // expected = 1; actual = 2.
        let err = tc_err("(loop (if true (return 42) (add 1 (break false))))");
        match &err.kind {
            TypeCheckErrorKind::StackBalanceError { expected, actual } => {
                assert_eq!(*expected, 1);
                assert_eq!(*actual, 2);
            }
            other => panic!("expected StackBalanceError from break, got {:?}", other),
        }
    }

    #[test]
    fn nested_loops_inner_break_validates_against_inner_context() {
        // Paket B.1c: inner break targets inner loop; outer loop
        // body merges (Breaks-outer, Continues(stack_in + [Bool])).
        // Outer break_type Bool from then-branch; output Bool.
        let stack = tcr_ok("(loop (if true (break false) (loop (break false))))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn nested_loops_inner_break_wrong_stack_for_inner() {
        // Paket B.1c: break checks stack == entry_stack + [T].
        // Inside `(add 1 (break false))`: lhs `1` pushes I64; then
        // break evaluates `false` (pushes Bool). Stack at break
        // time = [I64, Bool] (depth 2), but inner loop entry is
        // [] so expected = 0 + 1 = 1.
        let err = tc_err("(loop (loop (add 1 (break false))))");
        match &err.kind {
            TypeCheckErrorKind::StackBalanceError { expected, actual } => {
                assert_eq!(*expected, 1);
                assert_eq!(*actual, 2);
            }
            other => panic!("expected StackBalanceError, got {:?}", other),
        }
    }

    #[test]
    fn break_in_if_in_loop_correct_context() {
        // Paket B.1c: both branches break with Bool; output Bool.
        let stack = tcr_ok("(loop (if true (break false) (break false)))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn break_outside_loop_in_if_branch() {
        // Paket B: if cond must be Bool; (break v) outside any
        // loop → BreakOutsideLoopError.
        let err = tc_err("(if true (break false) (add 1 2))");
        assert_eq!(err.kind, TypeCheckErrorKind::BreakOutsideLoopError);
    }

    // ── 11. Phase B — Arity-Defensive Tests ─────────────────────

    #[test]
    fn arity_if_too_few_1() {
        let err = tcr_err("(if)");
        match &err.kind {
            TypeCheckErrorKind::ArityError {
                instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "if");
                assert_eq!(*expected, 4);
                assert_eq!(*actual, 1);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn arity_if_too_few_2() {
        let err = tcr_err("(if 1)");
        match &err.kind {
            TypeCheckErrorKind::ArityError {
                instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "if");
                assert_eq!(*expected, 4);
                assert_eq!(*actual, 2);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn arity_if_too_few_3() {
        let err = tcr_err("(if true 2)");
        match &err.kind {
            TypeCheckErrorKind::ArityError {
                instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "if");
                assert_eq!(*expected, 4);
                assert_eq!(*actual, 3);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn arity_if_too_many() {
        let err = tcr_err("(if true 2 3 4)");
        match &err.kind {
            TypeCheckErrorKind::ArityError {
                instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "if");
                assert_eq!(*expected, 4);
                assert_eq!(*actual, 5);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn arity_loop_too_few() {
        let err = tcr_err("(loop)");
        match &err.kind {
            TypeCheckErrorKind::ArityError {
                instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "loop");
                assert_eq!(*expected, 2);
                assert_eq!(*actual, 1);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn arity_loop_too_many() {
        let err = tcr_err("(loop a b)");
        match &err.kind {
            TypeCheckErrorKind::ArityError {
                instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "loop");
                assert_eq!(*expected, 2);
                assert_eq!(*actual, 3);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn arity_return_too_few() {
        let err = tcr_err("(return)");
        match &err.kind {
            TypeCheckErrorKind::ArityError {
                instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "return");
                assert_eq!(*expected, 2);
                assert_eq!(*actual, 1);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn arity_return_too_many() {
        let err = tcr_err("(return x y)");
        match &err.kind {
            TypeCheckErrorKind::ArityError {
                instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "return");
                assert_eq!(*expected, 2);
                assert_eq!(*actual, 3);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn arity_break_too_many() {
        // Paket B.1c: break takes exactly 1 arg; (break 1 2) is
        // a real arity error.
        let err = tcr_err("(break 1 2)");
        match &err.kind {
            TypeCheckErrorKind::ArityError {
                instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "break");
                assert_eq!(*expected, 1);
                assert_eq!(*actual, 2);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    // ── 12. Phase B — Dual-Review Coverage Completion ───────────

    #[test]
    fn merge_continue_in_then_return_in_else() {
        // Merge-table Arm 2: (Continues, Returns) → Continues.
        // Symmetric pair to merge_return_in_one_branch_continue_in_other
        // which covers (Returns, Continues) → Continues (Arm 4).
        let stack = tc_ok("(if true (add 1 2) (return 42))");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn nested_loop_outer_return_inner_break() {
        // Paket B.1c regression guard: inner loop produces Bool;
        // merge (Returns([I64]), Continues([Bool])) → Continues([Bool]).
        // Root sees depth 1 → no StackBalanceError; the Bool exit
        // type is the observable.
        let s = tc_ok("(if true (return 42) (loop (break false)))");
        assert_eq!(s, vec![ValueType::Bool]);
    }

    // ── 13. Phase B — Root-Check Tests ──────────────────────────

    #[test]
    fn root_depth_1_ok() {
        // Program ending with exactly 1 value
        let stack = tc_ok("42");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn root_depth_0_error() {
        let err = tc_err("(store 0 42)");
        match &err.kind {
            TypeCheckErrorKind::StackBalanceError { expected, actual } => {
                assert_eq!(*expected, 1);
                assert_eq!(*actual, 0);
            }
            other => panic!("expected StackBalanceError, got {:?}", other),
        }
    }

    #[test]
    fn root_depth_3_error() {
        let err = tc_err("(recv)");
        match &err.kind {
            TypeCheckErrorKind::StackBalanceError { expected, actual } => {
                assert_eq!(*expected, 1);
                assert_eq!(*actual, 3);
            }
            other => panic!("expected StackBalanceError, got {:?}", other),
        }
    }

    // ── 14. Phase C — Check 6: Bytes Size Tests ─────────────────

    #[test]
    fn bytes_size_exactly_16k_ok() {
        // Construct a Bytes atom with exactly 16 * 1024 = 16384 bytes
        let data = alloc::vec![0u8; 16 * 1024];
        let program = SExpr::Atom(Atom::Bytes(data));
        // Pre-scan should pass: exactly at limit, not over
        assert!(pre_scan_ast(&program, &mut Vec::new()).is_ok());
    }

    #[test]
    fn bytes_size_16k_plus_1_error() {
        let data = alloc::vec![0u8; 16 * 1024 + 1];
        let program = SExpr::Atom(Atom::Bytes(data));
        let err = pre_scan_ast(&program, &mut Vec::new()).expect_err("should fail");
        match &err.kind {
            TypeCheckErrorKind::BytesSizeError { actual, max } => {
                assert_eq!(*actual, 16385);
                assert_eq!(*max, 16384);
            }
            other => panic!("expected BytesSizeError, got {:?}", other),
        }
    }

    #[test]
    fn bytes_size_small_ok() {
        assert!(pre_scan_ast(
            &SExpr::Atom(Atom::Bytes(alloc::vec![0u8; 10])),
            &mut Vec::new()
        )
        .is_ok());
    }

    #[test]
    fn bytes_size_empty_ok() {
        // #x (empty bytes)
        assert!(pre_scan_ast(&SExpr::Atom(Atom::Bytes(alloc::vec![])), &mut Vec::new()).is_ok());
    }

    #[test]
    fn bytes_size_nested_in_instruction() {
        // Oversized bytes inside a list → still caught
        let data = alloc::vec![0u8; 16 * 1024 + 1];
        let program = SExpr::List(alloc::vec![
            SExpr::Atom(Atom::Symbol(String::from("spawn"))),
            SExpr::Atom(Atom::Bytes(data)),
        ]);
        let err = pre_scan_ast(&program, &mut Vec::new()).expect_err("should fail");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::BytesSizeError { .. }
        ));
    }

    // ── 15. Phase C — Check 7: Invalid Handle @0 Tests ──────────

    #[test]
    fn handle_zero_top_level() {
        let err = tc_err("@0");
        assert_eq!(err.kind, TypeCheckErrorKind::InvalidHandleError);
    }

    #[test]
    fn handle_zero_in_spawn() {
        let err = tc_err("(spawn @0)");
        assert_eq!(err.kind, TypeCheckErrorKind::InvalidHandleError);
    }

    #[test]
    fn handle_nonzero_ok() {
        // @42 is fine — no InvalidHandleError
        assert!(pre_scan_ast(&SExpr::Atom(Atom::Handle(42)), &mut Vec::new()).is_ok());
    }

    #[test]
    fn handle_zero_in_if_then_branch() {
        let err = tc_err("(if true @0 @42)");
        assert_eq!(err.kind, TypeCheckErrorKind::InvalidHandleError);
    }

    #[test]
    fn handle_zero_in_if_else_branch() {
        let err = tc_err("(if true @42 @0)");
        assert_eq!(err.kind, TypeCheckErrorKind::InvalidHandleError);
    }

    #[test]
    fn handle_zero_nested_deep() {
        // @0 deep in AST: (if 1 (return @0) (return @42))
        let err = tc_err("(if true (return @0) (return @42))");
        assert_eq!(err.kind, TypeCheckErrorKind::InvalidHandleError);
    }

    #[test]
    fn handle_stack_produced_no_error() {
        // spawn produces a handle on stack — no @0 literal in AST
        assert_eq!(tc_ok("(spawn #x00)"), vec![ValueType::Handle]);
    }

    // ── 16. Phase C — Check 8: Reserved Symbols Tests ───────────

    #[test]
    fn reserved_symbol_underscore_prefix() {
        let err = pre_scan_ast(
            &SExpr::Atom(Atom::Symbol(String::from("_foo"))),
            &mut Vec::new(),
        )
        .expect_err("should fail");
        match &err.kind {
            TypeCheckErrorKind::ReservedSymbolError { name } => {
                assert_eq!(name, "_foo");
            }
            other => panic!("expected ReservedSymbolError, got {:?}", other),
        }
    }

    #[test]
    fn reserved_symbol_dunder() {
        let err = pre_scan_ast(
            &SExpr::Atom(Atom::Symbol(String::from("__dunder"))),
            &mut Vec::new(),
        )
        .expect_err("should fail");
        match &err.kind {
            TypeCheckErrorKind::ReservedSymbolError { name } => {
                assert_eq!(name, "__dunder");
            }
            other => panic!("expected ReservedSymbolError, got {:?}", other),
        }
    }

    #[test]
    fn normal_symbol_no_underscore_ok() {
        assert!(pre_scan_ast(
            &SExpr::Atom(Atom::Symbol(String::from("foo"))),
            &mut Vec::new()
        )
        .is_ok());
    }

    #[test]
    fn symbol_with_internal_underscore_ok() {
        // foo_bar: underscore is not prefix → ok
        assert!(pre_scan_ast(
            &SExpr::Atom(Atom::Symbol(String::from("foo_bar"))),
            &mut Vec::new()
        )
        .is_ok());
    }

    #[test]
    fn reserved_symbol_in_nested_ast() {
        // _x inside an instruction argument
        let program = SExpr::List(alloc::vec![
            SExpr::Atom(Atom::Symbol(String::from("add"))),
            SExpr::Atom(Atom::Symbol(String::from("_x"))),
            SExpr::Atom(Atom::Integer(1)),
        ]);
        let err = pre_scan_ast(&program, &mut Vec::new()).expect_err("should fail");
        match &err.kind {
            TypeCheckErrorKind::ReservedSymbolError { name } => {
                assert_eq!(name, "_x");
            }
            other => panic!("expected ReservedSymbolError, got {:?}", other),
        }
    }

    // ── 17. MP5 — simulate_raw Pre-Scan Regression Guards ───────

    #[test]
    fn simulate_raw_enforces_check_7() {
        // simulate_raw must run pre_scan_ast — @0 should fail even
        // when bypassing the root depth check.
        let err = tcr_err("(loop (drop @0))");
        assert_eq!(err.kind, TypeCheckErrorKind::InvalidHandleError);
    }

    #[test]
    fn simulate_raw_enforces_check_8() {
        // simulate_raw must run pre_scan_ast — reserved symbol triggers
        // ReservedSymbolError even through tcr_err path.
        let program = SExpr::List(alloc::vec![
            SExpr::Atom(Atom::Symbol(String::from("loop"))),
            SExpr::List(alloc::vec![SExpr::Atom(Atom::Symbol(String::from("_foo"))),]),
        ]);
        let err = simulate_raw(&program).expect_err("should fail");
        match &err.kind {
            TypeCheckErrorKind::ReservedSymbolError { name } => {
                assert_eq!(name, "_foo");
            }
            other => panic!("expected ReservedSymbolError, got {:?}", other),
        }
    }

    // ── 18. MP5 — Path Convention Documentation Test ─────────

    #[test]
    fn error_path_conventions_differ_by_type() {
        // Documents and enforces the architectural contract between pre_scan
        // and simulate path reporting.
        //
        // Atom-level errors report the path to the offending atom directly.
        // Instruction-level errors report the path to the enclosing
        // instruction list, with argument_index locating the bad arg.
        //
        // Invariant: atom_path = list_path ++ [argument_index + 2]
        //
        // Nested example: (if 1 <slot> 42), where <slot> is the then-branch.
        // Both queries target the same AST slot (first arg of an add in the
        // then-branch).
        //
        // 1-based positions:
        //   top-list:     if=1, cond=2, then=3, else=4
        //   within then:  add=1, arg0=2, arg1=3

        let err_prescan = tc_err("(if true (add @0 1) 42)");
        assert_eq!(err_prescan.kind, TypeCheckErrorKind::InvalidHandleError);
        assert_eq!(
            err_prescan.list_path,
            vec![3, 2],
            "pre_scan must report 1-based path to the offending atom"
        );

        let err_simulate = tc_err("(if true (add #x00 1) 42)");
        let argument_index = match &err_simulate.kind {
            TypeCheckErrorKind::TypeMismatch { argument_index, .. } => *argument_index,
            other => panic!("expected TypeMismatch, got {:?}", other),
        };
        assert_eq!(argument_index, 0, "bad arg is the first argument");
        assert_eq!(
            err_simulate.list_path,
            vec![3],
            "simulate must report 1-based path to the enclosing instruction list"
        );

        // Enforce the cross-convention invariant.
        let reconstructed_atom_path: Vec<usize> = {
            let mut p = err_simulate.list_path.clone();
            p.push(argument_index + 2);
            p
        };
        assert_eq!(
            err_prescan.list_path, reconstructed_atom_path,
            "atom_path must equal list_path ++ [argument_index + 2]"
        );
    }

    // ── 19. MP5 Phase 3 — JSON-LSP-Error-Output Tests ───────────

    #[test]
    fn json_output_stack_balance_error() {
        let err = tc_err("(recv)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"StackBalanceError\""));
        assert!(json.contains("\"expected\":1"));
        assert!(json.contains("\"actual\":3"));
        assert!(json.contains("\"list_path\":[]"));
    }

    #[test]
    fn json_output_invalid_handle_error() {
        let err = tc_err("(spawn @0)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"InvalidHandleError\""));
        assert!(json.contains("\"details\":null"));
    }

    #[test]
    fn json_output_reserved_symbol_error() {
        // Construct directly since parser may reject _foo as instruction name
        let err = TypeCheckError {
            kind: TypeCheckErrorKind::ReservedSymbolError {
                name: String::from("_foo"),
            },
            list_path: Vec::new(),
            message: String::from("reserved"),
        };
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"ReservedSymbolError\""));
        assert!(json.contains("\"name\":\"_foo\""));
    }

    #[test]
    fn json_output_type_mismatch_error() {
        let err = tc_err("(add #x00 1)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"TypeMismatch\""));
        assert!(json.contains("\"expected\":\"I64\""));
        assert!(json.contains("\"actual\":\"Bytes\""));
        assert!(json.contains("\"instruction\":\"add\""));
    }

    #[test]
    fn json_output_branch_stack_mismatch() {
        let err = tc_err("(if true 42 #x00)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"BranchStackMismatch\""));
        assert!(json.contains("\"then_stack\":[\"I64\"]"));
        assert!(json.contains("\"else_stack\":[\"Bytes\"]"));
    }

    #[test]
    fn json_output_arity_error() {
        let err = tcr_err("(if)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"ArityError\""));
        assert!(json.contains("\"instruction\":\"if\""));
        assert!(json.contains("\"expected\":4"));
        assert!(json.contains("\"actual\":1"));
    }

    #[test]
    fn json_output_bytes_size_error() {
        let data = alloc::vec![0u8; 16 * 1024 + 1];
        let program = SExpr::Atom(Atom::Bytes(data));
        let err = pre_scan_ast(&program, &mut Vec::new()).expect_err("should fail");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"BytesSizeError\""));
        assert!(json.contains("\"actual\":16385"));
        assert!(json.contains("\"max\":16384"));
    }

    #[test]
    fn json_output_break_outside_loop() {
        let err = tc_err("(break false)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"BreakOutsideLoopError\""));
        assert!(json.contains("\"details\":null"));
    }

    #[test]
    fn json_output_stack_underflow() {
        let err = tcr_err("(drop)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"StackUnderflow\""));
        assert!(json.contains("\"instruction\":\"drop\""));
        assert!(json.contains("\"needed\":1"));
        assert!(json.contains("\"available\":0"));
    }

    #[test]
    fn json_output_escapes_special_chars() {
        // Construct error with quote in symbol name for escaping test
        let err = TypeCheckError {
            kind: TypeCheckErrorKind::ReservedSymbolError {
                name: String::from("_with\"quote"),
            },
            list_path: Vec::new(),
            message: String::from("test \"with\" quotes"),
        };
        let json = err.to_json();
        assert!(json.contains("\\\"quote"));
        assert!(json.contains("test \\\"with\\\" quotes"));
    }

    #[test]
    fn json_output_list_path_nested() {
        // (if 1 (add @0 1) 42): pre_scan finds @0 at path [3,2]
        // if=[1], cond=[2], then=[3], within then: add=[1], @0=[2]
        let err = tc_err("(if true (add @0 1) 42)");
        assert_eq!(err.kind, TypeCheckErrorKind::InvalidHandleError);
        let json = err.to_json();
        assert!(json.contains("\"list_path\":[3,2]"));
    }

    #[test]
    fn json_output_simulate_path_nested() {
        // Coverage for simulate-produced list_paths in JSON output.
        // (if 1 (add #x00 1) 42): simulate catches TypeMismatch at arg 0
        // of the add in the then-branch.
        // Expected: list_path = [3], argument_index = 0, via the
        // path-convention invariant the atom sits at [3] ++ [0+2] = [3, 2].
        let err = tc_err("(if true (add #x00 1) 42)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"TypeMismatch\""));
        assert!(json.contains("\"list_path\":[3]"));
        assert!(json.contains("\"argument_index\":0"));
        assert!(json.contains("\"expected\":\"I64\""));
        assert!(json.contains("\"actual\":\"Bytes\""));
    }

    // ── Parameter outside function ─────────────────────────────

    #[test]
    fn parameter_at_top_level_produces_error() {
        let err = tc_err("%0");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::ParameterOutsideFunction { parameter_index: 0 }
        ));
        // Top-level atom → empty list_path
        assert!(err.list_path.is_empty());
    }

    #[test]
    fn parameter_nested_in_expression_produces_error() {
        let err = tc_err("(add %0 1)");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::ParameterOutsideFunction { parameter_index: 0 }
        ));
        // Path to the parameter atom: [2] (1-based)
        assert_eq!(err.list_path, vec![2]);
    }

    #[test]
    fn parameter_with_high_index_produces_error() {
        let err = tc_err("%999");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::ParameterOutsideFunction {
                parameter_index: 999
            }
        ));
    }

    #[test]
    fn json_output_parameter_outside_function() {
        let err = tc_err("(add %0 1)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"ParameterOutsideFunction\""));
        assert!(json.contains("\"parameter_index\":0"));
        assert!(json.contains("\"list_path\":[2]"));
    }

    // ── MP2a: program & fn ─────────────────────────────────────

    #[test]
    fn program_wrapper_with_single_fn_validates() {
        // fn body: (add %0 1) — param 0 is i64, add expects i64+i64, produces i64.
        // main expression: 42
        tc_ok("(program (fn add1 (i64) i64 (add %0 1)) 42)");
    }

    #[test]
    fn program_rejects_empty() {
        let err = tc_err("(program)");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::InvalidProgramStructureError
        ));
    }

    #[test]
    fn program_rejects_non_fn_before_tail() {
        let err = tc_err("(program 42 (fn foo () i64 1) 42)");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::InvalidProgramStructureError
        ));
    }

    #[test]
    fn fn_signature_rejects_unknown_param_type() {
        let err = tc_err("(program (fn foo (notatype) i64 1) 0)");
        match &err.kind {
            TypeCheckErrorKind::UnknownTypeError { symbol } => {
                assert_eq!(symbol, "notatype");
            }
            other => panic!("expected UnknownTypeError, got {:?}", other),
        }
    }

    #[test]
    fn fn_signature_rejects_unknown_return_type() {
        let err = tc_err("(program (fn foo (i64) notatype 1) 0)");
        match &err.kind {
            TypeCheckErrorKind::UnknownTypeError { symbol } => {
                assert_eq!(symbol, "notatype");
            }
            other => panic!("expected UnknownTypeError, got {:?}", other),
        }
    }

    #[test]
    fn fn_with_valid_body_returns_matching_type() {
        tc_ok("(program (fn mul2 (i64) i64 (add %0 %0)) 0)");
    }

    #[test]
    fn fn_body_return_type_mismatch() {
        // fn declares i64 return, but body produces bytes
        let err = tc_err("(program (fn foo () i64 #x00) 0)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn parameter_resolves_inside_fn_body() {
        // %0 refers to first param (i64), body returns i64
        tc_ok("(program (fn id (i64) i64 %0) 0)");
    }

    #[test]
    fn parameter_invalid_index_inside_fn() {
        let err = tc_err("(program (fn foo (i64) i64 %5) 0)");
        match &err.kind {
            TypeCheckErrorKind::InvalidParameterIndex {
                parameter_index,
                arity,
            } => {
                assert_eq!(*parameter_index, 5);
                assert_eq!(*arity, 1);
            }
            other => panic!("expected InvalidParameterIndex, got {:?}", other),
        }
    }

    #[test]
    fn parameter_still_outside_fn_at_top_level() {
        // MP1 regression guard
        let err = tc_err("%0");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::ParameterOutsideFunction { .. }
        ));
    }

    #[test]
    fn nested_fn_rejected() {
        let err = tc_err("(program (fn foo () i64 (fn nested () i64 1)) 0)");
        assert!(matches!(err.kind, TypeCheckErrorKind::NestedFunctionError));
    }

    #[test]
    fn duplicate_fn_rejected() {
        let err = tc_err("(program (fn foo () i64 1) (fn foo () i64 2) 0)");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::DuplicateFunctionError
        ));
    }

    #[test]
    fn fn_wrong_arity_structure() {
        // fn should have exactly 5 items: (fn name params return-type body)
        let err = tc_err("(program (fn foo () i64) 0)"); // missing body
        assert!(matches!(err.kind, TypeCheckErrorKind::ArityError { .. }));
    }

    #[test]
    fn legacy_single_expression_still_works() {
        tc_ok("42");
        tc_ok("(add 1 2)");
        let err = tc_err("(add #x00 1)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    // ── MP2a JSON tests ────────────────────────────────────────

    #[test]
    fn json_output_unknown_type_error() {
        let err = tc_err("(program (fn foo (notatype) i64 1) 0)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"UnknownTypeError\""));
        assert!(json.contains("\"symbol\":\"notatype\""));
    }

    #[test]
    fn json_output_invalid_parameter_index() {
        let err = tc_err("(program (fn foo (i64) i64 %5) 0)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"InvalidParameterIndex\""));
        assert!(json.contains("\"parameter_index\":5"));
        assert!(json.contains("\"arity\":1"));
    }

    #[test]
    fn json_output_nested_function_error() {
        let err = tc_err("(program (fn foo () i64 (fn nested () i64 1)) 0)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"NestedFunctionError\""));
    }

    // ── MP2b: call instruction ─────────────────────────────────

    #[test]
    fn call_simple_function() {
        tc_ok("(program (fn add1 (i64) i64 (add %0 1)) (call add1 42))");
    }

    #[test]
    fn call_returns_correct_type() {
        // call pushes the return type, used as argument to add
        tc_ok("(program (fn id (i64) i64 %0) (add (call id 5) 10))");
    }

    #[test]
    fn call_zero_arity_function() {
        tc_ok("(program (fn one () i64 1) (call one))");
    }

    #[test]
    fn call_undefined_function_errors() {
        let err = tc_err("(program (call ghost 1 2))");
        match &err.kind {
            TypeCheckErrorKind::UndefinedFunctionError { name } => {
                assert_eq!(name, "ghost");
            }
            other => panic!("expected UndefinedFunctionError, got {:?}", other),
        }
    }

    #[test]
    fn call_too_few_args_errors() {
        let err = tc_err("(program (fn add2 (i64 i64) i64 (add %0 %1)) (call add2 1))");
        match &err.kind {
            TypeCheckErrorKind::FunctionArityMismatch {
                function_name,
                expected,
                actual,
            } => {
                assert_eq!(function_name, "add2");
                assert_eq!(*expected, 2);
                assert_eq!(*actual, 1);
            }
            other => panic!("expected FunctionArityMismatch, got {:?}", other),
        }
    }

    #[test]
    fn call_too_many_args_errors() {
        let err = tc_err("(program (fn id (i64) i64 %0) (call id 1 2))");
        match &err.kind {
            TypeCheckErrorKind::FunctionArityMismatch {
                expected, actual, ..
            } => {
                assert_eq!(*expected, 1);
                assert_eq!(*actual, 2);
            }
            other => panic!("expected FunctionArityMismatch, got {:?}", other),
        }
    }

    #[test]
    fn call_wrong_arg_type_errors() {
        let err = tc_err("(program (fn id (i64) i64 %0) (call id #x00))");
        match &err.kind {
            TypeCheckErrorKind::FunctionTypeMismatch {
                function_name,
                argument_index,
                expected,
                actual,
                ..
            } => {
                assert_eq!(function_name, "id");
                assert_eq!(*argument_index, 0);
                assert_eq!(*expected, ValueType::I64);
                assert_eq!(*actual, ValueType::Bytes);
            }
            other => panic!("expected FunctionTypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn call_wrong_arg_type_at_second_position() {
        let err = tc_err("(program (fn add2 (i64 i64) i64 (add %0 %1)) (call add2 1 #x00))");
        match &err.kind {
            TypeCheckErrorKind::FunctionTypeMismatch { argument_index, .. } => {
                assert_eq!(*argument_index, 1);
            }
            other => panic!("expected FunctionTypeMismatch at arg 1, got {:?}", other),
        }
    }

    #[test]
    fn self_recursive_function() {
        // Paket B.1b: if-cond must be Bool; convert I64 param via (ne %0 0).
        tc_ok(
            "(program (fn countdown (i64) i64 (if (ne %0 0) (call countdown %0) 0)) (call countdown 5))",
        );
    }

    #[test]
    fn mutually_recursive_functions() {
        // A calls B, B calls A. Pass 1 collects both signatures before
        // Pass 2 validates bodies — mutual recursion is a free win.
        let src = "(program \
            (fn ping (i64) i64 (call pong %0)) \
            (fn pong (i64) i64 (call ping %0)) \
            (call ping 42))";
        tc_ok(src);
    }

    #[test]
    fn call_with_handle_param() {
        tc_ok("(program (fn hid (handle) handle %0) (call hid @1))");
    }

    #[test]
    fn undefined_function_path_points_to_call_list() {
        let err = tc_err("(program (call ghost))");
        assert!(matches!(
            &err.kind,
            TypeCheckErrorKind::UndefinedFunctionError { .. }
        ));
        // main expr is at path [2], call list IS the main expr
        assert_eq!(err.list_path, vec![2]);
    }

    // ── MP2b JSON tests ────────────────────────────────────────

    #[test]
    fn json_output_undefined_function() {
        let err = tc_err("(program (call ghost))");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"UndefinedFunctionError\""));
        assert!(json.contains("\"name\":\"ghost\""));
    }

    #[test]
    fn json_output_function_arity_mismatch() {
        let err = tc_err("(program (fn id (i64) i64 %0) (call id 1 2))");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"FunctionArityMismatch\""));
        assert!(json.contains("\"function_name\":\"id\""));
        assert!(json.contains("\"expected\":1"));
        assert!(json.contains("\"actual\":2"));
    }

    #[test]
    fn json_output_function_type_mismatch() {
        let err = tc_err("(program (fn id (i64) i64 %0) (call id #x00))");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"FunctionTypeMismatch\""));
        assert!(json.contains("\"function_name\":\"id\""));
        assert!(json.contains("\"argument_index\":0"));
        assert!(json.contains("\"expected\":\"I64\""));
        assert!(json.contains("\"actual\":\"Bytes\""));
    }

    // ── Post-MP2b: position_offset sanity ──────────────────────

    #[test]
    fn type_mismatch_has_position_offset_2() {
        let err = tc_err("(add #x00 1)");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                position_offset, ..
            } => {
                assert_eq!(*position_offset, 2);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn function_type_mismatch_has_position_offset_3() {
        let err = tc_err("(program (fn id (i64) i64 %0) (call id #x00))");
        match &err.kind {
            TypeCheckErrorKind::FunctionTypeMismatch {
                position_offset, ..
            } => {
                assert_eq!(*position_offset, 3);
            }
            other => panic!("expected FunctionTypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn json_output_type_mismatch_includes_position_offset() {
        let err = tc_err("(add #x00 1)");
        let json = err.to_json();
        assert!(json.contains("\"position_offset\":2"));
    }

    #[test]
    fn json_output_function_type_mismatch_includes_position_offset() {
        let err = tc_err("(program (fn id (i64) i64 %0) (call id #x00))");
        let json = err.to_json();
        assert!(json.contains("\"position_offset\":3"));
    }
}

// ── simulate_let ───────────────────────────────────────────────

/// Simulate `(let %n expr body)`.
///
/// Evaluates `expr`, pops its result from the stack, binds it to `%n`
/// as a local in the current `FunctionContext`, evaluates `body` with
/// the binding active, then unbinds `%n` and returns the body's result.
fn simulate_let(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // (let %n expr body) → items[0]=let, items[1]=%n, items[2]=expr, items[3]=body
    if items.len() != 4 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("let"),
                expected: 3,
                actual: items.len() - 1,
            },
            list_path: path.clone(),
            message: String::from("let requires: parameter-index, expression, body"),
        });
    }

    // items[1] must be a Parameter atom
    let idx = match &items[1] {
        SExpr::Atom(Atom::Parameter(n)) => *n,
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: String::from("let binding target must be a parameter atom (%n)"),
            });
        }
    };

    // let is only valid inside a function body
    if fn_stack.is_empty() {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ParameterOutsideFunction {
                parameter_index: idx,
            },
            list_path: path.clone(),
            message: String::from("let can only appear inside a function body"),
        });
    }

    // Simulate expr (items[2])
    path.push(3); // 1-based: let=1, %n=2, expr=3
    let expr_result = simulate(
        &items[2],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();

    // If expr always breaks/returns, the body never executes.
    match expr_result {
        SimResult::Continues(_) => {}
        SimResult::Breaks(_) | SimResult::Returns(_) => {
            return Ok(expr_result);
        }
    }

    // Pop the expr's result from the stack as the local's value type
    let local_type = match stack.pop() {
        Some(ty) => ty,
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::StackUnderflow {
                    instruction: String::from("let"),
                    needed: 1,
                    available: 0,
                },
                list_path: path.clone(),
                message: String::from("let expression did not produce a value"),
            });
        }
    };

    // Bind the local. Pre-A.1 (F-C7): `fn_stack` is guaranteed
    // non-empty by the `is_empty()` guard above; surface a typed
    // error rather than panicking if a future refactor breaks that
    // invariant.
    let ctx = fn_stack.last_mut().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated("simulate_let: fn_stack empty at bind"),
        list_path: path.clone(),
        message: String::from("internal invariant violated: simulate_let fn_stack empty at bind"),
    })?;
    match ctx.bind_local(idx, local_type) {
        Ok(()) => {}
        Err(LetBindError::CollidesWithParameter) => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::LetCollidesWithParameter {
                    parameter_index: idx,
                },
                list_path: path.clone(),
                message: format!("let cannot rebind parameter %{}", idx),
            });
        }
        Err(LetBindError::CollidesWithExistingLocal) => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::LetRedefinitionError {
                    parameter_index: idx,
                },
                list_path: path.clone(),
                message: format!("local %{} is already bound in this scope", idx),
            });
        }
    }

    // Simulate the body (items[3]) with the new binding active
    path.push(4); // 1-based: let=1, %n=2, expr=3, body=4
    let body_result = simulate(
        &items[3],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    );
    path.pop();

    // Unbind the local before propagating the result (cleanup even
    // on error). Pre-A.1 (F-C7): if `fn_stack` was unexpectedly
    // drained by the body simulation, propagate that as a typed
    // error rather than panicking, preserving the underlying
    // `body_result` only when the unbind succeeds.
    let ctx = fn_stack.last_mut().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_let: fn_stack empty at unbind",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: simulate_let fn_stack empty at unbind"),
    })?;
    ctx.unbind_local(idx);

    body_result
}

// ── simulate_seq ───────────────────────────────────────────────

/// Simulate `(seq effect1 effect2 ... value)`.
///
/// Variadic with minimum 2 arguments. Each effect argument (all but
/// the last) must be stack-neutral. The last argument produces the
/// result value.
fn simulate_seq(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // (seq arg1 arg2 ... argN) → items[0]=seq, items[1..]=args
    let arg_count = items.len() - 1;
    if arg_count < 2 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("seq"),
                expected: 2,
                actual: arg_count,
            },
            list_path: path.clone(),
            message: String::from("seq requires at least 2 arguments"),
        });
    }

    let args = &items[1..];
    let effect_count = args.len() - 1;

    // Process effect arguments (all but last): must be stack-neutral
    for i in 0..effect_count {
        let stack_before = stack.len();
        path.push(i + 2); // 1-based: seq=1, args start at position 2
        let result = simulate(
            &args[i],
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        )?;
        path.pop();

        // Break/Return propagation: do NOT check stack-neutrality
        match result {
            SimResult::Continues(_) => {}
            SimResult::Breaks(_) | SimResult::Returns(_) => {
                return Ok(result);
            }
        }

        // Stack-neutrality check
        if stack.len() != stack_before {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::SeqEffectNotStackNeutral {
                    argument_index: i as u32,
                    expected_depth: stack_before,
                    actual_depth: stack.len(),
                    position_offset: 2,
                },
                list_path: path.clone(),
                message: format!(
                    "seq effect argument {} changed stack depth from {} to {}",
                    i,
                    stack_before,
                    stack.len()
                ),
            });
        }
    }

    // Process last argument (value-producing)
    let last_idx = args.len() - 1;
    path.push(last_idx + 2); // 1-based position
    let result = simulate(
        &args[last_idx],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();

    Ok(result)
}

// ── simulate_intent ────────────────────────────────────────────

/// Simulate `(intent arg1 arg2 ... argN)`.
///
/// Phase 1 stub: accepts any number of arguments of any type,
/// type-checks each argument recursively, then pushes i64 (status).
fn simulate_intent(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // (intent arg1 arg2 ... argN) → items[0]=intent, items[1..]=args
    let args = &items[1..];

    // Type-check each argument.
    //
    // Each arg MUST produce exactly one value on the stack — that
    // is the contract for `(intent …)` arguments today. The
    // previous code called `stack.pop()` unconditionally, which is
    // unsound: a sub-expression that pushed zero values (e.g.
    // a bare symbol that the validator silently passes through —
    // see review §4.1, QUARKS_REVIEW.md) would have its
    // unconditional pop steal an unrelated value from the
    // surrounding stack and validate a malformed program. Mirror
    // the `simulate_policy_or_query` shape: snapshot the depth
    // before, then enforce `+1` after, then pop.
    for (i, arg) in args.iter().enumerate() {
        let depth_before = stack.len();
        path.push(i + 2); // 1-based: intent=1, args start at 2
        let result = simulate(
            arg,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        )?;
        path.pop();

        // Break/Return propagation
        match result {
            SimResult::Continues(_) => {}
            SimResult::Breaks(_) | SimResult::Returns(_) => {
                return Ok(result);
            }
        }

        if stack.len() != depth_before + 1 {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::StackBalanceError {
                    expected: 1,
                    actual: stack.len().saturating_sub(depth_before),
                },
                list_path: path.clone(),
                message: format!(
                    "intent arg {} must push exactly 1 value (stack depth went from {} to {})",
                    i,
                    depth_before,
                    stack.len()
                ),
            });
        }

        // Pop the value that each arg pushed (we consume args, don't keep them on stack)
        stack.pop();
    }

    // Push i64 status result
    stack.push(ValueType::I64);
    Ok(SimResult::Continues(stack.clone()))
}

// ── simulate_policy_or_query (12i) ─────────────────────────────

/// 12i — type-check `(policy <subsystem> <operation> arg1 .. argN)` or
/// `(query <subsystem> <metric> arg1 .. argN)`. The subsystem and
/// operation/metric slots must be symbol atoms (literals, not stack-
/// produced values). The remaining args are evaluated as I64-typed
/// values; the form pushes a single I64 result (status code for
/// policy, telemetry value for query). Capability enforcement is a
/// runtime concern handled by [`PolicyContext`] in the interpreter.
#[allow(clippy::too_many_arguments)]
fn simulate_policy_or_query(
    head: &str,
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // (policy <subsys> <op> <args...>) — items[0]=head,
    // items[1]=subsystem, items[2]=operation, items[3..]=value-args.
    let args = &items[1..];
    if args.len() < 2 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from(head),
                expected: 2,
                actual: args.len(),
            },
            list_path: path.clone(),
            message: format!("{} requires at least <subsystem> <operation>", head),
        });
    }

    // Subsystem and operation must be symbol literals (not stack-
    // produced). Reject anything else with a structural error.
    if !matches!(&args[0], SExpr::Atom(Atom::Symbol(_))) {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::InvalidProgramStructureError,
            list_path: path.clone(),
            message: format!("{} subsystem (arg 0) must be a symbol literal", head),
        });
    }
    if !matches!(&args[1], SExpr::Atom(Atom::Symbol(_))) {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::InvalidProgramStructureError,
            list_path: path.clone(),
            message: format!("{} operation (arg 1) must be a symbol literal", head),
        });
    }

    // Type-check value-args. Each must produce exactly one I64 value
    // — the policy/query ABI passes integer scalars only (sandbox-id,
    // percentage, ms, kbps, bytes, etc.). Non-I64 args fail the type
    // check.
    for (i, arg) in args[2..].iter().enumerate() {
        let depth_before = stack.len();
        path.push(i + 4); // 1-based: head=1, subsys=2, op=3, arg0=4
        let result = simulate(
            arg,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        )?;
        path.pop();

        match result {
            SimResult::Continues(_) => {}
            SimResult::Breaks(_) | SimResult::Returns(_) => {
                return Ok(result);
            }
        }

        if stack.len() != depth_before + 1 {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::StackBalanceError {
                    expected: 1,
                    actual: stack.len().saturating_sub(depth_before),
                },
                list_path: path.clone(),
                message: format!(
                    "{} arg {} must push exactly 1 value (stack depth went from {} to {})",
                    head,
                    i,
                    depth_before,
                    stack.len()
                ),
            });
        }

        let actual = stack[stack.len() - 1];
        if actual != ValueType::I64 {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::TypeMismatch {
                    instruction: String::from(head),
                    argument_index: i,
                    expected: ValueType::I64,
                    actual,
                    position_offset: 4, // head=1, subsys=2, op=3, arg0=4
                },
                list_path: path.clone(),
                message: format!("{} arg {} must be i64, got {:?}", head, i, actual),
            });
        }

        // Pop the consumed value: policy/query consume their args.
        stack.pop();
    }

    // Push i64 result (status for policy, telemetry value for query).
    stack.push(ValueType::I64);
    Ok(SimResult::Continues(stack.clone()))
}

// ── simulate_discard ───────────────────────────────────────────

/// Simulate `(discard expr)`.
///
/// Evaluates `expr` (which pushes exactly one value onto the stack),
/// then pops that value and discards it. The result is stack-neutral.
///
/// Primary use case: wrapping value-producing instructions (like
/// `intent` or function calls) in `seq` effect positions. Without
/// `discard`, side-effect-only sequences couldn't be expressed
/// cleanly in the IR.
///
/// Path arithmetic: items[0]=discard (1-based pos 1), items[1]=expr (pos 2).
fn simulate_discard(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // Arity check: exactly 1 argument
    if items.len() != 2 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("discard"),
                expected: 1,
                actual: items.len() - 1,
            },
            list_path: path.clone(),
            message: format!("discard expects 1 argument, got {}", items.len() - 1),
        });
    }

    // Snapshot stack depth before evaluating expr
    let depth_before = stack.len();

    // Evaluate expr (push value)
    path.push(2); // 1-based: discard=1, expr=2
    let result = simulate(
        &items[1],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();

    // Break/Return propagation
    match result {
        SimResult::Continues(_) => {}
        SimResult::Breaks(_) | SimResult::Returns(_) => {
            return Ok(result);
        }
    }

    // Verify expr pushed exactly one value
    if stack.len() != depth_before + 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackUnderflow {
                instruction: String::from("discard"),
                needed: 1,
                available: if stack.len() > depth_before {
                    stack.len() - depth_before
                } else {
                    0
                },
            },
            list_path: path.clone(),
            message: format!(
                "discard expression must push exactly 1 value (stack depth went from {} to {})",
                depth_before,
                stack.len()
            ),
        });
    }

    // Pop the value (discard it)
    stack.pop();

    Ok(SimResult::Continues(stack.clone()))
}

// ── Paket B.2/B.3/B.4/B.6 — variadic / shape-specific forms ───

/// Paket B.2 — `(list e1 e2 … eN)` constructs a list of `i64`. Each
/// argument expression must evaluate to a single `i64` value (the
/// elements are popped off the stack after evaluation); the form
/// pushes one `ValueType::List` as its result.
fn simulate_list(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // `(list)` constructs an empty list of i64.
    let args = &items[1..];
    for (i, arg) in args.iter().enumerate() {
        let depth_before = stack.len();
        path.push(i + 2); // 1-based: list=1, args start at 2
        let result = simulate(
            arg,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        )?;
        path.pop();
        match result {
            SimResult::Continues(_) => {}
            other => return Ok(other),
        }
        if stack.len() != depth_before + 1 {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::StackBalanceError {
                    expected: depth_before + 1,
                    actual: stack.len(),
                },
                list_path: path.clone(),
                message: format!(
                    "list arg {} must push exactly 1 value (stack went from {} to {})",
                    i,
                    depth_before,
                    stack.len()
                ),
            });
        }
        let actual = stack[stack.len() - 1];
        if actual != ValueType::I64 {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::TypeMismatch {
                    instruction: String::from("list"),
                    argument_index: i,
                    expected: ValueType::I64,
                    actual,
                    position_offset: 2,
                },
                list_path: path.clone(),
                message: format!("list element {} must be i64, got {:?}", i, actual),
            });
        }
        // Pop the consumed element from the stack — the list ctor
        // absorbs its args.
        stack.pop();
    }
    stack.push(ValueType::List);
    Ok(SimResult::Continues(stack.clone()))
}

/// Paket B.4 — `(loop-with-bound bound-expr body)` is a bounded loop:
/// at runtime it iterates at most `bound-expr` times. If `(break v)`
/// fires inside the body within that bound, the loop produces `v`;
/// otherwise an implicit `(break false)` fires when the bound is
/// reached. The loop's output type is therefore always `Bool` (same
/// shape as `(while …)`); user-level breaks must produce `Bool`.
fn simulate_loop_with_bound(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    _outer_loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    if items.len() != 3 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("loop-with-bound"),
                expected: 2,
                actual: items.len() - 1,
            },
            list_path: path.clone(),
            message: format!(
                "loop-with-bound expects (loop-with-bound bound body), got {} items",
                items.len()
            ),
        });
    }

    let stack_in = stack.clone();

    // Evaluate the bound expression — must produce a single I64.
    path.push(2);
    let bound_result = simulate(
        &items[1],
        stack,
        path,
        None,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match bound_result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }
    if stack.len() != stack_in.len() + 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: stack_in.len() + 1,
                actual: stack.len(),
            },
            list_path: path.clone(),
            message: format!(
                "loop-with-bound bound expression must push 1 value (stack went from {} to {})",
                stack_in.len(),
                stack.len()
            ),
        });
    }
    let bound_type = stack.pop().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_loop_with_bound: stack.pop() returned None despite depth guard",
        ),
        list_path: path.clone(),
        message: String::from(
            "internal invariant violated: simulate_loop_with_bound bound stack vanished",
        ),
    })?;
    if bound_type != ValueType::I64 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::TypeMismatch {
                instruction: String::from("loop-with-bound"),
                argument_index: 0,
                expected: ValueType::I64,
                actual: bound_type,
                position_offset: 2,
            },
            list_path: path.clone(),
            message: String::from("loop-with-bound bound must be i64"),
        });
    }

    // Seed the loop ctx's break_type with Bool — matches the implicit
    // (break false) that fires when the bound is reached. User-level
    // breaks inside the body must agree.
    let loop_ctx = LoopCtx {
        entry_stack: stack_in.clone(),
        break_type: Cell::new(Some(ValueType::Bool)),
    };

    path.push(3);
    let body_result = simulate(
        &items[2],
        stack,
        path,
        Some(&loop_ctx),
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();

    finalise_loop_body(loop_ctx, stack_in, body_result, stack, path)
}

/// Paket B.4 — `(for %n source body)` iterates `source` (which must
/// be a `List`), binding each element to `%n` (an I64 local) and
/// evaluating `body` for its side effects. The body's residual value
/// is discarded each iteration (like `while`). The form's output type
/// is `Bool` — `Bool(true)` on normal exhaustion of the list, the
/// `(break v)` value otherwise (and all user breaks must be `Bool`).
fn simulate_for(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    _outer_loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    if items.len() != 4 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("for"),
                expected: 3,
                actual: items.len() - 1,
            },
            list_path: path.clone(),
            message: format!(
                "for expects (for %n source body), got {} items",
                items.len()
            ),
        });
    }

    // items[1] must be a Parameter atom (binding name).
    let idx = match &items[1] {
        SExpr::Atom(Atom::Parameter(n)) => *n,
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: String::from("for binding target must be a parameter atom (%n)"),
            });
        }
    };

    // `for` must live inside a function body (parameter binding needs
    // a frame to bind into).
    if fn_stack.is_empty() {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ParameterOutsideFunction {
                parameter_index: idx,
            },
            list_path: path.clone(),
            message: String::from("for can only appear inside a function body"),
        });
    }

    let stack_in = stack.clone();

    // Evaluate the source — must produce a single List.
    path.push(3);
    let source_result = simulate(
        &items[2],
        stack,
        path,
        None,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match source_result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }
    if stack.len() != stack_in.len() + 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: stack_in.len() + 1,
                actual: stack.len(),
            },
            list_path: path.clone(),
            message: format!(
                "for source must push 1 value (stack went from {} to {})",
                stack_in.len(),
                stack.len()
            ),
        });
    }
    let source_type = stack.pop().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_for: stack.pop() returned None despite depth guard",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: simulate_for source stack vanished"),
    })?;
    if source_type != ValueType::List {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::TypeMismatch {
                instruction: String::from("for"),
                argument_index: 1,
                expected: ValueType::List,
                actual: source_type,
                position_offset: 2,
            },
            list_path: path.clone(),
            message: String::from("for source must be a list"),
        });
    }

    // Bind %n as I64 (list elements) before body simulation.
    let ctx = fn_stack.last_mut().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_for: fn_stack empty at bind despite is_empty guard",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: simulate_for fn_stack empty at bind"),
    })?;
    match ctx.bind_local(idx, ValueType::I64) {
        Ok(()) => {}
        Err(LetBindError::CollidesWithParameter) => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::LetCollidesWithParameter {
                    parameter_index: idx,
                },
                list_path: path.clone(),
                message: format!("for cannot rebind parameter %{}", idx),
            });
        }
        Err(LetBindError::CollidesWithExistingLocal) => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::LetRedefinitionError {
                    parameter_index: idx,
                },
                list_path: path.clone(),
                message: format!("for binding %{} is already bound in this scope", idx),
            });
        }
    }

    // Seed the loop ctx with Bool (matches the implicit Bool result of
    // normal completion).
    let loop_ctx = LoopCtx {
        entry_stack: stack_in.clone(),
        break_type: Cell::new(Some(ValueType::Bool)),
    };

    path.push(4);
    let body_result = simulate(
        &items[3],
        stack,
        path,
        Some(&loop_ctx),
        fn_table,
        fn_stack,
        struct_table,
    );
    path.pop();

    // Always unbind the local before propagating.
    let ctx = fn_stack.last_mut().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_for: fn_stack empty at unbind",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: simulate_for fn_stack empty at unbind"),
    })?;
    ctx.unbind_local(idx);

    let body_result = body_result?;
    finalise_loop_body(loop_ctx, stack_in, body_result, stack, path)
}

/// Paket B.6 — `(write-host-state key-symbol value-expr)` writes a
/// scalar value to the host's structured-state surface under the
/// given key. The key slot is a symbol literal (mirroring
/// `policy`/`query`); the value slot is an `i64` expression. Returns
/// an `i64` status code (`0 = success`).
fn simulate_write_host_state(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    if items.len() != 3 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("write-host-state"),
                expected: 2,
                actual: items.len() - 1,
            },
            list_path: path.clone(),
            message: format!(
                "write-host-state expects (write-host-state key-symbol value), got {} items",
                items.len()
            ),
        });
    }

    // items[1] must be a symbol literal (the namespace key).
    if !matches!(&items[1], SExpr::Atom(Atom::Symbol(_))) {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::InvalidProgramStructureError,
            list_path: path.clone(),
            message: String::from("write-host-state key (arg 0) must be a symbol literal"),
        });
    }

    // Evaluate the value expression — must produce a single I64.
    let depth_before = stack.len();
    path.push(3);
    let value_result = simulate(
        &items[2],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match value_result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }
    if stack.len() != depth_before + 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: depth_before + 1,
                actual: stack.len(),
            },
            list_path: path.clone(),
            message: format!(
                "write-host-state value must push 1 value (stack went from {} to {})",
                depth_before,
                stack.len()
            ),
        });
    }
    let value_type = stack[stack.len() - 1];
    if value_type != ValueType::I64 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::TypeMismatch {
                instruction: String::from("write-host-state"),
                argument_index: 1,
                expected: ValueType::I64,
                actual: value_type,
                position_offset: 3,
            },
            list_path: path.clone(),
            message: format!("write-host-state value must be i64, got {:?}", value_type),
        });
    }
    // Consume the value and push the i64 status result.
    stack.pop();
    stack.push(ValueType::I64);
    Ok(SimResult::Continues(stack.clone()))
}

// ── Phase 4 Step 6 — Struct simulation ─────────────────────────

/// Type-check `(struct-new name v1 v2 … vN)`. The struct name must
/// be a registered struct symbol; each value arg must match the
/// declared field type at its positional index. The form pushes a
/// single `ValueType::Struct(idx)` value.
fn simulate_struct_new(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // items[0]=struct-new, items[1]=name symbol, items[2..]=value args.
    if items.len() < 2 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("struct-new"),
                expected: 1,
                actual: items.len() - 1,
            },
            list_path: path.clone(),
            message: String::from("struct-new requires (struct-new name v1 v2 …)"),
        });
    }
    let name = match &items[1] {
        SExpr::Atom(Atom::Symbol(s)) => s.clone(),
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: String::from("struct-new target must be a struct-name symbol literal"),
            });
        }
    };
    let idx = match struct_table.lookup_index(&name) {
        Some(i) => i,
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::UnknownStructError { name: name.clone() },
                list_path: path.clone(),
                message: format!("unknown struct '{}'", name),
            });
        }
    };
    let info = match struct_table.lookup(idx) {
        Some(s) => s.clone(),
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "simulate_struct_new: registered index missing entry",
                ),
                list_path: path.clone(),
                message: String::from("internal invariant violated: struct index without entry"),
            });
        }
    };

    let args = &items[2..];
    if args.len() != info.fields.len() {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StructFieldCountMismatch {
                struct_name: name,
                expected: info.fields.len(),
                actual: args.len(),
            },
            list_path: path.clone(),
            message: format!(
                "struct '{}' expects {} field values, got {}",
                info.name,
                info.fields.len(),
                args.len()
            ),
        });
    }

    let depth_before = stack.len();
    for (i, arg) in args.iter().enumerate() {
        let pre = stack.len();
        path.push(i + 3); // 1-based: struct-new=1, name=2, args start at 3
        let result = simulate(
            arg,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        )?;
        path.pop();
        match result {
            SimResult::Continues(_) => {}
            other => return Ok(other),
        }
        if stack.len() != pre + 1 {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::StackBalanceError {
                    expected: pre + 1,
                    actual: stack.len(),
                },
                list_path: path.clone(),
                message: format!(
                    "struct-new arg {} must push exactly 1 value (stack went from {} to {})",
                    i,
                    pre,
                    stack.len()
                ),
            });
        }
        let actual = stack[stack.len() - 1];
        let expected = info.fields[i].1;
        if actual != expected {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::TypeMismatch {
                    instruction: String::from("struct-new"),
                    argument_index: i,
                    expected,
                    actual,
                    position_offset: 3,
                },
                list_path: path.clone(),
                message: format!(
                    "struct '{}' field '{}' expects {:?}, got {:?}",
                    info.name, info.fields[i].0, expected, actual
                ),
            });
        }
        // Consume the value — struct-new absorbs its args.
        stack.pop();
    }

    // Sanity check: stack should be back at depth_before.
    if stack.len() != depth_before {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::InternalInvariantViolated(
                "simulate_struct_new: stack imbalance after consuming args",
            ),
            list_path: path.clone(),
            message: String::from("internal invariant violated: struct-new stack imbalance"),
        });
    }

    stack.push(ValueType::Struct(idx));
    Ok(SimResult::Continues(stack.clone()))
}

/// Type-check `(struct-get expr field-name)`. The expression must
/// produce a Struct value; the field name must exist in that struct.
/// The form pushes a single value of the field's declared type.
fn simulate_struct_get(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    if items.len() != 3 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("struct-get"),
                expected: 2,
                actual: items.len() - 1,
            },
            list_path: path.clone(),
            message: String::from("struct-get requires (struct-get expr field-name)"),
        });
    }
    let field_name = match &items[2] {
        SExpr::Atom(Atom::Symbol(s)) => s.clone(),
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: String::from("struct-get field-name must be a symbol literal"),
            });
        }
    };

    let depth_before = stack.len();
    path.push(2); // 1-based: struct-get=1, expr=2
    let result = simulate(
        &items[1],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }
    if stack.len() != depth_before + 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: depth_before + 1,
                actual: stack.len(),
            },
            list_path: path.clone(),
            message: format!(
                "struct-get expr must push 1 value (stack went from {} to {})",
                depth_before,
                stack.len()
            ),
        });
    }
    let struct_ty = stack.pop().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_struct_get: stack.pop returned None despite length guard",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: struct-get stack vanished"),
    })?;
    let idx = match struct_ty {
        ValueType::Struct(i) => i,
        other => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::TypeMismatch {
                    instruction: String::from("struct-get"),
                    argument_index: 0,
                    // We don't have a single concrete Struct(idx) to
                    // expect here, so report ValueType::Struct(0) as
                    // a generic placeholder — the consumer sees
                    // value_type_name == "Struct".
                    expected: ValueType::Struct(0),
                    actual: other,
                    position_offset: 2,
                },
                list_path: path.clone(),
                message: format!(
                    "struct-get target must be a Struct value, got {}",
                    value_type_name(other)
                ),
            });
        }
    };
    let info = match struct_table.lookup(idx) {
        Some(s) => s,
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "simulate_struct_get: stack-borne struct idx missing entry",
                ),
                list_path: path.clone(),
                message: String::from("internal invariant violated: struct-get idx without entry"),
            });
        }
    };
    let field_ty = match info
        .fields
        .iter()
        .find(|(n, _)| n == &field_name)
        .map(|(_, t)| *t)
    {
        Some(t) => t,
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::UnknownFieldError {
                    struct_name: info.name.clone(),
                    field: field_name.clone(),
                },
                list_path: path.clone(),
                message: format!("struct '{}' has no field '{}'", info.name, field_name),
            });
        }
    };
    stack.push(field_ty);
    Ok(SimResult::Continues(stack.clone()))
}

/// Type-check `(struct-set expr field-name new-value)`. The
/// expression must produce a Struct; the field name must exist; the
/// new value's type must match the field's declared type. Functional
/// update: the form pushes a single value of the SAME Struct type
/// (no mutation, returns a new Struct).
fn simulate_struct_set(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    if items.len() != 4 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("struct-set"),
                expected: 3,
                actual: items.len() - 1,
            },
            list_path: path.clone(),
            message: String::from("struct-set requires (struct-set expr field-name new-value)"),
        });
    }
    let field_name = match &items[2] {
        SExpr::Atom(Atom::Symbol(s)) => s.clone(),
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: String::from("struct-set field-name must be a symbol literal"),
            });
        }
    };

    // Evaluate the struct expression.
    let depth_before = stack.len();
    path.push(2);
    let expr_result = simulate(
        &items[1],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match expr_result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }
    if stack.len() != depth_before + 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: depth_before + 1,
                actual: stack.len(),
            },
            list_path: path.clone(),
            message: format!(
                "struct-set expr must push 1 value (stack went from {} to {})",
                depth_before,
                stack.len()
            ),
        });
    }
    let struct_ty = stack.pop().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_struct_set: stack.pop returned None despite length guard",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: struct-set stack vanished"),
    })?;
    let idx = match struct_ty {
        ValueType::Struct(i) => i,
        other => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::TypeMismatch {
                    instruction: String::from("struct-set"),
                    argument_index: 0,
                    expected: ValueType::Struct(0),
                    actual: other,
                    position_offset: 2,
                },
                list_path: path.clone(),
                message: format!(
                    "struct-set target must be a Struct value, got {}",
                    value_type_name(other)
                ),
            });
        }
    };
    let info = match struct_table.lookup(idx) {
        Some(s) => s.clone(),
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InternalInvariantViolated(
                    "simulate_struct_set: stack-borne struct idx missing entry",
                ),
                list_path: path.clone(),
                message: String::from("internal invariant violated: struct-set idx without entry"),
            });
        }
    };
    let field_ty = match info
        .fields
        .iter()
        .find(|(n, _)| n == &field_name)
        .map(|(_, t)| *t)
    {
        Some(t) => t,
        None => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::UnknownFieldError {
                    struct_name: info.name.clone(),
                    field: field_name.clone(),
                },
                list_path: path.clone(),
                message: format!("struct '{}' has no field '{}'", info.name, field_name),
            });
        }
    };

    // Evaluate the new value expression.
    let pre = stack.len();
    path.push(4); // 1-based: struct-set=1, expr=2, field=3, new-value=4
    let val_result = simulate(
        &items[3],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match val_result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }
    if stack.len() != pre + 1 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::StackBalanceError {
                expected: pre + 1,
                actual: stack.len(),
            },
            list_path: path.clone(),
            message: format!(
                "struct-set value must push 1 value (stack went from {} to {})",
                pre,
                stack.len()
            ),
        });
    }
    let val_ty = stack.pop().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_struct_set: stack.pop value returned None despite length guard",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: struct-set value stack vanished"),
    })?;
    if val_ty != field_ty {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::TypeMismatch {
                instruction: String::from("struct-set"),
                argument_index: 1,
                expected: field_ty,
                actual: val_ty,
                position_offset: 4,
            },
            list_path: path.clone(),
            message: format!(
                "struct '{}' field '{}' expects {:?}, got {:?}",
                info.name, field_name, field_ty, val_ty
            ),
        });
    }

    // Push the updated struct (same nominal type).
    stack.push(ValueType::Struct(idx));
    Ok(SimResult::Continues(stack.clone()))
}

/// Phase 4 Step 7 — type-check
/// `(match scrutinee (case pattern body) … (case _ body))`.
///
/// Algorithm:
/// 1. Evaluate the scrutinee → push its type, then pop it as
///    `scrutinee_ty`. The scrutinee's value is consumed by the
///    match dispatch and is not visible inside any case body.
/// 2. For each case:
///    a. Reset the simulated stack to the post-scrutinee-pop state
///      (i.e. the stack the match enters with, minus the scrutinee).
///    b. Validate the pattern against `scrutinee_ty`. Pattern shapes:
///       - `_`               — any scrutinee. No bindings.
///       - integer literal   — scrutinee must be `I64`. No bindings.
///       - `(some %n)`       — scrutinee must be `Maybe`. Binds `%n: I64`.
///       - `(none)`          — scrutinee must be `Maybe`. No bindings.
///       - `(struct T %a …)` — scrutinee must be `Struct(id_T)`. Binds
///         each `%n` to the corresponding field type.
///    c. Open the binding scope (`bind_local` per pattern binding).
///    d. Simulate the body with bindings active.
///    e. Always unbind every pattern binding before propagating —
///      same scope discipline as `simulate_let`.
///    f. Collect the body's `SimResult`.
/// 3. Merge all branch results via `merge_n_branches` — the same
///    N-way merge used by `cond`. Pattern-match branches must
///    produce a consistent stack effect.
///
/// Structural shape (3-element cases, mandatory wildcard) was already
/// enforced by `validate_match` in the structural pass; this function
/// re-validates defensively so the type-checker can run standalone
/// against an unvalidated AST (Ring-0 robustness).
fn simulate_match(
    items: &[SExpr],
    stack: &mut Vec<ValueType>,
    path: &mut Vec<usize>,
    loop_context: Option<&LoopCtx>,
    fn_table: &FunctionTable,
    fn_stack: &mut Vec<FunctionContext>,
    struct_table: &StructTable,
) -> Result<SimResult, TypeCheckError> {
    // items[0] = "match", items[1] = scrutinee, items[2..] = cases.
    if items.len() < 3 {
        return Err(TypeCheckError {
            kind: TypeCheckErrorKind::ArityError {
                instruction: String::from("match"),
                expected: 2,
                actual: items.len().saturating_sub(1),
            },
            list_path: path.clone(),
            message: String::from("match requires (match scrutinee (case pattern body) …)"),
        });
    }

    // Evaluate scrutinee (items[1] at 1-based position 2).
    path.push(2);
    let scrutinee_result = simulate(
        &items[1],
        stack,
        path,
        loop_context,
        fn_table,
        fn_stack,
        struct_table,
    )?;
    path.pop();
    match scrutinee_result {
        SimResult::Continues(_) => {}
        other => return Ok(other),
    }
    let scrutinee_ty = stack.pop().ok_or_else(|| TypeCheckError {
        kind: TypeCheckErrorKind::InternalInvariantViolated(
            "simulate_match: scrutinee produced no value",
        ),
        list_path: path.clone(),
        message: String::from("internal invariant violated: match scrutinee produced no value"),
    })?;

    // Snapshot the stack the match enters with — every case body must
    // be simulated from this state to keep branch effects comparable.
    let stack_before = stack.clone();
    let cases = &items[2..];
    let last_index = cases.len() - 1;
    let mut branch_results: Vec<SimResult> = Vec::with_capacity(cases.len());

    for (i, case) in cases.iter().enumerate() {
        let case_items = match case {
            SExpr::List(l) => l,
            _ => {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::InvalidProgramStructureError,
                    list_path: path.clone(),
                    message: format!("match case {} must be a (case pattern body) list", i),
                });
            }
        };
        if case_items.len() != 3 {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: format!(
                    "match case {} must be (case pattern body); got {} items",
                    i,
                    case_items.len()
                ),
            });
        }
        let head_is_case = matches!(
            &case_items[0],
            SExpr::Atom(Atom::Symbol(s)) if s == "case"
        );
        if !head_is_case {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: format!("match case {} must start with the symbol `case`", i),
            });
        }

        let pattern = &case_items[1];
        let body = &case_items[2];

        // Reset stack to the pre-case state.
        *stack = stack_before.clone();

        // Validate the pattern shape against scrutinee_ty and collect
        // the bindings the pattern introduces.
        let bindings = validate_pattern(
            pattern,
            scrutinee_ty,
            i,
            i == last_index,
            struct_table,
            fn_stack,
            path,
        )?;

        // Open scope: bind each (idx, ty) into the current function
        // context. If `bindings` is empty, this is a no-op.
        let mut opened: Vec<u32> = Vec::with_capacity(bindings.len());
        let mut bind_error: Option<TypeCheckError> = None;
        for &(idx, ty) in bindings.iter() {
            // Phase 4 Step 7 — match bindings need a function frame
            // just like `let`. The pattern walker already rejected
            // bindings when `fn_stack` is empty, but the bind path
            // also defensively returns a typed error rather than
            // panicking.
            let ctx = match fn_stack.last_mut() {
                Some(c) => c,
                None => {
                    bind_error = Some(TypeCheckError {
                        kind: TypeCheckErrorKind::ParameterOutsideFunction {
                            parameter_index: idx,
                        },
                        list_path: path.clone(),
                        message: String::from("match pattern bindings require a function frame"),
                    });
                    break;
                }
            };
            match ctx.bind_local(idx, ty) {
                Ok(()) => opened.push(idx),
                Err(LetBindError::CollidesWithParameter) => {
                    bind_error = Some(TypeCheckError {
                        kind: TypeCheckErrorKind::MatchBindingCollidesWithParameter {
                            case_index: i,
                            parameter_index: idx,
                        },
                        list_path: path.clone(),
                        message: format!(
                            "match case {} binding %{} collides with a function parameter",
                            i, idx
                        ),
                    });
                    break;
                }
                Err(LetBindError::CollidesWithExistingLocal) => {
                    bind_error = Some(TypeCheckError {
                        kind: TypeCheckErrorKind::MatchBindingRedefinition {
                            case_index: i,
                            parameter_index: idx,
                        },
                        list_path: path.clone(),
                        message: format!(
                            "match case {} binding %{} is already bound in this scope",
                            i, idx
                        ),
                    });
                    break;
                }
            }
        }

        // Always unbind anything we successfully bound, then surface
        // the bind_error if present.
        if let Some(err) = bind_error {
            if let Some(ctx) = fn_stack.last_mut() {
                for idx in opened.iter().rev() {
                    ctx.unbind_local(*idx);
                }
            }
            return Err(err);
        }

        // Simulate body (case[2] at 1-based pos 3 inside case; case
        // itself at outer items pos i+3 → 1-based path component i+3).
        path.push(i + 3);
        path.push(3);
        let body_result = simulate(
            body,
            stack,
            path,
            loop_context,
            fn_table,
            fn_stack,
            struct_table,
        );
        path.pop();
        path.pop();

        // Close scope before propagating.
        if let Some(ctx) = fn_stack.last_mut() {
            for idx in opened.iter().rev() {
                ctx.unbind_local(*idx);
            }
        }

        let body_result = body_result?;
        branch_results.push(body_result);
    }

    merge_n_branches(branch_results, stack, path, "match")
}

/// Phase 4 Step 7 — validate a single pattern against the scrutinee
/// type, returning the list of `(idx, ty)` bindings the pattern
/// introduces (in source order, ready for `bind_local`).
///
/// `case_index` is the 0-based clause index; used in error variants.
/// `is_last_case` is true for the wildcard fallback — note that
/// non-wildcard patterns are still accepted as the last case at this
/// layer; the wildcard-mandatory rule is enforced by `validate_match`
/// in the structural pass (this layer is type-only).
fn validate_pattern(
    pattern: &SExpr,
    scrutinee_ty: ValueType,
    case_index: usize,
    _is_last_case: bool,
    struct_table: &StructTable,
    fn_stack: &mut Vec<FunctionContext>,
    path: &mut Vec<usize>,
) -> Result<Vec<(u32, ValueType)>, TypeCheckError> {
    // Wildcard — accepts any scrutinee type, binds nothing.
    if let SExpr::Atom(Atom::Symbol(s)) = pattern {
        if s == "_" {
            return Ok(Vec::new());
        }
    }
    // Integer literal — scrutinee must be I64.
    if let SExpr::Atom(Atom::Integer(_)) = pattern {
        if scrutinee_ty != ValueType::I64 {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::MatchPatternTypeMismatch {
                    case_index,
                    scrutinee_type: scrutinee_ty,
                    pattern_kind: MatchPatternKind::IntegerLiteral,
                },
                list_path: path.clone(),
                message: format!(
                    "match case {} pattern is an integer literal but scrutinee is {:?}",
                    case_index, scrutinee_ty
                ),
            });
        }
        return Ok(Vec::new());
    }
    // Pattern is a list — destructure based on head symbol.
    let p_items = match pattern {
        SExpr::List(l) => l,
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: format!(
                    "match case {} pattern is not a recognised shape",
                    case_index
                ),
            });
        }
    };
    let head = match p_items.first() {
        Some(SExpr::Atom(Atom::Symbol(s))) => s.as_str(),
        _ => {
            return Err(TypeCheckError {
                kind: TypeCheckErrorKind::InvalidProgramStructureError,
                list_path: path.clone(),
                message: format!("match case {} pattern's head must be a symbol", case_index),
            });
        }
    };
    match head {
        "some" => {
            // (some %n) — exactly 2 items.
            if p_items.len() != 2 {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::InvalidProgramStructureError,
                    list_path: path.clone(),
                    message: format!(
                        "match case {} pattern `(some %n)` requires exactly one bind slot",
                        case_index
                    ),
                });
            }
            if scrutinee_ty != ValueType::Maybe {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::MatchPatternTypeMismatch {
                        case_index,
                        scrutinee_type: scrutinee_ty,
                        pattern_kind: MatchPatternKind::Some,
                    },
                    list_path: path.clone(),
                    message: format!(
                        "match case {} `(some _)` pattern requires a Maybe scrutinee, got {:?}",
                        case_index, scrutinee_ty
                    ),
                });
            }
            let idx = match &p_items[1] {
                SExpr::Atom(Atom::Parameter(n)) => *n,
                _ => {
                    return Err(TypeCheckError {
                        kind: TypeCheckErrorKind::MatchSomeBindNotParameter { case_index },
                        list_path: path.clone(),
                        message: format!(
                            "match case {} `(some …)` bind slot must be a parameter atom (%n)",
                            case_index
                        ),
                    });
                }
            };
            if fn_stack.is_empty() {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::ParameterOutsideFunction {
                        parameter_index: idx,
                    },
                    list_path: path.clone(),
                    message: String::from("match pattern bindings require a function frame"),
                });
            }
            // Phase 4 keeps Maybe monomorphic over I64 — see ValueType::Maybe.
            Ok(alloc::vec![(idx, ValueType::I64)])
        }
        "none" => {
            if p_items.len() != 1 {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::InvalidProgramStructureError,
                    list_path: path.clone(),
                    message: format!(
                        "match case {} pattern `(none)` takes no bind slots",
                        case_index
                    ),
                });
            }
            if scrutinee_ty != ValueType::Maybe {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::MatchPatternTypeMismatch {
                        case_index,
                        scrutinee_type: scrutinee_ty,
                        pattern_kind: MatchPatternKind::None,
                    },
                    list_path: path.clone(),
                    message: format!(
                        "match case {} `(none)` pattern requires a Maybe scrutinee, got {:?}",
                        case_index, scrutinee_ty
                    ),
                });
            }
            Ok(Vec::new())
        }
        "struct" => {
            // (struct T %a %b …) — arity ≥ 2 (head + name).
            if p_items.len() < 2 {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::InvalidProgramStructureError,
                    list_path: path.clone(),
                    message: format!(
                        "match case {} pattern `(struct …)` requires a struct name",
                        case_index
                    ),
                });
            }
            let struct_name = match &p_items[1] {
                SExpr::Atom(Atom::Symbol(s)) => s.clone(),
                _ => {
                    return Err(TypeCheckError {
                        kind: TypeCheckErrorKind::InvalidProgramStructureError,
                        list_path: path.clone(),
                        message: format!(
                            "match case {} `(struct …)` pattern name must be a symbol",
                            case_index
                        ),
                    });
                }
            };
            let info_idx = match struct_table.lookup_index(&struct_name) {
                Some(i) => i,
                None => {
                    return Err(TypeCheckError {
                        kind: TypeCheckErrorKind::MatchUnknownStructError {
                            case_index,
                            name: struct_name.clone(),
                        },
                        list_path: path.clone(),
                        message: format!(
                            "match case {} references unknown struct '{}'",
                            case_index, struct_name
                        ),
                    });
                }
            };
            let info = struct_table
                .lookup(info_idx)
                .ok_or_else(|| TypeCheckError {
                    kind: TypeCheckErrorKind::InternalInvariantViolated(
                        "validate_pattern: struct name resolved to a missing index",
                    ),
                    list_path: path.clone(),
                    message: String::from(
                        "internal invariant violated: struct lookup_index/lookup disagreement",
                    ),
                })?;
            // Nominal compatibility — the scrutinee must be exactly
            // this struct nominal.
            let scrutinee_is_same_struct =
                matches!(scrutinee_ty, ValueType::Struct(id) if id == info_idx);
            if !scrutinee_is_same_struct {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::MatchPatternTypeMismatch {
                        case_index,
                        scrutinee_type: scrutinee_ty,
                        pattern_kind: MatchPatternKind::Struct,
                    },
                    list_path: path.clone(),
                    message: format!(
                        "match case {} pattern is `(struct {} …)` but scrutinee is {:?}",
                        case_index, struct_name, scrutinee_ty
                    ),
                });
            }
            // Field count must match (positional binding).
            let bind_slots = &p_items[2..];
            if bind_slots.len() != info.fields.len() {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::MatchStructFieldCountMismatch {
                        case_index,
                        struct_name: struct_name.clone(),
                        expected: info.fields.len(),
                        actual: bind_slots.len(),
                    },
                    list_path: path.clone(),
                    message: format!(
                        "match case {} pattern `(struct {} …)` expects {} bind slots, got {}",
                        case_index,
                        struct_name,
                        info.fields.len(),
                        bind_slots.len()
                    ),
                });
            }
            if !bind_slots.is_empty() && fn_stack.is_empty() {
                return Err(TypeCheckError {
                    kind: TypeCheckErrorKind::ParameterOutsideFunction { parameter_index: 0 },
                    list_path: path.clone(),
                    message: String::from("match pattern bindings require a function frame"),
                });
            }
            // Collect bindings; reject non-Parameter slots and duplicates
            // within the same pattern.
            let mut bindings: Vec<(u32, ValueType)> = Vec::with_capacity(bind_slots.len());
            for (j, slot) in bind_slots.iter().enumerate() {
                let idx = match slot {
                    SExpr::Atom(Atom::Parameter(n)) => *n,
                    _ => {
                        return Err(TypeCheckError {
                            kind: TypeCheckErrorKind::MatchStructFieldNotParameter {
                                case_index,
                                struct_name: struct_name.clone(),
                                field_index: j,
                            },
                            list_path: path.clone(),
                            message: format!(
                                "match case {} struct pattern field {} must be a parameter atom (%n)",
                                case_index, j
                            ),
                        });
                    }
                };
                if bindings.iter().any(|(existing, _)| *existing == idx) {
                    return Err(TypeCheckError {
                        kind: TypeCheckErrorKind::MatchBindingRedefinition {
                            case_index,
                            parameter_index: idx,
                        },
                        list_path: path.clone(),
                        message: format!(
                            "match case {} struct pattern binds %{} more than once",
                            case_index, idx
                        ),
                    });
                }
                bindings.push((idx, info.fields[j].1));
            }
            Ok(bindings)
        }
        _ => Err(TypeCheckError {
            kind: TypeCheckErrorKind::InvalidProgramStructureError,
            list_path: path.clone(),
            message: format!(
                "match case {} pattern head '{}' is not a recognised pattern shape",
                case_index, head
            ),
        }),
    }
}

#[cfg(test)]
mod let_tests {
    use super::*;
    use crate::parser::parse;

    fn tc(input: &str) -> Result<Vec<ValueType>, TypeCheckError> {
        let sexpr = parse(input).expect("parse should succeed");
        type_check(&sexpr)
    }

    fn tc_ok(input: &str) -> Vec<ValueType> {
        tc(input).expect("type check should succeed")
    }

    fn tc_err(input: &str) -> TypeCheckError {
        tc(input).expect_err("type check should fail")
    }

    // ── A: Happy-path ──────────────────────────────────────────

    #[test]
    fn let_binds_i64_and_returns_via_body() {
        let s = tc_ok("(program (fn f () i64 (let %0 42 %0)) (call f))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn let_nested_two_locals() {
        let s = tc_ok("(program (fn f () i64 (let %0 1 (let %1 2 (add %0 %1)))) (call f))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn let_with_param_and_local() {
        let s = tc_ok("(program (fn f (i64) i64 (let %1 %0 (add %0 %1))) (call f 5))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn let_binds_handle_type() {
        let s = tc_ok("(program (fn f () handle (let %0 @1 %0)) (call f))");
        assert_eq!(s, vec![ValueType::Handle]);
    }

    // ── B: Errors ──────────────────────────────────────────────

    #[test]
    fn let_outside_fn_errors() {
        let err = tc_err("(let %0 42 %0)");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::ParameterOutsideFunction { .. }
        ));
    }

    #[test]
    fn let_param_collision_errors() {
        // fn has arity 1, %0 is a parameter, trying to let-bind %0
        let err = tc_err("(program (fn f (i64) i64 (let %0 42 %0)) (call f 1))");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::LetCollidesWithParameter { parameter_index: 0 }
        ));
    }

    #[test]
    fn let_redefinition_errors() {
        // %1 bound twice at same scope (not IR-shadowing)
        let err = tc_err("(program (fn f () i64 (let %0 1 (let %0 2 %0))) (call f))");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::LetRedefinitionError { parameter_index: 0 }
        ));
    }

    #[test]
    fn let_wrong_arity_too_few() {
        let err = tc_err("(program (fn f () i64 (let %0 42)) (call f))");
        assert!(matches!(err.kind, TypeCheckErrorKind::ArityError { .. }));
    }

    #[test]
    fn let_non_param_target_errors() {
        let err = tc_err("(program (fn f () i64 (let foo 42 42)) (call f))");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::InvalidProgramStructureError
        ));
    }

    // ── C: Scope ───────────────────────────────────────────────

    #[test]
    fn let_local_unbound_after_body() {
        // After the inner let's body ends, %1 should be unbound.
        // The outer expression references %1 a second time (inside the
        // same fn but outside the let body) — this must trigger
        // InvalidParameterIndex because %1 is no longer bound.
        //
        // If unbind_local were broken (e.g. early return skipping it),
        // the outer %1 reference would silently resolve to the still-bound
        // local, and this test would falsely pass. This test guards
        // against that regression.
        let err = tc_err("(program (fn f () i64 (add (let %1 42 %1) %1)) (call f))");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::InvalidParameterIndex {
                parameter_index: 1,
                ..
            }
        ));
    }

    #[test]
    fn let_sibling_reuse_after_unbind() {
        // First let binds %0, body returns it. Second let binds %0 again after unbind.
        // This uses nested lets to verify that after the first let's body ends,
        // %0 is unbound and can be re-bound by a sibling (via outer nesting).
        // We test this indirectly: the body of the first let returns, and the
        // outer expression continues.
        let s = tc_ok("(program (fn f () i64 (let %0 10 (add %0 1))) (call f))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn let_inner_visible_from_nested_body() {
        // %0 bound in outer let, visible in inner let body
        let s = tc_ok("(program (fn f () i64 (let %0 10 (let %1 20 (add %0 %1)))) (call f))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    // ── D: Integration ─────────────────────────────────────────

    #[test]
    fn let_inside_if_branches() {
        // Paket B.1b: cond must be Bool; let-binding inside cond also Bool.
        let s = tc_ok(
            "(program (fn f () i64 (if (let %0 true %0) (let %1 10 %1) (let %2 20 %2))) (call f))",
        );
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn let_with_call_in_body() {
        let s = tc_ok(
            "(program (fn g (i64) i64 (add %0 1)) (fn f () i64 (let %0 5 (call g %0))) (call f))",
        );
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn let_with_return_in_expr() {
        // If the expr returns, body never executes — the return propagates
        let s = tc_ok("(program (fn f () i64 (let %0 (return 42) %0)) (call f))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    // ── E: JSON ────────────────────────────────────────────────

    #[test]
    fn json_let_collides_with_parameter() {
        let err = tc_err("(program (fn f (i64) i64 (let %0 42 %0)) (call f 1))");
        let json = err.to_json();
        assert!(json.contains("\"LetCollidesWithParameter\""));
        assert!(json.contains("\"parameter_index\":0"));
    }

    #[test]
    fn json_let_redefinition_error() {
        let err = tc_err("(program (fn f () i64 (let %0 1 (let %0 2 %0))) (call f))");
        let json = err.to_json();
        assert!(json.contains("\"LetRedefinitionError\""));
        assert!(json.contains("\"parameter_index\":0"));
    }
}

#[cfg(test)]
mod mp4_tests {
    use super::*;
    use crate::parser::parse;

    fn tc(input: &str) -> Result<Vec<ValueType>, TypeCheckError> {
        let sexpr = parse(input).expect("parse should succeed");
        type_check(&sexpr)
    }

    fn tc_ok(input: &str) -> Vec<ValueType> {
        tc(input).expect("type check should succeed")
    }

    fn tc_err(input: &str) -> TypeCheckError {
        tc(input).expect_err("type check should fail")
    }

    // Helper: simulate_raw for sub-expression tests
    fn tc_raw(input: &str) -> Result<Vec<ValueType>, TypeCheckError> {
        let sexpr = parse(input).expect("parse should succeed");
        simulate_raw(&sexpr)
    }

    fn tc_raw_ok(input: &str) -> Vec<ValueType> {
        tc_raw(input).expect("simulate_raw should succeed")
    }

    fn tc_raw_err(input: &str) -> TypeCheckError {
        tc_raw(input).expect_err("simulate_raw should fail")
    }

    // ── A: Comparisons ─────────────────────────────────────────

    #[test]
    fn a01_eq_returns_bool() {
        // Paket B.1: comparison ops now return Bool.
        assert_eq!(tc_ok("(eq 1 2)"), vec![ValueType::Bool]);
    }

    #[test]
    fn a02_ne_returns_bool() {
        // Paket B.1: `neq` renamed to `ne`; returns Bool.
        assert_eq!(tc_ok("(ne 1 2)"), vec![ValueType::Bool]);
    }

    #[test]
    fn a03_lt_returns_bool() {
        assert_eq!(tc_ok("(lt 1 2)"), vec![ValueType::Bool]);
    }

    #[test]
    fn a04_gt_returns_bool() {
        assert_eq!(tc_ok("(gt 1 2)"), vec![ValueType::Bool]);
    }

    #[test]
    fn a05_le_returns_bool() {
        // Paket B.1: `lteq` renamed to `le`; returns Bool.
        assert_eq!(tc_ok("(le 1 2)"), vec![ValueType::Bool]);
    }

    #[test]
    fn a06_ge_returns_bool() {
        // Paket B.1: `gteq` renamed to `ge`; returns Bool.
        assert_eq!(tc_ok("(ge 1 2)"), vec![ValueType::Bool]);
    }

    #[test]
    fn a07_lt_with_handle_arg_errors() {
        let err = tc_err("(lt 1 @5)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn a08_cmp_no_longer_exists() {
        // cmp was removed; unknown instructions cause stack-neutral pass-through in
        // simulate_instruction (no lookup match). Wrapped in a program to verify
        // it doesn't type-check as a known instruction.
        // An unknown symbol in a list just returns Continues with unmodified stack.
        // At program root, that means stack depth 0 ≠ 1 → StackBalanceError.
        let err = tc_err("(cmp 1 2)");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::StackBalanceError { .. }
        ));
    }

    // ── B: seq Happy-Path ──────────────────────────────────────

    #[test]
    fn b01_seq_two_args() {
        let s = tc_raw_ok("(seq (discard (loop (break false))) 42)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn b02_seq_three_args_chain() {
        let s = tc_raw_ok("(seq (discard (loop (break false))) (discard (loop (break false))) 99)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn b03_seq_inside_function() {
        let s = tc_ok("(program (fn f () i64 (seq (discard (loop (break false))) 42)) (call f))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn b04_seq_last_arg_complex() {
        let s = tc_raw_ok("(seq (discard (loop (break false))) (add 1 2))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    // ── C: seq Error Cases ─────────────────────────────────────

    #[test]
    fn c01_seq_zero_args_errors() {
        let err = tc_raw_err("(seq)");
        assert!(matches!(err.kind, TypeCheckErrorKind::ArityError { .. }));
    }

    #[test]
    fn c02_seq_one_arg_errors() {
        let err = tc_raw_err("(seq 42)");
        assert!(matches!(err.kind, TypeCheckErrorKind::ArityError { .. }));
    }

    #[test]
    fn c03_seq_effect_pushes_value_errors() {
        let err = tc_raw_err("(seq 42 99)");
        match err.kind {
            TypeCheckErrorKind::SeqEffectNotStackNeutral {
                argument_index,
                expected_depth,
                actual_depth,
                ..
            } => {
                assert_eq!(argument_index, 0);
                assert_eq!(expected_depth, 0);
                assert_eq!(actual_depth, 1);
            }
            other => panic!("expected SeqEffectNotStackNeutral, got {:?}", other),
        }
    }

    #[test]
    fn c04_seq_second_effect_pushes_errors() {
        let err = tc_raw_err("(seq (discard (loop (break false))) 42 99)");
        match err.kind {
            TypeCheckErrorKind::SeqEffectNotStackNeutral { argument_index, .. } => {
                assert_eq!(argument_index, 1);
            }
            other => panic!("expected SeqEffectNotStackNeutral, got {:?}", other),
        }
    }

    #[test]
    fn c05_seq_with_break_propagates() {
        // Paket B.1c: outer loop produces the break-value type.
        let s = tc_raw_ok("(loop (seq (discard (loop (break false))) (break false)))");
        assert_eq!(s, vec![ValueType::Bool]);
    }

    #[test]
    fn c06_seq_with_return_propagates() {
        // Inside a function: (seq effect (return x)) — return propagates
        let s = tc_ok(
            "(program (fn f () i64 (seq (discard (loop (break false))) (return 42))) (call f))",
        );
        assert_eq!(s, vec![ValueType::I64]);
    }

    // ── D: intent Tests ────────────────────────────────────────

    #[test]
    fn d01_intent_no_args() {
        let s = tc_ok("(intent)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn d02_intent_with_args() {
        let s = tc_ok("(intent @1 42 #x00)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn d03_intent_mixed_types() {
        let s = tc_ok("(intent @1 #x4142 42 @99)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    // ── E: JSON Tests ──────────────────────────────────────────

    #[test]
    fn e01_json_seq_effect_not_stack_neutral() {
        let err = tc_raw_err("(seq 42 99)");
        let json = err.to_json();
        assert!(json.contains("\"SeqEffectNotStackNeutral\""));
        assert!(json.contains("\"position_offset\":2"));
        assert!(json.contains("\"argument_index\":0"));
        assert!(json.contains("\"expected_depth\":0"));
        assert!(json.contains("\"actual_depth\":1"));
    }

    #[test]
    fn e02_json_seq_kind_string() {
        let err = tc_raw_err("(seq 42 99)");
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"SeqEffectNotStackNeutral\""));
    }

    // ── F: Integration ─────────────────────────────────────────

    #[test]
    fn f01_comparison_in_if_condition() {
        let s = tc_ok("(if (eq 1 2) (add 10 20) (sub 5 3))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn f02_seq_with_intent_as_value() {
        // intent as the value-producing last arg of seq
        let s = tc_raw_ok("(seq (discard (loop (break false))) (intent @1 42))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn f03_seq_in_function_body_with_let() {
        let s = tc_ok(
            "(program (fn f () i64 (let %0 (seq (discard (loop (break false))) 42) %0)) (call f))",
        );
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn f04_intent_inside_program() {
        let s = tc_ok("(program (fn f () i64 (intent @1 42)) (call f))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    // ── G: Stack-balance enforcement on `(intent …)` args ──────
    //
    // Regression guard for the bug where `simulate_intent` did an
    // unconditional `stack.pop()` after each arg. Bare symbols (per
    // simulate(Atom::Symbol(_) => {})) push nothing; the old code
    // would have stolen the top of the *surrounding* stack and
    // happily reported the program as well-typed.

    #[test]
    fn g01_intent_with_symbol_arg_rejects() {
        // `foo` is a bare symbol; it pushes no value. The new
        // depth-snapshot check must catch this as a stack-balance
        // error rather than silently popping from the surrounding
        // stack.
        let err = tc_raw_err("(intent foo)");
        match err.kind {
            TypeCheckErrorKind::StackBalanceError {
                expected: 1,
                actual: 0,
            } => {}
            other => panic!("expected StackBalanceError(1, 0), got {:?}", other),
        }
    }

    #[test]
    fn g02_intent_inside_seq_does_not_steal_from_outer_stack() {
        // `(seq <effect> <value>)` requires the *value* to leave one
        // value on the stack and the *effect* to be stack-neutral.
        // If `simulate_intent` were still unconditionally popping,
        // it would corrupt the surrounding seq's accounting and the
        // surrounding program would still type-check by accident.
        // With the fix in place, `(intent foo)` fails on its own
        // arg-balance check before seq sees the result.
        let err = tc_raw_err("(seq (discard (intent foo)) 42)");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::StackBalanceError { .. }
        ));
    }
}

#[cfg(test)]
mod discard_tests {
    use super::*;
    use crate::parser::parse;

    fn tc(input: &str) -> Result<Vec<ValueType>, TypeCheckError> {
        let sexpr = parse(input).expect("parse should succeed");
        type_check(&sexpr)
    }

    fn tc_ok(input: &str) -> Vec<ValueType> {
        tc(input).expect("type check should succeed")
    }

    /// Symmetry pair with `tc_ok` — kept for future error-path tests.
    #[allow(dead_code)]
    fn tc_err(input: &str) -> TypeCheckError {
        tc(input).expect_err("type check should fail")
    }

    fn tc_raw(input: &str) -> Result<Vec<ValueType>, TypeCheckError> {
        let sexpr = parse(input).expect("parse should succeed");
        simulate_raw(&sexpr)
    }

    fn tc_raw_ok(input: &str) -> Vec<ValueType> {
        tc_raw(input).expect("simulate_raw should succeed")
    }

    fn tc_raw_err(input: &str) -> TypeCheckError {
        tc_raw(input).expect_err("simulate_raw should fail")
    }

    // ── A: Happy-Path ──────────────────────────────────────────

    #[test]
    fn discard_integer_literal_is_stack_neutral() {
        let s = tc_raw_ok("(discard 42)");
        assert_eq!(s, vec![]);
    }

    #[test]
    fn discard_with_intent_in_seq() {
        let s = tc_raw_ok("(seq (discard (intent @1 42)) 99)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn discard_in_function_body() {
        let s = tc_ok("(program (fn f () i64 (seq (discard (intent @1 42)) 0)) (call f))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    // ── B: Error Cases ─────────────────────────────────────────

    #[test]
    fn discard_no_args_errors() {
        let err = tc_raw_err("(discard)");
        match err.kind {
            TypeCheckErrorKind::ArityError {
                ref instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "discard");
                assert_eq!(expected, 1);
                assert_eq!(actual, 0);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn discard_too_many_args_errors() {
        let err = tc_raw_err("(discard 1 2)");
        match err.kind {
            TypeCheckErrorKind::ArityError {
                ref instruction,
                expected,
                actual,
            } => {
                assert_eq!(instruction, "discard");
                assert_eq!(expected, 1);
                assert_eq!(actual, 2);
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn discard_stack_neutral_expr_errors() {
        // Paket B.1c: loop now pushes 1 value; the test anchor
        // moves to (store 0 0) which pushes 0 values — discard
        // still requires exactly 1.
        let err = tc_raw_err("(discard (store 0 0))");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::StackUnderflow { .. }
        ));
    }

    // ── C: Integration ─────────────────────────────────────────

    #[test]
    fn discard_chains_in_seq() {
        let s = tc_raw_ok("(seq (discard (intent @1 1)) (discard (intent @2 2)) 99)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn discard_with_function_call() {
        let s = tc_ok(
            "(program (fn helper () i64 42) (fn f () i64 (seq (discard (call helper)) 0)) (call f))"
        );
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn discard_with_break_propagates() {
        // Paket B.1c: break inside discard propagates as Breaks;
        // loop catches and produces the break value's type.
        let s = tc_raw_ok("(loop (discard (break false)))");
        assert_eq!(s, vec![ValueType::Bool]);
    }
}

#[cfg(test)]
mod nop_tests {
    use super::*;
    use crate::parser::parse;

    fn tc_raw_ok(input: &str) -> Vec<ValueType> {
        let sexpr = parse(input).expect("parse should succeed");
        simulate_raw(&sexpr).expect("simulate_raw should succeed")
    }

    #[test]
    fn nop_alone_is_stack_neutral() {
        let s = tc_raw_ok("(nop)");
        assert_eq!(s, vec![]);
    }

    #[test]
    fn nop_multiple_in_seq() {
        let s = tc_raw_ok("(seq (nop) (nop) 42)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn nop_as_if_else_branch() {
        let s = tc_raw_ok("(if true (nop) (nop))");
        assert_eq!(s, vec![]);
    }
}

// ── 12i: policy / query type-checking tests ────────────────────

#[cfg(test)]
mod policy_query_tests {
    use super::*;
    use crate::parser::parse;

    fn tc(input: &str) -> Result<Vec<ValueType>, TypeCheckError> {
        let sexpr = parse(input).expect("parse should succeed");
        type_check(&sexpr)
    }

    fn tc_raw_ok(input: &str) -> Vec<ValueType> {
        let sexpr = parse(input).expect("parse should succeed");
        simulate_raw(&sexpr).expect("simulate_raw should succeed")
    }

    #[test]
    fn policy_pushes_i64_status() {
        let s = tc_raw_ok("(policy gpu allocate-slice 5 50)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn query_pushes_i64_telemetry() {
        let s = tc_raw_ok("(query gpu utilization)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn policy_status_can_feed_arithmetic() {
        // (policy ...) returns i64; add to another i64 → i64.
        let s = tc_raw_ok("(add (policy gpu release-slice 1) 1)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn query_value_can_feed_arithmetic() {
        let s = tc_raw_ok("(add (query gpu utilization) 10)");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn policy_with_nested_arithmetic_args_typechecks() {
        let s = tc_raw_ok("(policy gpu allocate-slice (add 1 2) (mul 5 10))");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn policy_missing_subsystem_or_op_is_arity_error() {
        let err = tc("(policy)").unwrap_err();
        assert!(matches!(err.kind, TypeCheckErrorKind::ArityError { .. }));
    }

    #[test]
    fn policy_non_symbol_subsystem_is_structural_error() {
        let err = tc("(policy 1 set-priority 0)").unwrap_err();
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::InvalidProgramStructureError
        ));
    }

    #[test]
    fn policy_non_symbol_operation_is_structural_error() {
        let err = tc("(policy gpu 42 0)").unwrap_err();
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::InvalidProgramStructureError
        ));
    }

    #[test]
    fn policy_non_i64_arg_is_type_mismatch() {
        // Bytes arg in a position the policy ABI expects i64.
        let err = tc("(policy gpu release-slice #x41)").unwrap_err();
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn policy_handle_arg_is_type_mismatch() {
        let err = tc("(policy gpu release-slice @5)").unwrap_err();
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn query_inside_program_main_typechecks() {
        let s = tc("(program (query gpu utilization))").expect("type-check");
        assert_eq!(s, vec![ValueType::I64]);
    }

    #[test]
    fn policy_inside_fn_body_typechecks() {
        let s = tc("(program (fn p (i64) i64 (policy gpu allocate-slice %0 50)) (call p 3))")
            .expect("type-check");
        assert_eq!(s, vec![ValueType::I64]);
    }

    // === Pre-A.1 (F-C6 / F-C7): InternalInvariantViolated surface ===

    #[test]
    fn internal_invariant_violated_serialises_to_json() {
        // The variant is structurally unreachable on the type-check
        // happy path, but it is part of the public error surface.
        // Verify that it round-trips through `to_json()` so a Pillar 2
        // forensic-readout cannot regress on an `unreachable!` panic
        // disguised as a missing `details_json` arm.
        let err = TypeCheckError {
            kind: TypeCheckErrorKind::InternalInvariantViolated("test_call_site"),
            list_path: alloc::vec![1, 2, 3],
            message: alloc::string::String::from("synthetic invariant"),
        };
        let json = err.to_json();
        assert!(json.contains("\"kind\":\"InternalInvariantViolated\""));
        assert!(json.contains("\"site\":\"test_call_site\""));
        assert!(json.contains("\"list_path\":[1,2,3]"));
    }

    #[test]
    fn type_checker_never_panics_on_arbitrary_bytes_input() {
        // Smoke test: small adversarial inputs should never panic the
        // type-checker — they should produce a `TypeCheckError` of
        // some variant. We don't pin the variant because the goal is
        // *no-panic*, not a specific error mapping. Pre-A.3 (Paket A)
        // will replace this with a proper property-test harness.
        let probes = [
            "",
            "(",
            ")",
            "()",
            "(program)",
            "(program (fn))",
            "(program (fn))",
            "(fn)",
            "(if)",
            "(if 0)",
            "(let)",
            "(let %0)",
            "(call)",
            "(call undefined)",
            "(loop)",
            "(loop (break false))",
            "(seq)",
            "(seq 1)",
            "(dup)",
            "(swap)",
            "(drop)",
            "(div)",
            "(div 1)",
        ];
        for src in probes {
            let parsed = crate::parser::parse(src);
            if let Ok(ast) = parsed {
                // Whichever variant comes out, the call must return,
                // not panic.
                let _ = crate::type_checker::type_check(&ast);
            }
        }
    }
}

// ── Stage 12 Paket B.2/B.3/B.4/B.5/B.6 type-checker tests ──────

#[cfg(test)]
mod paket_b_rest_tests {
    use super::*;
    use crate::parser::parse;

    fn tc(input: &str) -> Result<Vec<ValueType>, TypeCheckError> {
        let sexpr = parse(input).expect("parse should succeed");
        type_check(&sexpr)
    }

    fn tc_ok(input: &str) -> Vec<ValueType> {
        tc(input).expect("type check should succeed")
    }

    fn tc_err(input: &str) -> TypeCheckError {
        tc(input).expect_err("type check should fail")
    }

    /// Bypasses the root depth-1 check by calling `simulate_raw`.
    /// Used by tests that evaluate forms (`loop`, `(break v)`) that
    /// cannot stand at the program root.
    fn tcr_ok(input: &str) -> Vec<ValueType> {
        let sexpr = parse(input).expect("parse should succeed");
        simulate_raw(&sexpr).expect("simulate_raw should succeed")
    }

    // ── B.2 Lists ──────────────────────────────────────────────

    #[test]
    fn list_pushes_list_type() {
        assert_eq!(tc_ok("(list 1 2 3)"), vec![ValueType::List]);
    }

    #[test]
    fn list_empty_pushes_list_type() {
        assert_eq!(tc_ok("(list)"), vec![ValueType::List]);
    }

    #[test]
    fn list_rejects_non_i64_element() {
        let err = tc_err("(list 1 true 3)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn list_get_returns_i64() {
        assert_eq!(tc_ok("(list-get (list 1 2 3) 0)"), vec![ValueType::I64]);
    }

    #[test]
    fn list_len_returns_i64() {
        assert_eq!(tc_ok("(list-len (list))"), vec![ValueType::I64]);
    }

    #[test]
    fn list_append_returns_list() {
        assert_eq!(tc_ok("(list-append (list 1 2) 3)"), vec![ValueType::List]);
    }

    // ── B.3 Maps ───────────────────────────────────────────────

    #[test]
    fn map_new_returns_map() {
        assert_eq!(tc_ok("(map-new)"), vec![ValueType::Map]);
    }

    #[test]
    fn map_put_returns_map() {
        assert_eq!(tc_ok("(map-put (map-new) 1 100)"), vec![ValueType::Map]);
    }

    #[test]
    fn map_get_returns_i64() {
        assert_eq!(tc_ok("(map-get (map-new) 1)"), vec![ValueType::I64]);
    }

    #[test]
    fn map_contains_returns_bool() {
        assert_eq!(tc_ok("(map-contains (map-new) 1)"), vec![ValueType::Bool]);
    }

    // ── B.4 Bounded Loops ──────────────────────────────────────

    #[test]
    fn loop_with_bound_returns_bool() {
        assert_eq!(
            tc_ok("(loop-with-bound 10 (break true))"),
            vec![ValueType::Bool]
        );
    }

    #[test]
    fn loop_with_bound_rejects_non_i64_bound() {
        let err = tc_err("(loop-with-bound true (break true))");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn loop_with_bound_rejects_non_bool_break() {
        // The implicit (break false) seeds break_type=Bool; a user
        // break with i64 must be rejected.
        let err = tc_err("(loop-with-bound 10 (break 42))");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::BreakTypeMismatch { .. }
        ));
    }

    #[test]
    fn loop_with_bound_wrong_arity() {
        let err = tc_err("(loop-with-bound 10)");
        assert!(matches!(err.kind, TypeCheckErrorKind::ArityError { .. }));
    }

    #[test]
    fn for_returns_bool() {
        let src = "(program (fn run () bool (for %0 (list 1 2) (gt %0 0))) (call run))";
        assert_eq!(tc_ok(src), vec![ValueType::Bool]);
    }

    #[test]
    fn for_rejects_non_parameter_binding() {
        let src = "(program (fn run () bool (for foo (list 1 2) (gt foo 0))) (call run))";
        let err = tc_err(src);
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::InvalidProgramStructureError
        ));
    }

    #[test]
    fn for_rejects_non_list_source() {
        let src = "(program (fn run () bool (for %0 42 (gt %0 0))) (call run))";
        let err = tc_err(src);
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn for_outside_function_errors() {
        // for must live inside a fn body so its %n binding has a frame.
        let err = tc_err("(for %0 (list 1 2 3) (gt %0 0))");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::ParameterOutsideFunction { .. }
        ));
    }

    // ── B.5 Strings ────────────────────────────────────────────

    #[test]
    fn string_from_int_returns_string() {
        assert_eq!(tc_ok("(string-from-int 42)"), vec![ValueType::String]);
    }

    #[test]
    fn string_concat_returns_string() {
        assert_eq!(
            tc_ok("(string-concat (string-from-int 1) (string-from-int 2))"),
            vec![ValueType::String]
        );
    }

    #[test]
    fn string_eq_returns_bool() {
        assert_eq!(
            tc_ok("(string-eq (string-from-int 1) (string-from-int 1))"),
            vec![ValueType::Bool]
        );
    }

    #[test]
    fn string_concat_rejects_non_string_args() {
        let err = tc_err("(string-concat 1 (string-from-int 2))");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    // ── B.6 I/O ────────────────────────────────────────────────

    #[test]
    fn read_handle_returns_bytes() {
        assert_eq!(tc_ok("(read-handle @5)"), vec![ValueType::Bytes]);
    }

    #[test]
    fn read_handle_rejects_non_handle() {
        let err = tc_err("(read-handle 42)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn write_host_state_returns_i64() {
        assert_eq!(tc_ok("(write-host-state foo 100)"), vec![ValueType::I64]);
    }

    #[test]
    fn write_host_state_rejects_non_symbol_key() {
        let err = tc_err("(write-host-state 42 100)");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::InvalidProgramStructureError
        ));
    }

    #[test]
    fn write_host_state_rejects_non_i64_value() {
        let err = tc_err("(write-host-state foo true)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn write_host_state_wrong_arity() {
        let err = tc_err("(write-host-state foo)");
        assert!(matches!(err.kind, TypeCheckErrorKind::ArityError { .. }));
    }

    // ── Composite type symbols in fn signatures ────────────────

    #[test]
    fn fn_return_type_list_accepted() {
        let src = "(program (fn make () list (list 1 2 3)) (call make))";
        assert_eq!(tc_ok(src), vec![ValueType::List]);
    }

    #[test]
    fn fn_return_type_map_accepted() {
        let src = "(program (fn make () map (map-new)) (call make))";
        assert_eq!(tc_ok(src), vec![ValueType::Map]);
    }

    #[test]
    fn fn_return_type_string_accepted() {
        let src = "(program (fn make () string (string-from-int 1)) (call make))";
        assert_eq!(tc_ok(src), vec![ValueType::String]);
    }

    // ── Phase 4 Step 2 — Bytes operations ──────────────────────

    #[test]
    fn bytes_len_returns_i64() {
        assert_eq!(tc_ok("(bytes-len #x48656c6c6f)"), vec![ValueType::I64]);
    }

    #[test]
    fn bytes_len_rejects_non_bytes() {
        let err = tc_err("(bytes-len 42)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn bytes_get_returns_i64() {
        assert_eq!(tc_ok("(bytes-get #x48656c6c6f 0)"), vec![ValueType::I64]);
    }

    #[test]
    fn bytes_get_rejects_non_bytes() {
        let err = tc_err("(bytes-get 42 0)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn bytes_get_rejects_non_i64_index() {
        let err = tc_err("(bytes-get #x48 true)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn bytes_slice_returns_bytes() {
        assert_eq!(
            tc_ok("(bytes-slice #x48656c6c6f 1 3)"),
            vec![ValueType::Bytes]
        );
    }

    #[test]
    fn bytes_slice_rejects_non_i64_start() {
        let err = tc_err("(bytes-slice #x48 true 1)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn bytes_concat_returns_bytes() {
        assert_eq!(
            tc_ok("(bytes-concat #x4865 #x6c6c6f)"),
            vec![ValueType::Bytes]
        );
    }

    #[test]
    fn bytes_concat_rejects_non_bytes_args() {
        let err = tc_err("(bytes-concat 1 #x00)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn bytes_eq_returns_bool() {
        assert_eq!(tc_ok("(bytes-eq #x4865 #x6c6c6f)"), vec![ValueType::Bool]);
    }

    #[test]
    fn bytes_eq_rejects_non_bytes_args() {
        let err = tc_err("(bytes-eq #x00 42)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn bytes_from_int_returns_bytes() {
        assert_eq!(tc_ok("(bytes-from-int 42)"), vec![ValueType::Bytes]);
    }

    #[test]
    fn bytes_from_int_rejects_non_i64() {
        let err = tc_err("(bytes-from-int true)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    // ── Phase 4 Step 3 — list-slice ────────────────────────────

    #[test]
    fn list_slice_returns_list() {
        assert_eq!(
            tc_ok("(list-slice (list 1 2 3) 0 2)"),
            vec![ValueType::List]
        );
    }

    #[test]
    fn list_slice_rejects_non_list() {
        let err = tc_err("(list-slice 42 0 1)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn list_slice_rejects_non_i64_start() {
        let err = tc_err("(list-slice (list 1 2) true 1)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn list_slice_rejects_non_i64_len() {
        let err = tc_err("(list-slice (list 1 2) 0 true)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    // ── Phase 4 Step 4 — `cond` N-way Bool dispatch ─────────────

    #[test]
    fn cond_default_only_takes_default_body_type() {
        assert_eq!(tc_ok("(cond (default 42))"), vec![ValueType::I64]);
    }

    #[test]
    fn cond_two_clauses_matching_i64() {
        assert_eq!(
            tc_ok("(cond ((eq 0 1) 10) (default 20))"),
            vec![ValueType::I64]
        );
    }

    #[test]
    fn cond_three_clauses_matching_i64() {
        assert_eq!(
            tc_ok("(cond ((eq 0 1) 10) ((eq 0 2) 20) (default 30))"),
            vec![ValueType::I64]
        );
    }

    #[test]
    fn cond_matching_bytes_branches() {
        assert_eq!(
            tc_ok("(cond ((eq 0 1) #x00) (default #x01))"),
            vec![ValueType::Bytes]
        );
    }

    #[test]
    fn cond_matching_bool_branches() {
        assert_eq!(
            tc_ok("(cond ((eq 0 1) true) (default false))"),
            vec![ValueType::Bool]
        );
    }

    #[test]
    fn cond_branch_type_mismatch_default_differs() {
        // First clause body produces I64, default produces Bytes.
        let err = tc_err("(cond ((eq 0 1) 1) (default #x00))");
        match &err.kind {
            TypeCheckErrorKind::BranchStackMismatch {
                then_stack,
                else_stack,
            } => {
                assert_eq!(*then_stack, vec![ValueType::I64]);
                assert_eq!(*else_stack, vec![ValueType::Bytes]);
            }
            other => panic!("expected BranchStackMismatch, got {:?}", other),
        }
    }

    #[test]
    fn cond_branch_type_mismatch_between_clauses() {
        // Two non-default clauses with different body types.
        let err = tc_err("(cond ((eq 0 1) 1) ((eq 0 2) #x00) (default 0))");
        assert!(matches!(
            err.kind,
            TypeCheckErrorKind::BranchStackMismatch { .. }
        ));
    }

    #[test]
    fn cond_predicate_must_be_bool_i64_rejected() {
        let err = tc_err("(cond (42 1) (default 0))");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "cond");
                assert_eq!(*expected, ValueType::Bool);
                assert_eq!(*actual, ValueType::I64);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn cond_predicate_must_be_bool_handle_rejected() {
        let err = tc_err("(cond (@5 1) (default 0))");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "cond");
                assert_eq!(*expected, ValueType::Bool);
                assert_eq!(*actual, ValueType::Handle);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn cond_predicate_must_be_bool_bytes_rejected() {
        let err = tc_err("(cond (#x00 1) (default 0))");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "cond");
                assert_eq!(*expected, ValueType::Bool);
                assert_eq!(*actual, ValueType::Bytes);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn cond_predicate_second_clause_must_be_bool() {
        // First predicate Bool, second predicate I64.
        let err = tc_err("(cond ((eq 0 1) 1) (42 2) (default 0))");
        match &err.kind {
            TypeCheckErrorKind::TypeMismatch {
                instruction,
                expected,
                actual,
                ..
            } => {
                assert_eq!(instruction, "cond");
                assert_eq!(*expected, ValueType::Bool);
                assert_eq!(*actual, ValueType::I64);
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn cond_all_branches_return_same_type() {
        // All clauses return; cond resolves to Returns(I64).
        let stack =
            tc_ok("(cond ((eq 0 1) (return 1)) ((eq 0 2) (return 2)) (default (return 3)))");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn cond_mixed_return_and_continue_uses_continue_stack() {
        // First clause returns, default continues — the cond surface
        // takes the continuing branch's stack.
        let stack = tc_ok("(cond ((eq 0 1) (return 1)) (default 99))");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn cond_all_break_inside_loop_uses_break_value() {
        // Every clause breaks with Bool — outer loop's break type
        // resolves to Bool.
        let stack = tcr_ok(
            "(loop (cond ((eq 0 1) (break true)) ((eq 0 2) (break false)) (default (break true))))",
        );
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn cond_break_and_continue_mixed_uses_continue() {
        // Inside a loop: one clause breaks (terminates), others
        // continue normally. The cond merge takes the continuing
        // path's stack; the loop is statically infinite from the
        // continuing perspective but the break establishes the
        // loop's exit type.
        let stack = tcr_ok("(loop (cond ((eq 0 1) (break false)) (default (store 0 0))))");
        assert_eq!(stack, vec![ValueType::Bool]);
    }

    #[test]
    fn cond_no_clauses_is_arity_error() {
        let err = tc_err("(cond)");
        match &err.kind {
            TypeCheckErrorKind::ArityError { instruction, .. } => {
                assert_eq!(instruction, "cond");
            }
            other => panic!("expected ArityError, got {:?}", other),
        }
    }

    #[test]
    fn cond_inside_function_body() {
        let src = "(program \
            (fn pick (i64) i64 \
                (cond ((eq %0 1) 10) ((eq %0 2) 20) (default 0))) \
            (call pick 1))";
        let stack = tc_ok(src);
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn cond_nested_inside_cond() {
        let src = "(cond \
            ((eq 0 1) (cond ((eq 0 1) 100) (default 200))) \
            (default (cond ((eq 0 2) 300) (default 400))))";
        let stack = tc_ok(src);
        assert_eq!(stack, vec![ValueType::I64]);
    }

    #[test]
    fn cond_nested_inside_if() {
        let stack = tc_ok("(if true (cond ((eq 0 1) 1) (default 0)) 99)");
        assert_eq!(stack, vec![ValueType::I64]);
    }

    // ── Phase 4 Step 5 — Maybe / Option type ───────────────────

    #[test]
    fn some_returns_maybe() {
        assert_eq!(tc_ok("(some 42)"), vec![ValueType::Maybe]);
    }

    #[test]
    fn none_returns_maybe() {
        assert_eq!(tc_ok("(none)"), vec![ValueType::Maybe]);
    }

    #[test]
    fn is_some_returns_bool() {
        assert_eq!(tc_ok("(is-some (some 1))"), vec![ValueType::Bool]);
    }

    #[test]
    fn is_none_returns_bool() {
        assert_eq!(tc_ok("(is-none (none))"), vec![ValueType::Bool]);
    }

    #[test]
    fn unwrap_returns_i64() {
        assert_eq!(tc_ok("(unwrap (some 1))"), vec![ValueType::I64]);
    }

    #[test]
    fn unwrap_or_returns_i64() {
        assert_eq!(tc_ok("(unwrap-or (some 1) 0)"), vec![ValueType::I64]);
    }

    #[test]
    fn some_rejects_non_i64() {
        // Phase 4 is monomorphic over I64; `(some #x00)` fails.
        let err = tc_err("(some #x00)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn is_some_rejects_non_maybe() {
        let err = tc_err("(is-some 42)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn is_none_rejects_non_maybe() {
        let err = tc_err("(is-none 42)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn unwrap_rejects_non_maybe() {
        let err = tc_err("(unwrap 42)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn unwrap_or_rejects_non_maybe_first_arg() {
        let err = tc_err("(unwrap-or 1 0)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn unwrap_or_rejects_non_i64_default() {
        let err = tc_err("(unwrap-or (some 1) true)");
        assert!(matches!(err.kind, TypeCheckErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn maybe_fn_return_type_accepted() {
        let src = "(program (fn wrap () maybe (some 7)) (call wrap))";
        assert_eq!(tc_ok(src), vec![ValueType::Maybe]);
    }

    #[test]
    fn maybe_fn_parameter_type_accepted() {
        let src = "(program (fn force (maybe) i64 (unwrap %0)) (call force (some 7)))";
        assert_eq!(tc_ok(src), vec![ValueType::I64]);
    }

    #[test]
    fn cond_with_maybe_branches() {
        let src = "(cond ((eq 1 1) (some 1)) (default (none)))";
        assert_eq!(tc_ok(src), vec![ValueType::Maybe]);
    }
}
