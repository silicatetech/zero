// SPDX-License-Identifier: AGPL-3.0-or-later
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArgShape {
    Value, // Atom or nested instruction (returns a value)
    List,  // Must be a List (used for if/loop bodies - instruction sequences)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueType {
    I64,
    Bytes,
    Handle,
    /// Stage 12 Paket B.2: Bool type ratified by the Spracherweiterung
    /// (`docs/discovery/stage-12-completion-plan.md` §B.2). Comparison
    /// operators (`eq`/`ne`/`lt`/`gt`/`le`/`ge`) and logical operators
    /// (`and`/`or`/`not`) traffic in this type. `if`/`while` conditions
    /// require it. Bool is distinct from I64 so the validator can
    /// reject "looks-like-zero is false" idioms that would erode the
    /// LLM's ability to reason about boolean expressions cleanly.
    Bool,
    /// Stage 12 Paket B.2: list of `i64`. Constructed by `(list …)`,
    /// inspected by `list-get`/`list-len`, extended by `list-append`.
    /// Element type is fixed to `i64` in MVP.
    List,
    /// Stage 12 Paket B.3: deterministic key-value map. Constructed
    /// by `(map-new)`, mutated by `map-put`, queried by `map-get` /
    /// `map-contains`. Both keys and values are `i64`. `BTreeMap`
    /// at runtime — `HashMap` would erode determinism.
    Map,
    /// Stage 12 Paket B.5: textual data carrier. Constructed by
    /// `string-from-int`, composed by `string-concat`, compared by
    /// `string-eq`. No parser-level string literal syntax exists.
    String,
    /// Phase 4 Step 5: Maybe / Option type. Constructed by `(some v)`
    /// or `(none)`, inspected by `is-some`/`is-none`, unwrapped by
    /// `unwrap`/`unwrap-or`. Phase 4 keeps `Maybe` monomorphic — the
    /// inner type is fixed to `I64` so `ValueType` stays `Copy` and
    /// the validator avoids the bookkeeping of parameterised types.
    /// Polymorphic inner types are deferred to a later language MP
    /// (the runtime carrier `Value::Maybe(Option<Box<Value>>)` is
    /// already permissive — only the validator is monomorphic).
    Maybe,
    /// Phase 4 Step 6: nominal struct type. The `u32` is an index
    /// into the type-checker's `StructTable`, assigned during Pass 1
    /// (before fn signatures). Carrying an index — not the struct
    /// name — keeps `ValueType` `Copy`, avoiding a cascade of
    /// `.clone()` calls through every `match vt` / `let ty = stack[i]`
    /// site in the type-checker. Lookup against the struct table
    /// resolves the index back to a `(name, fields)` pair for error
    /// messages and field access. Structs are nominal: two struct
    /// types compare unequal even if their field layouts are
    /// identical, because indices come from distinct top-level
    /// declarations.
    Struct(u32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstructionSignature {
    pub name: &'static str,
    pub args: &'static [ArgShape],
    pub inputs: &'static [ValueType], // what pops from stack (bottom-to-top)
    pub outputs: &'static [ValueType], // what pushes to stack (bottom-to-top)
}

// Polymorphic instructions (dup/drop/swap) have empty inputs/outputs.
// The simulator hardcodes them.
//
// Stage 12 Paket B (Quarks Spracherweiterung,
// `docs/discovery/stage-12-completion-plan.md` §B):
// - `if`/`while` conditions traffic in `Bool`, not `I64`. Branches and
//   loop bodies are arbitrary value-producing expressions (no longer
//   forced to be lists).
// - Comparison operators return `Bool`.
// - Logical operators (`and`/`or`/`not`) are first-class.
// - `loop` produces the value carried by `break`; `break` therefore
//   takes one argument.
//
// `if`, `loop`, `while`, `break`, `return`, `call`, `let`, `seq`,
// `policy`/`query`/`intent` are all custom-dispatched by the
// type-checker (`simulate_*`); the entries here serve only the
// structural argument-count / argument-shape pass in
// `validate_structure`. The structural signatures use `ArgShape::Value`
// for argument positions that take an arbitrary expression (an atom or
// a nested list), consistent with the rest of the surface — they no
// longer mandate `ArgShape::List` for branch / body slots.
pub const INSTRUCTIONS: &[InstructionSignature] = &[
    // Control Flow
    InstructionSignature {
        name: "if",
        args: &[ArgShape::Value, ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Bool],
        outputs: &[],
    },
    InstructionSignature {
        name: "loop",
        args: &[ArgShape::Value],
        inputs: &[],
        outputs: &[],
    },
    InstructionSignature {
        name: "while",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Bool],
        outputs: &[],
    },
    InstructionSignature {
        name: "break",
        args: &[ArgShape::Value],
        inputs: &[],
        outputs: &[],
    },
    InstructionSignature {
        name: "return",
        args: &[ArgShape::Value],
        inputs: &[],
        outputs: &[],
    },
    // Memory, Arithmetic
    InstructionSignature {
        name: "load",
        args: &[ArgShape::Value],
        inputs: &[ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "store",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[],
    },
    InstructionSignature {
        name: "add",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "sub",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "mul",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "div",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    // Phase 4 Step 1 — Bitwise integer operators. All five take two
    // `i64` and produce one `i64`, matching `add/sub/mul/div`. Naming
    // follows the kebab-case convention (`list-get`, `map-put`) — the
    // bare `and`/`or` symbols already belong to the Bool operators.
    // Shift semantics: `bit-shl` is a wrapping left shift, `bit-shr`
    // is an arithmetic right shift (SAR, hardware-aligned per
    // ADR-018). Runtime shift counts outside `0..64` surface as
    // `InterpretErrorKind::ShiftCountOutOfRange`.
    InstructionSignature {
        name: "bit-and",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "bit-or",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "bit-xor",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "bit-shl",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "bit-shr",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    // Comparison operators (Paket B.1): I64 in, Bool out.
    InstructionSignature {
        name: "eq",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::Bool],
    },
    InstructionSignature {
        name: "ne",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::Bool],
    },
    InstructionSignature {
        name: "lt",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::Bool],
    },
    InstructionSignature {
        name: "gt",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::Bool],
    },
    InstructionSignature {
        name: "le",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::Bool],
    },
    InstructionSignature {
        name: "ge",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::I64, ValueType::I64],
        outputs: &[ValueType::Bool],
    },
    // Logical operators (Paket B.2): Bool in, Bool out.
    InstructionSignature {
        name: "and",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Bool, ValueType::Bool],
        outputs: &[ValueType::Bool],
    },
    InstructionSignature {
        name: "or",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Bool, ValueType::Bool],
        outputs: &[ValueType::Bool],
    },
    InstructionSignature {
        name: "not",
        args: &[ArgShape::Value],
        inputs: &[ValueType::Bool],
        outputs: &[ValueType::Bool],
    },
    // Discard — custom-dispatched, stack manipulation in simulate_discard
    InstructionSignature {
        name: "discard",
        args: &[ArgShape::Value],
        inputs: &[],
        outputs: &[],
    },
    // Stack - polymorphic
    InstructionSignature {
        name: "dup",
        args: &[],
        inputs: &[],
        outputs: &[],
    },
    InstructionSignature {
        name: "drop",
        args: &[],
        inputs: &[],
        outputs: &[],
    },
    InstructionSignature {
        name: "swap",
        args: &[],
        inputs: &[],
        outputs: &[],
    },
    // No-op — stack-neutral, 0 args
    InstructionSignature {
        name: "nop",
        args: &[],
        inputs: &[],
        outputs: &[],
    },
    // OS Intents
    InstructionSignature {
        name: "send",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Handle, ValueType::Bytes],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "recv",
        args: &[],
        inputs: &[],
        outputs: &[ValueType::Handle, ValueType::I64, ValueType::Bytes],
    },
    InstructionSignature {
        name: "spawn",
        args: &[ArgShape::Value],
        inputs: &[ValueType::Bytes],
        outputs: &[ValueType::Handle],
    },
    InstructionSignature {
        name: "register",
        args: &[ArgShape::Value],
        inputs: &[ValueType::Bytes],
        outputs: &[ValueType::Handle],
    },
    // Paket B.2 — Lists. `(list …)` is variadic and custom-dispatched.
    // The fixed-arity operators sit here for the structural pass.
    InstructionSignature {
        name: "list-get",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::List, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "list-len",
        args: &[ArgShape::Value],
        inputs: &[ValueType::List],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "list-append",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::List, ValueType::I64],
        outputs: &[ValueType::List],
    },
    // Phase 4 Step 3 — Functional `list-slice` mirrors `bytes-slice`:
    // start/len i64 arguments, deterministic out-of-bounds error.
    InstructionSignature {
        name: "list-slice",
        args: &[ArgShape::Value, ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::List, ValueType::I64, ValueType::I64],
        outputs: &[ValueType::List],
    },
    // Paket B.3 — Maps. `BTreeMap`-backed at runtime.
    InstructionSignature {
        name: "map-new",
        args: &[],
        inputs: &[],
        outputs: &[ValueType::Map],
    },
    InstructionSignature {
        name: "map-put",
        args: &[ArgShape::Value, ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Map, ValueType::I64, ValueType::I64],
        outputs: &[ValueType::Map],
    },
    InstructionSignature {
        name: "map-get",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Map, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "map-contains",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Map, ValueType::I64],
        outputs: &[ValueType::Bool],
    },
    // Paket B.5 — Strings.
    InstructionSignature {
        name: "string-from-int",
        args: &[ArgShape::Value],
        inputs: &[ValueType::I64],
        outputs: &[ValueType::String],
    },
    InstructionSignature {
        name: "string-concat",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::String, ValueType::String],
        outputs: &[ValueType::String],
    },
    InstructionSignature {
        name: "string-eq",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::String, ValueType::String],
        outputs: &[ValueType::Bool],
    },
    // Paket B.6 — I/O via Capability gates. `read-handle` is a
    // structural fixed-arity op; `write-host-state` is variadic /
    // custom-dispatched because its key slot is a symbol literal
    // (mirroring `policy` / `query`).
    InstructionSignature {
        name: "read-handle",
        args: &[ArgShape::Value],
        inputs: &[ValueType::Handle],
        outputs: &[ValueType::Bytes],
    },
    // Phase 4 Step 2 — Bytes operations. All six operate on the
    // existing `Bytes` type (already plumbed through every layer).
    // Functional only — no in-place mutation (IR-SPEC §3 keeps
    // bytes immutable). `bytes-from-int` mirrors `i64::to_le_bytes`
    // (little-endian 8 bytes) for binary framing; mirror it on
    // the language side to avoid locale-dependent encodings.
    InstructionSignature {
        name: "bytes-len",
        args: &[ArgShape::Value],
        inputs: &[ValueType::Bytes],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "bytes-get",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Bytes, ValueType::I64],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "bytes-slice",
        args: &[ArgShape::Value, ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Bytes, ValueType::I64, ValueType::I64],
        outputs: &[ValueType::Bytes],
    },
    InstructionSignature {
        name: "bytes-concat",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Bytes, ValueType::Bytes],
        outputs: &[ValueType::Bytes],
    },
    InstructionSignature {
        name: "bytes-eq",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Bytes, ValueType::Bytes],
        outputs: &[ValueType::Bool],
    },
    InstructionSignature {
        name: "bytes-from-int",
        args: &[ArgShape::Value],
        inputs: &[ValueType::I64],
        outputs: &[ValueType::Bytes],
    },
    // Phase 4 Step 5 — Maybe / Option type. Solves the `map-get`
    // silent-zero ambiguity (key-present-with-value-0 vs.
    // key-absent) by making absence representable. `(some v)` /
    // `(none)` construct, `is-some`/`is-none` test, `unwrap` /
    // `unwrap-or` extract. Phase 4 is monomorphic over I64; the
    // runtime carrier `Value::Maybe(Option<Box<Value>>)` already
    // accommodates richer inner types when the validator catches up.
    InstructionSignature {
        name: "some",
        args: &[ArgShape::Value],
        inputs: &[ValueType::I64],
        outputs: &[ValueType::Maybe],
    },
    InstructionSignature {
        name: "none",
        args: &[],
        inputs: &[],
        outputs: &[ValueType::Maybe],
    },
    InstructionSignature {
        name: "is-some",
        args: &[ArgShape::Value],
        inputs: &[ValueType::Maybe],
        outputs: &[ValueType::Bool],
    },
    InstructionSignature {
        name: "is-none",
        args: &[ArgShape::Value],
        inputs: &[ValueType::Maybe],
        outputs: &[ValueType::Bool],
    },
    InstructionSignature {
        name: "unwrap",
        args: &[ArgShape::Value],
        inputs: &[ValueType::Maybe],
        outputs: &[ValueType::I64],
    },
    InstructionSignature {
        name: "unwrap-or",
        args: &[ArgShape::Value, ArgShape::Value],
        inputs: &[ValueType::Maybe, ValueType::I64],
        outputs: &[ValueType::I64],
    },
];

pub fn lookup(name: &str) -> Option<&'static InstructionSignature> {
    INSTRUCTIONS.iter().find(|sig| sig.name == name)
}
