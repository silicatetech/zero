// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interpreter values.

extern crate alloc;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A runtime value produced by the interpreter.
///
/// Stage 9: integers only. Stage 10 MP5 adds handles and bytes
/// (per ADR-027). Stage 12 Paket B.2 adds [`Value::Bool`] (per
/// `docs/discovery/stage-12-completion-plan.md` §B.2) so the
/// validator's `Bool` type has a distinct runtime carrier: comparison
/// operators (`eq`/`ne`/`lt`/`gt`/`le`/`ge`) and logical operators
/// (`and`/`or`/`not`) traffic in `Bool`, and `(if cond …)` /
/// `(while cond …)` demand it.
///
/// Stage 12 Paket B.2/B.3/B.5 (data-structure extension) adds:
/// - [`Value::List`] — list of `i64`, the runtime carrier for the
///   `(list …)` constructor and `list-get`/`list-len`/`list-append`
///   operations. Arena-allocated, deterministic iteration.
/// - [`Value::Map`] — `BTreeMap<i64, i64>` (not `HashMap`!) for the
///   `(map-new)` / `map-put`/`map-get`/`map-contains` family.
///   `BTreeMap` is mandatory per `hardware-abstraction-constraints.md`
///   §1.3: `HashMap`-iteration is non-deterministic and would erode
///   bitwise reproducibility.
/// - [`Value::String`] — textual data carrier for `string-from-int`,
///   `string-concat`, `string-eq`. No string literal syntax exists at
///   the parser level; strings are constructed at runtime from i64
///   via `string-from-int` and composed via `string-concat`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Integer(i64),
    Handle(u64),
    Bytes(Vec<u8>),
    Bool(bool),
    /// Paket B.2: homogeneous list of `i64`. Element type is fixed to
    /// `i64` in MVP — polymorphic element types are out of scope.
    List(Vec<i64>),
    /// Paket B.3: deterministic key-value map. `BTreeMap` ordering
    /// guarantees iteration determinism (hardware-abstraction-constraints
    /// §1.3).
    Map(BTreeMap<i64, i64>),
    /// Paket B.5: textual data carrier.
    String(String),
    /// Phase 4 Step 5: Option / Maybe carrier. Constructed by
    /// `(some v)` (`Some(Box::new(v))`) or `(none)` (`None`); tested
    /// by `is-some`/`is-none`; unwrapped by `unwrap` (traps on
    /// `None` with [`InterpretErrorKind::UnwrapOnNone`]) or
    /// `unwrap-or` (returns a caller-provided default on `None`).
    ///
    /// The inner is `Box<Value>` so the variant carrier is permissive:
    /// the runtime can already represent `Maybe` of any value type
    /// once the validator (currently monomorphic over I64) catches
    /// up. Phase 4 only exercises the `Maybe<i64>` slice.
    Maybe(Option<Box<Value>>),
    /// Phase 4 Step 6: nominal struct instance. The `name` field
    /// carries the user-declared struct name verbatim (used for
    /// summarisation/diagnostics); equality is structural over
    /// `(name, fields)`. The `fields` map uses `BTreeMap` for
    /// deterministic iteration order (required by Ring-0 / hardware-
    /// abstraction-constraints §1.3). Field order in the source
    /// declaration is reconstructible from the struct table; the
    /// runtime carrier only needs name-keyed access.
    Struct {
        name: String,
        fields: BTreeMap<String, Value>,
    },
}
