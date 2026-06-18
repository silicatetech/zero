// SPDX-License-Identifier: AGPL-3.0-or-later
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::ast::{Atom, SExpr};
use crate::instructions::{lookup, ArgShape};

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    pub kind: ValidationErrorKind,
    pub list_path: Vec<usize>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationErrorKind {
    UnknownInstruction {
        name: String,
    },
    ArgumentCountMismatch {
        instruction: String,
        expected: usize,
        actual: usize,
    },
    ArgumentShapeMismatch {
        instruction: String,
        argument_index: usize,
        expected: ArgShape,
        actual_is_list: bool,
    },
    NonSymbolInstruction,
    /// 12i — `(policy ...)` / `(query ...)` form is missing the
    /// mandatory subsystem-symbol or operation-symbol head. The
    /// instruction is variadic in its remaining args, but the first
    /// two slots after the head MUST be lowercase symbols
    /// (subsystem then operation).
    PolicyMalformed {
        instruction: String,
        message_kind: PolicyMalformedKind,
    },
    /// Phase 4 Step 4 — `(cond …)` form is malformed. The cond
    /// instruction is variadic; each clause must be a 2-element list
    /// `(predicate body)` and the last clause must use the keyword
    /// `default` as its predicate slot.
    CondMalformed {
        message_kind: CondMalformedKind,
    },
    /// Phase 4 Step 7 — `(match scrutinee …case…)` form is malformed.
    /// `match` is variadic over case clauses; each case is a 3-element
    /// list `(case pattern body)` and the last case must use the
    /// wildcard pattern `_` to make match total at the language
    /// surface (no exhaustiveness analysis required).
    MatchMalformed {
        message_kind: MatchMalformedKind,
    },
}

/// 12i — discriminator for the `(policy ...)` / `(query ...)` form
/// malformation. Reported via [`ValidationErrorKind::PolicyMalformed`].
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyMalformedKind {
    /// Fewer than two args after the `policy`/`query` head — the
    /// subsystem-symbol or operation-symbol slot is missing.
    MissingSubsystemOrOperation { actual_args: usize },
    /// The subsystem slot (args[0]) is not a symbol atom.
    NonSymbolSubsystem,
    /// The operation slot (args[1]) is not a symbol atom.
    NonSymbolOperation,
}

/// Phase 4 Step 7 — discriminator for `(match …)` malformation.
/// Reported via [`ValidationErrorKind::MatchMalformed`]. Pattern
/// shape (`(some %n)` / `(none)` / `(struct T %a %b …)` / integer
/// literal / `_`) is validated structurally; deeper type-level
/// checks (scrutinee type compatibility, struct existence, field
/// count) live in the type-checker.
#[derive(Debug, Clone, PartialEq)]
pub enum MatchMalformedKind {
    /// `(match scrutinee)` — no cases at all. A match must have at
    /// least one case (the mandatory wildcard fallback).
    NoCases,
    /// `match` has no scrutinee at all — `(match)` is malformed.
    MissingScrutinee,
    /// A case is not a list — e.g. `(match scrutinee 42)` instead of
    /// `(match scrutinee (case 42 body))`.
    CaseNotList { case_index: usize },
    /// A case list has the wrong number of items. A case must be
    /// `(case pattern body)` — exactly 3 items.
    CaseWrongArity {
        case_index: usize,
        actual_items: usize,
    },
    /// A case list's head is not the symbol `case`.
    CaseHeadNotCaseKeyword { case_index: usize },
    /// The last case is not the wildcard `_`. Match must be total at
    /// the language surface; the wildcard is mandatory.
    MissingWildcard,
    /// A non-last case uses the wildcard `_` as its pattern. The
    /// wildcard absorbs every value, so any later case would be
    /// unreachable. Surface that statically.
    WildcardNotLast { case_index: usize },
    /// A pattern's structural shape is unrecognised. Phase 4 supports
    /// `(some %n)`, `(none)`, `(struct T %a %b …)`, integer literal,
    /// and `_`. Everything else is rejected at the structural pass.
    UnrecognisedPattern { case_index: usize },
}

/// Phase 4 Step 4 — discriminator for `(cond …)` malformation.
/// Reported via [`ValidationErrorKind::CondMalformed`].
#[derive(Debug, Clone, PartialEq)]
pub enum CondMalformedKind {
    /// `(cond)` with no clauses. A cond must have at least one
    /// clause (the mandatory `default`).
    NoClauses,
    /// A clause is not a 2-element list. Each clause is shaped
    /// `(predicate body)`.
    ClauseNotList { clause_index: usize },
    /// A clause list has the wrong number of items (must be 2:
    /// `(predicate body)`).
    ClauseWrongArity {
        clause_index: usize,
        actual_items: usize,
    },
    /// The last clause is not the mandatory `(default body)`
    /// fallback (its predicate is not the symbol `default`).
    MissingDefaultClause,
    /// A non-last clause used the reserved keyword `default` as its
    /// predicate. `default` may appear only in the last clause.
    DefaultNotLast { clause_index: usize },
}

/// Validate structural correctness of a parsed S-expression program.
///
/// Checks 2 and 3 from IR-SPEC Section 6:
/// - Check 2: every list's first symbol must be a known instruction.
/// - Check 3: argument count and shape must match the instruction signature.
///
/// Does NOT perform type checking (Check 4) or stack-balance analysis (Check 5).
pub fn validate_structure(program: &SExpr) -> Result<(), ValidationError> {
    let mut path = Vec::new();
    validate_sexpr(program, &mut path)
}

fn validate_sexpr(sexpr: &SExpr, path: &mut Vec<usize>) -> Result<(), ValidationError> {
    match sexpr {
        SExpr::Atom(_) => Ok(()),
        SExpr::List(items) => validate_list(items, path),
    }
}

/// 12i — structurally validate `(policy <subsystem> <operation> ...)`
/// or `(query <subsystem> <metric> ...)`. The subsystem and operation
/// slots must be symbol atoms; any number of extra value-args may
/// follow and are recursed into for nested instruction validation.
fn validate_policy_or_query(
    instruction_name: &str,
    items: &[SExpr],
    path: &mut Vec<usize>,
) -> Result<(), ValidationError> {
    let args = &items[1..];
    if args.len() < 2 {
        return Err(ValidationError {
            kind: ValidationErrorKind::PolicyMalformed {
                instruction: String::from(instruction_name),
                message_kind: PolicyMalformedKind::MissingSubsystemOrOperation {
                    actual_args: args.len(),
                },
            },
            list_path: path.clone(),
            message: format!(
                "instruction '{}' requires at least <subsystem> <operation> (got {} args)",
                instruction_name,
                args.len()
            ),
        });
    }

    if !matches!(&args[0], SExpr::Atom(Atom::Symbol(_))) {
        return Err(ValidationError {
            kind: ValidationErrorKind::PolicyMalformed {
                instruction: String::from(instruction_name),
                message_kind: PolicyMalformedKind::NonSymbolSubsystem,
            },
            list_path: path.clone(),
            message: format!(
                "instruction '{}' subsystem (arg 0) must be a symbol",
                instruction_name
            ),
        });
    }

    if !matches!(&args[1], SExpr::Atom(Atom::Symbol(_))) {
        return Err(ValidationError {
            kind: ValidationErrorKind::PolicyMalformed {
                instruction: String::from(instruction_name),
                message_kind: PolicyMalformedKind::NonSymbolOperation,
            },
            list_path: path.clone(),
            message: format!(
                "instruction '{}' operation (arg 1) must be a symbol",
                instruction_name
            ),
        });
    }

    // Recurse into children — value-args may themselves be instructions.
    for (i, child) in items.iter().enumerate() {
        path.push(i);
        validate_sexpr(child, path)?;
        path.pop();
    }

    Ok(())
}

/// Phase 4 Step 4 — structurally validate
/// `(cond (p1 b1) (p2 b2) … (default body))`. Each clause must be
/// a 2-element list. The last clause must have the symbol `default`
/// as its predicate; no earlier clause may use `default` in that
/// position. Predicate and body sub-expressions are recursed into
/// for nested instruction validation.
fn validate_cond(items: &[SExpr], path: &mut Vec<usize>) -> Result<(), ValidationError> {
    let clauses = &items[1..];
    if clauses.is_empty() {
        return Err(ValidationError {
            kind: ValidationErrorKind::CondMalformed {
                message_kind: CondMalformedKind::NoClauses,
            },
            list_path: path.clone(),
            message: String::from("cond requires at least one (default body) clause"),
        });
    }

    let last_index = clauses.len() - 1;
    for (i, clause) in clauses.iter().enumerate() {
        // Each clause must be a list.
        let clause_items = match clause {
            SExpr::List(items) => items,
            _ => {
                return Err(ValidationError {
                    kind: ValidationErrorKind::CondMalformed {
                        message_kind: CondMalformedKind::ClauseNotList { clause_index: i },
                    },
                    list_path: path.clone(),
                    message: format!(
                        "cond clause {} must be a list of the form (predicate body)",
                        i
                    ),
                });
            }
        };
        if clause_items.len() != 2 {
            return Err(ValidationError {
                kind: ValidationErrorKind::CondMalformed {
                    message_kind: CondMalformedKind::ClauseWrongArity {
                        clause_index: i,
                        actual_items: clause_items.len(),
                    },
                },
                list_path: path.clone(),
                message: format!(
                    "cond clause {} must be (predicate body); got {} items",
                    i,
                    clause_items.len()
                ),
            });
        }
        let is_default = matches!(&clause_items[0],
            SExpr::Atom(Atom::Symbol(s)) if s == "default");
        if i == last_index {
            if !is_default {
                return Err(ValidationError {
                    kind: ValidationErrorKind::CondMalformed {
                        message_kind: CondMalformedKind::MissingDefaultClause,
                    },
                    list_path: path.clone(),
                    message: String::from(
                        "cond last clause must be (default body) — fallback is mandatory",
                    ),
                });
            }
        } else if is_default {
            return Err(ValidationError {
                kind: ValidationErrorKind::CondMalformed {
                    message_kind: CondMalformedKind::DefaultNotLast { clause_index: i },
                },
                list_path: path.clone(),
                message: format!(
                    "cond clause {} uses (default …) but default may appear only in the last clause",
                    i
                ),
            });
        }
    }

    // Recurse into each clause's predicate and body separately —
    // the clause list `(predicate body)` is itself NOT an instruction
    // list, so we cannot dispatch on its head. Predicates and bodies
    // ARE value-producing expressions that may contain nested
    // instructions. For the default clause, only its body is
    // recursed into (the `default` keyword head is structurally
    // verified above).
    //
    // The `default` symbol itself is not an instruction and must not
    // be re-walked as one. items[0] is the `cond` head symbol; we
    // also do not re-recurse on it.
    for (i, clause) in clauses.iter().enumerate() {
        let clause_items = match clause {
            SExpr::List(items) => items,
            // Already errored above; unreachable in practice.
            _ => continue,
        };
        // path to clause = path + [i + 1] (clause i lives at items[i+1])
        path.push(i + 1);
        // Recurse predicate at position 0 within the clause unless
        // it is the `default` keyword in the last clause.
        let is_default_head = i == last_index
            && matches!(&clause_items[0], SExpr::Atom(Atom::Symbol(s)) if s == "default");
        if !is_default_head {
            path.push(0);
            validate_sexpr(&clause_items[0], path)?;
            path.pop();
        }
        // Recurse body at position 1.
        path.push(1);
        validate_sexpr(&clause_items[1], path)?;
        path.pop();
        path.pop();
    }

    Ok(())
}

/// Phase 4 Step 7 — structurally validate
/// `(match scrutinee (case pattern body) … (case _ body))`. The
/// scrutinee must be present (any expression); each case must be a
/// 3-element list `(case pattern body)`; the last case's pattern
/// must be the wildcard symbol `_`; no earlier case may use `_`
/// (would render later cases unreachable).
///
/// Pattern shape recognised at this layer:
///   - integer literal           (Atom::Integer)
///   - wildcard                  (Atom::Symbol("_"))
///   - `(some %n)`               (list head "some", arity 2)
///   - `(none)`                  (list head "none", arity 1)
///   - `(struct T %a %b …)`      (list head "struct", arity ≥ 2)
///
/// Type-level pattern validity (scrutinee type matches pattern,
/// struct exists, struct arity matches field count) is owned by the
/// type-checker (`simulate_match`).
///
/// Recursion into nested instructions: `match` recurses into the
/// scrutinee and every case body. Patterns themselves are NOT
/// recursed into as instructions — `(some %n)` and friends are
/// pattern syntax, not instruction calls.
fn validate_match(items: &[SExpr], path: &mut Vec<usize>) -> Result<(), ValidationError> {
    // items[0] = "match", items[1] = scrutinee, items[2..] = cases.
    if items.len() < 2 {
        return Err(ValidationError {
            kind: ValidationErrorKind::MatchMalformed {
                message_kind: MatchMalformedKind::MissingScrutinee,
            },
            list_path: path.clone(),
            message: String::from("match requires a scrutinee expression"),
        });
    }
    let cases = &items[2..];
    if cases.is_empty() {
        return Err(ValidationError {
            kind: ValidationErrorKind::MatchMalformed {
                message_kind: MatchMalformedKind::NoCases,
            },
            list_path: path.clone(),
            message: String::from(
                "match requires at least one case (the mandatory wildcard fallback)",
            ),
        });
    }

    let last_index = cases.len() - 1;
    for (i, case) in cases.iter().enumerate() {
        let case_items = match case {
            SExpr::List(l) => l,
            _ => {
                return Err(ValidationError {
                    kind: ValidationErrorKind::MatchMalformed {
                        message_kind: MatchMalformedKind::CaseNotList { case_index: i },
                    },
                    list_path: path.clone(),
                    message: format!("match case {} must be a (case pattern body) list", i),
                });
            }
        };
        if case_items.len() != 3 {
            return Err(ValidationError {
                kind: ValidationErrorKind::MatchMalformed {
                    message_kind: MatchMalformedKind::CaseWrongArity {
                        case_index: i,
                        actual_items: case_items.len(),
                    },
                },
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
            return Err(ValidationError {
                kind: ValidationErrorKind::MatchMalformed {
                    message_kind: MatchMalformedKind::CaseHeadNotCaseKeyword { case_index: i },
                },
                list_path: path.clone(),
                message: format!("match case {} must start with the symbol `case`", i),
            });
        }

        // case_items[1] is the pattern; check structural shape.
        let pattern = &case_items[1];
        let is_wildcard = matches!(pattern,
            SExpr::Atom(Atom::Symbol(s)) if s == "_");
        let is_integer_literal = matches!(pattern, SExpr::Atom(Atom::Integer(_)));
        let pattern_list_head = if let SExpr::List(p_items) = pattern {
            if let Some(SExpr::Atom(Atom::Symbol(s))) = p_items.first() {
                Some((s.as_str(), p_items.len()))
            } else {
                None
            }
        } else {
            None
        };
        let recognised = is_wildcard
            || is_integer_literal
            || matches!(pattern_list_head, Some(("some", 2)))
            || matches!(pattern_list_head, Some(("none", 1)))
            || matches!(pattern_list_head, Some(("struct", n)) if n >= 2);
        if !recognised {
            return Err(ValidationError {
                kind: ValidationErrorKind::MatchMalformed {
                    message_kind: MatchMalformedKind::UnrecognisedPattern { case_index: i },
                },
                list_path: path.clone(),
                message: format!(
                    "match case {} pattern is not recognised (Phase 4 supports integer literal, `_`, `(some %n)`, `(none)`, `(struct T %a %b …)`)",
                    i
                ),
            });
        }

        // Wildcard placement: required as the last case; forbidden
        // anywhere else.
        if i == last_index {
            if !is_wildcard {
                return Err(ValidationError {
                    kind: ValidationErrorKind::MatchMalformed {
                        message_kind: MatchMalformedKind::MissingWildcard,
                    },
                    list_path: path.clone(),
                    message: String::from(
                        "match last case must be (case _ body) — wildcard fallback is mandatory",
                    ),
                });
            }
        } else if is_wildcard {
            return Err(ValidationError {
                kind: ValidationErrorKind::MatchMalformed {
                    message_kind: MatchMalformedKind::WildcardNotLast { case_index: i },
                },
                list_path: path.clone(),
                message: format!(
                    "match case {} uses `_` but the wildcard may appear only in the last case",
                    i
                ),
            });
        }
    }

    // Recurse into the scrutinee expression (items[1]) and each case
    // body (case_items[2]). Patterns are not recursed into as
    // instruction lists.
    path.push(1);
    validate_sexpr(&items[1], path)?;
    path.pop();
    for (i, case) in cases.iter().enumerate() {
        let case_items = match case {
            SExpr::List(l) => l,
            // Already errored above.
            _ => continue,
        };
        // case lives at items[i+2] → 0-based position i+2.
        path.push(i + 2);
        // case body at position 2 inside the clause.
        path.push(2);
        validate_sexpr(&case_items[2], path)?;
        path.pop();
        path.pop();
    }
    Ok(())
}

fn validate_list(items: &[SExpr], path: &mut Vec<usize>) -> Result<(), ValidationError> {
    // List must not be empty (parser already catches this, but defensive)
    if items.is_empty() {
        return Err(ValidationError {
            kind: ValidationErrorKind::NonSymbolInstruction,
            list_path: path.clone(),
            message: String::from("empty list"),
        });
    }

    // First element must be a symbol
    let instruction_name = match &items[0] {
        SExpr::Atom(Atom::Symbol(name)) => name.as_str(),
        _ => {
            return Err(ValidationError {
                kind: ValidationErrorKind::NonSymbolInstruction,
                list_path: path.clone(),
                message: String::from("first element of list must be instruction symbol"),
            });
        }
    };

    // Variadic special-case: `policy`, `query`, and `intent` are
    // structurally `(<head> <subsystem-symbol> <operation-symbol>
    // <value-args...>)` for policy/query, and `(intent <args...>)` for
    // intent. They take a variable number of arguments and are not
    // listed in the static `INSTRUCTIONS` table; the type-checker
    // (`simulate_policy`/`simulate_query`/`simulate_intent`) enforces
    // their semantic shape.
    if instruction_name == "policy" || instruction_name == "query" {
        return validate_policy_or_query(instruction_name, items, path);
    }
    // Phase 4 Step 4 — `(cond (p1 b1) (p2 b2) … (default body))` is
    // a variadic N-way dispatch. Structural validation enforces:
    // at least one clause, each clause is a 2-element list, and the
    // last clause must use the `default` keyword as its predicate.
    // The type-checker (`simulate_cond`) enforces semantic shape
    // (predicates must be Bool, branches must agree).
    if instruction_name == "cond" {
        return validate_cond(items, path);
    }
    // Phase 4 Step 7 — `(match scrutinee (case pattern body) …)` is
    // a variadic dispatch over pattern shapes. Structural validation
    // enforces: scrutinee present, at least one case, each case is a
    // 3-element list `(case pattern body)`, wildcard `_` only as the
    // last case. The type-checker (`simulate_match`) enforces
    // pattern/scrutinee type compatibility.
    if instruction_name == "match" {
        return validate_match(items, path);
    }
    // intent recurses into all children for nested instruction
    // validation; its arg list is uniform (value-producing
    // expressions).
    if instruction_name == "intent" {
        for (i, child) in items.iter().enumerate() {
            path.push(i);
            validate_sexpr(child, path)?;
            path.pop();
        }
        return Ok(());
    }
    // program/fn/call/let/seq are variadic and have idiosyncratic
    // substructures: `fn` has a `(i64 bytes …)` param-type list
    // whose head is a type-symbol, not an instruction; `let` binds
    // a `%n` atom; `seq` mixes effects and a tail value-expr.
    // The type-checker (`simulate_program` / `simulate_*`) owns
    // their structural validation; the structural pass just
    // accepts them at the surface so type-checking has a chance
    // to run. Sub-expressions that DO need structural validation
    // (e.g. the value-producing parts of `let`/`seq`) are reached
    // when the type-checker recurses into them via `simulate`.
    //
    // Stage 12 Paket B (data-structure / loop / I/O extension)
    // adds four more idiosyncratic forms:
    // - `(list a b c …)` — variadic constructor, all args are I64;
    // - `(loop-with-bound bound body)` — body is a value-producing
    //   expression, bound is an I64 expression evaluated once;
    // - `(for %n source body)` — %n is a Parameter atom binding;
    // - `(write-host-state key value)` — key is a symbol literal.
    // Each is structurally accepted here; the type-checker performs
    // shape + type validation in its own `simulate_*` helper.
    if instruction_name == "program"
        || instruction_name == "fn"
        || instruction_name == "call"
        || instruction_name == "let"
        || instruction_name == "seq"
        || instruction_name == "list"
        || instruction_name == "loop-with-bound"
        || instruction_name == "for"
        || instruction_name == "write-host-state"
    {
        return Ok(());
    }
    // Phase 4 Step 6 — nominal structs.
    //
    // The four forms are variadic with idiosyncratic substructures
    // that the static INSTRUCTIONS table cannot describe:
    //   `(struct name ((f t) …))`          — declaration; field-list
    //                                        items are (name type-symbol)
    //                                        pairs, NOT instructions.
    //   `(struct-new name v1 v2 … vN)`     — name symbol + value args.
    //   `(struct-get expr field-name)`     — value expr + symbol literal.
    //   `(struct-set expr field-name v)`   — value expr + symbol literal
    //                                        + value expr.
    // Structural validation accepts the surface; the type-checker
    // (`simulate_struct_*` / struct collection in Pass 1) owns
    // semantic shape (field types, existence, recursive declaration
    // rejection).
    //
    // We selectively recurse into ONLY the value-producing slots so
    // structural checks (UnknownInstruction, ArgumentCountMismatch)
    // still fire inside nested expressions, without walking field-
    // name or struct-name symbol literals as if they were
    // instructions.
    if instruction_name == "struct" {
        // Struct declarations have no value-producing children — the
        // field-list is a list of (name type-symbol) clauses where
        // the heads are field-name symbols, not instructions.
        return Ok(());
    }
    if instruction_name == "struct-new" {
        // items[0]=struct-new, items[1]=struct-name symbol,
        // items[2..]=value args.
        for (i, child) in items.iter().enumerate().skip(2) {
            path.push(i);
            validate_sexpr(child, path)?;
            path.pop();
        }
        return Ok(());
    }
    if instruction_name == "struct-get" {
        // items[0]=struct-get, items[1]=value expr,
        // items[2]=field-name symbol literal.
        if items.len() >= 2 {
            path.push(1);
            validate_sexpr(&items[1], path)?;
            path.pop();
        }
        return Ok(());
    }
    if instruction_name == "struct-set" {
        // items[0]=struct-set, items[1]=value expr,
        // items[2]=field-name symbol literal, items[3]=value expr.
        if items.len() >= 2 {
            path.push(1);
            validate_sexpr(&items[1], path)?;
            path.pop();
        }
        if items.len() >= 4 {
            path.push(3);
            validate_sexpr(&items[3], path)?;
            path.pop();
        }
        return Ok(());
    }

    // Check 2: instruction must be known
    let signature = match lookup(instruction_name) {
        Some(sig) => sig,
        None => {
            return Err(ValidationError {
                kind: ValidationErrorKind::UnknownInstruction {
                    name: String::from(instruction_name),
                },
                list_path: path.clone(),
                message: format!("unknown instruction '{}'", instruction_name),
            });
        }
    };

    // Check 3a: argument count
    let actual_count = items.len() - 1;
    let expected_count = signature.args.len();
    if actual_count != expected_count {
        return Err(ValidationError {
            kind: ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from(instruction_name),
                expected: expected_count,
                actual: actual_count,
            },
            list_path: path.clone(),
            message: format!(
                "instruction '{}' expects {} arguments, got {}",
                instruction_name, expected_count, actual_count
            ),
        });
    }

    // Check 3b: argument shape
    for (arg_index, (arg, expected_shape)) in
        items[1..].iter().zip(signature.args.iter()).enumerate()
    {
        let actual_is_list = matches!(arg, SExpr::List(_));
        match (expected_shape, actual_is_list) {
            (ArgShape::List, false) => {
                return Err(ValidationError {
                    kind: ValidationErrorKind::ArgumentShapeMismatch {
                        instruction: String::from(instruction_name),
                        argument_index: arg_index,
                        expected: *expected_shape,
                        actual_is_list: false,
                    },
                    list_path: path.clone(),
                    message: format!(
                        "instruction '{}' argument {} must be a list (instruction sequence), got atom",
                        instruction_name, arg_index
                    ),
                });
            }
            _ => {
                // ArgShape::Value accepts both atoms and nested lists.
                // ArgShape::List with actual_is_list == true is fine.
                // Distinguishing value-producing vs non-value-producing lists
                // requires stack simulation (Check 4, Micro-Prompt 3).
            }
        }
    }

    // Recurse into children to validate nested instructions
    for (i, child) in items.iter().enumerate() {
        path.push(i);
        validate_sexpr(child, path)?;
        path.pop();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use alloc::string::String;
    use alloc::vec;

    /// Helper: parse then validate, return Result
    fn pv(input: &str) -> Result<(), ValidationError> {
        let sexpr = parse(input).expect("parse should succeed for validation tests");
        validate_structure(&sexpr)
    }

    // ── Valid programs ──────────────────────────────────────────

    #[test]
    fn valid_add() {
        assert!(pv("(add 1 2)").is_ok());
    }

    #[test]
    fn valid_return_nested() {
        assert!(pv("(return (add 1 2))").is_ok());
    }

    #[test]
    fn valid_if_with_list_bodies() {
        assert!(pv("(if (eq 0 1) (return 1) (return 0))").is_ok());
    }

    #[test]
    fn valid_loop_break() {
        // Paket B.1c: break carries a value; loop produces it.
        assert!(pv("(loop (break false))").is_ok());
    }

    #[test]
    fn valid_zero_arg_dup() {
        assert!(pv("(dup)").is_ok());
    }

    #[test]
    fn valid_zero_arg_drop() {
        assert!(pv("(drop)").is_ok());
    }

    #[test]
    fn valid_zero_arg_swap() {
        assert!(pv("(swap)").is_ok());
    }

    #[test]
    fn valid_zero_arg_recv() {
        assert!(pv("(recv)").is_ok());
    }

    #[test]
    fn valid_break_with_value() {
        // Paket B.1c: (break v) is a single-arg form.
        // Structural shape only; semantic outer-loop check is
        // type_checker territory.
        assert!(pv("(break false)").is_ok());
    }

    #[test]
    fn valid_send() {
        assert!(pv("(send @5 #x48656c6c6f)").is_ok());
    }

    #[test]
    fn valid_store() {
        assert!(pv("(store 0 42)").is_ok());
    }

    #[test]
    fn valid_deeply_nested() {
        assert!(pv("(return (add (mul 2 3) (sub 10 4)))").is_ok());
    }

    // ── Phase 4 Step 1 — Bitwise ops: structural validation ────

    #[test]
    fn valid_bit_and() {
        assert!(pv("(bit-and 255 15)").is_ok());
    }

    #[test]
    fn valid_bit_or() {
        assert!(pv("(bit-or 1 2)").is_ok());
    }

    #[test]
    fn valid_bit_xor() {
        assert!(pv("(bit-xor 5 3)").is_ok());
    }

    #[test]
    fn valid_bit_shl() {
        assert!(pv("(bit-shl 1 8)").is_ok());
    }

    #[test]
    fn valid_bit_shr() {
        assert!(pv("(bit-shr 256 4)").is_ok());
    }

    #[test]
    fn valid_bit_nested() {
        assert!(pv("(bit-and (bit-shl 1 8) (bit-or 255 0))").is_ok());
    }

    // ── Check 2: Unknown Instruction ───────────────────────────

    #[test]
    fn unknown_foo() {
        let err = pv("(foo 1 2)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::UnknownInstruction {
                name: String::from("foo")
            }
        );
    }

    #[test]
    fn unknown_case_sensitive() {
        // Parser rejects uppercase at parse level (InvalidSymbol).
        // So we test a lowercase-but-wrong name instead.
        let err = pv("(bogus-op)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::UnknownInstruction {
                name: String::from("bogus-op")
            }
        );
    }

    #[test]
    fn unknown_typo() {
        let err = pv("(ad 1 2)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::UnknownInstruction {
                name: String::from("ad")
            }
        );
    }

    // ── Check 3a: Argument Count ───────────────────────────────

    #[test]
    fn argcount_add_too_few() {
        let err = pv("(add 1)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("add"),
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn argcount_add_too_many() {
        let err = pv("(add 1 2 3)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("add"),
                expected: 2,
                actual: 3,
            }
        );
    }

    // ── Phase 4 Step 1 — bit-* arity ─────────────────────────

    #[test]
    fn argcount_bit_and_too_few() {
        let err = pv("(bit-and 1)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bit-and"),
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn argcount_bit_or_too_many() {
        let err = pv("(bit-or 1 2 3)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bit-or"),
                expected: 2,
                actual: 3,
            }
        );
    }

    #[test]
    fn argcount_bit_xor_zero_args() {
        let err = pv("(bit-xor)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bit-xor"),
                expected: 2,
                actual: 0,
            }
        );
    }

    #[test]
    fn argcount_bit_shl_too_many() {
        let err = pv("(bit-shl 1 2 3 4)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bit-shl"),
                expected: 2,
                actual: 4,
            }
        );
    }

    #[test]
    fn argcount_bit_shr_too_few() {
        let err = pv("(bit-shr 64)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bit-shr"),
                expected: 2,
                actual: 1,
            }
        );
    }

    // ── Phase 4 Step 2 — bytes-* arity ───────────────────────

    #[test]
    fn argcount_bytes_len_too_many() {
        let err = pv("(bytes-len #x00 #x01)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bytes-len"),
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn argcount_bytes_get_too_few() {
        let err = pv("(bytes-get #x00)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bytes-get"),
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn argcount_bytes_slice_too_few() {
        let err = pv("(bytes-slice #x0102 0)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bytes-slice"),
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn argcount_bytes_concat_too_many() {
        let err = pv("(bytes-concat #x00 #x01 #x02)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bytes-concat"),
                expected: 2,
                actual: 3,
            }
        );
    }

    #[test]
    fn argcount_bytes_eq_zero_args() {
        let err = pv("(bytes-eq)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bytes-eq"),
                expected: 2,
                actual: 0,
            }
        );
    }

    #[test]
    fn argcount_bytes_from_int_zero_args() {
        let err = pv("(bytes-from-int)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("bytes-from-int"),
                expected: 1,
                actual: 0,
            }
        );
    }

    // ── Phase 4 Step 3 — list-slice arity ────────────────────

    #[test]
    fn argcount_list_slice_too_few() {
        let err = pv("(list-slice (list 1 2) 0)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("list-slice"),
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn argcount_list_slice_too_many() {
        let err = pv("(list-slice (list 1 2) 0 1 2)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("list-slice"),
                expected: 3,
                actual: 4,
            }
        );
    }

    #[test]
    fn argcount_list_slice_zero_args() {
        let err = pv("(list-slice)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("list-slice"),
                expected: 3,
                actual: 0,
            }
        );
    }

    // ── Phase 4 Step 5 — Maybe instruction arity ─────────────

    #[test]
    fn valid_some() {
        assert!(pv("(some 42)").is_ok());
    }

    #[test]
    fn valid_none() {
        assert!(pv("(none)").is_ok());
    }

    #[test]
    fn valid_is_some() {
        assert!(pv("(is-some (some 1))").is_ok());
    }

    #[test]
    fn valid_is_none() {
        assert!(pv("(is-none (none))").is_ok());
    }

    #[test]
    fn valid_unwrap() {
        assert!(pv("(unwrap (some 1))").is_ok());
    }

    #[test]
    fn valid_unwrap_or() {
        assert!(pv("(unwrap-or (some 1) 0)").is_ok());
    }

    #[test]
    fn argcount_some_zero_args() {
        let err = pv("(some)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("some"),
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn argcount_some_too_many() {
        let err = pv("(some 1 2)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("some"),
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn argcount_none_with_arg() {
        let err = pv("(none 1)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("none"),
                expected: 0,
                actual: 1,
            }
        );
    }

    #[test]
    fn argcount_is_some_zero_args() {
        let err = pv("(is-some)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("is-some"),
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn argcount_is_none_too_many() {
        let err = pv("(is-none (none) 1)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("is-none"),
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn argcount_unwrap_zero_args() {
        let err = pv("(unwrap)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("unwrap"),
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn argcount_unwrap_or_too_few() {
        let err = pv("(unwrap-or (some 1))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("unwrap-or"),
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn argcount_unwrap_or_too_many() {
        let err = pv("(unwrap-or (some 1) 0 0)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("unwrap-or"),
                expected: 2,
                actual: 3,
            }
        );
    }

    #[test]
    fn argcount_dup_with_arg() {
        let err = pv("(dup 1)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("dup"),
                expected: 0,
                actual: 1,
            }
        );
    }

    #[test]
    fn argcount_if_too_few() {
        let err = pv("(if 0)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("if"),
                expected: 3,
                actual: 1,
            }
        );
    }

    #[test]
    fn argcount_return_too_many() {
        let err = pv("(return 1 2)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("return"),
                expected: 1,
                actual: 2,
            }
        );
    }

    // ── Check 3b: Argument Shape ───────────────────────────────

    #[test]
    fn paket_b_if_branches_accept_atoms() {
        // Paket B.1b: branches are arbitrary value-producing
        // expressions, no longer constrained to be lists. The
        // structural pass accepts the form; the type-checker
        // separately verifies cond is Bool and branches match.
        assert!(pv("(if true 1 2)").is_ok());
    }

    #[test]
    fn paket_b_if_with_mixed_branch_shapes_ok() {
        // Paket B.1b: list and atom branches coexist; both legal
        // structurally.
        assert!(pv("(if true (add 1 2) 3)").is_ok());
    }

    #[test]
    fn paket_b_loop_body_accepts_atom() {
        // Paket B.1c: loop body is an arbitrary value-producing
        // expression. Structurally `(loop 42)` is now accepted;
        // the type-checker rejects it separately (no reachable
        // break ⇒ LoopWithoutBreak).
        assert!(pv("(loop 42)").is_ok());
    }

    // ── NonSymbolInstruction ───────────────────────────────────

    #[test]
    fn nonsymbol_integer_first() {
        let err = pv("(1 2 3)").unwrap_err();
        assert_eq!(err.kind, ValidationErrorKind::NonSymbolInstruction);
    }

    #[test]
    fn nonsymbol_handle_first() {
        let err = pv("(@5 1 2)").unwrap_err();
        assert_eq!(err.kind, ValidationErrorKind::NonSymbolInstruction);
    }

    #[test]
    fn nonsymbol_list_first() {
        let err = pv("((add 1 2) 3)").unwrap_err();
        assert_eq!(err.kind, ValidationErrorKind::NonSymbolInstruction);
    }

    // ── Path tracking ──────────────────────────────────────────

    #[test]
    fn path_nested_unknown() {
        // (return (foo))
        // return is items[0], (foo) is items[1]
        // inside (foo): foo is items[0]
        // error should be at path [1] (the nested list)
        let err = pv("(return (foo))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::UnknownInstruction {
                name: String::from("foo")
            }
        );
        assert_eq!(err.list_path, vec![1]);
    }

    #[test]
    fn path_nested_argcount() {
        // (if (add 1) (return 0) (return 1))
        // if is items[0], (add 1) is items[1]
        // error in (add 1) at path [1]
        let err = pv("(if (add 1) (return 0) (return 1))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("add"),
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(err.list_path, vec![1]);
    }

    #[test]
    fn path_nested_argcount_in_loop() {
        // Paket B.1b retired the ArgumentShapeMismatch path for
        // if/loop body slots (they now accept atoms). Path-
        // tracking is still exercised by ArgumentCountMismatch:
        // `(loop (add 1))` — add is missing its second arg, inside
        // the loop's body slot.
        let err = pv("(loop (add 1))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::ArgumentCountMismatch {
                instruction: String::from("add"),
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(err.list_path, vec![1]);
    }

    #[test]
    fn path_deeply_nested_error() {
        // (loop (return (foo)))
        // loop[0], (return (foo))[1], return[0] inside, (foo)[1] inside
        // path: [1, 1]
        let err = pv("(loop (return (foo)))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::UnknownInstruction {
                name: String::from("foo")
            }
        );
        assert_eq!(err.list_path, vec![1, 1]);
    }

    // ── 12i: Policy / Query structural validation ───────────────

    #[test]
    fn valid_policy_form() {
        assert!(pv("(policy gpu allocate-slice 5 50)").is_ok());
    }

    #[test]
    fn valid_policy_with_no_value_args() {
        // pause/resume have <subsystem> <operation> <sandbox-id>;
        // <subsystem> <operation> alone (zero value-args) is also
        // structurally valid even if the manager rejects it as
        // operation-shape mismatch later.
        assert!(pv("(policy gpu release-slice)").is_ok());
    }

    #[test]
    fn valid_query_form() {
        assert!(pv("(query gpu utilization)").is_ok());
    }

    #[test]
    fn valid_policy_with_nested_arithmetic_arg() {
        assert!(pv("(policy gpu allocate-slice 5 (add 25 25))").is_ok());
    }

    #[test]
    fn policy_missing_subsystem_and_operation_is_malformed() {
        let err = pv("(policy)").unwrap_err();
        match err.kind {
            ValidationErrorKind::PolicyMalformed {
                instruction,
                message_kind: PolicyMalformedKind::MissingSubsystemOrOperation { actual_args },
            } => {
                assert_eq!(instruction, "policy");
                assert_eq!(actual_args, 0);
            }
            other => panic!(
                "expected PolicyMalformed::MissingSubsystemOrOperation, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn query_missing_metric_is_malformed() {
        let err = pv("(query gpu)").unwrap_err();
        match err.kind {
            ValidationErrorKind::PolicyMalformed {
                instruction,
                message_kind: PolicyMalformedKind::MissingSubsystemOrOperation { actual_args },
            } => {
                assert_eq!(instruction, "query");
                assert_eq!(actual_args, 1);
            }
            other => panic!("expected PolicyMalformed, got {:?}", other),
        }
    }

    #[test]
    fn policy_non_symbol_subsystem_is_malformed() {
        let err = pv("(policy 1 set-priority 0)").unwrap_err();
        assert!(matches!(
            err.kind,
            ValidationErrorKind::PolicyMalformed {
                message_kind: PolicyMalformedKind::NonSymbolSubsystem,
                ..
            }
        ));
    }

    #[test]
    fn query_non_symbol_metric_is_malformed() {
        let err = pv("(query gpu 42)").unwrap_err();
        assert!(matches!(
            err.kind,
            ValidationErrorKind::PolicyMalformed {
                message_kind: PolicyMalformedKind::NonSymbolOperation,
                ..
            }
        ));
    }

    #[test]
    fn policy_recurses_into_nested_value_arg() {
        // The value-arg is itself an unknown instruction; the
        // structural recursion should surface it.
        let err = pv("(policy gpu allocate-slice (frobnicate))").unwrap_err();
        assert!(matches!(
            err.kind,
            ValidationErrorKind::UnknownInstruction { .. }
        ));
    }

    #[test]
    fn intent_form_no_longer_fails_structure_check() {
        // Pre-12i `(intent ...)` was rejected by validate_structure
        // because `intent` is not in the static INSTRUCTIONS table.
        // 12i normalises the situation by special-casing intent
        // alongside policy/query — the type-checker still owns
        // shape semantics.
        assert!(pv("(intent host-state read)").is_ok());
    }

    // ── Phase 4 Step 4 — `cond` structural validation ───────────

    #[test]
    fn valid_cond_default_only() {
        assert!(pv("(cond (default 0))").is_ok());
    }

    #[test]
    fn valid_cond_two_clause() {
        assert!(pv("(cond ((eq 0 1) 1) (default 0))").is_ok());
    }

    #[test]
    fn valid_cond_three_clause() {
        assert!(pv("(cond ((eq 0 1) 10) ((eq 0 2) 20) (default 30))").is_ok());
    }

    #[test]
    fn valid_cond_with_nested_expressions() {
        assert!(pv("(cond ((eq (add 1 2) 3) (add 10 20)) (default (sub 99 1)))").is_ok());
    }

    #[test]
    fn valid_cond_nested_in_if() {
        assert!(pv("(if true (cond ((eq 1 1) 1) (default 0)) (cond (default 99)))").is_ok());
    }

    #[test]
    fn cond_no_clauses_is_malformed() {
        let err = pv("(cond)").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::CondMalformed {
                message_kind: CondMalformedKind::NoClauses
            }
        );
    }

    #[test]
    fn cond_missing_default_is_malformed() {
        // The single clause is `((eq 0 1) 1)`, no default clause.
        let err = pv("(cond ((eq 0 1) 1))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::CondMalformed {
                message_kind: CondMalformedKind::MissingDefaultClause
            }
        );
    }

    #[test]
    fn cond_missing_default_with_multiple_clauses_is_malformed() {
        let err = pv("(cond ((eq 0 1) 1) ((eq 0 2) 2))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::CondMalformed {
                message_kind: CondMalformedKind::MissingDefaultClause
            }
        );
    }

    #[test]
    fn cond_default_not_last_is_malformed() {
        let err = pv("(cond (default 0) ((eq 1 1) 1))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::CondMalformed {
                message_kind: CondMalformedKind::DefaultNotLast { clause_index: 0 }
            }
        );
    }

    #[test]
    fn cond_clause_not_list_is_malformed() {
        // The first clause is `42`, an atom rather than a list.
        let err = pv("(cond 42 (default 0))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::CondMalformed {
                message_kind: CondMalformedKind::ClauseNotList { clause_index: 0 }
            }
        );
    }

    #[test]
    fn cond_clause_wrong_arity_is_malformed() {
        // First clause has 3 items, not the required 2.
        let err = pv("(cond (true 1 2) (default 0))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::CondMalformed {
                message_kind: CondMalformedKind::ClauseWrongArity {
                    clause_index: 0,
                    actual_items: 3
                }
            }
        );
    }

    #[test]
    fn cond_default_clause_wrong_arity_is_malformed() {
        // The default clause itself has only 1 item.
        let err = pv("(cond (default))").unwrap_err();
        assert_eq!(
            err.kind,
            ValidationErrorKind::CondMalformed {
                message_kind: CondMalformedKind::ClauseWrongArity {
                    clause_index: 0,
                    actual_items: 1
                }
            }
        );
    }

    #[test]
    fn cond_recurses_into_predicate_for_unknown_instruction() {
        // The predicate contains an unknown instruction — should
        // surface as UnknownInstruction, not be silently accepted.
        let err = pv("(cond ((frobnicate 1) 1) (default 0))").unwrap_err();
        assert!(matches!(
            err.kind,
            ValidationErrorKind::UnknownInstruction { .. }
        ));
    }

    #[test]
    fn cond_recurses_into_body_for_unknown_instruction() {
        // The body contains an unknown instruction.
        let err = pv("(cond (true (frobnicate 1)) (default 0))").unwrap_err();
        assert!(matches!(
            err.kind,
            ValidationErrorKind::UnknownInstruction { .. }
        ));
    }

    #[test]
    fn cond_recurses_into_default_body() {
        let err = pv("(cond (default (frobnicate)))").unwrap_err();
        assert!(matches!(
            err.kind,
            ValidationErrorKind::UnknownInstruction { .. }
        ));
    }

    // ── Phase 4 Step 6 — Struct structural validation ──────────

    #[test]
    fn valid_struct_declaration() {
        assert!(pv("(struct point ((x i64) (y i64)))").is_ok());
    }

    #[test]
    fn valid_struct_single_field() {
        assert!(pv("(struct w ((v i64)))").is_ok());
    }

    #[test]
    fn valid_struct_empty_fields() {
        // The structural pass accepts an empty field list; the
        // type-checker decides whether to permit it.
        assert!(pv("(struct unit ())").is_ok());
    }

    #[test]
    fn valid_struct_new_zero_args() {
        // Structural pass only — type checker decides semantic shape.
        assert!(pv("(struct-new unit)").is_ok());
    }

    #[test]
    fn valid_struct_new_two_args() {
        assert!(pv("(struct-new point 3 4)").is_ok());
    }

    #[test]
    fn valid_struct_new_with_nested_expr() {
        assert!(pv("(struct-new point (add 1 2) 4)").is_ok());
    }

    #[test]
    fn valid_struct_get() {
        assert!(pv("(struct-get %0 x)").is_ok());
    }

    #[test]
    fn valid_struct_set() {
        assert!(pv("(struct-set %0 x 42)").is_ok());
    }

    #[test]
    fn struct_new_recurses_into_value_args() {
        // The nested instruction is unknown → structural pass should
        // surface it via the value-arg recursion.
        let err = pv("(struct-new point (frobnicate))").unwrap_err();
        assert!(matches!(
            err.kind,
            ValidationErrorKind::UnknownInstruction { .. }
        ));
    }

    #[test]
    fn struct_get_recurses_into_target_expr() {
        let err = pv("(struct-get (frobnicate) x)").unwrap_err();
        assert!(matches!(
            err.kind,
            ValidationErrorKind::UnknownInstruction { .. }
        ));
    }

    #[test]
    fn struct_set_recurses_into_target_and_value() {
        let err = pv("(struct-set %0 x (frobnicate))").unwrap_err();
        assert!(matches!(
            err.kind,
            ValidationErrorKind::UnknownInstruction { .. }
        ));
    }
}
