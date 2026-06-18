// SPDX-License-Identifier: AGPL-3.0-or-later
//! Function symbol table.
//!
//! Stage 9 Phase-2-Erweiterung: Pass 1 of the interpreter walks the
//! `(program ...)` IR and collects all `(fn ...)` definitions into
//! a symbol table. Pass 2 then evaluates the program body, with
//! `(call name args...)` performing lookups against this table.
//!
//! Stage 12 Paket A.4 (per-sandbox session storage): the table no
//! longer borrows from the input AST. Function bodies are stored as
//! [`alloc::sync::Arc<SExpr>`] so that a [`Session`](crate::Session)
//! can outlive the original `&SExpr` borrow handed to
//! [`Session::new`](crate::Session::new). The Arc clone in
//! [`crate::machine::Instr::InvokeCall`] is cheap (refcount bump) so
//! the lifetime-free design does not regress per-step performance for
//! recursive calls.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;

use quarks_validator::SExpr;

use crate::error::{InterpretError, InterpretErrorKind};

/// A function definition extracted from `(fn name (params...) return-type body)`.
///
/// Stored in the symbol table for `(call name args)` resolution.
pub struct FunctionDef {
    /// Number of parameters (arity). Determines whether `%n` resolves
    /// to a parameter (n < arity) or a local let-binding (n >= arity).
    pub arity: usize,
    /// Function body — the last element of `(fn name (params) return body)`.
    /// Stored as `Arc<SExpr>` so the symbol table owns its definitions
    /// independent of any external borrow.
    pub body: Arc<SExpr>,
}

/// Phase 4 Step 6: struct definition extracted from
/// `(struct name ((field1 type1) (field2 type2) …))`.
///
/// The interpreter needs only the ordered list of field NAMES so it
/// can match positional `(struct-new name v1 v2)` args to field
/// names when constructing the runtime `Value::Struct` map. Field
/// types are owned by the validator; runtime trust comes from the
/// pre-validated IR contract.
#[derive(Debug, Clone)]
pub struct StructDef {
    pub fields: alloc::vec::Vec<String>,
}

/// Function symbol table. Built once per `interpret(program)` call
/// in Pass 1, then queried during Pass 2 for `(call name args)` resolution.
///
/// Phase 4 Step 6 — the table also carries the program's struct
/// declarations so `(struct-new name …)` can look up the field-name
/// order at queue time. Structs and functions share the same Pass 1
/// walk over the `(program …)` body.
pub struct SymbolTable {
    fns: BTreeMap<String, FunctionDef>,
    structs: BTreeMap<String, StructDef>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            fns: BTreeMap::new(),
            structs: BTreeMap::new(),
        }
    }

    pub fn lookup(&self, name: &str) -> Option<&FunctionDef> {
        self.fns.get(name)
    }

    pub fn insert(&mut self, name: String, def: FunctionDef) -> Result<(), InterpretError> {
        if self.fns.contains_key(&name) {
            return Err(InterpretError::new(
                InterpretErrorKind::DuplicateFunction,
                "function defined twice",
            ));
        }
        self.fns.insert(name, def);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn function_count(&self) -> usize {
        self.fns.len()
    }

    /// Phase 4 Step 6 — look up a struct definition by name.
    pub fn lookup_struct(&self, name: &str) -> Option<&StructDef> {
        self.structs.get(name)
    }

    /// Phase 4 Step 6 — register a struct definition. Returns a
    /// `DuplicateFunction` error (reusing the existing variant — the
    /// validator catches the same condition with a richer
    /// `DuplicateStructError` upstream) on collision.
    pub fn insert_struct(&mut self, name: String, def: StructDef) -> Result<(), InterpretError> {
        if self.structs.contains_key(&name) {
            return Err(InterpretError::new(
                InterpretErrorKind::DuplicateFunction,
                "struct defined twice",
            ));
        }
        self.structs.insert(name, def);
        Ok(())
    }
}

/// Collect all `(fn ...)` definitions from a `(program ...)` IR tree.
///
/// Pass 1 of the interpreter — analogous to type_checker's
/// `collect_signature` pass.
///
/// # Format requirement
///
/// Each `(fn ...)` must be a 5-element list:
///   `(fn name params-list return-type body)`
///
/// Where:
/// - `name` is a Symbol atom
/// - `params-list` is a List of type Symbols (i64, bytes, handle)
/// - `return-type` is a Symbol atom
/// - `body` is any SExpr
///
/// This format is enforced by the validator's type-checker;
/// the interpreter trusts pre-validated IR.
pub fn collect_signatures(ast: &SExpr) -> Result<SymbolTable, InterpretError> {
    let mut table = SymbolTable::new();

    let items = match ast {
        SExpr::List(items) => items,
        _ => {
            // Bare expression (no program wrapper) — no fn definitions to collect.
            return Ok(table);
        }
    };

    // Check for (program ...) wrapper
    let is_program = matches!(
        items.first(),
        Some(SExpr::Atom(quarks_validator::Atom::Symbol(s))) if s == "program"
    );

    if !is_program {
        // Bare expression — no fn definitions
        return Ok(table);
    }

    // Walk program children, collect fn and struct definitions.
    for child in &items[1..] {
        if let SExpr::List(child_items) = child {
            if let Some(SExpr::Atom(quarks_validator::Atom::Symbol(s))) = child_items.first() {
                if s == "fn" {
                    let (name, def) = parse_fn_definition(child_items)?;
                    table.insert(name, def)?;
                } else if s == "struct" {
                    let (name, def) = parse_struct_definition(child_items)?;
                    table.insert_struct(name, def)?;
                }
            }
        }
    }

    Ok(table)
}

/// Phase 4 Step 6 — parse `(struct name ((f1 t1) (f2 t2) …))` into
/// an ordered list of field names. The interpreter trusts pre-
/// validated IR; structural shape (3-elem outer list, list-of-pairs
/// fields, symbol names) is checked defensively to keep the runtime
/// panic-free in Ring-0 even on adversarial IR.
fn parse_struct_definition(items: &[SExpr]) -> Result<(String, StructDef), InterpretError> {
    if items.len() != 3 {
        return Err(InterpretError::new(
            InterpretErrorKind::MalformedFunction,
            "struct definition must be 3 elements: (struct name fields)",
        ));
    }
    let name = match &items[1] {
        SExpr::Atom(quarks_validator::Atom::Symbol(s)) => s.clone(),
        _ => {
            return Err(InterpretError::new(
                InterpretErrorKind::MalformedFunction,
                "struct name must be a symbol",
            ));
        }
    };
    let field_items = match &items[2] {
        SExpr::List(l) => l,
        _ => {
            return Err(InterpretError::new(
                InterpretErrorKind::MalformedFunction,
                "struct field list must be a list",
            ));
        }
    };
    let mut fields: alloc::vec::Vec<String> = alloc::vec::Vec::with_capacity(field_items.len());
    for clause in field_items.iter() {
        let clause_items = match clause {
            SExpr::List(l) => l,
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::MalformedFunction,
                    "struct field clause must be a (name type) list",
                ));
            }
        };
        if clause_items.len() != 2 {
            return Err(InterpretError::new(
                InterpretErrorKind::MalformedFunction,
                "struct field clause must be exactly (name type)",
            ));
        }
        let fname = match &clause_items[0] {
            SExpr::Atom(quarks_validator::Atom::Symbol(s)) => s.clone(),
            _ => {
                return Err(InterpretError::new(
                    InterpretErrorKind::MalformedFunction,
                    "struct field name must be a symbol",
                ));
            }
        };
        fields.push(fname);
    }
    Ok((name, StructDef { fields }))
}

/// Parse a single `(fn name params return body)` definition.
fn parse_fn_definition(items: &[SExpr]) -> Result<(String, FunctionDef), InterpretError> {
    if items.len() != 5 {
        return Err(InterpretError::new(
            InterpretErrorKind::MalformedFunction,
            "fn definition must be 5 elements: (fn name params return body)",
        ));
    }

    let name = match &items[1] {
        SExpr::Atom(quarks_validator::Atom::Symbol(s)) => s.clone(),
        _ => {
            return Err(InterpretError::new(
                InterpretErrorKind::MalformedFunction,
                "fn name must be a symbol",
            ));
        }
    };

    let arity = match &items[2] {
        SExpr::List(params) => params.len(),
        _ => {
            return Err(InterpretError::new(
                InterpretErrorKind::MalformedFunction,
                "fn params must be a list",
            ));
        }
    };

    let body = Arc::new(items[4].clone());

    Ok((name, FunctionDef { arity, body }))
}
