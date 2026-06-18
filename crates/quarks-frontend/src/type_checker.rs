// SPDX-License-Identifier: AGPL-3.0-or-later
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::ast::*;
use crate::lexer::Span;

// ── Error types ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub span: Span,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeErrorKind {
    /// Expression has wrong type for its context.
    TypeMismatch { expected: Type, actual: Type },
    /// Reference to a name that isn't in scope.
    UndefinedIdentifier { name: String },
    /// Call to a function that doesn't exist.
    UndefinedFunction { name: String },
    /// Call with wrong number of arguments.
    ArityMismatch {
        function_name: String,
        expected: u32,
        actual: u32,
    },
    /// Two functions with the same name defined in the program.
    DuplicateFunction { name: String },
    /// Break statement outside of a loop.
    BreakOutsideLoop,
    /// Function body doesn't return on all paths.
    MissingReturn { function_name: String },
    /// Intent call has wrong argument shape.
    InvalidIntentCall,
}

// ── Function table ─────────────────────────────────────────────

/// Type signature of a function, collected in Pass 1.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    pub name: String,
    pub param_types: Vec<Type>,
    pub param_names: Vec<String>,
    pub return_type: Type,
    pub arity: u32,
    pub span: Span,
}

#[derive(Debug, Default)]
struct FunctionTable {
    entries: BTreeMap<String, FunctionSignature>,
}

impl FunctionTable {
    fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    fn insert(&mut self, sig: FunctionSignature) -> Result<(), TypeError> {
        if self.entries.contains_key(&sig.name) {
            return Err(TypeError {
                kind: TypeErrorKind::DuplicateFunction {
                    name: sig.name.clone(),
                },
                span: sig.span,
                message: format!("function '{}' is defined more than once", sig.name),
            });
        }
        self.entries.insert(sig.name.clone(), sig);
        Ok(())
    }

    fn get(&self, name: &str) -> Option<&FunctionSignature> {
        self.entries.get(name)
    }
}

// ── Scope stack ────────────────────────────────────────────────

#[derive(Debug, Default)]
struct ScopeStack {
    scopes: Vec<BTreeMap<String, Type>>,
}

impl ScopeStack {
    fn new() -> Self {
        Self { scopes: Vec::new() }
    }

    fn push_scope(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Bind in innermost scope. Overwrites if name exists in same scope
    /// (shadowing).
    fn bind(&mut self, name: String, ty: Type) {
        if let Some(top) = self.scopes.last_mut() {
            top.insert(name, ty);
        }
    }

    /// Look up walking innermost-to-outermost.
    fn lookup(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(*ty);
            }
        }
        None
    }
}

// ── Public API ─────────────────────────────────────────────────

/// Type-check a parsed program.
///
/// Two-pass architecture:
/// - Pass 1: Collect all function signatures into a FunctionTable.
/// - Pass 2: Type-check each function body against the table.
///
/// Fail-fast: the first TypeError stops checking.
pub fn check(program: &Program) -> Result<(), TypeError> {
    // Pass 1: collect signatures
    let mut fn_table = FunctionTable::new();
    for func in &program.functions {
        let sig = build_signature(func);
        fn_table.insert(sig)?;
    }

    // Pass 2: type-check bodies
    for func in &program.functions {
        check_function(func, &fn_table)?;
    }

    Ok(())
}

fn build_signature(func: &FunctionDef) -> FunctionSignature {
    let param_types: Vec<Type> = func.params.iter().map(|p| p.ty).collect();
    let param_names: Vec<String> = func.params.iter().map(|p| p.name.clone()).collect();
    FunctionSignature {
        name: func.name.clone(),
        arity: param_types.len() as u32,
        param_types,
        param_names,
        return_type: func.return_type,
        span: func.span,
    }
}

// ── Function body check ────────────────────────────────────────

fn check_function(func: &FunctionDef, fn_table: &FunctionTable) -> Result<(), TypeError> {
    let mut scopes = ScopeStack::new();
    scopes.push_scope();

    // Bind parameters
    for param in &func.params {
        scopes.bind(param.name.clone(), param.ty);
    }

    // Type-check the body
    check_block(&func.body, &func.return_type, fn_table, &mut scopes, false)?;

    // Return-coverage analysis
    if !body_returns(&func.body) {
        return Err(TypeError {
            kind: TypeErrorKind::MissingReturn {
                function_name: func.name.clone(),
            },
            span: func.body.span,
            message: format!("function '{}' does not return on all paths", func.name),
        });
    }

    scopes.pop_scope();
    Ok(())
}

// ── Block and statement check ──────────────────────────────────

fn check_block(
    block: &Block,
    return_type: &Type,
    fn_table: &FunctionTable,
    scopes: &mut ScopeStack,
    in_loop: bool,
) -> Result<(), TypeError> {
    scopes.push_scope();
    for stmt in &block.statements {
        check_statement(stmt, return_type, fn_table, scopes, in_loop)?;
    }
    scopes.pop_scope();
    Ok(())
}

fn check_statement(
    stmt: &Statement,
    return_type: &Type,
    fn_table: &FunctionTable,
    scopes: &mut ScopeStack,
    in_loop: bool,
) -> Result<(), TypeError> {
    match stmt {
        Statement::Let { name, value, .. } => {
            let ty = infer_expression(value, fn_table, scopes)?;
            scopes.bind(name.clone(), ty);
            Ok(())
        }
        Statement::Return { value, span } => {
            let ty = infer_expression(value, fn_table, scopes)?;
            if ty != *return_type {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: *return_type,
                        actual: ty,
                    },
                    span: *span,
                    message: format!(
                        "return type mismatch: expected {:?}, got {:?}",
                        return_type, ty
                    ),
                });
            }
            Ok(())
        }
        Statement::If {
            condition,
            then_block,
            else_block,
            span,
        } => {
            let cond_ty = infer_expression(condition, fn_table, scopes)?;
            if cond_ty != Type::I64 {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Type::I64,
                        actual: cond_ty,
                    },
                    span: *span,
                    message: format!("if condition must be i64, got {:?}", cond_ty),
                });
            }
            check_block(then_block, return_type, fn_table, scopes, in_loop)?;
            if let Some(else_b) = else_block {
                check_block(else_b, return_type, fn_table, scopes, in_loop)?;
            }
            Ok(())
        }
        Statement::Loop { body, .. } => {
            check_block(body, return_type, fn_table, scopes, true)?;
            Ok(())
        }
        Statement::Break { span } => {
            if !in_loop {
                return Err(TypeError {
                    kind: TypeErrorKind::BreakOutsideLoop,
                    span: *span,
                    message: String::from("break used outside of a loop"),
                });
            }
            Ok(())
        }
        Statement::Expr { expression, .. } => {
            // Type-check the expression, discard the result type.
            let _ty = infer_expression(expression, fn_table, scopes)?;
            Ok(())
        }
    }
}

// ── Expression inference ───────────────────────────────────────

fn infer_expression(
    expr: &Expression,
    fn_table: &FunctionTable,
    scopes: &mut ScopeStack,
) -> Result<Type, TypeError> {
    match expr {
        Expression::IntegerLiteral { .. } => Ok(Type::I64),
        Expression::BytesLiteral { .. } => Ok(Type::Bytes),
        Expression::HandleLiteral { .. } => Ok(Type::Handle),

        Expression::Identifier { name, span } => match scopes.lookup(name) {
            Some(ty) => Ok(ty),
            None => Err(TypeError {
                kind: TypeErrorKind::UndefinedIdentifier { name: name.clone() },
                span: *span,
                message: format!("undefined identifier '{}'", name),
            }),
        },

        Expression::UnaryOp { op, operand, span } => {
            let operand_ty = infer_expression(operand, fn_table, scopes)?;
            match op {
                UnaryOperator::Negate => {
                    if operand_ty != Type::I64 {
                        return Err(TypeError {
                            kind: TypeErrorKind::TypeMismatch {
                                expected: Type::I64,
                                actual: operand_ty,
                            },
                            span: *span,
                            message: format!("negation requires i64 operand, got {:?}", operand_ty),
                        });
                    }
                    Ok(Type::I64)
                }
            }
        }

        Expression::BinaryOp {
            op: _,
            lhs,
            rhs,
            span,
        } => {
            let lhs_ty = infer_expression(lhs, fn_table, scopes)?;
            let rhs_ty = infer_expression(rhs, fn_table, scopes)?;
            // All binary operators expect i64 × i64 → i64
            if lhs_ty != Type::I64 {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Type::I64,
                        actual: lhs_ty,
                    },
                    span: *span,
                    message: format!("binary operator left operand must be i64, got {:?}", lhs_ty),
                });
            }
            if rhs_ty != Type::I64 {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Type::I64,
                        actual: rhs_ty,
                    },
                    span: *span,
                    message: format!(
                        "binary operator right operand must be i64, got {:?}",
                        rhs_ty
                    ),
                });
            }
            Ok(Type::I64)
        }

        Expression::Call {
            function,
            args,
            span,
        } => {
            let sig = match fn_table.get(function) {
                Some(s) => s,
                None => {
                    return Err(TypeError {
                        kind: TypeErrorKind::UndefinedFunction {
                            name: function.clone(),
                        },
                        span: *span,
                        message: format!("undefined function '{}'", function),
                    });
                }
            };

            // Arity check
            let actual_arity = args.len() as u32;
            if actual_arity != sig.arity {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch {
                        function_name: function.clone(),
                        expected: sig.arity,
                        actual: actual_arity,
                    },
                    span: *span,
                    message: format!(
                        "function '{}' expects {} argument(s), got {}",
                        function, sig.arity, actual_arity
                    ),
                });
            }

            // Arg type check
            for (i, arg) in args.iter().enumerate() {
                let arg_ty = infer_expression(arg, fn_table, scopes)?;
                if arg_ty != sig.param_types[i] {
                    return Err(TypeError {
                        kind: TypeErrorKind::TypeMismatch {
                            expected: sig.param_types[i],
                            actual: arg_ty,
                        },
                        span: arg.span(),
                        message: format!(
                            "argument {} of '{}': expected {:?}, got {:?}",
                            i, function, sig.param_types[i], arg_ty
                        ),
                    });
                }
            }

            Ok(sig.return_type)
        }

        Expression::IntentCall { args, span } => {
            // Intent requires ≥1 arg, first must be Handle
            if args.is_empty() {
                return Err(TypeError {
                    kind: TypeErrorKind::InvalidIntentCall,
                    span: *span,
                    message: String::from("intent requires at least one argument (target handle)"),
                });
            }

            let target_ty = infer_expression(&args[0], fn_table, scopes)?;
            if target_ty != Type::Handle {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Type::Handle,
                        actual: target_ty,
                    },
                    span: args[0].span(),
                    message: format!("intent target must be handle, got {:?}", target_ty),
                });
            }

            // Type-check remaining args (any type is valid)
            for arg in &args[1..] {
                let _ty = infer_expression(arg, fn_table, scopes)?;
            }

            Ok(Type::I64)
        }
    }
}

// ── Return-coverage analysis ───────────────────────────────────

/// Does the block definitely return on all code paths?
fn body_returns(block: &Block) -> bool {
    for stmt in &block.statements {
        if statement_always_returns(stmt) {
            return true;
        }
    }
    false
}

fn statement_always_returns(stmt: &Statement) -> bool {
    match stmt {
        Statement::Return { .. } => true,
        Statement::If {
            then_block,
            else_block,
            ..
        } => match else_block {
            Some(else_b) => body_returns(then_block) && body_returns(else_b),
            None => false,
        },
        // Infinite loop (no break) always returns (or runs forever).
        Statement::Loop { body, .. } => !block_contains_break(body),
        Statement::Break { .. } => false,
        Statement::Let { .. } | Statement::Expr { .. } => false,
    }
}

/// Does the block contain a break targeted at the immediately enclosing loop?
fn block_contains_break(block: &Block) -> bool {
    for stmt in &block.statements {
        if statement_contains_break_to_this_loop(stmt) {
            return true;
        }
    }
    false
}

fn statement_contains_break_to_this_loop(stmt: &Statement) -> bool {
    match stmt {
        Statement::Break { .. } => true,
        Statement::If {
            then_block,
            else_block,
            ..
        } => {
            block_contains_break(then_block)
                || else_block.as_ref().map_or(false, block_contains_break)
        }
        // Nested loop's break is theirs, not ours
        Statement::Loop { .. } => false,
        Statement::Return { .. } => false,
        Statement::Let { .. } | Statement::Expr { .. } => false,
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::parse;

    fn check_source(src: &str) -> Result<(), TypeError> {
        let tokens = tokenize(src).expect("lex should succeed");
        let program = parse(&tokens).expect("parse should succeed");
        check(&program)
    }

    fn expect_ok(src: &str) {
        check_source(src).expect("type check should succeed");
    }

    fn expect_type_error(src: &str) -> TypeError {
        check_source(src).expect_err("should produce type error")
    }

    // ── A: Happy-path ──────────────────────────────────────────

    #[test]
    fn a01_minimal_function() {
        expect_ok("fn f() -> i64 { return 0; }");
    }

    #[test]
    fn a02_function_with_params() {
        expect_ok("fn add(a: i64, b: i64) -> i64 { return a + b; }");
    }

    #[test]
    fn a03_let_infers_type() {
        expect_ok("fn f() -> i64 { let x = 42; return x; }");
    }

    #[test]
    fn a04_nested_arithmetic() {
        expect_ok("fn f() -> i64 { return (1 + 2) * (3 - 4) / 5; }");
    }

    #[test]
    fn a05_comparison_as_i64() {
        expect_ok("fn f() -> i64 { return 1 == 2; }");
    }

    #[test]
    fn a06_handle_literal() {
        expect_ok("fn f() -> handle { return @1; }");
    }

    #[test]
    fn a07_bytes_literal() {
        expect_ok("fn f() -> bytes { return #x00ff; }");
    }

    // ── B: Type-Mismatches ─────────────────────────────────────

    #[test]
    fn b01_return_type_mismatch() {
        let err = expect_type_error("fn f() -> i64 { return @1; }");
        assert!(matches!(
            err.kind,
            TypeErrorKind::TypeMismatch {
                expected: Type::I64,
                actual: Type::Handle,
            }
        ));
    }

    #[test]
    fn b02_arithmetic_on_handle() {
        let err = expect_type_error("fn f() -> i64 { return @1 + @2; }");
        assert!(matches!(
            err.kind,
            TypeErrorKind::TypeMismatch {
                expected: Type::I64,
                ..
            }
        ));
    }

    #[test]
    fn b03_if_condition_non_i64() {
        let err = expect_type_error("fn f() -> i64 { if @1 { return 0; } return 1; }");
        assert!(matches!(
            err.kind,
            TypeErrorKind::TypeMismatch {
                expected: Type::I64,
                actual: Type::Handle,
            }
        ));
    }

    #[test]
    fn b04_negation_on_handle() {
        let err = expect_type_error("fn f() -> i64 { return -@1; }");
        assert!(matches!(
            err.kind,
            TypeErrorKind::TypeMismatch {
                expected: Type::I64,
                actual: Type::Handle,
            }
        ));
    }

    // ── C: Name-Resolution ─────────────────────────────────────

    #[test]
    fn c01_undefined_identifier() {
        let err = expect_type_error("fn f() -> i64 { return x; }");
        assert!(matches!(
            err.kind,
            TypeErrorKind::UndefinedIdentifier { .. }
        ));
    }

    #[test]
    fn c02_parameter_in_scope() {
        expect_ok("fn f(x: i64) -> i64 { return x; }");
    }

    #[test]
    fn c03_let_binding_in_scope() {
        expect_ok("fn f() -> i64 { let x = 5; return x; }");
    }

    #[test]
    fn c04_shadowing_same_scope() {
        expect_ok("fn f() -> i64 { let x = 5; let x = 10; return x; }");
    }

    #[test]
    fn c05_shadowing_different_type() {
        expect_ok("fn f() -> handle { let x = 5; let x = @1; return x; }");
    }

    #[test]
    fn c06_inner_scope_does_not_leak() {
        // x is defined in the if-block scope, not accessible outside
        let err = expect_type_error("fn f() -> i64 { if 1 { let x = 5; } return x; }");
        assert!(matches!(
            err.kind,
            TypeErrorKind::UndefinedIdentifier { .. }
        ));
    }

    // ── D: Function-Calls ──────────────────────────────────────

    #[test]
    fn d01_simple_call() {
        expect_ok("fn g() -> i64 { return 42; } fn f() -> i64 { return g(); }");
    }

    #[test]
    fn d02_undefined_function() {
        let err = expect_type_error("fn f() -> i64 { return g(); }");
        assert!(matches!(err.kind, TypeErrorKind::UndefinedFunction { .. }));
    }

    #[test]
    fn d03_wrong_arity() {
        let err = expect_type_error(
            "fn g(a: i64) -> i64 { return a; } fn f() -> i64 { return g(1, 2); }",
        );
        assert!(matches!(err.kind, TypeErrorKind::ArityMismatch { .. }));
    }

    #[test]
    fn d04_wrong_arg_type() {
        let err =
            expect_type_error("fn g(a: i64) -> i64 { return a; } fn f() -> i64 { return g(@1); }");
        assert!(matches!(err.kind, TypeErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn d05_mutually_recursive() {
        expect_ok("fn a() -> i64 { return b(); } fn b() -> i64 { return a(); }");
    }

    #[test]
    fn d06_duplicate_function() {
        let err = expect_type_error("fn f() -> i64 { return 0; } fn f() -> i64 { return 1; }");
        assert!(matches!(err.kind, TypeErrorKind::DuplicateFunction { .. }));
    }

    // ── E: Return-Coverage ─────────────────────────────────────

    #[test]
    fn e01_empty_body_errors() {
        let err = expect_type_error("fn f() -> i64 { }");
        assert!(matches!(err.kind, TypeErrorKind::MissingReturn { .. }));
    }

    #[test]
    fn e02_if_without_else_needs_trailing_return() {
        expect_ok("fn f() -> i64 { if 1 { return 0; } return 1; }");
    }

    #[test]
    fn e03_both_if_branches_return() {
        expect_ok("fn f() -> i64 { if 1 { return 0; } else { return 1; } }");
    }

    #[test]
    fn e04_if_without_else_no_trailing_return_errors() {
        let err = expect_type_error("fn f() -> i64 { if 1 { return 0; } }");
        assert!(matches!(err.kind, TypeErrorKind::MissingReturn { .. }));
    }

    #[test]
    fn e05_infinite_loop_ok() {
        // Infinite loop (no break) counts as "always returns"
        expect_ok("fn f() -> i64 { loop { } }");
    }

    #[test]
    fn e06_loop_with_break_needs_trailing_return() {
        expect_ok("fn f() -> i64 { loop { break; } return 0; }");
    }

    #[test]
    fn e07_loop_with_break_no_trailing_return_errors() {
        let err = expect_type_error("fn f() -> i64 { loop { break; } }");
        assert!(matches!(err.kind, TypeErrorKind::MissingReturn { .. }));
    }

    // ── F: Break-Outside-Loop ──────────────────────────────────

    #[test]
    fn f01_break_outside_loop() {
        let err = expect_type_error("fn f() -> i64 { break; return 0; }");
        assert!(matches!(err.kind, TypeErrorKind::BreakOutsideLoop));
    }

    #[test]
    fn f02_break_inside_nested_loop_ok() {
        expect_ok("fn f() -> i64 { loop { loop { break; } } }");
    }

    // ── G: Intent-Calls ────────────────────────────────────────

    #[test]
    fn g01_intent_with_handle_ok() {
        expect_ok("fn f() -> i64 { return intent(@1, 42); }");
    }

    #[test]
    fn g02_intent_without_args_errors() {
        let err = expect_type_error("fn f() -> i64 { return intent(); }");
        assert!(matches!(err.kind, TypeErrorKind::InvalidIntentCall));
    }

    #[test]
    fn g03_intent_non_handle_target_errors() {
        let err = expect_type_error("fn f() -> i64 { return intent(42); }");
        assert!(matches!(
            err.kind,
            TypeErrorKind::TypeMismatch {
                expected: Type::Handle,
                actual: Type::I64,
            }
        ));
    }

    // ── H: Edge-Cases ──────────────────────────────────────────

    #[test]
    fn h01_all_comparison_operators() {
        expect_ok("fn f() -> i64 { return (1 < 2) + (3 > 4) + (5 <= 6) + (7 >= 8) + (9 != 10); }");
    }

    #[test]
    fn h02_deeply_nested_if_returns() {
        expect_ok(
            "fn f() -> i64 { if 1 { if 2 { return 3; } else { return 4; } } else { return 5; } }",
        );
    }

    #[test]
    fn h03_let_used_in_arithmetic() {
        expect_ok("fn f(n: i64) -> i64 { let x = n + 1; let y = x * 2; return y; }");
    }

    #[test]
    fn h04_multiple_functions_different_types() {
        expect_ok(
            "fn get_handle() -> handle { return @1; } fn use_handle() -> i64 { let h = get_handle(); return intent(h, 42); }"
        );
    }

    #[test]
    fn h05_loop_with_conditional_break() {
        // Loop has break in an if-branch → not infinite → needs trailing return
        expect_ok("fn f() -> i64 { loop { if 1 { break; } } return 0; }");
    }
}
