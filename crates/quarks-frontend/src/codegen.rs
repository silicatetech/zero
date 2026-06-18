// SPDX-License-Identifier: AGPL-3.0-or-later
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use quarks_validator::ast::{Atom, SExpr};

use crate::ast::*;

// ── Public API ─────────────────────────────────────────────────

/// Compile a typed source program to S-expression IR.
///
/// Pre-condition: `program` has been successfully type-checked via
/// `type_checker::check`.
///
/// `main_name` specifies which function serves as the entry point.
/// The named function must exist and must have zero parameters.
pub fn compile(program: &Program, main_name: &str) -> Result<SExpr, CodegenError> {
    // Verify entry point pre-conditions
    let main_fn = program
        .functions
        .iter()
        .find(|f| f.name == main_name)
        .ok_or_else(|| CodegenError::EntryPointNotFound {
            name: String::from(main_name),
        })?;

    if !main_fn.params.is_empty() {
        return Err(CodegenError::EntryPointHasParameters {
            name: String::from(main_name),
            param_count: main_fn.params.len(),
        });
    }

    // Build (program (fn ...) (fn ...) ... (call main_name))
    let mut items: Vec<SExpr> = Vec::new();
    items.push(SExpr::Atom(Atom::Symbol(String::from("program"))));

    for func in &program.functions {
        items.push(compile_function(func)?);
    }

    items.push(SExpr::List(vec![
        SExpr::Atom(Atom::Symbol(String::from("call"))),
        SExpr::Atom(Atom::Symbol(String::from(main_name))),
    ]));

    Ok(SExpr::List(items))
}

#[derive(Debug, Clone, PartialEq)]
pub enum CodegenError {
    EntryPointNotFound { name: String },
    EntryPointHasParameters { name: String, param_count: usize },
    UnresolvedName { name: String },
    EmptyBlock,
    StatementsExhausted,
    LetAsLast { name: String },
}

// ── Compiler Context ───────────────────────────────────────────

struct CompilerContext {
    /// Function parameters: name → 0-based index.
    params_by_name: BTreeMap<String, u32>,

    /// Let-bound names: name → IR parameter index (>= arity).
    locals_by_name: BTreeMap<String, u32>,

    /// Next available local index. Monotonic — never reset.
    next_local_index: u32,
}

impl CompilerContext {
    /// Look up a name. Parameters first, then locals.
    ///
    /// Returns `CodegenError::UnresolvedName` rather than panicking
    /// — the codegen must remain defensive against future
    /// type-checker bugs or mis-ordered pipeline calls (a panic
    /// in a Ring-0 kernel context is a hard crash).
    fn resolve(&self, name: &str) -> Result<u32, CodegenError> {
        if let Some(&idx) = self.params_by_name.get(name) {
            return Ok(idx);
        }
        if let Some(&idx) = self.locals_by_name.get(name) {
            return Ok(idx);
        }
        Err(CodegenError::UnresolvedName {
            name: String::from(name),
        })
    }

    /// Allocate a new local with a given name (only for source-level let bindings).
    fn allocate_local(&mut self, name: String) -> u32 {
        let idx = self.next_local_index;
        self.next_local_index += 1;
        self.locals_by_name.insert(name, idx);
        idx
    }

    /// Snapshot the current locals_by_name (for block-scope restore).
    fn snapshot_locals(&self) -> BTreeMap<String, u32> {
        self.locals_by_name.clone()
    }

    /// Restore locals_by_name. next_local_index NOT reverted.
    fn restore_locals(&mut self, snapshot: BTreeMap<String, u32>) {
        self.locals_by_name = snapshot;
    }
}

// ── Function Compilation ───────────────────────────────────────

fn compile_function(func: &FunctionDef) -> Result<SExpr, CodegenError> {
    let name_atom = SExpr::Atom(Atom::Symbol(func.name.clone()));

    let param_types: Vec<SExpr> = func
        .params
        .iter()
        .map(|p| SExpr::Atom(Atom::Symbol(type_to_ir_name(p.ty))))
        .collect();
    let params_list = SExpr::List(param_types);

    let return_type = SExpr::Atom(Atom::Symbol(type_to_ir_name(func.return_type)));

    let arity = func.params.len() as u32;
    let mut ctx = CompilerContext {
        params_by_name: func
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.clone(), i as u32))
            .collect(),
        locals_by_name: BTreeMap::new(),
        next_local_index: arity,
    };

    let body_sexpr = compile_block(&func.body, &mut ctx)?;

    Ok(SExpr::List(vec![
        SExpr::Atom(Atom::Symbol(String::from("fn"))),
        name_atom,
        params_list,
        return_type,
        body_sexpr,
    ]))
}

fn type_to_ir_name(ty: Type) -> String {
    match ty {
        Type::I64 => String::from("i64"),
        Type::Bytes => String::from("bytes"),
        Type::Handle => String::from("handle"),
    }
}

// ── Block Compilation ──────────────────────────────────────────

fn compile_block(block: &Block, ctx: &mut CompilerContext) -> Result<SExpr, CodegenError> {
    if block.statements.is_empty() {
        return Err(CodegenError::EmptyBlock);
    }

    let snapshot = ctx.snapshot_locals();
    let result = compile_statements(&block.statements, 0, ctx);
    ctx.restore_locals(snapshot);
    result
}

fn compile_statements(
    statements: &[Statement],
    index: usize,
    ctx: &mut CompilerContext,
) -> Result<SExpr, CodegenError> {
    if index >= statements.len() {
        return Err(CodegenError::StatementsExhausted);
    }

    let stmt = &statements[index];
    let is_last = index == statements.len() - 1;

    match stmt {
        Statement::Let { name, value, .. } => {
            let expr_sexpr = compile_expression(value, ctx)?;
            let local_idx = ctx.allocate_local(name.clone());

            if is_last {
                return Err(CodegenError::LetAsLast { name: name.clone() });
            }
            let body_sexpr = compile_statements(statements, index + 1, ctx)?;

            Ok(SExpr::List(vec![
                SExpr::Atom(Atom::Symbol(String::from("let"))),
                SExpr::Atom(Atom::Parameter(local_idx)),
                expr_sexpr,
                body_sexpr,
            ]))
        }

        Statement::Return { value, .. } => {
            // return must be at end of its control-flow path.
            // Statements after return are unreachable; codegen ignores them.
            Ok(SExpr::List(vec![
                SExpr::Atom(Atom::Symbol(String::from("return"))),
                compile_expression(value, ctx)?,
            ]))
        }

        Statement::If {
            condition,
            then_block,
            else_block,
            ..
        } => {
            let cond_sexpr = compile_bool_cond(condition, ctx)?;
            let then_sexpr = compile_block(then_block, ctx)?;

            let else_sexpr = match else_block {
                Some(eb) => compile_block(eb, ctx)?,
                None => {
                    // No explicit else — flatten trailing statements into else.
                    if is_last {
                        // If-without-else as last statement (e.g. in a loop body).
                        // Emit (nop) as stack-neutral no-op for else branch.
                        SExpr::List(vec![SExpr::Atom(Atom::Symbol(String::from("nop")))])
                    } else {
                        compile_statements(statements, index + 1, ctx)?
                    }
                }
            };

            let if_expr = SExpr::List(vec![
                SExpr::Atom(Atom::Symbol(String::from("if"))),
                cond_sexpr,
                then_sexpr,
                else_sexpr,
            ]);

            // If had explicit else AND there are more statements after:
            // wrap if's value in (seq (discard if-expr) rest).
            if is_last || else_block.is_none() {
                Ok(if_expr)
            } else {
                let rest = compile_statements(statements, index + 1, ctx)?;
                Ok(SExpr::List(vec![
                    SExpr::Atom(Atom::Symbol(String::from("seq"))),
                    SExpr::List(vec![
                        SExpr::Atom(Atom::Symbol(String::from("discard"))),
                        if_expr,
                    ]),
                    rest,
                ]))
            }
        }

        Statement::Loop { body, .. } => {
            let body_sexpr = compile_block(body, ctx)?;
            let loop_expr = SExpr::List(vec![
                SExpr::Atom(Atom::Symbol(String::from("loop"))),
                body_sexpr,
            ]);

            if is_last {
                Ok(loop_expr)
            } else {
                // Paket B.1c: loop now produces the break-value
                // type (was stack-neutral). When the loop is in
                // a `seq` effect position it must be wrapped in
                // `(discard ...)` to remain stack-neutral.
                let rest = compile_statements(statements, index + 1, ctx)?;
                Ok(SExpr::List(vec![
                    SExpr::Atom(Atom::Symbol(String::from("seq"))),
                    SExpr::List(vec![
                        SExpr::Atom(Atom::Symbol(String::from("discard"))),
                        loop_expr,
                    ]),
                    rest,
                ]))
            }
        }

        Statement::Break { .. } => {
            // Paket B.1c: break carries a value; the source
            // language's nullary `break;` desugars to `(break
            // false)`, giving the surrounding loop a Bool exit
            // type. Statements after break are unreachable.
            Ok(SExpr::List(vec![
                SExpr::Atom(Atom::Symbol(String::from("break"))),
                SExpr::Atom(Atom::Symbol(String::from("false"))),
            ]))
        }

        Statement::Expr { expression, .. } => {
            let expr_sexpr = compile_expression(expression, ctx)?;
            if is_last {
                // Expression as last statement: its value is the block value.
                Ok(expr_sexpr)
            } else {
                // Expression as effect statement: wrap in (seq (discard expr) rest)
                let rest = compile_statements(statements, index + 1, ctx)?;
                Ok(SExpr::List(vec![
                    SExpr::Atom(Atom::Symbol(String::from("seq"))),
                    SExpr::List(vec![
                        SExpr::Atom(Atom::Symbol(String::from("discard"))),
                        expr_sexpr,
                    ]),
                    rest,
                ]))
            }
        }
    }
}

// ── Expression Compilation ─────────────────────────────────────

fn compile_expression(expr: &Expression, ctx: &mut CompilerContext) -> Result<SExpr, CodegenError> {
    match expr {
        Expression::IntegerLiteral { value, .. } => Ok(SExpr::Atom(Atom::Integer(*value))),
        Expression::BytesLiteral { value, .. } => Ok(SExpr::Atom(Atom::Bytes(value.clone()))),
        Expression::HandleLiteral { value, .. } => Ok(SExpr::Atom(Atom::Handle(*value))),
        Expression::Identifier { name, .. } => {
            let idx = ctx.resolve(name)?;
            Ok(SExpr::Atom(Atom::Parameter(idx)))
        }
        Expression::UnaryOp { op, operand, .. } => {
            let operand_sexpr = compile_expression(operand, ctx)?;
            match op {
                UnaryOperator::Negate => {
                    // -x → (sub 0 x)
                    Ok(SExpr::List(vec![
                        SExpr::Atom(Atom::Symbol(String::from("sub"))),
                        SExpr::Atom(Atom::Integer(0)),
                        operand_sexpr,
                    ]))
                }
            }
        }
        Expression::BinaryOp { op, lhs, rhs, .. } => {
            let op_name = binary_op_to_ir_name(*op);
            Ok(SExpr::List(vec![
                SExpr::Atom(Atom::Symbol(String::from(op_name))),
                compile_expression(lhs, ctx)?,
                compile_expression(rhs, ctx)?,
            ]))
        }
        Expression::Call { function, args, .. } => {
            let mut items = vec![
                SExpr::Atom(Atom::Symbol(String::from("call"))),
                SExpr::Atom(Atom::Symbol(function.clone())),
            ];
            for arg in args {
                items.push(compile_expression(arg, ctx)?);
            }
            Ok(SExpr::List(items))
        }
        Expression::IntentCall { args, .. } => {
            let mut items = vec![SExpr::Atom(Atom::Symbol(String::from("intent")))];
            for arg in args {
                items.push(compile_expression(arg, ctx)?);
            }
            Ok(SExpr::List(items))
        }
    }
}

/// Paket B.1b: convert a source-level (I64-valued) condition into
/// an IR-level Bool. Comparisons already produce Bool in the IR
/// (Paket B.1) so they pass through unchanged; every other
/// expression gets wrapped in `(ne <cond> 0)`, which converts an
/// I64 zero/non-zero pattern into the Bool the validator now
/// requires for `if`/`while` conditions.
fn compile_bool_cond(expr: &Expression, ctx: &mut CompilerContext) -> Result<SExpr, CodegenError> {
    let inner = compile_expression(expr, ctx)?;
    if is_comparison_op(expr) {
        // Comparison ops already produce Bool — emit as-is.
        Ok(inner)
    } else {
        // I64 → Bool conversion: (ne <cond> 0).
        Ok(SExpr::List(vec![
            SExpr::Atom(Atom::Symbol(String::from("ne"))),
            inner,
            SExpr::Atom(Atom::Integer(0)),
        ]))
    }
}

fn is_comparison_op(expr: &Expression) -> bool {
    matches!(
        expr,
        Expression::BinaryOp {
            op: BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Lt
                | BinaryOperator::Gt
                | BinaryOperator::LtEq
                | BinaryOperator::GtEq,
            ..
        }
    )
}

fn binary_op_to_ir_name(op: BinaryOperator) -> &'static str {
    match op {
        BinaryOperator::Add => "add",
        BinaryOperator::Sub => "sub",
        BinaryOperator::Mul => "mul",
        BinaryOperator::Div => "div",
        BinaryOperator::Eq => "eq",
        BinaryOperator::NotEq => "ne",
        BinaryOperator::Lt => "lt",
        BinaryOperator::Gt => "gt",
        BinaryOperator::LtEq => "le",
        BinaryOperator::GtEq => "ge",
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::parse;
    use crate::type_checker::check;
    use quarks_validator::type_checker::type_check;

    /// Helper: compile source through the full pipeline.
    fn compile_source(src: &str, main_name: &str) -> Result<SExpr, CodegenError> {
        let tokens = tokenize(src).expect("lex");
        let program = parse(&tokens).expect("parse");
        check(&program).expect("type-check");
        compile(&program, main_name)
    }

    /// Round-trip: compile + validate via Stage-6 type_check.
    fn round_trip(src: &str, main_name: &str) {
        let ir = compile_source(src, main_name).expect("compile");
        type_check(&ir).expect("validator accepts compiled IR");
    }

    // ── A: Happy-Path ──────────────────────────────────────────

    #[test]
    fn a01_minimal_main() {
        round_trip("fn main() -> i64 { return 0; }", "main");
    }

    #[test]
    fn a02_arithmetic() {
        round_trip("fn main() -> i64 { return 1 + 2 * 3; }", "main");
    }

    #[test]
    fn a03_let_binding() {
        round_trip("fn main() -> i64 { let x = 5; return x + 1; }", "main");
    }

    #[test]
    fn a04_multiple_lets() {
        round_trip(
            "fn main() -> i64 { let x = 5; let y = x + 1; let z = y * 2; return z; }",
            "main",
        );
    }

    #[test]
    fn a05_if_else() {
        round_trip(
            "fn main() -> i64 { if 1 { return 10; } else { return 20; } }",
            "main",
        );
    }

    // ── B: Control Flow ────────────────────────────────────────

    #[test]
    fn b01_if_without_else_flattens() {
        round_trip(
            "fn main() -> i64 { if 1 { return 10; } return 20; }",
            "main",
        );
    }

    #[test]
    fn b02_loop_with_break() {
        round_trip("fn main() -> i64 { loop { break; } return 0; }", "main");
    }

    #[test]
    fn b03_infinite_loop() {
        round_trip("fn main() -> i64 { loop { return 0; } }", "main");
    }

    #[test]
    fn b04_nested_if() {
        round_trip(
            "fn main() -> i64 { if 1 { if 2 { return 1; } return 2; } return 3; }",
            "main",
        );
    }

    #[test]
    fn b05_loop_with_conditional_break() {
        round_trip(
            "fn main() -> i64 { loop { if 1 { break; } } return 0; }",
            "main",
        );
    }

    // ── C: Functions ───────────────────────────────────────────

    #[test]
    fn c01_function_call() {
        round_trip(
            "fn add(a: i64, b: i64) -> i64 { return a + b; } fn main() -> i64 { return add(1, 2); }",
            "main",
        );
    }

    #[test]
    fn c02_mutually_recursive() {
        round_trip(
            "fn ping(n: i64) -> i64 { return n; } fn pong(n: i64) -> i64 { return ping(n); } fn main() -> i64 { return pong(5); }",
            "main",
        );
    }

    #[test]
    fn c03_shadowing_fresh_indices() {
        round_trip(
            "fn main() -> i64 { let x = 5; let x = x + 1; return x; }",
            "main",
        );
    }

    #[test]
    fn c04_nested_blocks_restore_scope() {
        round_trip(
            "fn main() -> i64 { let x = 1; if 1 { let x = 2; return x; } return x; }",
            "main",
        );
    }

    // ── D: Operators ───────────────────────────────────────────

    #[test]
    fn d01_negation_to_sub_zero() {
        round_trip("fn main() -> i64 { return -5; }", "main");
    }

    #[test]
    fn d02_all_comparisons() {
        round_trip(
            "fn main() -> i64 { if 1 == 1 { return 1; } if 1 != 2 { return 1; } if 1 < 2 { return 1; } if 2 > 1 { return 1; } if 1 <= 2 { return 1; } if 2 >= 1 { return 1; } return 0; }",
            "main",
        );
    }

    #[test]
    fn d03_handle_and_bytes_literals() {
        round_trip("fn main() -> handle { return @42; }", "main");
        round_trip("fn main() -> bytes { return #x4142; }", "main");
    }

    // ── E: Intent ──────────────────────────────────────────────

    #[test]
    fn e01_intent_as_expression_statement() {
        round_trip("fn main() -> i64 { intent(@1, 42); return 0; }", "main");
    }

    #[test]
    fn e02_intent_as_return_value() {
        round_trip("fn main() -> i64 { return intent(@1, 42); }", "main");
    }

    #[test]
    fn e03_multiple_intent_statements() {
        round_trip(
            "fn main() -> i64 { intent(@1, 1); intent(@2, 2); return 0; }",
            "main",
        );
    }

    // ── F: Entry-Point ─────────────────────────────────────────

    #[test]
    fn f01_entry_point_not_found() {
        let tokens = tokenize("fn main() -> i64 { return 0; }").unwrap();
        let program = parse(&tokens).unwrap();
        check(&program).unwrap();
        let result = compile(&program, "nonexistent");
        match result {
            Err(CodegenError::EntryPointNotFound { name }) => {
                assert_eq!(name, "nonexistent");
            }
            other => panic!("expected EntryPointNotFound, got {:?}", other),
        }
    }

    #[test]
    fn f02_entry_point_with_params() {
        let tokens = tokenize("fn main(x: i64) -> i64 { return x; }").unwrap();
        let program = parse(&tokens).unwrap();
        check(&program).unwrap();
        let result = compile(&program, "main");
        match result {
            Err(CodegenError::EntryPointHasParameters { name, param_count }) => {
                assert_eq!(name, "main");
                assert_eq!(param_count, 1);
            }
            other => panic!("expected EntryPointHasParameters, got {:?}", other),
        }
    }

    #[test]
    fn f03_any_function_as_entry() {
        round_trip(
            "fn helper(x: i64) -> i64 { return x + 1; } fn run() -> i64 { return helper(42); }",
            "run",
        );
    }

    // ── G: Structural ──────────────────────────────────────────

    #[test]
    fn g01_ir_starts_with_program() {
        let ir = compile_source("fn main() -> i64 { return 0; }", "main").unwrap();
        match ir {
            SExpr::List(items) => {
                assert!(matches!(&items[0], SExpr::Atom(Atom::Symbol(s)) if s == "program"));
                assert_eq!(items.len(), 3); // program, fn, call
            }
            _ => panic!("expected List"),
        }
    }

    #[test]
    fn g02_ir_ends_with_main_call() {
        let ir = compile_source("fn main() -> i64 { return 0; }", "main").unwrap();
        match ir {
            SExpr::List(items) => {
                let last = items.last().unwrap();
                match last {
                    SExpr::List(call_items) => {
                        assert!(
                            matches!(&call_items[0], SExpr::Atom(Atom::Symbol(s)) if s == "call")
                        );
                        assert!(
                            matches!(&call_items[1], SExpr::Atom(Atom::Symbol(s)) if s == "main")
                        );
                    }
                    _ => panic!("expected List"),
                }
            }
            _ => panic!("expected List"),
        }
    }

    #[test]
    fn g03_negation_structure() {
        let ir = compile_source("fn main() -> i64 { return -5; }", "main").unwrap();
        // Verify the sub 0 pattern is in there
        fn contains_sub_zero(sexpr: &SExpr) -> bool {
            match sexpr {
                SExpr::List(items) => {
                    if items.len() == 3 {
                        if matches!(&items[0], SExpr::Atom(Atom::Symbol(s)) if s == "sub") {
                            if matches!(&items[1], SExpr::Atom(Atom::Integer(0))) {
                                return true;
                            }
                        }
                    }
                    items.iter().any(contains_sub_zero)
                }
                SExpr::Atom(_) => false,
            }
        }
        assert!(contains_sub_zero(&ir));
    }

    // ── H: Defensive error returns (no panic) ──────────────────
    //
    // The codegen lives in a Ring-0 path; a panic here is a kernel
    // crash. These tests exercise the four conditions that
    // previously panicked and now must surface as structured
    // `CodegenError` values.

    use crate::lexer::Span;

    fn dummy_span() -> Span {
        Span::new(0, 0)
    }

    #[test]
    fn h01_unresolved_name_returns_error() {
        // Hand-craft a malformed Program that bypasses the
        // type-checker so we can exercise the codegen's
        // defensive path. `body` references `undefined` which
        // appears in no scope.
        let program = Program {
            functions: vec![FunctionDef {
                name: String::from("main"),
                params: vec![],
                return_type: Type::I64,
                body: Block {
                    statements: vec![Statement::Return {
                        value: Expression::Identifier {
                            name: String::from("undefined"),
                            span: dummy_span(),
                        },
                        span: dummy_span(),
                    }],
                    span: dummy_span(),
                },
                span: dummy_span(),
            }],
            span: dummy_span(),
        };

        match compile(&program, "main") {
            Err(CodegenError::UnresolvedName { name }) => {
                assert_eq!(name, "undefined");
            }
            other => panic!("expected UnresolvedName, got {:?}", other),
        }
    }

    #[test]
    fn h02_empty_block_returns_error() {
        // An empty function body — parser+TC reject this in the
        // happy path, but the codegen must not panic if it ever
        // sees one.
        let program = Program {
            functions: vec![FunctionDef {
                name: String::from("main"),
                params: vec![],
                return_type: Type::I64,
                body: Block {
                    statements: vec![],
                    span: dummy_span(),
                },
                span: dummy_span(),
            }],
            span: dummy_span(),
        };

        match compile(&program, "main") {
            Err(CodegenError::EmptyBlock) => {}
            other => panic!("expected EmptyBlock, got {:?}", other),
        }
    }

    #[test]
    fn h03_let_as_last_statement_returns_error() {
        // `let` as the final statement of a function body — the
        // type-checker rejects this in the happy path because the
        // body would have no value to return. The codegen used to
        // panic; it now surfaces a structured error.
        let program = Program {
            functions: vec![FunctionDef {
                name: String::from("main"),
                params: vec![],
                return_type: Type::I64,
                body: Block {
                    statements: vec![Statement::Let {
                        name: String::from("x"),
                        value: Expression::IntegerLiteral {
                            value: 1,
                            span: dummy_span(),
                        },
                        span: dummy_span(),
                    }],
                    span: dummy_span(),
                },
                span: dummy_span(),
            }],
            span: dummy_span(),
        };

        match compile(&program, "main") {
            Err(CodegenError::LetAsLast { name }) => {
                assert_eq!(name, "x");
            }
            other => panic!("expected LetAsLast, got {:?}", other),
        }
    }

    #[test]
    fn h04_codegen_never_panics_on_round_trip_inputs() {
        // Sanity: every happy-path round trip exercised above
        // returns Ok. This test re-uses one of the canonical
        // inputs to lock the property "Result-returning codegen
        // still type-checks the compiled IR" into the test
        // matrix.
        let ir = compile_source("fn main() -> i64 { let x = 5; return x + 1; }", "main")
            .expect("happy-path codegen must succeed");
        type_check(&ir).expect("validator accepts compiled IR");
    }
}
