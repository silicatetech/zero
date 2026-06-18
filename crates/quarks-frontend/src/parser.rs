// SPDX-License-Identifier: AGPL-3.0-or-later
use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::ast::*;
use crate::lexer::{Span, Token, TokenKind};

// ── Public API ─────────────────────────────────────────────────

/// Parse a token stream into a program AST.
///
/// Fail-Fast semantics: the first error encountered is returned.
pub fn parse(tokens: &[Token]) -> Result<Program, ParseError> {
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    UnexpectedToken { expected: String, found: String },
    UnexpectedEof,
    InvalidType,
    ExpectedExpression,
    ExpectedIdentifier,
    EmptyProgram,
}

// ── Parser Internals ───────────────────────────────────────────

struct Parser<'a> {
    tokens: &'a [Token],
    cursor: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, cursor: 0 }
    }

    // ── Cursor helpers ─────────────────────────────────────────

    fn peek(&self) -> &Token {
        &self.tokens[self.cursor]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.tokens[self.cursor].kind
    }

    fn peek_at(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.cursor + offset)
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.cursor];
        if self.cursor < self.tokens.len() - 1 {
            self.cursor += 1;
        }
        tok
    }

    fn at_eof(&self) -> bool {
        matches!(self.tokens[self.cursor].kind, TokenKind::Eof)
    }

    /// Consume a token of the expected kind and return its span.
    /// Only works for unit variants (keywords, delimiters).
    fn expect(&mut self, expected: TokenKind) -> Result<Span, ParseError> {
        let tok = &self.tokens[self.cursor];
        if tok.kind == expected {
            let span = tok.span;
            self.cursor += 1;
            Ok(span)
        } else {
            Err(ParseError {
                kind: ParseErrorKind::UnexpectedToken {
                    expected: token_kind_name(&expected),
                    found: token_kind_name(&tok.kind),
                },
                span: tok.span,
                message: format!(
                    "expected {}, found {}",
                    token_kind_name(&expected),
                    token_kind_name(&tok.kind)
                ),
            })
        }
    }

    /// Consume an identifier token and return (name, span).
    fn expect_identifier(&mut self) -> Result<(String, Span), ParseError> {
        let tok = &self.tokens[self.cursor];
        if let TokenKind::Ident(name) = &tok.kind {
            let result = (name.clone(), tok.span);
            self.cursor += 1;
            Ok(result)
        } else {
            Err(ParseError {
                kind: ParseErrorKind::ExpectedIdentifier,
                span: tok.span,
                message: format!("expected identifier, found {}", token_kind_name(&tok.kind)),
            })
        }
    }

    // ── Grammar Productions ────────────────────────────────────

    // program = function_def+ ;
    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let start_span = self.peek().span;
        let mut functions = Vec::new();

        while !self.at_eof() {
            functions.push(self.parse_function_def()?);
        }

        if functions.is_empty() {
            return Err(ParseError {
                kind: ParseErrorKind::EmptyProgram,
                span: start_span,
                message: String::from("program must contain at least one function"),
            });
        }

        let end_span = self.peek().span; // Eof
        Ok(Program {
            functions,
            span: Span::new(start_span.start, end_span.end),
        })
    }

    // function_def = "fn" IDENT "(" params? ")" "->" type block ;
    fn parse_function_def(&mut self) -> Result<FunctionDef, ParseError> {
        let start = self.expect(TokenKind::Fn)?;
        let (name, _) = self.expect_identifier()?;
        self.expect(TokenKind::LParen)?;

        let params = if matches!(self.peek_kind(), TokenKind::RParen) {
            Vec::new()
        } else {
            self.parse_params()?
        };

        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Arrow)?;
        let (return_type, _) = self.parse_type()?;
        let body = self.parse_block()?;

        let span = Span::new(start.start, body.span.end);
        Ok(FunctionDef {
            name,
            params,
            return_type,
            body,
            span,
        })
    }

    // params = param ("," param)* ","? ;
    fn parse_params(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();
        params.push(self.parse_param()?);

        while matches!(self.peek_kind(), TokenKind::Comma) {
            self.advance(); // consume ','
                            // Trailing comma: if next is ')', stop
            if matches!(self.peek_kind(), TokenKind::RParen) {
                break;
            }
            params.push(self.parse_param()?);
        }

        Ok(params)
    }

    // param = IDENT ":" type ;
    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let (name, name_span) = self.expect_identifier()?;
        self.expect(TokenKind::Colon)?;
        let (ty, type_span) = self.parse_type()?;
        Ok(Param {
            name,
            ty,
            span: Span::new(name_span.start, type_span.end),
        })
    }

    // type = "i64" | "bytes" | "handle" ;
    fn parse_type(&mut self) -> Result<(Type, Span), ParseError> {
        let tok = &self.tokens[self.cursor];
        let result = match tok.kind {
            TokenKind::TypeI64 => Ok((Type::I64, tok.span)),
            TokenKind::TypeBytes => Ok((Type::Bytes, tok.span)),
            TokenKind::TypeHandle => Ok((Type::Handle, tok.span)),
            _ => Err(ParseError {
                kind: ParseErrorKind::InvalidType,
                span: tok.span,
                message: format!(
                    "expected type (i64, bytes, or handle), found {}",
                    token_kind_name(&tok.kind)
                ),
            }),
        };
        if result.is_ok() {
            self.cursor += 1;
        }
        result
    }

    // block = "{" statement* "}" ;
    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let start = self.expect(TokenKind::LBrace)?;
        let mut statements = Vec::new();

        while !matches!(self.peek_kind(), TokenKind::RBrace) {
            if self.at_eof() {
                return Err(ParseError {
                    kind: ParseErrorKind::UnexpectedEof,
                    span: self.peek().span,
                    message: String::from("unexpected end of input, expected '}'"),
                });
            }
            statements.push(self.parse_statement()?);
        }

        let end = self.expect(TokenKind::RBrace)?;
        Ok(Block {
            statements,
            span: Span::new(start.start, end.end),
        })
    }

    // statement dispatch
    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        match self.peek_kind() {
            TokenKind::Let => self.parse_let_stmt(),
            TokenKind::Return => self.parse_return_stmt(),
            TokenKind::If => self.parse_if_stmt(),
            TokenKind::Loop => self.parse_loop_stmt(),
            TokenKind::Break => self.parse_break_stmt(),
            _ => self.parse_expr_stmt(),
        }
    }

    // let_stmt = "let" IDENT "=" expression ";" ;
    fn parse_let_stmt(&mut self) -> Result<Statement, ParseError> {
        let start = self.expect(TokenKind::Let)?;
        let (name, _) = self.expect_identifier()?;
        self.expect(TokenKind::Equals)?;
        let value = self.parse_expression()?;
        let end = self.expect(TokenKind::Semicolon)?;
        Ok(Statement::Let {
            name,
            value,
            span: Span::new(start.start, end.end),
        })
    }

    // return_stmt = "return" expression ";" ;
    fn parse_return_stmt(&mut self) -> Result<Statement, ParseError> {
        let start = self.expect(TokenKind::Return)?;
        let value = self.parse_expression()?;
        let end = self.expect(TokenKind::Semicolon)?;
        Ok(Statement::Return {
            value,
            span: Span::new(start.start, end.end),
        })
    }

    // break_stmt = "break" ";" ;
    fn parse_break_stmt(&mut self) -> Result<Statement, ParseError> {
        let start = self.expect(TokenKind::Break)?;
        let end = self.expect(TokenKind::Semicolon)?;
        Ok(Statement::Break {
            span: Span::new(start.start, end.end),
        })
    }

    // if_stmt = "if" expression block ("else" block)? ;
    fn parse_if_stmt(&mut self) -> Result<Statement, ParseError> {
        let start = self.expect(TokenKind::If)?;
        let condition = self.parse_expression()?;
        let then_block = self.parse_block()?;

        let else_block = if matches!(self.peek_kind(), TokenKind::Else) {
            self.advance();
            Some(self.parse_block()?)
        } else {
            None
        };

        let end = else_block.as_ref().map_or(then_block.span, |b| b.span);

        Ok(Statement::If {
            condition,
            then_block,
            else_block,
            span: Span::new(start.start, end.end),
        })
    }

    // loop_stmt = "loop" block ;
    fn parse_loop_stmt(&mut self) -> Result<Statement, ParseError> {
        let start = self.expect(TokenKind::Loop)?;
        let body = self.parse_block()?;
        let end = body.span.end;
        Ok(Statement::Loop {
            body,
            span: Span::new(start.start, end),
        })
    }

    // expr_stmt = expression ";" ;
    fn parse_expr_stmt(&mut self) -> Result<Statement, ParseError> {
        let expr = self.parse_expression()?;
        let expr_start = expr.span().start;
        let end = self.expect(TokenKind::Semicolon)?;
        Ok(Statement::Expr {
            expression: expr,
            span: Span::new(expr_start, end.end),
        })
    }

    // ── Expression Parsing (Pratt-style via loops) ─────────────

    // expression = comparison ;
    fn parse_expression(&mut self) -> Result<Expression, ParseError> {
        self.parse_comparison()
    }

    // comparison = arithmetic (comparison_op arithmetic)? ;
    // Non-associative: at most one comparison operator.
    fn parse_comparison(&mut self) -> Result<Expression, ParseError> {
        let left = self.parse_arithmetic()?;

        let op = match self.peek_kind() {
            TokenKind::EqEq => BinaryOperator::Eq,
            TokenKind::NotEq => BinaryOperator::NotEq,
            TokenKind::Lt => BinaryOperator::Lt,
            TokenKind::Gt => BinaryOperator::Gt,
            TokenKind::LtEq => BinaryOperator::LtEq,
            TokenKind::GtEq => BinaryOperator::GtEq,
            _ => return Ok(left),
        };

        self.advance();
        let right = self.parse_arithmetic()?;
        let span = Span::new(left.span().start, right.span().end);

        Ok(Expression::BinaryOp {
            op,
            lhs: Box::new(left),
            rhs: Box::new(right),
            span,
        })
    }

    // arithmetic = term (("+" | "-") term)* ;
    // Left-associative.
    fn parse_arithmetic(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_term()?;

        loop {
            let op = match self.peek_kind() {
                TokenKind::Plus => BinaryOperator::Add,
                TokenKind::Minus => BinaryOperator::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_term()?;
            let span = Span::new(left.span().start, right.span().end);
            left = Expression::BinaryOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    // term = unary (("*" | "/") unary)* ;
    // Left-associative.
    fn parse_term(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_unary()?;

        loop {
            let op = match self.peek_kind() {
                TokenKind::Star => BinaryOperator::Mul,
                TokenKind::Slash => BinaryOperator::Div,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            let span = Span::new(left.span().start, right.span().end);
            left = Expression::BinaryOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    // unary = "-" unary | primary ;
    fn parse_unary(&mut self) -> Result<Expression, ParseError> {
        if matches!(self.peek_kind(), TokenKind::Minus) {
            let start = self.tokens[self.cursor].span;
            self.cursor += 1;
            let operand = self.parse_unary()?;
            let span = Span::new(start.start, operand.span().end);
            Ok(Expression::UnaryOp {
                op: UnaryOperator::Negate,
                operand: Box::new(operand),
                span,
            })
        } else {
            self.parse_primary()
        }
    }

    // primary = literal | IDENT | call | intent_call | "(" expression ")" ;
    fn parse_primary(&mut self) -> Result<Expression, ParseError> {
        // Integer literal
        if let TokenKind::Integer(n) = self.tokens[self.cursor].kind {
            let span = self.tokens[self.cursor].span;
            self.cursor += 1;
            return Ok(Expression::IntegerLiteral { value: n, span });
        }

        // Bytes literal
        if matches!(self.tokens[self.cursor].kind, TokenKind::Bytes(_)) {
            let span = self.tokens[self.cursor].span;
            if let TokenKind::Bytes(ref b) = self.tokens[self.cursor].kind {
                let value = b.clone();
                self.cursor += 1;
                return Ok(Expression::BytesLiteral { value, span });
            }
        }

        // Handle literal
        if let TokenKind::Handle(h) = self.tokens[self.cursor].kind {
            let span = self.tokens[self.cursor].span;
            self.cursor += 1;
            return Ok(Expression::HandleLiteral { value: h, span });
        }

        // Identifier or function call
        if matches!(self.tokens[self.cursor].kind, TokenKind::Ident(_)) {
            let is_call = self
                .peek_at(1)
                .map_or(false, |t| matches!(t.kind, TokenKind::LParen));
            if is_call {
                return self.parse_call();
            } else {
                let span = self.tokens[self.cursor].span;
                if let TokenKind::Ident(ref name) = self.tokens[self.cursor].kind {
                    let name = name.clone();
                    self.cursor += 1;
                    return Ok(Expression::Identifier { name, span });
                }
            }
        }

        // Intent call
        if matches!(self.tokens[self.cursor].kind, TokenKind::Intent) {
            return self.parse_intent_call();
        }

        // Parenthesized expression
        if matches!(self.tokens[self.cursor].kind, TokenKind::LParen) {
            self.cursor += 1; // consume '('
            let expr = self.parse_expression()?;
            self.expect(TokenKind::RParen)?;
            return Ok(expr);
        }

        // Nothing matched — error
        let span = self.tokens[self.cursor].span;
        let found = token_kind_name(&self.tokens[self.cursor].kind);
        Err(ParseError {
            kind: ParseErrorKind::ExpectedExpression,
            span,
            message: format!("expected expression, found {}", found),
        })
    }

    // call = IDENT "(" args? ")" ;
    fn parse_call(&mut self) -> Result<Expression, ParseError> {
        let (name, name_span) = self.expect_identifier()?;
        self.expect(TokenKind::LParen)?;

        let args = if matches!(self.peek_kind(), TokenKind::RParen) {
            Vec::new()
        } else {
            self.parse_call_args()?
        };

        let end = self.expect(TokenKind::RParen)?;
        Ok(Expression::Call {
            function: name,
            args,
            span: Span::new(name_span.start, end.end),
        })
    }

    // intent_call = "intent" "(" args ")" ;
    fn parse_intent_call(&mut self) -> Result<Expression, ParseError> {
        let start = self.tokens[self.cursor].span;
        self.cursor += 1; // consume 'intent'
        self.expect(TokenKind::LParen)?;

        let args = if matches!(self.peek_kind(), TokenKind::RParen) {
            Vec::new()
        } else {
            self.parse_call_args()?
        };

        let end = self.expect(TokenKind::RParen)?;
        Ok(Expression::IntentCall {
            args,
            span: Span::new(start.start, end.end),
        })
    }

    // args = expression ("," expression)* ","? ;
    fn parse_call_args(&mut self) -> Result<Vec<Expression>, ParseError> {
        let mut args = Vec::new();
        args.push(self.parse_expression()?);

        while matches!(self.peek_kind(), TokenKind::Comma) {
            self.advance(); // consume ','
                            // Trailing comma: if next is ')', stop
            if matches!(self.peek_kind(), TokenKind::RParen) {
                break;
            }
            args.push(self.parse_expression()?);
        }

        Ok(args)
    }
}

// ── Helpers ────────────────────────────────────────────────────

fn token_kind_name(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Fn => String::from("'fn'"),
        TokenKind::Let => String::from("'let'"),
        TokenKind::If => String::from("'if'"),
        TokenKind::Else => String::from("'else'"),
        TokenKind::Loop => String::from("'loop'"),
        TokenKind::Break => String::from("'break'"),
        TokenKind::Return => String::from("'return'"),
        TokenKind::Intent => String::from("'intent'"),
        TokenKind::TypeI64 => String::from("'i64'"),
        TokenKind::TypeBytes => String::from("'bytes'"),
        TokenKind::TypeHandle => String::from("'handle'"),
        TokenKind::Ident(s) => format!("identifier '{}'", s),
        TokenKind::Integer(n) => format!("integer '{}'", n),
        TokenKind::Bytes(_) => String::from("bytes literal"),
        TokenKind::Handle(n) => format!("handle '@{}'", n),
        TokenKind::Plus => String::from("'+'"),
        TokenKind::Minus => String::from("'-'"),
        TokenKind::Star => String::from("'*'"),
        TokenKind::Slash => String::from("'/'"),
        TokenKind::EqEq => String::from("'=='"),
        TokenKind::NotEq => String::from("'!='"),
        TokenKind::Lt => String::from("'<'"),
        TokenKind::Gt => String::from("'>'"),
        TokenKind::LtEq => String::from("'<='"),
        TokenKind::GtEq => String::from("'>='"),
        TokenKind::LParen => String::from("'('"),
        TokenKind::RParen => String::from("')'"),
        TokenKind::LBrace => String::from("'{'"),
        TokenKind::RBrace => String::from("'}'"),
        TokenKind::Comma => String::from("','"),
        TokenKind::Colon => String::from("':'"),
        TokenKind::Semicolon => String::from("';'"),
        TokenKind::Arrow => String::from("'->'"),
        TokenKind::Equals => String::from("'='"),
        TokenKind::Eof => String::from("end of input"),
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use alloc::vec;

    /// Helper: lex + parse in one step.
    fn parse_src(src: &str) -> Result<Program, ParseError> {
        let tokens = tokenize(src).expect("lex failed");
        parse(&tokens)
    }

    // ── A: Grammar Rules ───────────────────────────────────────

    #[test]
    fn parses_minimal_function() {
        let p = parse_src("fn main() -> i64 { return 0; }").unwrap();
        assert_eq!(p.functions.len(), 1);
        let f = &p.functions[0];
        assert_eq!(f.name, "main");
        assert!(f.params.is_empty());
        assert_eq!(f.return_type, Type::I64);
    }

    #[test]
    fn parses_function_with_params() {
        let p = parse_src("fn add(a: i64, b: i64) -> i64 { return a + b; }").unwrap();
        let f = &p.functions[0];
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.params[0].name, "a");
        assert_eq!(f.params[0].ty, Type::I64);
        assert_eq!(f.params[1].name, "b");
    }

    #[test]
    fn parses_let_statement() {
        let p = parse_src("fn f() -> i64 { let x = 5; return x; }").unwrap();
        let stmts = &p.functions[0].body.statements;
        assert_eq!(stmts.len(), 2);
        match &stmts[0] {
            Statement::Let { name, .. } => assert_eq!(name, "x"),
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn parses_if_else() {
        let p = parse_src("fn f() -> i64 { if 1 { return 10; } else { return 20; } }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::If { else_block, .. } => assert!(else_block.is_some()),
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn parses_if_without_else() {
        let p = parse_src("fn f() -> i64 { if 1 { return 10; } return 20; }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::If { else_block, .. } => assert!(else_block.is_none()),
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn parses_loop_break() {
        let p = parse_src("fn f() -> i64 { loop { break; } return 0; }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Loop { body, .. } => match &body.statements[0] {
                Statement::Break { .. } => {}
                _ => panic!("expected Break inside Loop"),
            },
            _ => panic!("expected Loop"),
        }
    }

    #[test]
    fn parses_function_call() {
        let p = parse_src("fn f() -> i64 { return add(1, 2); }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Return { value, .. } => match value {
                Expression::Call { function, args, .. } => {
                    assert_eq!(function, "add");
                    assert_eq!(args.len(), 2);
                }
                _ => panic!("expected Call"),
            },
            _ => panic!("expected Return"),
        }
    }

    #[test]
    fn parses_intent_call() {
        let p = parse_src("fn f() -> i64 { intent(@1, 42); return 0; }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Expr { expression, .. } => match expression {
                Expression::IntentCall { args, .. } => assert_eq!(args.len(), 2),
                _ => panic!("expected IntentCall"),
            },
            _ => panic!("expected Expr"),
        }
    }

    #[test]
    fn parses_zero_arg_call() {
        let p = parse_src("fn f() -> i64 { return g(); }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Return {
                value: Expression::Call { args, .. },
                ..
            } => {
                assert!(args.is_empty());
            }
            _ => panic!("expected Call with 0 args"),
        }
    }

    // ── B: Precedence & Associativity ──────────────────────────

    #[test]
    fn arithmetic_precedence_mul_over_add() {
        // 1 + 2 * 3 → Add(1, Mul(2, 3))
        let p = parse_src("fn f() -> i64 { return 1 + 2 * 3; }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Return { value, .. } => match value {
                Expression::BinaryOp {
                    op: BinaryOperator::Add,
                    rhs,
                    ..
                } => {
                    assert!(matches!(
                        **rhs,
                        Expression::BinaryOp {
                            op: BinaryOperator::Mul,
                            ..
                        }
                    ));
                }
                _ => panic!("expected top-level Add"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn comparison_has_lowest_precedence() {
        // 1 + 2 == 3 → Eq(Add(1, 2), 3)
        let p = parse_src("fn f() -> i64 { return 1 + 2 == 3; }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Return { value, .. } => {
                assert!(matches!(
                    value,
                    Expression::BinaryOp {
                        op: BinaryOperator::Eq,
                        ..
                    }
                ));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn left_associative_subtraction() {
        // 1 - 2 - 3 → Sub(Sub(1, 2), 3)
        let p = parse_src("fn f() -> i64 { return 1 - 2 - 3; }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Return { value, .. } => match value {
                Expression::BinaryOp {
                    op: BinaryOperator::Sub,
                    lhs,
                    ..
                } => {
                    assert!(matches!(
                        **lhs,
                        Expression::BinaryOp {
                            op: BinaryOperator::Sub,
                            ..
                        }
                    ));
                }
                _ => panic!("expected top-level Sub"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn parentheses_override_precedence() {
        // (1 + 2) * 3 → Mul(Add(1, 2), 3)
        let p = parse_src("fn f() -> i64 { return (1 + 2) * 3; }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Return { value, .. } => match value {
                Expression::BinaryOp {
                    op: BinaryOperator::Mul,
                    lhs,
                    ..
                } => {
                    assert!(matches!(
                        **lhs,
                        Expression::BinaryOp {
                            op: BinaryOperator::Add,
                            ..
                        }
                    ));
                }
                _ => panic!("expected top-level Mul"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn unary_negation_binds_tight() {
        // -1 + 2 → Add(Negate(1), 2)
        let p = parse_src("fn f() -> i64 { return -1 + 2; }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Return { value, .. } => match value {
                Expression::BinaryOp {
                    op: BinaryOperator::Add,
                    lhs,
                    ..
                } => {
                    assert!(matches!(
                        **lhs,
                        Expression::UnaryOp {
                            op: UnaryOperator::Negate,
                            ..
                        }
                    ));
                }
                _ => panic!("expected top-level Add"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn double_negation() {
        let p = parse_src("fn f() -> i64 { return --x; }").unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Return { value, .. } => match value {
                Expression::UnaryOp {
                    op: UnaryOperator::Negate,
                    operand,
                    ..
                } => {
                    assert!(matches!(
                        **operand,
                        Expression::UnaryOp {
                            op: UnaryOperator::Negate,
                            ..
                        }
                    ));
                }
                _ => panic!("expected outer Negate"),
            },
            _ => panic!(),
        }
    }

    // ── C: Trailing Comma ──────────────────────────────────────

    #[test]
    fn trailing_comma_in_params() {
        let p = parse_src("fn f(a: i64, b: i64,) -> i64 { return a; }").unwrap();
        assert_eq!(p.functions[0].params.len(), 2);
    }

    #[test]
    fn trailing_comma_in_call_args() {
        parse_src("fn f() -> i64 { return add(1, 2,); }").unwrap();
    }

    // ── D: Error Cases ─────────────────────────────────────────

    #[test]
    fn missing_semicolon_errors() {
        let err = parse_src("fn f() -> i64 { let x = 5 return x; }").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UnexpectedToken { .. }));
    }

    #[test]
    fn unclosed_brace_errors() {
        let err = parse_src("fn f() -> i64 { return 0;").unwrap_err();
        assert!(matches!(
            err.kind,
            ParseErrorKind::UnexpectedEof | ParseErrorKind::UnexpectedToken { .. }
        ));
    }

    #[test]
    fn empty_program_errors() {
        let err = parse_src("").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::EmptyProgram));
    }

    #[test]
    fn comment_only_program_errors() {
        let err = parse_src("// just a comment").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::EmptyProgram));
    }

    #[test]
    fn missing_return_type_errors() {
        let err = parse_src("fn f() { return 0; }").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UnexpectedToken { .. }));
    }

    #[test]
    fn invalid_type_errors() {
        let err = parse_src("fn f() -> i32 { return 0; }").unwrap_err();
        assert!(matches!(
            err.kind,
            ParseErrorKind::InvalidType | ParseErrorKind::UnexpectedToken { .. }
        ));
    }

    #[test]
    fn chained_comparison_errors() {
        // 1 < 2 < 3 → parser stops at first compare, expects `;`, finds `<`
        let err = parse_src("fn f() -> i64 { return 1 < 2 < 3; }").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UnexpectedToken { .. }));
    }

    // ── E: Span Accuracy ───────────────────────────────────────

    #[test]
    fn function_span_covers_full_definition() {
        let src = "fn f() -> i64 { return 0; }";
        let p = parse_src(src).unwrap();
        let f = &p.functions[0];
        assert_eq!(f.span.start, 0);
        assert_eq!(f.span.end, src.len());
    }

    #[test]
    fn expression_span_tracks_through_operators() {
        let src = "fn f() -> i64 { return 1 + 2; }";
        let p = parse_src(src).unwrap();
        match &p.functions[0].body.statements[0] {
            Statement::Return { value, .. } => {
                let s = value.span();
                let covered = &src[s.start..s.end];
                assert_eq!(covered, "1 + 2");
            }
            _ => panic!(),
        }
    }

    // ── F: Multi-Function ──────────────────────────────────────

    #[test]
    fn parses_multiple_functions() {
        let p = parse_src("fn a() -> i64 { return 1; } fn b() -> i64 { return 2; }").unwrap();
        assert_eq!(p.functions.len(), 2);
        assert_eq!(p.functions[0].name, "a");
        assert_eq!(p.functions[1].name, "b");
    }

    #[test]
    fn parses_mutually_recursive_signatures() {
        let src = "fn ping(n: i64) -> i64 { return pong(n); } \
                   fn pong(n: i64) -> i64 { return ping(n); }";
        let p = parse_src(src).unwrap();
        assert_eq!(p.functions.len(), 2);
    }

    // ── G: Edge Cases ──────────────────────────────────────────

    #[test]
    fn empty_block() {
        let p = parse_src("fn f() -> i64 { }").unwrap();
        assert!(p.functions[0].body.statements.is_empty());
    }

    #[test]
    fn nested_parentheses() {
        parse_src("fn f() -> i64 { return (((1))); }").unwrap();
    }

    #[test]
    fn deeply_nested_if() {
        parse_src("fn f() -> i64 { if 1 { if 2 { if 3 { return 1; } } } return 0; }").unwrap();
    }

    #[test]
    fn handle_literal_in_expression() {
        let p = parse_src("fn f() -> handle { return @42; }").unwrap();
        assert_eq!(p.functions[0].return_type, Type::Handle);
        match &p.functions[0].body.statements[0] {
            Statement::Return {
                value: Expression::HandleLiteral { value, .. },
                ..
            } => {
                assert_eq!(*value, 42);
            }
            _ => panic!("expected HandleLiteral"),
        }
    }

    #[test]
    fn bytes_literal_in_expression() {
        let p = parse_src("fn f() -> bytes { return #x48656c6c6f; }").unwrap();
        assert_eq!(p.functions[0].return_type, Type::Bytes);
        match &p.functions[0].body.statements[0] {
            Statement::Return {
                value: Expression::BytesLiteral { value, .. },
                ..
            } => {
                assert_eq!(value, &vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]);
            }
            _ => panic!("expected BytesLiteral"),
        }
    }

    #[test]
    fn all_comparison_operators() {
        for op_src in &["==", "!=", "<", ">", "<=", ">="] {
            let src = format!("fn f() -> i64 {{ return 1 {} 2; }}", op_src);
            let p = parse_src(&src).unwrap();
            match &p.functions[0].body.statements[0] {
                Statement::Return {
                    value: Expression::BinaryOp { .. },
                    ..
                } => {}
                _ => panic!("expected BinaryOp for {}", op_src),
            }
        }
    }

    #[test]
    fn all_arithmetic_operators() {
        for (op_src, expected_op) in &[
            ("+", BinaryOperator::Add),
            ("-", BinaryOperator::Sub),
            ("*", BinaryOperator::Mul),
            ("/", BinaryOperator::Div),
        ] {
            let src = format!("fn f() -> i64 {{ return 1 {} 2; }}", op_src);
            let p = parse_src(&src).unwrap();
            match &p.functions[0].body.statements[0] {
                Statement::Return {
                    value: Expression::BinaryOp { op, .. },
                    ..
                } => {
                    assert_eq!(op, expected_op);
                }
                _ => panic!("expected BinaryOp for {}", op_src),
            }
        }
    }

    #[test]
    fn complex_realistic_function() {
        let src = "\
fn max(a: i64, b: i64) -> i64 {
    if a > b {
        return a;
    } else {
        return b;
    }
}";
        let p = parse_src(src).unwrap();
        let f = &p.functions[0];
        assert_eq!(f.name, "max");
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.body.statements.len(), 1);
        assert!(matches!(&f.body.statements[0], Statement::If { .. }));
    }

    #[test]
    fn loop_with_let_and_break() {
        let src = "\
fn count() -> i64 {
    let n = 0;
    loop {
        if n == 10 {
            break;
        }
        let n = n + 1;
    }
    return n;
}";
        let p = parse_src(src).unwrap();
        assert_eq!(p.functions[0].body.statements.len(), 3); // let, loop, return
    }
}
