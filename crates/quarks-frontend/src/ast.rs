// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::lexer::Span;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// Root AST node — a complete program.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub functions: Vec<FunctionDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Type {
    I64,
    Bytes,
    Handle,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Let {
        name: String,
        value: Expression,
        span: Span,
    },
    Return {
        value: Expression,
        span: Span,
    },
    If {
        condition: Expression,
        then_block: Block,
        else_block: Option<Block>,
        span: Span,
    },
    Loop {
        body: Block,
        span: Span,
    },
    Break {
        span: Span,
    },
    Expr {
        expression: Expression,
        span: Span,
    },
}

impl Statement {
    pub fn span(&self) -> Span {
        match self {
            Statement::Let { span, .. } => *span,
            Statement::Return { span, .. } => *span,
            Statement::If { span, .. } => *span,
            Statement::Loop { span, .. } => *span,
            Statement::Break { span } => *span,
            Statement::Expr { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    IntegerLiteral {
        value: i64,
        span: Span,
    },
    BytesLiteral {
        value: Vec<u8>,
        span: Span,
    },
    HandleLiteral {
        value: u64,
        span: Span,
    },
    Identifier {
        name: String,
        span: Span,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
        span: Span,
    },
    BinaryOp {
        op: BinaryOperator,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
        span: Span,
    },
    Call {
        function: String,
        args: Vec<Expression>,
        span: Span,
    },
    IntentCall {
        args: Vec<Expression>,
        span: Span,
    },
}

impl Expression {
    pub fn span(&self) -> Span {
        match self {
            Expression::IntegerLiteral { span, .. } => *span,
            Expression::BytesLiteral { span, .. } => *span,
            Expression::HandleLiteral { span, .. } => *span,
            Expression::Identifier { span, .. } => *span,
            Expression::UnaryOp { span, .. } => *span,
            Expression::BinaryOp { span, .. } => *span,
            Expression::Call { span, .. } => *span,
            Expression::IntentCall { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOperator {
    Negate,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
}
