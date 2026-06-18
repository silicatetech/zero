// SPDX-License-Identifier: AGPL-3.0-or-later
use alloc::string::String;
use alloc::vec::Vec;

use alloc::format;

use crate::ast::{Atom, SExpr};

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub list_path: Vec<usize>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    UnexpectedChar(char),
    UnexpectedEof,
    InvalidInteger,
    InvalidBytes,
    InvalidHandle,
    InvalidParameter,
    InvalidSymbol,
    UnmatchedParen,
}

/// Parse an Quarks-IR S-expression from a string.
///
/// Returns a single top-level `SExpr`. The input must contain exactly
/// one expression (leading/trailing whitespace is allowed, trailing
/// tokens after the first complete expression are an error).
pub fn parse(input: &str) -> Result<SExpr, ParseError> {
    let mut cursor = Cursor::new(input);
    let mut path: Vec<usize> = Vec::new();
    let expr = parse_sexpr(&mut cursor, &mut path)?;

    // After parsing one complete expression, only whitespace should remain.
    cursor.skip_whitespace();
    if let Some(ch) = cursor.peek() {
        if ch == ')' {
            return Err(ParseError {
                kind: ParseErrorKind::UnmatchedParen,
                list_path: path.clone(),
                message: format!("unexpected closing parenthesis at position {}", cursor.pos),
            });
        }
        return Err(ParseError {
            kind: ParseErrorKind::UnexpectedChar(ch),
            list_path: path.clone(),
            message: format!(
                "unexpected character '{}' after complete expression at position {}",
                ch, cursor.pos
            ),
        });
    }

    Ok(expr)
}

// ── Cursor ─────────────────────────────────────────────────────

struct Cursor<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        if self.pos < self.input.len() {
            Some(self.input[self.pos] as char)
        } else {
            None
        }
    }

    fn advance(&mut self) -> Option<char> {
        if self.pos < self.input.len() {
            let ch = self.input[self.pos] as char;
            self.pos += 1;
            Some(ch)
        } else {
            None
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }
}

// ── Recursive-descent parser ───────────────────────────────────

fn parse_sexpr(cursor: &mut Cursor, path: &mut Vec<usize>) -> Result<SExpr, ParseError> {
    cursor.skip_whitespace();

    match cursor.peek() {
        None => Err(ParseError {
            kind: ParseErrorKind::UnexpectedEof,
            list_path: path.clone(),
            message: String::from("unexpected end of input while expecting expression"),
        }),
        Some('(') => parse_list(cursor, path),
        Some(')') => Err(ParseError {
            kind: ParseErrorKind::UnmatchedParen,
            list_path: path.clone(),
            message: format!("unexpected ')' at position {}", cursor.pos),
        }),
        Some(_) => parse_atom(cursor, path).map(SExpr::Atom),
    }
}

fn parse_list(cursor: &mut Cursor, path: &mut Vec<usize>) -> Result<SExpr, ParseError> {
    // Consume '('
    cursor.advance();
    cursor.skip_whitespace();

    // Check for empty list
    match cursor.peek() {
        None => {
            return Err(ParseError {
                kind: ParseErrorKind::UnexpectedEof,
                list_path: path.clone(),
                message: String::from("unexpected end of input inside list (unclosed parenthesis)"),
            });
        }
        Some(')') => {
            // Empty lists `()` are structurally valid at the parser level.
            // Semantic rejection (e.g. `()` as a top-level program, or as
            // an instruction call without an instruction-name symbol) is
            // enforced by the validator, not the parser.
            //
            // Zero-arity function definitions rely on this:
            //   (fn foo () i64 body)
            // See ADR-017. Changed in Stage 6 Extension MP2a (abbd324).
            cursor.advance();
            return Ok(SExpr::List(Vec::new()));
        }
        _ => {}
    }

    let mut children: Vec<SExpr> = Vec::new();
    loop {
        cursor.skip_whitespace();
        match cursor.peek() {
            None => {
                return Err(ParseError {
                    kind: ParseErrorKind::UnexpectedEof,
                    list_path: path.clone(),
                    message: String::from(
                        "unexpected end of input inside list (unclosed parenthesis)",
                    ),
                });
            }
            Some(')') => {
                cursor.advance();
                break;
            }
            _ => {
                path.push(children.len());
                let child = parse_sexpr(cursor, path)?;
                path.pop();
                children.push(child);
            }
        }
    }

    Ok(SExpr::List(children))
}

fn parse_atom(cursor: &mut Cursor, path: &mut Vec<usize>) -> Result<Atom, ParseError> {
    match cursor.peek() {
        Some('#') => parse_bytes(cursor, path),
        Some('@') => parse_handle(cursor, path),
        Some('%') => parse_parameter(cursor, path),
        Some(ch) if ch == '-' || ch.is_ascii_digit() => parse_integer_or_symbol(cursor, path),
        Some(ch) if is_symbol_start(ch) => parse_symbol(cursor, path),
        Some(ch) if ch.is_ascii_uppercase() => Err(ParseError {
            kind: ParseErrorKind::InvalidSymbol,
            list_path: path.clone(),
            message: format!("symbol cannot contain uppercase character '{}' at position {}; symbols are [a-z0-9_-]+ only", ch, cursor.pos),
        }),
        Some(ch) => Err(ParseError {
            kind: ParseErrorKind::UnexpectedChar(ch),
            list_path: path.clone(),
            message: format!("unexpected character '{}' at position {}", ch, cursor.pos),
        }),
        None => Err(ParseError {
            kind: ParseErrorKind::UnexpectedEof,
            list_path: path.clone(),
            message: String::from("unexpected end of input while expecting atom"),
        }),
    }
}

#[allow(clippy::ptr_arg)]
fn parse_bytes(cursor: &mut Cursor, path: &mut Vec<usize>) -> Result<Atom, ParseError> {
    let start = cursor.pos;
    cursor.advance(); // consume '#'

    match cursor.peek() {
        Some('x') => cursor.advance(),
        _ => {
            return Err(ParseError {
                kind: ParseErrorKind::InvalidBytes,
                list_path: path.clone(),
                message: format!("expected 'x' after '#' at position {}", start),
            });
        }
    };

    // Collect hex chars
    let mut hex_chars: Vec<u8> = Vec::new();
    while let Some(ch) = cursor.peek() {
        if ch.is_ascii_hexdigit() && ch.is_ascii_lowercase() || ch.is_ascii_digit() {
            hex_chars.push(ch as u8);
            cursor.advance();
        } else if ch.is_ascii_hexdigit() && ch.is_ascii_uppercase() {
            return Err(ParseError {
                kind: ParseErrorKind::InvalidBytes,
                list_path: path.clone(),
                message: format!("uppercase hex digit '{}' not allowed in bytes literal at position {}; use lowercase a-f", ch, cursor.pos),
            });
        } else if ch.is_ascii_alphanumeric() {
            // Non-hex alphanumeric after #x: invalid bytes literal
            return Err(ParseError {
                kind: ParseErrorKind::InvalidBytes,
                list_path: path.clone(),
                message: format!("invalid character '{}' in bytes literal at position {}; expected lowercase hex pairs (0-9, a-f)", ch, cursor.pos),
            });
        } else {
            break;
        }
    }

    // Must have even number of hex chars
    if !hex_chars.len().is_multiple_of(2) {
        return Err(ParseError {
            kind: ParseErrorKind::InvalidBytes,
            list_path: path.clone(),
            message: format!("odd number of hex digits ({}) in bytes literal starting at position {}; hex pairs required", hex_chars.len(), start),
        });
    }

    // Decode hex pairs
    let mut bytes: Vec<u8> = Vec::with_capacity(hex_chars.len() / 2);
    for pair in hex_chars.chunks(2) {
        let hi = hex_digit(pair[0]);
        let lo = hex_digit(pair[1]);
        bytes.push((hi << 4) | lo);
    }

    Ok(Atom::Bytes(bytes))
}

fn hex_digit(ch: u8) -> u8 {
    match ch {
        b'0'..=b'9' => ch - b'0',
        b'a'..=b'f' => ch - b'a' + 10,
        _ => unreachable!("validated by is_ascii_hexdigit"),
    }
}

#[allow(clippy::ptr_arg)]
fn parse_handle(cursor: &mut Cursor, path: &mut Vec<usize>) -> Result<Atom, ParseError> {
    let start = cursor.pos;
    cursor.advance(); // consume '@'

    // Collect digits
    let mut digits = Vec::new();
    while let Some(ch) = cursor.peek() {
        if ch.is_ascii_digit() {
            digits.push(ch as u8);
            cursor.advance();
        } else {
            break;
        }
    }

    if digits.is_empty() {
        return Err(ParseError {
            kind: ParseErrorKind::InvalidHandle,
            list_path: path.clone(),
            message: format!("expected decimal digits after '@' at position {}", start),
        });
    }

    let s: String = digits.iter().map(|&b| b as char).collect();
    match s.parse::<u64>() {
        Ok(val) => Ok(Atom::Handle(val)),
        Err(_) => Err(ParseError {
            kind: ParseErrorKind::InvalidHandle,
            list_path: path.clone(),
            message: format!("handle value too large at position {}", start),
        }),
    }
}

#[allow(clippy::ptr_arg)]
fn parse_parameter(cursor: &mut Cursor, path: &mut Vec<usize>) -> Result<Atom, ParseError> {
    let start = cursor.pos;
    cursor.advance(); // consume '%'

    // Collect digits
    let mut digits = Vec::new();
    while let Some(ch) = cursor.peek() {
        if ch.is_ascii_digit() {
            digits.push(ch as u8);
            cursor.advance();
        } else {
            break;
        }
    }

    if digits.is_empty() {
        return Err(ParseError {
            kind: ParseErrorKind::InvalidParameter,
            list_path: path.clone(),
            message: format!("expected decimal digits after '%' at position {}", start),
        });
    }

    let s: String = digits.iter().map(|&b| b as char).collect();
    match s.parse::<u32>() {
        Ok(val) => Ok(Atom::Parameter(val)),
        Err(_) => Err(ParseError {
            kind: ParseErrorKind::InvalidParameter,
            list_path: path.clone(),
            message: format!("parameter index too large at position {}", start),
        }),
    }
}

fn parse_integer_or_symbol(cursor: &mut Cursor, path: &mut Vec<usize>) -> Result<Atom, ParseError> {
    let _start = cursor.pos;

    // Peek ahead: if '-' followed by digit, it is a negative integer.
    // If '-' followed by symbol char, it is a symbol starting with '-'.
    // Wait: spec says symbol first char must not be digit. But '-' is allowed
    // in symbols. We need to distinguish -123 (integer) from -foo (symbol).
    if cursor.peek() == Some('-') {
        // Look at the char after '-'
        if cursor.pos + 1 < cursor.input.len() {
            let next = cursor.input[cursor.pos + 1] as char;
            if next.is_ascii_digit() {
                return parse_integer(cursor, path);
            }
            // '-' followed by non-digit: parse as symbol
            return parse_symbol(cursor, path);
        }
        // '-' at EOF: single-char symbol
        return parse_symbol(cursor, path);
    }

    // Starts with digit: integer
    parse_integer(cursor, path)
}

#[allow(clippy::ptr_arg)]
fn parse_integer(cursor: &mut Cursor, path: &mut Vec<usize>) -> Result<Atom, ParseError> {
    let start = cursor.pos;

    let mut chars: Vec<u8> = Vec::new();
    if cursor.peek() == Some('-') {
        chars.push(b'-');
        cursor.advance();
    }

    while let Some(ch) = cursor.peek() {
        if ch.is_ascii_digit() {
            chars.push(ch as u8);
            cursor.advance();
        } else {
            break;
        }
    }

    if chars.is_empty() || (chars.len() == 1 && chars[0] == b'-') {
        return Err(ParseError {
            kind: ParseErrorKind::InvalidInteger,
            list_path: path.clone(),
            message: format!("expected decimal digits at position {}", start),
        });
    }

    let s: String = chars.iter().map(|&b| b as char).collect();
    match s.parse::<i64>() {
        Ok(val) => Ok(Atom::Integer(val)),
        Err(_) => Err(ParseError {
            kind: ParseErrorKind::InvalidInteger,
            list_path: path.clone(),
            message: format!("integer value out of i64 range at position {}", start),
        }),
    }
}

#[allow(clippy::ptr_arg)]
fn parse_symbol(cursor: &mut Cursor, path: &mut Vec<usize>) -> Result<Atom, ParseError> {
    let start = cursor.pos;

    let mut chars: Vec<u8> = Vec::new();
    while let Some(ch) = cursor.peek() {
        if is_symbol_char(ch) {
            chars.push(ch as u8);
            cursor.advance();
        } else {
            break;
        }
    }

    if chars.is_empty() {
        return Err(ParseError {
            kind: ParseErrorKind::InvalidSymbol,
            list_path: path.clone(),
            message: format!("expected symbol at position {}", start),
        });
    }

    let s: String = chars.iter().map(|&b| b as char).collect();

    // Validate: symbol must match [a-z0-9_-]+, first char not digit
    let first = s.as_bytes()[0] as char;
    if first.is_ascii_digit() {
        return Err(ParseError {
            kind: ParseErrorKind::InvalidSymbol,
            list_path: path.clone(),
            message: format!(
                "symbol cannot start with digit: '{}' at position {}",
                s, start
            ),
        });
    }

    // Check for uppercase (spec says case-sensitive, symbols are lowercase)
    for &b in s.as_bytes() {
        let ch = b as char;
        if ch.is_ascii_uppercase() {
            return Err(ParseError {
                kind: ParseErrorKind::InvalidSymbol,
                list_path: path.clone(),
                message: format!("symbol contains uppercase character '{}' in '{}' at position {}; symbols are [a-z0-9_-]+ only", ch, s, start),
            });
        }
    }

    Ok(Atom::Symbol(s))
}

fn is_symbol_start(ch: char) -> bool {
    ch.is_ascii_lowercase() || ch == '_' || ch == '-'
}

fn is_symbol_char(ch: char) -> bool {
    ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-'
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // ── Valid atoms ─────────────────────────────────────────────

    #[test]
    fn parse_integer_positive() {
        assert_eq!(parse("42"), Ok(SExpr::Atom(Atom::Integer(42))));
    }

    #[test]
    fn parse_integer_negative() {
        assert_eq!(parse("-1"), Ok(SExpr::Atom(Atom::Integer(-1))));
    }

    #[test]
    fn parse_integer_zero() {
        assert_eq!(parse("0"), Ok(SExpr::Atom(Atom::Integer(0))));
    }

    #[test]
    fn parse_bytes_hello() {
        assert_eq!(
            parse("#x48656c6c6f"),
            Ok(SExpr::Atom(Atom::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f])))
        );
    }

    #[test]
    fn parse_bytes_single_byte() {
        assert_eq!(parse("#x00"), Ok(SExpr::Atom(Atom::Bytes(vec![0x00]))));
    }

    #[test]
    fn parse_bytes_empty() {
        assert_eq!(parse("#x"), Ok(SExpr::Atom(Atom::Bytes(vec![]))));
    }

    #[test]
    fn parse_handle_zero() {
        assert_eq!(parse("@0"), Ok(SExpr::Atom(Atom::Handle(0))));
    }

    #[test]
    fn parse_handle_five() {
        assert_eq!(parse("@5"), Ok(SExpr::Atom(Atom::Handle(5))));
    }

    #[test]
    fn parse_handle_large() {
        assert_eq!(parse("@127"), Ok(SExpr::Atom(Atom::Handle(127))));
    }

    #[test]
    fn parse_symbol_simple() {
        assert_eq!(
            parse("send"),
            Ok(SExpr::Atom(Atom::Symbol(String::from("send"))))
        );
    }

    #[test]
    fn parse_symbol_with_dash() {
        assert_eq!(
            parse("my-agent"),
            Ok(SExpr::Atom(Atom::Symbol(String::from("my-agent"))))
        );
    }

    #[test]
    fn parse_symbol_with_underscore() {
        assert_eq!(
            parse("loop_body"),
            Ok(SExpr::Atom(Atom::Symbol(String::from("loop_body"))))
        );
    }

    // ── Valid lists ─────────────────────────────────────────────

    #[test]
    fn parse_simple_list() {
        assert_eq!(
            parse("(add 1 2)"),
            Ok(SExpr::List(vec![
                SExpr::Atom(Atom::Symbol(String::from("add"))),
                SExpr::Atom(Atom::Integer(1)),
                SExpr::Atom(Atom::Integer(2)),
            ]))
        );
    }

    #[test]
    fn parse_list_with_bytes_and_handle() {
        assert_eq!(
            parse("(send @5 #x48656c6c6f)"),
            Ok(SExpr::List(vec![
                SExpr::Atom(Atom::Symbol(String::from("send"))),
                SExpr::Atom(Atom::Handle(5)),
                SExpr::Atom(Atom::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f])),
            ]))
        );
    }

    #[test]
    fn parse_nested_list() {
        let result = parse("(if (eq a 0) (return 1) (return 0))");
        assert!(result.is_ok());
        let expr = result.unwrap();
        if let SExpr::List(items) = &expr {
            assert_eq!(items.len(), 4); // if, (eq a 0), (return 1), (return 0)
            assert_eq!(items[0], SExpr::Atom(Atom::Symbol(String::from("if"))));
            if let SExpr::List(eq_items) = &items[1] {
                assert_eq!(eq_items.len(), 3);
            } else {
                panic!("expected nested list");
            }
        } else {
            panic!("expected list");
        }
    }

    #[test]
    fn parse_with_varied_whitespace() {
        let result = parse("(add\n\t1\r\n  2)");
        assert_eq!(
            result,
            Ok(SExpr::List(vec![
                SExpr::Atom(Atom::Symbol(String::from("add"))),
                SExpr::Atom(Atom::Integer(1)),
                SExpr::Atom(Atom::Integer(2)),
            ]))
        );
    }

    // ── Invalid inputs ─────────────────────────────────────────

    #[test]
    fn error_unclosed_paren() {
        let err = parse("(").unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::UnexpectedEof);
    }

    #[test]
    fn empty_list_parses_to_empty_vec() {
        // Empty lists are now valid at parse level. The validator's
        // structure-check layer rejects them as instructions; the
        // type-checker uses them as fn-parameter lists.
        assert_eq!(parse("()"), Ok(SExpr::List(vec![])));
    }

    #[test]
    fn error_unclosed_list() {
        let err = parse("(add 1").unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::UnexpectedEof);
    }

    #[test]
    fn error_unmatched_close() {
        let err = parse("1 2)").unwrap_err();
        // After parsing "1", trailing "2)" is unexpected
        assert!(matches!(err.kind, ParseErrorKind::UnexpectedChar(_)));
    }

    #[test]
    fn error_trailing_close_paren() {
        let err = parse("(add 1 2))").unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::UnmatchedParen);
    }

    #[test]
    fn error_bytes_non_hex() {
        let err = parse("#xZZ").unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::InvalidBytes);
    }

    #[test]
    fn error_bytes_odd_length() {
        let err = parse("#x1").unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::InvalidBytes);
    }

    #[test]
    fn error_bytes_uppercase_hex() {
        let err = parse("#xFF").unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::InvalidBytes);
    }

    #[test]
    fn error_handle_no_digits() {
        let err = parse("@@").unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::InvalidHandle);
    }

    #[test]
    fn error_handle_negative() {
        // @-1 -> '@' then '-1'. '-' is not a digit so handle parsing
        // fails with no digits after '@'.
        let err = parse("@-1").unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::InvalidHandle);
    }

    #[test]
    fn error_symbol_uppercase() {
        let err = parse("Send").unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::InvalidSymbol);
    }

    // ── Path tracking ──────────────────────────────────────────

    #[test]
    fn path_tracking_nested_error() {
        // (outer (inner @@))
        // "outer" is children[0], "(inner @@)" is children[1]
        // inside inner: "inner" is children[0], "@@" is children[1]
        // So the error path for @@ should be [1, 1]
        let err = parse("(outer (inner @@))").unwrap_err();
        assert_eq!(err.kind, ParseErrorKind::InvalidHandle);
        assert_eq!(err.list_path, vec![1, 1]);
    }

    #[test]
    fn path_tracking_deeply_nested() {
        // (a (b (c ##)))
        // a=children[0], (b ...)=children[1]
        // b=children[0], (c ##)=children[1]
        // c=children[0], ##=children[1]
        // path for ##: [1, 1, 1]
        let err = parse("(a (b (c ##)))").unwrap_err();
        assert_eq!(err.list_path, vec![1, 1, 1]);
    }

    #[test]
    fn path_tracking_first_element_error() {
        // (@@)
        // @@ is children[0]
        // path: [0]
        let err = parse("(@@)").unwrap_err();
        assert_eq!(err.list_path, vec![0]);
    }

    #[test]
    fn path_tracking_root_level() {
        // A root-level invalid atom has empty path
        let err = parse("@@").unwrap_err();
        assert_eq!(err.list_path, vec![] as Vec<usize>);
    }

    // ── Parameter (%n) tokens ──────────────────────────────────

    #[test]
    fn parse_parameter_zero() {
        assert_eq!(parse("%0"), Ok(SExpr::Atom(Atom::Parameter(0))));
    }

    #[test]
    fn parse_parameter_nonzero() {
        assert_eq!(parse("%42"), Ok(SExpr::Atom(Atom::Parameter(42))));
    }

    #[test]
    fn parse_parameter_in_list() {
        let result = parse("(add %0 %1)").unwrap();
        match result {
            SExpr::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], SExpr::Atom(Atom::Symbol(String::from("add"))));
                assert_eq!(items[1], SExpr::Atom(Atom::Parameter(0)));
                assert_eq!(items[2], SExpr::Atom(Atom::Parameter(1)));
            }
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn parse_parameter_without_digits_errors() {
        let err = parse("%").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::InvalidParameter));
    }

    #[test]
    fn parse_parameter_with_non_digit_errors() {
        let err = parse("%abc").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::InvalidParameter));
    }

    #[test]
    fn parse_parameter_overflow_errors() {
        // u32::MAX + 1
        let err = parse("%4294967296").unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::InvalidParameter));
    }
}
