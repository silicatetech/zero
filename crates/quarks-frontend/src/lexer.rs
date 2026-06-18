// SPDX-License-Identifier: AGPL-3.0-or-later
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

// ── Public Types ───────────────────────────────────────────────

/// Byte-span of a token in the source text.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// A single token with its kind and source span.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// Lexical token categories.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Fn,
    Let,
    If,
    Else,
    Loop,
    Break,
    Return,
    Intent,

    // Type keywords
    TypeI64,
    TypeBytes,
    TypeHandle,

    // Identifiers
    Ident(String),

    // Literals
    Integer(i64),
    Bytes(Vec<u8>),
    Handle(u64),

    // Operators — arithmetic
    Plus,
    Minus,
    Star,
    Slash,

    // Operators — comparison
    EqEq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Semicolon,
    Arrow,
    Equals,

    // End of input
    Eof,
}

/// Lex error with span and kind.
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub kind: LexErrorKind,
    pub span: Span,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LexErrorKind {
    UnexpectedChar(char),
    UnterminatedBlockComment,
    InvalidInteger,
    IntegerOverflow,
    InvalidBytes,
    InvalidHandle,
    InvalidOperator,
}

// ── Cursor ─────────────────────────────────────────────────────

struct Cursor<'a> {
    source: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source: source.as_bytes(),
            pos: 0,
        }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn at_end(&self) -> bool {
        self.pos >= self.source.len()
    }

    fn peek(&self) -> Option<u8> {
        if self.at_end() {
            None
        } else {
            Some(self.source[self.pos])
        }
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        let idx = self.pos + offset;
        if idx < self.source.len() {
            Some(self.source[idx])
        } else {
            None
        }
    }

    fn advance(&mut self) -> Option<u8> {
        if self.at_end() {
            None
        } else {
            let b = self.source[self.pos];
            self.pos += 1;
            Some(b)
        }
    }
}

// ── Tokenizer ──────────────────────────────────────────────────

/// Tokenize source text into a stream of tokens terminated by `Eof`.
///
/// Whitespace (spaces, tabs, newlines) and comments (line `//` and
/// block `/* ... */`) are filtered out. All remaining syntactic
/// content produces tokens with accurate byte spans.
///
/// Returns the first error encountered (Fail-Fast).
pub fn tokenize(source: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let mut cursor = Cursor::new(source);

    loop {
        skip_whitespace_and_comments(&mut cursor)?;

        if cursor.at_end() {
            let pos = cursor.position();
            tokens.push(Token {
                kind: TokenKind::Eof,
                span: Span::new(pos, pos),
            });
            return Ok(tokens);
        }

        let start = cursor.position();
        let kind = next_token_kind(&mut cursor, start)?;
        let end = cursor.position();

        tokens.push(Token {
            kind,
            span: Span::new(start, end),
        });
    }
}

// ── Whitespace & Comments ──────────────────────────────────────

fn skip_whitespace_and_comments(cursor: &mut Cursor<'_>) -> Result<(), LexError> {
    loop {
        // Skip whitespace
        while let Some(b) = cursor.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                cursor.advance();
            } else {
                break;
            }
        }

        // Check for comments
        if cursor.peek() == Some(b'/') {
            if cursor.peek_at(1) == Some(b'/') {
                // Line comment — skip to EOL
                cursor.advance(); // /
                cursor.advance(); // /
                while let Some(b) = cursor.peek() {
                    if b == b'\n' {
                        cursor.advance();
                        break;
                    }
                    cursor.advance();
                }
                continue; // re-enter loop to skip more whitespace/comments
            } else if cursor.peek_at(1) == Some(b'*') {
                // Block comment — skip to */
                let start = cursor.position();
                cursor.advance(); // /
                cursor.advance(); // *
                loop {
                    if cursor.at_end() {
                        return Err(LexError {
                            kind: LexErrorKind::UnterminatedBlockComment,
                            span: Span::new(start, cursor.position()),
                            message: String::from("unterminated block comment"),
                        });
                    }
                    if cursor.peek() == Some(b'*') && cursor.peek_at(1) == Some(b'/') {
                        cursor.advance(); // *
                        cursor.advance(); // /
                        break;
                    }
                    cursor.advance();
                }
                continue; // re-enter loop
            }
        }

        // Not whitespace, not a comment — done
        break;
    }
    Ok(())
}

// ── Token Dispatch ─────────────────────────────────────────────

fn next_token_kind(cursor: &mut Cursor<'_>, start: usize) -> Result<TokenKind, LexError> {
    let b = cursor.peek().unwrap(); // caller ensures not at_end

    match b {
        b'(' => {
            cursor.advance();
            Ok(TokenKind::LParen)
        }
        b')' => {
            cursor.advance();
            Ok(TokenKind::RParen)
        }
        b'{' => {
            cursor.advance();
            Ok(TokenKind::LBrace)
        }
        b'}' => {
            cursor.advance();
            Ok(TokenKind::RBrace)
        }
        b',' => {
            cursor.advance();
            Ok(TokenKind::Comma)
        }
        b':' => {
            cursor.advance();
            Ok(TokenKind::Colon)
        }
        b';' => {
            cursor.advance();
            Ok(TokenKind::Semicolon)
        }
        b'+' => {
            cursor.advance();
            Ok(TokenKind::Plus)
        }
        b'*' => {
            cursor.advance();
            Ok(TokenKind::Star)
        }

        // `-` → Arrow (`->`) or Minus
        b'-' => {
            cursor.advance();
            if cursor.peek() == Some(b'>') {
                cursor.advance();
                Ok(TokenKind::Arrow)
            } else {
                Ok(TokenKind::Minus)
            }
        }

        // `/` — comments already stripped, so this is always divide
        b'/' => {
            cursor.advance();
            Ok(TokenKind::Slash)
        }

        // `=` → `==` or `=`
        b'=' => {
            cursor.advance();
            if cursor.peek() == Some(b'=') {
                cursor.advance();
                Ok(TokenKind::EqEq)
            } else {
                Ok(TokenKind::Equals)
            }
        }

        // `!` → `!=` or error
        b'!' => {
            cursor.advance();
            if cursor.peek() == Some(b'=') {
                cursor.advance();
                Ok(TokenKind::NotEq)
            } else {
                Err(LexError {
                    kind: LexErrorKind::InvalidOperator,
                    span: Span::new(start, cursor.position()),
                    message: String::from("expected '!=' but found '!' alone"),
                })
            }
        }

        // `<` → `<=` or `<`
        b'<' => {
            cursor.advance();
            if cursor.peek() == Some(b'=') {
                cursor.advance();
                Ok(TokenKind::LtEq)
            } else {
                Ok(TokenKind::Lt)
            }
        }

        // `>` → `>=` or `>`
        b'>' => {
            cursor.advance();
            if cursor.peek() == Some(b'=') {
                cursor.advance();
                Ok(TokenKind::GtEq)
            } else {
                Ok(TokenKind::Gt)
            }
        }

        // `#` → bytes literal `#xHEXHEX...`
        b'#' => lex_bytes_literal(cursor, start),

        // `@` → handle literal `@DIGITS`
        b'@' => lex_handle_literal(cursor, start),

        // Digit → integer literal
        b'0'..=b'9' => lex_integer(cursor, start),

        // Letter or `_` → identifier or keyword
        b'a'..=b'z' | b'A'..=b'Z' | b'_' => lex_identifier_or_keyword(cursor),

        _ => {
            cursor.advance();
            Err(LexError {
                kind: LexErrorKind::UnexpectedChar(b as char),
                span: Span::new(start, cursor.position()),
                message: format!("unexpected character '{}'", b as char),
            })
        }
    }
}

// ── Literal Lexers ─────────────────────────────────────────────

fn lex_integer(cursor: &mut Cursor<'_>, start: usize) -> Result<TokenKind, LexError> {
    let begin = cursor.position();

    // Consume digits
    while let Some(b'0'..=b'9') = cursor.peek() {
        cursor.advance();
    }

    let end = cursor.position();
    let slice = &cursor.source[begin..end];
    // Safety: all digits are ASCII
    let s = core::str::from_utf8(slice).unwrap();

    match s.parse::<i64>() {
        Ok(n) => Ok(TokenKind::Integer(n)),
        Err(_) => Err(LexError {
            kind: LexErrorKind::IntegerOverflow,
            span: Span::new(start, end),
            message: format!("integer literal '{}' overflows i64", s),
        }),
    }
}

fn lex_bytes_literal(cursor: &mut Cursor<'_>, start: usize) -> Result<TokenKind, LexError> {
    cursor.advance(); // skip '#'

    // Expect 'x'
    if cursor.peek() != Some(b'x') {
        return Err(LexError {
            kind: LexErrorKind::InvalidBytes,
            span: Span::new(start, cursor.position()),
            message: String::from("bytes literal must start with '#x'"),
        });
    }
    cursor.advance(); // skip 'x'

    // Collect hex digits
    let hex_start = cursor.position();
    while let Some(b) = cursor.peek() {
        if b.is_ascii_hexdigit() {
            cursor.advance();
        } else {
            break;
        }
    }
    let hex_end = cursor.position();
    let hex_len = hex_end - hex_start;

    // Must be even number of hex digits
    if hex_len % 2 != 0 {
        return Err(LexError {
            kind: LexErrorKind::InvalidBytes,
            span: Span::new(start, hex_end),
            message: String::from("bytes literal must have even number of hex digits"),
        });
    }

    // Parse pairs of hex digits
    let mut bytes = Vec::with_capacity(hex_len / 2);
    let hex_slice = &cursor.source[hex_start..hex_end];
    for chunk in hex_slice.chunks(2) {
        let hi = hex_digit_value(chunk[0]);
        let lo = hex_digit_value(chunk[1]);
        bytes.push((hi << 4) | lo);
    }

    Ok(TokenKind::Bytes(bytes))
}

fn hex_digit_value(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => unreachable!(), // caller ensures is_ascii_hexdigit
    }
}

fn lex_handle_literal(cursor: &mut Cursor<'_>, start: usize) -> Result<TokenKind, LexError> {
    cursor.advance(); // skip '@'

    let digit_start = cursor.position();
    while let Some(b'0'..=b'9') = cursor.peek() {
        cursor.advance();
    }
    let digit_end = cursor.position();

    if digit_end == digit_start {
        return Err(LexError {
            kind: LexErrorKind::InvalidHandle,
            span: Span::new(start, digit_end),
            message: String::from("handle literal '@' must be followed by digits"),
        });
    }

    let slice = &cursor.source[digit_start..digit_end];
    let s = core::str::from_utf8(slice).unwrap();

    match s.parse::<u64>() {
        Ok(n) => Ok(TokenKind::Handle(n)),
        Err(_) => Err(LexError {
            kind: LexErrorKind::IntegerOverflow,
            span: Span::new(start, digit_end),
            message: format!("handle literal '@{}' overflows u64", s),
        }),
    }
}

fn lex_identifier_or_keyword(cursor: &mut Cursor<'_>) -> Result<TokenKind, LexError> {
    let begin = cursor.position();

    // First char already validated as letter/underscore by caller
    cursor.advance();

    // Continue with alphanumeric/underscore
    while let Some(b) = cursor.peek() {
        if b.is_ascii_alphanumeric() || b == b'_' {
            cursor.advance();
        } else {
            break;
        }
    }

    let end = cursor.position();
    let slice = &cursor.source[begin..end];
    let s = core::str::from_utf8(slice).unwrap();

    Ok(classify_identifier(s))
}

fn classify_identifier(s: &str) -> TokenKind {
    match s {
        "fn" => TokenKind::Fn,
        "let" => TokenKind::Let,
        "if" => TokenKind::If,
        "else" => TokenKind::Else,
        "loop" => TokenKind::Loop,
        "break" => TokenKind::Break,
        "return" => TokenKind::Return,
        "intent" => TokenKind::Intent,
        "i64" => TokenKind::TypeI64,
        "bytes" => TokenKind::TypeBytes,
        "handle" => TokenKind::TypeHandle,
        _ => TokenKind::Ident(String::from(s)),
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // ── A: Smoke ───────────────────────────────────────────────

    #[test]
    fn empty_input_yields_only_eof() {
        let tokens = tokenize("").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
        assert_eq!(tokens[0].span, Span::new(0, 0));
    }

    #[test]
    fn whitespace_only_yields_only_eof() {
        let tokens = tokenize("   \n\t\n  ").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    // ── B: Keywords ────────────────────────────────────────────

    #[test]
    fn keyword_fn_tokenized_correctly() {
        let tokens = tokenize("fn").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Fn);
        assert_eq!(tokens[0].span, Span::new(0, 2));
    }

    #[test]
    fn all_keywords_distinguished_from_idents() {
        let src = "fn let if else loop break return intent";
        let tokens = tokenize(src).unwrap();
        let kinds: Vec<_> = tokens.iter().take(8).map(|t| &t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                &TokenKind::Fn,
                &TokenKind::Let,
                &TokenKind::If,
                &TokenKind::Else,
                &TokenKind::Loop,
                &TokenKind::Break,
                &TokenKind::Return,
                &TokenKind::Intent,
            ]
        );
    }

    #[test]
    fn type_keywords() {
        let tokens = tokenize("i64 bytes handle").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::TypeI64);
        assert_eq!(tokens[1].kind, TokenKind::TypeBytes);
        assert_eq!(tokens[2].kind, TokenKind::TypeHandle);
    }

    #[test]
    fn ident_that_starts_like_keyword_is_still_ident() {
        let tokens = tokenize("ifx").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Ident(String::from("ifx")));
    }

    #[test]
    fn plain_identifier() {
        let tokens = tokenize("foo_bar").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Ident(String::from("foo_bar")));
        assert_eq!(tokens[0].span, Span::new(0, 7));
    }

    // ── C: Literals ────────────────────────────────────────────

    #[test]
    fn integer_literal() {
        let tokens = tokenize("42").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Integer(42));
    }

    #[test]
    fn negative_number_is_minus_then_integer() {
        // `-` is always Minus operator. Negative integers come from
        // the parser as unary-minus on positive literals.
        let tokens = tokenize("-17").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Minus);
        assert_eq!(tokens[1].kind, TokenKind::Integer(17));
    }

    #[test]
    fn integer_zero() {
        let tokens = tokenize("0").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Integer(0));
    }

    #[test]
    fn bytes_literal() {
        let tokens = tokenize("#x48656c6c6f").unwrap();
        assert_eq!(
            tokens[0].kind,
            TokenKind::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f])
        );
    }

    #[test]
    fn empty_bytes_literal() {
        let tokens = tokenize("#x").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Bytes(vec![]));
    }

    #[test]
    fn bytes_odd_digits_errors() {
        let err = tokenize("#x123").unwrap_err();
        assert!(matches!(err.kind, LexErrorKind::InvalidBytes));
    }

    #[test]
    fn handle_literal() {
        let tokens = tokenize("@42").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Handle(42));
    }

    #[test]
    fn handle_without_digits_errors() {
        let err = tokenize("@ ").unwrap_err();
        assert!(matches!(err.kind, LexErrorKind::InvalidHandle));
    }

    #[test]
    fn integer_overflow_errors() {
        let err = tokenize("99999999999999999999999").unwrap_err();
        assert!(matches!(err.kind, LexErrorKind::IntegerOverflow));
    }

    // ── D: Operators ───────────────────────────────────────────

    #[test]
    fn arithmetic_operators() {
        let tokens = tokenize("+ - * /").unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::Plus));
        assert!(matches!(tokens[1].kind, TokenKind::Minus));
        assert!(matches!(tokens[2].kind, TokenKind::Star));
        assert!(matches!(tokens[3].kind, TokenKind::Slash));
    }

    #[test]
    fn comparison_operators() {
        let tokens = tokenize("== != < > <= >=").unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::EqEq));
        assert!(matches!(tokens[1].kind, TokenKind::NotEq));
        assert!(matches!(tokens[2].kind, TokenKind::Lt));
        assert!(matches!(tokens[3].kind, TokenKind::Gt));
        assert!(matches!(tokens[4].kind, TokenKind::LtEq));
        assert!(matches!(tokens[5].kind, TokenKind::GtEq));
    }

    #[test]
    fn arrow_vs_minus() {
        let tokens = tokenize("-> -x").unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::Arrow));
        assert!(matches!(tokens[1].kind, TokenKind::Minus));
        assert!(matches!(tokens[2].kind, TokenKind::Ident(_)));
    }

    #[test]
    fn bang_without_equals_errors() {
        let err = tokenize("!").unwrap_err();
        assert!(matches!(err.kind, LexErrorKind::InvalidOperator));
    }

    // ── E: Delimiters ──────────────────────────────────────────

    #[test]
    fn delimiters() {
        let tokens = tokenize("( ) { } , : ; =").unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::LParen));
        assert!(matches!(tokens[1].kind, TokenKind::RParen));
        assert!(matches!(tokens[2].kind, TokenKind::LBrace));
        assert!(matches!(tokens[3].kind, TokenKind::RBrace));
        assert!(matches!(tokens[4].kind, TokenKind::Comma));
        assert!(matches!(tokens[5].kind, TokenKind::Colon));
        assert!(matches!(tokens[6].kind, TokenKind::Semicolon));
        assert!(matches!(tokens[7].kind, TokenKind::Equals));
    }

    // ── F: Comments ────────────────────────────────────────────

    #[test]
    fn line_comment_skipped() {
        let tokens = tokenize("42 // ignored\n7").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Integer(42));
        assert_eq!(tokens[1].kind, TokenKind::Integer(7));
        assert_eq!(tokens[2].kind, TokenKind::Eof);
    }

    #[test]
    fn block_comment_skipped() {
        let tokens = tokenize("42 /* ignored */ 7").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Integer(42));
        assert_eq!(tokens[1].kind, TokenKind::Integer(7));
    }

    #[test]
    fn unterminated_block_comment_errors() {
        let err = tokenize("42 /* never ends").unwrap_err();
        assert!(matches!(err.kind, LexErrorKind::UnterminatedBlockComment));
    }

    #[test]
    fn consecutive_comments() {
        let tokens = tokenize("// first\n// second\n/* third */42").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Integer(42));
    }

    // ── G: Span Accuracy ───────────────────────────────────────

    #[test]
    fn spans_track_accurately_through_whitespace() {
        let tokens = tokenize("  42  +  7").unwrap();
        assert_eq!(tokens[0].span, Span::new(2, 4)); // "42"
        assert_eq!(tokens[1].span, Span::new(6, 7)); // "+"
        assert_eq!(tokens[2].span, Span::new(9, 10)); // "7"
    }

    #[test]
    fn spans_track_accurately_through_comments() {
        let tokens = tokenize("42 /* gap */ 7").unwrap();
        assert_eq!(tokens[0].span, Span::new(0, 2)); // "42"
        assert_eq!(tokens[1].span, Span::new(13, 14)); // "7"
    }

    // ── H: Realistic Source ────────────────────────────────────

    #[test]
    fn full_function_tokenizes() {
        let src = "fn add(a: i64, b: i64) -> i64 { return a + b; }";
        let tokens = tokenize(src).unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::Fn));
        assert!(matches!(tokens[1].kind, TokenKind::Ident(ref s) if s == "add"));
        assert!(matches!(tokens[2].kind, TokenKind::LParen));
        assert!(matches!(tokens[3].kind, TokenKind::Ident(ref s) if s == "a"));
        assert!(matches!(tokens[4].kind, TokenKind::Colon));
        assert!(matches!(tokens[5].kind, TokenKind::TypeI64));
        assert!(matches!(tokens[6].kind, TokenKind::Comma));
        assert_eq!(tokens.last().unwrap().kind, TokenKind::Eof);
    }

    #[test]
    fn unexpected_char_errors() {
        let err = tokenize("42 ~ 7").unwrap_err();
        assert!(matches!(err.kind, LexErrorKind::UnexpectedChar('~')));
    }

    #[test]
    fn bytes_uppercase_hex() {
        let tokens = tokenize("#xFF").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Bytes(vec![0xFF]));
    }

    #[test]
    fn handle_at_zero() {
        // @0 is lexically valid — semantic rejection happens later
        let tokens = tokenize("@0").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Handle(0));
    }

    #[test]
    fn multiple_statements() {
        let src = "let x = 5;\nlet y = x + 1;";
        let tokens = tokenize(src).unwrap();
        // let x = 5 ; let y = x + 1 ; Eof
        assert_eq!(tokens.len(), 13); // 12 tokens + Eof
        assert_eq!(tokens[0].kind, TokenKind::Let);
        assert_eq!(tokens[4].kind, TokenKind::Semicolon);
        assert_eq!(tokens[5].kind, TokenKind::Let);
    }
}
