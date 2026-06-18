// SPDX-License-Identifier: AGPL-3.0-or-later
use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range};

use crate::engine::{ByteSpan, EngineDiagnostic};

/// Convert an `EngineDiagnostic` to an LSP `Diagnostic`.
///
/// The source-language frontend provides direct byte spans, so no
/// AST path walking or argument-offset resolution is needed.
pub fn build_diagnostic(source: &str, diag: &EngineDiagnostic) -> Diagnostic {
    let range = byte_span_to_range(source, diag.span);

    Diagnostic {
        range,
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some(String::from("quarks-frontend")),
        message: diag.message.clone(),
        data: Some(serde_json::from_str(&diag.error_json).unwrap_or(serde_json::Value::Null)),
        ..Diagnostic::default()
    }
}

fn byte_span_to_range(source: &str, span: ByteSpan) -> Range {
    Range {
        start: byte_offset_to_position(source, span.start),
        end: byte_offset_to_position(source, span.end),
    }
}

/// Convert a byte offset to an LSP Position (line, character).
///
/// `character` uses UTF-16 code unit offsets per LSP 3.17 default.
/// For pure ASCII source (the common case for Quarks), this equals
/// the byte offset from line start.
pub(crate) fn byte_offset_to_position(source: &str, byte_offset: usize) -> Position {
    let bytes = source.as_bytes();
    let offset = byte_offset.min(bytes.len());

    let mut line: u32 = 0;
    let mut line_start_byte: usize = 0;

    for i in 0..offset {
        if bytes[i] == b'\n' {
            line += 1;
            line_start_byte = i + 1;
        }
    }

    let line_segment = &source[line_start_byte..offset];
    let character: u32 = line_segment.chars().map(|c| c.len_utf16() as u32).sum();

    Position { line, character }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::DiagnosticsEngine;

    #[test]
    fn build_diagnostic_for_lex_error_correct_severity() {
        let engine = DiagnosticsEngine::new();
        let source = "fn main() -> i64 { return $; }";
        let diags = engine.diagnose(source);
        assert_eq!(diags.len(), 1);
        let lsp = build_diagnostic(source, &diags[0]);
        assert_eq!(lsp.severity, Some(DiagnosticSeverity::ERROR));
        assert_eq!(lsp.source, Some(String::from("quarks-frontend")));
    }

    #[test]
    fn build_diagnostic_data_field_contains_error_json() {
        let engine = DiagnosticsEngine::new();
        let source = "fn main() -> i64 { return undefined; }";
        let diags = engine.diagnose(source);
        let lsp = build_diagnostic(source, &diags[0]);
        let data = lsp.data.expect("data should be populated");
        let obj = data.as_object().expect("data should be JSON object");
        assert!(obj.contains_key("error"));
    }

    #[test]
    fn build_diagnostic_for_multi_line_source() {
        let engine = DiagnosticsEngine::new();
        let source = "fn main() -> i64 {\n    return undefined;\n}";
        let diags = engine.diagnose(source);
        let lsp = build_diagnostic(source, &diags[0]);
        // "undefined" is on line 1 (0-based)
        assert_eq!(lsp.range.start.line, 1);
    }

    #[test]
    fn byte_offset_to_position_handles_ascii_correctly() {
        let source = "abc\ndef";
        // byte 5 = 'e' on line 1, character 1
        let pos = byte_offset_to_position(source, 5);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 1);
    }

    #[test]
    fn byte_offset_to_position_handles_utf16_for_non_ascii() {
        let source = "ä bc";
        // byte 0 = ä (2 bytes UTF-8, 1 UTF-16 unit)
        // byte 2 = ' ' on line 0, UTF-16 character 1
        let pos = byte_offset_to_position(source, 2);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.character, 1);
    }
}
