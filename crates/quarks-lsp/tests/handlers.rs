// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for LSP handler logic.
//!
//! Uses Option B (handler-logic extraction): tests call `run_validation`
//! directly rather than going through the tower-lsp Service stack.
//! This isolates the validation pipeline from LSP protocol concerns
//! and avoids notification-mocking complexity.
//!
//! MP5: migrated from S-expression IR to source-language pipeline.

use quarks_lsp::server::run_validation;
use tower_lsp::lsp_types::DiagnosticSeverity;

#[test]
fn valid_source_produces_no_diagnostics() {
    let diags = run_validation("fn main() -> i64 { return 0; }");
    assert!(
        diags.is_empty(),
        "valid source should produce no diagnostics"
    );
}

#[test]
fn lex_error_produces_diagnostic_with_span() {
    let diags = run_validation("fn main() -> i64 { return $; }");
    assert_eq!(diags.len(), 1);

    let d = &diags[0];
    assert_eq!(d.severity, Some(DiagnosticSeverity::ERROR));
    assert_eq!(d.source, Some(String::from("quarks-frontend")));
}

#[test]
fn parse_error_produces_diagnostic_with_span() {
    // Missing expression after +
    let diags = run_validation("fn main() -> i64 { return 1 + ; }");
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].severity, Some(DiagnosticSeverity::ERROR));
}

#[test]
fn type_check_error_produces_diagnostic_for_undefined_var() {
    let diags = run_validation("fn main() -> i64 { return undefined_var; }");
    assert_eq!(diags.len(), 1);

    let d = &diags[0];
    assert_eq!(d.severity, Some(DiagnosticSeverity::ERROR));
    // Span should point to "undefined_var"
    let start_offset = "fn main() -> i64 { return ".len();
    assert_eq!(d.range.start.line, 0);
    assert_eq!(d.range.start.character, start_offset as u32);
}

#[test]
fn diagnostic_data_field_is_structured_json() {
    let diags = run_validation("fn main() -> i64 { return undefined; }");
    assert_eq!(diags.len(), 1);

    let data = diags[0]
        .data
        .as_ref()
        .expect("data field should be present");
    let obj = data.as_object().expect("data should be a JSON object");
    assert!(obj.contains_key("error"), "data should contain 'error' key");
}

#[test]
fn multi_line_source_reports_correct_line() {
    // "undefined" is on line 1
    let diags = run_validation("fn main() -> i64 {\n    return undefined;\n}");
    assert_eq!(diags.len(), 1);

    let d = &diags[0];
    assert_eq!(d.range.start.line, 1);
}

#[test]
fn fail_fast_one_diagnostic_max() {
    // Lex error on first char — only one diagnostic fires
    let diags = run_validation("$");
    assert_eq!(diags.len(), 1, "Fail-Fast: exactly one diagnostic");
}
