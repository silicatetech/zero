// SPDX-License-Identifier: AGPL-3.0-or-later
use quarks_lsp::server::QuarksLsp;
use tower_lsp::lsp_types::*;
use tower_lsp::{jsonrpc, LspService};

/// Build a minimal LSP initialize request.
fn initialize_request() -> jsonrpc::Request {
    let params = InitializeParams::default();
    jsonrpc::Request::build("initialize")
        .params(serde_json::to_value(params).unwrap())
        .id(1)
        .finish()
}

#[tokio::test]
async fn initialize_returns_capabilities() {
    let (mut service, _socket) = LspService::new(QuarksLsp::new);

    let request = initialize_request();
    let response = tower::Service::call(&mut service, request)
        .await
        .expect("initialize should not fail");

    let response = response.expect("initialize should return a response");

    // Verify: response deserializes to InitializeResult with expected fields.
    let result: InitializeResult =
        serde_json::from_value(response.result().expect("ok response").clone())
            .expect("response should be valid InitializeResult");

    assert_eq!(result.server_info.as_ref().unwrap().name, "quarks-lsp");
    assert_eq!(
        result.server_info.as_ref().unwrap().version.as_deref(),
        Some("0.1.0")
    );
    assert!(matches!(
        result.capabilities.text_document_sync,
        Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL))
    ));
}
