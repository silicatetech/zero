// SPDX-License-Identifier: AGPL-3.0-or-later
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use crate::diagnostic_builder::build_diagnostic;
use crate::document_store::DocumentStore;
use crate::engine::DiagnosticsEngine;

const DEBOUNCE_DURATION: Duration = Duration::from_millis(200);

/// Quarks Language Server.
///
/// Handles LSP lifecycle, text synchronization, and debounced
/// diagnostic publishing. Validation pipeline:
///
/// 1. `didOpen`/`didChange` → update `DocumentStore`
/// 2. Schedule debounced validation (cancel any in-flight for same URI)
/// 3. After debounce delay: `DiagnosticsEngine::diagnose` → `build_diagnostic` → `publishDiagnostics`
///
/// The debounce ensures rapid edits don't flood the validator.
pub struct QuarksLsp {
    client: Client,
    documents: DocumentStore,
    /// Per-URI pending validation task handles. Allows cancel-on-new-edit.
    pending: Arc<Mutex<HashMap<Url, JoinHandle<()>>>>,
}

impl QuarksLsp {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: DocumentStore::new(),
            pending: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Schedule a debounced validation for a URI.
    ///
    /// Cancels any in-flight validation for the same URI before
    /// spawning a new one. The new task waits `DEBOUNCE_DURATION`,
    /// then reads the latest document state and publishes diagnostics.
    ///
    /// Lock discipline: `pending` lock is acquired, modified, and
    /// dropped before any async work (sleep, publish). No lock is
    /// held across await points.
    async fn schedule_validation(&self, uri: Url) {
        let mut pending = self.pending.lock().await;

        // Cancel existing task for this URI if any.
        if let Some(handle) = pending.remove(&uri) {
            handle.abort();
        }

        let client = self.client.clone();
        let documents = self.documents.clone_handle();
        let uri_cloned = uri.clone();

        let handle = tokio::spawn(async move {
            tokio::time::sleep(DEBOUNCE_DURATION).await;

            let Some(doc) = documents.get(&uri_cloned).await else {
                return;
            };

            let lsp_diags = run_validation(&doc.content);

            client
                .publish_diagnostics(uri_cloned, lsp_diags, Some(doc.version))
                .await;
        });

        pending.insert(uri, handle);
        // `pending` lock dropped here — before any async work in the spawned task.
    }
}

/// Run the validation pipeline on source text and produce LSP Diagnostics.
///
/// This is the core handler logic, extracted as a free function for
/// testability (Option B from the MP4 spec). No LSP-specific state
/// needed — just source text in, diagnostics out.
pub fn run_validation(source: &str) -> Vec<Diagnostic> {
    let engine = DiagnosticsEngine::new();
    let engine_diags = engine.diagnose(source);
    engine_diags
        .iter()
        .map(|d| build_diagnostic(source, d))
        .collect()
}

#[tower_lsp::async_trait]
impl LanguageServer for QuarksLsp {
    async fn initialize(&self, _params: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: String::from("quarks-lsp"),
                version: Some(String::from(env!("CARGO_PKG_VERSION"))),
            }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                ..ServerCapabilities::default()
            },
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "quarks-lsp initialized")
            .await;
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        self.documents
            .insert(
                uri.clone(),
                params.text_document.text,
                params.text_document.version,
            )
            .await;
        self.schedule_validation(uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        // Full sync: content_changes has exactly one element with the full text.
        if let Some(change) = params.content_changes.into_iter().next() {
            self.documents
                .insert(uri.clone(), change.text, params.text_document.version)
                .await;
            self.schedule_validation(uri).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;

        // Cancel any pending validation — drop lock before async work.
        {
            let mut pending = self.pending.lock().await;
            if let Some(handle) = pending.remove(&uri) {
                handle.abort();
            }
        }

        // Clear diagnostics (empty vec signals "no diagnostics for this URI").
        self.client
            .publish_diagnostics(uri.clone(), Vec::new(), None)
            .await;
        self.documents.remove(&uri).await;
    }

    async fn shutdown(&self) -> Result<()> {
        // Cancel all pending tasks for clean exit.
        let mut pending = self.pending.lock().await;
        for (_uri, handle) in pending.drain() {
            handle.abort();
        }
        Ok(())
    }
}
