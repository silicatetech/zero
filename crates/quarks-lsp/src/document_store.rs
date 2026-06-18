// SPDX-License-Identifier: AGPL-3.0-or-later
use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tower_lsp::lsp_types::Url;

/// Per-URI in-memory document state.
#[derive(Debug, Clone)]
pub struct DocumentState {
    pub content: String,
    pub version: i32,
}

/// Thread-safe document store keyed by URI.
///
/// Internally uses `tokio::sync::RwLock` to allow concurrent reads from
/// `publishDiagnostics` while updates come in from `didChange`.
pub struct DocumentStore {
    inner: Arc<RwLock<HashMap<Url, DocumentState>>>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Clone the inner Arc — both handles access the same store.
    /// This is not a deep clone; it creates a shared reference.
    pub fn clone_handle(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }

    pub async fn insert(&self, uri: Url, content: String, version: i32) {
        let mut store = self.inner.write().await;
        store.insert(uri, DocumentState { content, version });
    }

    pub async fn get(&self, uri: &Url) -> Option<DocumentState> {
        let store = self.inner.read().await;
        store.get(uri).cloned()
    }

    pub async fn remove(&self, uri: &Url) {
        let mut store = self.inner.write().await;
        store.remove(uri);
    }
}

impl Default for DocumentStore {
    fn default() -> Self {
        Self::new()
    }
}
