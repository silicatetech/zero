// SPDX-License-Identifier: AGPL-3.0-or-later
use tower_lsp::{LspService, Server};

use quarks_lsp::server::QuarksLsp;

#[tokio::main]
async fn main() {
    // IMPORTANT: logging goes to stderr only.
    // stdout is reserved for LSP JSON-RPC traffic; any stdout write
    // that is not a valid LSP message corrupts the protocol.
    eprintln!("quarks-lsp starting");

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(QuarksLsp::new);
    Server::new(stdin, stdout, socket).serve(service).await;

    eprintln!("quarks-lsp exiting");
}
