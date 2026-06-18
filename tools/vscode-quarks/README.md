# vscode-quarks

VS Code extension for Quarks `.qk` files. Connects to the
`quarks-lsp` language server via JSON-RPC over stdio and displays
diagnostics inline.

**Status:** Development extension (Stage 8 Phase 1 deliverable). Not
published to the VS Code Marketplace.

## Prerequisites

- Node.js 18+ and npm
- VS Code 1.80+
- `quarks-lsp` binary built: `cargo build -p quarks-lsp` from
  the repo root. Binary will be at `target/debug/quarks-lsp`.

## Install (development workflow)

From this directory (`tools/vscode-quarks/`):

```bash
npm install
npm run compile
```

This produces `out/extension.js`.

To load the extension in VS Code for development:

1. Open `tools/vscode-quarks/` as a VS Code workspace
2. Press `F5` to launch the Extension Development Host
3. In the new VS Code window, set `quarks.serverPath` to the
   absolute path of your compiled `quarks-lsp` binary:
   - Settings → Extensions → Quarks → Server Path
   - e.g. `/path/to/target/debug/quarks-lsp`
4. Open any `.qk` file in the dev host to activate the extension

## Install (VSIX for external use)

To package as a `.vsix`:

```bash
npm install -g @vscode/vsce
vsce package
```

Install the resulting `.vsix`:

```bash
code --install-extension vscode-quarks-0.1.0.vsix
```

External users still need the `quarks-lsp` binary on PATH or
configured via settings.

## Manual test scenarios

After installing and configuring the server path, open each fixture
file and verify the expected behavior.

### Scenario 1: Valid source

File: `test-fixtures/valid.qk`

**Expected:**
- No red squiggles
- No diagnostics in Problems panel

### Scenario 2: Lex error

File: `test-fixtures/type-mismatch.qk`

**Expected:**
- Red squiggle precisely under the offending character
- Problems panel shows one error from quarks-frontend
- Diagnostic source shows "quarks-frontend"

### Scenario 3: Type-check error

File: `test-fixtures/nested-error.qk`

**Expected:**
- Red squiggle under the undefined identifier
- Problems panel shows the type-check error
- Span points to the exact identifier, not the surrounding expression

### Scenario 4: Parse error

File: `test-fixtures/parse-error.qk`

**Expected:**
- Diagnostic in Problems panel for missing semicolon or unclosed brace
- Red squiggle at or near the syntax error position
- Source-language frontend provides direct byte spans, so positioning
  is precise (no fallback to file-start)

### Scenario 5: Debounce verification

Open `test-fixtures/valid.qk` with the Output panel visible
(View → Output → "Quarks Language Server" channel, enable trace
via `quarks.trace.server: "verbose"`).

Edit the file rapidly — hold down a key to produce 20–50 edits within
1 second. Then stop editing.

**Expected:**
- Many `textDocument/didChange` requests in the trace
- Only ONE `textDocument/publishDiagnostics` notification after you
  stop editing (debounce fires 200ms after last edit)
- If you see multiple `publishDiagnostics` during rapid typing:
  debounce is broken — regression, file a bug

## Troubleshooting

**Extension activates but no diagnostics appear:**

1. Check `quarks.serverPath` is set and the binary exists
2. Check binary is executable: `chmod +x <path>`
3. Enable trace: set `quarks.trace.server` to `"verbose"` and
   check Output → "Quarks Language Server"
4. Check the server's stderr — tower-lsp routes stderr to VS Code's
   Output panel

**Server crashes immediately:**

- stdout contamination will kill the LSP connection. All `quarks-lsp`
  logs are routed through stderr; if something writes to stdout other
  than valid JSON-RPC messages, the protocol breaks. Regression in
  server code — file a bug.

## Stage 8 Phase 1 Migration (Completed)

This extension previously consumed S-expression IR syntax. Stage 8
Phase 1 (MP1–MP6) migrated the entire toolchain to a proper source
language with the following components:

- **MP1–MP3:** Frontend lexer, parser, and source-level type-checker
  in `crates/quarks-frontend`
- **MP4:** Codegen from typed source AST to S-expression IR
- **MP4.1:** `nop` instruction for stack-neutral if-else branches
- **MP5:** LSP engine swap — `DiagnosticsEngine` now runs the
  source-language pipeline (`tokenize → parse → check`) instead of
  validating S-expression IR directly
- **MP6:** This extension's grammar and fixtures migrated to source
  syntax

**Diagnostic shape unchanged:** LSP `Diagnostic.data` still carries
structured JSON for downstream consumers (e.g., M2M agent loops).
The internal schema changed from `list_path` (S-expression AST path)
to `span` (direct byte offsets), but the JSON envelope shape is
stable.

**S-expression IR is no longer user-facing.** The codegen step still
produces S-expression IR for the validator and (eventually) for
compilation to executable form, but users write source code only.
The LSP and editor never see IR.

## Publishing

This extension is not published to the VS Code Marketplace. Publishing
is deferred. Prerequisites for publication:

- Stable publisher ID (current `silicate-dev` is a placeholder)
- Icon assets
- Semver discipline
- CI build for VSIX artifacts
- User-facing documentation beyond this README

Until then, the extension is available for local development only.

## Development notes

- The extension's `out/` directory is gitignored. Users must compile
  before use (see Install section).
- `npm run watch` is available for incremental builds during extension
  development.
- The extension has no runtime state beyond the `LanguageClient`
  instance. Restart via VS Code command "Developer: Reload Window"
  if the server enters a bad state.
