# CLA Bot

Self-contained Contributor License Agreement enforcement for this
repository. **No external service** — only GitHub Actions.

## Files

| File | Purpose |
|------|---------|
| [`../CLA.md`](../CLA.md) | The CLA text (Apache-style Individual CLA, adapted for Silicate AGPLv3 + commercial dual-licensing). |
| [`workflows/cla-check.yml`](workflows/cla-check.yml) | The Action. Checks PRs, records signatures, publishes the `license/cla` status. |
| [`cla-signatures.json`](cla-signatures.json) | The signature ledger. Updated automatically by the bot. |

## Flow

1. A contributor opens a PR.
2. The bot looks up the author in `cla-signatures.json`.
   - **Signed** → publishes a passing `license/cla` commit status.
   - **Not signed** → publishes a failing status and posts a one-time
     comment with a link to `CLA.md` and signing instructions.
3. The contributor signs by **commenting on the PR**:
   > I have read the CLA and I agree
4. The bot appends them to `cla-signatures.json` (committed to the
   default branch), flips the `license/cla` status to passing, and
   thanks them. Signing is once-per-contributor and covers all future
   PRs.

Bots (`*[bot]`) are exempt and never need to sign.

## One-time setup: make it blocking

The Action publishes a commit status named **`license/cla`** but cannot
*enforce* it by itself. To block merges until the CLA is signed, add it
as a required status check in branch protection:

**Settings → Branches → Branch protection rules → `main`**
→ *Require status checks to pass before merging* → add **`license/cla`**.

(In a terminal: this must be done in the GitHub web UI or via
`gh api`; it is not part of the repo.)

## Updating the CLA text

Bump the `Version` line in `CLA.md` and the `cla_version` field in
`cla-signatures.json`. Existing signatures keep the version they signed.
If a new version requires re-signing, clear the `signatures` array (or
filter by `cla_version`) — the check compares usernames only, so any
contributor not present must sign again.
