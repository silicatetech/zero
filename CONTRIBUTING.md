# Contributing to Zero

Thanks for your interest in contributing! Zero is a from-scratch,
agent-native operating system kernel. This guide explains what you need to
know before opening a pull request.

## License & Contributor License Agreement (CLA)

Zero is dual-licensed:

- **Community edition** — [AGPL-3.0-or-later](LICENSE)
- **Enterprise edition** — separate commercial license

To support this model, **every contributor must sign the project's
[Contributor License Agreement](CLA.md)** before their first pull request can
be merged. The CLA bot will comment on your PR with a one-time signing link;
once you sign, the check turns green automatically. You retain full ownership
of your contribution — the CLA only grants the Maintainer the right to
relicense it under both the AGPL and the commercial license.

By submitting a contribution you also affirm it is your own work (or that you
have the right to submit it) and that it may be distributed under the terms
above.

## Getting started

1. Fork the repository and create a topic branch off `main`.
2. Read [`docs/codebase-guide.md`](docs/codebase-guide.md) for a map of how the
   code is organized and where the "sacred" (bit-exact) boundaries are.
3. Make your change in small, reviewable commits.

## Build & test gates

A change must pass the following before it can be merged. Please run them
locally first:

```bash
# 1. Host-side Rust workspace — builds and tests
cargo build --workspace
cargo test --workspace

# 2. Python tooling tests (SilicatePack writer)
python3 tools/test_silicatepack.py

# 3. Both bare-metal kernel builds must compile
make kernel           # x86_64-unknown-none
make build-aarch64    # aarch64-unknown-none
```

Notes:

- The `kernel/` and `boot/` crates are `no_std`/bare-metal and are **excluded
  from the workspace** — they are built via the `Makefile`, not
  `cargo build` at the workspace root.
- The kernel's host-compilable unit tests (`smp`, `net::ice`, `net::tcp`,
  `drivers::nvme_probe`, `weight_layout`) **do** run under `cargo test
  --workspace`: the `crates/kernel-tests/` harness includes those modules
  verbatim via `#[path]` and shims out the bare-metal surfaces, so all 43
  tests execute on the host. The kernel sources stay unmodified. Tests that
  genuinely need hardware (the bare-metal NIC bring-up for `ice`/E810 and
  `i40e`) still run only on target.
- The AVX-512 inference path can only be exercised on x86_64 hardware; it is
  not testable on an ARM dev machine.

## Pull request expectations

- **Scope:** keep PRs focused on a single change. Unrelated cleanups belong in
  their own PR.
- **Determinism:** the Boot-LLM has a canonical cross-platform **Token-ID 25**
  β-anchor. Changes to inference, quantization, memory layout, or the
  `.smodel` loader must preserve bit-exact output unless the PR explicitly
  documents and justifies a change to the anchor.
- **`unsafe`:** new `unsafe` blocks (MMIO, raw pointers) should carry a
  `// SAFETY:` comment explaining why they are sound.
- **Style:** match the surrounding code — naming, comment density, and idiom.
  Keep comments in English.
- **Description:** explain *what* changed and *why*. If you change behavior,
  say how you verified it.

## Reporting issues

When filing a bug, include the architecture (x86_64 / aarch64), how you built
(QEMU vs. bare metal), and the serial/console output. For security-sensitive
reports, contact the maintainer privately rather than opening a public issue.

We appreciate every contribution — thank you for helping build Zero.
