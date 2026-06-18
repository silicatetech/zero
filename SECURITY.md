# Security Policy

We take the security of Zero seriously. Thank you for helping keep the
project and its users safe.

## Reporting a Vulnerability

**Do NOT open a public issue for security vulnerabilities.**

Please report any security issue privately by emailing:

**Tom.Stuhl@nefesh.ai**

Include as much detail as you can so we can reproduce and assess the issue:

- A description of the vulnerability and its potential impact
- The affected component and, if known, the affected version/commit
- Step-by-step reproduction instructions or a proof of concept
- Relevant architecture (x86_64 / aarch64) and how you built/ran (QEMU vs.
  bare metal), plus any serial/console output

You can expect an initial acknowledgement of your report, and we will keep you
informed as we work on a fix.

## Scope

Security reports are in scope for the core components of Zero:

- **Kernel** (`kernel/`)
- **Inference engine** (the Boot-LLM forward pass and quantized inference paths)
- **Boot** (`boot/`)
- **Network stack** (`net/`, including the `ice`/E810 and `i40e` drivers)

## Disclosure

Please give us a reasonable opportunity to investigate and release a fix before
any public disclosure. We will coordinate timing with you.

## Links

- Website: [silicate.tech](https://silicate.tech)
- X / Twitter: [@silicate_tech](https://x.com/silicate_tech)
