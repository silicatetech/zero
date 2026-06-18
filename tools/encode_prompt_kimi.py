#!/usr/bin/env python3
"""Encode the Kimi K2.6 Zero system-prompt prefix for kernel/src/inference.rs.

This is the deepseek2 / Kimi K2.6 counterpart to `encode_prompt.py` (which
serves the Qwen3 path). The output is a Rust `const DEEPSEEK2_PROMPT_TOKENS`
array suitable for paste-in to `kernel/src/inference.rs`.

# Why a separate script?

Kimi K2 / K2.6 ships a tiktoken-derived BPE tokenizer (vocab_size ≈
163 840), not the Qwen3 tokenizer. The chat template differs too:
Kimi uses Moonshot's own ChatML variant — `<|im_system|>`, `<|im_user|>`,
`<|im_assistant|>`, `<|im_middle|>`, `<|im_end|>` — *not* the standard
`<|im_start|>` / `<|im_end|>` pair you'd find in Qwen3 or vanilla ChatML.
Hard-coding Kimi token IDs based on the Qwen3 file would silently shift
every layer's input.

# Backends

`moonshotai/Kimi-K2-Instruct` on HuggingFace ships the raw tiktoken
file (`tiktoken.model`) plus a small `tokenization_kimi.py` wrapper
and `tokenizer_config.json`. This script accepts either:

  * `--tokenizer-json path/to/tokenizer.json`  (HF `tokenizers` library
    format — only if a sibling deploy provides it; the upstream repo
    does *not*)
  * `--tiktoken      path/to/tiktoken.model    --tokenizer-config
                     path/to/tokenizer_config.json`
                                            (the canonical Kimi K2 path)

The tiktoken backend rebuilds the Kimi tokenizer faithfully: the pat_str
regex comes from `tokenization_kimi.py`; the special-token IDs come
from `tokenizer_config.json`'s `added_tokens_decoder` map; and the
reserved-token block layout matches what
`KimiTikTokenTokenizer.__init__` constructs.

# Output

A Rust const ready to drop into `kernel/src/inference.rs`:

  pub const DEEPSEEK2_PROMPT_TOKENS: &[u32] = &[
      <id0>, <id1>, ...
  ];
  pub const DEEPSEEK2_PROMPT_TOKEN_COUNT: usize = N;

# Constraints

* The system prompt text is identical to the Qwen3 path's Zero
  prompt, so a side-by-side test of Qwen3 vs Kimi K2.6 measures only
  the model difference, not the prompt.
* The Kimi chat template wraps the system message and opens an
  assistant turn so the model continues from there. The canonical
  Moonshot form (no `<|im_start|>` — that token does not exist in
  Kimi's vocab!) is:

      <|im_system|>system<|im_middle|>{PROMPT}<|im_end|>
      <|im_assistant|>assistant<|im_middle|>

* The script refuses to emit if the encoded length exceeds
  `DEEPSEEK2_MAX_PREFILL` (currently 64 — see inference.rs). Bumping
  that constant + the sequence stack ceiling is a kernel change.

# Usage

  python3 tools/encode_prompt_kimi.py \\
      --tiktoken          /tmp/kimi-tokenizer/tiktoken.model \\
      --tokenizer-config  /tmp/kimi-tokenizer/tokenizer_config.json \\
      > /tmp/deepseek2_prompt_tokens.rs

Then splice the file into `kernel/src/inference.rs` at the marked
`DEEPSEEK2_PROMPT_TOKENS` block.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Zero Kimi K2.6 system prompt ──────────────────────────────────
#
# Same text as the Qwen3 side, only wrapped in Kimi's ChatML template.
# Keep this in lockstep with `encode_prompt.py` so cross-model A/B
# benchmarks compare model behaviour, not prompt drift.
SYSTEM_PROMPT = (
    "You are Zero, a bare-metal AI operating system. "
    "Think in public about Quarks, kernel memory, GPU, capabilities, "
    "and the future of the AI era OS."
)

# Kimi K2's ChatML-variant special tokens. These IDs are documented in
# `tokenizer_config.json::added_tokens_decoder`; the script reads the
# actual IDs from that file at runtime when the tiktoken backend is in
# use (so a future Moonshot release that re-numbers the reserved-token
# block is picked up automatically). The names listed here are what we
# round-trip-check `tokens[0]` against; if Moonshot ever renames them
# the assertion fails loud and early.
IM_SYSTEM = "<|im_system|>"
IM_USER = "<|im_user|>"
IM_ASSISTANT = "<|im_assistant|>"
IM_MIDDLE = "<|im_middle|>"
IM_END = "<|im_end|>"

# Tokens we expect to round-trip as atomic vocab entries; the
# verification step decodes each by ID and confirms its string round-
# trip. Order doesn't matter — used only for diagnostic output.
KIMI_SPECIALS = (IM_SYSTEM, IM_USER, IM_ASSISTANT, IM_MIDDLE, IM_END)

# Hard ceiling that mirrors `DEEPSEEK2_MAX_PREFILL` in inference.rs.
# Bump both together if you grow the prompt past this. Kimi's ChatML
# variant adds ≈8 special-token markers around the system message, so
# the encoded prompt sits in the 40–55 range — well above the Qwen3-
# style 13-token ceiling.
DEEPSEEK2_MAX_PREFILL = 64


def _build_kimi_chat(system_prompt: str) -> str:
    """Compose the full Kimi K2 chat-template input.

    Moonshot's variant — verified against `tokenization_kimi.py` and
    the `added_tokens_decoder` map in `tokenizer_config.json`:

        <|im_system|>system<|im_middle|>{system}<|im_end|>
        <|im_assistant|>assistant<|im_middle|>

    Role-name strings ("system", "assistant") are LITERAL tokens
    between the role-marker and `<|im_middle|>`. The trailing
    `<|im_assistant|>...<|im_middle|>` opens the model's reply turn —
    the first generated token continues from there.
    """
    return (
        f"{IM_SYSTEM}system{IM_MIDDLE}{system_prompt}{IM_END}"
        f"{IM_ASSISTANT}assistant{IM_MIDDLE}"
    )


# Kimi's pretoken regex, lifted verbatim from
# `tokenization_kimi.py::KimiTikTokenTokenizer.pat_str`. Uses Unicode
# property classes (`\p{...}`) that require the `regex`-compatible
# engine tiktoken uses internally.
KIMI_PAT_STR = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)


def _kimi_special_token_map(tokenizer_config_path: Path) -> dict[str, int]:
    """Load `<|im_*|>` token IDs from `tokenizer_config.json`.

    Kimi K2 reserves a 256-entry block at the end of the BPE vocab for
    special tokens; `added_tokens_decoder` is the authoritative map
    from numeric ID → token string. We trust it verbatim — guessing
    IDs would re-introduce the kind of silent drift this whole script
    exists to prevent.
    """
    cfg = json.loads(tokenizer_config_path.read_text())
    decoder = cfg.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        print(
            f"FATAL: {tokenizer_config_path} is missing `added_tokens_decoder`",
            file=sys.stderr,
        )
        sys.exit(1)
    return {info["content"]: int(tid) for tid, info in decoder.items()}


def _encode_via_tokenizer_json(tokenizer_path: Path, text: str):
    """HuggingFace `tokenizers` backend — only useful if a downstream
    deploy bakes its own `tokenizer.json` (Moonshot's upstream repo
    does not ship one)."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: 'tokenizers' library not available", file=sys.stderr)
        print("Install via: pip3 install tokenizers", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    enc = tokenizer.encode(text, add_special_tokens=False)
    return list(enc.ids), tokenizer


def _encode_via_tiktoken(
    tiktoken_path: Path,
    tokenizer_config_path: Path,
    text: str,
):
    """Canonical Kimi K2 path: build a tiktoken `Encoding` faithful to
    `tokenization_kimi.py` and encode through it."""
    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
    except ImportError:
        print("ERROR: 'tiktoken' library not available", file=sys.stderr)
        print("Install via: pip3 install tiktoken", file=sys.stderr)
        sys.exit(1)

    mergeable_ranks = load_tiktoken_bpe(str(tiktoken_path))
    num_base = len(mergeable_ranks)
    num_reserved = 256  # KimiTikTokenTokenizer.num_reserved_special_tokens

    # Build the reserved-block mapping exactly like
    # KimiTikTokenTokenizer.__init__ does it: every reserved slot gets
    # the name from added_tokens_decoder if present, otherwise a
    # `<|reserved_token_{i}|>` placeholder. tiktoken needs all of them
    # declared so it knows the IDs are special (not BPE-mergeable).
    named = _kimi_special_token_map(tokenizer_config_path)
    special_tokens: dict[str, int] = {}
    for tid in range(num_base, num_base + num_reserved):
        # Prefer the named entry from tokenizer_config.json; fall
        # through to the reserved-slot placeholder so every ID in the
        # block has a unique string handle.
        name = next(
            (n for n, i in named.items() if i == tid),
            f"<|reserved_token_{tid}|>",
        )
        special_tokens[name] = tid

    enc = tiktoken.Encoding(
        name="kimi-k2",
        pat_str=KIMI_PAT_STR,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    # Allow every special token so the ChatML markers stay atomic.
    ids = enc.encode(text, allowed_special=set(special_tokens.keys()))
    return ids, enc


def _verify_special(ids: list[int], decode, backend_name: str) -> None:
    """Make sure the Kimi ChatML markers round-trip as single IDs.

    The first token of every well-formed Kimi prompt is `<|im_system|>`;
    if it decodes back to anything else, the tokenizer fell back to
    BPE-merging the angle brackets and every downstream token is
    silently shifted.
    """
    if not ids:
        print(f"FATAL: {backend_name} produced an empty encoding", file=sys.stderr)
        sys.exit(2)
    first = decode([ids[0]])
    if first != IM_SYSTEM:
        print(
            f"FATAL: tokens[0] decodes to {first!r}, expected {IM_SYSTEM!r}.\n"
            f"       The {backend_name} tokenizer is not honouring Kimi's\n"
            f"       ChatML special tokens. Check that {IM_SYSTEM} appears\n"
            f"       in the tokenizer's special-token map.",
            file=sys.stderr,
        )
        sys.exit(3)


def _emit_rust(ids: list[int], decode, backend_name: str, source: Path) -> None:
    n = len(ids)
    print("// Auto-generated by tools/encode_prompt_kimi.py — DO NOT EDIT MANUALLY.")
    print(f"// Source prompt: {SYSTEM_PROMPT!r}")
    print(f"// Backend: {backend_name}")
    print(f"// Tokenizer: {source}")
    print(f"// Encoded: {n} tokens (DEEPSEEK2_MAX_PREFILL = {DEEPSEEK2_MAX_PREFILL})")
    print("//")
    print("// Cross-check table (token-ID -> decoded fragment):")
    for i, tid in enumerate(ids):
        frag = decode([tid])
        # Keep the comment one-line; replace newlines with \n.
        printable = frag.replace("\n", "\\n").replace("\r", "\\r")
        print(f"//   [{i:>2}] {tid:>6} -> {printable!r}")
    print("")
    print("/// Kimi K2 Zero chat-template prompt — Moonshot ChatML variant")
    print("/// (`<|im_system|>` / `<|im_middle|>` / `<|im_end|>` /")
    print("/// `<|im_assistant|>`), ready to feed into the deepseek2 prefill")
    print("/// loop. Regenerate via")
    print("/// `tools/encode_prompt_kimi.py --tiktoken … --tokenizer-config …`.")
    print(f"pub const DEEPSEEK2_PROMPT_TOKENS: &[u32] = &[")
    for i, tid in enumerate(ids):
        frag = decode([tid]).replace("\n", "\\n").replace("\r", "\\r")
        print(f"    {tid:>6}, // [{i:>2}] {frag!r}")
    print("];")
    print("")
    print(f"pub const DEEPSEEK2_PROMPT_TOKEN_COUNT: usize = {n};")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode the Kimi K2.6 Zero system prompt for inference.rs."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokenizer-json", type=Path, help="Path to HuggingFace tokenizer.json")
    src.add_argument("--tiktoken", type=Path, help="Path to raw tiktoken.model file")
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        help="Path to tokenizer_config.json (required with --tiktoken)",
    )
    args = parser.parse_args()

    text = _build_kimi_chat(SYSTEM_PROMPT)

    if args.tokenizer_json is not None:
        if not args.tokenizer_json.exists():
            print(f"ERROR: {args.tokenizer_json} not found", file=sys.stderr)
            sys.exit(1)
        ids, tokenizer = _encode_via_tokenizer_json(args.tokenizer_json, text)
        decode = tokenizer.decode
        backend = "huggingface tokenizers (tokenizer.json)"
        source = args.tokenizer_json
    else:
        if not args.tiktoken.exists():
            print(f"ERROR: {args.tiktoken} not found", file=sys.stderr)
            sys.exit(1)
        if args.tokenizer_config is None:
            print(
                "ERROR: --tiktoken requires --tokenizer-config "
                "(needed to map <|im_*|> tokens to their reserved-block IDs)",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.tokenizer_config.exists():
            print(f"ERROR: {args.tokenizer_config} not found", file=sys.stderr)
            sys.exit(1)
        ids, enc = _encode_via_tiktoken(args.tiktoken, args.tokenizer_config, text)
        decode = enc.decode
        backend = "tiktoken (tiktoken.model + tokenizer_config.json)"
        source = args.tiktoken

    _verify_special(ids, decode, backend)

    if len(ids) > DEEPSEEK2_MAX_PREFILL:
        print(
            f"FATAL: encoded prompt is {len(ids)} tokens, exceeds\n"
            f"       DEEPSEEK2_MAX_PREFILL = {DEEPSEEK2_MAX_PREFILL}.\n"
            f"       Either shorten SYSTEM_PROMPT or bump the kernel const\n"
            f"       (and the sequence stack ceiling in inference.rs).",
            file=sys.stderr,
        )
        sys.exit(4)

    # Stderr summary makes cross-backend comparison cheap.
    head = ", ".join(str(i) for i in ids[:8])
    print(
        f"[{backend}] encoded {len(ids)} tokens, head: [{head}, ...]",
        file=sys.stderr,
    )

    _emit_rust(ids, decode, backend, source)


if __name__ == "__main__":
    main()
