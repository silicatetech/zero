# Kimi K2.6 vocabulary stubs

`kimi_vocab_bytes.bin` and `kimi_vocab_offsets.bin` in this directory
are **placeholder stubs** included so that
`cargo check --features kimi-k26-vocab` succeeds on a fresh checkout.

The real Kimi K2.6 / Llama-3-family vocabulary (128 256 tokens) must
be generated offline from the model's `tokenizer.json`. The kernel
detokenizer falls back to printing raw token IDs on the serial console
when the offset table is too small to back a real lookup (which is the
case for these stubs), so the boot path still works end-to-end — only
the rendered text is missing.

## How to regenerate

```bash
# 1. Download tokenizer.json from the Kimi K2.6 model repo:
huggingface-cli download moonshotai/Kimi-K2.6 tokenizer.json \
    --local-dir /tmp/kimi-k26-tok

# 2. Run the existing extractor (the same one that produced the Qwen3
#    vocab_*.bin pair):
python3 tools/extract_vocab.py \
    --tokenizer /tmp/kimi-k26-tok/tokenizer.json \
    --out-bytes  kernel/src/kimi_vocab_bytes.bin \
    --out-offsets kernel/src/kimi_vocab_offsets.bin \
    --expected-vocab-size 128256

# 3. Sanity-check sizes:
#    * offsets file: exactly (128256 + 1) × 4 = 513 028 bytes
#    * bytes file:   a few hundred KiB to a few MiB, depending on
#                    UTF-8 width of the tokens.

# 4. Build:
make image-cherry FEATURES=avx512-acceleration,cherry-net,kimi-k26-vocab
```

`extract_vocab.py` is the same script `kernel/src/vocab_bytes.bin`
(Qwen3) was built with. If a `--expected-vocab-size` flag isn't yet
plumbed through, just verify by hand that the offset count comes out
to 128 257.

## Why not commit the real binaries

The full Kimi vocab adds ≈ 1 MiB to the kernel binary and the repo,
and would lock the deployment to one tokenizer revision. The stub
keeps the build green without forcing a checkout of files most
contributors don't need; the operator regenerates them once per
deploy.
