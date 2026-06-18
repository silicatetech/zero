#!/usr/bin/env python3
"""Sub-MP-D4 Task A0: Offline Qwen3 vocab extraction for kernel embedding.

Produces two static binary artifacts:
  - vocab_bytes.bin: concatenated UTF-8 byte sequences of all tokens
  - vocab_offsets.bin: u32 LE array of byte offsets [vocab_size + 1 entries]

Per V3.1 Pillar 1 (zero runtime IO) + Pillar 7 (canonical shared data).
Per Lesson 17 canonical: point-of-consumption discipline.
"""
import sys
import struct
import json
from pathlib import Path


def build_byte_decoder():
    """Build GPT-2 family byte_decoder mapping (inverse of byte_encoder).
    
    Qwen3 uses the same byte-level BPE encoding as GPT-2:
    - Printable ASCII and some Latin-1 chars map to themselves
    - Other bytes (0-32, 127-160, etc.) are mapped to Unicode chars starting at 256
    
    This function builds the reverse mapping: Unicode char → original byte value.
    """
    # GPT-2 byte_encoder: maps byte values to Unicode chars
    bs = list(range(ord('!'), ord('~') + 1))
    bs += list(range(ord('¡'), ord('¬') + 1))
    bs += list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_decoder = {chr(c): b for b, c in zip(bs, cs)}
    return byte_decoder


def token_string_to_bytes(token_str, byte_decoder):
    """Convert a BPE token string to its actual UTF-8 byte sequence.
    
    Each character in the token string is looked up in byte_decoder
    to get the original byte value. The resulting bytes are then
    the actual content this token represents.
    """
    raw_bytes = bytes([byte_decoder[c] for c in token_str if c in byte_decoder])
    return raw_bytes


def main():
    if len(sys.argv) < 4:
        print("Usage: extract_vocab.py <tokenizer.json> <vocab_bytes.bin> <vocab_offsets.bin>",
              file=sys.stderr)
        sys.exit(1)

    tokenizer_path = Path(sys.argv[1])
    bytes_out = Path(sys.argv[2])
    offsets_out = Path(sys.argv[3])

    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer from {tokenizer_path}...", file=sys.stderr)

    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    # Extract vocabulary: token_string -> token_id
    vocab = tokenizer_data.get('model', {}).get('vocab', {})
    if not vocab:
        print("ERROR: No vocab found in tokenizer.json model section", file=sys.stderr)
        sys.exit(1)

    # Also check added_tokens for special tokens
    added_tokens = tokenizer_data.get('added_tokens', [])
    added_map = {t['id']: t['content'] for t in added_tokens}

    vocab_size = max(max(vocab.values()), max(added_map.keys(), default=0)) + 1
    print(f"Vocab size: {vocab_size}", file=sys.stderr)

    # Build byte decoder
    byte_decoder = build_byte_decoder()

    # Build token_id -> bytes mapping
    # Invert vocab: token_id -> token_string
    id_to_string = {}
    for token_str, token_id in vocab.items():
        id_to_string[token_id] = token_str

    # Override with added_tokens (special tokens)
    for token_id, token_str in added_map.items():
        id_to_string[token_id] = token_str

    # Convert each token to bytes
    bytes_buffer = bytearray()
    offsets = [0]

    for tid in range(vocab_size):
        token_str = id_to_string.get(tid, '')
        if token_str and tid in added_map:
            # Special tokens: encode directly as UTF-8 (they are literal strings like <|endoftext|>)
            token_bytes = token_str.encode('utf-8')
        elif token_str:
            # Regular BPE tokens: apply byte_decoder
            token_bytes = token_string_to_bytes(token_str, byte_decoder)
        else:
            # Unknown token ID: empty bytes
            token_bytes = b''

        bytes_buffer.extend(token_bytes)
        offsets.append(len(bytes_buffer))

    assert len(offsets) == vocab_size + 1

    # Sanity checks
    def get_token_bytes(tid):
        start = offsets[tid]
        end = offsets[tid + 1]
        return bytes(bytes_buffer[start:end])

    # Token-ID 9707 should be "Hello"
    t9707 = get_token_bytes(9707)
    print(f"Token-ID 9707: {t9707!r} → '{t9707.decode('utf-8', errors='replace')}'", file=sys.stderr)

    # Token-ID 25 (β-anchor next token)
    t25 = get_token_bytes(25)
    print(f"Token-ID 25:   {t25!r} → '{t25.decode('utf-8', errors='replace')}'", file=sys.stderr)

    # Token-ID 4337 (" World")
    t4337 = get_token_bytes(4337)
    print(f"Token-ID 4337: {t4337!r} → '{t4337.decode('utf-8', errors='replace')}'", file=sys.stderr)

    # Sub-MP-D3 generated tokens sample
    for tid in [271, 40, 2776, 264, 4285]:
        tb = get_token_bytes(tid)
        print(f"Token-ID {tid}: {tb!r} → '{tb.decode('utf-8', errors='replace')}'", file=sys.stderr)

    # Full Sub-MP-D3 prompt decode test
    prompt_tokens = [9707, 4337, 358, 2776, 279, 1156, 444, 10994, 4303, 389, 60792, 5251, 21323]
    prompt_text = b''.join(get_token_bytes(t) for t in prompt_tokens)
    print(f"\nFull prompt decode: '{prompt_text.decode('utf-8', errors='replace')}'", file=sys.stderr)

    # Full Sub-MP-D3 generated decode test
    gen_tokens = [271, 40, 2776, 264, 4285, 2025, 429, 23473, 330, 9707,
                  4337, 1, 389, 279, 4171, 382, 40, 2776, 537, 264,
                  1931, 1697, 11, 358, 2776, 264, 2025, 382, 40, 2776, 537, 264]
    gen_text = b''.join(get_token_bytes(t) for t in gen_tokens)
    print(f"Full generated decode: '{gen_text.decode('utf-8', errors='replace')}'", file=sys.stderr)

    # Write output files
    with open(bytes_out, 'wb') as f:
        f.write(bytes(bytes_buffer))

    with open(offsets_out, 'wb') as f:
        for offset in offsets:
            f.write(struct.pack('<I', offset))

    total_size = len(bytes_buffer) + len(offsets) * 4
    print(f"\nOutput:", file=sys.stderr)
    print(f"  {bytes_out}: {len(bytes_buffer)} bytes ({len(bytes_buffer)/1024:.1f} KB)", file=sys.stderr)
    print(f"  {offsets_out}: {len(offsets)*4} bytes ({len(offsets)*4/1024:.1f} KB)", file=sys.stderr)
    print(f"  Combined: {total_size/1024/1024:.2f} MiB", file=sys.stderr)
    print(f"  VOCAB_SIZE = {vocab_size}", file=sys.stderr)


if __name__ == "__main__":
    main()
