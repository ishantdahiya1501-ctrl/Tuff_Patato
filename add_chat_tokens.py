"""Add chat special tokens to the TinyStories BPE tokenizer (no BPE retraining).

Loads data/tinystories/tokenizer.json, appends the special tokens
<|user|>, <|assistant|>, <|end|> to the vocabulary, and saves the result as
data/chat/tokenizer_chat.json (vocab 4096 -> 4099).
"""

import argparse
import os

from tokenizers import Tokenizer

SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]


def main():
    parser = argparse.ArgumentParser(description="Add chat special tokens to the BPE tokenizer")
    parser.add_argument("--src", default="data/tinystories/tokenizer.json")
    parser.add_argument("--dst", default="data/chat/tokenizer_chat.json")
    args = parser.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit(f"Tokenizer not found: {args.src}")
    tokenizer = Tokenizer.from_file(args.src)
    old_vocab = tokenizer.get_vocab_size()

    added = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    missing = [t for t in SPECIAL_TOKENS if tokenizer.token_to_id(t) is None]
    if missing:
        raise SystemExit(f"Failed to add special tokens: {missing}")

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    tokenizer.save(args.dst)
    print(f"Added {added} special token(s): {', '.join(SPECIAL_TOKENS)}")
    print(f"New vocab size: {tokenizer.get_vocab_size()} ({old_vocab} -> {tokenizer.get_vocab_size()})")

    # Sanity check: the special tokens must survive an encode/decode round trip
    # and be encoded as single dedicated ids (not merged BPE pieces).
    sample = ("<|user|>Write a story about a dog.<|assistant|>"
              "Once upon a time there was a dog.<|end|>")
    enc = tokenizer.encode(sample)
    for token in SPECIAL_TOKENS:
        assert tokenizer.token_to_id(token) in enc.ids, f"{token} not encoded as one id"
    assert tokenizer.decode(enc.ids, skip_special_tokens=False) == sample, "round-trip failed"
    print(f"Saved chat tokenizer to {args.dst} (round-trip OK)")


if __name__ == "__main__":
    main()
