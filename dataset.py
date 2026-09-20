"""Tokenize TinyStories into a flat uint16 token stream, cached as .npy.

Each story is encoded and followed by an EOS token, then everything is
concatenated and later chunked into fixed-length sequences for causal LM.
"""

import argparse
import os

import numpy as np
from tqdm import tqdm

from tokenizer import eos_id, iter_tinystories, load_tokenizer

DTYPE = np.uint16  # vocab 4096 fits comfortably in uint16


def _encode_batch(tokenizer, texts, eos: int) -> np.ndarray:
    ids = []
    for encoding in tokenizer.encode_batch(texts):
        ids.extend(encoding.ids)
        ids.append(eos)
    return np.asarray(ids, dtype=DTYPE)


def tokenize_texts(tokenizer, texts, total=None, desc="tokenizing") -> np.ndarray:
    eos = eos_id(tokenizer)
    vocab = tokenizer.get_vocab_size()
    if vocab > np.iinfo(DTYPE).max + 1:
        raise ValueError(f"vocab size {vocab} does not fit in {DTYPE.__name__}")
    parts, batch = [], []
    for text in tqdm(texts, total=total, desc=desc, unit="story"):
        batch.append(text)
        if len(batch) >= 1000:
            parts.append(_encode_batch(tokenizer, batch, eos))
            batch = []
    if batch:
        parts.append(_encode_batch(tokenizer, batch, eos))
    if not parts:
        return np.zeros(0, dtype=DTYPE)
    return np.concatenate(parts)


def prepare_tokens(data_dir, tokenizer_path, split="train", max_stories=None, force=False):
    """Return the packed token array for a split, tokenizing + caching on first use."""
    cache_path = os.path.join(data_dir, f"{split}_tokens.npy")
    if os.path.exists(cache_path) and not force:
        tokens = np.load(cache_path, mmap_mode="r")
        print(f"Loaded cached tokens for '{split}': {len(tokens):,} tokens ({cache_path})")
        return tokens

    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Tokenizing TinyStories '{split}' split with {tokenizer_path} ...")
    tokens = tokenize_texts(
        tokenizer, iter_tinystories(split, max_stories), desc=f"tokenize {split}"
    )
    os.makedirs(data_dir, exist_ok=True)
    np.save(cache_path, tokens)
    print(f"Saved {len(tokens):,} tokens to {cache_path}")
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize TinyStories into cached .npy token streams"
    )
    parser.add_argument("--data_dir", default="data/tinystories")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer.json (default: <data_dir>/tokenizer.json)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_stories", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-tokenize even if cached")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or os.path.join(args.data_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise SystemExit(
            f"Tokenizer not found at {tokenizer_path}.\n"
            f"Run first: python tokenizer.py --data_dir {args.data_dir} --vocab_size 4096"
        )
    tokens = prepare_tokens(args.data_dir, tokenizer_path, args.split,
                            args.max_stories, args.force)
    print(f"'{args.split}': {len(tokens):,} tokens "
          f"(~{(len(tokens) - 1) // 256:,} sequences of 256)")


if __name__ == "__main__":
    main()
