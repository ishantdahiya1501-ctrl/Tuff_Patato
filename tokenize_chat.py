"""Encode formatted chat lines into packed uint16 token arrays (95/5 split).

Reads data/chat/formatted_chat.txt (one conversation per line), encodes with
the chat tokenizer, shuffles conversations (seeded), splits 95/5, and saves
data/chat/chat_train_tokens.npy and data/chat/chat_val_tokens.npy.
Each line already ends with <|end|>, which delimits conversations in the
packed stream (no extra EOS is needed).
"""

import argparse
import os

import numpy as np

from tokenizer import load_tokenizer

DTYPE = np.uint16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="data/chat/tokenizer_chat.json")
    parser.add_argument("--inp", default="data/chat/formatted_chat.txt")
    parser.add_argument("--train_out", default="data/chat/chat_train_tokens.npy")
    parser.add_argument("--val_out", default="data/chat/chat_val_tokens.npy")
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for path in (args.tokenizer, args.inp):
        if not os.path.exists(path):
            raise SystemExit(f"Required file not found: {path}")
    tokenizer = load_tokenizer(args.tokenizer)
    vocab = tokenizer.get_vocab_size()
    if vocab > np.iinfo(DTYPE).max + 1:
        raise SystemExit(f"vocab size {vocab} does not fit in {DTYPE.__name__}")

    with open(args.inp, encoding="utf-8") as f:
        lines = [line.strip("\n") for line in f]
    lines = [line for line in lines if line]
    if not lines:
        raise SystemExit(f"No conversations found in {args.inp}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(lines))
    n_val = int(round(len(lines) * args.val_fraction))
    is_val = np.zeros(len(lines), dtype=bool)
    is_val[order[:n_val]] = True

    parts_train, parts_val = [], []
    batch_size = 2000
    for i in range(0, len(order), batch_size):
        idx_chunk = order[i:i + batch_size]
        encodings = tokenizer.encode_batch([lines[j] for j in idx_chunk])
        for j, enc in zip(idx_chunk, encodings):
            arr = np.asarray(enc.ids, dtype=DTYPE)
            (parts_val if is_val[j] else parts_train).append(arr)

    train = np.concatenate(parts_train) if parts_train else np.zeros(0, dtype=DTYPE)
    val = np.concatenate(parts_val) if parts_val else np.zeros(0, dtype=DTYPE)
    os.makedirs(os.path.dirname(os.path.abspath(args.train_out)), exist_ok=True)
    np.save(args.train_out, train)
    np.save(args.val_out, val)

    print(f"conversations: {len(lines):,} "
          f"(train {len(lines) - n_val:,} / val {n_val:,})")
    print(f"total tokens: {train.size + val.size:,} "
          f"(train {train.size:,} / val {val.size:,})")
    for path, arr in ((args.train_out, train), (args.val_out, val)):
        print(f"{path}: {arr.size:,} tokens, {os.path.getsize(path) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
