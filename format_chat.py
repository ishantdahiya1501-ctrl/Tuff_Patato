"""Format raw chat JSON into <|user|>...<|assistant|>...<|end|> lines.

Reads data/chat/raw_chat.json (from download_chat_data.py), skips empty turns,
non-English conversations, and anything longer than --max_tokens, then writes
one conversation per line to data/chat/formatted_chat.txt.
"""

import argparse
import json
import os
import re

from tokenizer import load_tokenizer

USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN = "<|user|>", "<|assistant|>", "<|end|>"

# Cheap English heuristic: TinyStories is essentially all-English, this guards
# against stray non-Latin / mojibake rows without pulling in a langdetect dep.
COMMON_WORDS = {"the", "and", "a", "to", "of", "was", "she", "he", "said", "in",
                "it", "they", "her", "his", "had", "on", "with", "mom", "day"}


def looks_english(text: str) -> bool:
    if not text:
        return False
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < 0.95:
        return False
    words = set(re.findall(r"[a-z']+", text.lower()))
    return len(words & COMMON_WORDS) >= 2 or len(words) >= 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/chat/raw_chat.json")
    parser.add_argument("--tokenizer", default="data/chat/tokenizer_chat.json")
    parser.add_argument("--out", default="data/chat/formatted_chat.txt")
    parser.add_argument("--max_tokens", type=int, default=250,
                        help="Skip conversations longer than this many tokens")
    args = parser.parse_args()

    if not os.path.exists(args.raw):
        raise SystemExit(f"Raw chat data not found: {args.raw} (run download_chat_data.py)")
    if not os.path.exists(args.tokenizer):
        raise SystemExit(f"Chat tokenizer not found: {args.tokenizer} (run add_chat_tokens.py)")
    tokenizer = load_tokenizer(args.tokenizer)

    with open(args.raw, encoding="utf-8") as f:
        rows = json.load(f)

    n_empty = n_nonenglish = n_long = 0
    candidates = []
    for row in rows:
        user = " ".join((row.get("user") or "").split())
        assistant = " ".join((row.get("assistant") or "").split())
        if not user or not assistant:
            n_empty += 1
            continue
        line = f"{USER_TOKEN}{user}{ASSISTANT_TOKEN}{assistant}{END_TOKEN}"
        if not looks_english(line):
            n_nonenglish += 1
            continue
        candidates.append(line)

    # Batch-encode to count tokens, keeping only conversations that fit.
    kept_lines, kept_lens = [], []
    batch_size = 2000
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        for line, enc in zip(batch, tokenizer.encode_batch(batch)):
            if len(enc.ids) <= args.max_tokens:
                kept_lines.append(line)
                kept_lens.append(len(enc.ids))
            else:
                n_long += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))

    skipped = len(rows) - len(kept_lines)
    avg_tokens = sum(kept_lens) / max(1, len(kept_lens))
    print(f"kept={len(kept_lines):,} skipped={skipped:,} "
          f"(empty={n_empty:,}, non_english={n_nonenglish:,}, too_long={n_long:,})")
    print(f"avg_tokens={avg_tokens:.1f} (limit {args.max_tokens})")
    for i, line in enumerate(kept_lines[:3], 1):
        print(f"\n--- sample {i} ---")
        print(line if len(line) <= 240 else line[:240] + " ...")


if __name__ == "__main__":
    main()
