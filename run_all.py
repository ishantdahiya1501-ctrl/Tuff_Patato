"""Run the full chat fine-tuning prep pipeline in order:

  1A  add_chat_tokens.py      -> data/chat/tokenizer_chat.json
  2A  download_chat_data.py   -> data/chat/raw_chat.json
  2B  format_chat.py          -> data/chat/formatted_chat.txt
  2C  tokenize_chat.py        -> data/chat/chat_{train,val}_tokens.npy

Stops on the first failing step. Prints a final summary when everything is done.
"""

import glob
import subprocess
import sys

STEPS = [
    "add_chat_tokens.py",
    "download_chat_data.py",
    "format_chat.py",
    "tokenize_chat.py",
]

CHANGED_FILES = [
    "train.py",
    "sample.py",
    "dataset.py",
    "add_chat_tokens.py",
    "download_chat_data.py",
    "format_chat.py",
    "tokenize_chat.py",
    "run_all.py",
    "README.md",
]


def run_step(script):
    print()
    print("=" * 70)
    print(f"STEP: python {script}")
    print("=" * 70)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\nFAILED: python {script} (exit code {result.returncode})")
        sys.exit(result.returncode)


def line_count(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def print_summary():
    import numpy as np
    from tokenizers import Tokenizer

    backups = sorted(glob.glob("*.bak"))
    tokenizer = Tokenizer.from_file("data/chat/tokenizer_chat.json")
    vocab = tokenizer.get_vocab_size()
    train = np.load("data/chat/chat_train_tokens.npy", mmap_mode="r")
    val = np.load("data/chat/chat_val_tokens.npy", mmap_mode="r")

    print()
    print("=== READY FOR FINE-TUNING ===")
    print("Changed files: " + ", ".join(f"{p} ({line_count(p)} lines)"
                                        for p in CHANGED_FILES))
    print("Backups:       " + (", ".join(backups) if backups else "(none)"))
    print(f"Tokenizer:     data/chat/tokenizer_chat.json  (vocab = {vocab})")
    print(f"Train tokens:  data/chat/chat_train_tokens.npy ({len(train):,})")
    print(f"Val tokens:    data/chat/chat_val_tokens.npy   ({len(val):,})")
    print("Next command:  python train.py --data_dir data/chat \\")
    print("                 --resume checkpoints/latest.pt \\")
    print(f"                 --vocab_size {vocab} --epochs 2 --lr 1e-4 --batch_size 32")


if __name__ == "__main__":
    for script in STEPS:
        run_step(script)
    print_summary()
