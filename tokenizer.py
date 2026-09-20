"""Train / load a byte-level BPE tokenizer (vocab 4096) on the TinyStories training split."""

import argparse
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

EOS_TOKEN = "<|endoftext|>"


def train_tokenizer(texts, vocab_size: int = 4096, save_path: str = "tokenizer.json") -> Tokenizer:
    """Train a GPT-2-style byte-level BPE tokenizer over an iterable of texts."""
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def eos_id(tokenizer: Tokenizer) -> int:
    token_id = tokenizer.token_to_id(EOS_TOKEN)
    if token_id is None:
        raise ValueError(f"{EOS_TOKEN!r} is missing from the tokenizer vocab")
    return token_id


def iter_tinystories(split: str = "train", max_stories: int | None = None):
    """Yield cleaned story texts from the TinyStories dataset (downloads on first use)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required to download TinyStories. "
            "Run: pip install -r requirements.txt"
        ) from exc
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    count = 0
    for row in dataset:
        text = (row["text"] or "").strip()
        if not text:
            continue
        yield text
        count += 1
        if max_stories is not None and count >= max_stories:
            break


def main():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on TinyStories")
    parser.add_argument("--data_dir", default="data/tinystories",
                        help="Output directory for tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Limit stories used for tokenizer training (default: all)")
    args = parser.parse_args()

    save_path = os.path.join(args.data_dir, "tokenizer.json")
    print(f"Training BPE tokenizer (vocab_size={args.vocab_size}) on the "
          f"TinyStories '{args.split}' split ...")
    tokenizer = train_tokenizer(
        iter_tinystories(args.split, args.max_stories), args.vocab_size, save_path
    )
    print(f"Saved tokenizer to {save_path}")
    print(f"Final vocab size: {tokenizer.get_vocab_size()} (includes {EOS_TOKEN!r})")

    sample = "Once upon a time there was a little girl named Lucy. She loved to play in the park."
    decoded = tokenizer.decode(tokenizer.encode(sample).ids)
    assert decoded == sample, f"Tokenizer round-trip failed: {decoded!r}"
    print("Round-trip encode/decode: OK")


if __name__ == "__main__":
    main()
