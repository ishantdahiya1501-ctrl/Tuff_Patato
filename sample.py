"""Generate text from a trained TinyGPT checkpoint."""

import argparse

import torch

from model import ModelConfig, TinyGPT, generate, resolve_device
from tokenizer import EOS_TOKEN, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Sample from a trained TinyGPT checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--tokenizer", default="data/tinystories/tokenizer.json")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="0 = greedy, ~0.7-1.0 recommended")
    parser.add_argument("--top_k", type=int, default=50, help="0 disables top-k filtering")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    tokenizer = load_tokenizer(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = ModelConfig(**checkpoint["config"])
    if cfg.vocab_size != tokenizer.get_vocab_size():
        raise SystemExit(
            f"Checkpoint vocab_size ({cfg.vocab_size}) != tokenizer vocab size "
            f"({tokenizer.get_vocab_size()}); pass the matching --tokenizer."
        )
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded {args.checkpoint} ({model.num_parameters():,} params, "
          f"epoch {checkpoint.get('epoch', '?')}, "
          f"step {checkpoint.get('global_step', '?')}) on {device.type}")

    ids = tokenizer.encode(args.prompt).ids
    eos = tokenizer.token_to_id(EOS_TOKEN)
    if not ids:  # empty prompt: seed with EOS
        ids = [eos if eos is not None else 0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = generate(model, idx, max_new_tokens=args.max_tokens,
                   temperature=args.temperature, top_k=args.top_k, eos_id=eos)
    text = tokenizer.decode(out[0].tolist()).replace(EOS_TOKEN, "").strip()
    print("-" * 60)
    print(text)
    print("-" * 60)


if __name__ == "__main__":
    main()
