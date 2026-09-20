"""Train the tiny GPT on packed TinyStories tokens (next-token prediction)."""

import argparse
import itertools
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from dataset import prepare_tokens
from model import ModelConfig, TinyGPT, resolve_device, save_checkpoint
from tokenizer import load_tokenizer


def lr_at(step: int, warmup_steps: int, total_steps: int) -> float:
    """Cosine schedule with linear warmup, as a multiplicative LR factor."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def batch_starts(n_sequences, batch_size, rng):
    """Yield shuffled arrays of sequence start indices, one array per batch."""
    order = rng.permutation(n_sequences)
    for i in range(0, n_sequences - batch_size + 1, batch_size):
        yield order[i:i + batch_size]


def gather_batch(tokens, starts, context_length):
    idx = starts[:, None] * context_length + np.arange(context_length)[None, :]
    return np.asarray(tokens[idx]), np.asarray(tokens[idx + 1])


@torch.no_grad()
def evaluate(model, tokens, context_length, batch_size, device, max_batches, autocast_kwargs):
    model.eval()
    n_sequences = (len(tokens) - 1) // context_length
    if n_sequences == 0:
        model.train()
        return float("nan")
    batch_size = min(batch_size, n_sequences)
    losses = []
    for start in range(0, n_sequences - batch_size + 1, batch_size):
        if len(losses) >= max_batches:
            break
        x, y = gather_batch(tokens, np.arange(start, start + batch_size), context_length)
        x = torch.from_numpy(x.astype(np.int64)).to(device)
        y = torch.from_numpy(y.astype(np.int64)).to(device)
        with torch.autocast(**autocast_kwargs):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ~5M param GPT on TinyStories")
    parser.add_argument("--data_dir", default="data/tinystories")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer.json (default: <data_dir>/tokenizer.json)")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Micro-batch size (use --grad_accum_steps if it doesn't fit)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Micro-batches per optimizer step (effective batch = batch_size * this)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Cap the number of training stories (default: all)")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Optimizer steps between validation evals (0 = off)")
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-tokenize even if cached")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    tokenizer_path = args.tokenizer or os.path.join(args.data_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise SystemExit(
            f"Tokenizer not found at {tokenizer_path}.\n"
            f"Run first: python tokenizer.py --data_dir {args.data_dir} --vocab_size 4096"
        )
    vocab_size = load_tokenizer(tokenizer_path).get_vocab_size()

    train_tokens = prepare_tokens(args.data_dir, tokenizer_path, "train",
                                  args.max_stories, args.force)
    val_tokens = (
        prepare_tokens(args.data_dir, tokenizer_path, "validation", None, args.force)
        if args.eval_every > 0 else None
    )

    cfg = ModelConfig(vocab_size=vocab_size, context_length=args.context_length,
                      dropout=args.dropout)
    model = TinyGPT(cfg).to(device)
    print(f"Total parameters: {model.num_parameters():,}")

    n_sequences = (len(train_tokens) - 1) // args.context_length
    micro_per_epoch = n_sequences // args.batch_size
    if micro_per_epoch == 0:
        raise SystemExit(f"Only {n_sequences} sequences available: too few for one "
                         f"batch of {args.batch_size}. Lower --batch_size.")
    accum = max(1, args.grad_accum_steps)
    opt_per_epoch = max(1, micro_per_epoch // accum)
    total_opt_steps = opt_per_epoch * args.epochs
    print(f"{n_sequences:,} sequences of {args.context_length} tokens | "
          f"effective batch = {args.batch_size} x {accum} = {args.batch_size * accum} | "
          f"{total_opt_steps:,} optimizer steps over {args.epochs} epochs")

    # AdamW: weight decay on 2-D matrices only (not embeddings/norms).
    decay = [p for name, p in model.named_parameters()
             if p.ndim >= 2 and "tok_emb" not in name]
    no_decay = [p for name, p in model.named_parameters()
                if not (p.ndim >= 2 and "tok_emb" not in name)]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_at(step, args.warmup_steps, total_opt_steps)
    )

    # Mixed precision on CUDA (bf16 if supported, else fp16); plain fp32 on CPU.
    if device.type == "cuda":
        use_amp = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        use_amp = False
        amp_dtype = torch.bfloat16  # unused: autocast stays disabled on CPU
    autocast_kwargs = {"device_type": device.type, "dtype": amp_dtype, "enabled": use_amp}
    use_scaler = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(args.checkpoint_dir, "latest.pt")

    def run_eval(step, epoch):
        if val_tokens is None:
            return None
        val_loss = evaluate(model, val_tokens, args.context_length, args.batch_size,
                            device, args.eval_batches, autocast_kwargs)
        print(f"[eval] step {step} (epoch {epoch}): val_loss = {val_loss:.4f}")
        return val_loss

    def save(path, epoch, step, val_loss=None):
        save_checkpoint(path, model, optimizer, epoch=epoch,
                        global_step=step, val_loss=val_loss)
        print(f"[ckpt] saved {path}")

    def optimizer_step():
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    print(f"Training on {device.type} | mixed precision: "
          f"{str(amp_dtype).split('.')[-1] if use_amp else 'off'}")
    start_time = time.time()
    global_step, epoch = 0, 0
    try:
        for epoch in range(1, args.epochs + 1):
            batches = batch_starts(n_sequences, args.batch_size, rng)
            usable_micro = opt_per_epoch * accum
            progress = tqdm(itertools.islice(batches, usable_micro), total=usable_micro,
                            desc=f"epoch {epoch}/{args.epochs}", unit="batch",
                            dynamic_ncols=True)
            recent, seen = [], 0
            optimizer.zero_grad(set_to_none=True)
            for starts in progress:
                x, y = gather_batch(train_tokens, starts, args.context_length)
                x = torch.from_numpy(x.astype(np.int64)).to(device)
                y = torch.from_numpy(y.astype(np.int64)).to(device)
                with torch.autocast(**autocast_kwargs):
                    _, loss = model(x, y)
                scaler.scale(loss / accum).backward()
                recent.append(loss.item())
                seen += 1
                if seen % accum == 0:
                    optimizer_step()
                    global_step += 1
                    if global_step % args.log_every == 0:
                        progress.set_postfix(
                            loss=f"{np.mean(recent[-args.log_every * accum:]):.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    if args.eval_every > 0 and global_step % args.eval_every == 0:
                        val_loss = run_eval(global_step, epoch)
                        save(latest_path, epoch, global_step, val_loss)
            # Flush trailing accumulated grads (when the epoch isn't divisible).
            if seen % accum != 0:
                optimizer_step()
                global_step += 1
            print(f"epoch {epoch}: train_loss (last 100 avg) = {np.mean(recent[-100:]):.4f}")
            val_loss = None
            if val_tokens is not None and (args.eval_every == 0
                                           or global_step % args.eval_every != 0):
                val_loss = run_eval(global_step, epoch)
            save(os.path.join(args.checkpoint_dir, f"epoch{epoch}.pt"),
                 epoch, global_step, val_loss)
            save(latest_path, epoch, global_step, val_loss)

        save(os.path.join(args.checkpoint_dir, "final.pt"), args.epochs, global_step)
        print(f"Done in {(time.time() - start_time) / 60:.1f} min "
              f"({global_step:,} optimizer steps).")
        print("Generate with: python sample.py --checkpoint "
              f"{os.path.join(args.checkpoint_dir, 'final.pt')} --prompt \"Once upon a time\"")
    except KeyboardInterrupt:
        print("\nInterrupted - saving checkpoint ...")
        save(latest_path, max(epoch, 1), global_step)


if __name__ == "__main__":
    main()
