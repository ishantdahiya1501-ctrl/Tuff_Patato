# TinyGPT — a ~5M-parameter decoder-only Transformer trained on TinyStories

A minimal, from-scratch GPT-style language model in PyTorch. No HuggingFace
`transformers` — the Transformer (attention, RoPE, RMSNorm, SwiGLU) is
implemented by hand in `model.py`. It is pretrained **only** on
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
so it learns to tell simple, fluent children's stories. Chat fine-tuning can be
added later; there is deliberately **no chat template or instruction tuning**
in this repo.

## Architecture

| Component          | Choice                                    |
|--------------------|-------------------------------------------|
| Type               | Decoder-only GPT (pre-norm)               |
| Layers             | 4                                         |
| d_model            | 256                                       |
| Heads              | 8 (head_dim = 32)                         |
| Context length     | 256 tokens                                |
| FFN                | SwiGLU, hidden size 1024                  |
| Normalization      | RMSNorm                                   |
| Position encoding  | RoPE (theta = 10000)                      |
| Embeddings         | Input/output tied                         |
| Bias terms         | None                                      |
| Dropout            | 0.0 (configurable)                        |
| Vocab              | 4096 byte-level BPE, trained on TinyStories |
| **Parameters**     | **5,245,184 (~5.25M)**                    |

## Files

- `model.py` — model definition (TinyGPT), RoPE, generation, checkpoint helper
- `tokenizer.py` — train/load the 4096-vocab byte-level BPE tokenizer
- `dataset.py` — tokenize TinyStories and cache as packed `.npy` token streams
- `train.py` — training loop (AdamW, cosine schedule with warmup, AMP, checkpoints)
- `sample.py` — generate text from a prompt using a checkpoint
- `requirements.txt` — dependencies (torch, numpy, tokenizers, datasets, tqdm)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
# 1. Train the BPE tokenizer (vocab 4096) on the TinyStories training split
python tokenizer.py --data_dir data/tinystories --vocab_size 4096

# 2. Train the model (downloads TinyStories on first run, then caches tokens)
python train.py --data_dir data/tinystories --context_length 256 \
    --batch_size 64 --epochs 3 --lr 3e-4 --device cuda
```

On CPU, training 3 epochs over the full dataset takes a long time — add
`--max_stories 50000` for a quicker (weaker) run, and expect `--device auto`
to pick CPU when no CUDA GPU is present.

### Useful `train.py` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--batch_size` | 64 | Micro-batch size; lower it if you hit OOM |
| `--grad_accum_steps` | 1 | Accumulate N micro-batches per optimizer step (effective batch = batch_size × N) |
| `--epochs` | 3 | Passes over the packed token stream |
| `--lr` / `--warmup_steps` | 3e-4 / 500 | Cosine LR schedule with linear warmup |
| `--dropout` | 0.0 | Set 0.1 for regularization |
| `--eval_every` / `--eval_batches` | 500 / 50 | Validation loss cadence (0 disables) |
| `--max_stories` | all | Cap training stories for a quick run |
| `--checkpoint_dir` | `checkpoints/` | Where `epochN.pt`, `latest.pt`, `final.pt` are saved |
| `--device` | auto | `auto`, `cpu`, or `cuda` |

Mixed precision is enabled automatically on CUDA (bf16 when supported, fp16
with a GradScaler otherwise); CPU runs in fp32. Progress shows a live loss and
LR; Ctrl-C saves `latest.pt` so you can stop safely.

## Generating text

```bash
python sample.py --prompt "Once upon a time" --max_tokens 100
```

Useful flags: `--checkpoint checkpoints/final.pt` (default), `--temperature 0.8`
(0 = greedy), `--top_k 50` (0 disables), `--seed 42`, `--device auto`.

Example output after full training:

```
Once upon a time there was a little girl named Lily. She liked to play
outside every day. One day, she saw a big dog in the yard. The dog was
very friendly and Lily wanted to play with it...
```

## How the data pipeline works

1. `tokenizer.py` trains byte-level BPE (`tokenizers` lib) with vocab 4096 on
   the TinyStories `train` split and saves `data/tinystories/tokenizer.json`.
2. `dataset.py` encodes each story, appends an `<|endoftext|>` token, and packs
   everything into one flat `uint16` array cached at
   `data/tinystories/{train,validation}_tokens.npy`.
3. `train.py` chunks the stream into fixed 256-token sequences; inputs are
   `tokens[i:i+256]` and targets `tokens[i+1:i+257]` (pure next-token
   prediction, shuffled each epoch). The validation split is held out for eval.

Delete the `data/` cache and rerun with `--force` if you change the tokenizer.

## Notes

- `--device auto` uses CUDA when available; check `nvidia-smi` if training is
  unexpectedly slow.
- Checkpoints store the full model config, so `sample.py` reconstructs the
  model without any extra flags.
- Deliberately minimal: no RLHF, DPO, MoE, chat formatting, or other extras.
