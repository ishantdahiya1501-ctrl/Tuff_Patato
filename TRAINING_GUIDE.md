# Training Guide — how to train TinyGPT on CPU, the 840M, or both

Your laptop: Kali Linux, 4-core CPU (no AVX-512), **GeForce 840M** (2 GB VRAM,
Maxwell sm_50), 7 GB RAM. This guide covers every way to start training.

> **Good to know:** the dataset is downloaded **automatically** on the first
> run. The tokenizer + token cache are created **once** and shared by CPU and
> GPU training — you never redo that step when you switch devices.

---

## 0. Start-to-finish checklist (any device)

```bash
cd <project folder>
source .venv/bin/activate              # CPU venv (already created)

# 1. Train the BPE tokenizer — downloads TinyStories (~2 GB) on first run
python tokenizer.py --data_dir data/tinystories --vocab_size 4096

# 2. Train the model (tokenizes + caches data automatically on first run)
python train.py --data_dir data/tinystories --context_length 256 \
    --batch_size 64 --epochs 3 --lr 3e-4 --device auto

# 3. Generate text
python sample.py --prompt "Once upon a time" --max_tokens 100
```

`--device auto` = CUDA if a working GPU + CUDA torch exist, else CPU.

---

## Option A — Train on CPU only (works right now)

No setup needed; the `.venv` already has CPU torch.

| Run | Command addition | Time (this machine) |
|-----|------------------|---------------------|
| Quick sanity run | `--max_stories 20000` | ~4 hours |
| Medium run | `--max_stories 200000` | ~1.5–2 days |
| Full 3 epochs (all 2.1M stories) | *(nothing)* | **~2–3 weeks — don't** ❌ |

```bash
python train.py --data_dir data/tinystories --context_length 256 \
    --batch_size 64 --epochs 3 --lr 3e-4 --device cpu --max_stories 20000
```

Best use of the CPU: the **quick sanity run** to prove learning, while you set
up the GPU (Option B) for real training.

---

## Option B — Train on your GeForce 840M (~2–4 days for full training)

The 840M is **~5–10× faster than this CPU** for this model. Full 3-epoch
training becomes feasible (~2–4 days unattended).

### Step 1 — Install the NVIDIA driver (needs sudo, one reboot)

Kali's repo ships **550.163.01**, a branch that still supports Maxwell GPUs:

```bash
sudo apt update
sudo apt install -y nvidia-driver nvidia-smi
sudo reboot
```

After reboot, verify the 840M is visible:

```bash
nvidia-smi
# should list: GeForce 840M, 2048 MiB
```

> Your laptop has hybrid graphics (Intel + NVIDIA). The display keeps running
> on Intel; the 840M is used purely for CUDA compute. Secure Boot is off, so
> no signing issues.

### Step 2 — Create a CUDA-enabled torch venv

⚠️ Recent torch builds **removed Maxwell support** from their CUDA 12.8/12.9
wheels. The **cu126** build of torch 2.9.0 is the one that works on your
Python 3.14 *and* still contains sm_50 kernels:

```bash
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install torch==2.9.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.venv-gpu/bin/pip install -r requirements.txt
```

### Step 3 — Verify CUDA works

```bash
.venv-gpu/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
# expect: 2.9.0+cu126 True  /  NVIDIA GeForce 840M
```

If it prints `NO GPU`: driver didn't load (check `nvidia-smi`) or the wrong
torch build got installed (must be `+cu126`).

### Step 4 — Train on the GPU

```bash
.venv-gpu/bin/python train.py --data_dir data/tinystories --context_length 256 \
    --batch_size 64 --epochs 3 --lr 3e-4 --device cuda
```

**If you hit CUDA out-of-memory** (2 GB VRAM is small), keep the same
effective batch by splitting it:

```bash
    --batch_size 32 --grad_accum_steps 2     # or 16 x 4
```

Mixed precision turns on automatically (fp16 on Maxwell; bf16 is not
supported on this generation).

**Monitor GPU usage** in a second terminal:

```bash
watch -n1 nvidia-smi
```

### Step 5 — Run it unattended (recommended for multi-day runs)

```bash
nohup .venv-gpu/bin/python train.py --data_dir data/tinystories \
    --context_length 256 --batch_size 64 --epochs 3 --lr 3e-4 --device cuda \
    > train.log 2>&1 &
tail -f train.log        # watch progress; Ctrl-C here does NOT stop training
```

Interrupting training directly is safe anyway — `latest.pt` is saved on
Ctrl-C, and checkpoints (`epochN.pt`, `final.pt`) land in `checkpoints/`.

### Shorter GPU runs

| Goal | Command addition | GPU time |
|------|------------------|----------|
| First stories tonight | `--max_stories 50000` | ~2–3 h |
| Good quality | `--max_stories 500000` | ~1 day |
| Full TinyStories, 3 epochs | *(nothing)* | ~2–4 days |

---

## "Can I use the CPU **and** 840M at the same time?"

**For one training run: no — and you shouldn't want to.** Splitting a single
5M-parameter model across CPU + GPU means every layer/batch must sync over
PCIe, which is slower than the GPU alone. PyTorch also has no built-in
CPU+GPU data-parallel mode. (This is why the code doesn't fake it.)

What "using both" means in practice, and what the code already does:

| Who does what | How |
|---------------|-----|
| **GPU: all matrix math** | `--device cuda` |
| **CPU: tokenization, data packing, batch prep** | automatic, every step |
| **CPU: free for other work while GPU trains** | run `sample.py`, tokenize data, browse — the training process only needs ~1 core |

So: start training on the 840M, and your CPU stays usable for everything
else. That *is* using both.

---

## Option C — Full training on free Colab/Kaggle (~4–8 h)

Best quality-per-hour if you don't want a 2–4 day local run:

1. Zip the project (without `.venv*`, `data/`, `checkpoints/`) and upload to
   Google Colab or Kaggle (T4 GPU runtime).
2. `pip install -r requirements.txt`
3. Run the three commands from section 0 with `--device cuda`.
4. Download `checkpoints/final.pt` + `data/tinystories/tokenizer.json` back
   to this machine and sample locally with `--device cpu`.

Sampling a 5M model runs fine on any CPU — training is the only heavy part.

---

## Quick command reference

| Script | Key flags | Default |
|--------|-----------|---------|
| `tokenizer.py` | `--data_dir`, `--vocab_size`, `--max_stories` | vocab 4096 |
| `train.py` | `--device auto\|cpu\|cuda`, `--batch_size`, `--grad_accum_steps`, `--epochs`, `--lr`, `--warmup_steps`, `--max_stories`, `--eval_every`, `--checkpoint_dir` | 3 epochs, lr 3e-4, warmup 500 |
| `sample.py` | `--checkpoint`, `--prompt`, `--max_tokens`, `--temperature`, `--top_k`, `--seed` | temp 0.8, top-k 50 |

**Recommended path for you:** Option A quick run now → install driver →
Option B `--max_stories 500000` overnight → sample.
