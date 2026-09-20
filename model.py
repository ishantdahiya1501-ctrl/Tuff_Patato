"""Minimal decoder-only GPT-style Transformer, written from scratch in PyTorch.

Default configuration (~5.25M parameters):
  - 4 transformer blocks, d_model=256, 8 heads (head_dim=32)
  - SwiGLU feed-forward (hidden size 1024)
  - RMSNorm, pre-norm
  - RoPE positional embeddings
  - Tied input/output embeddings, no bias terms anywhere
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 8  # head_dim = 256 / 8 = 32
    context_length: int = 256
    ffn_hidden: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def resolve_device(name: str = "auto") -> torch.device:
    """Map a CLI --device value (auto|cpu|cuda|cuda:0|...) to a torch.device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(name)


class RMSNorm(nn.Module):
    """Root-mean-square norm (no mean subtraction, no bias)."""

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def build_rope_cache(context_length: int, head_dim: int, theta: float = 10000.0):
    """Precompute cos/sin tables of shape (context_length, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(context_length, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embeddings on q/k of shape (batch, heads, seq, head_dim)."""
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.attn_dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.dropout(self.attn(self.attn_norm(x), rope_cos, rope_sin))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        cos, sin = build_rope_cache(cfg.context_length, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down residual-output projections (GPT-2 style init).
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=residual_std)
            nn.init.normal_(block.ffn.w2.weight, std=residual_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.dropout(self.tok_emb(idx))
        rope_cos = self.rope_cos[: idx.size(1)]
        rope_sin = self.rope_sin[: idx.size(1)]
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=50, eos_id=None):
    """Sample tokens autoregressively. idx: (batch, <= context_length) LongTensor."""
    assert idx.size(1) > 0, "need at least one seed token"
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if temperature <= 0:  # greedy decoding
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k and top_k < logits.size(-1):
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
        if eos_id is not None and bool((next_id == eos_id).all()):
            break
    return idx


def save_checkpoint(path, model, optimizer=None, epoch=0, global_step=0, val_loss=None):
    torch.save(
        {
            "config": asdict(model.cfg),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
        },
        path,
    )
