"""Common building blocks shared by dynamics, decoder, and JEPA models.

This module intentionally keeps the implementation dependency-light so that
unit tests can run on CPU without ``flash_attn`` or other GPU-only kernels.
Production training paths under :mod:`another_world.training` will swap these
primitives for fused / optimized variants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (LLaMA-style)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        norm = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * norm).to(dtype) * self.weight


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute cosine / sine tables for rotary positional embeddings."""

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    angles = torch.outer(t, freqs)  # [seq_len, half]
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to ``x`` of shape ``[B, H, T, D]``."""

    seq_len = x.size(-2)
    cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)
    sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)
    # Split last dim into even/odd pairs.
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated_even = x1 * cos - x2 * sin
    rotated_odd = x1 * sin + x2 * cos
    out = torch.stack((rotated_even, rotated_odd), dim=-1)
    return out.flatten(-2)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block (LLaMA-style)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))


@dataclass
class AttentionShape:
    n_heads: int
    n_kv_heads: int
    head_dim: int


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with grouped-query attention (GQA).

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` so it
    automatically picks the best kernel available (Flash / mem-efficient / math).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        n_kv_heads = n_kv_heads or n_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"
            )

        self.shape = AttentionShape(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=dim // n_heads,
        )
        self.dropout = dropout

        self.wq = nn.Linear(dim, n_heads * self.shape.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.shape.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.shape.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.shape.head_dim, dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        s = self.shape

        q = self.wq(x).view(bsz, seq_len, s.n_heads, s.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, s.n_kv_heads, s.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, s.n_kv_heads, s.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if s.n_kv_heads != s.n_heads:
            repeats = s.n_heads // s.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: norm -> attn -> residual -> norm -> ffn -> residual."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = _swiglu_hidden_dim(dim, ffn_mult)
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, n_kv_heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden, dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def _swiglu_hidden_dim(dim: int, ffn_mult: int) -> int:
    """LLaMA-style SwiGLU hidden size: round to multiple of 256."""

    raw = int(dim * ffn_mult * 2 / 3)
    return ((raw + 255) // 256) * 256


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """In-place initialization following the LLaMA recipe."""

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)


def estimate_rope_cache_bytes(seq_len: int, head_dim: int) -> int:
    """Bytes needed for the RoPE sin/cos cache at fp32."""

    return 2 * seq_len * (head_dim // 2) * 4


__all__ = [
    "AttentionShape",
    "CausalSelfAttention",
    "RMSNorm",
    "SwiGLU",
    "TransformerBlock",
    "apply_rope",
    "build_rope_cache",
    "count_parameters",
    "estimate_rope_cache_bytes",
    "init_weights",
]
