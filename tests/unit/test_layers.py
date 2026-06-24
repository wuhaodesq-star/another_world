"""Tests for shared layer primitives."""

from __future__ import annotations

import torch

from another_world.models.layers.common import (
    CausalSelfAttention,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    apply_rope,
    build_rope_cache,
    count_parameters,
)


def test_rmsnorm_preserves_shape_and_changes_values() -> None:
    norm = RMSNorm(16)
    x = torch.randn(2, 4, 16) * 5
    y = norm(x)
    assert y.shape == x.shape
    # weight is initialised to 1, so output should have unit RMS along last dim.
    rms = y.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=5e-2)


def test_rope_apply_norm_preserving() -> None:
    head_dim = 8
    cos, sin = build_rope_cache(seq_len=16, head_dim=head_dim)
    x = torch.randn(1, 2, 16, head_dim)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    # RoPE is a rotation: per-position vector norm is preserved.
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_swiglu_forward() -> None:
    block = SwiGLU(dim=16, hidden_dim=32)
    x = torch.randn(3, 5, 16)
    y = block(x)
    assert y.shape == x.shape


def test_causal_attention_shapes_and_gqa() -> None:
    attn = CausalSelfAttention(dim=32, n_heads=4, n_kv_heads=2)
    cos, sin = build_rope_cache(seq_len=10, head_dim=32 // 4)
    x = torch.randn(2, 7, 32)
    y = attn(x, cos, sin)
    assert y.shape == x.shape


def test_transformer_block_forward() -> None:
    block = TransformerBlock(dim=32, n_heads=4, n_kv_heads=4, ffn_mult=2, dropout=0.0)
    cos, sin = build_rope_cache(seq_len=10, head_dim=8)
    x = torch.randn(2, 7, 32)
    y = block(x, cos, sin)
    assert y.shape == x.shape


def test_count_parameters_positive() -> None:
    block = SwiGLU(dim=8, hidden_dim=16)
    assert count_parameters(block) > 0
