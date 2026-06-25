"""Tests for the mixed RoPE positional encoding."""

from __future__ import annotations

import pytest
import torch

from another_world.models.layers.mixed_rope import (
    MixedRoPE,
    RopeAxes,
    axes_from_segments,
)


def _axes(modality: list[int], t: list[int], h: list[int], w: list[int]) -> RopeAxes:
    return RopeAxes(
        modality=torch.tensor([modality], dtype=torch.long),
        linear=torch.arange(len(modality), dtype=torch.long).unsqueeze(0),
        t=torch.tensor([t], dtype=torch.long),
        h=torch.tensor([h], dtype=torch.long),
        w=torch.tensor([w], dtype=torch.long),
    )


def test_rope_output_shape() -> None:
    rope = MixedRoPE(head_dim=32)
    axes = _axes([0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0])
    cos, sin = rope.build(axes)
    assert cos.shape == (1, 4, 16)
    assert sin.shape == (1, 4, 16)


def test_rope_visual_uses_thw_axes() -> None:
    rope = MixedRoPE(head_dim=32)
    # Two visual tokens at same linear but different (h, w).
    axes_a = RopeAxes(
        modality=torch.tensor([[1, 1]], dtype=torch.long),
        linear=torch.tensor([[0, 0]], dtype=torch.long),
        t=torch.tensor([[0, 0]], dtype=torch.long),
        h=torch.tensor([[0, 5]], dtype=torch.long),
        w=torch.tensor([[0, 0]], dtype=torch.long),
    )
    cos_a, _ = rope.build(axes_a)
    # Each axis contributes (head_dim/4)/2 = 4 cos cols, total 16.
    s = 4
    assert torch.allclose(cos_a[:, 0, :s], cos_a[:, 1, :s])              # linear
    assert torch.allclose(cos_a[:, 0, s:2*s], cos_a[:, 1, s:2*s])        # t
    assert not torch.allclose(cos_a[:, 0, 2*s:3*s], cos_a[:, 1, 2*s:3*s])  # h
    assert torch.allclose(cos_a[:, 0, 3*s:4*s], cos_a[:, 1, 3*s:4*s])    # w


def test_rope_text_uses_linear_everywhere() -> None:
    rope = MixedRoPE(head_dim=32)
    axes = _axes([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
    cos, _ = rope.build(axes)
    s = 4
    # All four shards should be identical for text tokens.
    a = cos[:, :, :s]
    b = cos[:, :, s:2*s]
    c = cos[:, :, 2*s:3*s]
    d = cos[:, :, 3*s:4*s]
    assert torch.allclose(a, b)
    assert torch.allclose(a, c)
    assert torch.allclose(a, d)


def test_rope_head_dim_validation() -> None:
    with pytest.raises(ValueError):
        MixedRoPE(head_dim=10)


def test_axes_from_segments() -> None:
    axes = axes_from_segments(
        [
            ("text", {"count": 2}),
            ("visual", {"t": 1, "h": 2, "w": 2}),
            ("action", {"count": 1}),
        ]
    )
    # 2 text + 4 visual + 1 action = 7 tokens
    assert axes.modality.shape == (1, 7)
    assert axes.modality[0].tolist() == [0, 0, 1, 1, 1, 1, 2]
    assert axes.linear[0].tolist() == [0, 1, 2, 3, 4, 5, 6]
    # Visual coords row-major (t, h, w):
    assert axes.t[0, 2:6].tolist() == [0, 0, 0, 0]
    assert axes.h[0, 2:6].tolist() == [0, 0, 1, 1]
    assert axes.w[0, 2:6].tolist() == [0, 1, 0, 1]
