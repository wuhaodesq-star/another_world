"""Tests for the multimodal dynamics model and its presets."""

from __future__ import annotations

import pytest
import torch

from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
    build_multimodal_model,
)
from another_world.models.layers.mixed_rope import axes_from_segments
from another_world.tokenizers.vocab import VocabLayout


def _toy_model(vocab_size: int = 224) -> MultimodalDynamicsModel:
    return MultimodalDynamicsModel(MultimodalDynamicsConfig.toy(vocab_size))


def _axes_for(text_count: int, vt: int, vh: int, vw: int):
    return axes_from_segments([
        ("special", {"count": 1}),                    # bos
        ("special", {"count": 1}),                    # boc
        ("text", {"count": text_count}),
        ("special", {"count": 1}),                    # eoc
        ("special", {"count": 1}),                    # bov
        ("visual", {"t": vt, "h": vh, "w": vw}),
        ("special", {"count": 1}),                    # eov
        ("special", {"count": 1}),                    # eos
    ])


def test_toy_model_forward_shapes() -> None:
    layout = VocabLayout.tiny()
    model = _toy_model(layout.total_size)
    axes = _axes_for(text_count=3, vt=2, vh=2, vw=2)
    seq_len = axes.modality.shape[1]
    tokens = torch.randint(0, layout.total_size, (1, seq_len))
    out = model(tokens, axes=axes)
    assert out["logits"].shape == (1, seq_len, layout.total_size)


def test_toy_model_returns_loss() -> None:
    layout = VocabLayout.tiny()
    model = _toy_model(layout.total_size)
    axes = _axes_for(text_count=2, vt=1, vh=2, vw=2)
    seq_len = axes.modality.shape[1]
    tokens = torch.randint(0, layout.total_size, (1, seq_len))
    targets = torch.randint(0, layout.total_size, (1, seq_len))
    out = model(tokens, axes=axes, targets=targets)
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    assert out["loss"].dim() == 0


def test_toy_model_backward_updates_params() -> None:
    layout = VocabLayout.tiny()
    model = _toy_model(layout.total_size)
    axes = _axes_for(text_count=2, vt=1, vh=2, vw=2)
    seq_len = axes.modality.shape[1]
    tokens = torch.randint(0, layout.total_size, (1, seq_len))
    targets = tokens.clone()
    out = model(tokens, axes=axes, targets=targets)
    out["loss"].backward()
    grad_seen = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert grad_seen


def test_toy_model_loss_mask_zero_yields_zero_loss() -> None:
    layout = VocabLayout.tiny()
    model = _toy_model(layout.total_size)
    axes = _axes_for(text_count=2, vt=1, vh=2, vw=2)
    seq_len = axes.modality.shape[1]
    tokens = torch.randint(0, layout.total_size, (1, seq_len))
    targets = torch.randint(0, layout.total_size, (1, seq_len))
    mask = torch.zeros_like(tokens, dtype=torch.float32)
    out = model(tokens, axes=axes, targets=targets, loss_mask=mask)
    assert torch.isfinite(out["loss"])
    # All weights masked out -> denominator clamped to 1, numerator is 0.
    assert float(out["loss"].item()) == 0.0


def test_axes_shape_mismatch_raises() -> None:
    layout = VocabLayout.tiny()
    model = _toy_model(layout.total_size)
    axes = _axes_for(text_count=2, vt=1, vh=2, vw=2)
    bad_tokens = torch.randint(0, layout.total_size, (1, axes.modality.shape[1] + 1))
    with pytest.raises(ValueError):
        model(bad_tokens, axes=axes)


def test_targets_shape_mismatch_raises() -> None:
    layout = VocabLayout.tiny()
    model = _toy_model(layout.total_size)
    axes = _axes_for(text_count=2, vt=1, vh=2, vw=2)
    n = axes.modality.shape[1]
    tokens = torch.randint(0, layout.total_size, (1, n))
    bad_targets = torch.randint(0, layout.total_size, (1, n + 1))
    with pytest.raises(ValueError):
        model(tokens, axes=axes, targets=bad_targets)


def test_build_multimodal_model_default() -> None:
    model = build_multimodal_model()
    assert isinstance(model, MultimodalDynamicsModel)
    assert model.num_parameters > 0


def test_preset_classmethods_compile() -> None:
    cfg_350 = MultimodalDynamicsConfig.m350(vocab_size=1024)
    assert cfg_350.dim == 1024
    cfg_7b = MultimodalDynamicsConfig.b7(vocab_size=1024)
    assert cfg_7b.dim == 4096
    assert cfg_7b.n_layers == 32


def test_tied_embeddings_share_storage() -> None:
    model = _toy_model(64)
    assert model.output.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()
