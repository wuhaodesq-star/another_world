"""Tests for the toy transformer model."""

from __future__ import annotations

import torch

from another_world.models.dynamics.toy import (
    ToyTransformer,
    ToyTransformerConfig,
    build_toy_transformer,
)


def _tiny_config(**overrides: object) -> ToyTransformerConfig:
    base = dict(
        vocab_size=64,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2,
        max_seq_len=32,
        dropout=0.0,
    )
    base.update(overrides)
    return ToyTransformerConfig(**base)  # type: ignore[arg-type]


def test_toy_transformer_forward_shapes() -> None:
    cfg = _tiny_config()
    model = build_toy_transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(tokens)
    assert "logits" in out
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert "loss" not in out


def test_toy_transformer_returns_loss_when_targets_given() -> None:
    cfg = _tiny_config()
    model = build_toy_transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(tokens, targets=targets)
    assert "loss" in out
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"]).item()


def test_toy_transformer_backward_step_runs() -> None:
    cfg = _tiny_config()
    model = build_toy_transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    loss = model(tokens, targets=targets)["loss"]
    loss.backward()
    # at least one parameter must have a non-zero gradient.
    any_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert any_grad


def test_toy_transformer_rejects_oversized_sequence() -> None:
    cfg = _tiny_config(max_seq_len=8)
    model = build_toy_transformer(cfg)
    too_long = torch.randint(0, cfg.vocab_size, (1, 9))
    try:
        model(too_long)
    except ValueError:
        return
    raise AssertionError("expected ValueError for sequence longer than max_seq_len")


def test_tied_embeddings_share_storage() -> None:
    cfg = _tiny_config(tie_embeddings=True)
    model = build_toy_transformer(cfg)
    assert model.output.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()


def test_untied_embeddings_are_independent() -> None:
    cfg = _tiny_config(tie_embeddings=False)
    model = build_toy_transformer(cfg)
    assert model.output.weight.data_ptr() != model.tok_embeddings.weight.data_ptr()
