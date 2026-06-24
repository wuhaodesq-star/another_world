"""Tests for the stage-0 smoke trainer."""

from __future__ import annotations

import torch

from another_world.data.datasets.dummy import DummyTokenDataset
from another_world.models.dynamics.toy import (
    ToyTransformerConfig,
    build_toy_transformer,
)
from another_world.training.smoke import SmokeTrainerConfig, run_smoke_training


def test_smoke_training_runs_and_loss_decreases() -> None:
    torch.manual_seed(0)
    cfg = ToyTransformerConfig(
        vocab_size=32,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2,
        max_seq_len=16,
    )
    model = build_toy_transformer(cfg)
    # A tiny deterministic dataset: with limited unique sequences the model
    # should memorise quickly, giving us a reliable loss-decrease check.
    dataset = DummyTokenDataset(
        vocab_size=cfg.vocab_size,
        seq_len=8,
        length=8,
        seed=0,
    )
    train_cfg = SmokeTrainerConfig(
        steps=60,
        batch_size=4,
        lr=3e-3,
        warmup_steps=5,
        log_every=10,
        device="cpu",
        precision="fp32",
        seed=0,
    )
    history = run_smoke_training(model, dataset, train_cfg)
    assert len(history) >= 2
    assert history[0].loss > history[-1].loss
    assert history[-1].tokens_per_sec > 0


def test_smoke_training_grad_accum_smoke() -> None:
    cfg = ToyTransformerConfig(
        vocab_size=16,
        dim=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2,
        max_seq_len=16,
    )
    model = build_toy_transformer(cfg)
    dataset = DummyTokenDataset(vocab_size=cfg.vocab_size, seq_len=8, length=8, seed=1)
    train_cfg = SmokeTrainerConfig(
        steps=4,
        batch_size=2,
        grad_accum=2,
        lr=1e-3,
        warmup_steps=1,
        log_every=1,
        device="cpu",
        precision="fp32",
        seed=1,
    )
    history = run_smoke_training(model, dataset, train_cfg)
    assert len(history) == 4
