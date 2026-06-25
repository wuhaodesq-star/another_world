"""Tests for the DiT trainer."""

from __future__ import annotations

import torch

from another_world.models.decoder import (
    DiTDecoder,
    DiTDecoderConfig,
    DiffusionObjectiveConfig,
)
from another_world.training.dit_trainer import (
    DiTTrainerConfig,
    run_dit_training,
)


def _dummy_batches(n: int, *, channels: int = 4, t: int = 2, h: int = 8, w: int = 8,
                   vocab: int = 32, batch_size: int = 2):
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        x0 = torch.randn(batch_size, channels, t, h, w, generator=g)
        ids = torch.randint(0, vocab, (batch_size, 4), generator=g)
        out.append((x0, ids))
    return out


def test_dit_trainer_runs_and_loss_decreases_overfit() -> None:
    torch.manual_seed(0)
    cfg = DiTDecoderConfig.toy(vocab_size=32)
    model = DiTDecoder(cfg)

    batches = _dummy_batches(4)

    history = run_dit_training(
        model, batches,
        DiTTrainerConfig(
            steps=30, lr=3e-3, warmup_steps=2, log_every=5,
            device="cpu", precision="fp32",
            objective=DiffusionObjectiveConfig(objective="rectified_flow"),
        ),
    )
    assert len(history) >= 2
    assert history[-1]["loss"] < history[0]["loss"]


def test_dit_trainer_v_prediction_path() -> None:
    torch.manual_seed(1)
    cfg = DiTDecoderConfig.toy(vocab_size=16)
    model = DiTDecoder(cfg)
    batches = _dummy_batches(2, vocab=16)
    history = run_dit_training(
        model, batches,
        DiTTrainerConfig(
            steps=8, lr=1e-3, warmup_steps=1, log_every=4,
            device="cpu", precision="fp32",
            objective=DiffusionObjectiveConfig(objective="v_prediction"),
        ),
    )
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in history)


def test_dit_trainer_grad_accum_consumes_more_batches() -> None:
    cfg = DiTDecoderConfig.toy(vocab_size=16)
    model = DiTDecoder(cfg)
    batches = iter(_dummy_batches(20, vocab=16))
    consumed = 0

    def gen():
        nonlocal consumed
        for b in batches:
            consumed += 1
            yield b

    run_dit_training(
        model, gen(),
        DiTTrainerConfig(
            steps=3, grad_accum=4, lr=1e-3, warmup_steps=1, log_every=1,
            device="cpu", precision="fp32",
        ),
    )
    assert consumed >= 3 * 4
