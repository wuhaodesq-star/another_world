"""Integration test: smoke trainer logs to JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from another_world.data.datasets import DummyTokenDataset
from another_world.models.dynamics import ToyTransformerConfig, build_toy_transformer
from another_world.training.smoke import SmokeTrainerConfig, run_smoke_training
from another_world.utils.experiment import JsonlLogger


def test_trainer_writes_jsonl_metrics(tmp_path: Path) -> None:
    torch.manual_seed(0)
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
    ds = DummyTokenDataset(vocab_size=cfg.vocab_size, seq_len=8, length=8, seed=0)
    train_cfg = SmokeTrainerConfig(
        steps=4,
        batch_size=2,
        lr=1e-3,
        warmup_steps=1,
        log_every=1,
        device="cpu",
        precision="fp32",
        seed=0,
    )
    path = tmp_path / "logs" / "run.jsonl"
    logger = JsonlLogger(path=path)
    try:
        history = run_smoke_training(model, ds, train_cfg, logger=logger)
    finally:
        logger.finish()

    assert len(history) == 4
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 4
    for rec, h in zip(records, history):
        assert rec["step"] == h.step
        assert rec["metrics"]["loss"] == h.loss
        assert "grad_norm" in rec["metrics"]
        assert "tokens_per_sec" in rec["metrics"]
