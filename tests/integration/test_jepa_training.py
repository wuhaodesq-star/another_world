"""Integration test: multimodal trainer with JEPA auxiliary loss."""

from __future__ import annotations

import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import SequencePacker
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout
from another_world.training.multimodal import (
    MultimodalTrainerConfig,
    run_multimodal_training,
)


def _batches(layout, packer, n_batches, batch_size=2):
    out = []
    for i in range(n_batches):
        chunk = [
            TokenSample(
                visual_tokens=torch.randint(0, 16, (1, 2, 2), dtype=torch.long),
                text_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
                key=f"k{i}-{j}",
            )
            for j in range(batch_size)
        ]
        out.append(packer.pack_batch(chunk))
    return out


def test_jepa_trainer_runs_and_loss_drops() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=32)
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    history = run_multimodal_training(
        model,
        _batches(layout, packer, 10),
        MultimodalTrainerConfig(
            steps=15, lr=3e-3, warmup_steps=2, log_every=5,
            device="cpu", precision="fp32", seed=0,
            jepa_weight=0.1, jepa_ema_decay=0.9,
            jepa_predictor_layers=1, jepa_predictor_heads=2,
        ),
    )
    assert len(history) >= 2
    assert history[-1].loss < history[0].loss


def test_jepa_zero_weight_skips_predictor() -> None:
    """With jepa_weight=0 the trainer must remain backwards-compatible."""
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=24)
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    history = run_multimodal_training(
        model,
        _batches(layout, packer, 4),
        MultimodalTrainerConfig(
            steps=4, lr=1e-3, warmup_steps=1, log_every=1,
            device="cpu", precision="fp32", seed=0,
            jepa_weight=0.0,
        ),
    )
    assert len(history) == 4
