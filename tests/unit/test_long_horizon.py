"""Tests for the long-horizon evaluation harness."""

from __future__ import annotations

import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import SequencePacker
from another_world.eval.long_horizon import HorizonResult, evaluate_long_horizon
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout


def test_evaluate_long_horizon_returns_per_horizon() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=32)
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    model.eval()

    sample = TokenSample(
        visual_tokens=torch.randint(0, 16, (1, 2, 2), dtype=torch.long),
        text_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
        key="k",
    )
    batch = packer.pack_batch([sample])
    horizons = [4, 8, 12]
    result = evaluate_long_horizon(
        model,
        tokens=batch.tokens,
        axes=batch.axes,
        targets=batch.targets,
        horizons=horizons,
    )
    assert isinstance(result, HorizonResult)
    assert result.horizons == sorted(horizons)
    assert len(result.accuracy) == len(horizons)
    assert len(result.top5) == len(horizons)
    for a in result.accuracy:
        assert 0.0 <= a <= 1.0


def test_horizon_result_to_dict() -> None:
    r = HorizonResult(horizons=[1, 2], accuracy=[0.5, 0.25], top5=[0.8, 0.6])
    d = r.to_dict()
    assert d["acc_h1"] == 0.5
    assert d["top5_h2"] == 0.6
    assert d["acc_mean"] == 0.375
