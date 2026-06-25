"""Tests for the distributed-wrap helpers (single-process paths)."""

from __future__ import annotations

import pytest

from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout
from another_world.training.distributed_wrap import (
    FsdpConfig,
    wrap_model_for_distributed,
)
from another_world.utils.distributed import DistInfo


def _model() -> MultimodalDynamicsModel:
    return MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(VocabLayout.tiny().total_size)
    )


def test_wrap_returns_original_when_single_process(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    model = _model()
    result = wrap_model_for_distributed(model, fsdp=FsdpConfig(strategy="auto"))
    assert result.strategy == "none"
    assert result.model is model


def test_explicit_none_strategy_short_circuits(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    model = _model()
    result = wrap_model_for_distributed(
        model,
        fsdp=FsdpConfig(strategy="none"),
        info=DistInfo(rank=0, world_size=4, local_rank=0, backend="gloo"),
    )
    assert result.strategy == "none"
    assert result.model is model


def test_unknown_strategy_raises(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    model = _model()
    with pytest.raises(ValueError):
        wrap_model_for_distributed(
            model,
            fsdp=FsdpConfig(strategy="bogus"),
            info=DistInfo(rank=0, world_size=2, local_rank=0, backend="gloo"),
        )
