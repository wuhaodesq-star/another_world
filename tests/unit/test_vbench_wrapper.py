"""Tests for the VBench wrapper."""

from __future__ import annotations

import pytest
import torch

from another_world.eval.vbench_wrapper import (
    VBENCH_DIMENSIONS,
    VBenchAdapter,
    vbench_or_fallback,
)


def test_default_dimensions_constant() -> None:
    assert "subject_consistency" in VBENCH_DIMENSIONS
    assert "motion_smoothness" in VBENCH_DIMENSIONS


def test_adapter_rejects_unknown_dim() -> None:
    with pytest.raises(ValueError):
        VBenchAdapter(dimensions=("bogus",))


def test_adapter_returns_all_dimensions() -> None:
    adapter = VBenchAdapter()
    videos = torch.randn(2, 3, 4, 16, 16)
    out = adapter(videos)
    for dim in VBENCH_DIMENSIONS:
        assert dim in out
    assert "overall" in out
    for v in out.values():
        assert isinstance(v, float)


def test_adapter_consumes_btchw() -> None:
    adapter = VBenchAdapter()
    videos = torch.randn(2, 4, 3, 16, 16)  # [B, T, C, H, W]
    out = adapter(videos)
    assert "overall" in out


def test_adapter_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError):
        VBenchAdapter()(torch.randn(3, 4, 5))


def test_subject_consistency_static_video_is_high() -> None:
    adapter = VBenchAdapter(dimensions=("subject_consistency",))
    static = torch.ones(1, 3, 4, 8, 8)
    out = adapter(static)
    assert out["subject_consistency"] >= 0.99


def test_motion_smoothness_static_video_is_high() -> None:
    adapter = VBenchAdapter(dimensions=("motion_smoothness",))
    static = torch.ones(1, 3, 4, 8, 8)
    out = adapter(static)
    assert out["motion_smoothness"] >= 0.99


def test_dynamic_degree_static_is_zero() -> None:
    adapter = VBenchAdapter(dimensions=("dynamic_degree",))
    static = torch.ones(1, 3, 4, 8, 8)
    out = adapter(static)
    assert out["dynamic_degree"] == 0.0


def test_vbench_or_fallback_basic() -> None:
    videos = torch.randn(2, 3, 4, 16, 16)
    out = vbench_or_fallback(videos)
    assert "overall" in out
