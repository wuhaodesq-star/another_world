"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import pytest
import torch

from another_world.eval.metrics import (
    fvd_score,
    gaussian_frechet_distance,
    long_horizon_drift,
    mae,
    mse,
    psnr,
    temporal_consistency,
    token_accuracy,
    token_top_k,
)


def test_mse_zero_for_identical() -> None:
    x = torch.randn(2, 3, 4)
    assert mse(x, x) == 0.0
    assert mae(x, x) == 0.0


def test_mse_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mse(torch.zeros(2, 3), torch.zeros(3, 2))


def test_psnr_infinity_for_identical() -> None:
    x = torch.zeros(2, 3)
    assert math.isinf(psnr(x, x))


def test_psnr_finite_for_noise() -> None:
    x = torch.zeros(8, 8)
    y = torch.randn(8, 8) * 0.1
    val = psnr(x, y)
    assert math.isfinite(val)
    assert val > 0


def test_temporal_consistency_zero_for_static() -> None:
    static = torch.zeros(1, 3, 4, 8, 8)
    assert temporal_consistency(static) == 0.0


def test_temporal_consistency_handles_btchw_input() -> None:
    """Auto-detects [B, T, C, H, W] layout."""
    video = torch.zeros(1, 4, 3, 8, 8)
    assert temporal_consistency(video) == 0.0


def test_temporal_consistency_positive_for_dynamic() -> None:
    video = torch.randn(1, 3, 4, 8, 8)
    assert temporal_consistency(video) > 0


def test_long_horizon_drift_zero_for_perfect_predictions() -> None:
    preds = [torch.ones(2, 3) for _ in range(3)]
    targets = [torch.ones(2, 3) for _ in range(3)]
    result = long_horizon_drift(preds, targets)
    assert result["mse_mean"] == 0.0
    assert pytest.approx(result["cos_mean"], abs=1e-5) == 1.0


def test_long_horizon_drift_records_per_step() -> None:
    preds = [torch.zeros(2, 3), torch.ones(2, 3)]
    targets = [torch.ones(2, 3), torch.ones(2, 3)]
    result = long_horizon_drift(preds, targets)
    assert result["mse_step_0"] == 1.0
    assert pytest.approx(result["mse_step_1"], abs=1e-6) == 0.0


def test_token_accuracy_perfect() -> None:
    logits = torch.eye(5).unsqueeze(0)  # [1, 5, 5]
    targets = torch.arange(5).unsqueeze(0)  # [1, 5]
    assert token_accuracy(logits, targets) == 1.0


def test_token_accuracy_respects_ignore_index() -> None:
    logits = torch.zeros(1, 5, 4)  # always predicts 0
    targets = torch.tensor([[0, 1, -100, 1, 0]])
    # Visible targets: [0, 1, 1, 0], correct on first and last -> 2/4 = 0.5.
    assert token_accuracy(logits, targets) == 0.5


def test_token_top_k() -> None:
    logits = torch.tensor([[
        [0.0, 0.1, 0.2, 0.3, 0.4],   # argmax = 4
        [0.4, 0.3, 0.2, 0.1, 0.0],   # argmax = 0
    ]])
    targets = torch.tensor([[4, 0]])
    assert token_top_k(logits, targets, k=1) == 1.0
    assert token_top_k(logits, targets, k=2) == 1.0
    # Now a non-trivial case: target at the 2nd-most-likely slot.
    targets_b = torch.tensor([[3, 1]])
    assert token_top_k(logits, targets_b, k=1) == 0.0
    assert token_top_k(logits, targets_b, k=2) == 1.0


def test_gaussian_frechet_distance_zero_for_identical_dists() -> None:
    torch.manual_seed(0)
    a = torch.randn(128, 16)
    assert gaussian_frechet_distance(a, a) < 1e-3


def test_fvd_score_positive_for_different_videos() -> None:
    torch.manual_seed(0)
    real = torch.randn(8, 3, 4, 16, 16)
    fake = torch.randn(8, 3, 4, 16, 16) * 3
    score = fvd_score(real, fake)
    assert score > 0


def test_fvd_score_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        fvd_score(torch.zeros(2, 3, 4, 8, 8), torch.zeros(2, 3, 4, 8, 16))
