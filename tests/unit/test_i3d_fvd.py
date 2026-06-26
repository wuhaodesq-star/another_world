"""Tests for the I3D FVD wrapper."""

from __future__ import annotations

import pytest
import torch

from another_world.eval.i3d_fvd import FVDConfig, I3DFVD


def test_pixel_backend_matches_shape_and_returns_float() -> None:
    torch.manual_seed(0)
    real = torch.randn(4, 3, 2, 8, 8)
    fake = torch.randn(4, 3, 2, 8, 8)
    score = I3DFVD(FVDConfig(backend="pixel"))(real, fake)
    assert isinstance(score, float)
    assert score >= 0


def test_i3d_backend_with_injected_extractor() -> None:
    torch.manual_seed(0)

    def extractor(videos: torch.Tensor) -> torch.Tensor:
        # Simple pooled feature extractor: [B, C]
        return videos.mean(dim=[2, 3, 4])

    real = torch.randn(8, 3, 2, 8, 8)
    fake = real + 0.1
    score = I3DFVD(
        FVDConfig(backend="i3d", batch_size=3, device="cpu"),
        extractor=extractor,
    )(real, fake)
    assert isinstance(score, float)
    assert score >= 0


def test_i3d_backend_falls_back_when_no_extractor() -> None:
    real = torch.randn(4, 3, 2, 8, 8)
    fake = torch.randn(4, 3, 2, 8, 8)
    score = I3DFVD(FVDConfig(backend="i3d", strict_i3d=False))(real, fake)
    assert isinstance(score, float)


def test_i3d_backend_strict_raises_without_extractor() -> None:
    real = torch.randn(4, 3, 2, 8, 8)
    fake = torch.randn(4, 3, 2, 8, 8)
    with pytest.raises(RuntimeError):
        I3DFVD(FVDConfig(backend="i3d", strict_i3d=True))(real, fake)


def test_unknown_backend_raises() -> None:
    real = torch.randn(2, 3, 2, 4, 4)
    fake = torch.randn(2, 3, 2, 4, 4)
    with pytest.raises(ValueError):
        I3DFVD(FVDConfig(backend="bogus"))(real, fake)


def test_extractor_wrong_shape_raises() -> None:
    def bad(videos: torch.Tensor) -> torch.Tensor:
        return torch.randn(videos.shape[0], 3, 4)

    real = torch.randn(2, 3, 2, 4, 4)
    fake = torch.randn(2, 3, 2, 4, 4)
    with pytest.raises(ValueError):
        I3DFVD(FVDConfig(backend="i3d"), extractor=bad)(real, fake)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        I3DFVD()(torch.zeros(1, 3, 2, 4, 4), torch.zeros(1, 3, 2, 4, 5))
