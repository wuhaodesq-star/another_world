"""Tests for video transforms."""

from __future__ import annotations

import pytest
import torch

from another_world.data.datasets.sample import VideoSample
from another_world.data.datasets.transforms import (
    CenterCrop,
    Compose,
    Resize,
    TemporalRandomClip,
    TemporalSample,
    build_default_transform,
    to_float_minus_one_one,
    to_float_zero_one,
)


def _uint8_sample(t: int = 8, h: int = 32, w: int = 32) -> VideoSample:
    return VideoSample(
        frames=torch.randint(0, 256, (t, 3, h, w), dtype=torch.uint8)
    )


def test_to_float_minus_one_one_uint8() -> None:
    s = _uint8_sample()
    s = to_float_minus_one_one(s)
    assert s.frames.dtype == torch.float32
    assert -1.0 <= s.frames.min() <= s.frames.max() <= 1.0


def test_to_float_minus_one_one_already_float() -> None:
    s = VideoSample(frames=torch.zeros(2, 3, 4, 4, dtype=torch.float32))
    s = to_float_minus_one_one(s)
    assert torch.equal(s.frames, torch.zeros(2, 3, 4, 4))


def test_to_float_zero_one() -> None:
    s = _uint8_sample()
    s = to_float_zero_one(s)
    assert s.frames.dtype == torch.float32
    assert 0.0 <= s.frames.min() <= s.frames.max() <= 1.0


def test_resize_changes_size_preserves_temporal() -> None:
    s = _uint8_sample(t=4, h=16, w=16)
    s = Resize(height=8, width=8)(s)
    assert s.frames.shape == (4, 3, 8, 8)
    assert s.frames.dtype == torch.uint8


def test_resize_rejects_bad_dim() -> None:
    bad = VideoSample(frames=torch.zeros(3, 16, 16))  # missing C
    with pytest.raises(ValueError):
        Resize(8, 8)(bad)


def test_center_crop_picks_middle() -> None:
    s = VideoSample(frames=torch.zeros(2, 3, 16, 16, dtype=torch.uint8))
    s.frames[..., 7:9, 7:9] = 255
    s = CenterCrop(height=4, width=4)(s)
    assert s.frames.shape == (2, 3, 4, 4)
    assert (s.frames[..., 1:3, 1:3] == 255).all()


def test_center_crop_too_large_raises() -> None:
    s = _uint8_sample(h=4, w=4)
    with pytest.raises(ValueError):
        CenterCrop(8, 8)(s)


def test_temporal_sample_evenly() -> None:
    s = _uint8_sample(t=20)
    s = TemporalSample(num_frames=5)(s)
    assert s.frames.shape[0] == 5


def test_temporal_sample_pads_short_clip() -> None:
    s = _uint8_sample(t=2)
    s = TemporalSample(num_frames=5, pad="repeat")(s)
    assert s.frames.shape[0] == 5


def test_temporal_sample_error_pad() -> None:
    s = _uint8_sample(t=2)
    with pytest.raises(ValueError):
        TemporalSample(num_frames=5, pad="error")(s)


def test_temporal_random_clip_is_deterministic_with_seed() -> None:
    s1 = _uint8_sample(t=20)
    s2 = VideoSample(frames=s1.frames.clone())
    out1 = TemporalRandomClip(num_frames=5, seed=42)(s1)
    out2 = TemporalRandomClip(num_frames=5, seed=42)(s2)
    assert torch.equal(out1.frames, out2.frames)


def test_compose_runs_in_order() -> None:
    pipeline = Compose([
        TemporalSample(num_frames=4),
        Resize(8, 8),
        to_float_minus_one_one,
    ])
    s = _uint8_sample(t=10, h=32, w=32)
    s = pipeline(s)
    assert s.frames.shape == (4, 3, 8, 8)
    assert s.frames.dtype == torch.float32


def test_build_default_transform_minus_one_one() -> None:
    pipeline = build_default_transform(num_frames=4, height=8, width=8)
    s = _uint8_sample(t=12, h=32, w=32)
    s = pipeline(s)
    assert s.frames.shape == (4, 3, 8, 8)
    assert -1.0 <= s.frames.min() <= s.frames.max() <= 1.0


def test_build_default_transform_zero_one() -> None:
    pipeline = build_default_transform(
        num_frames=4, height=8, width=8, normalise="zero_one"
    )
    s = _uint8_sample(t=12, h=32, w=32)
    s = pipeline(s)
    assert 0.0 <= s.frames.min() <= s.frames.max() <= 1.0


def test_build_default_transform_invalid_args() -> None:
    with pytest.raises(ValueError):
        build_default_transform(
            num_frames=4, height=8, width=8, normalise="bogus"
        )
    with pytest.raises(ValueError):
        build_default_transform(
            num_frames=4, height=8, width=8, temporal="bogus"
        )
