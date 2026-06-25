"""Tests for VideoSample / TokenSample dataclasses."""

from __future__ import annotations

import torch

from another_world.data.datasets.sample import TokenSample, VideoSample


def test_video_sample_basic_properties() -> None:
    frames = torch.zeros(8, 3, 64, 64, dtype=torch.uint8)
    s = VideoSample(frames=frames, caption="hi", source="test")
    assert s.num_frames == 8
    assert s.resolution == (64, 64)
    assert s.caption == "hi"
    d = s.to_dict()
    assert d["frames"].shape == (8, 3, 64, 64)
    assert d["source"] == "test"


def test_video_sample_extra_round_trip() -> None:
    frames = torch.zeros(2, 3, 4, 4)
    s = VideoSample(frames=frames, extra={"k": 1})
    assert s.to_dict()["k"] == 1


def test_token_sample_optional_fields() -> None:
    s = TokenSample(visual_tokens=torch.zeros(3, 4, 5, dtype=torch.long))
    assert s.text_tokens is None
    assert s.action_tokens is None
