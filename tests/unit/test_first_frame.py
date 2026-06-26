"""Tests for the first-frame preprocessor."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from another_world.inference.first_frame import (
    FirstFramePreprocessor,
    MockFirstFrameTokenizer,
    load_image_as_video_tensor,
)


def test_mock_first_frame_tokenizer_output_shape() -> None:
    tk = MockFirstFrameTokenizer(vocab_size=64, downsample_spatial=4, downsample_temporal=4)
    video = torch.randn(1, 3, 5, 16, 16)
    out = tk.encode(video)[0]
    assert out.shape == (1, 2, 4, 4)  # B, T'=1+(5-1)//4, H'=4, W'=4
    assert (out >= 0).all() and (out < 64).all()


def test_mock_first_frame_tokenizer_rejects_wrong_rank() -> None:
    tk = MockFirstFrameTokenizer()
    with pytest.raises(ValueError):
        tk.encode(torch.randn(3, 16, 16))


def test_preprocessor_from_video_tensor() -> None:
    tk = MockFirstFrameTokenizer(vocab_size=32, downsample_spatial=8, downsample_temporal=4)
    pre = FirstFramePreprocessor(tk, target_h=32, target_w=32)
    video = torch.randn(1, 3, 1, 32, 32)
    ids = pre.from_video_tensor(video)
    assert ids.shape == (1, 4, 4)
    assert ids.dtype == torch.long


def test_preprocessor_rejects_bad_tokenizer_output() -> None:
    class Bad:
        def encode(self, video):
            return (torch.zeros(1, 2),)

    pre = FirstFramePreprocessor(Bad())
    with pytest.raises(ValueError):
        pre.from_video_tensor(torch.randn(1, 3, 1, 16, 16))


def test_load_image_as_video_tensor(tmp_path: Path) -> None:
    from PIL import Image
    import numpy as np

    path = tmp_path / "image.png"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[..., 0] = 255
    Image.fromarray(arr).save(path)

    video = load_image_as_video_tensor(path, height=8, width=8, pad_frames=3)
    assert video.shape == (1, 3, 3, 8, 8)
    assert video.min() >= -1.0 and video.max() <= 1.0
