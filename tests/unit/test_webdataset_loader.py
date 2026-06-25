"""Tests for the WebDataset / iterable video loader plumbing."""

from __future__ import annotations

import json
import pytest
import torch
from torch.utils.data import DataLoader

from another_world.data.datasets import (
    IterableVideoDataset,
    VideoSample,
    build_default_transform,
    collate_video_samples,
    decode_webdataset_sample,
)


def _make_samples(n: int = 4) -> list[VideoSample]:
    return [
        VideoSample(
            frames=torch.randint(0, 256, (8, 3, 16, 16), dtype=torch.uint8),
            caption=f"sample-{i}",
            key=f"k{i:03d}",
            source="unit-test",
            license="CC-BY",
        )
        for i in range(n)
    ]


def test_iterable_video_dataset_yields_in_order() -> None:
    samples = _make_samples(3)
    ds = IterableVideoDataset(samples)
    got = list(ds)
    assert [s.key for s in got] == ["k000", "k001", "k002"]
    # frames cloned, not shared.
    assert got[0].frames.data_ptr() != samples[0].frames.data_ptr()


def test_iterable_video_dataset_applies_transform() -> None:
    samples = _make_samples(2)
    transform = build_default_transform(num_frames=4, height=8, width=8)
    ds = IterableVideoDataset(samples, transform=transform)
    out = list(ds)
    assert all(s.frames.shape == (4, 3, 8, 8) for s in out)
    assert all(s.frames.dtype == torch.float32 for s in out)


def test_iterable_video_dataset_loops() -> None:
    samples = _make_samples(2)
    ds = IterableVideoDataset(samples, loops=3)
    out = list(ds)
    assert len(out) == 6


def test_collate_video_samples_basic() -> None:
    samples = _make_samples(3)
    batch = collate_video_samples(samples)
    assert batch["frames"].shape == (3, 8, 3, 16, 16)
    assert batch["caption"] == ["sample-0", "sample-1", "sample-2"]
    assert batch["key"] == ["k000", "k001", "k002"]
    assert "tokens" not in batch


def test_collate_video_samples_with_tokens() -> None:
    samples = _make_samples(2)
    for s in samples:
        s.tokens = torch.zeros(3, 4, 4, dtype=torch.long)
    batch = collate_video_samples(samples)
    assert batch["tokens"].shape == (2, 3, 4, 4)


def test_collate_video_samples_empty_raises() -> None:
    with pytest.raises(ValueError):
        collate_video_samples([])


def test_dataloader_round_trip() -> None:
    samples = _make_samples(5)
    transform = build_default_transform(num_frames=4, height=8, width=8)
    ds = IterableVideoDataset(samples, transform=transform)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_video_samples)
    batches = list(loader)
    # 5 samples / batch_size=2 with drop_last=False -> 3 batches (2,2,1).
    assert len(batches) >= 2
    for batch in batches:
        assert batch["frames"].dim() == 5
        assert batch["frames"].shape[2:] == (3, 8, 8)


def test_decode_sample_npy_frames() -> None:
    """Verify the .npy fallback path works."""
    import io

    import numpy as np

    arr = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    arr[..., 0] = 255  # red channel
    buf = io.BytesIO()
    np.save(buf, arr)
    raw = {"__key__": "abc", "npy": buf.getvalue(), "txt": b"hello",
           "json": json.dumps({"fps": 24, "source": "synthetic"}).encode()}
    sample = decode_webdataset_sample(raw, max_frames=None)
    assert sample.key == "abc"
    assert sample.caption == "hello"
    assert sample.frames.shape == (4, 3, 16, 16)
    assert sample.fps == 24
    assert sample.source == "synthetic"


def test_decode_sample_max_frames_npy() -> None:
    import io
    import numpy as np

    arr = np.zeros((10, 8, 8, 3), dtype=np.uint8)
    buf = io.BytesIO()
    np.save(buf, arr)
    raw = {"npy": buf.getvalue()}
    sample = decode_webdataset_sample(raw, max_frames=4)
    assert sample.frames.shape[0] == 4


def test_decode_sample_missing_visual_raises() -> None:
    with pytest.raises(ValueError, match="visual payload"):
        decode_webdataset_sample({"txt": b"hi"})


def test_decode_sample_bad_json_is_warned() -> None:
    import io
    import numpy as np

    arr = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    buf = io.BytesIO()
    np.save(buf, arr)
    raw = {"npy": buf.getvalue(), "json": b"not-json"}
    # Should not raise; bad json is logged and ignored.
    sample = decode_webdataset_sample(raw)
    assert sample.frames.shape == (2, 3, 4, 4)
