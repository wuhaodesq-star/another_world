"""Tests for the HuggingFace video stream adapter (uses pure-Python mocks)."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from another_world.data.datasets import hf as hf_mod


def _record_with_tensor() -> dict[str, object]:
    return {
        "video": torch.zeros(6, 3, 16, 16, dtype=torch.uint8),
        "caption": "a clip",
        "fps": 30,
        "duration": 0.2,
        "license": "CC-BY",
        "id": "abc",
    }


def _record_with_numpy() -> dict[str, object]:
    arr = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    return {"video": arr, "caption": "np-clip"}


def test_record_to_sample_tensor_payload() -> None:
    sample = hf_mod._record_to_sample(_record_with_tensor(), max_frames=None)
    assert sample.frames.shape == (6, 3, 16, 16)
    assert sample.caption == "a clip"
    assert sample.fps == 30
    assert sample.license == "CC-BY"
    assert sample.key == "abc"
    assert sample.source == "huggingface"


def test_record_to_sample_numpy_payload() -> None:
    sample = hf_mod._record_to_sample(_record_with_numpy(), max_frames=None)
    assert sample.frames.shape == (4, 3, 16, 16)


def test_record_to_sample_respects_max_frames_tensor() -> None:
    rec = _record_with_tensor()
    sample = hf_mod._record_to_sample(rec, max_frames=3)
    assert sample.frames.shape[0] == 3


def test_record_to_sample_missing_video_raises() -> None:
    with pytest.raises(KeyError):
        hf_mod._record_to_sample({"caption": "nothing"}, max_frames=None)


def test_record_to_sample_unsupported_payload_raises() -> None:
    with pytest.raises(TypeError):
        hf_mod._record_to_sample({"video": 12345}, max_frames=None)


def test_hf_iterable_dataset_iterates_records() -> None:
    fake_records = [_record_with_tensor(), _record_with_numpy()]
    fake_module = MagicMock()
    fake_module.load_dataset.return_value = iter(fake_records)

    spec = hf_mod.HfStreamSpec(
        repo_id="dummy/repo", streaming=True, max_frames=None, limit=None,
    )

    with patch.dict("sys.modules", {"datasets": fake_module}):
        ds = hf_mod.HfVideoIterableDataset(spec)
        out = list(ds)

    assert len(out) == 2
    assert out[0].caption == "a clip"
    assert out[1].caption == "np-clip"


def test_hf_iterable_dataset_respects_limit() -> None:
    fake_records = [_record_with_tensor() for _ in range(5)]
    fake_module = MagicMock()
    fake_module.load_dataset.return_value = iter(fake_records)
    spec = hf_mod.HfStreamSpec(
        repo_id="dummy/repo", streaming=True, max_frames=None, limit=2,
    )
    with patch.dict("sys.modules", {"datasets": fake_module}):
        ds = hf_mod.HfVideoIterableDataset(spec)
        out = list(ds)
    assert len(out) == 2


def test_hf_iterable_dataset_skips_bad_records() -> None:
    bad_record: dict[str, object] = {"caption": "missing-video"}
    good_record = _record_with_tensor()
    fake_module = MagicMock()
    fake_module.load_dataset.return_value = iter([bad_record, good_record])
    spec = hf_mod.HfStreamSpec(
        repo_id="dummy/repo", streaming=True, max_frames=None,
    )
    with patch.dict("sys.modules", {"datasets": fake_module}):
        ds = hf_mod.HfVideoIterableDataset(spec)
        out = list(ds)
    assert len(out) == 1
    assert out[0].caption == "a clip"
