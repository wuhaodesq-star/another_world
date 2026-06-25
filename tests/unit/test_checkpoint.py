"""Tests for the checkpoint save/load module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from another_world.training.checkpoint import (
    CheckpointMeta,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from another_world.training.checkpoint import _split_uri


class _ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_round_trip_model_only(tmp_path: Path) -> None:
    model = _ToyNet()
    meta = CheckpointMeta(step=42, lr=1e-3, config={"x": 1})
    save_checkpoint(tmp_path / "ck", model=model, meta=meta)

    model_b = _ToyNet()
    # Different random init -> at least one parameter differs.
    assert not torch.equal(model.fc1.weight, model_b.fc1.weight)

    loaded = load_checkpoint(tmp_path / "ck", model=model_b)
    assert loaded.step == 42
    assert torch.equal(model.fc1.weight, model_b.fc1.weight)


def test_round_trip_with_optimizer(tmp_path: Path) -> None:
    model = _ToyNet()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 4)
    loss = model(x).pow(2).sum()
    loss.backward()
    optim.step()
    meta = CheckpointMeta(step=7)
    save_checkpoint(tmp_path / "ck", model=model, optimizer=optim, meta=meta)

    model_b = _ToyNet()
    optim_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    load_checkpoint(tmp_path / "ck", model=model_b, optimizer=optim_b)
    a_state = optim.state_dict()
    b_state = optim_b.state_dict()
    assert sorted(a_state) == sorted(b_state)


def test_meta_round_trip(tmp_path: Path) -> None:
    model = _ToyNet()
    meta = CheckpointMeta(step=99, notes="ok", extras={"k": [1, 2, 3]})
    save_checkpoint(tmp_path / "ck", model=model, meta=meta)
    loaded = load_checkpoint(tmp_path / "ck", model=_ToyNet())
    assert loaded.notes == "ok"
    assert loaded.extras == {"k": [1, 2, 3]}
    assert loaded.model_class == "_ToyNet"


def test_strip_fsdp_prefix(tmp_path: Path) -> None:
    model = _ToyNet()
    # Simulate DDP/FSDP by manually writing wrapped keys.
    state = {f"module.{k}": v for k, v in model.state_dict().items()}
    torch.save(state, tmp_path / "model.pt")
    meta = CheckpointMeta(step=0)
    (tmp_path / "meta.json").write_text(meta.to_json(), encoding="utf-8")
    # Direct load (no safetensors file).
    loaded = load_checkpoint(tmp_path, model=_ToyNet())
    assert loaded.step == 0


def test_load_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope", model=_ToyNet())


def test_not_main_is_noop(tmp_path: Path) -> None:
    target = tmp_path / "ck"
    save_checkpoint(
        target,
        model=_ToyNet(),
        meta=CheckpointMeta(step=0),
        is_main=False,
    )
    assert not target.exists()


def test_find_latest_checkpoint(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path) is None
    for step in (10, 30, 20):
        save_checkpoint(
            tmp_path / f"step-{step:08d}",
            model=_ToyNet(),
            meta=CheckpointMeta(step=step),
        )
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "step-00000030"


def test_upload_uri_split() -> None:
    assert _split_uri("r2://bucket/prefix/path") == ("bucket", "prefix/path")
    assert _split_uri("s3://b/p") == ("b", "p")
    with pytest.raises(ValueError):
        _split_uri("ftp://x/y")
    with pytest.raises(ValueError):
        _split_uri("r2://bucket")


def test_upload_uri_triggers_r2_client(tmp_path: Path) -> None:
    model = _ToyNet()
    meta = CheckpointMeta(step=1)

    fake = MagicMock()
    fake_r2 = MagicMock()
    fake_r2.from_env.return_value = fake
    with patch(
        "another_world.training.checkpoint._upload_directory"
    ) as upload_mock:
        save_checkpoint(
            tmp_path / "ck",
            model=model,
            meta=meta,
            upload_uri="r2://my-bucket/runs/exp1",
        )
        upload_mock.assert_called_once()
