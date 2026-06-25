"""Tests for the experiment logger abstraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from another_world.utils import experiment


def test_disabled_logger_is_no_op() -> None:
    logger = experiment.create_logger("disabled")
    assert logger.backend == "disabled"
    logger.log({"loss": 1.0}, step=0)
    logger.log_config({"lr": 1e-3})
    logger.finish()


def test_jsonl_logger_writes_records(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "run.jsonl"
    logger = experiment.create_logger(
        "jsonl", jsonl_path=path, config={"lr": 1e-3, "model": "toy"}
    )
    assert logger.backend == "jsonl"
    logger.log({"loss": 2.5, "lr": 1e-3}, step=0)
    logger.log({"loss": 2.0}, step=1)
    logger.finish()

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    # First line was the initial config payload from create_logger.
    records = [json.loads(line) for line in lines]
    assert len(records) == 3
    assert records[0]["config"]["lr"] == 1e-3
    assert records[1]["metrics"]["loss"] == 2.5
    assert records[2]["step"] == 1


def test_jsonl_logger_log_config_appends(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    logger = experiment.JsonlLogger(path=path)
    logger.log_config({"a": 1})
    logger.log({"b": 2}, step=0)
    logger.finish()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["config"]["a"] == 1
    assert records[1]["metrics"]["b"] == 2


def test_create_logger_auto_falls_back_to_jsonl_without_wandb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("AW_LOGGER_BACKEND", raising=False)
    with patch.object(experiment, "_wandb_available", return_value=False):
        logger = experiment.create_logger(
            "auto", jsonl_path=tmp_path / "auto.jsonl"
        )
    assert logger.backend == "jsonl"
    logger.finish()


def test_create_logger_auto_picks_wandb_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "dummy-key")
    monkeypatch.delenv("AW_LOGGER_BACKEND", raising=False)

    called = {}

    class FakeWandbLogger:
        backend = "wandb"

        def __init__(self, **kwargs: object) -> None:
            called.update(kwargs)

        def log(self, *a: object, **kw: object) -> None:
            return None

        def log_config(self, *a: object, **kw: object) -> None:
            return None

        def finish(self) -> None:
            return None

    with patch.object(experiment, "_wandb_available", return_value=True), \
         patch.object(experiment, "WandbLogger", FakeWandbLogger):
        logger = experiment.create_logger("auto", project="aw", tags=["t1"])

    assert logger.backend == "wandb"
    assert called["project"] == "aw"
    assert called["tags"] == ["t1"]


def test_env_var_overrides_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AW_LOGGER_BACKEND", "disabled")
    logger = experiment.create_logger(
        "jsonl", jsonl_path=tmp_path / "x.jsonl"
    )
    assert logger.backend == "disabled"


def test_wandb_request_without_package_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AW_LOGGER_BACKEND", raising=False)
    with patch.object(experiment, "_wandb_available", return_value=False):
        logger = experiment.create_logger(
            "wandb", jsonl_path=tmp_path / "fallback.jsonl"
        )
    assert logger.backend == "jsonl"
    logger.finish()


def test_unknown_backend_raises() -> None:
    os.environ.pop("AW_LOGGER_BACKEND", None)
    with pytest.raises(ValueError):
        experiment.create_logger("not-a-backend")  # type: ignore[arg-type]


def test_disabled_logger_satisfies_protocol() -> None:
    logger = experiment.DisabledLogger()
    assert isinstance(logger, experiment.ExperimentLogger)


def test_jsonl_logger_handles_tensor_like(tmp_path: Path) -> None:
    """Make sure torch / numpy scalars are coerced."""
    import torch

    path = tmp_path / "torch.jsonl"
    logger = experiment.JsonlLogger(path=path)
    logger.log({"loss": torch.tensor(0.5)}, step=0)
    logger.finish()
    rec = json.loads(path.read_text().strip())
    assert rec["metrics"]["loss"] == 0.5
