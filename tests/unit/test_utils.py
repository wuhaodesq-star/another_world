"""Tests for utility helpers."""

from __future__ import annotations

import pytest
import torch

from another_world.utils.device import resolve_device, resolve_dtype
from another_world.utils.logging import get_logger


def test_resolve_device_auto() -> None:
    dev = resolve_device("auto")
    assert dev.type in {"cpu", "cuda"}


def test_resolve_device_explicit() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_dtype_known() -> None:
    assert resolve_dtype("fp32") is torch.float32
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype("fp16") is torch.float16


def test_resolve_dtype_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_dtype("nope")


def test_logger_is_idempotent() -> None:
    a = get_logger("aw.test")
    b = get_logger("aw.test")
    assert a is b
    assert len(a.handlers) == 1
