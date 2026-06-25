"""Tests for the distributed bring-up helpers (single-process paths)."""

from __future__ import annotations

import pytest

from another_world.utils import distributed


def test_init_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    info = distributed.init_distributed(backend="gloo")
    assert info.rank == 0
    assert info.world_size == 1
    assert info.local_rank == 0
    assert info.is_main
    # No process group should have been created.
    import torch.distributed as dist
    assert not dist.is_initialized()


def test_all_reduce_sum_single_process_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    distributed.init_distributed(backend="gloo")
    assert distributed.all_reduce_sum(3.5) == 3.5


def test_shutdown_is_safe_when_not_initialised() -> None:
    distributed.shutdown_distributed()  # must not raise


def test_dist_info_is_immutable() -> None:
    info = distributed.DistInfo(rank=0, world_size=1, local_rank=0, backend="gloo")
    with pytest.raises(Exception):
        info.rank = 5  # type: ignore[misc]
