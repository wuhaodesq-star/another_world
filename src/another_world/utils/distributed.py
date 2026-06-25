"""Distributed bring-up helpers.

For stage 0 we just need to verify that:

1. ``torch.distributed.init_process_group`` works.
2. ``DistributedDataParallel`` wraps the toy model on CPU (gloo) and on
   GPU (nccl) without code changes.
3. An all-reduce produces the expected aggregated result.

The real distributed trainer (FSDP2 / TorchTitan) takes over in stage 3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _pick_backend(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def init_distributed(backend: str | None = None) -> DistInfo:
    """Initialise ``torch.distributed`` from environment variables.

    Compatible with ``torchrun``: requires ``RANK``, ``WORLD_SIZE``, and
    ``LOCAL_RANK`` to be set in the environment.

    Returns:
        :class:`DistInfo` even when running single-process (rank=0, ws=1).
        In single-process mode no process group is initialised.
    """

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    chosen = _pick_backend(backend)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=chosen,
            rank=rank,
            world_size=world_size,
        )
        if chosen == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    info = DistInfo(
        rank=rank, world_size=world_size, local_rank=local_rank, backend=chosen,
    )
    _LOG.info(
        "Distributed init: rank=%d world_size=%d local_rank=%d backend=%s",
        info.rank, info.world_size, info.local_rank, info.backend,
    )
    return info


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(value: float) -> float:
    """All-reduce a scalar across ranks (sum). No-op when single process."""

    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


__all__ = [
    "DistInfo",
    "all_reduce_sum",
    "init_distributed",
    "shutdown_distributed",
]
