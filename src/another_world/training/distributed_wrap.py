"""FSDP2 / DDP wrappers for the multimodal model.

The new ``torch.distributed.fsdp.fully_shard`` API (FSDP2) is the preferred
way to shard our 7B+ model across many H100s. Locally on a single CPU we
fall back to a no-op wrapper (or DDP if multi-process), so the same
trainer code can serve both unit tests and full-scale training.

Reference: https://pytorch.org/blog/fsdp2/ (PyTorch >= 2.4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import nn

from another_world.models.dynamics.multimodal import (
    MultimodalBlock,
    MultimodalDynamicsModel,
)
from another_world.utils.distributed import DistInfo, init_distributed
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass
class FsdpConfig:
    """Knobs for :func:`wrap_model_for_distributed`."""

    strategy: str = "auto"  # auto | fsdp2 | ddp | none
    mixed_precision_dtype: str = "bf16"
    cpu_offload: bool = False
    activation_checkpointing: bool = False


@dataclass
class WrapResult:
    """Return value of :func:`wrap_model_for_distributed`."""

    model: nn.Module
    info: DistInfo
    strategy: str


def _resolve_strategy(strategy: str, info: DistInfo) -> str:
    if strategy == "auto":
        if info.world_size <= 1:
            return "none"
        return "fsdp2" if torch.cuda.is_available() else "ddp"
    return strategy


def _bf16_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wrap_model_for_distributed(
    model: MultimodalDynamicsModel,
    *,
    fsdp: FsdpConfig | None = None,
    info: DistInfo | None = None,
) -> WrapResult:
    """Wrap ``model`` in FSDP2 / DDP / none depending on environment.

    Single-process usage returns the original model unchanged (with
    ``strategy="none"``).
    """

    fsdp = fsdp or FsdpConfig()
    info = info or init_distributed()
    strategy = _resolve_strategy(fsdp.strategy, info)

    _LOG.info(
        "Distributed wrap: strategy=%s world_size=%d rank=%d",
        strategy, info.world_size, info.rank,
    )

    if strategy == "none":
        return WrapResult(model=model, info=info, strategy=strategy)

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        device = torch.device("cuda", info.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        wrapped = DistributedDataParallel(
            model,
            device_ids=[info.local_rank] if device.type == "cuda" else None,
        )
        return WrapResult(model=wrapped, info=info, strategy=strategy)

    if strategy == "fsdp2":
        return _wrap_fsdp2(model, info=info, fsdp=fsdp)

    raise ValueError(f"unknown strategy '{strategy}'")


def _wrap_fsdp2(
    model: MultimodalDynamicsModel,
    *,
    info: DistInfo,
    fsdp: FsdpConfig,
) -> WrapResult:
    try:
        from torch.distributed.fsdp import (  # type: ignore[import-not-found]
            MixedPrecisionPolicy,
            OffloadPolicy,
            fully_shard,
        )
    except ImportError as exc:  # pragma: no cover - on older torch
        raise ImportError(
            "FSDP2 requires torch >= 2.4 with torch.distributed.fsdp"
        ) from exc

    param_dtype = _bf16_dtype(fsdp.mixed_precision_dtype)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
    )
    offload = OffloadPolicy(offload=fsdp.cpu_offload) if fsdp.cpu_offload else None

    kwargs: dict[str, Any] = {"mp_policy": mp_policy}
    if offload is not None:
        kwargs["offload_policy"] = offload

    # Per-block sharding gives a good compute / comm trade-off.
    for block in model.layers:
        if isinstance(block, MultimodalBlock):
            fully_shard(block, **kwargs)
    fully_shard(model, **kwargs)

    return WrapResult(model=model, info=info, strategy="fsdp2")


def shutdown() -> None:
    """Tear down the process group if we initialised one."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "FsdpConfig",
    "WrapResult",
    "shutdown",
    "wrap_model_for_distributed",
]
