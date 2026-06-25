"""Multimodal training loop.

Production-shape trainer for the multimodal dynamics model. Designed to
run on a single GPU (or CPU for tests) and to plug into FSDP2 / TorchTitan
through the same ``run_multimodal_training`` entry point.

Differences from ``smoke.py``:

- Consumes :class:`PackedBatch` directly, no need to glue dataloaders on
  the fly.
- Supports bf16 / fp16 autocast on CUDA.
- Optional activation checkpointing (per-block) to trade compute for memory.
- Optional gradient accumulation across micro-batches.
- Reads its own batches from any iterator of :class:`PackedBatch`, so the
  same trainer accepts in-memory iterators, shard-backed streams, or a
  fully distributed dataloader.
- Reports per-step loss / lr / tokens-per-sec / grad-norm to the
  experiment logger.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from another_world.data.datasets.sequence_packer import PackedBatch
from another_world.models.dynamics.multimodal import (
    MultimodalBlock,
    MultimodalDynamicsModel,
)
from another_world.utils.device import resolve_device, resolve_dtype
from another_world.utils.experiment import ExperimentLogger, create_logger
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config / result types
# ---------------------------------------------------------------------------


@dataclass
class MultimodalTrainerConfig:
    steps: int = 100
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 10
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    device: str = "auto"
    precision: str = "fp32"
    compile: bool = False
    activation_checkpointing: bool = False
    seed: int = 42


@dataclass
class MultimodalStepResult:
    step: int
    loss: float
    lr: float
    tokens_per_sec: float
    grad_norm: float
    elapsed: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warmup_cosine_lr(
    step: int, base_lr: float, warmup: int, total: int, min_ratio: float
) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def _infinite_batches(loader: Iterable[PackedBatch]) -> Iterator[PackedBatch]:
    while True:
        yielded = False
        for batch in loader:
            yield batch
            yielded = True
        if not yielded:
            raise RuntimeError("PackedBatch iterator yielded zero batches")


def apply_activation_checkpointing(model: MultimodalDynamicsModel) -> None:
    """Wrap every transformer block's forward with ``torch.utils.checkpoint``.

    Doing this monkey-patches each ``MultimodalBlock`` so its ``forward`` is
    recomputed during the backward pass, saving activation memory at the
    cost of an extra forward.
    """

    for layer in model.layers:
        if not isinstance(layer, MultimodalBlock):
            continue
        if getattr(layer, "_ac_wrapped", False):
            continue
        original_forward = layer.forward

        def make_wrapped(orig: Callable, mod: MultimodalBlock) -> Callable:
            def wrapped(x, cos, sin):  # noqa: ANN001
                if mod.training and torch.is_grad_enabled():
                    return checkpoint(
                        orig, x, cos, sin, use_reentrant=False,
                    )
                return orig(x, cos, sin)

            return wrapped

        layer.forward = make_wrapped(original_forward, layer)  # type: ignore[assignment]
        layer._ac_wrapped = True  # type: ignore[attr-defined]


def build_optimizer(
    model: nn.Module, config: MultimodalTrainerConfig, *, fused: bool
) -> torch.optim.Optimizer:
    """AdamW with weight-decay applied only to 2-D+ parameters."""

    decay: list[nn.Parameter] = []
    nodecay: list[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        fused=fused,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_multimodal_training(
    model: MultimodalDynamicsModel,
    batches: Iterable[PackedBatch],
    config: MultimodalTrainerConfig,
    *,
    logger: ExperimentLogger | None = None,
) -> list[MultimodalStepResult]:
    """Train ``model`` for ``config.steps`` optimizer steps.

    ``batches`` should yield :class:`PackedBatch` objects. It is wrapped in
    an infinite iterator so finite iterators (e.g. a shard reader) are
    cycled automatically.
    """

    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.precision)
    use_autocast = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)

    owns_logger = False
    if logger is None:
        logger = create_logger("disabled")
        owns_logger = True

    if config.activation_checkpointing:
        apply_activation_checkpointing(model)

    model.to(device=device)
    if config.compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            _LOG.info("torch.compile enabled")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("torch.compile failed; continuing eager: %s", exc)

    optim = build_optimizer(model, config, fused=device.type == "cuda")

    _LOG.info(
        "Starting multimodal training: device=%s precision=%s steps=%d "
        "grad_accum=%d logger=%s ac=%s",
        device, config.precision, config.steps, config.grad_accum,
        logger.backend, config.activation_checkpointing,
    )

    history: list[MultimodalStepResult] = []
    data_iter = _infinite_batches(batches)

    model.train()
    try:
        for step in range(config.steps):
            t0 = time.perf_counter()
            optim.zero_grad(set_to_none=True)

            accum_loss = 0.0
            tokens_in_step = 0
            for _ in range(config.grad_accum):
                batch = next(data_iter).to(device)
                tokens_in_step += int(batch.tokens.numel())

                if use_autocast:
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        out = model(
                            batch.tokens,
                            axes=batch.axes,
                            targets=batch.targets,
                            loss_mask=batch.loss_mask,
                        )
                else:
                    out = model(
                        batch.tokens,
                        axes=batch.axes,
                        targets=batch.targets,
                        loss_mask=batch.loss_mask,
                    )
                loss = out["loss"] / config.grad_accum
                loss.backward()
                accum_loss += float(loss.detach())

            grad_norm = 0.0
            if config.grad_clip and config.grad_clip > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )
                )

            lr = _warmup_cosine_lr(
                step, config.lr, config.warmup_steps, config.steps,
                config.min_lr_ratio,
            )
            for pg in optim.param_groups:
                pg["lr"] = lr
            optim.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tps = tokens_in_step / max(dt, 1e-6)

            if step % config.log_every == 0 or step == config.steps - 1:
                res = MultimodalStepResult(
                    step=step, loss=accum_loss, lr=lr,
                    tokens_per_sec=tps, grad_norm=grad_norm, elapsed=dt,
                )
                history.append(res)
                logger.log(
                    {
                        "loss": res.loss,
                        "lr": res.lr,
                        "tokens_per_sec": res.tokens_per_sec,
                        "grad_norm": res.grad_norm,
                        "elapsed": res.elapsed,
                    },
                    step=step,
                )
                _LOG.info(
                    "step=%4d  loss=%.4f  lr=%.2e  tok/s=%.0f  gnorm=%.3f",
                    res.step, res.loss, res.lr, res.tokens_per_sec, res.grad_norm,
                )
        return history
    finally:
        if owns_logger:
            logger.finish()


__all__ = [
    "MultimodalStepResult",
    "MultimodalTrainerConfig",
    "apply_activation_checkpointing",
    "build_optimizer",
    "run_multimodal_training",
]
