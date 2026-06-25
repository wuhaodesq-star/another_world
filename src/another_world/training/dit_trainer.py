"""DiT decoder trainer.

Standalone training loop for the pixel-space DiT (stage 4). The trainer
consumes batches of clean latent / pixel tensors plus token-id contexts,
samples a timestep, and minimises the diffusion loss configured via
:class:`DiffusionObjectiveConfig`.

This trainer mirrors :func:`run_multimodal_training` but operates on a
different data structure (no PackedBatch packing here; the data side
yields raw ``(x0, token_ids)`` pairs).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import nn

from another_world.models.decoder import (
    DiTDecoder,
    DiffusionObjectiveConfig,
    compute_diffusion_loss,
)
from another_world.training.checkpoint import CheckpointMeta, save_checkpoint
from another_world.training.multimodal import _warmup_cosine_lr
from another_world.utils.device import resolve_device, resolve_dtype
from another_world.utils.experiment import ExperimentLogger, create_logger
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


# A batch is just (x0, token_ids) -- both tensors. We keep the type loose
# so callers can use dataclasses or simple tuples.
DiTBatch = tuple[torch.Tensor, torch.Tensor]


@dataclass
class DiTTrainerConfig:
    steps: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 10
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    device: str = "auto"
    precision: str = "fp32"
    grad_accum: int = 1
    seed: int = 0
    objective: DiffusionObjectiveConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.objective is None:
            self.objective = DiffusionObjectiveConfig()


def _infinite(loader: Iterable[DiTBatch]) -> Iterator[DiTBatch]:
    while True:
        yielded = False
        for batch in loader:
            yield batch
            yielded = True
        if not yielded:
            raise RuntimeError("DiT batch iterator yielded zero batches")


def run_dit_training(
    decoder: DiTDecoder,
    batches: Iterable[DiTBatch],
    config: DiTTrainerConfig,
    *,
    logger: ExperimentLogger | None = None,
) -> list[dict]:
    """Train ``decoder`` for ``config.steps`` optimizer steps."""

    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.precision)
    use_autocast = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)

    owns_logger = False
    if logger is None:
        logger = create_logger("disabled")
        owns_logger = True

    decoder.to(device=device)
    decay, nodecay = [], []
    for p in decoder.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=config.lr, betas=config.betas, eps=config.eps,
        fused=device.type == "cuda",
    )

    _LOG.info(
        "Starting DiT training: device=%s precision=%s steps=%d objective=%s",
        device, config.precision, config.steps, config.objective.objective,
    )
    history: list[dict] = []
    data_iter = _infinite(batches)
    decoder.train()
    try:
        for step in range(config.steps):
            t0 = time.perf_counter()
            optim.zero_grad(set_to_none=True)

            accum_loss = 0.0
            metrics: dict = {}
            for _ in range(config.grad_accum):
                x0, token_ids = next(data_iter)
                x0 = x0.to(device=device, dtype=torch.float32, non_blocking=True)
                token_ids = token_ids.to(device=device, non_blocking=True)

                def model_fn(x, t, **_kwargs):
                    return decoder(x, t, token_ids=token_ids)

                if use_autocast:
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        loss, m = compute_diffusion_loss(
                            model_fn, x0=x0, config=config.objective,
                        )
                else:
                    loss, m = compute_diffusion_loss(
                        model_fn, x0=x0, config=config.objective,
                    )
                loss = loss / config.grad_accum
                loss.backward()
                accum_loss += float(loss.detach())
                metrics.update(m)

            grad_norm = 0.0
            if config.grad_clip and config.grad_clip > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        decoder.parameters(), config.grad_clip
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

            if step % config.log_every == 0 or step == config.steps - 1:
                row = {
                    "step": step,
                    "loss": accum_loss,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "elapsed": dt,
                    **metrics,
                }
                history.append(row)
                logger.log(row, step=step)
                _LOG.info(
                    "step=%4d  loss=%.4f  lr=%.2e  gnorm=%.3f",
                    step, accum_loss, lr, grad_norm,
                )
        return history
    finally:
        if owns_logger:
            logger.finish()


__all__ = ["DiTBatch", "DiTTrainerConfig", "run_dit_training"]
