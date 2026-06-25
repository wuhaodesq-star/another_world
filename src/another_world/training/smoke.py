"""Minimal single-process trainer used for the stage-0 smoke test.

The real distributed trainer (FSDP2 / TorchTitan) will live alongside this
file in stage 3. We deliberately keep this implementation small so it can
run on CPU in CI without any GPU-only dependencies.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from another_world.utils.device import resolve_device, resolve_dtype
from another_world.utils.experiment import ExperimentLogger, create_logger
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass
class SmokeTrainerConfig:
    steps: int = 100
    batch_size: int = 4
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 10
    grad_clip: float = 1.0
    log_every: int = 10
    device: str = "auto"
    precision: str = "fp32"
    seed: int = 42


@dataclass
class TrainStepResult:
    step: int
    loss: float
    lr: float
    tokens_per_sec: float


def _warmup_cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _infinite_loader(loader: Iterable[tuple[Tensor, Tensor]]) -> Iterable[tuple[Tensor, Tensor]]:
    while True:
        yield from loader


def run_smoke_training(
    model: nn.Module,
    dataset: Dataset[tuple[Tensor, Tensor]],
    config: SmokeTrainerConfig,
    logger: ExperimentLogger | None = None,
) -> list[TrainStepResult]:
    """Train ``model`` on ``dataset`` for ``config.steps`` optimizer steps.

    Returns the per-logged-step metrics so callers (tests, scripts) can
    assert things like "loss decreased".
    """

    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.precision)
    use_autocast = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)

    owns_logger = False
    if logger is None:
        logger = create_logger("disabled")
        owns_logger = True

    model.to(device=device)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )

    history: list[TrainStepResult] = []
    data_iter = iter(_infinite_loader(loader))

    _LOG.info(
        "Starting smoke training: device=%s precision=%s steps=%d batch=%d "
        "logger=%s",
        device, config.precision, config.steps, config.batch_size,
        logger.backend,
    )

    model.train()
    try:
        for step in range(config.steps):
            t0 = time.perf_counter()
            optim.zero_grad(set_to_none=True)

            accum_loss = 0.0
            tokens_in_step = 0
            for _ in range(config.grad_accum):
                inputs, targets = next(data_iter)
                inputs = inputs.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)
                tokens_in_step += inputs.numel()

                if use_autocast:
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        out = model(inputs, targets=targets)
                    loss = out["loss"] / config.grad_accum
                else:
                    out = model(inputs, targets=targets)
                    loss = out["loss"] / config.grad_accum
                loss.backward()
                accum_loss += loss.item()

            grad_norm = 0.0
            if config.grad_clip and config.grad_clip > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )
                )

            lr = _warmup_cosine_lr(step, config.lr, config.warmup_steps, config.steps)
            for pg in optim.param_groups:
                pg["lr"] = lr

            optim.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tps = tokens_in_step / max(dt, 1e-6)

            if step % config.log_every == 0 or step == config.steps - 1:
                res = TrainStepResult(step=step, loss=accum_loss, lr=lr, tokens_per_sec=tps)
                history.append(res)
                logger.log(
                    {
                        "loss": res.loss,
                        "lr": res.lr,
                        "tokens_per_sec": res.tokens_per_sec,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )
                _LOG.info(
                    "step=%4d  loss=%.4f  lr=%.2e  tok/s=%.0f  gnorm=%.3f",
                    res.step, res.loss, res.lr, res.tokens_per_sec, grad_norm,
                )

        return history
    finally:
        if owns_logger:
            logger.finish()


__all__ = ["SmokeTrainerConfig", "TrainStepResult", "run_smoke_training"]
