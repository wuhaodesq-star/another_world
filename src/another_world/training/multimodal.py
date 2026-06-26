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
- Optional periodic checkpoint saving.
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
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from another_world.data.datasets.sequence_packer import PackedBatch
from another_world.models.dynamics.multimodal import (
    MultimodalBlock,
    MultimodalDynamicsModel,
)
from another_world.models.jepa import (
    EmaShadow,
    JEPAConfig,
    JEPALatentPredictor,
    jepa_loss,
)
from another_world.tokenizers.vocab import VocabInfo, VocabLayout
from another_world.training.cfg_dropout import ConditioningDropout
from another_world.training.checkpoint import CheckpointMeta, save_checkpoint
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
    # Checkpointing
    checkpoint_dir: str | None = None
    checkpoint_every: int = 0          # 0 disables periodic saves
    checkpoint_keep: int = 3           # number of recent checkpoints to keep
    checkpoint_upload_uri: str | None = None  # r2://bucket/prefix
    is_main: bool = True               # rank 0 in distributed runs
    # JEPA auxiliary loss
    jepa_weight: float = 0.0           # 0 disables JEPA
    jepa_ema_decay: float = 0.999
    jepa_predictor_hidden: int = 0     # 0 -> use dim as hidden width
    jepa_predictor_layers: int = 2
    jepa_predictor_heads: int = 4
    # Classifier-free guidance training dropout
    cfg_text_drop_prob: float = 0.0
    cfg_action_drop_prob: float = 0.0
    cfg_null_token_id: int = 0

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

    return build_optimizer_for_params(
        list(model.parameters()), config, fused=fused,
    )


def build_optimizer_for_params(
    params: list[nn.Parameter], config: MultimodalTrainerConfig, *, fused: bool
) -> torch.optim.Optimizer:
    """AdamW splitting params into decay (>=2D) / no-decay groups."""

    decay: list[nn.Parameter] = []
    nodecay: list[nn.Parameter] = []
    for p in params:
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

    cfg_dropout: ConditioningDropout | None = None
    if config.cfg_text_drop_prob > 0.0 or config.cfg_action_drop_prob > 0.0:
        cfg_dropout = ConditioningDropout(
            null_token_id=config.cfg_null_token_id,
            text_drop_prob=config.cfg_text_drop_prob,
            action_drop_prob=config.cfg_action_drop_prob,
            seed=config.seed,
        )
        _LOG.info(
            "CFG dropout enabled: text=%.2f action=%.2f null=%d",
            config.cfg_text_drop_prob,
            config.cfg_action_drop_prob,
            config.cfg_null_token_id,
        )

    # JEPA auxiliary head + EMA target encoder.
    jepa_predictor: JEPALatentPredictor | None = None
    jepa_ema: EmaShadow | None = None
    jepa_target: MultimodalDynamicsModel | None = None
    if config.jepa_weight > 0.0:
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model  # torch.compile
        hidden_dim = inner.config.dim
        jepa_cfg = JEPAConfig(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            hidden_dim=config.jepa_predictor_hidden or hidden_dim,
            n_layers=config.jepa_predictor_layers,
            n_heads=config.jepa_predictor_heads,
        )
        jepa_predictor = JEPALatentPredictor(jepa_cfg).to(device=device)
        # Frozen EMA target encoder (initialised from the model).
        jepa_target = MultimodalDynamicsModel(inner.config).to(device=device)
        jepa_target.load_state_dict(inner.state_dict())
        for p in jepa_target.parameters():
            p.requires_grad_(False)
        jepa_ema = EmaShadow(inner, decay=config.jepa_ema_decay)
        _LOG.info(
            "JEPA enabled: weight=%.3f decay=%.4f predictor=%.2fM",
            config.jepa_weight, config.jepa_ema_decay,
            jepa_predictor.num_parameters / 1e6,
        )

    optim_params = list(model.parameters())
    if jepa_predictor is not None:
        optim_params += list(jepa_predictor.parameters())
    optim = build_optimizer_for_params(
        optim_params, config, fused=device.type == "cuda",
    )

    _LOG.info(
        "Starting multimodal training: device=%s precision=%s steps=%d "
        "grad_accum=%d logger=%s ac=%s jepa=%.2f",
        device, config.precision, config.steps, config.grad_accum,
        logger.backend, config.activation_checkpointing, config.jepa_weight,
    )

    history: list[MultimodalStepResult] = []
    data_iter = _infinite_batches(batches)

    model.train()
    try:
        for step in range(config.steps):
            t0 = time.perf_counter()
            optim.zero_grad(set_to_none=True)

            accum_loss = 0.0
            accum_jepa = 0.0
            tokens_in_step = 0
            jepa_metrics: dict[str, float] = {}
            for _ in range(config.grad_accum):
                batch = next(data_iter).to(device)
                if cfg_dropout is not None:
                    batch = cfg_dropout(batch)
                tokens_in_step += int(batch.tokens.numel())

                need_hidden = jepa_predictor is not None
                if use_autocast:
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        out = model(
                            batch.tokens,
                            axes=batch.axes,
                            targets=batch.targets,
                            loss_mask=batch.loss_mask,
                            return_hidden=need_hidden,
                        )
                else:
                    out = model(
                        batch.tokens,
                        axes=batch.axes,
                        targets=batch.targets,
                        loss_mask=batch.loss_mask,
                        return_hidden=need_hidden,
                    )

                main_loss = out["loss"]
                loss = main_loss
                if jepa_predictor is not None and jepa_target is not None:
                    with torch.no_grad():
                        target_out = jepa_target(
                            batch.tokens, axes=batch.axes,
                            return_hidden=True,
                        )
                    aux_loss, jm = jepa_loss(
                        jepa_predictor,
                        student_hidden=out["hidden_states"],
                        target_hidden=target_out["hidden_states"],
                        mask=batch.loss_mask,
                    )
                    loss = main_loss + config.jepa_weight * aux_loss
                    accum_jepa += float(aux_loss.detach())
                    jepa_metrics = jm

                loss = loss / config.grad_accum
                loss.backward()
                accum_loss += float(main_loss.detach()) / config.grad_accum

            grad_norm = 0.0
            if config.grad_clip and config.grad_clip > 0:
                params_to_clip = list(model.parameters())
                if jepa_predictor is not None:
                    params_to_clip += list(jepa_predictor.parameters())
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        params_to_clip, config.grad_clip
                    )
                )

            lr = _warmup_cosine_lr(
                step, config.lr, config.warmup_steps, config.steps,
                config.min_lr_ratio,
            )
            for pg in optim.param_groups:
                pg["lr"] = lr
            optim.step()

            # JEPA: advance EMA shadow + sync target encoder.
            if jepa_ema is not None and jepa_target is not None:
                inner = model._orig_mod if hasattr(model, "_orig_mod") else model
                jepa_ema.update(inner)
                jepa_ema.copy_to(jepa_target)

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
                payload: dict[str, object] = {
                    "loss": res.loss,
                    "lr": res.lr,
                    "tokens_per_sec": res.tokens_per_sec,
                    "grad_norm": res.grad_norm,
                    "elapsed": res.elapsed,
                }
                if config.jepa_weight > 0.0:
                    payload["jepa_loss"] = accum_jepa
                    payload.update(jepa_metrics)
                logger.log(payload, step=step)
                _LOG.info(
                    "step=%4d  loss=%.4f  lr=%.2e  tok/s=%.0f  gnorm=%.3f"
                    + ("  jepa=%.4f" if config.jepa_weight > 0 else ""),
                    res.step, res.loss, res.lr, res.tokens_per_sec, res.grad_norm,
                    *([accum_jepa] if config.jepa_weight > 0 else []),
                )

            _maybe_save_checkpoint(
                model, optim, config, step=step + 1, lr=lr,
                final=(step == config.steps - 1),
            )
        return history
    finally:
        if owns_logger:
            logger.finish()


def _maybe_save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: MultimodalTrainerConfig,
    *,
    step: int,
    lr: float,
    final: bool,
) -> None:
    if not config.checkpoint_dir:
        return
    interval = max(config.checkpoint_every, 0)
    should_save = final or (interval > 0 and step % interval == 0)
    if not should_save:
        return

    directory = Path(config.checkpoint_dir) / f"step-{step:08d}"
    meta = CheckpointMeta(
        step=step,
        lr=lr,
        config={"trainer": _config_as_dict(config)},
    )
    save_checkpoint(
        directory,
        model=model,
        optimizer=optimizer,
        meta=meta,
        is_main=config.is_main,
        upload_uri=config.checkpoint_upload_uri,
    )
    _prune_old_checkpoints(Path(config.checkpoint_dir), keep=config.checkpoint_keep,
                           is_main=config.is_main)


def _config_as_dict(config: MultimodalTrainerConfig) -> dict:
    return {
        k: v for k, v in vars(config).items()
        if not k.startswith("_")
    }


def _prune_old_checkpoints(root: Path, *, keep: int, is_main: bool) -> None:
    if not is_main or keep <= 0 or not root.exists():
        return
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("step-")),
        key=lambda p: p.name,
    )
    excess = candidates[: max(0, len(candidates) - keep)]
    for old in excess:
        try:
            import shutil

            shutil.rmtree(old)
            _LOG.info("Pruned old checkpoint %s", old)
        except OSError as exc:
            _LOG.warning("Failed to prune %s: %s", old, exc)


__all__ = [
    "MultimodalStepResult",
    "MultimodalTrainerConfig",
    "apply_activation_checkpointing",
    "build_optimizer",
    "build_optimizer_for_params",
    "run_multimodal_training",
]
