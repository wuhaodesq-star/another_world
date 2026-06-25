"""``aw-train-dit`` CLI.

Train the DiT decoder on raw latent/pixel batches paired with conditioning
token ids. Designed to plug into datasets produced by stage 1.3 (token
shards) or arbitrary user-provided ``(x0, token_ids)`` iterables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from another_world.models.decoder import (
    DiTDecoder,
    DiTDecoderConfig,
    DiffusionObjectiveConfig,
)
from another_world.tokenizers.vocab import VocabLayout
from another_world.training.checkpoint import (
    CheckpointMeta,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from another_world.training.dit_trainer import DiTTrainerConfig, run_dit_training
from another_world.utils.experiment import create_logger
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


def _synthetic_batches(args: argparse.Namespace, vocab_size: int):
    """Random ``(x0, token_ids)`` pairs for smoke testing."""

    g = torch.Generator().manual_seed(args.seed)
    for _ in range(args.synthetic_batches):
        x0 = torch.randn(
            args.batch_size, args.in_channels, args.latent_t,
            args.latent_h, args.latent_w, generator=g,
        )
        ids = torch.randint(
            0, vocab_size, (args.batch_size, args.cond_len), generator=g,
        )
        yield x0, ids


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aw-train-dit",
        description="Train the DiT pixel decoder.",
    )
    # vocab
    p.add_argument("--vocab", default="tiny", choices=["tiny", "default"])
    # model
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--in-channels", type=int, default=4)
    # data
    p.add_argument("--latent-t", type=int, default=2)
    p.add_argument("--latent-h", type=int, default=8)
    p.add_argument("--latent-w", type=int, default=8)
    p.add_argument("--cond-len", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--synthetic-batches", type=int, default=64)
    # train
    p.add_argument("--objective", default="rectified_flow",
                   choices=["rectified_flow", "v_prediction"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    # checkpoints
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--save-every", type=int, default=0)
    # logger
    p.add_argument("--logger", default="auto",
                   choices=["auto", "disabled", "jsonl", "wandb"])
    p.add_argument("--wandb-project", default="another_world")
    p.add_argument("--wandb-run-name", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    layout = VocabLayout.tiny() if args.vocab == "tiny" else VocabLayout.default()

    cfg = DiTDecoderConfig(
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        patch_size=args.patch_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=layout.visual_size,
    )
    model = DiTDecoder(cfg)
    _LOG.info("DiT params: %.2fM", model.num_parameters() / 1e6)

    if args.auto_resume and args.checkpoint_dir:
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest is not None:
            load_checkpoint(latest, model=model)
            _LOG.info("auto-resume from %s", latest)

    logger = create_logger(
        backend=args.logger,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        tags=["stage4", "dit"],
        config={"cfg": cfg.to_dict(), "args": vars(args)},
    )

    trainer_cfg = DiTTrainerConfig(
        steps=args.steps,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        device=args.device,
        precision=args.precision,
        log_every=args.log_every,
        seed=args.seed,
        objective=DiffusionObjectiveConfig(objective=args.objective),
    )

    batches = list(_synthetic_batches(args, vocab_size=layout.visual_size))

    try:
        history = run_dit_training(model, batches, trainer_cfg, logger=logger)
    finally:
        logger.finish()

    if args.checkpoint_dir and args.save_every >= 0:
        out = Path(args.checkpoint_dir) / f"step-{args.steps:08d}"
        save_checkpoint(
            out,
            model=model,
            meta=CheckpointMeta(step=args.steps, config={"objective": args.objective}),
        )
        _LOG.info("saved final checkpoint -> %s", out)

    if history:
        _LOG.info("loss: %.4f -> %.4f", history[0]["loss"], history[-1]["loss"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
