"""Command-line entry points for training."""

from __future__ import annotations

import argparse
import sys

from another_world.data.datasets import DummyTokenDataset
from another_world.models.dynamics import ToyTransformerConfig, build_toy_transformer
from another_world.training.smoke import SmokeTrainerConfig, run_smoke_training
from another_world.utils.experiment import create_logger
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


def _smoke_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aw-train",
        description="Run a stage-0 smoke training of the toy transformer.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--logger",
        type=str,
        default="auto",
        choices=["auto", "disabled", "jsonl", "wandb"],
        help="experiment logger backend (default: auto)",
    )
    parser.add_argument("--wandb-project", type=str, default="another_world")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--jsonl-path", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _smoke_parser().parse_args(argv)

    model_cfg = ToyTransformerConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=max(64, args.seq_len),
    )
    model = build_toy_transformer(model_cfg)
    _LOG.info("Built toy model with %.2fM parameters", model.num_parameters / 1e6)

    dataset = DummyTokenDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        length=max(args.batch_size * args.steps, args.batch_size * 4),
        seed=args.seed,
    )

    train_cfg = SmokeTrainerConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
    )

    logger = create_logger(
        backend=args.logger,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        tags=["stage0", "smoke"],
        config={
            "model": vars(model_cfg) if hasattr(model_cfg, "__dict__") else {},
            "train": vars(train_cfg) if hasattr(train_cfg, "__dict__") else {},
            "cli": vars(args),
        },
        jsonl_path=args.jsonl_path,
        wandb_mode=args.wandb_mode,
    )

    try:
        history = run_smoke_training(model, dataset, train_cfg, logger=logger)
    finally:
        logger.finish()

    if not history:
        _LOG.error("No training history recorded.")
        return 1
    first, last = history[0], history[-1]
    _LOG.info("loss: %.4f -> %.4f", first.loss, last.loss)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
