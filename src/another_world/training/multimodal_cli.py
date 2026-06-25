"""Multimodal training CLI (``aw-train-mm``).

Wires the token-shard dataloader (from stage 1.3 outputs), the
multimodal dynamics model (stage 3.1), and the multimodal trainer
(stage 3.2 prep) into a single entry point.

Examples
--------
CPU smoke (synthetic shards generated on the fly)::

    aw-train-mm --shards-dir outputs/shards-smoke \
        --preset toy --device cpu --precision fp32 \
        --steps 20 --batch-size 2 --max-len 64

GPU smoke (after running ``tokenize_shards.py`` on a real source)::

    aw-train-mm --shards-dir /scratch/shards/tali \
        --preset m350 --device cuda --precision bf16 \
        --steps 5000 --batch-size 8 --max-len 2048 \
        --grad-accum 4 --activation-checkpointing \
        --logger auto --wandb-project another_world
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import (
    PackedBatch,
    SequencePacker,
)
from another_world.data.datasets.token_shard_stream import (
    build_packed_batch_stream,
)
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout
from another_world.training.distributed_wrap import (
    FsdpConfig,
    wrap_model_for_distributed,
)
from another_world.training.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
)
from another_world.training.multimodal import (
    MultimodalTrainerConfig,
    run_multimodal_training,
)
from another_world.utils.experiment import create_logger
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# preset mapping
# ---------------------------------------------------------------------------


_PRESETS = {
    "toy": MultimodalDynamicsConfig.toy,
    "m350": MultimodalDynamicsConfig.m350,
    "b1": MultimodalDynamicsConfig.b1,
    "b3": MultimodalDynamicsConfig.b3,
    "b7": MultimodalDynamicsConfig.b7,
}


def _build_model(args: argparse.Namespace, vocab_size: int) -> MultimodalDynamicsModel:
    if args.preset not in _PRESETS:
        raise SystemExit(f"unknown preset '{args.preset}'")
    cfg = _PRESETS[args.preset](vocab_size=vocab_size)
    if args.dim is not None:
        cfg.dim = args.dim
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.n_heads is not None:
        cfg.n_heads = args.n_heads
    if args.n_kv_heads is not None:
        cfg.n_kv_heads = args.n_kv_heads
    if args.max_len is not None:
        cfg.max_linear = max(cfg.max_linear, args.max_len)
    return MultimodalDynamicsModel(cfg)


# ---------------------------------------------------------------------------
# data sources
# ---------------------------------------------------------------------------


def _list_shards(shards_dir: Path) -> list[str]:
    paths = sorted(p for p in shards_dir.glob("*.tar"))
    if not paths:
        paths = sorted(p for p in shards_dir.glob("*.tar.*"))
    return [str(p) for p in paths]


def _synthetic_batches(
    layout: VocabLayout, packer: SequencePacker, batch_size: int, total: int,
) -> Iterable[PackedBatch]:
    """Fallback batch stream when ``--shards-dir`` is empty.

    Generates random :class:`TokenSample`s in the correct id ranges so the
    trainer can be smoke-tested on machines without real shards.
    """

    g = torch.Generator().manual_seed(0)
    samples: list[TokenSample] = []
    for i in range(total * batch_size):
        samples.append(
            TokenSample(
                visual_tokens=torch.randint(
                    0, layout.visual_size, (2, 2, 2), generator=g, dtype=torch.long,
                ),
                text_tokens=torch.randint(
                    0, layout.text_size, (4,), generator=g, dtype=torch.long,
                ),
                key=f"syn-{i:06d}",
            )
        )
    # group into batches
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        if len(chunk) == batch_size:
            yield packer.pack_batch(chunk)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aw-train-mm",
        description="Train the multimodal dynamics model on token shards.",
    )
    # data
    p.add_argument("--shards-dir", type=Path, default=None)
    p.add_argument("--synthetic-steps", type=int, default=64,
                   help="batches to generate when no shards found")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--loops", type=int, default=10**6)
    p.add_argument("--score-text", action="store_true")
    p.add_argument("--no-score-visual", action="store_true")
    # vocab
    p.add_argument("--vocab", default="tiny", choices=["tiny", "default"])
    # model
    p.add_argument("--preset", default="toy", choices=list(_PRESETS))
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--n-kv-heads", type=int, default=None)
    # train
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    # distributed
    p.add_argument("--dist-strategy", default="auto",
                   choices=["auto", "fsdp2", "ddp", "none"])
    p.add_argument("--cpu-offload", action="store_true")
    # logger
    p.add_argument("--logger", default="auto",
                   choices=["auto", "disabled", "jsonl", "wandb"])
    p.add_argument("--wandb-project", default="another_world")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--jsonl-path", default=None)
    # checkpoints
    p.add_argument("--checkpoint-dir", default=None,
                   help="root directory for periodic checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="save every N steps (0 disables)")
    p.add_argument("--checkpoint-keep", type=int, default=3,
                   help="max number of recent checkpoints to keep")
    p.add_argument("--checkpoint-upload-uri", default=None,
                   help="optional r2://<bucket>/<prefix> upload target")
    p.add_argument("--resume", default=None,
                   help="explicit checkpoint dir to resume from")
    p.add_argument("--auto-resume", action="store_true",
                   help="if set, auto-resume from latest under --checkpoint-dir")
    # JEPA
    p.add_argument("--jepa-weight", type=float, default=0.0,
                   help="auxiliary V-JEPA loss weight (0 disables)")
    p.add_argument("--jepa-ema-decay", type=float, default=0.999)
    p.add_argument("--jepa-predictor-layers", type=int, default=2)
    p.add_argument("--jepa-predictor-heads", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    layout = VocabLayout.tiny() if args.vocab == "tiny" else VocabLayout.default()
    _LOG.info("vocab layout: total=%d", layout.total_size)

    model = _build_model(args, vocab_size=layout.total_size)
    _LOG.info("model %s: %.2fM params", args.preset, model.num_parameters / 1e6)

    wrap = wrap_model_for_distributed(
        model,
        fsdp=FsdpConfig(
            strategy=args.dist_strategy,
            mixed_precision_dtype=args.precision,
            cpu_offload=args.cpu_offload,
            activation_checkpointing=args.activation_checkpointing,
        ),
    )
    inner_model = (
        wrap.model.module if hasattr(wrap.model, "module") else wrap.model
    )
    _LOG.info("distributed strategy: %s", wrap.strategy)

    packer = SequencePacker(
        layout=layout,
        max_len=args.max_len,
        include_text=True,
        score_text=args.score_text,
        score_visual=not args.no_score_visual,
        score_action=True,
    )

    shard_paths: list[str] = []
    if args.shards_dir is not None and args.shards_dir.exists():
        shard_paths = _list_shards(args.shards_dir)
    if shard_paths:
        _LOG.info("found %d shards in %s", len(shard_paths), args.shards_dir)
        batches = build_packed_batch_stream(
            shard_paths,
            packer=packer,
            batch_size=args.batch_size,
            loops=args.loops,
            drop_last=True,
        )
    else:
        _LOG.warning("no shards found; falling back to synthetic batches")
        batches = _synthetic_batches(
            layout, packer, args.batch_size, args.synthetic_steps,
        )

    logger = create_logger(
        backend=args.logger,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        tags=["stage3", "multimodal"],
        config={
            "preset": args.preset,
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "precision": args.precision,
            "activation_checkpointing": args.activation_checkpointing,
            "shards_dir": str(args.shards_dir) if args.shards_dir else None,
            "vocab_total": layout.total_size,
        },
        jsonl_path=args.jsonl_path,
    )

    cfg = MultimodalTrainerConfig(
        steps=args.steps,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
        activation_checkpointing=args.activation_checkpointing,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        checkpoint_keep=args.checkpoint_keep,
        checkpoint_upload_uri=args.checkpoint_upload_uri,
        is_main=(wrap.info.rank == 0),
        jepa_weight=args.jepa_weight,
        jepa_ema_decay=args.jepa_ema_decay,
        jepa_predictor_layers=args.jepa_predictor_layers,
        jepa_predictor_heads=args.jepa_predictor_heads,
    )

    # Resume (optional).
    resume_path = args.resume
    if resume_path is None and args.auto_resume and args.checkpoint_dir:
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest is not None:
            resume_path = str(latest)
            _LOG.info("auto-resume found checkpoint: %s", resume_path)
    if resume_path:
        meta = load_checkpoint(resume_path, model=inner_model, optimizer=None)
        _LOG.info("resumed from step=%d", meta.step)

    try:
        history = run_multimodal_training(inner_model, batches, cfg, logger=logger)
    finally:
        logger.finish()

    if history:
        first, last = history[0], history[-1]
        _LOG.info("loss: %.4f -> %.4f", first.loss, last.loss)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
