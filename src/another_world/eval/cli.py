"""Eval CLI entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from another_world.eval.i3d_fvd import FVDConfig, I3DFVD
from another_world.eval.metrics import (
    fvd_score,
    mae,
    mse,
    psnr,
    temporal_consistency,
)
from another_world.eval.vbench_wrapper import vbench_or_fallback
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


def _load_videos(path: Path) -> torch.Tensor:
    """Load a video tensor from ``.pt`` (preferred) or ``.npy``."""

    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(path))
    raise ValueError(f"unsupported extension {path.suffix}")


def _ensure_5d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4:
        t = t.unsqueeze(0)
    if t.dim() != 5:
        raise ValueError(f"expected 4-D or 5-D tensor, got {tuple(t.shape)}")
    return t


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aw-eval",
        description="Evaluate generated videos against optional references.",
    )
    parser.add_argument("--predictions", type=Path, required=True,
                        help="predictions tensor (.pt or .npy)")
    parser.add_argument("--targets", type=Path, default=None,
                        help="optional reference / ground-truth tensor")
    parser.add_argument(
        "--metrics", nargs="+",
        default=["mse", "psnr", "mae", "temporal_consistency", "vbench", "fvd"],
        choices=["mse", "psnr", "mae", "temporal_consistency", "vbench", "fvd"],
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="write metrics dict to this JSON file")
    parser.add_argument("--fvd-backend", default="pixel", choices=["pixel", "i3d"],
                        help="FVD backend: pixel approximation or I3D wrapper")
    parser.add_argument("--strict-i3d", action="store_true",
                        help="fail if --fvd-backend i3d is requested but no extractor is configured")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    preds = _ensure_5d(_load_videos(args.predictions))
    targets = (
        _ensure_5d(_load_videos(args.targets)) if args.targets else None
    )
    _LOG.info("preds shape: %s", tuple(preds.shape))
    if targets is not None:
        _LOG.info("targets shape: %s", tuple(targets.shape))

    out: dict[str, float] = {}

    pred_for_metrics = preds.float()
    target_for_metrics = targets.float() if targets is not None else None

    if "mse" in args.metrics and target_for_metrics is not None:
        out["mse"] = mse(pred_for_metrics, target_for_metrics)
    if "mae" in args.metrics and target_for_metrics is not None:
        out["mae"] = mae(pred_for_metrics, target_for_metrics)
    if "psnr" in args.metrics and target_for_metrics is not None:
        out["psnr"] = psnr(pred_for_metrics, target_for_metrics)
    if "temporal_consistency" in args.metrics:
        out["temporal_consistency"] = temporal_consistency(pred_for_metrics)
    if "fvd" in args.metrics and target_for_metrics is not None:
        evaluator = I3DFVD(
            FVDConfig(
                backend=args.fvd_backend,
                strict_i3d=args.strict_i3d,
            )
        )
        out["fvd"] = evaluator(target_for_metrics, pred_for_metrics)
    if "vbench" in args.metrics:
        out.update(vbench_or_fallback(pred_for_metrics))

    _LOG.info("results: %s", out)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        _LOG.info("wrote %s", args.output)
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
