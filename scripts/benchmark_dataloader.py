#!/usr/bin/env python
"""Dataloader throughput benchmark.

Measures how fast our pipeline can produce frames-per-second and bytes-per-second
for a given dataset source. Run locally first to sanity-check the implementation,
then re-run on a Lambda Labs node to confirm we hit the stage-1 gate
(>= 5 GB/s without starving the GPU).

Examples
--------
Local synthetic stream (no external data, useful as a CI canary)::

    python scripts/benchmark_dataloader.py \
        --source synthetic --steps 50 --batch-size 4 --num-workers 0 \
        --frames 17 --height 256 --width 256

WebDataset shards on Cloudflare R2 (replace with your real URL pattern)::

    python scripts/benchmark_dataloader.py \
        --source webdataset \
        --urls "https://<account>.r2.cloudflarestorage.com/another-world-shards/train-{0000..0099}.tar" \
        --batch-size 8 --num-workers 4 --frames 17

HuggingFace streaming::

    python scripts/benchmark_dataloader.py \
        --source hf --repo-id HuggingFaceM4/WebVid --split train \
        --limit 32 --batch-size 4 --num-workers 0
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from another_world.data.datasets import (
    IterableVideoDataset,
    VideoSample,
    build_default_transform,
    collate_video_samples,
)
from another_world.utils.logging import get_logger

_LOG = get_logger("benchmark_dataloader")


def _synthetic_samples(
    n: int, frames: int, height: int, width: int, seed: int = 0
) -> list[VideoSample]:
    """Build ``n`` random uint8 video samples for offline benchmarking."""

    generator = torch.Generator().manual_seed(seed)
    out: list[VideoSample] = []
    for i in range(n):
        x = torch.randint(
            0, 256, (frames, 3, height, width), generator=generator, dtype=torch.uint8
        )
        out.append(
            VideoSample(
                frames=x, caption=f"synthetic-{i}", source="synthetic",
                key=f"syn-{i:06d}",
            )
        )
    return out


def _build_source(args: argparse.Namespace) -> Iterable[VideoSample]:
    transform = build_default_transform(
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        normalise="minus_one_one",
        temporal="sample",
    )

    if args.source == "synthetic":
        samples = _synthetic_samples(
            n=max(args.steps * args.batch_size, 16),
            frames=max(args.frames, args.frames),
            height=args.height,
            width=args.width,
            seed=args.seed,
        )
        return IterableVideoDataset(samples, transform=transform, loops=10**6)

    if args.source == "webdataset":
        from another_world.data.datasets import WebDatasetSpec, build_video_webdataset

        if not args.urls:
            raise SystemExit("--urls is required for source=webdataset")
        spec = WebDatasetSpec(
            urls=args.urls,
            shardshuffle=True,
            shuffle_buffer=args.shuffle_buffer,
            max_frames=args.frames,
        )
        return build_video_webdataset(spec, transform=transform)

    if args.source == "hf":
        from another_world.data.datasets.hf import build_hf_video_stream

        if not args.repo_id:
            raise SystemExit("--repo-id is required for source=hf")
        return build_hf_video_stream(
            args.repo_id,
            split=args.split,
            streaming=args.streaming,
            limit=args.limit,
            transform=transform,
            max_frames=args.frames,
        )

    raise SystemExit(f"unknown source '{args.source}'")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("benchmark_dataloader")
    p.add_argument("--source", choices=["synthetic", "webdataset", "hf"], default="synthetic")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    # webdataset
    p.add_argument("--urls", type=str, default=None)
    p.add_argument("--shuffle-buffer", type=int, default=1024)
    # hf
    p.add_argument("--repo-id", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--limit", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    src = _build_source(args)
    loader = DataLoader(
        src,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_video_samples,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    _LOG.info(
        "source=%s steps=%d batch=%d workers=%d frame-shape=[%d, 3, %d, %d]",
        args.source, args.steps, args.batch_size, args.num_workers,
        args.frames, args.height, args.width,
    )

    samples_total = 0
    bytes_total = 0
    times: list[float] = []
    t_iter = iter(loader)

    for step in range(args.steps + args.warmup):
        t0 = time.perf_counter()
        try:
            batch = next(t_iter)
        except StopIteration:
            _LOG.warning("Source exhausted after %d steps", step)
            break
        dt = time.perf_counter() - t0
        if step >= args.warmup:
            times.append(dt)
            samples_total += batch["frames"].shape[0]
            bytes_total += int(batch["frames"].element_size() * batch["frames"].numel())
        if step % max(1, (args.steps + args.warmup) // 10) == 0:
            _LOG.info(
                "step=%4d  dt=%.3fs  shape=%s",
                step, dt, tuple(batch["frames"].shape),
            )

    if not times:
        _LOG.error("No measured steps; aborting.")
        return 1

    total_time = sum(times)
    samples_per_sec = samples_total / total_time
    mb_per_sec = bytes_total / total_time / (1024**2)
    p50 = sorted(times)[len(times) // 2]
    p95 = sorted(times)[int(len(times) * 0.95)]

    _LOG.info("=" * 60)
    _LOG.info(
        "samples/s = %.2f  (%.1f frames/s, %.1f MB/s)",
        samples_per_sec,
        samples_per_sec * args.frames,
        mb_per_sec,
    )
    _LOG.info("step latency p50=%.3fs  p95=%.3fs", p50, p95)
    _LOG.info("Stage-1 gate target: >= 5 GB/s = %.0f MB/s.", 5 * 1024)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
