#!/usr/bin/env python
"""Offline tokenisation CLI.

Takes a stream of raw video samples (synthetic / webdataset / hf) and writes
pre-tokenised WebDataset-style shards to disk.

The visual tokenizer is either:

- a real :class:`CosmosVideoTokenizer` loaded from a local checkpoint dir,
- or, for local smoke tests, a tiny mock that hashes the input frames
  into deterministic int64 indices.

Examples
--------
Mock tokenizer + synthetic source, useful for laptop validation::

    python scripts/tokenize_shards.py \
        --source synthetic --count 64 --frames 17 --height 64 --width 64 \
        --visual-tokenizer mock \
        --out-dir outputs/shards/mock --target-mb 4

Real Cosmos tokenizer on a GPU node::

    python scripts/tokenize_shards.py \
        --source hf --repo-id Antreas/TALI --split train --limit 200 \
        --frames 17 --height 256 --width 256 \
        --visual-tokenizer cosmos --cosmos-name Cosmos-1.0-Tokenizer-DV8x16x16 \
        --cosmos-ckpt-dir .cache/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16 \
        --device cuda --dtype bf16 \
        --out-dir outputs/shards/tali --target-mb 1024
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterable
from pathlib import Path

import torch

from another_world.data.datasets import (
    IterableVideoDataset,
    VideoSample,
    build_default_transform,
)
from another_world.data.filters import (
    AspectRatioFilter,
    FilterPipeline,
    MinResolutionFilter,
)
from another_world.data.tokenize import RotatingShardWriter, TokenizationPipeline
from another_world.utils.logging import get_logger

_LOG = get_logger("tokenize_shards")


# ---------------------------------------------------------------------------
# Mock tokenizer for local smoke tests
# ---------------------------------------------------------------------------


class MockDiscreteTokenizer:
    """Deterministic mock that hashes [B, C, T, H, W] -> int64 indices."""

    def __init__(self, vocab_size: int = 1024, downsample: int = 8) -> None:
        self.vocab_size = vocab_size
        self.downsample = downsample

    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, _, t, h, w = video.shape
        t_prime = 1 + (t - 1) // self.downsample
        h_prime = h // self.downsample
        w_prime = w // self.downsample
        # Spatially+temporally pool the input then quantise into vocab range.
        pooled = torch.nn.functional.adaptive_avg_pool3d(
            video.mean(dim=1, keepdim=False).unsqueeze(1),
            output_size=(t_prime, h_prime, w_prime),
        )[:, 0]
        normed = (pooled - pooled.min()) / max(
            float(pooled.max() - pooled.min()), 1e-6
        )
        indices = (normed * (self.vocab_size - 1)).round().long()
        return (indices,)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _hash_text_tokenizer(text: str) -> torch.Tensor:
    """Tiny deterministic text 'tokenizer' for local smoke tests."""

    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return torch.tensor(list(digest), dtype=torch.long)


def _synthetic_source(args: argparse.Namespace) -> Iterable[VideoSample]:
    g = torch.Generator().manual_seed(args.seed)
    samples = []
    for i in range(args.count):
        x = torch.randint(
            0,
            256,
            (args.frames, 3, args.height, args.width),
            generator=g,
            dtype=torch.uint8,
        )
        samples.append(
            VideoSample(
                frames=x,
                caption=f"synthetic-{i}",
                source="synthetic",
                license="CC-BY",
                key=f"syn-{i:06d}",
                fps=24,
                duration=args.frames / 24.0,
            )
        )
    transform = build_default_transform(
        num_frames=args.frames, height=args.height, width=args.width,
        normalise="minus_one_one",
    )
    return IterableVideoDataset(samples, transform=transform)


def _webdataset_source(args: argparse.Namespace) -> Iterable[VideoSample]:
    from another_world.data.datasets import WebDatasetSpec, build_video_webdataset

    transform = build_default_transform(
        num_frames=args.frames, height=args.height, width=args.width,
        normalise="minus_one_one",
    )
    spec = WebDatasetSpec(urls=args.urls, max_frames=args.frames)
    return build_video_webdataset(spec, transform=transform)


def _hf_source(args: argparse.Namespace) -> Iterable[VideoSample]:
    from another_world.data.datasets.hf import build_hf_video_stream

    transform = build_default_transform(
        num_frames=args.frames, height=args.height, width=args.width,
        normalise="minus_one_one",
    )
    return build_hf_video_stream(
        args.repo_id,
        split=args.split,
        streaming=True,
        limit=args.limit,
        transform=transform,
        max_frames=args.frames,
    )


def _build_source(args: argparse.Namespace) -> Iterable[VideoSample]:
    if args.source == "synthetic":
        return _synthetic_source(args)
    if args.source == "webdataset":
        if not args.urls:
            raise SystemExit("--urls required for source=webdataset")
        return _webdataset_source(args)
    if args.source == "hf":
        if not args.repo_id:
            raise SystemExit("--repo-id required for source=hf")
        return _hf_source(args)
    raise SystemExit(f"unknown source '{args.source}'")


# ---------------------------------------------------------------------------
# Tokenizer construction
# ---------------------------------------------------------------------------


def _build_visual_tokenizer(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    if args.visual_tokenizer == "mock":
        return MockDiscreteTokenizer(
            vocab_size=args.mock_vocab_size, downsample=args.mock_downsample
        )
    if args.visual_tokenizer == "cosmos":
        from another_world.tokenizers.visual.cosmos import CosmosVideoTokenizer

        dtype_map = {
            "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
        }
        if not args.cosmos_ckpt_dir:
            raise SystemExit("--cosmos-ckpt-dir required for cosmos tokenizer")
        return CosmosVideoTokenizer.from_local(
            args.cosmos_name,
            args.cosmos_ckpt_dir,
            device=args.device,
            dtype=dtype_map[args.dtype],
        )
    raise SystemExit(f"unknown tokenizer '{args.visual_tokenizer}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("tokenize_shards")
    # source
    p.add_argument("--source", choices=["synthetic", "webdataset", "hf"], default="synthetic")
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    # webdataset
    p.add_argument("--urls", type=str, default=None)
    # hf
    p.add_argument("--repo-id", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--limit", type=int, default=None)
    # filters
    p.add_argument("--min-height", type=int, default=32)
    p.add_argument("--min-width", type=int, default=32)
    p.add_argument("--min-aspect", type=float, default=0.4)
    p.add_argument("--max-aspect", type=float, default=2.5)
    # tokenizer
    p.add_argument("--visual-tokenizer", choices=["mock", "cosmos"], default="mock")
    p.add_argument("--mock-vocab-size", type=int, default=1024)
    p.add_argument("--mock-downsample", type=int, default=8)
    p.add_argument("--cosmos-name", default="Cosmos-1.0-Tokenizer-DV8x16x16")
    p.add_argument("--cosmos-ckpt-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    # text
    p.add_argument("--no-text", action="store_true",
                   help="skip text tokenisation entirely")
    # output
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="shard")
    p.add_argument("--target-mb", type=int, default=64)
    p.add_argument("--compression", default=None, choices=[None, "gz", "bz2", "xz"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    source = _build_source(args)
    visual = _build_visual_tokenizer(args)
    text = None if args.no_text else _hash_text_tokenizer

    filters = FilterPipeline([
        MinResolutionFilter(min_height=args.min_height, min_width=args.min_width),
        AspectRatioFilter(min_ratio=args.min_aspect, max_ratio=args.max_aspect),
    ])

    target_bytes = args.target_mb * 1024 * 1024
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TokenizationPipeline(
        visual_tokenizer=visual,
        text_tokenizer=text,
        filters=filters,
    )

    _LOG.info(
        "Source=%s tokenizer=%s out=%s target=%dMB",
        args.source, args.visual_tokenizer, args.out_dir, args.target_mb,
    )

    with RotatingShardWriter(
        args.out_dir,
        prefix=args.prefix,
        target_size_bytes=target_bytes,
        compression=args.compression,
        tokenizer=args.visual_tokenizer,
        source=args.source,
    ) as writer:
        manifests = pipeline.run(source, writer)

    _LOG.info(
        "Done. in=%d kept=%d filtered=%d failed=%d  elapsed=%.2fs  shards=%d",
        pipeline.stats.samples_in,
        pipeline.stats.samples_out,
        pipeline.stats.samples_filtered,
        pipeline.stats.samples_failed,
        pipeline.stats.elapsed_seconds,
        len(manifests),
    )
    for m in manifests:
        _LOG.info(
            "  %s  samples=%d  size=%.2fMB",
            m.shard_path, m.num_samples, m.bytes_written / (1024 * 1024),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
