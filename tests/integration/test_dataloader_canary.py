"""Lightweight performance canary for the synthetic dataloader path.

The real benchmark on the cluster runs ``scripts/benchmark_dataloader.py``;
this CPU-friendly version just guarantees the synthetic pipeline can do
at least a few hundred samples per second so we catch egregious regressions
in CI.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from another_world.data.datasets import (
    IterableVideoDataset,
    VideoSample,
    build_default_transform,
    collate_video_samples,
)


def _synthetic_samples(n: int) -> list[VideoSample]:
    samples = []
    g = torch.Generator().manual_seed(0)
    for i in range(n):
        x = torch.randint(0, 256, (8, 3, 32, 32), generator=g, dtype=torch.uint8)
        samples.append(VideoSample(frames=x, caption=f"s{i}", key=f"k{i}"))
    return samples


def test_synthetic_dataloader_meets_floor_throughput() -> None:
    transform = build_default_transform(num_frames=8, height=32, width=32)
    ds = IterableVideoDataset(_synthetic_samples(64), transform=transform, loops=10)
    loader = DataLoader(
        ds, batch_size=4, collate_fn=collate_video_samples, drop_last=True
    )

    # warmup
    it = iter(loader)
    for _ in range(2):
        next(it)

    t0 = time.perf_counter()
    samples = 0
    steps = 20
    for _ in range(steps):
        batch = next(it)
        samples += batch["frames"].shape[0]
    dt = time.perf_counter() - t0

    rate = samples / max(dt, 1e-6)
    # Sanity floor: CI machines should easily exceed 50 samples/s for this
    # tiny payload. Significantly tighter than what we hit locally (~350 /s).
    assert rate > 50, f"throughput collapsed: {rate:.1f} samples/s in {dt:.3f}s"
