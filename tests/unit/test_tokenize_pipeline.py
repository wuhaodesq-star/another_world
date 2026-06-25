"""Tests for the offline tokenisation pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from another_world.data.datasets.sample import VideoSample
from another_world.data.filters import (
    FilterPipeline,
    MinResolutionFilter,
)
from another_world.data.tokenize import (
    RotatingShardWriter,
    TokenizationPipeline,
    read_token_shards,
)


class _MockTokenizer:
    """Minimal discrete tokenizer: hashes input to ints."""

    def __init__(self, vocab: int = 64, down: int = 4) -> None:
        self.vocab = vocab
        self.down = down

    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, _, t, h, w = video.shape
        t_p = max(1, 1 + (t - 1) // self.down)
        h_p = max(1, h // self.down)
        w_p = max(1, w // self.down)
        pooled = torch.nn.functional.adaptive_avg_pool3d(
            video.mean(dim=1, keepdim=True), (t_p, h_p, w_p)
        )[:, 0]
        normed = (pooled - pooled.min()) / max(
            float(pooled.max() - pooled.min()), 1e-6
        )
        indices = (normed * (self.vocab - 1)).round().long()
        return (indices,)


def _samples(n: int, height: int = 32, width: int = 32) -> list[VideoSample]:
    samples = []
    g = torch.Generator().manual_seed(0)
    for i in range(n):
        x = torch.randn(
            (9, 3, height, width), generator=g, dtype=torch.float32
        ).clamp_(-1, 1)
        samples.append(
            VideoSample(
                frames=x,
                caption=f"clip {i}",
                source="unit",
                license="CC-BY",
                key=f"k{i:03d}",
                fps=24,
                duration=9 / 24,
            )
        )
    return samples


def test_pipeline_tokenises_and_writes(tmp_path: Path) -> None:
    samples = _samples(4)
    tokenizer = _MockTokenizer()
    pipeline = TokenizationPipeline(visual_tokenizer=tokenizer)
    with RotatingShardWriter(
        tmp_path, target_size_bytes=10**9
    ) as writer:
        manifests = pipeline.run(samples, writer)
    assert pipeline.stats.samples_in == 4
    assert pipeline.stats.samples_out == 4
    assert pipeline.stats.samples_filtered == 0
    assert pipeline.stats.samples_failed == 0
    assert len(manifests) == 1
    assert manifests[0].num_samples == 4

    paths = sorted(tmp_path.glob("*.tar"))
    out = list(read_token_shards(paths))
    assert len(out) == 4
    assert out[0].visual_tokens.dtype == torch.long
    # Check that captions made it into extra metadata.
    captions = {s.extra.get("caption") for s in out}
    assert "clip 0" in captions


def test_pipeline_applies_filters(tmp_path: Path) -> None:
    samples = _samples(2, height=16, width=16) + _samples(3, height=64, width=64)
    pipeline = TokenizationPipeline(
        visual_tokenizer=_MockTokenizer(),
        filters=FilterPipeline([MinResolutionFilter(min_height=32, min_width=32)]),
    )
    with RotatingShardWriter(tmp_path) as writer:
        pipeline.run(samples, writer)
    # 2 small samples filtered, 3 kept (note the second _samples call reuses
    # the same RNG seed, so the keys collide, but the writer assigns new
    # sample ids anyway).
    assert pipeline.stats.samples_filtered == 2
    assert pipeline.stats.samples_out == 3


def test_pipeline_text_tokenizer_invoked(tmp_path: Path) -> None:
    samples = _samples(2)

    calls: list[str] = []

    def fake_text(text: str) -> torch.Tensor:
        calls.append(text)
        return torch.tensor([len(text)], dtype=torch.long)

    pipeline = TokenizationPipeline(
        visual_tokenizer=_MockTokenizer(),
        text_tokenizer=fake_text,
    )
    with RotatingShardWriter(tmp_path) as writer:
        pipeline.run(samples, writer)
    assert calls == ["clip 0", "clip 1"]
    paths = sorted(tmp_path.glob("*.tar"))
    out = list(read_token_shards(paths))
    for s in out:
        assert s.text_tokens is not None


def test_pipeline_error_handler_skips_bad_samples(tmp_path: Path) -> None:
    class _Boom:
        def encode(self, video):
            raise RuntimeError("boom")

    samples = _samples(3)
    pipeline = TokenizationPipeline(visual_tokenizer=_Boom())
    with RotatingShardWriter(tmp_path) as writer:
        pipeline.run(samples, writer)
    assert pipeline.stats.samples_failed == 3
    assert pipeline.stats.samples_out == 0


def test_pipeline_stream_does_not_write(tmp_path: Path) -> None:
    pipeline = TokenizationPipeline(visual_tokenizer=_MockTokenizer())
    out = list(pipeline.stream(_samples(2)))
    assert len(out) == 2
    # Mock tokenizer mirrors the Cosmos discrete output:
    # ``[B=1, T', H', W']``.
    assert all(s.visual_tokens.dim() == 4 for s in out)
    # tmp_path should still be empty.
    assert not list(tmp_path.iterdir())
