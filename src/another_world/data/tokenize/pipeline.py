"""Offline tokenisation pipeline.

Ties together the components built in stage 1.1 / 1.2:

    VideoSample stream  ->  (filters)  ->  visual tokenizer  ->
    TokenSample stream  ->  RotatingShardWriter  ->  R2

The pipeline is split from the CLI so we can unit-test it with mock
tokenizers and an in-memory source.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import torch

from another_world.data.datasets.sample import TokenSample, VideoSample
from another_world.data.filters.pipeline import FilterPipeline
from another_world.data.tokenize.shards import (
    RotatingShardWriter,
    ShardManifest,
    make_sample_id,
)
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer interface
# ---------------------------------------------------------------------------


@runtime_checkable
class VisualTokenizerLike(Protocol):
    """Subset of :class:`CosmosVideoTokenizer` we depend on."""

    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, ...]: ...


TextTokenizer = Callable[[str], torch.Tensor]


def _tokenize_visual(
    tokenizer: VisualTokenizerLike, frames: torch.Tensor
) -> torch.Tensor:
    """Tokenise a single video clip.

    ``frames`` is ``[T, C, H, W]`` (float in ``[-1, 1]`` is preferred); the
    tokenizer expects ``[B, C, T, H, W]`` so we permute + unsqueeze before
    calling ``encode``. We return only the *indices* tensor for discrete
    tokenizers or the latent tensor for continuous ones (first element of
    the tuple).
    """

    if frames.dim() != 4:
        raise ValueError(
            f"expected [T, C, H, W] frames, got {tuple(frames.shape)}"
        )
    # [T, C, H, W] -> [1, C, T, H, W]
    batched = frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    encoded = tokenizer.encode(batched)
    if not encoded:
        raise RuntimeError("tokenizer returned empty tuple")
    return encoded[0].detach().to("cpu")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class TokenizationStats:
    samples_in: int = 0
    samples_out: int = 0
    samples_filtered: int = 0
    samples_failed: int = 0
    elapsed_seconds: float = 0.0

    @property
    def keep_rate(self) -> float:
        return self.samples_out / self.samples_in if self.samples_in else 0.0


@dataclass
class TokenizationPipeline:
    """End-to-end offline tokenisation pipeline.

    Args:
        visual_tokenizer: any object with ``encode(video)`` returning a
            tuple whose first element is the per-clip token tensor.
        text_tokenizer: optional callable mapping a caption string to a
            1D long tensor. ``None`` skips text tokenisation.
        filters: optional :class:`FilterPipeline`.
        on_error: callback invoked when a sample fails; ``True`` keeps the
            stream flowing, ``False`` re-raises.
    """

    visual_tokenizer: VisualTokenizerLike
    text_tokenizer: TextTokenizer | None = None
    filters: FilterPipeline | None = None
    on_error: Callable[[Exception, VideoSample], bool] | None = None

    stats: TokenizationStats = field(default_factory=TokenizationStats)

    def _handle_error(self, exc: Exception, sample: VideoSample) -> bool:
        if self.on_error is not None:
            return self.on_error(exc, sample)
        _LOG.warning("Skipping sample %s: %s", sample.key, exc)
        return True

    def _tokenize_one(self, sample: VideoSample) -> TokenSample:
        visual_tokens = _tokenize_visual(self.visual_tokenizer, sample.frames)
        text_tokens: torch.Tensor | None = None
        if self.text_tokenizer is not None and sample.caption:
            text_tokens = self.text_tokenizer(sample.caption)
            if not isinstance(text_tokens, torch.Tensor):
                text_tokens = torch.tensor(text_tokens, dtype=torch.long)

        extra: dict[str, Any] = {
            "caption": sample.caption,
            "source": sample.source,
            "license": sample.license,
            "fps": sample.fps,
            "duration": sample.duration,
            **sample.extra,
        }
        return TokenSample(
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            key=sample.key or make_sample_id(),
            extra=extra,
        )

    def run(
        self,
        source: Iterable[VideoSample],
        writer: RotatingShardWriter,
    ) -> list[ShardManifest]:
        """Stream ``source`` through filters + tokenizer, writing to ``writer``."""

        t0 = time.perf_counter()
        for sample in source:
            self.stats.samples_in += 1
            if self.filters is not None:
                filtered = self.filters(sample)
                if filtered is None:
                    self.stats.samples_filtered += 1
                    continue
                sample = filtered
            try:
                token_sample = self._tokenize_one(sample)
            except Exception as exc:  # noqa: BLE001
                self.stats.samples_failed += 1
                if not self._handle_error(exc, sample):
                    raise
                continue
            writer.write(token_sample)
            self.stats.samples_out += 1
        manifests = writer.close()
        self.stats.elapsed_seconds = time.perf_counter() - t0
        return manifests

    def stream(
        self, source: Iterable[VideoSample]
    ) -> Iterator[TokenSample]:
        """Stream tokenised samples without writing to disk (debugging)."""

        for sample in source:
            self.stats.samples_in += 1
            if self.filters is not None:
                filtered = self.filters(sample)
                if filtered is None:
                    self.stats.samples_filtered += 1
                    continue
                sample = filtered
            try:
                yield self._tokenize_one(sample)
                self.stats.samples_out += 1
            except Exception as exc:  # noqa: BLE001
                self.stats.samples_failed += 1
                if not self._handle_error(exc, sample):
                    raise


__all__ = [
    "TextTokenizer",
    "TokenizationPipeline",
    "TokenizationStats",
    "VisualTokenizerLike",
]
