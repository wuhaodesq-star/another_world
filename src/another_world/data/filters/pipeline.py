"""Filter primitives.

A *filter* is a deterministic callable that takes a :class:`VideoSample`
and either returns it unchanged (passed) or returns ``None`` (rejected).
Filters are designed to compose with :class:`FilterPipeline`, which keeps
per-filter accept/reject statistics so we can tune thresholds without
re-walking the dataset.

This module ships with implementations that:

- are pure-Python / pure-torch (no heavy ML deps),
- run in microseconds per call,
- have well-defined thresholds documented in the docstring,

so the real ML-based filters (LAION aesthetic predictor, watermark CNN, NSFW
classifier) can land in stage 1.2 as drop-in replacements with the same
interface.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import torch

from another_world.data.datasets.sample import VideoSample
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


@runtime_checkable
class Filter(Protocol):
    """Filter contract: keep or drop a :class:`VideoSample`."""

    name: str

    def __call__(self, sample: VideoSample) -> VideoSample | None: ...


# ---------------------------------------------------------------------------
# Built-in filters
# ---------------------------------------------------------------------------


@dataclass
class MinDurationFilter:
    """Reject clips shorter than ``min_seconds``.

    Requires either ``sample.duration`` or both ``num_frames`` and
    ``sample.fps`` to be present.
    """

    name: str = "min_duration"
    min_seconds: float = 1.0

    def __call__(self, sample: VideoSample) -> VideoSample | None:
        seconds = sample.duration
        if seconds is None and sample.fps and sample.fps > 0:
            seconds = sample.num_frames / sample.fps
        if seconds is None:
            return sample  # cannot decide -> keep
        if seconds < self.min_seconds:
            return None
        return sample


@dataclass
class MinResolutionFilter:
    """Reject samples whose frames are smaller than (height, width)."""

    name: str = "min_resolution"
    min_height: int = 256
    min_width: int = 256

    def __call__(self, sample: VideoSample) -> VideoSample | None:
        h, w = sample.resolution
        if h < self.min_height or w < self.min_width:
            return None
        return sample


@dataclass
class AspectRatioFilter:
    """Reject samples outside ``[min_ratio, max_ratio]`` (W/H)."""

    name: str = "aspect_ratio"
    min_ratio: float = 0.5
    max_ratio: float = 2.5

    def __call__(self, sample: VideoSample) -> VideoSample | None:
        h, w = sample.resolution
        ratio = w / max(h, 1)
        if ratio < self.min_ratio or ratio > self.max_ratio:
            return None
        return sample


@dataclass
class LicenseFilter:
    """Keep only samples whose ``license`` field is in the allow-list."""

    name: str = "license"
    allow: tuple[str, ...] = (
        "cc-by",
        "cc-by-sa",
        "cc-by-nc",
        "cc0",
        "public-domain",
    )
    require_field: bool = True

    def __call__(self, sample: VideoSample) -> VideoSample | None:
        lic = (sample.license or "").lower()
        if not lic:
            return None if self.require_field else sample
        return sample if any(token in lic for token in self.allow) else None


@dataclass
class AestheticFilter:
    """Heuristic aesthetic gate based on luminance variance.

    A real implementation should replace this with the LAION aesthetic
    predictor (stage 1.2). Until then, low pixel variance is a reasonable
    proxy for "blank / static / corrupt" clips.
    """

    name: str = "aesthetic"
    min_variance: float = 50.0

    def __call__(self, sample: VideoSample) -> VideoSample | None:
        frames = sample.frames.float()
        # luminance approximation: 0.299 R + 0.587 G + 0.114 B
        lum = (
            0.299 * frames[..., 0, :, :]
            + 0.587 * frames[..., 1, :, :]
            + 0.114 * frames[..., 2, :, :]
        )
        var = float(lum.var().item())
        if var < self.min_variance:
            return None
        sample.extra.setdefault("metrics", {})["aesthetic_var"] = var
        return sample


@dataclass
class DedupFilter:
    """Hash-based duplicate detection.

    Computes a fast perceptual-ish fingerprint (mean-pooled 8x8 luminance
    quantised to bits) and rejects any sample whose fingerprint we have
    already seen in this run.

    For multi-shard / multi-process dedup, the fingerprint should be
    persisted into the manifest layer; this in-memory class is intended
    for single-shard or test usage.
    """

    name: str = "dedup"
    max_hashes: int = 100_000
    _seen: set[str] = field(default_factory=set, init=False, repr=False)

    def __call__(self, sample: VideoSample) -> VideoSample | None:
        digest = self.fingerprint(sample.frames)
        if digest in self._seen:
            return None
        if len(self._seen) >= self.max_hashes:
            self._seen.pop()
        self._seen.add(digest)
        sample.extra.setdefault("metrics", {})["fingerprint"] = digest
        return sample

    @staticmethod
    def fingerprint(frames: torch.Tensor) -> str:
        """Compute a stable 64-bit fingerprint of a video tensor."""

        if frames.dim() == 4:  # [T, C, H, W]
            f32 = frames.float().mean(dim=0)  # collapse T
        elif frames.dim() == 3:  # [C, H, W]
            f32 = frames.float()
        else:
            raise ValueError(f"unexpected frame shape {tuple(frames.shape)}")
        lum = (
            0.299 * f32[0] + 0.587 * f32[1] + 0.114 * f32[2]
        ).unsqueeze(0).unsqueeze(0)
        pooled = torch.nn.functional.adaptive_avg_pool2d(lum, output_size=(8, 8))
        flat = pooled.flatten()
        median = flat.median()
        bits = (flat > median).to(torch.uint8).tolist()
        as_int = 0
        for b in bits:
            as_int = (as_int << 1) | int(b)
        return f"{as_int:016x}"


@dataclass
class CallableFilter:
    """Wrap any plain function into a :class:`Filter`-conformant callable."""

    name: str
    fn: Callable[[VideoSample], VideoSample | None]

    def __call__(self, sample: VideoSample) -> VideoSample | None:
        return self.fn(sample)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class FilterStats:
    """Bookkeeping for a single filter inside a pipeline."""

    name: str
    seen: int = 0
    dropped: int = 0

    @property
    def kept(self) -> int:
        return self.seen - self.dropped

    @property
    def keep_rate(self) -> float:
        return self.kept / self.seen if self.seen else 0.0


@dataclass
class FilterPipeline:
    """Apply filters in order, recording per-filter accept/reject counts.

    The pipeline short-circuits as soon as a filter drops the sample, which
    means later filters never see rejects (matches real-world cost-saving).
    """

    filters: list[Filter]
    stats: dict[str, FilterStats] = field(default_factory=dict, init=False)
    total: int = field(default=0, init=False)
    kept: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        for f in self.filters:
            self.stats.setdefault(f.name, FilterStats(name=f.name))

    def __call__(self, sample: VideoSample) -> VideoSample | None:
        self.total += 1
        for f in self.filters:
            stat = self.stats[f.name]
            stat.seen += 1
            result = f(sample)
            if result is None:
                stat.dropped += 1
                return None
            sample = result
        self.kept += 1
        return sample

    def apply(
        self, stream: Iterable[VideoSample]
    ) -> Iterator[VideoSample]:
        """Apply the pipeline to a stream, yielding only kept samples."""

        for sample in stream:
            out = self(sample)
            if out is not None:
                yield out

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "seen": stat.seen,
                "dropped": stat.dropped,
                "kept": stat.kept,
                "keep_rate": stat.keep_rate,
            }
            for name, stat in self.stats.items()
        }


__all__ = [
    "AestheticFilter",
    "AspectRatioFilter",
    "CallableFilter",
    "DedupFilter",
    "Filter",
    "FilterPipeline",
    "FilterStats",
    "LicenseFilter",
    "MinDurationFilter",
    "MinResolutionFilter",
]
