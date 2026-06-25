"""Video frame transforms used by the data pipeline.

These are designed to be cheap, deterministic when seeded, and to compose
into a single callable that maps :class:`VideoSample` -> :class:`VideoSample`.
We deliberately reimplement what we need with pure ``torch`` ops instead of
pulling in ``torchvision.transforms`` for two reasons:

1. We need a tensor pipeline that operates on ``[T, C, H, W]`` directly,
   not on PIL images.
2. We want zero hidden dependencies that would break Windows / CPU-only CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from another_world.data.datasets.sample import VideoSample


Transform = Callable[[VideoSample], VideoSample]


# ---------------------------------------------------------------------------
# basic ops
# ---------------------------------------------------------------------------


def to_float_minus_one_one(sample: VideoSample) -> VideoSample:
    """Convert ``uint8`` frames in ``[0, 255]`` to ``float32`` in ``[-1, 1]``."""

    frames = sample.frames
    if frames.dtype == torch.uint8:
        frames = frames.to(torch.float32) / 127.5 - 1.0
    elif frames.dtype != torch.float32:
        frames = frames.to(torch.float32)
        if frames.max() > 1.5:  # heuristic: looks like [0, 255]
            frames = frames / 127.5 - 1.0
    sample.frames = frames
    return sample


def to_float_zero_one(sample: VideoSample) -> VideoSample:
    """Convert frames to ``float32`` in ``[0, 1]``."""

    frames = sample.frames
    if frames.dtype == torch.uint8:
        frames = frames.to(torch.float32) / 255.0
    elif frames.dtype != torch.float32:
        frames = frames.to(torch.float32)
        if frames.max() > 1.5:
            frames = frames / 255.0
    sample.frames = frames
    return sample


# ---------------------------------------------------------------------------
# resize / crop
# ---------------------------------------------------------------------------


@dataclass
class Resize:
    """Resize ``[T, C, H, W]`` frames to ``(height, width)`` via bilinear interp."""

    height: int
    width: int
    mode: str = "bilinear"

    def __call__(self, sample: VideoSample) -> VideoSample:
        frames = sample.frames
        if frames.dim() != 4:
            raise ValueError(
                f"expected [T, C, H, W] frames, got {tuple(frames.shape)}"
            )
        was_uint8 = frames.dtype == torch.uint8
        if was_uint8:
            frames = frames.to(torch.float32)
        # ``F.interpolate`` operates on [N, C, H, W]; T plays the role of N.
        out = torch.nn.functional.interpolate(
            frames,
            size=(self.height, self.width),
            mode=self.mode,
            align_corners=False if self.mode in {"bilinear", "bicubic"} else None,
        )
        if was_uint8:
            out = out.clamp_(0, 255).to(torch.uint8)
        sample.frames = out
        return sample


@dataclass
class CenterCrop:
    """Center-crop ``[T, C, H, W]`` frames to ``(height, width)``."""

    height: int
    width: int

    def __call__(self, sample: VideoSample) -> VideoSample:
        frames = sample.frames
        _, _, h, w = frames.shape
        if h < self.height or w < self.width:
            raise ValueError(
                f"crop size ({self.height}, {self.width}) larger than frame "
                f"size ({h}, {w})"
            )
        top = (h - self.height) // 2
        left = (w - self.width) // 2
        sample.frames = frames[..., top : top + self.height, left : left + self.width]
        return sample


# ---------------------------------------------------------------------------
# temporal ops
# ---------------------------------------------------------------------------


@dataclass
class TemporalSample:
    """Pick ``num_frames`` evenly spaced indices from the time dimension.

    If the source clip has fewer than ``num_frames`` frames we either pad
    by repeating the last frame (``pad="repeat"``) or raise (``pad="error"``).
    """

    num_frames: int
    pad: str = "repeat"

    def __call__(self, sample: VideoSample) -> VideoSample:
        frames = sample.frames
        t = frames.shape[0]
        if t == 0:
            raise ValueError("video has zero frames")
        if t >= self.num_frames:
            idx = torch.linspace(0, t - 1, steps=self.num_frames).round().long()
            sample.frames = frames.index_select(0, idx)
            return sample

        if self.pad == "error":
            raise ValueError(
                f"only {t} frames available, need {self.num_frames}"
            )
        # repeat-pad
        idx = torch.arange(self.num_frames).clamp_max(t - 1)
        sample.frames = frames.index_select(0, idx)
        return sample


@dataclass
class TemporalRandomClip:
    """Take a random contiguous clip of ``num_frames``.

    ``seed`` makes the choice deterministic, useful for tests.
    """

    num_frames: int
    seed: int | None = None

    def __call__(self, sample: VideoSample) -> VideoSample:
        frames = sample.frames
        t = frames.shape[0]
        if t < self.num_frames:
            # fall back to repeat-pad
            idx = torch.arange(self.num_frames).clamp_max(t - 1)
            sample.frames = frames.index_select(0, idx)
            return sample
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed + t)
        start = int(
            torch.randint(0, t - self.num_frames + 1, (1,), generator=generator).item()
        )
        sample.frames = frames[start : start + self.num_frames]
        return sample


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


@dataclass
class Compose:
    """Run a list of transforms in order."""

    transforms: Sequence[Transform]

    def __call__(self, sample: VideoSample) -> VideoSample:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_default_transform(
    *,
    num_frames: int,
    height: int,
    width: int,
    normalise: str = "minus_one_one",
    temporal: str = "sample",
    seed: int | None = None,
) -> Compose:
    """Convenience builder for the standard ``video -> tensor`` pipeline."""

    transforms: list[Transform] = []
    if temporal == "sample":
        transforms.append(TemporalSample(num_frames=num_frames))
    elif temporal == "random":
        transforms.append(TemporalRandomClip(num_frames=num_frames, seed=seed))
    else:
        raise ValueError(f"unknown temporal mode '{temporal}'")
    transforms.append(Resize(height=height, width=width))
    if normalise == "minus_one_one":
        transforms.append(to_float_minus_one_one)
    elif normalise == "zero_one":
        transforms.append(to_float_zero_one)
    elif normalise == "none":
        pass
    else:
        raise ValueError(f"unknown normalise mode '{normalise}'")
    return Compose(transforms)


__all__ = [
    "CenterCrop",
    "Compose",
    "Resize",
    "TemporalRandomClip",
    "TemporalSample",
    "Transform",
    "build_default_transform",
    "to_float_minus_one_one",
    "to_float_zero_one",
]
