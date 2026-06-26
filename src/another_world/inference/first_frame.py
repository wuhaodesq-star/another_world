"""Image / video file -> first-frame visual-token preprocessor.

Bridges the gap between raw user inputs (PNG / JPG / MP4) and the
:func:`rollout_visual_tokens` interface, which expects a tensor of
*local* visual ids shaped ``[T_prefix, H', W']``.

The visual tokenizer is pluggable: the :class:`FirstFramePreprocessor`
accepts any object with ``encode(video)`` matching the
:class:`~another_world.tokenizers.visual.cosmos.CosmosVideoTokenizer`
contract (input ``[B=1, 3, T, H, W]`` in [-1, 1]).  For unit tests and
local smoke runs we ship :class:`MockFirstFrameTokenizer` which produces
deterministic int64 ids without any external dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FirstFrameTokenizer(Protocol):
    """Interface implemented by visual tokenizers we accept here."""

    def encode(self, video: Tensor) -> Sequence[Tensor]: ...


# ---------------------------------------------------------------------------
# Mock tokenizer (offline, deterministic, no external deps)
# ---------------------------------------------------------------------------


@dataclass
class MockFirstFrameTokenizer:
    """Hash-based mock that quantises a downsampled luma image to ids.

    Output rank matches the Cosmos discrete tokenizer: ``[B, T', H', W']``.
    """

    vocab_size: int = 1024
    downsample_spatial: int = 8
    downsample_temporal: int = 4

    def encode(self, video: Tensor) -> tuple[Tensor]:
        if video.dim() != 5:
            raise ValueError(f"expected [B, C, T, H, W], got {tuple(video.shape)}")
        b, _, t, h, w = video.shape
        t_p = max(1, 1 + (t - 1) // self.downsample_temporal)
        h_p = max(1, h // self.downsample_spatial)
        w_p = max(1, w // self.downsample_spatial)
        luma = video.mean(dim=1, keepdim=True)  # [B, 1, T, H, W]
        pooled = torch.nn.functional.adaptive_avg_pool3d(
            luma, output_size=(t_p, h_p, w_p)
        )[:, 0]
        normed = (pooled - pooled.min()) / max(
            float(pooled.max() - pooled.min()), 1e-6
        )
        indices = (normed * (self.vocab_size - 1)).round().long()
        return (indices,)


# ---------------------------------------------------------------------------
# Image / video loaders
# ---------------------------------------------------------------------------


def load_image_as_video_tensor(
    path: str | Path,
    *,
    height: int,
    width: int,
    pad_frames: int = 1,
) -> Tensor:
    """Load a single image and produce ``[1, 3, pad_frames, H, W]`` in [-1, 1].

    The pad_frames axis is repeated so a still image becomes a short
    constant-clip suitable for video tokenizers that expect ``T >= 1``.
    """

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow always available
        raise ImportError("Pillow is required to load images (`pip install Pillow`).") from exc
    import numpy as np

    img = Image.open(str(path)).convert("RGB").resize((width, height))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [H, W, 3] in [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)         # [3, H, W]
    tensor = tensor.unsqueeze(1).repeat(1, pad_frames, 1, 1)  # [3, T, H, W]
    return tensor.unsqueeze(0)                              # [1, 3, T, H, W]


def load_video_as_video_tensor(
    path: str | Path,
    *,
    height: int,
    width: int,
    max_frames: int = 17,
) -> Tensor:
    """Load up to ``max_frames`` frames of a video into ``[1, 3, T, H, W]``."""

    try:
        import av  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyav is required to load videos (`pip install av`)."
        ) from exc
    import numpy as np

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    frames: list[np.ndarray] = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= max_frames:
            break
    container.close()

    if not frames:
        raise ValueError(f"video {path} produced zero frames")
    arr = np.stack(frames, axis=0).astype(np.float32)        # [T, H, W, 3]
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2) / 127.5 - 1.0
    tensor = tensor.unsqueeze(0)                             # [1, 3, T, H, W]
    if tensor.shape[-1] != width or tensor.shape[-2] != height:
        tensor = torch.nn.functional.interpolate(
            tensor[0],          # [3, T, H, W]
            size=(tensor.shape[2], height, width),
            mode="trilinear",
            align_corners=False,
        ).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


@dataclass
class FirstFramePreprocessor:
    """Wrap a visual tokenizer to produce ``[T_prefix, H', W']`` id cubes."""

    tokenizer: FirstFrameTokenizer
    target_h: int = 256
    target_w: int = 256
    pad_frames: int = 1

    def from_image(self, path: str | Path) -> Tensor:
        video = load_image_as_video_tensor(
            path, height=self.target_h, width=self.target_w,
            pad_frames=self.pad_frames,
        )
        return self.from_video_tensor(video)

    def from_video(self, path: str | Path, max_frames: int = 9) -> Tensor:
        video = load_video_as_video_tensor(
            path, height=self.target_h, width=self.target_w,
            max_frames=max_frames,
        )
        return self.from_video_tensor(video)

    def from_video_tensor(self, video: Tensor) -> Tensor:
        out = self.tokenizer.encode(video)
        ids = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(ids, Tensor):
            raise TypeError(
                f"tokenizer.encode must return a tensor; got {type(ids).__name__}"
            )
        # Strip batch axis, return [T', H', W'].
        if ids.dim() == 4:
            ids = ids[0]
        elif ids.dim() != 3:
            raise ValueError(
                f"unexpected tokenizer output rank {ids.dim()} "
                f"(expected 3 or 4)"
            )
        _LOG.info(
            "first frame tokens: %s, ids in [%d, %d]",
            tuple(ids.shape), int(ids.min()), int(ids.max()),
        )
        return ids.to(torch.long)


__all__ = [
    "FirstFramePreprocessor",
    "FirstFrameTokenizer",
    "MockFirstFrameTokenizer",
    "load_image_as_video_tensor",
    "load_video_as_video_tensor",
]
