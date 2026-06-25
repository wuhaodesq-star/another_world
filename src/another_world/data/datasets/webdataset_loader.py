"""WebDataset-based streaming loader for video shards.

WebDataset stores samples as tar archives with a shared ``__key__`` plus
multiple "fields", e.g.::

    sample001.mp4
    sample001.txt        # caption
    sample001.json       # metadata

We treat ``.mp4`` (preferred) or pre-decoded ``.npy`` frame arrays as the
visual payload, ``.txt`` / ``.json`` for text and metadata. The decoded
output is a :class:`VideoSample`.

The :mod:`webdataset` import is lazy so importing this module never fails
even on machines that don't have it (CI, Windows dev boxes). Call
:func:`build_video_webdataset` only after ``pip install webdataset av``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch

from another_world.data.datasets.sample import VideoSample
from another_world.data.datasets.transforms import Transform
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)

DecodedSample = dict[str, Any]


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


def _decode_mp4_bytes(data: bytes, max_frames: int | None = None) -> torch.Tensor:
    """Decode an mp4 byte blob into a ``[T, C, H, W] uint8`` tensor."""

    try:
        import av  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:  # pragma: no cover - only when av missing
        raise ImportError(
            "pyav and numpy are required to decode mp4 shards "
            "(`pip install av numpy`)."
        ) from exc

    container = av.open(io.BytesIO(data))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    frames: list[np.ndarray] = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
        if max_frames is not None and len(frames) >= max_frames:
            break
    container.close()
    if not frames:
        raise ValueError("decoded mp4 has zero frames")
    arr = np.stack(frames, axis=0)  # [T, H, W, 3]
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
    return tensor


def _decode_npy_bytes(data: bytes) -> torch.Tensor:
    import numpy as np

    arr = np.load(io.BytesIO(data), allow_pickle=False)
    tensor = torch.from_numpy(arr)
    if tensor.dim() == 4 and tensor.shape[-1] == 3:
        # treat as [T, H, W, C]
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor


def decode_webdataset_sample(
    raw: dict[str, Any],
    *,
    max_frames: int | None = None,
) -> VideoSample:
    """Decode a WebDataset raw sample dict into a :class:`VideoSample`.

    The ``raw`` dict has keys like ``"mp4"``, ``"txt"``, ``"json"``, etc.;
    values are the raw bytes payload of the corresponding file in the tar.
    """

    if "mp4" in raw:
        frames = _decode_mp4_bytes(raw["mp4"], max_frames=max_frames)
    elif "npy" in raw:
        frames = _decode_npy_bytes(raw["npy"])
        if max_frames is not None and frames.shape[0] > max_frames:
            frames = frames[:max_frames]
    else:
        raise ValueError(
            f"sample has no recognised visual payload (keys: {sorted(raw)})"
        )

    caption = None
    if "txt" in raw:
        caption = raw["txt"].decode("utf-8").strip() if isinstance(raw["txt"], bytes) else str(raw["txt"]).strip()

    meta: dict[str, Any] = {}
    asr = None
    fps = None
    duration = None
    source = None
    license_ = None
    if "json" in raw:
        try:
            payload = raw["json"]
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            meta = json.loads(payload) if isinstance(payload, str) else dict(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            _LOG.warning("Failed to parse sample json: %s", exc)
        else:
            asr = meta.get("asr")
            fps = meta.get("fps")
            duration = meta.get("duration")
            source = meta.get("source")
            license_ = meta.get("license")

    return VideoSample(
        frames=frames,
        caption=caption,
        asr=asr,
        fps=fps,
        duration=duration,
        source=source,
        license=license_,
        key=raw.get("__key__"),
        extra={"meta": meta} if meta else {},
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclass
class WebDatasetSpec:
    """Configuration for :func:`build_video_webdataset`.

    Attributes:
        urls: shard URLs (``str``) or a brace expansion (``"s3://b/{0000..0099}.tar"``).
        shardshuffle: whether to shuffle shard order each epoch.
        shuffle_buffer: in-memory sample shuffle buffer size.
        max_frames: hard cap on decoded frames per sample (saves memory).
        handler: callable invoked on per-sample exceptions (default: log + skip).
    """

    urls: str | list[str]
    shardshuffle: bool = True
    shuffle_buffer: int = 1024
    max_frames: int | None = 256
    handler: Callable[[Exception], bool] | None = None
    resampled: bool = False
    cache_dir: str | None = None


def build_video_webdataset(
    spec: WebDatasetSpec,
    transform: Transform | None = None,
) -> "Iterable[VideoSample]":
    """Build a streaming WebDataset pipeline that yields :class:`VideoSample`s.

    The pipeline is:

    1. ``WebDataset(urls)`` - tar stream over the given shards.
    2. ``shuffle(buffer)`` - approximate sample-level shuffle.
    3. ``map(decode_webdataset_sample)`` - bytes -> VideoSample.
    4. ``map(transform)`` - user-supplied transforms (resize / crop / etc).

    Notes:
        - We use ``webdataset`` if installed, else raise an ImportError so
          callers can fall back to other loaders. The wrapper accepts the
          same ``spec`` for both real and mocked paths.
    """

    try:
        import webdataset as wds  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only when wds missing
        raise ImportError(
            "webdataset is required (`pip install webdataset`)."
        ) from exc

    def _handler(exn: Exception) -> bool:
        _LOG.warning("Skipping bad sample: %s", exn)
        return True

    handler = spec.handler or _handler

    ds = wds.WebDataset(
        spec.urls,
        shardshuffle=spec.shardshuffle,
        handler=handler,
        resampled=spec.resampled,
        cache_dir=spec.cache_dir,
    )
    if spec.shuffle_buffer > 0:
        ds = ds.shuffle(spec.shuffle_buffer)
    ds = ds.map(
        lambda raw: decode_webdataset_sample(raw, max_frames=spec.max_frames),
        handler=handler,
    )
    if transform is not None:
        ds = ds.map(transform, handler=handler)
    return ds


# ---------------------------------------------------------------------------
# In-memory fallback for tests
# ---------------------------------------------------------------------------


class IterableVideoDataset(torch.utils.data.IterableDataset[VideoSample]):
    """A tiny iterable dataset wrapping an explicit list of samples.

    Used by unit tests (and as a stand-in until real shards exist) to verify
    that downstream code consumes :class:`VideoSample` correctly without
    paying the cost of decoding real mp4s.
    """

    def __init__(
        self,
        samples: Sequence[VideoSample],
        transform: Transform | None = None,
        loops: int = 1,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.loops = max(1, int(loops))

    def __iter__(self) -> Iterator[VideoSample]:
        for _ in range(self.loops):
            for s in self.samples:
                # Defensive copy so transforms don't mutate the shared cache.
                copy = VideoSample(
                    frames=s.frames.clone(),
                    caption=s.caption,
                    asr=s.asr,
                    fps=s.fps,
                    duration=s.duration,
                    source=s.source,
                    license=s.license,
                    key=s.key,
                    tokens=None if s.tokens is None else s.tokens.clone(),
                    extra=dict(s.extra),
                )
                if self.transform is not None:
                    copy = self.transform(copy)
                yield copy


def collate_video_samples(batch: Sequence[VideoSample]) -> dict[str, Any]:
    """Stack a list of :class:`VideoSample` into a dict of batched tensors.

    Frames are stacked into ``[B, T, C, H, W]``. Strings are collected into
    lists. ``tokens`` (optional) is stacked if all items provide one.
    """

    if not batch:
        raise ValueError("empty batch")
    frames = torch.stack([s.frames for s in batch], dim=0)
    out: dict[str, Any] = {
        "frames": frames,
        "caption": [s.caption for s in batch],
        "key": [s.key for s in batch],
        "source": [s.source for s in batch],
        "license": [s.license for s in batch],
        "fps": [s.fps for s in batch],
    }
    if all(s.tokens is not None for s in batch):
        out["tokens"] = torch.stack([s.tokens for s in batch], dim=0)  # type: ignore[arg-type]
    return out


__all__ = [
    "IterableVideoDataset",
    "WebDatasetSpec",
    "build_video_webdataset",
    "collate_video_samples",
    "decode_webdataset_sample",
]
