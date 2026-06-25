"""HuggingFace ``datasets`` adapter.

A thin wrapper that exposes any HF dataset (streaming or otherwise) as an
``IterableDataset`` of :class:`VideoSample` objects. This is the easiest path
to a real public-data smoke test on a laptop:

    >>> from another_world.data.datasets.hf import build_hf_video_stream
    >>> ds = build_hf_video_stream(
    ...     "HuggingFaceM4/WebVid",
    ...     split="train",
    ...     streaming=True,
    ...     limit=8,
    ... )
    >>> next(iter(ds)).frames.shape
    torch.Size([16, 3, 256, 256])

The adapter is read-only: it never downloads anything outside the HF cache.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from another_world.data.datasets.sample import VideoSample
from another_world.data.datasets.transforms import Transform
from another_world.data.datasets.webdataset_loader import _decode_mp4_bytes
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


# Field names commonly seen in public video datasets. The first match wins.
_VIDEO_FIELDS = ("video", "video_bytes", "mp4", "clip", "frames")
_CAPTION_FIELDS = ("caption", "text", "title", "name", "description")
_FPS_FIELDS = ("fps", "frame_rate", "video_fps")
_DURATION_FIELDS = ("duration", "duration_seconds", "video_duration")
_LICENSE_FIELDS = ("license", "licence")


def _pick(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _record_to_sample(record: dict[str, Any], max_frames: int | None) -> VideoSample:
    video = _pick(record, _VIDEO_FIELDS)
    if video is None:
        raise KeyError(
            f"record has no recognised video field (keys={sorted(record)})"
        )

    if isinstance(video, dict) and "bytes" in video:
        frames = _decode_mp4_bytes(video["bytes"], max_frames=max_frames)
    elif isinstance(video, (bytes, bytearray)):
        frames = _decode_mp4_bytes(bytes(video), max_frames=max_frames)
    elif isinstance(video, str):
        with open(video, "rb") as fh:
            frames = _decode_mp4_bytes(fh.read(), max_frames=max_frames)
    elif isinstance(video, torch.Tensor):
        frames = video
        if max_frames is not None and frames.shape[0] > max_frames:
            frames = frames[:max_frames]
    else:
        try:
            import numpy as np
            if isinstance(video, np.ndarray):
                frames = torch.from_numpy(video)
                if frames.dim() == 4 and frames.shape[-1] == 3:
                    frames = frames.permute(0, 3, 1, 2).contiguous()
            else:
                raise TypeError(type(video).__name__)
        except (ImportError, TypeError) as exc:
            raise TypeError(
                f"unsupported video payload type {type(video).__name__}"
            ) from exc

    caption = _pick(record, _CAPTION_FIELDS)
    fps = _pick(record, _FPS_FIELDS)
    duration = _pick(record, _DURATION_FIELDS)
    license_ = _pick(record, _LICENSE_FIELDS)
    key = record.get("__key__") or record.get("id") or record.get("video_id")
    return VideoSample(
        frames=frames,
        caption=str(caption) if caption is not None else None,
        fps=float(fps) if fps is not None else None,
        duration=float(duration) if duration is not None else None,
        license=str(license_) if license_ is not None else None,
        key=str(key) if key is not None else None,
        source="huggingface",
    )


@dataclass
class HfStreamSpec:
    """Configuration for :func:`build_hf_video_stream`."""

    repo_id: str
    split: str = "train"
    config: str | None = None
    streaming: bool = True
    limit: int | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    max_frames: int | None = 256
    cache_dir: str | None = None
    hf_token: str | None = None


class HfVideoIterableDataset(torch.utils.data.IterableDataset[VideoSample]):
    """Wraps a HuggingFace dataset/iterable into a video-sample stream."""

    def __init__(
        self,
        spec: HfStreamSpec,
        transform: Transform | None = None,
    ) -> None:
        self.spec = spec
        self.transform = transform

    def _iter_raw(self) -> Iterator[dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - only when missing
            raise ImportError(
                "huggingface `datasets` is required "
                "(`pip install datasets`)."
            ) from exc

        ds = load_dataset(
            self.spec.repo_id,
            self.spec.config,
            split=self.spec.split,
            streaming=self.spec.streaming,
            revision=self.spec.revision,
            trust_remote_code=self.spec.trust_remote_code,
            cache_dir=self.spec.cache_dir,
            token=self.spec.hf_token,
        )
        n = 0
        for raw in ds:
            yield raw
            n += 1
            if self.spec.limit is not None and n >= self.spec.limit:
                break

    def __iter__(self) -> Iterator[VideoSample]:
        for record in self._iter_raw():
            try:
                sample = _record_to_sample(record, max_frames=self.spec.max_frames)
            except Exception as exc:  # noqa: BLE001 - keep stream flowing
                _LOG.warning("Skipping HF record: %s", exc)
                continue
            if self.transform is not None:
                sample = self.transform(sample)
            yield sample


def build_hf_video_stream(
    repo_id: str,
    *,
    split: str = "train",
    config: str | None = None,
    streaming: bool = True,
    limit: int | None = None,
    transform: Transform | None = None,
    max_frames: int | None = 256,
    cache_dir: str | None = None,
    hf_token: str | None = None,
) -> HfVideoIterableDataset:
    """Convenience constructor for :class:`HfVideoIterableDataset`."""

    spec = HfStreamSpec(
        repo_id=repo_id,
        split=split,
        config=config,
        streaming=streaming,
        limit=limit,
        max_frames=max_frames,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )
    return HfVideoIterableDataset(spec, transform=transform)


__all__ = [
    "HfStreamSpec",
    "HfVideoIterableDataset",
    "build_hf_video_stream",
]
