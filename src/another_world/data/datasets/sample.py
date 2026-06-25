"""Data sample types shared between loaders, filters, and trainers.

We keep these as plain :func:`dataclass` containers so they round-trip
cleanly through ``torch.utils.data.DataLoader`` (which uses ``default_collate``
under the hood) and through WebDataset's tar payload conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class VideoSample:
    """A decoded video clip plus its optional captions / metadata.

    Shape convention for ``frames``: ``[T, C, H, W]`` with ``C=3``, ``dtype``
    either ``torch.uint8`` (raw decode) or ``torch.float32`` in ``[-1, 1]``
    (normalised).  ``frames`` is the only required field; downstream stages
    will populate ``caption`` / ``asr`` / ``tokens`` as the pipeline grows.
    """

    frames: torch.Tensor
    caption: str | None = None
    asr: str | None = None
    fps: float | None = None
    duration: float | None = None
    source: str | None = None
    license: str | None = None
    key: str | None = None
    tokens: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.frames.shape[-2]), int(self.frames.shape[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "caption": self.caption,
            "asr": self.asr,
            "fps": self.fps,
            "duration": self.duration,
            "source": self.source,
            "license": self.license,
            "key": self.key,
            "tokens": self.tokens,
            **self.extra,
        }


@dataclass
class TokenSample:
    """A pre-tokenised sample ready for direct ingestion by the dynamics model.

    ``visual_tokens`` is what the visual tokenizer produced (long for
    discrete, float for continuous).  ``text_tokens`` are BPE ids.
    ``action_tokens`` is optional (only environments with actions).
    """

    visual_tokens: torch.Tensor
    text_tokens: torch.Tensor | None = None
    action_tokens: torch.Tensor | None = None
    key: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = ["TokenSample", "VideoSample"]
