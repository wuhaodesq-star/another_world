"""RL / robotics trajectory dataset.

Wraps a list of recorded trajectories (frames + per-step actions) into
:class:`TokenSample` objects so the multimodal dynamics model can be
trained on action-conditioned video prediction without going through
the full WebDataset / shard pipeline first.

Each trajectory is a dict with:

- ``"frames"``  : tensor ``[T, C, H, W]`` in [-1, 1]
- ``"actions"`` : tensor ``[T]`` (discrete) or ``[T, D]`` (continuous)
- optional ``"caption"`` : str

The dataset takes a visual tokenizer (Cosmos-style) and an action
tokenizer (see :mod:`another_world.tokenizers.action`) and yields
:class:`TokenSample` entries ready for :class:`SequencePacker`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from another_world.data.datasets.sample import TokenSample
from another_world.tokenizers.action import ActionTokenizer
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass
class RLTrajectory:
    """A single recorded episode."""

    frames: Tensor       # [T, C, H, W] in [-1, 1]
    actions: Tensor      # [T] long for discrete, [T, D] float for continuous
    caption: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frames.dim() != 4:
            raise ValueError(
                f"frames must be [T, C, H, W], got {tuple(self.frames.shape)}"
            )
        if self.actions.dim() not in (1, 2):
            raise ValueError(
                f"actions must be [T] or [T, D], got {tuple(self.actions.shape)}"
            )
        if self.actions.shape[0] != self.frames.shape[0]:
            raise ValueError(
                f"frames T={self.frames.shape[0]} != actions T={self.actions.shape[0]}"
            )


@dataclass
class RLTrajectoryDataset(IterableDataset[TokenSample]):
    """Stream :class:`TokenSample` items from a list of trajectories.

    Args:
        trajectories: iterable of :class:`RLTrajectory`.
        visual_tokenizer: any object exposing ``encode(video)`` that
            returns a tuple ``(indices, ...)`` with shape ``[B, T', H', W']``.
            ``cosmos.CosmosVideoTokenizer`` and
            ``inference.first_frame.MockFirstFrameTokenizer`` both qualify.
        action_tokenizer: optional :class:`ActionTokenizer` to convert raw
            actions into per-step ids.  When ``None``, actions are assumed
            to be discrete ids already.
        loops: number of times to iterate over the input list (defaults to 1).
    """

    trajectories: list[RLTrajectory]
    visual_tokenizer: Any
    action_tokenizer: ActionTokenizer | None = None
    loops: int = 1

    def __iter__(self) -> Iterator[TokenSample]:
        for _ in range(max(1, int(self.loops))):
            for i, traj in enumerate(self.trajectories):
                try:
                    yield self._tokenize_one(traj, idx=i)
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("Skipping trajectory %d: %s", i, exc)

    def _tokenize_one(self, traj: RLTrajectory, *, idx: int) -> TokenSample:
        # Visual tokenizer expects [B, C, T, H, W].
        video = traj.frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        encoded = self.visual_tokenizer.encode(video)
        ids = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
        if not isinstance(ids, Tensor):
            raise TypeError(f"visual tokenizer must return a Tensor, got {type(ids)}")
        if ids.dim() == 4:
            ids = ids[0]
        visual_tokens = ids.to(torch.long)

        if self.action_tokenizer is not None:
            action_tokens = self.action_tokenizer.encode(traj.actions).to(torch.long)
        else:
            action_tokens = traj.actions.to(torch.long).reshape(-1)

        return TokenSample(
            visual_tokens=visual_tokens,
            action_tokens=action_tokens,
            text_tokens=None,
            key=f"rl-{idx:08d}",
            extra={"caption": traj.caption, **traj.extra},
        )


def trajectories_from_dicts(items: Iterable[dict[str, Any]]) -> list[RLTrajectory]:
    """Adapt a sequence of raw dicts into :class:`RLTrajectory` objects."""

    out: list[RLTrajectory] = []
    for d in items:
        out.append(
            RLTrajectory(
                frames=d["frames"],
                actions=d["actions"],
                caption=d.get("caption"),
                extra={k: v for k, v in d.items()
                       if k not in {"frames", "actions", "caption"}},
            )
        )
    return out


__all__ = [
    "RLTrajectory",
    "RLTrajectoryDataset",
    "trajectories_from_dicts",
]
