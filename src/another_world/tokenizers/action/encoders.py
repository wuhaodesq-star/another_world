"""Action tokenizer.

Maps environment actions (numerical vectors, discrete categorical
indices, or mixed dicts) to integer ids in ``[0, vocab_size)`` that the
multimodal dynamics model can consume from its action slab.

Three implementations sharing :class:`ActionTokenizer`:

- :class:`DiscreteActionTokenizer`  : already-discrete env (Atari, gym
                                      ``Discrete(N)``); identity mapping.
- :class:`BinnedActionTokenizer`    : per-channel uniform quantisation of
                                      a continuous action vector. Each
                                      channel becomes ``bins`` tokens;
                                      they are interleaved into a flat
                                      stream of integer ids.
- :class:`CodebookActionTokenizer`  : k-means (FAISS or pure-torch)
                                      codebook over recorded action
                                      vectors. Stage 1.x feeds these
                                      tokenizers with logged trajectories.

The factory :func:`build_action_tokenizer` picks one by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class ActionTokenizer(Protocol):
    """Maps raw actions to / from integer ids."""

    @property
    def vocab_size(self) -> int: ...

    def encode(self, actions) -> Tensor: ...  # noqa: ANN001

    def decode(self, ids: Tensor) -> Tensor: ...


# ---------------------------------------------------------------------------
# Discrete (identity) tokenizer
# ---------------------------------------------------------------------------


@dataclass
class DiscreteActionTokenizer:
    """Identity mapping for already-discrete actions."""

    vocab_size_: int = 18  # Atari-like

    @property
    def vocab_size(self) -> int:
        return self.vocab_size_

    def encode(self, actions) -> Tensor:  # noqa: ANN001
        if isinstance(actions, int):
            actions = [actions]
        ids = torch.as_tensor(actions, dtype=torch.long)
        if (ids < 0).any() or (ids >= self.vocab_size_).any():
            raise ValueError(
                f"action ids must lie in [0, {self.vocab_size_}); got "
                f"min={int(ids.min())}, max={int(ids.max())}"
            )
        return ids

    def decode(self, ids: Tensor) -> Tensor:
        return ids.to(torch.long)


# ---------------------------------------------------------------------------
# Binned (per-channel) tokenizer
# ---------------------------------------------------------------------------


@dataclass
class BinnedActionTokenizer:
    """Uniformly quantise each channel of a continuous action vector.

    For an action ``a in [low, high]^D`` with ``bins`` bins per channel
    the encoder emits ``D`` integer ids, one per channel, in
    ``[0, bins)``. Total vocabulary size is ``D * bins`` -- per-channel
    ids are offset by ``channel_idx * bins`` so distinct channels do not
    collide.

    Args:
        dim: number of action channels ``D``.
        bins: bins per channel.
        low / high: per-channel clip range; if scalar, broadcast.
    """

    dim: int
    bins: int = 32
    low: float | list[float] = -1.0
    high: float | list[float] = 1.0

    _low: Tensor = field(init=False, repr=False)
    _high: Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.bins < 2:
            raise ValueError("bins must be >= 2")
        self._low = self._as_vec(self.low)
        self._high = self._as_vec(self.high)
        if (self._high <= self._low).any():
            raise ValueError("each `high` must be strictly greater than `low`")

    def _as_vec(self, x: float | list[float]) -> Tensor:
        if isinstance(x, (int, float)):
            return torch.full((self.dim,), float(x), dtype=torch.float32)
        v = torch.as_tensor(x, dtype=torch.float32)
        if v.shape != (self.dim,):
            raise ValueError(
                f"expected shape ({self.dim},) for vector, got {tuple(v.shape)}"
            )
        return v

    @property
    def vocab_size(self) -> int:
        return self.dim * self.bins

    def encode(self, actions) -> Tensor:  # noqa: ANN001
        """Encode an action ``[D]`` or batch ``[B, D]`` to ids ``[B*D]`` flat."""

        a = torch.as_tensor(actions, dtype=torch.float32)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if a.shape[-1] != self.dim:
            raise ValueError(
                f"expected last dim {self.dim}, got {tuple(a.shape)}"
            )
        clipped = torch.minimum(torch.maximum(a, self._low), self._high)
        normed = (clipped - self._low) / (self._high - self._low)
        ids = (normed * self.bins).clamp(max=self.bins - 1).to(torch.long)
        # Offset per-channel ids into disjoint slabs.
        offsets = torch.arange(self.dim, dtype=torch.long) * self.bins
        ids = ids + offsets
        return ids.view(-1)

    def decode(self, ids: Tensor) -> Tensor:
        """Reverse of :meth:`encode`. Returns ``[B, D]`` continuous actions."""

        ids = ids.to(torch.long)
        if ids.numel() % self.dim != 0:
            raise ValueError(
                f"id count {ids.numel()} not divisible by dim {self.dim}"
            )
        ids = ids.view(-1, self.dim)
        offsets = torch.arange(self.dim, dtype=torch.long) * self.bins
        local = ids - offsets
        local = local.clamp(0, self.bins - 1)
        normed = (local.float() + 0.5) / self.bins
        return self._low + normed * (self._high - self._low)


# ---------------------------------------------------------------------------
# Codebook tokenizer (k-means over observed actions)
# ---------------------------------------------------------------------------


@dataclass
class CodebookActionTokenizer:
    """k-means codebook mapping each action vector to its nearest centroid.

    Fitted offline (``fit()``) on a tensor of recorded actions.  After
    fitting, ``encode`` returns nearest-centroid ids and ``decode``
    returns the centroid vector for each id.
    """

    vocab_size_: int = 256
    n_iters: int = 25
    centroids: Tensor | None = None
    seed: int = 0

    @property
    def vocab_size(self) -> int:
        return self.vocab_size_

    def fit(self, actions: Tensor) -> "CodebookActionTokenizer":
        """Run mini-batch k-means on ``actions`` shape ``[N, D]``."""

        if actions.dim() != 2:
            raise ValueError(f"expected [N, D], got {tuple(actions.shape)}")
        n = actions.shape[0]
        k = min(self.vocab_size_, n)
        g = torch.Generator().manual_seed(self.seed)
        # initialise centroids from random samples.
        idx = torch.randperm(n, generator=g)[:k]
        centroids = actions[idx].clone()
        for _ in range(self.n_iters):
            dists = torch.cdist(actions, centroids)        # [N, K]
            assignments = dists.argmin(dim=1)
            new_centroids = centroids.clone()
            for ki in range(k):
                mask = assignments == ki
                if mask.any():
                    new_centroids[ki] = actions[mask].mean(dim=0)
            if torch.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        # Pad to exactly vocab_size by repeating the last centroid.
        if centroids.shape[0] < self.vocab_size_:
            pad = self.vocab_size_ - centroids.shape[0]
            extra = centroids[-1:].repeat(pad, 1)
            centroids = torch.cat([centroids, extra], dim=0)
        self.centroids = centroids
        return self

    def _ensure_fit(self) -> Tensor:
        if self.centroids is None:
            raise RuntimeError(
                "CodebookActionTokenizer must be fit() before encode/decode."
            )
        return self.centroids

    def encode(self, actions) -> Tensor:  # noqa: ANN001
        centroids = self._ensure_fit()
        a = torch.as_tensor(actions, dtype=torch.float32)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if a.shape[-1] != centroids.shape[-1]:
            raise ValueError(
                f"expected last dim {centroids.shape[-1]}, got {tuple(a.shape)}"
            )
        dists = torch.cdist(a, centroids)
        return dists.argmin(dim=-1).view(-1).to(torch.long)

    def decode(self, ids: Tensor) -> Tensor:
        centroids = self._ensure_fit()
        return centroids.index_select(0, ids.view(-1).to(torch.long))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_action_tokenizer(
    kind: str = "discrete",
    *,
    vocab_size: int = 32,
    dim: int = 1,
    bins: int = 32,
    low: float | list[float] = -1.0,
    high: float | list[float] = 1.0,
) -> ActionTokenizer:
    """Instantiate an action tokenizer by name."""

    if kind == "discrete":
        return DiscreteActionTokenizer(vocab_size_=vocab_size)
    if kind == "binned":
        return BinnedActionTokenizer(dim=dim, bins=bins, low=low, high=high)
    if kind == "codebook":
        return CodebookActionTokenizer(vocab_size_=vocab_size)
    raise ValueError(
        f"unknown action tokenizer kind '{kind}' "
        "(expected discrete, binned, or codebook)"
    )


__all__ = [
    "ActionTokenizer",
    "BinnedActionTokenizer",
    "CodebookActionTokenizer",
    "DiscreteActionTokenizer",
    "build_action_tokenizer",
]
