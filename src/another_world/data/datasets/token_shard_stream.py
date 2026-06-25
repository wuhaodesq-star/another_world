"""Stream :class:`PackedBatch` objects from token shards.

Bridges the data pipeline (tar-packed pre-tokenised shards from stage 1.3)
to the training loop (consumes :class:`PackedBatch`).

The reader is a plain Python generator so it can be wrapped in either
``torch.utils.data.IterableDataset`` or used directly in the trainer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import IterableDataset

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import PackedBatch, SequencePacker
from another_world.data.tokenize.shards import read_token_shard, read_token_shards
from another_world.tokenizers.vocab import VocabLayout
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


def _reshape_visual_tokens(t: torch.Tensor) -> torch.Tensor:
    """Coerce visual tokens to ``[T, H, W]``."""

    if t.dim() == 4 and t.shape[0] == 1:
        return t[0]
    if t.dim() == 3:
        return t
    raise ValueError(
        f"unexpected visual_tokens shape {tuple(t.shape)} "
        "(expected [T, H, W] or [1, T, H, W])"
    )


def iter_token_samples(
    shard_paths: Iterable[str | Path],
) -> Iterator[TokenSample]:
    """Iterate :class:`TokenSample`s across a list of shards.

    Reshapes the visual tensor to ``[T, H, W]`` defensively so the rest
    of the pipeline can assume the canonical layout.
    """

    for sample in read_token_shards(shard_paths):
        try:
            sample.visual_tokens = _reshape_visual_tokens(sample.visual_tokens)
        except ValueError as exc:
            _LOG.warning("Skipping sample %s: %s", sample.key, exc)
            continue
        yield sample


def iter_packed_batches(
    sample_stream: Iterable[TokenSample],
    *,
    packer: SequencePacker,
    batch_size: int,
    drop_last: bool = True,
) -> Iterator[PackedBatch]:
    """Group samples into mini-batches and pack each one."""

    buffer: list[TokenSample] = []
    for sample in sample_stream:
        buffer.append(sample)
        if len(buffer) == batch_size:
            yield packer.pack_batch(buffer)
            buffer = []
    if buffer and not drop_last:
        yield packer.pack_batch(buffer)


@dataclass
class TokenShardLoaderSpec:
    """Configuration for :class:`TokenShardDataset` and the iterator factory."""

    shards: list[str]
    batch_size: int = 4
    max_len: int = 1024
    drop_last: bool = True
    score_text: bool = False
    score_visual: bool = True
    score_action: bool = True
    include_text: bool = True
    layout: VocabLayout | None = None
    loops: int = 1  # number of times to walk through all shards


class TokenShardDataset(IterableDataset[PackedBatch]):
    """A torch ``IterableDataset`` over packed batches read from shards.

    Useful when you want to plug this into :class:`torch.utils.data.DataLoader`
    with ``num_workers > 0`` so disk I/O happens off the trainer thread.
    """

    def __init__(self, spec: TokenShardLoaderSpec) -> None:
        self.spec = spec
        layout = spec.layout or VocabLayout.default()
        self._packer = SequencePacker(
            layout=layout,
            max_len=spec.max_len,
            include_text=spec.include_text,
            score_text=spec.score_text,
            score_visual=spec.score_visual,
            score_action=spec.score_action,
        )

    @property
    def packer(self) -> SequencePacker:
        return self._packer

    def __iter__(self) -> Iterator[PackedBatch]:
        for _ in range(max(1, self.spec.loops)):
            samples = iter_token_samples(self.spec.shards)
            yield from iter_packed_batches(
                samples,
                packer=self._packer,
                batch_size=self.spec.batch_size,
                drop_last=self.spec.drop_last,
            )


def build_packed_batch_stream(
    shards: list[str],
    *,
    packer: SequencePacker,
    batch_size: int,
    loops: int = 1,
    drop_last: bool = True,
) -> Iterator[PackedBatch]:
    """Convenience: shards -> samples -> packed batches generator."""

    for _ in range(max(1, loops)):
        samples = iter_token_samples(shards)
        yield from iter_packed_batches(
            samples,
            packer=packer,
            batch_size=batch_size,
            drop_last=drop_last,
        )


__all__ = [
    "TokenShardDataset",
    "TokenShardLoaderSpec",
    "build_packed_batch_stream",
    "iter_packed_batches",
    "iter_token_samples",
]
