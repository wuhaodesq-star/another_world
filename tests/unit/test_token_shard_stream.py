"""Tests for the token-shard -> packed-batch dataloader bridge."""

from __future__ import annotations

from pathlib import Path

import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import SequencePacker
from another_world.data.datasets.token_shard_stream import (
    TokenShardDataset,
    TokenShardLoaderSpec,
    build_packed_batch_stream,
    iter_packed_batches,
    iter_token_samples,
)
from another_world.data.tokenize import TokenShardWriter
from another_world.tokenizers.vocab import VocabLayout


def _write_shard(path: Path, n: int) -> None:
    with TokenShardWriter(path=path) as w:
        for i in range(n):
            w.append(
                TokenSample(
                    visual_tokens=torch.randint(0, 16, (1, 2, 2), dtype=torch.long),
                    text_tokens=torch.tensor([i, i + 1, i + 2], dtype=torch.long),
                    key=f"sample-{i}",
                )
            )


def test_iter_token_samples_reshapes_visual(tmp_path: Path) -> None:
    path = tmp_path / "shard.tar"
    _write_shard(path, 3)
    samples = list(iter_token_samples([path]))
    assert len(samples) == 3
    for s in samples:
        assert s.visual_tokens.dim() == 3  # [T, H, W] after reshape


def test_iter_packed_batches_groups_by_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "shard.tar"
    _write_shard(path, 5)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=24)
    samples = iter_token_samples([path])
    batches = list(iter_packed_batches(samples, packer=packer, batch_size=2))
    assert len(batches) == 2  # 5 // 2 = 2 with drop_last=True
    for b in batches:
        assert b.tokens.shape == (2, 24)
        assert b.axes.modality.shape == (2, 24)


def test_iter_packed_batches_keep_remainder(tmp_path: Path) -> None:
    path = tmp_path / "shard.tar"
    _write_shard(path, 5)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=24)
    samples = iter_token_samples([path])
    batches = list(
        iter_packed_batches(samples, packer=packer, batch_size=2, drop_last=False)
    )
    # 2 + 2 + 1 = 3 batches with the last carrying only 1 sample.
    assert len(batches) == 3
    assert batches[-1].tokens.shape[0] == 1


def test_token_shard_dataset_yields_batches(tmp_path: Path) -> None:
    path = tmp_path / "shard.tar"
    _write_shard(path, 4)
    spec = TokenShardLoaderSpec(
        shards=[str(path)],
        batch_size=2,
        max_len=24,
        layout=VocabLayout.tiny(),
    )
    ds = TokenShardDataset(spec)
    batches = list(ds)
    assert len(batches) == 2
    assert batches[0].tokens.shape == (2, 24)


def test_token_shard_dataset_loops(tmp_path: Path) -> None:
    path = tmp_path / "shard.tar"
    _write_shard(path, 2)
    spec = TokenShardLoaderSpec(
        shards=[str(path)],
        batch_size=2,
        max_len=24,
        layout=VocabLayout.tiny(),
        loops=3,
    )
    ds = TokenShardDataset(spec)
    batches = list(ds)
    assert len(batches) == 3  # 1 batch per loop * 3 loops


def test_build_packed_batch_stream_helper(tmp_path: Path) -> None:
    path = tmp_path / "shard.tar"
    _write_shard(path, 4)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=16)
    out = list(build_packed_batch_stream([str(path)], packer=packer, batch_size=2))
    assert len(out) == 2
    assert all(b.tokens.shape == (2, 16) for b in out)
