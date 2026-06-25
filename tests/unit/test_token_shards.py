"""Tests for token shard writer / reader round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.tokenize.shards import (
    RotatingShardWriter,
    ShardManifest,
    TokenShardWriter,
    make_sample_id,
    read_token_shard,
    read_token_shards,
)


def _samples(n: int = 3) -> list[TokenSample]:
    samples = []
    for i in range(n):
        samples.append(
            TokenSample(
                visual_tokens=torch.full((2, 3, 4), i, dtype=torch.long),
                text_tokens=torch.tensor([i, i + 1, i + 2], dtype=torch.long),
                key=f"sample-{i:03d}",
                extra={"caption": f"clip {i}", "source": "unit-test"},
            )
        )
    return samples


def test_writer_round_trip(tmp_path: Path) -> None:
    samples = _samples(3)
    path = tmp_path / "out.tar"
    with TokenShardWriter(path=path) as w:
        for s in samples:
            full = w.append(s)
            assert full is False  # no target size set
    assert path.exists()

    out = list(read_token_shard(path))
    assert len(out) == 3
    for got, exp in zip(out, samples):
        assert torch.equal(got.visual_tokens, exp.visual_tokens)
        assert got.text_tokens is not None and torch.equal(
            got.text_tokens, exp.text_tokens
        )
        assert got.key == exp.key
        assert got.extra["caption"] == exp.extra["caption"]


def test_writer_emits_manifest(tmp_path: Path) -> None:
    path = tmp_path / "out.tar"
    with TokenShardWriter(path=path, tokenizer="mock", source="unit") as w:
        for s in _samples(2):
            w.append(s)
    manifest_path = path.with_suffix(".manifest.json")
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["num_samples"] == 2
    assert data["tokenizer"] == "mock"
    assert data["source"] == "unit"
    assert len(data["keys"]) == 2


def test_writer_close_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "out.tar"
    w = TokenShardWriter(path=path)
    w.__enter__()
    w.append(_samples(1)[0])
    m1 = w.close()
    m2 = w.close()
    assert isinstance(m1, ShardManifest)
    assert m2 is None


def test_writer_without_context_raises(tmp_path: Path) -> None:
    w = TokenShardWriter(path=tmp_path / "out.tar")
    with pytest.raises(RuntimeError):
        w.append(_samples(1)[0])


def test_rotating_writer_creates_multiple_shards(tmp_path: Path) -> None:
    # Each sample's visual tensor is roughly 24 * 4 bytes; the JSON header is
    # also a few hundred bytes. We pick a tiny target to force rotation.
    with RotatingShardWriter(
        tmp_path, prefix="part", target_size_bytes=512, tokenizer="mock"
    ) as writer:
        for s in _samples(8):
            writer.write(s)
    shards = sorted(p for p in tmp_path.glob("part-*.tar"))
    assert len(shards) >= 2

    total = 0
    for p in shards:
        total += sum(1 for _ in read_token_shard(p))
    assert total == 8


def test_read_token_shards_chains(tmp_path: Path) -> None:
    paths = []
    for i, s in enumerate(_samples(4)):
        p = tmp_path / f"single-{i:02d}.tar"
        with TokenShardWriter(path=p) as w:
            w.append(s)
        paths.append(p)
    out = list(read_token_shards(paths))
    assert len(out) == 4
    assert [s.key for s in out] == ["sample-000", "sample-001", "sample-002", "sample-003"]


def test_writer_with_compression(tmp_path: Path) -> None:
    path = tmp_path / "out.tar.gz"
    with TokenShardWriter(path=path, compression="gz") as w:
        for s in _samples(2):
            w.append(s)
    assert path.exists()
    out = list(read_token_shard(path))
    assert len(out) == 2


def test_sample_id_helper_is_unique() -> None:
    a = make_sample_id()
    b = make_sample_id()
    assert a != b
    assert len(a) == 12
