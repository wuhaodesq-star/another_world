"""Tests for the dummy token dataset."""

from __future__ import annotations

import pytest
import torch

from another_world.data.datasets.dummy import DummyTokenDataset


def test_length_and_indexing() -> None:
    ds = DummyTokenDataset(vocab_size=16, seq_len=8, length=4, seed=0)
    assert len(ds) == 4
    inputs, targets = ds[0]
    assert inputs.shape == (8,)
    assert targets.shape == (8,)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


def test_targets_are_shifted_by_one() -> None:
    ds = DummyTokenDataset(vocab_size=16, seq_len=8, length=2, seed=1)
    inputs, targets = ds[0]
    # The raw row has length seq_len + 1; targets[i] should equal inputs[i+1].
    assert torch.equal(targets[:-1], inputs[1:])


def test_determinism_across_instances() -> None:
    a = DummyTokenDataset(vocab_size=16, seq_len=8, length=2, seed=42)
    b = DummyTokenDataset(vocab_size=16, seq_len=8, length=2, seed=42)
    assert torch.equal(a[0][0], b[0][0])
    assert torch.equal(a[1][1], b[1][1])


def test_token_values_within_vocab() -> None:
    ds = DummyTokenDataset(vocab_size=7, seq_len=12, length=3, seed=2)
    for i in range(len(ds)):
        x, y = ds[i]
        assert int(x.min()) >= 0
        assert int(y.min()) >= 0
        assert int(x.max()) < 7
        assert int(y.max()) < 7


def test_invalid_args_raise() -> None:
    with pytest.raises(ValueError):
        DummyTokenDataset(vocab_size=1)
    with pytest.raises(ValueError):
        DummyTokenDataset(seq_len=1)
    with pytest.raises(ValueError):
        DummyTokenDataset(length=0)


def test_out_of_range_index() -> None:
    ds = DummyTokenDataset(vocab_size=4, seq_len=4, length=2, seed=0)
    with pytest.raises(IndexError):
        _ = ds[2]
