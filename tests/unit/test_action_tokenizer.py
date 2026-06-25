"""Tests for the action tokenizer module."""

from __future__ import annotations

import pytest
import torch

from another_world.tokenizers.action import (
    ActionTokenizer,
    BinnedActionTokenizer,
    CodebookActionTokenizer,
    DiscreteActionTokenizer,
    build_action_tokenizer,
)


# ---------------------------------------------------------------------------
# Discrete
# ---------------------------------------------------------------------------


def test_discrete_round_trip() -> None:
    tk = DiscreteActionTokenizer(vocab_size_=8)
    ids = tk.encode([0, 3, 7])
    assert torch.equal(ids, torch.tensor([0, 3, 7]))
    assert torch.equal(tk.decode(ids), ids)


def test_discrete_single_int() -> None:
    tk = DiscreteActionTokenizer(vocab_size_=4)
    assert torch.equal(tk.encode(2), torch.tensor([2]))


def test_discrete_out_of_range_raises() -> None:
    tk = DiscreteActionTokenizer(vocab_size_=4)
    with pytest.raises(ValueError):
        tk.encode([4])
    with pytest.raises(ValueError):
        tk.encode([-1])


# ---------------------------------------------------------------------------
# Binned
# ---------------------------------------------------------------------------


def test_binned_round_trip_approximate() -> None:
    tk = BinnedActionTokenizer(dim=2, bins=16, low=-1.0, high=1.0)
    action = torch.tensor([-0.5, 0.7])
    ids = tk.encode(action)
    rec = tk.decode(ids)[0]
    # Quantisation error <= bin width.
    assert (rec - action).abs().max() < (2.0 / 16)


def test_binned_vocab_size_is_dim_times_bins() -> None:
    tk = BinnedActionTokenizer(dim=3, bins=8)
    assert tk.vocab_size == 24


def test_binned_channels_are_disjoint_slabs() -> None:
    tk = BinnedActionTokenizer(dim=3, bins=4)
    ids = tk.encode(torch.tensor([-1.0, -1.0, -1.0]))
    # All channels saturated low -> local id 0 per channel; offsets 0,4,8.
    assert ids.tolist() == [0, 4, 8]


def test_binned_batch_input() -> None:
    tk = BinnedActionTokenizer(dim=2, bins=4)
    ids = tk.encode(torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))
    assert ids.shape == (4,)


def test_binned_rejects_bad_dim() -> None:
    with pytest.raises(ValueError):
        BinnedActionTokenizer(dim=0)
    with pytest.raises(ValueError):
        BinnedActionTokenizer(dim=2, bins=1)


def test_binned_rejects_low_ge_high() -> None:
    with pytest.raises(ValueError):
        BinnedActionTokenizer(dim=2, low=1.0, high=0.0)


def test_binned_per_channel_vector_low_high() -> None:
    tk = BinnedActionTokenizer(
        dim=2, bins=4, low=[-1.0, 0.0], high=[1.0, 4.0],
    )
    ids = tk.encode(torch.tensor([0.0, 2.0]))
    # First channel: 0.0 in [-1, 1] -> normed=0.5 -> bin 2. Offset 0.
    # Second channel: 2.0 in [0, 4] -> normed=0.5 -> bin 2. Offset 4.
    assert ids.tolist() == [2, 6]


# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------


def test_codebook_fit_assigns_clusters() -> None:
    torch.manual_seed(0)
    # Two well-separated clusters.
    a = torch.randn(100, 4) + 5.0
    b = torch.randn(100, 4) - 5.0
    data = torch.cat([a, b], dim=0)
    tk = CodebookActionTokenizer(vocab_size_=2, n_iters=20, seed=0)
    tk.fit(data)
    ids_a = tk.encode(a)
    ids_b = tk.encode(b)
    # Each cluster should map predominantly to a single id.
    a_majority = ids_a.bincount().max().item() / float(ids_a.numel())
    b_majority = ids_b.bincount().max().item() / float(ids_b.numel())
    assert a_majority > 0.9
    assert b_majority > 0.9


def test_codebook_decode_recovers_centroid() -> None:
    torch.manual_seed(1)
    data = torch.randn(50, 3)
    tk = CodebookActionTokenizer(vocab_size_=8, n_iters=5).fit(data)
    ids = tk.encode(data[:3])
    decoded = tk.decode(ids)
    assert decoded.shape == (3, 3)


def test_codebook_requires_fit() -> None:
    tk = CodebookActionTokenizer(vocab_size_=4)
    with pytest.raises(RuntimeError):
        tk.encode(torch.zeros(2, 3))


def test_codebook_rejects_wrong_input_dim() -> None:
    tk = CodebookActionTokenizer(vocab_size_=4).fit(torch.randn(20, 4))
    with pytest.raises(ValueError):
        tk.encode(torch.zeros(2, 5))


# ---------------------------------------------------------------------------
# Factory + Protocol
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(
        build_action_tokenizer("discrete", vocab_size=4),
        DiscreteActionTokenizer,
    )
    assert isinstance(
        build_action_tokenizer("binned", dim=2, bins=4),
        BinnedActionTokenizer,
    )
    assert isinstance(
        build_action_tokenizer("codebook", vocab_size=4),
        CodebookActionTokenizer,
    )


def test_factory_unknown_raises() -> None:
    with pytest.raises(ValueError):
        build_action_tokenizer("bogus")


def test_protocol_runtime_check() -> None:
    tk = DiscreteActionTokenizer(vocab_size_=4)
    assert isinstance(tk, ActionTokenizer)
