"""Tests for the KV cache."""

from __future__ import annotations

import pytest
import torch

from another_world.inference.kv_cache import (
    KVCache,
    LayerKVCache,
    build_kv_cache,
    incremental_forward,
)
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.models.layers.mixed_rope import RopeAxes
from another_world.tokenizers.vocab import VocabLayout


def _toy_model(layout: VocabLayout | None = None) -> MultimodalDynamicsModel:
    layout = layout or VocabLayout.tiny()
    return MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )


def _full_axes(seq_len: int) -> RopeAxes:
    return RopeAxes(
        modality=torch.zeros(1, seq_len, dtype=torch.long),
        linear=torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        t=torch.zeros(1, seq_len, dtype=torch.long),
        h=torch.zeros(1, seq_len, dtype=torch.long),
        w=torch.zeros(1, seq_len, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Container correctness
# ---------------------------------------------------------------------------


def test_build_kv_cache_geometry() -> None:
    model = _toy_model()
    cache = build_kv_cache(model, batch_size=2, max_len=16)
    assert len(cache.layers) == len(model.layers)
    head_dim = model.config.dim // model.config.n_heads
    assert cache.layers[0].k.shape == (2, model.config.n_kv_heads, 16, head_dim)
    assert cache.length == 0
    assert cache.max_len == 16


def test_kv_cache_append_advances_length() -> None:
    layer = LayerKVCache(
        k=torch.zeros(1, 2, 4, 8), v=torch.zeros(1, 2, 4, 8),
    )
    layer.append(torch.ones(1, 2, 1, 8), torch.ones(1, 2, 1, 8) * 2)
    assert layer.length == 1
    assert torch.equal(layer.k[:, :, 0], torch.ones(1, 2, 8))
    assert torch.equal(layer.v[:, :, 0], torch.ones(1, 2, 8) * 2)


def test_kv_cache_overflow_raises() -> None:
    layer = LayerKVCache(
        k=torch.zeros(1, 2, 2, 8), v=torch.zeros(1, 2, 2, 8),
    )
    layer.append(torch.zeros(1, 2, 2, 8), torch.zeros(1, 2, 2, 8))
    with pytest.raises(RuntimeError, match="overflow"):
        layer.append(torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8))


def test_kv_cache_reset() -> None:
    model = _toy_model()
    cache = build_kv_cache(model, batch_size=1, max_len=8)
    cache.layers[0].length = 3
    cache.reset()
    assert cache.length == 0


# ---------------------------------------------------------------------------
# Equivalence between cached and uncached forward
# ---------------------------------------------------------------------------


def test_incremental_forward_matches_full_forward() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = _toy_model(layout)
    model.eval()

    seq_len = 6
    tokens = torch.randint(0, layout.total_size, (1, seq_len))
    axes = _full_axes(seq_len)

    # Full forward gives a reference.
    full = model(tokens, axes=axes)["logits"]

    # Incremental forward, one token at a time.
    cache = build_kv_cache(model, batch_size=1, max_len=seq_len)
    cached_logits = []
    for i in range(seq_len):
        sub_axes = RopeAxes(
            modality=axes.modality[:, i:i + 1],
            linear=axes.linear[:, i:i + 1],
            t=axes.t[:, i:i + 1],
            h=axes.h[:, i:i + 1],
            w=axes.w[:, i:i + 1],
        )
        out = incremental_forward(
            model, tokens=tokens[:, i:i + 1], axes=sub_axes, cache=cache,
        )
        cached_logits.append(out)
    cached = torch.cat(cached_logits, dim=1)

    assert torch.allclose(full, cached, atol=1e-5, rtol=1e-5)


def test_incremental_forward_chunked_matches_full() -> None:
    """Feeding the prompt in one chunk + extra tokens one-by-one must match."""

    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = _toy_model(layout)
    model.eval()

    seq_len = 7
    tokens = torch.randint(0, layout.total_size, (1, seq_len))
    axes = _full_axes(seq_len)
    full = model(tokens, axes=axes)["logits"]

    cache = build_kv_cache(model, batch_size=1, max_len=seq_len)
    chunk = 4
    sub_axes = RopeAxes(
        modality=axes.modality[:, :chunk],
        linear=axes.linear[:, :chunk],
        t=axes.t[:, :chunk],
        h=axes.h[:, :chunk],
        w=axes.w[:, :chunk],
    )
    first = incremental_forward(
        model, tokens=tokens[:, :chunk], axes=sub_axes, cache=cache,
    )
    extra_pieces = [first]
    for i in range(chunk, seq_len):
        sa = RopeAxes(
            modality=axes.modality[:, i:i + 1],
            linear=axes.linear[:, i:i + 1],
            t=axes.t[:, i:i + 1],
            h=axes.h[:, i:i + 1],
            w=axes.w[:, i:i + 1],
        )
        extra_pieces.append(
            incremental_forward(model, tokens=tokens[:, i:i + 1], axes=sa, cache=cache)
        )
    chunked = torch.cat(extra_pieces, dim=1)
    assert torch.allclose(full, chunked, atol=1e-5, rtol=1e-5)


def test_incremental_forward_validates_shapes() -> None:
    model = _toy_model()
    cache = build_kv_cache(model, batch_size=1, max_len=4)
    with pytest.raises(ValueError):
        incremental_forward(
            model, tokens=torch.zeros(4, dtype=torch.long),
            axes=_full_axes(4), cache=cache,
        )


def test_incremental_forward_validates_layers() -> None:
    model = _toy_model()
    cache = build_kv_cache(model, batch_size=1, max_len=4)
    cache.layers = cache.layers[:-1]  # drop one
    with pytest.raises(ValueError):
        incremental_forward(
            model, tokens=torch.zeros(1, 2, dtype=torch.long),
            axes=_full_axes(2), cache=cache,
        )
