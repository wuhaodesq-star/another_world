"""KV cache for fast autoregressive decoding.

Without a cache, every new token costs ``O(T^2)`` because we recompute
keys and values for every position in the prefix. With a per-layer cache
the same step takes ``O(T)``.

Design
------

- :class:`KVCache` holds a pre-allocated ``[B, n_kv_heads, max_len, head_dim]``
  buffer for K and V per layer, plus an integer ``length`` counter.
- The dynamics model exposes an ``incremental_forward(tokens, axes, cache)``
  method that runs the model on a *small* chunk (e.g. one new token) and
  appends the resulting K/V tensors to the cache.
- We do not modify the existing :meth:`forward` (used for training); KV
  cache is opt-in for generation only.

This is a CPU/GPU-portable pure-PyTorch implementation. Production runs
on H100 will eventually swap the underlying attention call for
FlashAttention's `varlen_kvpacked` API; the cache structure is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn

from another_world.models.dynamics.multimodal import (
    MultimodalAttention,
    MultimodalBlock,
    MultimodalDynamicsModel,
)
from another_world.models.layers.common import apply_rope
from another_world.models.layers.mixed_rope import RopeAxes


# ---------------------------------------------------------------------------
# Cache containers
# ---------------------------------------------------------------------------


@dataclass
class LayerKVCache:
    """One transformer layer's K/V cache."""

    k: Tensor  # [B, n_kv_heads, max_len, head_dim]
    v: Tensor
    length: int = 0  # number of valid positions

    @property
    def max_len(self) -> int:
        return int(self.k.shape[2])

    def append(self, new_k: Tensor, new_v: Tensor) -> None:
        bsz, _, t_new, _ = new_k.shape
        if self.length + t_new > self.max_len:
            raise RuntimeError(
                f"KV cache overflow: have {self.length}, adding {t_new}, "
                f"max {self.max_len}"
            )
        self.k[:bsz, :, self.length : self.length + t_new] = new_k
        self.v[:bsz, :, self.length : self.length + t_new] = new_v
        self.length += t_new

    def reset(self) -> None:
        self.length = 0


@dataclass
class KVCache:
    """Per-layer KV caches plus housekeeping."""

    layers: List[LayerKVCache]

    @property
    def length(self) -> int:
        return self.layers[0].length if self.layers else 0

    @property
    def max_len(self) -> int:
        return self.layers[0].max_len if self.layers else 0

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()


def build_kv_cache(
    model: MultimodalDynamicsModel,
    *,
    batch_size: int,
    max_len: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> KVCache:
    """Pre-allocate a :class:`KVCache` matching the model's geometry."""

    n_layers = len(model.layers)
    head_dim = model.config.dim // model.config.n_heads
    n_kv_heads = model.config.n_kv_heads
    device = device or next(model.parameters()).device
    layers: list[LayerKVCache] = []
    for _ in range(n_layers):
        k = torch.zeros(
            batch_size, n_kv_heads, max_len, head_dim,
            device=device, dtype=dtype,
        )
        v = torch.zeros_like(k)
        layers.append(LayerKVCache(k=k, v=v))
    return KVCache(layers=layers)


# ---------------------------------------------------------------------------
# Incremental attention
# ---------------------------------------------------------------------------


def _attention_with_cache(
    attn: MultimodalAttention,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    cache: LayerKVCache,
) -> Tensor:
    """Run attention for a new chunk and update the cache in-place."""

    bsz, seq_len_new, _ = x.shape
    q = attn.wq(x).view(bsz, seq_len_new, attn.n_heads, attn.head_dim).transpose(1, 2)
    k_new = attn.wk(x).view(bsz, seq_len_new, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    v_new = attn.wv(x).view(bsz, seq_len_new, attn.n_kv_heads, attn.head_dim).transpose(1, 2)

    q = apply_rope(q, cos, sin)
    k_new = apply_rope(k_new, cos, sin)

    # Append after rotary so cached keys are already in the rotated basis.
    cache.append(k_new, v_new)
    total_len = cache.length

    k_all = cache.k[:bsz, :, :total_len]
    v_all = cache.v[:bsz, :, :total_len]

    if attn.n_kv_heads != attn.n_heads:
        repeats = attn.n_heads // attn.n_kv_heads
        k_all = k_all.repeat_interleave(repeats, dim=1)
        v_all = v_all.repeat_interleave(repeats, dim=1)

    # Causal mask: each new query may attend to all cached positions up to
    # its own. With seq_len_new == 1 (the common case) this is a no-op.
    is_causal = seq_len_new > 1
    out = nn.functional.scaled_dot_product_attention(
        q, k_all, v_all,
        dropout_p=0.0,
        is_causal=is_causal and total_len == seq_len_new,
    )
    out = out.transpose(1, 2).contiguous().view(bsz, seq_len_new, -1)
    return attn.wo(out)


def _block_with_cache(
    block: MultimodalBlock,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    cache: LayerKVCache,
) -> Tensor:
    h = block.attn_norm(x)
    x = x + _attention_with_cache(block.attn, h, cos, sin, cache)
    x = x + block.ffn(block.ffn_norm(x))
    return x


@torch.no_grad()
def incremental_forward(
    model: MultimodalDynamicsModel,
    *,
    tokens: Tensor,
    axes: RopeAxes,
    cache: KVCache,
) -> Tensor:
    """Run the model on ``tokens`` using ``cache`` for previous positions.

    Returns the logits for *only* the new positions (shape
    ``[B, seq_len_new, vocab_size]``).
    """

    if tokens.dim() != 2:
        raise ValueError(f"expected [B, T] tokens, got {tuple(tokens.shape)}")
    if axes.modality.shape != tokens.shape:
        raise ValueError(
            f"axes.modality {tuple(axes.modality.shape)} must equal "
            f"tokens {tuple(tokens.shape)}"
        )
    if len(cache.layers) != len(model.layers):
        raise ValueError(
            f"cache has {len(cache.layers)} layers but model has "
            f"{len(model.layers)}"
        )

    h = model.tok_embeddings(tokens)
    if model.modality_embeddings is not None:
        h = h + model.modality_embeddings(axes.modality)

    cos, sin = model.rope.build(axes, device=h.device, dtype=h.dtype)

    for block, layer_cache in zip(model.layers, cache.layers):
        h = _block_with_cache(block, h, cos, sin, layer_cache)

    h = model.norm(h)
    return model.output(h)


__all__ = [
    "KVCache",
    "LayerKVCache",
    "build_kv_cache",
    "incremental_forward",
]
