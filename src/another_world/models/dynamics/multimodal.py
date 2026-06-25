"""Multimodal dynamics model.

A decoder-only Transformer that operates on a single packed token stream
mixing text, visual, and action ids. The visual tokens carry 3-D positional
information via :class:`~another_world.models.layers.MixedRoPE` while the
attention kernel itself remains modality-agnostic.

This is the production model skeleton intended to scale from ~350M to 7B/30B
parameters by selecting a different :class:`MultimodalDynamicsConfig`. It is
written to run on CPU for unit tests; production training plugs into
TorchTitan / FSDP2 elsewhere.

High-level architecture
-----------------------

::

    tokens [B, T] (long, ids in [0, vocab_size))
      ->  token + modality embedding         [B, T, dim]
      ->  N x ( pre-norm -> attn(MixedRoPE) -> residual ->
                 pre-norm -> SwiGLU         -> residual )
      ->  RMSNorm
      ->  linear (tied to embedding)         [B, T, vocab_size]

Loss masking is delegated to the caller: pass ``targets`` with ``-100``
for positions that should be ignored (e.g. text predicting visual, or
padding).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn

from another_world.models.layers.common import (
    RMSNorm,
    SwiGLU,
    apply_rope,
    count_parameters,
    init_weights,
)
from another_world.models.layers.mixed_rope import MixedRoPE, RopeAxes
from another_world.tokenizers.vocab import VocabInfo, VocabLayout


# ---------------------------------------------------------------------------
# Attention with per-sample mixed RoPE buffers
# ---------------------------------------------------------------------------


class MultimodalAttention(nn.Module):
    """Causal self-attention with grouped-query attention + mixed RoPE.

    Unlike :class:`another_world.models.layers.CausalSelfAttention`, this
    variant accepts ``(cos, sin)`` of shape ``[B, T, head_dim // 2]`` so
    each token in the batch can have its own RoPE axes.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        n_kv_heads = n_kv_heads or n_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"
            )

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.dropout_p = dropout

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)


class MultimodalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = _swiglu_hidden(dim, ffn_mult)
        self.attn_norm = RMSNorm(dim)
        self.attn = MultimodalAttention(dim, n_heads, n_kv_heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden, dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def _swiglu_hidden(dim: int, ffn_mult: int) -> int:
    raw = int(dim * ffn_mult * 2 / 3)
    return ((raw + 255) // 256) * 256


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MultimodalDynamicsConfig:
    """Shape parameters for the multimodal dynamics Transformer.

    The "preset" classmethods produce LLaMA-style configurations at
    different parameter scales.
    """

    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_mult: int = 4
    dropout: float = 0.0
    max_linear: int = 8_192
    max_t: int = 64
    max_h: int = 64
    max_w: int = 64
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True
    use_modality_embedding: bool = True

    # ----- presets --------------------------------------------------------

    @classmethod
    def toy(cls, vocab_size: int) -> "MultimodalDynamicsConfig":
        """~5M params, used by CPU unit tests."""

        return cls(
            vocab_size=vocab_size,
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            ffn_mult=2,
            max_linear=512,
            max_t=16,
            max_h=16,
            max_w=16,
        )

    @classmethod
    def m350(cls, vocab_size: int) -> "MultimodalDynamicsConfig":
        """~350M params, single-GPU debug."""

        return cls(
            vocab_size=vocab_size,
            dim=1024,
            n_layers=24,
            n_heads=16,
            n_kv_heads=8,
        )

    @classmethod
    def b1(cls, vocab_size: int) -> "MultimodalDynamicsConfig":
        """~1B params."""

        return cls(
            vocab_size=vocab_size,
            dim=2048,
            n_layers=24,
            n_heads=16,
            n_kv_heads=8,
        )

    @classmethod
    def b3(cls, vocab_size: int) -> "MultimodalDynamicsConfig":
        """~3B params."""

        return cls(
            vocab_size=vocab_size,
            dim=2560,
            n_layers=32,
            n_heads=20,
            n_kv_heads=4,
        )

    @classmethod
    def b7(cls, vocab_size: int) -> "MultimodalDynamicsConfig":
        """~7B params (MVP target)."""

        return cls(
            vocab_size=vocab_size,
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MultimodalDynamicsModel(nn.Module):
    """Decoder-only multimodal Transformer."""

    def __init__(self, config: MultimodalDynamicsConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        if config.use_modality_embedding:
            self.modality_embeddings = nn.Embedding(4, config.dim)
        else:
            self.modality_embeddings = None  # type: ignore[assignment]

        self.layers = nn.ModuleList(
            [
                MultimodalBlock(
                    dim=config.dim,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    ffn_mult=config.ffn_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight

        head_dim = config.dim // config.n_heads
        self.rope = MixedRoPE(
            head_dim=head_dim,
            max_linear=config.max_linear,
            max_t=config.max_t,
            max_h=config.max_h,
            max_w=config.max_w,
            theta=config.rope_theta,
        )

        self.apply(init_weights)

    # ----- helpers --------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        return count_parameters(self)

    # ----- forward --------------------------------------------------------

    def forward(
        self,
        tokens: Tensor,
        axes: RopeAxes,
        targets: Tensor | None = None,
        loss_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run a forward pass.

        Args:
            tokens: ``[B, T]`` long.
            axes: per-token coordinate tensors (see
                :class:`~another_world.models.layers.RopeAxes`). Each field
                has shape ``[B, T]``.
            targets: optional ``[B, T]`` long. Use ``-100`` to mask out
                positions you don't want to score.
            loss_mask: optional ``[B, T]`` float mask multiplied into the
                per-position cross entropy before averaging.
        """

        if tokens.dim() != 2:
            raise ValueError(f"expected [B, T] tokens, got {tuple(tokens.shape)}")
        if axes.modality.shape != tokens.shape:
            raise ValueError(
                f"axes.modality shape {tuple(axes.modality.shape)} must equal "
                f"tokens shape {tuple(tokens.shape)}"
            )

        # Embeddings.
        h = self.tok_embeddings(tokens)
        if self.modality_embeddings is not None:
            h = h + self.modality_embeddings(axes.modality)

        cos, sin = self.rope.build(axes, device=h.device, dtype=h.dtype)

        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm(h)
        logits = self.output(h)

        out: dict[str, Tensor] = {"logits": logits}
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} must equal "
                    f"tokens shape {tuple(tokens.shape)}"
                )
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            if loss_mask is not None:
                per_token = nn.functional.cross_entropy(
                    flat_logits, flat_targets, ignore_index=-100, reduction="none",
                )
                mask = loss_mask.reshape(-1).to(per_token.dtype)
                denom = mask.sum().clamp_min(1.0)
                loss = (per_token * mask).sum() / denom
            else:
                loss = nn.functional.cross_entropy(
                    flat_logits, flat_targets, ignore_index=-100,
                )
            out["loss"] = loss
        return out


def build_multimodal_model(
    config: MultimodalDynamicsConfig | None = None,
    *,
    vocab: VocabLayout | None = None,
) -> MultimodalDynamicsModel:
    """Convenience factory used by tests and the CLI."""

    if config is None:
        if vocab is None:
            vocab = VocabLayout.tiny()
        config = MultimodalDynamicsConfig.toy(vocab.total_size)
    return MultimodalDynamicsModel(config)


__all__ = [
    "MultimodalAttention",
    "MultimodalBlock",
    "MultimodalDynamicsConfig",
    "MultimodalDynamicsModel",
    "build_multimodal_model",
]
