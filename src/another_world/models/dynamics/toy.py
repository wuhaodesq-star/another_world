"""Toy 350M decoder-only Transformer.

This module exists to provide a runnable end-to-end smoke test for the
training stack (data loader -> model -> loss -> optimizer step) at a scale
that fits on a single consumer GPU or even CPU.

The full-scale dynamics models (7B, 30B, 70B) will live alongside this file
once they are implemented in stage 3 of the roadmap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from another_world.models.layers.common import (
    RMSNorm,
    TransformerBlock,
    build_rope_cache,
    count_parameters,
    init_weights,
)


@dataclass
class ToyTransformerConfig:
    """Configuration for :class:`ToyTransformer`.

    Defaults are tuned to ~350M parameters with ``vocab_size=1024``; in
    production we override these via Hydra configs in ``configs/model/``.
    """

    vocab_size: int = 1024
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 16
    ffn_mult: int = 4
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ToyTransformer(nn.Module):
    """A LLaMA-style decoder-only Transformer used for stage-0 smoke tests."""

    def __init__(self, config: ToyTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
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
        cos, sin = build_rope_cache(
            seq_len=config.max_seq_len,
            head_dim=head_dim,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(init_weights)

    @property
    def num_parameters(self) -> int:
        return count_parameters(self)

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run a forward pass.

        Args:
            tokens: ``[batch, seq_len]`` long tensor of input token ids.
            targets: optional ``[batch, seq_len]`` long tensor. When given,
                the cross-entropy loss is computed against the shifted
                ``tokens`` (next-token prediction is the convention).

        Returns:
            Dict with ``logits`` and optionally ``loss``.
        """

        if tokens.dim() != 2:
            raise ValueError(f"expected [B, T] tokens, got shape {tuple(tokens.shape)}")
        bsz, seq_len = tokens.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds model max_seq_len "
                f"{self.config.max_seq_len}"
            )

        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)
        h = self.norm(h)
        logits = self.output(h)

        out: dict[str, Tensor] = {"logits": logits}
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} must equal "
                    f"tokens shape {tuple(tokens.shape)}"
                )
            loss = nn.functional.cross_entropy(
                logits.reshape(bsz * seq_len, -1),
                targets.reshape(bsz * seq_len),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out


def build_toy_transformer(config: ToyTransformerConfig | None = None) -> ToyTransformer:
    return ToyTransformer(config or ToyTransformerConfig())


def _smoke_forward() -> None:
    """Tiny end-to-end check used when running ``python -m ... toy``."""

    torch.manual_seed(0)
    cfg = ToyTransformerConfig(
        vocab_size=128,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2,
        max_seq_len=32,
    )
    model = build_toy_transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(tokens, targets=targets)
    print(
        f"[toy] params={model.num_parameters:,} "
        f"logits={tuple(out['logits'].shape)} "
        f"loss={out['loss'].item():.4f}"
    )


if __name__ == "__main__":  # pragma: no cover
    _smoke_forward()
