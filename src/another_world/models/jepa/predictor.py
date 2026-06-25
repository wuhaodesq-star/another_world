"""V-JEPA-style latent predictor.

V-JEPA (Bardes et al. 2024) trains a video model by predicting *latent
features* of masked regions instead of raw pixels. For our world model
we use this as an **auxiliary loss** on top of the standard next-token
prediction: it pushes the dynamics model to keep semantically-aware
representations even when the token-level signal is noisy.

Sketch
------

::

    target_features  = target_encoder(visual_inputs)   # frozen EMA copy
    student_features = dynamics_model_hidden_states    # the trunk we are training
    pred             = predictor(student_features)
    loss             = smooth_l1(pred, target_features)

The :class:`JEPALatentPredictor` here is the small Transformer that maps
student hidden states to a prediction in the target feature space. The
target encoder is a small frozen MLP (placeholder; production wires this
to an EMA copy of the dynamics model trunk).

This module is intentionally lightweight so it can be unit-tested on CPU
and slotted into either the multimodal trainer (visual tokens) or the
JEPA-pretraining pipeline (frames).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from another_world.models.layers.common import (
    RMSNorm,
    SwiGLU,
    init_weights,
)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


@dataclass
class JEPAConfig:
    in_dim: int = 256
    out_dim: int = 256
    hidden_dim: int = 512
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 2
    dropout: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ffn_mult": self.ffn_mult,
            "dropout": self.dropout,
        }


class _JepaBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True, bias=False,
        )
        self.norm2 = RMSNorm(dim)
        hidden = ((int(dim * ffn_mult * 2 / 3) + 63) // 64) * 64
        self.ffn = SwiGLU(dim, hidden, dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class JEPALatentPredictor(nn.Module):
    """Project student hidden states to predicted target features."""

    def __init__(self, config: JEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.in_dim, config.hidden_dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                _JepaBlock(config.hidden_dim, config.n_heads, config.ffn_mult,
                           config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.out_dim, bias=False)
        self.apply(init_weights)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, hidden_states: Tensor) -> Tensor:
        """``hidden_states`` shape ``[B, T, in_dim]`` -> ``[B, T, out_dim]``."""

        if hidden_states.dim() != 3:
            raise ValueError(
                f"expected [B, T, D] hidden_states, got {tuple(hidden_states.shape)}"
            )
        x = self.input_proj(hidden_states)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def jepa_loss(
    predictor: JEPALatentPredictor,
    *,
    student_hidden: Tensor,
    target_hidden: Tensor,
    mask: Tensor | None = None,
    huber_beta: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the V-JEPA auxiliary loss.

    Args:
        predictor: the trainable predictor head.
        student_hidden: ``[B, T, D]`` features from the trainable trunk.
        target_hidden: ``[B, T, D']`` features from the (frozen) target
            encoder. Predictions are MSE-matched against these.
        mask: optional ``[B, T]`` float in ``{0, 1}`` weighting positions.
        huber_beta: Huber/smooth-L1 transition point.
    """

    if student_hidden.shape[:2] != target_hidden.shape[:2]:
        raise ValueError(
            f"student/target shape mismatch: "
            f"{tuple(student_hidden.shape)} vs {tuple(target_hidden.shape)}"
        )

    pred = predictor(student_hidden)
    diff = pred - target_hidden.detach()
    per_token = torch.nn.functional.smooth_l1_loss(
        pred, target_hidden.detach(), beta=huber_beta, reduction="none",
    ).mean(dim=-1)  # [B, T]

    if mask is not None:
        weight = mask.to(per_token.dtype)
        denom = weight.sum().clamp_min(1.0)
        loss = (per_token * weight).sum() / denom
    else:
        loss = per_token.mean()

    cos = torch.nn.functional.cosine_similarity(
        pred.detach().reshape(-1, pred.shape[-1]),
        target_hidden.detach().reshape(-1, target_hidden.shape[-1]),
        dim=-1,
    ).mean()
    metrics = {
        "jepa_loss": float(loss.detach()),
        "jepa_cos": float(cos),
        "jepa_pred_norm": float(pred.detach().pow(2).mean().sqrt()),
        "jepa_target_norm": float(target_hidden.detach().pow(2).mean().sqrt()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# EMA target encoder helper
# ---------------------------------------------------------------------------


class EmaShadow:
    """Exponential moving average shadow of a module.

    Used to keep a slowly-updated copy of the dynamics trunk that serves
    as the JEPA target encoder.  The shadow lives on the same device as
    the source.
    """

    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.decay = decay
        self._shadow = {
            name: p.detach().clone() for name, p in module.named_parameters()
        }
        self._module_id = id(module)

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        if id(module) != self._module_id:
            raise RuntimeError("EmaShadow.update called with a different module")
        for name, p in module.named_parameters():
            self._shadow[name].mul_(self.decay).add_(
                p.detach(), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def copy_to(self, target: nn.Module) -> None:
        """Load the shadow weights into ``target`` (same parameter shape)."""

        target_params = dict(target.named_parameters())
        for name, p in self._shadow.items():
            if name in target_params:
                target_params[name].data.copy_(p)


__all__ = [
    "EmaShadow",
    "JEPAConfig",
    "JEPALatentPredictor",
    "jepa_loss",
]
