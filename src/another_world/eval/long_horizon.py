"""Long-horizon prediction evaluation.

Walks a dynamics model forward in fixed-stride chunks and records the
per-step token-prediction accuracy and latent-cosine drift. This is the
primary diagnostic for the "does the model forget over long horizons?"
question (one of the stage-5 evaluation dimensions).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from another_world.eval.metrics import token_accuracy, token_top_k
from another_world.models.layers.mixed_rope import RopeAxes


@dataclass
class HorizonResult:
    horizons: list[int]
    accuracy: list[float]
    top5: list[float]

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for h, a, k in zip(self.horizons, self.accuracy, self.top5):
            out[f"acc_h{h}"] = a
            out[f"top5_h{h}"] = k
        if self.accuracy:
            out["acc_mean"] = sum(self.accuracy) / len(self.accuracy)
        return out


def _slice_axes(axes: RopeAxes, end: int) -> RopeAxes:
    return RopeAxes(
        modality=axes.modality[:, :end],
        linear=axes.linear[:, :end],
        t=axes.t[:, :end],
        h=axes.h[:, :end],
        w=axes.w[:, :end],
    )


@torch.no_grad()
def evaluate_long_horizon(
    model,
    *,
    tokens: Tensor,
    axes: RopeAxes,
    targets: Tensor,
    horizons: Iterable[int],
) -> HorizonResult:
    """Compute token accuracy at the requested horizon lengths.

    ``tokens`` is ``[B, T]``; for each horizon ``h`` we feed only the
    first ``h`` tokens to the model and score its prediction at position
    ``h`` against ``targets[:, h-1]`` (the next-token convention).
    """

    horizons = sorted({int(h) for h in horizons})
    acc_vals: list[float] = []
    top5_vals: list[float] = []
    model_was_training = model.training
    model.eval()
    try:
        for h in horizons:
            if h < 1 or h > tokens.shape[1]:
                raise ValueError(
                    f"horizon {h} out of range [1, {tokens.shape[1]}]"
                )
            sub_tokens = tokens[:, :h]
            sub_axes = _slice_axes(axes, h)
            out = model(sub_tokens, axes=sub_axes)
            last_logits = out["logits"][:, -1, :]
            target_at_h = targets[:, h - 1]
            acc_vals.append(token_accuracy(last_logits, target_at_h))
            top5_vals.append(token_top_k(last_logits, target_at_h, k=5))
    finally:
        if model_was_training:
            model.train()

    return HorizonResult(
        horizons=horizons, accuracy=acc_vals, top5=top5_vals,
    )


__all__ = ["HorizonResult", "evaluate_long_horizon"]
