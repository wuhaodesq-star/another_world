"""Common metrics over video / latent batches.

Lightweight implementations that do **not** depend on pretrained networks
(those land separately under :mod:`another_world.eval.vbench_wrapper` and
will pull in I3D / CLIP / etc. on demand).

Implemented here:

- :func:`mse` / :func:`psnr` / :func:`mae`            (per-pixel reconstruction)
- :func:`temporal_consistency`                        (frame-to-frame stability)
- :func:`long_horizon_drift`                          (per-step prediction error)
- :func:`token_accuracy` / :func:`token_top_k`       (autoregressive token-level)
- :func:`fvd_score` (statistical FVD approximation using raw pixel
  Frechet distance; the real I3D-based metric ships in stage 5.x).

All functions accept ``torch.Tensor`` inputs and return Python floats /
dicts so they round-trip through W&B and JSONL loggers without surprises.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------


def mse(pred: Tensor, target: Tensor) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    return float((pred.float() - target.float()).pow(2).mean())


def mae(pred: Tensor, target: Tensor) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    return float((pred.float() - target.float()).abs().mean())


def psnr(pred: Tensor, target: Tensor, *, max_value: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB."""

    err = mse(pred, target)
    if err <= 0:
        return float("inf")
    return float(20.0 * math.log10(max_value) - 10.0 * math.log10(err))


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------


def temporal_consistency(video: Tensor) -> float:
    """Mean L2 distance between consecutive frames.

    Lower is more "stable" but also more "static"; useful as one
    sanity check among others. ``video`` must be ``[B, C, T, H, W]``
    or ``[B, T, C, H, W]`` (auto-detected).
    """

    if video.dim() != 5:
        raise ValueError(f"expected 5-D tensor, got {tuple(video.shape)}")
    # canonicalise to [B, T, C, H, W]
    if video.shape[1] <= 4 and video.shape[2] > 4:
        # likely [B, C, T, H, W]
        video = video.transpose(1, 2)
    diff = video[:, 1:] - video[:, :-1]
    return float(diff.float().pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Long-horizon drift
# ---------------------------------------------------------------------------


def long_horizon_drift(
    predictions: Iterable[Tensor],
    targets: Iterable[Tensor],
) -> dict[str, float]:
    """Compute per-step MSE / cosine similarity between predictions and targets.

    ``predictions`` and ``targets`` are equal-length iterables of tensors
    (typically successive autoregressive rollouts at increasing horizons).
    Returns a dict keyed by ``"mse_step_<i>"`` and ``"cos_step_<i>"`` plus
    summary ``"mse_mean"`` / ``"cos_mean"``.
    """

    out: dict[str, float] = {}
    mse_list: list[float] = []
    cos_list: list[float] = []
    for i, (p, t) in enumerate(zip(predictions, targets)):
        err = float((p.float() - t.float()).pow(2).mean())
        pf = p.float().reshape(-1)
        tf = t.float().reshape(-1)
        cos = float(
            torch.nn.functional.cosine_similarity(pf, tf, dim=0)
        )
        out[f"mse_step_{i}"] = err
        out[f"cos_step_{i}"] = cos
        mse_list.append(err)
        cos_list.append(cos)
    if mse_list:
        out["mse_mean"] = sum(mse_list) / len(mse_list)
        out["cos_mean"] = sum(cos_list) / len(cos_list)
    return out


# ---------------------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------------------


def token_accuracy(logits: Tensor, targets: Tensor, *, ignore_index: int = -100) -> float:
    """Top-1 accuracy over predicted tokens."""

    if logits.dim() < 2:
        raise ValueError(f"logits must be at least 2-D, got {tuple(logits.shape)}")
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    correct = (preds[mask] == targets[mask]).float().mean()
    return float(correct)


def token_top_k(
    logits: Tensor, targets: Tensor, *, k: int = 5, ignore_index: int = -100,
) -> float:
    if logits.dim() < 2:
        raise ValueError(f"logits must be at least 2-D, got {tuple(logits.shape)}")
    topk = logits.topk(k=min(k, logits.shape[-1]), dim=-1).indices
    targets_expanded = targets.unsqueeze(-1).expand_as(topk)
    hits = (topk == targets_expanded).any(dim=-1)
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    return float(hits[mask].float().mean())


# ---------------------------------------------------------------------------
# Frechet distance over raw video features
# ---------------------------------------------------------------------------


def _flatten_videos(videos: Tensor) -> Tensor:
    """Flatten ``[N, ...]`` to ``[N, D]``."""

    return videos.float().reshape(videos.shape[0], -1)


def gaussian_frechet_distance(a: Tensor, b: Tensor) -> float:
    """Closed-form Frechet distance between two Gaussian fits.

    ``a`` and ``b`` are ``[N, D]`` feature matrices. Returns
    ``||mu_a - mu_b||^2 + tr(Sigma_a + Sigma_b - 2 (Sigma_a Sigma_b)^0.5)``.
    """

    mu_a = a.mean(dim=0)
    mu_b = b.mean(dim=0)
    cov_a = _cov(a)
    cov_b = _cov(b)
    diff = (mu_a - mu_b).pow(2).sum()
    eigs = _matsqrt_eigs(cov_a @ cov_b)
    return float(diff + cov_a.trace() + cov_b.trace() - 2.0 * eigs.sum())


def _cov(x: Tensor) -> Tensor:
    n = x.shape[0]
    if n < 2:
        return torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype)
    centred = x - x.mean(dim=0, keepdim=True)
    return centred.t() @ centred / (n - 1)


def _matsqrt_eigs(m: Tensor) -> Tensor:
    """Compute eigenvalues of (M)^0.5 via eigendecomposition.

    We clamp tiny negatives that arise from numerical noise.
    """

    eigvals = torch.linalg.eigvals(m).real
    eigvals = eigvals.clamp(min=0.0)
    return eigvals.sqrt()


def fvd_score(real_videos: Tensor, fake_videos: Tensor) -> float:
    """Raw-pixel Frechet Video Distance approximation.

    Not the I3D-based real metric (that requires a pretrained network and
    is implemented separately in :mod:`vbench_wrapper`), but a useful
    smoke metric for unit tests and quick regression spotting.
    """

    if real_videos.shape != fake_videos.shape:
        raise ValueError(
            f"shape mismatch: {tuple(real_videos.shape)} vs "
            f"{tuple(fake_videos.shape)}"
        )
    a = _flatten_videos(real_videos)
    b = _flatten_videos(fake_videos)
    return gaussian_frechet_distance(a, b)


__all__ = [
    "fvd_score",
    "gaussian_frechet_distance",
    "long_horizon_drift",
    "mae",
    "mse",
    "psnr",
    "temporal_consistency",
    "token_accuracy",
    "token_top_k",
]
