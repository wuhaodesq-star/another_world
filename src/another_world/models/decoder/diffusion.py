"""Diffusion / flow training objectives for the DiT decoder.

Implements two losses:

- **Rectified flow** (preferred for SOTA video DiTs, see Esser et al.
  "Scaling Rectified Flow Transformers"). The network predicts the
  velocity ``v = x1 - x0`` given a noisy interpolation
  ``x_t = (1 - t) * x0 + t * x1``, with ``t ~ logit-normal(0, 1)`` by
  default to bias sampling towards the noisier end. ``x0`` is the
  clean target latent, ``x1`` is pure Gaussian noise.

- **DDPM v-prediction** (compatibility option). The network predicts
  ``v = alpha * eps - sigma * x0`` for a cosine-noise schedule, which
  is the parameterization used by Stable Diffusion-XL and many video
  papers.

Both losses are mean-squared error in latent space. The choice between
them is exposed through :class:`DiffusionObjectiveConfig`.

Sampling is in ``samplers.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DiffusionObjectiveConfig:
    """Hyperparameters for diffusion / flow training."""

    objective: str = "rectified_flow"  # "rectified_flow" | "v_prediction"
    timestep_dist: str = "logit_normal"  # "logit_normal" | "uniform"
    logit_normal_mu: float = 0.0
    logit_normal_sigma: float = 1.0
    schedule: str = "cosine"  # only used for v_prediction
    schedule_s: float = 0.008
    num_train_steps: int = 1000  # for v_prediction discrete schedule


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def cosine_alpha_bar(t: Tensor, s: float = 0.008) -> Tensor:
    """Improved cosine alpha_bar(t) from Nichol & Dhariwal."""

    return torch.cos(((t + s) / (1 + s)) * math.pi / 2).clamp(min=1e-6) ** 2


def cosine_alpha_sigma(t: Tensor, s: float = 0.008) -> tuple[Tensor, Tensor]:
    """Return ``(alpha, sigma)`` for the cosine schedule at fractional ``t``."""

    alpha_bar = cosine_alpha_bar(t, s)
    alpha = torch.sqrt(alpha_bar)
    sigma = torch.sqrt(1.0 - alpha_bar)
    return alpha, sigma


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------


def sample_timesteps(
    batch: int,
    config: DiffusionObjectiveConfig,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw ``[batch]`` floats in ``(0, 1)`` according to the configured prior."""

    if config.timestep_dist == "uniform":
        return torch.rand(batch, device=device, generator=generator).clamp(1e-4, 1 - 1e-4)
    if config.timestep_dist == "logit_normal":
        noise = torch.randn(batch, device=device, generator=generator)
        u = config.logit_normal_mu + config.logit_normal_sigma * noise
        return torch.sigmoid(u).clamp(1e-4, 1 - 1e-4)
    raise ValueError(f"unknown timestep_dist '{config.timestep_dist}'")


def _broadcast(t: Tensor, x: Tensor) -> Tensor:
    """Broadcast ``[B]`` scalars to match ``[B, C, T, H, W]`` etc."""

    return t.view(t.shape[0], *([1] * (x.dim() - 1)))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def rectified_flow_loss(
    model_fn,
    *,
    x0: Tensor,
    config: DiffusionObjectiveConfig,
    generator: torch.Generator | None = None,
    **model_kwargs,
) -> tuple[Tensor, dict]:
    """Compute the rectified-flow loss on a clean batch ``x0``.

    ``model_fn`` must accept ``(latents, timesteps, **model_kwargs)`` and
    return a velocity prediction of the same shape as ``latents``.
    """

    device = x0.device
    bsz = x0.shape[0]
    t = sample_timesteps(bsz, config, device=device, generator=generator)
    t_b = _broadcast(t, x0)
    noise = torch.randn(x0.shape, device=device, generator=generator, dtype=x0.dtype)

    x_t = (1.0 - t_b) * x0 + t_b * noise
    # Velocity target is constant in time for a straight path.
    target = noise - x0

    timesteps_for_model = (t * config.num_train_steps).to(torch.long)
    pred = model_fn(x_t, timesteps_for_model, **model_kwargs)
    loss = (pred - target).pow(2).mean()

    metrics = {
        "t_mean": float(t.mean()),
        "target_norm": float(target.detach().pow(2).mean().sqrt()),
        "pred_norm": float(pred.detach().pow(2).mean().sqrt()),
    }
    return loss, metrics


def v_prediction_loss(
    model_fn,
    *,
    x0: Tensor,
    config: DiffusionObjectiveConfig,
    generator: torch.Generator | None = None,
    **model_kwargs,
) -> tuple[Tensor, dict]:
    """Compute the v-prediction loss with a cosine schedule."""

    device = x0.device
    bsz = x0.shape[0]
    t = sample_timesteps(bsz, config, device=device, generator=generator)
    alpha, sigma = cosine_alpha_sigma(t, config.schedule_s)
    alpha_b = _broadcast(alpha, x0)
    sigma_b = _broadcast(sigma, x0)
    eps = torch.randn(x0.shape, device=device, generator=generator, dtype=x0.dtype)

    x_t = alpha_b * x0 + sigma_b * eps
    target = alpha_b * eps - sigma_b * x0

    timesteps_for_model = (t * config.num_train_steps).to(torch.long)
    pred = model_fn(x_t, timesteps_for_model, **model_kwargs)
    loss = (pred - target).pow(2).mean()

    metrics = {
        "t_mean": float(t.mean()),
        "target_norm": float(target.detach().pow(2).mean().sqrt()),
        "pred_norm": float(pred.detach().pow(2).mean().sqrt()),
    }
    return loss, metrics


def compute_diffusion_loss(
    model_fn,
    *,
    x0: Tensor,
    config: DiffusionObjectiveConfig,
    generator: torch.Generator | None = None,
    **model_kwargs,
) -> tuple[Tensor, dict]:
    """Dispatch to the configured loss."""

    if config.objective == "rectified_flow":
        return rectified_flow_loss(
            model_fn, x0=x0, config=config, generator=generator, **model_kwargs,
        )
    if config.objective == "v_prediction":
        return v_prediction_loss(
            model_fn, x0=x0, config=config, generator=generator, **model_kwargs,
        )
    raise ValueError(f"unknown objective '{config.objective}'")


__all__ = [
    "DiffusionObjectiveConfig",
    "compute_diffusion_loss",
    "cosine_alpha_bar",
    "cosine_alpha_sigma",
    "rectified_flow_loss",
    "sample_timesteps",
    "v_prediction_loss",
]
