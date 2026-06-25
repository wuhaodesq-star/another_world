"""Tests for diffusion / flow training objectives."""

from __future__ import annotations

import pytest
import torch

from another_world.models.decoder.diffusion import (
    DiffusionObjectiveConfig,
    compute_diffusion_loss,
    cosine_alpha_bar,
    cosine_alpha_sigma,
    rectified_flow_loss,
    sample_timesteps,
    v_prediction_loss,
)


def _identity_model(x: torch.Tensor, _t: torch.Tensor, **_kwargs) -> torch.Tensor:
    return torch.zeros_like(x)


def test_sample_timesteps_logit_normal_in_range() -> None:
    cfg = DiffusionObjectiveConfig(timestep_dist="logit_normal")
    t = sample_timesteps(64, cfg, device=torch.device("cpu"))
    assert t.shape == (64,)
    assert (t > 0).all() and (t < 1).all()


def test_sample_timesteps_uniform_in_range() -> None:
    cfg = DiffusionObjectiveConfig(timestep_dist="uniform")
    t = sample_timesteps(64, cfg, device=torch.device("cpu"))
    assert t.shape == (64,)
    assert (t > 0).all() and (t < 1).all()


def test_sample_timesteps_unknown_dist_raises() -> None:
    cfg = DiffusionObjectiveConfig(timestep_dist="bogus")
    with pytest.raises(ValueError):
        sample_timesteps(4, cfg, device=torch.device("cpu"))


def test_cosine_schedule_endpoints() -> None:
    alpha0, sigma0 = cosine_alpha_sigma(torch.tensor([0.0]))
    alpha1, sigma1 = cosine_alpha_sigma(torch.tensor([1.0]))
    assert alpha0.item() > 0.95
    assert sigma0.item() < 0.3
    assert alpha1.item() < 0.05
    assert sigma1.item() > 0.95


def test_cosine_alpha_bar_monotone() -> None:
    t = torch.linspace(0.0, 1.0, 16)
    ab = cosine_alpha_bar(t)
    diffs = ab[1:] - ab[:-1]
    assert (diffs <= 0).all()


def test_rectified_flow_loss_finite_and_grad_flows() -> None:
    torch.manual_seed(0)
    cfg = DiffusionObjectiveConfig(objective="rectified_flow")
    x0 = torch.randn(2, 4, 2, 8, 8)

    weight = torch.nn.Parameter(torch.zeros(1))

    def model_fn(x, t, **_):
        return x * weight  # learnable predictor

    loss, m = rectified_flow_loss(model_fn, x0=x0, config=cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert weight.grad is not None and weight.grad.abs().sum().item() > 0
    assert "t_mean" in m and 0 < m["t_mean"] < 1


def test_v_prediction_loss_finite() -> None:
    cfg = DiffusionObjectiveConfig(objective="v_prediction")
    x0 = torch.randn(2, 4, 2, 8, 8)
    loss, _ = v_prediction_loss(_identity_model, x0=x0, config=cfg)
    assert torch.isfinite(loss)
    assert float(loss) > 0  # zero predictions != target


def test_compute_loss_dispatch() -> None:
    x0 = torch.randn(1, 4, 2, 8, 8)
    cfg_rf = DiffusionObjectiveConfig(objective="rectified_flow")
    cfg_vp = DiffusionObjectiveConfig(objective="v_prediction")
    loss_rf, _ = compute_diffusion_loss(_identity_model, x0=x0, config=cfg_rf)
    loss_vp, _ = compute_diffusion_loss(_identity_model, x0=x0, config=cfg_vp)
    assert torch.isfinite(loss_rf)
    assert torch.isfinite(loss_vp)


def test_compute_loss_unknown_objective_raises() -> None:
    cfg = DiffusionObjectiveConfig(objective="bogus")
    with pytest.raises(ValueError):
        compute_diffusion_loss(_identity_model, x0=torch.randn(1, 4, 2, 4, 4), config=cfg)


def test_rectified_flow_loss_perfect_pred_is_zero() -> None:
    torch.manual_seed(42)
    cfg = DiffusionObjectiveConfig(objective="rectified_flow")
    x0 = torch.randn(2, 4, 2, 4, 4)

    # Pre-compute the target the loss function will see by replicating its
    # internal RNG: just use the model that perfectly returns ``noise - x0``.
    # We pass the noise via closure.
    captured = {}

    def passthrough(x, t, **_):
        captured["x"] = x
        # Recover noise from x_t = (1 - t_b) * x0 + t_b * noise:
        t_b = t.float()[:, None, None, None, None] / cfg.num_train_steps
        noise = (x - (1.0 - t_b) * x0) / t_b.clamp_min(1e-6)
        return noise - x0

    loss, _ = rectified_flow_loss(passthrough, x0=x0, config=cfg)
    assert float(loss) < 1e-3
