"""Tests for diffusion samplers."""

from __future__ import annotations

import pytest
import torch

from another_world.models.decoder.samplers import dpm_solver_sampler, euler_sampler


def test_euler_sampler_zero_velocity_returns_initial_noise() -> None:
    torch.manual_seed(0)
    initial = torch.randn(2, 4, 2, 4, 4)

    def zero_velocity(x, t, **_):
        return torch.zeros_like(x)

    out = euler_sampler(
        zero_velocity, shape=initial.shape, steps=5, device="cpu",
        initial_noise=initial,
    )
    # dx/dt = 0 means x is unchanged.
    assert torch.allclose(out, initial)


def test_euler_sampler_constant_negative_velocity_moves_towards_target() -> None:
    """If v == constant, Euler integration produces x_end = x_start + v * (t_end - t_start)."""

    initial = torch.zeros(1, 1, 1, 2, 2)
    target_v = -torch.ones_like(initial)

    def constant_v(x, t, **_):
        return target_v

    out = euler_sampler(
        constant_v, shape=initial.shape, steps=10, device="cpu",
        initial_noise=initial,
    )
    # t goes from 1 to 0; integral of v dt over that range = v * (-1).
    # So out = 0 + (-1) * (-1) = 1.
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def test_euler_sampler_default_initial_noise_random() -> None:
    torch.manual_seed(0)

    def zero(x, t, **_):
        return torch.zeros_like(x)

    a = euler_sampler(zero, shape=(2, 3), steps=2, device="cpu")
    b = euler_sampler(zero, shape=(2, 3), steps=2, device="cpu")
    assert not torch.equal(a, b)  # independent draws


def test_dpm_solver_sampler_runs_and_finite() -> None:
    torch.manual_seed(0)

    def zero_v(x, t, **_):
        return torch.zeros_like(x)

    out = dpm_solver_sampler(zero_v, shape=(1, 4, 2, 4, 4), steps=5, device="cpu")
    assert out.shape == (1, 4, 2, 4, 4)
    assert torch.isfinite(out).all()


def test_dpm_solver_sampler_endpoint_at_t_zero_is_approx_x0_hat() -> None:
    """A pure-zero v means x0_hat = alpha * x.

    Walking the schedule to t=0 should yield a sample whose magnitude is
    bounded by the original noise magnitude (no blow-up).
    """

    torch.manual_seed(0)
    initial = torch.randn(1, 4, 2, 4, 4)

    def zero_v(x, t, **_):
        return torch.zeros_like(x)

    out = dpm_solver_sampler(
        zero_v, shape=initial.shape, steps=20, device="cpu",
        initial_noise=initial,
    )
    assert out.abs().max() <= initial.abs().max() * 2.0


def test_euler_with_decoder_runs_end_to_end() -> None:
    """Plug a real (tiny) DiT into the sampler."""

    from another_world.models.decoder import DiTDecoder, DiTDecoderConfig

    cfg = DiTDecoderConfig.toy(vocab_size=32)
    model = DiTDecoder(cfg)
    model.eval()

    token_ids = torch.randint(0, 32, (1, 4))

    def fn(x, t, **_):
        return model(x, t, token_ids=token_ids)

    out = euler_sampler(
        fn,
        shape=(1, cfg.in_channels, 2, 8, 8),
        steps=4,
        device="cpu",
    )
    assert out.shape == (1, cfg.in_channels, 2, 8, 8)
    assert torch.isfinite(out).all()
