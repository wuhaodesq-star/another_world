"""Tests for classifier-free guidance support in the DiT samplers."""

from __future__ import annotations

import torch

from another_world.models.decoder.samplers import (
    dpm_solver_sampler,
    euler_sampler,
)


def _make_constant_pred(value: float):
    def fn(x, t, **_):
        return torch.full_like(x, value)
    return fn


def test_euler_cfg_scale_one_matches_no_cfg() -> None:
    """cfg_scale=1.0 must give identical results to omitting CFG entirely."""

    torch.manual_seed(0)
    initial = torch.randn(1, 4, 2, 4, 4)
    cond = _make_constant_pred(0.5)
    uncond = _make_constant_pred(-1.0)

    a = euler_sampler(
        cond, shape=initial.shape, steps=4, device="cpu",
        initial_noise=initial,
    )
    b = euler_sampler(
        cond, shape=initial.shape, steps=4, device="cpu",
        initial_noise=initial,
        cfg_scale=1.0, cfg_uncond_fn=uncond,
    )
    assert torch.allclose(a, b)


def test_euler_cfg_scale_two_extrapolates() -> None:
    """At cfg_scale=2 the effective velocity is 2*cond - uncond."""

    torch.manual_seed(0)
    initial = torch.zeros(1, 1, 1, 2, 2)
    cond = _make_constant_pred(2.0)
    uncond = _make_constant_pred(0.0)

    cfg_out = euler_sampler(
        cond, shape=initial.shape, steps=10, device="cpu",
        initial_noise=initial,
        cfg_scale=2.0, cfg_uncond_fn=uncond,
    )
    # Effective velocity = uncond + 2 * (cond - uncond) = 0 + 2 * 2 = 4.
    # x_end = x_start + v * (t_end - t_start) = 0 + 4 * (-1) = -4.
    assert torch.allclose(cfg_out, torch.full_like(cfg_out, -4.0), atol=1e-5)


def test_dpm_solver_cfg_runs() -> None:
    """DPM-Solver path should run cleanly with CFG enabled."""

    torch.manual_seed(0)
    cond = _make_constant_pred(0.1)
    uncond = _make_constant_pred(-0.1)
    out = dpm_solver_sampler(
        cond, shape=(1, 4, 2, 4, 4), steps=4, device="cpu",
        cfg_scale=3.0, cfg_uncond_fn=uncond,
    )
    assert out.shape == (1, 4, 2, 4, 4)
    assert torch.isfinite(out).all()


def test_cfg_zero_scale_uses_only_uncond() -> None:
    """cfg_scale=0 collapses the prediction onto the unconditional branch."""

    initial = torch.zeros(1, 1, 1, 2, 2)
    cond = _make_constant_pred(10.0)   # should be ignored
    uncond = _make_constant_pred(1.0)
    out = euler_sampler(
        cond, shape=initial.shape, steps=4, device="cpu",
        initial_noise=initial,
        cfg_scale=0.0, cfg_uncond_fn=uncond,
    )
    # Effective velocity = uncond + 0 * (cond - uncond) = uncond = 1.
    # x_end = 0 + 1 * (-1) = -1.
    assert torch.allclose(out, torch.full_like(out, -1.0), atol=1e-5)


def test_cfg_unused_when_uncond_is_none() -> None:
    """cfg_scale != 1 but no uncond fn => behaves as if no CFG."""

    torch.manual_seed(0)
    initial = torch.randn(1, 2, 1, 2, 2)
    cond = _make_constant_pred(0.5)
    no_cfg = euler_sampler(
        cond, shape=initial.shape, steps=4, device="cpu",
        initial_noise=initial,
    )
    with_cfg = euler_sampler(
        cond, shape=initial.shape, steps=4, device="cpu",
        initial_noise=initial, cfg_scale=5.0, cfg_uncond_fn=None,
    )
    assert torch.allclose(no_cfg, with_cfg)
