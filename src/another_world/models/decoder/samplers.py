"""Inference samplers for the DiT decoder.

Two samplers are implemented:

- :func:`euler_sampler` for rectified-flow models (ODE Euler integration
  of ``dx/dt = v_theta(x_t, t)`` from ``t=1`` (noise) to ``t=0`` (clean)).
- :func:`dpm_solver_sampler` (single-step DPM-Solver) for v-prediction
  models on a cosine schedule.

Both samplers share a uniform Python protocol::

    x = sampler(model_fn, shape=..., steps=50, ...)

where ``model_fn(x, t, **kwargs)`` returns the network prediction.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from another_world.models.decoder.diffusion import cosine_alpha_sigma


ModelFn = Callable[..., Tensor]


# ---------------------------------------------------------------------------
# Rectified flow / Euler
# ---------------------------------------------------------------------------


def euler_sampler(
    model_fn: ModelFn,
    *,
    shape: tuple[int, ...],
    steps: int = 50,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    num_train_steps: int = 1000,
    generator: torch.Generator | None = None,
    initial_noise: Tensor | None = None,
    **model_kwargs,
) -> Tensor:
    """Euler ODE sampler for rectified-flow models.

    Integrates from ``t=1`` (Gaussian noise) backwards to ``t=0`` (clean
    sample) using ``steps`` uniform steps.
    """

    device = torch.device(device)
    if initial_noise is None:
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    else:
        x = initial_noise.to(device=device, dtype=dtype)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    for i in range(steps):
        t_now = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_now  # negative

        t_batch = t_now.expand(x.shape[0])
        timesteps_for_model = (t_batch * num_train_steps).to(torch.long).clamp(
            max=num_train_steps - 1
        )
        v = model_fn(x, timesteps_for_model, **model_kwargs)
        # x_{t+dt} = x_t + v * dt;  with dt<0 this walks toward t=0.
        x = x + v * dt

    return x


# ---------------------------------------------------------------------------
# DPM-Solver (1st order) for v-prediction
# ---------------------------------------------------------------------------


def dpm_solver_sampler(
    model_fn: ModelFn,
    *,
    shape: tuple[int, ...],
    steps: int = 30,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    schedule_s: float = 0.008,
    num_train_steps: int = 1000,
    generator: torch.Generator | None = None,
    initial_noise: Tensor | None = None,
    **model_kwargs,
) -> Tensor:
    """First-order DPM-Solver (equivalent to DDIM with v-pred reparam).

    Walks the cosine schedule from ``t=1`` to ``t=0`` in ``steps`` jumps.
    For each segment we estimate the predicted clean sample ``x0_hat``
    from the network output then re-noise to the next timestep.
    """

    device = torch.device(device)
    if initial_noise is None:
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    else:
        x = initial_noise.to(device=device, dtype=dtype)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    for i in range(steps):
        t_now = ts[i]
        t_next = ts[i + 1]

        a_now, s_now = cosine_alpha_sigma(t_now.unsqueeze(0), schedule_s)
        a_next, s_next = cosine_alpha_sigma(t_next.unsqueeze(0), schedule_s)

        t_batch = t_now.expand(x.shape[0])
        timesteps_for_model = (t_batch * num_train_steps).to(torch.long).clamp(
            max=num_train_steps - 1
        )
        v = model_fn(x, timesteps_for_model, **model_kwargs)

        # x0_hat = alpha * x - sigma * v;  eps_hat = alpha * v + sigma * x
        x0_hat = a_now * x - s_now * v
        eps_hat = a_now * v + s_now * x
        x = a_next * x0_hat + s_next * eps_hat

    return x


__all__ = ["dpm_solver_sampler", "euler_sampler"]
