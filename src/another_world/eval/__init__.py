"""Evaluation suite."""

from another_world.eval.i3d_fvd import FVDConfig, I3DFVD
from another_world.eval.long_horizon import HorizonResult, evaluate_long_horizon
from another_world.eval.metrics import (
    fvd_score,
    gaussian_frechet_distance,
    long_horizon_drift,
    mae,
    mse,
    psnr,
    temporal_consistency,
    token_accuracy,
    token_top_k,
)
from another_world.eval.vbench_wrapper import (
    VBENCH_DIMENSIONS,
    VBenchAdapter,
    vbench_or_fallback,
)

__all__ = [
    "FVDConfig",
    "HorizonResult",
    "I3DFVD",
    "VBENCH_DIMENSIONS",
    "VBenchAdapter",
    "evaluate_long_horizon",
    "fvd_score",
    "gaussian_frechet_distance",
    "long_horizon_drift",
    "mae",
    "mse",
    "psnr",
    "temporal_consistency",
    "token_accuracy",
    "token_top_k",
    "vbench_or_fallback",
]
