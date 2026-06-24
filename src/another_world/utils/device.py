"""Device / precision helpers."""

from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device string ("auto" / "cpu" / "cuda" / "cuda:0")."""

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(spec)


def resolve_dtype(precision: str) -> torch.dtype:
    """Map ("fp32" / "bf16" / "fp16") -> torch.dtype."""

    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if precision not in mapping:
        raise ValueError(f"unknown precision '{precision}', expected one of {sorted(mapping)}")
    return mapping[precision]


__all__ = ["resolve_device", "resolve_dtype"]
