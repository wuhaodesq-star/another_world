"""I3D-based Frechet Video Distance wrapper.

The real FVD metric uses features from a pretrained I3D network trained
on Kinetics.  Shipping the weights in this repository would be large and
would make CI slow, so this module implements a **lazy wrapper**:

- If a user provides an I3D feature extractor callable, we compute the
  true Frechet distance over those features.
- If the optional ``torchvision`` / third-party implementation is not
  configured, we raise a clear error or fall back to the raw-pixel
  approximation in :func:`another_world.eval.metrics.fvd_score`.

This keeps the public API stable today while allowing stage 5.x to plug
in a real I3D checkpoint on the H100 box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from another_world.eval.metrics import fvd_score, gaussian_frechet_distance
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)

FeatureExtractor = Callable[[Tensor], Tensor]


@dataclass
class FVDConfig:
    backend: str = "pixel"  # "pixel" | "i3d"
    batch_size: int = 8
    device: str = "auto"
    strict_i3d: bool = False


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class I3DFVD:
    """FVD evaluator with optional injected I3D feature extractor.

    Args:
        extractor: callable mapping videos ``[B, C, T, H, W]`` to features
            ``[B, D]``. When ``None`` and ``config.backend == 'i3d'`` we try
            to lazily load a known external implementation; if unavailable,
            fallback behaviour is controlled by ``strict_i3d``.
    """

    def __init__(
        self,
        config: FVDConfig | None = None,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        self.config = config or FVDConfig()
        self.extractor = extractor

    def __call__(self, real: Tensor, fake: Tensor) -> float:
        if real.shape != fake.shape:
            raise ValueError(f"shape mismatch: {tuple(real.shape)} vs {tuple(fake.shape)}")
        if self.config.backend == "pixel":
            return fvd_score(real, fake)
        if self.config.backend != "i3d":
            raise ValueError(f"unknown FVD backend '{self.config.backend}'")

        extractor = self.extractor or self._load_default_i3d()
        if extractor is None:
            if self.config.strict_i3d:
                raise RuntimeError(
                    "I3D FVD requested but no extractor is configured. "
                    "Install/provide an I3D feature extractor or use "
                    "--fvd-backend pixel."
                )
            _LOG.warning("I3D extractor unavailable; falling back to pixel FVD.")
            return fvd_score(real, fake)

        device = _device(self.config.device)
        real_feats = self._extract_features(real, extractor, device)
        fake_feats = self._extract_features(fake, extractor, device)
        return gaussian_frechet_distance(real_feats, fake_feats)

    def _extract_features(
        self,
        videos: Tensor,
        extractor: FeatureExtractor,
        device: torch.device,
    ) -> Tensor:
        chunks: list[Tensor] = []
        for start in range(0, videos.shape[0], self.config.batch_size):
            chunk = videos[start : start + self.config.batch_size].to(device)
            with torch.no_grad():
                feats = extractor(chunk)
            if feats.dim() != 2:
                raise ValueError(
                    f"feature extractor must return [B, D], got {tuple(feats.shape)}"
                )
            chunks.append(feats.detach().cpu().float())
        return torch.cat(chunks, dim=0)

    def _load_default_i3d(self) -> FeatureExtractor | None:
        """Try to discover an installed I3D implementation.

        We intentionally do not hard-depend on any particular package.
        Stage 5.x can replace this method with a concrete model loader
        once we settle on a weight source.
        """

        try:
            import pytorch_fvd  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            return None
        return None


__all__ = ["FVDConfig", "I3DFVD"]
