"""Wrapper around NVIDIA Cosmos-Tokenizer.

This module provides a thin, dependency-light wrapper that lets the rest of
Another World treat the tokenizer as a black box:

    encoder = CosmosVideoTokenizer.from_pretrained("Cosmos-1.0-Tokenizer-DV8x16x16")
    indices, codes = encoder.encode(video)          # discrete models
    latents,        = encoder.encode(video)         # continuous models
    reconstruction  = encoder.decode(indices or latents)

We deliberately keep the import of the real ``cosmos_tokenizer`` package
lazy so that:

1. CPU-only CI jobs can still import this module and exercise its argument
   handling / shape checks / model registry via mocks.
2. Developers without the proprietary HF download flow can still run unit
   tests for everything that *uses* the tokenizer, not the tokenizer itself.

References
----------
- https://github.com/NVIDIA/Cosmos-Tokenizer
- https://huggingface.co/collections/nvidia/cosmos-tokenizer-672b93023add81b66a8ff8e6
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from another_world.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CosmosModelSpec:
    """Static metadata about a Cosmos-Tokenizer checkpoint."""

    name: str
    kind: str  # "discrete" or "continuous"
    spatial_compression: int  # e.g. 8 or 16
    temporal_compression: int  # e.g. 4 or 8 (videos only); 1 for images
    latent_channels: int  # discrete: FSQ level count; continuous: channel dim
    is_video: bool
    vocab_size: int | None = None  # discrete only

    @property
    def hf_repo(self) -> str:
        return f"nvidia/{self.name}"


# Subset focused on the variants we plan to use; expand on demand.
COSMOS_REGISTRY: dict[str, CosmosModelSpec] = {
    # ----- discrete video tokenizers (DV) -----
    "Cosmos-0.1-Tokenizer-DV4x8x8": CosmosModelSpec(
        name="Cosmos-0.1-Tokenizer-DV4x8x8",
        kind="discrete",
        spatial_compression=8,
        temporal_compression=4,
        latent_channels=6,
        is_video=True,
        vocab_size=64000,
    ),
    "Cosmos-0.1-Tokenizer-DV8x8x8": CosmosModelSpec(
        name="Cosmos-0.1-Tokenizer-DV8x8x8",
        kind="discrete",
        spatial_compression=8,
        temporal_compression=8,
        latent_channels=6,
        is_video=True,
        vocab_size=64000,
    ),
    "Cosmos-0.1-Tokenizer-DV8x16x16": CosmosModelSpec(
        name="Cosmos-0.1-Tokenizer-DV8x16x16",
        kind="discrete",
        spatial_compression=16,
        temporal_compression=8,
        latent_channels=6,
        is_video=True,
        vocab_size=64000,
    ),
    "Cosmos-1.0-Tokenizer-DV8x16x16": CosmosModelSpec(
        name="Cosmos-1.0-Tokenizer-DV8x16x16",
        kind="discrete",
        spatial_compression=16,
        temporal_compression=8,
        latent_channels=6,
        is_video=True,
        vocab_size=64000,
    ),
    # ----- continuous video tokenizers (CV) -----
    "Cosmos-0.1-Tokenizer-CV4x8x8": CosmosModelSpec(
        name="Cosmos-0.1-Tokenizer-CV4x8x8",
        kind="continuous",
        spatial_compression=8,
        temporal_compression=4,
        latent_channels=16,
        is_video=True,
    ),
    "Cosmos-0.1-Tokenizer-CV8x8x8": CosmosModelSpec(
        name="Cosmos-0.1-Tokenizer-CV8x8x8",
        kind="continuous",
        spatial_compression=8,
        temporal_compression=8,
        latent_channels=16,
        is_video=True,
    ),
    "Cosmos-0.1-Tokenizer-CV8x16x16": CosmosModelSpec(
        name="Cosmos-0.1-Tokenizer-CV8x16x16",
        kind="continuous",
        spatial_compression=16,
        temporal_compression=8,
        latent_channels=16,
        is_video=True,
    ),
    "Cosmos-1.0-Tokenizer-CV8x8x8": CosmosModelSpec(
        name="Cosmos-1.0-Tokenizer-CV8x8x8",
        kind="continuous",
        spatial_compression=8,
        temporal_compression=8,
        latent_channels=16,
        is_video=True,
    ),
    # ----- image tokenizers -----
    "Cosmos-0.1-Tokenizer-DI8x8": CosmosModelSpec(
        name="Cosmos-0.1-Tokenizer-DI8x8",
        kind="discrete",
        spatial_compression=8,
        temporal_compression=1,
        latent_channels=6,
        is_video=False,
        vocab_size=64000,
    ),
    "Cosmos-0.1-Tokenizer-DI16x16": CosmosModelSpec(
        name="Cosmos-0.1-Tokenizer-DI16x16",
        kind="discrete",
        spatial_compression=16,
        temporal_compression=1,
        latent_channels=6,
        is_video=False,
        vocab_size=64000,
    ),
    "Cosmos-0.1-Tokenizer-CI8x8": CosmosModelSpec(
        name="Cosmos-0.1-Tokenizer-CI8x8",
        kind="continuous",
        spatial_compression=8,
        temporal_compression=1,
        latent_channels=16,
        is_video=False,
    ),
    "Cosmos-0.1-Tokenizer-CI16x16": CosmosModelSpec(
        name="Cosmos-0.1-Tokenizer-CI16x16",
        kind="continuous",
        spatial_compression=16,
        temporal_compression=1,
        latent_channels=16,
        is_video=False,
    ),
}


# Default we recommend for stage 1 of the roadmap.
DEFAULT_VIDEO_MODEL = "Cosmos-1.0-Tokenizer-DV8x16x16"


def list_models() -> list[str]:
    """Return all known model names (sorted)."""

    return sorted(COSMOS_REGISTRY)


def get_spec(name: str) -> CosmosModelSpec:
    if name not in COSMOS_REGISTRY:
        raise KeyError(
            f"unknown Cosmos model '{name}'. Known: {list_models()}"
        )
    return COSMOS_REGISTRY[name]


# ---------------------------------------------------------------------------
# Shape / input validation
# ---------------------------------------------------------------------------


def validate_video_input(video: Tensor, spec: CosmosModelSpec) -> None:
    """Validate that ``video`` matches the shape conventions expected by Cosmos.

    Cosmos video tokenizers are *causal* and require ``T = 1 + k * temporal_compression``
    frames for an integer ``k >= 0``, and ``H % spatial_compression == 0`` (same
    for ``W``).
    """

    if not isinstance(video, Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(video).__name__}")
    if video.dim() != 5:
        raise ValueError(
            f"expected video shape [B, C, T, H, W], got {tuple(video.shape)}"
        )
    bsz, channels, frames, height, width = video.shape
    if channels != 3:
        raise ValueError(f"expected 3 color channels, got {channels}")
    if bsz < 1:
        raise ValueError(f"batch size must be >= 1, got {bsz}")

    tc = spec.temporal_compression
    sc = spec.spatial_compression
    if (frames - 1) % tc != 0:
        raise ValueError(
            f"frame count T={frames} not compatible with temporal_compression={tc}. "
            f"Cosmos requires T = 1 + k * {tc} (e.g. {1 + tc}, {1 + 2*tc}, ...)."
        )
    if height % sc != 0 or width % sc != 0:
        raise ValueError(
            f"spatial size {height}x{width} not divisible by "
            f"spatial_compression={sc}."
        )


def expected_latent_shape(
    spec: CosmosModelSpec,
    batch: int,
    frames: int,
    height: int,
    width: int,
) -> tuple[int, ...]:
    """Compute the latent shape produced by ``encode``.

    For discrete models this is the ``indices`` shape ``[B, T', H', W']``.
    For continuous models this is the ``latent`` shape ``[B, C, T', H', W']``.
    """

    t_prime = 1 + (frames - 1) // spec.temporal_compression
    h_prime = height // spec.spatial_compression
    w_prime = width // spec.spatial_compression
    if spec.kind == "discrete":
        return (batch, t_prime, h_prime, w_prime)
    return (batch, spec.latent_channels, t_prime, h_prime, w_prime)


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


@dataclass
class CosmosCheckpoints:
    """Filesystem paths to the encoder / decoder JIT artifacts."""

    encoder: Path | None
    decoder: Path | None
    autoencoder: Path | None = None

    @classmethod
    def from_directory(cls, directory: str | os.PathLike[str]) -> "CosmosCheckpoints":
        d = Path(directory)
        if not d.exists():
            raise FileNotFoundError(f"checkpoint directory not found: {d}")
        return cls(
            encoder=(d / "encoder.jit") if (d / "encoder.jit").exists() else None,
            decoder=(d / "decoder.jit") if (d / "decoder.jit").exists() else None,
            autoencoder=(
                (d / "autoencoder.jit") if (d / "autoencoder.jit").exists() else None
            ),
        )

    def require_encoder(self) -> Path:
        if self.encoder is None:
            raise FileNotFoundError("encoder.jit missing from checkpoint directory")
        return self.encoder

    def require_decoder(self) -> Path:
        if self.decoder is None:
            raise FileNotFoundError("decoder.jit missing from checkpoint directory")
        return self.decoder


def download_cosmos_checkpoint(
    name: str,
    local_dir: str | os.PathLike[str] | None = None,
    hf_token: str | None = None,
) -> Path:
    """Download a Cosmos checkpoint from HuggingFace.

    Imports of ``huggingface_hub`` happen lazily so this module stays usable
    in environments without it.
    """

    spec = get_spec(name)
    if local_dir is None:
        cache_root = Path(
            os.environ.get("ANOTHER_WORLD_TOKENIZER_CACHE", ".cache/cosmos")
        )
        local_dir = cache_root / name
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised when hub missing
        raise ImportError(
            "huggingface_hub is required to download Cosmos checkpoints; "
            "install with `pip install huggingface_hub`."
        ) from exc

    _LOG.info("Downloading %s -> %s", spec.hf_repo, local_dir)
    snapshot_download(
        repo_id=spec.hf_repo,
        local_dir=str(local_dir),
        token=hf_token or os.environ.get("HF_TOKEN"),
    )
    return local_dir


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


@dataclass
class CosmosVideoTokenizer:
    """Wrapper over ``cosmos_tokenizer.video_lib.CausalVideoTokenizer``.

    Construct via :meth:`from_pretrained` or :meth:`from_local`. The wrapper
    holds the spec, paths, and lazily-instantiated encoder/decoder objects.
    """

    spec: CosmosModelSpec
    checkpoints: CosmosCheckpoints
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    dtype: torch.dtype = torch.bfloat16

    _encoder: Any = field(default=None, init=False, repr=False)
    _decoder: Any = field(default=None, init=False, repr=False)

    # ----- factories -------------------------------------------------------

    @classmethod
    def from_local(
        cls,
        name: str,
        directory: str | os.PathLike[str],
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "CosmosVideoTokenizer":
        spec = get_spec(name)
        if not spec.is_video:
            raise ValueError(f"{name} is an image tokenizer; use CosmosImageTokenizer")
        return cls(
            spec=spec,
            checkpoints=CosmosCheckpoints.from_directory(directory),
            device=torch.device(device),
            dtype=dtype,
        )

    @classmethod
    def from_pretrained(
        cls,
        name: str = DEFAULT_VIDEO_MODEL,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        hf_token: str | None = None,
    ) -> "CosmosVideoTokenizer":
        directory = download_cosmos_checkpoint(
            name, local_dir=cache_dir, hf_token=hf_token
        )
        return cls.from_local(name, directory, device=device, dtype=dtype)

    # ----- lazy real-model construction -----------------------------------

    def _make_native(self, *, with_encoder: bool, with_decoder: bool) -> Any:
        """Build the underlying ``CausalVideoTokenizer`` object."""

        try:
            from cosmos_tokenizer.video_lib import (  # type: ignore[import-not-found]
                CausalVideoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "cosmos_tokenizer is not installed. Follow "
                "https://github.com/NVIDIA/Cosmos-Tokenizer to install it."
            ) from exc

        kwargs: dict[str, Any] = {}
        if with_encoder:
            kwargs["checkpoint_enc"] = str(self.checkpoints.require_encoder())
        if with_decoder:
            kwargs["checkpoint_dec"] = str(self.checkpoints.require_decoder())
        return CausalVideoTokenizer(**kwargs)

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = self._make_native(with_encoder=True, with_decoder=False)
        return self._encoder

    @property
    def decoder(self) -> Any:
        if self._decoder is None:
            self._decoder = self._make_native(with_encoder=False, with_decoder=True)
        return self._decoder

    # ----- public API ------------------------------------------------------

    def encode(self, video: Tensor) -> tuple[Tensor, ...]:
        """Encode a video tensor.

        Returns:
            - For discrete models: ``(indices, codes)``.
            - For continuous models: ``(latents,)``.
        """

        validate_video_input(video, self.spec)
        x = video.to(device=self.device, dtype=self.dtype)
        out = self.encoder.encode(x)
        if not isinstance(out, tuple):
            out = (out,)
        return out

    def decode(self, tokens: Tensor) -> Tensor:
        """Decode latents or indices back to a video tensor."""

        x = tokens.to(device=self.device)
        if self.spec.kind == "continuous":
            x = x.to(dtype=self.dtype)
        reconstruction = self.decoder.decode(x)
        return reconstruction

    # ----- utilities -------------------------------------------------------

    def latent_shape_for(
        self, batch: int, frames: int, height: int, width: int
    ) -> tuple[int, ...]:
        return expected_latent_shape(self.spec, batch, frames, height, width)

    def round_frames(self, frames: int) -> int:
        """Round ``frames`` up to the nearest Cosmos-compatible value."""

        tc = self.spec.temporal_compression
        if frames < 1:
            raise ValueError("frames must be >= 1")
        k = math.ceil((frames - 1) / tc)
        return 1 + max(0, k) * tc


__all__ = [
    "COSMOS_REGISTRY",
    "CosmosCheckpoints",
    "CosmosModelSpec",
    "CosmosVideoTokenizer",
    "DEFAULT_VIDEO_MODEL",
    "download_cosmos_checkpoint",
    "expected_latent_shape",
    "get_spec",
    "list_models",
    "validate_video_input",
]
