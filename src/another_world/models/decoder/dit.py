"""Diffusion-Transformer (DiT) pixel decoder skeleton.

The dynamics model emits visual tokens (integer indices for discrete
tokenizers or continuous latents). The DiT decoder turns those back into
pixel-space frames. Following Sora / Open-Sora-Plan / CogVideoX, we use a
spatiotemporal Transformer operating on flattened ``[T, H, W]`` latent
cubes, conditioned on token embeddings via cross-attention or AdaLN
modulation.

This file implements the **skeleton**:

- A pluggable token-embedding bridge (continuous latents pass through;
  discrete indices go through an embedding table).
- AdaLN-Zero conditioned DiT blocks with full self-attention over the
  spatiotemporal grid plus a feed-forward.
- Sinusoidal timestep embeddings.
- A patch-projection head that maps the final hidden states back to
  ``[B, C_out, T, H, W]`` pixel-space tensors.

Training the DiT to actually produce realistic frames (rectified-flow or
DDPM objective, large data, large compute) lives in stage 4 of the
roadmap; this module's role is to give us a working, dependency-light
implementation that can be unit-tested on CPU and slotted into the larger
generation pipeline as we go.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from another_world.models.layers.common import SwiGLU, init_weights


# ---------------------------------------------------------------------------
# Timestep / token conditioning
# ---------------------------------------------------------------------------


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Standard sinusoidal timestep embedding."""

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_dim: int | None = None) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim or hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: Tensor) -> Tensor:
        emb = timestep_embedding(t, self.frequency_dim)
        return self.mlp(emb)


class TokenContextEmbedder(nn.Module):
    """Project external token embeddings into the DiT hidden width.

    Two input modes:

    - Discrete indices (``token_ids``): an embedding table of size
      ``vocab_size`` is used. Set ``vocab_size`` accordingly.
    - Continuous latents (``token_embeds``): an MLP projects the latent
      width to the hidden width.

    The caller chooses by passing either ``ids`` or ``embeds`` to
    :meth:`forward`.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        vocab_size: int | None = None,
        latent_channels: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size is None and latent_channels is None:
            raise ValueError(
                "TokenContextEmbedder needs either vocab_size or latent_channels"
            )
        self.vocab_size = vocab_size
        self.latent_channels = latent_channels
        if vocab_size is not None:
            self.embed = nn.Embedding(vocab_size, hidden_size)
        if latent_channels is not None:
            self.proj = nn.Sequential(
                nn.Linear(latent_channels, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

    def forward(
        self,
        ids: Tensor | None = None,
        embeds: Tensor | None = None,
    ) -> Tensor:
        if ids is not None:
            if self.vocab_size is None:
                raise RuntimeError("TokenContextEmbedder built without vocab_size")
            return self.embed(ids)
        if embeds is not None:
            if self.latent_channels is None:
                raise RuntimeError(
                    "TokenContextEmbedder built without latent_channels"
                )
            return self.proj(embeds)
        raise ValueError("provide either ids or embeds")


# ---------------------------------------------------------------------------
# AdaLN-Zero DiT block
# ---------------------------------------------------------------------------


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Pre-norm Transformer block with AdaLN-Zero conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True, bias=False,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = ((int(dim * mlp_ratio) + 63) // 64) * 64
        self.mlp = SwiGLU(dim, hidden, dropout)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln(c).chunk(6, dim=-1)
        )
        h = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_out
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """Project the final hidden states to per-patch pixel deltas.

    The ``output_size`` argument is the *total* fan-out per token; for a
    2-D ``patch x patch x C`` head pass ``patch_size**2 * C``, for a 3-D
    ``patch_t x patch x patch x C`` head pass ``patch_t * patch**2 * C``.
    """

    def __init__(self, dim: int, output_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, output_size, bias=True)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaln(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Patch embed / unembed for [B, C, T, H, W]
# ---------------------------------------------------------------------------


class SpatialPatchEmbed(nn.Module):
    """Project ``[B, C, T, H, W]`` -> ``[B, T*H'*W', dim]`` tokens.

    Treats the temporal dim as independent (each frame is patchified the
    same way). For a small skeleton this is enough; the 3-D variant
    (:class:`SpatiotemporalPatchEmbed`) handles full spatio-temporal
    patches like CogVideoX.
    """

    def __init__(self, in_channels: int, patch_size: int, dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        b, c, t, h, w = x.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"frame size {h}x{w} must be divisible by patch_size={self.patch_size}"
            )
        x = x.transpose(1, 2).reshape(b * t, c, h, w)
        x = self.proj(x)                                    # [B*T, dim, h', w']
        h2, w2 = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)                    # [B*T, h'*w', dim]
        x = x.view(b, t, h2 * w2, x.shape[-1])              # [B, T, P, dim]
        x = x.reshape(b, t * h2 * w2, x.shape[-1])          # [B, T*P, dim]
        return x, (t, h2, w2)


class SpatiotemporalPatchEmbed(nn.Module):
    """3-D patch projection: ``[B, C, T, H, W]`` -> ``[B, T'*H'*W', dim]``.

    Uses a single Conv3d so each output token aggregates a
    ``(patch_t, patch_s, patch_s)`` cube of input pixels. This is the
    layout used by CogVideoX / Open-Sora-Plan v1.3.
    """

    def __init__(
        self,
        in_channels: int,
        patch_t: int,
        patch_s: int,
        dim: int,
    ) -> None:
        super().__init__()
        if patch_t < 1 or patch_s < 1:
            raise ValueError("patch sizes must be >= 1")
        self.patch_t = patch_t
        self.patch_s = patch_s
        self.proj = nn.Conv3d(
            in_channels, dim,
            kernel_size=(patch_t, patch_s, patch_s),
            stride=(patch_t, patch_s, patch_s),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        if x.dim() != 5:
            raise ValueError(
                f"expected [B, C, T, H, W], got {tuple(x.shape)}"
            )
        b, c, t, h, w = x.shape
        if t % self.patch_t:
            raise ValueError(
                f"T={t} not divisible by patch_t={self.patch_t}"
            )
        if h % self.patch_s or w % self.patch_s:
            raise ValueError(
                f"spatial size {h}x{w} not divisible by patch_s={self.patch_s}"
            )
        x = self.proj(x)                                # [B, dim, t', h', w']
        t2, h2, w2 = x.shape[-3:]
        x = x.flatten(2).transpose(1, 2)                # [B, t'*h'*w', dim]
        return x, (t2, h2, w2)


def unpatchify_3d(
    x: Tensor,
    shape: tuple[int, int, int],
    patch_t: int,
    patch_s: int,
    out_channels: int,
) -> Tensor:
    """Inverse of :class:`SpatiotemporalPatchEmbed`."""

    b, n, _ = x.shape
    t, h, w = shape
    if n != t * h * w:
        raise ValueError(f"token count {n} != t*h*w = {t * h * w}")
    x = x.view(b, t, h, w, patch_t, patch_s, patch_s, out_channels)
    # -> [B, C, T*patch_t, H*patch_s, W*patch_s]
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return x.view(b, out_channels, t * patch_t, h * patch_s, w * patch_s)


def unpatchify(x: Tensor, shape: tuple[int, int, int], patch_size: int,
               out_channels: int) -> Tensor:
    """Inverse of :class:`SpatialPatchEmbed` (patch -> pixel)."""

    b, n, _ = x.shape
    t, h_p, w_p = shape
    if n != t * h_p * w_p:
        raise ValueError(f"token count {n} != t*h*w = {t*h_p*w_p}")
    x = x.view(b, t, h_p, w_p, patch_size, patch_size, out_channels)
    # [B, T, h', w', P, P, C] -> [B, C, T, h'*P, w'*P]
    x = x.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
    return x.view(b, out_channels, t, h_p * patch_size, w_p * patch_size)


# ---------------------------------------------------------------------------
# Top-level DiT
# ---------------------------------------------------------------------------


@dataclass
class DiTDecoderConfig:
    in_channels: int = 4
    out_channels: int = 4
    patch_size: int = 2          # spatial patch (always)
    patch_t: int = 1             # temporal patch; 1 == purely spatial mode
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # Conditioning bridge: choose exactly one of these.
    vocab_size: int | None = None
    latent_channels: int | None = None

    @classmethod
    def toy(cls, vocab_size: int = 256) -> "DiTDecoderConfig":
        return cls(
            in_channels=4, out_channels=4, patch_size=2,
            dim=64, n_layers=2, n_heads=4, vocab_size=vocab_size,
        )

    @classmethod
    def toy_3d(cls, vocab_size: int = 256) -> "DiTDecoderConfig":
        """Toy 3-D variant (1 temporal patch == 2 frames at a time)."""

        return cls(
            in_channels=4, out_channels=4, patch_size=2, patch_t=2,
            dim=64, n_layers=2, n_heads=4, vocab_size=vocab_size,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DiTDecoder(nn.Module):
    """Token-conditioned video Diffusion Transformer.

    ``forward(latents, timesteps, token_ids|token_embeds)`` returns a
    pixel-shape tensor that the diffusion sampler interprets as the
    predicted noise (or velocity, depending on objective).
    """

    def __init__(self, config: DiTDecoderConfig) -> None:
        super().__init__()
        self.config = config

        if config.patch_t > 1:
            self.patch_embed = SpatiotemporalPatchEmbed(
                in_channels=config.in_channels,
                patch_t=config.patch_t,
                patch_s=config.patch_size,
                dim=config.dim,
            )
            self._spatiotemporal = True
        else:
            self.patch_embed = SpatialPatchEmbed(
                in_channels=config.in_channels,
                patch_size=config.patch_size,
                dim=config.dim,
            )
            self._spatiotemporal = False
        self.t_embed = TimestepEmbedder(config.dim)
        self.ctx_embed = TokenContextEmbedder(
            hidden_size=config.dim,
            vocab_size=config.vocab_size,
            latent_channels=config.latent_channels,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=config.dim,
                    num_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        # FinalLayer fan-out: patch elements * output channels.
        if self._spatiotemporal:
            output_size = (
                config.patch_t
                * config.patch_size
                * config.patch_size
                * config.out_channels
            )
        else:
            output_size = (
                config.patch_size * config.patch_size * config.out_channels
            )
        self.final = FinalLayer(dim=config.dim, output_size=output_size)
        self.apply(init_weights)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        *,
        token_ids: Tensor | None = None,
        token_embeds: Tensor | None = None,
    ) -> Tensor:
        if latents.dim() != 5:
            raise ValueError(
                f"expected latents [B, C, T, H, W], got {tuple(latents.shape)}"
            )

        tokens, shape = self.patch_embed(latents)
        t_emb = self.t_embed(timesteps)                                # [B, dim]
        ctx = self.ctx_embed(ids=token_ids, embeds=token_embeds)        # [B, K, dim]
        cond = t_emb + ctx.mean(dim=1)                                  # [B, dim]

        x = tokens
        for block in self.blocks:
            x = block(x, cond)
        x = self.final(x, cond)

        if self._spatiotemporal:
            return unpatchify_3d(
                x, shape=shape,
                patch_t=self.config.patch_t,
                patch_s=self.config.patch_size,
                out_channels=self.config.out_channels,
            )
        return unpatchify(
            x, shape=shape,
            patch_size=self.config.patch_size,
            out_channels=self.config.out_channels,
        )


__all__ = [
    "DiTBlock",
    "DiTDecoder",
    "DiTDecoderConfig",
    "FinalLayer",
    "SpatialPatchEmbed",
    "SpatiotemporalPatchEmbed",
    "TimestepEmbedder",
    "TokenContextEmbedder",
    "timestep_embedding",
    "unpatchify",
    "unpatchify_3d",
]
