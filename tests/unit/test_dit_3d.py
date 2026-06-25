"""Tests for the 3D spatiotemporal patch embed + DiT 3D mode."""

from __future__ import annotations

import pytest
import torch

from another_world.models.decoder import (
    DiTDecoder,
    DiTDecoderConfig,
    SpatiotemporalPatchEmbed,
    unpatchify_3d,
)


def test_spatiotemporal_patch_embed_output_shape() -> None:
    embed = SpatiotemporalPatchEmbed(in_channels=4, patch_t=2, patch_s=4, dim=16)
    x = torch.randn(2, 4, 4, 16, 16)
    tokens, shape = embed(x)
    # t' = 4/2 = 2,  h' = w' = 16/4 = 4
    assert shape == (2, 4, 4)
    assert tokens.shape == (2, 2 * 4 * 4, 16)


def test_spatiotemporal_patch_embed_rejects_bad_dim() -> None:
    embed = SpatiotemporalPatchEmbed(in_channels=4, patch_t=2, patch_s=4, dim=16)
    with pytest.raises(ValueError):
        embed(torch.randn(2, 4, 3, 16, 16))    # T not divisible by patch_t
    with pytest.raises(ValueError):
        embed(torch.randn(2, 4, 4, 15, 16))    # H not divisible by patch_s


def test_spatiotemporal_patch_embed_rank_check() -> None:
    embed = SpatiotemporalPatchEmbed(in_channels=4, patch_t=2, patch_s=4, dim=16)
    with pytest.raises(ValueError):
        embed(torch.randn(2, 4, 4, 16))


def test_unpatchify_3d_round_trip_shape() -> None:
    b, t, h, w = 2, 3, 4, 4
    patch_t, patch_s, channels = 2, 4, 3
    x = torch.randn(b, t * h * w, patch_t * patch_s * patch_s * channels)
    out = unpatchify_3d(
        x, shape=(t, h, w), patch_t=patch_t, patch_s=patch_s, out_channels=channels,
    )
    assert out.shape == (b, channels, t * patch_t, h * patch_s, w * patch_s)


def test_unpatchify_3d_token_count_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        unpatchify_3d(
            torch.zeros(1, 5, 12), shape=(2, 2, 2),
            patch_t=2, patch_s=1, out_channels=3,
        )


def test_dit_decoder_3d_mode_forward() -> None:
    """3-D mode should accept and produce matching pixel-shape tensors."""

    cfg = DiTDecoderConfig.toy_3d(vocab_size=64)
    assert cfg.patch_t == 2
    model = DiTDecoder(cfg)
    # Input must have T divisible by patch_t and H/W by patch_size.
    latents = torch.randn(2, cfg.in_channels, 4, 8, 8)
    timesteps = torch.tensor([0, 500])
    token_ids = torch.randint(0, 64, (2, 4))
    out = model(latents, timesteps, token_ids=token_ids)
    assert out.shape == (2, cfg.out_channels, 4, 8, 8)


def test_dit_decoder_3d_backward() -> None:
    cfg = DiTDecoderConfig.toy_3d(vocab_size=32)
    model = DiTDecoder(cfg)
    latents = torch.randn(1, cfg.in_channels, 4, 8, 8, requires_grad=True)
    timesteps = torch.tensor([0])
    token_ids = torch.randint(0, 32, (1, 4))
    out = model(latents, timesteps, token_ids=token_ids)
    out.pow(2).mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )


def test_dit_decoder_2d_mode_still_works() -> None:
    """The original 2-D path must remain unchanged."""
    cfg = DiTDecoderConfig.toy(vocab_size=32)
    model = DiTDecoder(cfg)
    latents = torch.randn(1, cfg.in_channels, 2, 8, 8)
    timesteps = torch.tensor([10])
    token_ids = torch.randint(0, 32, (1, 4))
    out = model(latents, timesteps, token_ids=token_ids)
    assert out.shape == (1, cfg.out_channels, 2, 8, 8)
