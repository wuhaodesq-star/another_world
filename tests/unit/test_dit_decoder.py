"""Tests for the DiT decoder skeleton."""

from __future__ import annotations

import pytest
import torch

from another_world.models.decoder.dit import (
    DiTBlock,
    DiTDecoder,
    DiTDecoderConfig,
    SpatialPatchEmbed,
    TimestepEmbedder,
    TokenContextEmbedder,
    timestep_embedding,
    unpatchify,
)


def test_timestep_embedding_shape_and_range() -> None:
    t = torch.tensor([0, 1, 100, 999])
    emb = timestep_embedding(t, dim=16)
    assert emb.shape == (4, 16)
    assert (-1.001 <= emb).all() and (emb <= 1.001).all()


def test_timestep_embedding_odd_dim() -> None:
    t = torch.tensor([0])
    emb = timestep_embedding(t, dim=7)
    assert emb.shape == (1, 7)


def test_timestep_embedder_module() -> None:
    mod = TimestepEmbedder(hidden_size=32)
    t = torch.tensor([0, 5, 100])
    out = mod(t)
    assert out.shape == (3, 32)


def test_token_context_embedder_ids() -> None:
    ctx = TokenContextEmbedder(hidden_size=16, vocab_size=64)
    ids = torch.randint(0, 64, (2, 4))
    out = ctx(ids=ids)
    assert out.shape == (2, 4, 16)


def test_token_context_embedder_embeds() -> None:
    ctx = TokenContextEmbedder(hidden_size=16, latent_channels=8)
    embeds = torch.randn(2, 4, 8)
    out = ctx(embeds=embeds)
    assert out.shape == (2, 4, 16)


def test_token_context_embedder_requires_an_input() -> None:
    ctx = TokenContextEmbedder(hidden_size=16, vocab_size=8)
    with pytest.raises(ValueError):
        ctx()


def test_token_context_embedder_no_vocab_no_embeds_raises() -> None:
    with pytest.raises(ValueError):
        TokenContextEmbedder(hidden_size=16)


def test_spatial_patch_embed_round_trip_via_unpatchify() -> None:
    embed = SpatialPatchEmbed(in_channels=3, patch_size=2, dim=8)
    x = torch.randn(2, 3, 4, 8, 8)
    tokens, shape = embed(x)
    t, h2, w2 = shape
    assert tokens.shape == (2, t * h2 * w2, 8)


def test_unpatchify_roundtrip_shape() -> None:
    b, t, h2, w2 = 2, 3, 4, 4
    patch, channels = 2, 3
    x = torch.randn(b, t * h2 * w2, patch * patch * channels)
    out = unpatchify(x, shape=(t, h2, w2), patch_size=patch, out_channels=channels)
    assert out.shape == (b, channels, t, h2 * patch, w2 * patch)


def test_unpatchify_size_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        unpatchify(
            torch.zeros(1, 5, 12), shape=(2, 2, 2), patch_size=2, out_channels=3,
        )


def test_patch_embed_rejects_non_divisible_size() -> None:
    embed = SpatialPatchEmbed(in_channels=3, patch_size=4, dim=8)
    with pytest.raises(ValueError):
        embed(torch.randn(1, 3, 2, 7, 8))


def test_dit_block_forward_shapes() -> None:
    block = DiTBlock(dim=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    c = torch.randn(2, 32)
    out = block(x, c)
    assert out.shape == x.shape


def test_dit_decoder_discrete_conditioning() -> None:
    cfg = DiTDecoderConfig.toy(vocab_size=128)
    model = DiTDecoder(cfg)
    latents = torch.randn(2, cfg.in_channels, 2, 8, 8)
    timesteps = torch.tensor([0, 500])
    token_ids = torch.randint(0, 128, (2, 6))
    out = model(latents, timesteps, token_ids=token_ids)
    assert out.shape == (2, cfg.out_channels, 2, 8, 8)


def test_dit_decoder_continuous_conditioning() -> None:
    cfg = DiTDecoderConfig(
        in_channels=4, out_channels=4, patch_size=2,
        dim=32, n_layers=2, n_heads=4, latent_channels=16,
    )
    model = DiTDecoder(cfg)
    latents = torch.randn(1, 4, 2, 8, 8)
    timesteps = torch.tensor([10])
    embeds = torch.randn(1, 4, 16)
    out = model(latents, timesteps, token_embeds=embeds)
    assert out.shape == (1, 4, 2, 8, 8)


def test_dit_decoder_backward() -> None:
    cfg = DiTDecoderConfig.toy(vocab_size=32)
    model = DiTDecoder(cfg)
    latents = torch.randn(1, cfg.in_channels, 2, 8, 8, requires_grad=True)
    timesteps = torch.tensor([0])
    token_ids = torch.randint(0, 32, (1, 4))
    out = model(latents, timesteps, token_ids=token_ids)
    out.pow(2).mean().backward()
    grad_seen = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert grad_seen


def test_dit_decoder_invalid_latent_dim_raises() -> None:
    cfg = DiTDecoderConfig.toy()
    model = DiTDecoder(cfg)
    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 8, 8), torch.tensor([0]),
              token_ids=torch.randint(0, 16, (1, 4)))


def test_dit_decoder_param_count_reasonable() -> None:
    cfg = DiTDecoderConfig.toy()
    model = DiTDecoder(cfg)
    assert model.num_parameters() > 0
