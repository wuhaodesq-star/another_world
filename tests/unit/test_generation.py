"""Tests for the end-to-end generation pipeline."""

from __future__ import annotations

import torch

from another_world.inference.generation import (
    GenerationConfig,
    decode_tokens_to_pixels,
    generate,
    rollout_visual_tokens,
)
from another_world.models.decoder import DiTDecoder, DiTDecoderConfig
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout


def test_rollout_visual_tokens_shape_and_range() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    model.eval()
    cfg = GenerationConfig(visual_frames=1, visual_height=2, visual_width=2, seed=0)
    out = rollout_visual_tokens(
        model,
        text_ids=[1, 2, 3],
        config=cfg,
        layout=layout,
    )
    assert out.shape == (1, 2, 2)
    assert (out >= 0).all() and (out < layout.visual_size).all()


def test_rollout_seeded_is_reproducible() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    model.eval()
    cfg = GenerationConfig(visual_frames=1, visual_height=2, visual_width=2, seed=7)
    a = rollout_visual_tokens(model, text_ids=[1, 2], config=cfg, layout=layout)
    b = rollout_visual_tokens(model, text_ids=[1, 2], config=cfg, layout=layout)
    assert torch.equal(a, b)


def test_decode_tokens_to_pixels_shape() -> None:
    layout = VocabLayout.tiny()
    dit_cfg = DiTDecoderConfig.toy(vocab_size=layout.visual_size)
    decoder = DiTDecoder(dit_cfg)
    decoder.eval()
    tokens = torch.randint(0, layout.visual_size, (1, 2, 2))
    cfg = GenerationConfig(
        latent_channels=dit_cfg.in_channels,
        pixel_t=2,
        pixel_h=8,
        pixel_w=8,
        sampler="euler",
        sampler_steps=2,
    )
    out = decode_tokens_to_pixels(decoder, token_ids=tokens, config=cfg)
    assert out.shape == (1, dit_cfg.out_channels, 2, 8, 8)


def test_generate_end_to_end_runs() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    dyn = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    dyn.eval()
    decoder = DiTDecoder(DiTDecoderConfig.toy(vocab_size=layout.visual_size))
    decoder.eval()

    cfg = GenerationConfig(
        visual_frames=1, visual_height=2, visual_width=2,
        latent_channels=4, pixel_t=2, pixel_h=8, pixel_w=8,
        sampler="euler", sampler_steps=2, seed=0,
    )
    result = generate(
        dynamics=dyn, decoder=decoder,
        text_ids=[1, 2, 3],
        layout=layout,
        config=cfg,
    )
    assert result.visual_tokens.shape == (1, 2, 2)
    assert result.pixels.shape == (1, 4, 2, 8, 8)
    assert torch.isfinite(result.pixels).all()


def test_generate_supports_dpm_solver() -> None:
    torch.manual_seed(1)
    layout = VocabLayout.tiny()
    dyn = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    dyn.eval()
    decoder = DiTDecoder(DiTDecoderConfig.toy(vocab_size=layout.visual_size))
    decoder.eval()
    cfg = GenerationConfig(
        visual_frames=1, visual_height=2, visual_width=2,
        latent_channels=4, pixel_t=2, pixel_h=8, pixel_w=8,
        sampler="dpm_solver", sampler_steps=2, seed=1,
    )
    result = generate(
        dynamics=dyn, decoder=decoder,
        text_ids=None,
        layout=layout,
        config=cfg,
    )
    assert result.pixels.shape == (1, 4, 2, 8, 8)
