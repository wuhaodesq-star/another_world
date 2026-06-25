"""Integration test: first-frame conditioning through the full generate() pipeline."""

from __future__ import annotations

import torch

from another_world.inference.generation import GenerationConfig, generate
from another_world.models.decoder import DiTDecoder, DiTDecoderConfig
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout


def test_generate_with_first_frame() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    dyn = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    ).eval()
    dec = DiTDecoder(DiTDecoderConfig.toy(vocab_size=layout.visual_size)).eval()

    cfg = GenerationConfig(
        visual_frames=2, visual_height=2, visual_width=2,
        sampler="euler", sampler_steps=2, seed=0,
        latent_channels=4, pixel_t=2, pixel_h=8, pixel_w=8,
    )
    prefix = torch.tensor([[1, 3], [5, 7]], dtype=torch.long)
    result = generate(
        dynamics=dyn, decoder=dec,
        text_ids=[1, 2, 3], layout=layout, config=cfg,
        first_frame=prefix,
    )
    assert result.visual_tokens.shape == (2, 2, 2)
    assert torch.equal(result.visual_tokens[0], prefix)
    assert result.pixels.shape == (1, 4, 2, 8, 8)


def test_generate_with_action_conditioning_yields_finite_pixels() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    dyn = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    ).eval()
    dec = DiTDecoder(DiTDecoderConfig.toy(vocab_size=layout.visual_size)).eval()
    cfg = GenerationConfig(
        visual_frames=1, visual_height=2, visual_width=2,
        sampler="euler", sampler_steps=2, seed=0,
        latent_channels=4, pixel_t=2, pixel_h=8, pixel_w=8,
    )
    result = generate(
        dynamics=dyn, decoder=dec,
        text_ids=None, layout=layout, config=cfg,
        action_ids=[1, 2, 3],
    )
    assert torch.isfinite(result.pixels).all()
