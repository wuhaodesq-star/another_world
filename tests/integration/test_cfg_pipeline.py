"""Integration test: classifier-free guidance through the generation pipeline."""

from __future__ import annotations

import torch

from another_world.inference.generation import (
    GenerationConfig,
    decode_tokens_to_pixels,
)
from another_world.models.decoder import DiTDecoder, DiTDecoderConfig


def test_cfg_changes_decoded_output() -> None:
    """A non-trivial cfg_scale should change the sampler output."""

    torch.manual_seed(0)
    cfg = DiTDecoderConfig.toy(vocab_size=32)
    decoder = DiTDecoder(cfg)
    decoder.eval()

    tokens = torch.randint(0, 32, (1, 2, 2))

    common_kwargs = dict(
        latent_channels=cfg.in_channels,
        pixel_t=2, pixel_h=8, pixel_w=8,
        sampler="euler", sampler_steps=4, seed=0,
    )
    out_nocfg = decode_tokens_to_pixels(
        decoder, token_ids=tokens,
        config=GenerationConfig(cfg_scale=1.0, **common_kwargs),
    )
    out_cfg = decode_tokens_to_pixels(
        decoder, token_ids=tokens,
        config=GenerationConfig(cfg_scale=3.5, null_token_id=0, **common_kwargs),
    )
    assert out_nocfg.shape == out_cfg.shape
    # With nonzero gates we expect the two paths to diverge.
    assert not torch.allclose(out_nocfg, out_cfg)
