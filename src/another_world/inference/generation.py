"""End-to-end video generation pipeline.

Stitches together the three model components:

1. **Dynamics rollout**: feed text + reference-frame tokens into the
   multimodal dynamics model and autoregressively sample visual tokens
   for the next ``T*H*W`` positions.

2. **DiT decode**: run the diffusion sampler over those tokens to
   produce latent / pixel video.

For the MVP target (text + first frame -> 5 s 512x288 video) this module
provides the orchestration code. The actual *quality* depends on having
trained checkpoints for both the dynamics model and the DiT decoder; in
stage 4 we ship the orchestration so the integration tests can run
on untrained models with mock tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from another_world.models.decoder import (
    DiTDecoder,
    dpm_solver_sampler,
    euler_sampler,
)
from another_world.models.dynamics import MultimodalDynamicsModel
from another_world.models.layers.mixed_rope import RopeAxes, axes_from_segments
from another_world.tokenizers.vocab import VocabInfo, VocabLayout


@dataclass
class GenerationConfig:
    """Hyperparameters controlling a single generation request."""

    visual_frames: int = 5            # T' axis of the visual cube
    visual_height: int = 16           # H' axis
    visual_width: int = 16            # W' axis
    temperature: float = 1.0
    top_k: int | None = None
    sampler: str = "euler"            # "euler" | "dpm_solver"
    sampler_steps: int = 30
    latent_channels: int = 4
    pixel_t: int = 17                 # output T frames (post DiT-decode)
    pixel_h: int = 256
    pixel_w: int = 256
    seed: int | None = None


# ---------------------------------------------------------------------------
# Token-level rollout in the dynamics model
# ---------------------------------------------------------------------------


def _build_prompt(
    text_ids: list[int] | None,
    *,
    layout: VocabLayout,
    vocab: VocabInfo,
) -> tuple[list[int], list[tuple[str, dict]]]:
    """Construct the initial token list + axes-segment description.

    Returns ``(token_ids, segments)`` where ``segments`` is in the
    :func:`axes_from_segments` format.
    """

    tokens: list[int] = [vocab.bos_id]
    segments: list[tuple[str, dict]] = [("special", {"count": 1})]

    if text_ids:
        tokens.append(vocab.boc_id)
        segments.append(("special", {"count": 1}))
        for tid in text_ids:
            tokens.append(layout.encode_text(int(tid)))
        segments.append(("text", {"count": len(text_ids)}))
        tokens.append(vocab.eoc_id)
        segments.append(("special", {"count": 1}))

    tokens.append(vocab.bov_id)
    segments.append(("special", {"count": 1}))
    return tokens, segments


def _sample_next(
    logits: Tensor, *,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token id from logits ``[B, V]``."""

    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        thresh = v[..., -1, None]
        logits = torch.where(logits < thresh, torch.full_like(logits, -1e9), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def rollout_visual_tokens(
    model: MultimodalDynamicsModel,
    *,
    text_ids: list[int] | None,
    config: GenerationConfig,
    layout: VocabLayout,
) -> Tensor:
    """Autoregressively sample a ``[T, H, W]`` visual-token cube.

    The function expects the caller to use a small enough cube that the
    *entire* sequence fits inside the model's ``max_linear`` window;
    no KV-cache is implemented here yet (a stage-3.3 optimisation).
    """

    device = next(model.parameters()).device
    vocab = VocabInfo(layout=layout)

    prompt_tokens, segments = _build_prompt(text_ids, layout=layout, vocab=vocab)
    total_visual = config.visual_frames * config.visual_height * config.visual_width
    generator = (
        torch.Generator(device=device).manual_seed(config.seed)
        if config.seed is not None else None
    )

    sampled: list[int] = []
    for vi in range(total_visual):
        # Build the partial sequence (prompt + already-sampled visual tokens).
        cur_tokens = prompt_tokens + sampled
        cur_segments = list(segments)
        if vi > 0:
            cur_segments.append(("visual", {
                "t": config.visual_frames,
                "h": config.visual_height,
                "w": config.visual_width,
            }))
            # Truncate the visual segment to vi by re-issuing the helper.
            # axes_from_segments expands to the *full* visual cube so we
            # rebuild the axes manually here for partial decode.
            axes = _partial_axes_after_prompt(
                prompt_segments=segments,
                visual_so_far=vi,
                config=config,
                device=device,
            )
        else:
            axes = axes_from_segments(cur_segments, device=device)

        tokens_t = torch.tensor([cur_tokens], dtype=torch.long, device=device)
        out = model(tokens_t, axes=axes)
        last_logits = out["logits"][:, -1, :]
        # Restrict sampling to the visual slab.
        slab_start = layout.visual_start
        slab_end = layout.action_start
        mask = torch.full_like(last_logits, -1e9)
        mask[:, slab_start:slab_end] = 0.0
        next_id = _sample_next(
            last_logits + mask,
            temperature=config.temperature,
            top_k=config.top_k,
            generator=generator,
        ).item()
        sampled.append(int(next_id))

    # Convert global ids back to *local* visual ids (subtract slab offset).
    arr = torch.tensor(sampled, dtype=torch.long) - layout.visual_start
    return arr.view(config.visual_frames, config.visual_height, config.visual_width)


def _partial_axes_after_prompt(
    *,
    prompt_segments: list[tuple[str, dict]],
    visual_so_far: int,
    config: GenerationConfig,
    device: torch.device,
) -> RopeAxes:
    """Build axes for the prompt + first ``visual_so_far`` visual tokens.

    The visual segment is row-major (t, h, w) so we can simply truncate
    after generating ``visual_so_far`` cells.
    """

    segs = list(prompt_segments)
    # Decompose visual_so_far into (full t rows, plus a partial row, plus a
    # partial column).  Easiest is to add raw 'special' filler with count
    # ``visual_so_far`` so axes_from_segments yields a flat sequence; we
    # then patch the modality/t/h/w arrays.
    if visual_so_far == 0:
        return axes_from_segments(segs, device=device)

    axes_so_far = axes_from_segments(
        segs + [("special", {"count": visual_so_far})], device=device
    )
    # Override the trailing slots with proper visual modality & THW coords.
    modality = axes_so_far.modality.clone()
    t_coords = axes_so_far.t.clone()
    h_coords = axes_so_far.h.clone()
    w_coords = axes_so_far.w.clone()

    start = modality.shape[1] - visual_so_far
    for idx in range(visual_so_far):
        slot = start + idx
        ti = idx // (config.visual_height * config.visual_width)
        hi = (idx // config.visual_width) % config.visual_height
        wi = idx % config.visual_width
        modality[0, slot] = 1
        t_coords[0, slot] = ti
        h_coords[0, slot] = hi
        w_coords[0, slot] = wi

    return RopeAxes(
        modality=modality,
        linear=axes_so_far.linear,
        t=t_coords,
        h=h_coords,
        w=w_coords,
    )


# ---------------------------------------------------------------------------
# DiT decode
# ---------------------------------------------------------------------------


@torch.no_grad()
def decode_tokens_to_pixels(
    decoder: DiTDecoder,
    *,
    token_ids: Tensor,
    config: GenerationConfig,
) -> Tensor:
    """Run a sampler over the DiT to turn token ids into a pixel-space tensor.

    ``token_ids`` should be a 1-D or 3-D long tensor with values in the
    DiT's vocabulary range; the sampler conditions on these via the
    DiT's :class:`TokenContextEmbedder`.
    """

    device = next(decoder.parameters()).device
    flat_ids = token_ids.reshape(1, -1).to(device)
    shape = (
        1,
        config.latent_channels,
        config.pixel_t,
        config.pixel_h,
        config.pixel_w,
    )

    def model_fn(x: Tensor, timesteps: Tensor, **_kwargs) -> Tensor:
        return decoder(x, timesteps, token_ids=flat_ids)

    if config.sampler == "euler":
        return euler_sampler(
            model_fn, shape=shape, steps=config.sampler_steps,
            device=device,
        )
    if config.sampler == "dpm_solver":
        return dpm_solver_sampler(
            model_fn, shape=shape, steps=config.sampler_steps,
            device=device,
        )
    raise ValueError(f"unknown sampler '{config.sampler}'")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    visual_tokens: Tensor          # [T', H', W'] local visual ids
    pixels: Tensor                 # [1, C, T, H, W] decoded latent / pixel tensor


def generate(
    *,
    dynamics: MultimodalDynamicsModel,
    decoder: DiTDecoder,
    text_ids: list[int] | None,
    layout: VocabLayout,
    config: GenerationConfig,
) -> GenerationResult:
    """End-to-end generation: rollout tokens then decode to pixels."""

    tokens = rollout_visual_tokens(
        dynamics, text_ids=text_ids, config=config, layout=layout,
    )
    pixels = decode_tokens_to_pixels(
        decoder, token_ids=tokens, config=config,
    )
    return GenerationResult(visual_tokens=tokens, pixels=pixels)


__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "decode_tokens_to_pixels",
    "generate",
    "rollout_visual_tokens",
]
