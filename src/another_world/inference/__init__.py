"""Inference / rollout entry points (stages 4 and 8)."""

from another_world.inference.generation import (
    GenerationConfig,
    GenerationResult,
    decode_tokens_to_pixels,
    generate,
    rollout_visual_tokens,
)

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "decode_tokens_to_pixels",
    "generate",
    "rollout_visual_tokens",
]
