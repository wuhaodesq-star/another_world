"""Inference / rollout entry points (stages 4 and 8)."""

from another_world.inference.first_frame import (
    FirstFramePreprocessor,
    FirstFrameTokenizer,
    MockFirstFrameTokenizer,
    load_image_as_video_tensor,
    load_video_as_video_tensor,
)
from another_world.inference.generation import (
    GenerationConfig,
    GenerationResult,
    decode_tokens_to_pixels,
    generate,
    rollout_visual_tokens,
)

__all__ = [
    "FirstFramePreprocessor",
    "FirstFrameTokenizer",
    "GenerationConfig",
    "GenerationResult",
    "MockFirstFrameTokenizer",
    "decode_tokens_to_pixels",
    "generate",
    "load_image_as_video_tensor",
    "load_video_as_video_tensor",
    "rollout_visual_tokens",
]
