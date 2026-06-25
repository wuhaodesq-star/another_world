"""Visual tokenizer wrappers.

Stage 1 defaults to NVIDIA Cosmos-Tokenizer (CV8x8x8 / DV8x16x16); a local
re-implementation of MAGVIT-v2 may be added later for research comparisons.
"""

from another_world.tokenizers.visual.cosmos import (
    COSMOS_REGISTRY,
    CosmosCheckpoints,
    CosmosModelSpec,
    CosmosVideoTokenizer,
    DEFAULT_VIDEO_MODEL,
    download_cosmos_checkpoint,
    expected_latent_shape,
    get_spec,
    list_models,
    validate_video_input,
)

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
