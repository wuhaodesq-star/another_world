"""Training entry points (smoke trainer for stage 0; multimodal for stage 3)."""

from another_world.training.checkpoint import (
    CheckpointMeta,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from another_world.training.distributed_wrap import (
    FsdpConfig,
    WrapResult,
    wrap_model_for_distributed,
)
from another_world.training.multimodal import (
    MultimodalStepResult,
    MultimodalTrainerConfig,
    apply_activation_checkpointing,
    build_optimizer,
    run_multimodal_training,
)
from another_world.training.smoke import (
    SmokeTrainerConfig,
    TrainStepResult,
    run_smoke_training,
)

__all__ = [
    "CheckpointMeta",
    "FsdpConfig",
    "MultimodalStepResult",
    "MultimodalTrainerConfig",
    "SmokeTrainerConfig",
    "TrainStepResult",
    "WrapResult",
    "apply_activation_checkpointing",
    "build_optimizer",
    "find_latest_checkpoint",
    "load_checkpoint",
    "run_multimodal_training",
    "run_smoke_training",
    "save_checkpoint",
    "wrap_model_for_distributed",
]
