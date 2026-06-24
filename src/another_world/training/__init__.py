"""Training entry points (smoke trainer for stage 0; FSDP/Megatron later)."""

from another_world.training.smoke import (
    SmokeTrainerConfig,
    TrainStepResult,
    run_smoke_training,
)

__all__ = ["SmokeTrainerConfig", "TrainStepResult", "run_smoke_training"]
