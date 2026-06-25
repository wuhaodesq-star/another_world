"""V-JEPA-style latent predictor (auxiliary loss)."""

from another_world.models.jepa.predictor import (
    EmaShadow,
    JEPAConfig,
    JEPALatentPredictor,
    jepa_loss,
)

__all__ = [
    "EmaShadow",
    "JEPAConfig",
    "JEPALatentPredictor",
    "jepa_loss",
]
