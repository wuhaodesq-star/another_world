"""Data filters: aesthetic / watermark / NSFW / dedup.

The :mod:`pipeline` module contains the core ``Filter`` protocol,
heuristic implementations, and the ``FilterPipeline`` composer. Stage 1.2
will replace heuristics with real ML-based filters (LAION aesthetic
predictor, watermark CNN, safety classifier) by registering new classes
that satisfy the same ``Filter`` protocol.
"""

from another_world.data.filters.pipeline import (
    AestheticFilter,
    AspectRatioFilter,
    CallableFilter,
    DedupFilter,
    Filter,
    FilterPipeline,
    FilterStats,
    LicenseFilter,
    MinDurationFilter,
    MinResolutionFilter,
)

__all__ = [
    "AestheticFilter",
    "AspectRatioFilter",
    "CallableFilter",
    "DedupFilter",
    "Filter",
    "FilterPipeline",
    "FilterStats",
    "LicenseFilter",
    "MinDurationFilter",
    "MinResolutionFilter",
]
