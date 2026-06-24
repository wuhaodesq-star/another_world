"""Logging helpers shared across training, evaluation, and data tooling."""

from __future__ import annotations

import logging
import sys
from typing import Final

_DEFAULT_FORMAT: Final = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    We attach a single stderr handler so the same logger is safe to
    call multiple times (idempotent).
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


__all__ = ["get_logger"]
