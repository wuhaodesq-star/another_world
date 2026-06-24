"""Eval entry points. Stage 5 will populate metric suites."""

from __future__ import annotations

import argparse
import sys

from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aw-eval",
        description="Evaluate a checkpoint (placeholder; populated in stage 5).",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args(argv)
    _LOG.info("aw-eval is a placeholder; will be implemented in stage 5. "
              "Received checkpoint=%s", args.checkpoint)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
