#!/usr/bin/env python
"""Stub for checkpoint evaluation (stage 5)."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--suite", default="default")
    args = parser.parse_args()
    print(f"[eval] checkpoint={args.checkpoint!r} suite={args.suite!r}")
    print("[eval] full evaluation suite is implemented in stage 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
