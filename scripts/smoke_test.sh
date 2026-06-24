#!/usr/bin/env bash
# Local smoke test that runs the toy trainer for a handful of steps on CPU.
# Intended for developer laptops and CI.

set -euo pipefail

python -m another_world.training.cli \
    --steps 20 \
    --batch-size 2 \
    --seq-len 32 \
    --vocab-size 128 \
    --dim 64 \
    --n-layers 2 \
    --n-heads 4 \
    --n-kv-heads 2 \
    --device cpu \
    --precision fp32 \
    --seed 0
