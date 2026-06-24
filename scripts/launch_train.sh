#!/usr/bin/env bash
# SLURM launch template for Another World training.
#
# Usage:
#   sbatch scripts/launch_train.sh configs/train/smoke.yaml
#
# Adjust partition, account, and node count for your cluster.

#SBATCH --job-name=aw-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

set -euo pipefail

CONFIG="${1:-configs/train/smoke.yaml}"

echo "[launch] config=${CONFIG}"
echo "[launch] nodes=${SLURM_NNODES:-1} ntasks=${SLURM_NTASKS:-1}"
echo "[launch] host=$(hostname) date=$(date -u +%FT%TZ)"

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

export MASTER_ADDR MASTER_PORT
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p logs/slurm

srun --label python -m another_world.training.cli \
    --steps 100 \
    --batch-size 4 \
    --device auto \
    --precision bf16
