#!/usr/bin/env bash
# Lambda Labs bootstrap script for Another World.
#
# Run this on a fresh Lambda Labs instance to install all system + Python
# dependencies and clone the repo. After this completes you can launch
# training jobs with ``aw-train-mm`` / ``aw-train-dit`` (single-node) or
# torchrun for multi-GPU.
#
# Usage on the box:
#   curl -fsSL https://raw.githubusercontent.com/wuhaodesq-star/another_world/main/scripts/bootstrap_lambda.sh \
#        | bash -s -- /workspace
#
# The single argument is the workspace directory (defaults to /workspace).

set -euo pipefail

WORKSPACE="${1:-/workspace}"
REPO_URL="${AW_REPO_URL:-https://github.com/wuhaodesq-star/another_world.git}"
BRANCH="${AW_BRANCH:-main}"
PYTORCH_INDEX_URL="${AW_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

log() {
  echo "[bootstrap] $*"
}

log "workspace=${WORKSPACE} repo=${REPO_URL} branch=${BRANCH}"

# --- system deps --------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  log "installing system packages"
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    build-essential \
    git git-lfs \
    ffmpeg \
    libgl1 libglib2.0-0 \
    tmux htop \
    python3.10 python3.10-venv python3.10-dev \
    python3-pip
fi

# --- clone --------------------------------------------------------------
mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"
if [ ! -d another_world ]; then
  log "cloning ${REPO_URL}"
  git clone "${REPO_URL}" another_world
fi
cd another_world
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}"

# --- python env --------------------------------------------------------
if [ ! -d .venv ]; then
  log "creating venv"
  python3.10 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

log "installing pytorch (cu124) + project"
pip install \
  --index-url "${PYTORCH_INDEX_URL}" \
  torch==2.4.1 torchvision==0.19.1
pip install -e ".[dev,train,data]"

# Optional but useful on the cluster.
pip install --upgrade \
  wandb safetensors webdataset av huggingface_hub boto3 \
  transformers || true

# --- gpu check ---------------------------------------------------------
log "torch / cuda status:"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}:", torch.cuda.get_device_name(i))
PY

log "running aw-doctor"
aw-doctor || true

log "done. activate via: source ${WORKSPACE}/another_world/.venv/bin/activate"
