#!/bin/bash
# Host the DSRL-finetuned GR00T N1.7 checkpoint (VRH3 / VFE vacuum pick) as a policy server.
# The initial noise is drawn from pi^W(s); per-request options: use_dsrl=False, dsrl_deterministic=True.
#
# Usage: bash run_scripts/serve_dsrl.sh [PORT] [CHECKPOINT_STEP]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Conda env used for this codebase
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gr00tn1d7_rl

PORT="${1:-5555}"
STEP="${2:-20000}"
RUN_DIR="/home/lmaotan/Documents/baoht9/checkpoints/gr00tn1d7-vrh3-vfe-vacuum-pick-dsrl-2026-10-07_16-32-52"
MODEL_PATH="$RUN_DIR/checkpoint-$STEP"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONWARNINGS="ignore"

python gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag new_embodiment \
    --device cuda \
    --host 0.0.0.0 \
    --port "$PORT"
