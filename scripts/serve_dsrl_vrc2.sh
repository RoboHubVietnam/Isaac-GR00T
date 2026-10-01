#!/bin/bash
# DSRL inference server (GR00T N1.5): initial flow noise is drawn from the learned pi^W(s).
# Usage: bash scripts/serve_dsrl_vrc2.sh [ckpt_dir]      env: PORT, DATA_CONFIG
set -euo pipefail
source /home/baoht9/Isaac-GR00T/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/data/sftp/data/baoht9/hf HF_HUB_OFFLINE=1

OUT=/mnt/data/sftp/data/vla/vr_checkpoints/gr00t_n15_vrc2_pnp_cup_dsrl
MODEL_PATH="${1:-$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)}"
DATA_CONFIG="${DATA_CONFIG:-gr00t.experiment.data_config:VRH31LeftArm2Cam}"

python scripts/inference_service.py --server \
    --model-path "$MODEL_PATH" \
    --embodiment-tag new_embodiment \
    --data-config "$DATA_CONFIG" \
    --port "${PORT:-5555}"
