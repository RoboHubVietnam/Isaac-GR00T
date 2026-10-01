#!/bin/bash
# ============================================================
# DSRL-NA offline training for GR00T N1.5 — vrc2 pick-place
# Reference: "Steering Your Diffusion Policy with Latent Space RL", arXiv:2506.15799 (Alg. 1)
#
# Single phase (replaces RECAP's value-head + policy phases):
#   backbone + DiT are FROZEN; trains Q^A (TD), Q^W (distilled), pi^W (latent-noise actor)
#   on the offline dataset. Reuses the RECAP `reward` section of modality.json.
#
# Usage: bash scripts/train_dsrl_vrc2.sh
# ============================================================
set -euo pipefail

source /home/baoht9/Isaac-GR00T/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export WANDB_DIR=/mnt/data/sftp/data/baoht9/wandb   # set WANDB_API_KEY in your shell, not here
export HF_HOME=/mnt/data/sftp/data/baoht9/hf
export WANDB_PROJECT="vrc2_pick_place-DSRL"

BASE_MODEL=nvidia/GR00T-N1.5-3B          # or your finetuned N1.5 checkpoint (the policy being steered)
MODALITY=/mnt/data/sftp/data/vla/vrc2_tcp/modality.json   # must contain a `reward` section
DATA1=/mnt/data/sftp/data/vla/vrc2_tcp/pnp_cup/pnp_cup_20260622
DATA_CONFIG="${DATA_CONFIG:-gr00t.experiment.data_config:VRH31LeftArm2Cam}"   # <-- set to your vrc2 config
OUT=/mnt/data/sftp/data/vla/vr_checkpoints/gr00t_n15_vrc2_pnp_cup_dsrl

cp "$MODALITY" "$DATA1/meta/modality.json"

python scripts/gr00t_finetune.py \
    --dataset-path "$DATA1" \
    --data-config "$DATA_CONFIG" \
    --embodiment-tag new_embodiment \
    --base-model-path "$BASE_MODEL" \
    --output-dir "$OUT" \
    --num-gpus 1 \
    --batch-size 64 \
    --max-steps 20000 \
    --save-steps 5000 \
    --learning-rate 3e-4 \
    --weight-decay 0 \
    --warmup-ratio 0.02 \
    --dataloader-num-workers 4 \
    --report-to wandb \
    --use-dsrl \
    --dsrl-n-step 16 \
    --dsrl-noise-bound 1.0 \
    --dsrl-noise-mode full
