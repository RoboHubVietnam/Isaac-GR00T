#!/bin/bash
# Offline DSRL-NA on top of a (fine-tuned) GR00T N1.7 policy.
#   "Steering Your Diffusion Policy with Latent Space RL", arXiv:2506.15799, Algorithm 1
#
# The policy is frozen; Q^A (TD), Q^W (distilled) and the latent-noise actor pi^W are trained on the
# offline dataset. Rewards come from the dataset's `next.reward` column (RECAP convention: < 0 marks
# a failed episode). Optionally map it explicitly in meta/modality.json:
#   "reward": {"reward": {"original_key": "next.reward"}}
#
# Usage:
#   bash examples/dsrl.sh --base-model-path <ckpt> --dataset-path <ds> --embodiment-tag new_embodiment \
#       --modality-config-path data_config/vrh3_both_arm_3cam.py --output-dir <out>
#
# Serving: point the usual inference server at <out>/checkpoint-*; the initial noise is drawn from
# pi^W(s). Per-request options: use_dsrl=False (plain N(0, I)), dsrl_deterministic=True.
set -euo pipefail

BASE_MODEL_PATH=""
DATASET_PATH=""
EMBODIMENT_TAG="new_embodiment"
MODALITY_CONFIG_PATH=""
OUTPUT_DIR=""
NUM_GPUS=1
MAX_STEPS=20000
SAVE_STEPS=5000
GLOBAL_BATCH_SIZE=64
LEARNING_RATE=3e-4
DSRL_N_STEP=16
DSRL_NOISE_BOUND=1.5
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-model-path) BASE_MODEL_PATH="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --embodiment-tag) EMBODIMENT_TAG="$2"; shift 2 ;;
        --modality-config-path) MODALITY_CONFIG_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --save-steps) SAVE_STEPS="$2"; shift 2 ;;
        --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
        --dsrl-n-step) DSRL_N_STEP="$2"; shift 2 ;;
        --dsrl-noise-bound) DSRL_NOISE_BOUND="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

for var in BASE_MODEL_PATH DATASET_PATH OUTPUT_DIR; do
    [ -n "${!var}" ] || { echo "missing --${var//_/-}" >&2; exit 1; }
done

CMD=(
    gr00t/experiment/launch_dsrl.py
    --base_model_path "$BASE_MODEL_PATH"
    --dataset_path "$DATASET_PATH"
    --embodiment_tag "$EMBODIMENT_TAG"
    --num_gpus "$NUM_GPUS"
    --output_dir "$OUTPUT_DIR"
    --max_steps "$MAX_STEPS"
    --save_steps "$SAVE_STEPS"
    --global_batch_size "$GLOBAL_BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --weight_decay 0
    --warmup_ratio 0.02
    --dsrl_n_step "$DSRL_N_STEP"
    --dsrl_noise_bound "$DSRL_NOISE_BOUND"
)
[ -z "$MODALITY_CONFIG_PATH" ] || CMD+=(--modality_config_path "$MODALITY_CONFIG_PATH")

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" "${CMD[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
else
    python "${CMD[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi
