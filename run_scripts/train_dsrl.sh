source /home/baoht9/Isaac-GR00T/venv/bin/activate
export WANDB_API_KEY="938a809d45af5c62586335955d205d25d0db5d04"
export HF_HOME="/mnt/data/sftp/data/baoht9/"
export PYTHONWARNINGS="ignore"

BASE_MODEL="/mnt/data/sftp/data/vla/vr_checkpoints/gr00t_n17_baseline_bds_0110_35000_64/gr00t_n17_baseline_bds_0110_35000_64"
MODALITY_PATH="/home/baoht9/Isaac-GR00T/demo_data/modalities/baoht-vrh3/modality.json"
OUTPUT_DIR="/mnt/data/sftp/data/vla/vr_checkpoints/gr00tn1d7-vrh3-vfe-vacuum-pick-dsrl"

# Offline DSRL needs both failed and successful episodes: next.reward < 0 marks a failure.
DATA_PICK1="/mnt/data/sftp/data/vla/data_sim_ac/20260928_VR_H5D_VFE_sim_pick_fail"
DATA_PICK2="/mnt/data/sftp/data/vla/data_sim_ac/20260929_VR_H5D_VFE_sim_pick_fail"
DATA_PICK3="/mnt/data/sftp/data/vla/data_sim_ac/20260928_VR_H5D_VFE_sim_pick_success"
DATA_PICK4="/mnt/data/sftp/data/vla/data_sim_ac/20260929_VR_H5D_VFE_sim_pick_success"
DATA_PICK5="/mnt/data/sftp/data/vla/data_sim_ac/20261001_VR_H5D_VFE_sim_teleop_pick_success"

for d in "$DATA_PICK1" "$DATA_PICK2" "$DATA_PICK3" "$DATA_PICK4" "$DATA_PICK5"; do
    cp "$MODALITY_PATH" "$d/meta/modality.json"
done

bash examples/dsrl.sh \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATA_PICK5" \
    --embodiment-tag new_embodiment \
    --modality-config-path data_config/vrh3_both_arm_3cam.py \
    --output-dir "$OUTPUT_DIR-$(date +"%Y-%m-%d_%H-%M-%S")" \
    --max-steps 20000 \
    --global-batch-size 64 \
    --video_backend torchvision_av \
    --use_wandb \
    --wandb_project "GR00TN1.7 RECAP"
