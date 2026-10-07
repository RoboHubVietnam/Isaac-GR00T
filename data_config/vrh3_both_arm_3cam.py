from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Matches the VR_H5D modality.json: two cameras, joint-space arms + 1-dof vacuum hands.
# The *_eff state groups in modality.json are not used.
vrh3_both_arm_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_front", "cam_outside"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "left_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=[i * 2 for i in range(16)],
        modality_keys=[
            "left_arm",
            "left_hand",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
    # "reward.current" -> next.reward via modality.json; the other two are synthesized per sample.
    "reward": ModalityConfig(
        delta_indices=[0],
        modality_keys=["reward.current", "reward.current_frame_idx", "reward.episode_lengths"],
    ),
}

register_modality_config(vrh3_both_arm_config)
