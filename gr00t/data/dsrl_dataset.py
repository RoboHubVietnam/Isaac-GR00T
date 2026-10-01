# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline DSRL transitions: each item carries the current sample plus the observation N steps later."""

from gr00t.data.dataset import LeRobotSingleDataset

# keys of the transformed "next" sample that DSRL needs (prefixed with `next_`)
NEXT_KEYS = ("eagle_content", "state", "state_mask")


class DSRLLeRobotSingleDataset(LeRobotSingleDataset):
    """
    Returns {**sample(t), next_eagle_content, next_state, next_state_mask} where the next sample is
    taken at min(t + n_step, T - 1). Terminal handling is done in the model from
    `reward.current_frame_idx` / `reward.episode_lengths`, so clamping here is harmless.
    """

    def __init__(self, *args, n_step: int = 16, **kwargs):
        self.n_step = n_step
        super().__init__(*args, **kwargs)

    def get_transformed_item(self, trajectory_id: int, base_index: int) -> dict:
        cur = self.transforms(self.get_step_data(trajectory_id, base_index))
        length = self.trajectory_lengths[self.get_trajectory_index(trajectory_id)]
        next_base = min(base_index + self.n_step, int(length) - 1)
        nxt = self.transforms(self.get_step_data(trajectory_id, next_base))
        for k in NEXT_KEYS:
            cur["next_" + k] = nxt[k]
        return cur
