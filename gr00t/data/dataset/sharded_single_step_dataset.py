# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig, VLAStepData

from .lerobot_episode_loader import LeRobotEpisodeLoader


def extract_step_data(
    episode_data: tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray],
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    allow_padding: bool = False,
) -> VLAStepData:
    episode_data, video_data, mask_data, all_step_indices_video, all_step_indices_mask = episode_data
    step_data = {}

    # Extract data for each configured modality
    for modality, config in modality_configs.items():
        step_data[modality] = {}
        # Sample timesteps according to delta indices configuration
        indices_to_load = [step_index + delta_index for delta_index in config.delta_indices]
        # TODO: support allow_padding=True
        if allow_padding:
            indices_to_load = [max(0, min(idx, len(episode_data) - 1)) for idx in indices_to_load]

        # Reward modality: synthetic keys + optional parquet column (modality.json maps e.g.
        # "reward" -> original_key "next.reward"). A missing column means all-success (0.0).
        if modality == "reward":
            episode_length = len(episode_data)
            n = len(config.delta_indices)
            for key in config.modality_keys:
                if key == "reward.current_frame_idx":
                    step_data[modality][key] = np.array(
                        [max(0, min(step_index + d, episode_length - 1)) for d in config.delta_indices],
                        dtype=np.float32,
                    )
                elif key == "reward.episode_lengths":
                    step_data[modality][key] = np.full(n, episode_length, dtype=np.float32)
                else:
                    col = key if key.startswith("reward.") else f"reward.{key}"
                    if col in episode_data.columns:
                        modality_data = episode_data[col].iloc[indices_to_load]
                        step_data[modality][key] = np.asarray(modality_data.tolist(), dtype=np.float32)
                    else:
                        step_data[modality][key] = np.zeros(n, dtype=np.float32)
            continue

        for key in config.modality_keys:
            if f"{modality}.{key}" in episode_data.columns:
                modality_data = episode_data[f"{modality}.{key}"].iloc[indices_to_load]
            elif modality == "video" and key in video_data:
                assert not np.in1d(indices_to_load, all_step_indices_video, invert=True).any()
                modality_data = video_data[key][np.searchsorted(all_step_indices_video, indices_to_load)]
            elif modality == "mask" and key in mask_data:
                assert not np.in1d(indices_to_load, all_step_indices_mask, invert=True).any()
                modality_data = mask_data[key][np.searchsorted(all_step_indices_mask, indices_to_load)]
            else:
                raise KeyError(
                    f"{modality}.{key} not found in episode data, available keys: {episode_data.columns}"
                )
            if modality in ["state", "action"]:
                # Stack arrays for numerical modalities
                # step_data[modality][key] = np.vstack(
                #     [
                #         np.array(modality_data.iloc[i]).astype(np.float32)
                #         for i in range(len(modality_data))
                #     ]
                # )
                # vectorize
                step_data[modality][key] = np.asarray(modality_data.tolist(), dtype=np.float32)
            elif modality in ["video", "mask"]:
                step_data[modality][key] = modality_data
            else:
                # Keep as lists for other modalities (video, language)
                step_data[modality][key] = modality_data.tolist()

    # Parse extracted data into VLAStepData structure
    video_data = step_data.get("video", {})
    mask_data = step_data.get("mask", {})
    state_data = step_data.get("state", {})
    action_data = step_data.get("action", {})
    language_data = step_data.get("language", {})
    assert len(language_data) == 1, f"Expected 1 language, got {len(language_data)}"
    text = language_data[list(language_data.keys())[0]][0]
    reward_data = step_data.get("reward", {})

    vla_step_data = VLAStepData(
        images=video_data,
        masks=mask_data if mask_data else None,
        states=state_data,
        actions=action_data,
        text=text,
        embodiment=embodiment_tag,
        metadata={"reward": reward_data} if reward_data else {},
    )
    return vla_step_data


class ShardedSingleStepDataset(ShardedDataset):
    """
    Single-step dataset that creates shards from individual timesteps across episodes.

    This dataset implementation provides step-level data access for VLA training by:
    1. Loading episodes using LeRobotEpisodeLoader
    2. Splitting episodes into individual timesteps
    3. Organizing timesteps into balanced shards for efficient loading
    4. Supporting episode subsampling for data efficiency

    The sharding strategy ensures balanced shard sizes while maintaining randomization
    across episodes and timesteps within episodes. Each shard contains a mix of
    timesteps from different episodes to improve training diversity.

    Key features:
    - Step-level data access (vs episode-level)
    - Balanced sharding for consistent batch sizes
    - Episode subsampling via sampling rate
    - Integration with LeRobot data format
    - Support for multi-modal data (video, state, action, language)

    Args:
        dataset_path: Path to LeRobot format dataset directory
        embodiment_tag: Embodiment identifier for cross-embodiment training
        modality_configs: Configuration for each modality (sampling, keys)
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for video backend
        shard_size: Target number of timesteps per shard
        episode_sampling_rate: Fraction of episode timesteps to use (for efficiency)
        seed: Random seed for reproducible sharding and sampling
        allow_padding: Whether to allow padding of indices to valid range [0, max_length - 1]

    Example:
        >>> dataset = ShardedSingleStepDataset(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.FRANKA,
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(8)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ...     shard_size=1024,
        ...     episode_sampling_rate=0.1,
        ... )
        >>> shard_data = dataset.get_shard(0)  # Get first shard of processed timesteps
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        shard_size: int = 2**10,  # 1024 steps
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
        shard_load_workers: int = 1,
        video_decode_workers: int = 1,
        num_ffmpeg_threads: int = 0,
        overlap_episode_io: bool = False,
        dsrl_n_step: int = 0,
    ):
        """Initialize single-step dataset with sharding configuration."""
        super().__init__(dataset_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.allow_padding = allow_padding
        self.shard_load_workers = max(1, shard_load_workers)
        self.video_decode_workers = max(1, video_decode_workers)
        self.num_ffmpeg_threads = num_ffmpeg_threads
        self.overlap_episode_io = overlap_episode_io
        # DSRL: every datapoint additionally carries the observation `dsrl_n_step` steps ahead.
        self.dsrl_n_step = dsrl_n_step
        self.processor = None
        self.rng = np.random.default_rng(seed)
        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            video_decode_workers=self.video_decode_workers,
            num_ffmpeg_threads=self.num_ffmpeg_threads,
            overlap_episode_io=self.overlap_episode_io,
            next_step_offset=self.dsrl_n_step,
        )

        # Create balanced shards from episode timesteps
        self.shard_dataset()

    def shard_dataset(self):
        """
        Create balanced shards by distributing episode timesteps across shards.

        The sharding process:
        1. Shuffle episode order for randomization
        2. Split each episode into multiple sub-sequences based on sampling rate
        3. Distribute sub-sequences across shards to balance shard sizes
        4. Use greedy assignment to minimize shard size variance

        This approach ensures:
        - Balanced shard sizes for consistent training batches
        - Diversity within shards (mix of episodes and timesteps)
        - Reproducible sharding based on seed
        """
        shuffled_episode_indices = self.rng.permutation(len(self.episode_loader.episode_lengths))
        num_splits = int(1 / self.episode_sampling_rate)

        assert len(shuffled_episode_indices) > 0, (
            f"No valid trajectories found for dataset {self.dataset_path}"
        )

        # Calculate total timesteps and required number of shards
        total_steps = np.sum(
            [self.get_effective_episode_length(idx) for idx in shuffled_episode_indices]
        ).astype(int)
        num_shards = np.ceil(total_steps / self.shard_size).astype(int)

        # Initialize shard containers
        sharded_episodes = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        # Distribute episode sub-sequences across shards
        for ep_idx in shuffled_episode_indices:
            # Split episode timesteps into multiple sub-sequences
            step_indices = np.arange(0, self.get_effective_episode_length(ep_idx))
            self.rng.shuffle(step_indices)
            for i in range(num_splits):
                split_step_indices = step_indices[i::num_splits]
                # Assign to shard with minimum current length (greedy balancing)
                shard_index = np.argmin(shard_lengths)
                sharded_episodes[shard_index].append((ep_idx, split_step_indices))
                shard_lengths[shard_index] += len(split_step_indices)

        # Validate shard creation
        assert all(shard_lengths[i] > 0 for i in range(num_shards)), (
            "All shards must have length greater than 0"
        )

        print(f"Generated {num_shards} shards for dataset {self.dataset_path}")
        print(
            f"Total steps: {total_steps}, average shard length: {total_steps / num_shards}, shard length std: {np.std(shard_lengths)}"
        )
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths

    def get_effective_episode_length(self, episode_index: int) -> int:
        """Get the effective episode length accounting for action horizon."""
        original_length = self.episode_loader.get_episode_length(episode_index)
        return max(0, original_length - self.action_horizon + 1)

    def __len__(self):
        """Return the number of shards in the dataset."""
        return len(self.shard_lengths)

    def get_datapoint(
        self, 
        episode_data: tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray], 
        step_index: int,
    ) -> dict:
        """
        Extract and process a single timestep from episode data.

        Converts raw episode data into a VLAStepData structure and applies
        the configured processor to create model-ready inputs.

        Args:
            episode_data: Complete episode DataFrame from LeRobotEpisodeLoader
            step_index: Timestep index within the episode to extract

        Returns:
            Processed datapoint ready for model training

        Raises:
            AssertionError: If processor is not set before calling this method
        """
        assert self.processor is not None, "Processor must be set before getting datapoints"
        vla_step_data = extract_step_data(
            episode_data,
            step_index,
            self.modality_configs,
            self.embodiment_tag,
            self.allow_padding,
        )
        # Apply processor to convert to model inputs
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        datapoint = self.processor(messages)
        if self.dsrl_n_step > 0:
            datapoint.update(self._get_next_observation(episode_data, step_index))
        return datapoint

    def _get_next_observation(self, episode_data: tuple, step_index: int) -> dict:
        """Process the observation at min(t + dsrl_n_step, T - 1) as the DSRL transition target.

        The terminal flag is derived downstream from reward.current_frame_idx / episode_lengths,
        so clamping at the last frame is harmless.
        """
        episode_length = len(episode_data[0])
        next_index = min(step_index + self.dsrl_n_step, episode_length - 1)
        obs_configs = {
            m: ModalityConfig(delta_indices=c.delta_indices, modality_keys=c.modality_keys)
            for m, c in self.modality_configs.items()
            if m in ("video", "state", "language", "mask")
        }
        next_step = extract_step_data(
            episode_data, next_index, obs_configs, self.embodiment_tag, self.allow_padding
        )
        return self.processor.process_next_step(next_step)

    def get_shard_length(self, idx: int) -> int:
        """Get the number of timesteps in a specific shard."""
        return self.shard_lengths[idx]

    def get_shard(self, idx: int) -> list:
        """
        Load and process all timesteps in a specific shard.

        Loads the required episodes and extracts all timesteps assigned to this shard,
        applying the configured processor to each timestep.

        Args:
            idx: Shard index to load

        Returns:
            List of processed timesteps ready for model training
        """
        episodes = self.sharded_episodes[idx]
        datapoints = []

        if self.shard_load_workers > 1:
            with ThreadPoolExecutor(max_workers=self.shard_load_workers) as executor:
                inflight = deque()
                i = 0

                while len(inflight) < self.shard_load_workers and i < len(episodes):
                    inflight.append((
                        episodes[i],
                        executor.submit(
                            self.episode_loader.__getitem__, 
                            episodes[i][0], 
                            episodes[i][1],
                        ),
                    ))
                    i += 1

                while inflight:
                    (ep_idx, step_indices), fut = inflight.popleft()
                    episode_data = fut.result()

                    if i < len(episodes):
                        inflight.append((
                            episodes[i],
                            executor.submit(
                                self.episode_loader.__getitem__, 
                                episodes[i][0], 
                                episodes[i][1],
                            ),
                        ))
                        i += 1

                    for step_index in step_indices:
                        datapoints.append(self.get_datapoint(episode_data, step_index))

            return datapoints
 
        for ep_idx, step_indices in episodes:
            # Load episode data once per episode in shard
            episode_data = self.episode_loader.__getitem__(ep_idx, step_indices)
            for step_index in step_indices:
                datapoints.append(self.get_datapoint(episode_data, step_index))
        return datapoints

    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics from the underlying episode loader."""
        return self.episode_loader.get_dataset_statistics()

    def get_initial_actions(self):
        """Get initial actions from the underlying episode loader."""
        return self.episode_loader.get_initial_actions()
