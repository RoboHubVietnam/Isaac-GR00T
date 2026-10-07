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

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import barrier


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def _build_single_step_dataset(self, dataset_path: str, embodiment_tag: str) -> ShardedSingleStepDataset:
        assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                generate_stats(dataset_path)
                generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
        else:
            generate_stats(dataset_path)
            generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
        barrier()
        return ShardedSingleStepDataset(
            dataset_path=dataset_path,
            embodiment_tag=EmbodimentTag(embodiment_tag),
            modality_configs=self.config.data.modality_configs[embodiment_tag],
            video_backend=self.config.data.video_backend,
            shard_size=self.config.data.shard_size,
            episode_sampling_rate=self.config.data.episode_sampling_rate,
            seed=self.config.data.seed,
            allow_padding=self.config.data.allow_padding,
            shard_load_workers=self.config.data.shard_load_workers,
            video_decode_workers=self.config.data.video_decode_workers,
            num_ffmpeg_threads=self.config.data.num_ffmpeg_threads,
            overlap_episode_io=self.config.data.overlap_episode_io,
            dsrl_n_step=self.config.data.dsrl_n_step,
        )

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        all_datasets = []
        all_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                datasets.append(self._build_single_step_dataset(dataset_path, embodiment_tag))
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        train_dataset = ShardedMixtureDataset(
            datasets=all_datasets,
            weights=all_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
        )

        eval_dataset = None
        val_paths = getattr(self.config.data, "val_dataset_paths", [])
        if self.config.training.eval_strategy != "no" and val_paths:
            # Use the embodiment tag from the first training dataset spec
            embodiment_tag = self.config.data.datasets[0].embodiment_tag
            assert embodiment_tag is not None, "Embodiment tag is required for validation"
            val_datasets = []
            for val_path in tqdm(val_paths, desc="Initializing validation datasets"):
                val_datasets.append(self._build_single_step_dataset(val_path, embodiment_tag))
            val_weights = [1.0 / len(val_datasets)] * len(val_datasets)
            eval_dataset = ShardedMixtureDataset(
                datasets=val_datasets,
                weights=val_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=False,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            )

        return train_dataset, eval_dataset
