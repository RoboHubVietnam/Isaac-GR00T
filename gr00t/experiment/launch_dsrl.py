#!/usr/bin/env python3
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

# Offline DSRL-NA launcher for GR00T N1.7.
#
# The (fine-tuned) GR00T policy is frozen and treated as a black-box map from initial flow noise
# w to an action chunk. Three small heads are trained on the offline LeRobot dataset:
#
#   Q^A(s, a)   action critic, TD-learned on (s_t, a_{t:t+H}, R, s_{t+n_step})
#   Q^W(s, w)   noise critic, distilled from Q^A through the frozen denoiser
#   pi^W(s)     latent-noise actor, maximises Q^W
#
# Rewards come from the dataset (`next.reward`, RECAP convention: < 0 marks a failed episode),
# turned into -1 per step / 0 on success / -c_fail on failure (RECAP Eq. 5).
#
# At inference, load the resulting checkpoint with the usual policy server: the initial noise is
# then drawn from pi^W(s) instead of N(0, I).
#
# Reference: "Steering Your Diffusion Policy with Latent Space RL", arXiv:2506.15799, Algorithm 1.

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from transformers import TrainingArguments, set_seed
import tyro
import wandb

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.trainer import Gr00tTrainer
from gr00t.experiment.utils import CheckpointFormatCallback
from gr00t.model import MODEL_REGISTRY
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, save_initial_actions


def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def infer_action_dim(dataset_path: str, action_keys: list[str]) -> int:
    """Sum of the raw action widths of `action_keys` according to meta/modality.json."""
    with open(Path(dataset_path) / "meta" / "modality.json") as f:
        action_meta = json.load(f)["action"]
    return sum(action_meta[k]["end"] - action_meta[k]["start"] for k in action_keys)


@dataclass
class DSRLFinetuneConfig(FinetuneConfig):
    """FinetuneConfig plus DSRL-NA hyperparameters. Policy tune_* flags are ignored (frozen)."""

    dsrl_n_step: int = 16
    """Environment steps executed per decision; the TD transition is (s_t, a_t, R, s_{t+n_step})."""

    dsrl_noise_bound: float = 1.5
    """b_W: actor outputs live in [-b, b] (paper App. B 'action magnitude', ~1.5)."""

    dsrl_noise_mode: str = "full"
    """'full': w in R^{H*D}; 'chunk_repeat': w in R^D repeated over the horizon."""

    dsrl_discount: float = 0.99
    dsrl_tau: float = 0.005
    dsrl_hidden_dim: int = 512
    dsrl_n_noise_samples: int = 4
    """Prior-noise samples per step for Q^W distillation; each costs one full denoising pass."""

    dsrl_target_entropy: float = 0.0
    dsrl_init_alpha: float = 0.1

    dsrl_max_episode_length: int = 520
    dsrl_c_fail: float = 260.0
    dsrl_reward_scale: float = 1.0

    video_backend: str = "torchcodec"
    """Video decoder (e.g. torchvision_av for AV1 videos)."""

    dsrl_action_dim: int = 0
    """Steered action dims. 0 -> inferred from the dataset's modality.json."""


def run_dsrl(ft_config: DSRLFinetuneConfig):
    # ----- distributed setup ---------------------------------------------------
    if dist.is_initialized():
        global_rank = dist.get_rank()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    if global_rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    set_seed(42)

    # ----- build config --------------------------------------------------------
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                    for dataset_path in ft_config.dataset_path
                ],
            }
        }
    )
    config.load_config_path = None

    # Policy is frozen: no tuning, and no state dropout / augmentation (the critics must see the
    # observation distribution of inference).
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = False
    config.model.tune_diffusion_model = False
    config.model.tune_vlln = False
    config.model.state_dropout_prob = 0.0

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    action_cfg = config.data.modality_configs[embodiment_tag]["action"]
    action_dim = ft_config.dsrl_action_dim or infer_action_dim(
        ft_config.dataset_path[0], action_cfg.modality_keys
    )
    dsrl_fields = {
        "use_dsrl": True,
        "dsrl_hidden_dim": ft_config.dsrl_hidden_dim,
        "dsrl_discount": ft_config.dsrl_discount,
        "dsrl_tau": ft_config.dsrl_tau,
        "dsrl_noise_bound": ft_config.dsrl_noise_bound,
        "dsrl_target_entropy": ft_config.dsrl_target_entropy,
        "dsrl_init_alpha": ft_config.dsrl_init_alpha,
        "dsrl_n_noise_samples": ft_config.dsrl_n_noise_samples,
        "dsrl_noise_mode": ft_config.dsrl_noise_mode,
        "dsrl_active_horizon": len(action_cfg.delta_indices),
        "dsrl_active_dim": action_dim,
        "dsrl_n_step": ft_config.dsrl_n_step,
        "dsrl_max_episode_length": ft_config.dsrl_max_episode_length,
        "dsrl_c_fail": ft_config.dsrl_c_fail,
        "dsrl_reward_scale": ft_config.dsrl_reward_scale,
    }
    logging.info(f"[DSRL] {dsrl_fields}")

    config.data.dsrl_n_step = ft_config.dsrl_n_step
    config.data.video_backend = ft_config.video_backend

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.shard_load_workers = ft_config.shard_load_workers
    config.data.video_decode_workers = ft_config.video_decode_workers
    config.data.num_ffmpeg_threads = ft_config.num_ffmpeg_threads
    config.data.overlap_episode_io = ft_config.overlap_episode_io

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    # ----- output dir ----------------------------------------------------------
    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        experiment_name = output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / "processor"
    config.save(save_cfg_dir / "config.yaml")
    omegaconf_config = OmegaConf.create(config.__dict__)
    omegaconf_config["max_steps"] = config.training.max_steps
    omegaconf_config["save_steps"] = config.training.save_steps
    OmegaConf.save(omegaconf_config, save_cfg_dir / "conf.yaml", resolve=True)

    if config.training.use_wandb and global_rank == 0:
        wandb.init(
            project=config.training.wandb_project,
            name=experiment_name,
            config={**config.__dict__, **dsrl_fields},
            tags=["dsrl"],
        )

    # ----- model setup ---------------------------------------------------------
    # The base checkpoint has no DSRL weights and setup.py matches weights strictly, so load the
    # model first and create the DSRL heads afterwards (same pattern as launch_recap.py).
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)
    pipeline.setup()
    model = pipeline.return_model()

    for key, value in dsrl_fields.items():
        setattr(model.config, key, value)  # persisted in config.json -> inference loads the heads

    action_head = model.action_head
    if action_head.dsrl is None:
        logging.info("[DSRL] DSRL heads not found in checkpoint -- initialising fresh.")
        action_head._init_dsrl()
    # Must happen before the HF Trainer builds its optimizer.
    action_head.set_phase_dsrl()

    train_dataset, eval_dataset = pipeline.return_dataset()
    data_collator = pipeline.return_collator()
    processor = pipeline.return_processor()
    processor.clean_observations = True
    if getattr(train_dataset, "processor", None) is not None:
        train_dataset.processor.clean_observations = True
    processor.save_pretrained(processor_dir)

    # ----- training args -------------------------------------------------------
    if config.training.num_gpus > 1 and not config.training.use_ddp:
        deepspeed_config = config.get_deepspeed_config()
    else:
        deepspeed_config = None

    per_device_batch_size = config.training.global_batch_size // config.training.num_gpus

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        save_only_model=config.training.save_only_model,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to="wandb" if config.training.use_wandb else "none",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=config.training.ddp_bucket_cap_mb,
        eval_strategy="no",
        batch_eval_metrics=True,
        remove_unused_columns=config.training.remove_unused_columns,
        ignore_data_skip=True,
    )

    trainer = Gr00tTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        multiprocessing_context=config.data.multiprocessing_context,
    )

    trainer.add_callback(
        CheckpointFormatCallback(
            run_name=experiment_name,
            exp_cfg_dir=save_cfg_dir,
            processor_dir=processor_dir,
        )
    )

    if hasattr(train_dataset, "get_initial_actions"):
        initial_actions = train_dataset.get_initial_actions()
        if initial_actions:
            save_initial_actions(initial_actions, save_cfg_dir / INITIAL_ACTIONS_FILENAME)

    logging.info("[DSRL] Starting offline DSRL-NA training")
    trainer.train(resume_from_checkpoint=True)
    logging.info("[DSRL] Training complete.")


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    run_dsrl(tyro.cli(DSRLFinetuneConfig))
