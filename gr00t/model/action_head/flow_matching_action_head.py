# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# DSRL-NA (offline variant) — Algorithm 1 from:
# "Steering Your Diffusion Policy with Latent Space Reinforcement Learning"
# Wagenmaker et al., arXiv:2506.15799
#
# Code style follows the RECAP action head (document 11):
#   init_dsrl_heads()   mirrors init_value_head()
#   set_phase_dsrl()    mirrors set_phase_value_head() / set_phase_policy()
#   forward_dsrl()      mirrors forward_value_head() / forward_action_head()
#   forward() dispatcher on _phase
#
# Required extra keys in action_input (like RECAP uses reward.current_frame_idx):
#   "next_backbone_features"  (B, S, D)   next-state VL features
#   "next_state"              (B, state_dim)
#   "reward"                  (B,) or (B,1)   {-1=fail, 0=success}

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta, Normal
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import SinusoidalPositionalEncoding, swish

from .cross_attention_dit import DiT, SelfAttentionTransformer


def get_prefix_weights(start: int, end: int, total: int, schedule: str) -> torch.Tensor:
    """
    With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    assert schedule in ["ones", "zeros", "linear", "exp"], f"Invalid schedule: {schedule}"
    start = min(start, end)
    idx = torch.arange(total, dtype=torch.float32)
    if schedule == "ones":
        w = torch.ones(total, dtype=torch.float32)
    elif schedule == "zeros":
        w = (idx < start).float()
    elif schedule == "linear" or schedule == "exp":
        w = torch.clamp((start - 1 - idx) / (end - start + 1) + 1, min=0, max=1)
        if schedule == "exp":
            # torch.expm1(x) = exp(x) - 1, torch.e = math.e
            w = w * torch.expm1(w) / (torch.tensor(torch.e) - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    w = torch.where(idx >= end, torch.tensor(0.0, dtype=w.dtype), w)
    return w


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        if timesteps.dim() == 2 and timesteps.shape == (B, T):
            pass  # already in desired shape
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


# ===========================================================================
# DSRL-NA components
# ===========================================================================

@dataclass
class DSRLConfig(PretrainedConfig):
    """
    Config for DSRL-NA heads: DSRLObsEncoder, ActionCritic, NoiseCritic, LatentNoisePolicy.

    All heads share backbone_dim so they consume the same backbone_features
    already produced by process_backbone_output() — identical to how
    DistributionalValueHead works in RECAP.
    """
    backbone_dim:         int   = field(default=1536)   # backbone_embedding_dim
    state_dim:            int   = field(default=128)    # max_state_dim
    hidden_dim:           int   = field(default=256)    # keep small
    action_horizon:       int   = field(default=16)
    action_dim:           int   = field(default=7)
    noise_dim:            int   = field(default=-1)     # set to action_horizon * action_dim
    discount:             float = field(default=0.99)
    tau:                  float = field(default=0.005)
    log_std_min:          float = field(default=-4.0)
    log_std_max:          float = field(default=0.5)
    target_entropy_scale: float = field(default=0.5)    # * noise_dim
    n_noise_samples:      int   = field(default=8)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.noise_dim == -1:
            self.noise_dim = self.action_horizon * self.action_dim


class DSRLObsEncoder(nn.Module):
    """
    Shared observation encoder for all DSRL heads.

    Same design as RECAP's DistributionalValueHead._encode():
        CLS token cross-attends over backbone_features (already vlln-processed)
        then concatenates state_proj for explicit proprioception.
    This mirrors the pattern used by the RECAP value head exactly.
    """
    def __init__(self, config: DSRLConfig):
        super().__init__()
        D = config.backbone_dim
        H = config.hidden_dim
        S = config.state_dim

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.cls_token, std=0.02)
        self.norm_in    = nn.LayerNorm(D)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=max(1, D // 64), batch_first=True,
        )
        self.norm_attn  = nn.LayerNorm(D)
        self.state_proj = nn.Sequential(
            nn.Linear(S, H), nn.LayerNorm(H), nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(D + H, H * 2), nn.LayerNorm(H * 2), nn.SiLU(),
            nn.Linear(H * 2, H),     nn.LayerNorm(H),      nn.SiLU(),
        )
        self.out_dim = H
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.ones_(self.norm_in.weight);   nn.init.zeros_(self.norm_in.bias)
        nn.init.ones_(self.norm_attn.weight); nn.init.zeros_(self.norm_attn.bias)
        for name, p in self.cross_attn.named_parameters():
            if "weight" in name: nn.init.normal_(p, std=0.02)
            elif "bias"  in name: nn.init.zeros_(p)
        for m in list(self.fuse.modules()) + list(self.state_proj.modules()):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, backbone_features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B = backbone_features.shape[0]
        cls      = self.cls_token.float().expand(B, -1, -1)
        normed   = self.norm_in(backbone_features.float())
        attended, _ = self.cross_attn(query=cls, key=normed, value=normed)
        attended = self.norm_attn(attended.squeeze(1))                  # (B, D)
        s        = self.state_proj(state.float())                        # (B, H)
        return self.fuse(torch.cat([attended, s], dim=-1))               # (B, H)


class ActionCritic(nn.Module):
    """
    Q^A(s, a) — action-space critic (DSRL-NA Algorithm 1, line 4).
    Trained via TD on offline (s, a, r, s') data. Double-Q.
    """
    def __init__(self, config: DSRLConfig, obs_encoder: DSRLObsEncoder):
        super().__init__()
        self.obs_encoder = obs_encoder
        H   = config.hidden_dim
        obs = obs_encoder.out_dim
        act = config.action_horizon * config.action_dim

        def _q():
            return nn.Sequential(
                nn.Linear(obs + act, H), nn.LayerNorm(H), nn.SiLU(),
                nn.Linear(H, H),         nn.LayerNorm(H), nn.SiLU(),
                nn.Linear(H, 1),
            )
        self.q1 = _q()
        self.q2 = _q()

    def forward(self, backbone_features, state, action):
        obs_emb = self.obs_encoder(backbone_features, state)
        a_flat  = action.reshape(action.shape[0], -1)
        x       = torch.cat([obs_emb, a_flat], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, backbone_features, state, action):
        q1, q2 = self.forward(backbone_features, state, action)
        return torch.min(q1, q2)


class NoiseCritic(nn.Module):
    """
    Q^W(s, w) — noise-space critic (DSRL-NA Algorithm 1, line 5).
    Distilled from Q^A: sample w~N(0,I), run frozen denoiser, target=Q^A(s,a).
    No inversion needed. Double-Q.
    """
    def __init__(self, config: DSRLConfig, obs_encoder: DSRLObsEncoder):
        super().__init__()
        self.obs_encoder = obs_encoder
        H     = config.hidden_dim
        obs   = obs_encoder.out_dim
        noise = config.noise_dim

        def _q():
            return nn.Sequential(
                nn.Linear(obs + noise, H), nn.LayerNorm(H), nn.SiLU(),
                nn.Linear(H, H),           nn.LayerNorm(H), nn.SiLU(),
                nn.Linear(H, 1),
            )
        self.q1 = _q()
        self.q2 = _q()

    def forward(self, backbone_features, state, w):
        obs_emb = self.obs_encoder(backbone_features, state)
        x       = torch.cat([obs_emb, w], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, backbone_features, state, w):
        q1, q2 = self.forward(backbone_features, state, w)
        return torch.min(q1, q2)


class LatentNoisePolicy(nn.Module):
    """
    pi^W(w | s) — noise-space actor (DSRL-NA Algorithm 1, line 6).
    Outputs Gaussian over initial noise w. Maximises Q^W.
    At inference: w ~ pi^W(s), then actions = denoise(w, s).
    """
    def __init__(self, config: DSRLConfig, obs_encoder: DSRLObsEncoder):
        super().__init__()
        self.config      = config
        self.obs_encoder = obs_encoder
        H     = config.hidden_dim
        noise = config.noise_dim

        self.trunk = nn.Sequential(
            nn.Linear(obs_encoder.out_dim, H), nn.LayerNorm(H), nn.SiLU(),
            nn.Linear(H, H),                   nn.LayerNorm(H), nn.SiLU(),
        )
        self.mean_head    = nn.Linear(H, noise)
        self.log_std_head = nn.Linear(H, noise)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)   # init std ~= 1 (close to N(0,I))

    def forward(self, backbone_features, state):
        """Returns (mean, log_std), both (B, noise_dim)."""
        h       = self.trunk(self.obs_encoder(backbone_features, state))
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.config.log_std_min, self.config.log_std_max)
        return mean, log_std

    def sample(self, backbone_features, state):
        """Returns (w, log_prob): w (B, noise_dim), log_prob (B,)."""
        mean, log_std = self.forward(backbone_features, state)
        std      = log_std.exp()
        w        = mean + std * torch.randn_like(mean)
        log_prob = Normal(mean, std).log_prob(w).sum(dim=-1)
        return w, log_prob

    def entropy(self, log_std):
        return Normal(torch.zeros_like(log_std), log_std.exp()).entropy().sum(dim=-1)


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )
    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    # ── DSRL fields ───────────────────────────────────────────────────────
    _phase: str = field(default="dsrl")

    dsrl_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dim for all DSRL heads (Q^A, Q^W, pi^W). Keep small."},
    )
    dsrl_discount:             float = field(default=0.99)
    dsrl_tau:                  float = field(default=0.005)
    dsrl_log_std_min:          float = field(default=-4.0)
    dsrl_log_std_max:          float = field(default=0.5)
    dsrl_target_entropy_scale: float = field(default=0.5)
    dsrl_n_noise_samples:      int   = field(
        default=8,
        metadata={"help": "w~N(0,I) samples per state when distilling Q^A -> Q^W."},
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: FlowmatchingActionHeadConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

        self.init_dsrl_heads()
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    # ── DSRL head init (mirrors init_value_head style) ────────────────────

    def init_dsrl_heads(self):
        """
        Initialise Q^A, Q^W, pi^W and shared DSRLObsEncoder.
        Also creates the Q^A target network (EMA copy, never trained directly).
        Mirrors init_value_head() in structure.
        """
        dsrl_cfg = DSRLConfig(
            backbone_dim=self.config.backbone_embedding_dim,
            state_dim=self.config.max_state_dim,
            hidden_dim=self.config.dsrl_hidden_dim,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            discount=self.config.dsrl_discount,
            tau=self.config.dsrl_tau,
            log_std_min=self.config.dsrl_log_std_min,
            log_std_max=self.config.dsrl_log_std_max,
            target_entropy_scale=self.config.dsrl_target_entropy_scale,
            n_noise_samples=self.config.dsrl_n_noise_samples,
        )

        self.dsrl_obs_encoder = DSRLObsEncoder(dsrl_cfg)
        self.q_a              = ActionCritic(dsrl_cfg, self.dsrl_obs_encoder)
        self.q_a_target       = copy.deepcopy(self.q_a)
        for p in self.q_a_target.parameters():
            p.requires_grad = False

        self.q_w              = NoiseCritic(dsrl_cfg, self.dsrl_obs_encoder)
        self.noise_actor      = LatentNoisePolicy(dsrl_cfg, self.dsrl_obs_encoder)

        target_entropy = -dsrl_cfg.target_entropy_scale * dsrl_cfg.noise_dim
        self.dsrl_log_alpha      = nn.Parameter(torch.zeros(1))
        self.dsrl_target_entropy = target_entropy
        self.dsrl_config         = dsrl_cfg

        print(
            f"[DSRL] Heads ENABLED  "
            f"noise_dim={dsrl_cfg.noise_dim}  "
            f"n_noise_samples={dsrl_cfg.n_noise_samples}  "
            f"hidden_dim={dsrl_cfg.hidden_dim}"
        )

    # ── Phase switching (mirrors set_phase_value_head / set_phase_policy) ─

    def set_phase_dsrl(self):
        """
        Freeze the base DiT denoiser, unfreeze DSRL heads.

        Trains all three Algorithm 1 components in forward_dsrl():
            Line 4: Q^A via TD on offline data
            Line 5: Q^W via distillation from Q^A
            Line 6: pi^W to maximise Q^W

        Mirrors set_phase_value_head() in structure.
        """
        for p in self.parameters():
            p.requires_grad = False

        for p in self.dsrl_obs_encoder.parameters():
            p.requires_grad = True
        for p in self.q_a.parameters():
            p.requires_grad = True
        for p in self.q_w.parameters():
            p.requires_grad = True
        for p in self.noise_actor.parameters():
            p.requires_grad = True
        self.dsrl_log_alpha.requires_grad = True

        # Target network never receives gradients — EMA only
        for p in self.q_a_target.parameters():
            p.requires_grad = False

        self._phase = "dsrl"
        print("[DSRL] ── Phase DSRL ──  Denoiser FROZEN")
        print(f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    # ── Standard helpers ──────────────────────────────────────────────────

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    # ── Internal: frozen denoiser forward pass ────────────────────────────

    @torch.no_grad()
    def _denoise_from_noise(self, w, backbone_feats, vl_attn_mask, state_features, embodiment_id):
        """
        pi^W_dp(s, w) — run the frozen denoiser from initial noise w -> action.
        Called by forward_dsrl() for Q^W distillation targets (Algorithm 1, line 5).
        Mirrors the logic inside get_action() exactly.
        """
        device = backbone_feats.device
        B = backbone_feats.shape[0]
        H = self.action_horizon
        D = self.action_dim

        x  = w.reshape(B, H, D).to(dtype=backbone_feats.dtype)
        dt = 1.0 / self.num_inference_timesteps

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            ts     = torch.full((B,), fill_value=t_disc, device=device)

            af = self.action_encoder(x, ts, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(af.shape[1], dtype=torch.long, device=device)
                af      = af + self.position_embedding(pos_ids).unsqueeze(0)

            ft  = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
            sa  = torch.cat((state_features, ft, af), dim=1)
            out = self.model(
                hidden_states=sa,
                encoder_hidden_states=backbone_feats,
            encoder_attention_mask=vl_attn_mask,
                timestep=ts,
            )
            v  = self.action_decoder(out, embodiment_id)[:, -H:]
            x  = x + dt * v

        return x   # (B, H, D)

    def _ema_update_target(self):
        """Soft update Q^A target network. Called after each forward_dsrl step."""
        tau = self.dsrl_config.tau
        for p, pt in zip(self.q_a.parameters(), self.q_a_target.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

    # ── Phase-specific forward (mirrors forward_value_head style) ─────────

    def forward_dsrl(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        DSRL-NA Algorithm 1 (offline), all three update lines in one step.
        Mirrors forward_value_head() / forward_action_head() in structure.

        The base DiT denoiser is frozen (no gradient flows through it).

        Required keys in action_input:
            action                    (B, H, D)   dataset action
            state                     (B, state_dim)
            reward                    (B,) or (B,1)   {-1, 0}
            next_backbone_features    (B, S, D)   next-state VL features
            next_state                (B, state_dim)
        """
        self.set_frozen_modules_to_eval_mode()
        self.model.eval()   # denoiser always frozen in DSRL phase

        backbone_output = self.process_backbone_output(backbone_output)

        assert "reward"                 in action_input, "DSRL needs 'reward'"
        assert "next_backbone_features" in action_input, "DSRL needs 'next_backbone_features'"
        assert "next_state"             in action_input, "DSRL needs 'next_state'"

        vl_embs_cur   = backbone_output.backbone_features
        vl_attn_mask  = backbone_output.backbone_attention_mask
        state_cur     = action_input.state.float()
        action        = action_input.action
        reward        = torch.squeeze(action_input["reward"], dim=-1).float()
        embodiment_id = action_input.embodiment_id

        # Process next-state backbone features (mirrors process_backbone_output)
        next_bf   = self.vl_self_attention(self.vlln(action_input["next_backbone_features"].float()))
        state_next = action_input["next_state"].float()
        device    = vl_embs_cur.device

        # ── Line 4: Update Q^A ────────────────────────────────────────────
        # TD target: r + γ * Q̄^A(s', a')  where a' = pi^W_dp(s', pi^W(s'))
        with torch.no_grad():
            w_next, _        = self.noise_actor.sample(next_bf, state_next)
            sf_next          = self.state_encoder(action_input["next_state"], embodiment_id)
            a_next           = self._denoise_from_noise(
                w_next, next_bf, vl_attn_mask, sf_next, embodiment_id
            )
            q_next  = self.q_a_target.min_q(next_bf, state_next, a_next)
            target  = reward.unsqueeze(1) + self.dsrl_config.discount * q_next

        q1, q2  = self.q_a(vl_embs_cur, state_cur, action)
        loss_qa = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # ── Line 5: Update Q^W ────────────────────────────────────────────
        # Sample fresh w ~ N(0,I) — no noise labels needed from dataset
        B = vl_embs_cur.shape[0]
        N = self.dsrl_config.n_noise_samples

        bf_exp  = vl_embs_cur.unsqueeze(1).expand(-1, N, -1, -1).reshape(B*N, *vl_embs_cur.shape[1:])
        s_exp   = state_cur.unsqueeze(1).expand(-1, N, -1).reshape(B*N, state_cur.shape[-1])
        eid_exp = embodiment_id.unsqueeze(1).expand(-1, N).reshape(B*N)
        sf_exp  = self.state_encoder(
            action_input.state.unsqueeze(1).expand(-1, N, -1).reshape(B*N, -1), eid_exp
        )
        mask_exp = (
            vl_attn_mask.unsqueeze(1).expand(-1, N, -1).reshape(B*N, -1)
            if vl_attn_mask is not None else None
        )

        with torch.no_grad():
            w_rand  = torch.randn(B * N, self.dsrl_config.noise_dim, device=device)
            a_rand  = self._denoise_from_noise(w_rand, bf_exp, mask_exp, sf_exp, eid_exp)
            qa_tgt  = self.q_a.min_q(bf_exp, s_exp, a_rand)

        qw1, qw2 = self.q_w(bf_exp, s_exp, w_rand)
        loss_qw  = F.mse_loss(qw1, qa_tgt) + F.mse_loss(qw2, qa_tgt)

        # ── Line 6: Update pi^W ───────────────────────────────────────────
        w_actor, log_prob = self.noise_actor.sample(vl_embs_cur, state_cur)
        qw_min            = self.q_w.min_q(vl_embs_cur, state_cur, w_actor)
        alpha             = self.dsrl_log_alpha.exp().detach()
        loss_actor        = (alpha * log_prob.unsqueeze(1) - qw_min).mean()

        # SAC temperature update
        loss_alpha = -(self.dsrl_log_alpha * (log_prob + self.dsrl_target_entropy).detach()).mean()

        total_loss = loss_qa + loss_qw + loss_actor + loss_alpha

        # EMA update for Q^A target network
        self._ema_update_target()

        return BatchFeature(data={
            "loss":       total_loss,
            "loss_qa":    loss_qa.detach(),
            "loss_qw":    loss_qw.detach(),
            "loss_actor": loss_actor.detach(),
            "loss_alpha": loss_alpha.detach(),
            "alpha":      alpha.item(),
            "log_prob":   log_prob.mean().detach(),
        })

    # ── Forward dispatcher ────────────────────────────────────────────────

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        if self._phase == "dsrl":
            return self.forward_dsrl(backbone_output, action_input)
        else:
            raise ValueError(f"Unknown _phase: {self._phase!r}")

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embs        = backbone_output.backbone_features
        vl_attn_mask   = backbone_output.backbone_attention_mask
        embodiment_id  = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)
        batch_size     = vl_embs.shape[0]
        device         = vl_embs.device

        # Sample w ~ pi^W(w|s) instead of w ~ N(0,I)
        w, _ = self.noise_actor.sample(vl_embs, action_input.state.float())
        actions = self._denoise_from_noise(
            w, vl_embs, vl_attn_mask, state_features, embodiment_id
        )
        return BatchFeature(data={"action_pred": actions})
    
    @torch.enable_grad()
    def get_realtime_action(
        self,
        action_input: BatchFeature,
        backbone_output:BatchFeature,
        prev_action_chunk: torch.Tensor,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str,
        max_guidance_weight: float,
        sigma_d_o: float,
        actual_action_dim: int,
        use_prev_action: bool = True
    )  -> BatchFeature:
        torch.set_grad_enabled(True)
        num_steps = self.num_inference_timesteps
        self.sigma_d_o = sigma_d_o
        dt = 1.0 / num_steps
        prev_action_chunk = torch.as_tensor(prev_action_chunk, device=self.device, dtype=self.dtype)

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        actions_mask = torch.zeros_like(x_t, dtype=vl_embs.dtype, device=device)
        actions_mask[:, :, :actual_action_dim] = True

        # weights: [horizon]
        weights = get_prefix_weights(
            inference_delay, prefix_attention_horizon, self.config.action_horizon, prefix_attention_schedule
        )
        weights = weights.to(device)

        for t in range(num_steps):

            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            
            def denoiser(x_t_):
                t_discretized = int(t_cont * self.num_timestep_buckets)

                # Embed noised action trajectory.
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(x_t_, timesteps_tensor, embodiment_id)
                # Maybe add position embedding.
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                # Join vision, language, state and action embedding along sequence dimension.
                future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

                # Run model forward.
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred = self.action_decoder(model_output, embodiment_id)

                pred_velocity = pred[:, -self.action_horizon :]
                return x_t_ + pred_velocity * dt, pred_velocity
            
            (outputs, vjp_func) = torch.func.vjp(denoiser, x_t)
            (x_1_i_vjp, v_t_i_vjp) = outputs
            error = (prev_action_chunk - x_1_i_vjp) * weights[:, None] * actions_mask
            # error = F.mse_loss(x_1_i_vjp, prev_action_chunk, reduction="none") / actions_mask.sum() * weights[:, None] * actions_mask
            
            pinv_correction = vjp_func((error, torch.zeros_like(x_t)))[0]
            if pinv_correction is None:
                pinv_correction = torch.zeros_like(x_1_i_vjp)
            inv_r2 = (self.sigma_d_o**2 * t_cont**2 + (1 - t_cont)**2) / (self.sigma_d_o**2 * (1 - t_cont)**2)
            # inv_r2 = (t_cont**2 + (1 - t_cont) ** 2) / ((1 - t_cont) ** 2)
            c = torch.nan_to_num(torch.tensor((1 - t_cont) / max(t_cont, 1e-12), device=self.device, dtype=self.dtype),  # Avoid division by zero
                                 nan=0.0, posinf=max_guidance_weight)
            
            guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
            guidance_weight = torch.minimum(guidance_weight, torch.tensor(max_guidance_weight, device=device))
            v_t_corr = v_t_i_vjp + guidance_weight * pinv_correction
            x_t = x_t + v_t_corr * dt

        assert x_t.shape == (batch_size, self.config.action_horizon, self.config.action_dim), x_t.shape
        x_t = x_t.clone().detach()
        print("EACH DENOISING STEP: ", (x_t[:,:inference_delay,:actual_action_dim] - prev_action_chunk[:,:inference_delay,:actual_action_dim]).abs().mean())
        if use_prev_action:
            x_t = prev_action_chunk * weights[:, None] + x_t * (1 - weights[:, None])
            print("AFTER ASSIGN: ", (x_t[:,:inference_delay,:actual_action_dim] - prev_action_chunk[:,:inference_delay,:actual_action_dim]).abs().mean())
        return BatchFeature(data={"action_pred": x_t})
    
    @torch.no_grad()
    def get_repaint_action(
        self,
        action_input: BatchFeature,
        backbone_output: BatchFeature,
        prev_action_chunk: torch.Tensor,    # [B, H, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        actual_action_dim: int,
        use_prev_action: bool = True,
    ) -> BatchFeature:
        
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs      = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device     = vl_embs.device
        dtype      = vl_embs.dtype
        d          = inference_delay
        dt         = 1.0 / self.num_inference_timesteps

        # prefix_mask[b, i, 0] = True iff i < d  →  broadcastable over [B, H, D]
        prefix_mask = (
            torch.arange(self.config.action_horizon, device=device)
            .unsqueeze(0).unsqueeze(-1) < d
        )  # [1, H, 1]

        # ── shared denoising step ─────────────────────────────────────────────
        def model_velocity(actions: torch.Tensor, t: int) -> torch.Tensor:
            """One forward pass of the denoiser at step t. Returns predicted velocity."""
            t_cont        = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps     = torch.full((batch_size,), t_discretized, device=device)

            action_features = self.action_encoder(actions, timesteps, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs       = torch.cat((state_features, future_tokens, action_features), dim=1)

            model_output  = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps,
            )
            return self.action_decoder(model_output, embodiment_id)[:, -self.action_horizon:]

        # ── Step 1: naive forward pass ───────────────────────────────────────
        x_0_free = torch.randn(
            batch_size, self.config.action_horizon, self.config.action_dim,
            dtype=dtype, device=device,
        )
        x = x_0_free.clone()
        for t in range(self.num_inference_timesteps):
            x = x + dt * model_velocity(x, t)
        x_1_naive = x

        # ── Step 2: construct inversion target ────────────────────────────────
        x_1_target = torch.where(prefix_mask, prev_action_chunk, x_1_naive)

        # ── Step 3: backward Euler inversion ─────────────────────────────────
        x = x_1_target.clone().to(dtype=dtype)
        for t in reversed(range(self.num_inference_timesteps)):
            x = x - dt * model_velocity(x, t)
        x_0_star = x

        # ── Step 4: Mao re-painting ───────────────────────────────────────────
        x_0_repaint = torch.where(prefix_mask, x_0_star, x_0_free)

        # ── Step 5: final forward pass ────────────────────────────────────────
        x = x_0_repaint.clone().to(dtype=dtype)
        for t in range(self.num_inference_timesteps):
            x = x + dt * model_velocity(x, t)
        x_1_final = x
        print("EACH DENOISING STEP: ", (x_1_final[:,:inference_delay,:actual_action_dim] - prev_action_chunk[:,:inference_delay,:actual_action_dim]).abs().mean())
        if use_prev_action:
            weights = get_prefix_weights(
                inference_delay, prefix_attention_horizon, self.config.action_horizon, "exp"
            ).to(device)
            x_1_final = prev_action_chunk * weights[:, None] + x_1_final * (1 - weights[:, None])
            print("AFTER ASSIGN: ", (x_1_final[:,:inference_delay,:] - prev_action_chunk[:,:inference_delay,:]).abs().mean())

        return BatchFeature(data={
            "action_pred": x_1_final,
        })

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
