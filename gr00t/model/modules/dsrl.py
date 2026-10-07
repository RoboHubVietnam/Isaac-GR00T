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
"""
DSRL-NA: Noise-Aliased Diffusion Steering via Reinforcement Learning (offline).

    "Steering Your Diffusion Policy with Latent Space Reinforcement Learning"
    Wagenmaker et al., arXiv:2506.15799, Algorithm 1

The pretrained GR00T flow-matching policy pi_dp^W(s, w) is treated as a frozen, black-box
action-space transformation. Only three small networks are trained on top of the frozen
backbone features:

    Q^A(s, a)  action critic       TD-learned on (s, a, r, s') from the dataset   (line 4)
    Q^W(s, w)  latent-noise critic distilled from Q^A through the frozen denoiser  (line 5)
    pi^W(s)    latent-noise actor  maximises Q^W                                   (line 6)

At inference the initial flow-matching noise is drawn from pi^W(s) instead of N(0, I).

Design notes
------------
* One decision = one action chunk. TD transitions are (s_t, a_{t:t+H}, R, s_{t+N}); N is the
  number of chunk steps executed before replanning (`n_step`).
* Rewards follow the RECAP convention (-1 per step, 0 on success, -c_fail on failure), scaled by
  `reward_scale / (max_episode_length + c_fail)`.
* The actor is a tanh-squashed Gaussian on the box [-b, b]^d (paper App. B, "action magnitude").
* Critics and actor each own an observation encoder, so no network's loss leaks gradient into
  another one (the actor loss sees Q^W through a parameter-detached functional call).
"""

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.func import functional_call
import torch.nn.functional as F


NOISE_MODES = ("full", "chunk_repeat")


@dataclass
class DSRLConfig:
    backbone_dim: int = 2048
    state_dim: int = 132
    action_horizon: int = 40  # padded model chunk length H
    action_dim: int = 132  # padded model action dim D
    # Region of the (H, D) chunk the latent-noise policy steers. The rest of the padded chunk is
    # masked out of the loss, so it keeps its N(0, I) prior noise. 0 -> use the full extent.
    active_horizon: int = 0
    active_dim: int = 0
    hidden_dim: int = 512
    num_heads: int = 8
    discount: float = 0.99  # per action-chunk decision, as in the paper
    tau: float = 0.005
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    noise_bound: float = 1.5  # b_W
    target_entropy: float = 0.0  # paper Table 3
    init_alpha: float = 0.1
    n_noise_samples: int = 4
    noise_mode: str = "full"  # "full": w in R^{H*D};  "chunk_repeat": w in R^D repeated over H

    def __post_init__(self):
        assert self.noise_mode in NOISE_MODES, f"noise_mode must be one of {NOISE_MODES}"

    @property
    def h_act(self) -> int:
        return self.active_horizon or self.action_horizon

    @property
    def d_act(self) -> int:
        return self.active_dim or self.action_dim

    @property
    def noise_dim(self) -> int:
        if self.noise_mode == "chunk_repeat":
            return self.d_act
        return self.h_act * self.d_act

    @property
    def action_flat_dim(self) -> int:
        return self.h_act * self.d_act


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def compute_chunk_transition(
    reward_label: torch.Tensor,  # (B,)  RECAP label: < 0 failed episode, >= 0 successful episode
    frame_idx: torch.Tensor,  # (B,)  t
    episode_length: torch.Tensor,  # (B,)  T
    n_step: int,
    max_episode_length: float,
    c_fail: float,
    reward_scale: float = 1.0,
):
    """
    N-step chunk transition (RECAP Eq. 5 rewards) for the transition (s_t -> s_{t+N}).

        r_t' = -1                      for every environment step
        r_T  =  0 if success, -c_fail  if failure   (terminal step)

    Returns (reward, not_done), each (B,) float32. `not_done` is 0 when the chunk reaches the end of
    the episode, in which case the TD target must not bootstrap.
    """
    t = frame_idx.float()
    T = episode_length.float()
    steps = torch.clamp(T - t, min=0.0, max=float(n_step))
    done = (t + n_step >= T).float()
    failed = (reward_label.float() < 0).float()
    norm = float(max_episode_length + c_fail)
    reward = (-steps - done * failed * c_fail) / norm * reward_scale
    return reward, 1.0 - done


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.Tanh()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class DSRLObsEncoder(nn.Module):
    """
    Frozen-backbone features (B, S, D) + proprio state -> (B, H).

    A learned query cross-attends over the (padding-masked) backbone tokens, then is fused with a
    projection of the last observed state.
    """

    def __init__(self, cfg: DSRLConfig):
        super().__init__()
        H = cfg.hidden_dim
        self.norm_in = nn.LayerNorm(cfg.backbone_dim)
        self.proj_in = nn.Linear(cfg.backbone_dim, H)
        self.query = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        self.attn = nn.MultiheadAttention(H, cfg.num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(H)
        self.state_proj = nn.Sequential(nn.Linear(cfg.state_dim, H), nn.LayerNorm(H), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(2 * H, H), nn.LayerNorm(H), nn.SiLU())
        self.out_dim = H

    def forward(self, backbone_features, attn_mask, state):
        B = backbone_features.shape[0]
        if state.dim() == 3:  # (B, T_state, S) -> last observed frame
            state = state[:, -1]
        x = self.proj_in(self.norm_in(backbone_features.float()))
        key_padding_mask = None if attn_mask is None else ~attn_mask.bool()
        q = self.query.expand(B, -1, -1)
        attended, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        attended = self.norm_attn(attended.squeeze(1))
        s = self.state_proj(state.float())
        return self.fuse(torch.cat([attended, s], dim=-1))


class _DoubleQ(nn.Module):
    """Encoder + two Q heads over concat(obs_emb, x)."""

    def __init__(self, cfg: DSRLConfig, x_dim: int):
        super().__init__()
        self.encoder = DSRLObsEncoder(cfg)
        self.q1 = _mlp(cfg.hidden_dim + x_dim, cfg.hidden_dim, 1)
        self.q2 = _mlp(cfg.hidden_dim + x_dim, cfg.hidden_dim, 1)

    def encode(self, backbone_features, attn_mask, state):
        return self.encoder(backbone_features, attn_mask, state)

    def q_from_emb(self, emb, x):
        h = torch.cat([emb, x.float()], dim=-1)
        return self.q1(h).squeeze(-1), self.q2(h).squeeze(-1)

    def forward(self, backbone_features, attn_mask, state, x):
        return self.q_from_emb(self.encode(backbone_features, attn_mask, state), x)

    def min_q(self, backbone_features, attn_mask, state, x):
        q1, q2 = self(backbone_features, attn_mask, state, x)
        return torch.min(q1, q2)


class ActionCritic(_DoubleQ):
    """Q^A(s, a): a is the flattened, action-masked chunk."""

    def __init__(self, cfg: DSRLConfig):
        super().__init__(cfg, cfg.action_flat_dim)


class NoiseCritic(_DoubleQ):
    """Q^W(s, w)."""

    def __init__(self, cfg: DSRLConfig):
        super().__init__(cfg, cfg.noise_dim)


class LatentNoisePolicy(nn.Module):
    """pi^W(w | s): tanh-squashed Gaussian on [-b, b]^noise_dim."""

    def __init__(self, cfg: DSRLConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim
        self.encoder = DSRLObsEncoder(cfg)
        self.trunk = nn.Sequential(
            nn.Linear(H, H), nn.LayerNorm(H), nn.Tanh(), nn.Linear(H, H), nn.LayerNorm(H), nn.Tanh()
        )
        self.mean_head = nn.Linear(H, cfg.noise_dim)
        self.log_std_head = nn.Linear(H, cfg.noise_dim)
        # Start close to the base policy's N(0, I): small mean, unit pre-squash std.
        for head in (self.mean_head, self.log_std_head):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    def dist_params(self, backbone_features, attn_mask, state):
        h = self.trunk(self.encoder(backbone_features, attn_mask, state))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.cfg.log_std_min, self.cfg.log_std_max)
        return mean, log_std

    def sample(self, backbone_features, attn_mask, state, deterministic: bool = False):
        """Returns (w, log_prob): w (B, noise_dim) in [-b, b], log_prob (B,) of the squashed sample."""
        mean, log_std = self.dist_params(backbone_features, attn_mask, state)
        b = self.cfg.noise_bound
        if deterministic:
            return b * torch.tanh(mean), torch.zeros(mean.shape[0], device=mean.device)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)  # reparameterised
        log_prob = Normal(mean, std).log_prob(u).sum(-1)
        # log|d(b tanh u)/du| = log b + log(1 - tanh^2 u)   (numerically stable form)
        log_det = (math.log(b) + 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return b * torch.tanh(u), log_prob - log_det


# ---------------------------------------------------------------------------
# All DSRL parameters + losses
# ---------------------------------------------------------------------------


class DSRLHeads(nn.Module):
    def __init__(self, cfg: DSRLConfig):
        super().__init__()
        self.cfg = cfg
        self.q_a = ActionCritic(cfg)
        self.q_a_target = copy.deepcopy(self.q_a)
        self.q_a_target.requires_grad_(False)
        self.q_w = NoiseCritic(cfg)
        self.actor = LatentNoisePolicy(cfg)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.init_alpha)))

    # -- noise <-> chunk shape ------------------------------------------------
    def to_noise(self, w: torch.Tensor) -> torch.Tensor:
        """(B, noise_dim) -> (B, H, D) flow-matching initial noise.

        The steered region [:h_act, :d_act] comes from `w`; the padded remainder is N(0, I).
        """
        cfg = self.cfg
        B = w.shape[0]
        if cfg.noise_mode == "chunk_repeat":
            active = w.unsqueeze(1).expand(B, cfg.h_act, -1)
        else:
            active = w.reshape(B, cfg.h_act, cfg.d_act)
        full = torch.randn(B, cfg.action_horizon, cfg.action_dim, device=w.device, dtype=w.dtype)
        full[:, : cfg.h_act, : cfg.d_act] = active
        return full

    def sample_prior_noise(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        """w ~ N(0, I) clipped to the actor's box, flat (B, noise_dim)."""
        b = self.cfg.noise_bound
        return torch.randn(batch, self.cfg.noise_dim, device=device, dtype=dtype).clamp(-b, b)

    @torch.no_grad()
    def ema_update(self):
        tau = self.cfg.tau
        for p, pt in zip(self.q_a.parameters(), self.q_a_target.parameters()):
            pt.lerp_(p, tau)

    def train(self, mode: bool = True):
        super().train(mode)
        self.q_a_target.eval()  # target network never trains
        return self

    # -- Algorithm 1, lines 4-6 -----------------------------------------------
    def compute_losses(
        self,
        obs: dict,  # backbone_features (B,S,D), attn_mask (B,S)|None, state (B,1,S)
        next_obs: dict,
        action: torch.Tensor,  # (B, H, D) dataset chunk
        action_mask: torch.Tensor,  # (B, H, D)
        reward: torch.Tensor,  # (B,)
        not_done: torch.Tensor,  # (B,)
        denoise,  # denoise(w_full (B,H,D), next: bool) -> (B,H,D), no grad; next selects s' vs s
    ) -> dict:
        cfg = self.cfg
        B = action.shape[0]
        mask = action_mask.float()

        def masked_flat(a):
            return (a.float() * mask)[:, : cfg.h_act, : cfg.d_act].reshape(B, -1)

        a_data = masked_flat(action)

        # ---- line 4: Q^A by TD on dataset (s, a, r, s') --------------------
        with torch.no_grad():
            w_next, _ = self.actor.sample(**next_obs)
            a_next = masked_flat(denoise(self.to_noise(w_next), True))
            q_next = self.q_a_target.min_q(**next_obs, x=a_next)
            td_target = reward + cfg.discount * not_done * q_next
        q1, q2 = self.q_a(**obs, x=a_data)
        loss_qa = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        # ---- line 5: Q^W distilled from Q^A through the frozen denoiser ----
        emb_w = self.q_w.encode(**obs)
        loss_qw, qa_tgt_mean = 0.0, 0.0
        for _ in range(cfg.n_noise_samples):
            with torch.no_grad():
                w = self.sample_prior_noise(B, action.device)
                a_w = masked_flat(denoise(self.to_noise(w), False))
                tgt = self.q_a.min_q(**obs, x=a_w)
            qw1, qw2 = self.q_w.q_from_emb(emb_w, w)
            loss_qw = loss_qw + F.mse_loss(qw1, tgt) + F.mse_loss(qw2, tgt)
            qa_tgt_mean = qa_tgt_mean + tgt.mean().detach() / cfg.n_noise_samples
        loss_qw = loss_qw / cfg.n_noise_samples

        # ---- line 6: pi^W maximises Q^W (Q^W params detached: actor loss only trains the actor)
        w_pi, log_prob = self.actor.sample(**obs)
        frozen_qw = {k: v.detach() for k, v in self.q_w.named_parameters()}
        qp1, qp2 = functional_call(self.q_w, frozen_qw, (obs["backbone_features"], obs["attn_mask"], obs["state"], w_pi))
        q_pi = torch.min(qp1, qp2)
        alpha = self.log_alpha.exp().detach()
        loss_actor = (alpha * log_prob - q_pi).mean()
        loss_alpha = -(self.log_alpha * (log_prob.detach() + cfg.target_entropy)).mean()

        if self.training:
            self.ema_update()

        return {
            "loss": loss_qa + loss_qw + loss_actor + loss_alpha,
            "dsrl_loss_qa": loss_qa.detach(),
            "dsrl_loss_qw": loss_qw.detach(),
            "dsrl_loss_actor": loss_actor.detach(),
            "dsrl_alpha": alpha,
            "dsrl_entropy": -log_prob.detach().mean(),
            "dsrl_q_data": q1.detach().mean(),
            "dsrl_q_pi": q_pi.detach().mean(),
            "dsrl_q_prior": torch.as_tensor(qa_tgt_mean, device=action.device),
            "dsrl_reward": reward.mean(),
            "dsrl_done_frac": 1.0 - not_done.mean(),
        }
