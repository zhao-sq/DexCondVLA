"""Shared conditional flow-matching DiT for dexterous hand actions."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


@dataclasses.dataclass(frozen=True)
class HandDiTConfig:
    """Architecture and conditioning switches for the first hand expert.

    The default is the design-document A3 model.  Conditioning switches are
    kept in the config so A1/A2 and future encoders do not require a new DiT.
    """

    action_horizon: int = 50
    arm_dim: int = 8
    hand_dim: int = 22
    hand_state_dim: int = 22
    vision_feature_dim: int = 768
    model_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    max_vision_tokens: int = 64
    dropout: float = 0.0
    condition_on_arm: bool = True
    condition_on_vision: bool = True
    condition_on_state: bool = True
    bimanual_mode: Literal["independent_shared", "joint"] = "independent_shared"

    def __post_init__(self) -> None:
        positive = {
            "action_horizon": self.action_horizon,
            "arm_dim": self.arm_dim,
            "hand_dim": self.hand_dim,
            "hand_state_dim": self.hand_state_dim,
            "vision_feature_dim": self.vision_feature_dim,
            "model_dim": self.model_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_vision_tokens": self.max_vision_tokens,
        }
        if invalid := {name: value for name, value in positive.items() if value <= 0}:
            raise ValueError(f"HandDiT dimensions must be positive: {invalid}")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not (self.condition_on_arm or self.condition_on_vision or self.condition_on_state):
            raise ValueError("At least one conditioning modality must be enabled")
        if self.bimanual_mode != "independent_shared":
            raise NotImplementedError(
                "The first version implements shared weights with independent sides; "
                "the config field is reserved for a future joint decoder"
            )


def sinusoidal_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    if timestep.ndim != 1:
        raise ValueError(f"Expected timestep [B], got {timestep.shape}")
    if dimension % 2:
        raise ValueError("Sinusoidal embedding dimension must be even")
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(dimension // 2, device=timestep.device, dtype=torch.float32)
        / max(1, dimension // 2 - 1)
    )
    angles = timestep.float()[:, None] * frequency[None, :] * 2.0 * math.pi
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1.0 + scale[:, None, :]) + shift[:, None, :]


class HandDiTBlock(nn.Module):
    """Self-attention, condition cross-attention, and MLP with adaLN gates."""

    def __init__(self, config: HandDiTConfig) -> None:
        super().__init__()
        dimension = config.model_dim
        self.norm_self = nn.LayerNorm(dimension, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(dimension, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(dimension, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(
            dimension, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            dimension, config.num_heads, dropout=config.dropout, batch_first=True
        )
        hidden_dimension = int(dimension * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dimension, dimension),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 9 * dimension))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        hand_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        time_embedding: torch.Tensor,
        *,
        condition_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation = self.modulation(time_embedding).chunk(9, dim=-1)
        self_shift, self_scale, self_gate = modulation[:3]
        cross_shift, cross_scale, cross_gate = modulation[3:6]
        mlp_shift, mlp_scale, mlp_gate = modulation[6:]

        query = _modulate(self.norm_self(hand_tokens), self_shift, self_scale)
        attended, _ = self.self_attention(query, query, query, need_weights=False)
        hand_tokens = hand_tokens + self_gate[:, None, :] * attended

        query = _modulate(self.norm_cross(hand_tokens), cross_shift, cross_scale)
        attended, _ = self.cross_attention(
            query,
            condition_tokens,
            condition_tokens,
            key_padding_mask=condition_padding_mask,
            need_weights=False,
        )
        hand_tokens = hand_tokens + cross_gate[:, None, :] * attended

        hidden = _modulate(self.norm_mlp(hand_tokens), mlp_shift, mlp_scale)
        return hand_tokens + mlp_gate[:, None, :] * self.mlp(hidden)


class HandConditionEncoder(nn.Module):
    """Build concatenated arm, wrist-patch, and hand-state condition memory."""

    def __init__(self, config: HandDiTConfig) -> None:
        super().__init__()
        self.config = config
        dimension = config.model_dim
        self.arm_projection = nn.Linear(config.arm_dim, dimension)
        self.vision_norm = nn.LayerNorm(config.vision_feature_dim)
        self.vision_projection = nn.Linear(config.vision_feature_dim, dimension)
        self.state_projection = nn.Sequential(
            nn.Linear(config.hand_state_dim, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.arm_positions = nn.Parameter(torch.empty(config.action_horizon, dimension))
        self.modality_embedding = nn.Parameter(torch.empty(3, dimension))
        self.side_embedding = nn.Parameter(torch.empty(2, dimension))
        nn.init.normal_(self.arm_positions, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.side_embedding, std=0.02)

    def _pool_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] <= self.config.max_vision_tokens:
            return tokens
        return F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.config.max_vision_tokens).transpose(1, 2)

    def forward(
        self,
        arm_actions: torch.Tensor,
        wrist_patch_tokens: torch.Tensor | None,
        hand_state: torch.Tensor,
        side_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = arm_actions.shape[0]
        if arm_actions.shape != (batch_size, self.config.action_horizon, self.config.arm_dim):
            raise ValueError(f"Unexpected arm action shape: {arm_actions.shape}")
        if self.config.condition_on_vision:
            if wrist_patch_tokens is None or wrist_patch_tokens.ndim != 3 or wrist_patch_tokens.shape[0] != batch_size:
                raise ValueError("Vision conditioning requires wrist patch tokens [B,N,F]")
            if wrist_patch_tokens.shape[-1] != self.config.vision_feature_dim:
                raise ValueError(
                    f"Expected vision feature dim {self.config.vision_feature_dim}, got {wrist_patch_tokens.shape[-1]}"
                )
        if hand_state.shape != (batch_size, self.config.hand_state_dim):
            raise ValueError(f"Unexpected hand state shape: {hand_state.shape}")
        if side_ids.shape != (batch_size,) or torch.any((side_ids < 0) | (side_ids > 1)):
            raise ValueError(f"Expected side_ids [B] containing 0/1, got {side_ids}")

        side = self.side_embedding[side_ids.long()][:, None, :]
        condition_tokens: list[torch.Tensor] = []
        if self.config.condition_on_arm:
            arm = self.arm_projection(arm_actions.detach())
            arm = arm + self.arm_positions[None, :, :] + self.modality_embedding[0] + side
            condition_tokens.append(arm)
        if self.config.condition_on_vision:
            assert wrist_patch_tokens is not None
            vision = self._pool_vision(wrist_patch_tokens)
            vision = self.vision_projection(self.vision_norm(vision))
            condition_tokens.append(vision + self.modality_embedding[1] + side)
        if self.config.condition_on_state:
            state = self.state_projection(hand_state)[:, None, :]
            condition_tokens.append(state + self.modality_embedding[2] + side)
        return torch.cat(condition_tokens, dim=1)


class HandDiT(nn.Module):
    """One shared-weight hand model, batched across independent left/right sides."""

    def __init__(self, config: HandDiTConfig) -> None:
        super().__init__()
        self.config = config
        dimension = config.model_dim
        self.condition_encoder = HandConditionEncoder(config)
        self.action_projection = nn.Linear(config.hand_dim, dimension)
        self.action_positions = nn.Parameter(torch.empty(config.action_horizon, dimension))
        self.time_mlp = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.blocks = nn.ModuleList(HandDiTBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 2 * dimension))
        self.output_projection = nn.Linear(dimension, config.hand_dim)
        nn.init.normal_(self.action_positions, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_hand_actions: torch.Tensor,
        flow_time: torch.Tensor,
        arm_actions: torch.Tensor,
        wrist_patch_tokens: torch.Tensor | None,
        hand_state: torch.Tensor,
        side_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_hand_actions.shape[0]
        expected = (batch_size, self.config.action_horizon, self.config.hand_dim)
        if noisy_hand_actions.shape != expected:
            raise ValueError(f"Expected noisy hand actions {expected}, got {noisy_hand_actions.shape}")
        condition = self.condition_encoder(arm_actions, wrist_patch_tokens, hand_state, side_ids)
        time_embedding = self.time_mlp(sinusoidal_embedding(flow_time, self.config.model_dim))
        hand_tokens = self.action_projection(noisy_hand_actions) + self.action_positions[None, :, :]
        for block in self.blocks:
            hand_tokens = block(hand_tokens, condition, time_embedding)
        shift, scale = self.final_modulation(time_embedding).chunk(2, dim=-1)
        hand_tokens = _modulate(self.final_norm(hand_tokens), shift, scale)
        return self.output_projection(hand_tokens)

    def flow_matching_loss(
        self,
        target_hand_actions: torch.Tensor,
        arm_actions: torch.Tensor,
        wrist_patch_tokens: torch.Tensor | None,
        hand_state: torch.Tensor,
        side_ids: torch.Tensor,
        *,
        side_valid: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Linear conditional-flow-matching loss from noise at t=0 to data at t=1."""
        batch_size = target_hand_actions.shape[0]
        noise = torch.randn_like(target_hand_actions) if noise is None else noise
        flow_time = (
            torch.rand(batch_size, device=target_hand_actions.device, dtype=target_hand_actions.dtype)
            if flow_time is None
            else flow_time
        )
        if noise.shape != target_hand_actions.shape or flow_time.shape != (batch_size,):
            raise ValueError("Noise or flow-time shape does not match the target batch")
        weight = flow_time[:, None, None]
        noisy_actions = (1.0 - weight) * noise + weight * target_hand_actions
        target_velocity = target_hand_actions - noise
        predicted_velocity = self(noisy_actions, flow_time, arm_actions, wrist_patch_tokens, hand_state, side_ids)
        loss = torch.mean(torch.square(predicted_velocity - target_velocity), dim=(-2, -1))
        if side_valid is None:
            return loss.mean()
        if side_valid.shape != (batch_size,):
            raise ValueError(f"Expected side_valid [B], got {side_valid.shape}")
        valid = side_valid.to(loss.dtype)
        return torch.sum(loss * valid) / torch.clamp(valid.sum(), min=1.0)

    @torch.no_grad()
    def sample(
        self,
        arm_actions: torch.Tensor,
        wrist_patch_tokens: torch.Tensor | None,
        hand_state: torch.Tensor,
        side_ids: torch.Tensor,
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler-integrate the learned flow from Gaussian noise (t=0) to actions (t=1)."""
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        batch_size = arm_actions.shape[0]
        shape = (batch_size, self.config.action_horizon, self.config.hand_dim)
        actions = torch.randn(shape, device=arm_actions.device, dtype=arm_actions.dtype) if noise is None else noise
        if actions.shape != shape:
            raise ValueError(f"Expected sampling noise {shape}, got {actions.shape}")
        step_size = 1.0 / num_steps
        for step in range(num_steps):
            flow_time = torch.full((batch_size,), step / num_steps, device=actions.device, dtype=actions.dtype)
            velocity = self(actions, flow_time, arm_actions, wrist_patch_tokens, hand_state, side_ids)
            actions = actions + step_size * velocity
        return actions
