"""Shared VLM adapter used by the second Task26 dual-expert policy."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class VLMActionAdapterConfig:
    vlm_feature_dim: int = 2048
    model_dim: int = 512
    action_dim: int = 16
    num_vlm_tokens: int = 8


class VLMActionAdapter(nn.Module):
    """Create shared Hand-DiT tokens and a residual correction to VLA actions."""

    def __init__(self, config: VLMActionAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.shared = nn.Sequential(
            nn.LayerNorm(config.vlm_feature_dim),
            nn.Linear(config.vlm_feature_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.token_projection = nn.Linear(config.model_dim, config.num_vlm_tokens * config.model_dim)
        self.action_residual = nn.Sequential(
            nn.Linear(config.action_dim + config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.action_dim),
        )
        nn.init.zeros_(self.action_residual[-1].weight)
        nn.init.zeros_(self.action_residual[-1].bias)

    def forward(self, normalized_actions: torch.Tensor, vlm_features: torch.Tensor):
        if normalized_actions.shape[-1] != self.config.action_dim:
            raise ValueError(f"Expected {self.config.action_dim}D actions, got {normalized_actions.shape}")
        if vlm_features.shape != (normalized_actions.shape[0], self.config.vlm_feature_dim):
            raise ValueError(f"Unexpected VLM feature shape {vlm_features.shape}")
        shared = self.shared(vlm_features.float())
        expanded = shared[:, None, :].expand(-1, normalized_actions.shape[1], -1)
        residual = self.action_residual(torch.cat((normalized_actions, expanded), dim=-1))
        corrected = normalized_actions + residual
        tokens = self.token_projection(shared).reshape(
            normalized_actions.shape[0], self.config.num_vlm_tokens, self.config.model_dim
        )
        return corrected, tokens
