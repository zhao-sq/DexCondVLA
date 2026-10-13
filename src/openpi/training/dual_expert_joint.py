"""Utilities shared by Task26 VLA + Hand DiT joint training."""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import torch

from openpi.policies.dexterous_action_layout import TASK26_BIMANUAL_LAYOUT
from openpi.policies.dual_expert_hand_policy import DualExpertNormalization


def predicted_vla_actions_to_hand_condition(
    normalized_vla_actions: np.ndarray,
    *,
    vla_action_q01: np.ndarray,
    vla_action_q99: np.ndarray,
    hand_normalization: DualExpertNormalization,
    device: torch.device,
) -> torch.Tensor:
    """Convert normalized VLA [L7, active, R7, active] to Hand-DiT conditions.

    This conversion deliberately crosses NumPy and creates a detached torch
    tensor: Hand loss can update Hand DiT, but can never backpropagate into VLA.
    """
    values = np.asarray(normalized_vla_actions, dtype=np.float32)[..., :16]
    low = np.asarray(vla_action_q01, dtype=np.float32)[:16]
    high = np.asarray(vla_action_q99, dtype=np.float32)[:16]
    physical = (values + 1.0) * 0.5 * (high - low + 1e-6) + low
    physical = torch.from_numpy(physical).to(device=device)
    layout = TASK26_BIMANUAL_LAYOUT
    left_arm = hand_normalization.normalize(physical[..., :7], kind="action", indices=layout.left_arm)
    right_arm = hand_normalization.normalize(physical[..., 8:15], kind="action", indices=layout.right_arm)
    left_active = physical[..., 7:8].clamp(0.0, 1.0)
    right_active = physical[..., 15:16].clamp(0.0, 1.0)
    return torch.cat((torch.cat((left_arm, left_active), dim=-1), torch.cat((right_arm, right_active), dim=-1)), dim=0)


def hand_targets_and_state(
    state: torch.Tensor,
    actions: torch.Tensor,
    normalization: DualExpertNormalization,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stacked left/right normalized hand targets, state, and side ids."""
    layout = TASK26_BIMANUAL_LAYOUT
    left_target = normalization.normalize(
        layout.gather(actions, "left", "hand"), kind="action", indices=layout.left_hand
    )
    right_target = normalization.normalize(
        layout.gather(actions, "right", "hand"), kind="action", indices=layout.right_hand
    )
    left_state = normalization.normalize(layout.gather(state, "left", "hand"), kind="state", indices=layout.left_hand)
    right_state = normalization.normalize(
        layout.gather(state, "right", "hand"), kind="state", indices=layout.right_hand
    )
    side_ids = torch.cat(
        (
            torch.zeros(state.shape[0], device=state.device, dtype=torch.long),
            torch.ones(state.shape[0], device=state.device, dtype=torch.long),
        )
    )
    return torch.cat((left_target, right_target), dim=0), torch.cat((left_state, right_state), dim=0), side_ids


def retain_latest_and_best(checkpoint_root: pathlib.Path, *, latest_step: int, best_step: int) -> list[int]:
    """Delete numeric checkpoint directories except the latest and validation best."""
    keep = {int(latest_step), int(best_step)}
    removed: list[int] = []
    for path in checkpoint_root.iterdir():
        if path.is_dir() and path.name.isdigit() and int(path.name) not in keep:
            shutil.rmtree(path)
            removed.append(int(path.name))
    (checkpoint_root / "LATEST").write_text(f"{latest_step}\n", encoding="utf-8")
    (checkpoint_root / "BEST").write_text(f"{best_step}\n", encoding="utf-8")
    return sorted(removed)
