"""Action-space layouts for composing a VLA arm policy with hand-only experts."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class BimanualActionLayout:
    """Indices that split a full robot action into two arms and two hands.

    The VLA emits contiguous left/right arm actions while the environment uses
    a full, interleaved action ordering. This layout owns the corresponding
    gather/scatter boundary for the arm VLA and hand expert.
    """

    action_dim: int
    left_arm: tuple[int, ...]
    right_arm: tuple[int, ...]
    left_hand: tuple[int, ...]
    right_hand: tuple[int, ...]
    name: str = "custom"

    def __post_init__(self) -> None:
        groups = (self.left_arm, self.right_arm, self.left_hand, self.right_hand)
        flat = tuple(index for group in groups for index in group)
        if not all(groups):
            raise ValueError("Every arm/hand index group must be non-empty")
        if len(flat) != len(set(flat)):
            raise ValueError("Arm/hand index groups overlap")
        if min(flat) < 0 or max(flat) >= self.action_dim:
            raise ValueError(f"Layout indices must be in [0, {self.action_dim})")
        if set(flat) != set(range(self.action_dim)):
            missing = sorted(set(range(self.action_dim)) - set(flat))
            raise ValueError(f"Layout must partition all {self.action_dim} channels; missing={missing}")
        if len(self.left_arm) != len(self.right_arm):
            raise ValueError("Shared hand policy requires equal left/right arm dimensions")
        if len(self.left_hand) != len(self.right_hand):
            raise ValueError("Shared hand policy requires equal left/right hand dimensions")

    @property
    def arm_dim(self) -> int:
        return len(self.left_arm)

    @property
    def hand_dim(self) -> int:
        return len(self.left_hand)

    @property
    def arm_action_dim(self) -> int:
        """Physical VLA action width in [left arm, right arm] order."""
        return len(self.left_arm) + len(self.right_arm)

    def indices(self, side: str, part: str) -> tuple[int, ...]:
        if side not in {"left", "right"}:
            raise ValueError(f"Unknown side: {side!r}")
        if part not in {"arm", "hand"}:
            raise ValueError(f"Unknown action part: {part!r}")
        return getattr(self, f"{side}_{part}")

    def gather(self, values: np.ndarray | torch.Tensor, side: str, part: str):
        indices = self.indices(side, part)
        if values.shape[-1] != self.action_dim:
            raise ValueError(f"Expected last dimension {self.action_dim}, got {values.shape}")
        if isinstance(values, torch.Tensor):
            index = torch.as_tensor(indices, device=values.device)
            return torch.index_select(values, -1, index)
        return np.take(values, indices, axis=-1)

    def replace_hands(
        self,
        full_actions: np.ndarray | torch.Tensor,
        left_actions: np.ndarray | torch.Tensor,
        right_actions: np.ndarray | torch.Tensor,
    ):
        if full_actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected full actions ending in {self.action_dim}, got {full_actions.shape}")
        expected = (*full_actions.shape[:-1], self.hand_dim)
        if left_actions.shape != expected or right_actions.shape != expected:
            raise ValueError(
                f"Expected each hand action to have shape {expected}, got {left_actions.shape} and {right_actions.shape}"
            )
        output = full_actions.clone() if isinstance(full_actions, torch.Tensor) else np.array(full_actions, copy=True)
        output[..., self.left_hand] = left_actions
        output[..., self.right_hand] = right_actions
        return output

    def replace_arms(
        self,
        full_actions: np.ndarray | torch.Tensor,
        left_actions: np.ndarray | torch.Tensor,
        right_actions: np.ndarray | torch.Tensor,
    ):
        """Scatter contiguous per-side VLA outputs into the full robot layout."""
        if full_actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected full actions ending in {self.action_dim}, got {full_actions.shape}")
        expected = (*full_actions.shape[:-1], self.arm_dim)
        if left_actions.shape != expected or right_actions.shape != expected:
            raise ValueError(
                f"Expected each arm action to have shape {expected}, got {left_actions.shape} and {right_actions.shape}"
            )
        output = full_actions.clone() if isinstance(full_actions, torch.Tensor) else np.array(full_actions, copy=True)
        output[..., self.left_arm] = left_actions
        output[..., self.right_arm] = right_actions
        return output

    def validate_joint_names(self, joint_names: Sequence[str]) -> None:
        if len(joint_names) != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} joint names, got {len(joint_names)}")
        if not all(joint_names[index].startswith("left_") for index in self.left_hand):
            raise ValueError("Left-hand indices do not match left_ joint names")
        if not all(joint_names[index].startswith("multi_right_") for index in self.right_hand):
            raise ValueError("Right-hand indices do not match multi_right_ joint names")


# Task26 ordering is interleaved across the two KUKA arms and, separately,
# across the two dexterous hands.  Keep the explicit indices here rather than
# relying on a contiguous [arms, hands] assumption.
TASK26_BIMANUAL_LAYOUT = BimanualActionLayout(
    name="bench2dex_task26",
    action_dim=58,
    left_arm=(0, 1, 3, 5, 7, 9, 11),
    right_arm=(2, 4, 6, 8, 10, 12, 13),
    left_hand=(14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 49, 50),
    right_hand=(24, 25, 26, 27, 28, 34, 35, 36, 37, 38, 44, 45, 46, 47, 48, 51, 52, 53, 54, 55, 56, 57),
)
