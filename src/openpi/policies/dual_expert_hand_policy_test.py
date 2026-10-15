from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from openpi.models_pytorch.hand_dit import HandDiT
from openpi.models_pytorch.hand_dit import HandDiTConfig
from openpi.models_pytorch.vlm_action_adapter import VLMActionAdapter
from openpi.models_pytorch.vlm_action_adapter import VLMActionAdapterConfig
from openpi.policies.dexterous_action_layout import TASK26_BIMANUAL_LAYOUT
from openpi.policies.dexterous_action_layout import BimanualActionLayout
from openpi.policies.dual_expert_hand_policy import DualExpertHandPolicy
from openpi.policies.dual_expert_hand_policy import DualExpertNormalization
from openpi.policies.dual_expert_hand_policy import load_hand_dit_checkpoint
from openpi.policies.dual_expert_hand_policy import save_hand_dit_checkpoint


def test_task26_layout_partitions_and_replaces_only_hands() -> None:
    layout = TASK26_BIMANUAL_LAYOUT
    full = np.arange(2 * 58, dtype=np.float32).reshape(2, 58)
    left = np.full((2, 22), -1.0, dtype=np.float32)
    right = np.full((2, 22), 2.0, dtype=np.float32)

    replaced = layout.replace_hands(full, left, right)

    np.testing.assert_array_equal(replaced[:, layout.left_arm], full[:, layout.left_arm])
    np.testing.assert_array_equal(replaced[:, layout.right_arm], full[:, layout.right_arm])
    np.testing.assert_array_equal(replaced[:, layout.left_hand], left)
    np.testing.assert_array_equal(replaced[:, layout.right_hand], right)

    left_arm = np.full((2, 7), 3.0, dtype=np.float32)
    right_arm = np.full((2, 7), 4.0, dtype=np.float32)
    arms_replaced = layout.replace_arms(full, left_arm, right_arm)
    np.testing.assert_array_equal(arms_replaced[:, layout.left_arm], left_arm)
    np.testing.assert_array_equal(arms_replaced[:, layout.right_arm], right_arm)
    np.testing.assert_array_equal(arms_replaced[:, layout.left_hand], full[:, layout.left_hand])
    np.testing.assert_array_equal(arms_replaced[:, layout.right_hand], full[:, layout.right_hand])


class _DummyVlaPolicy:
    metadata: ClassVar = {"config": "arm_vla"}

    def infer(self, _obs: dict) -> dict:
        actions = np.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 1.0, 3.0, 0.0]], dtype=np.float32)
        return {"actions": actions, "policy_timing": {"infer_ms": 1.0}}


class _DummyWristEncoder(nn.Module):
    def forward(self, images):
        return torch.ones(len(images), 4, 8)


class _DummyHandModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = HandDiTConfig(
            action_horizon=2,
            arm_dim=2,
            hand_dim=3,
            hand_state_dim=3,
            vision_feature_dim=8,
            model_dim=16,
            num_layers=1,
            num_heads=4,
        )

    def sample(self, arm_actions, wrist_patch_tokens, hand_state, side_ids, *, num_steps):
        assert arm_actions.shape == (2, 2, 2)
        torch.testing.assert_close(arm_actions[..., -1], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        assert wrist_patch_tokens.shape == (2, 4, 8)
        assert hand_state.shape == (2, 3)
        torch.testing.assert_close(side_ids, torch.tensor([0, 1]))
        return torch.zeros(2, 2, 3)


def _normalization(action_dim: int) -> DualExpertNormalization:
    zeros = (0.0,) * action_dim
    ones = (1.0,) * action_dim
    twos = (2.0,) * action_dim
    return DualExpertNormalization(
        state_mean=zeros,
        state_std=ones,
        state_q01=zeros,
        state_q99=twos,
        action_mean=zeros,
        action_std=ones,
        action_q01=zeros,
        action_q99=twos,
    )


def test_dual_policy_preserves_vla_arms_and_replaces_only_hands() -> None:
    layout = BimanualActionLayout(
        action_dim=8,
        left_arm=(0,),
        right_arm=(1,),
        left_hand=(2, 3, 4),
        right_hand=(5, 6, 7),
    )
    policy = DualExpertHandPolicy(
        _DummyVlaPolicy(),
        _DummyHandModel(),
        _DummyWristEncoder(),
        _normalization(8),
        layout=layout,
        left_wrist_key="left",
        right_wrist_key="right",
        device="cpu",
    )
    observation = {
        "state": np.arange(8, dtype=np.float32),
        "images": {
            "left": np.zeros((16, 16, 3), dtype=np.uint8),
            "right": np.zeros((16, 16, 3), dtype=np.uint8),
        },
    }

    result = policy.infer(observation)

    np.testing.assert_array_equal(result["actions"][:, layout.left_arm], np.asarray([[0.0], [2.0]]))
    np.testing.assert_array_equal(result["actions"][:, layout.right_arm], np.asarray([[1.0], [3.0]]))
    np.testing.assert_allclose(result["actions"][:, layout.left_hand], 1.0, atol=1e-6)
    np.testing.assert_allclose(result["actions"][:, layout.right_hand], 1.0, atol=1e-6)
    assert result["policy_timing"]["infer_ms"] == 1.0
    assert policy.metadata["config"] == "arm_vla"
    assert policy.metadata["hand_policy"] == "shared_independent_hand_dit_v1"
    assert policy.metadata["vla_physical_action_dim"] == 4
    assert policy.metadata["composed_action_dim"] == 8


def test_hand_checkpoint_round_trip(tmp_path: Path) -> None:
    model = HandDiT(
        HandDiTConfig(
            action_horizon=2,
            arm_dim=1,
            hand_dim=3,
            hand_state_dim=3,
            vision_feature_dim=8,
            model_dim=16,
            num_layers=1,
            num_heads=4,
        )
    )
    checkpoint_path = tmp_path / "hand.pt"

    save_hand_dit_checkpoint(
        checkpoint_path,
        model,
        _normalization(8),
        step=12,
        metadata={"arm_condition_source": "ground_truth"},
    )
    restored, normalization, checkpoint = load_hand_dit_checkpoint(checkpoint_path, device="cpu")

    assert restored.config == model.config
    assert normalization == _normalization(8)
    assert checkpoint["step"] == 12
    assert checkpoint["metadata"]["arm_condition_source"] == "ground_truth"
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_v2_checkpoint_restores_adapter(tmp_path: Path) -> None:
    model = HandDiT(
        HandDiTConfig(
            action_horizon=2,
            arm_dim=1,
            hand_dim=3,
            hand_state_dim=3,
            vision_feature_dim=8,
            model_dim=16,
            num_layers=1,
            num_heads=4,
            condition_on_vlm=True,
            vlm_feature_dim=16,
            detach_arm_condition=False,
        )
    )
    adapter = VLMActionAdapter(VLMActionAdapterConfig(vlm_feature_dim=12, model_dim=16, action_dim=4, num_vlm_tokens=2))
    checkpoint_path = tmp_path / "v2.pt"
    save_hand_dit_checkpoint(checkpoint_path, model, _normalization(8), step=3, adapter=adapter)

    restored, _, checkpoint = load_hand_dit_checkpoint(checkpoint_path, device="cpu")

    assert checkpoint["format_version"] == 2
    assert restored.config.condition_on_vlm
    assert isinstance(checkpoint["adapter_model"], VLMActionAdapter)
