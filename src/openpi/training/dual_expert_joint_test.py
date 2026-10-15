import pathlib

import numpy as np
import torch

from openpi.policies.dual_expert_hand_policy import DualExpertNormalization
from openpi.training.dual_expert_joint import predicted_vla_actions_to_hand_condition
from openpi.training.dual_expert_joint import retain_latest_and_best


def _normalization() -> DualExpertNormalization:
    zeros = (0.0,) * 58
    ones = (1.0,) * 58
    return DualExpertNormalization(zeros, ones, zeros, ones, zeros, ones, zeros, ones)


def test_predicted_condition_is_split_stacked_and_detached():
    values = np.zeros((2, 3, 32), dtype=np.float32)
    values[..., 7] = 1.0
    condition = predicted_vla_actions_to_hand_condition(
        values,
        vla_action_q01=np.zeros(16),
        vla_action_q99=np.ones(16),
        hand_normalization=_normalization(),
        device=torch.device("cpu"),
    )
    assert condition.shape == (4, 3, 8)
    assert not condition.requires_grad
    torch.testing.assert_close(condition[:2, ..., -1], torch.ones(2, 3))
    torch.testing.assert_close(condition[2:, ..., -1], torch.ones(2, 3))


def test_retention_keeps_only_latest_and_best(tmp_path: pathlib.Path):
    for step in (100, 200, 300):
        (tmp_path / str(step)).mkdir()
    assert retain_latest_and_best(tmp_path, latest_step=300, best_step=100) == [200]
    assert {path.name for path in tmp_path.iterdir()} == {"100", "300", "LATEST", "BEST"}
    retain_latest_and_best(tmp_path, latest_step=300, best_step=300)
    assert {path.name for path in tmp_path.iterdir()} == {"300", "LATEST", "BEST"}
