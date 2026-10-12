import numpy as np

from scripts.serve_bench2dex_rpc_policy import (
    TASK26_ALOHA_FULL_INDICES,
    _expand_task26_aloha_arm_actions,
    _extract_policy_observation,
    _warm_up_policy,
)


class _DummyPolicy:
    def __init__(self) -> None:
        self._rng = object()
        self.original_rng = self._rng
        self.observations = []

    def infer(self, observation: dict) -> dict:
        self.observations.append(observation)
        self._rng = object()
        return {"actions": np.zeros((50, 58), dtype=np.float32)}


def test_warm_up_uses_policy_observation_shapes_and_restores_rng() -> None:
    policy = _DummyPolicy()

    _warm_up_policy(policy, runs=2, image_height=480, image_width=640, prompt="task")

    assert len(policy.observations) == 2
    assert policy._rng is policy.original_rng  # noqa: SLF001
    observation = policy.observations[0]
    assert observation["state"].shape == (58,)
    assert observation["prompt"] == "task"
    assert set(observation["images"]) == {
        "cam_overhead",
        "cam_wrist_left",
        "cam_wrist_right",
    }
    assert all(image.shape == (480, 640, 3) for image in observation["images"].values())


def _raw_task26_observation() -> dict:
    return {
        "observation": {
            name: {"rgb": np.zeros((48, 64, 3), dtype=np.uint8)}
            for name in ("cam_overhead", "cam_wrist_left", "cam_wrist_right")
        },
        "joint_action": {"vector": np.arange(58, dtype=np.float32)},
        "language": "move the cans",
    }


def test_task26_aloha_observation_maps_only_six_joints_per_arm() -> None:
    observation = _extract_policy_observation(
        _raw_task26_observation(), None, task26_aloha_arm_only=True
    )

    np.testing.assert_array_equal(
        observation["state"],
        np.asarray([0, 1, 3, 5, 7, 9, 1, 2, 4, 6, 8, 10, 12, 1], dtype=np.float32),
    )
    assert observation["images"]["cam_high"].shape == (3, 48, 64)
    assert observation["prompt"] == "move the cans"


def test_task26_aloha_actions_hold_uncontrolled_joints() -> None:
    state = np.arange(58, dtype=np.float32)
    aloha_actions = np.arange(2 * 14, dtype=np.float32).reshape(2, 14)

    expanded = _expand_task26_aloha_arm_actions(aloha_actions, state)

    np.testing.assert_array_equal(expanded[:, TASK26_ALOHA_FULL_INDICES], aloha_actions[:, [*range(6), *range(7, 13)]])
    uncontrolled = np.ones(58, dtype=bool)
    uncontrolled[TASK26_ALOHA_FULL_INDICES] = False
    np.testing.assert_array_equal(expanded[:, uncontrolled], np.broadcast_to(state[uncontrolled], (2, 46)))
