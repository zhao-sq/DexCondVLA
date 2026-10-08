import numpy as np

from scripts.serve_bench2dex_rpc_policy import _warm_up_policy


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
