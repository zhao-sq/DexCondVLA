"""OpenPI input/output transforms for the 58-DoF Bench2Dex Sharpa embodiment."""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class Bench2DexInputs(transforms.DataTransformFn):
    action_dim: int = 58

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        camera_map = {
            "base_0_rgb": "cam_overhead",
            "left_wrist_0_rgb": "cam_wrist_left",
            "right_wrist_0_rgb": "cam_wrist_right",
        }
        missing = set(camera_map.values()) - set(images)
        if missing:
            raise KeyError(f"Missing Bench2Dex cameras: {sorted(missing)}")

        def parse_image(value) -> np.ndarray:
            image = np.asarray(value)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB image, got shape={image.shape}")
            if image.dtype != np.uint8:
                raise ValueError(f"Expected uint8 RGB image, got dtype={image.dtype}")
            return image

        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != self.action_dim:
            raise ValueError(f"Expected {self.action_dim}D state, got {state.shape}")

        output = {
            "state": state,
            "image": {target: parse_image(images[source]) for target, source in camera_map.items()},
            "image_mask": dict.fromkeys(camera_map, np.True_),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != self.action_dim:
                raise ValueError(f"Expected {self.action_dim}D actions, got {actions.shape}")
            output["actions"] = actions
        if "action_states" in data:
            action_states = np.asarray(data["action_states"], dtype=np.float32)
            if "actions" not in output or action_states.shape != output["actions"].shape:
                raise ValueError(
                    f"Expected action_states to match actions, got {action_states.shape} and "
                    f"{None if 'actions' not in output else output['actions'].shape}"
                )
            output["action_states"] = action_states
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class Bench2DexOutputs(transforms.DataTransformFn):
    action_dim: int = 58

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < self.action_dim:
            raise ValueError(f"Model returned only {actions.shape[-1]} action dimensions")
        return {"actions": actions[..., : self.action_dim]}
