"""OpenPI input/output transforms for the 58-DoF Bench2Dex Sharpa embodiment."""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies.dexterous_action_layout import TASK26_BIMANUAL_LAYOUT

TASK26_ARM_JOINT_DIM = 7
TASK26_ARM_CONDITION_DIM = 8
TASK26_BIMANUAL_ARM_ACTION_DIM = 2 * TASK26_ARM_CONDITION_DIM
TASK26_GRIPPER_ACTIVE_INDICES = (TASK26_ARM_CONDITION_DIM - 1, TASK26_BIMANUAL_ARM_ACTION_DIM - 1)


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
        for side in ("left", "right"):
            key = f"{side}_gripper_active_actions"
            if key in data:
                gripper_active = np.asarray(data[key], dtype=np.float32)
                if "actions" not in output or gripper_active.shape != output["actions"].shape[:-1]:
                    raise ValueError(
                        f"Expected {key} to match the action horizon, got "
                        f"{gripper_active.shape} and {None if 'actions' not in output else output['actions'].shape}"
                    )
                output[key] = gripper_active
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


@dataclasses.dataclass(frozen=True)
class Bench2DexArmInputs(transforms.DataTransformFn):
    """Project Task26 to 14D arm state and 16D arm-plus-activity actions."""

    def __call__(self, data: dict) -> dict:
        output = Bench2DexInputs(action_dim=TASK26_BIMANUAL_LAYOUT.action_dim)(data)
        arm_indices = (*TASK26_BIMANUAL_LAYOUT.left_arm, *TASK26_BIMANUAL_LAYOUT.right_arm)
        output["state"] = np.take(output["state"], arm_indices, axis=-1)
        left_active = output.pop("left_gripper_active_actions", None)
        right_active = output.pop("right_gripper_active_actions", None)
        if "actions" in output:
            if left_active is None or right_active is None:
                raise KeyError("Arm VLA training requires separate left/right gripper-active labels")
            arm_actions = np.take(output["actions"], arm_indices, axis=-1)
            output["actions"] = np.concatenate(
                (
                    arm_actions[..., :TASK26_ARM_JOINT_DIM],
                    left_active[..., None],
                    arm_actions[..., TASK26_ARM_JOINT_DIM:],
                    right_active[..., None],
                ),
                axis=-1,
            )
        if "action_states" in output:
            arm_states = np.take(output["action_states"], arm_indices, axis=-1)
            output["action_states"] = np.concatenate(
                (
                    arm_states[..., :TASK26_ARM_JOINT_DIM],
                    left_active[..., None],
                    arm_states[..., TASK26_ARM_JOINT_DIM:],
                    right_active[..., None],
                ),
                axis=-1,
            )
        return output


@dataclasses.dataclass(frozen=True)
class Bench2DexArmOutputs(transforms.DataTransformFn):
    """Crop π0.5 output to 16D and discretize its two activity labels."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        action_dim = TASK26_BIMANUAL_ARM_ACTION_DIM
        if actions.shape[-1] < action_dim:
            raise ValueError(f"Model returned only {actions.shape[-1]} dimensions; expected at least {action_dim}")
        actions = np.array(actions[..., :action_dim], copy=True)
        actions[..., TASK26_GRIPPER_ACTIVE_INDICES] = (actions[..., TASK26_GRIPPER_ACTIVE_INDICES] >= 0.5).astype(
            actions.dtype
        )
        return {"actions": actions}
