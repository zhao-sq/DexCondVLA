"""Runtime composition of a 16D arm/activity VLA and a hand-only DiT."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import pathlib
import time
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
import torch

from openpi.models_pytorch.dinov3_wrist_encoder import Dinov3WristEncoder
from openpi.models_pytorch.hand_dit import HandDiT
from openpi.models_pytorch.hand_dit import HandDiTConfig
from openpi.policies.dexterous_action_layout import TASK26_BIMANUAL_LAYOUT
from openpi.policies.dexterous_action_layout import BimanualActionLayout


@dataclasses.dataclass(frozen=True)
class DualExpertNormalization:
    """Full 58D statistics used for Hand DiT targets and conditions."""

    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    state_q01: tuple[float, ...]
    state_q99: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    action_q01: tuple[float, ...]
    action_q99: tuple[float, ...]
    use_quantiles: bool = True

    @classmethod
    def from_openpi_json(cls, path: pathlib.Path | str, *, use_quantiles: bool = True):
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["norm_stats"]
        state = payload["state"]
        action = payload["actions"]
        return cls(
            state_mean=tuple(state["mean"]),
            state_std=tuple(state["std"]),
            state_q01=tuple(state["q01"]),
            state_q99=tuple(state["q99"]),
            action_mean=tuple(action["mean"]),
            action_std=tuple(action["std"]),
            action_q01=tuple(action["q01"]),
            action_q99=tuple(action["q99"]),
            use_quantiles=use_quantiles,
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]):
        return cls(**{key: tuple(value) if isinstance(value, list) else value for key, value in values.items()})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def _statistics(self, kind: str, indices: tuple[int, ...], reference: torch.Tensor):
        if self.use_quantiles:
            low = getattr(self, f"{kind}_q01")
            high = getattr(self, f"{kind}_q99")
            offset = torch.as_tensor(np.take(low, indices), device=reference.device, dtype=reference.dtype)
            scale = torch.as_tensor(
                np.take(high, indices) - np.take(low, indices), device=reference.device, dtype=reference.dtype
            )
            return offset, scale, True
        mean = getattr(self, f"{kind}_mean")
        std = getattr(self, f"{kind}_std")
        offset = torch.as_tensor(np.take(mean, indices), device=reference.device, dtype=reference.dtype)
        scale = torch.as_tensor(np.take(std, indices), device=reference.device, dtype=reference.dtype)
        return offset, scale, False

    def normalize(self, values: torch.Tensor, *, kind: str, indices: tuple[int, ...]) -> torch.Tensor:
        offset, scale, quantile = self._statistics(kind, indices, values)
        normalized = (values - offset) / (scale + 1e-6)
        return normalized * 2.0 - 1.0 if quantile else normalized

    def unnormalize_actions(self, values: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        offset, scale, quantile = self._statistics("action", indices, values)
        values = (values + 1.0) / 2.0 if quantile else values
        return values * (scale + 1e-6) + offset


def save_hand_dit_checkpoint(
    path: pathlib.Path | str,
    model: HandDiT,
    normalization: DualExpertNormalization,
    *,
    step: int,
    optimizer_state: dict[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "model_config": dataclasses.asdict(model.config),
            "model_state": model.state_dict(),
            "normalization": normalization.to_dict(),
            "step": int(step),
            "optimizer_state": optimizer_state,
            "metadata": dict(metadata or {}),
        },
        temporary,
    )
    temporary.replace(path)


def load_hand_dit_checkpoint(
    path: pathlib.Path | str, *, device: torch.device | str
) -> tuple[HandDiT, DualExpertNormalization, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported hand checkpoint format: {checkpoint.get('format_version')}")
    model = HandDiT(HandDiTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    normalization = DualExpertNormalization.from_dict(checkpoint["normalization"])
    return model, normalization, checkpoint


def _extract_state(observation: Mapping[str, Any]) -> np.ndarray:
    if "state" in observation:
        state = observation["state"]
    else:
        joint_action = observation.get("joint_action", {})
        state = joint_action.get("vector", joint_action.get("qpos"))
    if state is None:
        raise KeyError("Dual-expert observation is missing state or joint_action.vector/qpos")
    return np.asarray(state, dtype=np.float32).reshape(-1)


def _extract_image(observation: Mapping[str, Any], name: str) -> np.ndarray:
    images = observation.get("images", observation.get("observation", observation))
    value = images.get(name) if isinstance(images, Mapping) else None
    if isinstance(value, Mapping):
        value = value.get("rgb")
    if value is None:
        raise KeyError(f"Dual-expert observation is missing wrist image {name!r}")
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC wrist image for {name}, got {image.shape}")
    return image


class DualExpertHandPolicy(_base_policy.BasePolicy):
    """Run 16D π₀.₅ arm/activity output, then compose it with Hand DiT.

    The wrapped VLA receives the original 58D observation; its data transform
    predicts [left arm 7, activity, right arm 7, activity], pads that physical
    16D representation to the official π0.5 32D model width, and crops inference
    back to 16D. The
    wrapper scatters those arms into Task26's interleaved 58D runtime layout.
    Left/right hand examples are stacked on the batch dimension for one
    shared-weight Hand DiT call, with no cross-hand attention.
    """

    def __init__(
        self,
        vla_policy: _base_policy.BasePolicy,
        hand_model: HandDiT,
        wrist_encoder: Dinov3WristEncoder | None,
        normalization: DualExpertNormalization,
        *,
        layout: BimanualActionLayout = TASK26_BIMANUAL_LAYOUT,
        left_wrist_key: str = "cam_wrist_left",
        right_wrist_key: str = "cam_wrist_right",
        num_hand_steps: int = 10,
        device: torch.device | str = "cuda",
    ) -> None:
        self._vla_policy = vla_policy
        self._hand_model = hand_model.to(device).eval()
        self._wrist_encoder = wrist_encoder.to(device).eval() if wrist_encoder is not None else None
        self._normalization = normalization
        self._layout = layout
        self._left_wrist_key = left_wrist_key
        self._right_wrist_key = right_wrist_key
        self._num_hand_steps = num_hand_steps
        self._device = torch.device(device)
        config = hand_model.config
        if (config.arm_dim, config.hand_dim, config.hand_state_dim) != (
            layout.arm_dim + 1,
            layout.hand_dim,
            layout.hand_dim,
        ):
            raise ValueError("HandDiT dimensions do not match the action layout")
        if config.condition_on_vision and self._wrist_encoder is None:
            raise ValueError("Vision-conditioned HandDiT requires a wrist encoder")

    @property
    def _rng(self):
        """Forward JAX RNG access so existing warm-up/reset code remains deterministic."""
        return self._vla_policy._rng  # noqa: SLF001

    @_rng.setter
    def _rng(self, value) -> None:
        self._vla_policy._rng = value  # noqa: SLF001

    def infer(self, obs: dict) -> dict:  # type: ignore[override]
        total_start = time.monotonic()
        vla_result = self._vla_policy.infer(obs)
        vla_actions = np.asarray(vla_result["actions"], dtype=np.float32)
        condition_dim = self._hand_model.config.arm_dim
        expected = (self._hand_model.config.action_horizon, 2 * condition_dim)
        if vla_actions.shape != expected:
            raise ValueError(f"Expected VLA actions {expected}, got {vla_actions.shape}")
        state = _extract_state(obs)
        if state.shape != (self._layout.action_dim,):
            raise ValueError(f"Expected state [{self._layout.action_dim}], got {state.shape}")

        with torch.inference_mode():
            if self._hand_model.config.condition_on_vision:
                images = np.stack(
                    (
                        _extract_image(obs, self._left_wrist_key),
                        _extract_image(obs, self._right_wrist_key),
                    )
                )
                assert self._wrist_encoder is not None
                wrist_tokens = self._wrist_encoder(images)
            else:
                wrist_tokens = None
            vla_arm_actions = torch.as_tensor(vla_actions, device=self._device)
            full_state = torch.as_tensor(state, device=self._device)
            left_condition, right_condition = torch.split(vla_arm_actions, condition_dim, dim=-1)
            left_arm = left_condition[..., : self._layout.arm_dim]
            right_arm = right_condition[..., : self._layout.arm_dim]
            full_actions = full_state.expand(vla_arm_actions.shape[0], -1).clone()
            full_actions = self._layout.replace_arms(full_actions, left_arm, right_arm)
            arm_conditions = torch.stack(
                (
                    left_condition,
                    right_condition,
                )
            )
            hand_state = torch.stack(
                (
                    self._layout.gather(full_state, "left", "hand"),
                    self._layout.gather(full_state, "right", "hand"),
                )
            )
            arm_actions = torch.stack(
                (
                    torch.cat(
                        (
                            self._normalization.normalize(
                                arm_conditions[0, ..., : self._layout.arm_dim],
                                kind="action",
                                indices=self._layout.left_arm,
                            ),
                            arm_conditions[0, ..., -1:],
                        ),
                        dim=-1,
                    ),
                    torch.cat(
                        (
                            self._normalization.normalize(
                                arm_conditions[1, ..., : self._layout.arm_dim],
                                kind="action",
                                indices=self._layout.right_arm,
                            ),
                            arm_conditions[1, ..., -1:],
                        ),
                        dim=-1,
                    ),
                )
            )
            hand_state = torch.stack(
                (
                    self._normalization.normalize(hand_state[0], kind="state", indices=self._layout.left_hand),
                    self._normalization.normalize(hand_state[1], kind="state", indices=self._layout.right_hand),
                )
            )
            side_ids = torch.tensor((0, 1), device=self._device)
            normalized_hands = self._hand_model.sample(
                arm_actions,
                wrist_tokens,
                hand_state,
                side_ids,
                num_steps=self._num_hand_steps,
            )
            left_hand = self._normalization.unnormalize_actions(normalized_hands[0], self._layout.left_hand)
            right_hand = self._normalization.unnormalize_actions(normalized_hands[1], self._layout.right_hand)
            actions = self._layout.replace_hands(full_actions, left_hand, right_hand).cpu().numpy()

        result = dict(vla_result)
        result["actions"] = actions
        prior_timing = dict(result.get("policy_timing", {}))
        prior_timing["dual_expert_total_ms"] = (time.monotonic() - total_start) * 1000.0
        result["policy_timing"] = prior_timing
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        base_metadata = dict(getattr(self._vla_policy, "metadata", {}))
        base_metadata.update(
            {
                "hand_policy": "shared_independent_hand_dit_v1",
                "hand_dim_per_side": self._layout.hand_dim,
                "hand_action_horizon": self._hand_model.config.action_horizon,
                "hand_flow_steps": self._num_hand_steps,
                "vla_physical_action_dim": 2 * self._hand_model.config.arm_dim,
                "composed_action_dim": self._layout.action_dim,
                "wrist_encoder": (
                    "dinov3_vitb16_frozen_patch_tokens" if self._hand_model.config.condition_on_vision else "disabled"
                ),
            }
        )
        return base_metadata
