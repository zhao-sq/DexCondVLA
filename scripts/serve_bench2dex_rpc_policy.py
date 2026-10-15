"""Serve an OpenPI checkpoint through Bench2Dex's length-prefixed RPC protocol.

This adapter intentionally lives in the OpenPI environment.  The Bench2Dex
Isaac Sim client runs in a separate environment and sends raw observations over
a local TCP socket.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import logging
import pathlib
import signal
import socket
import threading
import time
import traceback
from typing import Any

import jax
import numpy as np
import torch

from openpi.models_pytorch.dinov3_wrist_encoder import Dinov3WristEncoder
from openpi.policies import policy_config
from openpi.policies.dual_expert_hand_policy import DualExpertHandPolicy
from openpi.policies.dual_expert_hand_policy import load_hand_dit_checkpoint
from openpi.training import config as training_config

DEFAULT_WARMUP_PROMPT = (
    "Use left hand to place the master chef can and MSG onto the tray, use right hand to place "
    "the milk box and potted meat can onto the tray."
)

# Task26's runtime joint order interleaves the two KUKA arms before the hand
# joints.  The pi0.5 ALOHA embodiment is [left arm 6, left gripper, right arm 6,
# right gripper].  For a conservative zero-shot probe, map only those 12 arm
# channels and hold both KUKA joint 7 axes plus all 44 hand joints in place.
TASK26_ALOHA_LEFT_ARM_INDICES = np.asarray((0, 1, 3, 5, 7, 9), dtype=np.int64)
TASK26_ALOHA_RIGHT_ARM_INDICES = np.asarray((2, 4, 6, 8, 10, 12), dtype=np.int64)
TASK26_ALOHA_ACTION_INDICES = np.asarray((*range(6), *range(7, 13)), dtype=np.int64)
TASK26_ALOHA_FULL_INDICES = np.concatenate((TASK26_ALOHA_LEFT_ARM_INDICES, TASK26_ALOHA_RIGHT_ARM_INDICES))


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _json_to_numpy(payload: bytes) -> Any:
    def object_hook(item: dict[str, Any]) -> Any:
        if item.get("__numpy_array__"):
            raw = base64.b64decode(item["data"])
            return np.frombuffer(raw, dtype=np.dtype(item["dtype"])).reshape(item["shape"])
        return item

    return json.loads(payload.decode("utf-8"), object_hook=object_hook)


def _recv_message(sock: socket.socket) -> Any | None:
    length_bytes = sock.recv(4)
    if not length_bytes:
        return None
    remaining = int.from_bytes(length_bytes, "big")
    chunks: list[bytes] = []
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("Incomplete Bench2Dex RPC payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return _json_to_numpy(b"".join(chunks))


def _send_message(sock: socket.socket, payload: Any) -> None:
    encoded = json.dumps(payload, cls=_NumpyEncoder).encode("utf-8")
    sock.sendall(len(encoded).to_bytes(4, "big"))
    sock.sendall(encoded)


def _extract_full_state(raw: dict[str, Any]) -> np.ndarray:
    joint_action = raw.get("joint_action", {})
    state_value = joint_action.get("vector", joint_action.get("qpos"))
    if state_value is None:
        raise KeyError("Bench2Dex observation is missing joint_action.vector/qpos")
    state = np.asarray(state_value, dtype=np.float32).reshape(-1)
    if state.shape != (58,):
        raise ValueError(f"Expected a 58D Task26 state, got {state.shape}")
    return state


def _extract_policy_observation(
    raw: dict[str, Any], instruction: str | None, *, task26_aloha_arm_only: bool = False
) -> dict[str, Any]:
    observation = raw.get("observation", raw)

    def camera(name: str) -> np.ndarray:
        try:
            image = np.asarray(observation[name]["rgb"])
        except KeyError as exc:
            raise KeyError(f"Bench2Dex observation is missing {name}.rgb") from exc
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected uint8 HWC RGB for {name}, got {image.shape} {image.dtype}")
        return image

    state = _extract_full_state(raw)
    prompt = instruction or raw.get("language") or ""
    if task26_aloha_arm_only:
        aloha_state = np.empty(14, dtype=np.float32)
        aloha_state[:6] = state[TASK26_ALOHA_LEFT_ARM_INDICES]
        aloha_state[6] = 1.0  # Virtual ALOHA gripper is open; its output is discarded.
        aloha_state[7:13] = state[TASK26_ALOHA_RIGHT_ARM_INDICES]
        aloha_state[13] = 1.0
        return {
            "images": {
                "cam_high": np.moveaxis(camera("cam_overhead"), -1, 0),
                "cam_left_wrist": np.moveaxis(camera("cam_wrist_left"), -1, 0),
                "cam_right_wrist": np.moveaxis(camera("cam_wrist_right"), -1, 0),
            },
            "state": aloha_state,
            "prompt": prompt,
        }
    return {
        "images": {
            "cam_overhead": camera("cam_overhead"),
            "cam_wrist_left": camera("cam_wrist_left"),
            "cam_wrist_right": camera("cam_wrist_right"),
        },
        "state": state,
        "prompt": prompt,
    }


def _expand_task26_aloha_arm_actions(actions: np.ndarray, full_state: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected ALOHA policy actions shaped [T, 14], got {actions.shape}")
    full_actions = np.broadcast_to(full_state, (actions.shape[0], 58)).copy()
    full_actions[:, TASK26_ALOHA_FULL_INDICES] = actions[:, TASK26_ALOHA_ACTION_INDICES]
    return full_actions


def _warm_up_policy(
    policy: Any,
    *,
    runs: int,
    image_height: int,
    image_width: int,
    prompt: str,
    task26_aloha_arm_only: bool = False,
) -> None:
    if runs <= 0:
        logging.info("Policy warm-up disabled")
        return

    if task26_aloha_arm_only:
        observation = {
            "images": {
                camera_id: np.zeros((3, image_height, image_width), dtype=np.uint8)
                for camera_id in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            },
            "state": np.asarray([0.0] * 6 + [1.0] + [0.0] * 6 + [1.0], dtype=np.float32),
            "prompt": prompt,
        }
        expected_action_dim = 14
    else:
        observation = {
            "images": {
                camera_id: np.zeros((image_height, image_width, 3), dtype=np.uint8)
                for camera_id in ("cam_overhead", "cam_wrist_left", "cam_wrist_right")
            },
            "state": np.zeros(58, dtype=np.float32),
            "prompt": prompt,
        }
        expected_action_dim = 58
    had_rng = hasattr(policy, "_rng")
    original_rng = getattr(policy, "_rng", None)
    try:
        for index in range(runs):
            started = time.perf_counter()
            result = policy.infer(observation)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != expected_action_dim:
                raise ValueError(
                    f"Warm-up returned actions shaped {actions.shape}, expected [T, {expected_action_dim}]"
                )
            logging.info(
                "Policy warm-up %d/%d completed in %.3fs with actions %s",
                index + 1,
                runs,
                time.perf_counter() - started,
                actions.shape,
            )
    finally:
        if had_rng:
            policy._rng = original_rng  # noqa: SLF001


class Bench2DexPolicyServer:
    def __init__(
        self,
        policy: Any,
        host: str,
        port: int,
        action_horizon: int,
        metadata: dict[str, Any],
        *,
        task26_aloha_arm_only: bool = False,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._action_horizon = action_horizon
        self._metadata = metadata
        self._task26_aloha_arm_only = task26_aloha_arm_only
        self._running = True
        self._inference_lock = threading.Lock()
        self._server: socket.socket | None = None

    def stop(self, *_args: Any) -> None:
        self._running = False
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()

    def _dispatch(self, request: dict[str, Any]) -> Any:
        command = request.get("command")
        if command == "describe":
            return self._metadata
        if command == "reset":
            seed = request.get("seed")
            if seed is not None and hasattr(self._policy, "_rng"):
                self._policy._rng = jax.random.key(int(seed))  # noqa: SLF001
            return None
        if command == "update_after_action":
            return None
        if command == "get_action_chunk":
            raw_observation = request.get("observation", {})
            observation = _extract_policy_observation(
                raw_observation,
                request.get("instruction"),
                task26_aloha_arm_only=self._task26_aloha_arm_only,
            )
            with self._inference_lock:
                actions = np.asarray(self._policy.infer(observation)["actions"], dtype=np.float32)
            if self._task26_aloha_arm_only:
                actions = _expand_task26_aloha_arm_actions(actions, _extract_full_state(raw_observation))
            if actions.ndim != 2 or actions.shape[1] != 58:
                raise ValueError(f"Expected policy actions shaped [T, 58], got {actions.shape}")
            return actions[: self._action_horizon]
        if command == "close":
            return "close"
        raise ValueError(f"Unsupported Bench2Dex RPC command: {command!r}")

    def _handle_client(self, client: socket.socket) -> None:
        with client:
            while self._running:
                request = _recv_message(client)
                if request is None:
                    return
                try:
                    result = self._dispatch(request)
                    close_requested = request.get("command") == "close"
                    _send_message(client, {"result": None if close_requested else result})
                    if close_requested:
                        return
                except Exception as exc:  # Keep the server alive for later eval chunks.
                    traceback.print_exc()
                    _send_message(client, {"error": f"{type(exc).__name__}: {exc}"})
                    return

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self._server = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self._host, self._port))
            server.listen(4)
            server.settimeout(1.0)
            logging.info("Bench2Dex policy server listening on %s:%d", self._host, self._port)
            while self._running:
                try:
                    client, address = server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                logging.info("Bench2Dex evaluator connected from %s:%d", *address)
                self._handle_client(client)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_bench2dex_task26_action_expert")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-image-height", type=int, default=480)
    parser.add_argument("--warmup-image-width", type=int, default=640)
    parser.add_argument("--warmup-prompt", default=DEFAULT_WARMUP_PROMPT)
    parser.add_argument(
        "--hand-dit-checkpoint",
        help="Enable dual-expert inference using this trained hand-only checkpoint.",
    )
    parser.add_argument(
        "--dinov3-repo",
        default="/mnt/ssd1/szhao/DexTool/dinov3",
        help="Official facebookresearch/dinov3 checkout used by the frozen wrist encoder.",
    )
    parser.add_argument(
        "--dinov3-checkpoint",
        default="/mnt/ssd1/szhao/LeveFD/dinov3_vitb16.pth",
    )
    parser.add_argument("--hand-device", default="cuda:0")
    parser.add_argument("--hand-flow-steps", type=int, default=10)
    parser.add_argument(
        "--task26-aloha-arm-only",
        action="store_true",
        help=(
            "Use the pi0.5 ALOHA embodiment for a Task26 zero-shot probe: control "
            "six joints per KUKA arm, discard gripper outputs, and hold all remaining joints."
        ),
    )
    args = parser.parse_args()

    config = training_config.get_config(args.config)
    action_representation = getattr(config.data, "action_representation", "absolute")
    task26_arm_vla = bool(getattr(config.data, "arm_only", False))
    metadata = {
        "config": config.name,
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "action_representation": action_representation,
        "action_horizon": args.action_horizon,
        "model_action_dim": config.model.action_dim,
        "action_dim": 14 if task26_arm_vla else config.model.action_dim,
    }
    if args.task26_aloha_arm_only:
        if config.name != "pi05_aloha":
            raise ValueError("--task26-aloha-arm-only requires --config pi05_aloha")
        metadata["action_dim"] = 58
        metadata["embodiment_mapping"] = "task26_aloha_arm_only"
    if args.hand_dit_checkpoint:
        if args.task26_aloha_arm_only:
            raise ValueError("The hand DiT cannot be combined with the ALOHA zero-shot mapping")
        if not task26_arm_vla:
            raise ValueError(
                "The hand DiT requires --config pi05_bench2dex_task26_dual_expert_arm "
                "so the VLA supplies physical 16D arm/activity actions"
            )
        if action_representation != "absolute":
            raise ValueError("The first hand DiT runtime supports only the absolute-action Task26 pipeline")
    elif task26_arm_vla:
        raise ValueError("The Task26 16D arm/activity VLA requires --hand-dit-checkpoint to compose a 58D action")

    policy = policy_config.create_trained_policy(config, args.checkpoint)
    if args.hand_dit_checkpoint:
        device = torch.device(args.hand_device)
        hand_model, normalization, hand_checkpoint = load_hand_dit_checkpoint(args.hand_dit_checkpoint, device=device)
        adapter = hand_checkpoint.get("adapter_model")
        checkpoint_metadata = hand_checkpoint.get("metadata", {})
        wrist_encoder = (
            Dinov3WristEncoder(
                args.dinov3_repo,
                args.dinov3_checkpoint,
            )
            if hand_model.config.condition_on_vision
            else None
        )
        policy = DualExpertHandPolicy(
            policy,
            hand_model,
            wrist_encoder,
            normalization,
            num_hand_steps=args.hand_flow_steps,
            device=device,
            adapter=adapter,
            vla_action_q01=checkpoint_metadata.get("vla_action_q01"),
            vla_action_q99=checkpoint_metadata.get("vla_action_q99"),
        )
        metadata.update(policy.metadata)
        metadata["action_dim"] = 58
        metadata["hand_checkpoint"] = str(pathlib.Path(args.hand_dit_checkpoint).resolve())
        metadata["hand_checkpoint_step"] = int(hand_checkpoint["step"])
    logging.info("Policy metadata: %s", metadata)
    _warm_up_policy(
        policy,
        runs=args.warmup_runs,
        image_height=args.warmup_image_height,
        image_width=args.warmup_image_width,
        prompt=args.warmup_prompt,
        task26_aloha_arm_only=args.task26_aloha_arm_only,
    )
    server = Bench2DexPolicyServer(
        policy,
        args.host,
        args.port,
        args.action_horizon,
        metadata,
        task26_aloha_arm_only=args.task26_aloha_arm_only,
    )
    signal.signal(signal.SIGINT, server.stop)
    signal.signal(signal.SIGTERM, server.stop)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
