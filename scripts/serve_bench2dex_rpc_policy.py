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
import signal
import socket
import threading
import time
import traceback
from typing import Any

import jax
import numpy as np

from openpi.policies import policy_config
from openpi.training import config as training_config

DEFAULT_WARMUP_PROMPT = (
    "Use left hand to place the master chef can and MSG onto the tray, use right hand to place "
    "the milk box and potted meat can onto the tray."
)


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


def _extract_policy_observation(raw: dict[str, Any], instruction: str | None) -> dict[str, Any]:
    observation = raw.get("observation", raw)

    def camera(name: str) -> np.ndarray:
        try:
            image = np.asarray(observation[name]["rgb"])
        except KeyError as exc:
            raise KeyError(f"Bench2Dex observation is missing {name}.rgb") from exc
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected uint8 HWC RGB for {name}, got {image.shape} {image.dtype}")
        return image

    joint_action = raw.get("joint_action", {})
    state_value = joint_action.get("vector", joint_action.get("qpos"))
    if state_value is None:
        raise KeyError("Bench2Dex observation is missing joint_action.vector/qpos")
    state = np.asarray(state_value, dtype=np.float32).reshape(-1)
    if state.shape != (58,):
        raise ValueError(f"Expected a 58D Task26 state, got {state.shape}")

    prompt = instruction or raw.get("language") or ""
    return {
        "images": {
            "cam_overhead": camera("cam_overhead"),
            "cam_wrist_left": camera("cam_wrist_left"),
            "cam_wrist_right": camera("cam_wrist_right"),
        },
        "state": state,
        "prompt": prompt,
    }


def _warm_up_policy(
    policy: Any,
    *,
    runs: int,
    image_height: int,
    image_width: int,
    prompt: str,
) -> None:
    if runs <= 0:
        logging.info("Policy warm-up disabled")
        return

    observation = {
        "images": {
            camera_id: np.zeros((image_height, image_width, 3), dtype=np.uint8)
            for camera_id in ("cam_overhead", "cam_wrist_left", "cam_wrist_right")
        },
        "state": np.zeros(58, dtype=np.float32),
        "prompt": prompt,
    }
    had_rng = hasattr(policy, "_rng")
    original_rng = getattr(policy, "_rng", None)
    try:
        for index in range(runs):
            started = time.perf_counter()
            result = policy.infer(observation)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 58:
                raise ValueError(f"Warm-up returned actions shaped {actions.shape}, expected [T, 58]")
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
    def __init__(self, policy: Any, host: str, port: int, action_horizon: int) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._action_horizon = action_horizon
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
        if command == "reset":
            seed = request.get("seed")
            if seed is not None and hasattr(self._policy, "_rng"):
                self._policy._rng = jax.random.key(int(seed))  # noqa: SLF001
            return None
        if command == "update_after_action":
            return None
        if command == "get_action_chunk":
            observation = _extract_policy_observation(request.get("observation", {}), request.get("instruction"))
            with self._inference_lock:
                actions = np.asarray(self._policy.infer(observation)["actions"], dtype=np.float32)
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
    args = parser.parse_args()

    config = training_config.get_config(args.config)
    policy = policy_config.create_trained_policy(config, args.checkpoint)
    _warm_up_policy(
        policy,
        runs=args.warmup_runs,
        image_height=args.warmup_image_height,
        image_width=args.warmup_image_width,
        prompt=args.warmup_prompt,
    )
    server = Bench2DexPolicyServer(policy, args.host, args.port, args.action_horizon)
    signal.signal(signal.SIGINT, server.stop)
    signal.signal(signal.SIGTERM, server.stop)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
