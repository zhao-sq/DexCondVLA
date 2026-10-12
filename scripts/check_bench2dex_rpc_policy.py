"""Validate that a running Bench2Dex RPC server matches an evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import pathlib
import socket


def _send_message(sock: socket.socket, payload: object) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    sock.sendall(len(encoded).to_bytes(4, "big"))
    sock.sendall(encoded)


def _recv_message(sock: socket.socket) -> object:
    length_bytes = sock.recv(4)
    if len(length_bytes) != 4:
        raise ConnectionError("Incomplete Bench2Dex RPC response header")
    remaining = int.from_bytes(length_bytes, "big")
    chunks: list[bytes] = []
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("Incomplete Bench2Dex RPC response payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return json.loads(b"".join(chunks).decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--action-representation",
        required=True,
        choices=("absolute", "state_delta", "aligned_delta"),
    )
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=58)
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port), timeout=30.0) as sock:
        _send_message(sock, {"command": "describe"})
        response = _recv_message(sock)
    if not isinstance(response, dict) or not isinstance(response.get("result"), dict):
        raise RuntimeError(f"Unexpected policy-server describe response: {response!r}")

    actual = response["result"]
    expected = {
        "config": args.config,
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "action_representation": args.action_representation,
        "action_horizon": args.action_horizon,
        "action_dim": args.action_dim,
    }
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Policy server does not match evaluation pipeline: {mismatches}")
    print(f"[eval] Verified policy server: {actual}", flush=True)


if __name__ == "__main__":
    main()
