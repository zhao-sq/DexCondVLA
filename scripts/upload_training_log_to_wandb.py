"""Backfill and follow OpenPI text metrics in a W&B run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

import wandb

METRIC_PATTERN = re.compile(
    r"Step\s+(?P<step>\d+):\s+"
    r"grad_norm=(?P<grad_norm>[-+0-9.eE]+),\s+"
    r"loss=(?P<loss>[-+0-9.eE]+),\s+"
    r"param_norm=(?P<param_norm>[-+0-9.eE]+)"
)


def _parse_metric(line: str) -> tuple[int, dict[str, float]] | None:
    match = METRIC_PATTERN.search(line)
    if match is None:
        return None
    values = match.groupdict()
    return int(values.pop("step")), {name: float(value) for name, value in values.items()}


def _process_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--wandb-dir", type=Path, required=True)
    parser.add_argument("--project", default="openpi")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    args.wandb_dir.mkdir(parents=True, exist_ok=True)
    run_id_path = args.wandb_dir / f"{args.run_name}.run_id"
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
    else:
        run_id = wandb.util.generate_id()
        run_id_path.write_text(run_id)

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        id=run_id,
        resume="allow",
        dir=str(args.wandb_dir),
        job_type="metrics-sidecar",
        config={
            "source_log": str(args.log_path),
            "source_training_pid": args.training_pid,
            "metric_log_interval": 100,
        },
    )
    print(json.dumps({"wandb_url": run.url, "run_id": run.id}), flush=True)

    last_step = -1
    with args.log_path.open(errors="replace") as log_file:
        while True:
            line = log_file.readline()
            if line:
                parsed = _parse_metric(line)
                if parsed is not None:
                    step, metrics = parsed
                    if step > last_step:
                        wandb.log(metrics, step=step)
                        last_step = step
                continue

            if not _process_alive(args.training_pid):
                break
            time.sleep(args.poll_seconds)

    wandb.finish()


if __name__ == "__main__":
    main()
