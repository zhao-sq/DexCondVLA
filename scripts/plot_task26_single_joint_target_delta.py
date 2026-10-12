"""Plot one Task26 hand joint's per-step target-action delta."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-path", type=Path, required=True)
    parser.add_argument("--joint", default="multi_right_middle_DIP")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.episode_path, "r") as episode:
        names = [_decode(value) for value in episode["action/action_names"][()]]
        if args.joint not in names:
            raise ValueError(f"Unknown joint {args.joint!r}")
        joint_index = names.index(args.joint)
        actions = np.asarray(episode["action/commanded"], dtype=np.float64)
        valid = np.asarray(episode["frame_valid"], dtype=bool)
        valid &= np.asarray(episode["action/action_valid"], dtype=bool)
        valid &= np.isfinite(actions).all(axis=1)
        homing_start = int(episode["meta/homing_start_sim_step"][()])
        if homing_start >= 0:
            sim_steps = np.asarray(episode["time/sim_step"])
            cutoff = int(np.searchsorted(sim_steps, homing_start, side="left"))
            valid[cutoff:] = False

    source_frames = np.flatnonzero(valid)
    targets = actions[source_frames, joint_index]
    deltas = np.zeros_like(targets)
    deltas[1:] = np.diff(targets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.episode_path.stem}_{args.joint}_delta_target_action"
    csv_path = args.output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target_action_step", "source_frame_index", "target_action_rad", "delta_target_action_rad"])
        writer.writerows(zip(range(targets.size), source_frames, targets, deltas, strict=True))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(deltas.size), deltas, linewidth=1.0, color="#2563eb")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.65)
    ax.set(
        title=f"{args.episode_path.name} — {args.joint}\nDelta target action (current minus previous)",
        xlabel="Valid target-action step",
        ylabel="Delta target action (rad)",
        xlim=(0, max(1, deltas.size - 1)),
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    png_path = args.output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"episode={args.episode_path.name}")
    print(f"joint={args.joint}")
    print(f"valid_target_action_steps={targets.size}")
    print(f"nonzero_deltas={np.count_nonzero(deltas[1:])}/{max(0, deltas.size - 1)}")
    print(f"delta_min_rad={deltas.min():.9f}")
    print(f"delta_max_rad={deltas.max():.9f}")
    print(f"delta_mean_abs_rad={np.abs(deltas).mean():.9f}")
    print(f"csv={csv_path}")
    print(f"plot={png_path}")


if __name__ == "__main__":
    main()
