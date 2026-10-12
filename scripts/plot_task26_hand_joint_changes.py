"""Plot changes in target hand-joint actions for random Task26 episodes."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _episode_index(path: Path) -> int:
    with h5py.File(path, "r") as episode:
        return int(episode["meta/episode_index"][()])


def _choose_unique_trajectories(files: list[Path], count: int, seed: int) -> list[Path]:
    """Choose at most one visual variant for each underlying robot trajectory."""
    by_episode_index: dict[int, list[Path]] = {}
    for path in files:
        by_episode_index.setdefault(_episode_index(path), []).append(path)

    rng = random.Random(seed)
    episode_indices = sorted(by_episode_index)
    if count > len(episode_indices):
        raise ValueError(f"Requested {count} trajectories, but only {len(episode_indices)} are unique")

    chosen_indices = rng.sample(episode_indices, count)
    return [rng.choice(sorted(by_episode_index[index])) for index in chosen_indices]


def _analyze_episode(
    path: Path, delta_atol: float, skip_valid_steps: int
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], int]:
    with h5py.File(path, "r") as episode:
        target_actions = np.asarray(episode["action/commanded"], dtype=np.float64)
        joint_names = [_decode(value) for value in episode["action/action_names"][()]]
        episode_index = int(episode["meta/episode_index"][()])
        valid = np.asarray(episode["frame_valid"], dtype=bool)
        valid &= np.asarray(episode["action/action_valid"], dtype=bool)
        valid &= np.isfinite(target_actions).all(axis=1)

        homing_start = int(episode["meta/homing_start_sim_step"][()])
        if homing_start >= 0:
            sim_steps = np.asarray(episode["time/sim_step"])
            homing_cutoff = int(np.searchsorted(sim_steps, homing_start, side="left"))
            valid[homing_cutoff:] = False

    hand_indices = [
        index
        for index, name in enumerate(joint_names)
        if name.startswith("left_") or name.startswith("right_") or name.startswith("multi_right_")
    ]
    hand_index_set = set(hand_indices)
    arm_joint_names = [name for index, name in enumerate(joint_names) if index not in hand_index_set]
    if len(hand_indices) != 44:
        raise ValueError(f"Expected 44 hand joints in {path.name}, found {len(hand_indices)}")
    if len(arm_joint_names) != 14:
        raise ValueError(f"Expected 14 excluded arm joints in {path.name}, found {len(arm_joint_names)}")
    source_frame_indices = np.flatnonzero(valid)
    if skip_valid_steps >= source_frame_indices.size:
        raise ValueError(
            f"Cannot skip {skip_valid_steps} valid target-action steps in {path.name}; "
            f"only {source_frame_indices.size} are available"
        )
    source_frame_indices = source_frame_indices[skip_valid_steps:]
    target_hand_actions = target_actions[source_frame_indices][:, hand_indices]
    if target_hand_actions.shape[0] < 2:
        raise ValueError(f"Fewer than two valid target-action frames in {path.name}")

    changed_count = np.zeros(target_hand_actions.shape[0], dtype=np.int64)
    changed_count[1:] = (np.abs(np.diff(target_hand_actions, axis=0)) > delta_atol).sum(axis=1)
    return (
        changed_count,
        source_frame_indices,
        [joint_names[index] for index in hand_indices],
        arm_joint_names,
        episode_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument(
        "--skip-valid-steps",
        type=int,
        default=0,
        help="Drop this many valid target-action steps from the start of every trajectory.",
    )
    parser.add_argument(
        "--delta-atol",
        type=float,
        default=1e-6,
        help="A joint changes when abs(target_action[t] - target_action[t-1]) exceeds this tolerance.",
    )
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be positive")
    if args.delta_atol < 0:
        parser.error("--delta-atol must be non-negative")
    if args.skip_valid_steps < 0:
        parser.error("--skip-valid-steps must be non-negative")

    files = sorted(args.dataset_dir.glob("episode_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 files under {args.dataset_dir}")
    chosen = _choose_unique_trajectories(files, args.count, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "dataset_dir": str(args.dataset_dir),
        "seed": args.seed,
        "delta_atol": args.delta_atol,
        "skipped_initial_valid_target_action_steps": args.skip_valid_steps,
        "source": "action/commanded (target action), not robot/qpos (actual state)",
        "definition": "abs(target_action[t] - target_action[t-1]) > delta_atol; first valid action is 0",
        "validity_filter": "frame_valid & action_valid & all-finite, before homing_start_sim_step",
        "scope": "44 target hand joints only; all 14 target arm joints are excluded",
        "samples": [],
    }

    for sample_number, path in enumerate(chosen, start=1):
        changed_count, source_frames, hand_joint_names, arm_joint_names, episode_index = _analyze_episode(
            path, args.delta_atol, args.skip_valid_steps
        )
        stem = f"sample_{sample_number:02d}_{path.stem}_target_hand_joint_changes"

        csv_path = args.output_dir / f"{stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["target_action_step", "source_frame_index", "changed_target_hand_joint_count"])
            writer.writerows(zip(range(changed_count.size), source_frames.tolist(), changed_count.tolist(), strict=True))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(changed_count.size), changed_count, linewidth=1.1, color="#2563eb")
        ax.set(
            title=(
                f"Task26 {path.name} — target hand actions only (arm excluded)\n"
                f"episode_index={episode_index}, threshold={args.delta_atol:g} rad, "
                f"initial valid steps skipped={args.skip_valid_steps}"
            ),
            xlabel="Valid target-action step",
            ylabel="Changed target hand joints",
            xlim=(0, max(1, changed_count.size - 1)),
            ylim=(-0.5, len(hand_joint_names) + 0.5),
        )
        ax.set_yticks(np.arange(0, len(hand_joint_names) + 1, 4))
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        png_path = args.output_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

        manifest["samples"].append(
            {
                "sample_number": sample_number,
                "file": path.name,
                "episode_index": episode_index,
                "valid_target_action_count": int(changed_count.size),
                "target_hand_joint_count": len(hand_joint_names),
                "mean_changed_target_hand_joint_count": float(changed_count.mean()),
                "min_changed_target_hand_joint_count": int(changed_count.min()),
                "max_changed_target_hand_joint_count": int(changed_count.max()),
                "plot": png_path.name,
                "csv": csv_path.name,
            }
        )

    manifest["hand_joint_names"] = hand_joint_names
    manifest["excluded_arm_joint_names"] = arm_joint_names
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.count} plots, {args.count} CSV files, and {manifest_path}")


if __name__ == "__main__":
    main()
