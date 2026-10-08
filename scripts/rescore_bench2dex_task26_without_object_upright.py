"""Offline-rescore Task26 recordings without per-object upright checks."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

import h5py

CHANNELS = ("none", "cov", "inv", "inv_cov")


def _max_consecutive_true(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _first_stable_frame(values: list[bool], required_frames: int) -> int | None:
    current = 0
    for index, value in enumerate(values):
        current = current + 1 if value else 0
        if current >= required_frames:
            return index
    return None


def _read_channel_rows(eval_root: Path, channel: str, expected: int) -> list[dict[str, Any]]:
    path = eval_root / channel / "per_episode.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} {channel} results, found {len(rows)} in {path}")
    return rows


def _find_record(
    eval_root: Path,
    channel: str,
    episode: int,
    *,
    official_success: bool,
) -> Path:
    outcome = "success" if official_success else "failure"
    matches = list((eval_root / channel).glob(f"chunks/*/records/{outcome}/episode_{episode:06d}*.hdf5"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one recording for {channel} episode {episode}, found {len(matches)}: {matches}")
    return matches[0]


def _load_states(file: h5py.File, object_ids: list[str], frame: int) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for object_id in object_ids:
        group = file[f"objects/{object_id}"]
        states[object_id] = {
            "pose_world": group["pose_world"][frame],
            "lin_vel_world": group["lin_vel_world"][frame],
            "ang_vel_world": group["ang_vel_world"][frame],
        }
    return states


def _score_record(
    path: Path,
    *,
    conditions: list[dict[str, Any]],
    object_ids: list[str],
    evaluate_conditions: Any,
    fps: float,
    dwell_time_s: float,
) -> dict[str, Any]:
    required_frames = max(1, math.ceil(fps * dwell_time_s))
    with h5py.File(path, "r") as file:
        frame_count = int(file["meta/frame_count"][()])
        raw_condition = [
            bool(evaluate_conditions(conditions, _load_states(file, object_ids, frame), {}))
            for frame in range(frame_count)
        ]

    first_stable_frame = _first_stable_frame(raw_condition, required_frames)
    return {
        "success_without_object_upright": first_stable_frame is not None,
        "first_stable_frame": first_stable_frame,
        "first_stable_time_s": None if first_stable_frame is None else first_stable_frame / fps,
        "raw_condition_frames": sum(raw_condition),
        "max_consecutive_condition_frames": _max_consecutive_true(raw_condition),
        "required_consecutive_frames": required_frames,
        "record_fps": fps,
        "dwell_time_s": dwell_time_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bench2dex-root", type=Path, required=True)
    parser.add_argument("--expected-per-channel", type=int, default=25)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--dwell-time-s", type=float, default=0.5)
    args = parser.parse_args()

    sys.path.insert(0, str(args.bench2dex_root))
    from success.condition_evaluator import evaluate_conditions
    from success.custom import task_26_canned_food_tray_line_arrangement as task26

    object_ids = list(task26._ALL_OBJ_IDS)  # noqa: SLF001
    state_object_ids = [*object_ids, "obj_130_tray_1"]
    conditions: list[dict[str, Any]] = []
    for object_id in object_ids:
        conditions.append(copy.deepcopy(task26._OBJECT_ON[object_id]))  # noqa: SLF001
        conditions.extend(copy.deepcopy(task26._static_for(object_id)))  # noqa: SLF001
    conditions.extend(copy.deepcopy(task26._TRAY_UPRIGHT))  # noqa: SLF001

    results: list[dict[str, Any]] = []
    for channel in CHANNELS:
        for official in _read_channel_rows(args.eval_root, channel, args.expected_per_channel):
            episode = int(official["episode_index"])
            source = _find_record(
                args.eval_root,
                channel,
                episode,
                official_success=bool(official["success"]),
            )
            rescored = _score_record(
                source,
                conditions=conditions,
                object_ids=state_object_ids,
                evaluate_conditions=evaluate_conditions,
                fps=args.fps,
                dwell_time_s=args.dwell_time_s,
            )
            results.append(
                {
                    "channel": channel,
                    "episode_index": episode,
                    "official_success": bool(official["success"]),
                    **rescored,
                    "source_hdf5": str(source),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_episode = args.output_dir / "per_episode.jsonl"
    per_episode.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "metric": "success_without_object_upright",
        "definition": (
            "All four objects are on the tray and static, the tray is upright, and these "
            "conditions hold for the dwell window; per-object upright checks are omitted."
        ),
        "sampling_note": (
            "Offline scoring uses recorded 20 Hz frames rather than the evaluator's 60 Hz physics steps."
        ),
        "episodes": len(results),
        "successes": sum(row["success_without_object_upright"] for row in results),
        "by_channel": {},
    }
    summary["success_rate"] = summary["successes"] / summary["episodes"]
    for channel in CHANNELS:
        rows = [row for row in results if row["channel"] == channel]
        successes = sum(row["success_without_object_upright"] for row in rows)
        summary["by_channel"][channel] = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
