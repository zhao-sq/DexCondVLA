"""Add per-frame gripper-active labels to one Task26 HDF5 trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from render_task26_gripper_activity_video import GripperInterval
from render_task26_gripper_activity_video import infer_gripper_intervals

DEFAULT_LEFT_DATASET_KEY = "labels/left_gripper_active"
DEFAULT_RIGHT_DATASET_KEY = "labels/right_gripper_active"
DEFAULT_LEGACY_DATASET_KEY = "labels/gripper_active"
METHOD_VERSION = "target_action_contact_projection_v2_side_specific"


def _make_labels(frame_count: int, intervals: list[GripperInterval], *, side: str | None = None) -> np.ndarray:
    labels = np.zeros(frame_count, dtype=np.bool_)
    for interval in intervals:
        if side is not None and interval.side != side:
            continue
        if not 0 <= interval.close_start <= interval.open_complete < frame_count:
            raise ValueError(f"Invalid inferred interval: {interval}")
        labels[interval.close_start : interval.open_complete + 1] = True
    return labels


def _write_label(
    episode: h5py.File,
    dataset_key: str,
    labels: np.ndarray,
    *,
    intervals: list[GripperInterval],
    side: str | None,
    overwrite: bool,
) -> str:
    group_name, dataset_name = dataset_key.rsplit("/", 1)
    group = episode.require_group(group_name)
    if dataset_name in group:
        existing = np.asarray(group[dataset_name], dtype=np.bool_)
        if existing.shape == labels.shape and np.array_equal(existing, labels):
            dataset = group[dataset_name]
            status = "unchanged"
        elif not overwrite:
            raise FileExistsError(f"/{dataset_key} already exists with different values; pass --overwrite")
        else:
            del group[dataset_name]
            dataset = group.create_dataset(dataset_name, data=labels, dtype=np.bool_)
            status = "overwritten"
    else:
        dataset = group.create_dataset(dataset_name, data=labels, dtype=np.bool_)
        status = "created"

    selected = intervals if side is None else [interval for interval in intervals if interval.side == side]
    subject = "either hand" if side is None else f"the {side.lower()} hand"
    semantics = "union across hands" if side is None else f"{side.lower()} hand only"
    dataset.attrs["description"] = (
        f"True from inferred hand-closing start through hand-opening completion, inclusive, for {subject}"
    )
    dataset.attrs["method_version"] = METHOD_VERSION
    dataset.attrs["side"] = "EITHER" if side is None else side
    dataset.attrs["source_target_action_key"] = "/action/commanded"
    dataset.attrs["source_contact_group"] = "/robot/tactile/contact_mask"
    dataset.attrs["interval_semantics"] = f"inclusive [close_start, open_complete], {semantics}"
    dataset.attrs["intervals_json"] = json.dumps(_interval_records(selected), separators=(",", ":"))
    return status


def _interval_records(intervals: list[GripperInterval]) -> list[dict[str, int | str]]:
    return [
        {
            "side": interval.side,
            "close_start": interval.close_start,
            "open_start": interval.open_start,
            "open_complete": interval.open_complete,
        }
        for interval in intervals
    ]


def _intervals_from_existing_label(episode: h5py.File, dataset_key: str) -> tuple[list[GripperInterval], int]:
    if dataset_key not in episode:
        raise KeyError(f"Missing existing compatibility label /{dataset_key}")
    dataset = episode[dataset_key]
    raw_intervals = dataset.attrs.get("intervals_json")
    if raw_intervals is None:
        raise KeyError(f"/{dataset_key} has no intervals_json metadata")
    records = json.loads(raw_intervals)
    intervals = [
        GripperInterval(
            side=str(record["side"]),
            close_start=int(record["close_start"]),
            open_start=int(record["open_start"]),
            open_complete=int(record["open_complete"]),
        )
        for record in records
    ]
    frame_count = int(dataset.shape[0])
    combined = _make_labels(frame_count, intervals)
    if not np.array_equal(combined, np.asarray(dataset, dtype=np.bool_)):
        raise ValueError(f"/{dataset_key} does not equal the union of its stored intervals")
    return intervals, frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_path", type=Path, help="One Task26 episode_*.hdf5 trajectory")
    parser.add_argument("--left-dataset-key", default=DEFAULT_LEFT_DATASET_KEY)
    parser.add_argument("--right-dataset-key", default=DEFAULT_RIGHT_DATASET_KEY)
    parser.add_argument(
        "--legacy-dataset-key",
        default=DEFAULT_LEGACY_DATASET_KEY,
        help="Also maintain the old OR-combined compatibility label; pass an empty string to disable",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing label if its values differ")
    parser.add_argument(
        "--from-existing-label",
        action="store_true",
        help="Split the existing OR label using its stored side-specific intervals instead of re-running inference",
    )
    parser.add_argument("--dry-run", action="store_true", help="Infer and validate labels without modifying the file")
    parser.add_argument("--quiet", action="store_true", help="Suppress the successful annotation summary")
    args = parser.parse_args()

    dataset_keys = {
        "LEFT": args.left_dataset_key.strip("/"),
        "RIGHT": args.right_dataset_key.strip("/"),
    }
    legacy_dataset_key = args.legacy_dataset_key.strip("/")
    for option, dataset_key in dataset_keys.items():
        if not dataset_key or "/" not in dataset_key:
            parser.error(f"--{option.lower()}-dataset-key must include a group and dataset name")
    if legacy_dataset_key and "/" not in legacy_dataset_key:
        parser.error("--legacy-dataset-key must include a group and dataset name")

    mode = "r" if args.dry_run else "r+"
    with h5py.File(args.episode_path, mode) as episode:
        frame_count = int(episode["action/commanded"].shape[0])
        if args.from_existing_label:
            intervals, legacy_frame_count = _intervals_from_existing_label(episode, legacy_dataset_key)
            if legacy_frame_count != frame_count:
                raise ValueError(
                    f"Existing label length {legacy_frame_count} does not match action frame count {frame_count}"
                )
            homing_start = int(episode["meta/homing_start_sim_step"][()])
            homing_cutoff = (
                frame_count
                if homing_start < 0
                else int(np.searchsorted(episode["time/sim_step"][:], homing_start, side="left"))
            )
        else:
            intervals, homing_cutoff = infer_gripper_intervals(episode)
        labels_by_side = {side: _make_labels(frame_count, intervals, side=side) for side in ("LEFT", "RIGHT")}
        combined_labels = np.logical_or(labels_by_side["LEFT"], labels_by_side["RIGHT"])
        if any(np.any(labels[homing_cutoff:]) for labels in labels_by_side.values()):
            raise ValueError("Inferred side-specific active labels extend into the homing tail")
        if len(intervals) != 4:
            raise ValueError(f"Expected four gripper-active intervals, found {len(intervals)}")
        for side in ("LEFT", "RIGHT"):
            count = sum(interval.side == side for interval in intervals)
            if count != 2:
                raise ValueError(f"Expected two {side} gripper-active intervals, found {count}")

        statuses = dict.fromkeys(dataset_keys.values(), "dry-run")
        if legacy_dataset_key:
            statuses[legacy_dataset_key] = "dry-run"
        if not args.dry_run:
            for side, dataset_key in dataset_keys.items():
                statuses[dataset_key] = _write_label(
                    episode,
                    dataset_key,
                    labels_by_side[side],
                    intervals=intervals,
                    side=side,
                    overwrite=args.overwrite,
                )
            if legacy_dataset_key:
                statuses[legacy_dataset_key] = _write_label(
                    episode,
                    legacy_dataset_key,
                    combined_labels,
                    intervals=intervals,
                    side=None,
                    overwrite=args.overwrite,
                )
            episode.flush()

    if not args.quiet:
        print(f"episode={args.episode_path}")
        print(f"frame_count={frame_count}")
        print(f"homing_cutoff={homing_cutoff}")
        for side, dataset_key in dataset_keys.items():
            print(
                f"dataset=/{dataset_key},status={statuses[dataset_key]},"
                f"true_frame_count={int(labels_by_side[side].sum())}"
            )
        if legacy_dataset_key:
            print(
                f"dataset=/{legacy_dataset_key},status={statuses[legacy_dataset_key]},"
                f"true_frame_count={int(combined_labels.sum())}"
            )
        for record in _interval_records(intervals):
            print(f"interval={record['side']},{record['close_start']},{record['open_start']},{record['open_complete']}")


if __name__ == "__main__":
    main()
