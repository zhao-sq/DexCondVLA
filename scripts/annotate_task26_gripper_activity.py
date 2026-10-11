"""Add per-frame gripper-active labels to one Task26 HDF5 trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from render_task26_gripper_activity_video import GripperInterval
from render_task26_gripper_activity_video import infer_gripper_intervals

DEFAULT_DATASET_KEY = "labels/gripper_active"
METHOD_VERSION = "target_action_contact_projection_v1"


def _make_labels(frame_count: int, intervals: list[GripperInterval]) -> np.ndarray:
    labels = np.zeros(frame_count, dtype=np.bool_)
    for interval in intervals:
        if not 0 <= interval.close_start <= interval.open_complete < frame_count:
            raise ValueError(f"Invalid inferred interval: {interval}")
        labels[interval.close_start : interval.open_complete + 1] = True
    return labels


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_path", type=Path, help="One Task26 episode_*.hdf5 trajectory")
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing label if its values differ")
    parser.add_argument("--dry-run", action="store_true", help="Infer and validate labels without modifying the file")
    parser.add_argument("--quiet", action="store_true", help="Suppress the successful annotation summary")
    args = parser.parse_args()

    dataset_key = args.dataset_key.strip("/")
    if not dataset_key or "/" not in dataset_key:
        parser.error("--dataset-key must include a group and dataset name, e.g. labels/gripper_active")

    mode = "r" if args.dry_run else "r+"
    with h5py.File(args.episode_path, mode) as episode:
        frame_count = int(episode["action/commanded"].shape[0])
        intervals, homing_cutoff = infer_gripper_intervals(episode)
        labels = _make_labels(frame_count, intervals)
        if np.any(labels[homing_cutoff:]):
            raise ValueError("Inferred active labels extend into the homing tail")
        if len(intervals) != 4:
            raise ValueError(f"Expected four gripper-active intervals, found {len(intervals)}")

        status = "dry-run"
        if not args.dry_run:
            group_name, dataset_name = dataset_key.rsplit("/", 1)
            group = episode.require_group(group_name)
            if dataset_name in group:
                existing = np.asarray(group[dataset_name], dtype=np.bool_)
                if existing.shape == labels.shape and np.array_equal(existing, labels):
                    dataset = group[dataset_name]
                    status = "unchanged"
                elif not args.overwrite:
                    raise FileExistsError(
                        f"/{dataset_key} already exists with different values; pass --overwrite to replace it"
                    )
                else:
                    del group[dataset_name]
                    dataset = group.create_dataset(dataset_name, data=labels, dtype=np.bool_)
                    status = "overwritten"
            else:
                dataset = group.create_dataset(dataset_name, data=labels, dtype=np.bool_)
                status = "created"

            dataset.attrs["description"] = (
                "True from inferred hand-closing start through hand-opening completion, inclusive, for either hand"
            )
            dataset.attrs["method_version"] = METHOD_VERSION
            dataset.attrs["source_target_action_key"] = "/action/commanded"
            dataset.attrs["source_contact_group"] = "/robot/tactile/contact_mask"
            dataset.attrs["interval_semantics"] = "inclusive [close_start, open_complete] union across hands"
            dataset.attrs["intervals_json"] = json.dumps(_interval_records(intervals), separators=(",", ":"))
            episode.flush()

    if not args.quiet:
        print(f"episode={args.episode_path}")
        print(f"status={status}")
        print(f"dataset_key=/{dataset_key}")
        print(f"frame_count={frame_count}")
        print(f"homing_cutoff={homing_cutoff}")
        print(f"true_frame_count={int(labels.sum())}")
        for record in _interval_records(intervals):
            print(
                "interval="
                f"{record['side']},{record['close_start']},{record['open_start']},{record['open_complete']}"
            )


if __name__ == "__main__":
    main()
