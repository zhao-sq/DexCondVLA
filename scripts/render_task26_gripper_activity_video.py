"""Render Task26 video with green-left and red-right activity overlays."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


@dataclass(frozen=True)
class GripperInterval:
    side: str
    close_start: int
    open_start: int
    open_complete: int

    def phase(self, frame_index: int) -> str | None:
        if self.close_start <= frame_index < self.open_start:
            return "CLOSING / HOLDING"
        if self.open_start <= frame_index <= self.open_complete:
            return "OPENING"
        return None


CONTACT_CONFIRM_FRAMES = 3
PROGRESS_CONFIRM_FRAMES = 3
CLOSE_START_PROGRESS = 0.1
OPEN_START_PROGRESS = 0.1
OPEN_COMPLETE_PROGRESS = 0.9


def _decode_jpeg(raw: object) -> np.ndarray:
    encoded = np.frombuffer(bytes(raw), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG frame")
    return frame


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _homing_cutoff(episode: h5py.File) -> int:
    total = len(episode["action/commanded"])
    homing_start = int(episode["meta/homing_start_sim_step"][()])
    if homing_start < 0:
        return total
    return int(np.searchsorted(episode["time/sim_step"][:], homing_start, side="left"))


def _contact_intervals(episode: h5py.File, side: str, cutoff: int) -> list[tuple[int, int]]:
    sites = [name for name in episode["robot/tactile/contact_mask"] if name.startswith(side.lower())]
    active_sensor_count = np.zeros(cutoff, dtype=np.int64)
    for site in sites:
        masks = episode[f"robot/tactile/contact_mask/{site}"]
        active_sensor_count += np.array([np.any(masks[index]) for index in range(cutoff)], dtype=np.int64)

    intervals: list[tuple[int, int]] = []
    contact_start: int | None = None
    for index in range(cutoff - CONTACT_CONFIRM_FRAMES + 1):
        if contact_start is None:
            stable_contact = np.all(active_sensor_count[index : index + CONTACT_CONFIRM_FRAMES] >= 2)
            if stable_contact and (index == 0 or active_sensor_count[index - 1] < 2):
                contact_start = index
        else:
            stable_no_contact = np.all(active_sensor_count[index : index + CONTACT_CONFIRM_FRAMES] == 0)
            if stable_no_contact and (index == 0 or active_sensor_count[index - 1] > 0):
                intervals.append((contact_start, index))
                contact_start = None
    return intervals


def _first_sustained_crossing(
    values: np.ndarray, threshold: float, start: int, stop: int, *, choose_last: bool = False
) -> int:
    candidates = [
        index
        for index in range(start, max(start, stop - PROGRESS_CONFIRM_FRAMES + 1))
        if np.all(values[index : index + PROGRESS_CONFIRM_FRAMES] >= threshold)
        and (index == start or values[index - 1] < threshold)
    ]
    if candidates:
        return candidates[-1] if choose_last else candidates[0]
    search = values[start:stop]
    if search.size == 0:
        return start
    return start + int(np.argmin(np.abs(search - threshold)))


def infer_gripper_intervals(episode: h5py.File) -> tuple[list[GripperInterval], int]:
    cutoff = _homing_cutoff(episode)
    names = [_decode(value) for value in episode["action/action_names"][()]]
    actions = np.asarray(episode["action/commanded"], dtype=np.float64)
    if not np.isfinite(actions[0]).all():
        actions[0] = actions[1]

    inferred: list[GripperInterval] = []
    for side, prefix in (("LEFT", "left_"), ("RIGHT", "multi_right_")):
        flexion_indices = [
            index
            for index, name in enumerate(names)
            if name.startswith(prefix) and ("_FE" in name or "_PIP" in name or "_DIP" in name or name.endswith("_IP"))
        ]
        hand_targets = actions[:, flexion_indices]
        previous_release = 1
        for contact_start, contact_end in _contact_intervals(episode, side, cutoff):
            pre_start = max(previous_release, contact_start - 35)
            pre_end = max(pre_start + 1, contact_start - 25)
            open_before = np.median(hand_targets[pre_start:pre_end], axis=0)
            closed_after = np.median(hand_targets[contact_start + 5 : contact_start + 15], axis=0)
            close_direction = closed_after - open_before
            close_denominator = float(close_direction @ close_direction)
            if close_denominator == 0:
                raise ValueError(f"Zero close-direction norm for {side} contact at frame {contact_start}")
            close_progress = (hand_targets - open_before) @ close_direction / close_denominator
            close_search_start = max(previous_release, contact_start - 50)
            close_start = _first_sustained_crossing(
                close_progress,
                CLOSE_START_PROGRESS,
                close_search_start,
                contact_start,
                choose_last=True,
            )

            closed_before_release = np.median(hand_targets[contact_end - 15 : contact_end - 5], axis=0)
            post_start = min(cutoff - 3, contact_end + 25)
            post_end = min(cutoff, contact_end + 36)
            if post_end - post_start < 3:
                post_start = max(contact_end + 1, cutoff - 10)
            open_after = np.median(hand_targets[post_start:post_end], axis=0)
            open_direction = open_after - closed_before_release
            open_denominator = float(open_direction @ open_direction)
            if open_denominator == 0:
                raise ValueError(f"Zero open-direction norm for {side} release at frame {contact_end}")
            open_progress = (hand_targets - closed_before_release) @ open_direction / open_denominator
            open_start = _first_sustained_crossing(
                open_progress,
                OPEN_START_PROGRESS,
                max(close_start + 1, contact_end - 20),
                min(cutoff, contact_end + 25),
            )
            open_complete = _first_sustained_crossing(
                open_progress,
                OPEN_COMPLETE_PROGRESS,
                max(open_start, contact_end - 10),
                cutoff,
            )
            inferred.append(GripperInterval(side, close_start, open_start, open_complete))
            previous_release = contact_end

    return sorted(inferred, key=lambda interval: interval.close_start), cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--camera", default="cam_overhead")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--metadata-path", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be between 0 and 1")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.episode_path, "r") as episode:
        intervals, homing_cutoff = infer_gripper_intervals(episode)
        frames = episode[f"cameras/{args.camera}/rgb"]
        fps = float(episode["meta/effective_fps"][()])
        first = _decode_jpeg(frames[0])
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(args.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {args.output_path}")

        overlay_colors = {
            "LEFT": (0, 255, 0),
            "RIGHT": (0, 0, 255),
        }
        try:
            for frame_index in range(len(frames)):
                frame = first.copy() if frame_index == 0 else _decode_jpeg(frames[frame_index])
                active = [interval for interval in intervals if interval.phase(frame_index)]
                for interval in active:
                    color = overlay_colors[interval.side]
                    overlay = np.empty_like(frame)
                    overlay[:] = color
                    frame = cv2.addWeighted(frame, 1.0 - args.alpha, overlay, args.alpha, 0.0)
                for line_index, interval in enumerate(active):
                    phase = interval.phase(frame_index)
                    label = f"GRIPPER ACTIVE: {interval.side} - {phase}"
                    color = overlay_colors[interval.side]
                    top = 42 + line_index * 42
                    bottom = top + 38
                    cv2.rectangle(frame, (10, top), (525, bottom), color, -1)
                    cv2.putText(
                        frame,
                        label,
                        (20, top + 27),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    frame,
                    f"frame {frame_index}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
        finally:
            writer.release()

    print(f"python_video_frames={len(frames)}")
    print(f"fps={fps:g}")
    print(f"resolution={width}x{height}")
    for interval in intervals:
        print(f"interval={interval.side},{interval.close_start},{interval.open_start},{interval.open_complete}")
    print(f"output={args.output_path}")
    if args.metadata_path is not None:
        args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "episode": str(args.episode_path),
            "camera": args.camera,
            "fps": fps,
            "frame_count": len(frames),
            "homing_cutoff": homing_cutoff,
            "overlay_alpha": args.alpha,
            "method": {
                "contact_confirmation_frames": CONTACT_CONFIRM_FRAMES,
                "progress_confirmation_frames": PROGRESS_CONFIRM_FRAMES,
                "close_start_progress": CLOSE_START_PROGRESS,
                "open_start_progress": OPEN_START_PROGRESS,
                "open_complete_progress": OPEN_COMPLETE_PROGRESS,
            },
            "intervals": [
                {
                    "side": interval.side,
                    "close_start": interval.close_start,
                    "open_start": interval.open_start,
                    "open_complete": interval.open_complete,
                }
                for interval in intervals
            ],
        }
        args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"metadata={args.metadata_path}")


if __name__ == "__main__":
    main()
