"""Lazy reader for Bench2Dex replay HDF5 episodes.

The reader keeps the released JPEG images in-place and decodes only the three
views used by the policy.  No LeRobot conversion or duplicate image store is
required.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import os
from pathlib import Path
from typing import ClassVar

import cv2
import h5py
import numpy as np
import torch

DEFAULT_CAMERA_NAMES = ("cam_overhead", "cam_wrist_left", "cam_wrist_right")


def _decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_jpeg(raw_bytes) -> np.ndarray:
    encoded = np.frombuffer(bytes(raw_bytes), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode a Bench2Dex JPEG frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _episode_sort_key(path: str) -> tuple[int, int]:
    stem = Path(path).stem.removeprefix("episode_")
    episode, _, variant = stem.partition("_")
    return int(episode), int(variant or 0)


def list_episode_files(dataset_dir: str) -> list[str]:
    """Return all released episode HDF5 files in deterministic numeric order."""
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Bench2Dex dataset directory not found: {dataset_dir}")
    paths = [
        os.path.join(dataset_dir, name)
        for name in os.listdir(dataset_dir)
        if name.startswith("episode_") and name.endswith(".hdf5")
    ]
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {dataset_dir}")
    return sorted(paths, key=_episode_sort_key)


def _valid_frame_indices(h5file: h5py.File) -> np.ndarray:
    """Filter invalid frames and remove the return-to-home tail."""
    if "robot/qpos" not in h5file or "action/commanded" not in h5file:
        raise KeyError("Episode must contain /robot/qpos and /action/commanded")

    total = int(h5file["robot/qpos"].shape[0])
    actions = np.asarray(h5file["action/commanded"][:], dtype=np.float32)
    if actions.shape[0] != total:
        raise ValueError("qpos and commanded action lengths do not match")

    frame_valid = (
        np.asarray(h5file["frame_valid"][:], dtype=bool) if "frame_valid" in h5file else np.ones(total, dtype=bool)
    )
    action_valid = (
        np.asarray(h5file["action/action_valid"][:], dtype=bool)
        if "action/action_valid" in h5file
        else np.ones(total, dtype=bool)
    )
    valid = np.flatnonzero(frame_valid & action_valid & np.isfinite(actions).all(axis=1)).astype(np.int64)

    if "meta/homing_start_sim_step" in h5file and "time/sim_step" in h5file:
        homing_start = int(h5file["meta/homing_start_sim_step"][()])
        if homing_start >= 0:
            sim_steps = np.asarray(h5file["time/sim_step"][:])
            cutoff = int(np.searchsorted(sim_steps, homing_start, side="left"))
            if 0 < cutoff < total:
                valid = valid[valid < cutoff]
    return valid


@dataclasses.dataclass(frozen=True)
class EpisodeInfo:
    path: str
    valid_indices: np.ndarray
    prompt: str
    sample_count: int


@dataclasses.dataclass(frozen=True)
class Bench2DexDatasetProcessor:
    """Adapt a Bench2Dex HDF5 file or episode directory to an OpenPI dataset.

    This is the only class that interprets a Bench2Dex data source.  The
    generic loader selects it by ``NAME`` and remains independent of the HDF5
    schema, camera names, and prompt storage used by Bench2Dex.
    """

    NAME: ClassVar[str] = "bench2dex_hdf5"

    source: str
    camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES
    prompt: str | None = None

    def create_dataset(self, *, action_horizon: int, action_dim: int) -> Bench2DexHdf5Dataset:
        # Keep compatibility with the earlier bench2dex:/path spelling while
        # making the configured processor, rather than a URI prefix, select
        # the dataset implementation.
        source = self.source.removeprefix("bench2dex:")
        source = os.path.abspath(source)
        paths = [source] if os.path.isfile(source) else list_episode_files(source)
        return Bench2DexHdf5Dataset(
            hdf5_paths=paths,
            action_horizon=action_horizon,
            action_dim=action_dim,
            camera_names=self.camera_names,
            prompt=self.prompt,
        )


class Bench2DexHdf5Dataset(torch.utils.data.Dataset):
    """Random-access Bench2Dex dataset that produces OpenPI-ready raw samples."""

    def __init__(
        self,
        hdf5_paths: Sequence[str],
        action_horizon: int,
        *,
        action_dim: int = 58,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        prompt: str | None = None,
    ) -> None:
        if not hdf5_paths:
            raise ValueError("hdf5_paths is empty")
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")

        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.camera_names = tuple(camera_names)
        if self.camera_names != DEFAULT_CAMERA_NAMES:
            raise ValueError(f"Expected camera_names={DEFAULT_CAMERA_NAMES}, got {self.camera_names}")

        episodes: list[EpisodeInfo] = []
        for path in hdf5_paths:
            with h5py.File(path, "r") as h5file:
                self._validate_episode(h5file, path)
                valid = _valid_frame_indices(h5file)
                sample_count = int(valid.size) - self.action_horizon + 1
                if sample_count <= 0:
                    continue
                episode_prompt = prompt
                if episode_prompt is None:
                    episode_prompt = _decode_text(h5file["meta/instruction"][()])
                episodes.append(EpisodeInfo(path, valid, episode_prompt, sample_count))

        if not episodes:
            raise RuntimeError("No usable Bench2Dex samples after validation and homing truncation")
        self.episodes = episodes
        self._cumulative_sizes = np.cumsum([episode.sample_count for episode in episodes])

    def _validate_episode(self, h5file: h5py.File, path: str) -> None:
        qpos = h5file["robot/qpos"]
        actions = h5file["action/commanded"]
        if qpos.ndim != 2 or qpos.shape[1] != self.action_dim:
            raise ValueError(f"{path}: qpos shape {qpos.shape} does not have action_dim={self.action_dim}")
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"{path}: action shape {actions.shape} does not have action_dim={self.action_dim}")
        if qpos.shape[0] != actions.shape[0]:
            raise ValueError(f"{path}: qpos/action frame counts differ")

        if "robot/joint_names" in h5file and "action/action_names" in h5file:
            state_names = [_decode_text(x) for x in h5file["robot/joint_names"][:]]
            action_names = [_decode_text(x) for x in h5file["action/action_names"][:]]
            if state_names != action_names:
                raise ValueError(f"{path}: qpos and action joint ordering differ")

        total = int(qpos.shape[0])
        for camera in self.camera_names:
            key = f"cameras/{camera}/rgb"
            if key not in h5file:
                raise KeyError(f"{path}: missing /{key}")
            if h5file[key].shape[0] != total:
                raise ValueError(f"{path}: /{key} is not frame-aligned")
        if "meta/instruction" not in h5file:
            raise KeyError(f"{path}: missing /meta/instruction")

    def __len__(self) -> int:
        return int(self._cumulative_sizes[-1])

    def _locate(self, index: int) -> tuple[EpisodeInfo, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self._cumulative_sizes, index, side="right"))
        previous = int(self._cumulative_sizes[episode_index - 1]) if episode_index else 0
        return self.episodes[episode_index], index - previous

    def __getitem__(self, index: int) -> dict:
        episode, local_index = self._locate(int(index))
        frame_indices = episode.valid_indices[local_index : local_index + self.action_horizon]
        frame_index = int(frame_indices[0])

        with h5py.File(episode.path, "r") as h5file:
            state = np.asarray(h5file["robot/qpos"][frame_index], dtype=np.float32)
            actions = np.asarray(h5file["action/commanded"][frame_indices], dtype=np.float32)
            images = {
                camera: _decode_jpeg(h5file[f"cameras/{camera}/rgb"][frame_index]) for camera in self.camera_names
            }

        return {
            "state": state,
            "images": images,
            "actions": actions,
            "prompt": episode.prompt,
        }
