"""Run one Bench2Dex episode through the complete OpenPI input pipeline."""

from __future__ import annotations

import dataclasses
import pathlib

import h5py
import jax
import numpy as np
import tyro

from openpi.models import pi0_config
from openpi.shared import normalize
from openpi.training import bench2dex_dataset
from openpi.training import config as training_config
from openpi.training import data_loader


def _episode_norm_stats(episode: pathlib.Path) -> dict[str, normalize.NormStats]:
    with h5py.File(episode, "r") as h5file:
        valid = bench2dex_dataset._valid_frame_indices(h5file)  # noqa: SLF001
        state = np.asarray(h5file["robot/qpos"][valid], dtype=np.float32)
        actions = np.asarray(h5file["action/commanded"][valid], dtype=np.float32)

    def stats(values: np.ndarray) -> normalize.NormStats:
        return normalize.NormStats(
            mean=np.mean(values, axis=0),
            std=np.std(values, axis=0),
            q01=np.quantile(values, 0.01, axis=0),
            q99=np.quantile(values, 0.99, axis=0),
        )

    return {"state": stats(state), "actions": stats(actions)}


def main(
    episode_path: str,
    *,
    action_horizon: int = 50,
    batch_size: int = 2,
) -> None:
    episode = pathlib.Path(episode_path).resolve()
    if not episode.is_file():
        raise FileNotFoundError(episode)

    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=58,
        action_horizon=action_horizon,
        max_token_len=280,
    )
    factory = training_config.Bench2DexDataConfig(
        repo_id=str(episode),
        dataset_processor="bench2dex_hdf5",
    )
    config = factory.create(pathlib.Path("/tmp/bench2dex_pipeline_assets"), model_config)
    config = dataclasses.replace(config, norm_stats=_episode_norm_stats(episode))
    loader = data_loader.create_torch_data_loader(
        config,
        model_config=model_config,
        action_horizon=action_horizon,
        batch_size=batch_size,
        num_batches=1,
        num_workers=0,
        shuffle=False,
        skip_norm_stats=False,
        framework="jax",
    )
    observation, actions = next(iter(loader))

    expected_image_keys = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    if set(observation.images) != expected_image_keys:
        raise AssertionError(f"Unexpected image keys: {set(observation.images)}")
    if observation.state.shape != (batch_size, 58):
        raise AssertionError(f"Unexpected state shape: {observation.state.shape}")
    if actions.shape != (batch_size, action_horizon, 58):
        raise AssertionError(f"Unexpected action shape: {actions.shape}")
    if observation.tokenized_prompt.shape != (batch_size, 280):
        raise AssertionError(f"Unexpected prompt shape: {observation.tokenized_prompt.shape}")
    for name, image in observation.images.items():
        if image.shape != (batch_size, 224, 224, 3):
            raise AssertionError(f"Unexpected {name} shape: {image.shape}")
        if not np.isfinite(np.asarray(image)).all():
            raise AssertionError(f"Non-finite pixels in {name}")
    if not np.isfinite(np.asarray(observation.state)).all() or not np.isfinite(np.asarray(actions)).all():
        raise AssertionError("Non-finite state/action values after processing")

    print(f"episode={episode}")
    print(f"jax_backend={jax.default_backend()}")
    print(f"state_shape={observation.state.shape} dtype={observation.state.dtype}")
    print(f"action_shape={actions.shape} dtype={actions.dtype}")
    print("quantile_normalization=enabled")
    for name, image in observation.images.items():
        print(f"image[{name}]={image.shape} dtype={image.dtype}")
    print(f"prompt_tokens={observation.tokenized_prompt.shape}")
    print("BENCH2DEX_DATA_PIPELINE_OK")


if __name__ == "__main__":
    tyro.cli(main)
