"""Verify all Bench2Dex action representations against a real HDF5 trajectory."""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import numpy as np

from openpi.training import bench2dex_dataset
from openpi.training import config as training_config

CONFIGS = {
    "absolute": "pi05_bench2dex_task26_action_expert",
    "state_delta": "pi05_bench2dex_task26_action_expert_state_delta",
    "aligned_delta": "pi05_bench2dex_task26_action_expert_aligned_delta",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--bench2dex-root", default="/mnt/ssd1/szhao/DexTool/Bench2Dex")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(args.bench2dex_root).resolve()))
    from utils.policy_timing import action_for_execution

    dataset = bench2dex_dataset.Bench2DexHdf5Dataset(
        [args.episode],
        action_horizon=50,
        action_dim=58,
        include_action_states=True,
    )
    sample_count = min(args.samples, len(dataset))
    if sample_count <= 0:
        raise ValueError("No valid samples found")

    max_errors: dict[str, float] = {}
    for representation, config_name in CONFIGS.items():
        config = training_config.get_config(config_name)
        data_config = config.data.create(config.assets_dirs, config.model)
        max_error = 0.0
        for sample_index in np.linspace(0, len(dataset) - 1, sample_count, dtype=int):
            raw = dataset[int(sample_index)]
            ground_truth = np.asarray(raw["actions"], dtype=np.float32).copy()
            aligned_states = np.asarray(raw["action_states"], dtype=np.float32)
            transformed = copy.deepcopy(raw)
            for transform in data_config.data_transforms.inputs:
                transformed = transform(transformed)
            output = {"state": raw["state"].copy(), "actions": transformed["actions"].copy()}
            for transform in data_config.data_transforms.outputs:
                output = transform(output)

            model_actions = np.asarray(output["actions"], dtype=np.float32)
            recovered = np.stack(
                [
                    action_for_execution(
                        model_actions[offset],
                        current_state=aligned_states[offset],
                        chunk_start_state=raw["state"],
                        representation=representation,
                    )
                    for offset in range(model_actions.shape[0])
                ]
            )
            max_error = max(max_error, float(np.max(np.abs(recovered - ground_truth))))
        max_errors[representation] = max_error

    print(
        {
            "episode": str(pathlib.Path(args.episode).resolve()),
            "samples": sample_count,
            "action_horizon": 50,
            "action_dim": 58,
            "max_abs_error": max_errors,
        }
    )
    if any(error > 1e-6 for error in max_errors.values()):
        raise AssertionError(f"Ground-truth reconstruction failed: {max_errors}")


if __name__ == "__main__":
    main()
