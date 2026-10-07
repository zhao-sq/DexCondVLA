"""Registry for dataset-format-specific processors.

The generic OpenPI data loader should not know how a particular on-disk
dataset is laid out.  A processor owns that boundary and returns a random
access dataset in OpenPI's raw sample convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import openpi.models.model as _model
from openpi.training import bench2dex_dataset


def create_dataset(
    processor_name: str,
    source: str,
    *,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    options: Mapping[str, Any] | None = None,
):
    """Create a raw dataset using the processor selected by data config."""
    processor_options = dict(options or {})
    if processor_name == bench2dex_dataset.Bench2DexDatasetProcessor.NAME:
        return bench2dex_dataset.Bench2DexDatasetProcessor(
            source=source,
            **processor_options,
        ).create_dataset(
            action_horizon=action_horizon,
            action_dim=model_config.action_dim,
        )
    raise ValueError(
        f"Unknown dataset processor {processor_name!r}. "
        f"Available processors: {[bench2dex_dataset.Bench2DexDatasetProcessor.NAME]}"
    )
