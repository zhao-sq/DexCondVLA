# Task26 Gripper-Activity Annotation

## Output dataset key

Each Task26 HDF5 trajectory receives two side-specific frame-aligned datasets:

```text
/labels/left_gripper_active
/labels/right_gripper_active
```

Each dataset has shape `[T]` and boolean dtype, where `T` is the trajectory's
original frame count. A side's frame is `True` only when that hand is between
its inferred closing start and opening completion, inclusive. All other frames,
including the return-to-home tail, are `False`.

The old `/labels/gripper_active` dataset is retained as a compatibility field.
It is exactly the logical OR of the new left and right labels. New VLA and Hand
DiT code must consume the two side-specific datasets, not this combined field.

The writer stores the method version, source keys, interval semantics, and the
four inferred intervals as HDF5 attributes on this dataset.

## Source signals

The annotation uses:

- absolute hand target positions from `/action/commanded`;
- joint names from `/action/action_names`;
- per-finger contact masks from `/robot/tactile/contact_mask`;
- `/meta/homing_start_sim_step` and `/time/sim_step` to exclude homing.

It does not use `/robot/qpos` to find the hand-motion boundaries.

For each hand, the target-action vector contains the 15 grasp-related flexion
joints whose names contain `MCP_FE`, `CMC_FE`, `PIP`, `DIP`, or end in `IP`.
Abduction/adduction (`AA`) joints are excluded from the progress calculation.

## Quantitative method

Contact provides only a robust anchor for each manipulation event:

1. Stable contact begins when at least two fingertip sensors report contact
   for three consecutive frames.
2. Stable release begins when all fingertip sensors report no contact for
   three consecutive frames.

The exact motion boundaries come from target actions. For a pre-contact open
reference `q_open` and an in-contact closed reference `q_closed`, closing
progress is

```text
C(t) = dot(q(t) - q_open, q_closed - q_open)
       / squared_norm(q_closed - q_open)
```

The first frame before contact for which `C(t) >= 0.1` for three consecutive
frames is the closing start.

Opening progress uses the same projection construction, with a grasp reference
before contact loss and an open reference after release:

- opening starts at 10% progress sustained for three frames;
- opening completes at 90% progress sustained for three frames.

Reference poses are robust medians over ten-frame windows. The pre-grasp open
window is 25--35 frames before stable contact, the closed reference is 5--15
frames after contact begins, the pre-release grasp reference is 5--15 frames
before contact disappears, and the post-release open reference is 25--35
frames after release. Windows are clipped to the usable pre-homing trajectory.

For each inferred interval, frames in the inclusive range
`[close_start, open_complete]` are labeled `True` only in the dataset matching
the interval's `LEFT` or `RIGHT` side. Their logical OR is written only to the
legacy compatibility dataset.

Existing annotated data can be migrated without re-running contact/action
inference. The legacy dataset stores all intervals, including their side, in
its `intervals_json` attribute. Use `--from-existing-label` to reconstruct both
side-specific masks from that metadata and verify that their union exactly
matches the existing label.

## Annotation script

Run all project Python commands in the OpenPI virtual environment:

```bash
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python \
  scripts/annotate_task26_gripper_activity.py \
  /path/to/episode_000000.hdf5 \
  --from-existing-label
```

Use `--dry-run` to infer and validate without changing the file. The command is
idempotent: existing identical labels are reported as `unchanged`. If a key
exists with different values, the script refuses to modify it unless
`--overwrite` is explicitly supplied. The method version for split labels is
`target_action_contact_projection_v2_side_specific`.

The corresponding visualization implementation is
`scripts/render_task26_gripper_activity_video.py`.
