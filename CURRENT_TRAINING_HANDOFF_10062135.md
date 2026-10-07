# Current Training Handoff — Bench2Dex Task26 π₀.₅

Generated: **2026-10-06 21:35 PDT**
Filename timestamp convention: `CURRENT_TRAINING_HANDOFF_MMDDHHMM.md`

## Read this first

This is a live-training handoff. Before changing or launching anything:

1. Read this file and `pi05_bench2dex_task26_finetuning.md`.
2. Check `tmux list-sessions`, `pgrep`, the latest log, and GPU 0.
3. Do **not** start a second copy of the training job.
4. Keep every cache, checkpoint, asset, and log under
   `/mnt/ssd1/szhao/DexTool`.
5. Use the project virtual environment at
   `/mnt/ssd1/szhao/DexTool/openpi_env`.

## Current live status

Training was restored from the complete **step 2000** checkpoint and is
currently running in a detached tmux session:

```text
tmux session: openpi_task26
training PID at handoff creation: 2272398
config: pi05_bench2dex_task26_action_expert
experiment: task26_action_expert_v1
target: 20,000 steps
```

The restored job was confirmed to have:

- loaded checkpoint step 2000 successfully;
- entered the training loop again;
- returned GPU 0 to 100% utilization;
- returned to approximately 4.3 seconds per step after recompilation.

The current estimate is approximately **24–25 hours remaining from the
21:36 PDT resume**, including checkpoint-save stalls.

Useful status commands:

```bash
tmux list-sessions
tmux capture-pane -pt openpi_task26:0 -S -60 | tail -60
pgrep -af 'scripts/train.py pi05_bench2dex_task26_action_expert'
nvidia-smi -i 0
tail -100 /mnt/ssd1/szhao/DexTool/openpi_logs/task26_action_expert_v1.log
```

Do not kill the tmux session unless explicitly requested or a verified fatal
condition requires it.

## Interruption and recovery history

The original training process reached approximately step 2240, then exited
abruptly without a Python traceback, OOM, NaN, or explicit training error. The
GPU returned to idle. The evidence points to the outer tool-managed persistent
session being terminated rather than the training code failing.

The latest complete checkpoint was step 2000, so steps 2001–approximately 2240
from the original process were not recoverable. Training was resumed with:

```text
--resume --no-wandb-enabled
```

Do not use `--overwrite` when resuming. The log contains the earlier uncommitted
step 2100/2200 metrics and will later contain repeated post-resume step numbers;
this is expected.

## W&B curve

Online run:

```text
https://wandb.ai/shuqi_zhao-univerisity-of-california-berkeley/openpi/runs/ebclf88k
```

The training process itself has native W&B disabled. A separate sidecar parses
the text log and uploads `loss`, `grad_norm`, and `param_norm` every 100 steps.

```text
tmux session: openpi_task26_wandb
run id: ebclf88k
run name: task26_action_expert_v1_metrics
```

Uploader source:

```text
scripts/upload_training_log_to_wandb.py
```

Uploader log:

```text
/mnt/ssd1/szhao/DexTool/openpi_logs/task26_action_expert_v1_wandb_uploader.log
```

The sidecar resumed the same W&B run. During restart it attempted to backfill
old points again; W&B correctly rejected lower duplicate steps. The sidecar
tracks the highest parsed step and will continue with new higher steps. Because
training resumed from 2000 after previously reaching about 2240, W&B may show a
small discontinuity around this region.

## Training recipe

The selected recipe is:

- π₀.₅ base model;
- action dimension 58;
- action horizon 50;
- three cameras:
  - main: `cam_overhead`;
  - left wrist: `cam_wrist_left`;
  - right wrist: `cam_wrist_right`;
- VLM/SigLIP/Gemma backbone frozen;
- train the action expert plus `action_in_proj` and `action_out_proj`;
- 58D action input/output projections initialized from scratch;
- global batch size 64;
- 20,000 steps;
- peak learning rate `5e-5`;
- 1,000 warmup steps;
- EMA 0.99;
- gradient clipping 1.0;
- JAX memory fraction 0.8 on GPU 0.

The actual filter was verified against the NNX parameter tree. It selects
428,052,538 parameters out of 3,353,487,146 total parameters (12.76%).

Checkpoint initialization correctly skipped only these shape-changed tensors:

```text
action_in_proj/kernel: 32x1024 -> 58x1024
action_out_proj/kernel: 1024x32 -> 1024x58
action_out_proj/bias: 32 -> 58
```

## Data

Dataset:

```text
/mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization
```

Dataset facts:

- 100 HDF5 episodes;
- robot: `multi_iiwa7_with_sharpa`;
- 58D state and commanded actions;
- 59,038 usable 50-step windows after validity and homing-tail filtering;
- images are decoded lazily from the source HDF5 files;
- no converted duplicate dataset was created.

The released Task26 files all store `meta/success=False`, but recomputation
using the current official evaluator shows all 100 trajectories satisfy the
terminal condition for longer than the required dwell time. Do not filter these
episodes using the released `success` or `demo_eligible` flags.

## Modular data implementation

Relevant files:

```text
src/openpi/training/dataset_processors.py
src/openpi/training/bench2dex_dataset.py
src/openpi/policies/bench2dex_policy.py
src/openpi/training/data_loader.py
src/openpi/training/config.py
src/openpi/training/weight_loaders.py
scripts/test_bench2dex_data_pipeline.py
```

Architecture:

```text
TrainConfig
  -> DataConfig.dataset_processor = "bench2dex_hdf5"
  -> dataset processor registry
  -> Bench2DexDatasetProcessor
  -> Bench2DexHdf5Dataset
  -> Bench2DexInputs
  -> common OpenPI transforms
```

Future raw dataset formats should be implemented as another processor class and
registered in `dataset_processors.py`, rather than adding format-specific logic
to the generic loader.

## Paths

```text
repository:
  /home/shuqi/DexCondVLA

virtual environment:
  /mnt/ssd1/szhao/DexTool/openpi_env

training log:
  /mnt/ssd1/szhao/DexTool/openpi_logs/task26_action_expert_v1.log

checkpoint root:
  /mnt/ssd1/szhao/DexTool/openpi_checkpoints/
    pi05_bench2dex_task26_action_expert/task26_action_expert_v1

latest complete checkpoint at handoff creation:
  .../task26_action_expert_v1/2000

normalization statistics:
  /mnt/ssd1/szhao/DexTool/openpi_assets/
    pi05_bench2dex_task26_action_expert/bench2dex_task26/norm_stats.json

base checkpoint cache:
  /mnt/ssd1/szhao/DexTool/openpi_assets/openpi-cache/
    openpi-assets/checkpoints/pi05_base/params
```

The π₀.₅ base checkpoint download is already complete (approximately 11.59
GiB). Do not download it again.

## Checkpoint behavior and capacity

- Checkpoint interval: every 1,000 steps.
- One checkpoint is approximately 14 GB.
- Each save currently stalls training for approximately 7–12 minutes because
  observed write throughput is only about 16–24 MiB/s.
- `keep_period=5000`; ordinary old checkpoints are deleted, while periodic
  checkpoints are retained.
- SSD free space at the handoff check was approximately 424 GB, sufficient for
  the expected retained checkpoints and temporary save space.

Before assuming a hang near a multiple of 1,000 steps, inspect the log for
Orbax checkpoint write activity.

## Last known metrics before interruption

These metrics came from the original process after checkpoint step 2000 and
were not themselves checkpointed:

```text
step 2000: loss=0.0521, grad_norm=0.2203, param_norm=1802.0933
step 2100: loss=0.0498, grad_norm=0.2105, param_norm=1802.1196
step 2200: loss=0.0464, grad_norm=0.1912, param_norm=1802.1453
```

`param_norm` covers all model kernels, including the frozen backbone. Its
change from about 1801.90 to 1802.15 is tiny relative to its magnitude and is
not parameter explosion.

## Validation completed

- Relevant unit tests: 5 passed.
- Ruff checks and formatting checks passed for modified files.
- Real Task26 episode passed the complete three-camera processing pipeline.
- Registered config selected the correct processor and data path.
- Actual trainable parameter paths contain only the action expert and action
  projections.
- Normalization statistics completed successfully.
- Base checkpoint restored successfully.
- Initial and resumed training both completed real GPU steps without OOM.

## Instructions for the next conversation

Start with:

```text
Read CURRENT_TRAINING_HANDOFF_10062135.md and
pi05_bench2dex_task26_finetuning.md. Verify the two tmux sessions, latest
checkpoint, training log, W&B run, GPU 0, and SSD before taking action. Do not
launch a duplicate training process.
```

For every future handoff, create a new file without overwriting this one, using:

```text
CURRENT_TRAINING_HANDOFF_MMDDHHMM.md
```

where the suffix is local Pacific time month, day, hour, and minute, matching
the requested example `CURRENT_TRAINING_HANDOFF_10062134.md`.
