# Training Checkpoint Registry

This file is the persistent index of training runs and checkpoints produced by
this repository. Add one entry whenever a new training run starts, including
debug or ablation runs that may not finish. Do not reuse or overwrite an
existing experiment name or checkpoint directory.

## Recording rules

At training launch time, record:

1. the unique checkpoint/experiment name and launch time;
2. the config name, initialization checkpoint, output path, log path, and
   W&B link or run ID when available;
3. what the run is intended to learn;
4. the dataset, observation cameras, state/action dimensions, action
   representation, and action horizon;
5. frozen and trainable modules plus the main optimization settings;
6. every special change relative to the closest earlier run;
7. the exact Python virtual environment and device;
8. current status and the checkpoint selected for evaluation, once known.

The training entrypoints automatically save the exact launch-time working tree
under `<checkpoint_dir>/code_snapshots/<UTC_timestamp>_<PID>/`. This includes
tracked and non-ignored untracked files, the resolved training config, Python
environment metadata, Git HEAD/status/diff, and submodule revisions. `LATEST`
points to the newest snapshot. Both new runs and resumes create a snapshot;
snapshot failure aborts training rather than continuing without provenance.
The snapshot lives once per launch at the experiment root instead of being
duplicated into every numeric step directory.

Update the same entry when a run is resumed, stopped, completed, or selected
for evaluation. Never delete an old entry; mark superseded or failed runs
explicitly so checkpoint semantics remain traceable.

## Entry template

```markdown
## `<checkpoint_or_experiment_name>`

- Started: `YYYY-MM-DD HH:MM TZ`
- Status: `running | stopped | completed | failed`
- Purpose: ...
- Config: `...`
- Initialization: `...`
- Checkpoint directory: `.../{step}`
- Selected checkpoint: `...` or `not selected yet`
- Code snapshot: `<checkpoint directory>/code_snapshots/<timestamp>_<PID>`
- Log: `...`
- W&B: `...` or `disabled/not available`
- Environment: `...`
- Device: `...`

### Setting

- Dataset: `...`
- Cameras: `...`
- State/action dimension: `...`
- Action representation and label: `...`
- Action horizon: `...`
- Frozen modules: `...`
- Trainable modules: `...`
- Batch size / steps / LR / warmup / EMA / gradient clipping: `...`

### Explanation and special changes

Describe the overall goal and list every difference from the closest prior
run. Include changes to preprocessing, normalization, model structure,
training recipe, or checkpoint initialization.

### Status history

- `YYYY-MM-DD HH:MM TZ`: run launched.
```

## `task26_action_expert_v1`

- Started: `2026-10-06 PDT`
- Status: stopped for evaluation after step 16000
- Purpose: establish the π₀.₅ absolute-action Task26 baseline while freezing
  the vision-language backbone and fine-tuning the action path.
- Config: `pi05_bench2dex_task26_action_expert`
- Initialization: `gs://openpi-assets/checkpoints/pi05_base/params`
- Checkpoint directory:
  `/mnt/ssd1/szhao/DexTool/openpi_checkpoints/pi05_bench2dex_task26_action_expert/task26_action_expert_v1/{step}`
- Selected checkpoint:
  `/mnt/ssd1/szhao/DexTool/openpi_checkpoints/pi05_bench2dex_task26_action_expert/task26_action_expert_v1/16000`
- Code snapshot: unavailable; this run predates automatic code snapshots, so
  the current working tree must not be recorded retroactively as its source
- Log: `/mnt/ssd1/szhao/DexTool/openpi_logs/task26_action_expert_v1.log`
- W&B: `https://wandb.ai/shuqi_zhao-univerisity-of-california-berkeley/openpi/runs/ebclf88k`
- Environment: `/mnt/ssd1/szhao/DexTool/openpi_env`
- Device: GPU 0

### Setting

- Dataset:
  `/mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization`
  (100 replay HDF5 episodes; 59,038 usable 50-step windows)
- Cameras: `cam_overhead`, `cam_wrist_left`, `cam_wrist_right`
- State/action dimension: full 58 DOF
- Action representation: absolute joint target, with label `a[t+k]`
- Action horizon: 50
- Frozen modules: SigLIP and VLM/Gemma backbone
- Trainable modules: action expert, `action_in_proj`, `action_out_proj`
  (428,052,538 of 3,353,487,146 parameters; 12.76%)
- Optimization: batch size 64; target 20,000 steps; peak LR `5e-5`;
  1,000-step warmup; final LR `5e-6`; EMA 0.99; gradient clipping 1.0

### Explanation and special changes

This was the first Task26 action-expert-only run. The π₀.₅ base checkpoint has
32-dimensional action projections, while Task26 uses 58 dimensions. The
shape-compatible loader therefore restored all compatible pretrained weights
but initialized the new 58D `action_in_proj` and `action_out_proj` from
scratch. Stored dataset `meta/success` and `meta/demo_eligible` values were not
used for filtering because they are inconsistent with the current evaluator.

### Status history

- `2026-10-06 PDT`: training launched.
- The original process exited after approximately step 2240 without a Python
  traceback; training resumed from the complete step-2000 checkpoint.
- Step 16000 was selected for the absolute-action evaluation pipeline.

## `task26_aligned_delta_1009002600`

- Started: `2026-10-09 00:26:00 PDT` (timestamp encoded in experiment name)
- Status: completed
- Purpose: train the same Task26 action expert to predict per-timestep
  tracking corrections instead of absolute joint targets.
- Config: `pi05_bench2dex_task26_action_expert_aligned_delta`
- Initialization: `gs://openpi-assets/checkpoints/pi05_base/params`
- Checkpoint directory:
  `/mnt/ssd1/szhao/DexTool/ckp/pi05_bench2dex_task26_action_expert_aligned_delta/task26_aligned_delta_1009002600/{step}`
- Selected checkpoint:
  `/mnt/ssd1/szhao/DexTool/ckp/pi05_bench2dex_task26_action_expert_aligned_delta/task26_aligned_delta_1009002600/19999`
- Code snapshot: unavailable; this run predates automatic code snapshots, so
  the current working tree must not be recorded retroactively as its source
- Log: under `/mnt/ssd1/szhao/DexTool/openpi_logs`
- W&B: recorded by the training run; add the exact URL here when recovered
- Environment: `/mnt/ssd1/szhao/DexTool/openpi_env`
- Device: GPU 0

### Setting

- Dataset:
  `/mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization`
- Cameras: `cam_overhead`, `cam_wrist_left`, `cam_wrist_right`
- State/action dimension: full 58 DOF
- Action representation: aligned delta, with training label
  `a[t+k] - s[t+k]`
- Action horizon: 50
- Frozen modules: SigLIP and VLM/Gemma backbone
- Trainable modules: action expert, `action_in_proj`, `action_out_proj`;
  the 58D input/output projections train from scratch
- Optimization: batch size 64; 20,000 steps; peak LR `5e-5`;
  1,000-step warmup; final LR `5e-6`; EMA 0.99; gradient clipping 1.0

### Explanation and special changes

This run keeps the baseline architecture and optimization recipe but changes
the supervision and normalization domain. Each target is normalized after
conversion to `a[t+k] - s[t+k]`, using aligned-delta-specific normalization
statistics. At inference, the server returns an unnormalized 50-step delta
chunk. When queued element `k` executes, Bench2Dex reconstructs an absolute
joint target as `live_qpos[t+k] + predicted_delta[t+k]`. This is distinct from
the `state_delta` variant, which subtracts and later adds only the chunk-start
state. The horizon remains 50; it was not reduced to one.

### Status history

- `2026-10-09 00:26:00 PDT`: training launched with isolated checkpoint and
  log destinations so earlier checkpoints could not be overwritten.
- Step 19999 was selected for evaluation.
- Task26 evaluation result: official 0/100 and relaxed-without-object-upright
  0/100. Evaluation artifacts are under
  `/mnt/ssd1/szhao/DexTool/results/task26_aligned_delta_1009002600_step19999`.

## `task26_action_expert_absolute_25k_1011164016`

- Started: `2026-10-11 16:41:14 PDT`
- Status: running
- Purpose: independently replicate the first Task26 absolute-action training
  with a longer 25,000-step schedule and a smaller JAX memory reservation.
- Config: `pi05_bench2dex_task26_action_expert_25k`
- Initialization: `gs://openpi-assets/checkpoints/pi05_base/params`
- Checkpoint directory:
  `/mnt/ssd1/szhao/DexTool/ckp/pi05_bench2dex_task26_action_expert_25k/task26_action_expert_absolute_25k_1011164016/{step}`
- Selected checkpoint: not selected yet
- Code snapshot:
  `/mnt/ssd1/szhao/DexTool/ckp/pi05_bench2dex_task26_action_expert_25k/task26_action_expert_absolute_25k_1011164016/code_snapshots/20261011T234121.069908Z_2834054`
- Log:
  `/mnt/ssd1/szhao/DexTool/openpi_logs/task26_action_expert_absolute_25k_1011164016.log`
- W&B: `https://wandb.ai/shuqi_zhao-univerisity-of-california-berkeley/openpi/runs/s8pc3rl1`
- Environment: `/mnt/ssd1/szhao/DexTool/openpi_env`
- Device: GPU 0

### Setting

- Dataset:
  `/mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization`
- Cameras: `cam_overhead`, `cam_wrist_left`, `cam_wrist_right`
- State/action dimension: full 58 DOF
- Action representation: absolute joint target, label `a[t+k]`
- Action horizon: 50
- Frozen modules: SigLIP and VLM/Gemma backbone
- Trainable modules: action expert, `action_in_proj`, `action_out_proj`;
  the 58D input/output projections initialize from scratch
- Optimization: batch size 64; 25,000 steps; peak LR `5e-5`;
  1,000-step warmup; 25,000-step cosine decay to `5e-6`; EMA 0.99;
  gradient clipping 1.0
- JAX memory setting: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`
- Normalization: reuse the original absolute-action statistics from
  `/mnt/ssd1/szhao/DexTool/openpi_assets/pi05_bench2dex_task26_action_expert/bench2dex_task26/norm_stats.json`

### Explanation and special changes

This preserves the first run's absolute-action labels, dataset, three-camera
mapping, 58D architecture, batch size, frozen/trainable module split, and
optimizer settings. The differences are: training is extended from 20k to
25k steps, cosine decay is extended to the same 25k horizon, checkpoints use
a new isolated path under `/mnt/ssd1/szhao/DexTool/ckp`, and the JAX memory
fraction is reduced from 0.8 to 0.4. It is a fresh initialization from
`pi05_base`, not a resume from step 16000.

### Status history

- `2026-10-11 16:40:16 PDT`: old aligned-delta evaluation policy server was
  stopped, releasing approximately 39.5 GB on GPU 0; launch preparation began.
- `2026-10-11 16:41:14 PDT`: training launched in tmux session
  `openpi_task26_abs25k_1011164016`, with console log at the path above and
  PID 2834054 at launch.
- `2026-10-11 16:42:46 PDT`: step 0 completed successfully with loss 1.5103,
  gradient norm 1.1381, and parameter norm 1801.9016. XLA reported a compiled
  memory requirement of approximately 37.25 GiB, which fits within the 0.40
  allocator fraction; no OOM occurred.
