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

## `task26_dual_expert_gtcond_v1_20261012_223318`

- Started: launch preparation began `2026-10-12 22:33:18 PDT`
- Status: running both formal 20,000-step components remotely
- Purpose: jointly train the Task26 dual-expert policy with a fine-tuned π₀.₅
  arm/activity Action Expert and a from-scratch shared Hand DiT.
- Server: Vast instance `136.175.252.91:35499`, RTX PRO 6000 Blackwell Server
  Edition 96 GB
- Repository: `/workspace/DexCondVLA`
- Environment: `/workspace/DexCondVLA/.venv`; PyTorch `2.7.1+cu128`; JAX
  `0.5.3`
- Dataset:
  `/workspace/DexTool/bench2dex/task26/replay-generalization`
- Arm checkpoint directory:
  `/workspace/DexTool/ckp/pi05_bench2dex_task26_dual_expert_arm/task26_dual_expert_gtcond_v1_20261012_223318/{step}`
- Hand checkpoint directory:
  `/workspace/DexTool/ckp/task26_dual_expert_gtcond_v1_20261012_223318_hand/step_*.pt`
- Log directory:
  `/workspace/DexTool/openpi_logs/task26_dual_expert_gtcond_v1_20261012_223318`
- Selected checkpoint: not selected yet
- Code snapshots: both component checkpoint directories contain
  `code_snapshots/LATEST`
- Device: GPU 0

### Setting

- VLA observation: overhead, left-wrist, and right-wrist RGB; language prompt;
  current 14D arm-joint state.
- VLA physical target: absolute `(50, 16)` chunk ordered as
  `[left arm joints 7, left_active, right arm joints 7, right_active]`, padded
  to the pretrained internal 32D width.
- Activity supervision: independent
  `/labels/left_gripper_active` and `/labels/right_gripper_active`; inference
  threshold is 0.5.
- VLA initialization: `gs://openpi-assets/checkpoints/pi05_base/params`.
- VLA trainable modules: π₀.₅ Action Expert, `action_in_proj`, and
  `action_out_proj`; VLM/SigLIP remain frozen.
- VLA optimization: batch 32; 20,000 steps; peak LR `2e-5`; 500-step warmup;
  cosine decay to `2e-6`; EMA 0.99; gradient clipping 1.0.
- VLA checkpointing on this server: non-OCDBT Zarr via
  `OPENPI_CHECKPOINT_USE_OCDBT=0`, with no byte limiter; save interval 1,000.
- Hand target: ground-truth left and right `(50, 22)` hand-joint chunks.
- Hand condition in this stage: corresponding ground-truth `(50, 8)` arm
  joints plus side-specific activity label, detached from the VLA.
- Hand model: one shared six-block Hand DiT, width 512, eight heads, learned
  side embeddings; initialized from scratch.
- Hand vision: frozen DINOv3 ViT-B/16 wrist patch features from
  `/workspace/DexTool/assets/dinov3_vitb16.pth`.
- Hand optimization: batch 8; 20,000 steps; constant LR `1e-4`; AdamW weight
  decay `1e-4`; gradient clipping 1.0.
- Execution: one tmux session with separate `vla` and `hand` windows on the
  same GPU. The losses and optimizers are independent in this GT-conditioned
  stage.

### Explanation and special changes

Relative to the earlier 58D Action Expert runs, π₀.₅ now predicts only the 14
arm joints plus two side-specific activity channels. Relative to a deployable
predicted-arm-conditioned hand policy, this initial run deliberately uses
ground-truth arm/activity chunks for Hand DiT so the complete data, DINOv3,
and flow-matching training path can be validated before introducing VLA
prediction error. The smaller VLA learning rate protects pretrained Action
Expert weights during fine-tuning, while the randomly initialized Hand DiT
uses the larger `1e-4` learning rate.

### Status history

- `2026-10-12 22:33:18 PDT`: remote repository cloned at commit `28d99fc`;
  dataset and DINOv3 assets placed under `/workspace/DexTool`; project `.venv`
  created. The default cu126 PyTorch wheel was replaced with the official cu128
  wheel after an `sm_120` incompatibility was detected. Both PyTorch and JAX
  then completed CUDA tensor smoke tests successfully.
- `2026-10-12 22:46 PDT`: generated the remote arm-only normalization stats
  from all 922 batches. Verified state width 14, action width 16, and distinct
  left/right activity channels at action indices 7 and 15.
- `2026-10-12 22:56 PDT`: Hand DiT completed a three-step smoke test with
  losses 1.362572, 1.442464, and 1.370296. The final smoke checkpoint is
  `step_00000003.pt` and is 472 MiB.
- `2026-10-12 23:18 PDT`: VLA batch-32 forward/backward smoke steps completed
  without OOM (losses 0.1495, 0.1658, and 0.1355). Default OCDBT checkpointing
  stalled on the container overlay, so commit `0009d6b` added opt-in checkpoint
  backend controls. Non-OCDBT saving without a byte limiter then finalized a
  resumable checkpoint at step 1 in 46 seconds, with measured write throughput
  of 158--303 MiB/s. The reduced-batch checkpoint-only retry does not change
  model, optimizer, or checkpoint shapes; the formal resume returns to batch
  32.
- `2026-10-12 23:19 PDT`: launched the formal VLA resume in tmux session
  `task26_dual_gtcond_223318`, window `vla`, from Orbax checkpoint step 1.
  Restore completed in 25 seconds, batch 32 resumed successfully, and the
  resume snapshot is `20261013T061917.162471Z_15181`.
- `2026-10-12 23:21 PDT`: launched the formal Hand DiT resume in the same tmux
  session, window `hand`, from `step_00000003.pt`. The resume snapshot is
  `20261013T062113.061851Z_16860`. Early live checks showed VLA step-20 loss
  0.1393 and gradient norm 0.4324; Hand step-200 loss was 0.245692. The two
  jobs used approximately 51.5 GiB of 97.9 GiB VRAM together with no OOM.
  Launch PIDs were 15181 for VLA and 16860 for Hand.
