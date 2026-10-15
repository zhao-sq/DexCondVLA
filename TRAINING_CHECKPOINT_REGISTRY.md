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
- Status: stopped; superseded before evaluation and checkpoints deleted
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
- `2026-10-12 23:31 PDT`: stopped both processes after confirming that this
  launch was two independent optimizations rather than the required same-batch
  predicted-condition joint training. Both experiment checkpoint directories
  were deleted as requested; the immutable registry history is retained.

## `task26_dual_expert_predcond_stopgrad_v1_20261012_233955`

- Started: launch preparation began `2026-10-12 23:39:55 PDT`
- Status: stopped during initial batch-4 compilation; no training step or
  numeric checkpoint was produced
- Purpose: same-batch joint training of the pretrained π₀.₅ Action Expert and
  scratch Hand DiT, with Hand conditioned on VLA predictions and a stopped
  Hand-to-VLA gradient.
- Server: Vast instance `136.175.252.91:35499`, RTX PRO 6000 Blackwell Server
  Edition 96 GB
- Repository: `/workspace/DexCondVLA`
- Environment: `/workspace/DexCondVLA/.venv`
- Dataset: `/workspace/DexTool/bench2dex/task26/replay-generalization`
- Unified checkpoint directory:
  `/workspace/DexTool/ckp/pi05_bench2dex_task26_dual_expert_arm/task26_dual_expert_predcond_stopgrad_v1_20261012_233955/{step}`
- Log:
  `/workspace/DexTool/openpi_logs/task26_dual_expert_predcond_stopgrad_v1_20261012_233955.log`
- Selected checkpoint: validation-best pointer `BEST`; not selected yet
- Code snapshot: must be present at `<checkpoint_dir>/code_snapshots/LATEST`
- Device: GPU 0

### Setting

- Every optimizer step uses one shared batch for both experts.
- VLA input: three RGB cameras, language, and current 14D arm state.
- VLA target/output: `(50, 16)` absolute
  `[left_joint_7, left_active, right_joint_7, right_active]`, internally padded
  to the official pretrained 32D width.
- VLA initialization: official `pi05_base`; VLM/SigLIP frozen; Action Expert
  plus action projections fine-tuned with the config's cosine schedule peaking
  at `5e-5`.
- Hand arm condition: VLA's single-forward denoised 16D estimate, converted
  back to physical units, split into two 8D side conditions, then explicitly
  detached before entering Hand DiT.
- Hand target: ground-truth `(50, 22)` joint chunk for each side. One shared
  six-layer, width-512, eight-head Hand DiT is initialized from scratch and
  optimized with AdamW at `1e-4`.
- Other Hand conditions: frozen DINOv3 ViT-B/16 wrist patch tokens, normalized
  current 22D hand state, and learned left/right side embedding.
- Total loss is optimized by two framework-native optimizers; Hand loss cannot
  update VLA, while VLA flow loss continues to fine-tune the Action Expert.
- Schedule: batch 4, 20,000 steps, save/validate every 1,000 steps, two fixed
  validation batches per save.
- Checkpoint retention: retain only the latest step and the minimum combined
  validation score (`vla_validation_loss + hand_validation_loss`). If they are
  the same step, retain only one checkpoint. Each retained step contains the
  Orbax VLA state, `hand_dit.pt`, and `joint_metrics.json`.
- Storage backend: `OPENPI_CHECKPOINT_USE_OCDBT=0`; no concurrent-byte limiter.

### Explanation and special changes

This replaces the stopped GT-conditioned two-process run. The deploy-time
conditioning distribution is now present during training, the two experts see
the exact same samples, and a single VLA forward supplies both the VLA loss and
the denoised condition. Crossing JAX to NumPy/Torch plus `.detach()` makes the
gradient boundary explicit. The larger scratch Hand LR and smaller pretrained
Action Expert LR remain intentionally separate.

### Status history

- `2026-10-12 23:47 PDT`: the remote two-step batch-1 joint GPU smoke passed
  from official `pi05_base`. Step 1 produced VLA/Hand losses
  `0.196499/1.341047`; step 2 produced `0.117124/1.317484`. The first XLA
  compilation took about seven minutes, while the cached second coordinated
  step ran at 4.896 steps/s. No training checkpoint was written by the smoke.
- `2026-10-12 23:54 PDT`: a formal batch-4 process was launched under
  Supervisor and its `code_snapshots/LATEST` was verified, then stopped at the
  user's request before the initial compilation completed so the default
  batch 64 could be tested. Only `code_snapshots/` exists in this run.

## `task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520`

- Started: launch preparation began `2026-10-12 23:55:20 PDT`
- Status: stopped by user after the step-19,000 checkpoint save stalled;
  step 18,000 selected for and completed full evaluation
- Purpose: same architecture and optimization as
  `task26_dual_expert_predcond_stopgrad_v1_20261012_233955`, testing and, if it
  fits, formally using the original config batch size 64.
- Server: Vast instance `136.175.252.91:35499`, RTX PRO 6000 Blackwell Server
  Edition 96 GB
- Repository/environment: `/workspace/DexCondVLA`,
  `/workspace/DexCondVLA/.venv`
- Dataset: `/workspace/DexTool/bench2dex/task26/replay-generalization`
- Unified checkpoint directory:
  `/workspace/DexTool/ckp/pi05_bench2dex_task26_dual_expert_arm/task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520/{step}`
- Log:
  `/workspace/DexTool/openpi_logs/task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520.log`
- Initialization/freeze boundary: official `pi05_base`; VLM/SigLIP and DINOv3
  frozen; Action Expert fine-tuned; Hand DiT scratch; Hand-to-VLA gradient
  stopped.
- Optimization: VLA default config cosine schedule with peak `5e-5`; Hand
  AdamW `1e-4`; batch 64; 20,000 steps; save/validate every 1,000 steps.
- Checkpoint retention: latest plus minimum combined fixed-validation loss.

### Status history

- `2026-10-13 00:12 PDT`: batch-64 joint computation completed two steps
  without OOM at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`. Step 1 VLA/Hand losses
  were `0.156778/1.404863`; step 2 losses were `0.142524/1.378899`. Cached
  throughput was 0.255 step/s (about 3.92 seconds/step). The smoke then exposed
  a DataLoader-worker exception only during interpreter teardown, so an
  explicit loader close was added before formal launch.
- `2026-10-13 00:16 PDT`: repeated the same two-step batch-64 smoke after the
  explicit close change. It reproduced the same losses and 0.256 cached
  steps/s, then exited cleanly with no worker error.
- `2026-10-13 00:18 PDT`: launched the formal run under Supervisor program
  `task26_dual_joint`, PID 29616 at launch. Verified
  `code_snapshots/LATEST`, 78.4 GiB allocated GPU memory, and repository commit
  `d753637`.
- `2026-10-13 00:24 PDT`: step 40 completed. The compile-inclusive first
  20-step window ran at 0.102 steps/s; the steady step 21--40 window ran at
  0.251 steps/s. VLA loss averaged 0.117249 and Hand loss 0.788547 in the
  latter window. Estimated completion is about 22--23 hours after launch,
  including validation/checkpoint overhead.
- `2026-10-14 04:21 PDT`: training reached step 19,000 and entered Orbax
  checkpoint saving, but the save remained in
  `19000.orbax-checkpoint-tmp-*`. TensorStore worker threads were later
  confirmed in uninterruptible Linux memory-management waits while GPU
  utilization was zero. The partial step 19,000 checkpoint is not loadable.
- `2026-10-14 11:36 PDT`: at the user's direction, stopped the stalled formal
  trainer and did not resume training. W&B was left running. The latest and
  validation-best complete checkpoint are both step 18,000. Its fixed
  validation VLA/Hand losses are `0.0014852549647912383` and
  `0.007435241714119911` (combined score `0.00892049667891115`). Step 18,000
  was selected for the full 100-rollout remote evaluation and result return.
- `2026-10-15 PDT`: completed the 100-rollout Task26 evaluation for the
  step-18,000 checkpoint. Official success was `0/100`. The relaxed score that
  omits per-object upright checks was `10/100`: `none=4/25`, `cov=2/25`,
  `inv=4/25`, and `inv_cov=0/25`. All 100 H.264 videos, both summaries, and the
  completion marker were copied to
  `/mnt/ssd1/szhao/DexTool/results/task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520_step18000`.
