# pi0.5 Bench2Dex Task26 Evaluation

This document records the reproducible evaluation path for the Task26 action-expert checkpoint. Python must always be run from the explicit virtual environments listed below; do not use system Python or a base Conda environment.

## Evaluation targets

The original absolute-action baseline remains available and its completed
results are not overwritten:

- Task: Bench2Dex Task26, `26_canned_food_tray_line_arrangement`
- Robot: `multi_iiwa7_with_sharpa`
- Checkpoint: `/mnt/ssd1/szhao/DexTool/openpi_checkpoints/pi05_bench2dex_task26_action_expert/task26_action_expert_v1/16000`
- Result root: `/mnt/ssd1/szhao/DexTool/results/task26_action_expert_v1_step16000`
- Model observation cameras:
  - primary: `cam_overhead`
  - left wrist: `cam_wrist_left`
  - right wrist: `cam_wrist_right`
- State/action dimension: full 58 DOF
- Action chunk: 50 steps
- Evaluation horizon: 871 policy steps = 2613 physics steps at policy stride 3
- Rendered policy cameras: `cam_overhead`, `cam_wrist_left`, and `cam_wrist_right` only
- Policy-server warm-up: one synthetic inference before the RPC port starts listening

The aligned-delta evaluation uses the same protocol with these substitutions:

- Pipeline mode: `aligned_delta`
- Config: `pi05_bench2dex_task26_action_expert_aligned_delta`
- Checkpoint: `/mnt/ssd1/szhao/DexTool/ckp/pi05_bench2dex_task26_action_expert_aligned_delta/task26_aligned_delta_1009002600/19999`
- Result root: `/mnt/ssd1/szhao/DexTool/results/task26_aligned_delta_1009002600_step19999`

## Action-representation pipelines

`PIPELINE_MODE` selects matching preprocessing, OpenPI output semantics, and
Bench2Dex execution semantics. The default remains `absolute` for backward
compatibility.

| `PIPELINE_MODE` | Model label/output | Executor conversion |
| --- | --- | --- |
| `absolute` | `a[t+k]` | execute directly |
| `state_delta` | `a[t+k] - s[t]` | add the qpos captured when the chunk was requested to every queued delta |
| `aligned_delta` | `a[t+k] - s[t+k]` | when queued delta `k` executes, add the live qpos at that timestep |

All modes keep `action_horizon=50`. For `aligned_delta`, the server returns a
50-step delta chunk; it does not cumulatively convert that chunk from the
initial state. Bench2Dex consumes one delta at each 20 Hz policy timestep and
forms the absolute command using the current measured qpos:

```text
absolute_target[t + k] = live_qpos[t + k] + predicted_delta[t + k]
```

This preserves closed-loop tracking within an open-loop action chunk and does
not accumulate earlier actuator tracking errors. The original absolute mode
still sends its output directly, so the completed baseline pipeline is
unchanged.

Before Isaac Sim starts, the launcher queries the policy server's `describe`
RPC and requires exact agreement on config, checkpoint, action representation,
action horizon, and action dimension. This prevents accidentally connecting an
aligned-delta executor to an absolute checkpoint or vice versa.

Ground-truth validation uses a real Task26 replay HDF5 trajectory and applies
the same executor conversion to ten different 50-step, 58-dimensional chunks:

```bash
OPENPI_DATA_HOME=/mnt/ssd1/szhao/DexTool/openpi_assets/openpi-cache \
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python \
  scripts/verify_bench2dex_action_representations.py \
  --episode /mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization/episode_000035_1.hdf5 \
  --samples 10
```

Verified maximum absolute reconstruction errors:

| Representation | Maximum error |
| --- | ---: |
| `absolute` | `0.0` |
| `state_delta` | `5.96e-08` |
| `aligned_delta` | `5.96e-08` |

The VLM pipeline remains frozen. The action expert is finetuned, while its action input/output projections are initialized and trained from scratch for 58-dimensional state/action data.

## Inference performance settings

The evaluator generates a filtered collect config before launching Isaac Sim. It preserves every setting from Bench2Dex's source `configs/collect/default.yaml`, but retains only the three cameras consumed by this policy: overhead, left wrist, and right wrist. The unused stereo pair is therefore neither created nor rendered. Camera generalization, policy inputs, recording resolution, seeds, and success evaluation are unchanged.

The policy server performs one 480x640 synthetic inference after loading the checkpoint and before opening the RPC listener. This pays the JAX/XLA compilation cost before official timing begins. The warm-up uses the Task26 instruction and 58-dimensional state shape, validates the returned `[T, 58]` action chunk, and restores the policy RNG afterward. Every official episode still resets the RNG from its original episode seed. Set `--warmup-runs 0` only for debugging when startup latency matters more than rollout latency.

## Environments and external repositories

- OpenPI policy server: `/mnt/ssd1/szhao/DexTool/openpi_env/bin/python`
- Bench2Dex/Isaac Sim evaluator: `/mnt/ssd1/szhao/DexTool/bench2dex_isaaclab_env/bin/python`
- Bench2Dex repository: `/mnt/ssd1/szhao/DexTool/Bench2Dex`
- Isaac Lab v2.3.2 source: `/mnt/ssd1/szhao/DexTool/IsaacLab-2.3.2`
- Bench2Dex assets: `/mnt/ssd1/szhao/DexTool/dex2bench_dataset`
- Task26 replay/generalization anchors: `/mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization`

The asset mirror stores the Sharpa robot under `usd_sdf_linux`, while the evaluator expects `usd`. The local setup therefore contains this compatibility symlink:

```text
dex2bench_dataset/Robots_p/kuka+sharpa/usd
  -> dex2bench_dataset/Robots_p/kuka+sharpa/usd_sdf_linux
```

No asset content is changed by this symlink.

## Important 58-DOF setting

Bench2Dex defines `--active-dof` with a default value of true. For Sharpa this reduces the policy interface from 58 to 54 dimensions. Omitting the flag is therefore insufficient: this checkpoint must be evaluated with the explicit `--no-active-dof` option.

The verified smoke recording contains:

- `robot/qpos`: `(871, 58)`
- `action/commanded`: `(871, 58)`
- `cameras`: only `cam_overhead` is persisted to save space
- model input still uses overhead, left wrist, and right wrist images online

## Protocol

The run uses Bench2Dex's official evaluator, Task26 custom success checker, and the four official generalization channels. To satisfy the requested total of 100 rollouts, it runs 25 episodes per channel:

| Output channel | Bench2Dex profile | Episodes |
| --- | --- | ---: |
| `none` | `none` | 25 |
| `cov` | `cov_only` | 25 |
| `inv` | `inv_only` | 25 |
| `inv_cov` | `inv_cov` | 25 |
| Total |  | 100 |

The official repository's default large-scale setting is 50 episodes per channel. This run intentionally uses 25 per channel so that the requested total is exactly 100 while retaining all four protocol channels.

Evaluation is split into five-episode Isaac Sim processes to bound simulator memory growth. Completed chunks are detected from their JSONL line counts and skipped when the launcher is rerun.

## Reproduction

Start the original absolute policy server in a persistent session:

```bash
tmux new-session -d -s bench2dex_task26_policy \
  "cd /home/shuqi/DexCondVLA && \
   export CUDA_VISIBLE_DEVICES=0 \
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
   OPENPI_DATA_HOME=/mnt/ssd1/szhao/DexTool/openpi_assets/openpi-cache && \
   exec /mnt/ssd1/szhao/DexTool/openpi_env/bin/python -u \
     scripts/serve_bench2dex_rpc_policy.py \
     --config pi05_bench2dex_task26_action_expert \
     --checkpoint /mnt/ssd1/szhao/DexTool/openpi_checkpoints/pi05_bench2dex_task26_action_expert/task26_action_expert_v1/16000 \
     --warmup-runs 1 \
     --port 9000"
```

For the aligned-delta checkpoint, use a separate session and select its config:

```bash
tmux new-session -d -s bench2dex_task26_aligned_policy \
  "cd /home/shuqi/DexCondVLA && \
   export CUDA_VISIBLE_DEVICES=0 \
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
   OPENPI_DATA_HOME=/mnt/ssd1/szhao/DexTool/openpi_assets/openpi-cache && \
   exec /mnt/ssd1/szhao/DexTool/openpi_env/bin/python -u \
     scripts/serve_bench2dex_rpc_policy.py \
     --config pi05_bench2dex_task26_action_expert_aligned_delta \
     --checkpoint /mnt/ssd1/szhao/DexTool/ckp/pi05_bench2dex_task26_action_expert_aligned_delta/task26_aligned_delta_1009002600/19999 \
     --warmup-runs 1 \
     --port 9000"
```

Run or resume the original absolute evaluator:

```bash
cd /home/shuqi/DexCondVLA
PIPELINE_MODE=absolute bash scripts/run_bench2dex_task26_evaluation.sh
```

Run or resume the aligned-delta evaluator without touching the baseline result
directory:

```bash
cd /home/shuqi/DexCondVLA
PIPELINE_MODE=aligned_delta bash scripts/run_bench2dex_task26_evaluation.sh
```

The `0.40` XLA fraction was validated on GPU 0 before launching the current
run: checkpoint restore, one `(50, 58)` synthetic warm-up, and RPC startup all
completed successfully. With Isaac Sim active, GPU 0 retained roughly 28 GB
free at startup.

Current persistent sessions and logs:

```text
policy tmux: bench2dex_task26_aligned_policy
eval tmux:   bench2dex_task26_aligned_eval
policy log:  /mnt/ssd1/szhao/DexTool/openpi_logs/bench2dex_task26_aligned_policy_step19999.log
eval log:    /mnt/ssd1/szhao/DexTool/results/task26_aligned_delta_1009002600_step19999/evaluation_launcher.log
```

The launcher explicitly invokes the Isaac virtual environment. IsaacLab's temporary directory is redirected to the SSD result root because a shared `/tmp/isaaclab` directory may not be writable.

Before launching a chunk, the launcher writes `${ISAAC_TMPDIR}/task26_pi05_three_camera.yaml` from the installed Bench2Dex default collect config and passes it through `--collect-config`. Its startup log must report exactly:

```text
[init] Cameras: ['cam_overhead', 'cam_wrist_left', 'cam_wrist_right']
```

Set `FILTER_POLICY_CAMERAS=0` only when resuming an evaluation whose earlier chunks used the legacy five-camera renderer. This passes the original Bench2Dex collect config unchanged and prevents mixed rendering settings within one 100-rollout result set.

After all recordings exist, the optional relaxed offline score can be generated without rerunning inference or physics:

```bash
/mnt/ssd1/szhao/DexTool/bench2dex_isaaclab_env/bin/python \
  scripts/rescore_bench2dex_task26_without_object_upright.py \
  --eval-root /mnt/ssd1/szhao/DexTool/results/task26_action_expert_v1_step16000/eval \
  --output-dir /mnt/ssd1/szhao/DexTool/results/task26_action_expert_v1_step16000/rescore_without_object_upright \
  --bench2dex-root /mnt/ssd1/szhao/DexTool/Bench2Dex
```

This produces a separate diagnostic metric and never overwrites official success. It keeps the on-tray, object-static, tray-upright, and 0.5-second dwell requirements, but removes each object's upright requirement. Because recordings are sampled at 20 Hz, the offline dwell window is ten consecutive recorded frames rather than the evaluator's 30 consecutive 60 Hz physics steps.

For aligned-delta results, use the same scorer with the isolated result root:

```bash
/mnt/ssd1/szhao/DexTool/bench2dex_isaaclab_env/bin/python \
  scripts/rescore_bench2dex_task26_without_object_upright.py \
  --eval-root /mnt/ssd1/szhao/DexTool/results/task26_aligned_delta_1009002600_step19999/eval \
  --output-dir /mnt/ssd1/szhao/DexTool/results/task26_aligned_delta_1009002600_step19999/rescore_without_object_upright \
  --bench2dex-root /mnt/ssd1/szhao/DexTool/Bench2Dex
```

After all 100 JSONL records exist, export one browser-compatible H.264 overhead video per rollout:

```bash
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python \
  scripts/export_bench2dex_task26_rollouts.py \
  --eval-root /mnt/ssd1/szhao/DexTool/results/task26_action_expert_v1_step16000/eval \
  --output-dir /mnt/ssd1/szhao/DexTool/results/task26_action_expert_v1_step16000/rollouts \
  --bench2dex-root /mnt/ssd1/szhao/DexTool/Bench2Dex
```

Each filename contains the global rollout number, protocol channel, episode number, and official outcome, for example:

```text
rollout_001_none_episode_001_failure.mp4
rollout_057_inv_episode_007_success.mp4
```

The export directory also contains `rollout_manifest.csv`, `rollout_manifest.json`, and `success_summary.json`.

The aligned-delta export command is:

```bash
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python \
  scripts/export_bench2dex_task26_rollouts.py \
  --eval-root /mnt/ssd1/szhao/DexTool/results/task26_aligned_delta_1009002600_step19999/eval \
  --output-dir /mnt/ssd1/szhao/DexTool/results/task26_aligned_delta_1009002600_step19999/rollouts \
  --bench2dex-root /mnt/ssd1/szhao/DexTool/Bench2Dex
```

## Smoke-test evidence

- Valid full-length smoke result: `smoke_full58_explicit/output/per_episode.jsonl`
- Raw overhead recording: `smoke_full58_explicit/records/failure/episode_000001.hdf5`
- H.264 video: `smoke_full58_explicit/smoke_task26_overhead_failure.mp4`
- Video validation: H.264, `yuv420p`, 640x480, 871 frames, 20 FPS, 43.55 seconds
- Official outcome: failure; current and latched stage completion 0.25; safety violation step rate 0.0

Two earlier smoke directories contain configuration-abort records and must not be included in benchmark statistics:

- `smoke/`: IsaacLab temporary log directory was not writable
- `smoke_full58/`: the evaluator defaulted to active 54-DOF mode because `--no-active-dof` was not yet explicit

## Final results: absolute-action baseline

<!-- RESULTS_START -->
- Completed at: `2026-10-07T22:37:17-07:00`
- Episodes: **100**
- Successes: **0**
- Overall success rate: **0.0%**
- Validated H.264 overhead videos: **100**
- Video directory: `/mnt/ssd1/szhao/DexTool/results/task26_action_expert_v1_step16000/rollouts`

| Channel | Episodes | Successes | Success rate |
| --- | ---: | ---: | ---: |
| `none` | 25 | 0 | 0.0% |
| `cov` | 25 | 0 | 0.0% |
| `inv` | 25 | 0 | 0.0% |
| `inv_cov` | 25 | 0 | 0.0% |
<!-- RESULTS_END -->

## Version caveat

The downloaded Task26 training HDF5 instruction says the right hand handles the meat can and MSG and the left hand handles the tuna/master-chef cans. The current Bench2Dex scene YAML instead instructs the left hand to place the master-chef can and MSG and the right hand to place the milk box and potted-meat can. The current evaluation follows the current official scene YAML and success checker. This dataset/repository prompt-version mismatch must be considered when interpreting the final success rate.
