# π₀.₅ Bench2Dex Task26 Fine-tuning Runbook

This file is the source of truth for reproducing and debugging the current
Task26 experiment. The implementation reads Bench2Dex replay HDF5 files
directly; it does not create a second converted copy of the image dataset.

## 1. Selected configuration

The registered OpenPI config is:

```text
pi05_bench2dex_task26_action_expert
```

It is defined in `src/openpi/training/config.py` and contains both the dataset
selection and the fine-tuning recipe.

| Item | Value |
|---|---|
| Base checkpoint | `gs://openpi-assets/checkpoints/pi05_base/params` |
| Model | π₀.₅ |
| Action dimension | 58 |
| Action horizon | 50 |
| Prompt token length | 280 |
| Trainable modules | action expert, `action_in_proj`, `action_out_proj` |
| Frozen modules | SigLIP and the VLM/Gemma backbone |
| Batch size | 64 |
| Steps | 20,000 |
| Peak LR | `5e-5` |
| Warmup | 1,000 steps |
| Final LR | `5e-6` |
| EMA | 0.99 |
| Gradient clipping | 1.0 |
| Precision | bf16 for frozen parameters / JAX training defaults |

The instantiated model has 3,353,487,146 parameters. The configured filter
selects 428,052,538 parameters (12.76%): only the action expert and the two
action projections. This was checked against the actual NNX parameter paths.

Because the base checkpoint uses 32D actions while Task26 uses 58D actions,
`ShapeCompatibleCheckpointWeightLoader` loads all shape-compatible tensors and
leaves only `action_in_proj` and `action_out_proj` at their new random
initialization. Any unexpected shape mismatch raises an error.

## 2. Dataset processor architecture

Dataset format selection is explicit in config:

```python
Bench2DexDataConfig(
    repo_id="/mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization",
    dataset_processor="bench2dex_hdf5",
)
```

The call path is:

```text
TrainConfig
  -> DataConfig.dataset_processor
  -> dataset_processors.create_dataset(...)
  -> Bench2DexDatasetProcessor
  -> Bench2DexHdf5Dataset
  -> Bench2DexInputs
  -> common OpenPI resize/tokenize/normalize transforms
```

Responsibilities are separated as follows:

- `src/openpi/training/dataset_processors.py`: generic processor registry and
  dispatch. Add future dataset formats here without adding format-specific
  branches to `data_loader.py`.
- `src/openpi/training/bench2dex_dataset.py`:
  `Bench2DexDatasetProcessor` interprets a Bench2Dex file or directory;
  `Bench2DexHdf5Dataset` validates episodes and lazily reads samples.
- `src/openpi/policies/bench2dex_policy.py`: maps raw Bench2Dex fields and
  cameras into OpenPI model inputs and maps predictions back to 58D actions.
- `src/openpi/training/data_loader.py`: format-independent loader. It delegates
  construction to the processor named by config.

To add another raw dataset format, implement a processor class with a
`create_dataset` method, register its name in `dataset_processors.py`, and set
that name in the new data config. The shared batching and model transforms do
not need another hard-coded dataset branch.

## 3. Task26 sample mapping

| OpenPI field | Bench2Dex source |
|---|---|
| `state` | `/robot/qpos` |
| `actions` | `/action/commanded` |
| `base_0_rgb` | `/cameras/cam_overhead/rgb` |
| `left_wrist_0_rgb` | `/cameras/cam_wrist_left/rgb` |
| `right_wrist_0_rgb` | `/cameras/cam_wrist_right/rgb` |
| prompt | `/meta/instruction` |

The processor rejects the return-to-home tail using
`meta/homing_start_sim_step`, rejects invalid frames, validates the 58D joint
ordering, and decodes JPEGs lazily. It does not filter on the released
`meta/success` or `meta/demo_eligible` values because Task26's stored success
labels are inconsistent with the current official evaluator.

The 100 episodes contain 59,038 usable 50-step training windows after validity
checks and homing-tail truncation.

## 4. Storage and environment

Use only the project virtual environment:

```text
/mnt/ssd1/szhao/DexTool/openpi_env
```

Persistent outputs belong on the SSD:

```text
dataset:     /mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization
assets:      /mnt/ssd1/szhao/DexTool/openpi_assets
checkpoints: /mnt/ssd1/szhao/DexTool/openpi_checkpoints
logs:        /mnt/ssd1/szhao/DexTool/openpi_logs
cache:       /mnt/ssd1/szhao/DexTool/openpi_assets/openpi-cache
```

`assets_base_dir` and `checkpoint_base_dir` are embedded in the registered
config. Set `OPENPI_DATA_HOME` and `WANDB_DIR` before commands so downloaded
checkpoint cache and W&B files also stay on the SSD.

## 5. Reproduction commands

Run from the repository root, currently resolved as `/home/shuqi/DexCondVLA`.

### 5.1 Tests that do not download the base checkpoint

```bash
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python -m pytest -q \
  src/openpi/training/bench2dex_dataset_test.py \
  src/openpi/training/bench2dex_config_test.py \
  src/openpi/training/weight_loaders_test.py
```

Real-episode end-to-end data test:

```bash
JAX_PLATFORMS=cpu \
OPENPI_DATA_HOME=/mnt/ssd1/szhao/DexTool/openpi_assets/openpi-cache \
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python \
  scripts/test_bench2dex_data_pipeline.py \
  --episode-path /mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization/episode_000000.hdf5 \
  --batch-size 2
```

The final line must be `BENCH2DEX_DATA_PIPELINE_OK`.

### 5.2 Compute normalization statistics

This reads the local dataset and writes statistics under the SSD assets tree.
It does not require the π₀.₅ parameter checkpoint.

```bash
OPENPI_DATA_HOME=/mnt/ssd1/szhao/DexTool/openpi_assets/openpi-cache \
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python \
  scripts/compute_norm_stats.py \
  --config-name pi05_bench2dex_task26_action_expert
```

Expected output directory:

```text
/mnt/ssd1/szhao/DexTool/openpi_assets/
  pi05_bench2dex_task26_action_expert/bench2dex_task26
```

### 5.3 Training

Do not run this until the base-checkpoint download has been explicitly
approved. The first run downloads `pi05_base`; the cache is directed to the
SSD by `OPENPI_DATA_HOME`.

```bash
mkdir -p /mnt/ssd1/szhao/DexTool/openpi_logs/wandb

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
OPENPI_DATA_HOME=/mnt/ssd1/szhao/DexTool/openpi_assets/openpi-cache \
WANDB_DIR=/mnt/ssd1/szhao/DexTool/openpi_logs/wandb \
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python \
  scripts/train.py pi05_bench2dex_task26_action_expert \
  --exp-name=task26_action_expert_v1 \
  --overwrite \
  2>&1 | tee /mnt/ssd1/szhao/DexTool/openpi_logs/task26_action_expert_v1.log
```

On the currently observed RTX PRO 6000 Blackwell 96GB, use memory fraction
0.8 while other GPU-0 processes remain active. Do not silently reduce batch
size or stop training on a disk-space prediction; report the condition first.

## 6. Debugging checklist

1. Confirm the selected config prints action dimension 58, processor
   `bench2dex_hdf5`, batch size 64, and 20,000 steps.
2. If an HDF5 error occurs, run the single-episode test and verify qpos,
   commanded action, camera lengths, and joint ordering.
3. If normalization is missing, check the expected assets directory above;
   never write `norm_stats.json` into the raw dataset directory.
4. If checkpoint initialization fails, inspect the shape-mismatch log. Only
   action input/output projection tensors may be skipped.
5. If unexpected parameters are trainable, list NNX paths selected by
   `config.trainable_filter`; every path must belong to `llm *_1`,
   `action_in_proj`, or `action_out_proj`.
6. If GPU memory is insufficient, first identify other GPU-0 processes and
   report them. The approved recipe remains batch 64 unless explicitly
   changed.
7. All checkpoint, cache, asset, W&B, and console log paths must remain under
   `/mnt/ssd1/szhao/DexTool`.

## 7. Handoff naming convention

Create a new handoff file instead of overwriting an older one. Use local
Pacific time and the exact pattern:

```text
CURRENT_TRAINING_HANDOFF_MMDDHHMM.md
```

For example, October 6 at 21:34 becomes
`CURRENT_TRAINING_HANDOFF_10062134.md`. Each handoff must capture the live
process/session, latest complete checkpoint, latest metrics, GPU/disk status,
W&B URL, recovery commands, and any interruption since the previous handoff.
