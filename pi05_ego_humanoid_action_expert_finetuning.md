# π₀.₅ Ego Humanoid Action-Expert-Only Fine-tuning Guide

## 1. Dataset download status

The five selected Ego Humanoid tasks have been downloaded to:

```text
/mnt/ssd1/szhao/DexTool/ego_humanoid_benchmark
```

Each task is stored in a separate directory. The downloaded files were verified by task-level file count and exact byte total, and no `.part` temporary files remain.

| Task | Episodes | Size |
|---|---:|---:|
| Open-Drawer | 100 | 4.95 GB |
| Open-Laptop | 100 | 4.69 GB |
| Sort-Cans | 101 | 23.35 GB |
| Insert-Cans | 100 | 12.41 GB |
| Unload-Cans | 100 | 10.38 GB |
| **Total** | **501** | **55.78 GB / 51.95 GiB** |

## 2. Recommended training implementation

Use the JAX training entry point:

```text
scripts/train.py
```

The JAX implementation applies `TrainConfig.freeze_filter` when initializing the optimizer and computing gradients. The current PyTorch entry point, `scripts/train_pytorch.py`, passes all `model.parameters()` to AdamW and does not implement the requested action-expert-only freeze policy.

The target recipe is the final table in `pi05_finetune_recipe_selection.md`:

| Setting | Value |
|---|---:|
| Trainable parameters | Action expert + action input/output projections |
| Global batch size | 64 |
| Peak learning rate | `5e-5` |
| Warmup | 1,000 steps |
| Training length | 15,000–20,000 steps |
| EMA | 0.99 |
| Gradient clipping | 1.0 |
| Weight decay | Effectively zero / repository default |
| Precision | bf16 |

For a first run over all five tasks, use 20,000 steps.

### 2.1 Proposed training configuration

Add a new configuration similar to the following:

```python
TrainConfig(
    name="pi05_ego_humanoid_action_expert",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=50,
        action_horizon=50,
    ),
    data=LeRobotEgoHumanoidDataConfig(
        repo_id="ego_humanoid_5tasks",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=64,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=20_000,
        decay_lr=5e-6,
    ),
    optimizer=_optimizer.AdamW(
        weight_decay=1e-10,
        clip_gradient_norm=1.0,
    ),
    ema_decay=0.99,
    num_train_steps=20_000,
    freeze_filter=nnx.Not(
        nnx_utils.PathRegex(
            r"(.*llm.*_1.*|action_in_proj.*|action_out_proj.*)"
        )
    ),
    weight_loader=ShapeCompatibleCheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
)
```

`freeze_filter` identifies frozen parameters. Wrapping the trainable-path regex in `nnx.Not(...)` therefore freezes everything except:

- The second Gemma expert under paths matching `.*llm.*_1.*`.
- `action_in_proj`.
- `action_out_proj`.

The existing `Pi0Config.get_freeze_filter()` is designed primarily around LoRA configurations. With the regular `gemma_300m` action expert it returns `nnx.Nothing`, which does not freeze the backbone, so it cannot be used directly for this setting.

Before a long run, instantiate the model and print every parameter selected by `config.trainable_filter`. Assert that no SigLIP or Gemma 2B backbone parameters are selected.

### 2.2 Optional timestep-MLP variant

The strict interpretation of the recipe trains only the expert and action projections. π₀.₅ also contains two action-path timestep layers:

```text
time_mlp_in
time_mlp_out
```

They generate the conditioning used by the action expert's adaptive RMSNorm. A second experiment may include them by extending the trainable regex:

```python
r"(.*llm.*_1.*|action_in_proj.*|action_out_proj.*|time_mlp_in.*|time_mlp_out.*)"
```

For exact reproduction of the table, leave these two layers frozen in the first experiment.

## 3. The 50-dimensional action-space requirement

The repository's default π₀.₅ configuration uses:

```python
action_dim = 32
action_horizon = 50
```

Ego Humanoid uses 50-dimensional `qpos` and action vectors. `PadStatesAndActions` can zero-pad vectors that are shorter than the model dimension, but it is not a principled 50-to-32 action-space conversion. The model should therefore be configured as:

```python
Pi0Config(
    pi05=True,
    action_dim=50,
    action_horizon=50,
)
```

This changes the shapes of two layers relative to the released `pi05_base` checkpoint:

```text
action_in_proj:  Linear(50, 1024)
action_out_proj: Linear(1024, 50)
```

### 3.1 Checkpoint loading

The current `CheckpointWeightLoader` merges checkpoint parameters by path without first filtering shape-incompatible tensors. The training script subsequently performs exact shape validation. Loading the 32-dimensional `pi05_base` projections into the 50-dimensional model will therefore fail.

Implement a shape-compatible checkpoint loader with the following behavior:

1. Load a checkpoint tensor only when its path exists in the target model and its shape matches.
2. Keep the target model's initialized tensor when the shape differs.
3. Log every skipped tensor and the source and target shapes.
4. Assert that the only expected skipped tensors are the kernels and biases of `action_in_proj` and `action_out_proj`.

This retains pretrained weights for SigLIP, Gemma 2B, and the complete π₀.₅ action expert while reinitializing only the two 50-dimensional projection layers.

## 4. Dataset conversion and transforms

The raw HDF5 files should be converted to LeRobot format. The existing ALOHA converter can be used as a structural reference because it already reads HDF5 state, action, and image arrays.

Recommended mapping:

```text
observation.state       <- /observations/qpos
action                  <- /action
observation.images.main <- /observations/images/main
task                    <- natural-language instruction for the episode
```

Keep the five raw task directories separate. For multi-task fine-tuning, convert them into one LeRobot dataset and retain the task identity as the per-episode prompt. If separate checkpoints are desired, create five LeRobot datasets and five otherwise-identical configs.

Example prompts:

| Task | Prompt |
|---|---|
| Open-Drawer | `open the drawer` |
| Open-Laptop | `open the laptop` |
| Sort-Cans | `sort the cans` |
| Insert-Cans | `insert the cans into the box` |
| Unload-Cans | `unload the cans from the box` |

Use the benchmark's precise task wording if it is available in its metadata.

### 4.1 Ego Humanoid policy transforms

Do not reuse `AlohaInputs` and `AlohaOutputs` unchanged. They contain ALOHA-specific joint sign changes, gripper conversions, and an output truncation to 14 dimensions.

Implement `EgoHumanoidInputs` and `EgoHumanoidOutputs` that:

- Preserve the meaning and order of all 50 H1 state and action dimensions.
- Convert image arrays to the expected `H × W × C` uint8 format when needed.
- Return all 50 predicted action dimensions.
- Map the main camera to `base_0_rgb`.
- Do not apply ALOHA joint or gripper conversions.

π₀.₅ declares three image slots. If the dataset contains only the main camera, use:

```text
main image -> base_0_rgb,       mask=True
black image -> left_wrist_0_rgb,  mask=False
black image -> right_wrist_0_rgb, mask=False
```

If wrist-camera streams actually exist in the HDF5 files, map them instead of generating black placeholders.

### 4.2 Training and validation split

The downloaded episodes do not provide a ready-made training/inference split. Split at the episode level, never at the frame level. A reasonable initial split is to reserve approximately 10 episodes per task for validation or simulator evaluation and train on the remainder.

Fit normalization statistics using the training split only. Do not include held-out episodes.

## 5. Commands

After conversion and after registering the new config, compute normalization statistics:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_ego_humanoid_action_expert
```

Start JAX training:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_ego_humanoid_action_expert \
  --exp-name=ego_humanoid_5tasks_action_expert \
  --overwrite
```

For a restart after interruption, use the repository's resume option rather than `--overwrite`.

## 6. Action expert architecture

The main flow is:

```text
images + prompt + discretized state
                │
                ▼
      frozen SigLIP + Gemma 2B
                │
       visual-language prefix
                │
                ▼
noisy action -> Linear(action_dim, 1024)
                │
                ▼
       18-layer Gemma action expert
                │
                ▼
Linear(1024, action_dim) -> predicted action flow
```

The action expert is not a separate conventional cross-attention network. Its action tokens and the frozen visual-language prefix participate in the same attention computation, with masks controlling which tokens may attend to which context.

The `gemma_300m` expert configuration is:

| Component | Value |
|---|---:|
| Transformer depth | 18 |
| Hidden width | 1024 |
| MLP hidden dimension | 4096 |
| Query heads | 8 |
| KV heads | 1 |
| Head dimension | 256 |
| Independent token embedding | No |
| Independent language-model head | No |

Training uses flow matching. For clean actions `a`, sampled noise `ε`, and flow time `t`:

```text
x_t = t ε + (1 - t) a
u_t = ε - a
```

The action expert predicts the flow vector `u_t`. In π₀.₅, the timestep is embedded by two `1024 → 1024` MLP layers and injected into the expert through adaptive RMSNorm.

## 7. Parameter count and memory implications

The repository labels the regular `gemma_300m` decoder core as approximately 311M parameters. π₀.₅ replaces the expert's ordinary norm scales with timestep-conditioned adaptive RMSNorm projections, increasing the actual expert parameter count.

Approximate counts are:

| Component | Parameters |
|---|---:|
| π₀.₅ Gemma expert core including adaptive RMSNorm | 427,932,672 |
| 50D action input projection | 52,224 |
| 50D action output projection | 51,250 |
| **Strict recipe trainable total** | **428,036,146** |
| Two timestep MLPs, if also trained | 2,099,200 |
| **Expert + projections + timestep MLPs** | **430,135,346** |

Thus the strict action-expert-only recipe trains approximately **428 million parameters**, roughly 13% of the approximately 3.3B-parameter complete model.

Raw trainable-weight storage is approximately:

| Representation | Approximate size |
|---|---:|
| bf16 | 0.86 GB / 0.80 GiB |
| fp32 | 1.71 GB / 1.59 GiB |

Actual training memory is substantially higher because it also includes:

- Gradients.
- Adam first and second moments.
- Activations and attention intermediates.
- The frozen SigLIP and Gemma 2B weights.
- EMA parameters.

The current JAX training state initializes EMA from the complete parameter tree, including frozen parameters. Consequently, freezing the backbone greatly reduces gradient and optimizer-state cost, but does not remove the backbone or its EMA copy from memory. Batch size 64 may still require multiple GPUs and FSDP, depending on GPU memory and sequence/image settings.

## 8. Implementation checklist

- [ ] Convert the five HDF5 task directories into LeRobot format.
- [ ] Preserve 50-dimensional state and action ordering.
- [ ] Add `EgoHumanoidInputs` and `EgoHumanoidOutputs`.
- [ ] Add `LeRobotEgoHumanoidDataConfig`.
- [ ] Add a shape-compatible π₀.₅ checkpoint loader.
- [ ] Add `pi05_ego_humanoid_action_expert` with `action_dim=50`.
- [ ] Audit the exact parameter paths selected by `trainable_filter`.
- [ ] Create an episode-level train/validation split.
- [ ] Compute normalization statistics from the training split only.
- [ ] Run a short data-loader and one-step training smoke test.
- [ ] Start the 15k–20k step training run.
