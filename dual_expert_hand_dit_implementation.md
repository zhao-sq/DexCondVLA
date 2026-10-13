# Task26 Dual-Expert Hand DiT: v1 Implementation

This implementation keeps the official π₀.₅ network and checkpoint at their
pretrained 32-dimensional action width. At the robot-facing boundary, the VLA
sees a physical 14D arm-joint state and predicts a physical 16D action in
`[left arm 7, left_gripper_active, right arm 7, right_gripper_active]` order.
The two activity labels come from independent side-specific annotations. The data pipeline pads
these interfaces to 32D before the model and crops output back to 16D after
inference. A separate hand policy predicts
the 22 left-hand and 22 right-hand channels, and the runtime scatters all 58
channels into Task26's native interleaved ordering.

## Implemented v1 boundary

- Arm/global policy: official-width π₀.₅ initialized directly from `pi05_base`,
  with a Task26 16D arm/activity physical output interface.
- Hand policy: one shared six-block conditional flow-matching DiT.
- Bimanual behavior: left and right are stacked on the batch dimension and
  predicted independently with shared weights and learned side embeddings.
- Arm condition: one aligned eight-value arm/activity token per future timestep
  (seven joints plus the predicted 0/1 activity flag).
- Vision condition: frozen DINOv3 ViT-B/16 patch tokens from the corresponding
  wrist camera; CLS and four storage tokens are excluded.
- Hand state: the current 22 joint positions for the corresponding hand.
- Condition fusion: concatenated arm, spatial vision, and state tokens, read by
  DiT cross-attention.
- Hand generation: linear conditional flow matching, integrated from `t=0`
  Gaussian noise to `t=1` hand actions with Euler steps.
- Gradient boundary: the arm condition is detached inside the condition
  encoder. DINOv3 is permanently frozen.

Stage-1 Hand DiT training uses ground-truth arm chunks. Deployment uses the arm
VLA's predicted chunk. This intentional mismatch must be measured before a
deployable result is claimed.

## VLA supervision contract

For every Task26 sample, π₀.₅ receives the three RGB views, language prompt,
and current 14D arm-joint state. Its physical training target has shape
`(50, 16)` and exact ordering:

```text
[left_arm_joint_0..6, left_gripper_active,
 right_arm_joint_0..6, right_gripper_active]
```

The two activity targets are read independently from
`/labels/left_gripper_active` and `/labels/right_gripper_active`. The 16D target
is zero-padded to the official 32D π₀.₅ width; inference crops back to 16D and
thresholds channels 7 and 15 at `0.5`. The Hand DiT then receives the left and
right `(50, 8)` chunks separately. Only the 14 joint channels are scattered
into the final environment action; the two activity channels remain internal
conditioning signals.

## Task26 action layout

The 58 channels are interleaved, so the implementation uses explicit indices:

- left arm: 7 channels;
- right arm: 7 channels;
- left hand: 22 channels selected by the `left_` joint names;
- right hand: 22 channels selected by the `multi_right_` joint names.

`BimanualActionLayout` validates that these four groups are disjoint and
partition all 58 channels. The final composition starts from the current 58D
robot state, scatters the VLA's 14 joint channels, and then writes the two Hand
DiT outputs. The two activity predictions condition Hand DiT but are not extra
environment joints.

## DINOv3 assets

The verified local checkpoint is:

```text
/mnt/ssd1/szhao/LeveFD/dinov3_vitb16.pth
```

The official source checkout used to construct the exact RoPE ViT-B/16 is:

```text
/mnt/ssd1/szhao/DexTool/dinov3
```

The checkout used for validation was commit
`6876159a11b4df116f30f667f8c9888617df0751`.

With the official LVD preprocessing, a 256×256 wrist image produces 256 patch
tokens of width 768. The condition encoder adaptively pools these to 64 tokens
by default. The external checkpoint and DINOv3 source are not copied into this
repository or into Hand DiT checkpoints.

## Arm VLA preparation

The arm VLA config is `pi05_bench2dex_task26_dual_expert_arm`. It keeps
`Pi0Config.action_dim=32` so `pi05_base` loads without any shape mismatch, but
computes normalization statistics over the selected 14D Task26 arm state and
the 16D arm/activity actions:

```bash
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python \
  scripts/compute_norm_stats.py \
  --config-name pi05_bench2dex_task26_dual_expert_arm
```

Before starting an arm-VLA training run, register its unique experiment name in
`TRAINING_CHECKPOINT_REGISTRY.md` as required by the repository rules.

## Hand DiT training

Before launching, add the unique experiment to
`TRAINING_CHECKPOINT_REGISTRY.md`. The training entry point refuses to start if
the name is missing, refuses to overwrite a non-empty new-run directory, logs
the interpreter, and creates the required code snapshot.

```bash
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python \
  scripts/train_task26_hand_dit.py \
  --experiment-name task26_hand_dit_v1_<unique_timestamp> \
  --output-dir /mnt/ssd1/szhao/DexTool/ckp/task26_hand_dit_v1_<unique_timestamp>
```

Default training settings match the design proposal: horizon 50, per-side
arm/activity condition dimension 8, per-side hand dimension 22, model width 512, six blocks, eight
heads, and at most 64 wrist patch tokens. The loader creates left and right
examples from each Task26 sample and performs one shared-weight forward pass.
The Hand DiT checkpoint keeps the full Task26 statistics: it selects the same
physical arm channels for conditioning and the hand channels for targets.

The following flags produce the A1/A2/A3 conditioning ablations without
changing the DiT code:

```text
--disable-arm-condition
--disable-vision-condition
--disable-state-condition
```

At least one condition must remain enabled.

## Combined inference

Add the hand checkpoint and DINOv3 paths to the existing Task26 RPC server:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
/mnt/ssd1/szhao/DexTool/openpi_env/bin/python \
  scripts/serve_bench2dex_rpc_policy.py \
  --config pi05_bench2dex_task26_dual_expert_arm \
  --checkpoint <pi05_checkpoint> \
  --hand-dit-checkpoint <hand_dit_step_checkpoint> \
  --dinov3-repo /mnt/ssd1/szhao/DexTool/dinov3 \
  --dinov3-checkpoint /mnt/ssd1/szhao/LeveFD/dinov3_vitb16.pth \
  --hand-device cuda:0 \
  --hand-flow-steps 10
```

The first runtime version deliberately accepts only the absolute-action
Task26 pipeline. State-delta and aligned-delta modes must define and train the
Hand DiT in the matching label domain before they are enabled.

JAX and PyTorch share the GPU in this process. Set an explicit JAX memory
fraction that leaves capacity for the frozen DINOv3 and Hand DiT, and measure
VLA time, total dual-expert time, and closed-loop control latency.

## Extension points reserved for v2

- `condition_on_*` switches support modality ablations and later additions.
- The DINOv3 wrapper is separate from `HandDiT`; cached/precomputed tokens can
  be passed directly during representation experiments.
- `side_valid` provides mask-aware loss for future single-hand examples.
- `bimanual_mode` is serialized in the config. v1 only permits
  `independent_shared`; a future joint decoder can add cross-hand attention
  without changing checkpoints silently.
- The checkpoint stores the full model config, normalization, step, optimizer,
  and metadata, leaving room to record predicted-arm mixing schedules or new
  condition encoders.
