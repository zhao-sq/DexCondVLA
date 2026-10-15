# Task26 Dual-Expert Hand DiT v1: Implemented Method

> **Version status:** This document is the authoritative description of the
> first implemented and evaluated method, **Hand-DiT v1**. It describes the
> code and checkpoint exactly as evaluated at step 18,000; later fixes or v2
> changes must not be presented as part of this result.

The evaluated v1 variant is named **predicted-condition, stop-gradient,
independent-shared Hand-DiT**. It is intentionally a first-version experiment,
not the final recommended architecture.

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
  (seven joints plus the predicted activity value). The supervision label is
  binary; the exact v1 train/inference conversion is recorded below.
- Vision condition: frozen DINOv3 ViT-B/16 patch tokens from the corresponding
  wrist camera; CLS and four storage tokens are excluded.
- Hand state: the current 22 joint positions for the corresponding hand.
- Condition fusion: concatenated arm, spatial vision, and state tokens, read by
  DiT cross-attention.
- Hand generation: linear conditional flow matching, integrated from `t=0`
  Gaussian noise to `t=1` hand actions with Euler steps.
- Gradient boundary: the arm condition is detached inside the condition
  encoder. DINOv3 is permanently frozen.

The formal v1 run does **not** use ground-truth arm chunks as Hand-DiT
conditions. The VLA and Hand-DiT are trained in one coordinated loop on the
same batch. Hand-DiT receives the VLA's single-forward denoised arm/activity
prediction, copied across the JAX-to-PyTorch boundary and detached. Therefore
the hand loss trains Hand-DiT only; the VLA action expert is trained by its own
flow-matching loss. The separate Stage-1 entry point with ground-truth arm
conditions remains available only as an ablation.

## Exact v1 method

### Arm/activity expert

- Backbone: official pi0.5 model initialized from `pi05_base`.
- Frozen: SigLIP and the Gemma/VLM pipeline.
- Trainable: the pi0.5 action expert and action projections selected by the
  `pi05_bench2dex_task26_dual_expert_arm` freeze filter.
- Network action width: the official 32D width, preserving direct checkpoint
  compatibility.
- Physical interface: a 14D current arm state and a 50-step, 16D action chunk
  `[left_arm_7, left_active, right_arm_7, right_active]`.
- The two activity targets are the independently annotated binary left/right
  labels. They condition the hand model and are not robot joints.

### Hand expert

- One Hand-DiT is shared by both sides; left and right examples are stacked on
  the batch axis and decoded independently with learned side embeddings.
- Each side outputs a `(50, 22)` absolute hand-joint chunk.
- Conditions per side are the predicted `(50, 8)` arm/activity chunk, frozen
  DINOv3 ViT-B/16 wrist patch tokens pooled to at most 64 tokens, and the
  current 22D hand state.
- Architecture: width 512, six DiT blocks, eight attention heads, self-attention
  over hand-action time tokens, and cross-attention to the condition memory.
- Objective: linear conditional flow matching from Gaussian noise to the hand
  target. Runtime uses ten Euler integration steps.
- DINOv3 is frozen and Hand-DiT is initialized from scratch.

### Joint optimization

- A single dataset batch supervises both experts.
- VLA optimizer: the config's AdamW/cosine schedule, peak LR `5e-5`, 1,000-step
  warmup, final LR `5e-6`, EMA `0.99`, gradient clip `1.0`.
- Hand optimizer: AdamW, LR `1e-4`, weight decay `1e-4`, gradient clip `1.0`.
- Formal batch size: 64; planned length: 20,000 steps.
- Hand-to-VLA gradient is stopped. The VLA never receives the hand loss.
- Validation checkpoint ranking is the unweighted sum of fixed-batch VLA and
  Hand-DiT flow losses.

### Runtime composition

The VLA first samples its 16D physical chunk. Hand-DiT then samples both 22D
hand chunks from their side-specific conditions. `BimanualActionLayout`
scatters the 14 arm joints and 44 hand joints back into Task26's interleaved
58D action order. The activity values remain internal conditions.

## Evaluated v1 checkpoint and result

- Experiment: `task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520`.
- Selected checkpoint: step 18,000. Step 19,000 was incomplete after its Orbax
  save stalled and was not used.
- Fixed validation losses at step 18,000: VLA `0.0014852549647912383`, Hand-DiT
  `0.007435241714119911`, combined `0.00892049667891115`.
- Bench2Dex Task26 evaluation: 100 rollouts, 25 in each of `none`, `cov`, `inv`,
  and `inv_cov`.
- Official success: `0/100` (`0%`).
- Relaxed success without per-object upright checks: `10/100` (`10%`), broken
  down as `none=4/25`, `cov=2/25`, `inv=4/25`, `inv_cov=0/25`.
- The corresponding full-action baseline scored `13/100` (`13%`) under the
  same relaxed criterion.

## Known v1 behavior retained for reproducibility

These points describe the evaluated implementation; they are candidates for a
later version, not silent changes to v1:

1. During joint training, predicted activity conditions are unnormalized and
   clamped to `[0, 1]` but remain continuous. During deployed inference,
   `Bench2DexArmOutputs` thresholds activity at `0.5` before Hand-DiT sees it.
2. Training conditions use the VLA's single-forward clean-action estimate from
   its noisy training forward pass. Deployment conditions use the VLA's full
   iterative flow sample.
3. The RPC reset path resets the JAX VLA RNG but does not separately reset the
   PyTorch RNG used to initialize Hand-DiT sampling noise.
4. Left and right hands have no cross-hand attention, and the hand loss cannot
   correct the VLA arm plan because of the stop-gradient boundary.
5. Hand-DiT has no direct language, overhead-camera, or frozen VLM-token
   condition in v1.

## v1 source map

- `src/openpi/models_pytorch/hand_dit.py`: Hand-DiT architecture, loss, sampler.
- `src/openpi/models_pytorch/dinov3_wrist_encoder.py`: frozen wrist encoder.
- `src/openpi/policies/dexterous_action_layout.py`: exact Task26 58D layout.
- `src/openpi/policies/dual_expert_hand_policy.py`: composed runtime policy.
- `src/openpi/policies/bench2dex_policy.py`: 16D arm/activity transforms.
- `src/openpi/training/dual_expert_joint.py`: joint-training conversions and
  stop-gradient boundary.
- `scripts/train_task26_dual_expert_joint.py`: formal joint trainer.
- `scripts/serve_bench2dex_rpc_policy.py`: evaluation policy server.
- `scripts/run_task26_dual_expert_joint_remote.sh`: formal v1 launch settings.
- `scripts/queue_task26_dual_expert_full_remote_evaluation.sh`: complete remote
  evaluation/export/rescore pipeline.

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
