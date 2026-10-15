# Arm-Conditioned Hand DiT: Policy Design Specification

**Status:** Historical design specification. The first implemented and
evaluated method is frozen as **Hand-DiT v1** and documented exactly in
`dual_expert_hand_dit_implementation.md`. Future v2 work must preserve that
version boundary rather than silently changing the v1 result.
**Goal:** Efficient dexterous manipulation finetuning by separating arm and hand action modeling while preserving fine-grained local visual information.

## 1. Motivation and Scope

The current baseline finetunes only the action expert of **π₀.₅**, predicting concatenated arm and dexterous-hand actions. The proposed design separates the two outputs:

- **Arm Expert:** Reuse the pretrained π₀.₅ architecture and frozen VLM context to predict a future arm-action chunk.
- **Hand Expert:** A lightweight, independent **flow-matching DiT** conditioned on the predicted arm-action chunk, wrist-camera features, and current hand state.

The Hand Expert **does not directly consume frozen VLM features in v0.1**. This isolates the hypothesis that local visual features and arm-motion context can be more useful for finger-level control. Later ablations can add VLM features.

### Design principles

1. Preserve the **temporal structure** of arm actions, rather than pooling an entire chunk into one vector.
2. Preserve the **spatial structure** of wrist-camera vision via patch tokens, rather than using only a CLS token.
3. Make local hand observation encoding modular and independent of the arm/VLM branch.
4. Keep gradients from the hand loss out of the Arm Expert in the first version.
5. Establish a minimal working baseline before adding tactile sensing, additional modalities, or bidirectional feedback.
6. For bimanual operation, **share one Hand DiT across left and right hands** in v0.2, but predict each hand independently conditioned on its corresponding arm and wrist inputs.

## 2. System Architecture

```text
  Global camera(s) + language + arm state
                    |
          Frozen π₀.₅ VLM
                    |
              Arm Expert
                    |
       Predicted arm action chunk (B, H, D_a)
                    |
        Linear projection + temporal positions
                    |                       Wrist camera
                    |                            |
                    |                    Frozen DINOv3 ViT-S
                    |                            |
                    |                 Patch tokens + projection
                    |                            |
                    |          Current hand joint state
                    |                            |
                    |                       State MLP
                    |                            |
                    +-------------+--------------+
                                  |
                         Condition memory C
                                  |
  Noisy hand action (B, H, D_h) -> Hand DiT (6 blocks)
                      + flow time -> self-attention
                                  -> cross-attention to C
                                  -> feedforward / adaLN
                                  |
                       Predicted hand flow velocity
                                  |
                       Flow ODE sampling
                                  |
                      Hand action chunk (B, H, D_h)
```

For bimanual tasks, the diagram represents **one pass per hand** through the same shared-weight Hand DiT. Each pass uses its own arm chunk, wrist image, and hand state, plus a side identifier. This does not imply left/right hand attention or coordination inside the Hand DiT.

The Arm Expert and Hand Expert have **separate trainable action-model parameters**. They do not need the same Transformer depth, hidden width, or vision-conditioning mechanism.

For Task26, the Arm Expert consumes the same 14D arm-joint state and predicts a
16D physical action in `[left arm 7, left_gripper_active, right arm 7,
right_gripper_active]` order. The two binary activity annotations are inferred
and supervised independently for the left and right hands. The π₀.₅ network itself retains the official `action_dim=32`:
the input pipeline pads these physical interfaces to 32D and the output
pipeline crops back to 16D. This preserves direct compatibility with all tensor
shapes in the official `pi05_base` checkpoint. The Hand Expert consumes each
side's full 8D prediction as an arm/activity condition, supplies the 44 hand
channels, and runtime composition restores Task26's native 58D order.

### 2.1 Task26 VLA training contract

The Task26 Arm Expert is trained with the following fixed interface:

| Item | Shape | Semantics |
|---|---:|---|
| Current arm state | `(B, 14)` | Left 7 arm joints followed by right 7 arm joints |
| Image observation | Three RGB views | Overhead, left wrist, and right wrist |
| Language | One instruction per sample | Standard π₀.₅ prompt input |
| Physical action target | `(B, 50, 16)` | `[left joints 7, left_active, right joints 7, right_active]` |
| Internal π₀.₅ action tensor | `(B, 50, 32)` | Physical target padded with 16 zeros for checkpoint compatibility |

`left_active` and `right_active` come from
`/labels/left_gripper_active` and `/labels/right_gripper_active` respectively.
They are independently supervised binary channels; neither is copied from the
other side. At inference, the two physical activity outputs are thresholded at
`0.5` to produce exact 0/1 values.

After inference, split the 16D VLA result into two temporally aligned Hand DiT
conditions:

```text
left condition:  [left joints 7, left_active]   -> (B, 50, 8)
right condition: [right joints 7, right_active] -> (B, 50, 8)
```

The activity values are conditioning signals only. Runtime scatters the 14 arm
joint predictions into the native action layout and combines them with the 44
Hand DiT joint predictions, so the environment still receives `(B, 50, 58)`.

## 3. Tensor Interfaces

Let:

- `B`: batch size
- `H`: predicted action horizon, shared initially between the two experts
- `D_a`: per-side arm/activity condition dimension (8 for Task26)
- `D_h`: hand action dimension (hardware dependent)
- `N_v`: number of retained wrist-image patch tokens
- `d`: Hand DiT model dimension, initially 512

| Signal | Shape | Processing |
|---|---|---|
| Predicted arm chunk `A_hat` | `(B, H, D_a)` | Linear/MLP → `(B, H, d)` plus temporal position embeddings |
| Wrist image `I_wrist` | Image tensor | Frozen DINOv3 ViT-S → patch features → projection `(B, N_v, d)` |
| Current hand state `s_h` | `(B, D_s)` | MLP → `(B, 1, d)` |
| Noisy hand chunk `x_tau` | `(B, H, D_h)` | Linear → `(B, H, d)` plus action positional embeddings |
| Flow timestep `tau` | `(B,)` | Sinusoidal/MLP time embedding → DiT adaLN conditioning |
| Output hand velocity `v_hat` | `(B, H, D_h)` | Final projection |

`D_s` should initially include finger joint positions. Joint velocities can be added later if available reliably.

### 3.1 Arm tokens

Given the Arm Expert's predicted chunk:

\[
\hat A = [\hat a_t, \ldots, \hat a_{t+H-1}],\qquad \hat A\in\mathbb{R}^{B\times H\times D_a},
\]

construct arm-condition tokens:

\[
C_a = \operatorname{Linear}_{D_a\to d}(\operatorname{stopgrad}(\hat A)) + P_{\mathrm{time}} + E_{\mathrm{arm}}.
\]

The time indices must remain aligned with the hand action horizon. If arm and hand action frequencies differ in the future, explicitly resample or time-align arm tokens first.

### 3.2 Wrist-vision tokens

Use a **frozen DINOv3 ViT-S** encoder on the wrist-camera observation. Retain its **patch tokens**, not just the CLS token:

\[
Z_v = \operatorname{DINOv3}_{\mathrm{patch}}(I_{\mathrm{wrist}}),\qquad
C_v = \operatorname{Linear}(\operatorname{Pool}_{\mathrm{spatial}}(Z_v)) + E_{\mathrm{vision}}.
\]

Start with approximately **16–64 pooled spatial tokens**, then compare with full patch tokens if needed. Spatial pooling is a compute-saving choice, not a claim that the same information is always retained. For multiple wrist cameras, add camera-identifying embeddings and concatenate their token sets.

### 3.3 Hand-state tokens

\[
C_s = \operatorname{MLP}(s_h) + E_{\mathrm{state}}.
\]

Initially use one state token. Additional proprioceptive details are optional.

### 3.4 Condition memory

Concatenate (rather than summing or globally pooling) modality tokens:

\[
C = [C_a\,;\,C_v\,;\,C_s]\in\mathbb{R}^{B\times(H+N_v+1)\times d}.
\]

Cross-attention allows each noisy hand-action token to selectively attend to arm timesteps, image patches, and hand proprioception. Use modality embeddings and appropriate padding masks.

## 4. Hand DiT Architecture

**First-version configuration:**

| Parameter | Choice |
|---|---|
| Generative model | Conditional flow matching |
| Transformer blocks | 6 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| Output | Velocity in hand-action space |
| Vision encoder | Frozen DINOv3 ViT-S |
| Arm conditioning | Full arm-action chunk as tokens |
| Hand state | MLP to 1 token |
| Conditioning fusion | Cross-attention to concatenated condition memory |
| Flow timestep | adaLN (adaptive LayerNorm) modulation |
| Frozen VLM in Hand Expert | No direct connection in v0.1 |

Each Hand DiT block performs:

1. **Self-attention** over noisy hand-action tokens (with their temporal positions).
2. **Cross-attention** from hand-action tokens to condition memory `C`.
3. **Feed-forward network**.
4. **Flow-time modulation** via adaLN for the relevant sublayers.

A possible block is:

```python
class HandDiTBlock(nn.Module):
    def forward(self, hand_tokens, condition_tokens, flow_time_embed):
        hand_tokens = self.self_attention_with_adaln(hand_tokens, flow_time_embed)
        hand_tokens = self.cross_attention_with_adaln(
            query=hand_tokens, context=condition_tokens, time=flow_time_embed
        )
        hand_tokens = self.feedforward_with_adaln(hand_tokens, flow_time_embed)
        return hand_tokens
```

The pseudocode specifies module connectivity; it is **not** a drop-in library implementation. Residual paths, attention masks, initialization, and tensor conventions should be handled explicitly.

## 5. Flow-Matching Objective

Let `x_1` denote the demonstrated hand-action chunk, and sample `x_0 ~ N(0,I)`. For `tau ~ Uniform(0,1)` choose the simple linear interpolation:

\[
x_\tau = (1-\tau)x_0+\tau x_1,\qquad
u_\tau = x_1-x_0.
\]

The Hand Expert predicts:

\[
\hat u_\theta(x_\tau,\tau,C)=\operatorname{HandDiT}(x_\tau,\tau,C).
\]

Optimize:

\[
\mathcal L_{\mathrm{hand}} =
\mathbb E_{x_0,x_1,\tau}\big[\|\hat u_\theta(x_\tau,\tau,C)-\nu_\tau\|_2^2\big].
\]

Use the same action normalization and chunk conventions as the existing baseline wherever possible. The precise interpolation schedule should match the flow convention of the chosen implementation; the linear example above is one valid starting point.

### Training the arm condition

- **Initial integration test:** use ground-truth arm chunks as a diagnostic to verify that Hand DiT can learn conditioned hand actions.
- **Actual deployment-oriented training:** condition primarily on **predicted arm chunks**, since these are what the hand policy sees at inference.
- Use `stop_gradient` / `.detach()` on arm predictions in v0.1 so the hand loss does not update the Arm Expert.
- Optionally mix GT and predicted arm chunks or add small realistic perturbations to reduce train/inference mismatch; track this as an explicit ablation.

GT-arm-conditioned results should not be reported as deployable policy performance.

## 6. Inference

At each control replanning event:

1. Encode global camera observations and language with the frozen π₀.₅ VLM.
2. Sample/predict the Arm Expert action chunk `A_hat`.
3. Encode the wrist image with frozen DINOv3; embed current hand state.
4. Build the Hand Expert condition memory from `A_hat`, image-patch features, and hand state.
5. Initialize a Gaussian hand-action chunk and numerically integrate the learned flow from `tau=0` to `tau=1`.
6. Execute synchronized arm and hand action prefixes, then replan.

**Caveat:** This arm-then-hand generation is sequential and can increase inference latency. Measure both expert sampling times and full closed-loop latency. Also measure errors caused by failed arm-hand coupling rather than assuming the hand can always compensate for arm predictions.

## 7. Minimal Ablations

Keep training steps, data volume, action normalization, and (where possible) compute budgets matched.

| Experiment | Hand condition | Question |
|---|---|---|
| A0 | Existing single π₀.₅ joint-action expert | Established baseline |
| A1 | Arm chunk + hand state | Is arm conditioning enough without local vision? |
| A2 | Wrist patch tokens + hand state | How much does local vision help alone? |
| A3 | Arm chunk + wrist patches + hand state | Does the combined design help? |
| A4 (later) | A3 + direct VLM feature | Does frozen global VLM context provide additional value? |
| A5 (later) | A3 with pooled/CLS wrist feature | Are spatial patch tokens necessary? |

**Control for model capacity:** a dual-expert solution may have more total parameters or FLOPs than a single expert. Track total and trainable parameter counts, accelerator memory, steps/sec, inference latency, success rate, and dexterous-data efficiency. If necessary, add a parameter-matched dual-expert configuration as a separate architecture ablation.

## 8. Suggested Implementation Modules

```text
policy/
  arm_expert.py              # Existing pi0.5 action expert wrapper
  hand_dit.py                # Conditional hand flow-matching Transformer
  hand_conditioning.py       # Arm projection, modality tokens, condition memory
  wrist_encoder.py           # DINOv3 ViT-S patch feature extraction
  hand_state_encoder.py      # Hand joint state MLP
  dual_expert_policy.py      # End-to-end arm-first, hand-second orchestration
  losses.py                  # Flow-matching losses
  configs/dual_expert.yaml   # Hyperparameters and modality flags
```

```python
# Conceptual interface (not executable code)
arm_actions = arm_expert(global_obs, language, arm_state)
arm_tokens = arm_projection(arm_actions.detach())
vision_tokens = wrist_encoder(wrist_images)  # frozen backbone
state_tokens = hand_state_encoder(hand_state)
condition_tokens = concatenate([arm_tokens, vision_tokens, state_tokens], dim=1)

hand_velocity = hand_dit(noisy_hand_chunk, flow_time, condition_tokens)
```

The encoder wrapper should also support **precomputed visual features** so representation experiments do not require changes to the Hand DiT itself. For experiments evaluating a trainable visual backbone, switch to raw images and enable encoder gradients explicitly.

## 9. Bimanual Hand Policy: Shared Weights, Separate Predictions (v0.2)

**Default choice:** Use **one shared Hand DiT** for both hands, invoked separately for the left and right side. Do not instantiate two independent sets of Hand DiT weights. This reduces parameters and lets data from both hands update the same hand-control model. Shared weights do **not** by themselves establish bimanual coordination.

```text
Left arm chunk  + Left wrist image  + Left hand state  + LEFT embedding
                              |
                              v
                      Shared Hand DiT ----> Left hand action chunk
                              ^
                              |
Right arm chunk + Right wrist image + Right hand state + RIGHT embedding

  (same weights, separate passes; no direct left-right attention in v0.2)
```

### 9.1 Inputs and outputs

For each side `k in {left, right}`, apply exactly the same conditioning pipeline:

- `A_k`: corresponding arm-action chunk `(B, H, D_a)`; align the two sides to the same physical timestamps.
- `I_k`: corresponding wrist-camera image, encoded with a **shared frozen DINOv3** and a shared projection head initially.
- `s_k`: corresponding finger joint state, projected with a shared state encoder.
- `e_k`: learned left/right side embedding, added to the condition tokens (or supplied as a distinct side token).
- `x_tau,k`: noisy **hand-only** action chunk `(B, H, D_h)`; output velocity `v_hat_k` of the same shape.

```python
# Conceptual pseudocode: one shared module, twice per batch
left_cond = encode_hand_condition(left_arm.detach(), left_wrist, left_state, side='left')
right_cond = encode_hand_condition(right_arm.detach(), right_wrist, right_state, side='right')

v_left = shared_hand_dit(noisy_left_hand, flow_time, left_cond)
v_right = shared_hand_dit(noisy_right_hand, flow_time, right_cond)
loss = 0.5 * (flow_matching_loss(v_left, target_left) +
              flow_matching_loss(v_right, target_right))
```

Alternatively, stack left/right examples along the batch dimension and make **one batched forward pass**; this is still two independent predictions with shared weights, not joint action modeling.

### 9.2 Important implementation details

1. **Coordinate conventions:** express each wrist and arm condition in a consistently defined hand/wrist-local frame where possible. Otherwise a shared model may waste capacity learning left/right geometric conventions.
2. **Hand morphology:** if the left and right hands have mirrored joint definitions, define a clear joint-index mapping and appropriate sign / frame conventions. Do not assume naive copying of actions between hands is valid.
3. **Side identity:** start with learned `LEFT`/`RIGHT` embeddings; ablate them later if the coordinate transforms already normalize handedness.
4. **Missing hand:** allow a side-validity mask so single-handed demonstrations can train the same shared model without inventing labels for the absent hand.
5. **Cross-hand coordination:** in v0.2, coordination is **indirect**, via correlated arm plans and visible observations. This can fail for object handovers or simultaneous force-sensitive grasps.
6. **Loss accounting:** sum or average over the available hands, with mask-aware normalization, so bimanual examples do not accidentally receive twice the weight of single-hand examples.

### 9.3 Future optional variant: Joint bimanual decoding

If shared independent predictions fail on strongly coupled bimanual tasks, keep the same shared backbone but concatenate both hands' **noisy action tokens** and both sides' condition tokens in one DiT forward pass; use side embeddings and output two hand chunks together. This permits direct cross-hand attention, but increases sequence length and makes the ablation less clean. Do **not** implement this until a measurable coordination failure justifies it.

### 9.4 Minimum bimanual ablation

| Variant | Hand-model weights | Interaction between hands | Main question |
|---|---|---|---|
| B0 (default) | Shared | None directly; separate inference | Best low-complexity baseline? |
| B1 | Separate left/right | None directly | Is side-specific capacity needed? |
| B2 (later) | Shared | Joint attention over both hand chunks | Does explicit coordination help? |

Report **total parameters, training time, per-hand success, bimanual task success, and inference latency**. Sharing weights saves parameters, but does not necessarily halve execution FLOPs when predicting both hands.

## 10. Risks and Immediate Validation Checks

1. **Mismatch in arm-conditioning distribution:** predicted versus GT arm chunks can differ. Evaluate with predicted chunks early.
2. **Local visual observability:** a wrist image may not reveal occluded contacts or forces. Tactile or depth may become necessary for specific tasks, but are outside v0.1.
3. **Task semantics:** removing direct VLM conditioning may hurt task-dependent finger behavior. Ablate optional global semantic tokens later.
4. **Arm-hand coupling:** strictly unidirectional conditioning cannot correct a bad arm plan. Measure this failure mode.
5. **Capacity and compute:** the lightweight Hand DiT is additional to the Arm Expert and the wrist encoder; report whole-system costs, not only the Hand DiT parameter count.
6. **Flow and control timing:** align action chunk horizons and execution rates so predicted arm motion and hand control refer to the same future times.

## 11. Immediate Implementation Plan

**Stage 1:** Implement Hand DiT and train with GT arm chunk + precomputed frozen DINOv3 wrist patch features + hand state. Confirm tensor shapes, conditioning masks, loss decrease, and sample quality.

**Stage 2:** Replace GT arm chunk with predicted Arm Expert output (detached). Run fully deployable closed-loop evaluation.

### Implemented Stage-2 joint trainer (2026-10-12)

`scripts/train_task26_dual_expert_joint.py` implements Stage 2 as one coordinated
training loop. A single raw Task26 sample batch is transformed into both the
π₀.₅ batch and the Hand-DiT supervision batch. The π₀.₅ flow-matching forward
returns both its normal training loss and the clean-action estimate
`x0_hat = x_t - t * v_theta(x_t, t)`, avoiding a separate multi-step ODE solve
inside every training iteration.

Only the first physical 16 channels are used, ordered as
`[left_joint_7, left_active, right_joint_7, right_active]`. They are unnormalized,
split into two 8D conditions, normalized to the Hand-DiT convention, and copied
across the JAX-to-PyTorch boundary. The resulting tensor is detached again in
the Hand condition encoder. Therefore the Hand loss updates only Hand DiT; the
VLA Action Expert is updated only by its own flow loss. VLM/SigLIP and DINOv3
remain frozen.

The joint checkpoint directory stores the Orbax VLA state, `hand_dit.pt`, and
validation metrics together under each numeric step. Validation uses fixed
batches/noise and ranks checkpoints by
`vla_validation_loss + hand_validation_loss`. Retention deletes every numeric
step except `LATEST` and `BEST` (one directory when both pointers coincide).

**Stage 3:** Compare Arm+State, Vision+State, and Arm+Vision+State conditions against the existing π₀.₅ joint-action baseline.

**Stage 4:** Only after a stable result, investigate direct VLM-feature conditioning, alternative patch-token selection, parameter-matched variants, and partial/incorrect hand supervision.

---

**One-line summary:** Retain π₀.₅'s global VLM-conditioned Arm Expert, and add a lightweight local-vision-dominant Hand DiT that cross-attends to **temporally aligned arm-action tokens, spatial wrist-image patch tokens, and hand-state tokens**, without direct VLM conditioning in the first version. For bimanual tasks, **share Hand DiT weights across left/right hands, but predict each hand separately using side-specific inputs and embeddings**.
