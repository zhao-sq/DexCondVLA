# Dual-Expert Hand-DiT v2: hard activity and VLM feature adapter

## Scope

This is the second implementation of the Task26 dual-expert policy. It keeps
the v1 π0.5 arm Action Expert and Hand-DiT decomposition, while addressing two
specific v1 limitations:

1. activity conditioning is identically hard at train and inference time;
2. Hand-DiT receives semantic context from the frozen VLM final layer and can
   update a shared adapter on the deployed VLA action path.

The v1 implementation and checkpoints remain loadable. V2 checkpoints use
format version 2 and bundle both Hand-DiT and the new adapter.

## Data flow

The π0.5 Action Expert retains its official internal 32D width. Its physical
Task26 output is the first 16 dimensions of each 50-step chunk:

`[left arm joints 7, left_active, right arm joints 7, right_active]`.

During the same π0.5 training forward, the final image/language prefix tokens
from the frozen VLM are masked-mean pooled to one 2048D feature per example.
No extra VLM forward is needed during training.

The new `VLMActionAdapter` maps that feature to a shared 512D representation
and emits:

- eight 512D semantic tokens used by both left and right Hand-DiT examples;
- a residual correction for every step of the normalized 16D VLA chunk.

The final residual layer is initialized to zero, so training starts from the
unmodified π0.5 arm/activity prediction. The corrected arm joints become the
Hand-DiT arm condition. Each activity value is first converted to physical
units and then hard-thresholded with `active = value >= 0.5`; Hand-DiT never
sees a fractional activity value.

For each original batch example, left and right hands are stacked along the
batch dimension. The shared six-block, width-512 Hand-DiT receives:

- corrected `(50, 8)` side-specific arm joints plus hard activity;
- frozen DINOv3 ViT-B/16 wrist patch tokens;
- normalized current 22D hand state;
- learned left/right side embedding;
- eight shared VLM semantic tokens.

It predicts only the corresponding `(50, 22)` hand-joint chunk.

## Losses and gradient boundary

Three losses are computed from the same batch:

- native π0.5 flow-matching loss, which fine-tunes the JAX Action Expert and
  action input/output projections;
- Hand-DiT conditional flow-matching loss;
- adapter VLA reconstruction MSE between corrected and ground-truth normalized
  16D action chunks, with weight 1.0.

Hand loss backpropagates through both the semantic tokens and corrected arm
joints into the shared adapter. Consequently it can modify the deployed VLA
action correction. The framework boundary remains explicit: it does not
backpropagate into the JAX Action Expert or the frozen VLM. Those Action Expert
parameters continue to train from the native VLA loss. This is a real coupled
post-VLA action path without an unsafe or misleading NumPy/JAX gradient claim.

## Inference

V2 checkpoint loading reconstructs the adapter automatically. The policy
extracts the VLM final-layer feature through the same normal input transforms,
normalizes the 16D VLA result with the training q01/q99 statistics stored in
the checkpoint, applies the adapter, restores physical units, and hardens both
activity channels. Corrected arms and Hand-DiT hands are then scattered into
the 58D Task26 runtime layout.

The initial implementation uses a separate prefix-only VLM call for inference
feature extraction. This favors correctness and v1 compatibility; a future
optimization can return the already-computed prefix feature from the policy's
sampling cache to remove that duplicate prefix pass without changing training
or checkpoint semantics.

## Formal run

- Experiment: `task26_dual_expert_vlm_adapter_hardact_v2_20261014_233437`
- Device: physical GPU 1, exposed as `cuda:0`
- Environment: `/mnt/ssd1/szhao/DexTool/openpi_env`
- Dataset: `/mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization`
- Batch/steps: 64 / 20,000
- VLA schedule: existing Task26 π0.5 cosine schedule, peak LR `5e-5`
- Hand-DiT LR: `1e-4`
- Adapter LR: `2e-5`
- DINOv3 and VLM: frozen
- Save/validation interval: 1,000 steps
- Retention: latest plus best fixed-validation combined score
- Checkpoint root: `/mnt/ssd1/szhao/DexTool/ckp/pi05_bench2dex_task26_dual_expert_arm/task26_dual_expert_vlm_adapter_hardact_v2_20261014_233437`
- Log: `/mnt/ssd1/szhao/DexTool/openpi_logs/task26_dual_expert_vlm_adapter_hardact_v2_20261014_233437.log`
- Persistent session: `tmux` session `task26_hand_dit_v2`

Before formal launch, two-step smoke tests passed at batch 1 and batch 64. The
batch-64 losses were:

| Step | VLA | Hand | Adapter VLA |
|---:|---:|---:|---:|
| 1 | 0.156778 | 1.404863 | 0.157210 |
| 2 | 0.142524 | 1.378855 | 0.142582 |

The required automatic code snapshot was verified at
`code_snapshots/LATEST` immediately after formal launch.

### Corrected restart

The initial formal run reached step 1,000 but its first validation exposed a
two-versus-three-output `out_shardings` declaration left from v1. It exited
before saving a numeric checkpoint. The declaration was corrected and a new
validation smoke mode was added so this path is exercised before long runs.
The validation smoke passed.

The joint entry point now also runs validation unconditionally immediately
after the first optimizer step for every launch. This first validation only
prints the VLA, Hand-DiT, and adapter losses and never writes a checkpoint,
including when `save_every=1` or the run contains only one step. Later
validation and checkpoint retention continue to follow the configured save
interval. The older `--validate-smoke` argument remains accepted for launch
command compatibility but is no longer needed.

The corrected restart is
`task26_dual_expert_vlm_adapter_hardact_v2r1_20261015_121347`. GPU 1 had since
become shared with unrelated jobs occupying about 50 GiB, so batch 64 no
longer fit. A batch-32 smoke passed under XLA memory fraction 0.35; the restart
uses batch 32 with otherwise identical architecture, losses, and optimizer
settings.

## Files changed

- `src/openpi/models/pi0.py`: expose pooled final-layer VLM features for joint
  training and inference.
- `src/openpi/models_pytorch/vlm_action_adapter.py`: shared semantic/action
  adapter with a zero-initialized VLA residual head.
- `src/openpi/models_pytorch/hand_dit.py`: optional VLM token modality and an
  explicit arm-condition gradient switch.
- `src/openpi/training/dual_expert_joint.py`: hard activity conversion and a
  differentiable Torch arm-joint condition path.
- `scripts/train_task26_dual_expert_joint.py`: v2 losses, optimizer groups,
  validation, checkpoint metadata, and adapter checkpoint saving.
- `src/openpi/policies/policy.py`: VLM feature extraction through standard
  policy input transforms.
- `src/openpi/policies/dual_expert_hand_policy.py`: v2 checkpoint restore and
  runtime adapter composition.
- `scripts/serve_bench2dex_rpc_policy.py`: pass restored adapter and its
  normalization metadata into the dual-expert runtime.
- `TRAINING_CHECKPOINT_REGISTRY.md`: immutable formal-run registration.

## Environment correction

The local OpenPI virtual environment originally contained PyTorch
`2.7.1+cu126`, which has no `sm_120` kernels for this Blackwell GPU. It was
replaced in the same explicit environment with official `torch 2.7.1+cu128`
and `torchvision 0.22.1+cu128`; no system Python packages were changed.
