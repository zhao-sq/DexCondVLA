#!/bin/bash
set -euo pipefail

: "${EXPERIMENT_NAME:?EXPERIMENT_NAME must be set}"

cd /workspace/DexCondVLA
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export OPENPI_DATA_HOME=/workspace/DexTool/openpi_assets/openpi-cache
export OPENPI_CHECKPOINT_USE_OCDBT=0

echo "interpreter=/workspace/DexCondVLA/.venv/bin/python"
echo "experiment=${EXPERIMENT_NAME}"

exec /workspace/DexCondVLA/.venv/bin/python scripts/train_task26_dual_expert_joint.py \
  --experiment-name "${EXPERIMENT_NAME}" \
  --dataset-dir /workspace/DexTool/bench2dex/task26/replay-generalization \
  --assets-base-dir /workspace/DexTool/openpi_assets/pi05_bench2dex_task26_dual_expert_arm \
  --checkpoint-base-dir /workspace/DexTool/ckp \
  --hand-norm-stats /workspace/DexTool/openpi_assets/pi05_bench2dex_task26_action_expert/bench2dex_task26/norm_stats.json \
  --dinov3-repo /workspace/DexTool/dinov3 \
  --dinov3-checkpoint /workspace/DexTool/assets/dinov3_vitb16.pth \
  --batch-size 32 \
  --num-workers 4 \
  --steps 20000 \
  --save-every 1000 \
  --log-every 20 \
  --hand-learning-rate 1e-4 \
  --adapter-learning-rate 2e-5 \
  --adapter-vla-loss-weight 1.0 \
  --vlm-conditioned-v2 \
  --validation-fraction 0.02 \
  --validation-batches 2 \
  --seed 42
