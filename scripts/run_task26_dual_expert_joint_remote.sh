#!/bin/bash
set -euo pipefail

cd /workspace/DexCondVLA
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export OPENPI_DATA_HOME=/workspace/DexTool/openpi_assets/openpi-cache
export OPENPI_CHECKPOINT_USE_OCDBT=0

exec /workspace/DexCondVLA/.venv/bin/python scripts/train_task26_dual_expert_joint.py \
  --experiment-name task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520 \
  --dataset-dir /workspace/DexTool/bench2dex/task26/replay-generalization \
  --assets-base-dir /workspace/DexTool/openpi_assets \
  --checkpoint-base-dir /workspace/DexTool/ckp \
  --hand-norm-stats /workspace/DexTool/openpi_assets/pi05_bench2dex_task26_action_expert/bench2dex_task26/norm_stats.json \
  --dinov3-repo /workspace/DexTool/dinov3 \
  --dinov3-checkpoint /workspace/DexTool/assets/dinov3_vitb16.pth \
  --batch-size 64 \
  --num-workers 4 \
  --steps 20000 \
  --save-every 1000 \
  --log-every 20 \
  --hand-learning-rate 1e-4 \
  --validation-batches 2
