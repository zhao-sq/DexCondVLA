#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/shuqi/DexCondVLA"
PYTHON="/mnt/ssd1/szhao/DexTool/openpi_env/bin/python"
CONFIG="pi05_bench2dex_task26_action_expert_25k"
EXP_NAME="task26_action_expert_absolute_25k_1011164016"

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export OPENPI_DATA_HOME="/mnt/ssd1/szhao/DexTool/openpi_assets/openpi-cache"
export WANDB_DIR="/mnt/ssd1/szhao/DexTool/openpi_logs/wandb"

mkdir -p "${WANDB_DIR}"
cd "${REPO_ROOT}"

printf 'Python interpreter: %s\n' "${PYTHON}"
printf 'Config: %s\n' "${CONFIG}"
printf 'Experiment: %s\n' "${EXP_NAME}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'XLA_PYTHON_CLIENT_MEM_FRACTION: %s\n' "${XLA_PYTHON_CLIENT_MEM_FRACTION}"

exec "${PYTHON}" -u scripts/train.py "${CONFIG}" --exp-name="${EXP_NAME}"
