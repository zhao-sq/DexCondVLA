#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="/mnt/ssd1/szhao/DexTool/results/task26_action_expert_v1_step16000"
EVAL_LOG="${RESULT_ROOT}/evaluation.log"
POSTPROCESS_LOG="${RESULT_ROOT}/postprocess.log"
FAILED_MARKER="${RESULT_ROOT}/EVALUATION_POSTPROCESS_FAILED.txt"
COMPLETE_MARKER="${RESULT_ROOT}/EVALUATION_COMPLETE.txt"
OPENPI_PYTHON="/mnt/ssd1/szhao/DexTool/openpi_env/bin/python"
ISAAC_PYTHON="/mnt/ssd1/szhao/DexTool/bench2dex_isaaclab_env/bin/python"
REPO_ROOT="/home/shuqi/DexCondVLA"
RESCORE_DIR="${RESULT_ROOT}/rescore_without_object_upright"
EVAL_SESSION="${EVAL_SESSION:-bench2dex_task26_eval}"

exec >>"${POSTPROCESS_LOG}" 2>&1
echo "[postprocess] Waiting for tmux session ${EVAL_SESSION}"
while tmux has-session -t "${EVAL_SESSION}" 2>/dev/null; do
    sleep 30
done

if ! grep -qF '[eval] All 100 Task26 rollouts completed:' "${EVAL_LOG}"; then
    {
        echo "Task26 evaluator exited before reporting all 100 completed rollouts."
        echo "Inspect ${EVAL_LOG}, then rerun scripts/run_bench2dex_task26_evaluation.sh."
    } >"${FAILED_MARKER}"
    echo "[postprocess] Evaluation incomplete; wrote ${FAILED_MARKER}"
    exit 1
fi

cd "${REPO_ROOT}"
echo "[postprocess] OPENPI_PYTHON=${OPENPI_PYTHON}"
echo "[postprocess] ISAAC_PYTHON=${ISAAC_PYTHON}"
"${OPENPI_PYTHON}" scripts/export_bench2dex_task26_rollouts.py \
    --eval-root "${RESULT_ROOT}/eval" \
    --output-dir "${RESULT_ROOT}/rollouts" \
    --bench2dex-root /mnt/ssd1/szhao/DexTool/Bench2Dex

"${ISAAC_PYTHON}" scripts/rescore_bench2dex_task26_without_object_upright.py \
    --eval-root "${RESULT_ROOT}/eval" \
    --output-dir "${RESCORE_DIR}" \
    --bench2dex-root /mnt/ssd1/szhao/DexTool/Bench2Dex

"${OPENPI_PYTHON}" scripts/finalize_bench2dex_task26_results.py \
    --rollout-dir "${RESULT_ROOT}/rollouts" \
    --documentation "${REPO_ROOT}/pi05_bench2dex_task26_evaluation.md" \
    --completion-marker "${COMPLETE_MARKER}"

rm -f "${FAILED_MARKER}"
echo "[postprocess] Task26 evaluation export and documentation complete"
