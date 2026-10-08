#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="/mnt/ssd1/szhao/DexTool/results/task26_action_expert_v1_step16000"
EVAL_LOG="${RESULT_ROOT}/evaluation.log"
POSTPROCESS_LOG="${RESULT_ROOT}/postprocess.log"
FAILED_MARKER="${RESULT_ROOT}/EVALUATION_POSTPROCESS_FAILED.txt"
COMPLETE_MARKER="${RESULT_ROOT}/EVALUATION_COMPLETE.txt"
OPENPI_PYTHON="/mnt/ssd1/szhao/DexTool/openpi_env/bin/python"
REPO_ROOT="/home/shuqi/DexCondVLA"

exec >>"${POSTPROCESS_LOG}" 2>&1
echo "[postprocess] Waiting for tmux session bench2dex_task26_eval"
while tmux has-session -t bench2dex_task26_eval 2>/dev/null; do
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
"${OPENPI_PYTHON}" scripts/export_bench2dex_task26_rollouts.py \
    --eval-root "${RESULT_ROOT}/eval" \
    --output-dir "${RESULT_ROOT}/rollouts" \
    --bench2dex-root /mnt/ssd1/szhao/DexTool/Bench2Dex

"${OPENPI_PYTHON}" scripts/finalize_bench2dex_task26_results.py \
    --rollout-dir "${RESULT_ROOT}/rollouts" \
    --documentation "${REPO_ROOT}/pi05_bench2dex_task26_evaluation.md" \
    --completion-marker "${COMPLETE_MARKER}"

echo "[postprocess] Task26 evaluation export and documentation complete"
