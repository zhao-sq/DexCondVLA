#!/usr/bin/env bash
set -u

# Self-healing wrapper for the local bulk-transfer/result-download coordinator.
# It never controls the formal trainer or W&B. It may restart only the dedicated
# remote evaluation queue created for this run.

REPO_ROOT="${REPO_ROOT:-/home/shuqi/DexCondVLA}"
DEXTOOL_ROOT="${DEXTOOL_ROOT:-/mnt/ssd1/szhao/DexTool}"
REMOTE_HOST="${REMOTE_HOST:-root@136.175.252.91}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-35499}"
REMOTE_KEY="${REMOTE_KEY:-/home/shuqi/.ssh/vast_ai}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
RETRY_SECONDS="${RETRY_SECONDS:-300}"
LOG="${DEXTOOL_ROOT}/openpi_logs/${EXPERIMENT_NAME}_pipeline_watchdog.log"
DIAGNOSTICS="${DEXTOOL_ROOT}/openpi_logs/${EXPERIMENT_NAME}_pipeline_diagnostics.log"
REMOTE_QUEUE_DIR="/workspace/DexTool/queued_evaluations/${EXPERIMENT_NAME}"
REMOTE_COMPLETE="${REMOTE_QUEUE_DIR}/REMOTE_EVALUATION_COMPLETE.env"
LOCAL_RESULTS_ROOT="${DEXTOOL_ROOT}/results"

mkdir -p "$(dirname "${LOG}")"
exec >>"${LOG}" 2>&1
ssh_base=(ssh -i "${REMOTE_KEY}" -o IdentitiesOnly=yes -o ConnectTimeout=10 -p "${REMOTE_SSH_PORT}")

is_complete() {
    local metadata result_name
    metadata="$("${ssh_base[@]}" "${REMOTE_HOST}" "test -s '${REMOTE_COMPLETE}' && sed -n '1,20p' '${REMOTE_COMPLETE}'" 2>/dev/null || true)"
    result_name="$(awk -F= '$1=="result_name" {print $2}' <<<"${metadata}")"
    [[ -n "${result_name}" ]] || return 1
    [[ -s "${LOCAL_RESULTS_ROOT}/${result_name}/rollouts/success_summary.json" ]] || return 1
    [[ -s "${LOCAL_RESULTS_ROOT}/${result_name}/rescore_without_object_upright/summary.json" ]] || return 1
    [[ "$(find "${LOCAL_RESULTS_ROOT}/${result_name}/rollouts" -maxdepth 1 -type f -name 'rollout_*.mp4' 2>/dev/null | wc -l)" -eq 100 ]]
}

collect_diagnostics() {
    {
        echo "===== $(date --iso-8601=seconds) watchdog diagnostics ====="
        df -h "${DEXTOOL_ROOT}" || true
        nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader || true
        "${ssh_base[@]}" "${REMOTE_HOST}" "
            echo SUPERVISOR
            supervisorctl status task26_dual_joint task26_joint_wandb task26_dual_best_policy || true
            echo DISK
            df -h /workspace || true
            echo GPU
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
            echo QUEUE_LOG
            supervisorctl tail -80 task26_dual_best_policy || true
            echo POLICY_LOG
            find /workspace/DexTool/results -path '*${EXPERIMENT_NAME}*/policy_server.log' -type f -exec tail -40 {} \; 2>/dev/null || true
            echo EVAL_LOG
            find /workspace/DexTool/results -path '*${EXPERIMENT_NAME}*/evaluation.log' -type f -exec tail -80 {} \; 2>/dev/null || true
        " || true
    } >>"${DIAGNOSTICS}" 2>&1
}

restart_evaluation_queue_if_needed() {
    local state
    state="$("${ssh_base[@]}" "${REMOTE_HOST}" "supervisorctl status task26_dual_best_policy" 2>/dev/null \
        | awk '$1=="task26_dual_best_policy" {print $2}' | tail -1)"
    case "${state}" in
        RUNNING|STARTING) return 0 ;;
        STOPPED|EXITED|FATAL|BACKOFF)
            echo "[watchdog] restarting dedicated remote evaluation queue from state=${state}"
            "${ssh_base[@]}" "${REMOTE_HOST}" "supervisorctl start task26_dual_best_policy" || true
            ;;
        *) echo "[watchdog] remote queue state unavailable; will retry later" ;;
    esac
}

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    if is_complete; then
        echo "[watchdog] pipeline already complete"
        exit 0
    fi
    echo "[watchdog] $(date --iso-8601=seconds) coordinator attempt ${attempt}/${MAX_ATTEMPTS}"
    if bash "${REPO_ROOT}/scripts/queue_task26_remote_assets_and_results_transfer_local.sh"; then
        if is_complete; then
            echo "[watchdog] desired result verified"
            exit 0
        fi
        echo "[watchdog] coordinator returned success but final validation is incomplete"
    else
        echo "[watchdog] coordinator failed"
    fi
    collect_diagnostics
    restart_evaluation_queue_if_needed
    sleep "${RETRY_SECONDS}"
done

echo "[watchdog] exhausted ${MAX_ATTEMPTS} attempts; diagnostics=${DIAGNOSTICS}"
exit 1
