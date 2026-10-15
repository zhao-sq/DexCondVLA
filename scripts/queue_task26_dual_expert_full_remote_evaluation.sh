#!/usr/bin/env bash
set -euo pipefail

# Wait for both the formal joint training run and the bulk-uploaded Isaac assets,
# then run policy inference and all 100 Bench2Dex rollouts on the rented server.
# SELECTED_STEP bypasses the normal-training completion gate for an explicitly
# selected complete checkpoint after training has been intentionally stopped.

REPO_ROOT="${REPO_ROOT:-/workspace/DexCondVLA}"
DEXTOOL_ROOT="${DEXTOOL_ROOT:-/workspace/DexTool}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520}"
TRAIN_PROGRAM="${TRAIN_PROGRAM:-task26_dual_joint}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
SELECTED_STEP="${SELECTED_STEP:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PORT="${PORT:-9000}"
OPENPI_PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${DEXTOOL_ROOT}/bench2dex_isaaclab_env/bin/python}"
BENCH2DEX_ROOT="${BENCH2DEX_ROOT:-${DEXTOOL_ROOT}/Bench2Dex}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${DEXTOOL_ROOT}/IsaacLab-2.3.2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DEXTOOL_ROOT}/ckp/pi05_bench2dex_task26_dual_expert_arm/${EXPERIMENT_NAME}}"
TRAIN_LOG="${TRAIN_LOG:-${DEXTOOL_ROOT}/openpi_logs/${EXPERIMENT_NAME}.log}"
QUEUE_DIR="${QUEUE_DIR:-${DEXTOOL_ROOT}/queued_evaluations/${EXPERIMENT_NAME}}"
ASSETS_READY="${ASSETS_READY:-${QUEUE_DIR}/REMOTE_ASSETS_READY}"
FAILED_FILE="${FAILED_FILE:-${QUEUE_DIR}/REMOTE_EVALUATION_FAILED.txt}"
COMPLETE_ENV="${COMPLETE_ENV:-${QUEUE_DIR}/REMOTE_EVALUATION_COMPLETE.env}"
CONFIG="pi05_bench2dex_task26_dual_expert_arm"

mkdir -p "${QUEUE_DIR}"
rm -f "${FAILED_FILE}"

fail() {
    local message="$1"
    printf '%s\n' "${message}" | tee "${FAILED_FILE}" >&2
    exit 1
}

[[ -x "${OPENPI_PYTHON}" ]] || fail "Missing OpenPI interpreter: ${OPENPI_PYTHON}"
echo "[remote-eval] OPENPI_PYTHON=${OPENPI_PYTHON}"
"${OPENPI_PYTHON}" -c 'import sys; print(f"[remote-eval] openpi_interpreter={sys.executable}")'

if [[ -n "${SELECTED_STEP}" ]]; then
    [[ "${SELECTED_STEP}" =~ ^[0-9]+$ ]] || fail "Invalid SELECTED_STEP: ${SELECTED_STEP}"
    echo "[remote-eval] explicitly selected checkpoint step=${SELECTED_STEP}; waiting only for uploaded assets"
    while [[ ! -s "${ASSETS_READY}" ]]; do
        echo "[remote-eval] $(date --iso-8601=seconds) selected_step=${SELECTED_STEP} assets=missing"
        sleep "${POLL_SECONDS}"
    done
else
    echo "[remote-eval] waiting for training and bulk-uploaded Isaac environment"
    while true; do
        status="$(supervisorctl status "${TRAIN_PROGRAM}" 2>&1 || true)"
        state="$(awk '{print $2}' <<<"${status}")"
        assets_state="missing"
        [[ -s "${ASSETS_READY}" ]] && assets_state="ready"
        echo "[remote-eval] $(date --iso-8601=seconds) trainer=${state:-unknown} assets=${assets_state}"
        if [[ "${state}" != "RUNNING" && "${state}" != "STARTING" && "${state}" != "EXITED" ]]; then
            fail "Trainer did not finish normally: ${status}"
        fi
        if [[ "${state}" == "EXITED" && "${assets_state}" == "ready" ]]; then
            break
        fi
        sleep "${POLL_SECONDS}"
    done
    grep -qE "step=${TRAIN_STEPS} .*steps_per_second=" "${TRAIN_LOG}" \
        || fail "Trainer exited without logged step=${TRAIN_STEPS}"
fi
[[ -x "${ISAAC_PYTHON}" ]] || fail "Missing Isaac interpreter: ${ISAAC_PYTHON}"
[[ -d "${BENCH2DEX_ROOT}" ]] || fail "Missing Bench2Dex repository"
[[ -d "${ISAACLAB_ROOT}" ]] || fail "Missing IsaacLab repository"
[[ -d "${DEXTOOL_ROOT}/dex2bench_dataset" ]] || fail "Missing Dex2Bench assets"
echo "[remote-eval] ISAAC_PYTHON=${ISAAC_PYTHON}"
"${ISAAC_PYTHON}" -c 'import sys; print(f"[remote-eval] isaac_interpreter={sys.executable}")'

if [[ -n "${SELECTED_STEP}" ]]; then
    best_step="${SELECTED_STEP}"
else
    [[ -s "${CHECKPOINT_ROOT}/BEST" ]] || fail "Missing BEST pointer"
    best_step="$(tr -d '[:space:]' < "${CHECKPOINT_ROOT}/BEST")"
fi
[[ "${best_step}" =~ ^[0-9]+$ ]] || fail "Invalid BEST step: ${best_step}"
checkpoint="${CHECKPOINT_ROOT}/${best_step}"
hand_checkpoint="${checkpoint}/hand_dit.pt"
[[ -f "${checkpoint}/_CHECKPOINT_METADATA" ]] || fail "Incomplete VLA checkpoint: ${checkpoint}"
[[ -s "${hand_checkpoint}" ]] || fail "Missing Hand-DiT checkpoint: ${hand_checkpoint}"

result_name="${EXPERIMENT_NAME}_step${best_step}"
result_root="${DEXTOOL_ROOT}/results/${result_name}"
mkdir -p "${result_root}"
exec > >(tee -a "${result_root}/remote_evaluation_queue.log") 2>&1
echo "[remote-eval] result_root=${result_root}"

if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
    fail "Port ${PORT} is occupied; no process was stopped"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${DEXTOOL_ROOT}/openpi_assets/openpi-cache}"
export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_tasks:${ISAACLAB_ROOT}/source/isaaclab_mimic:${PYTHONPATH:-}"

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}" \
"${OPENPI_PYTHON}" -u "${REPO_ROOT}/scripts/serve_bench2dex_rpc_policy.py" \
    --config "${CONFIG}" \
    --checkpoint "${checkpoint}" \
    --hand-dit-checkpoint "${hand_checkpoint}" \
    --dinov3-repo "${DEXTOOL_ROOT}/dinov3" \
    --dinov3-checkpoint "${DEXTOOL_ROOT}/assets/dinov3_vitb16.pth" \
    --hand-device cuda:0 \
    --hand-flow-steps 10 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --warmup-runs 1 \
    >"${result_root}/policy_server.log" 2>&1 &
policy_pid=$!

cleanup_policy() {
    if kill -0 "${policy_pid}" 2>/dev/null; then
        kill -TERM "${policy_pid}"
        wait "${policy_pid}" || true
    fi
}
trap cleanup_policy EXIT INT TERM

for _ in $(seq 1 180); do
    kill -0 "${policy_pid}" 2>/dev/null || fail "Policy server exited during startup"
    if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then break; fi
    sleep 5
done
ss -ltn "sport = :${PORT}" | grep -q LISTEN || fail "Policy server startup timed out"

evaluation_ok=0
for evaluation_attempt in $(seq 1 5); do
    echo "[remote-eval] evaluation attempt ${evaluation_attempt}/5"
    if PIPELINE_MODE=absolute \
        POLICY_CONFIG="${CONFIG}" \
        POLICY_CHECKPOINT="${checkpoint}" \
        POLICY_NAME="${result_name}" \
        RESULT_ROOT="${result_root}" \
        REMOTE_PORT="${PORT}" \
        ISAAC_PYTHON="${ISAAC_PYTHON}" \
        ISAACLAB_ROOT="${ISAACLAB_ROOT}" \
        BENCH2DEX_ROOT="${BENCH2DEX_ROOT}" \
        REPO_ROOT="${REPO_ROOT}" \
        ANCHOR_DIR="${DEXTOOL_ROOT}/bench2dex/task26/replay-generalization" \
        bash "${REPO_ROOT}/scripts/run_bench2dex_task26_evaluation.sh" \
            >>"${result_root}/evaluation.log" 2>&1; then
        evaluation_ok=1
        break
    fi
    echo "[remote-eval] evaluator failed; completed chunks will be reused"
    df -h /workspace || true
    nvidia-smi || true
    kill -0 "${policy_pid}" 2>/dev/null || fail "Policy server exited during evaluation"
    sleep 60
done
[[ "${evaluation_ok}" -eq 1 ]] || fail "Evaluator failed five times"

"${OPENPI_PYTHON}" "${REPO_ROOT}/scripts/export_bench2dex_task26_rollouts.py" \
    --eval-root "${result_root}/eval" \
    --output-dir "${result_root}/rollouts" \
    --bench2dex-root "${BENCH2DEX_ROOT}"

"${ISAAC_PYTHON}" "${REPO_ROOT}/scripts/rescore_bench2dex_task26_without_object_upright.py" \
    --eval-root "${result_root}/eval" \
    --output-dir "${result_root}/rescore_without_object_upright" \
    --bench2dex-root "${BENCH2DEX_ROOT}"

"${OPENPI_PYTHON}" "${REPO_ROOT}/scripts/relabel_bench2dex_relaxed_success_videos.py" \
    --result-root "${result_root}"

video_count="$(find "${result_root}/rollouts" -maxdepth 1 -type f -name 'rollout_*.mp4' | wc -l)"
[[ "${video_count}" -eq 100 ]] || fail "Expected 100 videos, found ${video_count}"
[[ -s "${result_root}/rollouts/success_summary.json" ]] || fail "Missing official summary"
[[ -s "${result_root}/rescore_without_object_upright/summary.json" ]] || fail "Missing relaxed summary"

temporary="${COMPLETE_ENV}.tmp"
{
    printf 'experiment_name=%s\n' "${EXPERIMENT_NAME}"
    printf 'best_step=%s\n' "${best_step}"
    printf 'checkpoint=%s\n' "${checkpoint}"
    printf 'result_name=%s\n' "${result_name}"
    printf 'result_root=%s\n' "${result_root}"
    printf 'videos=%s\n' "${video_count}"
} >"${temporary}"
mv "${temporary}" "${COMPLETE_ENV}"
cp "${COMPLETE_ENV}" "${result_root}/EVALUATION_COMPLETE.env"
echo "[remote-eval] evaluation complete; leaving policy server running as requested"

trap - EXIT
wait "${policy_pid}"
