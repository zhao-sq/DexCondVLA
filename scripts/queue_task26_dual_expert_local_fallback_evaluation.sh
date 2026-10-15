#!/usr/bin/env bash
set -euo pipefail

# Local fallback for rented hosts whose NVIDIA driver cannot start Isaac RTX.
# Wait for one compatible-driver GPU to become safely available, then run the
# policy server and Isaac evaluator on that same GPU without cross-host traffic.

REPO_ROOT="${REPO_ROOT:-/home/shuqi/DexCondVLA}"
DEXTOOL_ROOT="${DEXTOOL_ROOT:-/mnt/ssd1/szhao/DexTool}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520}"
SELECTED_STEP="${SELECTED_STEP:-18000}"
MIN_FREE_VRAM_MB="${MIN_FREE_VRAM_MB:-70000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PORT="${PORT:-9010}"
OPENPI_PYTHON="${OPENPI_PYTHON:-${DEXTOOL_ROOT}/openpi_env/bin/python}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${DEXTOOL_ROOT}/bench2dex_isaaclab_env/bin/python}"
BENCH2DEX_ROOT="${BENCH2DEX_ROOT:-${DEXTOOL_ROOT}/Bench2Dex}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${DEXTOOL_ROOT}/IsaacLab-2.3.2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DEXTOOL_ROOT}/ckp/pi05_bench2dex_task26_dual_expert_arm/${EXPERIMENT_NAME}}"
CHECKPOINT="${CHECKPOINT_ROOT}/${SELECTED_STEP}"
RESULT_NAME="${EXPERIMENT_NAME}_step${SELECTED_STEP}"
RESULT_ROOT="${DEXTOOL_ROOT}/results/${RESULT_NAME}"
CONFIG="pi05_bench2dex_task26_dual_expert_arm"

mkdir -p "${CHECKPOINT_ROOT}" "${RESULT_ROOT}"
exec > >(tee -a "${RESULT_ROOT}/local_fallback_queue.log") 2>&1
echo "[local-fallback] OPENPI_PYTHON=${OPENPI_PYTHON}"
echo "[local-fallback] ISAAC_PYTHON=${ISAAC_PYTHON}"
"${OPENPI_PYTHON}" -c 'import sys; print(f"[local-fallback] openpi_interpreter={sys.executable}")'
"${ISAAC_PYTHON}" -c 'import sys; print(f"[local-fallback] isaac_interpreter={sys.executable}")'

while [[ ! -f "${CHECKPOINT}/_CHECKPOINT_METADATA" || ! -s "${CHECKPOINT}/hand_dit.pt" ]]; do
    echo "[local-fallback] $(date --iso-8601=seconds) waiting for checkpoint transfer: ${CHECKPOINT}"
    sleep "${POLL_SECONDS}"
done

if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
    echo "Port ${PORT} is occupied; refusing to stop its owner" >&2
    exit 1
fi

gpu_index=""
while [[ -z "${gpu_index}" ]]; do
    best_line="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2,2nr | head -n 1)"
    candidate_index="$(awk -F, '{gsub(/ /, "", $1); print $1}' <<<"${best_line}")"
    candidate_free="$(awk -F, '{gsub(/ /, "", $2); print $2}' <<<"${best_line}")"
    if [[ "${candidate_index}" =~ ^[0-9]+$ && "${candidate_free}" =~ ^[0-9]+$ ]] \
        && (( candidate_free >= MIN_FREE_VRAM_MB )); then
        gpu_index="${candidate_index}"
        break
    fi
    echo "[local-fallback] $(date --iso-8601=seconds) waiting for >=${MIN_FREE_VRAM_MB}MiB free VRAM; best=${candidate_free:-unknown}MiB"
    sleep "${POLL_SECONDS}"
done

echo "[local-fallback] selected physical GPU ${gpu_index}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export OPENPI_DATA_HOME="${DEXTOOL_ROOT}/openpi_assets/openpi-cache"
export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_tasks:${ISAACLAB_ROOT}/source/isaaclab_mimic:${PYTHONPATH:-}"

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}" \
"${OPENPI_PYTHON}" -u "${REPO_ROOT}/scripts/serve_bench2dex_rpc_policy.py" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --hand-dit-checkpoint "${CHECKPOINT}/hand_dit.pt" \
    --dinov3-repo "${DEXTOOL_ROOT}/dinov3" \
    --dinov3-checkpoint "/mnt/ssd1/szhao/LeveFD/dinov3_vitb16.pth" \
    --hand-device cuda:0 \
    --hand-flow-steps 10 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --warmup-runs 1 \
    >"${RESULT_ROOT}/policy_server.log" 2>&1 &
policy_pid=$!

cleanup_policy() {
    if kill -0 "${policy_pid}" 2>/dev/null; then
        kill -TERM "${policy_pid}"
        wait "${policy_pid}" || true
    fi
}
trap cleanup_policy EXIT INT TERM

for _ in $(seq 1 180); do
    kill -0 "${policy_pid}" 2>/dev/null || { echo "Policy server exited during startup" >&2; exit 1; }
    if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then break; fi
    sleep 5
done
ss -ltn "sport = :${PORT}" | grep -q LISTEN || { echo "Policy server startup timed out" >&2; exit 1; }

PIPELINE_MODE=absolute \
POLICY_CONFIG="${CONFIG}" \
POLICY_CHECKPOINT="${CHECKPOINT}" \
POLICY_NAME="${RESULT_NAME}" \
RESULT_ROOT="${RESULT_ROOT}" \
REMOTE_PORT="${PORT}" \
ISAAC_PYTHON="${ISAAC_PYTHON}" \
ISAACLAB_ROOT="${ISAACLAB_ROOT}" \
BENCH2DEX_ROOT="${BENCH2DEX_ROOT}" \
REPO_ROOT="${REPO_ROOT}" \
ANCHOR_DIR="${DEXTOOL_ROOT}/bench2dex/task26/replay-generalization" \
UNSET_ISAAC_CUDA_VISIBLE_DEVICES=0 \
DEX2BENCH_HEADLESS_DISABLE_ACTIVE_VIEWPORT=0 \
ISAAC_EXPERIENCE=isaaclab.python.rendering.kit \
bash "${REPO_ROOT}/scripts/run_bench2dex_task26_evaluation.sh"

"${OPENPI_PYTHON}" "${REPO_ROOT}/scripts/export_bench2dex_task26_rollouts.py" \
    --eval-root "${RESULT_ROOT}/eval" \
    --output-dir "${RESULT_ROOT}/rollouts" \
    --bench2dex-root "${BENCH2DEX_ROOT}"
"${ISAAC_PYTHON}" "${REPO_ROOT}/scripts/rescore_bench2dex_task26_without_object_upright.py" \
    --eval-root "${RESULT_ROOT}/eval" \
    --output-dir "${RESULT_ROOT}/rescore_without_object_upright" \
    --bench2dex-root "${BENCH2DEX_ROOT}"
"${OPENPI_PYTHON}" "${REPO_ROOT}/scripts/relabel_bench2dex_relaxed_success_videos.py" \
    --result-root "${RESULT_ROOT}"

video_count="$(find "${RESULT_ROOT}/rollouts" -maxdepth 1 -type f -name 'rollout_*.mp4' | wc -l)"
[[ "${video_count}" -eq 100 ]] || { echo "Expected 100 videos, found ${video_count}" >&2; exit 1; }
[[ -s "${RESULT_ROOT}/rollouts/success_summary.json" ]]
[[ -s "${RESULT_ROOT}/rescore_without_object_upright/summary.json" ]]
printf 'result_root=%s\nvideos=%s\n' "${RESULT_ROOT}" "${video_count}" >"${RESULT_ROOT}/EVALUATION_COMPLETE.env"
echo "[local-fallback] evaluation complete: ${RESULT_ROOT}"
