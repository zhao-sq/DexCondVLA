#!/usr/bin/env bash
set -euo pipefail

# Bulk-transfer the local Isaac environment/assets after training finishes, then
# wait for the all-remote evaluation and copy only the requested final artifacts back.
# SELECTED_STEP bypasses the normal completion gate for an explicitly selected
# complete checkpoint after training has been intentionally stopped.

DEXTOOL_ROOT="${DEXTOOL_ROOT:-/mnt/ssd1/szhao/DexTool}"
REMOTE_HOST="${REMOTE_HOST:-root@136.175.252.91}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-35499}"
REMOTE_KEY="${REMOTE_KEY:-/home/shuqi/.ssh/vast_ai}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520}"
TRAIN_PROGRAM="${TRAIN_PROGRAM:-task26_dual_joint}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
SELECTED_STEP="${SELECTED_STEP:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
REMOTE_DEXTOOL="/workspace/DexTool"
REMOTE_QUEUE_DIR="${REMOTE_DEXTOOL}/queued_evaluations/${EXPERIMENT_NAME}"
ASSETS_READY="${REMOTE_QUEUE_DIR}/REMOTE_ASSETS_READY"
FAILED_FILE="${REMOTE_QUEUE_DIR}/REMOTE_EVALUATION_FAILED.txt"
COMPLETE_ENV="${REMOTE_QUEUE_DIR}/REMOTE_EVALUATION_COMPLETE.env"
LOG="${DEXTOOL_ROOT}/openpi_logs/${EXPERIMENT_NAME}_remote_bulk_transfer.log"

mkdir -p "$(dirname "${LOG}")"
exec >>"${LOG}" 2>&1
ssh_base=(ssh -i "${REMOTE_KEY}" -o IdentitiesOnly=yes -o ConnectTimeout=10 -p "${REMOTE_SSH_PORT}")
rsync_ssh="ssh -i ${REMOTE_KEY} -o IdentitiesOnly=yes -o ConnectTimeout=10 -p ${REMOTE_SSH_PORT}"

rsync_retry() {
    local attempt
    for attempt in $(seq 1 10); do
        if rsync "$@"; then
            return 0
        fi
        echo "[transfer] rsync attempt ${attempt}/10 failed; retrying in 60 seconds"
        sleep 60
    done
    return 1
}

if [[ -n "${SELECTED_STEP}" ]]; then
    [[ "${SELECTED_STEP}" =~ ^[0-9]+$ ]] || { echo "Invalid SELECTED_STEP: ${SELECTED_STEP}"; exit 1; }
    selected_checkpoint="${REMOTE_DEXTOOL}/ckp/pi05_bench2dex_task26_dual_expert_arm/${EXPERIMENT_NAME}/${SELECTED_STEP}"
    "${ssh_base[@]}" "${REMOTE_HOST}" \
        "test -f '${selected_checkpoint}/_CHECKPOINT_METADATA' && test -s '${selected_checkpoint}/hand_dit.pt'" \
        || { echo "Selected checkpoint is incomplete: ${selected_checkpoint}"; exit 1; }
    status="$("${ssh_base[@]}" "${REMOTE_HOST}" "supervisorctl status '${TRAIN_PROGRAM}'" 2>&1 || true)"
    state="$(awk -v program="${TRAIN_PROGRAM}" '$1 == program {print $2}' <<<"${status}" | tail -1)"
    if [[ "${state}" == "RUNNING" || "${state}" == "STARTING" ]]; then
        echo "Selected-checkpoint evaluation requires the trainer to be stopped; state=${state}"
        exit 1
    fi
    echo "[transfer] $(date --iso-8601=seconds) using explicitly selected complete checkpoint ${SELECTED_STEP}"
else
    echo "[transfer] $(date --iso-8601=seconds) waiting for training completion"
    while true; do
        status="$("${ssh_base[@]}" "${REMOTE_HOST}" "supervisorctl status '${TRAIN_PROGRAM}'" 2>&1 || true)"
        state="$(awk -v program="${TRAIN_PROGRAM}" '$1 == program {print $2}' <<<"${status}" | tail -1)"
        if [[ -z "${state}" ]]; then
            echo "[transfer] unable to read trainer state; retrying"
            sleep "${POLL_SECONDS}"
            continue
        fi
        if [[ "${state}" == "EXITED" ]]; then
            "${ssh_base[@]}" "${REMOTE_HOST}" \
                "grep -qE 'step=${TRAIN_STEPS} .*steps_per_second=' '${REMOTE_DEXTOOL}/openpi_logs/${EXPERIMENT_NAME}.log'" \
                || { echo "Trainer exited without completing ${TRAIN_STEPS} steps"; exit 1; }
            break
        fi
        if [[ "${state}" != "RUNNING" && "${state}" != "STARTING" ]]; then
            echo "Unexpected trainer state: ${state}"
            exit 1
        fi
        sleep "${POLL_SECONDS}"
    done
fi

available_gb="$("${ssh_base[@]}" "${REMOTE_HOST}" "df --output=avail -BG /workspace | tail -1 | tr -dc '0-9'" | tail -1)"
if [[ ! "${available_gb}" =~ ^[0-9]+$ ]] || (( available_gb < 65 )); then
    echo "Remote free space is ${available_gb:-unknown}GB; at least 65GB is required"
    exit 1
fi
echo "[transfer] remote free space before upload: ${available_gb}GB"

"${ssh_base[@]}" "${REMOTE_HOST}" \
    "mkdir -p /home/shuqi/miniconda3/envs/env_isaaclab '${REMOTE_DEXTOOL}/bench2dex_isaaclab_env' '${REMOTE_DEXTOOL}/dex2bench_dataset' '${REMOTE_DEXTOOL}/Bench2Dex' '${REMOTE_DEXTOOL}/IsaacLab-2.3.2' '${REMOTE_QUEUE_DIR}'"

rsync_retry -aH --partial --info=stats2 -e "${rsync_ssh}" \
    /home/shuqi/miniconda3/envs/env_isaaclab/ \
    "${REMOTE_HOST}:/home/shuqi/miniconda3/envs/env_isaaclab/"
rsync_retry -aH --partial --info=stats2 -e "${rsync_ssh}" \
    "${DEXTOOL_ROOT}/bench2dex_isaaclab_env/" \
    "${REMOTE_HOST}:${REMOTE_DEXTOOL}/bench2dex_isaaclab_env/"
rsync_retry -aH --partial --info=stats2 -e "${rsync_ssh}" \
    "${DEXTOOL_ROOT}/dex2bench_dataset/" \
    "${REMOTE_HOST}:${REMOTE_DEXTOOL}/dex2bench_dataset/"
rsync_retry -aH --partial --info=stats2 -e "${rsync_ssh}" \
    "${DEXTOOL_ROOT}/Bench2Dex/" \
    "${REMOTE_HOST}:${REMOTE_DEXTOOL}/Bench2Dex/"
rsync_retry -aH --partial --info=stats2 -e "${rsync_ssh}" \
    "${DEXTOOL_ROOT}/IsaacLab-2.3.2/" \
    "${REMOTE_HOST}:${REMOTE_DEXTOOL}/IsaacLab-2.3.2/"

"${ssh_base[@]}" "${REMOTE_HOST}" "
    set -e
    link='${REMOTE_DEXTOOL}/dex2bench_dataset/Robots_p/kuka+sharpa/usd'
    if test -L \"\${link}\"; then rm \"\${link}\"; elif test -e \"\${link}\"; then echo 'Expected usd to be a symlink' >&2; exit 1; fi
    ln -s usd_sdf_linux \"\${link}\"
    test -x '${REMOTE_DEXTOOL}/bench2dex_isaaclab_env/bin/python'
    '${REMOTE_DEXTOOL}/bench2dex_isaaclab_env/bin/python' -c 'import sys, torch, isaacsim; print(sys.executable, torch.__version__)'
    printf 'ready_at=%s\n' \"$(date --iso-8601=seconds)\" > '${ASSETS_READY}.tmp'
    mv '${ASSETS_READY}.tmp' '${ASSETS_READY}'
    df -h /workspace
"

echo "[transfer] assets ready; waiting for remote evaluation"
while ! "${ssh_base[@]}" "${REMOTE_HOST}" "test -s '${COMPLETE_ENV}'"; do
    if "${ssh_base[@]}" "${REMOTE_HOST}" "test -s '${FAILED_FILE}'"; then
        "${ssh_base[@]}" "${REMOTE_HOST}" "sed -n '1,120p' '${FAILED_FILE}'"
        exit 1
    fi
    sleep "${POLL_SECONDS}"
done

complete="$("${ssh_base[@]}" "${REMOTE_HOST}" "sed -n '1,20p' '${COMPLETE_ENV}'")"
result_name="$(awk -F= '$1=="result_name" {print $2}' <<<"${complete}")"
remote_result="$(awk -F= '$1=="result_root" {sub(/^[^=]*=/, ""); print}' <<<"${complete}")"
[[ -n "${result_name}" && -n "${remote_result}" ]] || { echo "Invalid completion metadata"; exit 1; }
local_result="${DEXTOOL_ROOT}/results/${result_name}"
mkdir -p "${local_result}/rescore_without_object_upright"

rsync_retry -aH --partial --info=stats2 -e "${rsync_ssh}" \
    "${REMOTE_HOST}:${remote_result}/rollouts/" "${local_result}/rollouts/"
rsync_retry -aH --partial --info=stats2 -e "${rsync_ssh}" \
    "${REMOTE_HOST}:${remote_result}/rescore_without_object_upright/" \
    "${local_result}/rescore_without_object_upright/"
rsync_retry -aH --partial -e "${rsync_ssh}" \
    "${REMOTE_HOST}:${remote_result}/EVALUATION_COMPLETE.env" "${local_result}/"
rsync_retry -aH --partial -e "${rsync_ssh}" \
    "${REMOTE_HOST}:${remote_result}/remote_evaluation_queue.log" "${local_result}/"

video_count="$(find "${local_result}/rollouts" -maxdepth 1 -type f -name 'rollout_*.mp4' | wc -l)"
[[ "${video_count}" -eq 100 ]] || { echo "Expected 100 downloaded videos, found ${video_count}"; exit 1; }
[[ -s "${local_result}/rollouts/success_summary.json" ]] || { echo "Missing official summary"; exit 1; }
[[ -s "${local_result}/rescore_without_object_upright/summary.json" ]] || { echo "Missing relaxed summary"; exit 1; }
echo "[transfer] completed result download: ${local_result}"
