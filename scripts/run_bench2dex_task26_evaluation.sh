#!/usr/bin/env bash
set -euo pipefail

# Official Bench2Dex Task26 protocol, with 25 episodes in each of the four
# generalization channels (100 rollouts total). Each simulator process handles
# five episodes to bound Isaac Sim memory growth; chunk seeds remain identical
# to a single 25-episode run through --start-episode.

BENCH2DEX_ROOT="${BENCH2DEX_ROOT:-/mnt/ssd1/szhao/DexTool/Bench2Dex}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/mnt/ssd1/szhao/DexTool/bench2dex_isaaclab_env/bin/python}"
REPO_ROOT="${REPO_ROOT:-/home/shuqi/DexCondVLA}"
ANCHOR_DIR="${ANCHOR_DIR:-/mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/ssd1/szhao/DexTool/results/task26_action_expert_v1_step16000/eval}"
ISAAC_TMPDIR="${ISAAC_TMPDIR:-/mnt/ssd1/szhao/DexTool/results/task26_action_expert_v1_step16000/isaac_tmp}"
REMOTE_PORT="${REMOTE_PORT:-9000}"
EPISODES_PER_CHANNEL="${EPISODES_PER_CHANNEL:-25}"
EPISODES_PER_PROCESS="${EPISODES_PER_PROCESS:-5}"
EPISODE_STEPS="${EPISODE_STEPS:-871}"
BASE_SEED="${BASE_SEED:-100000000}"
SOURCE_COLLECT_CONFIG="${SOURCE_COLLECT_CONFIG:-${BENCH2DEX_ROOT}/configs/collect/default.yaml}"
POLICY_COLLECT_CONFIG="${POLICY_COLLECT_CONFIG:-${ISAAC_TMPDIR}/task26_pi05_three_camera.yaml}"
FILTER_POLICY_CAMERAS="${FILTER_POLICY_CAMERAS:-1}"

export DEX2BENCH_RECORD_RES="${DEX2BENCH_RECORD_RES:-480}"
export DEX2BENCH_RECORD_JPEG="${DEX2BENCH_RECORD_JPEG:-80}"
export DEX2BENCH_RECORD_STRIDE="${DEX2BENCH_RECORD_STRIDE:-1}"
export DEX2BENCH_RECORD_CAMS="cam_overhead"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_ROOT}" "${ISAAC_TMPDIR}"
export TMPDIR="${ISAAC_TMPDIR}"
echo "[eval] ISAAC_PYTHON=${ISAAC_PYTHON}"
"${ISAAC_PYTHON}" -c 'import sys; print(f"[eval] interpreter={sys.executable}")'
echo "[eval] FILTER_POLICY_CAMERAS=${FILTER_POLICY_CAMERAS}"
if [[ "${FILTER_POLICY_CAMERAS}" == "1" ]]; then
    "${ISAAC_PYTHON}" "${REPO_ROOT}/scripts/filter_bench2dex_collect_config.py" \
        --input "${SOURCE_COLLECT_CONFIG}" \
        --output "${POLICY_COLLECT_CONFIG}" \
        --camera cam_overhead \
        --camera cam_wrist_left \
        --camera cam_wrist_right
elif [[ "${FILTER_POLICY_CAMERAS}" == "0" ]]; then
    POLICY_COLLECT_CONFIG="${SOURCE_COLLECT_CONFIG}"
else
    echo "FILTER_POLICY_CAMERAS must be 0 or 1, got: ${FILTER_POLICY_CAMERAS}" >&2
    exit 2
fi
cd "${BENCH2DEX_ROOT}"

run_channel() {
    local channel_name="$1"
    local profile="$2"
    local channel_dir="${OUTPUT_ROOT}/${channel_name}"
    mkdir -p "${channel_dir}/chunks"

    local start=1
    while (( start <= EPISODES_PER_CHANNEL )); do
        local count="${EPISODES_PER_PROCESS}"
        local remaining=$((EPISODES_PER_CHANNEL - start + 1))
        if (( count > remaining )); then count="${remaining}"; fi
        local end=$((start + count - 1))
        local label
        label=$(printf '%03d_%03d' "${start}" "${end}")
        local chunk_dir="${channel_dir}/chunks/${label}"

        if [[ -s "${chunk_dir}/per_episode.jsonl" ]] && \
           [[ $(wc -l < "${chunk_dir}/per_episode.jsonl") -eq ${count} ]]; then
            echo "[eval] ${channel_name} ${label}: already complete, skipping"
            start=$((end + 1))
            continue
        fi

        mkdir -p "${chunk_dir}"
        echo "[eval] ${channel_name} ${label}: starting ${count} episode(s)"
        args=(
            run_policy.py
            --task scenes/26_canned_food_tray_line_arrangement.yaml
            --policy-type REMOTE
            --policy-name pi05_task26_action_expert_step16000
            --robot-key multi_iiwa7_with_sharpa
            --collect-config "${POLICY_COLLECT_CONFIG}"
            --no-active-dof
            --enable-rgb
            --enable-generalization
            --generalization-profile "${profile}"
            --generalization-split unseen
            --episode-steps "${EPISODE_STEPS}"
            --num-episodes "${count}"
            --start-episode "${start}"
            --seed "${BASE_SEED}"
            --output-dir "${chunk_dir}"
            --record-dir "${chunk_dir}/records"
            --record-all
            --remote-host 127.0.0.1
            --remote-port "${REMOTE_PORT}"
            --headless
        )
        if [[ "${profile}" != "inv_cov" ]]; then
            args+=(--anchor-dir "${ANCHOR_DIR}")
        fi

        "${ISAAC_PYTHON}" "${args[@]}" 2>&1 | tee "${chunk_dir}/evaluation.log"
        if [[ ! -s "${chunk_dir}/per_episode.jsonl" ]] || \
           [[ $(wc -l < "${chunk_dir}/per_episode.jsonl") -ne ${count} ]]; then
            echo "[eval] ERROR: ${channel_name} ${label} did not produce ${count} results" >&2
            exit 1
        fi
        start=$((end + 1))
    done

    : > "${channel_dir}/per_episode.jsonl"
    while IFS= read -r result_file; do
        cat "${result_file}" >> "${channel_dir}/per_episode.jsonl"
    done < <(find "${channel_dir}/chunks" -mindepth 2 -maxdepth 2 -name per_episode.jsonl | sort)
    if [[ $(wc -l < "${channel_dir}/per_episode.jsonl") -ne ${EPISODES_PER_CHANNEL} ]]; then
        echo "[eval] ERROR: ${channel_name} merged result count is not ${EPISODES_PER_CHANNEL}" >&2
        exit 1
    fi
}

run_channel none none
run_channel cov cov_only
run_channel inv inv_only
run_channel inv_cov inv_cov

echo "[eval] All 100 Task26 rollouts completed: ${OUTPUT_ROOT}"
