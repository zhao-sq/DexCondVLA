#!/usr/bin/env bash
set -uo pipefail

# Deterministic, zero-model-token health checks for the Task26 pipeline.
# A bounded `codex exec` repair run is launched only when an actionable incident
# is detected. Healthy checks never invoke Codex.

REPO_ROOT="${REPO_ROOT:-/home/shuqi/DexCondVLA}"
DEXTOOL_ROOT="${DEXTOOL_ROOT:-/mnt/ssd1/szhao/DexTool}"
REMOTE_HOST="${REMOTE_HOST:-root@136.175.252.91}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-35499}"
REMOTE_KEY="${REMOTE_KEY:-/home/shuqi/.ssh/vast_ai}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-task26_dual_expert_predcond_stopgrad_bs64_v1_20261012_235520}"
TRAIN_PROGRAM="${TRAIN_PROGRAM:-task26_dual_joint}"
EVAL_PROGRAM="${EVAL_PROGRAM:-task26_dual_best_policy}"
WANDB_PROGRAM="${WANDB_PROGRAM:-task26_joint_wandb}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
SELECTED_STEP="${SELECTED_STEP:-}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-7200}"
SSH_RETRIES="${SSH_RETRIES:-3}"
SSH_RETRY_SECONDS="${SSH_RETRY_SECONDS:-15}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-90m}"

STATE_DIR="${DEXTOOL_ROOT}/openpi_logs/${EXPERIMENT_NAME}_token_gated_repair"
MONITOR_LOG="${STATE_DIR}/monitor.log"
INCIDENT_REPORT="${STATE_DIR}/current_incident.txt"
LAST_STEP_FILE="${STATE_DIR}/last_training_step.env"
SSH_FAILURE_FILE="${STATE_DIR}/consecutive_ssh_failures"
NEEDS_USER_ACTION="${STATE_DIR}/NEEDS_USER_ACTION.md"
LOCK_FILE="${STATE_DIR}/monitor.lock"
REMOTE_QUEUE_DIR="/workspace/DexTool/queued_evaluations/${EXPERIMENT_NAME}"
REMOTE_COMPLETE="${REMOTE_QUEUE_DIR}/REMOTE_EVALUATION_COMPLETE.env"
LOCAL_QUEUE_SESSION="${LOCAL_QUEUE_SESSION:-task26_full_remote_eval_queue}"

RUN_ONCE=0
DRY_RUN=0
SIMULATE_ERROR=0
for arg in "$@"; do
    case "${arg}" in
        --once) RUN_ONCE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --simulate-error) SIMULATE_ERROR=1 ;;
        *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done
if (( SIMULATE_ERROR == 1 && DRY_RUN == 0 )); then
    echo "--simulate-error is allowed only with --dry-run" >&2
    exit 2
fi

mkdir -p "${STATE_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another token-gated Task26 monitor already holds ${LOCK_FILE}" >&2
    exit 0
fi

log() {
    printf '[token-gated-monitor] %s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${MONITOR_LOG}"
}

ssh_base=(
    ssh -i "${REMOTE_KEY}"
    -o IdentitiesOnly=yes
    -o BatchMode=yes
    -o ConnectTimeout=15
    -p "${REMOTE_SSH_PORT}"
)

remote_snapshot() {
    local attempt output
    for attempt in $(seq 1 "${SSH_RETRIES}"); do
        if output="$("${ssh_base[@]}" "${REMOTE_HOST}" bash -s -- \
            "${TRAIN_PROGRAM}" "${EVAL_PROGRAM}" "${WANDB_PROGRAM}" \
            "${EXPERIMENT_NAME}" "${REMOTE_QUEUE_DIR}" "${SELECTED_STEP}" 2>/dev/null <<'REMOTE'
set -u
train_program="$1"
eval_program="$2"
wandb_program="$3"
experiment_name="$4"
queue_dir="$5"
selected_step="$6"
train_log="/workspace/DexTool/openpi_logs/${experiment_name}.log"
checkpoint_root="/workspace/DexTool/ckp/pi05_bench2dex_task26_dual_expert_arm/${experiment_name}"

program_state() {
    supervisorctl status "$1" 2>/dev/null | awk -v name="$1" '$1 == name {print $2}' | tail -1
}

train_state="$(program_state "${train_program}")"
eval_state="$(program_state "${eval_program}")"
wandb_state="$(program_state "${wandb_program}")"
train_step="$(grep -oE 'step=[0-9]+' "${train_log}" 2>/dev/null | tail -1 | cut -d= -f2)"
latest="$(cat "${checkpoint_root}/LATEST" 2>/dev/null || true)"
best="$(cat "${checkpoint_root}/BEST" 2>/dev/null || true)"
available_gb="$(df --output=avail -BG /workspace 2>/dev/null | tail -1 | tr -dc '0-9')"

printf 'TRAIN_STATE=%s\n' "${train_state}"
printf 'EVAL_STATE=%s\n' "${eval_state}"
printf 'WANDB_STATE=%s\n' "${wandb_state}"
printf 'TRAIN_STEP=%s\n' "${train_step}"
printf 'LATEST=%s\n' "${latest}"
printf 'BEST=%s\n' "${best}"
printf 'AVAILABLE_GB=%s\n' "${available_gb}"
if test -n "${selected_step}" && test -f "${checkpoint_root}/${selected_step}/_CHECKPOINT_METADATA" \
    && test -s "${checkpoint_root}/${selected_step}/hand_dit.pt"; then
    printf 'SELECTED_CHECKPOINT_COMPLETE=yes\n'
else
    printf 'SELECTED_CHECKPOINT_COMPLETE=no\n'
fi
if test -s "${queue_dir}/REMOTE_ASSETS_READY"; then printf 'ASSETS_READY=yes\n'; else printf 'ASSETS_READY=no\n'; fi
if test -s "${queue_dir}/REMOTE_EVALUATION_FAILED.txt"; then printf 'REMOTE_FAILED=yes\n'; else printf 'REMOTE_FAILED=no\n'; fi
if test -s "${queue_dir}/REMOTE_EVALUATION_COMPLETE.env"; then
    printf 'REMOTE_COMPLETE=yes\n'
    sed -n 's/^result_name=/RESULT_NAME=/p' "${queue_dir}/REMOTE_EVALUATION_COMPLETE.env" | head -1
else
    printf 'REMOTE_COMPLETE=no\n'
    printf 'RESULT_NAME=\n'
fi
printf 'TRAIN_LOG_MTIME=%s\n' "$(stat -c %Y "${train_log}" 2>/dev/null || printf 0)"
printf 'EVAL_LOG_MTIME=%s\n' "$(stat -c %Y "${queue_dir}/remote_evaluation_queue.log" 2>/dev/null || printf 0)"
REMOTE
        )"; then
            printf '%s\n' "${output}"
            return 0
        fi
        log "SSH snapshot attempt ${attempt}/${SSH_RETRIES} failed"
        if (( attempt < SSH_RETRIES )); then
            sleep "${SSH_RETRY_SECONDS}"
        fi
    done
    return 1
}

snapshot_value() {
    local key="$1" snapshot="$2"
    awk -F= -v key="${key}" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' <<<"${snapshot}"
}

local_result_complete() {
    local result_name="$1" result_root video_count
    [[ -n "${result_name}" ]] || return 1
    result_root="${DEXTOOL_ROOT}/results/${result_name}"
    [[ -s "${result_root}/rollouts/success_summary.json" ]] || return 1
    [[ -s "${result_root}/rescore_without_object_upright/summary.json" ]] || return 1
    video_count="$(find "${result_root}/rollouts" -maxdepth 1 -type f -name 'rollout_*.mp4' 2>/dev/null | wc -l)"
    [[ "${video_count}" -eq 100 ]]
}

write_incident_report() {
    local snapshot="$1"
    shift
    {
        printf 'detected_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'experiment=%s\n' "${EXPERIMENT_NAME}"
        printf 'issues:\n'
        printf -- '- %s\n' "$@"
        printf '\nremote_snapshot:\n%s\n' "${snapshot}"
        printf '\nlocal_tmux:\n'
        tmux list-panes -t "${LOCAL_QUEUE_SESSION}" \
            -F '#{session_name} pid=#{pane_pid} dead=#{pane_dead} command=#{pane_current_command}' 2>&1 || true
        printf '\nrecent_watchdog_log:\n'
        tail -n 80 "${DEXTOOL_ROOT}/openpi_logs/${EXPERIMENT_NAME}_pipeline_watchdog.log" 2>&1 || true
        printf '\nrecent_transfer_log:\n'
        tail -n 80 "${DEXTOOL_ROOT}/openpi_logs/${EXPERIMENT_NAME}_remote_bulk_transfer.log" 2>&1 || true
        printf '\nrecent_diagnostics:\n'
        tail -n 160 "${DEXTOOL_ROOT}/openpi_logs/${EXPERIMENT_NAME}_pipeline_diagnostics.log" 2>&1 || true
    } >"${INCIDENT_REPORT}.tmp"
    mv "${INCIDENT_REPORT}.tmp" "${INCIDENT_REPORT}"
}

run_codex_repair() {
    local run_id codex_bin trace_file stderr_file final_file prompt exit_code
    run_id="$(date +%Y%m%d_%H%M%S)"
    trace_file="${STATE_DIR}/codex_repair_${run_id}.jsonl"
    stderr_file="${STATE_DIR}/codex_repair_${run_id}.stderr.log"
    final_file="${STATE_DIR}/codex_repair_${run_id}.final.txt"
    codex_bin="${CODEX_BIN:-$(command -v codex 2>/dev/null || true)}"

    prompt="You are a bounded incident-repair agent for the Task26 VLA + Hand-DiT pipeline. Read ${REPO_ROOT}/AGENTS.md if present and read /etc/vast-agents-guide.md on the rented server before any remote mutation. The incident report is ${INCIDENT_REPORT}. Inspect current authoritative process, log, disk, checkpoint, queue, and artifact state before acting. Diagnose the actual cause, make the smallest safe in-scope repair, and verify the pipeline is moving again. Preserve these hard constraints: never stop, restart, signal, or alter the formal trainer ${TRAIN_PROGRAM}; never stop or alter ${WANDB_PROGRAM}; never delete checkpoints, datasets, assets, results, or unknown data; never use destructive git commands; only the dedicated evaluation queue ${EVAL_PROGRAM} may be restarted if evidence requires it. You may patch the queue/watchdog scripts in ${REPO_ROOT} and deploy their corresponding dedicated evaluation configuration when necessary. Use the existing SSH endpoint, paths, and explicit virtual environments documented in the repository and incident report. If repair needs user approval, credentials, a destructive action, or broader authority, do not perform it; write the exact blocker and requested action to ${NEEDS_USER_ACTION}. End only after verifying the incident is resolved or recording that blocker. This is one bounded repair run, not a request to start or resume a Goal."

    if (( DRY_RUN == 1 )); then
        log "DRY RUN: would invoke Codex for incident=${INCIDENT_REPORT} trace=${trace_file}"
        return 0
    fi
    if [[ -z "${codex_bin}" || ! -x "${codex_bin}" ]]; then
        printf '# Codex repair could not start\n\nNo executable Codex CLI was found at %s.\n' \
            "${codex_bin:-<unset>}" >"${NEEDS_USER_ACTION}"
        log "Codex CLI unavailable; wrote ${NEEDS_USER_ACTION}"
        return 127
    fi

    rm -f "${NEEDS_USER_ACTION}"
    log "actionable incident detected; starting one bounded Codex repair run"
    if timeout "${CODEX_TIMEOUT}" "${codex_bin}" exec \
        --ephemeral \
        --approve-for-me \
        --cd "${REPO_ROOT}" \
        --add-dir "${DEXTOOL_ROOT}" \
        --json \
        --output-last-message "${final_file}" \
        "${prompt}" >"${trace_file}" 2>"${stderr_file}"; then
        exit_code=0
    else
        exit_code=$?
    fi
    log "Codex repair run exited with code=${exit_code}; trace=${trace_file}"
    return "${exit_code}"
}

check_once() {
    local snapshot train_state eval_state train_step available_gb assets_ready remote_failed
    local remote_complete result_name previous_step previous_checked_at now local_available_gb
    local selected_checkpoint_complete
    local ssh_failures=0
    local -a issues=()

    if ! snapshot="$(remote_snapshot)"; then
        if [[ -s "${SSH_FAILURE_FILE}" ]]; then
            ssh_failures="$(cat "${SSH_FAILURE_FILE}" 2>/dev/null || printf 0)"
        fi
        [[ "${ssh_failures}" =~ ^[0-9]+$ ]] || ssh_failures=0
        ssh_failures=$((ssh_failures + 1))
        printf '%s\n' "${ssh_failures}" >"${SSH_FAILURE_FILE}"
        log "remote snapshot unavailable (${ssh_failures} consecutive two-hour checks)"
        if (( ssh_failures < 2 )); then
            return 0
        fi
        snapshot="REMOTE_SNAPSHOT=unavailable"
        issues+=("Remote SSH remained unavailable across ${ssh_failures} scheduled checks")
    else
        printf '0\n' >"${SSH_FAILURE_FILE}"
        train_state="$(snapshot_value TRAIN_STATE "${snapshot}")"
        eval_state="$(snapshot_value EVAL_STATE "${snapshot}")"
        train_step="$(snapshot_value TRAIN_STEP "${snapshot}")"
        available_gb="$(snapshot_value AVAILABLE_GB "${snapshot}")"
        assets_ready="$(snapshot_value ASSETS_READY "${snapshot}")"
        remote_failed="$(snapshot_value REMOTE_FAILED "${snapshot}")"
        remote_complete="$(snapshot_value REMOTE_COMPLETE "${snapshot}")"
        result_name="$(snapshot_value RESULT_NAME "${snapshot}")"
        selected_checkpoint_complete="$(snapshot_value SELECTED_CHECKPOINT_COMPLETE "${snapshot}")"

        if [[ "${remote_complete}" == "yes" ]] && local_result_complete "${result_name}"; then
            log "pipeline complete and locally verified: ${DEXTOOL_ROOT}/results/${result_name}"
            return 10
        fi

        if [[ -n "${SELECTED_STEP}" ]]; then
            if [[ "${selected_checkpoint_complete}" != "yes" ]]; then
                issues+=("Explicitly selected checkpoint ${SELECTED_STEP} is incomplete or missing")
            fi
            if [[ "${train_state}" == "RUNNING" || "${train_state}" == "STARTING" ]]; then
                issues+=("Trainer unexpectedly restarted during selected-checkpoint evaluation; state=${train_state}")
            fi
        else
            case "${train_state}" in
                RUNNING|STARTING)
                    if [[ ! "${train_step}" =~ ^[0-9]+$ ]]; then
                        issues+=("Trainer is ${train_state}, but its latest logged step is unavailable")
                    elif [[ -s "${LAST_STEP_FILE}" ]]; then
                        previous_step="$(awk -F= '$1=="step" {print $2}' "${LAST_STEP_FILE}" 2>/dev/null)"
                        previous_checked_at="$(awk -F= '$1=="checked_at_epoch" {print $2}' "${LAST_STEP_FILE}" 2>/dev/null)"
                        now="$(date +%s)"
                        if [[ "${previous_step}" =~ ^[0-9]+$ && "${previous_checked_at}" =~ ^[0-9]+$ ]] \
                            && (( now - previous_checked_at >= CHECK_INTERVAL_SECONDS / 2 )) \
                            && (( train_step <= previous_step )); then
                            issues+=("Trainer remained at step ${train_step} across scheduled checks")
                        fi
                    fi
                    ;;
                EXITED)
                    if [[ ! "${train_step}" =~ ^[0-9]+$ ]] || (( train_step < TRAIN_STEPS )); then
                        issues+=("Trainer exited before the required ${TRAIN_STEPS} steps; latest=${train_step:-unknown}")
                    fi
                    ;;
                *) issues+=("Unexpected trainer Supervisor state: ${train_state:-unavailable}") ;;
            esac

            if [[ "${train_step}" =~ ^[0-9]+$ ]]; then
                printf 'step=%s\nchecked_at_epoch=%s\n' "${train_step}" "$(date +%s)" >"${LAST_STEP_FILE}.tmp"
                mv "${LAST_STEP_FILE}.tmp" "${LAST_STEP_FILE}"
            fi
        fi

        if [[ "${remote_failed}" == "yes" ]]; then
            issues+=("Remote evaluation wrote REMOTE_EVALUATION_FAILED.txt")
        fi
        if [[ "${remote_complete}" != "yes" ]]; then
            case "${eval_state}" in
                RUNNING|STARTING) ;;
                *) issues+=("Dedicated evaluation queue state is ${eval_state:-unavailable} before completion") ;;
            esac
            if ! tmux has-session -t "${LOCAL_QUEUE_SESSION}" 2>/dev/null; then
                issues+=("Local upload/download coordinator tmux session ${LOCAL_QUEUE_SESSION} is missing")
            fi
        fi

        if [[ "${assets_ready}" != "yes" && ( "${train_state}" == "EXITED" || -n "${SELECTED_STEP}" ) ]]; then
            if [[ ! "${available_gb}" =~ ^[0-9]+$ ]] || (( available_gb < 65 )); then
                issues+=("Remote space before asset upload is ${available_gb:-unknown}GB; 65GB is required")
            fi
        elif [[ "${assets_ready}" == "yes" ]]; then
            if [[ ! "${available_gb}" =~ ^[0-9]+$ ]] || (( available_gb < 15 )); then
                issues+=("Remote free space after asset upload is critically low: ${available_gb:-unknown}GB")
            fi
        fi

        local_available_gb="$(df --output=avail -BG "${DEXTOOL_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')"
        if [[ ! "${local_available_gb}" =~ ^[0-9]+$ ]] || (( local_available_gb < 10 )); then
            issues+=("Local DexTool free space is critically low: ${local_available_gb:-unknown}GB")
        fi
    fi

    if (( SIMULATE_ERROR == 1 )); then
        issues+=("Simulated incident used to validate token gating")
    fi
    if (( ${#issues[@]} == 0 )); then
        rm -f "${INCIDENT_REPORT}"
        log "healthy; no Codex invocation (trainer=${train_state:-unknown} step=${train_step:-unknown} eval=${eval_state:-unknown})"
        return 0
    fi

    write_incident_report "${snapshot}" "${issues[@]}"
    log "incident recorded with ${#issues[@]} issue(s): ${INCIDENT_REPORT}"
    run_codex_repair || true
    return 0
}

log "started; deterministic interval=${CHECK_INTERVAL_SECONDS}s; Codex runs only on actionable incidents"
while true; do
    check_once
    check_status=$?
    if (( check_status == 10 )); then
        exit 0
    fi
    if (( RUN_ONCE == 1 )); then
        exit 0
    fi
    log "sleeping ${CHECK_INTERVAL_SECONDS}s until the next zero-token health check"
    sleep "${CHECK_INTERVAL_SECONDS}"
done
