#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/limingxin/SkillZero}"
SESSION="${SESSION:-searchqa_oracle_sft_4exp_pipeline}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-$REPO_ROOT/scripts/run_searchqa_oracle_sft_4exp_train_then_eval.sh}"
START_SCRIPT="${START_SCRIPT:-$REPO_ROOT/scripts/start_searchqa_oracle_sft_4exp_train_then_eval_tmux.sh}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_oracle_sft_4exp}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_oracle_sft_4exp}"
MONITOR_LOG="${MONITOR_LOG:-$LOG_DIR/hourly_monitor.log}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$MONITOR_LOG"
}

current_train_log() {
    local latest
    latest="$(find "$LOG_DIR" -maxdepth 1 -type f -name '*_ep1.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}')"
    echo "$latest"
}

report_status() {
    log "========== status =========="
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        log "pipeline_session=alive"
    else
        log "pipeline_session=missing"
    fi

    log "gpu_status:"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1 | tee -a "$MONITOR_LOG" || true

    log "checkpoints:"
    find "$CKPT_ROOT" -maxdepth 2 -type d -name 'global_step_*' -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort -V | tee -a "$MONITOR_LOG" || true

    local train_log
    train_log="$(current_train_log)"
    if [ -n "$train_log" ] && [ -f "$train_log" ]; then
        log "current_train_log=$train_log"
        grep -aE 'Start |Done |Total training steps|step:[0-9]+|global_step_|CUDA out of memory|Traceback|ChildFailedError|RuntimeError|Error executing job' "$train_log" \
            | tail -n 30 | tee -a "$MONITOR_LOG" || true
    else
        log "current_train_log=none"
    fi

    log "eval_metrics:"
    grep -aR "test_data/global_score/mean" "$LOG_DIR"/*.eval.log 2>/dev/null | tail -n 20 | tee -a "$MONITOR_LOG" || true
    log "============================"
}

pipeline_done() {
    [ -f "$LOG_DIR/pipeline.log" ] && grep -aq "All four train+eval experiments finished" "$LOG_DIR/pipeline.log"
}

pipeline_failed() {
    local train_log
    train_log="$(current_train_log)"
    if [ -n "$train_log" ] && [ -f "$train_log" ]; then
        if grep -aqE 'CUDA out of memory|ChildFailedError|Traceback|Error executing job|RuntimeError' "$train_log"; then
            return 0
        fi
    fi
    [ -f "$LOG_DIR/pipeline.log" ] && grep -aqE 'CUDA out of memory|ChildFailedError|Traceback|Error executing job|RuntimeError|Missing checkpoint|failed to start' "$LOG_DIR/pipeline.log"
}

lower_batch_once() {
    local script="$REPO_ROOT/scripts/run_searchqa_oracle_sft_4exp_train_only.sh"
    if grep -q 'noskill_lora_r32_ep1.*1536 24 768' "$script"; then
        log "Applying OOM recovery batch profile: level 1"
        sed -i \
            -e 's/run_exp "noskill_lora_r32_ep1" "$NOSKILL_DATA_DIR" "lora" 1536 24 768 1e-4/run_exp "noskill_lora_r32_ep1" "$NOSKILL_DATA_DIR" "lora" 1536 8 256 1e-4/' \
            -e 's/run_exp "skill_full_ep1" "$SKILL_DATA_DIR" "full" 2304 6 192 1e-5/run_exp "skill_full_ep1" "$SKILL_DATA_DIR" "full" 2304 2 64 1e-5/' \
            -e 's/run_exp "noskill_full_ep1" "$NOSKILL_DATA_DIR" "full" 1536 12 384 1e-5/run_exp "noskill_full_ep1" "$NOSKILL_DATA_DIR" "full" 1536 4 128 1e-5/' \
            "$script"
        return 0
    fi
    if grep -q 'noskill_lora_r32_ep1.*1536 8 256' "$script"; then
        log "Applying OOM recovery batch profile: level 2"
        sed -i \
            -e 's/run_exp "noskill_lora_r32_ep1" "$NOSKILL_DATA_DIR" "lora" 1536 8 256 1e-4/run_exp "noskill_lora_r32_ep1" "$NOSKILL_DATA_DIR" "lora" 1536 4 128 1e-4/' \
            -e 's/run_exp "skill_full_ep1" "$SKILL_DATA_DIR" "full" 2304 2 64 1e-5/run_exp "skill_full_ep1" "$SKILL_DATA_DIR" "full" 2304 1 32 1e-5/' \
            -e 's/run_exp "noskill_full_ep1" "$NOSKILL_DATA_DIR" "full" 1536 4 128 1e-5/run_exp "noskill_full_ep1" "$NOSKILL_DATA_DIR" "full" 1536 2 64 1e-5/' \
            "$script"
        return 0
    fi
    log "Batch profile already at lowest configured level"
    return 1
}

restart_pipeline() {
    local reason="$1"
    log "Restarting pipeline: $reason"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 5
    SESSION="$SESSION" REPO_ROOT="$REPO_ROOT" LOG_DIR="$LOG_DIR" bash "$START_SCRIPT" 2>&1 | tee -a "$MONITOR_LOG"
}

restarts=0
while true; do
    report_status

    if pipeline_done; then
        log "pipeline_done=true; monitor exiting"
        exit 0
    fi

    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        if [ "$restarts" -lt "$MAX_RESTARTS" ]; then
            restarts=$((restarts + 1))
            restart_pipeline "session missing restart=$restarts"
        else
            log "session missing but max restarts reached; monitor exiting with failure"
            exit 1
        fi
    elif pipeline_failed; then
        if [ "$restarts" -lt "$MAX_RESTARTS" ]; then
            restarts=$((restarts + 1))
            lower_batch_once || true
            restart_pipeline "failure detected restart=$restarts"
        else
            log "failure detected but max restarts reached; monitor exiting with failure"
            exit 1
        fi
    fi

    sleep "$INTERVAL_SECONDS"
done
