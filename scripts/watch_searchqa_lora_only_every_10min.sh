#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_oracle_sft_4exp}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_oracle_sft_4exp}"
WATCH_LOG="${WATCH_LOG:-$LOG_DIR/lora_only_10min_watch.log}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$WATCH_LOG"
}

while true; do
    log "========== 10min status =========="
    log "tmux:"
    tmux ls 2>&1 | tee -a "$WATCH_LOG" || true

    log "gpu:"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1 | tee -a "$WATCH_LOG" || true

    log "checkpoints:"
    find "$CKPT_ROOT" -maxdepth 2 -type d -name 'global_step_*' -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort -V | tee -a "$WATCH_LOG" || true

    log "noskill_lora tail:"
    if [ -f "$LOG_DIR/noskill_lora_r32_ep1.log" ]; then
        grep -aE 'step:[0-9]+|saved|Done|OutOfMemory|Traceback|ChildFailedError|Error executing job' "$LOG_DIR/noskill_lora_r32_ep1.log" | tail -n 20 | tee -a "$WATCH_LOG" || true
    fi

    log "lora_only_after_two tail:"
    if [ -f "$LOG_DIR/lora_only_after_two.log" ]; then
        tail -n 40 "$LOG_DIR/lora_only_after_two.log" | tee -a "$WATCH_LOG" || true
    fi

    log "search_rate summaries:"
    for f in "$LOG_DIR"/*.search_rate.json; do
        [ -f "$f" ] || continue
        log "$f"
        cat "$f" | tee -a "$WATCH_LOG"
    done

    if [ -f "$LOG_DIR/lora_only_after_two.log" ] && grep -aq "LoRA-only continuation finished" "$LOG_DIR/lora_only_after_two.log"; then
        log "LoRA-only eval finished; watch exiting"
        exit 0
    fi
    log "=================================="
    sleep "$INTERVAL_SECONDS"
done
