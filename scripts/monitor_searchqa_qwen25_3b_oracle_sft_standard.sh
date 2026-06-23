#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen25_3b_oracle_sft_standard_sharded}"
OUT_DIR="${OUT_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_oracle_sft_standard}"
NOSKILL_OUT_DIR="${NOSKILL_OUT_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_oracle_sft_standard_noskill}"
SHARD0_DIR="${SHARD0_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_oracle_sft_standard_shard0}"
SHARD1_DIR="${SHARD1_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_oracle_sft_standard_shard1}"
SHARD1_LOG="${SHARD1_LOG:-$LOG_DIR/shard1.log}"
IDLE_UTIL_THRESHOLD="${IDLE_UTIL_THRESHOLD:-10}"
IDLE_MEM_THRESHOLD_MIB="${IDLE_MEM_THRESHOLD_MIB:-25000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG_DIR/monitor.log"
}

gpu1_is_idle() {
    local line mem util
    line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' '$1 ~ /1/ {print $0}')"
    mem="$(echo "$line" | awk -F',' '{gsub(/ /, "", $2); print $2}')"
    util="$(echo "$line" | awk -F',' '{gsub(/ /, "", $3); print $3}')"
    [ -n "$mem" ] && [ -n "$util" ] || return 1
    [ "$mem" -le "$IDLE_MEM_THRESHOLD_MIB" ] && [ "$util" -le "$IDLE_UTIL_THRESHOLD" ]
}

while true; do
    if [ -f "$SHARD1_DIR/accepted.jsonl" ]; then
        log "shard1 output already exists"
        break
    fi
    if gpu1_is_idle; then
        log "GPU1 is idle enough, starting shard1"
        bash "$REPO_ROOT/scripts/run_searchqa_qwen25_3b_oracle_sft_standard_sharded.sh" shard1 >"$SHARD1_LOG" 2>&1
        log "shard1 finished"
        break
    fi
    log "GPU1 still busy, sleep ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
done

if [ -f "$SHARD0_DIR/accepted.jsonl" ] && [ -f "$SHARD1_DIR/accepted.jsonl" ]; then
    log "both shards ready, merging"
    OUT_DIR="$OUT_DIR" \
    NOSKILL_OUT_DIR="$NOSKILL_OUT_DIR" \
    SHARD0_DIR="$SHARD0_DIR" \
    SHARD1_DIR="$SHARD1_DIR" \
    LOG_DIR="$LOG_DIR" \
    bash "$REPO_ROOT/scripts/run_searchqa_qwen25_3b_oracle_sft_standard_sharded.sh" merge >>"$LOG_DIR/monitor.log" 2>&1
    log "merge finished"
else
    log "merge skipped because shard outputs are incomplete"
fi
