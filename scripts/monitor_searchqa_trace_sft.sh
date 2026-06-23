#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUT_DIR="${OUT_DIR:-/public/limingxin/SkillZero/data/searchqa_trace_sft_train}"
INTERVAL="${INTERVAL:-30}"
LOG="$OUT_DIR/logs/progress.log"
mkdir -p "$(dirname "$LOG")"

while true; do
    ts="$(date -Is)"
    gpu="$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
    workers="$(pgrep -af 'generate_searchqa_skill_traces_local.py' | wc -l)"
    done="$(find "$OUT_DIR/done" -type f -name '*.done' 2>/dev/null | wc -l)"
    sft="$(find "$OUT_DIR/sft" -type f -name '*.parquet' 2>/dev/null | wc -l)"
    success_lines="$(find "$OUT_DIR/traces" -type f -name '*.jsonl' -print0 2>/dev/null | xargs -0 -r wc -l | tail -n 1 | awk '{print $1}')"
    all_lines="$(find "$OUT_DIR/traces_all" -type f -name '*.jsonl' -print0 2>/dev/null | xargs -0 -r wc -l | tail -n 1 | awk '{print $1}')"
    success_lines="${success_lines:-0}"
    all_lines="${all_lines:-0}"
    echo "$ts workers=$workers done_shards=$done sft_shards=$sft success_traces=$success_lines all_candidates=$all_lines gpu=[$gpu]" | tee -a "$LOG"
    sleep "$INTERVAL"
done
