#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set -euo pipefail

OUT_DIR="${OUT_DIR:-/public/limingxin/SkillZero/data/searchqa_trace_sft_train}"
LOG="$OUT_DIR/logs/wait_and_build.log"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
mkdir -p "$(dirname "$LOG")"

while true; do
    free0="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    free1="$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    ts="$(date -Is)"
    echo "$ts free0=${free0}MiB free1=${free1}MiB min=${MIN_FREE_MIB}MiB" >> "$LOG"

    if [ "$free0" -ge "$MIN_FREE_MIB" ] && [ "$free1" -ge "$MIN_FREE_MIB" ]; then
        echo "$ts starting full SearchQA trace-SFT build" >> "$LOG"
        OUT_DIR="$OUT_DIR" bash "$REPO_ROOT/scripts/build_searchqa_trace_sft_train.sh" >> "$LOG" 2>&1
        exit $?
    fi

    sleep "$SLEEP_SECONDS"
done
