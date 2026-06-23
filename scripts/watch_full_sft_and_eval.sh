#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-/public/limingxin/SkillZero/checkpoints/searchqa_full_sft_full}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_ROOT/scripts/eval_searchqa_full_sft_mainppo.sh}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-$REPO_ROOT/log}"

mkdir -p "$EVAL_LOG_DIR"

last_seen=""
while true; do
    latest="$(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1 || true)"
    if [ -n "$latest" ] && [ "$latest" != "$last_seen" ]; then
        last_seen="$latest"
        step_name="$(basename "$latest")"
        export EXP_LOG_NAME="searchqa-full-sft-mainppo-eval-${step_name}"
        echo "[$(date -u +'%F %T')] detected checkpoint: $latest"
        bash "$EVAL_SCRIPT" "$latest"
        exit $?
    fi
    sleep "$POLL_SECONDS"
done
