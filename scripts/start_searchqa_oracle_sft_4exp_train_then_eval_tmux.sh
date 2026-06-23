#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

SESSION="${SESSION:-searchqa_oracle_sft_4exp_pipeline}"
REPO_ROOT="${REPO_ROOT:-/workspace/limingxin/SkillZero}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_oracle_sft_4exp}"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    echo "attach: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" \
    "cd '$REPO_ROOT' && bash scripts/run_searchqa_oracle_sft_4exp_train_then_eval.sh 2>&1 | tee '$PIPELINE_LOG'"

echo "started tmux session: $SESSION"
echo "pipeline log: $PIPELINE_LOG"
echo "attach: tmux attach -t $SESSION"
