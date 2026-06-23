#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/limingxin/modelscope_cache/Qwen__Qwen3-4B}"
DATA_PATH="${DATA_PATH:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/test_1000.parquet}"
SKILL_FILE="${SKILL_FILE:-$REPO_ROOT/skills/search/search_skills_nl.md}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_base_eval_1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_DIR"

run_one() {
    local run_name="$1"
    local prompt_style="$2"
    local extra_args=()
    if [ -n "$LIMIT" ]; then
        extra_args+=(--limit "$LIMIT")
    fi
    echo "[$(date -u '+%F %T UTC')] start $run_name prompt_style=$prompt_style"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$PYTHON_BIN" "$REPO_ROOT/examples/eval_searchqa_direct_1000.py" \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$run_name" \
        --prompt_style "$prompt_style" \
        --skill_file "$SKILL_FILE" \
        --batch_size "$BATCH_SIZE" \
        --max_input_length "$MAX_INPUT_LENGTH" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        "${extra_args[@]}"
    echo "[$(date -u '+%F %T UTC')] done $run_name"
}

run_one "qwen3_4b_base_test1000_noskill" "dataset"
run_one "qwen3_4b_base_test1000_with_skill" "with_skill"

echo "[$(date -u '+%F %T UTC')] all evals finished"
