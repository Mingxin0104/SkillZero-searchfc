#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/workspace/limingxin/modelscope_cache/Qwen__Qwen3-4B}"
TRAINED_WITH_SKILL_MODEL_PATH="${TRAINED_WITH_SKILL_MODEL_PATH:-/workspace/limingxin/merged_models/searchqa_qwen3_4b_noanswer_lora_r8_two/qwen3_4b_noanswer_with_skill_lora_r8_ep1}"
TRAINED_NO_SKILL_MODEL_PATH="${TRAINED_NO_SKILL_MODEL_PATH:-/workspace/limingxin/merged_models/searchqa_qwen3_4b_noanswer_lora_r8_two/qwen3_4b_noanswer_no_skill_lora_r8_ep1}"
DATA_PATH="${DATA_PATH:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/test_1000.parquet}"
SKILL_FILE="${SKILL_FILE:-$REPO_ROOT/skills/search/search_skills_nl.md}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_four_runs_eval_1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
LIMIT="${LIMIT:-}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

mkdir -p "$OUTPUT_DIR"

run_one() {
    local run_name="$1"
    local model_path="$2"
    local prompt_style="$3"
    local gpu="$4"
    local extra_args=()
    if [ -n "$LIMIT" ]; then
        extra_args+=(--limit "$LIMIT")
    fi

    echo "[$(date -u '+%F %T UTC')] start $run_name gpu=$gpu model=$model_path prompt_style=$prompt_style" | tee -a "$OUTPUT_DIR/driver.log"
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" "$REPO_ROOT/examples/eval_searchqa_direct_1000.py" \
        --model_path "$model_path" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$run_name" \
        --prompt_style "$prompt_style" \
        --skill_file "$SKILL_FILE" \
        --batch_size "$BATCH_SIZE" \
        --max_input_length "$MAX_INPUT_LENGTH" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        "${extra_args[@]}" \
        2>&1 | tee "$OUTPUT_DIR/${run_name}.stdout.log"
    echo "[$(date -u '+%F %T UTC')] done $run_name" | tee -a "$OUTPUT_DIR/driver.log"
}

run_pair() {
    local run_name_a="$1"
    local model_path_a="$2"
    local prompt_style_a="$3"
    local run_name_b="$4"
    local model_path_b="$5"
    local prompt_style_b="$6"

    run_one "$run_name_a" "$model_path_a" "$prompt_style_a" "$GPU_A" &
    local pid_a=$!
    run_one "$run_name_b" "$model_path_b" "$prompt_style_b" "$GPU_B" &
    local pid_b=$!

    wait "$pid_a"
    wait "$pid_b"
}

run_pair \
    "qwen3_4b_trained_noskill_test1000" "$TRAINED_NO_SKILL_MODEL_PATH" "dataset" \
    "qwen3_4b_trained_with_skill_test1000" "$TRAINED_WITH_SKILL_MODEL_PATH" "with_skill"

run_pair \
    "qwen3_4b_base_test1000" "$BASE_MODEL_PATH" "dataset" \
    "qwen3_4b_base_with_skill_test1000" "$BASE_MODEL_PATH" "with_skill"

echo "[$(date -u '+%F %T UTC')] all four evals finished" | tee -a "$OUTPUT_DIR/driver.log"
