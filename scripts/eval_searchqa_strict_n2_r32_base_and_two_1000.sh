#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/limingxin/SkillZero}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt}"
TEST_FILE="${TEST_FILE:-$DATA_ROOT/test_1000.parquet}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen25_3b_grpo_skill0_n2_strict_eval_1000}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"

BASE_MODEL="${BASE_MODEL:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_qwen25_3b_grpo_skill0_n2_strict_lora_r32}"

mkdir -p "$LOG_DIR"

run_one() {
    local name="$1"
    local model_path="$2"
    local gpu="$3"
    local metric_file="$LOG_DIR/${name}.search_rate.json"
    local gen_dir="$LOG_DIR/${name}.generations"
    local ray_tmp="/tmp/r${gpu}_${RANDOM}"

    rm -rf "$gen_dir" "$ray_tmp"
    mkdir -p "$ray_tmp"
    rm -f "$metric_file"

    echo "[$(date -u '+%F %T UTC')] start eval $name on GPU $gpu" | tee -a "$LOG_DIR/driver.log"
    RAY_DEDUP_LOGS=0 \
    SEARCH_URL="$SEARCH_URL" \
    DATA_ROOT="$DATA_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXP_LOG_NAME="${name}.eval" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    N_GPUS_PER_NODE=1 \
    RAY_NUM_CPUS=4 \
    ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}" \
    ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}" \
    bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$model_path" \
        env.use_skill=False \
        data.test_files="$TEST_FILE" \
        data.val_batch_size=32 \
        env.rollout.n=1 \
        +ray_init.include_dashboard=False \
        +ray_init._temp_dir="$ray_tmp" \
        trainer.validation_data_dir="$gen_dir"

    "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_search_rate.py" \
        --generation_dir "$gen_dir/test" \
        --use_trajectory_metrics \
        --out "$metric_file"
    echo "[$(date -u '+%F %T UTC')] done eval $name" | tee -a "$LOG_DIR/driver.log"
}

run_pair() {
    run_one "base_qwen25_3b" "$BASE_MODEL" 0 > "$LOG_DIR/base_qwen25_3b.driver.stdout.log" 2>&1 &
    local pid_a=$!
    run_one "strict_with_skill_lora_r32_ep1" "$MERGED_ROOT/strict_with_skill_lora_r32_ep1" 1 > "$LOG_DIR/strict_with_skill_lora_r32_ep1.driver.stdout.log" 2>&1 &
    local pid_b=$!
    wait "$pid_a"
    wait "$pid_b"
}

run_pair
run_one "strict_no_skill_lora_r32_ep1" "$MERGED_ROOT/strict_no_skill_lora_r32_ep1" 0 > "$LOG_DIR/strict_no_skill_lora_r32_ep1.driver.stdout.log" 2>&1

echo "[$(date -u '+%F %T UTC')] all evals finished" | tee -a "$LOG_DIR/driver.log"
