#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/limingxin/SkillZero}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_noanswer_skill_lora_sweep}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt}"
TEST_FILE="${TEST_FILE:-$DATA_ROOT/test_1000.parquet}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_noanswer_skill_lora_sweep_eval_1000}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"

mkdir -p "$LOG_DIR"

run_one() {
    local rank="$1"
    local gpu="$2"
    local tag="skill_lora_r${rank}_ep1"
    local model_dir="$MERGED_ROOT/$tag"
    local metric_file="$LOG_DIR/${tag}.search_rate.json"

    rm -rf "$LOG_DIR/${tag}.generations"
    rm -f "$metric_file"
    mkdir -p "/tmp/ray_${tag}"

    echo "[$(date -u '+%F %T UTC')] start $tag on GPU $gpu" | tee -a "$LOG_DIR/driver.log"
    RAY_DEDUP_LOGS=0 \
    SEARCH_URL="$SEARCH_URL" \
    DATA_ROOT="$DATA_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXP_LOG_NAME="${tag}.eval" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    N_GPUS_PER_NODE=1 \
    RAY_NUM_CPUS=4 \
    ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}" \
    ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}" \
    bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$model_dir" \
        env.use_skill=False \
        data.test_files="$TEST_FILE" \
        data.val_batch_size=32 \
        env.rollout.n=1 \
        +ray_init.include_dashboard=False \
        +ray_init._temp_dir="/tmp/ray_${tag}" \
        trainer.validation_data_dir="$LOG_DIR/${tag}.generations"

    "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_search_rate.py" \
        --generation_dir "$LOG_DIR/${tag}.generations/test" \
        --out "$metric_file"
    echo "[$(date -u '+%F %T UTC')] done $tag" | tee -a "$LOG_DIR/driver.log"
}

run_pair() {
    local rank_a="$1"
    local gpu_a="$2"
    local rank_b="$3"
    local gpu_b="$4"
    run_one "$rank_a" "$gpu_a" > "$LOG_DIR/skill_lora_r${rank_a}_ep1.driver.stdout.log" 2>&1 &
    local pid_a=$!
    run_one "$rank_b" "$gpu_b" > "$LOG_DIR/skill_lora_r${rank_b}_ep1.driver.stdout.log" 2>&1 &
    local pid_b=$!
    wait "$pid_a"
    wait "$pid_b"
}

run_pair 8 0 16 1
run_pair 32 0 64 1

echo "[$(date -u '+%F %T UTC')] all 1000-sample evals finished" | tee -a "$LOG_DIR/driver.log"
