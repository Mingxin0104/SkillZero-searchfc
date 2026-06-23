#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/limingxin/SkillZero}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_oracle_sft_4exp}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_oracle_sft_4exp}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"

mkdir -p "$LOG_DIR"

run_one() {
    local tag="$1"
    local gpu="$2"
    local use_skill="$3"
    local model_dir="$MERGED_ROOT/$tag"

    if [ ! -f "$model_dir/config.json" ]; then
        echo "Missing merged model: $model_dir" >&2
        exit 1
    fi

    rm -rf "$LOG_DIR/${tag}.generations"
    rm -f "$LOG_DIR/${tag}.eval.log" "$LOG_DIR/${tag}.search_rate.json"

    SEARCH_URL="$SEARCH_URL" \
    DATA_ROOT="$DATA_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXP_LOG_NAME="${tag}.eval" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    N_GPUS_PER_NODE=1 \
    RAY_NUM_CPUS=4 \
    ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}" \
    ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-128}" \
    bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$model_dir" \
        env.use_skill="$use_skill" \
        trainer.validation_data_dir="$LOG_DIR/${tag}.generations"

    "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_search_rate.py" \
        --generation_dir "$LOG_DIR/${tag}.generations/test" \
        --out "$LOG_DIR/${tag}.search_rate.json"
}

run_one "$@"
