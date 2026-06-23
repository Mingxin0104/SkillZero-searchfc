#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
INPUT_PATH="${INPUT_PATH:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/train.parquet}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
OUT_ROOT="${OUT_ROOT:-/workspace/limingxin/data/searchqa_qwen25_3b_grpo_skill0_full_n${NUM_CANDIDATES}}"
LOG_ROOT="${LOG_ROOT:-/workspace/limingxin/logs/searchqa_qwen25_3b_grpo_skill0_full_n${NUM_CANDIDATES}}"
SESSION_PREFIX="${SESSION_PREFIX:-qwen25_grpo_skill0_full_n${NUM_CANDIDATES}}"
TOTAL_LIMIT="${TOTAL_LIMIT:-117384}"
SHARD0_LIMIT="${SHARD0_LIMIT:-$((TOTAL_LIMIT / 2))}"
SHARD1_START="${SHARD1_START:-$SHARD0_LIMIT}"
SHARD1_LIMIT="${SHARD1_LIMIT:-$((TOTAL_LIMIT - SHARD1_START))}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" >&2
}

start_shard() {
    local shard_id="$1"
    local gpu="$2"
    local start="$3"
    local limit="$4"
    local out_dir="$OUT_ROOT/shard${shard_id}"
    local log_file="$LOG_ROOT/shard${shard_id}.log"
    local session="${SESSION_PREFIX}_shard${shard_id}"

    mkdir -p "$out_dir"
    tmux kill-session -t "$session" 2>/dev/null || true
    log "Starting shard${shard_id}: gpu=$gpu start=$start limit=$limit"
    tmux new-session -d -s "$session" \
        "cd '$REPO_ROOT' && CUDA_VISIBLE_DEVICES='$gpu' PYTHONPATH='$REPO_ROOT' '$PYTHON' examples/data_preprocess/generate_searchqa_skill_traces_vllm.py \
        --model_path '$MODEL_PATH' \
        --input_path '$INPUT_PATH' \
        --output_path '$out_dir/success_strict_em.jsonl' \
        --save_all_path '$out_dir/all_raw.jsonl' \
        --sft_output_path '$out_dir/train_strict_em.parquet' \
        --skill_mapping_file '$REPO_ROOT/skills/search/skill_mapping.json' \
        --search_url http://127.0.0.1:8000/retrieve \
        --start '$start' \
        --limit '$limit' \
        --batch_size '${BATCH_SIZE:-50}' \
        --num_candidates '$NUM_CANDIDATES' \
        --max_steps '${MAX_STEPS:-4}' \
        --max_new_tokens '${MAX_NEW_TOKENS:-160}' \
        --temperature '${TEMPERATURE:-0.7}' \
        --top_p '${TOP_P:-0.95}' \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization '${GPU_MEMORY_UTILIZATION:-0.50}' \
        --max_model_len '${MAX_MODEL_LEN:-8192}' \
        --retrieval_topk '${RETRIEVAL_TOPK:-3}' \
        --retrieval_max_doc_chars '${RETRIEVAL_MAX_DOC_CHARS:-1800}' \
        --retrieval_workers '${RETRIEVAL_WORKERS:-64}' \
        --flush_every 1 2>&1 | tee '$log_file'"
}

build_outputs() {
    "$PYTHON" examples/data_preprocess/build_searchqa_skill0_sft_from_all.py \
        --all_paths "$OUT_ROOT/shard0/all_raw.jsonl" "$OUT_ROOT/shard1/all_raw.jsonl" \
        --out_dir "$OUT_ROOT/merged"
}

status() {
    for shard_id in 0 1; do
        local log_file="$LOG_ROOT/shard${shard_id}.log"
        echo "== shard${shard_id} =="
        if [ -f "$log_file" ]; then
            grep -E 'processed=|saved=|Traceback|OutOfMemory|CUDA out|Error|Killed' "$log_file" | tail -n 20 || true
        else
            echo "no log yet"
        fi
    done
}

case "${1:-start}" in
    start)
        start_shard 0 0 0 "$SHARD0_LIMIT"
        start_shard 1 1 "$SHARD1_START" "$SHARD1_LIMIT"
        log "Started sessions: ${SESSION_PREFIX}_shard0 ${SESSION_PREFIX}_shard1"
        ;;
    build)
        build_outputs
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 [start|build|status]" >&2
        exit 1
        ;;
esac
