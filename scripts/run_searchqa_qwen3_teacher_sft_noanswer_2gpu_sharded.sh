#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
INPUT_PATH="${INPUT_PATH:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/train.parquet}"
BASE_OUT_DIR="${BASE_OUT_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_teacher_sft_noanswer}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-$BASE_OUT_DIR}"
MERGED_NOSKILL_OUT_DIR="${MERGED_NOSKILL_OUT_DIR:-${BASE_OUT_DIR}_noskill}"
LOG_ROOT="${LOG_ROOT:-/workspace/limingxin/logs/searchqa_noanswer_data_gen_2gpu}"
SESSION_PREFIX="${SESSION_PREFIX:-searchqa_noanswer_2gpu}"
TOTAL_LIMIT="${TOTAL_LIMIT:-117384}"
SHARD0_LIMIT="${SHARD0_LIMIT:-$((TOTAL_LIMIT / 2))}"
SHARD1_START="${SHARD1_START:-$SHARD0_LIMIT}"
SHARD1_LIMIT="${SHARD1_LIMIT:-$((TOTAL_LIMIT - SHARD1_START))}"

mkdir -p "$LOG_ROOT"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" >&2
}

start_shard() {
    local shard_id="$1"
    local gpu="$2"
    local start="$3"
    local limit="$4"
    local out_dir="${BASE_OUT_DIR}_shard${shard_id}"
    local noskill_dir="${BASE_OUT_DIR}_shard${shard_id}_noskill"
    local log_file="$LOG_ROOT/shard${shard_id}.log"
    local session="${SESSION_PREFIX}_shard${shard_id}"

    tmux kill-session -t "$session" 2>/dev/null || true
    rm -rf "$out_dir" "$noskill_dir"
    log "Starting shard${shard_id} gpu=$gpu start=$start limit=$limit out=$out_dir"
    tmux new-session -d -s "$session" \
        "cd '$REPO_ROOT' && CUDA_VISIBLE_DEVICES='$gpu' TENSOR_PARALLEL_SIZE=1 GPU_MEMORY_UTILIZATION='${GPU_MEMORY_UTILIZATION:-0.82}' START='$start' LIMIT='$limit' OUT_DIR='$out_dir' NOSKILL_OUT_DIR='$noskill_dir' BATCH_SIZE='${BATCH_SIZE:-192}' NUM_CANDIDATES='${NUM_CANDIDATES:-4}' MAX_STEPS='${MAX_STEPS:-5}' MAX_NEW_TOKENS='${MAX_NEW_TOKENS:-160}' bash scripts/run_searchqa_qwen3_teacher_sft_noanswer.sh 2>&1 | tee '$log_file'"
}

merge_outputs() {
    local shard0="${BASE_OUT_DIR}_shard0"
    local shard1="${BASE_OUT_DIR}_shard1"
    if [ ! -f "$shard0/train.parquet" ] || [ ! -f "$shard1/train.parquet" ]; then
        echo "Shard outputs are incomplete; cannot merge yet." >&2
        exit 1
    fi

    rm -rf "$MERGED_OUT_DIR" "$MERGED_NOSKILL_OUT_DIR"
    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/merge_searchqa_teacher_sft_shards.py" \
        --shard_dirs "$shard0" "$shard1" \
        --out_dir "$MERGED_OUT_DIR"

    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/filter_searchqa_evidence_supported.py" \
        --input_jsonl "$MERGED_OUT_DIR/accepted.jsonl" \
        --out_dir "$MERGED_OUT_DIR" \
        --val_size 1000

    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/make_searchqa_teacher_noskill.py" \
        --input_jsonl "$MERGED_OUT_DIR/accepted.jsonl" \
        --out_dir "$MERGED_NOSKILL_OUT_DIR" \
        --val_size 1000
}

case "${1:-start}" in
    start)
        start_shard 0 0 0 "$SHARD0_LIMIT"
        start_shard 1 1 "$SHARD1_START" "$SHARD1_LIMIT"
        log "Started two shard sessions: ${SESSION_PREFIX}_shard0 and ${SESSION_PREFIX}_shard1"
        ;;
    merge)
        merge_outputs
        ;;
    *)
        echo "Usage: $0 [start|merge]" >&2
        exit 1
        ;;
esac
