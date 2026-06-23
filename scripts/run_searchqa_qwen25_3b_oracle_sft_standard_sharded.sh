#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
INPUT_PATH="${INPUT_PATH:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/train.parquet}"
OUT_DIR="${OUT_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_oracle_sft_standard}"
NOSKILL_OUT_DIR="${NOSKILL_OUT_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_oracle_sft_standard_noskill}"
SHARD0_DIR="${SHARD0_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_oracle_sft_standard_shard0}"
SHARD1_DIR="${SHARD1_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_oracle_sft_standard_shard1}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen25_3b_oracle_sft_standard_sharded}"

TOTAL_ROWS="${TOTAL_ROWS:-117384}"
SHARD_ROWS="${SHARD_ROWS:-58692}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"

BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.62}"
RETRIEVAL_WORKERS="${RETRIEVAL_WORKERS:-96}"
RETRIEVAL_TOPK="${RETRIEVAL_TOPK:-5}"
RETRIEVAL_MAX_DOC_CHARS="${RETRIEVAL_MAX_DOC_CHARS:-1800}"

mkdir -p "$OUT_DIR" "$NOSKILL_OUT_DIR" "$SHARD0_DIR" "$SHARD1_DIR" "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" >&2
}

run_shard() {
    local shard_id="$1"
    local gpu="$2"
    local out_dir start limit log_file

    if [ "$shard_id" = "0" ]; then
        out_dir="$SHARD0_DIR"
        start=0
        limit="$SHARD_ROWS"
    else
        out_dir="$SHARD1_DIR"
        start="$SHARD_ROWS"
        limit="$((TOTAL_ROWS - SHARD_ROWS))"
    fi
    log_file="$LOG_DIR/shard${shard_id}.log"

    mkdir -p "$out_dir"
    log "Start shard${shard_id} gpu=$gpu start=$start limit=$limit out=$out_dir"
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$REPO_ROOT" \
    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/generate_searchqa_oracle_sft_vllm.py" \
        --model_path "$MODEL_PATH" \
        --input_path "$INPUT_PATH" \
        --out_dir "$out_dir" \
        --skill_dir "$REPO_ROOT/skills/search" \
        --search_url "$SEARCH_URL" \
        --start "$start" \
        --limit "$limit" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature 0.0 \
        --top_p 1.0 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --max_model_len "$MAX_MODEL_LEN" \
        --retrieval_topk "$RETRIEVAL_TOPK" \
        --retrieval_max_doc_chars "$RETRIEVAL_MAX_DOC_CHARS" \
        --retrieval_workers "$RETRIEVAL_WORKERS" \
        2>&1 | tee "$log_file"
}

merge_outputs() {
    if [ ! -f "$SHARD0_DIR/accepted.jsonl" ] || [ ! -f "$SHARD1_DIR/accepted.jsonl" ]; then
        echo "Both shard outputs are required before merge." >&2
        exit 1
    fi

    log "Merging shards into $OUT_DIR"
    rm -f "$OUT_DIR/accepted.jsonl" "$OUT_DIR/all_candidates.jsonl" "$OUT_DIR/train.parquet" "$OUT_DIR/val_1000.parquet"
    rm -f "$NOSKILL_OUT_DIR/accepted.jsonl" "$NOSKILL_OUT_DIR/train.parquet" "$NOSKILL_OUT_DIR/val_1000.parquet"

    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/merge_searchqa_teacher_sft_shards.py" \
        --shard_dirs "$SHARD0_DIR" "$SHARD1_DIR" \
        --out_dir "$OUT_DIR"

    OUT_DIR="$OUT_DIR" "$PYTHON" - <<'PY'
import os
import pandas as pd
out_dir = os.environ["OUT_DIR"]
train_path = os.path.join(out_dir, "train.parquet")
val_path = os.path.join(out_dir, "val_1000.parquet")
df = pd.read_parquet(train_path)
df.head(min(1000, len(df))).to_parquet(val_path, index=False)
print(f"saved_val={val_path} rows={min(1000, len(df))}")
PY

    log "Building no-skill version into $NOSKILL_OUT_DIR"
    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/make_searchqa_teacher_noskill.py" \
        --input_jsonl "$OUT_DIR/accepted.jsonl" \
        --out_dir "$NOSKILL_OUT_DIR" \
        --val_size 1000
}

case "${1:-}" in
    shard0)
        GPU="${GPU:-0}"
        run_shard 0 "$GPU"
        ;;
    shard1)
        GPU="${GPU:-1}"
        run_shard 1 "$GPU"
        ;;
    merge)
        merge_outputs
        ;;
    all)
        GPU0="${GPU0:-0}"
        GPU1="${GPU1:-1}"
        run_shard 0 "$GPU0" &
        pid0=$!
        run_shard 1 "$GPU1" &
        pid1=$!
        wait "$pid0"
        wait "$pid1"
        merge_outputs
        ;;
    *)
        echo "Usage: $0 {shard0|shard1|merge|all}" >&2
        exit 1
        ;;
esac
