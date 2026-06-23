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
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen25_3b_oracle_sft_standard_2gpu}"

TOTAL_ROWS="${TOTAL_ROWS:-117384}"
SHARD_ROWS="${SHARD_ROWS:-58692}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVER_HEALTH_URL="${RETRIEVER_HEALTH_URL:-http://127.0.0.1:8000/docs}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-/workspace/limingxin/miniconda3/envs/retriever/bin/python}"
RETRIEVER_INDEX_PATH="${RETRIEVER_INDEX_PATH:-/public/limingxin/SkillZero/data/searchR1/e5_Flat.index}"
RETRIEVER_CORPUS_PATH="${RETRIEVER_CORPUS_PATH:-/public/limingxin/SkillZero/data/searchR1/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-/public/limingxin/SkillZero/hf_cache/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-5}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
RETRIEVER_LOG="${RETRIEVER_LOG:-$LOG_DIR/retrieval_server.log}"

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

wait_for_retriever() {
    local retries="${1:-90}"
    local sleep_seconds="${2:-5}"
    local i
    for ((i=1; i<=retries; i++)); do
        if curl -fsS -m 2 "$RETRIEVER_HEALTH_URL" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$sleep_seconds"
    done
    return 1
}

ensure_retriever() {
    if curl -fsS -m 2 "$RETRIEVER_HEALTH_URL" >/dev/null 2>&1; then
        log "Retriever already available at $RETRIEVER_HEALTH_URL"
        return 0
    fi
    log "Starting retriever on port $RETRIEVER_PORT"
    nohup "$RETRIEVER_PYTHON" "$REPO_ROOT/examples/search/retriever/retrieval_server.py" \
        --index_path "$RETRIEVER_INDEX_PATH" \
        --corpus_path "$RETRIEVER_CORPUS_PATH" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL" \
        --faiss_gpu \
        --port "$RETRIEVER_PORT" \
        > "$RETRIEVER_LOG" 2>&1 &
    wait_for_retriever 90 5 || {
        echo "Retriever failed to start. Check $RETRIEVER_LOG" >&2
        exit 1
    }
}

run_shard() {
    local gpu="$1"
    local start="$2"
    local limit="$3"
    local out_dir="$4"
    local log_file="$5"

    log "Start shard gpu=$gpu start=$start limit=$limit out=$out_dir"
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

write_val() {
    "$PYTHON" - <<'PY'
import os
import pandas as pd

out_dir = os.environ["OUT_DIR"]
train_path = os.path.join(out_dir, "train.parquet")
val_path = os.path.join(out_dir, "val_1000.parquet")
df = pd.read_parquet(train_path)
df.head(min(1000, len(df))).to_parquet(val_path, index=False)
print(f"saved_val={val_path} rows={min(1000, len(df))}")
PY
}

ensure_retriever

log "Generating Qwen2.5-3B oracle-standard SearchQA data"
run_shard 0 0 "$SHARD_ROWS" "$SHARD0_DIR" "$LOG_DIR/shard0.log" &
pid0=$!
run_shard 1 "$SHARD_ROWS" "$((TOTAL_ROWS - SHARD_ROWS))" "$SHARD1_DIR" "$LOG_DIR/shard1.log" &
pid1=$!

wait "$pid0"
wait "$pid1"

log "Merging shards into $OUT_DIR"
"$PYTHON" "$REPO_ROOT/examples/data_preprocess/merge_searchqa_teacher_sft_shards.py" \
    --shard_dirs "$SHARD0_DIR" "$SHARD1_DIR" \
    --out_dir "$OUT_DIR"

export OUT_DIR
write_val

log "Building no-skill version into $NOSKILL_OUT_DIR"
"$PYTHON" "$REPO_ROOT/examples/data_preprocess/make_searchqa_teacher_noskill.py" \
    --input_jsonl "$OUT_DIR/accepted.jsonl" \
    --out_dir "$NOSKILL_OUT_DIR" \
    --val_size 1000

log "Done Qwen2.5-3B oracle-standard SearchQA data"
