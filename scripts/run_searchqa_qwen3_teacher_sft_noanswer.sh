#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_PATH="${MODEL_PATH:-}"
MODELSCOPE_CACHE_DIR="${MODELSCOPE_CACHE_DIR:-/workspace/limingxin/modelscope_cache}"
INPUT_PATH="${INPUT_PATH:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/train.parquet}"
OUT_DIR="${OUT_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_teacher_sft_noanswer}"
NOSKILL_OUT_DIR="${NOSKILL_OUT_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_teacher_sft_noanswer_noskill}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"

RETRIEVER_HEALTH_URL="${RETRIEVER_HEALTH_URL:-http://127.0.0.1:8000/docs}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-/workspace/limingxin/miniconda3/envs/retriever/bin/python}"
RETRIEVER_INDEX_PATH="${RETRIEVER_INDEX_PATH:-/home/limingxin/data/searchR1/e5_Flat.index}"
RETRIEVER_CORPUS_PATH="${RETRIEVER_CORPUS_PATH:-/home/limingxin/data/searchR1/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_MODEL_CACHE_ROOT="${RETRIEVER_MODEL_CACHE_ROOT:-/public/limingxin/SkillZero/hf_cache/hub/models--intfloat--e5-base-v2}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
RETRIEVER_LOG="${RETRIEVER_LOG:-$OUT_DIR/logs/retrieval_server.log}"

mkdir -p "$OUT_DIR/logs" "$NOSKILL_OUT_DIR" "$MODELSCOPE_CACHE_DIR"
export MODEL_ID MODELSCOPE_CACHE_DIR

log() {
    echo "[$(date -u '+%F %T UTC')] $*" >&2
}

resolve_modelscope_model() {
    if [ -n "$MODEL_PATH" ]; then
        return 0
    fi
    MODEL_PATH="$("$PYTHON" - <<'PY' | tail -n 1
from modelscope import snapshot_download
import os

model_id = os.environ["MODEL_ID"]
cache_dir = os.environ["MODELSCOPE_CACHE_DIR"]
local_dir = os.path.join(cache_dir, model_id.replace("/", "__"))
print(snapshot_download(model_id=model_id, cache_dir=cache_dir, local_dir=local_dir))
PY
)"
}

resolve_retriever_model() {
    if [ -d "$RETRIEVER_MODEL" ]; then
        return 0
    fi
    if [ -d "$RETRIEVER_MODEL_CACHE_ROOT/snapshots" ]; then
        local snapshot
        snapshot="$(find "$RETRIEVER_MODEL_CACHE_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
        if [ -n "$snapshot" ]; then
            RETRIEVER_MODEL="$snapshot"
            export RETRIEVER_MODEL
        fi
    fi
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
    resolve_retriever_model
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

    if ! wait_for_retriever 90 5; then
        echo "Retriever failed to start. Check $RETRIEVER_LOG" >&2
        exit 1
    fi
}

write_skill_val() {
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

resolve_modelscope_model
ensure_retriever

log "Generating no-answer teacher SFT data into $OUT_DIR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONPATH="$REPO_ROOT" \
"$PYTHON" "$REPO_ROOT/examples/data_preprocess/generate_searchqa_teacher_sft_vllm.py" \
    --model_path "$MODEL_PATH" \
    --input_path "$INPUT_PATH" \
    --out_dir "$OUT_DIR" \
    --skill_dir "${SKILL_DIR:-$REPO_ROOT/skills/search}" \
    --search_url "$SEARCH_URL" \
    --start "${START:-0}" \
    --limit "${LIMIT:-117384}" \
    --batch_size "${BATCH_SIZE:-128}" \
    --num_candidates "${NUM_CANDIDATES:-4}" \
    --max_steps "${MAX_STEPS:-5}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-160}" \
    --temperature "${TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.95}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-2}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
    --max_model_len "${MAX_MODEL_LEN:-8192}" \
    --retrieval_topk "${RETRIEVAL_TOPK:-3}" \
    --retrieval_max_doc_chars "${RETRIEVAL_MAX_DOC_CHARS:-1600}" \
    --retrieval_workers "${RETRIEVAL_WORKERS:-64}" \
    --flush_every "${FLUSH_EVERY:-1}" \
    ${APPEND:+--append}

write_skill_val

log "Building no-skill version into $NOSKILL_OUT_DIR"
"$PYTHON" "$REPO_ROOT/examples/data_preprocess/make_searchqa_teacher_noskill.py" \
    --input_jsonl "$OUT_DIR/accepted.jsonl" \
    --out_dir "$NOSKILL_OUT_DIR" \
    --val_size 1000

log "Done no-answer SearchQA teacher data"
