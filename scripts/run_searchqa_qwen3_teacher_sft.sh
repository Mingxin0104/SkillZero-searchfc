#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set -euo pipefail

PYTHON="${PYTHON:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_PATH="${MODEL_PATH:-}"
MODELSCOPE_CACHE_DIR="${MODELSCOPE_CACHE_DIR:-/workspace/limingxin/modelscope_cache}"
INPUT_PATH="${INPUT_PATH:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/train.parquet}"
OUT_DIR="${OUT_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_teacher_sft}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"

mkdir -p "$OUT_DIR/logs" "$MODELSCOPE_CACHE_DIR"
export MODEL_ID MODELSCOPE_CACHE_DIR

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$("$PYTHON" - <<'PY' | tail -n 1
from modelscope import snapshot_download
import os

model_id = os.environ["MODEL_ID"]
cache_dir = os.environ["MODELSCOPE_CACHE_DIR"]
local_dir = os.path.join(cache_dir, model_id.replace("/", "__"))
print(snapshot_download(model_id=model_id, cache_dir=cache_dir, local_dir=local_dir))
PY
)"
fi

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
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.82}" \
    --max_model_len "${MAX_MODEL_LEN:-8192}" \
    --retrieval_topk "${RETRIEVAL_TOPK:-3}" \
    --retrieval_max_doc_chars "${RETRIEVAL_MAX_DOC_CHARS:-1600}" \
    --retrieval_workers "${RETRIEVAL_WORKERS:-64}" \
    --flush_every "${FLUSH_EVERY:-1}" \
    ${APPEND:+--append}
