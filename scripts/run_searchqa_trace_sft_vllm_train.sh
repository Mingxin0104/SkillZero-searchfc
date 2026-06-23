#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set -euo pipefail

PYTHON="${PYTHON:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
OUT_DIR="${OUT_DIR:-/public/limingxin/SkillZero/data/searchqa_trace_sft_train_vllm}"
mkdir -p "$OUT_DIR/logs"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/examples/data_preprocess/generate_searchqa_skill_traces_vllm.py" \
    --model_path "${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}" \
    --input_path "${INPUT_PATH:-/home/limingxin/data/searchR1_processed_direct_skill0fmt/train.parquet}" \
    --output_path "$OUT_DIR/traces.jsonl" \
    --save_all_path "$OUT_DIR/traces_all.jsonl" \
    --sft_output_path "$OUT_DIR/train.parquet" \
    --skill_file "${SKILL_FILE:-$REPO_ROOT/skills/search/direct_retrieval.md}" \
    --search_url "${SEARCH_URL:-http://127.0.0.1:8000/retrieve}" \
    --start "${START:-0}" \
    --limit "${LIMIT:-117384}" \
    --batch_size "${BATCH_SIZE:-128}" \
    --num_candidates "${NUM_CANDIDATES:-3}" \
    --max_steps "${MAX_STEPS:-6}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
    --temperature "${TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.95}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-2}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.62}" \
    --max_model_len "${MAX_MODEL_LEN:-8192}" \
    --retrieval_topk "${RETRIEVAL_TOPK:-3}" \
    --retrieval_max_doc_chars "${RETRIEVAL_MAX_DOC_CHARS:-1800}" \
    --retrieval_workers "${RETRIEVAL_WORKERS:-64}" \
    --flush_every "${FLUSH_EVERY:-1}" \
    ${APPEND:+--append}
