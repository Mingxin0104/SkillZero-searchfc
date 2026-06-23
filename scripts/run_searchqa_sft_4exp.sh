#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set -euo pipefail
set -x

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/searchR1_processed_direct}"
SFT_DATA_DIR="${SFT_DATA_DIR:-$DATA_ROOT/searchqa_skill0_sft_full}"
SKILL0FMT_DATA_DIR="${SKILL0FMT_DATA_DIR:-${DATA_ROOT}_skill0fmt}"
CKPT_ROOT="${CKPT_ROOT:-/public/limingxin/SkillZero/checkpoints}"
MERGED_ROOT="${MERGED_ROOT:-/public/limingxin/SkillZero/merged_models}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVER_HEALTH_URL="${RETRIEVER_HEALTH_URL:-http://127.0.0.1:8000/docs}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-/workspace/limingxin/miniconda3/envs/retriever/bin/python}"
RETRIEVER_INDEX_PATH="${RETRIEVER_INDEX_PATH:-$HOME/data/searchR1/e5_Flat.index}"
RETRIEVER_CORPUS_PATH="${RETRIEVER_CORPUS_PATH:-$HOME/data/searchR1/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_MODEL_CACHE_ROOT="${RETRIEVER_MODEL_CACHE_ROOT:-/public/limingxin/SkillZero/hf_cache/hub/models--intfloat--e5-base-v2}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
RETRIEVER_LOG="${RETRIEVER_LOG:-$LOG_DIR/retrieval_server.auto.log}"
mkdir -p "$LOG_DIR" "$CKPT_ROOT" "$MERGED_ROOT"

wait_for_retriever() {
    local retries="${1:-60}"
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

ensure_retriever() {
    resolve_retriever_model

    if curl -fsS -m 2 "$RETRIEVER_HEALTH_URL" >/dev/null 2>&1; then
        echo "Retriever already available at $RETRIEVER_HEALTH_URL"
        return 0
    fi

    nohup "$RETRIEVER_PYTHON" "$REPO_ROOT/examples/search/retriever/retrieval_server.py" \
        --index_path "$RETRIEVER_INDEX_PATH" \
        --corpus_path "$RETRIEVER_CORPUS_PATH" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL" \
        --faiss_gpu \
        --port "$RETRIEVER_PORT" \
        > "$RETRIEVER_LOG" 2>&1 &

    if ! wait_for_retriever 60 5; then
        echo "Retriever failed to start. Check $RETRIEVER_LOG" >&2
        exit 1
    fi
}

prepare_data() {
    PYTHONPATH="$REPO_ROOT" /workspace/limingxin/miniconda3/envs/skillzero/bin/python \
        "$REPO_ROOT/examples/data_preprocess/prepare_searchqa_skill0_sft.py" \
        --input_dir "$DATA_ROOT" \
        --output_dir "$SFT_DATA_DIR"

    PYTHONPATH="$REPO_ROOT" /workspace/limingxin/miniconda3/envs/skillzero/bin/python \
        "$REPO_ROOT/examples/data_preprocess/fix_searchqa_skill0_ground_truth.py" \
        --input_dir "$DATA_ROOT" \
        --output_dir "$SKILL0FMT_DATA_DIR"
}

run_full() {
    local epochs="$1"
    local exp_name="searchqa-full-sft-ep${epochs}"
    local save_dir="$CKPT_ROOT/${exp_name}"
    local eval_log="$LOG_DIR/${exp_name}-mainppo-testonly.log"
    local latest_ckpt

    latest_ckpt="$(find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1 || true)"
    if [ -z "$latest_ckpt" ]; then
        TRAIN_DATA="$SFT_DATA_DIR/train.parquet" \
        VAL_DATA="$SFT_DATA_DIR/val_1000.parquet" \
        MODEL_ID="$MODEL_ID" \
        MODEL_PATH="$MODEL_PATH" \
        EXP_NAME="$exp_name" \
        TOTAL_EPOCHS="$epochs" \
        bash "$REPO_ROOT/scripts/train_searchqa_full_sft_full.sh" "$save_dir"
        latest_ckpt="$(find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    fi

    if [ -z "$latest_ckpt" ]; then
        echo "No full-SFT checkpoint found under $save_dir" >&2
        exit 1
    fi

    if [ -f "$eval_log" ] && rg -q "test_data/global_score/mean" "$eval_log"; then
        echo "Skipping full-SFT eval for $exp_name because $eval_log already contains test metrics"
        return 0
    fi

    ensure_retriever
    SEARCH_URL="$SEARCH_URL" \
    EXP_LOG_NAME="${exp_name}-mainppo-testonly" \
    bash "$REPO_ROOT/scripts/eval_searchqa_full_sft_mainppo.sh" "$latest_ckpt"
}

run_lora() {
    local epochs="$1"
    local exp_name="searchqa-lora-r32-ep${epochs}"
    local save_dir="$CKPT_ROOT/${exp_name}"
    local merged_dir="$MERGED_ROOT/${exp_name}"
    local eval_log="$LOG_DIR/${exp_name}-mainppo-testonly.log"
    local latest_ckpt

    latest_ckpt="$(find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1 || true)"
    if [ -z "$latest_ckpt" ]; then
        TRAIN_DATA="$SFT_DATA_DIR/train.parquet" \
        VAL_DATA="$SFT_DATA_DIR/val_1000.parquet" \
        MODEL_ID="$MODEL_ID" \
        MODEL_PATH="$MODEL_PATH" \
        EXP_NAME="$exp_name" \
        TOTAL_EPOCHS="$epochs" \
        bash "$REPO_ROOT/scripts/train_searchqa_lora_sft_r32_full.sh" "$save_dir"
        latest_ckpt="$(find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    fi

    if [ -z "$latest_ckpt" ]; then
        echo "No LoRA checkpoint found under $save_dir" >&2
        exit 1
    fi

    if [ ! -f "$merged_dir/config.json" ]; then
        rm -rf "$merged_dir"
        PYTHONPATH="$REPO_ROOT" /workspace/limingxin/miniconda3/envs/skillzero/bin/python \
            "$REPO_ROOT/examples/merge_searchqa_lora.py" \
            --base_model "$MODEL_PATH" \
            --adapter_path "$latest_ckpt" \
            --output_path "$merged_dir"
    fi

    if [ -f "$eval_log" ] && rg -q "test_data/global_score/mean" "$eval_log"; then
        echo "Skipping LoRA eval for $exp_name because $eval_log already contains test metrics"
        return 0
    fi

    ensure_retriever
    SEARCH_URL="$SEARCH_URL" \
    EXP_LOG_NAME="${exp_name}-mainppo-testonly" \
    bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$merged_dir"
}

prepare_data
run_full 1
run_full 2
run_lora 1
run_lora 2
