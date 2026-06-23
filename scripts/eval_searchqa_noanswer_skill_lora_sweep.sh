#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/limingxin/SkillZero}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-/workspace/limingxin/miniconda3/envs/retriever/bin/python}"
BASE_MODEL="${BASE_MODEL:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_noanswer_skill_lora_sweep}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_noanswer_skill_lora_sweep}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_noanswer_skill_lora_sweep_eval}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt}"

SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVER_HEALTH_URL="${RETRIEVER_HEALTH_URL:-http://127.0.0.1:8000/docs}"
RETRIEVER_INDEX_PATH="${RETRIEVER_INDEX_PATH:-/public/limingxin/SkillZero/data/searchR1/e5_Flat.index}"
RETRIEVER_CORPUS_PATH="${RETRIEVER_CORPUS_PATH:-/public/limingxin/SkillZero/data/searchR1/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_MODEL_CACHE_ROOT="${RETRIEVER_MODEL_CACHE_ROOT:-/public/limingxin/SkillZero/hf_cache/hub/models--intfloat--e5-base-v2}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
RETRIEVER_LOG="${RETRIEVER_LOG:-$LOG_DIR/retrieval_server.log}"
RETRIEVER_GPU="${RETRIEVER_GPU:-0}"
RETRIEVER_FAISS_GPU="${RETRIEVER_FAISS_GPU:-false}"
EVAL_GPU="${EVAL_GPU:-1}"
RANKS="${RANKS:-8 16 32 64}"
EVAL_VAL_BATCH_SIZE="${EVAL_VAL_BATCH_SIZE:-16}"
EVAL_ROLLOUT_N="${EVAL_ROLLOUT_N:-1}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-8}"

mkdir -p "$MERGED_ROOT" "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG_DIR/driver.log"
}

latest_ckpt() {
    find "$CKPT_ROOT/$1" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1
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
        log "Retriever already available"
        return 0
    fi

    log "Starting retriever on GPU $RETRIEVER_GPU faiss_gpu=$RETRIEVER_FAISS_GPU"
    local retriever_args=(
        "$REPO_ROOT/examples/search/retriever/retrieval_server.py"
        --index_path "$RETRIEVER_INDEX_PATH" \
        --corpus_path "$RETRIEVER_CORPUS_PATH" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL" \
        --port "$RETRIEVER_PORT" \
    )
    if [ "$RETRIEVER_FAISS_GPU" = "true" ]; then
        retriever_args+=(--faiss_gpu)
    fi
    CUDA_VISIBLE_DEVICES="$RETRIEVER_GPU" nohup "$RETRIEVER_PYTHON" "${retriever_args[@]}" > "$RETRIEVER_LOG" 2>&1 &
    wait_for_retriever 120 5
}

merge_one() {
    local rank="$1"
    local tag="skill_lora_r${rank}_ep1"
    local ckpt
    ckpt="$(latest_ckpt "$tag")"
    if [ -z "$ckpt" ]; then
        echo "Missing checkpoint for $tag under $CKPT_ROOT/$tag" >&2
        exit 1
    fi

    local out="$MERGED_ROOT/$tag"
    if [ -f "$out/config.json" ]; then
        log "Skip merge $tag: $out exists"
        return 0
    fi

    log "Merging $tag from $ckpt"
    rm -rf "$out"
    "$PYTHON_BIN" "$REPO_ROOT/examples/merge_searchqa_lora.py" \
        --base_model "$BASE_MODEL" \
        --adapter_path "$ckpt" \
        --output_path "$out" \
        2>&1 | tee "$LOG_DIR/${tag}.merge.log"
}

eval_one() {
    local rank="$1"
    local tag="skill_lora_r${rank}_ep1"
    local model_dir="$MERGED_ROOT/$tag"
    local metric_file="$LOG_DIR/${tag}.search_rate.json"

    if [ -f "$metric_file" ]; then
        log "Skip eval $tag: $metric_file exists"
        return 0
    fi

    log "Evaluating $tag on GPU $EVAL_GPU with env.use_skill=False"
    rm -rf "$LOG_DIR/${tag}.generations"
    SEARCH_URL="$SEARCH_URL" \
    DATA_ROOT="$DATA_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXP_LOG_NAME="${tag}.eval" \
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
    N_GPUS_PER_NODE=1 \
    RAY_NUM_CPUS=4 \
    ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}" \
    ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-$EVAL_MAX_NUM_SEQS}" \
    bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$model_dir" \
        env.use_skill=False \
        +env.query_only_prompt=True \
        data.val_batch_size="$EVAL_VAL_BATCH_SIZE" \
        env.rollout.n="$EVAL_ROLLOUT_N" \
        trainer.validation_data_dir="$LOG_DIR/${tag}.generations"

    "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_search_rate.py" \
        --generation_dir "$LOG_DIR/${tag}.generations/test" \
        --out "$metric_file"
}

log "SearchQA no-answer skill LoRA sweep eval started"
for rank in $RANKS; do
    merge_one "$rank"
done

ensure_retriever

for rank in $RANKS; do
    eval_one "$rank"
done

log "All SearchQA no-answer skill LoRA sweep evals finished"
