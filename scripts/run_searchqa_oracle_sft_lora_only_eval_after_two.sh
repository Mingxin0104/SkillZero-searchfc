#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/limingxin/SkillZero}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_oracle_sft_4exp}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_oracle_sft_4exp}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_oracle_sft_4exp}"
PIPELINE_SESSION="${PIPELINE_SESSION:-searchqa_oracle_sft_4exp_pipeline}"

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

DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"

mkdir -p "$MERGED_ROOT" "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" >&2
}

latest_ckpt() {
    find "$CKPT_ROOT/$1" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1
}

require_ckpt() {
    local tag="$1"
    local ckpt
    ckpt="$(latest_ckpt "$tag")"
    if [ -z "$ckpt" ]; then
        echo "Missing checkpoint for $tag under $CKPT_ROOT/$tag" >&2
        exit 1
    fi
    echo "$ckpt"
}

wait_for_ckpt() {
    local tag="$1"
    local sleep_seconds="${2:-60}"
    while true; do
        if latest_ckpt "$tag" >/dev/null && [ -n "$(latest_ckpt "$tag")" ]; then
            latest_ckpt "$tag"
            return 0
        fi
        log "Waiting for checkpoint: $tag"
        sleep "$sleep_seconds"
    done
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
    log "Starting retriever"
    nohup "$RETRIEVER_PYTHON" "$REPO_ROOT/examples/search/retriever/retrieval_server.py" \
        --index_path "$RETRIEVER_INDEX_PATH" \
        --corpus_path "$RETRIEVER_CORPUS_PATH" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL" \
        --faiss_gpu \
        --port "$RETRIEVER_PORT" \
        > "$RETRIEVER_LOG" 2>&1 &
    wait_for_retriever 90 5
}

merge_lora() {
    local tag="$1"
    local ckpt="$2"
    local out="$MERGED_ROOT/$tag"
    if [ -f "$out/config.json" ]; then
        echo "$out"
        return 0
    fi
    rm -rf "$out"
    log "Merging $tag"
    "$PYTHON_BIN" "$REPO_ROOT/examples/merge_searchqa_lora.py" \
        --base_model "$MODEL_PATH" \
        --adapter_path "$ckpt" \
        --output_path "$out"
    echo "$out"
}

eval_lora() {
    local tag="$1"
    local model_dir="$2"
    local use_skill="$3"
    local eval_log="$LOG_DIR/${tag}.eval.log"
    if [ -f "$eval_log" ] && rg -q "test_data/global_score/mean" "$eval_log"; then
        log "Skip eval $tag: metrics already exist"
        return 0
    fi
    ensure_retriever
    log "Eval $tag use_skill=$use_skill"
    SEARCH_URL="$SEARCH_URL" \
    DATA_ROOT="$DATA_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXP_LOG_NAME="${tag}.eval" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    ROLLOUT_GPU_MEMORY_UTILIZATION="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    ROLLOUT_MAX_NUM_SEQS="$ROLLOUT_MAX_NUM_SEQS" \
    bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$model_dir" \
        env.use_skill="$use_skill" \
        trainer.validation_data_dir="$LOG_DIR/${tag}.generations"

    "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_search_rate.py" \
        --generation_dir "$LOG_DIR/${tag}.generations/test" \
        --out "$LOG_DIR/${tag}.search_rate.json"
}

log "LoRA-only continuation started"
skill_ckpt="$(require_ckpt skill_lora_r32_ep1)"
noskill_ckpt="$(wait_for_ckpt noskill_lora_r32_ep1 60)"

log "Both LoRA checkpoints are ready. Stopping original full pipeline before it starts full SFT."
tmux kill-session -t "$PIPELINE_SESSION" 2>/dev/null || true
sleep 10

skill_model="$(merge_lora skill_lora_r32_ep1 "$skill_ckpt")"
noskill_model="$(merge_lora noskill_lora_r32_ep1 "$noskill_ckpt")"

eval_lora skill_lora_r32_ep1 "$skill_model" True
eval_lora noskill_lora_r32_ep1 "$noskill_model" False

log "LoRA-only continuation finished"
