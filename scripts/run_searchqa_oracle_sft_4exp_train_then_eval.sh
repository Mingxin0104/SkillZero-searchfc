#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/scripts/run_searchqa_oracle_sft_4exp_train_only.sh}"
MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_oracle_sft_4exp}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_oracle_sft_4exp}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_oracle_sft_4exp}"

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
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"

mkdir -p "$CKPT_ROOT" "$MERGED_ROOT" "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" >&2
}

latest_ckpt() {
    local save_dir="$1"
    find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1
}

require_ckpt() {
    local tag="$1"
    local ckpt
    ckpt="$(latest_ckpt "$CKPT_ROOT/$tag")"
    if [ -z "$ckpt" ]; then
        echo "Missing checkpoint for $tag under $CKPT_ROOT/$tag" >&2
        exit 1
    fi
    echo "$ckpt"
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

merge_lora_if_needed() {
    local tag="$1"
    local ckpt="$2"
    local merged_dir="$MERGED_ROOT/$tag"
    if [ -f "$merged_dir/config.json" ]; then
        echo "$merged_dir"
        return 0
    fi
    rm -rf "$merged_dir"
    log "Merging LoRA $tag from $ckpt"
    "$PYTHON_BIN" "$REPO_ROOT/examples/merge_searchqa_lora.py" \
        --base_model "$MODEL_PATH" \
        --adapter_path "$ckpt" \
        --output_path "$merged_dir"
    echo "$merged_dir"
}

eval_model() {
    local tag="$1"
    local mode="$2"
    local model_dir="$3"
    local use_skill="$4"
    local eval_log="$LOG_DIR/${tag}.eval.log"

    if [ -f "$eval_log" ] && rg -q "test_data/global_score/mean" "$eval_log"; then
        log "Skip eval $tag: metrics already exist in $eval_log"
        return 0
    fi

    ensure_retriever
    log "Start eval $tag use_skill=$use_skill"
    if [ "$mode" = "lora" ]; then
        SEARCH_URL="$SEARCH_URL" \
        DATA_ROOT="$DATA_ROOT" \
        LOG_DIR="$LOG_DIR" \
        EXP_LOG_NAME="${tag}.eval" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        ROLLOUT_GPU_MEMORY_UTILIZATION="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
        ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
        ROLLOUT_MAX_NUM_SEQS="$ROLLOUT_MAX_NUM_SEQS" \
        bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$model_dir" env.use_skill="$use_skill"
    else
        SEARCH_URL="$SEARCH_URL" \
        DATA_ROOT="$DATA_ROOT" \
        LOG_DIR="$LOG_DIR" \
        EXP_LOG_NAME="${tag}.eval" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        ROLLOUT_GPU_MEMORY_UTILIZATION="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
        ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
        ROLLOUT_MAX_NUM_SEQS="$ROLLOUT_MAX_NUM_SEQS" \
        bash "$REPO_ROOT/scripts/eval_searchqa_full_sft_mainppo.sh" "$model_dir" env.use_skill="$use_skill"
    fi
    log "Done eval $tag"
}

log "Stage 1/2: train four experiments sequentially"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MODEL_PATH="$MODEL_PATH" \
CKPT_ROOT="$CKPT_ROOT" \
LOG_DIR="$LOG_DIR" \
bash "$TRAIN_SCRIPT"

log "Stage 2/2: evaluate four experiments sequentially"

skill_lora_ckpt="$(require_ckpt skill_lora_r32_ep1)"
noskill_lora_ckpt="$(require_ckpt noskill_lora_r32_ep1)"
skill_full_ckpt="$(require_ckpt skill_full_ep1)"
noskill_full_ckpt="$(require_ckpt noskill_full_ep1)"

skill_lora_model="$(merge_lora_if_needed skill_lora_r32_ep1 "$skill_lora_ckpt")"
noskill_lora_model="$(merge_lora_if_needed noskill_lora_r32_ep1 "$noskill_lora_ckpt")"

eval_model "skill_lora_r32_ep1" "lora" "$skill_lora_model" True
eval_model "noskill_lora_r32_ep1" "lora" "$noskill_lora_model" False
eval_model "skill_full_ep1" "full" "$skill_full_ckpt" True
eval_model "noskill_full_ep1" "full" "$noskill_full_ckpt" False

log "All four train+eval experiments finished"
