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
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_noanswer_noskill_lora_sweep}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_noanswer_noskill_lora_sweep}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt}"
TEST_FILE="${TEST_FILE:-$DATA_ROOT/test_1000.parquet}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_noanswer_noskill_lora_sweep_eval_1000}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVER_HEALTH_URL="${RETRIEVER_HEALTH_URL:-http://127.0.0.1:8000/docs}"
RETRIEVER_INDEX_PATH="${RETRIEVER_INDEX_PATH:-/public/limingxin/SkillZero/data/searchR1/e5_Flat.index}"
RETRIEVER_CORPUS_PATH="${RETRIEVER_CORPUS_PATH:-/public/limingxin/SkillZero/data/searchR1/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_MODEL_CACHE_ROOT="${RETRIEVER_MODEL_CACHE_ROOT:-/public/limingxin/SkillZero/hf_cache/hub/models--intfloat--e5-base-v2}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
RETRIEVER_LOG="${RETRIEVER_LOG:-$LOG_DIR/retrieval_server.log}"
RANKS="${RANKS:-8 16 32 64}"

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
    local retries="${1:-120}"
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

    log "Starting GPU retriever on CUDA_VISIBLE_DEVICES=0,1"
    FAISS_GPU_NO_TEMP_MEMORY="${FAISS_GPU_NO_TEMP_MEMORY:-1}" \
    CUDA_VISIBLE_DEVICES=0,1 \
    nohup "$RETRIEVER_PYTHON" "$REPO_ROOT/examples/search/retriever/retrieval_server.py" \
        --index_path "$RETRIEVER_INDEX_PATH" \
        --corpus_path "$RETRIEVER_CORPUS_PATH" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL" \
        --port "$RETRIEVER_PORT" \
        --faiss_gpu \
        > "$RETRIEVER_LOG" 2>&1 &

    wait_for_retriever 120 5
    log "Retriever is ready"
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

run_one() {
    local rank="$1"
    local gpu="$2"
    local tag="skill_lora_r${rank}_ep1"
    local model_dir="$MERGED_ROOT/$tag"
    local metric_file="$LOG_DIR/${tag}.search_rate.json"

    if [ -f "$metric_file" ]; then
        log "Skip eval $tag: $metric_file exists"
        return 0
    fi

    rm -rf "$LOG_DIR/${tag}.generations"
    rm -f "$metric_file"
    local ray_tmp="/tmp/ray_nsn_r${rank}"
    mkdir -p "$ray_tmp"

    log "Start eval $tag on GPU $gpu"
    RAY_DEDUP_LOGS=0 \
    SEARCH_URL="$SEARCH_URL" \
    DATA_ROOT="$DATA_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXP_LOG_NAME="${tag}.eval" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    N_GPUS_PER_NODE=1 \
    RAY_NUM_CPUS=4 \
    ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}" \
    ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}" \
    bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$model_dir" \
        env.use_skill=False \
        data.test_files="$TEST_FILE" \
        data.val_batch_size=32 \
        env.rollout.n=1 \
        +ray_init.include_dashboard=False \
        +ray_init._temp_dir="$ray_tmp" \
        trainer.validation_data_dir="$LOG_DIR/${tag}.generations"

    "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_search_rate.py" \
        --generation_dir "$LOG_DIR/${tag}.generations/test" \
        --out "$metric_file"
    log "Done eval $tag"
}

run_pair() {
    local rank_a="$1"
    local gpu_a="$2"
    local rank_b="$3"
    local gpu_b="$4"
    run_one "$rank_a" "$gpu_a" > "$LOG_DIR/skill_lora_r${rank_a}_ep1.driver.stdout.log" 2>&1 &
    local pid_a=$!
    run_one "$rank_b" "$gpu_b" > "$LOG_DIR/skill_lora_r${rank_b}_ep1.driver.stdout.log" 2>&1 &
    local pid_b=$!
    wait "$pid_a"
    wait "$pid_b"
}

log "SearchQA no-answer noskill LoRA 1000 eval started"
for rank in $RANKS; do
    merge_one "$rank"
done

ensure_retriever

run_pair 8 0 16 1
run_pair 32 0 64 1

log "All SearchQA no-answer noskill LoRA 1000 evals finished"
