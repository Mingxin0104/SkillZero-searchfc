#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source "$SCRIPT_DIR/searchfc_runtime_env.sh"

PYTHON="${PYTHON:-$VLLM_ENV_ROOT/bin/python}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen3.5-4B}"
RAW_TRAIN="${RAW_TRAIN:-$DATA_HOME/nq_hotpotqa_train_raw/train.parquet}"
RAW_TEST="${RAW_TEST:-$DATA_HOME/nq_hotpotqa_train_raw/test.parquet}"
INPUT_DIR="${INPUT_DIR:-$DATA_HOME/nq_hotpot_qwen35_skill0fmt_input_10k1k}"
INPUT_PATH="${INPUT_PATH:-$INPUT_DIR/train.parquet}"
WITH_SKILL_OUT_DIR="${WITH_SKILL_OUT_DIR:-$DATA_HOME/nq_hotpot_qwen35_4b_teacher_sft_20k}"
NO_SKILL_OUT_DIR="${NO_SKILL_OUT_DIR:-${WITH_SKILL_OUT_DIR}_noskill}"
WITH_SKILL_FMT_DIR="${WITH_SKILL_FMT_DIR:-${WITH_SKILL_OUT_DIR}_skill0fmt}"
NO_SKILL_FMT_DIR="${NO_SKILL_FMT_DIR:-${NO_SKILL_OUT_DIR}_skill0fmt}"
LOG_ROOT="${LOG_ROOT:-$LOG_HOME/nq_hotpot_qwen35_teacher_sft_2gpu}"
SESSION_PREFIX="${SESSION_PREFIX:-nq_hotpot_qwen35_teacher}"
TOTAL_LIMIT="${TOTAL_LIMIT:-20000}"
SHARD0_LIMIT="${SHARD0_LIMIT:-$((TOTAL_LIMIT / 2))}"
SHARD1_START="${SHARD1_START:-$SHARD0_LIMIT}"
SHARD1_LIMIT="${SHARD1_LIMIT:-$((TOTAL_LIMIT - SHARD1_START))}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVER_HEALTH_URL="${RETRIEVER_HEALTH_URL:-http://127.0.0.1:8000/docs}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-$RETRIEVER_ENV/bin/python}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
RETRIEVER_VISIBLE_DEVICES="${RETRIEVER_VISIBLE_DEVICES:-0,1}"
RETRIEVER_LOG="${RETRIEVER_LOG:-$LOG_ROOT/retrieval_server.log}"
VLLM_BIN="${VLLM_BIN:-$VLLM_ENV_ROOT/bin/vllm}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-$MODEL_PATH}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-8100}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

mkdir -p "$LOG_ROOT"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" >&2
}

check_vllm_runtime() {
    if [ ! -x "$VLLM_BIN" ]; then
        echo "vLLM binary not found: $VLLM_BIN" >&2
        exit 1
    fi
    local version
    version="$("$VLLM_BIN" --version 2>/dev/null || true)"
    if echo "$version" | grep -Eq '0\.8\.'; then
        echo "Current vLLM is too old for Qwen3.5-4B: $version" >&2
        echo "Please point VLLM_BIN to a newer nightly/main-build environment before starting." >&2
        exit 1
    fi
}

prepare_inputs() {
    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/prepare_nq_hotpot_teacher_inputs.py" \
        --raw_train "$RAW_TRAIN" \
        --raw_test "$RAW_TEST" \
        --out_dir "$INPUT_DIR" \
        --train_size_per_source "${TRAIN_SIZE_PER_SOURCE:-10000}" \
        --test_size_per_source "${TEST_SIZE_PER_SOURCE:-1000}" \
        --seed "${DATA_SEED:-42}"
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

    log "Starting shared retriever on GPUs $RETRIEVER_VISIBLE_DEVICES port $RETRIEVER_PORT"
    CUDA_VISIBLE_DEVICES="$RETRIEVER_VISIBLE_DEVICES" nohup \
        "$RETRIEVER_PYTHON" "$REPO_ROOT/examples/search/retriever/retrieval_server.py" \
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

wait_for_url() {
    local url="$1"
    local retries="${2:-90}"
    local sleep_seconds="${3:-5}"
    local i
    for ((i=1; i<=retries; i++)); do
        if curl -fsS -m 3 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$sleep_seconds"
    done
    return 1
}

ensure_vllm_server() {
    local shard_id="$1"
    local gpu="$2"
    local port="$3"
    local session="${SESSION_PREFIX}_vllm${shard_id}"
    local log_file="$LOG_ROOT/vllm_shard${shard_id}.log"
    local health_url="http://127.0.0.1:${port}/health"

    if curl -fsS -m 3 "$health_url" >/dev/null 2>&1; then
        log "vLLM shard${shard_id} already available at $health_url"
        return 0
    fi

    tmux kill-session -t "$session" 2>/dev/null || true
    log "Starting vLLM shard${shard_id} on gpu=$gpu port=$port"
    tmux new-session -d -s "$session" \
        "cd '$REPO_ROOT' && \
         export CUDA_VISIBLE_DEVICES='$gpu' && \
         '$VLLM_BIN' serve '$MODEL_PATH' \
           --host 127.0.0.1 \
           --port '$port' \
           --tensor-parallel-size 1 \
           --max-model-len '$VLLM_MAX_MODEL_LEN' \
           --gpu-memory-utilization '$VLLM_GPU_MEMORY_UTILIZATION' \
           --reasoning-parser qwen3 \
           --api-key '$VLLM_API_KEY' \
           $VLLM_EXTRA_ARGS \
           2>&1 | tee '$log_file'"

    if ! wait_for_url "$health_url" 120 5; then
        echo "vLLM shard${shard_id} failed to start. Check $log_file" >&2
        exit 1
    fi
}

start_shard() {
    local shard_id="$1"
    local gpu="$2"
    local start="$3"
    local limit="$4"
    local vllm_port="$5"
    local out_dir="${WITH_SKILL_OUT_DIR}_shard${shard_id}"
    local log_file="$LOG_ROOT/shard${shard_id}.log"
    local session="${SESSION_PREFIX}_shard${shard_id}"

    tmux kill-session -t "$session" 2>/dev/null || true
    rm -rf "$out_dir"
    log "Starting shard${shard_id} gpu=$gpu start=$start limit=$limit out=$out_dir"
    tmux new-session -d -s "$session" \
        "cd '$REPO_ROOT' && \
         export PYTHONPATH='$REPO_ROOT'\${PYTHONPATH:+:\$PYTHONPATH} && \
         export CUDA_VISIBLE_DEVICES='$gpu' && \
         export MODEL_PATH='$MODEL_PATH' && \
         export INPUT_PATH='$INPUT_PATH' && \
         export OUT_DIR='$out_dir' && \
         export START='$start' && \
         export LIMIT='$limit' && \
         export BATCH_SIZE='${BATCH_SIZE:-12}' && \
         export NUM_CANDIDATES='${NUM_CANDIDATES:-4}' && \
         export MAX_STEPS='${MAX_STEPS:-5}' && \
         export MAX_NEW_TOKENS='${MAX_NEW_TOKENS:-160}' && \
         export RETRIEVAL_WORKERS='${RETRIEVAL_WORKERS:-64}' && \
         export SEARCH_URL='${SEARCH_URL:-http://127.0.0.1:8000/retrieve}' && \
         export VLLM_BASE_URL='http://127.0.0.1:${vllm_port}/v1' && \
         '$PYTHON' examples/data_preprocess/generate_qwen35_teacher_sft_vllm_fc.py \
           --model_path \"\$MODEL_PATH\" \
           --model_name \"\$MODEL_PATH\" \
           --input_path \"\$INPUT_PATH\" \
           --out_dir \"\$OUT_DIR\" \
           --search_url \"\$SEARCH_URL\" \
           --vllm_base_url \"\$VLLM_BASE_URL\" \
           --vllm_api_key '${VLLM_API_KEY}' \
           --start \"\$START\" \
           --limit \"\$LIMIT\" \
           --batch_size \"\$BATCH_SIZE\" \
           --num_candidates \"\$NUM_CANDIDATES\" \
           --max_steps \"\$MAX_STEPS\" \
           --max_new_tokens \"\$MAX_NEW_TOKENS\" \
           --max_model_len '${VLLM_MAX_MODEL_LEN}' \
           --retrieval_workers \"\$RETRIEVAL_WORKERS\" \
           --retrieval_topk '${RETRIEVAL_TOPK:-3}' \
           --retrieval_max_doc_chars '${RETRIEVAL_MAX_DOC_CHARS:-1600}' \
           2>&1 | tee '$log_file'"
}

merge_outputs() {
    local shard0="${WITH_SKILL_OUT_DIR}_shard0"
    local shard1="${WITH_SKILL_OUT_DIR}_shard1"
    if [ ! -f "$shard0/train.parquet" ] || [ ! -f "$shard1/train.parquet" ]; then
        echo "Shard outputs are incomplete; cannot merge yet." >&2
        exit 1
    fi

    rm -rf "$WITH_SKILL_OUT_DIR" "$NO_SKILL_OUT_DIR" "$WITH_SKILL_FMT_DIR" "$NO_SKILL_FMT_DIR"
    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/merge_searchqa_teacher_sft_shards.py" \
        --shard_dirs "$shard0" "$shard1" \
        --out_dir "$WITH_SKILL_OUT_DIR"

    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/filter_searchqa_evidence_supported.py" \
        --input_jsonl "$WITH_SKILL_OUT_DIR/accepted.jsonl" \
        --out_dir "$WITH_SKILL_OUT_DIR" \
        --val_size 1000

    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/make_searchqa_teacher_noskill.py" \
        --input_jsonl "$WITH_SKILL_OUT_DIR/accepted.jsonl" \
        --out_dir "$NO_SKILL_OUT_DIR" \
        --val_size 1000

    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/prepare_teacher_grpo_data_generic.py" \
        --input_path "$WITH_SKILL_OUT_DIR/train.parquet" \
        --output_dir "$WITH_SKILL_FMT_DIR" \
        --copy_eval_from "$INPUT_DIR"

    "$PYTHON" "$REPO_ROOT/examples/data_preprocess/prepare_teacher_grpo_data_generic.py" \
        --input_path "$NO_SKILL_OUT_DIR/train.parquet" \
        --output_dir "$NO_SKILL_FMT_DIR" \
        --copy_eval_from "$INPUT_DIR"
}

case "${1:-start}" in
    prepare)
        prepare_inputs
        ;;
    start)
        prepare_inputs
        check_vllm_runtime
        ensure_retriever
        ensure_vllm_server 0 0 "$VLLM_PORT_BASE"
        ensure_vllm_server 1 1 "$((VLLM_PORT_BASE + 1))"
        start_shard 0 0 0 "$SHARD0_LIMIT" "$VLLM_PORT_BASE"
        start_shard 1 1 "$SHARD1_START" "$SHARD1_LIMIT" "$((VLLM_PORT_BASE + 1))"
        log "Started two shard sessions: ${SESSION_PREFIX}_shard0 and ${SESSION_PREFIX}_shard1"
        ;;
    merge)
        merge_outputs
        ;;
    *)
        echo "Usage: $0 [prepare|start|merge]" >&2
        exit 1
        ;;
esac
