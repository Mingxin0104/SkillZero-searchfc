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
BASE_MODEL="${BASE_MODEL:-/workspace/limingxin/modelscope_cache/Qwen__Qwen3-4B}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen3_4b_noanswer_full_sft_2exp}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_noanswer_fullsft_2exp_skill0_eval_5000_2gpu_sharded}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt}"
TEST_FILE="${TEST_FILE:-$DATA_ROOT/test_5000.parquet}"

RETRIEVER_INDEX_PATH="${RETRIEVER_INDEX_PATH:-/public/limingxin/SkillZero/data/searchR1/e5_Flat.index}"
RETRIEVER_CORPUS_PATH="${RETRIEVER_CORPUS_PATH:-/public/limingxin/SkillZero/data/searchR1/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_MODEL_CACHE_ROOT="${RETRIEVER_MODEL_CACHE_ROOT:-/public/limingxin/SkillZero/hf_cache/hub/models--intfloat--e5-base-v2}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
RETRIEVER_VISIBLE_DEVICES="${RETRIEVER_VISIBLE_DEVICES:-${GPU_A},${GPU_B}}"
RETRIEVER_SHARED_PORT="${RETRIEVER_SHARED_PORT:-8000}"
RETRIEVER_SHARED_LOG="${RETRIEVER_SHARED_LOG:-$LOG_DIR/retrieval_server_gpu_sharded.log}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.68}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-512}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
RAY_TMP_ROOT="${RAY_TMP_ROOT:-/tmp/q5f}"

mkdir -p "$LOG_DIR" "$RAY_TMP_ROOT"

RUN_SPECS=(
  "qwen3_4b_noanswer_with_skill_full_ep1_test5000|$CKPT_ROOT/qwen3_4b_noanswer_with_skill_full_ep1/global_step_806"
  "qwen3_4b_noanswer_no_skill_full_ep1_test5000|$CKPT_ROOT/qwen3_4b_noanswer_no_skill_full_ep1/global_step_403"
)

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG_DIR/driver.log"
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
    local health_url="$1"
    local retries="${2:-360}"
    local sleep_seconds="${3:-5}"
    local i
    for ((i=1; i<=retries; i++)); do
        if curl -fsS -m 2 "$health_url" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$sleep_seconds"
    done
    return 1
}

ensure_shared_gpu_retriever() {
    local health_url="http://127.0.0.1:${RETRIEVER_SHARED_PORT}/docs"
    resolve_retriever_model
    if curl -fsS -m 2 "$health_url" >/dev/null 2>&1; then
        log "Shared GPU retriever already available on GPUs ${RETRIEVER_VISIBLE_DEVICES} port ${RETRIEVER_SHARED_PORT}"
        return 0
    fi

    log "Starting shared GPU retriever on GPUs ${RETRIEVER_VISIBLE_DEVICES} port ${RETRIEVER_SHARED_PORT}"
    CUDA_VISIBLE_DEVICES="$RETRIEVER_VISIBLE_DEVICES" nohup "$RETRIEVER_PYTHON" \
        "$REPO_ROOT/examples/search/retriever/retrieval_server.py" \
        --index_path "$RETRIEVER_INDEX_PATH" \
        --corpus_path "$RETRIEVER_CORPUS_PATH" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL" \
        --port "$RETRIEVER_SHARED_PORT" \
        --faiss_gpu \
        > "$RETRIEVER_SHARED_LOG" 2>&1 &
    wait_for_retriever "$health_url" 360 5
    log "Shared GPU retriever is ready on GPUs ${RETRIEVER_VISIBLE_DEVICES} port ${RETRIEVER_SHARED_PORT}"
}

summarize_one() {
    local run_name="$1"
    local gen_dir="$LOG_DIR/${run_name}.generations/test"

    "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_skill0_original_metric.py" \
        --generation_dir "$gen_dir" \
        --out "$LOG_DIR/${run_name}.skill0_original_metric.json" \
        > "$LOG_DIR/${run_name}.metric.stdout.log" 2>&1

    "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_search_rate.py" \
        --generation_dir "$gen_dir" \
        --use_trajectory_metrics \
        --out "$LOG_DIR/${run_name}.trajectory_metric.json" \
        >> "$LOG_DIR/${run_name}.metric.stdout.log" 2>&1
}

eval_one() {
    local run_name="$1"
    local model_path="$2"
    local gpu="$3"
    local search_url="$4"
    local short_name
    short_name="$(printf '%s' "$run_name" | sed 's/qwen3_4b_//g; s/_test5000//g; s/_with_skill/ws/g; s/_no_skill/ns/g; s/_full_ep1/f/g; s/_noanswer_//g')"
    local short_tmp
    short_tmp="$(printf '%s' "$short_name" | cut -c1-4)"
    local ray_tmp="${RAY_TMP_ROOT}/${gpu}_${short_tmp}"

    rm -rf "$LOG_DIR/${run_name}.generations" "$ray_tmp"
    mkdir -p "$ray_tmp"

    log "Start eval $run_name on GPU $gpu model=$model_path"
    RAY_DEDUP_LOGS=0 \
    SEARCH_URL="$search_url" \
    DATA_ROOT="$DATA_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXP_LOG_NAME="${run_name}.eval" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    N_GPUS_PER_NODE=1 \
    RAY_NUM_CPUS="$RAY_NUM_CPUS" \
    ROLLOUT_GPU_MEMORY_UTILIZATION="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    ROLLOUT_MAX_NUM_SEQS="$ROLLOUT_MAX_NUM_SEQS" \
    bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$model_path" \
        env.use_skill=False \
        data.test_files="$TEST_FILE" \
        data.val_batch_size="$VAL_BATCH_SIZE" \
        data.max_prompt_length="$MAX_PROMPT_LENGTH" \
        data.max_response_length="$MAX_RESPONSE_LENGTH" \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        env.rollout.n=1 \
        actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
        +ray_init.include_dashboard=False \
        +ray_init._temp_dir="$ray_tmp" \
        trainer.validation_data_dir="$LOG_DIR/${run_name}.generations" \
        env.curriculum_learning.enable=False \
        2>&1 | tee "$LOG_DIR/${run_name}.eval.driver.log"

    summarize_one "$run_name"
    log "Done eval $run_name"
}

log "SearchQA qwen3-4b noanswer full-SFT test5000 2gpu sharded eval started"

ensure_shared_gpu_retriever

IFS='|' read -r run_name_a model_path_a <<< "${RUN_SPECS[0]}"
IFS='|' read -r run_name_b model_path_b <<< "${RUN_SPECS[1]}"

eval_one "$run_name_a" "$model_path_a" "$GPU_A" "http://127.0.0.1:${RETRIEVER_SHARED_PORT}/retrieve" &
pid_a=$!
eval_one "$run_name_b" "$model_path_b" "$GPU_B" "http://127.0.0.1:${RETRIEVER_SHARED_PORT}/retrieve" &
pid_b=$!

wait "$pid_a"
wait "$pid_b"

LOG_DIR="$LOG_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os

log_dir = os.environ["LOG_DIR"]
rows = []

for name in sorted(os.listdir(log_dir)):
    if not name.endswith(".skill0_original_metric.json"):
        continue
    run_name = name[:-len(".skill0_original_metric.json")]
    a_path = os.path.join(log_dir, name)
    t_path = os.path.join(log_dir, run_name + ".trajectory_metric.json")
    with open(a_path, encoding="utf-8") as f:
        a = json.load(f)
    with open(t_path, encoding="utf-8") as f:
        t = json.load(f)
    rows.append({
        "name": run_name,
        "total": t["total"],
        "acc": a["acc"],
        "success_rate": t["success_rate"],
        "answer_rate": a["answer_rate"],
        "search_rate": t["search_rate"],
        "avg_search_count": t["avg_search_count"],
        "trajectory_metric_path": t_path,
        "skill0_original_metric_path": a_path,
    })

out_path = os.path.join(log_dir, "summary.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY

log "All SearchQA qwen3-4b noanswer full-SFT test5000 evals finished"
