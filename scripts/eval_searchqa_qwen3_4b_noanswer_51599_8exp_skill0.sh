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
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen3_4b_noanswer_51599_lora_sweep_2gpu}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_qwen3_4b_noanswer_51599_lora_sweep_2gpu}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_noanswer_51599_8exp_skill0_eval_1000}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt}"
TEST_FILE="${TEST_FILE:-$DATA_ROOT/test_1000.parquet}"

SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVER_HEALTH_URL="${RETRIEVER_HEALTH_URL:-http://127.0.0.1:8000/docs}"
RETRIEVER_INDEX_PATH="${RETRIEVER_INDEX_PATH:-/public/limingxin/SkillZero/data/searchR1/e5_Flat.index}"
RETRIEVER_CORPUS_PATH="${RETRIEVER_CORPUS_PATH:-/public/limingxin/SkillZero/data/searchR1/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_MODEL_CACHE_ROOT="${RETRIEVER_MODEL_CACHE_ROOT:-/public/limingxin/SkillZero/hf_cache/hub/models--intfloat--e5-base-v2}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
RETRIEVER_LOG="${RETRIEVER_LOG:-$LOG_DIR/retrieval_server.log}"
RETRIEVER_GPU="${RETRIEVER_GPU:-1}"
RETRIEVER_FAISS_GPU="${RETRIEVER_FAISS_GPU:-false}"

EVAL_GPU="${EVAL_GPU:-0}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-512}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
RAY_TMP_ROOT="${RAY_TMP_ROOT:-/tmp/q34s0}"

mkdir -p "$MERGED_ROOT" "$LOG_DIR" "$RAY_TMP_ROOT"

RUN_SPECS=(
  "qwen3_4b_base_test1000|base|$BASE_MODEL"
  "qwen3_4b_noanswer_51599_with_skill_lora_r8_ep1_test1000|adapter|qwen3_4b_noanswer_51599_with_skill_lora_r8_ep1"
  "qwen3_4b_noanswer_51599_no_skill_lora_r8_ep1_test1000|adapter|qwen3_4b_noanswer_51599_no_skill_lora_r8_ep1"
  "qwen3_4b_noanswer_51599_with_skill_lora_r16_ep1_test1000|adapter|qwen3_4b_noanswer_51599_with_skill_lora_r16_ep1"
  "qwen3_4b_noanswer_51599_no_skill_lora_r16_ep1_test1000|adapter|qwen3_4b_noanswer_51599_no_skill_lora_r16_ep1"
  "qwen3_4b_noanswer_51599_with_skill_lora_r32_ep1_test1000|adapter|qwen3_4b_noanswer_51599_with_skill_lora_r32_ep1"
  "qwen3_4b_noanswer_51599_no_skill_lora_r32_ep1_test1000|adapter|qwen3_4b_noanswer_51599_no_skill_lora_r32_ep1"
  "qwen3_4b_noanswer_51599_with_skill_lora_r64_ep1_test1000|adapter|qwen3_4b_noanswer_51599_with_skill_lora_r64_ep1"
  "qwen3_4b_noanswer_51599_no_skill_lora_r64_ep1_test1000|adapter|qwen3_4b_noanswer_51599_no_skill_lora_r64_ep1"
)

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
    local retries="${1:-360}"
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
    local -a retriever_args=(
        "$REPO_ROOT/examples/search/retriever/retrieval_server.py"
        --index_path "$RETRIEVER_INDEX_PATH"
        --corpus_path "$RETRIEVER_CORPUS_PATH"
        --topk "$RETRIEVER_TOPK"
        --retriever_name e5
        --retriever_model "$RETRIEVER_MODEL"
        --port "$RETRIEVER_PORT"
    )
    if [ "$RETRIEVER_FAISS_GPU" = "true" ]; then
        retriever_args+=(--faiss_gpu)
    fi

    CUDA_VISIBLE_DEVICES="$RETRIEVER_GPU" nohup "$RETRIEVER_PYTHON" "${retriever_args[@]}" > "$RETRIEVER_LOG" 2>&1 &
    wait_for_retriever 360 5
    log "Retriever is ready"
}

merge_one() {
    local tag="$1"
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
    local short_name
    short_name="$(printf '%s' "$run_name" | sed 's/qwen3_4b_//g; s/_test1000//g; s/_with_skill/ws/g; s/_no_skill/ns/g; s/_noanswer_//g; s/_51599//g; s/_lora_r/r/g; s/_ep1//g; s/base/b/g')"
    local ray_tmp="${RAY_TMP_ROOT}/${EVAL_GPU}_${short_name}"

    rm -rf "$LOG_DIR/${run_name}.generations" "$ray_tmp"
    mkdir -p "$ray_tmp"

    log "Start eval $run_name on GPU $EVAL_GPU"
    RAY_DEDUP_LOGS=0 \
    SEARCH_URL="$SEARCH_URL" \
    DATA_ROOT="$DATA_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXP_LOG_NAME="${run_name}.eval" \
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
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

log "SearchQA qwen3-4b noanswer 51599 8exp skill0 eval started"

for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r run_name kind value <<< "$spec"
    if [ "$kind" = "adapter" ]; then
        merge_one "$value"
    fi
done

ensure_retriever

for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r run_name kind value <<< "$spec"
    if [ "$kind" = "base" ]; then
        model_path="$value"
    else
        model_path="$MERGED_ROOT/$value"
    fi

    if [ -f "$LOG_DIR/${run_name}.trajectory_metric.json" ] && [ -f "$LOG_DIR/${run_name}.skill0_original_metric.json" ]; then
        log "Skip eval $run_name: metrics already exist"
        continue
    fi

    eval_one "$run_name" "$model_path"
done

LOG_DIR="$LOG_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os

log_dir = os.environ["LOG_DIR"]
rows = []

for name in sorted(os.listdir(log_dir)):
    if not name.endswith(".skill0_original_metric.json"):
        continue
    run_name = name[:-len(".skill0_original_metric.json")]
    metric_path = os.path.join(log_dir, name)
    traj_path = os.path.join(log_dir, f"{run_name}.trajectory_metric.json")

    with open(metric_path, encoding="utf-8") as f:
        metric = json.load(f)

    traj = {}
    if os.path.exists(traj_path):
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)

    rows.append(
        {
            "name": run_name,
            "total": metric.get("total"),
            "acc": metric.get("acc"),
            "success_rate": traj.get("success_rate", metric.get("acc")),
            "answer_rate": metric.get("answer_rate"),
            "search_rate": traj.get("search_rate", metric.get("search_rate")),
            "avg_search_count": traj.get("avg_search_count", metric.get("avg_search_count")),
            "trajectory_metric_path": traj_path if os.path.exists(traj_path) else None,
            "skill0_original_metric_path": metric_path,
        }
    )

out = os.path.join(log_dir, "summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY

log "All SearchQA qwen3-4b noanswer 51599 8exp skill0 evals finished"
