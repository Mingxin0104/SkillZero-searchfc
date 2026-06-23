#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh"
SKILL0_METRIC_SCRIPT="$REPO_ROOT/examples/summarize_searchqa_search_rate.py"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/workspace/limingxin/modelscope_cache/Qwen__Qwen3-4B}"
TRAINED_WITH_SKILL_MODEL_PATH="${TRAINED_WITH_SKILL_MODEL_PATH:-/workspace/limingxin/merged_models/searchqa_qwen3_4b_noanswer_lora_r8_two/qwen3_4b_noanswer_with_skill_lora_r8_ep1}"
TRAINED_NO_SKILL_MODEL_PATH="${TRAINED_NO_SKILL_MODEL_PATH:-/workspace/limingxin/merged_models/searchqa_qwen3_4b_noanswer_lora_r8_two/qwen3_4b_noanswer_no_skill_lora_r8_ep1}"

TEST_DATA_PATH="${TEST_DATA_PATH:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/test_1000.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_three_models_skill0_consistent_1000}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-512}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
RAY_TMP_ROOT="${RAY_TMP_ROOT:-/workspace/limingxin/rt/q34c}"

mkdir -p "$OUTPUT_DIR" "$RAY_TMP_ROOT"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$OUTPUT_DIR/driver.log"
}

run_one() {
    local run_name="$1"
    local model_path="$2"
    local gpu="$3"

    local gen_dir="$OUTPUT_DIR/${run_name}.generations"
    local metric_file="$OUTPUT_DIR/${run_name}.skill0_metrics.json"
    local short_name
    short_name="$(printf '%s' "$run_name" | sed 's/qwen3_4b_//g; s/_test1000//g; s/_with_skill/ws/g; s/_no_skill/ns/g; s/_noanswer_//g; s/_lora_r8_ep1//g; s/base/b/g')"
    local ray_tmp="${RAY_TMP_ROOT}/${gpu}_${short_name}"
    local -a overrides=(
        "env.use_skill=False"
        "data.test_files=${TEST_DATA_PATH}"
        "data.val_batch_size=${VAL_BATCH_SIZE}"
        "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
        "data.max_response_length=${MAX_RESPONSE_LENGTH}"
        "+data.apply_chat_template_kwargs.enable_thinking=False"
        "env.rollout.n=1"
        "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
        "+ray_init.include_dashboard=False"
        "+ray_init._temp_dir=${ray_tmp}"
        "trainer.validation_data_dir=${gen_dir}"
        "env.curriculum_learning.enable=False"
    )

    rm -rf "$gen_dir" "$ray_tmp"
    mkdir -p "$ray_tmp"

    log "start $run_name gpu=$gpu model=$model_path"
    RAY_DEDUP_LOGS=0 \
    SEARCH_URL="$SEARCH_URL" \
    DATA_ROOT="/workspace/limingxin/data/searchR1_processed_direct_skill0fmt" \
    LOG_DIR="$OUTPUT_DIR" \
    EXP_LOG_NAME="$run_name.eval" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    N_GPUS_PER_NODE=1 \
    RAY_NUM_CPUS="$RAY_NUM_CPUS" \
    ROLLOUT_GPU_MEMORY_UTILIZATION="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    ROLLOUT_MAX_NUM_SEQS="$ROLLOUT_MAX_NUM_SEQS" \
    bash "$EVAL_SCRIPT" "$model_path" \
        "${overrides[@]}" \
        2>&1 | tee "$OUTPUT_DIR/${run_name}.eval.driver.log"

    "$PYTHON_BIN" "$SKILL0_METRIC_SCRIPT" \
        --generation_dir "$gen_dir/test" \
        --use_trajectory_metrics \
        --out "$metric_file" \
        2>&1 | tee "$OUTPUT_DIR/${run_name}.summary.log"

    log "done $run_name metrics=$metric_file"
}

run_one "qwen3_4b_base_test1000" "$BASE_MODEL_PATH" "$GPU_A" &
pid_a=$!
run_one "qwen3_4b_noanswer_with_skill_lora_r8_ep1_test1000" "$TRAINED_WITH_SKILL_MODEL_PATH" "$GPU_B" &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_one "qwen3_4b_noanswer_no_skill_lora_r8_ep1_test1000" "$TRAINED_NO_SKILL_MODEL_PATH" "$GPU_A"

log "all three consistent skill0 evals finished"
