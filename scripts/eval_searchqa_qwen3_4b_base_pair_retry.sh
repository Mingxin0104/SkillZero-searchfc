#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh"
SUMMARY_SCRIPT="$REPO_ROOT/examples/summarize_searchqa_search_rate.py"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/workspace/limingxin/modelscope_cache/Qwen__Qwen3-4B}"
TEST_DATA_PATH="${TEST_DATA_PATH:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/test_1000.parquet}"
EXTERNAL_SKILL_DATA_PATH="${EXTERNAL_SKILL_DATA_PATH:-/workspace/limingxin/data/searchqa_external_skill_eval_1000/searchqa_train_no_overlap_eval_1000_rollout.parquet}"
SKILL_MAPPING_FILE="${SKILL_MAPPING_FILE:-$REPO_ROOT/skills/search/skill_mapping.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_four_runs_eval_1000_mixed_skill0}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-512}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"

mkdir -p "$OUTPUT_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$OUTPUT_DIR/driver.log"
}

run_one() {
    local run_name="$1"
    local data_path="$2"
    local use_skill="$3"
    local gpu="$4"

    local gen_dir="$OUTPUT_DIR/${run_name}.generations"
    local metric_file="$OUTPUT_DIR/${run_name}.skill0_metrics.json"
    local short_name
    short_name="$(printf '%s' "$run_name" | sed 's/qwen3_4b_//g; s/_test1000//g; s/_searchqa_train_skill1000/_bskill/g; s/base_with/basew/g')"
    local ray_tmp="/tmp/r_${gpu}_${short_name}"
    local -a overrides=(
        "env.use_skill=${use_skill}"
        "data.test_files=${data_path}"
        "data.val_batch_size=${VAL_BATCH_SIZE}"
        "data.max_prompt_length=3072"
        "data.max_response_length=128"
        "env.rollout.n=1"
        "actor_rollout_ref.rollout.response_length=128"
        "+ray_init.include_dashboard=False"
        "+ray_init._temp_dir=${ray_tmp}"
        "trainer.validation_data_dir=${gen_dir}"
    )

    if [ "$use_skill" = "True" ]; then
        overrides+=(
            "env.curriculum_learning.enable=True"
            "env.curriculum_learning.skill_mapping_file=${SKILL_MAPPING_FILE}"
        )
    else
        overrides+=("env.curriculum_learning.enable=False")
    fi

    rm -rf "$gen_dir" "$ray_tmp"
    mkdir -p "$ray_tmp"

    log "retry start $run_name gpu=$gpu skill=$use_skill data=$data_path"
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
    bash "$EVAL_SCRIPT" "$BASE_MODEL_PATH" \
        "${overrides[@]}" \
        2>&1 | tee "$OUTPUT_DIR/${run_name}.eval.driver.log"

    "$PYTHON_BIN" "$SUMMARY_SCRIPT" \
        --generation_dir "$gen_dir/test" \
        --use_trajectory_metrics \
        --out "$metric_file" \
        2>&1 | tee "$OUTPUT_DIR/${run_name}.summary.log"

    log "retry done $run_name metrics=$metric_file"
}

run_one "qwen3_4b_base_test1000" "$TEST_DATA_PATH" "False" "$GPU_A" &
pid_a=$!
run_one "qwen3_4b_base_with_searchqa_train_skill1000" "$EXTERNAL_SKILL_DATA_PATH" "True" "$GPU_B" &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

log "base pair retry finished"
