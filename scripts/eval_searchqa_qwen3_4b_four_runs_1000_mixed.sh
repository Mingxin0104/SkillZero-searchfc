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
TRAINED_WITH_SKILL_MODEL_PATH="${TRAINED_WITH_SKILL_MODEL_PATH:-/workspace/limingxin/merged_models/searchqa_qwen3_4b_noanswer_lora_r8_two/qwen3_4b_noanswer_with_skill_lora_r8_ep1}"
TRAINED_NO_SKILL_MODEL_PATH="${TRAINED_NO_SKILL_MODEL_PATH:-/workspace/limingxin/merged_models/searchqa_qwen3_4b_noanswer_lora_r8_two/qwen3_4b_noanswer_no_skill_lora_r8_ep1}"

TEST_DATA_PATH="${TEST_DATA_PATH:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/test_1000.parquet}"
EXTERNAL_SKILL_DATA_PATH="${EXTERNAL_SKILL_DATA_PATH:-/workspace/limingxin/data/searchqa_external_skill_eval_1000/searchqa_train_no_overlap_eval_1000.parquet}"
SKILL_MAPPING_FILE="${SKILL_MAPPING_FILE:-$REPO_ROOT/skills/search/skill_mapping.json}"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_four_runs_eval_1000_mixed_skill0}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"

mkdir -p "$OUTPUT_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$OUTPUT_DIR/driver.log"
}

run_one() {
    local run_name="$1"
    local model_path="$2"
    local data_path="$3"
    local use_skill="$4"
    local gpu="$5"

    local gen_dir="$OUTPUT_DIR/${run_name}.generations"
    local metric_file="$OUTPUT_DIR/${run_name}.skill0_metrics.json"
    local short_name
    short_name="$(printf '%s' "$run_name" | sed 's/qwen3_4b_//g; s/_test1000//g; s/_searchqa_train_skill1000/_bskill/g; s/trained_/t_/g; s/with_skill/wskill/g; s/noskill/nskill/g; s/base_with/basew/g')"
    local ray_tmp="/tmp/r_${gpu}_${short_name}"
    local -a overrides=(
        "env.use_skill=${use_skill}"
        "data.test_files=${data_path}"
        "data.val_batch_size=${VAL_BATCH_SIZE}"
        "env.rollout.n=1"
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

    log "start $run_name gpu=$gpu skill=$use_skill data=$data_path"
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

    "$PYTHON_BIN" "$SUMMARY_SCRIPT" \
        --generation_dir "$gen_dir/test" \
        --use_trajectory_metrics \
        --out "$metric_file" \
        2>&1 | tee "$OUTPUT_DIR/${run_name}.summary.log"

    log "done $run_name metrics=$metric_file"
}

run_pair() {
    local run_name_a="$1"
    local model_path_a="$2"
    local data_path_a="$3"
    local use_skill_a="$4"
    local run_name_b="$5"
    local model_path_b="$6"
    local data_path_b="$7"
    local use_skill_b="$8"

    run_one "$run_name_a" "$model_path_a" "$data_path_a" "$use_skill_a" "$GPU_A" &
    local pid_a=$!
    run_one "$run_name_b" "$model_path_b" "$data_path_b" "$use_skill_b" "$GPU_B" &
    local pid_b=$!
    wait "$pid_a"
    wait "$pid_b"
}

run_pair \
    "qwen3_4b_trained_noskill_test1000" "$TRAINED_NO_SKILL_MODEL_PATH" "$TEST_DATA_PATH" "False" \
    "qwen3_4b_trained_with_skill_test1000" "$TRAINED_WITH_SKILL_MODEL_PATH" "$TEST_DATA_PATH" "False"

run_pair \
    "qwen3_4b_base_test1000" "$BASE_MODEL_PATH" "$TEST_DATA_PATH" "False" \
    "qwen3_4b_base_with_searchqa_train_skill1000" "$BASE_MODEL_PATH" "$EXTERNAL_SKILL_DATA_PATH" "True"

log "all four skill0-style evals finished"
