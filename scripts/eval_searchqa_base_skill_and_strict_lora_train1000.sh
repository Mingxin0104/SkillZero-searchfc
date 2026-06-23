#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/limingxin/SkillZero}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
BASE_MODEL="${BASE_MODEL:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt}"
TEST_1000="${TEST_1000:-$DATA_ROOT/test_1000.parquet}"
TRAIN_1000="${TRAIN_1000:-$DATA_ROOT/train_1000.parquet}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"

MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_qwen25_3b_grpo_skill0_n2_strict_lora_sweep_r8_16_64}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_base_skill_and_strict_lora_train1000_eval}"
RAY_BASE="${RAY_BASE:-/workspace/limingxin/rt}"

mkdir -p "$LOG_DIR" "$RAY_BASE"

NAMES=(
  strict_with_skill_lora_r8_ep1
  strict_no_skill_lora_r8_ep1
  strict_with_skill_lora_r16_ep1
  strict_no_skill_lora_r16_ep1
  strict_with_skill_lora_r64_ep1
  strict_no_skill_lora_r64_ep1
)

log() {
  echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG_DIR/driver.log"
}

summarize_one() {
  local name="$1"
  local gen_dir="$LOG_DIR/$name.generations/test"

  "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_skill0_original_metric.py" \
    --generation_dir "$gen_dir" \
    --out "$LOG_DIR/$name.skill0_original_metric.json" \
    > "$LOG_DIR/$name.metric.stdout.log" 2>&1

  "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_search_rate.py" \
    --generation_dir "$gen_dir" \
    --use_trajectory_metrics \
    --out "$LOG_DIR/$name.trajectory_metric.json" \
    >> "$LOG_DIR/$name.metric.stdout.log" 2>&1
}

run_one() {
  local name="$1"
  local model_path="$2"
  local data_file="$3"
  local use_skill="$4"
  local gpu="$5"
  local gen_dir="$LOG_DIR/$name.generations"
  local ray_tmp="$RAY_BASE/r${gpu}_${RANDOM}"

  rm -rf "$gen_dir" "$ray_tmp"
  mkdir -p "$ray_tmp"

  log "start eval $name on GPU $gpu use_skill=$use_skill data=$data_file"
  RAY_DEDUP_LOGS=0 \
  SEARCH_URL="$SEARCH_URL" \
  DATA_ROOT="$DATA_ROOT" \
  LOG_DIR="$LOG_DIR" \
  EXP_LOG_NAME="${name}.eval" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  N_GPUS_PER_NODE=1 \
  RAY_NUM_CPUS=4 \
  ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}" \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}" \
  ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}" \
  bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$model_path" \
    env.use_skill="$use_skill" \
    data.test_files="$data_file" \
    data.val_batch_size="${VAL_BATCH_SIZE:-32}" \
    env.rollout.n=1 \
    +ray_init.include_dashboard=False \
    +ray_init._temp_dir="$ray_tmp" \
    trainer.validation_data_dir="$gen_dir" \
    > "$LOG_DIR/$name.driver.stdout.log" 2>&1

  summarize_one "$name"
  log "done eval $name on GPU $gpu"
}

run_pair_train() {
  local name_a="$1"
  local name_b="$2"
  run_one "${name_a}_train1000_noskill" "$MERGED_ROOT/$name_a" "$TRAIN_1000" False 0 &
  local pid_a=$!
  run_one "${name_b}_train1000_noskill" "$MERGED_ROOT/$name_b" "$TRAIN_1000" False 1 &
  local pid_b=$!
  wait "$pid_a"
  wait "$pid_b"
}

run_one "base_qwen25_3b_test1000_with_skill" "$BASE_MODEL" "$TEST_1000" True 0

run_pair_train strict_with_skill_lora_r8_ep1 strict_no_skill_lora_r8_ep1
run_pair_train strict_with_skill_lora_r16_ep1 strict_no_skill_lora_r16_ep1
run_pair_train strict_with_skill_lora_r64_ep1 strict_no_skill_lora_r64_ep1

"$PYTHON_BIN" - <<'PY'
import glob
import json
import os

log_dir = os.environ.get(
    "LOG_DIR",
    "/workspace/limingxin/logs/searchqa_base_skill_and_strict_lora_train1000_eval",
)
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, "*.skill0_original_metric.json"))):
    name = os.path.basename(path).replace(".skill0_original_metric.json", "")
    with open(path, encoding="utf-8") as f:
        metric = json.load(f)
    traj_path = os.path.join(log_dir, f"{name}.trajectory_metric.json")
    traj = {}
    if os.path.exists(traj_path):
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)
    rows.append(
        {
            "name": name,
            "total": metric.get("total"),
            "acc": metric.get("acc"),
            "success_rate": metric.get("acc"),
            "answer_rate": metric.get("answer_rate"),
            "search_rate": traj.get("search_rate", metric.get("search_rate")),
            "avg_search_count": traj.get("avg_search_count", metric.get("avg_search_count")),
        }
    )

out = os.path.join(log_dir, "summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY

log "all evals finished"
