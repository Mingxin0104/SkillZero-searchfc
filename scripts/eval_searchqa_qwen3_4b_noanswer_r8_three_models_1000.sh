#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/limingxin/SkillZero}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
BASE_MODEL="${BASE_MODEL:-/workspace/limingxin/modelscope_cache/Qwen__Qwen3-4B}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt}"
TEST_FILE="${TEST_FILE:-$DATA_ROOT/test_1000.parquet}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"

CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen3_4b_noanswer_lora_r8_two}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_qwen3_4b_noanswer_lora_r8_two}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_noanswer_lora_r8_two_eval_1000}"
RAY_TMP_ROOT="${RAY_TMP_ROOT:-/workspace/limingxin/rt/q34}"

mkdir -p "$MERGED_ROOT" "$LOG_DIR" "$RAY_TMP_ROOT"

MODELS=(
  base_qwen3_4b
  qwen3_4b_noanswer_with_skill_lora_r8_ep1
  qwen3_4b_noanswer_no_skill_lora_r8_ep1
)

log() {
  echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG_DIR/driver.log"
}

merge_one() {
  local name="$1"
  local adapter_path="$2"
  local output_path="$MERGED_ROOT/$name"

  if [[ -f "$output_path/config.json" ]] && [[ -f "$output_path/model.safetensors.index.json" || -f "$output_path/model.safetensors" ]]; then
    log "skip merge $name, merged model exists"
    return
  fi

  log "start merge $name"
  "$PYTHON_BIN" "$REPO_ROOT/examples/merge_searchqa_lora.py" \
    --base_model "$BASE_MODEL" \
    --adapter_path "$adapter_path" \
    --output_path "$output_path" \
    > "$LOG_DIR/$name.merge.log" 2>&1
  log "done merge $name"
}

model_path_for() {
  local name="$1"
  case "$name" in
    base_qwen3_4b)
      echo "$BASE_MODEL"
      ;;
    qwen3_4b_noanswer_with_skill_lora_r8_ep1|qwen3_4b_noanswer_no_skill_lora_r8_ep1)
      echo "$MERGED_ROOT/$name"
      ;;
    *)
      echo "unknown model: $name" >&2
      return 1
      ;;
  esac
}

run_one() {
  local name="$1"
  local gpu="$2"
  local model_path
  model_path="$(model_path_for "$name")"
  local gen_dir="$LOG_DIR/$name.generations"
  local short_name
  case "$name" in
    base_qwen3_4b) short_name="base" ;;
    qwen3_4b_noanswer_with_skill_lora_r8_ep1) short_name="ws" ;;
    qwen3_4b_noanswer_no_skill_lora_r8_ep1) short_name="ns" ;;
    *) short_name="m" ;;
  esac
  local ray_tmp="$RAY_TMP_ROOT/${short_name}${gpu}"

  rm -rf "$gen_dir" "$ray_tmp"
  mkdir -p "$ray_tmp"

  log "start eval $name on GPU $gpu"
  RAY_DEDUP_LOGS=0 \
  SEARCH_URL="$SEARCH_URL" \
  DATA_ROOT="$DATA_ROOT" \
  LOG_DIR="$LOG_DIR" \
  EXP_LOG_NAME="${name}.eval" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  N_GPUS_PER_NODE=1 \
  RAY_NUM_CPUS=4 \
  ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.72}" \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}" \
  ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}" \
  bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$model_path" \
    env.use_skill=False \
    data.test_files="$TEST_FILE" \
    data.val_batch_size="${VAL_BATCH_SIZE:-32}" \
    env.rollout.n=1 \
    +ray_init.include_dashboard=False \
    +ray_init._temp_dir="$ray_tmp" \
    trainer.validation_data_dir="$gen_dir" \
    > "$LOG_DIR/$name.driver.stdout.log" 2>&1

  "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_skill0_original_metric.py" \
    --generation_dir "$gen_dir/test" \
    --out "$LOG_DIR/$name.skill0_original_metric.json" \
    > "$LOG_DIR/$name.metric.stdout.log" 2>&1

  "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_search_rate.py" \
    --generation_dir "$gen_dir/test" \
    --use_trajectory_metrics \
    --out "$LOG_DIR/$name.trajectory_metric.json" \
    >> "$LOG_DIR/$name.metric.stdout.log" 2>&1

  log "done eval $name on GPU $gpu"
}

merge_one \
  qwen3_4b_noanswer_with_skill_lora_r8_ep1 \
  "$CKPT_ROOT/qwen3_4b_noanswer_with_skill_lora_r8_ep1/global_step_806"
merge_one \
  qwen3_4b_noanswer_no_skill_lora_r8_ep1 \
  "$CKPT_ROOT/qwen3_4b_noanswer_no_skill_lora_r8_ep1/global_step_806"

run_one base_qwen3_4b 0 &
pid_a=$!
run_one qwen3_4b_noanswer_with_skill_lora_r8_ep1 1 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_one qwen3_4b_noanswer_no_skill_lora_r8_ep1 0

LOG_DIR="$LOG_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os

log_dir = os.environ["LOG_DIR"]
names = [
    "base_qwen3_4b",
    "qwen3_4b_noanswer_with_skill_lora_r8_ep1",
    "qwen3_4b_noanswer_no_skill_lora_r8_ep1",
]
rows = []
for name in names:
    metric_path = os.path.join(log_dir, f"{name}.skill0_original_metric.json")
    traj_path = os.path.join(log_dir, f"{name}.trajectory_metric.json")
    if not os.path.exists(metric_path):
        continue
    with open(metric_path, encoding="utf-8") as f:
        metric = json.load(f)
    traj = {}
    if os.path.exists(traj_path):
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)
    rows.append(
        {
            "name": name,
            "total": metric.get("total"),
            "success_rate": metric.get("acc"),
            "acc": metric.get("acc"),
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
