#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/limingxin/SkillZero}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/torchrun}"

DATA_DIR="${DATA_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_grpo_skill0_full_n2/loose_with_skill_sft}"
TEST_FILE="${TEST_FILE:-/workspace/limingxin/data/searchR1_processed_direct_skill0fmt/test_1000.parquet}"
MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"

CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen25_3b_grpo_skill0_n2_loose_with_skill_lora_r32}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/limingxin/merged_models/searchqa_qwen25_3b_grpo_skill0_n2_loose_with_skill_lora_r32}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen25_3b_grpo_skill0_n2_loose_with_skill_r32_eval_1000}"

NAME="${NAME:-loose_with_skill_lora_r32_ep1}"
SAVE_DIR="$CKPT_ROOT/$NAME"
MERGED_DIR="$MERGED_ROOT/$NAME"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LR="${LR:-1e-4}"

SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
EVAL_GPU="${EVAL_GPU:-0}"

mkdir -p "$CKPT_ROOT" "$MERGED_ROOT" "$LOG_DIR" "$SAVE_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG_DIR/driver.log"
}

latest_ckpt() {
    find "$SAVE_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1 || true
}

if [ -z "$(latest_ckpt)" ]; then
    log "Start training $NAME"
    log "DATA_DIR=$DATA_DIR"
    log "max_length=$MAX_LENGTH micro_bsz=$MICRO_BATCH_SIZE_PER_GPU train_bsz=$TRAIN_BATCH_SIZE lr=$LR"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PYTHONPATH="$REPO_ROOT" \
    "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="$DATA_DIR/train.parquet" \
        data.val_files="$DATA_DIR/val_1000.parquet" \
        data.multiturn.enable=True \
        data.multiturn.messages_key=messages \
        data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
        data.train_batch_size="$TRAIN_BATCH_SIZE" \
        data.max_length="$MAX_LENGTH" \
        data.truncation=right \
        model.partial_pretrain="$MODEL_PATH" \
        model.enable_gradient_checkpointing=True \
        model.lora_rank=32 \
        model.lora_alpha=16 \
        model.target_modules=all-linear \
        optim.lr="$LR" \
        trainer.default_local_dir="$SAVE_DIR" \
        trainer.default_hdfs_dir=null \
        trainer.project_name=SkillZero-searchqa-loose-with-skill-sft \
        trainer.experiment_name="$NAME" \
        trainer.logger="['console']" \
        trainer.total_epochs="$TOTAL_EPOCHS" \
        +trainer.skip_validation=True \
        2>&1 | tee -a "$LOG_DIR/$NAME.train.log"
else
    log "Skip training: checkpoint exists at $(latest_ckpt)"
fi

CKPT="$(latest_ckpt)"
if [ -z "$CKPT" ]; then
    log "ERROR: no checkpoint found under $SAVE_DIR"
    exit 1
fi
log "Latest checkpoint: $CKPT"

if [ ! -f "$MERGED_DIR/config.json" ]; then
    log "Merging LoRA to $MERGED_DIR"
    rm -rf "$MERGED_DIR"
    "$PYTHON_BIN" "$REPO_ROOT/examples/merge_searchqa_lora.py" \
        --base_model "$MODEL_PATH" \
        --adapter_path "$CKPT" \
        --output_path "$MERGED_DIR" \
        2>&1 | tee -a "$LOG_DIR/$NAME.merge.log"
else
    log "Skip merge: merged model exists at $MERGED_DIR"
fi

GEN_DIR="$LOG_DIR/$NAME.generations"
METRIC_FILE="$LOG_DIR/$NAME.skill0_paper_success_acc.json"
RAY_TMP="/tmp/lws${EVAL_GPU}_${RANDOM}"

rm -rf "$GEN_DIR" "$RAY_TMP"
mkdir -p "$RAY_TMP"

log "Start eval $NAME on GPU $EVAL_GPU, test has no skill"
RAY_DEDUP_LOGS=0 \
SEARCH_URL="$SEARCH_URL" \
DATA_ROOT="/workspace/limingxin/data/searchR1_processed_direct_skill0fmt" \
LOG_DIR="$LOG_DIR" \
EXP_LOG_NAME="$NAME.eval" \
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
N_GPUS_PER_NODE=1 \
RAY_NUM_CPUS=4 \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}" \
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}" \
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}" \
bash "$REPO_ROOT/scripts/eval_searchqa_lora_mainppo.sh" "$MERGED_DIR" \
    env.use_skill=False \
    data.test_files="$TEST_FILE" \
    data.val_batch_size=32 \
    env.rollout.n=1 \
    +ray_init.include_dashboard=False \
    +ray_init._temp_dir="$RAY_TMP" \
    trainer.validation_data_dir="$GEN_DIR" \
    2>&1 | tee -a "$LOG_DIR/$NAME.eval.driver.log"

log "Compute Skill0 paper success_rate/acc"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

path = Path("$GEN_DIR/test/0.trajectory_metrics.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
scores = [float(r.get("score", 0.0) or 0.0) for r in rows]
success = sum(1 for s in scores if s >= 1.0)
total = len(rows)
out = {
    "definition": "Skill0/SearchQA: success_rate = mean(I_succ(tau)); I_succ=1 iff final <answer> normalized-EM matches gold. acc = mean(score).",
    "source_file": str(path),
    "total": total,
    "success": success,
    "success_rate": success / total if total else 0.0,
    "success_rate_percent": 100 * success / total if total else 0.0,
    "acc": sum(scores) / total if total else 0.0,
    "acc_percent": 100 * sum(scores) / total if total else 0.0,
    "score_sum": sum(scores),
    "searched": sum(1 for r in rows if float(r.get("tool_callings", 0.0) or 0.0) > 0),
    "search_rate_by_tool_callings": sum(1 for r in rows if float(r.get("tool_callings", 0.0) or 0.0) > 0) / total if total else 0.0,
    "avg_search_count": sum(float(r.get("tool_callings", 0.0) or 0.0) for r in rows) / total if total else 0.0,
}
Path("$METRIC_FILE").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
PY

log "Finished $NAME"
