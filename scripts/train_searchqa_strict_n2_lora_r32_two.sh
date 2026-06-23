#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

BASE_DATA_DIR="${BASE_DATA_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_grpo_skill0_full_n2}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen25_3b_grpo_skill0_n2_strict_lora_r32}"
LOG_ROOT="${LOG_ROOT:-/workspace/limingxin/logs/searchqa_qwen25_3b_grpo_skill0_n2_strict_lora_r32}"

MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LR="${LR:-1e-4}"

mkdir -p "$CKPT_ROOT" "$LOG_ROOT"

run_one() {
    local name="$1"
    local data_dir="$2"
    local save_dir="$CKPT_ROOT/$name"
    local log_file="$LOG_ROOT/$name.log"

    mkdir -p "$save_dir"
    if find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | grep -q .; then
        echo "[$(date -u '+%F %T UTC')] Skip $name: checkpoint exists under $save_dir" | tee -a "$log_file"
        return 0
    fi

    echo "[$(date -u '+%F %T UTC')] Start $name" | tee -a "$log_file"
    echo "DATA_DIR=$data_dir" | tee -a "$log_file"
    echo "MODEL_PATH=$MODEL_PATH" | tee -a "$log_file"
    echo "max_length=$MAX_LENGTH micro_bsz=$MICRO_BATCH_SIZE_PER_GPU train_bsz=$TRAIN_BATCH_SIZE" | tee -a "$log_file"

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PYTHONPATH="$REPO_ROOT" \
    /workspace/limingxin/miniconda3/envs/skillzero/bin/torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="$data_dir/train.parquet" \
        data.val_files="$data_dir/val_1000.parquet" \
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
        trainer.default_local_dir="$save_dir" \
        trainer.default_hdfs_dir=null \
        trainer.project_name=SkillZero-searchqa-strict-n2-sft \
        trainer.experiment_name="$name" \
        trainer.logger="['console']" \
        trainer.total_epochs="$TOTAL_EPOCHS" \
        +trainer.skip_validation=True \
        2>&1 | tee -a "$log_file"
}

run_one "strict_with_skill_lora_r32_ep${TOTAL_EPOCHS}" "$BASE_DATA_DIR/strict_with_skill_sft"
run_one "strict_no_skill_lora_r32_ep${TOTAL_EPOCHS}" "$BASE_DATA_DIR/strict_no_skill_sft"

echo "[$(date -u '+%F %T UTC')] All strict n2 r32 LoRA trainings finished"
