#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

BASE_DATA_DIR="${BASE_DATA_DIR:-/workspace/limingxin/data/searchqa_qwen25_3b_grpo_skill0_full_n2}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen25_3b_grpo_skill0_n2_strict_lora_sweep_r8_16_64}"
LOG_ROOT="${LOG_ROOT:-/workspace/limingxin/logs/searchqa_qwen25_3b_grpo_skill0_n2_strict_lora_sweep_r8_16_64}"

MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LR="${LR:-1e-4}"
LORA_ALPHA="${LORA_ALPHA:-16}"
RANKS="${RANKS:-8 16 64}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"

mkdir -p "$CKPT_ROOT" "$LOG_ROOT"

micro_bsz_for_rank() {
    local rank="$1"
    if [ "$rank" = "64" ]; then
        echo "${MICRO_BATCH_SIZE_R64:-2}"
    else
        echo "${MICRO_BATCH_SIZE_DEFAULT:-4}"
    fi
}

run_one() {
    local rank="$1"
    local variant="$2"
    local data_dir="$3"
    local micro_bsz
    micro_bsz="$(micro_bsz_for_rank "$rank")"

    local name="strict_${variant}_lora_r${rank}_ep${TOTAL_EPOCHS}"
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
    echo "max_length=$MAX_LENGTH micro_bsz=$micro_bsz train_bsz=$TRAIN_BATCH_SIZE rank=$rank alpha=$LORA_ALPHA" | tee -a "$log_file"
    if [ -n "$TOTAL_TRAINING_STEPS" ]; then
        echo "total_training_steps=$TOTAL_TRAINING_STEPS" | tee -a "$log_file"
    fi

    local extra_args=()
    if [ -n "$TOTAL_TRAINING_STEPS" ]; then
        extra_args+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PYTHONPATH="$REPO_ROOT" \
    /workspace/limingxin/miniconda3/envs/skillzero/bin/torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="$data_dir/train.parquet" \
        data.val_files="$data_dir/val_1000.parquet" \
        data.multiturn.enable=True \
        data.multiturn.messages_key=messages \
        data.micro_batch_size_per_gpu="$micro_bsz" \
        data.train_batch_size="$TRAIN_BATCH_SIZE" \
        data.max_length="$MAX_LENGTH" \
        data.truncation=right \
        model.partial_pretrain="$MODEL_PATH" \
        model.enable_gradient_checkpointing=True \
        model.lora_rank="$rank" \
        model.lora_alpha="$LORA_ALPHA" \
        model.target_modules=all-linear \
        optim.lr="$LR" \
        trainer.default_local_dir="$save_dir" \
        trainer.default_hdfs_dir=null \
        trainer.project_name=SkillZero-searchqa-strict-n2-sft-lora-sweep \
        trainer.experiment_name="$name" \
        trainer.logger="['console']" \
        trainer.total_epochs="$TOTAL_EPOCHS" \
        +trainer.skip_validation=True \
        "${extra_args[@]}" \
        2>&1 | tee -a "$log_file"
}

for rank in $RANKS; do
    run_one "$rank" "with_skill" "$BASE_DATA_DIR/strict_with_skill_sft"
    run_one "$rank" "no_skill" "$BASE_DATA_DIR/strict_no_skill_sft"
done

echo "[$(date -u '+%F %T UTC')] All strict n2 LoRA sweep trainings finished: ranks=$RANKS"
