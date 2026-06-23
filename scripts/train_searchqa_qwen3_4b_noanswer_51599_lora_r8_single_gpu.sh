#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/workspace/limingxin/modelscope_cache/Qwen__Qwen3-4B}"
WITH_SKILL_DATA_DIR="${WITH_SKILL_DATA_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_teacher_sft_noanswer}"
NO_SKILL_DATA_DIR="${NO_SKILL_DATA_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_teacher_sft_noanswer_noskill}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen3_4b_noanswer_51599_lora_r8_single_gpu}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_noanswer_51599_lora_r8_single_gpu}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-2304}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
FALLBACK1_MICRO_BATCH_SIZE_PER_GPU="${FALLBACK1_MICRO_BATCH_SIZE_PER_GPU:-2}"
FALLBACK1_TRAIN_BATCH_SIZE="${FALLBACK1_TRAIN_BATCH_SIZE:-16}"
FALLBACK2_MICRO_BATCH_SIZE_PER_GPU="${FALLBACK2_MICRO_BATCH_SIZE_PER_GPU:-1}"
FALLBACK2_TRAIN_BATCH_SIZE="${FALLBACK2_TRAIN_BATCH_SIZE:-8}"

mkdir -p "$CKPT_ROOT" "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG_DIR/driver.log" >&2
}

has_checkpoint() {
    local save_dir="$1"
    find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | grep -q .
}

run_once() {
    local tag="$1"
    local data_dir="$2"
    local save_dir="$CKPT_ROOT/$tag"
    local log_file="$LOG_DIR/${tag}.log"
    local attempts=(
        "$MICRO_BATCH_SIZE_PER_GPU:$TRAIN_BATCH_SIZE"
        "$FALLBACK1_MICRO_BATCH_SIZE_PER_GPU:$FALLBACK1_TRAIN_BATCH_SIZE"
        "$FALLBACK2_MICRO_BATCH_SIZE_PER_GPU:$FALLBACK2_TRAIN_BATCH_SIZE"
    )

    mkdir -p "$save_dir"
    if has_checkpoint "$save_dir"; then
        log "Skip $tag: checkpoint already exists under $save_dir"
        return 0
    fi

    local attempt_idx=0
    local attempt
    for attempt in "${attempts[@]}"; do
        attempt_idx=$((attempt_idx + 1))
        local micro_bsz="${attempt%%:*}"
        local train_bsz="${attempt##*:}"
        if [ "$attempt_idx" -gt 1 ]; then
            log "Retry $tag with lower batch: micro_bsz=$micro_bsz train_bsz=$train_bsz"
        fi

        {
            echo "[$(date -u '+%F %T UTC')] Start $tag"
            echo "MODEL_PATH=$MODEL_PATH"
            echo "DATA_DIR=$data_dir"
            echo "SAVE_DIR=$save_dir"
            echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE"
            echo "rank=$LORA_RANK alpha=$LORA_ALPHA max_length=$MAX_LENGTH micro_bsz=$micro_bsz train_bsz=$train_bsz lr=$LR epochs=$TOTAL_EPOCHS"
        } | tee -a "$log_file"

        if CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
            PYTHONPATH="$REPO_ROOT" \
            "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
                -m verl.trainer.fsdp_sft_trainer \
                data.train_files="$data_dir/train.parquet" \
                data.val_files="$data_dir/val_1000.parquet" \
                data.multiturn.enable=True \
                data.multiturn.messages_key=messages \
                data.micro_batch_size_per_gpu="$micro_bsz" \
                data.train_batch_size="$train_bsz" \
                data.max_length="$MAX_LENGTH" \
                data.truncation=right \
                model.partial_pretrain="$MODEL_PATH" \
                model.enable_gradient_checkpointing=True \
                model.lora_rank="$LORA_RANK" \
                model.lora_alpha="$LORA_ALPHA" \
                model.target_modules=all-linear \
                optim.lr="$LR" \
                trainer.default_local_dir="$save_dir" \
                trainer.default_hdfs_dir=null \
                trainer.project_name=SkillZero-searchqa-qwen3-4b-noanswer-51599-sft \
                trainer.experiment_name="$tag" \
                trainer.logger="['console']" \
                trainer.total_epochs="$TOTAL_EPOCHS" \
                +trainer.skip_validation=True \
                2>&1 | tee -a "$log_file"; then
            log "Done $tag"
            return 0
        fi

        if ! grep -Eqi 'OutOfMemory|CUDA out|CUDA error: out of memory|CUDNN_STATUS_ALLOC_FAILED|ChildFailedError' "$log_file"; then
            log "Stop $tag: failure was not recognized as OOM; see $log_file"
            return 1
        fi
    done

    log "Stop $tag: exhausted OOM fallback batches"
    return 1
}

log "Start Qwen3-4B noanswer 51599 LoRA r8 trainings"
run_once "qwen3_4b_noanswer_51599_with_skill_lora_r8_ep${TOTAL_EPOCHS}" "$WITH_SKILL_DATA_DIR"
run_once "qwen3_4b_noanswer_51599_no_skill_lora_r8_ep${TOTAL_EPOCHS}" "$NO_SKILL_DATA_DIR"
log "All Qwen3-4B noanswer 51599 LoRA r8 trainings finished"
