#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/workspace/limingxin/modelscope_cache/Qwen__Qwen3-4B}"
DATA_DIR="${DATA_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_oracle_sft_standard}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen3_4b_oracle_lora_sweep}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_oracle_lora_sweep}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
RANKS="${RANKS:-8 16 32 64}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-2304}"

MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
R64_MICRO_BATCH_SIZE_PER_GPU="${R64_MICRO_BATCH_SIZE_PER_GPU:-4}"
R64_TRAIN_BATCH_SIZE="${R64_TRAIN_BATCH_SIZE:-64}"
FALLBACK1_MICRO_BATCH_SIZE_PER_GPU="${FALLBACK1_MICRO_BATCH_SIZE_PER_GPU:-4}"
FALLBACK1_TRAIN_BATCH_SIZE="${FALLBACK1_TRAIN_BATCH_SIZE:-64}"
FALLBACK2_MICRO_BATCH_SIZE_PER_GPU="${FALLBACK2_MICRO_BATCH_SIZE_PER_GPU:-2}"
FALLBACK2_TRAIN_BATCH_SIZE="${FALLBACK2_TRAIN_BATCH_SIZE:-32}"

mkdir -p "$CKPT_ROOT" "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" >&2
}

has_checkpoint() {
    local save_dir="$1"
    find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | grep -q .
}

run_rank_once() {
    local rank="$1"
    local tag="$2"
    local save_dir="$3"
    local log_file="$4"
    local micro_bsz="$5"
    local train_bsz="$6"

    {
        echo "[$(date -u '+%F %T UTC')] Start $tag"
        echo "MODEL_PATH=$MODEL_PATH"
        echo "DATA_DIR=$DATA_DIR"
        echo "SAVE_DIR=$save_dir"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE"
        echo "rank=$rank alpha=$LORA_ALPHA max_length=$MAX_LENGTH micro_bsz=$micro_bsz train_bsz=$train_bsz lr=$LR epochs=$TOTAL_EPOCHS"
    } | tee -a "$log_file"

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PYTHONPATH="$REPO_ROOT" \
    "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="$DATA_DIR/train.parquet" \
        data.val_files="$DATA_DIR/val_1000.parquet" \
        data.multiturn.enable=True \
        data.multiturn.messages_key=messages \
        data.micro_batch_size_per_gpu="$micro_bsz" \
        data.train_batch_size="$train_bsz" \
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
        trainer.project_name=SkillZero-searchqa-qwen3-4b-oracle-sft \
        trainer.experiment_name="$tag" \
        trainer.logger="['console']" \
        trainer.total_epochs="$TOTAL_EPOCHS" \
        +trainer.skip_validation=True \
        2>&1 | tee -a "$log_file"
}

run_rank() {
    local rank="$1"
    local tag="qwen3_4b_oracle_skill_lora_r${rank}_ep${TOTAL_EPOCHS}"
    local save_dir="$CKPT_ROOT/$tag"
    local log_file="$LOG_DIR/${tag}.log"
    local attempts=()

    mkdir -p "$save_dir"
    if has_checkpoint "$save_dir"; then
        log "Skip $tag: checkpoint already exists under $save_dir"
        return 0
    fi

    if [ "$rank" = "64" ]; then
        attempts=("$R64_MICRO_BATCH_SIZE_PER_GPU:$R64_TRAIN_BATCH_SIZE" "$FALLBACK2_MICRO_BATCH_SIZE_PER_GPU:$FALLBACK2_TRAIN_BATCH_SIZE")
    else
        attempts=("$MICRO_BATCH_SIZE_PER_GPU:$TRAIN_BATCH_SIZE" "$FALLBACK1_MICRO_BATCH_SIZE_PER_GPU:$FALLBACK1_TRAIN_BATCH_SIZE" "$FALLBACK2_MICRO_BATCH_SIZE_PER_GPU:$FALLBACK2_TRAIN_BATCH_SIZE")
    fi

    log "Start $tag"
    local attempt_idx=0
    local attempt
    for attempt in "${attempts[@]}"; do
        attempt_idx=$((attempt_idx + 1))
        local micro_bsz="${attempt%%:*}"
        local train_bsz="${attempt##*:}"
        if [ "$attempt_idx" -gt 1 ]; then
            log "Retry $tag with lower batch: micro_bsz=$micro_bsz train_bsz=$train_bsz"
        fi

        if run_rank_once "$rank" "$tag" "$save_dir" "$log_file" "$micro_bsz" "$train_bsz"; then
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

log "Qwen3-4B generated-data LoRA sweep started. DATA_DIR=$DATA_DIR RANKS=$RANKS"
for rank in $RANKS; do
    run_rank "$rank"
done
log "All Qwen3-4B generated-data LoRA sweep trainings finished"
