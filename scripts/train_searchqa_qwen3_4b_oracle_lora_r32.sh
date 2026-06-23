#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/workspace/limingxin/modelscope_cache/Qwen__Qwen3-4B}"
DATA_DIR="${DATA_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_oracle_sft_standard}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen3_4b_oracle_lora}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_oracle_lora}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-2304}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
FALLBACK1_MICRO_BATCH_SIZE_PER_GPU="${FALLBACK1_MICRO_BATCH_SIZE_PER_GPU:-4}"
FALLBACK1_TRAIN_BATCH_SIZE="${FALLBACK1_TRAIN_BATCH_SIZE:-128}"
FALLBACK2_MICRO_BATCH_SIZE_PER_GPU="${FALLBACK2_MICRO_BATCH_SIZE_PER_GPU:-2}"
FALLBACK2_TRAIN_BATCH_SIZE="${FALLBACK2_TRAIN_BATCH_SIZE:-64}"

TAG="qwen3_4b_oracle_skill_lora_r${LORA_RANK}_ep${TOTAL_EPOCHS}"
SAVE_DIR="$CKPT_ROOT/$TAG"
LOG_FILE="$LOG_DIR/${TAG}.log"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" >&2
}

has_checkpoint() {
    find "$SAVE_DIR" -maxdepth 1 -type d -name 'global_step_*' | grep -q .
}

run_once() {
    local micro_bsz="$1"
    local train_bsz="$2"

    {
        echo "[$(date -u '+%F %T UTC')] Start $TAG"
        echo "MODEL_PATH=$MODEL_PATH"
        echo "DATA_DIR=$DATA_DIR"
        echo "SAVE_DIR=$SAVE_DIR"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE"
        echo "rank=$LORA_RANK alpha=$LORA_ALPHA max_length=$MAX_LENGTH micro_bsz=$micro_bsz train_bsz=$train_bsz lr=$LR epochs=$TOTAL_EPOCHS"
    } | tee -a "$LOG_FILE"

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
        model.lora_rank="$LORA_RANK" \
        model.lora_alpha="$LORA_ALPHA" \
        model.target_modules=all-linear \
        optim.lr="$LR" \
        trainer.default_local_dir="$SAVE_DIR" \
        trainer.default_hdfs_dir=null \
        trainer.project_name=SkillZero-searchqa-qwen3-4b-oracle-sft \
        trainer.experiment_name="$TAG" \
        trainer.logger="['console']" \
        trainer.total_epochs="$TOTAL_EPOCHS" \
        +trainer.skip_validation=True \
        2>&1 | tee -a "$LOG_FILE"
}

if has_checkpoint; then
    log "Skip $TAG: checkpoint already exists under $SAVE_DIR"
    exit 0
fi

attempts=(
    "$MICRO_BATCH_SIZE_PER_GPU:$TRAIN_BATCH_SIZE"
    "$FALLBACK1_MICRO_BATCH_SIZE_PER_GPU:$FALLBACK1_TRAIN_BATCH_SIZE"
    "$FALLBACK2_MICRO_BATCH_SIZE_PER_GPU:$FALLBACK2_TRAIN_BATCH_SIZE"
)

idx=0
for attempt in "${attempts[@]}"; do
    idx=$((idx + 1))
    micro_bsz="${attempt%%:*}"
    train_bsz="${attempt##*:}"
    if [ "$idx" -gt 1 ]; then
        log "Retry $TAG with lower batch: micro_bsz=$micro_bsz train_bsz=$train_bsz"
    fi
    if run_once "$micro_bsz" "$train_bsz"; then
        log "Done $TAG"
        exit 0
    fi
    if ! grep -Eqi 'OutOfMemory|CUDA out|CUDA error: out of memory|CUDNN_STATUS_ALLOC_FAILED|ChildFailedError' "$LOG_FILE"; then
        log "Stop $TAG: failure was not recognized as OOM; see $LOG_FILE"
        exit 1
    fi
done

log "Stop $TAG: exhausted OOM fallback batches"
exit 1
