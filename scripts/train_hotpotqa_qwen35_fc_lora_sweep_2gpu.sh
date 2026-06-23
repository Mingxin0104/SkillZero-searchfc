#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
PYTHON_TRAIN_BIN="${PYTHON_TRAIN_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/limingxin/models/Qwen3.5-4B}"
SOURCE_DIR="${SOURCE_DIR:-/workspace/limingxin/data/hotpotqa_qwen35_fc_general_top10000_run1}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/hotpotqa_qwen35_fc_lora_variants_run1}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/hotpotqa_qwen35_fc_lora_sweep_2gpu_run1}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/hotpotqa_qwen35_fc_lora_sweep_2gpu_run1}"
DATASET_NAME="${DATASET_NAME:-hotpotqa}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-2304}"
VAL_SIZE="${VAL_SIZE:-1000}"
RANKS="${RANKS:-8 16 32 64}"
SAVE_FREQ="${SAVE_FREQ:-100}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"

mkdir -p "$DATA_ROOT" "$CKPT_ROOT" "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG_DIR/driver.log" >&2
}

has_checkpoint() {
    local save_dir="$1"
    find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | grep -q .
}

rank_to_alpha() {
    local rank="$1"
    echo $((rank * 2))
}

rank_to_attempts() {
    local rank="$1"
    case "$rank" in
        8)  printf '%s\n' "4:32" "2:16" "1:8" ;;
        16) printf '%s\n' "4:32" "2:16" "1:8" ;;
        32) printf '%s\n' "3:24" "2:16" "1:8" ;;
        64) printf '%s\n' "2:16" "1:8" "1:4" ;;
        *)  echo "Unsupported rank: $rank" >&2; return 1 ;;
    esac
}

prepare_dataset_variants() {
    local out_root="$DATA_ROOT/$DATASET_NAME"
    local with_skill_dir="$out_root/with_skill"
    local no_skill_dir="$out_root/no_skill"

    if [ -f "$with_skill_dir/train.parquet" ] && [ -f "$with_skill_dir/val_1000.parquet" ] && \
       [ -f "$no_skill_dir/train.parquet" ] && [ -f "$no_skill_dir/val_1000.parquet" ]; then
        log "Dataset variants already prepared for $DATASET_NAME"
        return 0
    fi

    log "Preparing dataset variants for $DATASET_NAME from $SOURCE_DIR"
    "$PYTHON_BIN" "$REPO_ROOT/examples/data_preprocess/prepare_qwen35_fc_sft_variants.py" \
        --input_dir "$SOURCE_DIR" \
        --with_skill_out_dir "$with_skill_dir" \
        --no_skill_out_dir "$no_skill_dir" \
        --val_size "$VAL_SIZE"
}

run_one() {
    local gpu="$1"
    local rank="$2"
    local variant="$3"
    local data_dir="$DATA_ROOT/$DATASET_NAME/$variant"
    local alpha
    alpha="$(rank_to_alpha "$rank")"
    local tag="${DATASET_NAME}_qwen35_fc_${variant}_lora_r${rank}_ep${TOTAL_EPOCHS}"
    local save_dir="$CKPT_ROOT/$tag"
    local log_file="$LOG_DIR/${tag}.log"

    mkdir -p "$save_dir"
    if has_checkpoint "$save_dir"; then
        log "Skip $tag: checkpoint already exists under $save_dir"
        return 0
    fi

    local attempt_idx=0
    local attempt
    while IFS= read -r attempt; do
        attempt_idx=$((attempt_idx + 1))
        local micro_bsz="${attempt%%:*}"
        local train_bsz="${attempt##*:}"
        if [ "$attempt_idx" -gt 1 ]; then
            log "Retry $tag on gpu=$gpu with lower batch: micro_bsz=$micro_bsz train_bsz=$train_bsz"
        fi

        {
            echo "[$(date -u '+%F %T UTC')] Start $tag"
            echo "MODEL_PATH=$MODEL_PATH"
            echo "DATA_DIR=$data_dir"
            echo "SAVE_DIR=$save_dir"
            echo "CUDA_VISIBLE_DEVICES=$gpu NPROC_PER_NODE=1"
            echo "dataset=$DATASET_NAME variant=$variant rank=$rank alpha=$alpha max_length=$MAX_LENGTH micro_bsz=$micro_bsz train_bsz=$train_bsz lr=$LR epochs=$TOTAL_EPOCHS"
            echo "loss_mask_rule=assistant_only"
            echo "save_freq=$SAVE_FREQ max_ckpt_to_keep=$MAX_CKPT_TO_KEEP"
        } | tee -a "$log_file"

        if CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$REPO_ROOT" \
            "$PYTHON_TRAIN_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
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
                +model.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='["Qwen3_5DecoderLayer"]' \
                model.enable_gradient_checkpointing=True \
                model.lora_rank="$rank" \
                model.lora_alpha="$alpha" \
                model.target_modules=all-linear \
                optim.lr="$LR" \
                trainer.default_local_dir="$save_dir" \
                trainer.default_hdfs_dir=null \
                trainer.project_name=SkillZero-hotpotqa-qwen35-fc-sft-lora-sweep \
                trainer.experiment_name="$tag" \
                trainer.logger="['console']" \
                trainer.total_epochs="$TOTAL_EPOCHS" \
                +trainer.save_freq="$SAVE_FREQ" \
                +trainer.max_ckpt_to_keep="$MAX_CKPT_TO_KEEP" \
                +trainer.skip_validation=True \
                2>&1 | tee -a "$log_file"; then
            log "Done $tag"
            return 0
        fi

        if ! grep -Eqi 'OutOfMemory|CUDA out|CUDA error: out of memory|CUDNN_STATUS_ALLOC_FAILED|ChildFailedError' "$log_file"; then
            log "Stop $tag: failure was not recognized as OOM; see $log_file"
            return 1
        fi
    done < <(rank_to_attempts "$rank")

    log "Stop $tag: exhausted OOM fallback batches"
    return 1
}

worker_dataset() {
    log "Start dataset sweep: $DATASET_NAME"
    for rank in $RANKS; do
        run_one 0 "$rank" with_skill &
        pid0=$!
        run_one 1 "$rank" no_skill &
        pid1=$!
        wait "$pid0"
        wait "$pid1"
    done
    log "Finished dataset sweep: $DATASET_NAME"
}

prepare_dataset_variants
log "Start Qwen3.5-4B FC LoRA sweep for $DATASET_NAME"
worker_dataset
log "All Qwen3.5-4B FC LoRA sweep trainings finished for $DATASET_NAME"
