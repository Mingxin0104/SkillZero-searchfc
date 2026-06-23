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
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen3_4b_noanswer_51599_lora_sweep_2gpu}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_noanswer_51599_lora_sweep_2gpu}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-2304}"

mkdir -p "$CKPT_ROOT" "$LOG_DIR"

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

run_one() {
    local gpu="$1"
    local rank="$2"
    local variant="$3"
    local data_dir="$4"
    local alpha
    alpha="$(rank_to_alpha "$rank")"
    local tag="qwen3_4b_noanswer_51599_${variant}_lora_r${rank}_ep${TOTAL_EPOCHS}"
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
            echo "rank=$rank alpha=$alpha max_length=$MAX_LENGTH micro_bsz=$micro_bsz train_bsz=$train_bsz lr=$LR epochs=$TOTAL_EPOCHS"
            echo "loss_mask_rule=assistant_only; retriever_docs_are_user_messages_and_do_not_contribute_to_loss"
        } | tee -a "$log_file"

        if CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$REPO_ROOT" \
            "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node=1 \
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
                model.lora_rank="$rank" \
                model.lora_alpha="$alpha" \
                model.target_modules=all-linear \
                optim.lr="$LR" \
                trainer.default_local_dir="$save_dir" \
                trainer.default_hdfs_dir=null \
                trainer.project_name=SkillZero-searchqa-qwen3-4b-noanswer-51599-sft-sweep-2gpu \
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
    done < <(rank_to_attempts "$rank")

    log "Stop $tag: exhausted OOM fallback batches"
    return 1
}

worker_gpu0() {
    run_one 0 8  with_skill "$WITH_SKILL_DATA_DIR"
    run_one 0 16 with_skill "$WITH_SKILL_DATA_DIR"
    run_one 0 32 with_skill "$WITH_SKILL_DATA_DIR"
    run_one 0 64 with_skill "$WITH_SKILL_DATA_DIR"
}

worker_gpu1() {
    run_one 1 8  no_skill "$NO_SKILL_DATA_DIR"
    run_one 1 16 no_skill "$NO_SKILL_DATA_DIR"
    run_one 1 32 no_skill "$NO_SKILL_DATA_DIR"
    run_one 1 64 no_skill "$NO_SKILL_DATA_DIR"
}

log "Start Qwen3-4B noanswer 51599 LoRA sweep on 2 GPUs"
worker_gpu0 &
pid0=$!
worker_gpu1 &
pid1=$!

wait "$pid0"
wait "$pid1"

log "All Qwen3-4B noanswer 51599 LoRA sweep trainings finished"
