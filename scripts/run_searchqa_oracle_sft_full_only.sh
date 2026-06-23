#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"

SKILL_DATA_DIR="${SKILL_DATA_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_oracle_sft_standard}"
NOSKILL_DATA_DIR="${NOSKILL_DATA_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_oracle_sft_standard_noskill}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_oracle_sft_4exp}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_oracle_sft_4exp}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
RUN_SMOKE="${RUN_SMOKE:-0}"

mkdir -p "$CKPT_ROOT" "$LOG_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" >&2
}

has_complete_checkpoint() {
    local save_dir="$1"
    local ckpt
    ckpt="$(find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1 || true)"
    [ -n "$ckpt" ] && [ -f "$ckpt/config.json" ] && [ -f "$ckpt/tokenizer_config.json" ]
}

run_full_exp() {
    local tag="$1"
    local data_dir="$2"
    local max_length="$3"
    local micro_bsz="$4"
    local train_bsz="$5"
    local total_steps="${6:-}"

    local save_dir="$CKPT_ROOT/$tag"
    local log_file="$LOG_DIR/${tag}.log"
    mkdir -p "$save_dir"

    if has_complete_checkpoint "$save_dir"; then
        log "Skip $tag: complete checkpoint already exists under $save_dir"
        return 0
    fi

    local step_args=()
    if [ -n "$total_steps" ]; then
        step_args=(trainer.total_training_steps="$total_steps")
    fi

    log "Start $tag data_dir=$data_dir max_length=$max_length micro_bsz=$micro_bsz train_bsz=$train_bsz total_steps=${total_steps:-epoch}"
    {
        echo "[$(date -u '+%F %T UTC')] Start $tag"
        echo "MODEL_PATH=$MODEL_PATH"
        echo "data_dir=$data_dir"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE"
        echo "max_length=$max_length micro_bsz=$micro_bsz train_bsz=$train_bsz total_epochs=$TOTAL_EPOCHS total_steps=${total_steps:-epoch}"
    } | tee "$log_file"

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PYTHONPATH="$REPO_ROOT" \
    "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="$data_dir/train.parquet" \
        data.val_files="$data_dir/val_1000.parquet" \
        data.multiturn.enable=True \
        data.multiturn.messages_key=messages \
        data.micro_batch_size_per_gpu="$micro_bsz" \
        data.train_batch_size="$train_bsz" \
        data.max_length="$max_length" \
        data.truncation=right \
        model.partial_pretrain="$MODEL_PATH" \
        model.enable_gradient_checkpointing=True \
        model.lora_rank=0 \
        optim.lr=1e-5 \
        trainer.default_local_dir="$save_dir" \
        trainer.default_hdfs_dir=null \
        trainer.project_name=SkillZero-searchqa-oracle-sft \
        trainer.experiment_name="$tag" \
        trainer.logger="['console']" \
        trainer.total_epochs="$TOTAL_EPOCHS" \
        +trainer.skip_validation=True \
        "${step_args[@]}" \
        2>&1 | tee -a "$log_file"

    log "Done $tag"
}

if [ "$RUN_SMOKE" = "1" ]; then
    SMOKE_ROOT="${SMOKE_ROOT:-/workspace/limingxin/checkpoints/searchqa_oracle_sft_full_smoke}"
    CKPT_ROOT="$SMOKE_ROOT"
    LOG_DIR="${SMOKE_LOG_DIR:-/workspace/limingxin/logs/searchqa_oracle_sft_full_smoke}"
    SMOKE_MAX_LENGTH="${SMOKE_MAX_LENGTH:-2304}"
    mkdir -p "$CKPT_ROOT" "$LOG_DIR"
    run_full_exp "skill_full_smoke_1step" "$SKILL_DATA_DIR" "$SMOKE_MAX_LENGTH" 1 2 1
    log "Smoke finished"
    exit 0
fi

run_full_exp "skill_full_ep1" "$SKILL_DATA_DIR" 2304 2 64
run_full_exp "noskill_full_ep1" "$NOSKILL_DATA_DIR" 1536 4 128

log "Both full SFT experiments finished"
