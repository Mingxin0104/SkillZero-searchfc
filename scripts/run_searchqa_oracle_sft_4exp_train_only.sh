#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"

SKILL_DATA_DIR="${SKILL_DATA_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_oracle_sft_standard}"
NOSKILL_DATA_DIR="${NOSKILL_DATA_DIR:-/workspace/limingxin/data/searchqa_qwen3_4b_oracle_sft_standard_noskill}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_oracle_sft_4exp}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_oracle_sft_4exp}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
mkdir -p "$CKPT_ROOT" "$LOG_DIR"

run_exp() {
    local tag="$1"
    local data_dir="$2"
    local mode="$3"
    local max_length="$4"
    local micro_bsz="$5"
    local train_bsz="$6"
    local lr="$7"

    local save_dir="$CKPT_ROOT/$tag"
    local log_file="$LOG_DIR/${tag}.log"
    mkdir -p "$save_dir"

    if find "$save_dir" -maxdepth 1 -type d -name 'global_step_*' | grep -q .; then
        echo "[$(date -u '+%F %T UTC')] Skip $tag: checkpoint already exists under $save_dir" | tee -a "$log_file"
        return 0
    fi

    local lora_rank=0
    local lora_args=()
    if [ "$mode" = "lora" ]; then
        lora_rank=32
        lora_args=(
            model.lora_alpha=16
            model.target_modules=all-linear
        )
    fi

    echo "[$(date -u '+%F %T UTC')] Start $tag" | tee "$log_file"
    echo "data_dir=$data_dir mode=$mode max_length=$max_length micro_bsz=$micro_bsz train_bsz=$train_bsz lr=$lr" | tee -a "$log_file"

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
        model.lora_rank="$lora_rank" \
        "${lora_args[@]}" \
        optim.lr="$lr" \
        trainer.default_local_dir="$save_dir" \
        trainer.default_hdfs_dir=null \
        trainer.project_name=SkillZero-searchqa-oracle-sft \
        trainer.experiment_name="$tag" \
        trainer.logger="['console']" \
        trainer.total_epochs="$TOTAL_EPOCHS" \
        +trainer.skip_validation=True \
        2>&1 | tee -a "$log_file"

    echo "[$(date -u '+%F %T UTC')] Done $tag" | tee -a "$log_file"
}

run_exp "skill_lora_r32_ep1" "$SKILL_DATA_DIR" "lora" 2304 12 384 1e-4
run_exp "noskill_lora_r32_ep1" "$NOSKILL_DATA_DIR" "lora" 1536 8 256 1e-4
run_exp "skill_full_ep1" "$SKILL_DATA_DIR" "full" 2304 2 64 1e-5
run_exp "noskill_full_ep1" "$NOSKILL_DATA_DIR" "full" 1536 4 128 1e-5
