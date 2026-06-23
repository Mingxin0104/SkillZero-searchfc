#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set -euo pipefail
set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <save_dir> [extra_hydra_overrides...]"
    exit 1
fi

SAVE_DIR="$1"
shift

TRAIN_DATA="${TRAIN_DATA:-$HOME/data/searchR1_processed_direct/searchqa_sft_10k_1k/train_10k.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/searchR1_processed_direct/searchqa_sft_10k_1k/val_1k.parquet}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-}"
MODELSCOPE_CACHE_DIR="${MODELSCOPE_CACHE_DIR:-/public/limingxin/SkillZero/modelscope_cache}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log}"
EXP_NAME="${EXP_NAME:-searchqa-sft-skillzero-qwen2.5-3b-lora32-10k1k}"
mkdir -p "$LOG_DIR" "$SAVE_DIR" "$MODELSCOPE_CACHE_DIR"

export VERL_USE_MODELSCOPE=True

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$(PYTHONPATH="$REPO_ROOT" /workspace/limingxin/miniconda3/envs/skillzero/bin/python - <<'PY'
from modelscope import snapshot_download
import os

model_id = os.environ["MODEL_ID"]
cache_dir = os.environ["MODELSCOPE_CACHE_DIR"]
local_dir = os.path.join(cache_dir, model_id.replace("/", "__"))
path = snapshot_download(model_id=model_id, cache_dir=cache_dir, local_dir=local_dir)
print(path)
PY
)"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONPATH="$REPO_ROOT" \
/workspace/limingxin/miniconda3/envs/skillzero/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.prompt_key=question \
    data.response_key=answer \
    'data.prompt_dict_keys=[]' \
    'data.response_dict_keys=[]' \
    data.micro_batch_size_per_gpu=4 \
    data.train_batch_size=64 \
    data.max_length=1024 \
    data.truncation=right \
    model.partial_pretrain="$MODEL_PATH" \
    model.enable_gradient_checkpointing=True \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    optim.lr=1e-4 \
    trainer.default_local_dir="$SAVE_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.project_name=SkillZero-searchqa-sft \
    trainer.experiment_name="$EXP_NAME" \
    trainer.logger="['console']" \
    trainer.total_epochs=1 \
    "$@" 2>&1 | tee "$LOG_DIR/${EXP_NAME}.log"
