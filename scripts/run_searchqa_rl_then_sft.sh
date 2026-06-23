#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set -euo pipefail
set -x

RL_LOG="${RL_LOG:-$REPO_ROOT/log/skillzero_search_vl_3b_searchqa_2gpu_modelscope.log}"
SFT_SAVE_DIR="${SFT_SAVE_DIR:-/public/limingxin/SkillZero/checkpoints/searchqa_lora_sft_r32_full}"
SFT_LOG="${SFT_LOG:-$REPO_ROOT/log/searchqa-lora-sft-r32-full.log}"
SFT_EVAL_PRED="${SFT_EVAL_PRED:-$REPO_ROOT/log/searchqa_sft_eval_test_predictions.parquet}"
SFT_EVAL_LOG="${SFT_EVAL_LOG:-$REPO_ROOT/log/searchqa_sft_eval_main_eval.log}"
MODEL_PATH_VL="${MODEL_PATH_VL:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-VL-3B-Instruct}"
MODEL_PATH_TEXT="${MODEL_PATH_TEXT:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"

MODEL_PATH="$MODEL_PATH_VL" bash "$REPO_ROOT/scripts/train_searchqa_skillzero_3b_2gpu_modelscope.sh"

# The original RL script runs validation during training and test_after_train at the end.
grep -n "test" "$RL_LOG" | tail -n 50 || true

MODEL_PATH="$MODEL_PATH_TEXT" bash "$REPO_ROOT/scripts/train_searchqa_lora_sft_r32_full.sh" "$SFT_SAVE_DIR"

LATEST_SFT_CKPT="$(find "$SFT_SAVE_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
if [ -z "$LATEST_SFT_CKPT" ]; then
    echo "No SFT checkpoint found under $SFT_SAVE_DIR" >&2
    exit 1
fi

PYTHONPATH="$REPO_ROOT" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
/workspace/limingxin/miniconda3/envs/skillzero/bin/python \
    "$REPO_ROOT/examples/generate_searchqa_sft_eval_parquet.py" \
    --base_model "$MODEL_PATH_TEXT" \
    --adapter_path "$LATEST_SFT_CKPT" \
    --input_path "$HOME/data/searchR1_processed_direct/test.parquet" \
    --output_path "$SFT_EVAL_PRED"

PYTHONPATH="$REPO_ROOT" /workspace/limingxin/miniconda3/envs/skillzero/bin/python -m verl.trainer.main_eval \
    data.path="$SFT_EVAL_PRED" \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.reward_model_key=reward_model \
    2>&1 | tee "$SFT_EVAL_LOG"
