#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set -euo pipefail
set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_path> [extra_hydra_overrides...]"
    exit 1
fi

MODEL_PATH="$1"
shift

SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/searchR1_processed_direct_skill0fmt}"
EXP_LOG_NAME="${EXP_LOG_NAME:-searchqa-full-sft-mainppo-eval}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log}"
mkdir -p "$LOG_DIR"

export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'
export LOG_PATH="$LOG_DIR/$EXP_LOG_NAME.log"
export USE_SKILL=True
export SKILL_FILE="$REPO_ROOT/skills/search/search_skills_nl.md"

num_cpus_per_env_worker=0.1
val_data_size=512
train_data_size=128
group_size=8
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONPATH="$REPO_ROOT" \
/workspace/limingxin/miniconda3/envs/skillzero/bin/python -m verl.trainer.main_ppo \
    ray_init.num_cpus=8 \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_ROOT/train.parquet" \
    data.val_files="$DATA_ROOT/val_1000.parquet" \
    data.test_files="$DATA_ROOT/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    env.env_name=search \
    env.use_skill=$USE_SKILL \
    env.skill_file=$SKILL_FILE \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url="$SEARCH_URL" \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    ocr.use_ocr=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name='SkillZero_search_eval' \
    trainer.experiment_name="$EXP_LOG_NAME" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=False \
    trainer.test_only=True \
    trainer.resume_mode=disable \
    "$@" 2>&1 | tee "$LOG_PATH"
