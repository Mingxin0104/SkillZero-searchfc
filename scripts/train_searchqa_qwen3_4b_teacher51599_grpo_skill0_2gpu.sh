#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/limingxin/modelscope_cache/Qwen__Qwen3-4B}"
DATA_ROOT="${DATA_ROOT:-/workspace/limingxin/data/searchqa_qwen3_4b_teacher_grpo_51599_noskill_skill0fmt}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/limingxin/checkpoints/searchqa_qwen3_4b_teacher51599_grpo_skill0_2gpu}"
LOG_DIR="${LOG_DIR:-/workspace/limingxin/logs/searchqa_qwen3_4b_teacher51599_grpo_skill0_2gpu}"
RUN_NAME="${RUN_NAME:-qwen3_4b_teacher51599_grpo_skill0_noskill}"
USE_SKILL="${USE_SKILL:-False}"
SKILL_FILE="${SKILL_FILE:-$REPO_ROOT/skills/search/search_skills_nl.md}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-512}"
GROUP_SIZE="${GROUP_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-403}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.68}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
RAY_TMP_DIR="${RAY_TMP_DIR:-/workspace/limingxin/rt/grpo51599_noskill}"

SAVE_DIR="$CKPT_ROOT/$RUN_NAME"
LOG_PATH="$LOG_DIR/$RUN_NAME.log"
mkdir -p "$SAVE_DIR" "$LOG_DIR" "$RAY_TMP_DIR"

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG_DIR/driver.log" >&2
}

export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'
export LOG_PATH

log "Start $RUN_NAME"
log "MODEL_PATH=$MODEL_PATH"
log "DATA_ROOT=$DATA_ROOT"
log "SAVE_DIR=$SAVE_DIR"
log "RAY_TMP_DIR=$RAY_TMP_DIR"
log "USE_SKILL=$USE_SKILL TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONPATH="$REPO_ROOT" \
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    ray_init.num_cpus=$RAY_NUM_CPUS \
    +ray_init.include_dashboard=False \
    +ray_init._temp_dir="$RAY_TMP_DIR" \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_ROOT/train.parquet" \
    data.val_files="$DATA_ROOT/val_1000.parquet" \
    data.test_files="$DATA_ROOT/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
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
    env.skill_file="$SKILL_FILE" \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$GROUP_SIZE \
    env.history_length=4 \
    env.search.search_url="$SEARCH_URL" \
    env.resources_per_worker.num_cpus=0.1 \
    ocr.use_ocr=False \
    trainer.default_local_dir="$SAVE_DIR" \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name='SkillZero_searchqa_grpo_teacher51599' \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.test_after_train=False \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.resume_mode=auto \
    2>&1 | tee "$LOG_PATH"

log "Finished $RUN_NAME"
