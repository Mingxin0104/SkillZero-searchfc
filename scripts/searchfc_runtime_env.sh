#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export REPO_ROOT
export WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/workspace}"

export CONDA_HOME="${CONDA_HOME:-$WORK_ROOT/miniconda3}"
export MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/models}"
export DATA_HOME="${DATA_HOME:-$WORK_ROOT/data}"
export LOG_HOME="${LOG_HOME:-$WORK_ROOT/logs}"
export CKPT_HOME="${CKPT_HOME:-$WORK_ROOT/checkpoints}"
export MERGED_HOME="${MERGED_HOME:-$WORK_ROOT/merged_models}"
export SERVED_HOME="${SERVED_HOME:-$WORK_ROOT/served_models}"
export CACHE_HOME="${CACHE_HOME:-$WORK_ROOT/hf_cache}"
export TMP_HOME="${TMP_HOME:-$WORK_ROOT/tmp}"

export SKILLZERO_ENV="${SKILLZERO_ENV:-$CONDA_HOME/envs/skillzero}"
export RETRIEVER_ENV="${RETRIEVER_ENV:-$CONDA_HOME/envs/retriever}"
export VLLM_ENV_ROOT="${VLLM_ENV_ROOT:-$CONDA_HOME/envs/vllmcleanpy310}"

export SEARCHR1_ROOT="${SEARCHR1_ROOT:-$DATA_HOME/searchR1}"
export RETRIEVER_INDEX_PATH="${RETRIEVER_INDEX_PATH:-$SEARCHR1_ROOT/e5_Flat.index}"
export RETRIEVER_CORPUS_PATH="${RETRIEVER_CORPUS_PATH:-$SEARCHR1_ROOT/wiki-18.jsonl}"
export RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
export RETRIEVER_MODEL_CACHE_ROOT="${RETRIEVER_MODEL_CACHE_ROOT:-$CACHE_HOME/hub/models--intfloat--e5-base-v2}"

mkdir -p "$WORK_ROOT" "$MODEL_ROOT" "$DATA_HOME" "$LOG_HOME" "$CKPT_HOME" "$MERGED_HOME" "$SERVED_HOME" "$CACHE_HOME" "$TMP_HOME"
