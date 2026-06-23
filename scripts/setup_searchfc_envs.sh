#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/searchfc_runtime_env.sh"

MINICONDA_VERSION="${MINICONDA_VERSION:-py310_25.3.1-1}"
MINICONDA_INSTALLER="Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}}"

if [ ! -x "$CONDA_HOME/bin/conda" ]; then
    mkdir -p "$WORK_ROOT"
    curl -L "$MINICONDA_URL" -o "$WORK_ROOT/${MINICONDA_INSTALLER}"
    bash "$WORK_ROOT/${MINICONDA_INSTALLER}" -b -p "$CONDA_HOME"
fi

source "$CONDA_HOME/etc/profile.d/conda.sh"

create_or_update_env() {
    local env_name="$1"
    local python_version="$2"
    conda create -y -n "$env_name" "python=${python_version}" || conda install -y -n "$env_name" "python=${python_version}"
}

create_or_update_env skillzero 3.10
create_or_update_env retriever 3.10
create_or_update_env vllmcleanpy310 3.10

conda run -n skillzero pip install -U pip setuptools wheel
conda run -n skillzero pip install -r "$REPO_ROOT/requirements.txt"
conda run -n skillzero pip install -e "$REPO_ROOT"

conda run -n retriever pip install -U pip setuptools wheel
conda run -n retriever pip install torch transformers datasets fastapi uvicorn pydantic requests sentencepiece accelerate faiss-gpu-cu12

conda run -n vllmcleanpy310 pip install -U pip setuptools wheel
conda run -n vllmcleanpy310 pip install torch transformers datasets requests pandas pyarrow fastapi uvicorn pydantic peft safetensors sentencepiece accelerate
conda run -n vllmcleanpy310 pip install vllm

cat <<EOF
Finished.

Conda root: $CONDA_HOME
skillzero env: $SKILLZERO_ENV
retriever env: $RETRIEVER_ENV
vllm env: $VLLM_ENV_ROOT
EOF
