#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set -euo pipefail

PYTHON="${PYTHON:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
MODEL_PATH="${MODEL_PATH:-/public/limingxin/SkillZero/modelscope_cache/Qwen__Qwen2.5-3B-Instruct}"
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:-/public/limingxin/SkillZero/hf_cache/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd}"
INDEX_PATH="${INDEX_PATH:-/public/limingxin/SkillZero/data/searchR1/e5_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-/public/limingxin/SkillZero/data/searchR1/wiki-18.jsonl}"
RETRIEVER_BACKEND="${RETRIEVER_BACKEND:-http}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
INPUT_PATH="${INPUT_PATH:-/home/limingxin/data/searchR1_processed_direct_skill0fmt/train.parquet}"
SKILL_FILE="${SKILL_FILE:-$REPO_ROOT/skills/search/direct_retrieval.md}"
OUT_DIR="${OUT_DIR:-/public/limingxin/SkillZero/data/searchqa_trace_sft_train}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
START="${START:-0}"
TOTAL="${TOTAL:-117384}"
NUM_CANDIDATES="${NUM_CANDIDATES:-3}"
MAX_STEPS="${MAX_STEPS:-6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SKIP_FINAL_MERGE="${SKIP_FINAL_MERGE:-0}"

TRACE_DIR="$OUT_DIR/traces"
ALL_TRACE_DIR="$OUT_DIR/traces_all"
SFT_DIR="$OUT_DIR/sft"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$TRACE_DIR" "$ALL_TRACE_DIR" "$SFT_DIR" "$LOG_DIR"

end=$((START + TOTAL))
for ((shard_start=START; shard_start<end; shard_start+=SHARD_SIZE)); do
    shard_end=$((shard_start + SHARD_SIZE))
    if (( shard_end > end )); then
        shard_end=$end
    fi
    shard_limit=$((shard_end - shard_start))
    shard_name="$(printf '%06d_%06d' "$shard_start" "$shard_end")"
    trace_path="$TRACE_DIR/searchqa_skill_traces_${shard_name}.jsonl"
    all_trace_path="$ALL_TRACE_DIR/searchqa_skill_traces_all_${shard_name}.jsonl"
    sft_path="$SFT_DIR/searchqa_trace_sft_${shard_name}.parquet"
    log_path="$LOG_DIR/searchqa_trace_sft_${shard_name}.log"

    if [ -s "$sft_path" ]; then
        echo "skip existing $sft_path"
        continue
    fi

    echo "building shard $shard_name"
    PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/examples/data_preprocess/generate_searchqa_skill_traces_local.py" \
        --model_path "$MODEL_PATH" \
        --retriever_model_path "$RETRIEVER_MODEL_PATH" \
        --index_path "$INDEX_PATH" \
        --corpus_path "$CORPUS_PATH" \
        --retriever_backend "$RETRIEVER_BACKEND" \
        --search_url "$SEARCH_URL" \
        --input_path "$INPUT_PATH" \
        --output_path "$trace_path" \
        --save_all_path "$all_trace_path" \
        --skill_file "$SKILL_FILE" \
        --max_steps "$MAX_STEPS" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --start "$shard_start" \
        --limit "$shard_limit" \
        --num_candidates "$NUM_CANDIDATES" \
        --temperature "$TEMPERATURE" \
        2>&1 | tee "$log_path"

    PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/examples/data_preprocess/build_searchqa_trace_sft.py" \
        --trace_file "$trace_path" \
        --output_path "$sft_path" \
        2>&1 | tee -a "$log_path"
done

if [ "$SKIP_FINAL_MERGE" = "1" ]; then
    echo "skip final merge"
    exit 0
fi

PYTHONPATH="$REPO_ROOT" "$PYTHON" - <<'PY'
import os
import pandas as pd

out_dir = os.path.expanduser(os.environ.get("OUT_DIR", "/public/limingxin/SkillZero/data/searchqa_trace_sft_train"))
sft_dir = os.path.join(out_dir, "sft")
paths = [os.path.join(sft_dir, name) for name in sorted(os.listdir(sft_dir)) if name.endswith(".parquet")]
if not paths:
    print("no shard parquet files found")
    raise SystemExit(0)
frames = [pd.read_parquet(path) for path in paths]
df = pd.concat(frames, ignore_index=True)
final_path = os.path.join(out_dir, "train.parquet")
df.to_parquet(final_path, index=False)
print(f"saved={final_path} rows={len(df)} shards={len(paths)}")
PY
