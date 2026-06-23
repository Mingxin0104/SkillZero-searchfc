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
SHARD_SIZE="${SHARD_SIZE:-200}"
START="${START:-0}"
TOTAL="${TOTAL:-117384}"
NUM_CANDIDATES="${NUM_CANDIDATES:-3}"
MAX_STEPS="${MAX_STEPS:-6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
WORKER_ID="${WORKER_ID:-worker}"

TRACE_DIR="$OUT_DIR/traces"
ALL_TRACE_DIR="$OUT_DIR/traces_all"
SFT_DIR="$OUT_DIR/sft"
LOG_DIR="$OUT_DIR/logs"
LOCK_DIR="$OUT_DIR/locks"
DONE_DIR="$OUT_DIR/done"
mkdir -p "$TRACE_DIR" "$ALL_TRACE_DIR" "$SFT_DIR" "$LOG_DIR" "$LOCK_DIR" "$DONE_DIR"

end=$((START + TOTAL))
while true; do
    claimed=0
    for ((shard_start=START; shard_start<end; shard_start+=SHARD_SIZE)); do
        shard_end=$((shard_start + SHARD_SIZE))
        if (( shard_end > end )); then
            shard_end=$end
        fi
        shard_limit=$((shard_end - shard_start))
        shard_name="$(printf '%06d_%06d' "$shard_start" "$shard_end")"
        lock_path="$LOCK_DIR/$shard_name.lock"
        done_path="$DONE_DIR/$shard_name.done"
        sft_path="$SFT_DIR/searchqa_trace_sft_${shard_name}.parquet"

        if [ -f "$done_path" ] || [ -s "$sft_path" ]; then
            continue
        fi

        if mkdir "$lock_path" 2>/dev/null; then
            claimed=1
            trace_path="$TRACE_DIR/searchqa_skill_traces_${shard_name}.jsonl"
            all_trace_path="$ALL_TRACE_DIR/searchqa_skill_traces_all_${shard_name}.jsonl"
            log_path="$LOG_DIR/${WORKER_ID}_${shard_name}.log"

            echo "$(date -Is) $WORKER_ID building shard $shard_name" | tee -a "$LOG_DIR/${WORKER_ID}.log"
            set +e
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
            gen_status=${PIPESTATUS[0]}

            if [ "$gen_status" -eq 0 ]; then
                PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/examples/data_preprocess/build_searchqa_trace_sft.py" \
                    --trace_file "$trace_path" \
                    --output_path "$sft_path" \
                    2>&1 | tee -a "$log_path"
                sft_status=${PIPESTATUS[0]}
            else
                sft_status=$gen_status
            fi
            set -e

            if [ "$sft_status" -eq 0 ]; then
                date -Is > "$done_path"
                echo "$(date -Is) $WORKER_ID done shard $shard_name" | tee -a "$LOG_DIR/${WORKER_ID}.log"
            else
                echo "$(date -Is) $WORKER_ID failed shard $shard_name status=$sft_status" | tee -a "$LOG_DIR/${WORKER_ID}.log"
                rm -f "$sft_path"
            fi
            rmdir "$lock_path" 2>/dev/null || true
            break
        fi
    done

    if [ "$claimed" -eq 0 ]; then
        echo "$(date -Is) $WORKER_ID no shards left" | tee -a "$LOG_DIR/${WORKER_ID}.log"
        exit 0
    fi
done
