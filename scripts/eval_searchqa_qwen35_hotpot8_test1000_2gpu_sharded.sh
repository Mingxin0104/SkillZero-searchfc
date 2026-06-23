#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source "$SCRIPT_DIR/searchfc_runtime_env.sh"
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$VLLM_ENV_ROOT/bin/python}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-$RETRIEVER_ENV/bin/python}"
VLLM_BIN="${VLLM_BIN:-$VLLM_ENV_ROOT/bin/vllm}"

QWEN35_BASE_MODEL="${QWEN35_BASE_MODEL:-$MODEL_ROOT/Qwen3.5-4B}"
QWEN35_TEXT_BASE_MODEL="${QWEN35_TEXT_BASE_MODEL:-$MODEL_ROOT/Qwen3.5-4B-text}"
QWEN35_CKPT_ROOT="${QWEN35_CKPT_ROOT:-$CKPT_HOME/hotpotqa_qwen35_fc_lora_sweep_2gpu_run1}"
QWEN35_MERGED_ROOT="${QWEN35_MERGED_ROOT:-$MERGED_HOME/hotpotqa_qwen35_fc_lora_sweep_2gpu_run1_searchqa_eval}"
QWEN35_SERVE_ROOT="${QWEN35_SERVE_ROOT:-$SERVED_HOME/hotpotqa_qwen35_fc_lora_sweep_2gpu_run1_searchqa_eval}"

LOG_DIR="${LOG_DIR:-$LOG_HOME/searchqa_qwen35_hotpot8_test1000_2gpu_sharded}"
DATA_ROOT="${DATA_ROOT:-$DATA_HOME/searchR1_processed_direct_skill0fmt}"
TEST_FILE="${TEST_FILE:-$DATA_HOME/nq_hotpot_qwen35_skill0fmt_input_10k1k/hotpotqa_test_1000.parquet}"

RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
RETRIEVER_VISIBLE_DEVICES="${RETRIEVER_VISIBLE_DEVICES:-${GPU_A},${GPU_B}}"
RETRIEVER_SHARED_PORT="${RETRIEVER_SHARED_PORT:-8000}"
RETRIEVER_SHARED_LOG="${RETRIEVER_SHARED_LOG:-$LOG_DIR/retrieval_server_gpu_sharded.log}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
RAY_TMP_ROOT="${RAY_TMP_ROOT:-$TMP_HOME/sq9}"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-8100}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
FC_BATCH_SIZE="${FC_BATCH_SIZE:-16}"
FC_REQUEST_WORKERS="${FC_REQUEST_WORKERS:-32}"
FC_RETRIEVAL_WORKERS="${FC_RETRIEVAL_WORKERS:-64}"
FC_MAX_STEPS="${FC_MAX_STEPS:-4}"

mkdir -p "$QWEN35_MERGED_ROOT" "$LOG_DIR" "$RAY_TMP_ROOT"

RUN_SPECS=(
  "qwen35_4b_base_test1000|ready|$QWEN35_BASE_MODEL"
  "hotpotqa_qwen35_fc_with_skill_lora_r8_ep1_test1000|merge|hotpotqa_qwen35_fc_with_skill_lora_r8_ep1"
  "hotpotqa_qwen35_fc_no_skill_lora_r8_ep1_test1000|merge|hotpotqa_qwen35_fc_no_skill_lora_r8_ep1"
  "hotpotqa_qwen35_fc_with_skill_lora_r16_ep1_test1000|merge|hotpotqa_qwen35_fc_with_skill_lora_r16_ep1"
  "hotpotqa_qwen35_fc_no_skill_lora_r16_ep1_test1000|merge|hotpotqa_qwen35_fc_no_skill_lora_r16_ep1"
  "hotpotqa_qwen35_fc_with_skill_lora_r32_ep1_test1000|merge|hotpotqa_qwen35_fc_with_skill_lora_r32_ep1"
  "hotpotqa_qwen35_fc_no_skill_lora_r32_ep1_test1000|merge|hotpotqa_qwen35_fc_no_skill_lora_r32_ep1"
  "hotpotqa_qwen35_fc_with_skill_lora_r64_ep1_test1000|merge|hotpotqa_qwen35_fc_with_skill_lora_r64_ep1"
  "hotpotqa_qwen35_fc_no_skill_lora_r64_ep1_test1000|merge|hotpotqa_qwen35_fc_no_skill_lora_r64_ep1"
)

log() {
    echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG_DIR/driver.log"
}

sync_qwen35_processor_files() {
    local target_dir="$1"
    local processor_file
    for processor_file in \
        preprocessor_config.json \
        video_preprocessor_config.json \
        chat_template.jinja \
        tokenizer.json \
        tokenizer_config.json; do
        if [ -f "$QWEN35_BASE_MODEL/$processor_file" ] && [ ! -f "$target_dir/$processor_file" ]; then
            cp "$QWEN35_BASE_MODEL/$processor_file" "$target_dir/$processor_file"
        fi
    done
}

normalize_qwen35_text_config() {
    local target_dir="$1"
    if [ ! -f "$target_dir/config.json" ]; then
        return 0
    fi

    TARGET_DIR="$target_dir" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["TARGET_DIR"]) / "config.json"
cfg = json.loads(path.read_text())

if cfg.get("model_type") != "qwen3_5_text":
    raise SystemExit(0)

cfg.setdefault("image_token_id", 248056)
cfg.setdefault("video_token_id", 248057)
cfg.setdefault("vision_start_token_id", 248053)
cfg.setdefault("vision_end_token_id", 248054)
cfg.setdefault("vision_config", {"spatial_merge_size": 2})

path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
PY
}

wrap_qwen35_text_for_vllm() {
    local src_dir="$1"
    local out_dir="$2"
    mkdir -p "$out_dir"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    cp -a "$src_dir"/. "$out_dir"/
    sync_qwen35_processor_files "$out_dir"

SRC_DIR="$src_dir" OUT_DIR="$out_dir" QWEN35_BASE_MODEL="$QWEN35_BASE_MODEL" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from safetensors import safe_open

src_dir = Path(os.environ["SRC_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
base_dir = Path(os.environ["QWEN35_BASE_MODEL"])

src_cfg = json.loads((src_dir / "config.json").read_text())
base_cfg = json.loads((base_dir / "config.json").read_text())

if src_cfg.get("model_type") != "qwen3_5_text":
    raise SystemExit(0)

wrapped = dict(base_cfg)
wrapped["architectures"] = ["Qwen3_5ForConditionalGeneration"]
wrapped["text_config"] = dict(src_cfg)
wrapped["text_config"]["architectures"] = ["Qwen3_5ForCausalLM"]
wrapped["text_config"]["model_type"] = "qwen3_5_text"
wrapped["text_config"]["pad_token_id"] = src_cfg.get("pad_token_id", 248044)
wrapped.setdefault("image_token_id", 248056)
wrapped.setdefault("video_token_id", 248057)
wrapped.setdefault("vision_start_token_id", 248053)
wrapped.setdefault("vision_end_token_id", 248054)
wrapped.setdefault("vision_config", {"spatial_merge_size": 2})

(out_dir / "config.json").write_text(json.dumps(wrapped, ensure_ascii=False, indent=2) + "\n")

src_weight = out_dir / "model.safetensors"
if src_weight.exists():
    with safe_open(str(src_weight), framework="pt", device="cpu") as f:
        merged_keys = list(f.keys())

    base_index_path = base_dir / "model.safetensors.index.json"
    if base_index_path.exists():
        base_index = json.loads(base_index_path.read_text())
        weight_map = {key: "model.safetensors" for key in merged_keys}
        for key, shard in base_index["weight_map"].items():
            if key not in weight_map:
                weight_map[key] = shard
                shard_path = base_dir / shard
                target_shard = out_dir / shard
                if not target_shard.exists():
                    os.symlink(shard_path, target_shard)

        total_size = src_weight.stat().st_size
        used_shards = {weight_map[key] for key in weight_map if weight_map[key] != "model.safetensors"}
        for shard in used_shards:
            total_size += (out_dir / shard).stat().st_size

        index_payload = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        (out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n"
        )
PY
}

ensure_qwen35_text_base() {
    if [ -f "$QWEN35_TEXT_BASE_MODEL/config.json" ]; then
        sync_qwen35_processor_files "$QWEN35_TEXT_BASE_MODEL"
        normalize_qwen35_text_config "$QWEN35_TEXT_BASE_MODEL"
        return 0
    fi

    mkdir -p "$QWEN35_TEXT_BASE_MODEL"
    cp -a "$QWEN35_BASE_MODEL"/model.safetensors* "$QWEN35_TEXT_BASE_MODEL"/
    cp -a "$QWEN35_BASE_MODEL"/merges.txt "$QWEN35_TEXT_BASE_MODEL"/
    cp -a "$QWEN35_BASE_MODEL"/vocab.json "$QWEN35_TEXT_BASE_MODEL"/
    sync_qwen35_processor_files "$QWEN35_TEXT_BASE_MODEL"
    QWEN35_BASE_MODEL="$QWEN35_BASE_MODEL" QWEN35_TEXT_BASE_MODEL="$QWEN35_TEXT_BASE_MODEL" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["QWEN35_BASE_MODEL"])
out = Path(os.environ["QWEN35_TEXT_BASE_MODEL"])
cfg = json.loads((base / "config.json").read_text())
text_cfg = dict(cfg["text_config"])
text_cfg["architectures"] = ["Qwen3_5ForCausalLM"]
text_cfg["model_type"] = "qwen3_5_text"
text_cfg["pad_token_id"] = 248044
text_cfg["eos_token_id"] = text_cfg.get("eos_token_id", 248044)
text_cfg["transformers_version"] = cfg.get("transformers_version", text_cfg.get("transformers_version"))
(out / "config.json").write_text(json.dumps(text_cfg, ensure_ascii=False, indent=2) + "\n")
PY
    normalize_qwen35_text_config "$QWEN35_TEXT_BASE_MODEL"
}

latest_ckpt() {
    local ckpt_root="$1"
    local tag="$2"
    find "$ckpt_root/$tag" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1
}

resolve_retriever_model() {
    if [ -d "$RETRIEVER_MODEL" ]; then
        return 0
    fi
    if [ -d "$RETRIEVER_MODEL_CACHE_ROOT/snapshots" ]; then
        local snapshot
        snapshot="$(find "$RETRIEVER_MODEL_CACHE_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
        if [ -n "$snapshot" ]; then
            RETRIEVER_MODEL="$snapshot"
            export RETRIEVER_MODEL
        fi
    fi
}

wait_for_retriever() {
    local health_url="$1"
    local retries="${2:-360}"
    local sleep_seconds="${3:-5}"
    local i
    for ((i=1; i<=retries; i++)); do
        if curl -fsS -m 2 "$health_url" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$sleep_seconds"
    done
    return 1
}

wait_for_url() {
    local url="$1"
    local retries="${2:-180}"
    local sleep_seconds="${3:-5}"
    local i
    for ((i=1; i<=retries; i++)); do
        if curl -fsS -m 3 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$sleep_seconds"
    done
    return 1
}

ensure_shared_gpu_retriever() {
    local health_url="http://127.0.0.1:${RETRIEVER_SHARED_PORT}/docs"
    resolve_retriever_model
    if curl -fsS -m 2 "$health_url" >/dev/null 2>&1; then
        log "Shared GPU retriever already available on GPUs ${RETRIEVER_VISIBLE_DEVICES} port ${RETRIEVER_SHARED_PORT}"
        return 0
    fi

    log "Starting shared GPU retriever on GPUs ${RETRIEVER_VISIBLE_DEVICES} port ${RETRIEVER_SHARED_PORT}"
    CUDA_VISIBLE_DEVICES="$RETRIEVER_VISIBLE_DEVICES" nohup "$RETRIEVER_PYTHON" \
        "$REPO_ROOT/examples/search/retriever/retrieval_server.py" \
        --index_path "$RETRIEVER_INDEX_PATH" \
        --corpus_path "$RETRIEVER_CORPUS_PATH" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL" \
        --port "$RETRIEVER_SHARED_PORT" \
        --faiss_gpu \
        > "$RETRIEVER_SHARED_LOG" 2>&1 &
    wait_for_retriever "$health_url" 360 5
    log "Shared GPU retriever is ready on GPUs ${RETRIEVER_VISIBLE_DEVICES} port ${RETRIEVER_SHARED_PORT}"
}

merge_one() {
    local tag="$1"
    local ckpt
    ckpt="$(latest_ckpt "$QWEN35_CKPT_ROOT" "$tag")"
    if [ -z "$ckpt" ]; then
        echo "Missing checkpoint for $tag under $QWEN35_CKPT_ROOT/$tag" >&2
        exit 1
    fi

    local out="$QWEN35_MERGED_ROOT/$tag"
    if [ -f "$out/config.json" ]; then
        sync_qwen35_processor_files "$out"
        normalize_qwen35_text_config "$out"
        log "Skip merge $tag: $out exists"
        return 0
    fi

    log "Merging $tag from $ckpt"
    rm -rf "$out"
    "$PYTHON_BIN" "$REPO_ROOT/examples/merge_searchqa_lora.py" \
        --base_model "$QWEN35_BASE_MODEL" \
        --adapter_path "$ckpt" \
        --output_path "$out" \
        2>&1 | tee "$LOG_DIR/${tag}.merge.log"
    sync_qwen35_processor_files "$out"
    normalize_qwen35_text_config "$out"
}

ensure_vllm_servable_model() {
    local model_path="$1"
    local model_name="$2"
    local serve_dir="$QWEN35_SERVE_ROOT/$model_name"
    local top_model_type
    local need_rebuild="0"

    if [ ! -f "$model_path/config.json" ]; then
        echo "$model_path"
        return 0
    fi

    top_model_type="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$model_path/config.json").read_text())
print(cfg.get("model_type", ""))
PY
)"

    if [ "$top_model_type" != "qwen3_5_text" ]; then
        echo "$model_path"
        return 0
    fi

    if [ ! -f "$serve_dir/config.json" ] || [ ! -f "$serve_dir/model.safetensors.index.json" ]; then
        need_rebuild="1"
    else
        if ! find "$serve_dir" -maxdepth 1 -type l -name 'model.safetensors-*.safetensors' | grep -q .; then
            need_rebuild="1"
        fi
    fi

    if [ "$need_rebuild" = "1" ]; then
        echo "[$(date -u '+%F %T UTC')] Wrapping $model_name for vLLM serve compatibility" >&2
        wrap_qwen35_text_for_vllm "$model_path" "$serve_dir"
    fi
    echo "$serve_dir"
}

summarize_one() {
    local run_name="$1"
    local gen_dir="$LOG_DIR/${run_name}.generations/test"

    "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_skill0_original_metric.py" \
        --generation_dir "$gen_dir" \
        --out "$LOG_DIR/${run_name}.skill0_original_metric.json" \
        > "$LOG_DIR/${run_name}.metric.stdout.log" 2>&1

    "$PYTHON_BIN" "$REPO_ROOT/examples/summarize_searchqa_search_rate.py" \
        --generation_dir "$gen_dir" \
        --use_trajectory_metrics \
        --out "$LOG_DIR/${run_name}.trajectory_metric.json" \
        >> "$LOG_DIR/${run_name}.metric.stdout.log" 2>&1
}

eval_one() {
    local run_name="$1"
    local model_path="$2"
    local gpu="$3"
    local search_url="$4"
    local serve_model_path
    local port="$((VLLM_PORT_BASE + gpu))"
    local session="sq35hotpot_gpu${gpu}"
    local vllm_log="$LOG_DIR/${run_name}.vllm.log"
    local eval_log="$LOG_DIR/${run_name}.eval.log"
    local base_url="http://127.0.0.1:${port}/v1"
    local short_name
    short_name="$(printf '%s' "$run_name" | sed 's/_test1000//g; s/hotpotqa_qwen35_fc_/hq35_/g; s/_with_skill/_ws/g; s/_no_skill/_ns/g; s/_lora_r/r/g; s/_ep1//g; s/qwen35_4b_base/q35b/g')"
    local short_tmp
    short_tmp="$(printf '%s' "$short_name" | cut -c1-4)"
    local ray_tmp="${RAY_TMP_ROOT}/${gpu}_${short_tmp}"

    if [ -f "$LOG_DIR/${run_name}.skill0_original_metric.json" ] && [ -f "$LOG_DIR/${run_name}.trajectory_metric.json" ]; then
        log "Skip eval $run_name: metrics already exist"
        return 0
    fi

    rm -rf "$LOG_DIR/${run_name}.generations" "$ray_tmp"
    mkdir -p "$ray_tmp"
    serve_model_path="$(ensure_vllm_servable_model "$model_path" "$run_name")"

    log "Start eval $run_name on GPU $gpu model=$model_path serve_model=$serve_model_path"
    tmux kill-session -t "$session" 2>/dev/null || true
    tmux new-session -d -s "$session" \
        "cd '$REPO_ROOT' && \
         export CUDA_VISIBLE_DEVICES='$gpu' && \
         export VLLM_USE_FLASHINFER_SAMPLER=0 && \
         '$VLLM_BIN' serve '$serve_model_path' \
           --host 127.0.0.1 \
           --port '$port' \
           --tensor-parallel-size 1 \
           --max-model-len '$VLLM_MAX_MODEL_LEN' \
           --gpu-memory-utilization '$VLLM_GPU_MEMORY_UTILIZATION' \
           --trust-remote-code \
           --enforce-eager \
           --skip-mm-profiling \
           --gdn-prefill-backend triton \
           --attention-backend TRITON_ATTN \
           --mm-encoder-attn-backend TORCH_SDPA \
           --reasoning-parser qwen3 \
           --enable-auto-tool-choice \
           --tool-call-parser qwen3_coder \
           --api-key '$VLLM_API_KEY' \
           2>&1 | tee '$vllm_log'"

    if ! wait_for_url "http://127.0.0.1:${port}/health" 180 5; then
        log "vLLM failed to start for $run_name on GPU $gpu"
        tmux kill-session -t "$session" 2>/dev/null || true
        return 1
    fi

    PYTHONPATH="$REPO_ROOT" \
    "$PYTHON_BIN" "$REPO_ROOT/examples/eval_searchqa_vllm_fc.py" \
        --model_path "$model_path" \
        --model_name "$serve_model_path" \
        --input_path "$TEST_FILE" \
        --output_dir "$LOG_DIR/${run_name}.generations" \
        --search_url "$search_url" \
        --vllm_base_url "$base_url" \
        --vllm_api_key "$VLLM_API_KEY" \
        --batch_size "$FC_BATCH_SIZE" \
        --max_steps "$FC_MAX_STEPS" \
        --max_new_tokens "$MAX_RESPONSE_LENGTH" \
        --request_workers "$FC_REQUEST_WORKERS" \
        --retrieval_workers "$FC_RETRIEVAL_WORKERS" \
        2>&1 | tee "$eval_log"

    tmux kill-session -t "$session" 2>/dev/null || true

    summarize_one "$run_name"
    log "Done eval $run_name"
}

run_pair() {
    local idx_a="$1"
    local idx_b="$2"
    local model_path_a model_path_b
    local run_name_a kind_a value_a
    local run_name_b kind_b value_b

    IFS='|' read -r run_name_a kind_a value_a <<< "${RUN_SPECS[$idx_a]}"
    if [ "$kind_a" = "merge" ]; then
        model_path_a="$QWEN35_MERGED_ROOT/$value_a"
        sync_qwen35_processor_files "$model_path_a"
    else
        model_path_a="$value_a"
    fi
    eval_one "$run_name_a" "$model_path_a" "$GPU_A" "http://127.0.0.1:${RETRIEVER_SHARED_PORT}/retrieve" &
    pid_a=$!

    if [ "$idx_b" -ge 0 ]; then
        IFS='|' read -r run_name_b kind_b value_b <<< "${RUN_SPECS[$idx_b]}"
        if [ "$kind_b" = "merge" ]; then
            model_path_b="$QWEN35_MERGED_ROOT/$value_b"
            sync_qwen35_processor_files "$model_path_b"
        else
            model_path_b="$value_b"
        fi
        eval_one "$run_name_b" "$model_path_b" "$GPU_B" "http://127.0.0.1:${RETRIEVER_SHARED_PORT}/retrieve" &
        pid_b=$!
        wait "$pid_a"
        wait "$pid_b"
    else
        wait "$pid_a"
    fi
}

log "SearchQA Qwen3.5 Hotpot8 test1000 eval started"
log "TEST_FILE=$TEST_FILE"
ensure_qwen35_text_base

for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r _ kind value <<< "$spec"
    if [ "$kind" = "merge" ]; then
        merge_one "$value"
    fi
done

ensure_shared_gpu_retriever

total="${#RUN_SPECS[@]}"
idx=0
while [ "$idx" -lt "$total" ]; do
    next=$((idx + 1))
    if [ "$next" -lt "$total" ]; then
        run_pair "$idx" "$next"
    else
        run_pair "$idx" -1
    fi
    idx=$((idx + 2))
done

LOG_DIR="$LOG_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os

log_dir = os.environ["LOG_DIR"]
rows = []

for name in sorted(os.listdir(log_dir)):
    if not name.endswith(".skill0_original_metric.json"):
        continue
    run_name = name[:-len(".skill0_original_metric.json")]
    metric_path = os.path.join(log_dir, name)
    traj_path = os.path.join(log_dir, f"{run_name}.trajectory_metric.json")

    with open(metric_path, encoding="utf-8") as f:
        metric = json.load(f)

    traj = {}
    if os.path.exists(traj_path):
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)

    rows.append(
        {
            "name": run_name,
            "total": metric.get("total"),
            "acc": metric.get("acc"),
            "success_rate": traj.get("success_rate", metric.get("acc")),
            "answer_rate": metric.get("answer_rate"),
            "search_rate": traj.get("search_rate", metric.get("search_rate")),
            "avg_search_count": traj.get("avg_search_count", metric.get("avg_search_count")),
            "trajectory_metric_path": traj_path if os.path.exists(traj_path) else None,
            "skill0_original_metric_path": metric_path,
        }
    )

out = os.path.join(log_dir, "summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY

log "All SearchQA Qwen3.5 Hotpot8 test1000 evals finished"
