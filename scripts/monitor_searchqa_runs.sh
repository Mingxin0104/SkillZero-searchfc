#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -euo pipefail

OUT_DIR="${1:-/public/limingxin/SkillZero/monitoring}"
RL_LOG="${2:-/home/limingxin/SkillZero/log/skillzero_search_vl_3b_searchqa_2gpu_modelscope.log}"
SFT_LOG="${3:-/home/limingxin/SkillZero/log/searchqa-lora-sft-r32-full.log}"
RL_CKPT_DIR="${4:-/home/limingxin/SkillZero/checkpoints/skillzero_search_vl_3b_searchqa_2gpu_modelscope}"
SFT_CKPT_DIR="${5:-/public/limingxin/SkillZero/checkpoints/searchqa_lora_sft_r32_full}"

mkdir -p "$OUT_DIR"

snapshot() {
    local ts="$1"
    local out="$OUT_DIR/${ts}.log"
    {
        echo "timestamp=$ts"
        date -u '+utc_now=%Y-%m-%dT%H:%M:%SZ'
        nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
        echo
        echo "[rl_tail]"
        tail -n 80 "$RL_LOG" 2>/dev/null || true
        echo
        echo "[sft_tail]"
        tail -n 80 "$SFT_LOG" 2>/dev/null || true
        echo
        echo "[rl_ckpt]"
        find "$RL_CKPT_DIR" -maxdepth 2 -type f 2>/dev/null | sort | tail -n 40 || true
        echo
        echo "[sft_ckpt]"
        find "$SFT_CKPT_DIR" -maxdepth 2 -type f 2>/dev/null | sort | tail -n 40 || true
    } > "$out"
}

while true; do
    now_epoch="$(date -u +%s)"
    now_tag="$(date -u +%Y%m%dT%H%M%SZ)"
    snapshot "$now_tag"

    target_epoch="$(date -u -d 'next hour' +%s)"
    special_epoch="$(date -u -d '2026-06-04 09:30:00' +%s)"

    if [ "$special_epoch" -gt "$now_epoch" ] && [ "$special_epoch" -lt "$target_epoch" ]; then
        sleep "$((special_epoch - now_epoch))"
        special_tag="$(date -u +%Y%m%dT%H%M%SZ)"
        snapshot "$special_tag"
        now_epoch="$(date -u +%s)"
        target_epoch="$(date -u -d 'next hour' +%s)"
    fi

    sleep "$((target_epoch - now_epoch))"
done
