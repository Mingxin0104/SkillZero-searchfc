#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set -euo pipefail

PYTHON="${PYTHON:-/workspace/limingxin/miniconda3/envs/skillzero/bin/python}"
OUT_DIR="${OUT_DIR:-/public/limingxin/SkillZero/data/searchqa_trace_sft_train}"

PYTHONPATH="$REPO_ROOT" "$PYTHON" - <<'PY'
import os
import pandas as pd

out_dir = os.path.expanduser(os.environ.get("OUT_DIR", "/public/limingxin/SkillZero/data/searchqa_trace_sft_train"))
sft_dir = os.path.join(out_dir, "sft")
paths = [os.path.join(sft_dir, name) for name in sorted(os.listdir(sft_dir)) if name.endswith(".parquet")]
if not paths:
    raise SystemExit(f"no shard parquet files found in {sft_dir}")

frames = []
for path in paths:
    df = pd.read_parquet(path)
    if len(df):
        frames.append(df)

if not frames:
    raise SystemExit("all shard parquet files are empty")

df = pd.concat(frames, ignore_index=True)
final_path = os.path.join(out_dir, "train.parquet")
df.to_parquet(final_path, index=False)
print(f"saved={final_path} rows={len(df)} shards={len(paths)} nonempty={len(frames)}")
PY
