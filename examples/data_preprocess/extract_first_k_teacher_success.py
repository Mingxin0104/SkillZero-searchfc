import argparse
import json
import os

import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_rows", type=int, default=10000)
    parser.add_argument("--source_name", default=None)
    args = parser.parse_args()

    rows = load_jsonl(os.path.expanduser(args.input_jsonl))
    if args.source_name:
        rows = [row for row in rows if row.get("data_source") == args.source_name]
    rows = rows[: args.target_rows]

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "accepted_topk.jsonl")
    out_parquet = os.path.join(out_dir, "train.parquet")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    print(f"saved {out_jsonl} rows={len(rows)}")
    print(f"saved {out_parquet} rows={len(rows)}")


if __name__ == "__main__":
    main()
