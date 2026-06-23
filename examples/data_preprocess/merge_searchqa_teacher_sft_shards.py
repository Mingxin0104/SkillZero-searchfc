import argparse
import json
import os

import pandas as pd


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    accepted_out = os.path.join(args.out_dir, "accepted.jsonl")
    all_out = os.path.join(args.out_dir, "all_candidates.jsonl")
    parquet_out = os.path.join(args.out_dir, "train.parquet")

    accepted_records = []
    all_records = []
    for shard_dir in args.shard_dirs:
        shard_dir = os.path.expanduser(shard_dir)
        accepted_path = os.path.join(shard_dir, "accepted.jsonl")
        all_path = os.path.join(shard_dir, "all_candidates.jsonl")
        if os.path.exists(accepted_path):
            accepted_records.extend(iter_jsonl(accepted_path))
        if os.path.exists(all_path):
            all_records.extend(iter_jsonl(all_path))
    accepted_records.sort(key=lambda x: (x.get("source_index", -1), x.get("candidate_id", -1)))
    all_records.sort(key=lambda x: (x.get("source_index", -1), x.get("candidate_id", -1)))

    with open(accepted_out, "w", encoding="utf-8") as accepted_f:
        for record in accepted_records:
            accepted_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(all_out, "w", encoding="utf-8") as all_f:
        for record in all_records:
            all_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    pd.DataFrame(accepted_records).to_parquet(parquet_out, index=False)
    print(f"saved_jsonl={accepted_out} rows={len(accepted_records)}")
    print(f"saved_parquet={parquet_out} rows={len(accepted_records)}")


if __name__ == "__main__":
    main()
