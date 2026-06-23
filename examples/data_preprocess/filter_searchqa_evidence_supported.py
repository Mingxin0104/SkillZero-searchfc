import argparse
import json
import os
import re

import pandas as pd


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def target_list(ground_truth):
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", [])
    if ground_truth is None:
        return []
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    return [x for x in ground_truth if str(x).strip()]


def answer_supported_by_observation(record):
    observations = []
    for step in record.get("search_process", []):
        observation = step.get("observation")
        if observation:
            observations.append(observation)
    normalized_observation = normalize_text("\n".join(observations))
    if not normalized_observation:
        return False
    return any(
        normalize_text(target) in normalized_observation
        for target in target_list(record.get("ground_truth"))
        if normalize_text(target)
    )


def is_valid(record):
    if not record.get("teacher_success"):
        return False
    steps = record.get("search_process", [])
    if not any(step.get("action_type") == "search" for step in steps):
        return False
    if any(step.get("action_type") == "invalid" for step in steps):
        return False
    if not record.get("answer"):
        return False
    return answer_supported_by_observation(record)


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_size", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = [record for record in iter_jsonl(os.path.expanduser(args.input_jsonl)) if is_valid(record)]

    accepted_path = os.path.join(args.out_dir, "accepted.jsonl")
    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "val_1000.parquet")

    with open(accepted_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    df = pd.DataFrame(records)
    df.to_parquet(train_path, index=False)
    df.head(min(args.val_size, len(df))).to_parquet(val_path, index=False)
    print(f"saved_jsonl={accepted_path} rows={len(records)}")
    print(f"saved_parquet={train_path} rows={len(records)}")
    print(f"saved_val={val_path} rows={min(args.val_size, len(df))}")


if __name__ == "__main__":
    main()
