import argparse
import json
import os

import pandas as pd


NO_SKILL_SYSTEM_PROMPT = (
    "You are an expert search agent. For each question, reason step by step, call search when evidence is "
    "needed using <search>...</search>, read the returned <information>...</information>, and finish with "
    "only the answer inside <answer>...</answer> when the evidence supports it."
)


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def rewrite_messages(record):
    messages = [
        {"role": "system", "content": NO_SKILL_SYSTEM_PROMPT},
        {"role": "user", "content": record["question"]},
    ]
    for step in record.get("search_process", []):
        assistant = step.get("assistant")
        observation = step.get("observation")
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        if observation:
            messages.append({"role": "user", "content": observation})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_size", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    for record in iter_jsonl(os.path.expanduser(args.input_jsonl)):
        record = dict(record)
        record.pop("skill", None)
        record["messages"] = rewrite_messages(record)
        records.append(record)

    accepted_path = os.path.join(args.out_dir, "accepted.jsonl")
    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "val_1000.parquet")

    with open(accepted_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    df = pd.DataFrame(records)
    df.to_parquet(train_path, index=False)
    val_df = df.head(min(args.val_size, len(df)))
    val_df.to_parquet(val_path, index=False)

    print(f"saved_jsonl={accepted_path} rows={len(records)}")
    print(f"saved_parquet={train_path} rows={len(records)}")
    print(f"saved_val={val_path} rows={len(val_df)}")


if __name__ == "__main__":
    main()
