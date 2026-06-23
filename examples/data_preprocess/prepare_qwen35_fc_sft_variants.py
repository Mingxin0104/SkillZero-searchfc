import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert search agent. Use the provided search function when evidence is needed. "
    "After enough evidence is collected, answer with only the shortest final answer and no explanation."
)

ANSWER_RULE = (
    "Important output rule: when you answer directly, return only the final entity, title, number, date, or short phrase. "
    "Do not include explanation, punctuation, or extra words."
)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_no_skill_user_prompt(question):
    return f"Question: {question}\n\n{ANSWER_RULE}"


def to_python(obj):
    if isinstance(obj, np.ndarray):
        return [to_python(x) for x in obj.tolist()]
    if isinstance(obj, list):
        return [to_python(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if pd.isna(obj) and not isinstance(obj, (str, bytes)):
        return None
    return obj


def normalize_message(msg):
    msg = to_python(msg)
    if not isinstance(msg, dict):
        return msg
    cleaned = {}
    for key in ["role", "content", "tool_calls", "tool_call_id", "name", "reasoning"]:
        if key in msg and msg[key] is not None:
            cleaned[key] = to_python(msg[key])
    return cleaned


def sanitize_record(record):
    record = to_python(dict(record))
    record["messages"] = [normalize_message(m) for m in record.get("messages", [])]
    return record


def rewrite_no_skill_record(record):
    record = sanitize_record(record)
    question = record["question"]
    raw_messages = record.get("messages", [])
    if len(raw_messages) < 2:
        raise ValueError("Expected at least system+user messages.")

    messages = [normalize_message(m) for m in raw_messages]
    messages[0] = {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    messages[1] = {"role": "user", "content": build_no_skill_user_prompt(question)}
    record["messages"] = messages
    record["skill"] = ""
    record["skill_variant"] = "no_skill"
    return record


def save_variant_dir(records, out_dir, val_size):
    os.makedirs(out_dir, exist_ok=True)
    accepted_path = os.path.join(out_dir, "accepted.jsonl")
    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "val_1000.parquet")

    save_jsonl(accepted_path, records)
    df = pd.DataFrame(records)
    df.to_parquet(train_path, index=False)
    df.head(min(val_size, len(df))).to_parquet(val_path, index=False)

    print(f"saved_jsonl={accepted_path} rows={len(records)}")
    print(f"saved_parquet={train_path} rows={len(records)}")
    print(f"saved_val={val_path} rows={min(val_size, len(records))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--with_skill_out_dir", required=True)
    parser.add_argument("--no_skill_out_dir", required=True)
    parser.add_argument("--val_size", type=int, default=1000)
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    accepted_path = os.path.join(input_dir, "accepted.jsonl")
    train_path = os.path.join(input_dir, "train.parquet")
    if not os.path.exists(accepted_path):
        raise FileNotFoundError(accepted_path)
    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)

    records = [sanitize_record(record) for record in load_jsonl(accepted_path)]

    with_skill_out_dir = os.path.expanduser(args.with_skill_out_dir)
    no_skill_out_dir = os.path.expanduser(args.no_skill_out_dir)
    os.makedirs(with_skill_out_dir, exist_ok=True)
    with_skill_records = records
    save_jsonl(os.path.join(with_skill_out_dir, "accepted.jsonl"), with_skill_records)
    df = pd.DataFrame(with_skill_records)
    df.to_parquet(os.path.join(with_skill_out_dir, "train.parquet"), index=False)
    df.head(min(args.val_size, len(df))).to_parquet(os.path.join(with_skill_out_dir, "val_1000.parquet"), index=False)
    print(f"saved_with_skill={with_skill_out_dir} rows={len(df)}")

    no_skill_records = [rewrite_no_skill_record(record) for record in records]
    save_variant_dir(no_skill_records, no_skill_out_dir, args.val_size)


if __name__ == "__main__":
    main()
