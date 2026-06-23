import argparse
import json
import os
import re
import string

import pandas as pd

from examples.data_preprocess.build_searchqa_trace_sft import build_messages


SYSTEM_PROMPT = (
    "You are an expert agent tasked with answering the given question step-by-step. "
    "You may search when needed using <search>...</search> and only provide the final answer "
    "when confident using <answer>...</answer>."
)


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(text):
    matches = list(re.finditer(r"<answer>(.*?)</answer>", str(text), re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def strict_em(prediction, targets):
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(target) for target in targets)


def load_jsonl(path):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_strict_usable(record):
    steps = record.get("steps", [])
    if not any(step.get("action_type") == "search" for step in steps):
        return False, None
    if any(step.get("action_type") == "invalid" for step in steps):
        return False, None

    answer_steps = [step for step in steps if step.get("action_type") == "answer"]
    if not answer_steps:
        return False, None

    answer = extract_answer(answer_steps[-1].get("assistant", ""))
    if not answer:
        return False, None

    targets = record.get("ground_truth", [])
    if isinstance(targets, dict):
        targets = targets.get("target", [])
    if isinstance(targets, str):
        targets = [targets]

    if not strict_em(answer, targets):
        return False, None
    return True, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--success_output_path", required=True)
    parser.add_argument("--sft_output_path", required=True)
    args = parser.parse_args()

    records = load_jsonl(args.input_path)
    best_by_source = {}
    for record in records:
        usable, answer = is_strict_usable(record)
        if not usable:
            continue
        source_index = record.get("source_index")
        if source_index in best_by_source:
            continue
        record = dict(record)
        record["answer"] = answer
        record["success"] = True
        record["final_reward"] = 1.0
        record["final_answer"] = f"<answer>{answer}</answer>"
        best_by_source[source_index] = record

    success_records = list(best_by_source.values())
    os.makedirs(os.path.dirname(os.path.expanduser(args.success_output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.expanduser(args.sft_output_path)), exist_ok=True)

    with open(os.path.expanduser(args.success_output_path), "w", encoding="utf-8") as f:
        for record in success_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    rows = []
    for record in success_records:
        rows.append(
            {
                "messages": build_messages(record, SYSTEM_PROMPT),
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "data_source": record.get("data_source", "searchqa"),
                "skill_type": record.get("skill_type"),
                "teacher_success": True,
                "final_reward": 1.0,
                "source_index": record.get("source_index"),
                "candidate_id": record.get("candidate_id"),
                "trace_id": record.get("trace_id"),
            }
        )
    pd.DataFrame(rows).to_parquet(os.path.expanduser(args.sft_output_path), index=False)
    print(f"strict_success={args.success_output_path} rows={len(success_records)}")
    print(f"strict_sft={args.sft_output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
