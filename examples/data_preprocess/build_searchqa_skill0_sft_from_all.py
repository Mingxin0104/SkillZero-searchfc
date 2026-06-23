import argparse
import json
import os
import re
import string

import pandas as pd

from examples.data_preprocess.build_searchqa_trace_sft import build_messages
from agent_system.environments.prompts.search import SEARCH_TEMPLATE_NO_HIS


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


def targets_from_record(record):
    targets = record.get("ground_truth", [])
    if isinstance(targets, dict):
        targets = targets.get("target", [])
    if isinstance(targets, str):
        return [targets]
    return list(targets or [])


def strict_em(answer, targets):
    pred = normalize_answer(answer)
    return any(pred == normalize_answer(target) for target in targets)


def loose_subem(answer, targets):
    pred = normalize_answer(answer)
    return any(normalize_answer(target) in pred for target in targets)


def load_jsonl(path):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_required_trace(record):
    steps = record.get("steps", [])
    if not any(step.get("action_type") == "search" for step in steps):
        return False
    if any(step.get("action_type") == "invalid" for step in steps):
        return False
    if any(not step.get("is_action_valid", True) for step in steps):
        return False
    return True


def final_answer(record):
    answer_steps = [step for step in record.get("steps", []) if step.get("action_type") == "answer"]
    if not answer_steps:
        return None
    return extract_answer(answer_steps[-1].get("assistant", ""))


def keep_loose(record):
    if not has_required_trace(record):
        return False, None
    answer = final_answer(record)
    if not answer:
        return False, None
    return loose_subem(answer, targets_from_record(record)), answer


def keep_strict(record):
    if not has_required_trace(record):
        return False, None
    answer = final_answer(record)
    if not answer:
        return False, None
    return strict_em(answer, targets_from_record(record)), answer


def dedupe(records, predicate):
    best = {}
    for record in records:
        keep, answer = predicate(record)
        if not keep:
            continue
        source_index = record.get("source_index")
        if source_index in best:
            continue
        record = dict(record)
        record["answer"] = answer
        record["success"] = True
        record["final_reward"] = 1.0
        record["final_answer"] = f"<answer>{answer}</answer>"
        best[source_index] = record
    return list(best.values())


def build_messages_with_optional_skill(record, include_skill):
    if not include_skill:
        return build_messages(record, SYSTEM_PROMPT)

    record = dict(record)
    skill_context = record.get("skill_context", "")
    question = record.get("question", "")
    record["question"] = SEARCH_TEMPLATE_NO_HIS.format(
        skill_context=skill_context,
        task_description=question,
    ).strip()
    return build_messages(record, SYSTEM_PROMPT)


def write_dataset(records, jsonl_path, parquet_path, include_skill):
    os.makedirs(os.path.dirname(os.path.expanduser(jsonl_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.expanduser(parquet_path)), exist_ok=True)
    with open(os.path.expanduser(jsonl_path), "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    rows = []
    for record in records:
        rows.append(
            {
                "messages": build_messages_with_optional_skill(record, include_skill),
                "skill": record.get("skill_context", "") if include_skill else "",
                "has_skill": bool(include_skill),
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
    pd.DataFrame(rows).to_parquet(os.path.expanduser(parquet_path), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_paths", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    records = []
    for path in args.all_paths:
        records.extend(load_jsonl(path))
    records.sort(key=lambda x: (x.get("source_index", -1), x.get("candidate_id", -1)))

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    all_path = os.path.join(out_dir, "all.jsonl")
    with open(all_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    loose_records = dedupe(records, keep_loose)
    strict_records = dedupe(records, keep_strict)
    write_dataset(
        loose_records,
        os.path.join(out_dir, "loose_success.jsonl"),
        os.path.join(out_dir, "loose_with_skill_train.parquet"),
        include_skill=True,
    )
    write_dataset(
        loose_records,
        os.path.join(out_dir, "loose_success.jsonl"),
        os.path.join(out_dir, "loose_no_skill_train.parquet"),
        include_skill=False,
    )
    write_dataset(
        strict_records,
        os.path.join(out_dir, "strict_success.jsonl"),
        os.path.join(out_dir, "strict_with_skill_train.parquet"),
        include_skill=True,
    )
    write_dataset(
        strict_records,
        os.path.join(out_dir, "strict_success.jsonl"),
        os.path.join(out_dir, "strict_no_skill_train.parquet"),
        include_skill=False,
    )

    print(f"all={all_path} rows={len(records)}")
    print(f"loose_with_skill={os.path.join(out_dir, 'loose_with_skill_train.parquet')} rows={len(loose_records)}")
    print(f"loose_no_skill={os.path.join(out_dir, 'loose_no_skill_train.parquet')} rows={len(loose_records)}")
    print(f"strict_with_skill={os.path.join(out_dir, 'strict_with_skill_train.parquet')} rows={len(strict_records)}")
    print(f"strict_no_skill={os.path.join(out_dir, 'strict_no_skill_train.parquet')} rows={len(strict_records)}")


if __name__ == "__main__":
    main()
