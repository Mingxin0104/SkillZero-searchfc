import argparse
import json
import os

import pandas as pd


def load_trace_records(path):
    path = os.path.expanduser(path)
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported trace file: {path}")


def build_messages(record, no_skill_system_prompt):
    question = record["question"]
    steps = record.get("steps", [])

    messages = [
        {"role": "system", "content": no_skill_system_prompt},
        {"role": "user", "content": question},
    ]

    for step in steps:
        assistant = step.get("assistant")
        observation = step.get("observation")

        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        if observation:
            messages.append({"role": "user", "content": observation})

    has_answer_step = any(step.get("action_type") == "answer" for step in steps)
    final_answer = record.get("final_answer")
    if final_answer and not has_answer_step:
        if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("content") != final_answer:
            messages.append({"role": "assistant", "content": final_answer})

    return messages


def is_usable_trace(record, require_search=True, require_no_invalid=True):
    if not record.get("success"):
        return False
    if not record.get("final_answer") and not record.get("answer"):
        return False

    steps = record.get("steps", [])
    if require_search and not any(step.get("action_type") == "search" for step in steps):
        return False
    if require_no_invalid and any(step.get("action_type") == "invalid" for step in steps):
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_file", required=True, help="Teacher rollout traces in json/jsonl format.")
    parser.add_argument("--output_path", required=True, help="Output parquet path for multiturn SFT.")
    parser.add_argument("--include_failed", action="store_true", help="Include failed traces for debugging.")
    parser.add_argument("--allow_no_search", action="store_true", help="Keep successful direct-answer traces.")
    parser.add_argument("--allow_invalid", action="store_true", help="Keep traces that contain invalid actions.")
    parser.add_argument(
        "--system_prompt",
        default=(
            "You are an expert agent tasked with answering the given question step-by-step. "
            "You may search when needed using <search>...</search> and only provide the final answer "
            "when confident using <answer>...</answer>."
        ),
    )
    args = parser.parse_args()

    records = load_trace_records(args.trace_file)
    rows = []
    for record in records:
        if not args.include_failed and not is_usable_trace(
            record,
            require_search=not args.allow_no_search,
            require_no_invalid=not args.allow_invalid,
        ):
            continue
        messages = build_messages(record, args.system_prompt)
        rows.append(
            {
                "messages": messages,
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "data_source": record.get("data_source", "searchqa"),
                "skill_type": record.get("skill_type"),
                "teacher_success": bool(record.get("success")),
                "final_reward": record.get("final_reward"),
                "source_index": record.get("source_index"),
                "candidate_id": record.get("candidate_id"),
                "trace_id": record.get("trace_id"),
            }
        )

    out_df = pd.DataFrame(rows)
    output_path = os.path.expanduser(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"saved={output_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
