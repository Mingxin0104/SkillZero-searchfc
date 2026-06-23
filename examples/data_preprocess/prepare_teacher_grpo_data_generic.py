import argparse
import os

import pandas as pd


def to_target_dict(value):
    if isinstance(value, dict) and "target" in value:
        return value
    if value is None:
        values = []
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]

    normalized = []
    for item in values:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return {"target": normalized}


def build_prompt(question):
    return [
        {"role": "system", "content": "You are a helpful and harmless assistant."},
        {"role": "user", "content": str(question).strip()},
    ]


def build_row(row):
    question = str(row.get("question", "")).strip()
    if not question:
        return None

    ground_truth = to_target_dict(row.get("ground_truth"))
    data_source = str(row.get("data_source", "searchqa")).strip() or "searchqa"
    skill_type = str(row.get("skill_type", "")).strip() or None
    trace_id = str(row.get("trace_id", "")).strip()
    source_index = row.get("source_index")
    candidate_id = row.get("candidate_id")

    env_kwargs = {
        "ground_truth": ground_truth,
        "question": question,
        "data_source": data_source,
    }
    if skill_type:
        env_kwargs["skill_type"] = skill_type

    return {
        "data_source": data_source,
        "prompt": build_prompt(question),
        "ability": "search",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "index": source_index,
            "need_tools_kwargs": True,
            "question": question,
            "split": "train",
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "ground_truth": ground_truth,
                        "question": question,
                        "data_source": data_source,
                    }
                }
            },
        },
        "metadata": {
            "trace_id": trace_id,
            "source_index": source_index,
            "candidate_id": candidate_id,
            "source_dataset": row.get("source_dataset", "teacher_sft"),
        },
        "env_kwargs": env_kwargs,
        "skill_type": skill_type,
    }


def copy_if_exists(src_dir, dst_dir, filename):
    src = os.path.join(src_dir, filename)
    if not os.path.exists(src):
        return
    pd.read_parquet(src).to_parquet(os.path.join(dst_dir, filename), index=False)
    print(f"copied {src} -> {os.path.join(dst_dir, filename)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--copy_eval_from", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    src_df = pd.read_parquet(os.path.expanduser(args.input_path))
    rows = []
    for _, row in src_df.iterrows():
        built = build_row(row)
        if built is not None:
            rows.append(built)

    out_df = pd.DataFrame(rows)
    train_path = os.path.join(args.output_dir, "train.parquet")
    out_df.to_parquet(train_path, index=False)
    print(f"saved {train_path} rows={len(out_df)}")

    for filename in ["val_1000.parquet", "test.parquet"]:
        copy_if_exists(os.path.expanduser(args.copy_eval_from), args.output_dir, filename)


if __name__ == "__main__":
    main()
