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
    skill_type = str(row.get("skill_type", "direct_retrieval")).strip() or "direct_retrieval"
    trace_id = str(row.get("trace_id", "")).strip()
    source_index = row.get("source_index")
    candidate_id = row.get("candidate_id")

    return {
        "data_source": str(row.get("data_source", "searchqa")).strip() or "searchqa",
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
                        "data_source": "searchqa",
                    }
                }
            },
        },
        "metadata": {
            "source_dataset": "searchqa_qwen3_4b_teacher_sft_noanswer",
            "trace_id": trace_id,
            "source_index": source_index,
            "candidate_id": candidate_id,
        },
        "env_kwargs": {
            "ground_truth": ground_truth,
            "question": question,
            "data_source": "searchqa",
            "skill_type": skill_type,
        },
        "skill_type": skill_type,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="/workspace/limingxin/data/searchqa_qwen3_4b_teacher_sft_noanswer/train.parquet",
    )
    parser.add_argument(
        "--output_dir",
        default="/workspace/limingxin/data/searchqa_qwen3_4b_teacher_grpo_51599_skill0fmt",
    )
    parser.add_argument(
        "--copy_eval_from",
        default="/workspace/limingxin/data/searchR1_processed_direct_skill0fmt",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    src_df = pd.read_parquet(args.input_path)
    rows = []
    for _, row in src_df.iterrows():
        built = build_row(row)
        if built is not None:
            rows.append(built)

    out_df = pd.DataFrame(rows)
    train_path = os.path.join(args.output_dir, "train.parquet")
    out_df.to_parquet(train_path, index=False)
    print(f"saved {train_path} rows={len(out_df)}")

    for name in ["val_1000.parquet", "test.parquet", "test_1000.parquet", "test_5000.parquet"]:
        src = os.path.join(args.copy_eval_from, name)
        if os.path.exists(src):
            dst = os.path.join(args.output_dir, name)
            pd.read_parquet(src).to_parquet(dst, index=False)
            print(f"copied {src} -> {dst}")


if __name__ == "__main__":
    main()
