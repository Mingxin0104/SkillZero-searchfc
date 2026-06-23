import argparse
import os

import pandas as pd


SYSTEM_PROMPT = "You are a helpful and harmless assistant."
SOURCE_SKILL_MAP = {
    "nq": "direct_retrieval",
    "hotpotqa": "multi_hop_reasoning",
}


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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(question).strip()},
    ]


def build_record(row, source_name, split_name, local_index):
    question = str(row.get("question", "")).strip()
    ground_truth = to_target_dict(row.get("golden_answers"))
    skill_type = SOURCE_SKILL_MAP[source_name]

    return {
        "id": row.get("id"),
        "question": question,
        "ground_truth": ground_truth,
        "golden_answers": list(ground_truth["target"]),
        "data_source": source_name,
        "prompt": build_prompt(question),
        "ability": "search",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "index": int(local_index),
            "need_tools_kwargs": True,
            "question": question,
            "split": split_name,
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "ground_truth": ground_truth,
                        "question": question,
                        "data_source": source_name,
                    }
                }
            },
        },
        "metadata": {
            "source_dataset": "PeterJinGo/nq_hotpotqa_train",
            "source_id": row.get("id"),
            "original_data_source": row.get("data_source"),
            "original_split": split_name,
        },
        "env_kwargs": {
            "ground_truth": ground_truth,
            "question": question,
            "data_source": source_name,
            "skill_type": skill_type,
        },
        "skill_type": skill_type,
    }


def sample_split(df, source_name, split_name, count, seed):
    subset = df[df["data_source"] == source_name].reset_index(drop=True)
    if len(subset) < count:
        raise ValueError(f"{split_name}:{source_name} only has {len(subset)} rows, need {count}")
    sampled = subset.sample(n=count, random_state=seed).reset_index(drop=True)
    records = [build_record(row, source_name, split_name, idx) for idx, row in sampled.iterrows()]
    return pd.DataFrame(records)


def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"saved {path} rows={len(df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_train", required=True)
    parser.add_argument("--raw_test", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_size_per_source", type=int, default=10000)
    parser.add_argument("--test_size_per_source", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_raw = pd.read_parquet(os.path.expanduser(args.raw_train))
    test_raw = pd.read_parquet(os.path.expanduser(args.raw_test))
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    merged_train = []
    merged_test = []
    for offset, source_name in enumerate(["nq", "hotpotqa"]):
        train_df = sample_split(
            train_raw,
            source_name=source_name,
            split_name="train",
            count=args.train_size_per_source,
            seed=args.seed + offset,
        )
        test_df = sample_split(
            test_raw,
            source_name=source_name,
            split_name="test",
            count=args.test_size_per_source,
            seed=args.seed + 100 + offset,
        )
        save_df(train_df, os.path.join(out_dir, f"{source_name}_train_{len(train_df)}.parquet"))
        save_df(test_df, os.path.join(out_dir, f"{source_name}_test_{len(test_df)}.parquet"))
        merged_train.append(train_df)
        merged_test.append(test_df)

    merged_train_df = pd.concat(merged_train, ignore_index=True)
    merged_test_df = pd.concat(merged_test, ignore_index=True)
    save_df(merged_train_df, os.path.join(out_dir, "train.parquet"))
    save_df(merged_test_df, os.path.join(out_dir, "test.parquet"))
    save_df(merged_test_df.head(min(1000, len(merged_test_df))), os.path.join(out_dir, "val_1000.parquet"))


if __name__ == "__main__":
    main()
