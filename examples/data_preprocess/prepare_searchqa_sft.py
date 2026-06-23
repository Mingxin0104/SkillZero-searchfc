import argparse
import os

import pandas as pd


def _normalize_answer(ground_truth):
    if ground_truth is None:
        return ""
    if isinstance(ground_truth, (list, tuple)):
        values = list(ground_truth)
    else:
        try:
            values = list(ground_truth)
        except TypeError:
            values = [ground_truth]
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


def _build_sft_frame(df):
    records = []
    for _, row in df.iterrows():
        extra_info = row.get("extra_info", {}) or {}
        reward_model = row.get("reward_model", {}) or {}
        question = extra_info.get("question", "")
        answer = _normalize_answer(reward_model.get("ground_truth"))
        if not question or not answer:
            continue
        records.append(
            {
                "question": question,
                "answer": answer,
                "data_source": row.get("data_source", "searchqa"),
                "skill_type": row.get("skill_type", ""),
                "source_dataset": (row.get("metadata", {}) or {}).get("source_dataset", ""),
            }
        )
    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(description="Prepare SearchQA SFT subsets from SkillZero search data.")
    parser.add_argument("--input_dir", type=str, default="~/data/searchR1_processed_direct")
    parser.add_argument("--output_dir", type=str, default="~/data/searchR1_processed_direct/searchqa_sft_10k_1k")
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    train_sft = _build_sft_frame(train_df)
    val_sft = _build_sft_frame(test_df)

    if len(train_sft) < args.train_size:
        raise ValueError(f"Not enough training rows: {len(train_sft)} < {args.train_size}")
    if len(val_sft) < args.val_size:
        raise ValueError(f"Not enough validation rows: {len(val_sft)} < {args.val_size}")

    train_subset = train_sft.sample(n=args.train_size, random_state=args.seed).reset_index(drop=True)
    val_subset = val_sft.sample(n=args.val_size, random_state=args.seed).reset_index(drop=True)

    train_path = os.path.join(output_dir, "train_10k.parquet")
    val_path = os.path.join(output_dir, "val_1k.parquet")
    train_subset.to_parquet(train_path, index=False)
    val_subset.to_parquet(val_path, index=False)

    print(f"train_rows={len(train_subset)} saved={train_path}")
    print(f"val_rows={len(val_subset)} saved={val_path}")
    print(train_subset.head(2).to_dict(orient="records"))


if __name__ == "__main__":
    main()
