import argparse
import os

import pandas as pd

from agent_system.environments.prompts.search import SEARCH_TEMPLATE_NO_HIS


def first_answer(ground_truth):
    if ground_truth is None:
        return ""
    if isinstance(ground_truth, (list, tuple)):
        values = ground_truth
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


def build_frame(df):
    rows = []
    for _, row in df.iterrows():
        extra_info = row.get("extra_info", {}) or {}
        reward_model = row.get("reward_model", {}) or {}
        question = extra_info.get("question", "")
        answer = first_answer(reward_model.get("ground_truth"))
        if not question or not answer:
            continue
        prompt = SEARCH_TEMPLATE_NO_HIS.format(skill_context="", task_description=question).strip()
        response = f"<answer>{answer}</answer>"
        rows.append(
            {
                "prompt": prompt,
                "response": response,
                "question": question,
                "answer": answer,
                "data_source": row.get("data_source", "searchqa"),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="~/data/searchR1_processed_direct")
    parser.add_argument("--output_dir", default="~/data/searchR1_processed_direct/searchqa_skill0_sft_full")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    splits = {
        "train": "train.parquet",
        "val_1000": "val_1000.parquet",
        "test": "test.parquet",
    }
    for out_name, file_name in splits.items():
        df = pd.read_parquet(os.path.join(input_dir, file_name))
        out_df = build_frame(df)
        out_path = os.path.join(output_dir, f"{out_name}.parquet")
        out_df.to_parquet(out_path, index=False)
        print(out_name, len(out_df), out_path)


if __name__ == "__main__":
    main()
