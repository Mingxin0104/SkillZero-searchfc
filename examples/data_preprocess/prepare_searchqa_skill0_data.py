import argparse
import os

import pandas as pd

from agent_system.environments.prompts.search import SEARCH_TEMPLATE_NO_HIS


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


def first_answer(ground_truth):
    targets = to_target_dict(ground_truth).get("target", [])
    for value in targets:
        text = str(value).strip()
        if text:
            return text
    return ""


def build_grpo_frame(df):
    fixed_rows = []
    for _, row in df.iterrows():
        reward_model = dict((row.get("reward_model") or {}))
        env_kwargs = dict((row.get("env_kwargs") or {}))
        extra_info = dict((row.get("extra_info") or {}))
        tools_kwargs = dict((extra_info.get("tools_kwargs") or {}))
        search_kwargs = dict((tools_kwargs.get("search") or {}))
        create_kwargs = dict((search_kwargs.get("create_kwargs") or {}))

        gt = reward_model.get("ground_truth", env_kwargs.get("ground_truth"))
        gt_dict = to_target_dict(gt)

        reward_model["ground_truth"] = gt_dict
        env_kwargs["ground_truth"] = gt_dict
        create_kwargs["ground_truth"] = gt_dict

        search_kwargs["create_kwargs"] = create_kwargs
        tools_kwargs["search"] = search_kwargs
        extra_info["tools_kwargs"] = tools_kwargs

        fixed_row = row.copy()
        fixed_row["reward_model"] = reward_model
        fixed_row["env_kwargs"] = env_kwargs
        fixed_row["extra_info"] = extra_info
        fixed_rows.append(fixed_row)

    return pd.DataFrame(fixed_rows)


def build_sft_frame(df):
    rows = []
    for _, row in df.iterrows():
        extra_info = row.get("extra_info", {}) or {}
        reward_model = row.get("reward_model", {}) or {}
        question = extra_info.get("question", "")
        answer = first_answer(reward_model.get("ground_truth"))
        if not question or not answer:
            continue

        prompt = SEARCH_TEMPLATE_NO_HIS.format(
            skill_context="",
            task_description=question,
        ).strip()
        rows.append(
            {
                "prompt": prompt,
                "response": f"<answer>{answer}</answer>",
                "question": question,
                "answer": answer,
                "data_source": row.get("data_source", "searchqa"),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="~/data/searchR1_processed_direct")
    parser.add_argument("--grpo_output_dir", default="~/data/searchR1_processed_direct_skill0fmt")
    parser.add_argument("--sft_output_dir", default="~/data/searchR1_processed_direct/searchqa_skill0_sft_full")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    grpo_output_dir = os.path.expanduser(args.grpo_output_dir)
    sft_output_dir = os.path.expanduser(args.sft_output_dir)
    os.makedirs(grpo_output_dir, exist_ok=True)
    os.makedirs(sft_output_dir, exist_ok=True)

    for name in ["train.parquet", "val_1000.parquet", "test.parquet"]:
        src_path = os.path.join(input_dir, name)
        df = pd.read_parquet(src_path)

        grpo_df = build_grpo_frame(df)
        grpo_path = os.path.join(grpo_output_dir, name)
        grpo_df.to_parquet(grpo_path, index=False)
        print(f"saved grpo={grpo_path} rows={len(grpo_df)}")

        sft_df = build_sft_frame(grpo_df)
        sft_path = os.path.join(sft_output_dir, name)
        sft_df.to_parquet(sft_path, index=False)
        print(f"saved sft={sft_path} rows={len(sft_df)}")


if __name__ == "__main__":
    main()
