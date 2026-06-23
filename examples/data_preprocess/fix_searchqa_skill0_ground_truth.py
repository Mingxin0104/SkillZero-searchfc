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


def process_file(src_path, dst_path):
    df = pd.read_parquet(src_path)
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
        row = row.copy()
        row["reward_model"] = reward_model
        row["env_kwargs"] = env_kwargs
        row["extra_info"] = extra_info
        fixed_rows.append(row)
    out_df = pd.DataFrame(fixed_rows)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    out_df.to_parquet(dst_path, index=False)
    print(f"saved={dst_path} rows={len(out_df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="~/data/searchR1_processed_direct")
    parser.add_argument("--output_dir", default="~/data/searchR1_processed_direct_skill0fmt")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    for name in ["train.parquet", "val_1000.parquet", "test.parquet"]:
        process_file(os.path.join(input_dir, name), os.path.join(output_dir, name))


if __name__ == "__main__":
    main()
