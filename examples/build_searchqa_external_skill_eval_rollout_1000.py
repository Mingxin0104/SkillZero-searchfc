import argparse
from pathlib import Path

import pandas as pd


def to_target_list(value):
    if isinstance(value, dict) and "target" in value:
        value = value["target"]
    if value is None:
        items = []
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]

    out = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_path)
    rows = []
    for _, row in df.iterrows():
        question = str(row.get("question") or "").strip()
        reward_model = dict((row.get("reward_model") or {}))
        skill_type = str(row.get("skill_type") or "direct_retrieval")
        data_source = str(row.get("data_source") or "searchqa")
        source_index = row.get("source_index")
        targets = to_target_list((reward_model.get("ground_truth") or {}).get("target"))
        if not question or not targets:
            continue

        gt = {"target": targets}
        prompt = [
            {"role": "system", "content": "You are a helpful and harmless assistant."},
            {"role": "user", "content": question},
        ]
        extra_info = {
            "index": source_index,
            "need_tools_kwargs": True,
            "question": question,
            "split": "searchqa_train_no_overlap_eval",
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "ground_truth": gt,
                        "question": question,
                        "data_source": data_source,
                    }
                }
            },
        }
        env_kwargs = {
            "ground_truth": gt,
            "question": question,
            "data_source": data_source,
            "skill_type": skill_type,
        }
        rows.append(
            {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "search",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": extra_info,
                "metadata": {
                    "source_dataset": "searchqa_train_no_overlap_eval",
                    "source_path": row.get("source_path"),
                    "source_index": source_index,
                },
                "env_kwargs": env_kwargs,
                "skill_type": skill_type,
            }
        )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"saved={out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
