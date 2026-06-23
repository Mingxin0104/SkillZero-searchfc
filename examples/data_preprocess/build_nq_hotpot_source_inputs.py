import argparse
import os

import pandas as pd

from prepare_nq_hotpot_teacher_inputs import build_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_train", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sources", nargs="+", default=["nq", "hotpotqa"])
    args = parser.parse_args()

    train_raw = pd.read_parquet(os.path.expanduser(args.raw_train))
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for source_name in args.sources:
        subset = train_raw[train_raw["data_source"] == source_name].reset_index(drop=True)
        records = [build_record(row, source_name, "train", idx) for idx, row in subset.iterrows()]
        out_path = os.path.join(out_dir, f"{source_name}_train_full.parquet")
        pd.DataFrame(records).to_parquet(out_path, index=False)
        print(f"saved {out_path} rows={len(records)}")


if __name__ == "__main__":
    main()
