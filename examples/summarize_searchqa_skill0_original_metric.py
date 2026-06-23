import argparse
import json
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    files = []
    if os.path.isdir(args.generation_dir):
        files = [
            os.path.join(args.generation_dir, name)
            for name in os.listdir(args.generation_dir)
            if name.endswith(".jsonl") and not name.endswith(".trajectory_metrics.jsonl")
        ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No jsonl files found under {args.generation_dir}")

    search_re = re.compile(r"<search>(.*?)</search>", flags=re.IGNORECASE | re.DOTALL)
    total = 0
    searched = 0
    answered = 0
    score_sum = 0.0
    search_counts = []

    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                output = row.get("output", "") or ""
                score = float(row.get("score", 0.0) or 0.0)
                count = len([m for m in search_re.finditer(output) if m.group(1).strip()])

                total += 1
                searched += int(count > 0)
                answered += int("<answer>" in output and "</answer>" in output)
                score_sum += score
                search_counts.append(count)

    summary = {
        "generation_dir": args.generation_dir,
        "files": files,
        "total": total,
        "searched": searched,
        "search_rate": searched / total if total else 0.0,
        "answered": answered,
        "answer_rate": answered / total if total else 0.0,
        "avg_search_count": sum(search_counts) / total if total else 0.0,
        "avg_score": score_sum / total if total else 0.0,
        "acc": score_sum / total if total else 0.0,
        "metric": "skill0_original",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
