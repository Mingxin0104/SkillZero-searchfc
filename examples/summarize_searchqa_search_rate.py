import argparse
import json
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--use_trajectory_metrics",
        action="store_true",
        help="Use *.trajectory_metrics.jsonl instead of parsing generated text.",
    )
    args = parser.parse_args()

    trajectory_files = []
    if os.path.isdir(args.generation_dir):
        trajectory_files = [
            os.path.join(args.generation_dir, name)
            for name in os.listdir(args.generation_dir)
            if name.endswith(".trajectory_metrics.jsonl")
        ]
    trajectory_files.sort()

    if args.use_trajectory_metrics and trajectory_files:
        total = 0
        searched = 0
        success = 0
        score_sum = 0.0
        search_counts = []
        for path in trajectory_files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    count = float(row.get("tool_callings", 0.0) or 0.0)
                    score = float(row.get("score", 0.0) or 0.0)
                    total += 1
                    searched += int(count > 0)
                    # Keep Skill0 metric definition aligned with paper scripts:
                    # success iff trajectory score >= 1.0.
                    success += int(score >= 1.0)
                    score_sum += score
                    search_counts.append(count)

        summary = {
            "generation_dir": args.generation_dir,
            "files": trajectory_files,
            "success_definition": "score >= 1.0 (Skill0/SearchQA paper metric)",
            "total": total,
            "searched": searched,
            "search_rate": searched / total if total else 0.0,
            "success": success,
            "success_rate": success / total if total else 0.0,
            "avg_search_count": sum(search_counts) / total if total else 0.0,
            "avg_score": score_sum / total if total else 0.0,
            "acc": score_sum / total if total else 0.0,
        }

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    files = []
    if os.path.isdir(args.generation_dir):
        files = [
            os.path.join(args.generation_dir, name)
            for name in os.listdir(args.generation_dir)
            if name.endswith(".jsonl") and not name.endswith(".trajectory_metrics.jsonl")
        ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No jsonl generation files found under {args.generation_dir}")

    total = 0
    searched = 0
    answered = 0
    score_sum = 0.0
    search_counts = []
    search_re = re.compile(r"<search>(.*?)</search>", flags=re.IGNORECASE | re.DOTALL)
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                output = row.get("output", "") or ""
                score = float(row.get("score", 0.0) or 0.0)
                # Match the search environment projection: a usable search action is
                # a complete <search>...</search> block, parsed case-insensitively.
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
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
