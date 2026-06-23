import argparse
import json
from pathlib import Path

import pandas as pd


def load_skill_file_sections(filepath: str):
    text = Path(filepath).read_text(encoding="utf-8")
    sections = {}
    current = None
    buffer = []
    for line in text.splitlines():
        if line.startswith("### "):
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current = line[4:].strip()
            buffer = [line]
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    return sections


def build_skill_bank(skill_mapping_path: str):
    mapping = json.loads(Path(skill_mapping_path).read_text(encoding="utf-8"))
    base = Path(skill_mapping_path).parent
    skill_files = mapping["skill_files"]
    task_to_skill = mapping["task_to_skill"]

    general = load_skill_file_sections(str(base / skill_files["general_skills"]))
    general_text = next(iter(general.values())).strip()
    bank = {}
    for task, skill_name in task_to_skill.items():
        task_sections = load_skill_file_sections(str(base / skill_files[skill_name]))
        task_text = next(iter(task_sections.values())).strip()
        bank[task] = f"{general_text}\n\n{task_text}".strip()
    return bank


def extract_target(reward_model):
    reward_model = reward_model or {}
    target = (reward_model.get("ground_truth") or {}).get("target")
    if target is None:
        return ""
    if isinstance(target, str):
        return target.strip()
    try:
        for value in target:
            text = str(value).strip()
            if text:
                return text
    except TypeError:
        return str(target).strip()
    return ""


def add_excludes_from_parquet(path: str, exclude_source: set, exclude_question: set):
    df = pd.read_parquet(path)
    if "source_index" in df.columns:
        exclude_source.update(int(x) for x in df["source_index"].dropna().tolist())
    if "question" in df.columns:
        exclude_question.update(str(x).strip() for x in df["question"].dropna().tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--skill_mapping", required=True)
    parser.add_argument("--exclude_parquet", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    skill_bank = build_skill_bank(args.skill_mapping)
    exclude_source = set()
    exclude_question = set()
    for path in args.exclude_parquet:
        add_excludes_from_parquet(path, exclude_source, exclude_question)

    df = pd.read_parquet(args.input_parquet)
    rows = []
    seen_source = set()
    seen_question = set()

    for _, row in df.iterrows():
        extra_info = row.get("extra_info") or {}
        question = str(extra_info.get("question") or "").strip()
        source_index = extra_info.get("index")
        skill_type = str(row.get("skill_type") or row.get("env_kwargs", {}).get("skill_type") or "direct_retrieval")
        answer = extract_target(row.get("reward_model"))
        if source_index is None or not question or not answer:
            continue
        if source_index in seen_source or question in seen_question:
            continue
        if source_index in exclude_source or question in exclude_question:
            continue
        skill = skill_bank.get(skill_type)
        if not skill:
            continue

        seen_source.add(source_index)
        seen_question.add(question)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "reward_model": row.get("reward_model"),
                "skill": skill,
                "skill_type": skill_type,
                "data_source": row.get("data_source", "searchqa"),
                "source_index": source_index,
                "source_path": args.input_parquet,
            }
        )
        if len(rows) >= args.limit:
            break

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(
        f"saved={output_path} rows={len(rows)} excluded_source={len(exclude_source)} excluded_question={len(exclude_question)}"
    )


if __name__ == "__main__":
    main()
