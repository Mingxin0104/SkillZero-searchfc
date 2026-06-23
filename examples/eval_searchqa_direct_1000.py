import argparse
import json
import re
import string
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_system.environments.prompts.search import SEARCH_TEMPLATE_NO_HIS


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def extract_first_answer(reward_model) -> str:
    reward_model = reward_model or {}
    ground_truth = (reward_model.get("ground_truth") or {}).get("target")
    if ground_truth is None:
        return ""
    if isinstance(ground_truth, str):
        return ground_truth.strip()
    try:
        for value in ground_truth:
            text = str(value).strip()
            if text:
                return text
    except TypeError:
        return str(ground_truth).strip()
    return ""


def extract_first_answer_from_row(row) -> str:
    if "reward_model" in row and row["reward_model"] is not None:
        answer = extract_first_answer(row["reward_model"])
        if answer:
            return answer

    if "ground_truth" in row:
        ground_truth = row["ground_truth"]
        if isinstance(ground_truth, dict):
            answer = extract_first_answer(ground_truth)
            if answer:
                return answer
        elif isinstance(ground_truth, str):
            if ground_truth.strip():
                return ground_truth.strip()
        else:
            try:
                for value in ground_truth:
                    text = str(value).strip()
                    if text:
                        return text
            except TypeError:
                text = str(ground_truth).strip()
                if text:
                    return text

    if "answer" in row:
        return str(row["answer"]).strip()
    return ""


def extract_answer_text(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def strict_em(prediction: str, gold: str) -> bool:
    return normalize_text(prediction) == normalize_text(gold)


def load_skill_text(skill_file: str) -> str:
    if not skill_file:
        return ""
    text = Path(skill_file).read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return text + "\n\n"


def question_from_row(row) -> str:
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, dict) and extra_info.get("question"):
        return str(extra_info["question"])
    if row.get("question"):
        return str(row["question"])
    prompt = row.get("prompt")
    if prompt is not None:
        try:
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            if isinstance(prompt, list) and prompt:
                return str(prompt[-1].get("content", ""))
        except Exception:
            pass
    return ""


def build_prompt(row, prompt_style: str, skill_text: str):
    if prompt_style == "dataset":
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        return prompt

    if prompt_style in {"row_skill", "skill0_first_step_row_skill"}:
        row_skill = str(row.get("skill") or "").strip()
        skill_text = (row_skill + "\n\n") if row_skill else ""

    question = question_from_row(row)
    prompt_text = SEARCH_TEMPLATE_NO_HIS.format(
        skill_context=skill_text,
        task_description=question,
    ).strip()

    if prompt_style in {"skill0_first_step", "skill0_first_step_row_skill"}:
        return [
            {"role": "user", "content": prompt_text},
        ]

    return [
        {"role": "system", "content": "You are a helpful and harmless assistant."},
        {"role": "user", "content": prompt_text},
    ]


def batched(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i, items[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument(
        "--prompt_style",
        choices=[
            "dataset",
            "with_skill",
            "row_skill",
            "skill0_first_step",
            "skill0_first_step_row_skill",
        ],
        default="dataset",
    )
    parser.add_argument("--skill_file", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{args.run_name}.summary.json"
    output_parquet = output_dir / f"{args.run_name}.predictions.parquet"

    df = pd.read_parquet(args.data_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    skill_text = load_skill_text(args.skill_file) if args.prompt_style == "with_skill" else ""
    prompts = [build_prompt(row, args.prompt_style, skill_text) for _, row in df.iterrows()]
    gold_answers = [extract_first_answer_from_row(row) for _, row in df.iterrows()]

    print(f"rows={len(df)} prompt_style={args.prompt_style}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=False,
    )
    model.eval()

    rows = []
    with torch.no_grad():
        for start, prompt_batch in batched(prompts, args.batch_size):
            texts = [
                tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                for prompt in prompt_batch
            ]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for offset, raw_output in enumerate(decoded):
                idx = start + offset
                prediction = extract_answer_text(raw_output)
                gold = gold_answers[idx]
                success = strict_em(prediction, gold)
                rows.append(
                    {
                        "index": idx,
                        "question": question_from_row(df.iloc[idx]),
                        "gold": gold,
                        "raw_output": raw_output,
                        "prediction": prediction,
                        "has_search": "<search>" in raw_output.lower(),
                        "has_answer_tag": "<answer>" in raw_output.lower(),
                        "success": success,
                    }
                )

            done = min(start + args.batch_size, len(prompts))
            if done % max(args.batch_size * 5, 1) == 0 or done == len(prompts):
                print(f"processed={done}/{len(prompts)}")

    result_df = pd.DataFrame(rows)
    result_df.to_parquet(output_parquet, index=False)

    summary = {
        "run_name": args.run_name,
        "model_path": args.model_path,
        "data_path": args.data_path,
        "prompt_style": args.prompt_style,
        "num_examples": len(result_df),
        "success": int(result_df["success"].sum()) if len(result_df) else 0,
        "success_rate": float(result_df["success"].mean()) if len(result_df) else 0.0,
        "search_rate": float(result_df["has_search"].mean()) if len(result_df) else 0.0,
        "answer_tag_rate": float(result_df["has_answer_tag"].mean()) if len(result_df) else 0.0,
        "predictions_path": str(output_parquet),
        "samples": result_df.head(20).to_dict(orient="records"),
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
