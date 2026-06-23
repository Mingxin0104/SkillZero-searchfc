import argparse
import json
import os
import re
import string
from typing import List

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def batchify(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    questions = df["question"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    predictions = []
    with torch.no_grad():
        for q_batch in batchify(questions, args.batch_size):
            enc = tokenizer(
                q_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            predictions.extend([t.strip() for t in texts])

    exact = 0
    contains = 0
    rows = []
    for question, answer, prediction in zip(questions, answers, predictions):
        norm_answer = normalize_text(answer)
        norm_pred = normalize_text(prediction)
        em = norm_pred == norm_answer
        ca = norm_answer != "" and norm_answer in norm_pred
        exact += int(em)
        contains += int(ca)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "exact_match": em,
                "contains_answer": ca,
            }
        )

    metrics = {
        "num_examples": len(rows),
        "exact_match": exact / len(rows) if rows else 0.0,
        "contains_answer": contains / len(rows) if rows else 0.0,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "samples": rows[:50]}, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
