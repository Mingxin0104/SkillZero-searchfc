import argparse
import os

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    out = []
    for value in values:
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--input_path", default="~/data/searchR1_processed_direct/test.parquet")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input_path)
    output_path = os.path.expanduser(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_parquet(input_path)
    prompts = []
    reward_models = []
    for _, row in df.iterrows():
        extra_info = row.get("extra_info", {}) or {}
        reward_model = row.get("reward_model", {}) or {}
        question = extra_info.get("question", "")
        prompt = SEARCH_TEMPLATE_NO_HIS.format(skill_context="", task_description=question).strip()
        prompts.append(
            [
                {"role": "system", "content": "You are a helpful and harmless assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        reward_models.append({"ground_truth": {"target": first_answer(reward_model.get("ground_truth"))}})

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

    responses = []
    with torch.no_grad():
        for batch in batched(prompts, args.batch_size):
            texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
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
            responses.extend([[item.strip()] for item in decoded])

    out_df = pd.DataFrame(
        {
            "responses": responses,
            "data_source": ["searchR1_nq"] * len(responses),
            "reward_model": reward_models,
        }
    )
    out_df.to_parquet(output_path, index=False)
    print(f"saved={output_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
