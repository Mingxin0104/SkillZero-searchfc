import argparse
import os

os.environ.setdefault("NVTE_PROJECT_BUILDING", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    output_path = os.path.expanduser(args.output_path)
    os.makedirs(output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
