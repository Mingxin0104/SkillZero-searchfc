import argparse
import json
import os
import uuid

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_system.environments.env_package.search.envs import SearchMultiProcessEnv
from agent_system.environments.prompts.search import SEARCH_TEMPLATE, SEARCH_TEMPLATE_NO_HIS


def load_skills(skill_file):
    if not skill_file:
        return ""
    with open(os.path.expanduser(skill_file), "r", encoding="utf-8") as f:
        return f.read().strip() + "\n\n"


def to_target_list(value):
    if isinstance(value, dict) and "target" in value:
        value = value["target"]
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    result = []
    for item in values:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def build_skill_context(skill_text):
    return skill_text if skill_text else ""


def build_prompt(question, skill_context, history):
    if not history:
        return SEARCH_TEMPLATE_NO_HIS.format(
            skill_context=skill_context,
            task_description=question,
        ).strip()

    memory_context = "\n".join(history)
    return SEARCH_TEMPLATE.format(
        skill_context=skill_context,
        task_description=question,
        memory_context=memory_context,
        step_count=len(history) // 2,
        history_length=len(history) // 2,
    ).strip()


def generate_action(model, tokenizer, prompt, max_new_tokens=256):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def row_to_env_kwargs(row):
    env_kwargs = dict((row.get("env_kwargs") or {}))
    return {
        "ground_truth": env_kwargs.get("ground_truth"),
        "question": env_kwargs.get("question"),
        "data_source": env_kwargs.get("data_source", row.get("data_source", "searchqa")),
        "skill_type": env_kwargs.get("skill_type", row.get("skill_type")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--skill_file", required=True)
    parser.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    input_path = os.path.expanduser(args.input_path)
    output_path = os.path.expanduser(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()

    skill_text = load_skills(args.skill_file)
    skill_context = build_skill_context(skill_text)

    df = pd.read_parquet(input_path)
    subset = df.iloc[args.start : args.start + args.limit]

    from omegaconf import OmegaConf

    env_config = OmegaConf.create(
        {
            "max_steps": args.max_steps,
            "search": {
                "search_url": args.search_url,
                "topk": 3,
                "timeout": 30,
                "log_requests": True,
                "search_reward_coef": 0.0,
            },
        }
    )
    env = SearchMultiProcessEnv(seed=0, env_num=1, group_n=1, is_train=False, env_config=env_config)

    records = []
    for idx, (_, row) in enumerate(subset.iterrows()):
        env_kwargs = row_to_env_kwargs(row)
        question = env_kwargs["question"]
        ground_truth = to_target_list(env_kwargs["ground_truth"])
        obs_list, info_list = env.reset([env_kwargs])
        current_observation = obs_list[0]
        history = []
        steps = []
        final_reward = 0.0
        success = False

        for step_id in range(1, args.max_steps + 1):
            prompt = build_prompt(question, skill_context, history)
            assistant = generate_action(model, tokenizer, prompt)

            next_obs, rewards, dones, infos = env.step([assistant])
            observation = next_obs[0]
            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0]

            action_type = "answer" if "<answer>" in assistant and "</answer>" in assistant else "search"
            search_query = None
            if action_type == "search" and "<search>" in assistant and "</search>" in assistant:
                search_query = assistant.split("<search>", 1)[1].split("</search>", 1)[0].strip()

            steps.append(
                {
                    "step_id": step_id,
                    "assistant": assistant,
                    "action_type": action_type,
                    "search_query": search_query,
                    "observation": observation,
                    "reward": reward,
                    "done": done,
                    "is_action_valid": bool(info.get("is_action_valid", True)),
                }
            )

            if action_type == "search":
                history.append(assistant)
                if observation:
                    history.append(observation)

            final_reward = reward
            success = bool(info.get("won", False))
            current_observation = observation
            if done:
                break

        final_answer = None
        if steps:
            last_assistant = steps[-1]["assistant"]
            if "<answer>" in last_assistant and "</answer>" in last_assistant:
                final_answer = last_assistant.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()

        records.append(
            {
                "trace_id": str(uuid.uuid4()),
                "source_index": int(args.start + idx),
                "question": question,
                "ground_truth": ground_truth,
                "data_source": env_kwargs.get("data_source", "searchqa"),
                "skill_type": env_kwargs.get("skill_type"),
                "use_skill": True,
                "final_reward": final_reward,
                "success": success,
                "final_answer": f"<answer>{final_answer}</answer>" if final_answer else None,
                "answer": final_answer,
                "steps": steps,
            }
        )

        if (idx + 1) % 10 == 0:
            print(f"processed={idx + 1}")

    env.close()

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"saved={output_path} rows={len(records)}")


if __name__ == "__main__":
    main()
