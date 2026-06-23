import argparse
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from examples.data_preprocess.build_searchqa_trace_sft import build_messages, is_usable_trace
from examples.data_preprocess.generate_searchqa_skill_traces_local import (
    normalize_env_kwargs,
    to_target_list,
)
from agent_system.environments.prompts.search import SEARCH_TEMPLATE, SEARCH_TEMPLATE_NO_HIS
from agent_system.environments.env_package.search.projection import search_projection
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import compute_score


def load_skills(skill_file):
    with open(os.path.expanduser(skill_file), "r", encoding="utf-8") as f:
        return f.read().strip() + "\n\n"


def load_skill_file_sections(filepath):
    with open(os.path.expanduser(filepath), "r", encoding="utf-8") as f:
        content = f.read()
    parts = re_split_sections(content)
    return parts


def re_split_sections(content):
    import re

    sections = {}
    parts = re.split(r"### (.+?) ###\s*\n", content)
    for i in range(1, len(parts), 2):
        key = parts[i].strip()
        value = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[key] = value
    return sections


class Skill0SkillContext:
    def __init__(self, skill_file=None, skill_mapping_file=None):
        self.skills = {}
        self.task_to_sections = {}
        if skill_mapping_file:
            self._load_mapping(skill_mapping_file)
        elif skill_file:
            self.skills = load_skill_file_sections(skill_file)

    def _load_mapping(self, skill_mapping_file):
        skill_mapping_file = os.path.expanduser(skill_mapping_file)
        mapping_dir = os.path.dirname(skill_mapping_file)
        with open(skill_mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        skill_files = mapping.get("skill_files", {})
        task_to_skill = mapping.get("task_to_skill", {})
        per_skill_sections = {}
        for skill_name, rel_path in skill_files.items():
            path = os.path.join(mapping_dir, rel_path)
            sections = load_skill_file_sections(path)
            per_skill_sections[skill_name] = sections
            for section, content in sections.items():
                self.skills[section] = content
        for task, skill_name in task_to_skill.items():
            self.task_to_sections[task] = list(per_skill_sections.get(skill_name, {}).keys())

    def get(self, skill_type=None):
        parts = []
        general = self.skills.get("GENERAL SKILLS", "")
        if general:
            parts.append(general)
        if skill_type and self.task_to_sections:
            for section_key in self.task_to_sections.get(skill_type, []):
                content = self.skills.get(section_key, "")
                if content:
                    parts.append(content)
        elif not self.task_to_sections:
            for section_key, content in self.skills.items():
                if section_key != "GENERAL SKILLS" and content:
                    parts.append(content)
        return "\n\n".join(parts).strip() + "\n\n" if parts else ""


def render_search_memory(history):
    lines = []
    for item in history:
        action = item["action"].replace("\n", " ")
        observation = item["observation"].replace("\n", " ")
        lines.append(f"{action} {observation}\n")
    return "\n".join(lines)


def build_prompt(question, skill_context, history):
    if not history:
        return SEARCH_TEMPLATE_NO_HIS.format(skill_context=skill_context, task_description=question).strip()
    return SEARCH_TEMPLATE.format(
        skill_context=skill_context,
        task_description=question,
        memory_context=render_search_memory(history),
        step_count=len(history),
        history_length=len(history),
    ).strip()


class HttpRetriever:
    def __init__(self, search_url, topk=3, timeout=60, max_doc_chars=1800):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self.max_doc_chars = max_doc_chars

    def search(self, query):
        response = requests.post(
            self.search_url,
            json={"query": query, "topk": self.topk, "return_scores": False},
            timeout=self.timeout,
        )
        response.raise_for_status()
        results = response.json().get("result", [[]])
        docs = results[0] if results else []
        parts = []
        for i, doc in enumerate(docs, start=1):
            content = doc.get("contents")
            if content is None:
                content = f"{doc.get('title', '')}\n{doc.get('text', '')}".strip()
            content = content.strip()
            if self.max_doc_chars > 0 and len(content) > self.max_doc_chars:
                content = content[: self.max_doc_chars].rstrip()
            parts.append(f"Doc {i}: {content.strip()}")
        return "\n<information>" + "\n".join(parts) + "</information>\n"

    def batch_search(self, queries, max_workers=32):
        if not queries:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.search, queries))


def load_records(path):
    path = os.path.expanduser(path)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return pd.read_parquet(path).to_dict(orient="records")


def make_state(row, source_index, candidate_id):
    env_kwargs = normalize_env_kwargs(row)
    return {
        "trace_id": str(uuid.uuid4()),
        "source_index": int(source_index),
        "candidate_id": int(candidate_id),
        "question": env_kwargs["question"],
        "ground_truth": to_target_list(env_kwargs["ground_truth"]),
        "data_source": env_kwargs.get("data_source", "searchqa"),
        "skill_type": env_kwargs.get("skill_type"),
        "skill_context": None,
        "history": [],
        "steps": [],
        "done": False,
        "success": False,
        "final_reward": 0.0,
        "final_answer": None,
    }


def state_to_record(state):
    final_answer = state["final_answer"]
    return {
        "trace_id": state["trace_id"],
        "source_index": state["source_index"],
        "candidate_id": state["candidate_id"],
        "question": state["question"],
        "ground_truth": state["ground_truth"],
        "data_source": state["data_source"],
        "skill_type": state["skill_type"],
        "skill_context": state["skill_context"],
        "use_skill": True,
        "final_reward": state["final_reward"],
        "success": state["success"],
        "final_answer": f"<answer>{final_answer}</answer>" if final_answer else None,
        "answer": final_answer,
        "steps": state["steps"],
    }


def score_grpo_reward(chat_history, action, ground_truth):
    solution = "".join(
        item["action"] + item["observation"]
        for item in chat_history
    ) + action
    return float(compute_score(solution, {"target": ground_truth}))


def projected_action_type(action):
    if "<search>" in action and "</search>" in action:
        return "search", action.split("<search>", 1)[1].split("</search>", 1)[0].strip()
    if "<answer>" in action and "</answer>" in action:
        return "answer", action.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return "invalid", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--save_all_path", required=True)
    parser.add_argument("--sft_output_path", required=True)
    parser.add_argument("--skill_file", default=None)
    parser.add_argument("--skill_mapping_file", default=None)
    parser.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.72)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--retrieval_topk", type=int, default=3)
    parser.add_argument("--retrieval_max_doc_chars", type=int, default=1800)
    parser.add_argument("--retrieval_workers", type=int, default=32)
    parser.add_argument("--flush_every", type=int, default=20)
    parser.add_argument("--prompt_token_margin", type=int, default=256)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    for path in [args.output_path, args.save_all_path, args.sft_output_path]:
        os.makedirs(os.path.dirname(os.path.expanduser(path)), exist_ok=True)

    records = load_records(args.input_path)
    subset = records[args.start : args.start + args.limit]
    if not args.skill_mapping_file and not args.skill_file:
        raise ValueError("Either --skill_mapping_file or --skill_file is required.")
    skill_provider = Skill0SkillContext(args.skill_file, args.skill_mapping_file)
    retriever = HttpRetriever(args.search_url, topk=args.retrieval_topk, max_doc_chars=args.retrieval_max_doc_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output_path = os.path.expanduser(args.output_path)
    save_all_path = os.path.expanduser(args.save_all_path)
    accepted = sum(1 for _ in open(output_path, "r", encoding="utf-8")) if args.append and os.path.exists(output_path) else 0
    all_count = (
        sum(1 for _ in open(save_all_path, "r", encoding="utf-8")) if args.append and os.path.exists(save_all_path) else 0
    )
    mode = "a" if args.append else "w"
    with open(output_path, mode, encoding="utf-8") as success_f, open(
        save_all_path, mode, encoding="utf-8"
    ) as all_f:
        for base_start in range(0, len(subset), args.batch_size):
            batch_rows = subset[base_start : base_start + args.batch_size]
            states = []
            for offset, row in enumerate(batch_rows):
                source_index = args.start + base_start + offset
                for candidate_id in range(args.num_candidates):
                    state = make_state(row, source_index, candidate_id)
                    state["skill_context"] = skill_provider.get(state.get("skill_type"))
                    states.append(state)

            for step_id in range(1, args.max_steps + 1):
                active = [state for state in states if not state["done"]]
                if not active:
                    break
                prompts = []
                prompt_states = []
                for state in active:
                    prompt = build_prompt(state["question"], state["skill_context"], state["history"])
                    messages = [{"role": "user", "content": prompt}]
                    rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompt_len = len(tokenizer.encode(rendered_prompt, add_special_tokens=False))
                    if prompt_len + args.max_new_tokens + args.prompt_token_margin > args.max_model_len:
                        assistant = "<answer></answer>"
                        observation = (
                            f"Invalid action: prompt length {prompt_len} exceeds the configured context budget."
                        )
                        state["history"].append({"action": assistant, "observation": observation})
                        state["steps"].append(
                            {
                                "step_id": step_id,
                                "assistant": assistant,
                                "action_type": "invalid",
                                "search_query": None,
                                "observation": observation,
                                "reward": 0.0,
                                "done": True,
                                "is_action_valid": False,
                            }
                        )
                        state["final_reward"] = 0.0
                        state["done"] = True
                        continue
                    prompts.append(rendered_prompt)
                    prompt_states.append(state)

                if not prompts:
                    continue
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                pending_searches = []
                successful_sources = set()
                raw_assistants = [output.outputs[0].text.strip() for output in outputs]
                projected_actions, action_valids = search_projection(raw_assistants)
                for state, raw_assistant, assistant, valid in zip(
                    prompt_states, raw_assistants, projected_actions, action_valids
                ):
                    action_type, payload = projected_action_type(assistant)
                    query = None
                    if action_type == "search":
                        query = payload
                        observation = None
                        reward = 0.0
                        done = False
                    elif action_type == "answer":
                        observation = ""
                        reward = score_grpo_reward(state["history"], assistant, state["ground_truth"])
                        success = reward >= 1.0
                        done = True
                        state["success"] = success
                        state["final_answer"] = payload
                        if success:
                            successful_sources.add(state["source_index"])
                    else:
                        observation = "Invalid action: response must contain <search>...</search> or <answer>...</answer>."
                        reward = 0.0
                        done = step_id >= args.max_steps
                        action_type = "invalid"
                        state["history"].append({"action": assistant, "observation": observation})

                    state["steps"].append(
                        {
                            "step_id": step_id,
                            "assistant": assistant,
                            "raw_assistant": raw_assistant,
                            "action_type": action_type,
                            "search_query": query if action_type == "search" else None,
                            "observation": observation,
                            "reward": reward,
                            "done": done,
                            "is_action_valid": valid,
                        }
                    )
                    state["final_reward"] = reward
                    state["done"] = done
                    if action_type == "search":
                        pending_searches.append((state, assistant, query))

                if successful_sources:
                    for state in states:
                        if state["source_index"] in successful_sources and not state["success"]:
                            state["done"] = True
                    kept_searches = []
                    for state, assistant, query in pending_searches:
                        if state["source_index"] in successful_sources:
                            state["steps"][-1]["observation"] = "Skipped after another candidate for this question succeeded."
                        else:
                            kept_searches.append((state, assistant, query))
                    pending_searches = kept_searches

                observations = retriever.batch_search(
                    [query for _, _, query in pending_searches],
                    max_workers=args.retrieval_workers,
                )
                for (state, assistant, _), observation in zip(pending_searches, observations):
                    state["steps"][-1]["observation"] = observation
                    state["history"].append({"action": assistant, "observation": observation})

            best_by_source = {}
            for state in states:
                record = state_to_record(state)
                all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_count += 1
                all_actions_valid = all(step.get("is_action_valid", True) for step in record.get("steps", []))
                if is_usable_trace(record) and all_actions_valid and record["source_index"] not in best_by_source:
                    best_by_source[record["source_index"]] = record

            for record in best_by_source.values():
                success_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1

            if args.flush_every > 0 and (base_start // args.batch_size + 1) % args.flush_every == 0:
                success_f.flush()
                all_f.flush()
            print(
                f"processed={args.start + min(base_start + len(batch_rows), len(subset))}/{args.start + len(subset)} "
                f"accepted={accepted} all_candidates={all_count}",
                flush=True,
            )

    success_records = load_records(args.output_path)
    rows = []
    for record in success_records:
        rows.append(
            {
                "messages": build_messages(record, (
                    "You are an expert agent tasked with answering the given question step-by-step. "
                    "You may search when needed using <search>...</search> and only provide the final answer "
                    "when confident using <answer>...</answer>."
                )),
                "skill": record.get("skill_context", ""),
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "data_source": record.get("data_source", "searchqa"),
                "skill_type": record.get("skill_type"),
                "teacher_success": bool(record.get("success")),
                "final_reward": record.get("final_reward"),
                "source_index": record.get("source_index"),
                "candidate_id": record.get("candidate_id"),
                "trace_id": record.get("trace_id"),
            }
        )
    pd.DataFrame(rows).to_parquet(os.path.expanduser(args.sft_output_path), index=False)
    print(f"saved={args.sft_output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
