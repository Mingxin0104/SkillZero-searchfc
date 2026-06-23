import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.data_preprocess.generate_searchqa_skill_traces_local import (
    SEARCH_TEMPLATE,
    SEARCH_TEMPLATE_NO_HIS,
    normalize_env_kwargs,
    score_answer,
    to_target_list,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert search agent. For each question, reason step by step, call search when evidence is "
    "needed using <search>...</search>, read the returned <information>...</information>, and finish with "
    "only the answer inside <answer>...</answer> when the evidence supports it."
)

ACTION_FORMAT_INSTRUCTION = (
    "\nImportant output rule: do not write a long hidden reasoning block. "
    "For this step, output exactly one executable tag as soon as possible: "
    "<search>concise search query</search> or <answer>final answer</answer>. "
    "Do not output both tags in one step. "
    "When answering, put only the shortest final entity, name, title, date, number, or phrase in "
    "<answer>; never include explanations, citations, full sentences, or punctuation.\n"
)


def load_records(path):
    path = os.path.expanduser(path)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return pd.read_parquet(path).to_dict(orient="records")


def load_skill_text(path):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        return f.read().strip()


def load_skill_bank(skill_dir):
    skill_dir = os.path.expanduser(skill_dir)
    bank = {
        "general_skills": load_skill_text(os.path.join(skill_dir, "general_skills.md")),
    }
    for name in ["direct_retrieval", "multi_hop_reasoning", "entity_attribute_lookup", "compare"]:
        path = os.path.join(skill_dir, f"{name}.md")
        if os.path.exists(path):
            bank[name] = load_skill_text(path)
    return bank


def select_skill(skill_bank, skill_type):
    skill_type = skill_type or "direct_retrieval"
    selected = skill_bank.get(skill_type, skill_bank.get("direct_retrieval", ""))
    general = skill_bank.get("general_skills", "")
    if selected and selected != general:
        return f"{general}\n\n{selected}".strip() + "\n\n"
    return selected.strip() + "\n\n"


def build_prompt(question, skill_context, history):
    if not history:
        prompt = SEARCH_TEMPLATE_NO_HIS.format(skill_context=skill_context, task_description=question).strip()
    else:
        prompt = SEARCH_TEMPLATE.format(
            skill_context=skill_context,
            task_description=question,
            memory_context="\n".join(history),
            step_count=len(history) // 2,
            history_length=len(history) // 2,
        ).strip()
    return prompt + ACTION_FORMAT_INSTRUCTION


def render_chat_prompt(tokenizer, prompt, disable_thinking=True):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


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
        docs = response.json().get("result", [[]])
        docs = docs[0] if docs else []
        parts = []
        for i, doc in enumerate(docs, start=1):
            content = doc.get("contents")
            if content is None:
                content = f"{doc.get('title', '')}\n{doc.get('text', '')}".strip()
            content = content.strip()
            if self.max_doc_chars > 0 and len(content) > self.max_doc_chars:
                content = content[: self.max_doc_chars].rstrip()
            parts.append(f"Doc {i}: {content}")
        return "\n<information>" + "\n".join(parts) + "</information>\n"

    def batch_search(self, queries, max_workers=64):
        if not queries:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.search, queries))


class VLLMCompletionsClient:
    def __init__(self, base_url, model_name, api_key="EMPTY", timeout=600, max_retries=4, retry_sleep=3.0):
        self.endpoint = base_url.rstrip("/") + "/completions"
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def generate_batch(self, prompts, max_new_tokens, temperature, top_p):
        payload = {
            "model": self.model_name,
            "prompt": prompts,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
        }
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                choices = sorted(data.get("choices", []), key=lambda x: x.get("index", 0))
                if len(choices) != len(prompts):
                    raise ValueError(f"Expected {len(prompts)} choices, got {len(choices)}")
                return [choice.get("text", "").strip() for choice in choices]
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_sleep)
        raise RuntimeError(f"vLLM completion request failed after {self.max_retries} attempts: {last_error}")


def make_state(row, source_index, candidate_id, skill_bank):
    env_kwargs = normalize_env_kwargs(row)
    skill_type = env_kwargs.get("skill_type") or row.get("skill_type") or "direct_retrieval"
    skill_context = select_skill(skill_bank, skill_type)
    return {
        "trace_id": str(uuid.uuid4()),
        "source_index": int(source_index),
        "candidate_id": int(candidate_id),
        "question": env_kwargs["question"],
        "ground_truth": to_target_list(env_kwargs["ground_truth"]),
        "data_source": env_kwargs.get("data_source", "searchqa"),
        "skill_type": skill_type,
        "skill": skill_context.strip(),
        "history": [],
        "steps": [],
        "done": False,
        "success": False,
        "final_reward": 0.0,
        "answer": None,
    }


def parse_action(text):
    match = re.search(r"<(search|answer)>(.*?)</\1>", text, re.DOTALL)
    if not match:
        return "invalid", None
    return match.group(1), match.group(2).strip()


def normalize_answer_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def strict_score_answer(answer, ground_truth):
    pred = str(answer or "").strip()
    if not pred:
        return 0.0, False
    pred_words = pred.split()
    if len(pred_words) > 12 or len(pred) > 120:
        return 0.0, False
    if re.search(r"[.!?;:]", pred):
        return 0.0, False
    norm_pred = normalize_answer_text(pred)
    if not norm_pred:
        return 0.0, False
    for target in ground_truth:
        norm_target = normalize_answer_text(target)
        if norm_target and norm_pred == norm_target:
            return 1.0, True
    return 0.0, False


def search_process_from_steps(steps):
    process = []
    for step in steps:
        item = {
            "step_id": step.get("step_id"),
            "assistant": step.get("assistant", ""),
            "action_type": step.get("action_type"),
        }
        if step.get("search_query") is not None:
            item["search_query"] = step.get("search_query")
        if step.get("observation"):
            item["observation"] = step.get("observation")
        process.append(item)
    return process


def messages_from_record(record):
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{record['skill']}\n\nQuestion: {record['question']}"},
    ]
    for step in record["search_process"]:
        assistant = step.get("assistant")
        observation = step.get("observation")
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        if observation:
            messages.append({"role": "user", "content": observation})
    return messages


def compact_record(state):
    record = {
        "trace_id": state["trace_id"],
        "source_index": state["source_index"],
        "candidate_id": state["candidate_id"],
        "data_source": state["data_source"],
        "skill_type": state["skill_type"],
        "skill": state["skill"],
        "question": state["question"],
        "ground_truth": state["ground_truth"],
        "search_process": search_process_from_steps(state["steps"]),
        "answer": state["answer"],
        "teacher_success": bool(state["success"]),
        "final_reward": state["final_reward"],
    }
    record["messages"] = messages_from_record(record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skill_dir", default="skills/search")
    parser.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--vllm_base_url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--vllm_api_key", default="EMPTY")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--retrieval_topk", type=int, default=3)
    parser.add_argument("--retrieval_max_doc_chars", type=int, default=1600)
    parser.add_argument("--retrieval_workers", type=int, default=64)
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--prompt_token_margin", type=int, default=256)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    accepted_path = os.path.join(args.out_dir, "accepted.jsonl")
    all_path = os.path.join(args.out_dir, "all_candidates.jsonl")
    parquet_path = os.path.join(args.out_dir, "train.parquet")

    records = load_records(args.input_path)
    subset = records[args.start : args.start + args.limit]
    skill_bank = load_skill_bank(args.skill_dir)
    retriever = HttpRetriever(
        args.search_url,
        topk=args.retrieval_topk,
        max_doc_chars=args.retrieval_max_doc_chars,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model_name = args.model_name or args.model_path
    llm = VLLMCompletionsClient(
        base_url=args.vllm_base_url,
        model_name=model_name,
        api_key=args.vllm_api_key,
    )

    accepted_count = 0
    all_count = 0
    start_time = time.time()
    mode = "a" if args.append else "w"
    with open(accepted_path, mode, encoding="utf-8") as accepted_f, open(
        all_path, mode, encoding="utf-8"
    ) as all_f:
        for base_start in range(0, len(subset), args.batch_size):
            batch_rows = subset[base_start : base_start + args.batch_size]
            states = []
            for offset, row in enumerate(batch_rows):
                source_index = args.start + base_start + offset
                for candidate_id in range(args.num_candidates):
                    states.append(make_state(row, source_index, candidate_id, skill_bank))

            successful_sources = set()
            for step_id in range(1, args.max_steps + 1):
                active = [s for s in states if not s["done"] and s["source_index"] not in successful_sources]
                if not active:
                    break
                prompts = []
                prompt_states = []
                for state in active:
                    prompt = build_prompt(state["question"], state["skill"] + "\n\n", state["history"])
                    rendered = render_chat_prompt(tokenizer, prompt, disable_thinking=not args.enable_thinking)
                    prompt_len = len(tokenizer.encode(rendered, add_special_tokens=False))
                    if prompt_len + args.max_new_tokens + args.prompt_token_margin > args.max_model_len:
                        state["steps"].append(
                            {
                                "step_id": step_id,
                                "assistant": "<answer></answer>",
                                "action_type": "invalid",
                                "search_query": None,
                                "observation": f"Invalid action: prompt length {prompt_len} exceeds context budget.",
                                "reward": 0.0,
                                "done": True,
                                "is_action_valid": False,
                            }
                        )
                        state["done"] = True
                        continue
                    prompts.append(rendered)
                    prompt_states.append(state)

                if not prompts:
                    continue

                outputs = llm.generate_batch(
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                pending_searches = []
                for state, assistant in zip(prompt_states, outputs):
                    action_type, payload = parse_action(assistant)
                    is_valid = action_type != "invalid"
                    if action_type == "search":
                        step = {
                            "step_id": step_id,
                            "assistant": assistant,
                            "action_type": "search",
                            "search_query": payload,
                            "observation": None,
                            "reward": 0.0,
                            "done": False,
                            "is_action_valid": is_valid,
                        }
                        state["steps"].append(step)
                        pending_searches.append((state, payload))
                    elif action_type == "answer":
                        loose_reward, _ = score_answer(assistant, state["ground_truth"])
                        reward, success = strict_score_answer(payload, state["ground_truth"])
                        state["answer"] = payload
                        state["success"] = success
                        state["final_reward"] = reward
                        state["done"] = True
                        state["steps"].append(
                            {
                                "step_id": step_id,
                                "assistant": assistant,
                                "action_type": "answer",
                                "search_query": None,
                                "observation": "",
                                "reward": reward,
                                "loose_reward": loose_reward,
                                "done": True,
                                "is_action_valid": is_valid,
                            }
                        )
                        if success:
                            successful_sources.add(state["source_index"])
                    else:
                        observation = "Invalid action: response must contain <search>...</search> or <answer>...</answer>."
                        state["history"].append(assistant)
                        state["history"].append(observation)
                        state["steps"].append(
                            {
                                "step_id": step_id,
                                "assistant": assistant,
                                "action_type": "invalid",
                                "search_query": None,
                                "observation": observation,
                                "reward": 0.0,
                                "done": step_id >= args.max_steps,
                                "is_action_valid": False,
                            }
                        )
                        state["done"] = step_id >= args.max_steps

                kept_searches = [(s, q) for s, q in pending_searches if s["source_index"] not in successful_sources]
                observations = retriever.batch_search(
                    [query for _, query in kept_searches],
                    max_workers=args.retrieval_workers,
                )
                for (state, _), observation in zip(kept_searches, observations):
                    state["steps"][-1]["observation"] = observation
                    state["history"].append(state["steps"][-1]["assistant"])
                    state["history"].append(observation)

            best_by_source = {}
            for state in states:
                record = compact_record(state)
                all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_count += 1
                has_search = any(step["action_type"] == "search" for step in state["steps"])
                has_invalid = any(step["action_type"] == "invalid" for step in state["steps"])
                if state["success"] and state["answer"] and has_search and not has_invalid:
                    best_by_source.setdefault(state["source_index"], record)

            for record in best_by_source.values():
                accepted_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted_count += 1

            accepted_f.flush()
            all_f.flush()
            done = args.start + min(base_start + len(batch_rows), len(subset))
            elapsed = max(time.time() - start_time, 1e-6)
            processed_local = min(base_start + len(batch_rows), len(subset))
            examples_per_sec = processed_local / elapsed
            eta_sec = (len(subset) - processed_local) / examples_per_sec if examples_per_sec > 0 else 0.0
            print(
                f"processed={done}/{args.start + len(subset)} "
                f"accepted={accepted_count} all_candidates={all_count} "
                f"examples_per_sec={examples_per_sec:.3f} eta_min={eta_sec / 60:.1f}",
                flush=True,
            )

    accepted_records = load_records(accepted_path)
    pd.DataFrame(accepted_records).to_parquet(parquet_path, index=False)
    print(f"saved_jsonl={accepted_path} rows={len(accepted_records)}")
    print(f"saved_parquet={parquet_path} rows={len(accepted_records)}")


if __name__ == "__main__":
    main()
