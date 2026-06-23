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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.data_preprocess.generate_searchqa_skill_traces_local import (  # noqa: E402
    normalize_env_kwargs,
    score_answer,
    to_target_list,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert search agent. Use the provided search function when evidence is needed. "
    "After enough evidence is collected, answer with only the shortest final answer and no explanation."
)

ANSWER_RULE = (
    "Important output rule: when you answer directly, return only the final entity, title, number, date, or short phrase. "
    "Do not include explanation, punctuation, or extra words."
)

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the corpus for evidence relevant to the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise search query.",
                }
            },
            "required": ["query"],
        },
    },
}


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


def select_skill(skill_bank, skill_type, skill_variant):
    general = skill_bank.get("general_skills", "").strip()
    selected = skill_bank.get(skill_type or "direct_retrieval", "").strip()
    if skill_variant == "general_only":
        return general
    if skill_variant == "selected_only":
        return selected or general
    if selected and selected != general:
        return f"{general}\n\n{selected}".strip()
    return general


def build_messages(question, skill_context):
    user_content = f"{skill_context}\n\nQuestion: {question}\n\n{ANSWER_RULE}".strip()
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


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
        return "\n".join(parts)

    def batch_search(self, queries, max_workers=64):
        if not queries:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.search, queries))


class VLLMToolClient:
    def __init__(self, base_url, model_name, api_key="EMPTY", timeout=600, max_retries=4, retry_sleep=3.0):
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.api_key = api_key

    def _request_one(self, payload):
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = session.post(self.endpoint, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                response_text = getattr(getattr(exc, "response", None), "text", "")
                if status_code == 400:
                    return {
                        "_request_error": f"HTTP 400: {response_text[:2000]}",
                        "_status_code": 400,
                    }
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_sleep)
        return {
            "_request_error": f"Request failed after {self.max_retries} attempts: {last_error}",
            "_status_code": getattr(getattr(last_error, 'response', None), 'status_code', None),
        }

    def generate_batch(self, messages_batch, max_new_tokens, temperature, top_p, max_workers):
        payloads = []
        for messages in messages_batch:
            payloads.append(
                {
                    "model": self.model_name,
                    "messages": messages,
                    "tools": [SEARCH_TOOL],
                    "tool_choice": "auto",
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_new_tokens,
                }
            )
        with ThreadPoolExecutor(max_workers=min(max_workers, len(payloads))) as executor:
            return list(executor.map(self._request_one, payloads))


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


def make_state(row, source_index, candidate_id, skill_bank, skill_variant, skill_type_override):
    env_kwargs = normalize_env_kwargs(row)
    skill_type = skill_type_override or env_kwargs.get("skill_type") or row.get("skill_type") or "direct_retrieval"
    skill_context = select_skill(skill_bank, skill_type, skill_variant)
    question = env_kwargs["question"]
    return {
        "trace_id": str(uuid.uuid4()),
        "source_index": int(source_index),
        "candidate_id": int(candidate_id),
        "question": question,
        "ground_truth": to_target_list(env_kwargs["ground_truth"]),
        "data_source": env_kwargs.get("data_source", "searchqa"),
        "skill_type": skill_type,
        "skill_variant": skill_variant,
        "skill": skill_context.strip(),
        "messages": build_messages(question, skill_context),
        "steps": [],
        "done": False,
        "success": False,
        "final_reward": 0.0,
        "answer": None,
        "invalid_reason": None,
    }


def parse_tool_call(response_json):
    if response_json.get("_request_error"):
        return {"action": "invalid", "content": "", "tool_calls": [], "error": response_json["_request_error"]}
    choices = response_json.get("choices", [])
    if not choices:
        return {"action": "invalid", "content": "", "tool_calls": [], "error": "No choices returned."}
    message = choices[0].get("message", {}) or {}
    tool_calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    if tool_calls:
        return {"action": "tool", "content": content, "tool_calls": tool_calls, "error": None}
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        content = "".join(text_parts)
    content = str(content).strip()
    if content:
        return {"action": "answer", "content": content, "tool_calls": [], "error": None}
    return {"action": "invalid", "content": "", "tool_calls": [], "error": "Empty assistant message."}


def compact_record(state):
    return {
        "trace_id": state["trace_id"],
        "source_index": state["source_index"],
        "candidate_id": state["candidate_id"],
        "data_source": state["data_source"],
        "skill_type": state["skill_type"],
        "skill_variant": state["skill_variant"],
        "skill": state["skill"],
        "question": state["question"],
        "ground_truth": state["ground_truth"],
        "search_process": state["steps"],
        "answer": state["answer"],
        "teacher_success": bool(state["success"]),
        "final_reward": state["final_reward"],
        "messages": state["messages"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skill_dir", default="skills/search")
    parser.add_argument("--skill_variant", choices=["general_only", "general_plus_selected", "selected_only"], default="general_plus_selected")
    parser.add_argument("--skill_type_override", default=None)
    parser.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--vllm_base_url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--vllm_api_key", default="EMPTY")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--request_workers", type=int, default=24)
    parser.add_argument("--retrieval_topk", type=int, default=3)
    parser.add_argument("--retrieval_max_doc_chars", type=int, default=1600)
    parser.add_argument("--retrieval_workers", type=int, default=64)
    parser.add_argument("--stop_after_accepted", type=int, default=0)
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    accepted_path = os.path.join(args.out_dir, "accepted.jsonl")
    all_path = os.path.join(args.out_dir, "all_candidates.jsonl")
    parquet_path = os.path.join(args.out_dir, "train.parquet")

    records = load_records(args.input_path)
    subset = records[args.start :]
    if args.limit and args.limit > 0:
        subset = subset[: args.limit]
    skill_bank = load_skill_bank(args.skill_dir)
    retriever = HttpRetriever(
        args.search_url,
        topk=args.retrieval_topk,
        max_doc_chars=args.retrieval_max_doc_chars,
    )
    llm = VLLMToolClient(
        base_url=args.vllm_base_url,
        model_name=args.model_name or args.model_path,
        api_key=args.vllm_api_key,
    )

    accepted_count = 0
    all_count = 0
    start_time = time.time()
    mode = "a" if args.append else "w"

    with open(accepted_path, mode, encoding="utf-8") as accepted_f, open(all_path, mode, encoding="utf-8") as all_f:
        for base_start in range(0, len(subset), args.batch_size):
            if args.stop_after_accepted and accepted_count >= args.stop_after_accepted:
                break

            batch_rows = subset[base_start : base_start + args.batch_size]
            states = []
            for offset, row in enumerate(batch_rows):
                source_index = args.start + base_start + offset
                for candidate_id in range(args.num_candidates):
                    states.append(
                        make_state(
                            row=row,
                            source_index=source_index,
                            candidate_id=candidate_id,
                            skill_bank=skill_bank,
                            skill_variant=args.skill_variant,
                            skill_type_override=args.skill_type_override,
                        )
                    )

            successful_sources = set()
            for step_id in range(1, args.max_steps + 1):
                active = [s for s in states if not s["done"] and s["source_index"] not in successful_sources]
                if not active:
                    break

                responses = llm.generate_batch(
                    messages_batch=[state["messages"] for state in active],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_workers=args.request_workers,
                )

                pending_searches = []
                for state, response_json in zip(active, responses):
                    parsed = parse_tool_call(response_json)
                    if parsed["action"] == "tool":
                        tool_call = parsed["tool_calls"][0]
                        function_name = tool_call.get("function", {}).get("name", "")
                        raw_args = tool_call.get("function", {}).get("arguments", "{}")
                        try:
                            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            parsed_args = {}
                        query = str(parsed_args.get("query", "")).strip()
                        if function_name != "search" or not query:
                            state["invalid_reason"] = f"Invalid tool call: {function_name} args={raw_args}"
                            state["done"] = step_id >= args.max_steps
                            state["steps"].append(
                                {
                                    "step_id": step_id,
                                    "action_type": "invalid_tool_call",
                                    "assistant": response_json.get("choices", [{}])[0].get("message", {}),
                                    "observation": state["invalid_reason"],
                                }
                            )
                            continue
                        assistant_message = response_json.get("choices", [{}])[0].get("message", {})
                        state["messages"].append(assistant_message)
                        state["steps"].append(
                            {
                                "step_id": step_id,
                                "action_type": "tool_call",
                                "tool_name": function_name,
                                "search_query": query,
                                "assistant": assistant_message,
                                "observation": None,
                            }
                        )
                        pending_searches.append((state, tool_call.get("id", ""), function_name, query))
                    elif parsed["action"] == "answer":
                        answer = parsed["content"].strip()
                        loose_reward, _ = score_answer(f"<answer>{answer}</answer>", state["ground_truth"])
                        reward, success = strict_score_answer(answer, state["ground_truth"])
                        assistant_message = response_json.get("choices", [{}])[0].get("message", {})
                        state["messages"].append(assistant_message)
                        state["answer"] = answer
                        state["success"] = success
                        state["final_reward"] = reward
                        state["done"] = True
                        state["steps"].append(
                            {
                                "step_id": step_id,
                                "action_type": "answer",
                                "assistant": assistant_message,
                                "answer": answer,
                                "reward": reward,
                                "loose_reward": loose_reward,
                            }
                        )
                        if success:
                            successful_sources.add(state["source_index"])
                    else:
                        state["invalid_reason"] = parsed["error"] or "Invalid response."
                        state["done"] = step_id >= args.max_steps
                        state["steps"].append(
                            {
                                "step_id": step_id,
                                "action_type": "invalid",
                                "assistant": response_json.get("choices", [{}])[0].get("message", {}),
                                "observation": state["invalid_reason"],
                            }
                        )

                kept_searches = [(s, tool_call_id, tool_name, q) for s, tool_call_id, tool_name, q in pending_searches if s["source_index"] not in successful_sources]
                observations = retriever.batch_search([query for _, _, _, query in kept_searches], max_workers=args.retrieval_workers)
                for (state, tool_call_id, tool_name, query), observation in zip(kept_searches, observations):
                    state["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": observation,
                        }
                    )
                    state["steps"][-1]["observation"] = observation

            best_by_source = {}
            for state in states:
                record = compact_record(state)
                all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_count += 1
                has_tool = any(step["action_type"] == "tool_call" for step in state["steps"])
                has_invalid = any(step["action_type"].startswith("invalid") for step in state["steps"])
                if state["success"] and state["answer"] and has_tool and not has_invalid:
                    best_by_source.setdefault(state["source_index"], record)

            for record in best_by_source.values():
                if args.stop_after_accepted and accepted_count >= args.stop_after_accepted:
                    break
                accepted_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted_count += 1

            accepted_f.flush()
            all_f.flush()
            processed_local = min(base_start + len(batch_rows), len(subset))
            done = args.start + processed_local
            elapsed = max(time.time() - start_time, 1e-6)
            examples_per_sec = processed_local / elapsed
            eta_sec = (len(subset) - processed_local) / examples_per_sec if examples_per_sec > 0 else 0.0
            print(
                f"processed={done}/{args.start + len(subset)} "
                f"accepted={accepted_count} all_candidates={all_count} "
                f"examples_per_sec={examples_per_sec:.3f} eta_min={eta_sec / 60:.1f}",
                flush=True,
            )

            if args.stop_after_accepted and accepted_count >= args.stop_after_accepted:
                break

    accepted_records = load_records(accepted_path)
    if args.stop_after_accepted and len(accepted_records) > args.stop_after_accepted:
        accepted_records = accepted_records[: args.stop_after_accepted]
        with open(accepted_path, "w", encoding="utf-8") as f:
            for row in accepted_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(accepted_records).to_parquet(parquet_path, index=False)
    print(f"saved_jsonl={accepted_path} rows={len(accepted_records)}")
    print(f"saved_parquet={parquet_path} rows={len(accepted_records)}")


if __name__ == "__main__":
    main()
