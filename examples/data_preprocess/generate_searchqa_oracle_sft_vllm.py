import argparse
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from examples.data_preprocess.generate_searchqa_skill_traces_local import (
    normalize_env_kwargs,
    to_target_list,
)
from examples.data_preprocess.generate_searchqa_teacher_sft_vllm import (
    DEFAULT_SYSTEM_PROMPT,
    load_skill_bank,
    render_chat_prompt,
    select_skill,
)


ORACLE_QUERY_PROMPT = """You are creating a search trajectory for a supervised search-agent dataset.

You may use the hidden gold answer only to understand what evidence the query must retrieve.
The search query must look like a normal query written by a model that does not know the answer yet.
Do not include the hidden gold answer, aliases of the answer, or answer-only terms in the query.
Use distinctive words, quotes, dates, roles, or constraints from the question instead.

Question: {question}
Hidden gold answer: {answer}

Output exactly one tag:
<search>one concise query that can retrieve evidence supporting the hidden gold answer</search>
"""


def load_records(path):
    path = os.path.expanduser(path)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return pd.read_parquet(path).to_dict(orient="records")


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def evidence_supports_answer(observation, answers):
    obs = normalize_text(observation)
    if not obs:
        return False
    for answer in answers:
        ans = normalize_text(answer)
        if ans and ans in obs:
            return True
    return False


def query_contains_answer(query, answers):
    norm_query = normalize_text(query)
    if not norm_query:
        return True
    for answer in answers:
        norm_answer = normalize_text(answer)
        if norm_answer and norm_answer in norm_query:
            return True
    return False


def parse_search(text):
    match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    if match:
        query = " ".join(match.group(1).strip().split())
        if query:
            return query
    text = re.sub(r"</?search>", "", text).strip()
    return " ".join(text.split()[:32])


class HttpRetriever:
    def __init__(self, search_url, topk=5, timeout=60, max_doc_chars=1800, retries=120, retry_sleep=5):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self.max_doc_chars = max_doc_chars
        self.retries = retries
        self.retry_sleep = retry_sleep

    def search(self, query):
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
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
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                print(
                    f"[retriever-wait] attempt={attempt}/{self.retries} query={query[:80]!r} err={exc}",
                    flush=True,
                )
                time.sleep(self.retry_sleep)
        raise last_error

    def batch_search(self, queries, max_workers=128):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.search, queries))


def messages_from_record(record):
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{record['skill']}\n\nQuestion: {record['question']}"},
    ]
    for step in record["search_process"]:
        messages.append({"role": "assistant", "content": step["assistant"]})
        if step.get("observation"):
            messages.append({"role": "user", "content": step["observation"]})
    return messages


def make_record(row, source_index, skill_bank, query, observation, answer):
    env_kwargs = normalize_env_kwargs(row)
    skill_type = env_kwargs.get("skill_type") or row.get("skill_type") or "direct_retrieval"
    skill = select_skill(skill_bank, skill_type).strip()
    record = {
        "trace_id": str(uuid.uuid4()),
        "source_index": int(source_index),
        "candidate_id": 0,
        "data_source": env_kwargs.get("data_source", "searchqa"),
        "skill_type": skill_type,
        "skill": skill,
        "question": env_kwargs["question"],
        "ground_truth": to_target_list(env_kwargs["ground_truth"]),
        "search_process": [
            {
                "step_id": 1,
                "assistant": f"<search>{query}</search>",
                "action_type": "search",
                "search_query": query,
                "observation": observation,
            },
            {
                "step_id": 2,
                "assistant": f"<answer>{answer}</answer>",
                "action_type": "answer",
            },
        ],
        "answer": answer,
        "teacher_success": True,
        "final_reward": 1.0,
        "oracle_answer_used_for_generation": True,
    }
    record["messages"] = messages_from_record(record)
    return record


def question_only_fallback_queries(question):
    question = " ".join(str(question).split())
    quoted = re.findall(r'"([^"]{4,120})"', question)
    queries = []
    for phrase in quoted:
        queries.append(f'"{phrase}"')
        queries.append(f'"{phrase}" {question.replace(phrase, "").strip()}')
    queries.append(question)
    no_punct = re.sub(r"[^A-Za-z0-9\s'\"]", " ", question)
    no_punct = " ".join(no_punct.split())
    if no_punct and no_punct != question:
        queries.append(no_punct)
    unique = []
    seen = set()
    for query in queries:
        query = " ".join(query.split())
        if query and query not in seen:
            seen.add(query)
            unique.append(query)
    return unique[:4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skill_dir", default="skills/search")
    parser.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.82)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--retrieval_topk", type=int, default=5)
    parser.add_argument("--retrieval_max_doc_chars", type=int, default=1800)
    parser.add_argument("--retrieval_workers", type=int, default=128)
    parser.add_argument("--retrieval_retries", type=int, default=120)
    parser.add_argument("--retrieval_retry_sleep", type=float, default=5.0)
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
        retries=args.retrieval_retries,
        retry_sleep=args.retrieval_retry_sleep,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    accepted_count = 0
    all_count = 0
    start_time = time.time()
    with open(accepted_path, "w", encoding="utf-8") as accepted_f, open(
        all_path, "w", encoding="utf-8"
    ) as all_f:
        for base_start in range(0, len(subset), args.batch_size):
            batch_rows = subset[base_start : base_start + args.batch_size]
            prompts = []
            meta = []
            for offset, row in enumerate(batch_rows):
                env_kwargs = normalize_env_kwargs(row)
                answers = to_target_list(env_kwargs["ground_truth"])
                if not answers:
                    continue
                answer = str(answers[0]).strip()
                if not answer:
                    continue
                prompt = ORACLE_QUERY_PROMPT.format(question=env_kwargs["question"], answer=answer)
                prompts.append(render_chat_prompt(tokenizer, prompt, disable_thinking=True))
                meta.append((row, args.start + base_start + offset, env_kwargs["question"], answers, answer))

            outputs = llm.generate(prompts, sampling_params, use_tqdm=False) if prompts else []
            queries = [parse_search(output.outputs[0].text) for output in outputs]
            observations = retriever.batch_search(queries, max_workers=args.retrieval_workers) if queries else []

            fallback_jobs = []
            batch_candidates = []
            for item, query, observation in zip(meta, queries, observations):
                row, source_index, question, answers, answer = item
                leaks_answer = query_contains_answer(query, answers)
                supported = (not leaks_answer) and evidence_supports_answer(observation, answers)
                candidate = {
                    "source_index": int(source_index),
                    "question": question,
                    "ground_truth": answers,
                    "answer": answer,
                    "query": query,
                    "observation": observation,
                    "supported": supported,
                    "query_leaks_answer": leaks_answer,
                    "fallback_used": False,
                }
                if not supported:
                    for fallback_query in question_only_fallback_queries(question):
                        if not query_contains_answer(fallback_query, answers):
                            fallback_jobs.append((candidate, fallback_query))
                batch_candidates.append((row, candidate))

            if fallback_jobs:
                fallback_obs = retriever.batch_search(
                    [q for _, q in fallback_jobs],
                    max_workers=args.retrieval_workers,
                )
                for (candidate, fallback_query), observation in zip(fallback_jobs, fallback_obs):
                    if (
                        not candidate["supported"]
                        and not query_contains_answer(fallback_query, candidate["ground_truth"])
                        and evidence_supports_answer(observation, candidate["ground_truth"])
                    ):
                        candidate["query"] = fallback_query
                        candidate["observation"] = observation
                        candidate["supported"] = True
                        candidate["query_leaks_answer"] = False
                        candidate["fallback_used"] = True

            for row, candidate in batch_candidates:
                all_f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                all_count += 1
                if not candidate["supported"]:
                    continue
                record = make_record(
                    row,
                    candidate["source_index"],
                    skill_bank,
                    candidate["query"],
                    candidate["observation"],
                    candidate["answer"],
                )
                accepted_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted_count += 1

            accepted_f.flush()
            all_f.flush()
            processed_local = min(base_start + len(batch_rows), len(subset))
            done = args.start + processed_local
            elapsed = max(time.time() - start_time, 1e-6)
            examples_per_sec = processed_local / elapsed
            eta_sec = (len(subset) - processed_local) / examples_per_sec if examples_per_sec else 0.0
            print(
                f"processed={done}/{args.start + len(subset)} accepted={accepted_count} "
                f"all_candidates={all_count} examples_per_sec={examples_per_sec:.3f} eta_min={eta_sec / 60:.1f}",
                flush=True,
            )

    accepted_records = load_records(accepted_path)
    pd.DataFrame(accepted_records).to_parquet(parquet_path, index=False)
    print(f"saved_jsonl={accepted_path} rows={len(accepted_records)}")
    print(f"saved_parquet={parquet_path} rows={len(accepted_records)}")


if __name__ == "__main__":
    main()
