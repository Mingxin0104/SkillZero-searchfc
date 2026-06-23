import argparse
import json
import os
import re
import uuid

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

SEARCH_TEMPLATE_NO_HIS = """
{skill_context}You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags.

Your response must contain exactly one executable tag: either <search>...</search> or <answer>...</answer>.
"""

SEARCH_TEMPLATE = """
{skill_context}You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history, where <search>...</search> wrapped your past search queries and <information>...</information> wrapped the corresponding search results. History:
{memory_context}

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags.

Your response must contain exactly one executable tag: either <search>...</search> or <answer>...</answer>.
"""


def load_skills(skill_file):
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


def generate_action(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def pooling(last_hidden_state, attention_mask):
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class LocalDenseRetriever:
    def __init__(self, model_path, index_path, corpus_path, topk=3):
        print("loading retriever tokenizer/model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().cuda()
        self.corpus_path = corpus_path
        self.topk = topk
        import faiss

        print("loading faiss index")
        self.index = faiss.read_index(index_path)
        print("building corpus line offsets")
        self.offsets = self._build_offsets(corpus_path)
        print(f"retriever ready: ntotal={self.index.ntotal}, corpus={len(self.offsets)}")

    def _build_offsets(self, corpus_path):
        offsets = []
        offset = 0
        with open(corpus_path, "rb") as f:
            for line in f:
                offsets.append(offset)
                offset += len(line)
        return offsets

    def _load_doc(self, idx):
        idx = int(idx)
        with open(self.corpus_path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        return json.loads(line)

    @torch.no_grad()
    def encode_query(self, query):
        text = f"query: {query}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
        output = self.encoder(**inputs, return_dict=True)
        emb = pooling(output.last_hidden_state, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.detach().cpu().numpy().astype(np.float32)

    def search(self, query):
        emb = self.encode_query(query)
        scores, idxs = self.index.search(emb, k=self.topk)
        docs = [self._load_doc(idx) for idx in idxs[0]]
        parts = []
        for i, doc in enumerate(docs, start=1):
            content = doc["contents"].strip()
            parts.append(f"Doc {i}: {content}")
        return "\n<information>" + "\n".join(parts) + "</information>\n"


class HttpRetriever:
    def __init__(self, search_url, topk=3, timeout=60):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout

    def search(self, query):
        response = requests.post(
            self.search_url,
            json={"query": query, "topk": self.topk, "return_scores": False},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("result", [[]])
        docs = results[0] if results else []
        parts = []
        for i, doc in enumerate(docs, start=1):
            content = doc.get("contents")
            if content is None:
                title = doc.get("title", "")
                text = doc.get("text", "")
                content = f"{title}\n{text}".strip()
            parts.append(f"Doc {i}: {content.strip()}")
        return "\n<information>" + "\n".join(parts) + "</information>\n"


def score_answer(action, ground_truth):
    if "<answer>" not in action or "</answer>" not in action:
        return 0.0, False
    pred = action.split("<answer>", 1)[1].split("</answer>", 1)[0].strip().lower()
    targets = [x.strip().lower() for x in ground_truth if str(x).strip()]
    success = any(pred == tgt or pred in tgt or tgt in pred for tgt in targets)
    return (1.0 if success else 0.0), success


def row_to_env_kwargs(row):
    env_kwargs = dict((row.get("env_kwargs") or {}))
    return {
        "ground_truth": env_kwargs.get("ground_truth"),
        "question": env_kwargs.get("question"),
        "data_source": env_kwargs.get("data_source", row.get("data_source", "searchqa")),
        "skill_type": env_kwargs.get("skill_type", row.get("skill_type")),
    }


def normalize_env_kwargs(row):
    env_kwargs = dict((row.get("env_kwargs") or {}))
    if not env_kwargs and "question" in row:
        return {
            "ground_truth": row.get("ground_truth"),
            "question": row.get("question"),
            "data_source": row.get("data_source", "searchqa"),
            "skill_type": row.get("skill_type"),
        }
    return row_to_env_kwargs(row)


def load_input_records(path):
    path = os.path.expanduser(path)
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--retriever_model_path", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--retriever_backend", choices=["local", "http"], default="local")
    parser.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--skill_file", required=True)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save_all_path", default=None)
    parser.add_argument("--flush_every", type=int, default=1)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    ).cuda().eval()
    print("teacher model ready")

    if args.retriever_backend == "http":
        retriever = HttpRetriever(search_url=args.search_url, topk=3)
        print(f"retriever ready: {args.search_url}")
    else:
        retriever = LocalDenseRetriever(
            model_path=os.path.expanduser(args.retriever_model_path),
            index_path=os.path.expanduser(args.index_path),
            corpus_path=os.path.expanduser(args.corpus_path),
            topk=3,
        )

    skill_context = load_skills(args.skill_file)
    records = load_input_records(args.input_path)
    subset = records[args.start : args.start + args.limit]

    accepted_records = []
    all_records = []
    output_path = os.path.expanduser(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success_f = open(output_path, "w", encoding="utf-8")
    all_f = None
    if args.save_all_path:
        save_all_path = os.path.expanduser(args.save_all_path)
        os.makedirs(os.path.dirname(save_all_path), exist_ok=True)
        all_f = open(save_all_path, "w", encoding="utf-8")

    for idx, row in enumerate(subset):
        env_kwargs = normalize_env_kwargs(row)
        question = env_kwargs["question"]
        ground_truth = to_target_list(env_kwargs["ground_truth"])
        print(f"trace sample {idx + 1}/{len(subset)}: question={question[:120]!r}")

        for candidate_id in range(args.num_candidates):
            print(f" sample {idx + 1} candidate {candidate_id + 1}/{args.num_candidates}")
            history = []
            steps = []
            final_reward = 0.0
            success = False
            final_answer = None

            for step_id in range(1, args.max_steps + 1):
                query = None
                prompt = build_prompt(question, skill_context, history)
                assistant = generate_action(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                print(f"  step {step_id}: action={assistant[:180]!r}")
                valid = bool(re.search(r"<(search|answer)>.*?</\1>", assistant, re.DOTALL))

                if "<search>" in assistant and "</search>" in assistant:
                    query = assistant.split("<search>", 1)[1].split("</search>", 1)[0].strip()
                    observation = retriever.search(query)
                    reward = 0.0
                    done = False
                    history.append(assistant)
                    history.append(observation)
                    action_type = "search"
                elif "<answer>" in assistant and "</answer>" in assistant:
                    observation = ""
                    reward, success = score_answer(assistant, ground_truth)
                    done = True
                    action_type = "answer"
                    final_answer = assistant.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
                else:
                    observation = "Invalid action: response must contain <search>...</search> or <answer>...</answer>."
                    reward = 0.0
                    done = step_id >= args.max_steps
                    action_type = "invalid"
                    history.append(assistant)
                    history.append(observation)

                steps.append(
                    {
                        "step_id": step_id,
                        "assistant": assistant,
                        "action_type": action_type,
                        "search_query": query if action_type == "search" else None,
                        "observation": observation,
                        "reward": reward,
                        "done": done,
                        "is_action_valid": valid,
                    }
                )

                final_reward = reward
                if done:
                    break

            record = {
                "trace_id": str(uuid.uuid4()),
                "source_index": int(args.start + idx),
                "candidate_id": candidate_id,
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
            all_records.append(record)
            if all_f:
                all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if args.flush_every > 0 and len(all_records) % args.flush_every == 0:
                    all_f.flush()
            has_search = any(step["action_type"] == "search" for step in steps)
            invalid_count = sum(step["action_type"] == "invalid" for step in steps)
            if success and final_answer and has_search and invalid_count == 0:
                accepted_records.append(record)
                success_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if args.flush_every > 0 and len(accepted_records) % args.flush_every == 0:
                    success_f.flush()
                print(f" accepted sample={idx + 1} candidate={candidate_id + 1}")
                break
        print(f"processed={idx + 1}, accepted={len(accepted_records)}")

    success_f.close()
    print(f"saved={output_path} rows={len(accepted_records)}")

    if all_f:
        all_f.close()
        print(f"saved_all={save_all_path} rows={len(all_records)}")


if __name__ == "__main__":
    main()
