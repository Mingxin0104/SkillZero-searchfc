import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.data_preprocess.generate_qwen35_teacher_sft_vllm_fc import (  # noqa: E402
    ANSWER_RULE,
    DEFAULT_SYSTEM_PROMPT,
    HttpRetriever,
    VLLMToolClient,
    load_records,
    load_skill_bank,
    normalize_env_kwargs,
    parse_tool_call,
    select_skill,
    strict_score_answer,
    to_target_list,
)


def build_messages(question, skill_context):
    user_content = f"{skill_context}\n\nQuestion: {question}\n\n{ANSWER_RULE}".strip()
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def make_state(row, source_index, skill_bank, skill_variant, skill_type_override, use_skill):
    env_kwargs = normalize_env_kwargs(row)
    skill_type = skill_type_override or env_kwargs.get("skill_type") or row.get("skill_type") or "direct_retrieval"
    skill_context = ""
    if use_skill:
        skill_context = select_skill(skill_bank, skill_type, skill_variant)
    question = env_kwargs["question"]
    return {
        "trace_id": str(source_index),
        "source_index": int(source_index),
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


def render_output(state):
    chunks = []
    for step in state["steps"]:
        if step["action_type"] == "tool_call":
            query = (step.get("search_query") or "").strip()
            if query:
                chunks.append(f"<search>{query}</search>")
            observation = (step.get("observation") or "").strip()
            if observation:
                chunks.append(f"<information>{observation}</information>")
        elif step["action_type"] == "answer":
            answer = (step.get("answer") or "").strip()
            if answer:
                chunks.append(f"<answer>{answer}</answer>")
        elif step["action_type"].startswith("invalid"):
            observation = (step.get("observation") or "").strip()
            if observation:
                chunks.append(f"<invalid>{observation}</invalid>")
    return "\n".join(chunks).strip()


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skill_dir", default="skills/search")
    parser.add_argument("--skill_variant", choices=["general_only", "general_plus_selected", "selected_only"], default="general_plus_selected")
    parser.add_argument("--skill_type_override", default=None)
    parser.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--vllm_base_url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--vllm_api_key", default="EMPTY")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--request_workers", type=int, default=32)
    parser.add_argument("--retrieval_topk", type=int, default=3)
    parser.add_argument("--retrieval_max_doc_chars", type=int, default=1600)
    parser.add_argument("--retrieval_workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--use_skill", action="store_true")
    args = parser.parse_args()

    records = load_records(args.input_path)
    subset = records[args.start :]
    if args.limit and args.limit > 0:
        subset = subset[: args.limit]

    skill_bank = load_skill_bank(args.skill_dir) if args.use_skill else {"general_skills": ""}
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

    states = [
        make_state(
            row=row,
            source_index=args.start + idx,
            skill_bank=skill_bank,
            skill_variant=args.skill_variant,
            skill_type_override=args.skill_type_override,
            use_skill=args.use_skill,
        )
        for idx, row in enumerate(subset)
    ]

    start_time = time.time()
    for base_start in range(0, len(states), args.batch_size):
        batch_states = states[base_start : base_start + args.batch_size]
        for step_id in range(1, args.max_steps + 1):
            active = [s for s in batch_states if not s["done"]]
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
                        }
                    )
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

            if pending_searches:
                observations = retriever.batch_search([query for _, _, _, query in pending_searches], max_workers=args.retrieval_workers)
                for (state, tool_call_id, tool_name, _query), observation in zip(pending_searches, observations):
                    state["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": observation,
                        }
                    )
                    state["steps"][-1]["observation"] = observation

        done = min(base_start + len(batch_states), len(states))
        elapsed = max(time.time() - start_time, 1e-6)
        eps = done / elapsed
        eta_min = (len(states) - done) / eps / 60 if eps > 0 else 0.0
        print(f"processed={done}/{len(states)} examples_per_sec={eps:.3f} eta_min={eta_min:.1f}", flush=True)

    output_root = Path(args.output_dir) / "test"
    output_root.mkdir(parents=True, exist_ok=True)
    text_rows = []
    traj_rows = []
    for state in states:
        tool_count = sum(1 for step in state["steps"] if step["action_type"] == "tool_call")
        text_rows.append(
            {
                "input": state["question"],
                "output": render_output(state),
                "score": float(state["final_reward"]),
                "step": 0,
                "question": state["question"],
                "ground_truth": state["ground_truth"],
                "prediction": state["answer"],
                "searched": bool(tool_count > 0),
            }
        )
        traj_rows.append(
            {
                "traj_uid": state["trace_id"],
                "data_source": state["data_source"],
                "score": float(state["final_reward"]),
                "tool_callings": float(tool_count),
                "searched": bool(tool_count > 0),
                "is_success": bool(state["success"]),
            }
        )

    write_rows(output_root / "0.jsonl", text_rows)
    write_rows(output_root / "0.trajectory_metrics.jsonl", traj_rows)
    summary = {
        "model_path": args.model_path,
        "input_path": args.input_path,
        "output_dir": str(output_root),
        "total": len(states),
        "success": sum(1 for row in traj_rows if row["is_success"]),
        "success_rate": sum(float(row["score"]) for row in traj_rows) / len(traj_rows) if traj_rows else 0.0,
        "search_rate": sum(1 for row in traj_rows if row["searched"]) / len(traj_rows) if traj_rows else 0.0,
    }
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
