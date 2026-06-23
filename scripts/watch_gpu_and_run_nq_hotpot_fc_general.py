import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import requests


REPO_ROOT = Path("/workspace/limingxin/SkillZero")
PYTHON = "/workspace/limingxin/miniconda3/envs/skillzero/bin/python"
VLLM_BIN = "/workspace/limingxin/miniconda3/envs/vllmcleanpy310/bin/vllm"
MODEL_PATH = "/workspace/limingxin/models/Qwen3.5-4B"
LOG_DIR = Path("/workspace/limingxin/logs/nq_hotpot_fc_general_queue")
QUEUE_LOG = LOG_DIR / "watcher.log"
API_KEY = "EMPTY"
PROTECTED_PIDS = {"3945872"}

SLOTS = [
    {
        "slot_name": "gpu0_hotpotqa",
        "gpu": "0",
        "wait_pattern": "generate_qwen35_teacher_sft_vllm.py --model_path /workspace/limingxin/models/Qwen3.5-4B --model_name /workspace/limingxin/models/Qwen3.5-4B --input_path /workspace/limingxin/data/nq_hotpot_qwen35_source_inputs_full/hotpotqa_train_full.parquet --out_dir /workspace/limingxin/data/hotpotqa_qwen35_correct_26k_run1",
        "old_vllm_port": 8111,
        "fc_port": 8120,
        "job_name": "hotpotqa_fc_general_top10000",
        "input_path": "/workspace/limingxin/data/nq_hotpot_qwen35_source_inputs_full/hotpotqa_train_full.parquet",
        "out_dir": "/workspace/limingxin/data/hotpotqa_qwen35_fc_general_top10000_run1",
    },
    {
        "slot_name": "gpu1_nq",
        "gpu": "1",
        "wait_pattern": "generate_qwen35_teacher_sft_vllm.py --model_path /workspace/limingxin/models/Qwen3.5-4B --model_name /workspace/limingxin/models/Qwen3.5-4B --input_path /workspace/limingxin/data/nq_hotpot_qwen35_source_inputs_full/nq_train_full.parquet --out_dir /workspace/limingxin/data/nq_qwen35_correct_26k_run1",
        "old_vllm_port": 8110,
        "fc_port": 8121,
        "job_name": "nq_fc_general_top10000",
        "input_path": "/workspace/limingxin/data/nq_hotpot_qwen35_source_inputs_full/nq_train_full.parquet",
        "out_dir": "/workspace/limingxin/data/nq_qwen35_fc_general_top10000_run1",
    },
]


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {message}"
    print(line, flush=True)
    with open(QUEUE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, check=True, capture_output=False):
    return subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        check=check,
        text=True,
        capture_output=capture_output,
    )


def has_matching_process(pattern):
    result = run(["pgrep", "-f", pattern], check=False, capture_output=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def get_matching_pids(pattern):
    result = run(["pgrep", "-f", pattern], check=False, capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return [pid.strip() for pid in result.stdout.splitlines() if pid.strip()]


def wait_for_slot_to_free(slot):
    log(f"{slot['slot_name']}: waiting for previous job to finish.")
    while has_matching_process(slot["wait_pattern"]):
        log(f"{slot['slot_name']}: previous job still running, sleep 300s.")
        time.sleep(300)
    log(f"{slot['slot_name']}: previous job finished.")


def stop_old_vllm(slot):
    log(f"{slot['slot_name']}: stopping old vLLM on port {slot['old_vllm_port']}.")
    pattern = f"vllm serve {MODEL_PATH}.*--port {slot['old_vllm_port']}"
    target_pids = get_matching_pids(pattern)
    for pid in target_pids:
        if pid in PROTECTED_PIDS:
            log(f"{slot['slot_name']}: skip protected pid {pid}.")
            continue
        run(["kill", pid], check=False)
    time.sleep(10)


def start_fc_vllm(slot):
    fc_log = LOG_DIR / f"{slot['slot_name']}_fc_vllm.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = slot["gpu"]
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_SKIP_WARMUP"] = "1"
    log(f"{slot['slot_name']}: starting function-call vLLM on gpu={slot['gpu']} port={slot['fc_port']}.")
    log_file = open(fc_log, "w", encoding="utf-8")
    process = subprocess.Popen(
        [
            VLLM_BIN,
            "serve",
            MODEL_PATH,
            "--host",
            "127.0.0.1",
            "--port",
            str(slot["fc_port"]),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.72",
            "--api-key",
            API_KEY,
            "--trust-remote-code",
            "--enforce-eager",
            "--skip-mm-profiling",
            "--gdn-prefill-backend",
            "triton",
            "--attention-backend",
            "TRITON_ATTN",
            "--mm-encoder-attn-backend",
            "TORCH_SDPA",
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return process


def wait_for_fc_vllm(slot, timeout_sec=600):
    base_url = f"http://127.0.0.1:{slot['fc_port']}/v1"
    log(f"{slot['slot_name']}: waiting for function-call vLLM health.")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            response = requests.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=5,
            )
            if response.ok:
                log(f"{slot['slot_name']}: function-call vLLM is healthy.")
                return base_url
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"{slot['slot_name']}: function-call vLLM failed to become healthy.")


def smoke_test_tools(slot, base_url):
    payload = {
        "model": MODEL_PATH,
        "messages": [
            {"role": "system", "content": "You may either answer directly or call the search function."},
            {"role": "user", "content": "Who wrote Hamlet? Use the search function if needed."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the corpus for evidence.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 128,
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    log(f"{slot['slot_name']}: function-call smoke test passed.")


def run_job(slot, base_url):
    out_dir = Path(slot["out_dir"])
    log_path = LOG_DIR / f"{slot['job_name']}.log"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    log(f"{slot['slot_name']}: starting job {slot['job_name']}.")
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                PYTHON,
                "examples/data_preprocess/generate_qwen35_teacher_sft_vllm_fc.py",
                "--model_path",
                MODEL_PATH,
                "--model_name",
                MODEL_PATH,
                "--input_path",
                slot["input_path"],
                "--out_dir",
                slot["out_dir"],
                "--skill_dir",
                str(REPO_ROOT / "skills/search"),
                "--skill_variant",
                "general_only",
                "--search_url",
                "http://127.0.0.1:8000/retrieve",
                "--vllm_base_url",
                base_url,
                "--vllm_api_key",
                API_KEY,
                "--batch_size",
                "48",
                "--num_candidates",
                "2",
                "--max_steps",
                "5",
                "--max_new_tokens",
                "256",
                "--temperature",
                "0.0",
                "--top_p",
                "1.0",
                "--request_workers",
                "48",
                "--retrieval_topk",
                "3",
                "--retrieval_max_doc_chars",
                "1600",
                "--retrieval_workers",
                "64",
                "--stop_after_accepted",
                "10000",
            ],
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"{slot['slot_name']}: job failed with code {return_code}. See {log_path}")
    log(f"{slot['slot_name']}: finished job {slot['job_name']}.")


def handle_slot(slot):
    wait_for_slot_to_free(slot)
    stop_old_vllm(slot)
    fc_process = start_fc_vllm(slot)
    try:
        base_url = wait_for_fc_vllm(slot)
        smoke_test_tools(slot, base_url)
        run_job(slot, base_url)
        log(f"{slot['slot_name']}: queue completed.")
    finally:
        if fc_process.poll() is None:
            log(f"{slot['slot_name']}: stopping function-call vLLM.")
            fc_process.terminate()
            try:
                fc_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                fc_process.kill()
                fc_process.wait(timeout=30)


def main():
    threads = []
    errors = []

    def wrapped(slot):
        try:
            handle_slot(slot)
        except Exception as exc:
            errors.append((slot["slot_name"], str(exc)))
            log(f"{slot['slot_name']}: watcher failed: {exc}")

    for slot in SLOTS:
        thread = threading.Thread(target=wrapped, args=(slot,), daemon=False)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if errors:
        raise RuntimeError(f"Watcher finished with errors: {errors}")
    log("All queued general-only function-call jobs completed.")


if __name__ == "__main__":
    main()
