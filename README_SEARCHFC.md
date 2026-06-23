# SkillZero Search Function-Call Toolkit

This release package contains the code needed for:

- function-call style teacher data generation
- SearchQA / NQ / HotpotQA skill prompting and skill selection
- LoRA SFT training
- vLLM + FAISS GPU evaluation with Skill0-style metrics

## 1. Clone and bootstrap

```bash
git clone <your-repo-url>
cd SkillZero-searchfc-release
bash scripts/setup_searchfc_envs.sh
```

All runtime paths are controlled by `scripts/searchfc_runtime_env.sh`.

Default layout after clone:

```text
repo/
  workspace/
    miniconda3/
    models/
    data/
    checkpoints/
    logs/
    merged_models/
    served_models/
```

## 2. Required external assets

Put these assets under `workspace/` or override them with env vars:

- `workspace/models/Qwen3.5-4B`
- `workspace/data/searchR1/e5_Flat.index`
- `workspace/data/searchR1/wiki-18.jsonl`
- raw train/test parquet files for NQ + HotpotQA if you want to regenerate the 10k/1k splits

## 3. Core code

Function-call data generation:

- `examples/data_preprocess/generate_qwen35_teacher_sft_vllm_fc.py`
- `examples/data_preprocess/prepare_nq_hotpot_teacher_inputs.py`
- `examples/data_preprocess/prepare_qwen35_fc_sft_variants.py`

Skill invocation and evaluation:

- `examples/eval_searchqa_vllm_fc.py`
- `skills/search/*.md`
- `examples/search/retriever/retrieval_server.py`
- `examples/summarize_searchqa_skill0_original_metric.py`
- `examples/summarize_searchqa_search_rate.py`

LoRA training and merge:

- `scripts/train_nq_hotpot_qwen35_fc_lora_sweep_2gpu.sh`
- `examples/merge_searchqa_lora.py`

End-to-end scripts:

- `scripts/run_nq_hotpot_qwen35_teacher_sft_2gpu.sh`
- `scripts/eval_searchqa_qwen35_nq8_plus_base_test1000_2gpu_sharded.sh`
- `scripts/eval_searchqa_qwen35_hotpot8_test1000_2gpu_sharded.sh`

## 4. Typical workflow

Prepare 10k/1k NQ + HotpotQA splits:

```bash
bash scripts/run_nq_hotpot_qwen35_teacher_sft_2gpu.sh prepare_inputs
```

Generate function-call teacher data with vLLM:

```bash
bash scripts/run_nq_hotpot_qwen35_teacher_sft_2gpu.sh start
```

Train LoRA sweep:

```bash
bash scripts/train_nq_hotpot_qwen35_fc_lora_sweep_2gpu.sh
```

Evaluate on NQ:

```bash
bash scripts/eval_searchqa_qwen35_nq8_plus_base_test1000_2gpu_sharded.sh
```

Evaluate on HotpotQA:

```bash
bash scripts/eval_searchqa_qwen35_hotpot8_test1000_2gpu_sharded.sh
```

## 5. Notes

- These scripts assume 2 GPUs for the main training/eval path.
- FAISS retrieval is configured to shard on both GPUs.
- vLLM is required for the released function-call pipeline.
