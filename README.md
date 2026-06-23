# SkillZero-searchfc

This repository is the projectized release of our SearchQA function-call pipeline. It focuses on one concrete line of work rather than the original upstream paper homepage:

- function-call style teacher data generation
- search skill injection and skill selection
- Qwen3.5-4B LoRA SFT training
- vLLM + FAISS GPU retrieval evaluation
- Skill0-style metric summarization for SearchQA / NQ / HotpotQA

## Project Scope

The main target of this repo is the search agent pipeline around `Qwen3.5-4B`:

- build NQ / HotpotQA teacher inputs
- generate function-call trajectories with `vLLM`
- build `with_skill` and `no_skill` SFT datasets
- train LoRA models with different ranks
- evaluate with the same SearchQA-style interaction and metric definitions used in Skill0

This repo is not intended to be a generic landing page for all upstream `Skill0` work. The root README now documents only this release branch and the experiments actually run here.

## Main Changes in This Release

Compared with the original upstream repo, this branch adds and consolidates:

- function-call data generation scripts for Qwen3.5-4B
- NQ / HotpotQA dataset preparation scripts
- LoRA sweep training scripts for `r=8/16/32/64`
- vLLM-based evaluation scripts for `base + 8 LoRA` models
- sharded FAISS GPU retrieval support
- experiment result summaries committed into the repo
- a portable runtime layout under `workspace/`

Key entry files:

- `examples/data_preprocess/generate_qwen35_teacher_sft_vllm_fc.py`
- `examples/data_preprocess/prepare_nq_hotpot_teacher_inputs.py`
- `examples/data_preprocess/prepare_qwen35_fc_sft_variants.py`
- `examples/eval_searchqa_vllm_fc.py`
- `examples/search/retriever/retrieval_server.py`
- `examples/merge_searchqa_lora.py`
- `scripts/train_nq_hotpot_qwen35_fc_lora_sweep_2gpu.sh`
- `scripts/run_nq_hotpot_qwen35_teacher_sft_2gpu.sh`
- `scripts/eval_searchqa_qwen35_nq8_plus_base_test1000_2gpu_sharded.sh`
- `scripts/eval_searchqa_qwen35_hotpot8_test1000_2gpu_sharded.sh`

## Repository Layout

```text
.
├── examples/
├── scripts/
├── skills/search/
├── experiment_results/
│   ├── reports/
│   ├── nq_test1000/
│   └── hotpotqa_test1000/
└── workspace/
```

Recommended runtime layout after clone:

```text
workspace/
├── miniconda3/
├── models/
├── data/
├── checkpoints/
├── logs/
├── merged_models/
└── served_models/
```

## Environment Setup

The repo includes a simple bootstrap script:

```bash
bash scripts/setup_searchfc_envs.sh
```

Runtime path defaults are centralized in:

```bash
scripts/searchfc_runtime_env.sh
```

You will still need to provide the external assets yourself:

- `workspace/models/Qwen3.5-4B`
- `workspace/data/searchR1/e5_Flat.index`
- `workspace/data/searchR1/wiki-18.jsonl`
- raw NQ / HotpotQA parquet sources if you want to regenerate data

## Typical Workflow

Prepare NQ / HotpotQA inputs:

```bash
bash scripts/run_nq_hotpot_qwen35_teacher_sft_2gpu.sh prepare_inputs
```

Generate function-call teacher data:

```bash
bash scripts/run_nq_hotpot_qwen35_teacher_sft_2gpu.sh start
```

Train LoRA sweep:

```bash
bash scripts/train_nq_hotpot_qwen35_fc_lora_sweep_2gpu.sh
```

Evaluate NQ:

```bash
bash scripts/eval_searchqa_qwen35_nq8_plus_base_test1000_2gpu_sharded.sh
```

Evaluate HotpotQA:

```bash
bash scripts/eval_searchqa_qwen35_hotpot8_test1000_2gpu_sharded.sh
```

## Latest Experiment Results

The latest committed results are under:

- `experiment_results/reports/qwen35_nq_hotpot_eval_summary_2026-06-23.md`
- `experiment_results/nq_test1000/summary.json`
- `experiment_results/hotpotqa_test1000/summary.json`

### Combined Result Table

| Config | NQ acc | NQ success_rate | NQ search_rate | HotpotQA acc | HotpotQA success_rate | HotpotQA search_rate |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen3.5-4B base` | 0.367 | 0.367 | 0.963 | 0.407 | 0.407 | 1.000 |
| `with_skill r8` | 0.371 | 0.371 | 0.970 | 0.407 | 0.407 | 1.000 |
| `no_skill r8` | 0.372 | 0.372 | 0.973 | 0.407 | 0.407 | 1.000 |
| `with_skill r16` | 0.378 | 0.378 | 0.974 | 0.408 | 0.408 | 1.000 |
| `no_skill r16` | 0.377 | 0.377 | 0.974 | 0.411 | 0.411 | 1.000 |
| `with_skill r32` | 0.377 | 0.377 | 0.974 | 0.408 | 0.408 | 1.000 |
| `no_skill r32` | 0.378 | 0.378 | 0.974 | 0.405 | 0.405 | 1.000 |
| `with_skill r64` | 0.375 | 0.375 | 0.974 | 0.409 | 0.409 | 1.000 |
| `no_skill r64` | 0.374 | 0.374 | 0.974 | 0.408 | 0.408 | 1.000 |

### Short Takeaways

- On `NQ`, LoRA gives a small gain over base, with the best result at `37.8%`.
- On `HotpotQA`, improvements are marginal; the best result is `41.1%`.
- `with_skill` and `no_skill` are very close in this 1000-example evaluation.
- Increasing rank from `8` to `64` does not produce a stable monotonic improvement.

## Result Files

Per-run metric files are committed for both datasets:

- `*.skill0_original_metric.json`
- `*.trajectory_metric.json`

These are stored in:

- `experiment_results/nq_test1000/`
- `experiment_results/hotpotqa_test1000/`

## Notes

- The main scripts assume 2 GPUs.
- FAISS retrieval is configured to shard over both GPUs.
- The released function-call path uses `vLLM` rather than plain `transformers` inference.
- Large raw logs, generations, checkpoints, and served model artifacts are intentionally not committed.
