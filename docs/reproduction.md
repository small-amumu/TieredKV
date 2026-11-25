# Reproduction

## Environment

- **Platform:** Linux, an NVIDIA GPU with CUDA 12.x, Python 3.10+, PyTorch 2.x with CUDA, Triton 3.x, transformers.
- **Models:** Hugging Face checkpoints of Qwen3 (0.6B, 1.7B, 4B, 8B), Qwen2.5, Llama-3.x, LLaMA-2-7B-32K, Mistral,
  Gemma, GLM-4, SmolLM, Phi, EXAONE, Falcon3, Granite and Yi. The panel files list them per experiment.

```bash
source tieredkv/env.sh                              # library directories on PYTHONPATH
python experiments/demotion_t4/t4_selftest.py       # CPU, no model
python experiments/selected_rows_t1/t1_selftest.py  # CPU, no model
pip install -e integrations/vllm && pytest integrations/vllm/tests   # CPU
```

## External data and code

| Item | Used by | Source |
|---|---|---|
| RULER tasks at 32K / 130K | `real_engine`, `same_gpu_offload`, `tieredkv/bench` | generated with the official RULER scripts |
| LongBench (16 English tasks) and its `dataset2prompt.json` / `dataset2maxlen.json` | `tieredkv/bench/lb16.py` | the official LongBench repository |
| MATH-500 | `fidelity_nab1`, `demotion_t4`, `selected_rows_t1` | Hugging Face `datasets` |
| Calibration and evaluation texts (`diag_long_text.txt`, `pg19_text.txt`, `qasper_text.txt`, `mfqa_text.txt`) | `tieredkv/reference/cq_lab3.py`, `tieredkv/bench/tb_bench_v39.py` | excerpts from PG-19 and LongBench documents |
| ShadowKV, RetroInfer | `same_gpu_offload` | official repositories, each in its own environment (`opx_retro.py` imports RetroInfer's `config`) |
| RotateKV official code | `base_alignment/rkv_selftest.py` | github.com/ZunhaiSu/RotateKV @ 3055e2b6 |

## How the scripts were run

Only comments and the license header were changed in the scripts, so they still assume the run machines' directory
layout:

| In the scripts | Content | In this repository |
|---|---|---|
| `~/kvt_bench/hc_mac/code/`, `~/kvt_bench/hc_mac/rootcause/` | the library modules, flat | `tieredkv/reference/`, `tieredkv/kernels/`, `tieredkv/engine/`, `tieredkv/bench/` |
| `~/work/<experiment>/`, `/home/user/work/...` | one experiment's scripts and outputs | `experiments/<experiment>/` |
| `/var/tmp/work_panel/<model>` | local model copies | any Hugging Face checkpoint directory |

To rerun, either copy the library modules into that layout, or `source tieredkv/env.sh` and edit the paths at the
top of the chain script.

The chain scripts:
- take a GPU lock and check that the card is idle;
- write an rc line per step, and can be resumed;
- check the md5 of the scripts they launch against a freeze list (`FREEZE.md5`, `MD5SUMS`, `EXPECTED_MD5.json`).
  The freeze lists are not shipped, and they would not match these copies anyway. Regenerate them, or drop the
  check, before running.

## Machines

| Name in the scripts | GPU |
|---|---|
| `STAR` | NVIDIA RTX 4090, 24 GB |
| `pro5000` | 4x NVIDIA RTX PRO 5000, 48 GB |
| `lab05` | NVIDIA RTX 5090 D, 32 GB |
| `hd5090` | 8x NVIDIA RTX 5090, 32 GB |

## Script versions

The experiment records identify each run by the md5 of every script it used. [`provenance.tsv`](provenance.tsv)
lists that original md5 for every file in this repository.

Each experiment fixed its arms, sample, metric and analysis before it ran. Script comments refer to these plans by
file name (`REG_*.md`, `PLAN_*.md`); the plans themselves are not part of this repository.
