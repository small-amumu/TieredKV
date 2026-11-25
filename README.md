<div align="center">

# TieredKV

### Long-context decoding with a low-bit GPU KV cache and selective host corrections

[![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](docs/reproduction.md)
[![pytorch](https://img.shields.io/badge/pytorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](docs/reproduction.md)
[![triton](https://img.shields.io/badge/kernels-Triton-6f42c1)](tieredkv/kernels)
[![vLLM](https://img.shields.io/badge/vLLM-plugin-30A2FF)](integrations/vllm)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

[Library](tieredkv) &nbsp;|&nbsp; [Experiments](experiments) &nbsp;|&nbsp; [vLLM plugin](integrations/vllm)
&nbsp;|&nbsp; [Reproduction](docs/reproduction.md)

</div>

---

At long contexts, the KV cache, not the weights, determines how many tokens fit on a GPU. A 2-bit cache fits, but
its errors hit the rows that attention relies on. Offloading the full cache to host memory keeps it exact, but
moves large portions of it over PCIe at every step.

**TieredKV** splits the cache in two. A low-bit base for every token stays on the GPU: 2-bit codes plus metadata,
2.5-3 bits per element. The quantized residual (exact minus base) sits in host memory. At each decode step:
- the base scores every cached row;
- the top 128 rows fetch their host keys for rescoring;
- the top 64 of those fetch their host values for correction.

Only those residual rows cross PCIe: about 15 KB per layer and KV head per step. The 4 attention sinks and the
recent window stay exact on the GPU.

<p align="center">
  <img src="docs/figures/overview.png" width="92%" alt="TieredKV: score every row on the GPU, refine the top rows from host memory">
</p>

## Highlights

| | Result | Details |
|---|---|---|
| Fidelity at the same GPU memory | Decode KL is 59-81x lower than the 2-bit KIVI base (64/64 traces per model) and 34-87x lower than H2O-style eviction (Qwen3 0.6B-4B). | [fidelity_nab1](experiments/fidelity_nab1) |
| Same PCIe bytes as exact-row fetch | KL is 0.37x that of SpeCache-style fetch on a common base, lower on 16/16 models. | [ladder_l1](experiments/ladder_l1) |
| Which rows to correct | Selecting rows by exact attention (the sparse-fetch rule) is 2.3x worse than using the base's own scores, on 16/16 models. | [ladder_l1](experiments/ladder_l1) |
| Against offloading systems at 130K | On the same RTX 4090: 7-22% of ShadowKV's and 5-55% of RetroInfer's PCIe bytes per step, and about 30% of their pinned host memory. Accuracy, each system in its own environment: 3-4 points above ShadowKV (budgets 1024-4096) and within RetroInfer's interval (Llama-3.1-8B, RULER, 250 items). | [same_gpu_offload](experiments/same_gpu_offload) |
| 130K context, real engine | 71.4 vs. 72.7 at full precision (no difference detected); +8.9 [+1.6, +16.3] points over KIVI-128 on the same machine (Llama-3.1-8B, RULER). | [real_engine](experiments/real_engine) |
| Host memory | Demoting cold host rows by attention score cuts host bytes by 13-16% at 1.05-1.07x the reasoning-span KL of the full store (Qwen3 0.6B-4B). | [demotion_t4](experiments/demotion_t4) |

## How it works

1. **GPU base.** Keys are split by RoPE frequency-pair importance. The important half uses 3 bit and the rest
   1 bit, each after a Hadamard transform. Values use 2 bit per token.
2. **Host residuals.** The difference from the exact cache is Hadamard-transformed and coded per token at 5 bit
   (keys) and 4 bit (values) in pinned host memory (`h54`, 152 B per token and KV head).
3. **Two-stage selection.** The base scores every row. The top M = 128 rows fetch their key residuals for
   rescoring, and the top k = 64 of those fetch their value residuals.
4. **The rest.** Unfetched rows keep their base scores, adjusted by a small logit correction from the stored
   quantization step.
5. **Kernels and engine.** Fused Triton kernels run each step and match the PyTorch reference bit for bit. The
   decode engine seals full pages into the GPU base and host store and runs decode steps in a CUDA graph.

## Repository layout

| Path | Content |
|---|---|
| [`tieredkv/`](tieredkv) | the method: accuracy reference and selection (`reference/`), Triton kernels and byte layout (`kernels/`), decode engine and PCIe meter (`engine/`), evaluation harness (`bench/`) |
| [`experiments/`](experiments) | one directory per experiment: scripts of the formal run, analysis script, results; notation in [`experiments/README.md`](experiments/README.md) |
| [`integrations/vllm/`](integrations/vllm) | vLLM plugin (prototype): in-engine oracle backend, host residual codec, kernel parity checks |
| [`docs/`](docs) | reproduction guide, figure, per-file provenance |

## Quick start

```bash
git clone https://github.com/small-amumu/TieredKV.git && cd TieredKV
source tieredkv/env.sh
python experiments/demotion_t4/t4_selftest.py          # CPU checks, no model needed
python experiments/selected_rows_t1/t1_selftest.py
pip install -e integrations/vllm && pytest integrations/vllm/tests
```

Full runs need a CUDA GPU and the models listed per experiment; see [`docs/reproduction.md`](docs/reproduction.md).

## Status

- **Research code.** Every script is the version behind the reported numbers, with only comments and license
  headers changed ([`docs/provenance.tsv`](docs/provenance.tsv)).
- **Running the scripts:** they assume the original run-machine layout and md5 freeze lists; the reproduction guide
  explains both.
- **vLLM plugin:** a staged prototype.

## License

Released under the [Apache License 2.0](LICENSE). Copyright 2026 Zijie Hu.

Every source file carries an `SPDX-License-Identifier: Apache-2.0` header. You may use, modify and redistribute
the code under the terms of the license. Redistributions must keep the copyright and license notices and state
any changes. If you use this code or build on the method, please cite this repository.

## Citation

```bibtex
@software{tieredkv2026,
  title  = {TieredKV: long-context decoding with a low-bit GPU KV cache and selective host corrections},
  author = {Hu, Zijie},
  year   = {2026},
  url    = {https://github.com/small-amumu/TieredKV}
}
```
