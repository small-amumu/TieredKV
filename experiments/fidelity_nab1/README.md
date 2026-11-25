# Fidelity at matched GPU memory

**Question.** At the same GPU memory, how close is the tiered cache to the 16-bit model, compared with a 2-bit
cache alone and with an eviction cache?

**Protocol.** Matched-memory replay in the style of *Not All Bits Are Equal* (ICLR 2026).
- **Traces:** for Qwen3-0.6B, 1.7B and 4B, the 16-bit model generates 64 MATH-500 reasoning traces (thinking on).
- **Replay:** every arm replays the same traces teacher-forced.
- **Metric:** per-token KL to the 16-bit model, over the decode span and over the answer span.

| Arm | GPU memory | Host memory |
|---|---|---|
| `kivi128`: 2-bit KIVI base alone | 2.52-2.56 bit / element (per model) | none |
| `kivi128k64_x2s128_j50_h54`: the tiered cache on the same base | same | 4.75 bit / element residuals |
| `h2o`: H2O-style eviction with the same peak GPU memory | same peak | none |

**Result** ([`results/analysis_nab1.txt`](results/analysis_nab1.txt)).
- **Against the 2-bit base:** decode KL is lower in 64/64 traces for every model, 59-81x lower at the median. On
  Qwen3-0.6B the median is 0.0020 vs 0.119.
- **Against eviction:** decode KL is 34-87x lower at the median, and lower in every trace where the eviction budget
  held (58/58 for Qwen3-0.6B, 64/64 for 1.7B and 4B).
- **Caveat:** the eviction arm is an H2O-style proxy, not an official implementation of a newer eviction method.

| File | Role |
|---|---|
| `nab1_gen.py` | generates the 16-bit traces |
| `nab1_lab.py` | replays every arm on the traces |
| `nab1_evict.py` | eviction arm |
| `nab1_analyze.py` | paired ratios and sign tests |
| `nab1_chain.sh` | one GPU, models in order: generation, then replay |
| `sim_throughput_nab.py` | analytic throughput bound under a fixed VRAM limit (memory-bound decode, fetch serialized or overlapped) |

Original invocation (in this directory, with a `logs/` directory; see [`docs/reproduction.md`](../../docs/reproduction.md)):
`bash nab1_chain.sh <gpu> qwen3_0p6b qwen3_1p7b qwen3_4b`.
