# Experiments

Each directory contains:
- the scripts of the formal run (identical to the run versions apart from comments and license headers; see
  [`docs/provenance.tsv`](../docs/provenance.tsv));
- the analysis script;
- the analysis output.

Comparisons are paired per item (or per model) and reported with a bootstrap or rank-test interval.

| Directory | Question | Main result | Hardware |
|---|---|---|---|
| [`fidelity_nab1`](fidelity_nab1) | Fidelity against a 2-bit cache and an eviction cache at the same GPU memory | decode KL 59-81x lower than the 2-bit base (64/64 traces per model), 34-87x lower than eviction | RTX 4090 |
| [`ablation_nab2`](ablation_nab2) | Where the fidelity comes from | content-blind candidates: 31-46x higher KL; key-only correction: 8-10x | RTX 4090 |
| [`ladder_l1`](ladder_l1) | Against SpeCache-style exact-row fetch at the same PCIe bytes | KL 0.37x, 16/16 models; exact-attention selection 2.3x worse | RTX PRO 5000 |
| [`base_alignment`](base_alignment) | Is the GPU base on par with published 2-bit bases | not shown better than Block-GTQ (median 0.91 [0.78, 1.10]); better than RotateKV on 18/18 | RTX PRO 5000 |
| [`real_engine`](real_engine) | Does the real engine reproduce the reference, and how does it score at 130K | 71.4 vs 72.7 (full precision), +8.9 over KIVI-128 | RTX PRO 5000 |
| [`same_gpu_offload`](same_gpu_offload) | PCIe, host memory, latency and accuracy against ShadowKV and RetroInfer | 7-22% of ShadowKV's and 5-55% of RetroInfer's PCIe bytes per step; about 30% of their pinned host memory | RTX 4090, RTX PRO 5000 |
| [`selected_rows_t1`](selected_rows_t1) | Which host rows decoding actually reads | demoting by score loses 1.5-2x (small models) to 3.5-6x (4B) less correction weight than random | RTX 4090 |
| [`demotion_t4`](demotion_t4) | Demoting cold host rows to fewer bits inside the forward pass | 13-16% fewer host bytes at 1.05-1.07x the reasoning-span KL | RTX 5090 D, RTX PRO 5000, RTX 5090 |
| [`read_path`](read_path) | Cost of reading residual rows from host memory, and of correcting on the CPU | 2 MB huge pages: per-row reads 1.9-5x faster on the RTX 4090 host | RTX 4090, RTX 5090 D |
| [`host_memory_model`](host_memory_model) | Host memory vs per-step time, as a roofline / Pareto model | analytic model, calibrated on RTX 4090 profiles | CPU |

## Notation

| Term | Meaning |
|---|---|
| `kivi128` | KIVI-style 2-bit base (keys per channel, values per token) with a 128-token FP16 window and 4 exact sinks |
| `stkm8v8128` | the stack base of the default configuration (keys split by RoPE-pair importance into 3-bit and 1-bit halves, 8-bit metadata) |
| `k64_x2s128` | two-stage fetch: rescore the top 128 rows with their host keys (`x2s128`), then fetch host values for the top 64 (`k64`); `x2r` = content-blind candidates |
| `h54`, `h33`, ... | host residual widths: 5-bit keys and 4-bit values, 3-bit and 3-bit, ...; `h5n` = keys only |
| `_j50`, `_p50` | logit correction of 50% for rows that are not refined, `-` or `+ 0.5 sigma^2 / 2` (sigma^2 from the stored step) |
| `h32c` | `h32` re-encoded from a stored `h54` record (demotion) |
| R1, R3 | real-engine rounds: engine vs reference at 32K, and 130K RULER |
| A2, A2b | demotion runs: 64 traces, then 128 held-out items with uniform-width arms |
| NAB1, NAB2, L1, B1a, B1c, T1, T4, RP, RFP, OPX | short names of the experiments above, used in script names and comments |

How to set up and run: [`docs/reproduction.md`](../docs/reproduction.md).
