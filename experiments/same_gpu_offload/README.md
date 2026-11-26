# Same-GPU comparison with offloading systems

**Question.** ShadowKV and RetroInfer keep the full KV cache in host memory and fetch a sparse subset at every
decode step. On one GPU and with the same inputs, how do they compare with the tiered cache in PCIe traffic, host
memory, latency and accuracy?

**Protocol**
- **Model and data:** Llama-3.1-8B-Instruct, about 130K tokens, 5 RULER tasks (vt, cwe, fwe, qa_1, qa_2) x 50
  items. Every arm gets the same input ids.
- **Competitors:** ShadowKV and RetroInfer from their official code ([`opx_skv.py`](opx_skv.py),
  [`opx_retro.py`](opx_retro.py)), each at three budgets. ShadowKV also runs in its official 2048 configuration.
- **Ours:** the CUDA-graph engine with exact-size pinned host memory ([`opx_rp_run.py`](opx_rp_run.py)), widths
  `h32`, `h33`, `h43` and `h54`.
- **Resources:** measured for all arms on the same RTX 4090 while no other process used the machine:
  - PCIe bytes per decode step (NVML counters with the idle traffic subtracted, [`pcie_meter.py`](../../tieredkv/engine/pcie_meter.py));
  - pinned host memory;
  - step latency (first 4 items of each task).
- **Accuracy:** each system in its own software environment; ours and full precision on an RTX PRO 5000.
- **Analysis:** paired per item with bootstrap intervals and no pass / fail thresholds ([`opx_analyze.py`](opx_analyze.py)).

**Results** ([`results/opx_numbers.txt`](results/opx_numbers.txt))

| Arm | PCIe per step (median) | Decode ms per step (median / p99) | RULER score |
|---|---|---|---|
| Ours `h32` / `h33` / `h43` / `h54` | 5.99 / 6.45 / 6.60 / 7.64 MB | 29.1-29.2 / 100-102 | 54.3 (`h32`), 53.7 (`h54`) |
| RetroInfer 0.009 / 0.018 / 0.036 | 13.8 / 39.4 / 156 MB | 64.6-73.8 / 89-99 | 53.3 (0.018 and 0.036) |
| ShadowKV 1024 / 2048 / 4096 | 34.0 / 59.9 / 103 MB | 24.9-33.7 / 38.5-40.1 | 50.2 (2048) |
| Full precision | - | - | 51.4 |

- **PCIe:** `h54` moves 7-22% of ShadowKV's bytes per step and 5-55% of RetroInfer's.
- **Pinned host memory:** 4.13 GiB for `h54`, against 14.0 GiB (ShadowKV 2048) and 14.45 GiB (RetroInfer 0.018),
  about 30%. Counted as tensor sizes instead, the ratios are 0.65 and 0.32. PyTorch's pinned allocator rounds every
  block up to a power of two, which inflates the competitors' pinned memory.
- **Accuracy, paired per item:**
  - 3.3-4.3 points above ShadowKV 1024 / 2048 / 4096, with intervals excluding 0;
  - within RetroInfer's interval (differences of +0.4 to +2.7, with intervals including 0);
  - not below full precision.
- **Against full precision:** the tiered cache scores 2.3-3.0 points higher. The gap comes from a few items that
  flip in both directions (`h54`: 18 better, 8 worse, 224 equal), so it is not evidence that quantization improves
  accuracy.
- **Latency:** the median matches ShadowKV and is less than half of RetroInfer's. The p99 is about 2.5x ShadowKV's. It
  comes from the recent-window refresh every 128 tokens and the CUDA-graph recapture that follows, together about
  3.1 ms per token on average.
- **Scope:** one model, five tasks, one length. The accuracy of the competitors was measured on a different GPU and
  software stack.

[`opx_kernel_test.py`](opx_kernel_test.py) checks the engine's kernel variant bit for bit against the kernels it
was derived from, on synthetic data. The checks cover fixed row stride, fused records and page sealing, for codecs
`h54`, `h43`, `h33` and `h32`.

Original invocation (run-machine layout, see [`docs/reproduction.md`](../../docs/reproduction.md)):
`bash opx_formal_star.sh acc|calib|tim <gpu>` for the competitors and timing, `bash opx_formal_p5.sh <phase>
<gpu> [widths]` for ours and full precision, then `python opx_analyze.py`.
