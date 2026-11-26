# Residual coding that uses the known base step

**Question.** The base quantizer rounds to the nearest grid point, so each residual lies within half a base step,
and the base step is already stored on the GPU. If the residual is divided by that step and coded on a fixed grid,
no per-token minimum or step has to be stored. How many host bytes does that save at equal fidelity?

**Protocol**
- **Models:** Qwen3-0.6B and Qwen3-4B; 64 held-out reasoning traces per model, teacher-forced replay, paired per
  item against the current `h54` coding (152 B per token and KV head).
- **Metric:** KL of the next-token distribution to the bf16 trace over the reasoning span; ratios are per-item
  geometric means relative to `h54`, with bootstrap 95% intervals.
- **Hardware:** RTX PRO 5000.

**Results**

| Step-normalized coding (key + value bits) | Host bytes vs `h54` | KL ratio, Qwen3-0.6B | KL ratio, Qwen3-4B |
|---|---|---|---|
| 5 + 4 bits | −5% (144 B) | 0.93 [0.91, 0.95] | 0.92 [0.90, 0.94] |
| 4 + 4 bits | −16% (128 B) | 1.00 [0.98, 1.02] | 1.00 [0.98, 1.02] |
| 4 + 3 bits | −26% (112 B) | 1.09 [1.07, 1.12] | 1.09 [1.06, 1.12] |
| 3 + 3 bits | −37% (96 B) | 1.31 [1.27, 1.35] | 1.35 [1.31, 1.40] |
| Previous coding at 4 + 4 bits, for reference | −11% (136 B) | 1.16 [1.13, 1.18] | 1.11 [1.08, 1.13] |

- At 128 B (−16% host memory) no KL difference from `h54` is detected on either model.
- The fixed grid is nested, so a row can later be demoted by releasing its low bit planes, without re-encoding.
- At 128 B, splitting the bits as 5 + 3 instead of 4 + 4 is 3% worse on Qwen3-0.6B and equal on Qwen3-4B; 4 + 4
  is kept.
