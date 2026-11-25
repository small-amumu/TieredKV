# Algorithm ladder against SpeCache-style fetch

**Question.** On a common 2-bit base (KIVI-128 with 4 exact sinks), and at about the same PCIe bytes per decode
step:
- how much does each part of the method add?
- how does the whole compare with fetching exact FP16 rows of the top keys (SpeCache-style)?

**Protocol**
- **Models:** 16 models on the main panel ([`l1_main.txt`](l1_main.txt)), plus held-out and new panels.
- **Metric:** teacher-forced output KL to full precision.
- **Statistics:** paired ratios per model, with one-sided sign tests and Holm correction.
- **Runs:** each panel runs through [`run_pkv3.sh`](../../tieredkv/reference/run_pkv3.sh) and
  [`sel14_lab.py`](../../tieredkv/reference/sel14_lab.py).

**Result** ([`results/l1_analysis.txt`](results/l1_analysis.txt), main panel)

| Step | Median ratio [95% CI] | Models lower |
|---|---|---|
| Residual rows instead of FP16 rows, same bytes | 0.523 [0.442, 0.586] | 16/16 |
| Two-stage fetch instead of single-stage, same bytes | 0.810 [0.794, 0.851] | 15/16 |
| Logit correction (50%) vs none | 0.908 [0.890, 0.962] | 15/16 |
| **Full ladder vs SpeCache-style** | **0.374 [0.314, 0.412]** | **16/16** |
| Deployed `h54` (45-49% of the bytes) vs SpeCache-style | 0.513 [0.453, 0.616] | 16/16 |

- **Selection rule:** choosing rows by exact attention, the sparse-fetch rule, instead of by the base's own scores
  makes the error 2.31x larger. The base's own scores are better on 16/16 models.
- **What this shows:** correcting a low-bit cache and fetching a sparse subset are different problems, and their
  best selection rules are opposite.

Original invocation: `bash l1_chain.sh` from the run-machine layout (two GPU lanes; see
[`docs/reproduction.md`](../../docs/reproduction.md)).
