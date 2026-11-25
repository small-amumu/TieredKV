# Base alignment: the GPU base against published 2-bit bases

**Question.** Is the GPU-resident base of the tiered cache (2-bit codes plus metadata, 2.5-3 bits per element) on
par with recent 2-bit KV quantizers at the same GPU bits?

**Protocol**
- **Models:** 18. Per model, the baseline arm with the closest GPU bits is matched to ours.
- **Baselines:** both are re-implemented inside the same harness and checked against their official code.
  - Block-GTQ: [`gtq_q_v3.py`](../../tieredkv/reference/gtq_q_v3.py), aligned with the official code.
  - RotateKV: [`rkv_q.py`](../../tieredkv/reference/rkv_q.py). [`rkv_selftest.py`](rkv_selftest.py) compares it with
    the official code on the CPU.
- **Metric:** teacher-forced output KL to full precision.

**Result**
- **Block-GTQ** ([`results/b1a_analysis.txt`](results/b1a_analysis.txt)):
  - our base / matched Block-GTQ: median 0.912 [0.775, 1.104];
  - lower on 12/18 models, against the 14/18 needed to call it better;
  - family-level median 1.02;
  - not shown to be better, and two models are worse (gemma-2b 1.42, SmolLM3-3B 2.48).
- **RotateKV** ([`results/b1c_analysis.txt`](results/b1c_analysis.txt)): our base / matched RotateKV, median 0.284
  [0.007, 0.454], lower on 18/18 models.
- **For scale:** our base's error is 0.195x that of KIVI-128 at the median.

Original invocation: `bash b1a_chain.sh`, `bash b1c_chain.sh` from the run-machine layout (two GPU lanes each; lane
lists `b1*_lane*.txt`).
