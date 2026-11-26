# RULER against full attention, item by item

**Question.** At the same token ids, does the tiered cache change any RULER answer that full attention gets right?

**Protocol**
- **Model and data:** Llama-3.1-8B-Instruct; RULER items 0-9 of each task. Prompt = input + answer prefix, one
  BOS, no chat template, greedy decoding, the RULER generation caps (NIAH 128, VT 30, CWE 120, FWE 50, QA 32),
  RULER string-match scoring.
- **Arms:** our engine (`h54` host residuals; fixed layout, CUDA graph) and vLLM bf16 full attention. Full
  attention runs twice, with prefill chunks of 8192 and 2048 tokens; the difference between those two runs is
  the run-to-run noise of an exact implementation.
- **Pairing:** all arms see identical token ids (checked by an md5 per item); intervals are paired bootstrap 95%.
- **Hardware:** one RTX PRO 5000 per run.

**Results**

| Setting | Ours | Full attention, run 1 / run 2 | Ours − full (run 1) | Ours − full (run 2) |
|---|---|---|---|---|
| 128K (130000 tokens), 8 NIAH tasks, 80 items | 91.88 | 93.12 / 91.88 | −1.25 [−3.75, 0] | 0.00 (identical on 80/80 items) |
| 32K, all 13 tasks, 130 items | 90.74 | 90.64 / 89.87 | +0.10 [−0.31, +0.69] | +0.87 [−0.23, +2.74] |

- **128K:** the only item where ours differs from full-attention run 1 is one NIAH-single-3 item on which the two
  full-attention runs disagree with each other (run 1 right, run 2 wrong; ours wrong). Both full-attention runs
  produce identical token streams on only 44 of 80 items, so this gap is inside the reference's own noise.
  NIAH-multikey-3 scores 50 for all three arms: a model limit at this length, not a compression effect.
- **32K:** no difference from full attention is detected on any of the 13 tasks.
