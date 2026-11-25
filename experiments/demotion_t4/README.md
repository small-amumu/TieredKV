# Demoting cold host rows

**Question.** Host memory is the main cost of the tiered cache. If cold residual rows are demoted inside the
forward pass, how much fidelity does that cost?
- **Ways to demote:** re-encode to 3+2 bits from the stored `h54` record (`h32c`), keep keys only, or delete the
  residual.
- **Comparisons:** random or oldest-first demotion at equal bytes, and uniform lower bit widths at nearly equal bytes.

**Protocol**
- **Where demotion happens:** [`t4_q.py`](t4_q.py) applies the demotion schedule inside the replay. Decisions use
  only past attention, block by block. With demotion off, the output is bit-identical to `h54`.
- **A2:** Qwen3-0.6B / 1.7B / 4B, 64 traces each, RTX 5090 D.
- **A2b:** 128 held-out items per model, with uniform arms `h44` / `h53` (136 B) added; RTX PRO 5000 and RTX 5090.
- **Free generation:** Qwen3-0.6B, 32 items ([`t4_gen.py`](t4_gen.py)).

**Result**, held-out items ([`results/a2b_analysis.txt`](results/a2b_analysis.txt)). The main arm demotes 75% of
the rows older than a 1024-token window to `h32c`, chosen by attention score.

| | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|---|---|---|---|
| Host bytes per record (`h54`: 152 B) | 132.1 | 128.4 | 128.7 |
| Reasoning-span KL vs `h54` [95% CI] | 1.05 [1.04, 1.06] | 1.06 [1.04, 1.07] | 1.07 [1.06, 1.09] |
| vs random demotion at equal bytes | 0.94 | 0.92 | 0.83 |
| vs uniform-width frontier at equal bytes | 0.90 | 0.90 | 0.88 |

- **Host bytes:** 13-16% fewer than `h54`.
- **Selection:** choosing the rows to demote by score beats random choice at equal bytes, on every model.
- **Answer span:** the gain does not carry over. The answer-span KL of the demotion arm is about 1.3x that of `h54`
  on all three models.
- **Uniform widths for scale (Qwen3-0.6B):** `h44` (136 B) 1.15x, `h43` (120 B) 1.40x, `h32` (88 B) 3.78x the
  reasoning-span KL of `h54`.
- **Free generation:** 24 / 23 / 20 correct (uncompressed cache / demotion by score / random demotion) on 32 items,
  with no significant difference.

[`t4_selftest.py`](t4_selftest.py) checks the schedule on synthetic data (CPU).

Original invocation: `bash t4_chain_a2.sh qwen3_0p6b qwen3_1p7b qwen3_4b`; held-out items: `t4_a2b_chain.sh`,
`t4_ho_chain.sh`; free generation: `t4_gen_run.sh`.
