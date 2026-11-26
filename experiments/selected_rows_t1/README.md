# Selected-row trajectories

**Question.** Over a long decode, which host rows does the method actually read?
- **Demotion:** if cold rows were demoted by the attention they received in an observation window, how much of
  the correction would be lost, compared with demoting the same fraction at random?
- **Locality:** how clustered are the reads in host pages?

**Protocol**
- **Models:** Qwen3-0.6B / 1.7B / 4B on 64 reasoning traces each.
- **What is recorded:** the replay logs the stage-1 and stage-2 selections without changing the computation
  ([`t1_q.py`](t1_q.py)).
- **Demotion policies:** by score, at random, or oldest first, with observation windows of 512 / 1024 / 2048
  tokens and demotion fractions of 75% / 90%.

**Result** ([`results/t1_analysis.txt`](results/t1_analysis.txt)).
- **Lost correction weight:** at the same demotion fraction, demoting by score loses 1/2 to 2/3 (Qwen3-0.6B,
  1.7B) and 1/3.5 to 1/6 (4B) of the correction weight that random demotion loses.
- **Caveat on 4B:** its ratio is also affected by its larger GQA group and by its traces differing from the
  smaller models' traces.
- **Page clustering:** on items longer than 4096 tokens, a touched 16-row page holds 2.2-2.3 stage-1 selections,
  about 1.9x random.

These numbers set the arms of [`demotion_t4`](../demotion_t4) and the page sizes of [`read_path`](../read_path).
[`t1_selftest.py`](t1_selftest.py) and [`t1_aggtest.py`](t1_aggtest.py) check the policies and the aggregation on
synthetic data (CPU).

Original invocation: `bash t1_smoke.sh`, then `bash t1_chain.sh 0 qwen3_0p6b qwen3_1p7b qwen3_4b`.
