# Real decode engine

The engine in [`tieredkv/engine`](../../tieredkv/engine) runs the method end to end:
- the GPU base is sealed page by page;
- host residuals live in pinned host memory;
- each decode step runs the deployed Triton kernels.

These rounds check it against the accuracy reference and evaluate it at long context.

**R1: engine vs reference at 32K** ([`results/r1_verdict.json`](results/r1_verdict.json))
- **Setup:** LLaMA-2-7B-32K, 6 RULER tasks at 32768 tokens, 10 items each, teacher-forced against the gold answer.
- **`h54` results:**
  - argmax agreement with the reference: 99.7%;
  - gold-answer NLL difference: -0.0009 [-0.0043, +0.0024];
  - KL difference: [-1.5e-3, +2e-4].
- **Criterion:** the KL interval had to lie inside +-1e-3, and it does not. The interval only reaches past the band
  on the side where the engine's KL is lower than the reference's.
- **Sanity check:** the engine is far better than KIVI-128 (KL difference -0.273 [-0.293, -0.253]).

**R3: 130K on RULER** ([`results/r3_verdict.json`](results/r3_verdict.json))
- **Setup:** Llama-3.1-8B-Instruct, about 130K tokens, tasks vt / fwe / qa_1, 20 items each, free generation. Every
  generation ran to the task's maximum generation length.

| Arm | Average score (vt / fwe / qa_1) |
|---|---|
| Full precision | 72.7 |
| **Tiered cache, engine, `h54`** | **71.4** (81.0 / 78.3 / 55.0) |
| KIVI-128, same machine | 62.6 |

- **Against full precision:** -1.2 [-4.0, +1.0]. No difference was detected, but this does not show equivalence:
  only 4 items differ in score.
- **Against KIVI-128 on the same machine:** +8.9 [+1.6, +16.3].
- **Engine vs simulation:** the engine and the simulation give the same score on 57 of 60 items.
- **System numbers** (Python layer-by-layer path, before the CUDA-graph engine): decode median 31.4 ms per token,
  GPU peak 25.4 GiB, host store 4.4-4.7 GiB.

[`rp2_tf.py`](rp2_tf.py) is the teacher-forced driver of these rounds. [`tests/`](tests) holds
- the chunked-prefill equality test (GPU);
- a synthetic test of the verdict script, which needs the md5 freeze file of the original run.

Original invocation (run-machine layout, see [`docs/reproduction.md`](../../docs/reproduction.md)):
`bash r1_driver.sh <gpu> a|b formal`, then `python r1_verdict.py`. For R3, `bash r3_driver.sh <gpu> vt,fwe,qa_1
formal`, run by `r3_chain.sh` after the drift check `r3_fpdrift.sh` and the smoke check, then `python r3_verdict.py`.
