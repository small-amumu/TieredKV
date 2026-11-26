# tierkv-vllm: vLLM plugin (prototype)

A vLLM general plugin (`vllm.general_plugins` entry point `tierkv`) that brings the tiered KV cache into vLLM's
serving engine. Development follows a staged plan: each milestone is checked against the research code before the next one starts.

| Milestone | Content | Status |
|---|---|---|
| M0 | `TIERKV_M0` attention backend: an in-engine oracle. Every layer runs the stock attention, then the scored rows are recomputed by the research attention on the request's K / V read back from vLLM's paged cache. It checks paging, positions, GQA layout, layer index, scale and chunked prefill before any kernel is written. | done |
| Codec | `codec.py`: the host residual row codec (Hadamard + per-token integer code with FP16 step and minimum), bit-identical to the research code | done |
| M1 | deployed Triton kernels inside the backend | kernel parity checked (`diag/m1_parity.py`); integration in progress |

## Install and test

```bash
pip install -e integrations/vllm
pytest integrations/vllm/tests      # CPU: codec round trips, paged-cache read-back
```

| File | Environment | What it checks |
|---|---|---|
| `tests/test_codec.py` | CPU | codec layout and round trips |
| `tests/test_m0_paging.py` | CPU | the backend's read-back of vLLM's paged cache returns exactly what was written |
| `diag/codec_selftest.py` | CPU; needs `source tieredkv/env.sh` | the codec against the research implementation (`pkv5_q` + `cq_lab3`) on 1e5 random rows and real residual rows |
| `diag/m1_parity.py` | GPU | deployed kernel (`kalign`) vs the byte-level reference (`cap_attn`) on real model K / V, at lengths that cross page boundaries |
| `diag/vm0_chunk_check.py` | CPU | the research attention is independent of how vLLM chunks the prefill |
| `src/tierkv_vllm/m0_parity.py` | GPU | runs the research attention in Hugging Face and inside vLLM on the same token ids and compares the logits per method |
