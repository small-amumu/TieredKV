# tieredkv: the library

Flat Python modules grouped by role.
- **Imports:** modules import each other by bare name, so run `source tieredkv/env.sh` first. It puts the four
  directories on `PYTHONPATH`.
- **Versions:** every file is the version used by the experiments, with only comments and the license header
  changed. [`docs/provenance.tsv`](../docs/provenance.tsv) gives the original md5 of each file.

## Default configuration

`stkm8v8128k64_x2s128_p50_h54` (parsed by `sel14_q.parse`):

| Part | Setting |
|---|---|
| GPU base | own stack base (2-bit codes plus metadata, 2.5-3 bits per element depending on the model): keys split by RoPE-pair importance, the top half at 3 bit and the rest at 1 bit (mean +- MAD), each half after its own Hadamard transform, e4m3 metadata; values 2 bit per token |
| Selection | stage 1 scores every cached row with the base keys; the top M = 128 rows are rescored with the host keys; the top k = 64 of those get their host values |
| Host store | residual = exact - base, Hadamard over the head dimension, per-token asymmetric integer code with an FP16 step and minimum; key 5 bit, value 4 bit (`h54`, 152 B per token per KV head) |
| Unfetched rows | logit correction `_p50`: + 0.5 sigma^2 / 2, with sigma^2 from the stored quantization step (`_j50` in the KIVI-base experiments applies the opposite sign) |
| Exact rows | 4 attention sinks and the recent window, served from an FP16 copy on the GPU |
| Bytes per decode step | 128 x 84 B (keys) + 64 x 68 B (values) = 15104 B per layer and KV head |

The low tier `h33` (3-bit key and value residuals) uses the same path.

## `reference/`: accuracy reference and selection

Pure PyTorch simulation of the method at the level of stored codes. The accuracy numbers of the experiments come
from this path.

- **Default import chain:** `sel14_q` -> `sel13_q` -> `sel12_q` -> `sel10_q` -> `sel4_q` -> `pkv5_q`.
- **Helpers it uses:** `pkv6_q` ... `pkv11_q`, `s0_cov`, `sel11_q`, `bl_q` and `cq_lab3`.
- **Everything else:** later variants and baselines.

| Module | Role |
|---|---|
| `pkv5_q.py` | quantizers, effective-bit accounting and the shared attention core (KIVI-style per-channel / per-token codes) |
| `cq_lab3.py` | Hadamard transform and the key-error view used by the residual codec |
| `sel4_q.py` | two-stage fetch (`x2s<M>`), sinks, residual window, the attention step `_attend` |
| `sel10_q.py`, `sel12_q.py`, `sel13_q.py` | stack base (`stkm8v8128`), 8-bit metadata, logit correction `_j` / `_p` |
| `sel14_q.py` | host residual store `_h<rk><rv>`: the deployed configuration |
| `sel14_lab.py` | lab driver: teacher-forced KL against full precision on a model panel |
| `sel17_q.py` ... `sel30_q.py` | later variants: tail probes, asynchronous and predicted candidates, predict-verify-fill |
| `pkv6_q.py` ... `pkv11_q.py`, `rc_q.py`, `s0_cov.py`, `sel11_q.py` | earlier design rounds the default inherits from (what to fetch, which keys to refine, bit allocation, rotary-aware centring) |
| `gtq_q_v3.py`, `gtq3_lab.py`, `rkv_q.py`, `bl_q.py` | baselines re-implemented in the same harness: Block-GTQ (aligned with its official code), RotateKV, two sparse-attention methods |
| `v8g.py` | the 8-bit-metadata value format with the stack's group size, used by the replay drivers (NAB1, NAB2, T1, T4) |
| `run_pkv3.sh` | panel runner used by the ladder and base-alignment experiments |

## `kernels/`: Triton kernels and byte layout

| Module | Role |
|---|---|
| `cap_codec.py` | byte layout of the GPU base and of the host residual rows |
| `cap_attn.py` | one decode step computed from the stored bytes only (accuracy reference at the byte level) |
| `kbench4.py`, `kbench5.py`, `kbench7.py` | fused decode-step kernels: base scoring, two-stage merge, FP16 window |
| `kalign.py` | decode kernels aligned bit-for-bit with the accuracy reference |
| `kenc.py`, `kres.py` | fused cache encode; residual-format fetch |
| `kalign_fr.py`, `kalign_opx.py` | fused host residual records; fixed row stride and the CUDA-graph variant used by the engine |
| `make_kalign_opx.py` | generates `kalign_opx.py` from `kalign_fr.py` (every edit asserted) |
| `kbench3.py` | earlier kernel iteration kept for `kbench4`'s imports |

## `engine/`: real decode engine

| Module | Role |
|---|---|
| `rp_engine.py` | engine E1: prefill, sealing of full pages into the GPU base and the pinned host store, decode on the deployed kernels |
| `rp_chunk.py` | chunked exact causal attention for long prefills |
| `rp_run.py` | runs the engine on RULER items |
| `opx_engine_v8.py` | fixed residual row stride, shared scratch buffers, exact-size pinned host memory (`--exact_pin`) |
| `opx_graph.py` | the decode step captured in a CUDA graph |
| `pcie_meter.py` | NVML PCIe byte counter with idle subtraction |

## `bench/`: evaluation harness

| Module | Role |
|---|---|
| `tb_bench_v39.py` | task-level harness (RULER, LongBench) with teacher-forced paired metrics against full precision |
| `lb16.py` | the 16 English LongBench tasks for the harness |
