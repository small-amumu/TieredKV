# Ablations: where the fidelity comes from

**Question.** Where does the fidelity of [`fidelity_nab1`](../fidelity_nab1) come from: choosing the right rows,
the host residuals, the value correction, or the logit correction?

**Protocol**
- **Data:** the same 64 traces per model as NAB1.
- **One run per model:** NAB1's two arms, three ablations and one 4-bit reference run together in one process on
  one card, so every ratio compares arms of the same run.
- **Order check:** a separate run replays the arms in a seeded random order on eight items, to check for order
  effects.

| Key | Arm | Median KL ratio to the full method | Traces above 1 |
|---|---|---|---|
| `kivi128k64_x2s128_j50_h54` | full method (NAB1) | 1 | - |
| `kivi128k64_x2r128_j50_h54` | content-blind stage-1 candidates (hash of position) | 31-46x | 64/64 |
| `kivi128k64_x2s128_j50_h5n` | key-only correction (no value residuals) | 8-10x | 64/64 |
| `kivi128k64_x2s128_h54` | no logit correction for unfetched rows | 1.05-1.16x | 38-50 of 64 |
| `kivi4b128` | 4-bit KIVI, for scale (not at equal memory) | 0.85-1.23x | - |

Ranges are over Qwen3-0.6B, 1.7B and 4B ([`results/analysis_nab2.txt`](results/analysis_nab2.txt)).

**Reading**
- **Selection:** choosing rows by the base's own scores is a main source of the gain.
- **Values:** correcting the values matters.
- **Logit correction:** a small, mixed effect.

[`nab2_selftest.py`](nab2_selftest.py) checks that the patched code reproduces NAB1's arms exactly before the
ablations run ([`results/selftest_pass.txt`](results/selftest_pass.txt)).

Original invocation: `bash nab2_chain.sh <gpu> qwen3_0p6b qwen3_1p7b qwen3_4b` in this directory.
