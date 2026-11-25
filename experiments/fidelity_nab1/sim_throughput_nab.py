# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Analytic feasibility check for H5 (throughput under a fixed VRAM limit, Not All Bits App. C.1 protocol).

Decode is modelled as memory-bound: one step reads all weights once plus each sequence's GPU KV; our tiered arm also
gathers host residual rows over PCIe. Two bounds for the fetch: serialized with the GPU reads (conservative) and fully
overlapped (optimistic). Compute, dequantization, selection and two-stage overheads are NOT modelled.

Constants (stated so they can be checked):
- Qwen3 shapes: layers / KV heads / head_dim from the model configs; bf16 weights = 2 bytes x params.
- KV bytes per token per arm: FP16 = 2 x 2 B x elems; 2-bit per-channel (g64, fp16 scale) = 2.25 bit; our GPU tier =
  KIVI-128 2-bit base 2.25 bit; eviction keeps min(T, budget) FP16 tokens.
- Our fetch per step per (layer, KV head): 128 key rows x 84 B + 64 value rows x 68 B (B1: x2s128, h54);
  gather time 9 ns per row (RHR-A on the RTX 5090: zero-copy gather 8.5-9.4 ns per 84 B row).
- GPU: RTX 4090 24 GB, 1008 GB/s; RTX PRO 5000 48 GB, 1344 GB/s. 1.5 GB reserved for activations / runtime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int = 128


@dataclass(frozen=True)
class Gpu:
    name: str
    mem_gb: float
    bw_gbs: float


MODELS = [Model("Qwen3-1.7B", 1.72, 28, 8), Model("Qwen3-4B", 4.02, 36, 8), Model("Qwen3-8B", 8.19, 36, 8)]
GPUS = [Gpu("RTX 4090", 24.0, 1008.0), Gpu("RTX PRO 5000", 48.0, 1344.0)]
RESERVE_GB = 1.5
GATHER_NS_PER_ROW = 9.0
EVICT_BUDGET = 4096


def kv_bytes_per_seq(m: Model, arm: str, t: int) -> float:
    elems = m.layers * m.kv_heads * m.head_dim  # per token, per K or V
    if arm == "fp16":
        return 2 * elems * 2 * t
    if arm in ("q2", "ours"):
        return 2 * elems * 2.25 / 8 * t
    if arm == "evict4k":
        return 2 * elems * 2 * min(t, EVICT_BUDGET)
    raise ValueError(arm)


def fetch_ms_per_seq(m: Model) -> float:
    rows = m.layers * m.kv_heads * (128 + 64)
    return rows * GATHER_NS_PER_ROW * 1e-6


def step(m: Model, g: Gpu, arm: str, t: int, overlap: bool) -> tuple[int, float, float]:
    w_gb = 2 * m.params_b
    free = (g.mem_gb - w_gb - RESERVE_GB) * 1e9
    kv = kv_bytes_per_seq(m, arm, t)
    b = int(free // kv)
    if b < 1:
        return 0, float("nan"), float("nan")
    gpu_ms = (w_gb * 1e9 + b * kv) / (g.bw_gbs * 1e9) * 1e3
    fetch = b * fetch_ms_per_seq(m) if arm == "ours" else 0.0
    ms = max(gpu_ms, fetch) if overlap else gpu_ms + fetch
    return b, ms, b / ms * 1e3


def main() -> None:
    arms = ["fp16", "q2", "evict4k", "ours"]
    print("tok/s = batch / step time; ours shown conservative (fetch serialized) / optimistic (overlapped)")
    for g in GPUS:
        for m in MODELS:
            for t in (8192, 16384, 30720):
                cells = []
                for arm in arms:
                    b, ms, tps = step(m, g, arm, t, overlap=False)
                    if arm == "ours":
                        _, ms2, tps2 = step(m, g, arm, t, overlap=True)
                        cells.append(f"ours b={b} {tps:.0f}/{tps2:.0f}")
                    else:
                        cells.append(f"{arm} b={b} {tps:.0f}")
                print(f"{g.name:12s} {m.name:10s} T={t//1024:>2}K | " + " | ".join(cells))


if __name__ == "__main__":
    main()
