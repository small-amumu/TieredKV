# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Heavy-hitter KV eviction for teacher-forced replay (NAB1 eviction arm; H2O-style proxy, not R-KV).

Rule, per KV head (GQA: scores summed over the head's query group):
- rows < start (the prompt) attend to every earlier key exactly and add their attention to the accumulated scores;
- at every decode chunk boundary c0 >= start (multiples of `chunk` from 0), if more than `budget` keys before c0 are
  still alive, the lowest-score alive keys are evicted for good, except the first `sinks` keys and the last `window`
  keys before c0, until `budget` remain (or only protected keys remain);
- rows in [c0, c1) attend to the alive keys before c0 plus the causal keys inside [c0, c1); their attention
  probabilities are added to the accumulated scores of the keys they attended.
Keys keep exact values (eviction stores the kept tokens at full precision). `budget` counts kept keys before the
current chunk; the chunk's own keys (fewer than `chunk`) are a transient buffer, as in H2O-style implementations.
"""
from __future__ import annotations

import torch


def _evict(alive: torch.Tensor, acc: torch.Tensor, c0: int, budget: int, sinks: int, window: int) -> None:
    """In place: evict alive keys < c0 beyond `budget`, lowest accumulated score first, protecting sinks / window."""
    kvh, _ = alive.shape
    pos = torch.arange(c0, device=alive.device)
    protect = (pos < sinks) | (pos >= c0 - window)
    for h in range(kvh):
        live = alive[h, :c0]
        excess = int(live.sum()) - budget
        if excess <= 0:
            continue
        cand = live & ~protect
        n_cand = int(cand.sum())
        if n_cand == 0:
            continue
        score = acc[h, :c0].masked_fill(~cand, float("inf"))
        drop = torch.sort(score, stable=True).indices[:min(excess, n_cand)]   # ties: lowest index first
        alive[h, drop] = False


def evict_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, budget: int, start: int = 0,
                 chunk: int = 128, sinks: int = 4, window: int = 128) -> torch.Tensor:
    """q [H, L, d], k / v [KVH, L, d] (float32). Returns [H, L, d]."""
    H, L, d = q.shape
    kvh = k.shape[0]
    assert H % kvh == 0, f"{H} query heads not a multiple of {kvh} KV heads"
    g = H // kvh
    out = torch.empty_like(q)
    alive = torch.ones(kvh, L, dtype=torch.bool, device=q.device)
    acc = torch.zeros(kvh, L, dtype=torch.float32, device=q.device)
    kk = k.repeat_interleave(g, 0)
    vv = v.repeat_interleave(g, 0)
    pos = torch.arange(L, device=q.device)
    for c0 in range(0, L, chunk):
        c1 = min(L, c0 + chunk)
        if c0 >= start:
            _evict(alive, acc, c0, budget, sinks, window)
        rows = pos[c0:c1]
        s = torch.einsum("hqd,hkd->hqk", q[:, c0:c1], kk[:, :c1]) * scale              # [H, r, c1]
        causal = pos[None, :c1] <= rows[:, None]                                         # [r, c1]
        keep = alive[:, :c1].repeat_interleave(g, 0)[:, None, :] | (pos[:c1] >= c0)[None, None, :]
        if c0 < start:                                                                    # prompt rows: exact
            keep = torch.ones_like(keep)
        m = causal[None] & keep
        p = torch.softmax(s.masked_fill(~m, float("-inf")), -1)
        out[:, c0:c1] = torch.einsum("hqk,hkd->hqd", p, vv[:, :c1])
        acc[:, :c1] += p.reshape(kvh, g, c1 - c0, c1).sum((1, 2))
    return out


def evict_reference(q, k, v, scale, budget, start=0, chunk=128, sinks=4, window=128):
    """Row-by-row reference of the same rule (float64 loops), for the self-test."""
    H, L, d = q.shape
    kvh = k.shape[0]
    g = H // kvh
    out = torch.zeros(H, L, d, dtype=torch.float64)
    alive = [[True] * L for _ in range(kvh)]
    acc = [[0.0] * L for _ in range(kvh)]
    for t in range(L):
        c0 = (t // chunk) * chunk
        if t == c0 and c0 >= start:
            for h in range(kvh):
                live = [j for j in range(c0) if alive[h][j]]
                excess = len(live) - budget
                if excess > 0:
                    cand = [j for j in live if not (j < sinks or j >= c0 - window)]
                    cand.sort(key=lambda j: (acc[h][j], j))
                    for j in cand[:excess]:
                        alive[h][j] = False
        for hq in range(H):
            h = hq // g
            keys = [j for j in range(t + 1) if t < start or j >= c0 or alive[h][j]]
            sc = torch.tensor([float(q[hq, t].double() @ k[h, j].double()) * scale for j in keys], dtype=torch.float64)
            p = torch.softmax(sc, 0)
            out[hq, t] = sum(p[n] * v[h, j].double() for n, j in enumerate(keys))
            for n, j in enumerate(keys):
                acc[h][j] += float(p[n])
    return out


def selftest() -> None:
    # (H, kvh, L, d, budget, start, chunk, sinks, window): GQA g = 1 / 2 / 4, eviction on, a chunk straddling start,
    # budget >= L (no eviction), and budget < sinks + window (only protected keys remain)
    cases = [(4, 2, 70, 8, 30, 10, 16, 2, 8), (6, 3, 96, 16, 24, 0, 8, 4, 8), (2, 1, 40, 8, 100, 5, 16, 4, 16),
             (4, 4, 65, 8, 12, 20, 8, 2, 4), (8, 2, 90, 8, 20, 6, 16, 2, 8), (4, 2, 60, 8, 4, 0, 8, 2, 4)]
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        for H, kvh, L, d, budget, start, chunk, sinks, window in cases:
            q, k, v = (torch.randn(H, L, d), torch.randn(kvh, L, d), torch.randn(kvh, L, d))
            a = evict_attend(q, k, v, 0.35, budget, start, chunk, sinks, window).double()
            b = evict_reference(q, k, v, 0.35, budget, start, chunk, sinks, window)
            err = float((a - b).abs().max())
            assert err < 1e-4, f"seed {seed} case {(H, kvh, L, d, budget, start, chunk, sinks, window)}: max error {err}"
            print(f"PASS seed {seed} H{H} kvh{kvh} L{L} budget{budget} start{start} chunk{chunk} sinks{sinks} "
                  f"window{window}: max error {err:.2e}")
    # budget >= L: identical to exact causal attention
    H, kvh, L, d = 4, 2, 50, 8
    q, k, v = torch.randn(H, L, d), torch.randn(kvh, L, d), torch.randn(kvh, L, d)
    a = evict_attend(q, k, v, 0.35, L + 10, 0, 16)
    kk, vv = k.repeat_interleave(2, 0), v.repeat_interleave(2, 0)
    s = torch.einsum("hqd,hkd->hqk", q, kk) * 0.35
    s = s.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), float("-inf"))
    ex = torch.einsum("hqk,hkd->hqd", torch.softmax(s, -1), vv)
    err = float((a - ex).abs().max())
    assert err < 1e-5, f"no-eviction case differs from exact attention: {err}"
    print(f"PASS budget >= L equals exact attention: max error {err:.2e}")
    print("ALL PASS")


if __name__ == "__main__":
    selftest()
