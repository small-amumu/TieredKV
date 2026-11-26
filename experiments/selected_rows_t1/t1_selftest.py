# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""T1 self-test (REG_T1.md v3 Q0b / Q0c on synthetic data; CPU, no model). Run: python -I t1_selftest.py"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

stub = types.ModuleType("sel14_q")                      # t1_q only needs these names at import time
stub.S4 = types.SimpleNamespace(DIAG={"on": False})
stub._attend_h = lambda *a, **k: None
stub._res = lambda r, b: r
sys.modules["sel14_q"] = stub
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_q as T1  # noqa: E402

torch.manual_seed(0)
KVH, G, LK, PREF = 2, 2, 3000, 40
M, K, R, Z = T1.M_RUN, T1.K_RUN, T1.WIN, T1.SINKS


def chunk(r0: int, r1: int, scale: float = 3.0, heat: torch.Tensor | None = None):
    """Selection tensors exactly as t1_attend_h builds them for rows r0..r1 (Lq = Lk: full teacher forcing)."""
    n = r1 - r0
    kpos = torch.arange(LK)[None, :]
    qpos = torch.arange(r0, r1)[:, None]
    al = kpos <= qpos
    s = torch.randn(KVH * G, n, LK) * scale
    if heat is not None:
        s = s + heat[None, None, :]
    ex = ((kpos >= (qpos + 1) // R * R) & al)[None].expand_as(s) | (kpos < Z)[None]
    ex = ex & al[None]
    bad = ~al[None] | ex
    full = torch.softmax(s.masked_fill(~al, float("-inf")), -1)
    cu = full.reshape(KVH, G, n, LK).mean(1)
    c1 = full.masked_fill(bad, -float("inf")).reshape(KVH, G, n, LK).mean(1)
    kc = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, c1.topk(min(M, LK), -1).indices, True)
    kc = kc.repeat_interleave(G, 0) & ~bad
    c2 = torch.where(kc[::G], torch.rand(KVH, n, LK), torch.full((KVH, n, LK), -float("inf")))
    m = torch.zeros_like(c2, dtype=torch.bool).scatter_(-1, c2.topk(min(K, LK), -1).indices, True).repeat_interleave(G, 0)
    m = m & kc
    prK = torch.softmax(torch.randn(KVH, n, LK).masked_fill(~al, float("-inf")), -1)
    return qpos, cu, c1, kc, m, prK, ex, al, n


def run(scale: float, heat=None) -> list:
    T1.reset_item(7, PREF)
    T1.CFG["on"] = True
    allc = []
    for r0 in range(0, LK, 256):
        r1 = min(LK, r0 + 256)
        qpos, cu, c1, kc, m, prK, ex, al, n = chunk(r0, r1, scale, heat)
        T1._record(0, qpos, cu, c1, kc, m, prK, ex, al, G, KVH, n, LK)
        allc.append(cu.double())
    return allc


def check(allc: list, tag: str) -> None:
    # Q0b: final accumulator and a reference per-row a_exit for every exited block
    ref = torch.zeros(KVH, LK, dtype=torch.float64)
    a_exit = torch.zeros(KVH, LK, dtype=torch.float64)
    t = 0
    for cc in allc:
        for j in range(cc.shape[1]):
            w = ((t + 1) // R) * R
            if w > 0 and w % R == 0 and (t + 1) % R == 0:          # this row moves the window to w
                a_exit[:, w - R:w] = ref[:, w - R:w]
            ref = ref + cc[:, j]
            t += 1
    st = T1.STATE[0]
    worst = float(((st["acc"] - ref).abs() / ref.abs().clamp_min(1e-30)).max())
    exited = st["exited"]
    worst_e = float(((st["a_exit"][:, exited] - a_exit[:, exited]).abs()
                     / a_exit[:, exited].abs().clamp_min(1e-30)).max())
    print(f"[{tag}] Q0b acc rel diff {worst:.2e}, a_exit rel diff {worst_e:.2e}")
    assert worst <= 1e-9 and worst_e <= 1e-9, (worst, worst_e)
    # Q0c: decided blocks hold exactly floor(f * |block|) demoted tokens; undecided blocks hold none
    w = st["w"]
    for p in T1.POLICIES:
        dm = st["D"][T1.pol_name(p)]
        for b0 in range(0, LK, R):
            b1 = b0 + R
            lo = max(b0, Z)
            cnt = dm[:, lo:min(b1, LK)].sum(-1)
            want = int(p[2] * (b1 - lo)) if b1 + p[1] <= w else 0
            assert bool((cnt == want).all()), (p, b0, cnt, want)
        assert not bool(dm[:, :Z].any())
    sm = T1.stats_to_json(T1.STATS[0], st)
    assert all(v == v for rows in sm["pol"].values() for b in rows for v in b)


def main() -> int:
    check(run(3.0), "dense")
    check(run(60.0), "sparse")                                       # many exact zeros: no cancellation
    # persistent per-token heat: every score policy must demote colder tokens than random and than oldest-first
    heat = torch.randn(LK) * 2.0
    run(3.0, heat)
    st = T1.STATE[0]
    mean = {}
    for p in T1.POLICIES:
        dm = st["D"][T1.pol_name(p)]
        if bool(dm.any()):
            mean[p] = float(heat[dm[0]].mean())
    for g in T1.PROB:
        for f in T1.FRACS:
            if ("score", g, f) not in mean:
                continue
            sc, rd, od = mean[("score", g, f)], mean[("random", g, f)], mean[("oldest", g, f)]
            print(f"G{g} f{f}: mean heat demoted score {sc:.3f} random {rd:.3f} oldest {od:.3f}")
            assert sc < rd - 0.1 and sc < od - 0.1, (g, f, sc, rd, od)
    print("T1 self-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
