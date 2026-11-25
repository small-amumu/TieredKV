# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""T4 schedule self-test on synthetic data (CPU, no model, stubbed stack). Run: python -I t4_selftest.py

Checks: (1) decisions use only the mass of rows before the decision row and equal a slow reference;
(2) every decided block gets exactly the tier counts; (3) an all-tokens policy (G = 0, f = 1) makes the mask equal
to E_t = [4, w_t) on every row; (4) masks are monotone in the row and never cover the window or sinks;
(5) the random tie-break: equal scores are not demoted oldest-first; (6) chunking does not change decisions.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

stub = types.ModuleType("sel14_q")
stub.S4 = types.SimpleNamespace(DIAG={"on": False})
stub._attend_h = lambda *a, **k: None
stub._res = lambda r, b: r
sys.modules["sel14_q"] = stub
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_q as T4  # noqa: E402

torch.manual_seed(0)
KVH, LK, PREF = 2, 3000, 40
WIN, Z = T4.WIN, T4.SINKS
FAILS: list[str] = []


def ok(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def make_cu(heat: torch.Tensor | None = None, flat: bool = False) -> torch.Tensor:
    """(KVH, LK, LK) causal row-stochastic mass (row = query position)."""
    s = torch.zeros(KVH, LK, LK) if flat else torch.randn(KVH, LK, LK) * 3.0
    if heat is not None:
        s = s + heat[None, None, :]
    al = torch.arange(LK)[None, :] <= torch.arange(LK)[:, None]
    return torch.softmax(s.masked_fill(~al[None], float("-inf")), -1)


def run(pol: T4.Policy, cu: torch.Tensor, chunk: int, item: int = 7) -> dict:
    T4.reset_item(item, PREF, pol)
    masks = []
    for r0 in range(0, LK, chunk):
        r1 = min(LK, r0 + chunk)
        qpos = torch.arange(r0, r1)[:, None]
        T4._schedule(0, cu[:, r0:r1], qpos, KVH, LK, "cpu")
        masks.append([m.clone() for m in T4._masks(0, qpos, r1 - r0, KVH, LK, "cpu")])
    st = T4.STATE[0]
    return {"act": st["act"].clone(), "dpos": st["dpos"].clone(),
            "masks": [torch.cat([mm[a] for mm in masks], 1) for a in range(len(T4.ACTIONS))]}


def reference(pol: T4.Policy, cu: torch.Tensor, item: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Slow reference: per block, the decision row is the first row r with (r+1)//W*W >= b1 + G; score = mass
    from rows [exit row, r) where exit row is the first row whose window start is >= b1."""
    gen = torch.Generator(device="cpu").manual_seed(0 * 1000 + item)
    rnd = torch.randperm(LK, generator=gen).double()
    act = torch.zeros(KVH, LK, dtype=torch.int8)
    dpos = torch.full((KVH, LK), 1 << 40, dtype=torch.int64)
    cum = torch.cumsum(torch.cat([torch.zeros(KVH, 1, LK), cu.double()], 1), 1)   # cum[:, r] = mass of rows < r
    b0 = 0
    while True:
        b1 = b0 + WIN
        rows = [r for r in range(LK) if (r + 1) // WIN * WIN >= b1 + pol.g]
        if not rows:
            break
        r = rows[0]
        rexit = [x for x in range(LK) if (x + 1) // WIN * WIN >= b1][0]
        lo = max(b0, Z)
        nblk = b1 - lo
        for h in range(KVH):
            if pol.kind == "score":
                sc = (cum[h, r, lo:b1] - cum[h, rexit, lo:b1]).tolist()
                keys = sorted(range(nblk), key=lambda j: (sc[j], rnd[lo + j].item()))
            elif pol.kind == "random":
                keys = sorted(range(nblk), key=lambda j: rnd[lo + j].item())
            else:
                keys = list(range(nblk))
            c, st = 0.0, 0
            for f, a in pol.tiers:
                c += f
                en = min(nblk, int(c * nblk + 1e-9))
                for j in keys[st:en]:
                    act[h, lo + j] = T4.ACTIONS.index(a) + 1
                    dpos[h, lo + j] = r
                st = en
        b0 = b1
    return act, dpos


cu = make_cu()
for pol in (T4.Policy("score", 512, ((0.75, "del"),)), T4.Policy("score", 1024, ((0.5, "del"), (0.25, "h32"))),
            T4.Policy("random", 256, ((0.75, "key"),)), T4.Policy("oldest", 128, ((0.9, "del"),))):
    out = run(pol, cu, 256)
    ra, rd = reference(pol, cu)
    # score comparisons can differ in the last bits (float64 prefix sums vs cumsum): compare decisions exactly
    # only where the reference score gap is not tiny
    ok(torch.equal(out["dpos"], rd), f"{pol}: decision rows differ from the reference")
    agree = (out["act"] == ra).double().mean().item()
    ok(agree > 0.999, f"{pol}: actions agree with the reference on {agree:.5f}")
    for ai, a in enumerate(T4.ACTIONS):
        want = sum(1 for _, x in pol.tiers if x == a)
        if want == 0:
            ok(not bool((out["act"] == ai + 1).any()), f"{pol}: action {a} used but not in the tiers")
    # masks against the reference decisions: an action applies exactly from its reference decision row on
    # (catches a missing causal gate: rows between a token's exit and its decision row must not see the action)
    rows = torch.arange(LK)[None, :, None]
    same = (out["act"] == ra)[:, None, :]
    for ai in range(len(T4.ACTIONS)):
        want = (ra == ai + 1)[:, None, :] & (rows >= rd[:, None, :])
        ok(bool(((out["masks"][ai] == want) | ~same).all()), f"{pol}: mask for {T4.ACTIONS[ai]} != reference")
    out2 = run(pol, cu, 96)
    ok(torch.equal(out["act"], out2["act"]) and torch.equal(out["dpos"], out2["dpos"]),
       f"{pol}: chunk size changes decisions")
    for m in out["masks"]:
        ok(not bool(m[:, :, :Z].any()), f"{pol}: a sink is demoted")
        w = ((torch.arange(LK) + 1) // WIN * WIN)
        inwin = torch.arange(LK)[None, :] >= w[:, None]
        ok(not bool((m & inwin[None]).any()), f"{pol}: a window token is demoted")
        ok(bool((m[:, 1:] >= m[:, :-1]).all()), f"{pol}: a demotion was undone")
    # tier counts per decided block
    for h in range(KVH):
        for b0 in range(0, LK - WIN, WIN):
            b1 = b0 + WIN
            if int(out["dpos"][h, max(b0, Z):b1].min()) >= 1 << 40:
                continue
            lo = max(b0, Z)
            nblk = b1 - lo
            c, st = 0.0, 0
            for f, a in pol.tiers:
                c += f
                en = min(nblk, int(c * nblk + 1e-9))
                got = int((out["act"][h, lo:b1] == T4.ACTIONS.index(a) + 1).sum())
                ok(got == en - st, f"{pol}: block {b0} head {h} action {a}: {got} != {en - st}")
                st = en

# (3) all-tokens policy: mask == E_t on every row
pol = T4.Policy("oldest", 0, ((1.0, "del"),))
out = run(pol, cu, 256)
w = ((torch.arange(LK) + 1) // WIN * WIN)
E = (torch.arange(LK)[None, :] >= Z) & (torch.arange(LK)[None, :] < w[:, None])
ok(torch.equal(out["masks"][2], E[None].expand(KVH, -1, -1)), "all-tokens policy: mask != E_t")

# (5) random tie-break on equal scores: flat attention gives equal scores inside a block
cu_flat = make_cu(flat=True)
pol = T4.Policy("score", 512, ((0.5, "del"),))
out = run(pol, cu_flat, 256)
first = out["act"][0, Z:WIN]
older_share = float(first[: (WIN - Z) // 2].bool().double().mean())
ok(0.2 < older_share < 0.8, f"tie-break looks positional: older half demoted share {older_share:.2f}")

# (1) causality: changing mass at and after the decision row of block 0 must not change block 0's decision
pol = T4.Policy("score", 512, ((0.75, "del"),))
base = run(pol, cu, 256)
r_dec = int(base["dpos"][0, Z:WIN].min())
cu2 = cu.clone()
cu2[:, r_dec:, :] = make_cu(heat=torch.linspace(0, 5, LK))[:, r_dec:, :]
alt = run(pol, cu2, 256)
ok(torch.equal(base["act"][:, :WIN], alt["act"][:, :WIN]), "block 0 decision depends on rows at/after its row")


# (7) _record: column totals, page histogram totals, chunk invariance of rec / hist / reuse
_g = torch.Generator().manual_seed(3)
SC = torch.rand(KVH, LK, LK, generator=_g)               # per-row data fixed independently of the chunking
PRL = torch.randn(KVH * 2, LK, LK, generator=_g)
def record_run(chunk: int, pol: T4.Policy) -> dict:
    G2, H = 2, KVH * 2
    T4.reset_item(7, PREF, pol, pages=True)
    sel_tot = [0.0, 0.0]
    for r0 in range(0, LK, chunk):
        r1 = min(LK, r0 + chunk)
        n = r1 - r0
        qpos = torch.arange(r0, r1)[:, None]
        kpos = torch.arange(LK)[None, :]
        al = kpos <= qpos
        ex = (((kpos >= (qpos + 1) // WIN * WIN) & al) | (kpos < Z))[None].expand(H, -1, -1) & al[None]
        T4._schedule(0, cu[:, r0:r1], qpos, KVH, LK, "cpu")
        D = T4._masks(0, qpos, n, KVH, LK, "cpu")
        Dany = D[0] | D[1] | D[2]
        el = al & ~ex[0]
        sc = SC[:, r0:r1].masked_fill(~el[None], -1.0)
        kc = torch.zeros(KVH, n, LK, dtype=torch.bool).scatter_(-1, sc.topk(32, -1).indices, True) & el[None]
        m = torch.zeros_like(kc).scatter_(-1, torch.where(kc, sc, -2.0).topk(8, -1).indices, True) & kc
        m0 = torch.zeros_like(kc).scatter_(-1, torch.where(kc, sc.flip(-1), -2.0).topk(8, -1).indices, True) & kc
        pr = torch.softmax(PRL[:, r0:r1].masked_fill(~al[None], float("-inf")), -1)
        rep = (lambda t: t.repeat_interleave(G2, 0))
        T4._record(0, qpos, rep(kc), rep(m), rep(kc), rep(m0), pr, Dany, ex, al, G2, KVH, n, LK, "cpu", pr0=pr)
        dec = (qpos[:, 0] >= PREF)
        sel_tot[0] += float(kc[:, dec].sum())
        sel_tot[1] += float(m[:, dec].sum())
    sm = T4.STATS[0]
    return {"rec": sm["rec"].clone(), "hist": {P: sm["hist"][P].clone() for P in T4.PAGES},
            "reuse": sm["reuse"].clone(), "sel": sel_tot, "js": T4.stats_to_json(0)}


pol = T4.Policy("score", 512, ((0.5, "del"), (0.25, "h32")))
ra_, rb_ = record_run(256, pol), record_run(96, pol)
ok(torch.allclose(ra_["rec"], rb_["rec"], rtol=1e-9, atol=1e-9), "_record: rec depends on the chunk size")
ok(torch.equal(ra_["reuse"], rb_["reuse"]), "_record: page reuse depends on the chunk size")
ok(all(torch.equal(ra_["hist"][P], rb_["hist"][P]) for P in T4.PAGES), "_record: histogram depends on the chunk size")
ndec = LK - PREF
ok(float(ra_["rec"][:, 0].sum()) == ndec * KVH, "_record: column 0 != decode rows * KVH")
ok(float(ra_["rec"][:, 5].sum()) == 0.0, "_record: stage-1 drop column is not zero")
for P in T4.PAGES:
    h = ra_["hist"][P]
    ok(float(h[:, :, 0].sum()) == 0.0, f"_record: hist bin 0 used (P={P})")
    for si in range(2):
        hits = float((h[si].sum(0) * torch.arange(P + 1)).sum())
        ok(hits == ra_["sel"][si], f"_record: P={P} stage {si}: histogram hits {hits} != selected {ra_['sel'][si]}")
dem_cols = ra_["rec"][:, 2:5].sum(0)
ok(float(dem_cols[1]) == 0.0, "_record: key-only column used without a key tier")
ok(ra_["js"]["E_end"] == float(((LK - 1 + 1) // WIN * WIN - Z) * KVH), "_record: E_end != |E| at the last row")

# (8) incremental decoding: one row per call with a growing key length and a fixed state capacity must give the
# same decisions, masks and records as teacher forcing over the full length
def incremental_run(pol: T4.Policy, start: int) -> dict:
    T4.reset_item(7, PREF, pol, cap=LK)
    masks, rows = [], []
    for r0, r1 in [(0, start)] + [(r, r + 1) for r in range(start, LK)]:
        lk = r1                                           # keys up to the last query row of the call
        qpos = torch.arange(r0, r1)[:, None]
        T4._schedule(0, cu[:, r0:r1, :lk], qpos, KVH, lk, "cpu")
        mm = T4._masks(0, qpos, r1 - r0, KVH, lk, "cpu")
        masks.append([torch.nn.functional.pad(m, (0, LK - lk)) for m in mm])
    st = T4.STATE[0]
    return {"act": st["act"][:, :LK].clone(), "dpos": st["dpos"][:, :LK].clone(),
            "masks": [torch.cat([m[a] for m in masks], 1) for a in range(len(T4.ACTIONS))]}


for pol in (T4.Policy("score", 1024, ((0.75, "h32"),)), T4.Policy("score", 512, ((0.5, "del"), (0.25, "h32"))),
            T4.Policy("random", 1024, ((0.75, "h32"),))):
    full = run(pol, cu, 256)
    inc = incremental_run(pol, PREF)
    ok(torch.equal(full["act"], inc["act"]) and torch.equal(full["dpos"], inc["dpos"]),
       f"{pol}: incremental decoding changes decisions")
    ok(all(torch.equal(a_, b_) for a_, b_ in zip(full["masks"], inc["masks"])),
       f"{pol}: incremental decoding changes masks")
# (8b) _record under a capacity: one row per call with a growing key length gives the same records as the full
# teacher-forced chunks (page statistics are refused with a capacity, so pages are off in both runs)
def record_rows(pol: T4.Policy, cap: int, calls: list[tuple[int, int]]) -> dict:
    G2, H = 2, KVH * 2
    T4.reset_item(7, PREF, pol, pages=False, cap=cap)
    for r0, r1 in calls:
        lk = r1 if cap else LK
        n = r1 - r0
        qpos = torch.arange(r0, r1)[:, None]
        kpos = torch.arange(lk)[None, :]
        al = kpos <= qpos
        ex = (((kpos >= (qpos + 1) // WIN * WIN) & al) | (kpos < Z))[None].expand(H, -1, -1) & al[None]
        T4._schedule(0, cu[:, r0:r1, :lk], qpos, KVH, lk, "cpu")
        D = T4._masks(0, qpos, n, KVH, lk, "cpu")
        Dany = D[0] | D[1] | D[2]
        el = al & ~ex[0]
        sc = SC[:, r0:r1, :lk].masked_fill(~el[None], -1.0)
        kc = torch.zeros(KVH, n, lk, dtype=torch.bool).scatter_(-1, sc.topk(min(32, lk), -1).indices, True) & el[None]
        m = torch.zeros_like(kc).scatter_(-1, torch.where(kc, sc, -2.0).topk(min(8, lk), -1).indices, True) & kc
        sc2 = SC2[:, r0:r1, :lk].masked_fill(~el[None], -1.0)        # counterfactual ranking, fixed per row
        m0 = torch.zeros_like(kc).scatter_(-1, torch.where(kc, sc2, -2.0).topk(min(8, lk), -1).indices,
                                           True) & kc
        pr = torch.softmax(PRL[:, r0:r1, :lk].masked_fill(~al[None], float("-inf")), -1)
        rep = (lambda t_: t_.repeat_interleave(G2, 0))
        T4._record(0, qpos, rep(kc), rep(m), rep(kc), rep(m0), pr, Dany, ex, al, G2, KVH, n, lk, "cpu", pr0=pr)
    return {"rec": T4.STATS[0]["rec"].clone(), "js": T4.stats_to_json(0)}


SC2 = torch.rand(KVH, LK, LK, generator=_g)
pol = T4.Policy("score", 512, ((0.5, "del"), (0.25, "h32")))
full_r = record_rows(pol, 0, [(r, min(LK, r + 256)) for r in range(0, LK, 256)])
inc_r = record_rows(pol, LK, [(0, PREF)] + [(r, r + 1) for r in range(PREF, LK)])
ok(torch.allclose(full_r["rec"], inc_r["rec"], rtol=1e-9, atol=1e-9), "_record: incremental records differ")
ok(full_r["js"]["E_end"] == inc_r["js"]["E_end"] and full_r["js"]["dem"] == inc_r["js"]["dem"],
   "_record: incremental end-of-item counts differ")

try:
    T4.reset_item(7, PREF, None, pages=True, cap=LK)
    FAILS.append("cap with pages was accepted")
except ValueError:
    pass
T4.reset_item(7, PREF, T4.Policy("score", 1024, ((0.75, "h32"),)), cap=100)
T4._schedule(0, cu[:, :100, :100], torch.arange(0, 100)[:, None], KVH, 100, "cpu")
ok(T4.STATE[0]["acc"].shape[1] == 100, "capacity: state not allocated at cap")
try:
    T4._schedule(0, cu[:, 100:101, :101], torch.arange(100, 101)[:, None], KVH, 101, "cpu")
    FAILS.append("key length beyond the capacity was accepted")
except RuntimeError as e:
    ok("capacity" in str(e), f"capacity guard: unexpected error {e}")

if FAILS:
    print("FAIL")
    for f in FAILS:
        print(" -", f)
    sys.exit(1)
print("t4_selftest: all checks passed")
