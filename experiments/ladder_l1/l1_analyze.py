# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""l1_analyze: scoring of round L1 (algorithm ladder on the SpeCache / KIVI-128 base) exactly as registered in
REG_L1_ladder.md. Per model: mean next-token KL D over the evaluation segments of each run; ratios per model; median
over models. A clause HOLDS when its median threshold, its count and (main panel only) the Holm-adjusted exact one-sided
Wilcoxon p <= 0.05 all hold; SUPPORTED = threshold and count only; else FAILS. Bootstrap 95% intervals over models
(5000 resamples, seed 0) are descriptive.
Usage: python -I l1_analyze.py <results root holding l1_main/ l1_held/ l1_new/>
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import random
import statistics as S
import sys
from pathlib import Path

L0, L1, L2, L3, L4 = ("kivi128", "kivi128k64_xqs", "kivi128k120_x2s120_h88", "kivi128k80_x2s160_h88",
                      "kivi128k80_x2s160_j50_h88")
DEP, XOR, J30 = "kivi128k64_x2s128_j50_h54", "kivi128k64_xor", "kivi128k80_x2s160_j30_h88"
FLOOR = "kivi128k64_x2s128_hff_id"   # identity arm: exact rows everywhere, measures the numerical floor
RUNS = (L0, L1, L2, L3, L4, DEP, XOR, J30, FLOOR)
# (tag, numerator, denominator, median threshold, main count fraction, primary)
CLAUSES = (("A1 residual rows / FP16 rows, about L1 bytes", L2, L1, 0.80, 0.75, True),
           ("A2 two-stage / single-stage, same bytes", L3, L2, 0.95, 0.75, True),
           ("A3 debias 50% / none", L4, L3, 0.90, 0.75, True),
           ("A4 full ladder / SpeCache-style, about L1 bytes", L4, L1, 0.60, 0.75, True),
           ("A5 deployed h54 (45-48% of L1 bytes) / SpeCache-style", DEP, L1, 1.00, 0.75, True))
PANELS = ("l1_main", "l1_held", "l1_new")


def read_bytes(name: str, d: int) -> float:
    """bits read over PCIe per KV head per decode step (index bits not counted, as for every arm)."""
    if name in (L0, FLOOR):
        return 0.0
    if name in (L1, XOR):
        return 64 * 2 * 16 * d
    k_m = {L2: (120, 120, 8, 8), L3: (80, 160, 8, 8), L4: (80, 160, 8, 8), J30: (80, 160, 8, 8),
           DEP: (64, 128, 5, 4)}[name]
    k, m, rk, rv = k_m
    return m * (rk * d + 32) + k * (rv * d + 32)


MIN_VALID = {"l1_main": 12, "l1_held": 5, "l1_new": 6}


def load(d: Path, expected: list[str]) -> tuple[dict, dict, dict]:
    """Valid models only; every expected model that is not valid is returned in `invalid` with the reason
    (registered VOID rules: missing output, fewer than 4 segments, run list without a registered run, non-finite D)."""
    out, dims, invalid = {}, {}, {}
    for name in expected:
        f = d / f"{name}.json"
        if not f.is_file():
            invalid[name] = "no output"
            continue
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            invalid[name] = f"unreadable ({type(e).__name__})"
            continue
        if j.get("smoke"):
            invalid[name] = "smoke output"
            continue
        seg = j.get("seg") or []
        if len(seg) != 4:
            invalid[name] = f"{len(seg)} segments"
            continue
        miss = [r for r in RUNS if r not in j.get("runs", []) or any(r not in s["D"] for s in seg)]
        if miss:
            invalid[name] = f"missing runs {miss}"
            continue
        vals = {r: sum(s["D"][r] for s in seg) / len(seg) for r in RUNS}
        bad = [r for r in RUNS if r != FLOOR and not (math.isfinite(vals[r]) and vals[r] >= 0)]
        if bad:
            invalid[name] = f"non-finite or negative D on {bad}"
            continue
        out[name] = vals
        dims[name] = (j.get("head_dim"), len(seg))
    return out, dims, invalid


def wilcoxon_less(x: list[float]) -> float:
    x = [v for v in x if v != 0.0]
    n = len(x)
    if n == 0:
        return 1.0
    a = sorted(range(n), key=lambda i: abs(x[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(x[a[j + 1]]) == abs(x[a[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[a[k]] = (i + j) / 2 + 1
        i = j + 1
    w_pos = sum(r for r, v in zip(ranks, x) if v > 0)
    hits = tot = 0
    for signs in itertools.product((0, 1), repeat=n):
        tot += 1
        if sum(r for r, s_ in zip(ranks, signs) if s_) <= w_pos + 1e-12:
            hits += 1
    return hits / tot


def boot(rs: list[float], rnd: random.Random) -> tuple[float, float]:
    bs = sorted(S.median(rnd.choices(rs, k=len(rs))) for _ in range(5000))
    return bs[124], bs[4874]


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main() -> None:
    root = Path(sys.argv[1])
    rnd = random.Random(0)
    lists = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent
    for panel in PANELS:
        expected = [ln.split()[1] for ln in (lists / f"{panel}.txt").read_text(encoding="utf-8").splitlines()
                    if ln.strip()]
        data, dims, invalid = load(root / panel, expected)
        models = sorted(data)
        print(f"\n## {panel}: {len(models)}/{len(expected)} valid models: {' '.join(models)}")
        for m, why in invalid.items():
            print(f"  excluded (VOID rule): {m}: {why}")
        if len(models) < MIN_VALID[panel]:
            print(f"PANEL VOID: {len(models)} valid models < {MIN_VALID[panel]}; no clause is scored for {panel}")
            continue
        rows = []
        for tag, a, b, thr, frac, prim in CLAUSES:
            rs = [data[m][a] / data[m][b] for m in models if data[m][b] > 0]
            med, cnt = S.median(rs), sum(r < 1 for r in rs)
            p = wilcoxon_less([math.log(r) for r in rs if r > 0])
            need = math.ceil(frac * len(rs)) if panel == "l1_main" else len(rs) - 1
            rows.append((tag, rs, med, cnt, need, thr, p))
        adj = holm([r[6] for r in rows]) if panel == "l1_main" else [r[6] for r in rows]
        for (tag, rs, med, cnt, need, thr, p), pa in zip(rows, adj):
            ok = med <= thr and cnt >= need
            if panel == "l1_main":
                verdict = "HOLDS" if ok and pa <= 0.05 else ("SUPPORTED" if ok else "FAILS")
            else:
                verdict = "HOLDS (count rule, no significance test)" if ok else "FAILS"
            lo, hi = boot(rs, rnd)
            print(f"{tag}: median {med:.3f} [{lo:.3f}, {hi:.3f}] (<= {thr}), lower {cnt}/{len(rs)} (need {need}), "
                  f"p {p:.4g} Holm {pa:.4g} -> {verdict}")
        for tag, a_, b_ in (("J30 / L3", J30, L3), ("J30 / L4", J30, L4)):
            rs = [data[m][a_] / data[m][b_] for m in models]
            print(f"descriptive {tag}: median {S.median(rs):.3f}, lower {sum(r < 1 for r in rs)}/{len(rs)}")
        fl = [data[m][FLOOR] / data[m][L1] for m in models if math.isfinite(data[m][FLOOR])]
        if fl:
            print(f"descriptive identity arm / L1: median {S.median(fl):.2e}, max {max(fl):.2e} ({len(fl)} finite)")
        crit = [data[m][L1] / data[m][XOR] for m in models]
        print(f"descriptive: exact-attention selection / base selection (64 FP16 rows) median "
              f"{S.median([1 / c for c in crit]):.3f}, base better in {sum(c < 1 for c in crit)}/{len(crit)}")
        print("per model D (" + ", ".join(RUNS) + ") and read KiB per KV head per step at the model's head_dim:")
        for m in models:
            d = dims[m][0]
            if not d:
                print(f"  {m}: head_dim missing in the output; KiB column assumes 128")
                d = 128
            print(f"  {m:16s} " + " ".join(f"{data[m][r]:.4g}" for r in RUNS) + "  | KiB "
                  + " ".join(f"{read_bytes(r, d) / 8 / 1024:.1f}" for r in RUNS))


if __name__ == "__main__":
    main()
