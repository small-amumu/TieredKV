# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""b1c_analyze: scoring of round B1c (our stack base vs RotateKV (official code path, rkv_q) at equal or higher GPU bits) exactly
as registered in REG_B1a.md. Per model: mean next-token KL D over the 4 evaluation segments of each run.
Matched comparator per model: among the RotateKV arms, the one with the SMALLEST GPU bits that are >= the stack's GPU
bits (as recorded by the lab's eff_bits); if no RotateKV arm reaches the stack's bits, the largest-bit arm is used and
the model is flagged. Clauses:
  R1  stk128 / matched RotateKV: median <= 1.00, lower in >= 75% of valid models, exact one-sided Wilcoxon p <= 0.05
      -> HOLDS; threshold and count only -> SUPPORTED; else FAILS.
  R2  worst model of stk128 / matched RotateKV <= 1.25 -> HOLDS, else FAILS (listing every model above 1.25).
Descriptive: stk128 / kivi128, each RotateKV arm / kivi128, GPU bits per arm, RotateKV K2.0 / K2.8.
VOID: a model without exactly 4 segments, missing a registered run, or with a non-finite / negative D is excluded and
listed; fewer than 14 valid models makes G1 and G2 VOID.
Usage: python -I b1a_analyze.py <results dir holding b1c_A/ and b1c_B/> <dir holding b1a_laneA.txt, b1a_laneB.txt>
"""
from __future__ import annotations

import itertools
import json
import math
import random
import statistics as S
import sys
from pathlib import Path

OURS, KIVI = "stk128", "kivi128"
GTQ = ("rkv2", "rkv2w128", "rkv2w256")   # the RotateKV 2-bit arms eligible as comparator
EXTRA = ("rkv3", "gtqk22v20w128")
RUNS = (OURS, KIVI) + GTQ + EXTRA
MIN_VALID = 14
FAMILY = {"q25_7b": "qwen2", "q25_1p5b": "qwen2", "q2_0p5b": "qwen2", "q25_0p5b": "qwen2", "r1q_1p5b": "qwen2",
          "q25m_1p5b": "qwen2", "q25_3b": "qwen2", "llama8b": "llama3", "llama3b": "llama3", "llama1b": "llama3",
          "mistral7b": "mistral", "gemma2b": "gemma2", "smol3_3b": "smollm", "smol2_1p7b": "smollm",
          "falcon3_3b": "falcon3", "falcon3_7b": "falcon3", "granite31_2b": "granite", "yi15_6b": "yi"}


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


def main() -> None:
    root, lists = Path(sys.argv[1]), Path(sys.argv[2])
    expected = [ln.split()[1] for f in ("b1c_laneA.txt", "b1c_laneB.txt")
                for ln in (lists / f).read_text(encoding="utf-8").splitlines() if ln.strip()]
    data, bits, invalid = {}, {}, {}
    for m in expected:
        cands = [root / d / f"{m}.json" for d in ("b1c_A", "b1c_B") if (root / d / f"{m}.json").is_file()]
        if len(cands) > 1:
            invalid[m] = "output in both lanes"
            continue
        f = cands[0] if cands else root / "b1c_A" / f"{m}.json"
        if not f.is_file():
            invalid[m] = "no output"
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        seg = j.get("seg") or []
        if j.get("smoke") or len(seg) != 4:
            invalid[m] = f"{len(seg)} segments / smoke"
            continue
        miss = [r for r in RUNS if r not in j.get("runs", []) or any(r not in s["D"] for s in seg)]
        if miss:
            invalid[m] = f"missing runs {miss}"
            continue
        vals = {r: sum(s["D"][r] for s in seg) / 4 for r in RUNS}
        bad = [r for r in RUNS if not (math.isfinite(vals[r]) and vals[r] >= 0)]
        if bad:
            invalid[m] = f"non-finite or negative D on {bad}"
            continue
        data[m], bits[m] = vals, {r: j["bits"][r][0] for r in RUNS}
        bits[m]["_dtype"] = j.get("dtype")
    models = sorted(data)
    print(f"B1c: {len(models)}/{len(expected)} valid models")
    for m, why in invalid.items():
        print(f"  excluded (VOID rule): {m}: {why}")
    rows = []
    for m in models:
        b0 = bits[m][OURS]
        ok = [g for g in GTQ if bits[m][g] >= b0 - 1e-9]
        g = min(ok, key=lambda x: bits[m][x]) if ok else max(GTQ, key=lambda x: bits[m][x])
        rows.append((m, g, data[m][OURS] / data[m][g], b0, bits[m][g], not ok))
    if len(models) < MIN_VALID:
        print(f"VOID: {len(models)} valid models < {MIN_VALID}; R1 and R2 are not scored")
    else:
        rs = [r[2] for r in rows]
        med, cnt, p = S.median(rs), sum(r < 1 for r in rs), wilcoxon_less([math.log(r) for r in rs])
        need = math.ceil(0.75 * len(rs))
        ok1 = med <= 1.0 and cnt >= need
        v1 = "HOLDS" if ok1 and p <= 0.05 else ("SUPPORTED" if ok1 else "FAILS")
        rnd = random.Random(0)
        bs = sorted(S.median(rnd.choices(rs, k=len(rs))) for _ in range(5000))
        print(f"R1 stk128 / matched RotateKV: median {med:.3f} [{bs[124]:.3f}, {bs[4874]:.3f}] (<= 1.0), lower {cnt}/"
              f"{len(rs)} (need {need}), one-sided Wilcoxon p {p:.4g} -> {v1}")
        worst = max(rows, key=lambda r: r[2])
        over = [f"{r[0]} {r[2]:.3f}" for r in rows if r[2] > 1.25]
        print(f"R2 worst model {worst[0]} {worst[2]:.3f} (<= 1.25) -> {'HOLDS' if not over else 'FAILS'}"
              + (f"; above 1.25: {', '.join(over)}" if over else ""))
    rs_nf = [r[2] for r in rows if not r[5]]
    if len(rs_nf) != len(rows) and rs_nf:
        print(f"descriptive R1 without flagged models: median {S.median(rs_nf):.3f}, lower {sum(r < 1 for r in rs_nf)}/{len(rs_nf)}, worst {max(rs_nf):.3f}")
    fam: dict = {}
    for m, g, r, *_ in rows:
        fam.setdefault(FAMILY.get(m, m), []).append(r)
    fm = {f: S.median(v) for f, v in fam.items()}
    if fm:
        print(f"descriptive family level ({len(fm)} families): median of family medians {S.median(list(fm.values())):.3f}, "
              f"families lower {sum(v < 1 for v in fm.values())}/{len(fm)}; " + ", ".join(f"{f} {v:.3f}" for f, v in sorted(fm.items())))
    for dt in ("float32", "bfloat16"):
        rs_dt = [r[2] for r in rows if bits[r[0]]["_dtype"] == dt]
        if rs_dt:
            print(f"descriptive {dt} models: median {S.median(rs_dt):.3f}, lower {sum(v < 1 for v in rs_dt)}/{len(rs_dt)}")
    print("descriptive medians over valid models:")
    for a, b in ((OURS, KIVI),) + tuple((g, KIVI) for g in GTQ) + (("rkv2", "rkv3"), ("gtqk22v20w128", "rkv2w256")):
        rs = [data[m][a] / data[m][b] for m in models]
        if rs:
            print(f"  {a} / {b}: median {S.median(rs):.3f}, lower {sum(r < 1 for r in rs)}/{len(rs)}")
    print("per model: matched arm, stk128 / matched, GPU bits (ours, matched), flag (no RotateKV arm reaches our bits)")
    for m, g, r, b0, bg, flag in rows:
        print(f"  {m:14s} {g:14s} {r:.3f}  bits {b0:.3f} / {bg:.3f}{'  FLAG' if flag else ''}  | D "
              + " ".join(f"{k}={data[m][k]:.4g}" for k in RUNS))


if __name__ == "__main__":
    main()
