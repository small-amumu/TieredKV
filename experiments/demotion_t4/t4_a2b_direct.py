# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""A2b direct comparisons: score-based h32 demotion against the measured uniform 136 B points, same run.

usage: python -I t4_a2b_direct.py <a2b_reduced.json> <merged.json> [<a2b_reduced.json> <merged.json> ...]
Part 1 (one run, one GPU, items paired): geometric-mean KL ratio with an item bootstrap 95% interval for
  sc_G1024_h32_75 / h44, sc_G1024_h32_75 / h53, h44 / h53, h44 / h54, sc_G1024_h32_75 / h54, on the thinking
  span (all positions), thinking positions >= 4096, and the answer span; by sample length (all, L < 2048, L >= 4096).
  The demotion arm's own bytes per record are printed next to 136 B.
Part 2 (merged file; h43 / h33 / h32 come from the held-out run on another GPU): the answer-span uniform frontier
  at the demotion arm's per-item bytes (log KL interpolated linearly between measured uniform points; h44 and h53
  collapse to the lower one), ratio < 1 means the demotion arm is better than uniform at equal bytes.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

FLOOR, NB, SEED = 1e-12, 2000, 12345
SC = "sc_G1024_h32_75"
PAIRS = [(SC, "h44"), (SC, "h53"), ("h44", "h53"), ("h44", "h54"), (SC, "h54")]
GROUPS = {"all": (0, 1 << 30), "L<2048": (0, 2048), "L>=4096": (4096, 1 << 30)}
LONG = ("4096-8192", "8192-1073741824")
REC_B = {"h54": 152.0, "h53": 136.0, "h44": 136.0, "h43": 120.0, "h33": 104.0, "h32": 88.0}


def boot(lr: list) -> str:
    if not lr:
        return "-"
    rng = random.Random(SEED)
    bs = sorted(sum(rng.choice(lr) for _ in lr) / len(lr) for _ in range(NB))
    return (f"{math.exp(sum(lr) / len(lr)):.2f} [{math.exp(bs[int(0.025 * NB)]):.2f},"
            f"{math.exp(bs[int(0.975 * NB) - 1]):.2f}] n={len(lr)}")


def lg(x: float | None, y: float | None) -> float | None:
    if x is None or y is None:
        return None
    return math.log(max(x, FLOOR) / max(y, FLOOR))


def long_kl(it: dict, arm: str) -> float | None:
    s, n = 0.0, 0
    for b in LONG:
        v = it["bucket"][arm].get(b)
        if v and v[0] is not None and v[1]:
            s += v[0] * v[1]
            n += v[1]
    return s / n if n else None


def bytes_rec(it: dict, arm: str) -> float | None:
    if arm in REC_B:
        return REC_B[arm]
    lay = it["layers"].get(arm) or {}
    rows = [r for v in lay.values() for r in v["rec"]]
    e = sum(r[1] for r in rows)
    if e <= 0:
        return None
    dh, dk, dd = (sum(r[j] for r in rows) for j in (2, 3, 4))
    return (152.0 * (e - dh - dk - dd) + 88.0 * dh + 84.0 * dk) / e


def frontier(vals: dict, b: float) -> float | None:
    best: dict = {}
    for u, v in vals.items():
        if v is not None:
            lk = math.log(max(v, FLOOR))
            best[REC_B[u]] = min(lk, best.get(REC_B[u], lk))
    pts = sorted(best.items())
    for (b0, l0), (b1, l1) in zip(pts, pts[1:]):
        if b0 <= b <= b1:
            return math.exp(l0 + (l1 - l0) * (b - b0) / (b1 - b0)) if b1 > b0 else math.exp(l0)
    return None


def main() -> int:
    args = sys.argv[1:]
    if not args or len(args) % 2:
        raise SystemExit(__doc__)
    for ap, mp in zip(args[::2], args[1::2]):
        a = json.loads(Path(ap).read_text(encoding="utf-8"))
        m = json.loads(Path(mp).read_text(encoding="utf-8"))
        print(f"== {Path(a['model']).name} (A2b {a['reduced_from']['md5'][:8]}, merged {Path(mp).name})")
        for g, (lo, hi) in GROUPS.items():
            items = [it for it in a["items"] if "skipped" not in it and lo <= it["L"] < hi]
            bs = [b for b in (bytes_rec(it, SC) for it in items) if b is not None]
            print(f"  [{g}] {len(items)} items; {SC} bytes/record {sum(bs) / len(bs):.1f} vs uniform 136")
            for x, y in PAIRS:
                dec = [lg(it["decode"][x]["kl"], it["decode"][y]["kl"]) for it in items]
                lng = [lg(long_kl(it, x), long_kl(it, y)) for it in items]
                ans = [lg(it["answer"][x][0], it["answer"][y][0]) for it in items]
                f = [v for v in dec if v is not None]
                print(f"    {x}/{y}: thinking {boot(f)} | pos>=4096 {boot([v for v in lng if v is not None])}"
                      f" | answer {boot([v for v in ans if v is not None])}"
                      f" | x>y in {sum(v > 0 for v in f) / max(len(f), 1):.2f}")
            mi = [it for it in m["items"] if "skipped" not in it and lo <= it["L"] < hi]
            uni = [u for u in REC_B if u in m["runs"]]
            for arm in (SC, "rnd_G1024_h32_75", "sc_G1024_2lvl"):
                lr = []
                for it in mi:
                    b = bytes_rec(it, arm)
                    if b is None or it["answer"][arm][0] is None:
                        continue
                    ref = frontier({u: it["answer"][u][0] for u in uni}, b)
                    if ref is not None:
                        lr.append(lg(it["answer"][arm][0], ref))
                print(f"    answer-span uniform frontier, {arm}: {boot(lr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
