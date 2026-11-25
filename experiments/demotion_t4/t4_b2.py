# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""B2 (read path, plan line B): page-hit statistics from the A2 replay records, plus two A2 items still open
(answer-span argmax agreement, per-item host bytes quantiles).

usage: python -I t4_b2.py <out_dir> <a2_lab_json> [...]
Writes <out_dir>/t4_b2.json and t4_b2.txt.

Inputs are t4_lab.py outputs (A2). Arms with page statistics: h54, sc_G1024_del75, sc_G1024_h32_75. Per arm, layer,
page size P in (8, 16, 32, 64), view and decode-position bucket the record holds a histogram of hits per touched
page (h = 1..P selected rows in one page of P consecutive tokens, for one (row, KV head)), and page reuse
(touched pages at a row that had a previous row, and of them pages also touched at the previous row).
Views: s1 / s2 = the logical stage-1 (128 key rows) / stage-2 (64 value rows) selections; K54 / K32 = stage-1 rows
that are readable from the h54 / h32 key pool; V54 / V32 = stage-2 rows readable from the h54 / h32 value pool
(deleted residuals and key-only values are not readable, so they are not counted).

Per (model, arm, view, P), pooled over layers, KV heads and decode rows (and per position bucket):
  rows R = sum h * n_h, pages N = sum n_h, density R / N (rows per touched page),
  whole-page overfetch N * P / R (rows read per useful row if every touched page is read whole),
  for a switch threshold k (pages with h >= k read whole, the rest row by row): share of useful rows served by
  whole pages, share of pages read whole, overfetch of the whole-page part, for k in (2, P/4, P/2),
  reuse = reused / touched.
Development / validation split (fixed before looking: plan B2): trace id even = development, odd = validation; every
statistic is reported for both halves so a switching rule chosen on one half can be checked on the other.
Bytes per row and transfer times are not here: they need the B1 microbenchmark.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

VIEWS = ("s1", "s2", "K54", "K32", "V54", "V32")
PAGES = (8, 16, 32, 64)
BUCKETS = ("0-1024", "1024-2048", "2048-4096", "4096-8192", "8192+")
PAGE_ARMS = ("h54", "sc_G1024_del75", "sc_G1024_h32_75")
REC_B = {"h54": 152.0, "h43": 120.0, "h33": 104.0, "h32": 88.0, "h32c": 88.0, "all_del": 0.0}
ACT_B = (88.0, 84.0, 0.0)


def add(acc: list, h: list) -> None:
    for j, x in enumerate(h):
        acc[j] += x


def stats(h: list, P: int) -> dict:
    """h[j] = number of touched pages with j hits (j = 0 unused)."""
    R = sum(j * h[j] for j in range(1, P + 1))
    N = sum(h[1:P + 1])
    if N == 0:
        return {"rows": 0, "pages": 0}
    out = {"rows": R, "pages": N, "density": R / N, "overfetch_whole": N * P / R}
    for k in sorted({2, max(2, P // 4), max(2, P // 2)}):
        rw = sum(j * h[j] for j in range(k, P + 1))
        pw = sum(h[j] for j in range(k, P + 1))
        out[f"k{k}"] = {"rows_whole": rw / R, "pages_whole": pw / N,
                        "overfetch_whole_part": (pw * P / rw) if rw else None}
    return out


def bytes_item(it: dict, arm: str) -> float | None:
    if arm in REC_B:
        return REC_B[arm]
    lay = it["layers"].get(arm)
    if not lay:
        return None
    e = dh = dk = dd = 0.0
    for v in lay.values():
        for r in v["rec"]:
            e += r[1]; dh += r[2]; dk += r[3]; dd += r[4]
    if e <= 0:
        return None
    return (152.0 * (e - dh - dk - dd) + ACT_B[0] * dh + ACT_B[1] * dk + ACT_B[2] * dd) / e


def quantiles(xs: list) -> dict | None:
    v = sorted(x for x in xs if x is not None)
    if not v:
        return None
    q = (lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))])
    return {"n": len(v), "min": v[0], "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "max": v[-1]}


def schemes(h: list, P: int) -> dict:
    """Rows read and read requests per useful row for row-wise, whole-page and hybrid-k reads (no byte costs)."""
    R = sum(j * h[j] for j in range(1, P + 1))
    N = sum(h[1:P + 1])
    if not R:
        return {}
    out = {"row": {"rows_read": 1.0, "requests": 1.0}, "page": {"rows_read": N * P / R, "requests": N / R}}
    for k in sorted({2, max(2, P // 4), max(2, P // 2)}):
        lo = sum(j * h[j] for j in range(1, k))
        hi_p = sum(h[j] for j in range(k, P + 1))
        out[f"k{k}"] = {"rows_read": (lo + hi_p * P) / R, "requests": (lo + hi_p) / R}
    return out


def rand_density(m: float, E: float, P: int) -> float | None:
    """Rows per touched page if m rows were placed uniformly at random among E eligible tokens (pages of P)."""
    if m <= 0 or E < P or m > E:
        return None
    touched = (E / P) * (1.0 - (1.0 - P / E) ** m)
    return m / touched if touched > 0 else None


def analyze(lab: dict, src: str) -> dict:
    items = [it for it in lab["items"] if "skipped" not in it]
    res = {"model": Path(lab["model"]).name, "source": src, "n_items": len(items), "pages": {}, "argmax_answer": {},
           "bytes_quantiles": {}}
    for arm in PAGE_ARMS:
        if arm not in lab["runs"]:
            continue
        # per bucket: decode rows x KV heads x layers (rec col 0) and eligible tokens (rec col 1), all items
        c0 = [0.0] * len(BUCKETS)
        c1 = [0.0] * len(BUCKETS)
        for it in items:
            for lay in it["layers"][arm].values():
                for b, row in enumerate(lay["rec"]):
                    c0[b] += row[0]
                    c1[b] += row[1]
        res.setdefault("bucket_rows", {})[arm] = {BUCKETS[b]: {"row_head_layers": c0[b],
                                                                "mean_eligible": c1[b] / c0[b] if c0[b] else None}
                                                    for b in range(len(BUCKETS))}
        for split in ("dev", "val", "all"):
            sel = [it for it in items if split == "all" or (it["i"] % 2 == 0) == (split == "dev")]
            for P in PAGES:
                for vi, view in enumerate(VIEWS):
                    tot = [0.0] * (P + 1)
                    per_b = [[0.0] * (P + 1) for _ in BUCKETS]
                    reuse = [0.0, 0.0]
                    for it in sel:
                        for lay in it["layers"][arm].values():
                            hv = lay["hist"][str(P)][vi]
                            for b in range(len(BUCKETS)):
                                add(tot, hv[b])
                                add(per_b[b], hv[b])
                                rb = lay["reuse"][str(P)][vi][b]
                                reuse[0] += rb[0]
                                reuse[1] += rb[1]
                    st = stats(tot, P)
                    st["reuse"] = reuse[1] / reuse[0] if reuse[0] else None
                    st["by_bucket"] = {BUCKETS[b]: stats(per_b[b], P) for b in range(len(BUCKETS))}
                    st["schemes"] = schemes(tot, P)
                    for b in range(len(BUCKETS)):
                        bb = st["by_bucket"][BUCKETS[b]]
                        bb["schemes"] = schemes(per_b[b], P)
                        if split == "all" and c0[b] and bb.get("rows"):
                            m_ = bb["rows"] / c0[b]          # selected rows per (row, KV head, layer)
                            E_ = c1[b] / c0[b]
                            bb["mean_rows_selected"], bb["mean_eligible"] = m_, E_
                            bb["random_density"] = rand_density(m_, E_, P)
                    st["hist"] = tot
                    res["pages"].setdefault(arm, {}).setdefault(split, {}).setdefault(str(P), {})[view] = st
    for arm in lab["runs"]:
        ag = [it["answer"][arm][2] for it in items if it["answer"][arm][2] is not None]
        ag0 = [(it["answer"][arm][2], it["answer"][arm][1]) for it in items if it["answer"][arm][2] is not None]
        res["argmax_answer"][arm] = {"items": len(ag), "mean_item": sum(ag) / len(ag) if ag else None,
                                     "pooled_rows": (sum(a * n for a, n in ag0) / sum(n for _, n in ag0)) if ag0 else None}
        res["bytes_quantiles"][arm] = {"all": quantiles([bytes_item(it, arm) for it in items]),
                                       "L>=4096": quantiles([bytes_item(it, arm) for it in items if it["L"] >= 4096])}
    return res


def main() -> int:
    od = Path(sys.argv[1])
    srcs = [str(Path(p).resolve()) for p in sys.argv[2:]]
    out = {"script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
           "inputs": {p: hashlib.md5(Path(p).read_bytes()).hexdigest() for p in srcs},
           "models": [analyze(json.loads(Path(p).read_text(encoding="utf-8")), p) for p in srcs]}
    od.mkdir(parents=True, exist_ok=True)
    (od / "t4_b2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    def f(x, n=2):
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"
    L = []
    for m in out["models"]:
        L.append(f"== {m['model']} ({m['n_items']} items)")
        L.append("  pages (pooled over layers, KV heads, decode rows): dev / val")
        for arm, sp in m["pages"].items():
            for P in map(str, PAGES):
                for view in VIEWS:
                    d, v = sp["dev"][P][view], sp["val"][P][view]
                    if not d.get("pages"):
                        continue
                    k = f"k{max(2, int(P) // 4)}"
                    L.append(f"  {arm:16s} P={P:>2s} {view:4s} density {f(d['density'])}/{f(v['density'])}  "
                             f"overfetch {f(d['overfetch_whole'])}/{f(v['overfetch_whole'])}  "
                             f"{k}: rows {f(d[k]['rows_whole'])}/{f(v[k]['rows_whole'])} "
                             f"pages {f(d[k]['pages_whole'])}/{f(v[k]['pages_whole'])} "
                             f"of {f(d[k]['overfetch_whole_part'])}  reuse {f(d['reuse'])}/{f(v['reuse'])}")
        L.append("  answer-span argmax agreement with exact (item mean / pooled rows) and host bytes per record quantiles")
        for arm, a in m["argmax_answer"].items():
            bq = m["bytes_quantiles"][arm]["all"] or {}
            L.append(f"  {arm:18s} argmax {f(a['mean_item'], 4)} / {f(a['pooled_rows'], 4)} (n={a['items']})  "
                     f"bytes p10/p50/p90 {f(bq.get('p10'), 1)}/{f(bq.get('p50'), 1)}/{f(bq.get('p90'), 1)}")
    (od / "t4_b2.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
