# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""T4 analysis: per model and arm, quality vs host residual bytes, with paired item-level intervals.

usage: python -I t4_analyze.py <out_dir> <lab_json> [<lab_json> ...]
Writes <out_dir>/t4_analysis.json and t4_analysis.txt.

Per arm (items paired by trace id; skipped items dropped and counted):
- quality: decode-span KL (exact || arm) over thinking rows (answer rows excluded), the same on decode positions
  >= 4096 (long), answer-span KL, decode NLL excess; every mean is printed with its item count;
- host residual bytes per eligible record (token, layer, KV head outside the FP16 window and sinks), pooled over the
  item's decode rows (thinking AND answer rows: the records do not separate them): 152 B kept h54, 88 B h32, 84 B
  key-only, 0 B deleted; uniform arms are constant (h53 / h44 136, h43 120, h33 104, h32 88); the GPU base (2-bit KIVI, window,
  sinks) is identical in every arm and not counted;
- for demotion arms (same full decode span as the bytes): the share of the undemoted counterfactual's stage-2 final
  weight on tokens demoted here (T1's w2 inside this run), the measured stage-2 weight on demoted tokens, and stage-2
  value slots lost / dropped per (row, layer, KV head).
Comparisons (geometric mean of per-item KL ratios, item bootstrap 95% interval, unadjusted for multiplicity; KL is
floored at 1e-12 and the number of floored values is reported):
- every arm against h54;
- same-schedule controls (score vs random / oldest), with both arms' bytes printed;
- uniform frontier at equal bytes: per item, log KL of the uniform arms (h54 152, h53 / h44 136 (the lower of the
  two), h43 120, h33 104, h32 88, all_del 0)
  interpolated linearly in bytes at the arm's own bytes; ratio < 1 means the arm beats uniform re-quantization.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

REC_B = {"h54": 152.0, "h53": 136.0, "h44": 136.0, "h43": 120.0, "h33": 104.0, "h32": 88.0, "h32c": 88.0, "all_del": 0.0}
ACT_B = {"h32": 88.0, "key": 84.0, "del": 0.0}
KEEP_B = 152.0
NREC = 15
UNIFORM = ("h54", "h53", "h44", "h43", "h33", "h32", "all_del")
LONG = ("4096-8192", "8192-1073741824")
PAIRS = [("sc_G1024_del75", "rnd_G1024_del75"), ("sc_G1024_del75", "old_G1024_del75"),
         ("sc_G1024_h32_75", "rnd_G1024_h32_75"), ("h32", "h32c"), ("sc_G1024_h32_75", "h33")]
NBOOT, SEED, FLOOR = 2000, 12345, 1e-12


def long_kl(it: dict, arm: str) -> float | None:
    s, n = 0.0, 0
    for b in LONG:
        v = it["bucket"][arm].get(b)
        if v and v[0] is not None and v[1]:
            s += v[0] * v[1]
            n += v[1]
    return s / n if n else None


def rec_rows(it: dict, arm: str):
    lay = it["layers"].get(arm)
    if not lay:
        return None
    rows = [row for v in lay.values() for row in v["rec"]]
    for row in rows:
        if len(row) != NREC:
            raise SystemExit(f"item {it['i']} arm {arm}: record has {len(row)} columns, expected {NREC}")
    return rows


def bytes_per_record(it: dict, arm: str) -> float | None:
    if arm in REC_B:
        return REC_B[arm]
    rows = rec_rows(it, arm)
    if not rows:
        return None
    e = sum(r[1] for r in rows)
    d_h, d_k, d_d = (sum(r[j] for r in rows) for j in (2, 3, 4))
    if e <= 0:
        return None
    kept = e - d_h - d_k - d_d
    return (KEEP_B * kept + ACT_B["h32"] * d_h + ACT_B["key"] * d_k + ACT_B["del"] * d_d) / e


def mech(it: dict, arm: str) -> dict | None:
    rows = rec_rows(it, arm)
    if not rows:
        return None
    t = [sum(r[j] for r in rows) for j in range(NREC)]
    return {"w2_counterfactual": t[9] / t[10] if t[10] else None,
            "w2_measured": t[8] / t[7] if t[7] else None,
            "v_slots_lost_per_head_row": t[13] / t[0] if t[0] else None,
            "s2_dropped_per_head_row": t[6] / t[0] if t[0] else None}


def gmean_ratio(a: list, b: list) -> dict:
    """Geometric mean of a_i / b_i (KL floored at FLOOR) with a bootstrap 95% interval (items resampled)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"n": 0}
    floored = sum((x < FLOOR) + (y < FLOOR) for x, y in pairs)
    lr = [math.log(max(x, FLOOR) / max(y, FLOOR)) for x, y in pairs]
    rng = random.Random(SEED)
    boots = sorted(sum(rng.choice(lr) for _ in lr) / len(lr) for _ in range(NBOOT))
    lo, hi = boots[int(0.025 * NBOOT)], boots[int(0.975 * NBOOT) - 1]
    return {"n": len(pairs), "n_floored": floored, "ratio": math.exp(sum(lr) / len(lr)),
            "ci95": [math.exp(lo), math.exp(hi)], "share_a_higher": sum(1 for x in lr if x > 0) / len(lr)}


def frontier_kl(it_kl: dict, b: float) -> float | None:
    """log-KL of the uniform arms interpolated linearly in bytes at b (one item); None outside their byte range."""
    # uniform codecs at the same bytes (h44 and h53, both 136 B) collapse to one point: the lower log KL (the
    # strongest uniform baseline at that size)
    best: dict = {}
    for u in UNIFORM:
        if it_kl.get(u) is not None:
            lk = math.log(max(it_kl[u], FLOOR))
            best[REC_B[u]] = min(lk, best.get(REC_B[u], lk))
    pts = sorted(best.items())
    for (b0, l0), (b1, l1) in zip(pts, pts[1:]):
        if b0 <= b <= b1:
            return math.exp(l0 + (l1 - l0) * (b - b0) / (b1 - b0)) if b1 > b0 else math.exp(l0)
    return None


def mean_n(xs: list) -> tuple:
    v = [x for x in xs if x is not None]
    return (sum(v) / len(v) if v else None, len(v))


GROUPS = {"all": (0, 1 << 30), "short_L<2048": (0, 2048), "long_L>=4096": (4096, 1 << 30)}


def analyze(lab: dict, src: str, group: str = "all") -> dict:
    if lab.get("smoke") or not lab["runs"]:
        raise SystemExit(f"{src}: smoke output or no arms; nothing to analyze")
    if lab.get("head_dim", 128) != 128:
        raise SystemExit(f"{src}: head_dim {lab.get('head_dim')}; the byte constants assume 128")
    runs = lab["runs"]
    lo, hi = GROUPS[group]
    items = [it for it in lab["items"] if "skipped" not in it and lo <= it["L"] < hi]
    for it in items:
        for arm in runs:
            for key in ("bucket", "decode", "answer"):
                if arm not in it[key]:
                    raise SystemExit(f"{src}: item {it['i']} has no {key} for arm {arm}")
    warn = []
    if "h54" not in runs:
        warn.append("h54 not in runs: no x-h54 ratios")
    if not items:
        warn.append(f"no items in group {group}")
    out = {"model": Path(lab["model"]).name, "group": group, "source": src, "n_total": len(lab["items"]),
           "n_items": len(items),
           "skipped": {str(it["i"]): it["skipped"] for it in lab["items"] if "skipped" in it and lo <= it["L"] < hi}, "n_out_of_group": sum(1 for it in lab["items"] if not lo <= it["L"] < hi), "arms": {},
           "warnings": warn}
    met = {}
    for arm in runs:
        met[arm] = {"dec": [it["decode"][arm]["kl"] for it in items], "long": [long_kl(it, arm) for it in items],
                    "ans": [it["answer"][arm][0] for it in items],
                    "nll": [None if it["decode"][arm]["nll"] is None else
                            it["decode"][arm]["nll"] - it["decode"][arm]["nll_exact"] for it in items],
                    "bytes": [bytes_per_record(it, arm) for it in items]}
        mm = [m for m in (mech(it, arm) for it in items) if m]
        stats = {k: mean_n(met[arm][k]) for k in ("dec", "long", "ans", "nll", "bytes")}
        bpr = stats["bytes"][0]
        out["arms"][arm] = {
            "kl_decode_mean": stats["dec"][0], "kl_long_mean": stats["long"][0], "kl_answer_mean": stats["ans"][0],
            "nll_excess_mean": stats["nll"][0], "bytes_per_record": bpr,
            "n": {k: stats[k][1] for k in stats},
            "bytes_saving_vs_h54": None if bpr is None else 1 - bpr / KEEP_B,
            "mech": {k: mean_n([m[k] for m in mm])[0] for k in mm[0]} if mm else None}
        if arm != "h54" and "h54" in met:
            out["arms"][arm]["vs_h54_decode"] = gmean_ratio(met[arm]["dec"], met["h54"]["dec"])
            out["arms"][arm]["vs_h54_long"] = gmean_ratio(met[arm]["long"], met["h54"]["long"])
    out["n_long_items"] = out["arms"][runs[0]]["n"]["long"] if out["arms"] else 0
    pairs = {}
    for a, b in PAIRS:
        if a not in met or b not in met:
            if a in met or b in met:
                warn.append(f"pair {a}/{b}: one arm not in runs")
            continue
        pairs[f"{a}/{b}"] = {"decode": gmean_ratio(met[a]["dec"], met[b]["dec"]),
                             "long": gmean_ratio(met[a]["long"], met[b]["long"]),
                             "bytes": [out["arms"][a]["bytes_per_record"], out["arms"][b]["bytes_per_record"]]}
    out["pairs"] = pairs
    have_u = [u for u in UNIFORM if u in met]
    out["frontier"] = {}
    if len(have_u) >= 2:
        for arm in runs:
            if arm in REC_B:
                continue
            fr = {"dec": [], "long": []}
            for j in range(len(items)):
                b = met[arm]["bytes"][j]
                for k in ("dec", "long"):
                    ref = None if b is None else frontier_kl({u: met[u][k][j] for u in have_u}, b)
                    fr[k].append(ref)
            out["frontier"][arm] = {"uniform_arms": have_u,
                                    "decode": gmean_ratio(met[arm]["dec"], fr["dec"]),
                                    "long": gmean_ratio(met[arm]["long"], fr["long"])}
    else:
        warn.append("fewer than two uniform arms: no equal-bytes frontier comparison")
    return out


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    od = Path(sys.argv[1])
    srcs = [str(Path(p).resolve()) for p in sys.argv[2:]]
    labs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in srcs]
    res = {"script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
           "inputs": {p: hashlib.md5(Path(p).read_bytes()).hexdigest() for p in srcs},
           "models": [analyze(lab, p, g) for lab, p in zip(labs, srcs) for g in GROUPS]}
    od.mkdir(parents=True, exist_ok=True)
    (od / "t4_analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    def f(x, n=4):
        return "-" if x is None else f"{x:.{n}f}"

    def ci(r: dict) -> str:
        if not r.get("n"):
            return "-"
        return f"{f(r['ratio'], 2)} [{f(r['ci95'][0], 2)},{f(r['ci95'][1], 2)}] n={r['n']}"
    lines = []
    for m in res["models"]:
        lines.append(f"== {m['model']} [{m['group']}] ({Path(m['source']).name}): "
                     f"{m['n_items']} of {m['n_total']} items analyzed ({m['n_out_of_group']} outside this group), "
                     f"skipped {m['skipped'] or 'none'}; {m['n_long_items']} items with decode positions >= 4096")
        lines += [f"  WARNING: {w}" for w in m["warnings"]]
        lines.append("  bytes / w2 / Vlost: full decode span (thinking + answer); KL dec / long: thinking rows only")
        lines.append(f"  {'arm':18s} {'B/rec':>6s} {'save':>6s} {'KL dec':>9s} {'KL long':>9s} {'KL ans(n)':>13s} "
                     f"{'dNLL':>8s} {'x h54 dec [95%]':>26s} {'x h54 long':>26s} {'w2cf':>6s} {'w2':>6s} "
                     f"{'Vlost/hr':>8s}")
        for arm, v in m["arms"].items():
            mc = v.get("mech") or {}
            lines.append(f"  {arm:18s} {f(v['bytes_per_record'], 1):>6s} {f(v['bytes_saving_vs_h54'], 3):>6s} "
                         f"{f(v['kl_decode_mean'], 5):>9s} {f(v['kl_long_mean'], 5):>9s} "
                         f"{f(v['kl_answer_mean'], 5) + '(' + str(v['n']['ans']) + ')':>13s} "
                         f"{f(v['nll_excess_mean'], 4):>8s} {ci(v.get('vs_h54_decode', {})):>26s} "
                         f"{ci(v.get('vs_h54_long', {})):>26s} {f(mc.get('w2_counterfactual'), 3):>6s} "
                         f"{f(mc.get('w2_measured'), 3):>6s} {f(mc.get('v_slots_lost_per_head_row'), 2):>8s}")
        for k, v in m["pairs"].items():
            ba, bb = v["bytes"]
            flag = "  bytes n/a" if ba is None or bb is None else "" if ba is not None and bb is not None and abs(ba - bb) <= 0.01 * max(ba, bb, 1e-9) \
                else "  NOT equal bytes"
            lines.append(f"  pair {k}: decode {ci(v['decode'])}  long {ci(v['long'])}  "
                         f"bytes {f(ba, 1)} vs {f(bb, 1)}{flag}  a>b in {f(v['decode'].get('share_a_higher'), 2)}")
        for k, v in m["frontier"].items():
            lines.append(f"  vs uniform frontier at equal bytes {k}: decode {ci(v['decode'])}  long {ci(v['long'])}")
        fl = sum(r.get("n_floored", 0) for v in m["arms"].values() for r in (v.get("vs_h54_decode", {}),
                                                                             v.get("vs_h54_long", {})))
        lines.append(f"  KL values floored at {FLOOR} in x-h54 ratios: {fl}; intervals unadjusted for multiplicity")
    (od / "t4_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
