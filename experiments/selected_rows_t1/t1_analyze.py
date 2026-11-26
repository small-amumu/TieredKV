# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""T1 scorer (REG_T1.md v5 sections 3-4). Usage: python -I t1_analyze.py <out_dir> <lab json> x3

Per (item, layer): pooled ratios of sums -> mean over layers -> median over items, per model.
G1 and G2 are evaluated on the long bucket (decode positions [4096, 8800)); everything else is descriptive.
Writes <out_dir>/t1_analysis.json (NaN written as null) and t1_analysis.txt.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

N_ITEMS = 64
LONG = 3                                    # bucket index of [4096, 8800)
MIN_LONG_ITEMS = 16
G1_GF = tuple((g, f) for g in (512, 1024) for f in (0.75, 0.9))
G1_MAX_W2 = 0.10
G1_MIN_DEMOTED = 0.5                        # schedule sanity check: fixed by (G, f, length), not by the data
SCORE_VS_OLDEST = 0.8                       # score is preferred over oldest-first only if w2 <= 0.8 x oldest
G2_THRESHOLD = 2.30
G2_RANDOM_FACTOR = 1.5
MODEL_KEYS = ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B")
SAME_KEYS = ("run", "chunk", "max_len", "policies", "pages", "buckets", "probation", "script_md5", "stack_modules_md5")
CODE_MD5 = ("t1_lab.py", "t1_q.py", "sel14_q.py", "sel30_q.py", "v8g.py", "bl_q.py", "cq_lab3.py")
COL = {"n1": 0, "n2": 1, "c1cnt": 2, "c2cnt": 3, "c1D": 4, "c1all": 5, "prD": 6, "prall": 7, "D": 8, "E": 9}


def ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def strict_median(v: list, what: str) -> float:
    if any(not math.isfinite(x) for x in v):
        raise SystemExit(f"{what}: non-finite item value (Q0e)")
    return st.median(v) if v else float("nan")


class NoRows(Exception):
    pass


def layer_mean(it: dict, fn) -> float:
    """Mean over layers; every layer must be finite. Raises NoRows when no layer has rows (bucket not reached)."""
    vals = [fn(lv) for lv in it["layers"].values()]
    fin = [x for x in vals if math.isfinite(x)]
    if not fin:
        raise NoRows
    if len(fin) != len(vals):
        raise SystemExit(f"item {it['i']}: some layers non-finite and others not (Q0e)")
    return sum(fin) / len(fin)


def pol_ratio(pn: str, a: str, b: str, bucket=None):
    def f(lv):
        rows = lv["pol"][pn] if bucket is None else [lv["pol"][pn][bucket]]
        return ratio(sum(r[COL[a]] for r in rows), sum(r[COL[b]] for r in rows))
    return f


def page_ratio(P: int, stage: str, rnd: bool, bucket=None):
    c0 = 2 if rnd else 0

    def f(lv):
        rows = lv["pages"][str(P)][stage] if bucket is None else [lv["pages"][str(P)][stage][bucket]]
        return ratio(sum(r[c0] for r in rows), sum(r[c0 + 1] for r in rows))
    return f


def stage2_rows(it: dict, bucket) -> float:
    """Stage-2 row count of an item (policy-independent), over one bucket or all buckets."""
    tot = 0.0
    for lv in it["layers"].values():
        rows = next(iter(lv["pol"].values()))
        tot += sum(r[COL["c2cnt"]] for r in (rows if bucket is None else [rows[bucket]]))
    return tot


def per_items(items: list, fn, check: bool = False, bucket=None) -> list:
    """Per-item layer means. Items with no rows in scope are skipped; with check=True an item that has stage-2
    rows in scope but a NaN ratio in every layer is a Q0e failure instead of a silent drop."""
    out = []
    for it in items:
        try:
            out.append(layer_mean(it, fn))
        except NoRows:
            if check and stage2_rows(it, bucket) > 0:
                raise SystemExit(f"item {it['i']}: stage-2 rows in scope but every layer's ratio is NaN (Q0e)")
    return out


def boot_ci(v: list, seed: int = 0, n: int = 2000) -> list:
    if not v:
        return [None, None]
    rng = random.Random(seed)
    meds = sorted(st.median([rng.choice(v) for _ in v]) for _ in range(n))
    return [meds[int(0.025 * n)], meds[int(0.975 * n) - 1]]


def analyze(lab: dict) -> dict:
    if lab.get("smoke"):
        raise SystemExit(f"{lab['model']}: smoke output given to the scorer")
    items = [it for it in lab["items"] if "layers" in it]
    ids = sorted(it["i"] for it in lab["items"])
    if ids != list(range(N_ITEMS)) or len(items) != N_ITEMS:
        raise SystemExit(f"{lab['model']}: item ids must be 0..63 and all with statistics (Q0e)")
    n_layers = len(lab["attn_layers"])
    for it in items:
        if len(it["layers"]) != n_layers:
            raise SystemExit(f"{lab['model']} item {it['i']}: {len(it['layers'])} layers, expected {n_layers} (Q0e)")
    nb = len(lab["buckets"])
    name = Path(lab["model"]).name
    out: dict = {"model": lab["model"], "n_items": len(items),
                 "n_truncated": sum(bool(it.get("truncated_by_max_len")) for it in lab["items"]),
                 "prompt_len_median": st.median(it["prompt"] for it in lab["items"]),
                 "L_median": st.median(it["L"] for it in lab["items"]), "policies": {}, "pages": {}}
    n_long = sum(stage2_rows(it, LONG) > 0 for it in items)
    out["n_long_items"] = n_long

    def long_vals(fn, what: str) -> list:
        v = per_items(items, fn, True, LONG)
        if len(v) != n_long:
            raise SystemExit(f"{name} {what}: {len(v)} long-bucket items, n_long_items {n_long} (Q0e)")
        return v
    for pn in lab["policies"]:
        r: dict = {}
        for key, a, b in (("w2", "prD", "prall"), ("n2", "n2", "c2cnt"), ("n1", "n1", "c1cnt"),
                          ("c1_mass", "c1D", "c1all"), ("demoted", "D", "E")):
            allv = per_items(items, pol_ratio(pn, a, b), True, None)
            r[f"{key}_all"] = strict_median(allv, f"{name} {pn} {key}")
            r[f"{key}_by_bucket"] = [strict_median(per_items(items, pol_ratio(pn, a, b, bi), True, bi),
                                                   f"{name} {pn} {key} b{bi}") for bi in range(nb)]
            if key in ("w2", "demoted", "c1_mass"):
                lv_ = long_vals(pol_ratio(pn, a, b, LONG), f"{pn} {key}")
                r[f"{key}_long"] = strict_median(lv_, f"{name} {pn} {key} long")
                if key == "w2":
                    r["w2_long_ci95"] = boot_ci(lv_)
        if pn.startswith("score"):                     # descriptive, whole trajectory
            za = per_items(items, lambda lv: ratio(lv.get("zero_score_demoted", {}).get(pn, 0.0),
                                                   lv.get("demoted_total", {}).get(pn, 0.0)))
            r["zero_score_share_of_demoted_all"] = st.median(za) if za else float("nan")
        out["policies"][pn] = r
    for P in lab["pages"]:
        for s in ("s1", "s2"):
            out["pages"][f"P{P}_{s}_long"] = strict_median(long_vals(page_ratio(P, s, False, LONG), "pages"),
                                                           f"{name} pages")
            out["pages"][f"P{P}_{s}_long_random"] = strict_median(long_vals(page_ratio(P, s, True, LONG), "pages rnd"),
                                                                  f"{name} pages random")
            out["pages"][f"P{P}_{s}_all"] = strict_median(per_items(items, page_ratio(P, s, False), True, None),
                                                          f"{name} pages all")
    out["churn"] = strict_median(per_items(items, lambda lv: ratio(sum(b[0] for b in lv["churn"]),
                                                                   sum(b[1] for b in lv["churn"]))), f"{name} churn")
    out["union_by_bucket"] = [st.median(v) if v else None for v in
                              (per_items(items, lambda lv, bi=bi: ratio(lv["union"][bi][0], lv["union"][bi][1]))
                               for bi in range(nb))]
    return out


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def main() -> int:
    od = Path(sys.argv[1])
    labs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[2:]]
    names = [Path(lab["model"]).name for lab in labs]
    if sorted(names) != sorted(MODEL_KEYS):
        raise SystemExit(f"the scorer needs exactly the three registered models {MODEL_KEYS}, got {names}")
    for k in SAME_KEYS:
        if any(lab.get(k) != labs[0].get(k) for lab in labs):
            raise SystemExit(f"labs differ in {k} (Q0e)")
    for k in CODE_MD5:
        if any(not isinstance(lab.get("md5"), dict) or k not in lab["md5"] for lab in labs):
            raise SystemExit(f"md5 of {k} missing in a lab (Q0e)")
        if any(lab["md5"].get(k) != labs[0]["md5"].get(k) for lab in labs):
            raise SystemExit(f"labs differ in md5 of {k} (Q0e)")
    res = {"script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
           "models": [analyze(lab) for lab in labs]}
    def ok(m: dict, pn: str) -> bool:
        p = m["policies"][pn]
        return (m["n_long_items"] >= MIN_LONG_ITEMS and math.isfinite(p["w2_long"]) and p["w2_long"] <= G1_MAX_W2
                and math.isfinite(p["demoted_long"]) and p["demoted_long"] >= G1_MIN_DEMOTED)
    g1 = {}
    for g, f in G1_GF:
        tail = f"_G{g}_f{f}"
        detail = []
        for m in res["models"]:
            ps, po, pr = (m["policies"][k + tail] for k in ("score", "oldest", "random"))
            beats = (math.isfinite(ps["w2_long"]) and math.isfinite(po["w2_long"])
                     and ps["w2_long"] <= SCORE_VS_OLDEST * po["w2_long"])
            detail.append({"model": Path(m["model"]).name, "score_w2_long": ps["w2_long"],
                           "oldest_w2_long": po["w2_long"], "random_w2_long": pr["w2_long"],
                           "score_c1_mass_long": ps["c1_mass_long"], "oldest_c1_mass_long": po["c1_mass_long"],
                           "demoted_long": ps["demoted_long"], "n_long_items": m["n_long_items"],
                           "score_ok": ok(m, "score" + tail), "oldest_ok": ok(m, "oldest" + tail),
                           "random_ok": ok(m, "random" + tail),
                           "score_beats_oldest": beats})
        g1[tail[1:]] = {"G": g, "f": f, "score_ok": all(d["score_ok"] for d in detail),
                        "oldest_ok": all(d["oldest_ok"] for d in detail),
                        "random_ok": all(d["random_ok"] for d in detail),
                        "score_beats_oldest": all(d["score_beats_oldest"] for d in detail), "detail": detail}
        v = g1[tail[1:]]
        v["score_selectable"] = v["score_ok"] and v["score_beats_oldest"]   # no escape when oldest fails
        v["usable"] = v["score_selectable"] or v["oldest_ok"]
    passing = sorted((v for v in g1.values() if v["usable"]), key=lambda v: (-v["f"], v["G"]))
    pick = None
    random_also_ok = None
    if passing:
        v = passing[0]
        pick = f"{'score' if v['score_selectable'] else 'oldest'}_G{v['G']}_f{v['f']}"
        random_also_ok = v["random_ok"]
    res["G1"] = {"per_setting": g1, "pass": bool(passing), "t4_policy": pick,
                 "random_also_ok_at_pick": random_also_ok}
    g2d = []
    for m in res["models"]:
        v, rv = m["pages"]["P16_s1_long"], m["pages"]["P16_s1_long_random"]
        good = (m["n_long_items"] >= MIN_LONG_ITEMS and math.isfinite(v) and v >= G2_THRESHOLD
                and v >= G2_RANDOM_FACTOR * rv)
        g2d.append({"model": Path(m["model"]).name, "rows_per_page_long": v, "random_long": rv,
                    "n_long_items": m["n_long_items"], "ok": good})
    res["G2"] = {"pass": all(d["ok"] for d in g2d), "detail": g2d}
    (od / "t1_analysis.json").write_text(json.dumps(clean(res), indent=1, allow_nan=False))
    lines = []

    def f4(x):
        return "nan" if x is None or not math.isfinite(x) else f"{x:.4f}"
    for m in res["models"]:
        lines.append(f"== {Path(m['model']).name}: {m['n_items']} items, {m['n_long_items']} reach the long bucket, "
                     f"{m['n_truncated']} truncated, prompt median {m['prompt_len_median']}, L median {m['L_median']}")
        for pn, v in m["policies"].items():
            lines.append(f"  {pn:20s} long: w2 {f4(v['w2_long'])} c1 {f4(v['c1_mass_long'])} "
                         f"demoted {f4(v['demoted_long'])} | all: w2 "
                         f"{f4(v['w2_all'])} n2 {f4(v['n2_all'])} n1 {f4(v['n1_all'])} demoted {f4(v['demoted_all'])}")
        lines.append("  pages " + " ".join(f"{k} {f4(v)}" for k, v in m["pages"].items()))
        lines.append(f"  churn {f4(m['churn'])} of 128; union by bucket {m['union_by_bucket']}")
    lines.append(f"G1 pass {res['G1']['pass']} -> T4 policy {res['G1']['t4_policy']} "
                 f"(random also within threshold at this setting: {res['G1']['random_also_ok_at_pick']})")
    for k, v in g1.items():
        lines.append(f"  {k}: score_ok {v['score_ok']} oldest_ok {v['oldest_ok']} random_ok {v['random_ok']} "
                     f"beats {v['score_beats_oldest']} | " +
                     "; ".join(f"{d['model']} w2 score {f4(d['score_w2_long'])} oldest {f4(d['oldest_w2_long'])} "
                               f"random {f4(d['random_w2_long'])} c1 {f4(d['score_c1_mass_long'])} "
                               f"demoted {f4(d['demoted_long'])} n={d['n_long_items']}" for d in v["detail"]))
    lines.append(f"G2 pass {res['G2']['pass']}: " + "; ".join(
        f"{d['model']} {f4(d['rows_per_page_long'])} (random {f4(d['random_long'])}, n={d['n_long_items']})"
        for d in g2d))
    (od / "t1_analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
