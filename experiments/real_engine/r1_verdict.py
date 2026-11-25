# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""r1_verdict (TieredKV, round R1; 2026-11-10, v2 after 3 reviews): the registered gates and predictions of REGISTRATION_R1.md,
computed from the rp2_tf.py outputs of R1 (LLaMA-2-7B-32K, MHA; out/r1_<task>.jsonl) and the archived X-0139 records
(tf_g0a / tf_g2a.jsonl). Read-only; prints a JSON summary.

Completeness (Q0, checked first; any failure fails A4 and blocks P1 / P2):
  per task file: a head meta record with task == the file's task, data_md5 == the registered md5, items "0:10", maxgen as tb_bench,
  md5 of rp2_tf.py / rp_engine.py as registered and every EXPECTED_MD5.json entry equal; records with task == the file's task;
  items 0-9 exactly once; one end record with n_errors 0; every line valid JSON. Each task has >= 8 usable items.
  rc logs (both required): exactly one line "r1_<task> rc=<n>" per task with n == 0, and "chunk_test_a / _b rc=0 CHUNK_TEST PASS".
Exclusions: an item whose fp trajectory has < 2 tokens or whose gold answer is outside [2, maxgen] is excluded as tb_bench does
(benign, listed); an item or arm error, an engine flag or a non-finite value is an error (A4 fails).

Matching with X-0139 (used by A0, P1, P3): a usable item matches when L0 == X-0139 fp len, tf_len == fp tf_steps + 1,
gold_len == fp gold_steps + 1, |fp gold_nll - X-0139 fp gold_nll| <= 0.01 and |fp tf_nll - X-0139 fp tf_nll| <= 2e-3 (the fp
trajectory's own NLL: a fingerprint of the trajectory). Evaluable when >= 90% of the usable items match.

Part 1, the reference and the engine (per width w in {33, 54}, on the usable items):
  A0  the reference reproduces X-0139 (same tree, same machine): on the matched items |mean tf_kl(ref_w) - mean tf_kl(X h_w)|
      <= 1e-3 and |mean dNLL(ref_w vs own fp) - mean dNLL(X h_w vs X fp)| <= 0.0033; not evaluable -> A0 fails.
  A1  argmax(eng_w) == argmax(ref_w) on >= 98% of the trajectory steps (pooled; per task reported).
  A2  95% item-bootstrap interval (2000 draws) of mean[gold_nll(eng_w) - gold_nll(ref_w)] inside +-0.01.
  A3  95% item-bootstrap interval of mean[tf_kl_mean(eng_w) - tf_kl_mean(ref_w)] inside +-1e-3.
  A4  completeness as above; no item / arm error, no engine flag, no reference flag, every value finite.
  A5  max over trajectory steps of KL(ref_w || eng_w) <= max(0.25 * max KL(fp || ref_w), 0.002).
  informative: mean tf_kl_mean(ref_w) >= 5e-4, else INCONCLUSIVE.
  Part 1 PASS = A0-A5 and informative for both widths.
  reported, not gated: sensitivity eng33 vs ref54 / eng54 vs ref33 (A2 / A3); A2 / A3 without the item of largest ref KL.
Part 2, Table A with the engine's numbers (cluster bootstrap as long_analyze.py: items resampled within task, mean of task means,
10000 draws). P1 / P2 hold only on complete data (A4 conditions met).
  P1  (sanity replication; Table A: h54 / h33 clearly better than kivi128 and kvq2n) on the matched items, for w in {33, 54} and
      b in {kivi128, kvq2n}: the 95% intervals of tf_kl_mean(eng_w) - tf_kl_mean(b) and of [gold dNLL(eng_w vs own fp) -
      gold dNLL(b vs X-0139 fp)] all lie below 0 (8 intervals, a conjunction: no multiplicity correction needed).
  P2  (Table A: h54 equivalent to h33) gold_nll(eng33) - gold_nll(eng54): 90% interval inside +-0.01 (TOST 5% / side).
      KL counterpart (descriptive): 90% interval of tf_kl(eng33) - tf_kl(eng54) against +-1e-3.
  P3  (descriptive; Table A: h54 and kvq3 not separable) 95% intervals of tf_kl(eng54) - tf_kl(kvq3) and of the gold dNLL
      difference (within-run deltas), on the matched items.
  descriptive: the Table A row for ref and eng (tf KL mean of task means / item mean / item median; gold dNLL vs fp [95%];
  perplexity factor; per task); Table A's simulated values; the reference and engine against X-0139 on the matched items.
Judgement inputs: criteria (i)-(iii) of the registration, computed (the judgement itself is made by the analyst).
Every interval has its own seed (SEED + 100 + a fixed offset listed in SEEDS).
Usage: python3 -I r1_verdict.py <out dir> --rc <rc_a.log> <rc_b.log> --x0139 <tf_g0a.jsonl> <tf_g2a.jsonl> [--expected EXPECTED_MD5.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re

M_NLL, M_KL, INFO_MIN, AGREE_MIN = 0.01, 1e-3, 5e-4, 0.98
MATCH_NLL, MATCH_TF, MATCH_MIN, MIN_USABLE = 0.01, 2e-3, 0.90, 8
J_KL, J_NLL, J_TASK_KL, J_TASK_NLL = 1e-3, 0.0033, 3e-3, 0.01   # judgement criteria (i) / (iii); (ii) widens P2 by J_NLL
SEED, NB_ITEM, NB_CLUSTER = 20261120, 2000, 10000
W = ("33", "54")
TASKS = ("niah_single_3", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt")
DATA_MD5 = {"niah_single_3": "744ac2c5be94f6e7a9c68605e6a6cfef", "niah_multikey_2": "8fdd6127ce098f53a7b5f3892de734f7",
            "niah_multikey_3": "9661e424b48c1ce753f3c9408a0b0632", "niah_multivalue": "20c23af3ce3ac4f1f35d464e0352255b",
            "niah_multiquery": "5e1ac06d17c776603d791b3fc991588b", "vt": "6f76ebd82ff04bc18864060f6b3d3f11"}
MAXGEN = {"vt": 30}                                     # tb_bench v3.7; niah tasks 128
ITEMS = tuple(range(10))
RP2_MD5 = "f32e0bf44a71cdd50f6bb9e59b4eb5ea"
ENG_MD5 = "48545123140a1eb4b734b9b86a775f9c"
ARMS = [f"{p}{w}" for p in ("ref", "eng") for w in W]
PFX = "stkm8v8128k64_x2s128_p50_"
SEEDS = {"A2_33": 0, "A3_33": 1, "A2_54": 2, "A3_54": 3, "sens_33v54_n": 4, "sens_33v54_k": 5, "sens_54v33_n": 6,
         "sens_54v33_k": 7, "lo_33_n": 8, "lo_33_k": 9, "lo_54_n": 10, "lo_54_k": 11, "P2": 12, "P2_kl": 13, "P3_kl": 14,
         "P3_nll": 15, **{f"P1_{w}_{b}_{m}": 16 + 4 * j + 2 * (b == "kvq2n") + (m == "nll")
                          for j, w in enumerate(W) for b in ("kivi128", "kvq2n") for m in ("kl", "nll")},
         **{f"table_{a}": 30 + j for j, a in enumerate(ARMS)}}
assert len(set(SEEDS.values())) == len(SEEDS), "seed offsets must be distinct"


def sd(name):
    return SEED + 100 + SEEDS[name]


def load(d, expected):
    by_task, problems = {}, []
    for t in TASKS:
        f = os.path.join(d, f"r1_{t}.jsonl")
        if not os.path.exists(f):
            problems.append(f"{t}: file missing")
            continue
        recs, metas = [], []
        for n_line, line in enumerate(open(f)):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{t}: line {n_line + 1} is not valid JSON")
                continue
            (metas if r.get("meta") else recs).append(r)
        head = [m for m in metas if not m.get("end")]
        end = [m for m in metas if m.get("end")]
        if len(head) != 1:
            problems.append(f"{t}: {len(head)} head meta records")
        h = head[0] if head else {}
        m5 = h.get("md5", {})
        if m5.get("engine/rp2_tf.py") != RP2_MD5 or m5.get("engine/rp_engine.py") != ENG_MD5:
            problems.append(f"{t}: rp2_tf.py / rp_engine.py md5 differs")
        bad = {k: (m5.get(k), v) for k, v in expected.items() if m5.get(k) != v}
        if bad:
            problems.append(f"{t}: reference / kernel md5 differ from EXPECTED_MD5: {sorted(bad)}")
        if (h.get("task") != t or h.get("data_md5") != DATA_MD5[t] or h.get("items") != "0:10"
                or h.get("maxgen") != MAXGEN.get(t, 128) or not str(h.get("model", "")).endswith("LLaMA-2-7B-32K")):
            problems.append(f"{t}: head meta task / data_md5 / items / maxgen / model differ from the registration")
        if len(end) != 1 or end[0].get("n_errors") != 0:
            problems.append(f"{t}: end record missing or n_errors != 0")
        if any(r.get("task") != t for r in recs):
            problems.append(f"{t}: a record carries another task")
        idx = sorted(r.get("i") for r in recs)
        if idx != list(ITEMS):
            problems.append(f"{t}: items {idx} != {list(ITEMS)}")
        by_task[t] = [r for r in recs if r.get("task") == t]
    return by_task, problems


def rc_problems(paths):
    problems, seen = [], {}
    lines = []
    for p in paths:
        if not os.path.exists(p):
            problems.append(f"rc log {p} missing")
            continue
        lines += open(p).read().splitlines()
    for ln in lines:
        m = re.search(r" r1_(\w+) rc=(\d+)(?:\s|$)", ln)
        if m:
            seen.setdefault(m.group(1), []).append(int(m.group(2)))
    for t in TASKS:
        if seen.get(t) != [0]:
            problems.append(f"rc logs: r1_{t} rc lines {seen.get(t)} (need exactly one rc=0)")
    for g in ("a", "b"):
        if not any(re.search(rf" chunk_test_{g} rc=0 CHUNK_TEST PASS$", ln) for ln in lines):
            problems.append(f"rc logs: chunk test {g} not PASS")
    return problems


def classify(r):
    """'ok', 'excluded: <why>' (benign, as tb_bench) or 'error: <why>'."""
    if r.get("status") != "ok":
        return "error: item " + str(r.get("status", ""))[:80]
    for a in ARMS:
        x = r.get(a, {})
        if x.get("status", "missing") != "ok":
            return f"error: {a} " + str(x.get("status", "missing"))[:80]
    if r.get("tf_status") != "ok":
        return "excluded: " + str(r.get("tf_status"))
    if r.get("gold_status") != "ok":
        return "excluded: " + str(r.get("gold_status"))
    for a in ARMS:
        if not (r[a].get("tf_finite") and r[a].get("gold_finite")):
            return f"error: {a} non-finite"
    vals = [r.get("fp", {}).get("tf_nll"), r.get("fp", {}).get("gold_nll")] + [r[f"eng{w}"].get("tf_kl_ref_max") for w in W]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        return "error: fp NLL or engine-reference KL missing or non-finite"
    return "ok"


def boot_item(vals, seed):
    if not vals:
        return float("nan"), float("nan")
    rnd = random.Random(seed)
    n = len(vals)
    ms = sorted(sum(vals[rnd.randrange(n)] for _ in range(n)) / n for _ in range(NB_ITEM))
    return ms[int(0.025 * NB_ITEM)], ms[min(NB_ITEM - 1, int(0.975 * NB_ITEM))]


def boot_cluster(by_task, lo, hi, seed):
    by_task = {t: v for t, v in by_task.items() if v}
    if not by_task:
        return float("nan"), float("nan"), float("nan")
    rnd = random.Random(seed)
    stat = lambda d: sum(sum(v) / len(v) for v in d.values()) / len(d)  # noqa: E731
    draws = sorted(stat({t: [v[rnd.randrange(len(v))] for _ in v] for t, v in by_task.items()}) for _ in range(NB_CLUSTER))
    return stat(by_task), draws[int(lo * NB_CLUSTER)], draws[min(NB_CLUSTER - 1, int(hi * NB_CLUSTER))]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    s = sorted(xs)
    n = len(s)
    return float("nan") if not n else s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def pair(R, ref_w, eng_w, seed_n, seed_k):
    dn = [r[f"eng{eng_w}"]["gold_nll"] - r[f"ref{ref_w}"]["gold_nll"] for r in R]
    dk = [r[f"eng{eng_w}"]["tf_kl_mean"] - r[f"ref{ref_w}"]["tf_kl_mean"] for r in R]
    ci_n, ci_k = boot_item(dn, seed_n), boot_item(dk, seed_k)
    return dict(mean_dnll=mean(dn), ci_dnll=ci_n, A2=-M_NLL <= ci_n[0] and ci_n[1] <= M_NLL,
                mean_dkl=mean(dk), ci_dkl=ci_k, A3=-M_KL <= ci_k[0] and ci_k[1] <= M_KL)


def load_x0139(paths):
    X = {}
    for p in paths:
        for line in open(p):
            x = json.loads(line)
            key = (x["arm"].replace(PFX, ""), x["task"].split(":")[1].split("@")[0], x["i"])
            assert key not in X, f"duplicate X-0139 record {key}"
            X[key] = x
    return X


def x_ok(X, arm, r):
    x = X.get((arm, r["task"], r["i"]), {})
    return x.get("tf_status") == "ok" and x.get("gold_status") == "ok"


def matches(r, X):
    if not x_ok(X, "fp", r):
        return False
    f = X[("fp", r["task"], r["i"])]
    return (r["L0"] == f["len"] and r["tf_len"] == f["tf_steps"] + 1 and r["gold_len"] == f["gold_steps"] + 1
            and abs(r["fp"]["gold_nll"] - f["gold_nll"]) <= MATCH_NLL and f.get("tf_nll") is not None
            and abs(r["fp"]["tf_nll"] - f["tf_nll"]) <= MATCH_TF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rc", required=True, nargs=2)
    ap.add_argument("--x0139", required=True, nargs=2)
    ap.add_argument("--expected", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "EXPECTED_MD5.json"))
    a = ap.parse_args()
    expected = json.load(open(a.expected))
    by_task, problems = load(a.dir, expected)
    problems += rc_problems(a.rc)
    R_all = [r for t in TASKS for r in by_task.get(t, [])]
    keys = [(r["task"], r["i"]) for r in R_all]
    if len(keys) != len(set(keys)):
        problems.append("duplicate (task, item) records")
    cls = {(r["task"], r["i"]): classify(r) for r in R_all}
    R = [r for r in R_all if cls[(r["task"], r["i"])] == "ok"]
    for t in TASKS:
        n_ok = sum(r["task"] == t for r in R)
        if n_ok < MIN_USABLE:
            problems.append(f"{t}: {n_ok} usable items < {MIN_USABLE}")
    excluded = sorted([k + (v,) for k, v in cls.items() if v.startswith("excluded")])
    errors = sorted([k + (v,) for k, v in cls.items() if v.startswith("error")])
    complete = not problems and not errors
    out = dict(n_records=len(R_all), n_usable=len(R), excluded=excluded, errors=errors, completeness_problems=problems,
               complete=complete, margins=dict(M_NLL=M_NLL, M_KL=M_KL, INFO_MIN=INFO_MIN, AGREE_MIN=AGREE_MIN,
                                               MATCH_NLL=MATCH_NLL, MATCH_TF=MATCH_TF, MATCH_MIN=MATCH_MIN), seeds={k: sd(k) for k in SEEDS})
    if not R:
        out["part1"] = dict(PASS=False, reason="no usable item")
        print(json.dumps(out, indent=1))
        return
    tasks_used = [t for t in TASKS if any(r["task"] == t for r in R)]
    X = load_x0139(a.x0139)
    M = [r for r in R if matches(r, X)]
    frac = len(M) / len(R)
    evaluable = frac >= MATCH_MIN
    xa = lambda r, arm, key: X[(arm, r["task"], r["i"])][key]  # noqa: E731
    byR = lambda RR_, f: {t: [f(r) for r in RR_ if r["task"] == t] for t in tasks_used}  # noqa: E731
    part1 = {}
    for w in W:
        agree = sum(r[f"eng{w}"]["tf_agree_ref"] * r[f"eng{w}"]["tf_steps"] for r in R) / sum(r[f"eng{w}"]["tf_steps"] for r in R)
        per_task = {}
        for t in tasks_used:
            Rt = [r for r in R if r["task"] == t]
            per_task[t] = sum(r[f"eng{w}"]["tf_agree_ref"] * r[f"eng{w}"]["tf_steps"] for r in Rt) / sum(r[f"eng{w}"]["tf_steps"] for r in Rt)
        p = pair(R, w, w, sd(f"A2_{w}"), sd(f"A3_{w}"))
        flags_ref = sum(r[f"ref{w}"].get("tf_flags", 0) + r[f"ref{w}"].get("gold_flags", 0) for r in R)
        A4 = complete and flags_ref == 0
        info = mean([r[f"ref{w}"]["tf_kl_mean"] for r in R])
        max_fr = max(max(r[f"ref{w}"]["tf_kl_steps"]) for r in R)
        max_re = max(r[f"eng{w}"]["tf_kl_ref_max"] for r in R)
        A5 = max_re <= max(0.25 * max_fr, 0.002)
        Mw = [r for r in M if x_ok(X, f"h{w}", r)]
        a0_kl = mean([r[f"ref{w}"]["tf_kl_mean"] for r in Mw]) - mean([xa(r, f"h{w}", "tf_kl_mean") for r in Mw])
        a0_nll = (mean([r[f"ref{w}"]["gold_nll"] - r["fp"]["gold_nll"] for r in Mw])
                  - mean([xa(r, f"h{w}", "gold_nll") - xa(r, "fp", "gold_nll") for r in Mw]))
        A0 = evaluable and len(Mw) == len(M) and abs(a0_kl) <= M_KL and abs(a0_nll) <= J_NLL
        top = max(R, key=lambda r: r[f"ref{w}"]["tf_kl_mean"])
        p_lo = pair([r for r in R if r is not top], w, w, sd(f"lo_{w}_n"), sd(f"lo_{w}_k"))
        ok = A0 and agree >= AGREE_MIN and p["A2"] and p["A3"] and A4 and A5
        per_task_d = {t: dict(dkl=mean([r[f"eng{w}"]["tf_kl_mean"] - r[f"ref{w}"]["tf_kl_mean"] for r in R if r["task"] == t]),
                              dnll=mean([r[f"eng{w}"]["gold_nll"] - r[f"ref{w}"]["gold_nll"] for r in R if r["task"] == t]))
                      for t in tasks_used}
        part1[f"h{w}"] = dict(p, A0=A0, A0_n_items=len(Mw), A0_ref_minus_x0139_tf_kl=a0_kl, A0_ref_minus_x0139_gold_dnll=a0_nll,
                              A1=agree >= AGREE_MIN, argmax_agree=agree, per_task_agree=per_task,
                              per_task_eng_minus_ref=per_task_d, A4=A4, A5=A5, max_kl_fp_ref=max_fr, max_kl_ref_eng=max_re,
                              flags_ref=flags_ref, mean_tf_kl_ref=info, informative=info >= INFO_MIN,
                              verdict="PASS" if ok and info >= INFO_MIN else "INCONCLUSIVE" if ok else "FAIL",
                              mean_kl_ref_eng=mean([r[f"eng{w}"]["tf_kl_ref"] for r in R]),
                              without_max_kl_item=dict(item=[top["task"], top["i"]], **p_lo))
    sens = {"eng33_vs_ref54": pair(R, "54", "33", sd("sens_33v54_n"), sd("sens_33v54_k")),
            "eng54_vs_ref33": pair(R, "33", "54", sd("sens_54v33_n"), sd("sens_54v33_k"))}
    for v in sens.values():
        v["distinguished"] = not (v["A2"] and v["A3"])
    out["part1"] = dict(widths=part1, sensitivity_reported_not_gated=sens, PASS=all(v["verdict"] == "PASS" for v in part1.values()))
    # ---- Part 2
    p1 = {}
    for w in W:
        for b in ("kivi128", "kvq2n"):
            Mb = [r for r in M if x_ok(X, b, r)]
            ck = boot_cluster(byR(Mb, lambda r, b_=b, w_=w: r[f"eng{w_}"]["tf_kl_mean"] - xa(r, b_, "tf_kl_mean")),
                              0.025, 0.975, sd(f"P1_{w}_{b}_kl"))
            cn = boot_cluster(byR(Mb, lambda r, b_=b, w_=w: (r[f"eng{w_}"]["gold_nll"] - r["fp"]["gold_nll"])
                                  - (xa(r, b_, "gold_nll") - xa(r, "fp", "gold_nll"))), 0.025, 0.975, sd(f"P1_{w}_{b}_nll"))
            p1[f"eng{w}_vs_{b}"] = dict(n_items=len(Mb), tf_kl=dict(point=ck[0], ci95=[ck[1], ck[2]], below0=ck[2] < 0),
                                        gold_dnll=dict(point=cn[0], ci95=[cn[1], cn[2]], below0=cn[2] < 0))
    p1_holds = complete and evaluable and all(v["tf_kl"]["below0"] and v["gold_dnll"]["below0"] for v in p1.values())
    p2 = boot_cluster(byR(R, lambda r: r["eng33"]["gold_nll"] - r["eng54"]["gold_nll"]), 0.05, 0.95, sd("P2"))
    p2k = boot_cluster(byR(R, lambda r: r["eng33"]["tf_kl_mean"] - r["eng54"]["tf_kl_mean"]), 0.05, 0.95, sd("P2_kl"))
    M3 = [r for r in M if x_ok(X, "kvq3", r)]
    p3k = boot_cluster(byR(M3, lambda r: r["eng54"]["tf_kl_mean"] - xa(r, "kvq3", "tf_kl_mean")), 0.025, 0.975, sd("P3_kl"))
    p3n = boot_cluster(byR(M3, lambda r: (r["eng54"]["gold_nll"] - r["fp"]["gold_nll"]) - (xa(r, "kvq3", "gold_nll") - xa(r, "fp", "gold_nll"))),
                       0.025, 0.975, sd("P3_nll"))
    table = {}
    for arm in ARMS:
        kl_ = byR(R, lambda r, a_=arm: r[a_]["tf_kl_mean"])
        pt, lo, hi = boot_cluster(byR(R, lambda r, a_=arm: r[a_]["gold_nll"] - r["fp"]["gold_nll"]), 0.025, 0.975, sd(f"table_{arm}"))
        table[arm] = dict(tf_kl_mean=mean([mean(v) for v in kl_.values()]),
                          tf_kl_item_mean=mean([x for v in kl_.values() for x in v]),
                          tf_kl_item_median=median([x for v in kl_.values() for x in v]),
                          gold_dnll=pt, gold_dnll_ci95=[lo, hi], ppl_factor=math.exp(pt),
                          per_task_tf_kl={t: mean(v) for t, v in kl_.items()},
                          per_task_gold_dnll={t: mean([r[arm]["gold_nll"] - r["fp"]["gold_nll"] for r in R if r["task"] == t])
                                              for t in tasks_used})
    repro = {}
    for w in W:
        Mw = [r for r in M if x_ok(X, f"h{w}", r)]
        repro[f"h{w}"] = dict(n_items=len(Mw),
                              x0139_tf_kl=mean([xa(r, f"h{w}", "tf_kl_mean") for r in Mw]),
                              ref_tf_kl=mean([r[f"ref{w}"]["tf_kl_mean"] for r in Mw]),
                              eng_tf_kl=mean([r[f"eng{w}"]["tf_kl_mean"] for r in Mw]),
                              mean_abs_diff_ref_vs_x0139_tf_kl=mean([abs(r[f"ref{w}"]["tf_kl_mean"] - xa(r, f"h{w}", "tf_kl_mean")) for r in Mw]),
                              x0139_gold_dnll=mean([xa(r, f"h{w}", "gold_nll") - xa(r, "fp", "gold_nll") for r in Mw]),
                              ref_gold_dnll=mean([r[f"ref{w}"]["gold_nll"] - r["fp"]["gold_nll"] for r in Mw]),
                              eng_gold_dnll=mean([r[f"eng{w}"]["gold_nll"] - r["fp"]["gold_nll"] for r in Mw]))
    out["part2"] = dict(matching=dict(n_usable=len(R), n_matched=len(M), fraction=frac, evaluable=evaluable,
                                      unmatched=sorted((r["task"], r["i"]) for r in R if not any(r is m for m in M))),
                        P1=dict(per_pair=p1, holds=p1_holds, evaluable=evaluable, complete=complete),
                        P2=dict(point=p2[0], ci90=[p2[1], p2[2]], holds=complete and -M_NLL <= p2[1] and p2[2] <= M_NLL),
                        P2_kl_descriptive=dict(point=p2k[0], ci90=[p2k[1], p2k[2]], inside_1e3=-M_KL <= p2k[1] and p2k[2] <= M_KL),
                        P3_descriptive=dict(n_items=len(M3), tf_kl=dict(point=p3k[0], ci95=[p3k[1], p3k[2]]),
                                            gold_dnll=dict(point=p3n[0], ci95=[p3n[1], p3n[2]])),
                        table=table, reproduction_of_x0139=repro,
                        table_a_simulated=dict(source="archived simulation table A (X-0139 group 1)",
                                               h54=dict(tf_kl=0.0153, tf_kl_median=0.0033, gold_dnll=0.025),
                                               h33=dict(tf_kl=0.0156, gold_dnll=0.023), kvq3=dict(tf_kl=0.0184, gold_dnll=0.071),
                                               kvq2n=dict(tf_kl=0.170, gold_dnll=0.50), kivi128=dict(tf_kl=0.288, gold_dnll=0.82)))
    ji = all(abs(part1[f"h{w}"]["mean_dkl"]) < J_KL and abs(part1[f"h{w}"]["mean_dnll"]) < J_NLL for w in W)
    jii = complete and -M_NLL - J_NLL <= p2[1] and p2[2] <= M_NLL + J_NLL
    bad_tasks = []
    for t in tasks_used:
        d = [part1[f"h{w}"]["per_task_eng_minus_ref"][t] for w in W]
        if (all(x["dkl"] > J_TASK_KL for x in d) or all(x["dkl"] < -J_TASK_KL for x in d)
                or all(x["dnll"] > J_TASK_NLL for x in d) or all(x["dnll"] < -J_TASK_NLL for x in d)):
            bad_tasks.append(t)
    jiv = all(part1[f"h{w}"]["A0"] for w in W)
    out["judgement_inputs"] = dict(note="criteria (i)-(iv) of REGISTRATION_R1.md, computed; the judgement itself is made by the analyst",
                                   crit_i=ji, crit_ii=jii, crit_iii=not bad_tasks, crit_iii_tasks=bad_tasks, crit_iv_A0=jiv,
                                   all_hold=ji and jii and not bad_tasks and jiv)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
