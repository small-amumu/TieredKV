# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""r3_verdict (TieredKV, round R3; 2026-11-10): the registered readings of REGISTRATION_R3.md, computed from the rp_run.py
--mode ruler outputs of R3 (Llama-3.1-8B-Instruct, RULER vt / fwe / qa_1 @130000, items 0-19, widths h33 / h54;
out/r3_<task>.jsonl) and the archived X-0140 records (pro5000: fp, kivi128 and the simulated h33 / h54, m128r37 items 0-9 cfg
eb2c65008aaf, m128r37b1 items 10-14, m128r37b2 items 15-19; STAR: RetroInfer retro_main, ShadowKV official skv_main, ShadowKV
fixed pf_f5_fix130k_*). Read-only; prints a JSON summary. Scores in points (record score x 100).

Completeness (Q0; any failure blocks Q1 and the judgement inputs):
  per task file: one head meta record (mode ruler, items "0:20", widths "33,54", length 130000, model .../Meta-Llama-3.1-8B-
  Instruct, md5 of rp_run.py / rp_engine.py as registered and every EXPECTED_MD5.json entry equal); for every item 0-19 and
  width h33 / h54 exactly one record, status ok, data_md5 as registered; one end record; every line valid JSON.
  rc logs: exactly one line "r3_<task> rc=0" per task, and every rc log has a "chunk_test_<g> rc=0 CHUNK_TEST PASS" line.
  Input identity: every item's L0 equals X-0140's fp len and its ids_md5 equals the STAR RetroInfer record's ids_md5.
Part 1, the engine against the simulated reference (same items, same machine, X-0140 Part A), per width w:
  S1  pred identical to the simulated h_w pred (exact string), and score identical, as fractions of 60 items.
  S2  the 3-task average score difference eng_w - sim_w (mean of task means) with a 95% stratified bootstrap interval
      (items resampled within task, 10000 draws); per task.
Part 2, the SOTA table with the engine's rows (95% stratified bootstrap of the 3-task average difference):
  Q1  (registered prediction; the report reads "no difference detected" against fp) the interval of eng54 - fp contains 0.
      Holds only on complete data. eng33 - fp reported alongside.
  descriptive: eng_w against kivi128 (pro5000, same machine) and RetroInfer / ShadowKV fixed / ShadowKV official (STAR 4090,
  another machine and environment: labelled); the table row (per task and 3-task average) for eng54 / eng33 next to the
  archived rows; system numbers of the engine run (prefill seconds, median decode ms, GPU peak, store sizes; shared machine,
  not a timing claim).
D0 (environment drift): X-0140's harness reran fp on items 0-1 of each task (r3_fpdrift.sh); pred and score compared with
  the archived fp records (6 items).
Judgement inputs (criteria of the registration, computed; all require complete data; a NaN fails a criterion):
  (i) |S2 point| <= 3 points for both widths; (ii) Q1 holds; (iii) no task whose task-mean eng54 - sim54 exceeds 10 points
  in either direction; (iv) D0: >= 5 of the 6 fp preds identical to the archive.
Usage: python3 -I r3_verdict.py <out dir> --rc <rc logs...> --x0140 <dir with m128r37*/m128r.jsonl> --star <dir with STAR jsonl>
       --fpdrift <fpdrift/fp.jsonl> [--expected EXPECTED_MD5.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re

TASKS = ("vt", "fwe", "qa_1")
ITEMS = tuple(range(20))
W = ("33", "54")
NB = 10000
SEED = 20261310
SEEDS = {"S2_33": 0, "S2_54": 1, "Q1_54": 2, "Q1_33": 3,
         **{f"vs_{b}_{w}": 4 + 2 * j + k for j, b in enumerate(("kivi128", "retro", "skv_fix", "skv_off")) for k, w in enumerate(W)}}
assert len(set(SEEDS.values())) == len(SEEDS)
RUN_MD5 = "16f8dc773bf2168ffd1774bff073eeb4"
CHUNK_MD5 = "7b4d736b4139b69241570174b71b5d69"
ENG_MD5 = "48545123140a1eb4b734b9b86a775f9c"
DATA_MD5 = {"vt": "f79205a5d226675bfcb6b5837c4c897a", "fwe": "b962072d1c074126382f09c2f147d0ee", "qa_1": "a4f8f9fada28282d64b388efad51d81c"}
PFX = "stkm8v8128k64_x2s128_p50_"
J_S2, J_Q1, J_TASK = 3.0, 3.0, 10.0
X0140_PARTS = (("m128r37", 0, "eb2c65008aaf"), ("m128r37b1", 10, None), ("m128r37b2", 15, None))


def sd(k):
    return SEED + SEEDS[k]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    s = sorted(x for x in xs if x is not None)
    n = len(s)
    return float("nan") if not n else s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def boot(by_task, seed):
    """3-task average (mean of task means) and its 95% stratified bootstrap interval."""
    by_task = {t: v for t, v in by_task.items() if v}
    if not by_task:
        return float("nan"), float("nan"), float("nan")
    rnd = random.Random(seed)
    stat = lambda d: sum(sum(v) / len(v) for v in d.values()) / len(d)  # noqa: E731
    dr = sorted(stat({t: [v[rnd.randrange(len(v))] for _ in v] for t, v in by_task.items()}) for _ in range(NB))
    return stat(by_task), dr[int(0.025 * NB)], dr[min(NB - 1, int(0.975 * NB))]


def load_run(d, expected, problems):
    E = {}
    for t in TASKS:
        f = os.path.join(d, f"r3_{t}.jsonl")
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
        if len(head) != 1 or len(end) != 1:
            problems.append(f"{t}: {len(head)} head and {len(end)} end meta records")
        h = head[0] if head else {}
        m5 = h.get("md5", {})
        if (m5.get("engine/rp_run.py") != RUN_MD5 or m5.get("engine/rp_engine.py") != ENG_MD5
                or m5.get("engine/rp_chunk.py") != CHUNK_MD5):
            problems.append(f"{t}: rp_run.py / rp_engine.py / rp_chunk.py md5 differs")
        bad = sorted(k for k, v in expected.items() if m5.get(k) != v)
        if bad:
            problems.append(f"{t}: reference / kernel md5 differ from EXPECTED_MD5: {bad}")
        if (h.get("mode") != "ruler" or h.get("items") != "0:20" or h.get("widths") != "33,54" or h.get("length") != 130000
                or not str(h.get("model", "")).endswith("Meta-Llama-3.1-8B-Instruct")):
            problems.append(f"{t}: head meta mode / items / widths / length / model differ from the registration")
        seen = {}
        for r in recs:
            key = (r.get("task"), r.get("i"), r.get("width"))
            seen[key] = seen.get(key, 0) + 1
            if r.get("task") != f"ruler:{t}@130000":
                problems.append(f"{t}: record of task {r.get('task')}")
                continue
            if r.get("status") != "ok":
                problems.append(f"{t} item {r.get('i')} {r.get('width')}: status {r.get('status')}")
                continue
            if r.get("data_md5") != DATA_MD5[t]:
                problems.append(f"{t} item {r.get('i')}: data_md5 differs")
            sc_ = r.get("score")
            if not (isinstance(sc_, (int, float)) and math.isfinite(sc_) and 0.0 <= sc_ <= 1.0) or not isinstance(r.get("pred"), str):
                problems.append(f"{t} item {r.get('i')} {r.get('width')}: score {sc_!r} not in [0, 1] or pred missing")
                continue
            if r.get("i") not in ITEMS or r.get("width") not in ("h33", "h54") or not isinstance(r.get("L0"), int):
                problems.append(f"{t}: record with item {r.get('i')!r} / width {r.get('width')!r} / L0 {r.get('L0')!r}")
                continue
            E[(t, r["i"], r["width"])] = r
        for i in ITEMS:
            for w in W:
                if seen.get((f"ruler:{t}@130000", i, f"h{w}"), 0) != 1:
                    problems.append(f"{t} item {i} h{w}: {seen.get((f'ruler:{t}@130000', i, f'h{w}'), 0)} records")
        if len(recs) != len(ITEMS) * len(W):
            problems.append(f"{t}: {len(recs)} records != {len(ITEMS) * len(W)}")
    return E


def rc_problems(paths):
    problems, seen = [], {}
    for p in paths:
        if not os.path.exists(p):
            problems.append(f"rc log {p} missing")
            continue
        lines = open(p).read().splitlines()
        if not any(re.search(r" chunk_test_\w+ rc=0 CHUNK_TEST PASS$", ln) for ln in lines):
            problems.append(f"rc log {p}: no chunk test PASS")
        for ln in lines:
            m = re.search(r" r3_(\w+) rc=(\d+)(?:\s|$)", ln)
            if m:
                seen.setdefault(m.group(1), []).append(int(m.group(2)))
    for t in TASKS:
        if seen.get(t) != [0]:
            problems.append(f"rc logs: r3_{t} rc lines {seen.get(t)} (need exactly one rc=0)")
    return problems


def load_x0140(d):
    S = {}
    for sub, off, cfg in X0140_PARTS:
        for line in open(os.path.join(d, sub, "m128r.jsonl")):
            r = json.loads(line)
            if cfg and r.get("cfg") != cfg:
                continue
            t = r["task"].split(":")[1].split("@")[0]
            if t not in TASKS:
                continue
            a = r["arm"].replace(PFX, "")
            key = (a, t, r["i"] + off)
            assert key not in S, f"duplicate X-0140 record {key}"
            S[key] = r
    return S


def load_star(d):
    S = {}
    for f, name in (("retro_main.jsonl", "retro"), ("skv_main.jsonl", "skv_off"),
                    ("pf_f5_fix130k_fwe.jsonl", "skv_fix"), ("pf_f5_fix130k_vt_qa_1.jsonl", "skv_fix")):
        for line in open(os.path.join(d, f)):
            r = json.loads(line)
            key = (name, r["task"], r["i"])
            assert key not in S, f"duplicate STAR record {key}"
            S[key] = r
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rc", required=True, nargs="+")
    ap.add_argument("--x0140", required=True)
    ap.add_argument("--star", required=True)
    ap.add_argument("--fpdrift", required=True)
    ap.add_argument("--expected", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "EXPECTED_MD5.json"))
    a = ap.parse_args()
    expected = json.load(open(a.expected))
    problems = []
    E = load_run(a.dir, expected, problems)
    problems += rc_problems(a.rc)
    X = load_x0140(a.x0140)
    S = load_star(a.star)
    A = {**X, **S}
    for t in TASKS:
        for i in ITEMS:
            for b in ("fp", "kivi128", "h33", "h54", "retro", "skv_off", "skv_fix"):
                if (b, t, i) not in A:
                    problems.append(f"archive: {b} {t} {i} missing")
    ident_bad = []
    for (t, i, w), r in E.items():
        f, s = X.get(("fp", t, i)), S.get(("retro", t, i))
        if not f or not s or r["L0"] != f["len"] or r["ids_md5"] != s["ids_md5"]:
            ident_bad.append((t, i, w))
    if ident_bad:
        problems.append(f"input identity fails on {len(ident_bad)} records: {sorted(set(ident_bad))[:6]}")
    complete = not problems
    sc = lambda arm, t, i: 100.0 * (E[(t, i, arm)]["score"] if arm in ("h33", "h54") and (t, i, arm) in E else float("nan"))  # noqa: E731
    asc = lambda b, t, i: 100.0 * A[(b, t, i)]["score"]  # noqa: E731
    out = dict(n_records=len(E), complete=complete, completeness_problems=problems, seeds={k: sd(k) for k in SEEDS})
    have = lambda w, t, i: (t, i, f"h{w}") in E  # noqa: E731
    part1 = {}
    for w in W:
        keys = [(t, i) for t in TASKS for i in ITEMS if have(w, t, i) and (f"h{w}", t, i) in A]
        same_pred = sum(E[(t, i, f"h{w}")]["pred"] == A[(f"h{w}", t, i)]["pred"] for t, i in keys)
        same_score = sum(E[(t, i, f"h{w}")]["score"] == A[(f"h{w}", t, i)]["score"] for t, i in keys)
        by = {t: [sc(f"h{w}", t, i) - asc(f"h{w}", t, i) for tt, i in keys if tt == t] for t in TASKS}
        pt, lo, hi = boot(by, sd(f"S2_{w}"))
        part1[f"h{w}"] = dict(n_items=len(keys), S1_pred_identical=same_pred / len(keys) if keys else float("nan"),
                              S1_score_identical=same_score / len(keys) if keys else float("nan"),
                              S2_eng_minus_sim=dict(point=pt, ci95=[lo, hi]), per_task_eng_minus_sim={t: mean(v) for t, v in by.items()},
                              diverging_items=sorted([t, i] for t, i in keys if E[(t, i, f"h{w}")]["pred"] != A[(f"h{w}", t, i)]["pred"]))
    out["part1"] = part1
    q1, comps, rows = {}, {}, {}
    for w in W:
        keys = [(t, i) for t in TASKS for i in ITEMS if have(w, t, i) and ("fp", t, i) in A]
        by = {t: [sc(f"h{w}", t, i) - asc("fp", t, i) for tt, i in keys if tt == t] for t in TASKS}
        pt, lo, hi = boot(by, sd(f"Q1_{w}"))
        disc = sum(sc(f"h{w}", t, i) != asc("fp", t, i) for t, i in keys)
        q1[f"eng{w}_minus_fp"] = dict(point=pt, ci95=[lo, hi], ci_width=hi - lo, contains0=lo <= 0 <= hi, n_items=len(keys),
                                      items_with_different_score=disc)
        for b in ("kivi128", "retro", "skv_fix", "skv_off"):
            kb = [(t, i) for t, i in keys if (b, t, i) in A]
            pb, lb, hb = boot({t: [sc(f"h{w}", t, i) - asc(b, t, i) for tt, i in kb if tt == t] for t in TASKS}, sd(f"vs_{b}_{w}"))
            comps[f"eng{w}_minus_{b}"] = dict(point=pb, ci95=[lb, hb], machine="pro5000 (same)" if b == "kivi128" else "STAR 4090 (other)")
        rows[f"eng{w}"] = {t: mean([sc(f"h{w}", t, i) for i in ITEMS if have(w, t, i)]) for t in TASKS}
        rows[f"eng{w}"]["avg3"] = mean([rows[f"eng{w}"][t] for t in TASKS])
    for b in ("fp", "h54", "h33", "kivi128", "retro", "skv_fix", "skv_off"):
        rows[b] = {t: mean([asc(b, t, i) for i in ITEMS if (b, t, i) in A]) for t in TASKS}
        rows[b]["avg3"] = mean([rows[b][t] for t in TASKS])
    sysnum = {f"eng{w}": dict(prefill_sec_median=median([r.get("prefill_sec") for (t, i, ww), r in E.items() if ww == f"h{w}"]),
                              decode_ms_median_of_items=median([r.get("decode_ms_median") for (t, i, ww), r in E.items() if ww == f"h{w}"]),
                              gpu_peak_gib_max=max([r.get("gpu_peak_gib", 0) for (t, i, ww), r in E.items() if ww == f"h{w}"] or [float("nan")]),
                              gpu_store_gib_max=max([r.get("gpu_store_gib", 0) for (t, i, ww), r in E.items() if ww == f"h{w}"] or [float("nan")]),
                              host_store_gib_max=max([r.get("host_store_gib", 0) for (t, i, ww), r in E.items() if ww == f"h{w}"] or [float("nan")]))
              for w in W}
    out["part2"] = dict(Q1=dict(holds=complete and q1["eng54_minus_fp"]["contains0"], **q1), comparisons=comps, table=rows,
                        system_numbers_descriptive=sysnum,
                        sota_table_simulated=dict(source="archived simulation SOTA section (X-0140)", fp=72.7, h54=72.3, retro=69.4,
                                                  skv_fix=68.8, skv_off=44.0, kivi128=62.6))
    # ---- D0: environment drift (X-0140's harness, fp, items 0-1)
    D = {}
    if os.path.exists(a.fpdrift):
        for line in open(a.fpdrift):
            r = json.loads(line)
            if r.get("arm") == "fp":
                D[(r["task"].split(":")[1].split("@")[0], r["i"])] = r
    d_keys = [(t, i) for t in TASKS for i in (0, 1)]
    d_same = [k for k in d_keys if k in D and ("fp",) + k in X and D[k].get("pred") == X[("fp",) + k]["pred"]]
    d_score = [k for k in d_keys if k in D and ("fp",) + k in X and D[k].get("score") == X[("fp",) + k]["score"]]
    out["D0_fp_drift"] = dict(n_rerun=sum(k in D for k in d_keys), pred_identical=len(d_same), score_identical=len(d_score),
                              differing=sorted([list(k) for k in d_keys if k not in d_same]))
    fin = lambda x: isinstance(x, float) and math.isfinite(x)  # noqa: E731
    ji = complete and all(fin(part1[f"h{w}"]["S2_eng_minus_sim"]["point"]) and abs(part1[f"h{w}"]["S2_eng_minus_sim"]["point"]) <= J_S2
                          for w in W)
    jii = complete and q1["eng54_minus_fp"]["contains0"]
    bad = [t for t in TASKS if not fin(part1["h54"]["per_task_eng_minus_sim"][t]) or abs(part1["h54"]["per_task_eng_minus_sim"][t]) > J_TASK]
    jiii = complete and not bad
    jiv = complete and len(d_same) >= 5
    out["judgement_inputs"] = dict(note="criteria (i)-(iv) of REGISTRATION_R3.md, computed; the judgement itself is made by the analyst",
                                   crit_i=ji, crit_ii=jii, crit_iii=jiii, crit_iii_tasks=bad, crit_iv_D0=jiv,
                                   all_hold=ji and jii and jiii and jiv)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
