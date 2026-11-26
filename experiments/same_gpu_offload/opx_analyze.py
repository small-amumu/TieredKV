# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""OPX analysis (PLAN_OPX.md): descriptive readings only, no pass / fail thresholds. Usability is judged afterwards from these numbers.
usage: python -I opx_analyze.py --star <dir with acc/ calib/ tim/ mem/> --p5 <opx_formal dir> [--out report.txt]
       python -I opx_analyze.py --selftest
Prints, per section: data integrity (input hashes / token counts, completeness), scores with 95% intervals, paired
differences of each of our widths against each competitor arm and fp, item agreement, PCIe bytes (median, IQR, n),
host memory (census, pinned, held, length-differenced RSS parts), the score x bytes points, calibration agreement of
the full-attention paths, cross-card token agreement, timing contamination, latency."""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import tempfile
from pathlib import Path

TASKS = ("vt", "cwe", "fwe", "qa_1", "qa_2")
N_ITEMS = 50
COMP = ("skv1024", "skv2048", "skv4096", "ret009", "ret018", "ret036")
REF = "skv2048_official"
OURS_W = ("h32", "h33", "h43", "h54")
SEED, B = 20261123, 10000
TIM_FILES = {"skv1024": "tim_skv1024", "skv2048": "tim_skv2048", "skv4096": "tim_skv4096",
             REF: "tim_skv2048_official", "ret009": "tim_ret009", "ret018": "tim_ret018", "ret036": "tim_ret036"}


def kind_of(task: str) -> str:
    """'vt', 'ruler:vt@130000' -> 'vt'."""
    return task.split(":")[-1].split("@")[0]


BAD_LINES: dict[str, int] = {}                     # file -> lines that are not valid JSON (e.g. cut by a kill)


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    # bytes split on b"\n" and decoded per line: not splitlines() (U+2028 etc. can occur inside a pred), and a kill
    # that cut a multi-byte character only loses that line
    for x in p.read_bytes().split(b"\n"):
        if not x.strip():
            continue
        try:
            out.append(json.loads(x.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            BAD_LINES[str(p)] = BAD_LINES.get(str(p), 0) + 1
    return out


def items(recs: list[dict]) -> list[dict]:
    return [r for r in recs if not r.get("meta")]


def has_end(recs: list[dict]) -> bool:
    return any(r.get("meta") and r.get("end") for r in recs)


def status(r: dict | None) -> str:
    if r is None:
        return "missing"
    if r.get("status") in ("error", "flag", "skipped_length"):
        return r["status"]
    return "error" if r.get("error") else "ok"


def acc_status(r: dict | None) -> str:
    """accuracy records: an ok record without a score is its own class (counted, scored 0)."""
    s = status(r)
    return "ok_no_score" if s == "ok" and r.get("score") is None else s


def score(r: dict | None) -> float:
    """0-100. Not ok (missing, error, flag, skipped) counts 0; the completeness table says how many."""
    if status(r) != "ok" or r.get("score") is None:
        return 0.0
    return 100.0 * float(r["score"])


# ---------------------------------------------------------------- loading
def load_acc(star: Path, p5: Path) -> tuple[dict, dict]:
    """arm -> {(kind, i): record}; arm -> end record present in every file of the arm."""
    arms: dict[str, dict] = {}
    ended: dict[str, bool] = {}
    for a in COMP + (REF,):
        recs = read_jsonl(star / "acc" / f"{a}.jsonl")
        arms[a] = {(kind_of(r["task"]), int(r["i"])): r for r in items(recs)}
        ended[a] = has_end(recs)
    for f in sorted(p5.glob("ours_h*.jsonl")):
        recs = read_jsonl(f)
        for r in items(recs):
            arms.setdefault(r["width"], {})[(kind_of(r["task"]), int(r["i"]))] = r
            ended[r["width"]] = has_end(recs)
    recs = read_jsonl(p5 / "fp.jsonl")                  # tb_bench writes <out>l next to <out> (tb_bench_v39 line 548)
    arms["fp"] = {}
    for r in items(recs):                               # a resumed run appends the retry after an error line:
        if r.get("arm") != "fp" or "task" not in r or "i" not in r:   # a record without error wins over an error one
            continue
        k = (kind_of(r["task"]), int(r["i"]))
        if k not in arms["fp"] or (arms["fp"][k].get("error") and not r.get("error")):
            arms["fp"][k] = r
    # tb_bench rewrites its summary .json after every item (tb_bench_v39 line 864, inside the item loop): a finished
    # fp run is one with a record for every expected item
    ended["fp"] = sum(1 for r in arms["fp"].values() if not r.get("error")) >= len(TASKS) * N_ITEMS
    return arms, ended


def integrity(arms: dict) -> tuple[set, list[str]]:
    """(task, i) keys whose inputs differ between arms (by input hash where present, else by token count), and the
    lines describing them. tb_bench's cfg is one hash per run, so the fp arm is checked by token count only."""
    bad, lines = set(), []
    keys = set().union(*[set(d) for d in arms.values()]) if arms else set()
    for k in sorted(keys):
        h = {a: d[k]["ids_md5"][:12] for a, d in arms.items() if k in d and d[k].get("ids_md5")}
        n = {a: int(d[k]["len"]) for a, d in arms.items() if k in d and d[k].get("len") is not None}
        if len(set(h.values())) > 1 or len(set(n.values())) > 1:
            bad.add(k)
            lines.append(f"{k}: hashes {h} token counts {n}")
    return bad, lines


def completeness(arms: dict) -> dict:
    out = {}
    for a, d in arms.items():
        c = {"ok": 0, "ok_no_score": 0, "error": 0, "flag": 0, "skipped_length": 0, "missing": 0}
        for t in TASKS:
            for i in range(N_ITEMS):
                c[acc_status(d.get((t, i)))] += 1
        out[a] = c
    return out


# ---------------------------------------------------------------- scores and paired differences
def overall(d: dict, idx: dict[str, list[int]]) -> float:
    """equal-weight mean over the tasks that have kept items (a task whose items are all excluded drops out for
    every arm alike, since idx is shared)."""
    per = [st.mean(score(d.get((t, i))) for i in idx[t]) for t in TASKS if idx[t]]
    return st.mean(per) if per else float("nan")


def boot(f, keep: dict[str, list[int]], seed: int = SEED, b: int = B) -> tuple[float, float, float]:
    """point and 95% percentile interval of f(index) under task-stratified resampling of the kept items (paired: the
    same resampled items for every arm inside f)."""
    pt = f(keep)
    rng = random.Random(seed)
    vals = sorted(f({t: [rng.choice(keep[t]) for _ in keep[t]] for t in TASKS}) for _ in range(b))
    return pt, vals[int(0.025 * b)], vals[min(b - 1, int(0.975 * b))]


def kept_index(bad: set) -> dict[str, list[int]]:
    return {t: [i for i in range(N_ITEMS) if (t, i) not in bad] for t in TASKS}


def agreement(da: dict, db: dict, keep: dict[str, list[int]]) -> tuple[int, int]:
    """(items scored ok in both arms, of which equal score)."""
    pairs = [(score(da.get((t, i))), score(db.get((t, i)))) for t in TASKS for i in keep[t]
             if acc_status(da.get((t, i))) == "ok" and acc_status(db.get((t, i))) == "ok"]
    return len(pairs), sum(x == y for x, y in pairs)


# ---------------------------------------------------------------- timing, bytes, latency
def tim_records(star: Path) -> dict[str, list[dict]]:
    recs = {a: items(read_jsonl(star / "tim" / f"{n}.jsonl")) for a, n in TIM_FILES.items()}
    for r in items(read_jsonl(star / "tim" / "tim_ours.jsonl")):
        recs.setdefault(r["width"], []).append(r)
    return recs


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def bytes_latency(recs: dict[str, list[dict]]) -> tuple[dict, dict]:
    byt, lat = {}, {}
    for a, rs in recs.items():
        ok = [r for r in rs if status(r) == "ok"]
        # a reading whose sampler thread died is partial: excluded from the median, counted separately
        vals = [r["pcie_rx_bytes_per_step_net"] for r in ok
                if r.get("pcie_rx_bytes_per_step_net") is not None and not r.get("pcie_error")]
        byt[a] = {"n_items": len(rs), "n_ok": len(ok), "n_bytes": len(vals),
                  "pcie_errors": sum(1 for r in rs if r.get("pcie_error")),
                  "median": st.median(vals) if vals else None, "q25": q(vals, 0.25) if vals else None,
                  "q75": q(vals, 0.75) if vals else None}
        ms = sorted(x for r in ok for x in (r.get("decode_ms_list") or []))
        pre = [r["prefill_sec"] for r in ok if r.get("prefill_sec") is not None]
        lat[a] = {"forwards": len(ms), "p50": st.median(ms) if ms else None, "mean": st.mean(ms) if ms else None,
                  "p99": q(ms, 0.99) if ms else None, "max": ms[-1] if ms else None,
                  "prefill_s_median": st.median(pre) if pre else None}
    return byt, lat


def contamination(star: Path) -> dict:
    """samples of the whole-host gpuwatch log with more than one compute process; a missing log is 'not verified'."""
    p = star / "tim" / "gpuwatch.log"
    if not p.exists():
        return {"verified": False, "samples": 0, "multi": 0, "examples": []}
    lines = [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    failed = [x for x in lines if x.split(" ", 1)[-1].strip().startswith("FAIL")]   # nvidia-smi failed: unknown, not idle
    ok = [x for x in lines if x not in failed]
    multi = [x for x in ok if len([e for e in x.split(" ", 1)[-1].split(";") if e.strip()]) > 1]
    return {"verified": len(ok) > 0, "samples": len(lines), "failed_samples": len(failed),
            "first": lines[0][:8] if lines else None, "last": lines[-1][:8] if lines else None,
            "multi": len(multi), "examples": multi[:5]}


# ---------------------------------------------------------------- host memory
def host_mem(star: Path) -> dict:
    out = {}
    for m, name in (("h54", "mem_ours_h54"), ("skv2048", "mem_skv2048"), ("ret018", "mem_ret018")):
        rf = items(read_jsonl(star / "mem" / f"{name}_full.jsonl"))
        rs = items(read_jsonl(star / "mem" / f"{name}_16k.jsonl"))
        need = ("host_tensor_gib", "host_pinned_gib", "rss_parts_trim")
        if not rf or not rs or any(k not in rf[0] or k not in rs[0] for k in need):
            out[m] = {"available": False}
            continue
        f, s = rf[0], rs[0]
        out[m] = {"available": True, "census_gib": f["host_tensor_gib"], "pinned_gib": f["host_pinned_gib"],
                  "held_pinned_gib": f["rss_parts_trim"].get("RssShmem"), "rss_gib": f.get("rss_after_prefill_gib"),
                  "peak_rss_gib": f.get("rss_peak_gib"),
                  "d_census_pinned": f["host_pinned_gib"] - s["host_pinned_gib"],
                  "d_census_pageable": (f["host_tensor_gib"] - f["host_pinned_gib"])
                  - (s["host_tensor_gib"] - s["host_pinned_gib"]),
                  "d_rss_shmem": f["rss_parts_trim"].get("RssShmem", 0) - s["rss_parts_trim"].get("RssShmem", 0),
                  "d_rss_anon": f["rss_parts_trim"].get("RssAnon", 0) - s["rss_parts_trim"].get("RssAnon", 0)}
    return out


# ---------------------------------------------------------------- calibration and cross-card
def calib(star: Path, p5: Path) -> dict:
    fp = {(kind_of(r["task"]), int(r["i"])): r for r in items(read_jsonl(p5 / "calibfp.jsonl")) if r.get("arm") == "fp"}
    out = {"tb_fp_items": len(fp)}
    for m in ("skv_full", "retro_full"):
        rs = {(kind_of(r["task"]), int(r["i"])): r for r in items(read_jsonl(star / "calib" / f"{m}.jsonl"))}
        common = sorted(set(rs) & set(fp))
        same_len = [k for k in common if rs[k].get("len") == fp[k].get("len")]
        out[m] = {"items": len(rs), "common_with_tb_fp": len(common), "same_token_count": len(same_len),
                  "same_score": sum(score(rs[k]) == score(fp[k]) for k in same_len),
                  "same_pred": sum((rs[k].get("pred") or "").strip() == (fp[k].get("pred") or "").strip()
                                   for k in same_len)}
    a = {(kind_of(r["task"]), int(r["i"])): r for r in items(read_jsonl(star / "calib" / "skv_full.jsonl"))}
    b = {(kind_of(r["task"]), int(r["i"])): r for r in items(read_jsonl(star / "calib" / "retro_full.jsonl"))}
    cm = sorted(set(a) & set(b))
    sl = [k for k in cm if a[k].get("len") == b[k].get("len") and (a[k].get("ids_md5") or "")[:12]
          == (b[k].get("ids_md5") or "")[:12]]
    out["skv_full_vs_retro_full"] = {"common": len(cm), "same_input": len(sl),
                                     "same_score": sum(score(a[k]) == score(b[k]) for k in sl),
                                     "same_pred": sum((a[k].get("pred") or "").strip() == (b[k].get("pred") or "").strip()
                                                      for k in sl)}
    return out


def xcard(star: Path, p5: Path, bad: set | None = None) -> dict:
    acc = {}
    for f in p5.glob("ours_h*.jsonl"):
        for r in items(read_jsonl(f)):
            acc[(kind_of(r["task"]), int(r["i"]), r["width"])] = r
    out: dict[str, dict] = {}
    for r in items(read_jsonl(star / "tim" / "tim_ours.jsonl")):
        k = (kind_of(r["task"]), int(r["i"]), r["width"])
        a = acc.get(k)
        o = out.setdefault(r["width"], {"compared": 0, "identical_prefix": 0, "not_comparable": 0})
        if (a is None or status(a) != "ok" or status(r) != "ok" or not a.get("toks") or not r.get("toks")
                or (k[0], k[1]) in (bad or set()) or a.get("ids_md5") != r.get("ids_md5")):
            o["not_comparable"] += 1
            continue
        o["compared"] += 1
        o["identical_prefix"] += int(r["toks"][:len(a["toks"])] == a["toks"])
    return out


# ---------------------------------------------------------------- report
def f2(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def analyze(star: Path, p5: Path, b: int = B) -> tuple[str, dict]:
    L: list[str] = []
    BAD_LINES.clear()
    res: dict = {}
    arms, ended = load_acc(star, p5)
    bad, lines = integrity(arms)
    res["input_mismatch"] = sorted(bad)
    L.append(f"[integrity] {len(bad)} (task, i) with differing inputs; excluded from every comparison below")
    L += ["  " + x for x in lines[:20]]
    if BAD_LINES:
        L.append("[integrity] lines that are not valid JSON (skipped): " + json.dumps(BAD_LINES))
    res["bad_lines"] = dict(BAD_LINES)
    res["completeness"] = completeness(arms)
    res["ended"] = ended
    for a, c in sorted(res["completeness"].items()):
        L.append(f"[completeness] {a:18s} {json.dumps(c)} end_record={ended.get(a)}")
    keep = kept_index(bad)
    res["scores"] = {}
    for a, d in sorted(arms.items()):
        if not d:
            continue
        pt, lo, hi = boot(lambda ix, d=d: overall(d, ix), keep, b=b)
        per = {t: st.mean(score(d.get((t, i))) for i in keep[t]) if keep[t] else float("nan") for t in TASKS}
        n_ok = sum(acc_status(d.get((t, i))) == "ok" for t in TASKS for i in keep[t])
        n_kept = sum(len(keep[t]) for t in TASKS)
        res["scores"][a] = {"mean": pt, "lo": lo, "hi": hi, "per_task": per, "n_ok": n_ok, "n_kept": n_kept}
        L.append(f"[score] {a:18s} {pt:6.2f} [{lo:6.2f}, {hi:6.2f}]  ok {n_ok}/{n_kept}  "
                 + " ".join(f"{t}={v:.1f}" for t, v in per.items()))
    res["paired"] = {}
    for w in OURS_W:
        if not arms.get(w):
            continue
        for c in COMP + (REF, "fp"):
            if not arms.get(c):
                continue
            pt, lo, hi = boot(lambda ix, w=w, c=c: overall(arms[w], ix) - overall(arms[c], ix), keep, b=b)
            n_both, n_same = agreement(arms[w], arms[c], keep)
            res["paired"][f"{w}-{c}"] = {"diff": pt, "lo": lo, "hi": hi, "both_ok": n_both, "same_score": n_same}
            L.append(f"[paired] {w} - {c:16s} {pt:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]  same score on {n_same}/{n_both} "
                     f"items ok in both")
    byt, lat = bytes_latency(tim_records(star))
    res["bytes"], res["latency"] = byt, lat
    for a in sorted(byt):
        x = byt[a]
        L.append(f"[pcie] {a:18s} median {f2(x['median'] and x['median'] / 1e6)} MB/step  IQR "
                 f"[{f2(x['q25'] and x['q25'] / 1e6)}, {f2(x['q75'] and x['q75'] / 1e6)}]  n_bytes {x['n_bytes']}/"
                 f"{x['n_items']}  nvml_errors {x['pcie_errors']}")
    h = byt.get("h54", {}).get("median")
    for c in ("skv2048", "ret018"):
        m = byt.get(c, {}).get("median")
        L.append(f"[pcie ratio] h54 / {c} = " + (f"{h / m:.3f}" if h is not None and m else "n/a"))
    res["contamination"] = contamination(star)
    L.append("[timing host] " + json.dumps(res["contamination"]))
    for a in sorted(lat):
        L.append(f"[latency] {a:18s} " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                                     for k, v in lat[a].items()}))
    res["host_mem"] = host_mem(star)
    for m, v in res["host_mem"].items():
        L.append(f"[host memory] {m:8s} " + json.dumps({k: (round(x, 3) if isinstance(x, float) else x)
                                                       for k, x in v.items()}))
    hm = res["host_mem"]
    if hm.get("h54", {}).get("available"):
        for c in ("skv2048", "ret018"):
            if hm.get(c, {}).get("available"):
                def ratio(k: str, c: str = c) -> str:
                    x, y = hm["h54"].get(k), hm[c].get(k)
                    return f"{x / y:.3f}" if x is not None and y else "n/a"
                L.append(f"[host ratio] h54 / {c}: census {ratio('census_gib')}, held pinned (RssShmem) "
                         f"{ratio('held_pinned_gib')}")
    res["points"] = []
    for a in COMP + OURS_W:
        if a in res["scores"] and byt.get(a, {}).get("median") is not None:
            res["points"].append({"arm": a, "mb_per_step": byt[a]["median"] / 1e6, "score": res["scores"][a]["mean"]})
    for p in sorted(res["points"], key=lambda p: p["mb_per_step"]):
        L.append(f"[point] {p['arm']:8s} {p['mb_per_step']:8.2f} MB/step  score {p['score']:6.2f}")
    for c in COMP:
        cp = next((p for p in res["points"] if p["arm"] == c), None)
        if cp is None:
            continue
        cheaper = [p["arm"] for p in res["points"] if p["arm"] in OURS_W and p["mb_per_step"] <= cp["mb_per_step"]]
        L.append(f"[vs {c}] our widths with no more bytes: {cheaper} -> paired diffs: "
                 + ", ".join(f"{w} {res['paired'][f'{w}-{c}']['diff']:+.2f} [{res['paired'][f'{w}-{c}']['lo']:+.2f}, "
                             f"{res['paired'][f'{w}-{c}']['hi']:+.2f}]" for w in cheaper if f"{w}-{c}" in res["paired"]))
    res["calib"] = calib(star, p5)
    L.append("[calibration 32K] " + json.dumps(res["calib"]))
    res["xcard"] = xcard(star, p5, bad)
    L.append("[cross-card tokens] " + json.dumps(res["xcard"]))
    return "\n".join(L), res


# ---------------------------------------------------------------- self-test on synthetic records
def _w(p: Path, rows: list[dict], end: bool = True) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"meta": True}] + rows + ([{"meta": True, "end": True}] if end else [])
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _make(root: Path, p_ours: float, p_comp: float, gpuwatch: str | None = "12:00:00 GPU-a, 1, 1;\n") -> tuple[Path, Path]:
    rng = random.Random(1)
    star, p5 = root / "star", root / "p5"
    keys = [(t, i) for t in TASKS for i in range(N_ITEMS)]
    ids = {k: f"{abs(hash(k)) % 16 ** 12:012x}" + "0" * 20 for k in keys}
    hard = {k: rng.random() for k in keys}                    # shared item difficulty: paired data

    def recs(p: float, extra: dict | None = None, fmt: str = "{t}", tb: bool = False) -> list[dict]:
        out = []
        for (t, i) in keys:
            r = {"task": fmt.format(t=t), "i": i, "len": 129000 + i, "score": 1.0 if hard[(t, i)] < p else 0.0}
            if tb:
                r.update(arm="fp", cfg="runlevel0000")
            else:
                r["ids_md5"] = ids[(t, i)]
            r.update(extra or {})
            out.append(r)
        return out
    for a in COMP + (REF,):
        _w(star / "acc" / f"{a}.jsonl", recs(p_comp))
    for w in OURS_W:                                          # one file per width, as opx_formal_p5.sh writes
        _w(p5 / f"ours_{w}.jsonl", recs(p_ours, {"width": w, "status": "ok", "toks": [1, 2, 3]}, "ruler:{t}@130000"))
    _w(p5 / "fp.jsonl", recs(0.9, fmt="ruler:{t}@130000", tb=True), end=False)
    mb = {"skv1024": 40, "skv2048": 53, "skv4096": 80, REF: 53, "ret009": 30, "ret018": 40, "ret036": 70}
    for a, n in TIM_FILES.items():
        _w(star / "tim" / f"{n}.jsonl", [{"task": t, "i": i, "pcie_rx_bytes_per_step_net": mb[a] * 1e6,
                                          "prefill_sec": 55.0, "decode_ms_list": [27.0, 28.0, 30.0]}
                                         for t in TASKS for i in range(4)])
    _w(star / "tim" / "tim_ours.jsonl", [{"task": f"ruler:{t}@130000", "i": i, "width": w, "status": "ok",
                                         "ids_md5": ids[(t, i)],
                                         "toks": [1, 2, 3, 9], "prefill_sec": 64.0, "decode_ms_list": [28.0, 300.0],
                                         "pcie_rx_bytes_per_step_net": {"h32": 4.9, "h33": 5.2, "h43": 5.9,
                                                                        "h54": 6.7}[w] * 1e6}
                                        for t in TASKS for i in range(4) for w in OURS_W])
    if gpuwatch is not None:
        (star / "tim" / "gpuwatch.log").write_text(gpuwatch, encoding="utf-8")

    def mem(cen, pin, anon, shm):
        return [{"task": "vt", "i": 0, "host_tensor_gib": cen, "host_pinned_gib": pin, "rss_after_prefill_gib": 9.0,
                 "rss_parts_trim": {"RssAnon": anon, "RssShmem": shm, "RssFile": 0.4}}]
    for name, full, k16 in (("mem_ours_h54", (5.21, 4.71, 1.80, 4.74), (0.65, 0.58, 1.36, 0.60)),
                            ("mem_skv2048", (8.06, 8.06, 0.97, 16.02), (1.13, 1.13, 0.97, 2.02)),
                            ("mem_ret018", (16.36, 16.36, 0.92, 16.54), (2.01, 2.01, 0.81, 2.08))):
        _w(star / "mem" / f"{name}_full.jsonl", mem(*full))
        _w(star / "mem" / f"{name}_16k.jsonl", mem(*k16))
    ck = [(t, i) for t in ("vt", "fwe", "qa_1") for i in range(10)]
    _w(p5 / "calibfp.jsonl", [{"arm": "fp", "task": f"ruler:{t}@32768", "i": i, "cfg": "run", "len": 32000 + i,
                               "score": 1.0, "pred": "x"} for t, i in ck], end=False)
    for m, wrong in (("skv_full", 0), ("retro_full", 3)):
        rows = [{"task": t, "i": i, "ids_md5": "a" * 32, "len": 32000 + i, "score": 1.0, "pred": "x"} for t, i in ck]
        for r in rows[:wrong]:
            r["score"] = 0.0
        _w(star / "calib" / f"{m}.jsonl", rows)
    return star, p5


def _rows(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]


def _put(f: Path, rows: list[dict]) -> None:
    f.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


def selftest() -> None:
    with tempfile.TemporaryDirectory() as d:
        star, p5 = _make(Path(d) / "a", 0.80, 0.70)
        txt, r = analyze(star, p5, b=400)
        assert r["input_mismatch"] == [] and r["bad_lines"] == {}, txt
        assert r["completeness"]["h54"]["ok"] == 250 and r["completeness"]["fp"]["ok"] == 250
        assert r["ended"]["h54"] is True and r["ended"]["fp"] is True           # 250 fp records = finished
        # shared item difficulty: our arm solves a superset of skv2048's items. Exact consistency checks:
        s, p = r["scores"], r["paired"]["h54-skv2048"]
        assert abs(p["diff"] - (s["h54"]["mean"] - s["skv2048"]["mean"])) < 1e-9, p
        assert abs(p["diff"] - 100 * (p["both_ok"] - p["same_score"]) / 250) < 1e-9, p
        assert p["lo"] >= 0 and p["both_ok"] == 250 and s["h54"]["n_ok"] == 250, p
        b54 = r["bytes"]["h54"]
        assert b54["median"] == 6.7e6 and b54["n_bytes"] == 20 and b54["pcie_errors"] == 0, b54
        lat = r["latency"]["h54"]
        assert lat["forwards"] == 40 and lat["p50"] == 164.0 and lat["mean"] == 164.0 and lat["p99"] == 300.0, lat
        assert "[pcie ratio] h54 / skv2048 = 0.126" in txt and "[pcie ratio] h54 / ret018 = 0.168" in txt, txt
        hm = r["host_mem"]
        assert abs(hm["skv2048"]["d_rss_shmem"] - 14.0) < 1e-9 and abs(hm["h54"]["d_census_pinned"] - 4.13) < 1e-9
        assert "census 0.646, held pinned (RssShmem) 0.296" in txt, txt
        c = r["calib"]
        assert (c["skv_full"]["same_token_count"], c["skv_full"]["same_score"], c["retro_full"]["same_score"]) == (30, 30, 27)
        assert c["skv_full_vs_retro_full"] == {"common": 30, "same_input": 30, "same_score": 27, "same_pred": 30}, c
        assert r["xcard"]["h54"] == {"compared": 20, "identical_prefix": 20, "not_comparable": 0}, r["xcard"]
        pt = {x["arm"]: x for x in r["points"]}
        assert pt["h54"]["mb_per_step"] == 6.7 and pt["h54"]["score"] == s["h54"]["mean"] and "[vs skv2048]" in txt
        assert r["contamination"] == {"verified": True, "samples": 1, "failed_samples": 0, "first": "12:00:00",
                                      "last": "12:00:00", "multi": 0, "examples": []}, r["contamination"]
        # input mismatch (hash, token count) is listed and excluded; a whole task excluded does not crash
        f = star / "acc" / "ret036.jsonl"
        rows = _rows(f)
        rows[1]["ids_md5"] = "f" * 32
        rows[2]["len"] = 1
        _put(f, rows)
        _, r = analyze(star, p5, b=200)
        assert len(r["input_mismatch"]) == 2 and r["scores"]["h54"]["n_kept"] == 248, r["input_mismatch"]
        for x in rows:
            if x.get("task") == "qa_2":
                x["len"] = 7
        _put(f, rows)
        _, r = analyze(star, p5, b=200)
        assert len(r["input_mismatch"]) == 52 and r["scores"]["h54"]["n_kept"] == 198
        _put(f, [dict(x, len=1) if not x.get("meta") else x for x in rows])       # every item mismatched
        txt_all, r = analyze(star, p5, b=50)
        assert len(r["input_mismatch"]) == 250 and r["scores"]["h54"]["n_kept"] == 0 and "nan" in txt_all
        _put(f, rows)
        # completeness: missing, error, skipped, ok without score, no end record; a truncated last line is skipped
        f = p5 / "ours_h32.jsonl"
        _put(f, [x for x in _rows(f) if x.get("i") != 0 and not x.get("end")])                # 5 missing, no end
        f = p5 / "ours_h33.jsonl"
        _put(f, [dict(x, status="error") if x.get("i") == 1 else x for x in _rows(f)])
        f = p5 / "ours_h43.jsonl"
        _put(f, [dict(x, status="skipped_length") if x.get("i") == 2 else x for x in _rows(f)])
        f = p5 / "ours_h54.jsonl"
        _put(f, [dict(x, score=None) if x.get("i") == 3 else x for x in _rows(f)])
        with (star / "acc" / "skv1024.jsonl").open("a", encoding="utf-8") as fh:
            fh.write('{"task": "vt", "i": 0, "sc')
        _, r = analyze(star, p5, b=200)
        cc = r["completeness"]
        assert (cc["h32"]["missing"], cc["h33"]["error"], cc["h43"]["skipped_length"], cc["h54"]["ok_no_score"]) \
            == (5, 5, 5, 5), cc
        assert r["ended"]["h32"] is False and r["ended"]["h33"] is True and len(r["bad_lines"]) == 1, r["bad_lines"]
        # timing host: a busy sample is counted, a missing log is 'not verified'
        star, p5 = _make(Path(d) / "b", 0.5, 0.5,
                         gpuwatch="12:00:00 GPU-a, 1, 1;GPU-b, 2, 1;\n12:00:02 \n12:00:04 FAIL;\n")
        _, r = analyze(star, p5, b=200)
        cn = r["contamination"]
        assert (cn["multi"], cn["samples"], cn["failed_samples"]) == (1, 3, 1), cn
        (star / "tim" / "gpuwatch.log").write_text("12:00:00 FAIL;\n12:00:02 FAIL;\n", encoding="utf-8")
        _, r = analyze(star, p5, b=50)
        assert r["contamination"]["verified"] is False, r["contamination"]       # no known sample: not verified
        f = star / "acc" / "skv4096.jsonl"                    # a kill that cut a multi-byte character
        f.write_bytes(f.read_bytes() + '{"pred": "'.encode() + "\u6c49".encode()[:2])
        _, r = analyze(star, p5, b=50)
        assert len(r["bad_lines"]) == 1, r["bad_lines"]
        assert json.loads(json.dumps(_no_nan({"a": float("nan"), ("t", 1): [float("nan")]}))) == {"a": None,
                                                                                                    "('t', 1)": [None]}
        star, p5 = _make(Path(d) / "c", 0.5, 0.5, gpuwatch=None)
        f = p5 / "fp.jsonl"                                   # error line + appended retry: one item, ok wins
        rows = _rows(f)
        _put(f, rows[:1] + [dict(rows[1], error="OOM")] + rows[1:] + [{"arm": "fp", "task": "x"}])
        _, r = analyze(star, p5, b=50)
        assert r["completeness"]["fp"]["ok"] == 250 and r["ended"]["fp"] is True, r["completeness"]["fp"]
        _put(f, rows[:100])                                   # tb_bench killed after some items: not finished
        (p5 / "fp.json").write_text("{}", encoding="utf-8")    # tb_bench rewrites its summary after every item
        _, r = analyze(star, p5, b=200)
        assert r["contamination"]["verified"] is False and r["ended"]["fp"] is False
        assert r["completeness"]["fp"]["missing"] == 151, r["completeness"]["fp"]
        # a reading with a dead NVML sampler is left out of the median and counted
        f = star / "tim" / "tim_ours.jsonl"
        rows = _rows(f)
        for x in rows:
            if x.get("width") == "h54" and x.get("i") == 0:
                x["pcie_rx_bytes_per_step_net"], x["pcie_error"] = 0.4e6, "NVMLError"
        _put(f, rows)
        _, r = analyze(star, p5, b=200)
        assert r["bytes"]["h54"]["n_bytes"] == 15 and r["bytes"]["h54"]["pcie_errors"] == 5
        assert r["bytes"]["h54"]["median"] == 6.7e6 and r["bytes"]["h54"]["q25"] == 6.7e6
        # missing files, zero bytes and missing host-memory records degrade to n/a instead of crashing
        (star / "tim" / "tim_skv2048.jsonl").unlink()
        for f in (star / "mem").glob("mem_ret018_*.jsonl"):
            f.unlink()
        f = star / "tim" / "tim_ret018.jsonl"
        _put(f, [dict(x, pcie_rx_bytes_per_step_net=0.0) if not x.get("meta") else x for x in _rows(f)])
        txt, r = analyze(star, p5, b=200)
        assert r["host_mem"]["ret018"] == {"available": False} and "skv2048" not in [x["arm"] for x in r["points"]]
        assert "[pcie ratio] h54 / skv2048 = n/a" in txt and "[pcie ratio] h54 / ret018 = n/a" in txt, txt
    print("selftest PASS")


def _no_nan(x):
    """NaN -> None, so the JSON output is strict JSON (keys that are tuples become strings via default=str)."""
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _no_nan(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_no_nan(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", type=Path)
    ap.add_argument("--p5", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    txt, res = analyze(a.star, a.p5)
    print(txt)
    if a.out:
        a.out.write_text(txt + "\n", encoding="utf-8")
        a.out.with_suffix(".json").write_text(json.dumps(_no_nan(res), indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
