# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""NAB2 analysis (REG_NAB2.md): provenance, Q0-a/b/c/d/e checks, decision rules R1-R4, bits.

usage: python nab2_analyze.py <nab1_out_dir> <nab2_out_dir>
       python nab2_analyze.py --gate <tag> <nab1_out_dir> <nab2_out_dir>
  nab1_out_dir holds NAB1's lab_<tag>.json (frozen results); nab2_out_dir holds ctl_<tag>.json,
  lab2_<tag>.json, rev_<tag>.json and selftest_pass.txt (the Q0-b marker of nab2_selftest.py).
  Every ratio uses arms of the same lab2 run; NAB1 is only the reproduction reference (Q0-a on
  ctl items 0-7, Q0-c on all 64 lab2 items); rev is lab2's six arms in a seeded random order per
  item on eight items spread over the traces, in a new process (Q0-d, order invariance).
  ctl also runs t_variant_off (the patched variant, every ablation off), which must equal the
  deployed arm on that model (Q0-e). --gate is run by nab2_chain.sh after a model's control replay
  and before its ablations: the provenance, code record, Q0-a and Q0-e of ctl_<tag>.json alone;
  exit 0 if all pass, else 2.
Exit 0 with verdicts; exit 2 when the round is void (provenance, configuration, code record,
Q0-a to Q0-e, item sets, non-finite readouts, non-positive decode KL, or any error while checking).
No per-model numbers are printed for a void round. Verdicts come only from the registered rules.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

TAGS = ("qwen3_0p6b", "qwen3_1p7b", "qwen3_4b")
OURS, KIVI2 = "kivi128k64_x2s128_j50_h54", "kivi128"
X2R, LAM0 = "kivi128k64_x2r128_j50_h54", "kivi128k64_x2s128_h54"
KONLY, KIVI4 = "kivi128k64_x2s128_j50_h5n", "kivi4b128"
ARMS = {"x2r": X2R, "lam0": LAM0, "konly": KONLY, "kivi4": KIVI4}
# revision 7 (keep the ablation controls fixed): lab2 reruns NAB1's two arms next to the
# ablations on all 64 traces, so every ratio compares arms of the same run, card and code;
# NAB1 is only the reproduction check. Revision 7d (Q0-e): each model's control replay also
# runs the patched variant with every ablation off (t_variant_off), which must equal the
# deployed arm in every readout on that model
TVOFF = "t_variant_off"
NAB1_RUNS = [KIVI2, OURS]
CTL_RUNS, LAB2_RUNS = [KIVI2, OURS, TVOFF], [KIVI2, OURS, X2R, LAM0, KONLY, KIVI4]
# Q0-d (order invariance, revision 7b): the six arms again, in a seeded random order per item, on
# eight items spread over the trace file, in a new process
REV_IDS, REV_SEED = [0, 9, 18, 27, 36, 45, 54, 63], 7
# REG_NAB2.md section 4 (frozen code), in the marker's order
FROZEN = {"nab2_q.py": "19b2740a", "nab2_lab.py": "d6dcd3da",
          "nab2_selftest.py": "5db6fac9", "nab2_chain.sh": "d9786b7a"}
# the marker's line 1 ends with this analyzer's own md5 (the fifth file of nab2_selftest.CODE),
# so the analyzer that judges is the one that was on disk at the self-test
SELF_MD5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:8]
MARKER = "PASS " + " ".join(FROZEN.values()) + " " + SELF_MD5
CONFIG = {"dtype": "float32", "chunk": 256, "max_len": 8800}       # as NAB1 (REG section 2)
CTL_IDS = list(range(8))
REL_TOL = 1e-6
ERRORS = (Exception,)        # any error while checking voids the round (exit 2), never a verdict


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def same(a: object, b: object) -> bool:
    """both None, or both finite and equal within REL_TOL (NaN never passes)."""
    if a is None or b is None:
        return a is None and b is None
    return finite(a) and finite(b) and abs(a - b) <= REL_TOL * max(abs(b), 1e-12)


def readouts(it: dict, run: str) -> list:
    """every numeric readout of a run in one item (None allowed for empty spans and buckets)."""
    vals = list(it["decode"][run]) + list(it["answer"][run])
    for b in it["bucket"][run].values():
        vals += list(b)
    return vals


def read_marker(p: Path) -> tuple[str, dict | None]:
    """the Q0-b marker: its PASS line and the self-tested code record (None if unreadable)."""
    if not p.exists():
        return "", None
    lines = p.read_text(encoding="utf-8").splitlines()
    try:
        code = json.loads(lines[1])
    except (IndexError, ValueError):
        code = None
    return (lines[0] if lines else ""), (code if isinstance(code, dict) else None)


def code_record(d: dict) -> dict:
    """a replay's md5 record without its trace file: the code, stack and calibration text."""
    tr = Path(d["traces"]).name
    return {k: v for k, v in d["md5"].items() if k != tr}


def provenance(nab1: dict, d: dict, name: str, runs: list[str], marker: str,
               mcode: dict | None) -> list[str]:
    bad = []
    tr = Path(nab1["traces"]).name
    ref = nab1["md5"].get(tr)
    if ref is None or Path(d["traces"]).name != tr or d["md5"].get(tr) != ref:
        bad.append(f"{name}: traces md5 {d['md5'].get(tr)} != NAB1 {ref} ({tr})")
    if d["model"] != nab1["model"]:
        bad.append(f"{name}: model {d['model']} != NAB1 {nab1['model']}")
    for k, v in CONFIG.items():
        if d[k] != v or nab1[k] != v:
            bad.append(f"{name}: {k} {d[k]} (NAB1 {nab1[k]}) != {v}")
    if d["attn_layers"] != nab1["attn_layers"]:
        bad.append(f"{name}: attention layers differ from NAB1")
    if d["runs"] != runs:
        bad.append(f"{name}: runs {d['runs']} != {runs}")
    if d.get("smoke"):
        bad.append(f"{name}: a smoke file")
    for f in ("nab2_q.py", "nab2_lab.py"):
        if not d["md5"].get(f, "").startswith(FROZEN[f]):
            bad.append(f"{name}: {f} md5 {d['md5'].get(f)} is not the frozen {FROZEN[f]}")
    if marker != MARKER:
        bad.append(f"Q0-b marker '{marker}' != the frozen '{MARKER}'")
    if mcode is None:
        bad.append("Q0-b marker has no code record")
    else:
        cr = code_record(d)
        diff = sorted(k for k in set(cr) | set(mcode) if cr.get(k) != mcode.get(k))
        if diff:
            bad.append(f"{name}: code record differs from the self-tested one in {diff[:4]}")
    return bad


def q0a(nab1: dict, ctl: dict, ids_want: list[int] | None = CTL_IDS) -> list[str]:
    """NAB1's two arms rerun must equal NAB1 in every readout (value, row count): items 0-7 of the
    control replay (Q0-a), or, with ids_want=None, every item of lab2 (Q0-c)."""
    bad = []
    ids = [it["i"] for it in ctl["items"]]
    if ids_want is not None and sorted(ids) != ids_want:
        bad.append(f"control item ids {ids} != {ids_want}")
    old = {it["i"]: it for it in nab1["items"]}
    for it in ctl["items"]:
        o = old.get(it["i"])
        if o is None:
            bad.append(f"item {it['i']} not in NAB1")
            continue
        for run in NAB1_RUNS:
            if set(it["bucket"][run]) != set(o["bucket"][run]):
                bad.append(f"item {it['i']} {run}: bucket keys differ")
                continue
            a, b = readouts(it, run), readouts(o, run)
            diff = [j for j, (x, y) in enumerate(zip(a, b)) if not same(x, y)]
            if len(a) != len(b) or diff:              # positions only: a void prints no values
                bad.append(f"item {it['i']} {run}: {len(diff)} readouts differ from NAB1 "
                           f"(first at index {diff[:1]}; {len(a)} vs {len(b)} readouts)")
    return bad


def q0e(ctl: dict) -> list[str]:
    """the patched _attend_h with every ablation off (t_variant_off) equals the deployed arm in
    every readout and bit count on this model's control items; it fired the patch once per layer
    and never the frozen function, while the deployed arm ran the frozen function and no patch."""
    bad = []
    n = len(ctl["attn_layers"])
    for it in ctl["items"]:
        i = it["i"]
        if set(it["bucket"][TVOFF]) != set(it["bucket"][OURS]):
            bad.append(f"item {i}: {TVOFF} bucket keys differ from the deployed arm")
            continue
        a, b = readouts(it, TVOFF), readouts(it, OURS)
        diff = [j for j, (x, y) in enumerate(zip(a, b)) if not same(x, y)]
        if len(a) != len(b) or diff:
            bad.append(f"item {i}: {TVOFF} differs from the deployed arm in {len(diff)} readouts "
                       f"({len(a)} vs {len(b)} readouts)")
        if any(it[k][TVOFF] != it[k][OURS] for k in ("eff_bits", "read_bits", "host_bits")):
            bad.append(f"item {i}: {TVOFF} bits differ from the deployed arm")
        if list(it["patched_calls"][TVOFF]) != [n, 0] or list(it["patched_calls"][OURS]) != [0, 0]:
            bad.append(f"item {i}: patched calls {TVOFF} {it['patched_calls'][TVOFF]}, "
                       f"deployed {it['patched_calls'][OURS]}")
        fc = it["frozen_calls"]
        if not (fc[OURS] > 0 and fc[TVOFF] == 0):
            bad.append(f"item {i}: frozen calls deployed {fc[OURS]}, {TVOFF} {fc[TVOFF]}")
    return bad


def q0b(lab2: dict) -> list[str]:
    """per item: patches fired once per layer (none for lam0 and the two native arms); x2r, h5n
    and lam0 keep the deployed arm's GPU bits in the same run; x2r and lam0 also keep its
    per-step reads and host bits (the single-variable condition of R1-R3)."""
    bad = []
    n = len(lab2["attn_layers"])
    want = {X2R: [n, 0], KONLY: [n, 0], KIVI4: [0, n], LAM0: [0, 0], OURS: [0, 0], KIVI2: [0, 0]}
    for it in lab2["items"]:
        i = it["i"]
        for run, w in want.items():
            if list(it["patched_calls"][run]) != w:
                bad.append(f"item {i} {run}: patched calls {it['patched_calls'][run]} != {w}")
        # lam0 runs the same frozen _attend_h as the deployed arm, the same number of times; the
        # patched arms never reach the frozen function
        fc = it["frozen_calls"]
        if not (fc[LAM0] == fc[OURS] > 0 and fc[X2R] == 0 and fc[KONLY] == 0):
            bad.append(f"item {i}: frozen calls lam0 {fc[LAM0]}, deployed {fc[OURS]}, "
                       f"x2r {fc[X2R]}, h5n {fc[KONLY]}")
        for run in (X2R, KONLY, LAM0):
            if it["eff_bits"][run] != it["eff_bits"][OURS]:
                bad.append(f"item {i} {run}: GPU bits differ from the deployed arm")
        for run in (X2R, LAM0):
            if (it["read_bits"][run] != it["read_bits"][OURS]
                    or it["host_bits"][run] != it["host_bits"][OURS]):
                bad.append(f"item {i} {run}: reads or host bits differ from the deployed arm")
        if not (it["host_bits"][KONLY] < it["host_bits"][OURS]
                and it["read_bits"][KONLY] < it["read_bits"][OURS]):
            bad.append(f"item {i}: h5n does not read and store fewer host bits")
    return bad


def one_process(d: dict, name: str) -> list[str]:
    """every item of a file comes from the process that wrote the file (NAB2 never resumes)."""
    ids = {it.get("run_id") for it in d["items"]}
    if ids != {d.get("run_id")} or None in ids:
        return [f"{name}: items from {len(ids)} processes (run ids {sorted(map(str, ids))[:3]})"]
    return []


def q0d(lab2: dict, rev: dict) -> list[str]:
    """every arm, rerun in a seeded random per-item order on eight items spread over the traces in a
    new process, equals lab2 in every readout, bit count and patch count: no arm depends on which
    arms ran before it in the item, or on which items ran before."""
    bad = one_process(lab2, "lab2") + one_process(rev, "rev")
    if lab2.get("perm_seed") != -1 or rev.get("perm_seed") != REV_SEED:
        bad.append(f"perm seeds lab2 {lab2.get('perm_seed')} / rev {rev.get('perm_seed')}")
    if lab2.get("run_id") == rev.get("run_id"):
        bad.append("rev ran in the same process as lab2")
    if any(it.get("order") != LAB2_RUNS for it in lab2["items"]):
        bad.append("lab2 items did not run in the registered arm order")
    ids = sorted(it["i"] for it in rev["items"])
    if ids != REV_IDS:
        bad.append(f"rev item ids {ids} != {REV_IDS}")
    fwd = {it["i"]: it for it in lab2["items"]}
    for it in rev["items"]:
        f = fwd.get(it["i"])
        if f is None:
            bad.append(f"rev item {it['i']} not in lab2")
            continue
        # the order must be the registered seeded shuffle (as nab2_lab.py: seed * 1000 + item)
        want = list(LAB2_RUNS)
        random.Random(REV_SEED * 1000 + it["i"]).shuffle(want)
        if it.get("order") != want or want == LAB2_RUNS:
            bad.append(f"rev item {it['i']}: order {it.get('order')} is not the seeded "
                       f"shuffle {want}")
        for run in LAB2_RUNS:
            if set(it["bucket"][run]) != set(f["bucket"][run]):
                bad.append(f"item {it['i']} {run}: bucket keys depend on arm order")
                continue
            a, b = readouts(it, run), readouts(f, run)
            diff = [j for j, (x, y) in enumerate(zip(a, b)) if not same(x, y)]
            if len(a) != len(b):
                bad.append(f"item {it['i']} {run}: {len(a)} vs {len(b)} readouts")
            elif diff:
                bad.append(f"item {it['i']} {run}: {len(diff)} readouts depend on arm order")
            if (any(it[k][run] != f[k][run] for k in ("eff_bits", "read_bits", "host_bits"))
                    or list(it["patched_calls"][run]) != list(f["patched_calls"][run])
                    or it["frozen_calls"][run] != f["frozen_calls"][run]):
                bad.append(f"item {it['i']} {run}: bits or patch counts depend on arm order")
    return bad


def gate(tag: str, n1: Path, n2: Path) -> int:
    """the chain's per-model gate after the control replay: provenance, code record, Q0-a, Q0-e."""
    try:
        if tag not in TAGS:
            raise ValueError(f"unknown model tag {tag}")
        nab1, ctl = load(n1 / f"lab_{tag}.json"), load(n2 / f"ctl_{tag}.json")
        marker, mcode = read_marker(n2 / "selftest_pass.txt")
        bad = provenance(nab1, ctl, "ctl", CTL_RUNS, marker, mcode) + q0a(nab1, ctl) + q0e(ctl)
    except ERRORS as e:
        bad = [f"gate error {type(e).__name__}: {e}"]
    print(f"GATE {tag}: " + ("PASS" if not bad else "FAIL " + "; ".join(bad[:6])))
    return 2 if bad else 0


def per_model(tag: str, n1: Path, n2: Path, marker: str, mcode: dict | None) -> dict | None:
    """the registered statistics of one model, or None (with the reasons printed) if void."""
    nab1 = load(n1 / f"lab_{tag}.json")
    ctl, lab2 = load(n2 / f"ctl_{tag}.json"), load(n2 / f"lab2_{tag}.json")
    rev = load(n2 / f"rev_{tag}.json")
    a = {it["i"]: it for it in nab1["items"]}
    b = {it["i"]: it for it in lab2["items"]}
    bad = (provenance(nab1, ctl, "ctl", CTL_RUNS, marker, mcode)
           + provenance(nab1, lab2, "lab2", LAB2_RUNS, marker, mcode)
           + provenance(nab1, rev, "rev", LAB2_RUNS, marker, mcode)
           + q0a(nab1, ctl) + q0e(ctl) + q0d(lab2, rev))
    if not (ctl["versions"] == lab2["versions"] == rev["versions"]):   # one card type, one software
        bad.append(f"versions differ: ctl {ctl['versions']}, lab2 {lab2['versions']}, rev {rev['versions']}")
    if len(lab2["items"]) != 64 or len(b) != 64 or sorted(b) != sorted(a):
        bad.append(f"item sets differ (NAB1 {len(a)}, NAB2 {len(lab2['items'])} records)")
    else:
        bad += q0b(lab2) + q0a(nab1, lab2, None)      # Q0-c: NAB1's two arms reproduce on all 64
    if bad:
        print(f"VOID {tag}: " + "; ".join(bad[:6]))
        return None
    ids = sorted(a)
    # host bits: finite for every arm, except KIVI-2 (a stack name with no host store: None)
    nonfinite = [(i, r) for i in ids for r in LAB2_RUNS
                 if not all(v is None or finite(v) for v in readouts(b[i], r))
                 or not all(finite(b[i][k][r]) for k in ("eff_bits", "read_bits"))
                 or not (finite(b[i]["host_bits"][r])
                         or (r == KIVI2 and b[i]["host_bits"][r] is None))]
    # every ratio's numerator and denominator come from the same lab2 run (same card and code)
    kl = {run: [b[i]["decode"][run][0] for i in ids] for run in LAB2_RUNS}
    nonpos = [run for run, v in kl.items() if not all(finite(x) and x > 0 for x in v)]
    if nonfinite or nonpos:
        print(f"VOID {tag}: non-finite readouts {nonfinite[:4]}, "
              f"non-positive decode KL in {nonpos}")
        return None
    r: dict = {"ours_median_kl": statistics.median(kl[OURS]),
               "kivi2_median_kl": statistics.median(kl[KIVI2])}
    for name, run in ARMS.items():
        rr = [x / y for x, y in zip(kl[run], kl[OURS])]
        r[name] = {"median_ratio_vs_ours": statistics.median(rr),
                   "n_above_1": sum(x > 1 for x in rr), "n": len(rr),
                   "median_kl": statistics.median(kl[run])}
    r["x2r_vs_kivi2_median_ratio"] = statistics.median(
        x / y for x, y in zip(kl[X2R], kl[KIVI2]))
    # bits per element, medians over the 64 traces of the same run: GPU stored, read per decode
    # step, host stored (0.0 where an arm has no host store)
    r["bits_median"] = {
        run: (statistics.median(b[i]["eff_bits"][run] for i in ids),
              statistics.median(b[i]["read_bits"][run] for i in ids),
              statistics.median((b[i]["host_bits"][run] or 0.0) for i in ids))
        for run in LAB2_RUNS}
    # descriptive only (REG section 5): answer-span KL and argmax agreement, medians over items
    # answer = (KL, rows, argmax agreement); items with an empty answer span (None) are skipped
    def med(vals: list) -> float | None:
        v = [x for x in vals if x is not None]
        return statistics.median(v) if v else None
    r["answer_median"] = {run: {"kl": med([b[i]["answer"][run][0] for i in ids]),
                                "argmax_agree": med([b[i]["answer"][run][2] for i in ids])}
                          for run in LAB2_RUNS}
    return r


def lab_r1(r: dict) -> str:
    x = r["x2r"]
    if x["median_ratio_vs_ours"] >= 2 and x["n_above_1"] >= 56:
        return "selection is a main source"
    return "host residual explains most" if x["median_ratio_vs_ours"] <= 1.25 else "mixed"


def lab_r2(r: dict) -> str:
    x = r["lam0"]
    if x["median_ratio_vs_ours"] >= 1.25 and x["n_above_1"] >= 48:
        return "correction material"
    return "correction negligible" if 0.95 <= x["median_ratio_vs_ours"] <= 1.05 else "mixed"


def lab_r3(r: dict) -> str:
    x = r["konly"]
    if x["median_ratio_vs_ours"] >= 1.5 and x["n_above_1"] >= 48:
        return "value correction material"
    return "value correction negligible" if x["median_ratio_vs_ours"] <= 1.1 else "mixed"


def lab_r4(r: dict) -> str:
    x = r["kivi4"]
    if x["median_ratio_vs_ours"] > 1 and x["n_above_1"] >= 48:
        return "KL: ours lower than KIVI-4"
    return "not shown"


def analyze(n1: Path, n2: Path) -> int:
    print(f"nab2_analyze.py md5 {SELF_MD5}")
    marker, mcode = read_marker(n2 / "selftest_pass.txt")
    res = {tag: per_model(tag, n1, n2, marker, mcode) for tag in TAGS}
    if any(v is None for v in res.values()):
        print("NAB2 VOID")
        return 2
    print(json.dumps(res, indent=1))
    # thresholds fixed in REG_NAB2.md section 5 before any output; a rule's overall label needs
    # the same label on all three models, else mixed (R4: not shown)
    for name, fn, mixed in (("R1 (random candidates)", lab_r1, "mixed"),
                            ("R2 (lam = 0)", lab_r2, "mixed"),
                            ("R3 (key-only)", lab_r3, "mixed"),
                            ("R4 (KIVI-4 ruler)", lab_r4, "not shown")):
        labels = {t: fn(res[t]) for t in TAGS}
        overall = labels[TAGS[0]] if len(set(labels.values())) == 1 else mixed
        print(f"{name}: {overall}  | per model: "
              + ", ".join(f"{t} {v}" for t, v in labels.items()))
    print("x2r and lam0 keep the deployed arm's GPU bits, reads and host bits; h5n keeps its GPU "
          "bits and has fewer host bits. Report every ratio with bits_median.")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--gate"] and len(argv) == 4:
        return gate(argv[1], Path(argv[2]), Path(argv[3]))
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        return analyze(Path(argv[0]), Path(argv[1]))
    except ERRORS as e:
        print(f"NAB2 VOID: error while checking: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
