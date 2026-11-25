# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""T1 scorer branch tests on synthetic statistics (no model). Run: python -I t1_aggtest.py <scratch dir>"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import types
from pathlib import Path

stub = types.ModuleType("sel14_q")
stub.S4 = types.SimpleNamespace(DIAG={"on": False})
stub._attend_h = lambda *a, **k: None
stub._res = lambda r, b: r
sys.modules["sel14_q"] = stub
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_q as T1  # noqa: E402
import t1_selftest as TS  # noqa: E402

TS.LK = 4400
MODELS = ("/m/Qwen3-0.6B", "/m/Qwen3-1.7B", "/m/Qwen3-4B")


def synth_lab(model: str) -> dict:
    """4 distinct synthetic items, replicated to 64 ids (the scorer only needs the statistics schema)."""
    distinct = []
    for it in range(4):
        T1.reset_item(it, TS.PREF)
        T1.CFG["on"] = True
        for r0 in range(0, TS.LK, 256):
            r1 = min(TS.LK, r0 + 256)
            qpos, cu, c1, kc, m, prK, ex, al, n = TS.chunk(r0, r1)
            T1._record(0, qpos, cu, c1, kc, m, prK, ex, al, TS.G, TS.KVH, n, TS.LK)
        distinct.append({"0": T1.stats_to_json(T1.STATS[0], T1.STATE[0])})
    items = [{"i": i, "L": TS.LK, "prompt": TS.PREF, "truncated_by_max_len": False,
              "layers": copy.deepcopy(distinct[i % 4])} for i in range(64)]
    return {"model": model, "smoke": False, "items": items, "policies": [T1.pol_name(p) for p in T1.POLICIES],
            "pages": list(T1.PAGES), "buckets": [list(b) for b in T1.BUCKETS], "probation": list(T1.PROB),
            "run": "kivi128k64_x2s128_j50_h54", "chunk": 256, "max_len": 8800, "attn_layers": [0],
            "script_md5": "x", "stack_modules_md5": {},
            "md5": {k: "x" for k in ("t1_lab.py", "t1_q.py", "sel14_q.py", "sel30_q.py", "v8g.py", "bl_q.py",
                                     "cq_lab3.py")}}


def run(d: Path, labs: list) -> tuple[int, dict | None]:
    paths = []
    for k, lab in enumerate(labs):
        lab = dict(lab)
        lab["model"] = MODELS[k] if k < 3 else "/m/extra"
        p = d / f"lab{k}.json"
        p.write_text(json.dumps(lab))
        paths.append(str(p))
    r = subprocess.run([sys.executable, "-I", str(HERE / "t1_analyze.py"), str(d), *paths], capture_output=True, text=True)
    res = json.loads((d / "t1_analysis.json").read_text()) if r.returncode == 0 else None
    return r.returncode, res


def main() -> int:
    d = Path(sys.argv[1])
    d.mkdir(parents=True, exist_ok=True)
    base = synth_lab("/m/a")
    checks = []
    # 1. random data: acc policies look like random -> G1 fails
    rc, res = run(d, [base, base, base])
    checks.append(("G1 fails on null data", rc == 0 and res["G1"]["pass"] is False))
    # 2. make the acc policies' demoted mass tiny -> G1 passes and picks the largest f
    lab = copy.deepcopy(base)
    for it in lab["items"]:
        for lv in it["layers"].values():
            for pn, rows in lv["pol"].items():
                if pn.startswith("score"):
                    for row in rows:
                        row[6] = 0.01 * row[7]
                        row[8] = 0.8 * row[9]           # demoted share 0.8 everywhere
    rc, res = run(d, [lab, lab, lab])
    checks.append(("G1 passes when acc share is tiny", rc == 0 and res["G1"]["pass"] is True))
    checks.append(("G1 picks f = 0.9, smallest G", rc == 0 and res["G1"]["t4_policy"] == "score_G512_f0.9"))
    # 3. one model failing G1 makes G1 fail
    rc, res = run(d, [lab, lab, base])
    checks.append(("G1 needs all three models", rc == 0 and res["G1"]["pass"] is False))
    rc, _ = run(d, [lab, lab])
    checks.append(("two models rejected", rc != 0))
    # 4. G2: random dense synthetic selections give many rows per page but no clustering over random -> fail factor
    rc, res = run(d, [base, base, base])
    checks.append(("G2 fails without clustering over random", rc == 0 and res["G2"]["pass"] is False))
    lab2 = copy.deepcopy(base)
    for it in lab2["items"]:
        for lv in it["layers"].values():
            for row in lv["pages"]["16"]["s1"]:
                row[1] = row[0] / 4.0           # 4 rows per touched page
                row[3] = row[2] / 1.2           # random control 1.2
    rc, res = run(d, [lab2, lab2, lab2])
    checks.append(("G2 passes with 4 rows/page vs random 1.2", rc == 0 and res["G2"]["pass"] is True))
    # 5. completeness: 63 items -> error
    lab3 = copy.deepcopy(base)
    lab3["items"] = lab3["items"][:63]
    rc, _ = run(d, [lab3, base, base])
    checks.append(("63 items rejected", rc != 0))
    lab5 = copy.deepcopy(base)
    del lab5["items"][5]["layers"]
    lab5["items"][5]["skipped"] = "no decode rows"
    rc, _ = run(d, [lab5, base, base])
    checks.append(("a skipped item rejected", rc != 0))
    lab6 = copy.deepcopy(base)
    lab6["chunk"] = 512
    rc, _ = run(d, [lab6, base, base])
    checks.append(("config mismatch across labs rejected", rc != 0))
    lab7 = copy.deepcopy(base)
    for it in lab7["items"]:
        it["layers"]["1"] = copy.deepcopy(it["layers"]["0"])
        for rows in it["layers"]["1"]["pol"].values():
            for row in rows:
                row[7] = 0.0
    lab7["attn_layers"] = [0, 1]
    rc, _ = run(d, [lab7, lab7, lab7])
    checks.append(("a layer with NaN while others finite rejected", rc != 0))
    # 6. score vs oldest-first: equal shares -> T4 uses oldest; score clearly lower -> T4 uses score
    def set_w2(src: dict, kind: str, frac: float) -> None:
        for it in src["items"]:
            for lv in it["layers"].values():
                for pn, rows in lv["pol"].items():
                    if pn.startswith(kind):
                        for row in rows:
                            row[6] = frac * row[7]
    lab8 = copy.deepcopy(lab)
    set_w2(lab8, "oldest", 0.01)
    rc, res = run(d, [lab8, lab8, lab8])
    checks.append(("score no better than oldest -> T4 oldest", rc == 0 and res["G1"]["t4_policy"] == "oldest_G512_f0.9"))
    set_w2(lab8, "oldest", 0.05)
    rc, res = run(d, [lab8, lab8, lab8])
    checks.append(("score <= 0.8 x oldest -> T4 score", rc == 0 and res["G1"]["t4_policy"] == "score_G512_f0.9"))
    # 6b. boundary case: oldest fails the threshold in one model only, score barely beats it there -> no escape
    la, lb = copy.deepcopy(lab), copy.deepcopy(lab)
    set_w2(la, "oldest", 0.50)
    set_w2(lb, "score", 0.10)
    set_w2(lb, "oldest", 0.11)
    rc, res = run(d, [la, la, lb])
    checks.append(("score within 0.8 x oldest in one model -> G1 fails", rc == 0 and res["G1"]["pass"] is False))
    # 6c. random control also within the threshold is reported at the picked setting
    lr = copy.deepcopy(lab8)
    set_w2(lr, "random", 0.05)
    rc, res = run(d, [lr, lr, lr])
    checks.append(("random also within threshold reported", rc == 0 and res["G1"]["random_also_ok_at_pick"] is True))
    # 7. zero final weight in the long bucket with stage-2 rows present -> Q0e
    lab9 = copy.deepcopy(base)
    for lv in lab9["items"][3]["layers"].values():
        for rows in lv["pol"].values():
            rows[3][6] = 0.0
            rows[3][7] = 0.0
    rc, _ = run(d, [lab9, base, base])
    checks.append(("long rows with zero final weight rejected", rc != 0))
    # 8. items that never reach the long bucket: excluded from n and the long medians; n = 14 fails G1 and G2
    lab10 = copy.deepcopy(lab)
    for it in lab10["items"][14:]:
        for lv in it["layers"].values():
            for rows in lv["pol"].values():
                rows[3][:] = [0.0] * len(rows[3])
            for st_ in lv["pages"].values():
                for rows in st_.values():
                    rows[3][:] = [0.0] * len(rows[3])
            for key in ("churn", "union"):
                if key in lv:
                    lv[key][3][:] = [0.0] * len(lv[key][3])
    for row_it in lab10["items"][:14]:                 # clustered pages on the 14 long items: only n blocks G2
        for lv in row_it["layers"].values():
            for st_ in lv["pages"].values():
                for rows in st_.values():
                    for row in rows:
                        row[1] = row[0] / 4.0
                        row[3] = row[2] / 1.2
    rc, res = run(d, [lab10, lab10, lab10])
    checks.append(("n_long = 14 counted, G1 and G2 fail", rc == 0 and res["models"][0]["n_long_items"] == 14
                   and res["G1"]["pass"] is False and res["G2"]["pass"] is False))
    lab4 = copy.deepcopy(base)
    lab4["smoke"] = True
    rc, _ = run(d, [lab4, base, base])
    checks.append(("smoke output rejected", rc != 0))
    for name, ok in checks:
        print(("ok   " if ok else "FAIL ") + name)
    n_fail = sum(not ok for _, ok in checks)
    print(f"{len(checks) - n_fail}/{len(checks)} passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
