# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""NAB2 self-test (REG_NAB2.md, Q0-b): check the patches on the smoke replay of one trace and, on
a pass, write the Q0-b marker.

The smoke run is exactly (on STAR, in the nab2 directory, after deleting out/smoke.smoke.json):
  nab2_lab.py ~/work/models/Qwen3-0.6B ~/work/nab/nab1/out/traces_qwen3_0p6b.jsonl
    out/smoke.smoke.json --stack ~/work/nab --smoke --items 0 --max_len 1500 --chunk 256
    --runs <the 8 runs>
with the runs kivi128 and
kivi128k64_x2s128_j50_h54 (NAB1 routes), t_base2 and t_variant_off (patches with every ablation
off), and the four arms kivi128k64_x2r128_j50_h54, kivi128k64_x2s128_j50_h5n, kivi4b128,
kivi128k64_x2s128_h54. Checks (all must pass, else exit 1 and no marker is written):
  0. the smoke run was written by the nab2_q.py and nab2_lab.py of this directory, with the
     configuration above (NAB1's Qwen3-0.6B model path, NAB1's 0.6B trace file by name and md5,
     item 0, max_len 1500, chunk 256, float32, exactly these runs);
  1. t_variant_off equals kivi128k64_x2s128_j50_h54 and t_base2 equals kivi128 in every
     readout, bit for bit;
  2. every patched arm fired its patch exactly once per attention layer on its own counter and
     never on the other; lam0 and the two stack routes fired none;
  3. every readout is finite (None only for an empty span) and every decode KL is positive;
  4. each ablation changes the output: the decode KL of x2r, h5n, lam0 and kivi4b128 differs
     from its route's;
  5. x2r, h5n and lam0 keep the GPU bits of NAB1's arm; x2r and lam0 also keep its host bits and
     per-step reads; h5n reads and stores fewer host bits; the KIVI-4 ruler's bit rule at 2 bits
     (t_base2) gives kivi128's GPU bits and reads exactly. The x2r equalities hold by
     construction (same route, same accounting); checks 2 and 4 show that its patch ran.
Any earlier marker is removed first, so a failed rerun leaves none.
On a pass, <marker_out> gets two lines: "PASS" with the md5 prefixes of nab2_q.py, nab2_lab.py,
nab2_selftest.py, nab2_chain.sh and nab2_analyze.py (this directory), then the smoke run's md5
record (every stack module the replay loaded, the calibration text) without the trace file, as
JSON.
nab2_chain.sh starts only on a matching marker; nab2_analyze.py requires every production md5
record to equal the one in the marker.
usage: python nab2_selftest.py <smoke_json> <marker_out>
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = ("nab2_q.py", "nab2_lab.py", "nab2_selftest.py", "nab2_chain.sh", "nab2_analyze.py")
OURS, KIVI2 = "kivi128k64_x2s128_j50_h54", "kivi128"
X2R, LAM0 = "kivi128k64_x2r128_j50_h54", "kivi128k64_x2s128_h54"
KONLY, KIVI4 = "kivi128k64_x2s128_j50_h5n", "kivi4b128"
NEED = (KIVI2, OURS, "t_base2", "t_variant_off", X2R, KONLY, KIVI4, LAM0)
# NAB1's 0.6B model path and trace file (nab1/out/lab_qwen3_0p6b.json: model, md5 record)
NAB1_MODEL = "/home/user/work/models/Qwen3-0.6B"
NAB1_TRACE = ("traces_qwen3_0p6b.jsonl", "cb4c407c45b387f034076323ec9bcf3c")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def readouts(it: dict, run: str) -> list:
    vals = list(it["decode"][run]) + list(it["answer"][run])
    for b in it["bucket"][run].values():
        vals += list(b)
    return vals


def check(d: dict) -> list[str]:
    bad: list[str] = []
    cfg = {"dtype": "float32", "chunk": 256, "max_len": 1500}
    for k, v in cfg.items():
        if d.get(k) != v:
            bad.append(f"0: smoke {k} {d.get(k)} != {v}")
    if d["items"][0]["i"] != 0 or sorted(d["runs"]) != sorted(NEED):
        bad.append(f"0: smoke item {d['items'][0]['i']} / runs {d['runs']} "
                   f"are not item 0 / {NEED}")
    tr_name, tr_md5 = NAB1_TRACE
    if (d["model"] != NAB1_MODEL or Path(d["traces"]).name != tr_name
            or d["md5"].get(tr_name) != tr_md5):
        bad.append(f"0: smoke model {d['model']} / traces {d['traces']} "
                   f"(md5 {d['md5'].get(tr_name)}) are not NAB1's 0.6B model and traces")
    for f in ("nab2_q.py", "nab2_lab.py"):
        if d["md5"].get(f) != md5(HERE / f):
            bad.append(f"0: smoke run md5 of {f} {d['md5'].get(f)} != this directory's")
    it = d["items"][0]
    for test, ref in (("t_variant_off", OURS), ("t_base2", KIVI2)):
        for key in ("bucket", "decode", "answer"):
            if it[key][test] != it[key][ref]:
                bad.append(f"1: {test} {key} {it[key][test]} != {ref} {it[key][ref]}")
    n = len(d["attn_layers"])
    want = {"t_variant_off": [n, 0], X2R: [n, 0], KONLY: [n, 0], "t_base2": [0, n],
            KIVI4: [0, n], LAM0: [0, 0], KIVI2: [0, 0], OURS: [0, 0]}
    for run, w in want.items():           # [patched _attend_h, patched base] per replay forward
        if list(it["patched_calls"][run]) != w:
            bad.append(f"2: {run} patched calls {it['patched_calls'][run]}, expected {w}")
    for run in NEED:
        if not all(v is None or finite(v) for v in readouts(it, run)):
            bad.append(f"3: {run} has a non-finite readout")
        v = it["decode"][run][0]
        if not (finite(v) and v > 0):
            bad.append(f"3: {run} decode KL {v} is not finite and positive")
    for run, ref in ((X2R, OURS), (KONLY, OURS), (LAM0, OURS), (KIVI4, KIVI2)):
        if it["decode"][run][0] == it["decode"][ref][0]:
            bad.append(f"4: {run} decode KL equals {ref}: the ablation changed nothing")
    for run in (X2R, KONLY, LAM0):
        if it["eff_bits"][run] != it["eff_bits"][OURS]:
            bad.append(f"5: {run} GPU bits {it['eff_bits'][run]} != {it['eff_bits'][OURS]}")
    for run in (X2R, LAM0):
        for key in ("read_bits", "host_bits"):
            if it[key][run] != it[key][OURS]:
                bad.append(f"5: {run} {key} {it[key][run]} != {it[key][OURS]}")
    for key in ("eff_bits", "read_bits"):
        if it[key]["t_base2"] != it[key][KIVI2]:
            bad.append(f"5: t_base2 {key} {it[key]['t_base2']} != kivi128 {it[key][KIVI2]}")
    for key in ("read_bits", "host_bits"):
        if not (finite(it[key][KONLY]) and finite(it[key][OURS])
                and it[key][KONLY] < it[key][OURS]):
            bad.append(f"5: h5n {key} {it[key][KONLY]} not below NAB1's arm {it[key][OURS]}")
    return bad


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    out = Path(sys.argv[2])
    tmp = out.with_name(out.name + ".tmp")          # complete marker or none
    try:
        out.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
        d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        missing = [r for r in NEED if r not in d["items"][0]["decode"]]
        if not d.get("smoke") or len(d["items"]) != 1:
            bad = ["not a one-item smoke file"]
        elif missing:
            bad = [f"runs missing {missing}"]
        else:
            bad = check(d)
            it = d["items"][0]
            print(json.dumps({r: {"decode": it["decode"][r], "gpu": it["eff_bits"][r],
                                  "read": it["read_bits"][r], "host": it["host_bits"][r],
                                  "calls": it["patched_calls"][r]} for r in NEED}, indent=1))
            code = {k: v for k, v in d["md5"].items() if k != Path(d["traces"]).name}
            line = "PASS " + " ".join(md5(HERE / f)[:8] for f in CODE)
    except Exception as e:                           # any error: no marker
        bad = [f"check error {type(e).__name__}: {e}"]
    if bad:
        print("SELFTEST FAIL")
        print("\n".join(bad))
        return 1
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n" + json.dumps(code, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"SELFTEST PASS; marker written: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
