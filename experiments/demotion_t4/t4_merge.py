# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""Merge the A2b arms (h44, h53) into the held-out replay of the same traces, after checking that the arms both runs
share (h54, sc_G1024_h32_75) agree.

usage: python -I t4_merge.py <held_out_reduced.json> <a2b_reduced.json> <out_merged.json> <report.txt>
The two runs used different GPUs (RTX 5090 vs RTX PRO 5000) and stack copies, so bitwise equality is not expected;
the report gives, per shared arm, the share of items with identical decode KL and the per-item log ratio of decode
KL (a2b / held-out): geometric mean, median and largest |log ratio|. The noise floor of the compressed arms under a
1e-7 input perturbation is about 14% of their KL (issue log 213), the scale to compare against.
The merged file is the held-out run with the A2b-only arms appended (item fields bucket / decode / answer / layers).
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

SHARED = ("h54", "sc_G1024_h32_75")


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    ho = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    ab = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    lines = [f"held-out {ho['reduced_from']['md5']} ({ho['versions'].get('gpu')}), "
             f"a2b {ab['reduced_from']['md5']} ({ab['versions'].get('gpu')})"]
    if Path(ho["model"]).name != Path(ab["model"]).name:
        raise SystemExit(f"model differs: {ho['model']} vs {ab['model']}")
    if Path(ho["traces"]).name != Path(ab["traces"]).name:
        raise SystemExit(f"traces differ: {ho['traces']} vs {ab['traces']}")
    hi = {it["i"]: it for it in ho["items"]}
    ai = {it["i"]: it for it in ab["items"]}
    if set(hi) != set(ai):
        raise SystemExit(f"item sets differ: {sorted(set(hi) ^ set(ai))}")
    for i in hi:
        if hi[i]["L"] != ai[i]["L"]:
            raise SystemExit(f"item {i}: L {hi[i]['L']} vs {ai[i]['L']}")
    sm_h, sm_a = ho.get("stack_modules_md5", {}), ab.get("stack_modules_md5", {})
    md = sorted(k for k in set(sm_h) | set(sm_a) if sm_h.get(k) != sm_a.get(k))
    lines.append(f"stack_modules_md5: {len(sm_h)} vs {len(sm_a)} modules, differing: {md or 'none'}")
    lines.append(f"t4_lab script md5: held-out {ho['script_md5']}, a2b {ab['script_md5']}")
    for arm in SHARED:
        if arm not in ho["runs"] or arm not in ab["runs"]:
            raise SystemExit(f"shared arm {arm} missing")
        lr, same = [], 0
        for i in hi:
            a, b = hi[i]["decode"][arm]["kl"], ai[i]["decode"][arm]["kl"]
            same += a == b
            if a and b:
                lr.append(math.log(b / a))
        lines.append(f"{arm}: identical decode KL in {same}/{len(hi)} items; log ratio a2b/held-out: "
                     f"gmean {math.exp(statistics.fmean(lr)):.4f}, median {math.exp(statistics.median(lr)):.4f}, "
                     f"max |log| {max(abs(x) for x in lr):.4f} (ratio {math.exp(max(abs(x) for x in lr)):.3f})")
    new = [a for a in ab["runs"] if a not in ho["runs"]]
    out = dict(ho)
    out["runs"] = ho["runs"] + new
    out["merged_from"] = {"held_out": ho["reduced_from"], "a2b": ab["reduced_from"], "a2b_arms": new}
    items = []
    for it in ho["items"]:
        m = dict(it)
        src = ai[it["i"]]
        for key in ("bucket", "decode", "answer", "layers"):
            if key in it:
                m[key] = dict(it[key])
                for arm in new:
                    if arm in src.get(key, {}):
                        m[key][arm] = src[key][arm]
        items.append(m)
    out["items"] = items
    Path(sys.argv[3]).write_text(json.dumps(out), encoding="utf-8", newline="\n")
    lines.append(f"merged arms: {out['runs']}")
    Path(sys.argv[4]).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
