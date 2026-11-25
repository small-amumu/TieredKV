# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""NAB1 read-off (REG_NAB1.md, with amendment A1): per model, the 16-bit trace summary, per-run summaries and the
registered paired comparisons. Reads the trace files and the lab outputs; writes nothing but stdout.

Validity (registered): an item counts only if kivi128, ours and h2o all have a decode KL; a model with fewer than 32
valid items is INVALID (its H1 / H2 verdicts are not reported as supported). H2 uses only items whose eviction budget
could be held (h2o_budget_met, A1); the others are counted and reported separately.

usage: python nab1_analyze.py --lab <model>=<lab_json> [...] --traces <model>=<traces_jsonl> [...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
from pathlib import Path

OURS, BASE, EVICT = "kivi128k64_x2s128_j50_h54", "kivi128", "h2o"
MIN_VALID = 32


def fin(x) -> bool:
    return x is not None and math.isfinite(x)


def fmt(x) -> str:
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.4g}"


def smean(xs: list[float]):
    return st.mean(xs) if xs else None


def smed(xs: list[float]):
    return st.median(xs) if xs else None


def sign_p(k: int, n: int) -> float:
    """Two-sided exact binomial sign test, H0 p = 0.5 (ties already removed from n)."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(items: list[dict], a: str, b: str, key: str) -> dict:
    """Per item ratio of metric key ("decode" or "answer") of run a over run b."""
    rs, absent, dropped = [], 0, 0
    for it in items:
        xa, xb = it[key][a][0], it[key][b][0]
        if xa is None and xb is None:                       # no rows of this kind (e.g. no answer span)
            absent += 1
            continue
        if not (fin(xa) and fin(xb)) or xb <= 0:
            dropped += 1
            continue
        rs.append(xa / xb)
    below, ties = sum(r < 1 for r in rs), sum(r == 1 for r in rs)
    n = len(rs)
    med = smed(rs)
    return {"n": n, "absent": absent, "dropped": dropped, "ties": ties, "median_ratio": med, "below_1": below,
            "sign_p": sign_p(below, n - ties),
            "supported": med is not None and med < 1 and below >= math.ceil(2 * n / 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", nargs="+", required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    a = ap.parse_args()
    traces = dict(x.split("=", 1) for x in a.traces)
    h3: list[str] = []
    for spec in a.lab:
        name, path = spec.split("=", 1)
        res = json.loads(Path(path).read_text(encoding="utf-8"))
        if res.get("smoke"):
            raise SystemExit(f"{path} is a smoke output")
        if name not in traces:
            raise SystemExit(f"no --traces for {name}")
        if Path(res["traces"]).name != Path(traces[name]).name:
            raise SystemExit(f"{name}: lab read {res['traces']}, given {traces[name]}")
        missing = [k for k in ("head_dim", "kv_heads", "attn_layers", "md5", "host_bits") if k not in res]
        if missing:
            raise SystemExit(f"{name}: lab output lacks {missing}")
        tmd5 = hashlib.md5(Path(traces[name]).read_bytes()).hexdigest()
        if res["md5"].get(Path(traces[name]).name) != tmd5:
            raise SystemExit(f"{name}: traces md5 {tmd5} differs from the one the lab read")
        tr = [json.loads(line) for line in Path(traces[name]).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not tr:
            raise SystemExit(f"{name}: empty traces")
        print(f"== {name}: traces {len(tr)}, 16-bit accuracy {100 * sum(t['ok'] for t in tr) / len(tr):.1f}%, "
              f"with </think> {sum(t['think_end'] is not None for t in tr)}, ended {sum(t['ended'] for t in tr)}, "
              f"mean generated tokens {st.mean(t['n_gen'] for t in tr):.0f}")
        items = res["items"]
        runs = res["runs"]
        complete = "seconds" in res and len(items) == len(tr) and {i["i"] for i in items} == {t["i"] for t in tr}
        if not complete:
            print(f"   PARTIAL: lab has {len(items)} of {len(tr)} traces"
                  f"{'' if 'seconds' in res else ', run not finished'}; nothing below is a verdict")
        valid = [i for i in items if all(r in i["decode"] and fin(i["decode"][r][0]) for r in (BASE, OURS, EVICT))]
        def n_nonfinite(key: str) -> int:
            return sum(any(r in i[key] and i[key][r][0] is not None and not fin(i[key][r][0])
                           for r in (BASE, OURS, EVICT)) for i in items)
        met = [i for i in valid if i["h2o_budget_met"]]
        print(f"   replayed {len(items)}, valid {len(valid)} (non-finite KL: decode {n_nonfinite('decode')}, "
              f"answer {n_nonfinite('answer')}), h2o budget held {len(met)} (median L "
              f"{fmt(smed([i['L'] for i in met]))}), cut by max_len "
              f"{sum(i['truncated_by_max_len'] for i in items)}, answer span inside max_len "
              f"{sum(i['think_end'] is not None for i in items)}, median L {fmt(smed([i['L'] for i in items]))}, "
              f"median h2o budget {fmt(smed([i['h2o_budget'] for i in items]))}")
        invalid = len(valid) < MIN_VALID
        if invalid:
            print(f"   INVALID: {len(valid)} valid items < {MIN_VALID}")
        for r in runs:
            dec = [i["decode"][r][0] for i in valid if fin(i["decode"][r][0])]
            ans = [i["answer"][r][0] for i in valid if fin(i["answer"][r][0])]
            agr = [i["answer"][r][2] for i in valid if fin(i["answer"][r][2])]
            rows = sum(i["decode"][r][1] for i in valid if fin(i["decode"][r][0]))
            roww = sum(i["decode"][r][0] * i["decode"][r][1] for i in valid if fin(i["decode"][r][0])) / rows \
                if rows else None
            eff = smed([i["eff_bits"][r] for i in valid])
            # GPU bits at peak window occupancy (A3): KIVI-based arms charge the 128-token FP16 window at mean
            # occupancy (64 tokens), so add 64 * 13.75 / L; the eviction arm's budget is already the peak
            peak = smed([i["eff_bits"][r] + (0.0 if r == EVICT else 64 * 13.75 / i["L"]) for i in valid])
            # per decode step and sequence, host -> GPU bytes at the trace's final length:
            # (read bits - GPU bits) per element x 2 (K, V) x L x head_dim x KV heads x layers / 8
            elems = 2 * res["head_dim"] * res["kv_heads"] * len(res["attn_layers"])
            fetch = smed([(i["read_bits"][r] - i["eff_bits"][r]) * elems * i["L"] / 8 / 1024 for i in valid
                          if i.get("read_bits", {}).get(r) is not None])
            print(f"   {r:28s} GPU bits {fmt(eff)} (at peak window {fmt(peak)}) host bits "
                  f"{res['host_bits'].get(r, 0)} host fetch/step "
                  f"{fmt(fetch)} KiB | decode KL per-trace mean "
                  f"{fmt(smean(dec))} median {fmt(smed(dec))} row-weighted {fmt(roww)} | answer KL median {fmt(smed(ans))}"
                  f" | answer argmax agreement mean {fmt(smean(agr))}, all-agree {sum(x == 1.0 for x in agr)}/{len(agr)}")
            bk: dict = {}
            for i in valid:
                for b, (m, n) in i["bucket"][r].items():
                    if fin(m):
                        bk.setdefault(b, []).append(m)
            print("      buckets " + " ".join(f"{b}:{fmt(smean(v))}(n{len(v)})" for b, v in bk.items()))
        for lbl, other, pool in (("H1 ours vs base only", BASE, valid), ("H2 ours vs eviction", EVICT, met),
                                 ("H2 (budget not held, descriptive)", EVICT, [i for i in valid if not i["h2o_budget_met"]])):
            for key in ("decode", "answer"):
                p = paired(pool, OURS, other, key)
                if not complete:
                    verdict = "PARTIAL"
                elif "descriptive" in lbl:
                    verdict = "descriptive"
                elif invalid or p["n"] < MIN_VALID:               # A2: every registered test needs n >= 32
                    verdict = f"INVALID (n {p['n']} < {MIN_VALID})" if not invalid else "INVALID"
                else:
                    verdict = p["supported"]
                print(f"   {lbl} [{key}]: n {p['n']} (no rows {p['absent']}, dropped {p['dropped']}, ties "
                      f"{p['ties']}), median ratio "
                      f"{fmt(p['median_ratio'])}, below 1 {p['below_1']}/{p['n']}, sign p {fmt(p['sign_p'])}, "
                      f"supported {verdict}")
                if lbl.startswith("H2 ours") and key == "decode":
                    tag = ", PARTIAL" if not complete else (", INVALID" if invalid or p["n"] < MIN_VALID else "")
                    h3.append(f"{name} {fmt(p['median_ratio'])} (n {p['n']}{tag})")
    print("H3 (descriptive) ours / eviction, decode KL, median paired ratio on budget-held items: " + "; ".join(h3))


if __name__ == "__main__":
    main()
