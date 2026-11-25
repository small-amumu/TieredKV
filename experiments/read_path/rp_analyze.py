# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Analysis of read-path microbenchmark RP (PLAN_RP.md section 5 and amendments 1-2). Descriptive only: no thresholds.

Input: one rp_probe.py JSON-lines file per machine. Output: <out>.json (every computed row) and <out>.md (tables).
usage: python -I -X utf8 rp_analyze.py <out prefix> <rp_full.jsonl> [<rp_full.jsonl> ...]
       [--allow_dirty] [--allow_short]     (test only: keep points measured with other GPU processes / short loops)

Validity of a timing point (counted per reason; never silently dropped):
  readback_bad == 0; procs_clean; t_ok; host page mode coverage (thp: huge_frac_min >= 0.9, 4k: huge_frac_max == 0).
PCIe ratios additionally need pcie_ok. DMA records: readback_bad == 0, procs_clean, t_ok (PCIe needs pcie_ok).
Fixed launch overhead: each point's median step time minus the same-mode empty-graph baseline (both reported).
Uncertainty: per-point interquartile range of step times; a ratio a/b is given with the range
[a_q1 / b_q3, a_q3 / b_q1] (conservative, descriptive).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kinds = defaultdict(list)
    for r in recs:
        kinds[r["kind"]].append(r)
    if len(kinds["env"]) != 1:
        raise SystemExit(f"{path}: expected one env record, found {len(kinds['env'])}")
    if len(kinds["end"]) != 1:
        raise SystemExit(f"{path}: expected one end record, found {len(kinds['end'])} (incomplete file?)")
    return kinds


def coverage_ok(r: dict) -> bool:
    if r.get("place") != "host":
        return True
    if r["pm"] == "thp":
        return r.get("huge_frac_min", 0.0) >= 0.9
    if r["pm"] == "4k":
        return r.get("huge_frac_max", 1.0) == 0.0
    return False


def validity(r: dict, allow_dirty: bool, allow_short: bool) -> list[str]:
    bad = []
    if r.get("readback_bad", 1) != 0:
        bad.append("readback")
    if not r.get("procs_clean", False) and not allow_dirty:
        bad.append("procs")
    if not r.get("t_ok", False) and not allow_short:
        bad.append("short")
    if not coverage_ok(r):
        bad.append("coverage")
    return bad


def ratio(a: dict, b: dict, key: str = "ms") -> dict:
    """a / b with the conservative IQR range; a, b carry <key>_med, _q1, _q3 (already net of the baseline).
    None when a net time is not positive (the step is not longer than the empty-graph baseline)."""
    if b[f"{key}_med"] <= 0 or a[f"{key}_med"] <= 0:
        return None
    m = a[f"{key}_med"] / b[f"{key}_med"]
    lo = a[f"{key}_q1"] / b[f"{key}_q3"] if a[f"{key}_q1"] > 0 and b[f"{key}_q3"] > 0 else None
    hi = a[f"{key}_q3"] / b[f"{key}_q1"] if b[f"{key}_q1"] > 0 else None
    return {"ratio": m, "lo": lo, "hi": hi}


def fmt_r(x: dict | None) -> str:
    if x is None:
        return "-"
    lo = "?" if x["lo"] is None else f"{x['lo']:.2f}"
    hi = "?" if x["hi"] is None else f"{x['hi']:.2f}"
    return f"{x['ratio']:.2f} [{lo}, {hi}]"


def analyze(path: Path, allow_dirty: bool, allow_short: bool) -> dict:
    k = load(path)
    env, end = k["env"][0], k["end"][0]
    if not end.get("complete"):
        print(f"WARNING {path}: end record says complete = {end.get('complete')} (why: {end.get('why')})", flush=True)
    base = {}
    for b in k["baseline"]:
        if not b.get("procs_clean", False) and not allow_dirty or not b.get("t_ok", False) and not allow_short:
            raise SystemExit(f"{path}: baseline {b['mode']} is not valid (procs_clean {b.get('procs_clean')}, "
                             f"t_ok {b.get('t_ok')})")
        if b["mode"] in base:
            raise SystemExit(f"{path}: two baselines for mode {b['mode']}")
        base[b["mode"]] = b["ms_med"]
    if set(base) != {"all", "layer"}:
        raise SystemExit(f"{path}: baselines for modes all and layer expected, got {sorted(base)}")
    excl = Counter()
    flags = Counter()
    pts = {}
    for r in k["point"]:
        why = validity(r, allow_dirty, allow_short)
        for w in why:
            excl[w] += 1
        if why:
            continue
        b = base[r["mode"]]
        n = r["useful_rows"]
        q = dict(r)
        q.update({"ms_net_med": r["ms_med"] - b, "ms_net_q1": r["ms_q1"] - b, "ms_net_q3": r["ms_q3"] - b})
        q["ns_row_net"] = q["ms_net_med"] * 1e6 / n
        q["ns_row_raw"] = r["ms_med"] * 1e6 / n
        if q["ms_net_med"] <= 0:
            flags["nonpositive_net"] += 1     # kept, but every ratio with it is None
        key = (r["place"], r["pm"], r["codec"], r["B"], r["L"], r["arr"], r["path"], r["mode"], r["pat"], r["P"],
               r["h"], bool(r.get("ladder")))
        if key in pts:
            raise SystemExit(f"{path}: duplicate point {key}")
        pts[key] = q
    skip_records = len(k["skip"])         # (pattern, arr) combinations not run: too few pages at this L

    def net(q: dict) -> dict:
        return {"ms_med": q["ms_net_med"], "ms_q1": q["ms_net_q1"], "ms_q3": q["ms_net_q3"]}

    def get(place, pm, codec, B, L, arr, path, mode, pat="rand", P=0, h=0, ladder=False):
        return pts.get((place, pm, codec, B, L, arr, path, mode, pat, P, h, ladder))

    main_B = sorted({k_[3] for k_ in pts if k_[0] == "host" and not k_[11]})
    main_L = sorted({k_[4] for k_ in pts if k_[0] == "host" and not k_[11]})
    ladder_L = sorted({k_[4] for k_ in pts if k_[11]} | {L for L in main_L})
    out: dict = {"file": str(path), "host": env["host"], "gpu": env["gpu"], "iommu_group_type": env["iommu_group_type"],
                 "iommu_classes": env["iommu_classes"], "driver": env["driver"], "script_md5": env["script_md5"],
                 "end": {x: end.get(x) for x in ("complete", "error", "why", "points", "readback_bad_points",
                                                 "pcie_bad_points", "procs_dirty_points", "t_bad_points",
                                                 "alloc_errors", "dma_records", "dma_bad")},
                 "baseline_ms": base, "excluded": dict(excl), "flags": dict(flags), "skip_records": skip_records,
                 "kept_points": len(pts),
                 "points_in_file": len(k["point"]), "allow_dirty": allow_dirty, "allow_short": allow_short}
    # every kept point, raw and net of the baseline (plan amendment 2: report both)
    out["points"] = [{"place": kk[0], "pm": kk[1], "codec": kk[2], "B": kk[3], "L": kk[4], "arr": kk[5],
                      "path": kk[6], "mode": kk[7], "pat": kk[8], "P": kk[9], "h": kk[10], "ladder": kk[11],
                      "ns_row_raw": q["ns_row_raw"], "ns_row_net": q["ns_row_net"],
                      "ns_row_net_iqr": [q["ms_net_q1"] * 1e6 / q["useful_rows"], q["ms_net_q3"] * 1e6 / q["useful_rows"]],
                      "pcie_rx_over_useful": q["pcie_rx_over_useful"] if q["pcie_ok"] else None,
                      "pcie_rx_over_read": q.get("pcie_rx_over_read") if q["pcie_ok"] else None}
                     for kk, q in sorted(pts.items(), key=lambda x: str(x[0]))]
    # PTX load / store widths per path (plan section 3.4)
    out["ptx"] = [{"path": p["path"], "arr": p.get("arr"), "codec": p.get("codec"), "nw": p["nw"], "P": p["P"],
                   "insts": p.get("insts")} for p in k["ptx"]]

    # ---- T2a: 4k vs thp, rand, row/layer, h54
    t2a = []
    for B in main_B:
        for L in main_L:
            for arr in ("k", "v"):
                a = get("host", "thp", 54, B, L, arr, "row", "layer")
                b = get("host", "4k", 54, B, L, arr, "row", "layer")
                g = get("gpu", "-", 54, B, L, arr, "row", "layer")
                t2a.append({"B": B, "L": L, "arr": arr,
                            "ns_4k": b["ns_row_net"] if b else None, "ns_thp": a["ns_row_net"] if a else None,
                            "ns_gpu": g["ns_row_net"] if g else None,
                            "thp_over_4k": ratio(net(a), net(b)) if a and b else None,
                            "rx_over_useful_4k": b["pcie_rx_over_useful"] if b and b["pcie_ok"] else None,
                            "rx_over_useful_thp": a["pcie_rx_over_useful"] if a and a["pcie_ok"] else None})
    out["t2a_page_size"] = t2a

    # ---- T2b: ladder (B 1, rand, row/layer), 4k and thp
    t2b = []
    for L in ladder_L:
        for arr in ("k", "v"):
            row = {"L": L, "arr": arr}
            for pm in ("4k", "thp"):
                q = get("host", pm, 54, 1, L, arr, "row", "layer", ladder=L not in main_L)
                row[f"ns_{pm}"] = q["ns_row_net"] if q else None
                row[f"iqr_{pm}"] = ([q["ms_net_q1"] * 1e6 / q["useful_rows"], q["ms_net_q3"] * 1e6 / q["useful_rows"]]
                                    if q else None)
            t2b.append(row)
    out["t2b_ladder"] = t2b

    # ---- T2c: scalar vs 16-byte loads (rows / row), T2d: one launch for all layers vs per layer (all / layer)
    t2c, t2d = [], []
    for (place, pm, codec, B, L, arr, path, mode, pat, P, h, lad), q in sorted(pts.items(), key=lambda x: str(x[0])):
        if place != "host" or pat != "rand" or path != "row" or mode != "layer":
            continue
        s = get(place, pm, codec, B, L, arr, "rows", "layer", ladder=lad)
        a = get(place, pm, codec, B, L, arr, "row", "all", ladder=lad)
        tag = {"pm": pm, "codec": codec, "B": B, "L": L, "arr": arr, "ladder": lad}
        if s:
            t2c.append(dict(tag, scalar_over_vec=ratio(net(s), net(q))))
        if a:
            t2d.append(dict(tag, all_over_layer=ratio(net(a), net(q)), ns_layer=q["ns_row_net"],
                            ns_all=a["ns_row_net"]))
    out["t2c_load_width"] = t2c
    out["t2d_concurrency"] = t2d

    # ---- codec 32 vs 54 (bytes per row), rand row/layer at the largest main L
    t2e = []
    for B in main_B:
        for pm in ("4k", "thp"):
            for arr in ("k", "v"):
                a = get("host", pm, 32, B, max(main_L), arr, "row", "layer")
                b = get("host", pm, 54, B, max(main_L), arr, "row", "layer")
                if a and b:
                    t2e.append({"B": B, "L": max(main_L), "pm": pm, "arr": arr, "row_bytes_32": a["row_bytes"],
                                "row_bytes_54": b["row_bytes"], "h32_over_h54": ratio(net(a), net(b))})
    out["t2e_codec"] = t2e

    # ---- DMA: per-transfer cost, valid records only
    dma = defaultdict(dict)
    dexcl = Counter()
    for r in k["dma"]:
        why = validity(r, allow_dirty, allow_short)
        for w in why:
            dexcl[w] += 1
        if why:
            continue
        dma[(r["pm"], r["L"], r["arr"], r["P"], r["streams"])][r["what"]] = r
    out["dma_excluded"] = dict(dexcl)

    # ---- T6 gate 1: page patterns, per (B, L, pm, arr, P, h)
    t6 = []
    for (place, pm, codec, B, L, arr, path, mode, pat, P, h, lad), q in sorted(pts.items(), key=lambda x: str(x[0])):
        if place != "host" or pat != "page" or path != "row" or mode != "layer" or lad:
            continue
        sh = get(place, pm, codec, B, L, arr, "row_shuf", "layer", "page", P, h)
        pz = get(place, pm, codec, B, L, arr, "pagezc", "layer", "page", P, h)
        rec = {"pm": pm, "B": B, "L": L, "arr": arr, "P": P, "h": h, "ns_row": q["ns_row_net"],
               "ns_row_shuf": sh["ns_row_net"] if sh else None, "ns_pagezc": pz["ns_row_net"] if pz else None,
               "shuf_over_row": ratio(net(sh), net(q)) if sh else None,
               "pagezc_over_row": ratio(net(pz), net(q)) if pz else None,
               "pagezc_over_shuf": ratio(net(pz), net(sh)) if pz and sh else None}
        for ns in (1, 4):
            d = dma.get((pm, L, arr, P, ns), {})
            if B == 1 and "words" in d and "meta" in d:
                per_page = d["words"]["ns_per_transfer"] + d["meta"]["ns_per_transfer"]
                rec[f"ns_dma_s{ns}"] = per_page / h
                rec[f"dma_s{ns}_over_row"] = per_page / h / q["ns_row_net"] if q["ns_row_net"] > 0 else None
            else:
                rec[f"ns_dma_s{ns}"] = None
                rec[f"dma_s{ns}_over_row"] = None
        t6.append(rec)
    out["t6_gate1"] = t6

    # crossover: smallest h at which a path is cheaper per useful row than the row gather (median ratio < 1),
    # against both the sorted and the shuffled row order
    cross = []
    groups = defaultdict(list)
    for r in t6:
        groups[(r["pm"], r["B"], r["L"], r["arr"], r["P"])].append(r)
    for g, rows in sorted(groups.items(), key=lambda x: str(x[0])):
        rows.sort(key=lambda r: r["h"])
        c = {"pm": g[0], "B": g[1], "L": g[2], "arr": g[3], "P": g[4]}
        for name, fld in (("pagezc_vs_row", "pagezc_over_row"), ("pagezc_vs_shuf", "pagezc_over_shuf"),
                          ("dma_s1_vs_row", "dma_s1_over_row"), ("dma_s4_vs_row", "dma_s4_over_row")):
            hs = []
            for r in rows:
                v = r[fld]
                v = v["ratio"] if isinstance(v, dict) else v
                if v is not None and v < 1:
                    hs.append(r["h"])
            have = any(r[fld] is not None for r in rows)
            c[name] = (min(hs) if hs else "none") if have else "n/a"
            c[name + "_all_h"] = hs                                  # every h with a median ratio below 1
        cross.append(c)
    out["t6_crossover"] = cross
    return out


def md_report(res: list[dict]) -> str:
    L = ["# RP analysis (descriptive; no thresholds)", ""]
    for r in res:
        test = r["allow_dirty"] or r["allow_short"]
        L += [f"## {r['host']} - {r['gpu']} (IOMMU group type: {r['iommu_group_type']}; driver {r['driver']})", "",
              *([f"- **TEST ANALYSIS, NOT A RESULT: allow_dirty={r['allow_dirty']}, allow_short={r['allow_short']} "
                 f"(points measured with other GPU processes or too-short loops were kept)**"] if test else []),
              f"- file `{r['file']}`, script md5 {r['script_md5']}; end: {r['end']}",
              f"- empty-graph baselines (ms): {r['baseline_ms']}; points in file {r['points_in_file']}, kept "
              f"{r['kept_points']}; excluded by reason {r['excluded']}; flags {r['flags']}; skipped (pattern, arr) "
              f"combinations {r['skip_records']}; DMA excluded {r['dma_excluded']}", "",
              "### T2a page size (rand, row gather, per-layer launches; ns per row net of the baseline)", "",
              "| B | L | arr | 4K | 2M THP | THP/4K [IQR range] | GPU-resident | RX/useful 4K | RX/useful THP |",
              "|---|---|---|---|---|---|---|---|---|"]
        f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        for t in r["t2a_page_size"]:
            L.append(f"| {t['B']} | {t['L']} | {t['arr']} | {f(t['ns_4k'])} | {f(t['ns_thp'])} | "
                     f"{fmt_r(t['thp_over_4k'])} | {f(t['ns_gpu'])} | {f(t['rx_over_useful_4k'])} | "
                     f"{f(t['rx_over_useful_thp'])} |")
        L += ["", "### PTX global load / store instructions per kernel", "",
              "| path | arr | codec | words | P | instructions |", "|---|---|---|---|---|---|"]
        for p in r["ptx"]:
            ins = ", ".join(f"{k} x{v}" for k, v in sorted((p["insts"] or {}).items()))
            L.append(f"| {p['path']} | {p['arr']} | {p['codec']} | {p['nw']} | {p['P']} | {ins or '-'} |")
        L += ["", "### T2b context ladder (B 1, rand, row gather; ns per row net)", "",
              "| L | arr | 4K | THP |", "|---|---|---|---|"]
        for t in r["t2b_ladder"]:
            L.append(f"| {t['L']} | {t['arr']} | {f(t['ns_4k'])} | {f(t['ns_thp'])} |")
        L += ["", "### T2c scalar 32-bit vs 16-byte vector loads (rows / row)", "",
              "| pm | codec | B | L | arr | ratio [IQR range] |", "|---|---|---|---|---|---|"]
        for t in r["t2c_load_width"]:
            L.append(f"| {t['pm']} | {t['codec']} | {t['B']} | {t['L']}{' (ladder)' if t['ladder'] else ''} | "
                     f"{t['arr']} | {fmt_r(t['scalar_over_vec'])} |")
        L += ["", "### T2d one launch for all layers vs per-layer launches (all / layer)", "",
              "| pm | B | L | arr | per-layer ns | all ns | all/layer |", "|---|---|---|---|---|---|---|"]
        for t in r["t2d_concurrency"]:
            if t["codec"] == 54:
                L.append(f"| {t['pm']} | {t['B']} | {t['L']}{' (ladder)' if t['ladder'] else ''} | {t['arr']} | "
                         f"{f(t['ns_layer'])} | {f(t['ns_all'])} | {fmt_r(t['all_over_layer'])} |")
        L += ["", "### T2e h32 vs h54 rows (bytes per row)", "",
              "| B | L | pm | arr | bytes 32 / 54 | time ratio |", "|---|---|---|---|---|---|"]
        for t in r["t2e_codec"]:
            L.append(f"| {t['B']} | {t['L']} | {t['pm']} | {t['arr']} | {t['row_bytes_32']} / {t['row_bytes_54']} | "
                     f"{fmt_r(t['h32_over_h54'])} |")
        L += ["", "### T6 gate 1 (ns per useful row; ratios against the sorted row gather)", "",
              "| pm | B | L | arr | P | h | row | row shuf | page zc | zc/row | zc/shuf | DMA s1 | DMA s4 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in r["t6_gate1"]:
            L.append(f"| {t['pm']} | {t['B']} | {t['L']} | {t['arr']} | {t['P']} | {t['h']} | {f(t['ns_row'])} | "
                     f"{f(t['ns_row_shuf'])} | {f(t['ns_pagezc'])} | {fmt_r(t['pagezc_over_row'])} | "
                     f"{fmt_r(t['pagezc_over_shuf'])} | {f(t['ns_dma_s1'])} | {f(t['ns_dma_s4'])} |")
        L += ["", "### T6 crossover: smallest h with a median ratio below 1 ('none' = never; 'n/a' = not measured)",
              "", "| pm | B | L | arr | P | page zc vs row | page zc vs shuffled | DMA s1 vs row | DMA s4 vs row |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in r["t6_crossover"]:
            L.append(f"| {c['pm']} | {c['B']} | {c['L']} | {c['arr']} | {c['P']} | {c['pagezc_vs_row']} | "
                     f"{c['pagezc_vs_shuf']} | {c['dma_s1_vs_row']} | {c['dma_s4_vs_row']} |")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--allow_dirty", action="store_true")
    ap.add_argument("--allow_short", action="store_true")
    a = ap.parse_args()
    res = [analyze(Path(f), a.allow_dirty, a.allow_short) for f in a.files]
    Path(a.out + ".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path(a.out + ".md").write_text(md_report(res), encoding="utf-8")
    for r in res:
        tag = " [TEST: allow flags set]" if r["allow_dirty"] or r["allow_short"] else ""
        print(f"{r['host']}{tag}: kept {r['kept_points']} of {r['points_in_file']} points, excluded {r['excluded']},"
              f" flags {r['flags']}, DMA excluded {r['dma_excluded']}, end complete {r['end']['complete']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
