# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Tables of rp_cpu.py output (T3): numeric check, CPU throughput, round trips. Formatting only, no thresholds.
usage: python -I -X utf8 t3_summary.py <out.md> <t3_*.jsonl> [<t3_*.jsonl> ...]
A CPU row is marked INVALID when the id pool touched less than 2 x LLC (rows may come from cache) or fewer than 32
layer regions were used (layers shared regions). Units: us = microseconds; GB = 1e9 bytes; LLC in MiB."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(f: str) -> list[dict]:
    recs = []
    for i, line in enumerate(Path(f).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"WARNING {f}: line {i + 1} is not JSON (truncated file?), skipped", file=sys.stderr)
    return recs


def main() -> int:
    out = Path(sys.argv[1])
    L = ["# T3 near-data CPU correction: measured tables", ""]
    for f in sys.argv[2:]:
        recs = load(f)
        env = next((r for r in recs if r["kind"] == "env"), None)
        if env is None:
            L += [f"## `{f}`: no env record, skipped", ""]
            continue
        end = [r for r in recs if r["kind"] == "end"]
        L += [f"## `{f}` on {env['host']} ({env.get('cpu_model')}, {env.get('affinity')} CPUs; GPU {env.get('gpu')})",
              f"- script md5 {env['script_md5']}, ext md5 {env.get('ext_md5')}; end {end[0] if end else 'MISSING'}", ""]
        num = [r for r in recs if r["kind"] == "numeric"]
        if num:
            n = num[0]
            L += ["### (a) numeric: row-wise max relative difference", "",
                  "| quantity | logits (round 1) | values (round 2) |", "|---|---|---|"]
            for lab, ks, ko in (("torch CPU vs GPU fp32", "logits_cpu_vs_gpu_fp32", "values_cpu_vs_gpu_fp32"),
                                ("C++ vs GPU fp32", "logits_cpp_vs_gpu_fp32", "values_cpp_vs_gpu_fp32"),
                                ("C++ vs torch CPU", "logits_cpp_vs_torch_cpu", "values_cpp_vs_torch_cpu"),
                                ("GPU TF32 vs GPU fp32", "logits_gpu_tf32_vs_fp32", "values_gpu_tf32_vs_fp32")):
                L.append(f"| {lab} | {n[ks]['max_rel_rowwise']:.2e} | {n[ko]['max_rel_rowwise']:.2e} |")
            L += [f"| GPU noise floor (1e-7 perturbation / rerun) | {n['noise_floor_rel']['s']:.2e} | "
                  f"{n['noise_floor_rel']['o']:.2e} |",
                  f"| fp16 rounding of the reply | {n['fp16_reply_rel']['s']:.2e} | {n['fp16_reply_rel']['o']:.2e} |",
                  f"| fraction of fp16 replies differing, C++ vs GPU | "
                  f"{n['fp16_reply_diff_frac_cpp_vs_gpu']['s']:.2e} | {n['fp16_reply_diff_frac_cpp_vs_gpu']['o']:.2e} |",
                  ""]
        cpu = [r for r in recs if r["kind"] == "cpu"]
        if cpu:
            L += ["### (b) CPU time per layer (both rounds; median over all timed layer-steps)", "",
                  "| impl | B | threads | layer us [IQR] | round 1 us | round 2 us | step ms (32 layers) | GMAC/s | "
                  "steps / distinct | t_min reached | regions | footprint GB | pool touched GB / LLC MiB | "
                  "loadavg after | valid |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in sorted(cpu, key=lambda r: (bool(r.get("cache_control")), r["impl"], r["B"], r["threads"])):
                llc = r.get("llc_bytes")
                touched = r.get("pool_touched_bytes")
                why = []
                if r.get("cache_control"):
                    pass                                  # the cache-resident control: small on purpose
                else:
                    # ids are never reused within the timed loop and the LLC is flushed before it, so every timed
                    # row is a first access; these checks catch older records and shared regions
                    if not r.get("llc_flushed_before_timing"):
                        why.append("no LLC flush (old record)")
                    if not llc:
                        why.append("LLC size unknown")
                    if r.get("regions", 0) < 32:
                        why.append("regions < 32")
                    if touched is None:
                        why.append("old record (no pool size)")
                la = r.get("loadavg_after", r.get("loadavg"))
                imp = r["impl"] + (" (cache control)" if r.get("cache_control") else "")
                pool = "n/a" if touched is None else f"{touched / 1e9:.2f}"
                tt = r.get("timed_touched_bytes")
                timed = "" if tt is None else f" (timed {tt / 1e9:.2f})"
                llc_s = "n/a" if not llc else f"{llc / 2**20:.0f}"
                verdict = "INVALID: " + ", ".join(why) if why else ("control" if r.get("cache_control") else "ok")
                L.append(f"| {imp} | {r['B']} | {r['threads']} | {r['layer_us_med']:.1f} "
                         f"[{r['layer_us_q1']:.1f}, {r['layer_us_q3']:.1f}] | {r['round1_us_med']:.1f} | "
                         f"{r['round2_us_med']:.1f} | {r['step_ms_est']:.2f} | {r['gmac_per_s']:.1f} | "
                         f"{r['steps']} / {r.get('steps_distinct_ids', '?')} | {r.get('t_reached', '?')} | "
                         f"{r.get('regions', '?')} | {r['footprint_bytes'] / 1e9:.2f} | "
                         f"{pool}{timed} / {llc_s} | {la[0]:.1f} | {verdict} |")
            L.append("")
        rtt = [r for r in recs if r["kind"] == "rtt"]
        if rtt:
            L += ["### (c) GPU <-> CPU round trip (Python responder process; CUDA graph of 32 = one decode step)", "",
                  "| B | responder | threads | request / reply bytes | graph us per round trip [IQR] | "
                  "eager us (launch-bound) | timeouts | echo ok | PTX atoms (predicated) / bar.sync |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for r in rtt:
                p = r.get("ptx", {})
                echo = "-" if r["echo_ok"] is None else r["echo_ok"]
                L.append(f"| {r['B']} | {r['compute']} | {r.get('responder_threads', '-')} | "
                         f"{r['req_bytes']} / {r['resp_bytes']} | {r['graph_us_per_rt_med']:.1f} "
                         f"[{r['graph_us_per_rt_q1']:.1f}, {r['graph_us_per_rt_q3']:.1f}] | "
                         f"{r['eager_us_per_rt_med']:.1f} | {r['timeouts']} | {echo} | "
                         f"{p.get('atom')} ({p.get('atom_predicated')}) / {p.get('bar_sync')} |")
            L.append("")
        errs = [r for r in recs if r["kind"] == "error"]
        if errs:
            L += ["### errors", ""] + [f"- {e['mode']}: {(e.get('trace') or 'no trace').splitlines()[-1]}"
                                       for e in errs] + [""]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
