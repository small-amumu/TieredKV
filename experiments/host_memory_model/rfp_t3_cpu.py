# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Re-evaluate RFP's near-data CPU correction option with the CPU times measured in T3 (lab05).

rfp_model.py (unchanged, md5 checked below) assumes CPU_GFLOPS = 640e9 and RTT = 15 us for this
option. T3 measured the per-layer CPU time of both correction rounds (all 8 KV heads, B sequences,
rows mostly from DRAM, footprint of a 32K context) for B = 1, 8, 32 and 1-64 threads. This script
feeds those times into the model's own step_time() by setting, per B, the effective CPU rate that
reproduces the measured per-step CPU time, so the composition (weights + attention + host,
serial) is exactly the model's. RTT is not measured yet (T3 GPU part pending) and is swept.
The same 32K-footprint CPU time is used in the 4K columns (no 4K measurement exists).
usage: python -I -X utf8 rfp_t3_cpu.py <t3b_cpu.jsonl> <out.md>
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
EXPECT_MODEL_MD5 = "bee587d87cccff6036c65a2cf13c2722"
EXPECT_T3_MD5 = "08d4d0c6beb691cf50c122eff819e557"     # rp_cpu.py
EXPECT_EXT_MD5 = "ecb4aeea3d8428582402b42aa373d1ee"    # rp_cpu_ext.cpp (the timed kernels)
T3_L = 32768                     # context length whose footprint the T3 cpu records emulate
LENGTHS = (4096, 32768)
BATCHES = (1, 8, 32)
THREADS = (32, 64)               # 32 = the model's CPU budget; 64 = fastest at B >= 8 (uses SMT)
RTTS = (5e-6, 15e-6, 30e-6)
# RFP report, section 2 table (tok/s per GPU): the wrapper must reproduce these first
REPORT = {("deployed", 4096, 8): 259, ("deployed", 4096, 32): 517, ("deployed", 32768, 8): 167,
          ("deployed", 32768, 32): 247, ("cpu", 4096, 8): 335, ("cpu", 4096, 32): 1034,
          ("cpu", 32768, 8): 237, ("cpu", 32768, 32): 454}


def load_model() -> ModuleType:
    path = HERE / "rfp_model.py"
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != EXPECT_MODEL_MD5:
        raise SystemExit(f"refusing: rfp_model.py md5 {md5} is not the reviewed {EXPECT_MODEL_MD5}")
    spec = importlib.util.spec_from_file_location("rfp_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rfp_model"] = mod          # dataclass resolution needs the module registered
    spec.loader.exec_module(mod)
    return mod


def load_t3(path: Path) -> tuple[dict, dict, dict]:
    """Returns env record, main cpp records keyed (B, threads), cache-control cpp records."""
    recs = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path.name} line {i}: {e}") from e
    env = next((r for r in recs if r.get("kind") == "env"), None)
    end = next((r for r in recs if r.get("kind") == "end"), None)
    if env is None:
        raise SystemExit("refusing: T3 file has no env record")
    if end is None or not end.get("complete") or end.get("errors"):
        raise SystemExit(f"refusing: T3 file has no clean end record ({end})")
    if env.get("script_md5") != EXPECT_T3_MD5 or env.get("ext_md5") != EXPECT_EXT_MD5:
        raise SystemExit(f"refusing: T3 run md5 {env.get('script_md5')} / {env.get('ext_md5')} "
                         "is not the reviewed run")
    cpu, ctl = {}, {}
    for r in recs:
        if r.get("kind") != "cpu" or r.get("impl") != "cpp":
            continue
        target = ctl if r.get("cache_control") else cpu
        if not r.get("cache_control") and r.get("L") != T3_L:
            raise SystemExit(f"T3 record at L={r.get('L')}, expected {T3_L}")
        key = (r["B"], r["threads"])
        if key in target:
            raise SystemExit(f"duplicate T3 record {key}")
        target[key] = r
    return env, cpu, ctl


def tok_s(m: ModuleType, opt: Any, b: int, l: int, layer_s: float | None,
          rtt: float) -> tuple[float, float, bool]:
    """Model tok/s at (b, l); layer_s None = the model's own CPU assumption.
    Returns (tok/s, host ms per step, fits in GPU memory)."""
    save = (m.CPU_GFLOPS, m.RTT)
    try:
        m.RTT = rtt
        if layer_s is not None:
            m.CPU_GFLOPS = b * 2 * m.MAC_PER_SEQ_STEP / (m.LAYERS * layer_s)
        st = m.step_time(opt, b, l)
    finally:
        m.CPU_GFLOPS, m.RTT = save
    return st["tok_s"], st["host"] * 1e3, b <= m.max_batch_gpu(l, opt.weight_bytes)


def cell(v: tuple[float, float, bool]) -> str:
    return f"{v[0]:.0f}{'' if v[2] else ' x'}"


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: rfp_t3_cpu.py <t3b_cpu.jsonl> <out.md>")
    t3_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not t3_path.is_file():
        raise SystemExit(f"no such file: {t3_path}")
    m = load_model()
    try:
        env, cpu, ctl = load_t3(t3_path)
        for k in ("host", "cpu_model", "cpu_count", "script_md5", "ext_md5"):
            env[k]
        for r in list(cpu.values()) + list(ctl.values()):
            for k in ("layer_us_med", "layer_us_q1", "layer_us_q3", "gmac_per_s", "steps",
                      "layers_timed", "L"):
                r[k]
    except (UnicodeDecodeError, KeyError, TypeError, AttributeError) as e:
        raise SystemExit(f"malformed T3 file {t3_path.name}: {type(e).__name__} {e}") from e
    missing = [(b, t) for b in BATCHES for t in THREADS if (b, t) not in cpu]
    if missing:
        raise SystemExit(f"missing T3 points {missing}")
    dep = m.Option("deployed")
    cpu_o = m.Option("cpu", cpu_correct=True)
    int4 = 0.5 * m.PARAMS * 1.07
    cpu_i4 = m.Option("cpu int4", cpu_correct=True, weight_bytes=int4)
    dep_i4 = m.Option("deployed int4", weight_bytes=int4)

    # 1. reproduce the report's numbers with the model's own assumptions
    bad = []
    for (kind, l, b), want in REPORT.items():
        got = tok_s(m, dep if kind == "deployed" else cpu_o, b, l, None, m.RTT)[0]
        if abs(got - want) > 1.0:
            bad.append((kind, l, b, round(got, 1), want))
    if bad:
        raise SystemExit(f"wrapper does not reproduce the RFP report: {bad}")

    def layer_s(b: int, t: int | None) -> float | None:
        return None if t is None else cpu[(b, t)]["layer_us_med"] * 1e-6

    ctl_note = "the file has no cache-resident control"
    for (b, t), r in ctl.items():
        if (b, t) in cpu:
            c, d = r["layer_us_med"], cpu[(b, t)]["layer_us_med"]
            ctl_note = (f"the one cache-resident control (B{b}, {t} threads, L={r.get('L')}, "
                        f"{r.get('regions')} region) takes {c:.1f} vs {d:.1f} us per layer "
                        f"({1 - c / d:.0%} less), but it also differs in L and footprint")
    # residual pool per sequence (host bytes per token x tokens), to compare with the LLC
    pool = {(l, b): b * m.host_bytes_per_token("h54") * l for l in LENGTHS for b in BATCHES}
    llc = env.get("llc_bytes") or 0
    rtt_ratio = {b: m.LAYERS * cpu[(b, 32)]["layer_us_med"] * 1e-6 / (2 * m.LAYERS * 15e-6)
                 for b in BATCHES}
    floor_hit = sorted(k for k, r in cpu.items() if r.get("t_reached"))
    shown_hit = [k for k in floor_hit if k[0] in BATCHES and k[1] in THREADS]
    one = cpu.get((1, 1))
    cols = [(l, b) for l in LENGTHS for b in BATCHES]
    head = "| variant | " + " | ".join(f"{l // 1024}K B{b}" for l, b in cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [
        "# RFP near-data CPU correction with T3 measured CPU times (lab05)",
        "",
        f"- T3 input: `{t3_path.name}`, host {env['host']} ({env['cpu_model']}, "
        f"{env['cpu_count']} logical CPUs, {env.get('numa_nodes')} NUMA node), "
        f"rp_cpu.py {env['script_md5'][:8]}, rp_cpu_ext.cpp {env['ext_md5'][:8]}.",
        "- CPU time per layer = median over timed (layer, step) of round 1 + round 2, all 8 KV "
        "heads and B sequences, rows mostly from DRAM (pool >> LLC, ids never reused). It is the "
        "time of this simple C++/OpenMP implementation (compiler auto-vectorized, no "
        "hand-written SIMD decode), not a hardware limit"
        + (f": one thread does {one['gmac_per_s']:.1f} GMAC/s ({2 * one['gmac_per_s']:.1f} "
           "GFLOP/s-eq) at B1." if one else "."),
        f"- The records emulate a {T3_L // 1024}K context with row ids never reused (worst case "
        "for the cache). The same CPU time is used in the 4K columns. h54 residual pool vs the "
        f"{llc / 2 ** 20:.0f} MiB LLC: "
        + ", ".join(f"{l // 1024}K B{b} {pool[(l, b)] / 2 ** 20:,.0f} MiB" for l, b in
                    sorted(pool) if l == 4096)
        + ". At 4K B8 / B32 rows still come from DRAM; at 4K B1 the pool can sit in the LLC, and "
        "in real decoding many selected rows repeat between steps (T1 trajectories: about 43 of "
        "the 128 first-stage rows change per step), so the real time can be lower. As a "
        f"magnitude only: {ctl_note}.",
        "- lab05 was not exclusive (load average 5-11 during the run, from other jobs) and is a "
        "different CPU from STAR, the platform the model is calibrated on (STAR: 2 x Xeon Platinum "
        "8336C, 64 cores / 128 threads, 2 NUMA nodes). STAR's own measurement is pending.",
        f"- Model: rfp_model.py {EXPECT_MODEL_MD5[:8]} unchanged; MAC per sequence-step "
        f"{m.MAC_PER_SEQ_STEP / 1e6:.2f} M. The step is composed serially, weights + attention + "
        "host, for both the deployed and the CPU paths, as in the model; overlap of the CPU work "
        "with GPU work and the CUDA-graph cost of a per-layer CPU hand-off are not modelled. The "
        "wrapper reproduces the report's eight numbers (deployed and assumed-CPU rows) within 1 "
        "tok/s before any measured value is used.",
        "- RTT is an assumption (T3 GPU part not run yet): 2 round trips per layer x 32 layers. "
        "At 15 us the 64 round trips take 0.96 ms per step; the 32-thread CPU time is "
        + ", ".join(f"{rtt_ratio[b]:.0f}x that at B{b}" for b in BATCHES)
        + ". The RTT effect below is small for that reason and grows if the CPU part gets faster.",
        "- The model's 640 GFLOP/s is 32 of STAR's 128 logical CPUs (64 physical cores) at about "
        "20 GFLOP/s each. 32 threads here = 32 of lab05's 48 physical cores; 64 threads use SMT "
        "and exceed that budget.",
        "- `x` = the batch does not fit in 24 GB at that length (model's capacity rule); the "
        "number is shown as in the RFP report.",
        "",
        "## Measured CPU time per decode step (32 layers)",
        "",
        f"- GFLOP/s-eq = 2 x MAC per second, the model's convention (decode counted as D = "
        f"{m.D} MAC per row).",
        "- Each cell: ms per step = 32 x median layer time [32 x Q1, 32 x Q3 of the per-layer "
        "times]; GFLOP/s-eq; steps timed "
        "and layer samples (steps x 32). Timing stops at the 2 s floor or when the pool of "
        "never-reused row ids runs out, whichever comes first. The floor was reached at (B, "
        f"threads) = {floor_hit}; cells below that reached it: {shown_hit or 'none'} (the others "
        "stopped at the id cap).",
        "",
        "| threads | " + " | ".join(f"B{b}" for b in BATCHES) + " |",
        "|---|" + "---|" * len(BATCHES),
    ]
    for t in THREADS:
        row = []
        for b in BATCHES:
            r = cpu[(b, t)]
            ls = r["layer_us_med"] * 1e-6
            gf = b * 2 * m.MAC_PER_SEQ_STEP / (m.LAYERS * ls) / 1e9
            q1, q3 = (m.LAYERS * r[k] * 1e-3 for k in ("layer_us_q1", "layer_us_q3"))
            row.append(f"{m.LAYERS * ls * 1e3:.2f} [{q1:.2f}, {q3:.2f}]; {gf:.0f}; "
                       f"{r['steps']} steps, {r['layers_timed']} samples")
        lines.append(f"| {t} | " + " | ".join(row) + " |")
    assumed = [f"{b * 2 * m.MAC_PER_SEQ_STEP / m.CPU_GFLOPS * 1e3:.2f} "
               f"({m.CPU_GFLOPS / 1e9:.0f})" for b in BATCHES]
    lines += ["| model assumption | " + " | ".join(assumed) + " |", "",
              "## Throughput, tok/s per GPU (RTT assumed 15 us unless named)", "", head, sep]

    def row(name: str, opt: Any, t: int | None, rtt: float) -> str:
        vals = [cell(tok_s(m, opt, b, l, layer_s(b, t), rtt)) for l, b in cols]
        return f"| {name} | " + " | ".join(vals) + " |"

    lines.append(row("deployed: serial zero-copy row gather", dep, None, 15e-6))
    lines.append(row("CPU correction, model assumption (640 GFLOP/s)", cpu_o, None, 15e-6))
    for t in THREADS:
        for rtt in RTTS:
            lines.append(row(f"CPU correction, T3 32K-footprint CPU time {t} threads, RTT "
                             f"{rtt * 1e6:.0f} us", cpu_o, t, rtt))
    lines.append(row("INT4 weights + deployed host path", dep_i4, None, 15e-6))
    lines.append(row("INT4 weights + CPU correction, model assumption", cpu_i4, None, 15e-6))
    for t in THREADS:
        lines.append(row(f"INT4 weights + CPU correction, T3 32K-footprint CPU time {t} "
                         "threads", cpu_i4, t, 15e-6))

    lines += ["", "## Host time per step on the critical path (ms), RTT assumed 15 us", "", head,
              sep]

    def hrow(name: str, opt: Any, t: int | None) -> str:
        vals = [f"{tok_s(m, opt, b, l, layer_s(b, t), 15e-6)[1]:.2f}" for l, b in cols]
        return f"| {name} | " + " | ".join(vals) + " |"

    lines.append(hrow("deployed: serial zero-copy row gather", dep, None))
    lines.append(hrow("CPU correction, model assumption", cpu_o, None))
    for t in THREADS:
        lines.append(hrow(f"CPU correction, T3 32K-footprint CPU time {t} threads", cpu_o, t))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
