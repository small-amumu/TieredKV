# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""RFP: host-memory roofline and Pareto model for TieredKV's tiered KV (analysis only, no new runs).

Calibrated on measured data already in this repository:
  - PRF (algorithm-level profiling round): STAR RTX 4090, Llama-3.1-8B shape, 32 layers, attention
    kernels only, CUDA graph, h54 and h33 with residuals on host (zero-copy) or on GPU; reference rates.
  - FST / PFG / TRC / VFL / ACS / HBF outcomes for fetch timing, cache hit rates and quality.
Every assumption that is not a measurement is named in ASSUMPTIONS and printed.
Usage: python -I rfp_model.py <out_dir>
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------- model shape (Llama-3.1-8B)
LAYERS, KVH, QH, D = 32, 8, 32, 128
G = QH // KVH
PARAMS = 8.03e9
WEIGHT_BYTES_FP16 = 2 * PARAMS
M_KEY, K_VAL = 128, 64                       # deployed rows per (layer, KV head, step)
BASE_B_PER_TOKEN_HEAD = 70.0                 # PRF: requested bytes of the 2-bit base per token and KV head
BASE_B_FULL = 119.0                          # PRF: fuller estimate incl. intermediates

# ---------------------------------------------------------------------------- hardware (STAR RTX 4090)
HBM_SPEC = 1008e9
HBM_BASE_ACHIEVED = 553e9                    # PRF abasew at 8 x 32K
PCIE_DMA = 22.5e9                            # PRF pinned H2D
FP16_TENSOR = 165.2e12                       # RTX 4090 dense fp16 with fp32 accumulate (spec)
FP32_SIMT = 82.6e12
GPU_MEM = 24e9
HOST_DRAM_STAR = 251e9

ASSUMPTIONS = {
    "weight_read_efficiency": "full-model weight GEMV time = 16.06 GB / (0.85 x 1008 GB/s) = 18.7 ms; NOT measured on "
                              "this card (PRF timed attention kernels only); sensitivity 0.70-0.95 reported",
    "attention_fit": "T_attn(B, L) = a + b*B + c*B*L fitted to PRF's six h54-GPU points; 128K is an extrapolation",
    "row_cost": "host minus GPU time / rows per step (PRF, zero-copy path); log-linear in L fitted per codec; "
                "128K is an extrapolation",
    "overlap_bound": "ideal layer-ahead prefetch: step = max(GPU work, host fetch) with weights read once; an UPPER "
                     "bound (PFG measured a 60-row prefetch queue costing about what it saved on one GPU; micro-batch "
                     "splitting would re-read weights and is not modelled)",
    "cpu_correction": "near-data correction (lit D): CPU computes key rescores and value deltas next to the residuals; "
                      "31 M MAC per sequence-step (Llama-8B, M=128, K=64, G=4); CPU_GFLOPS effective and RTT are "
                      "ASSUMED (not measured): 2 round trips per layer x 32 layers x RTT on the critical path",
    "gpu_reserve": "1.5 GB of GPU memory reserved for activations / workspace",
    "host_reserve": "30 GB of host DRAM reserved for the OS and the engine",
}
WEIGHT_EFF = 0.85
CPU_GFLOPS = 640e9          # assumed: 32 of STAR's 128 cores x ~20 GFLOP/s effective fp32 incl. dequantization
RTT = 15e-6                 # assumed PCIe round trip per message (lit D: 10-20 us, to be measured)
MAC_PER_SEQ_STEP = LAYERS * KVH * (M_KEY * (D + G * D) + K_VAL * (D + G * D))


def weight_time(eff: float = WEIGHT_EFF, wbytes: float = WEIGHT_BYTES_FP16) -> float:
    return wbytes / (eff * HBM_SPEC)


# ---------------------------------------------------------------------------- codecs (host residual rows)
def row_bytes(bits: int, dim: int = D) -> int:
    return dim * bits // 8 + 4                # + 32-bit row metadata (min + step)


CODECS = {  # name: (key bits, value bits)
    "h88": (8, 8), "h64": (6, 4), "h54": (5, 4), "h43": (4, 3), "h33": (3, 3), "h32": (3, 2),
}


def host_bytes_per_token(codec: str) -> float:
    kb, vb = CODECS[codec]
    return LAYERS * KVH * (row_bytes(kb) + row_bytes(vb))


def host_bits_per_element(codec: str) -> float:
    kb, vb = CODECS[codec]
    return ((kb + 32 / D) + (vb + 32 / D)) / 2


def fetch_bytes_per_step(codec: str) -> float:
    kb, vb = CODECS[codec]
    return LAYERS * KVH * (M_KEY * row_bytes(kb) + K_VAL * row_bytes(vb))


ROWS_PER_SEQ_STEP = LAYERS * KVH * (M_KEY + K_VAL)
GPU_BYTES_PER_TOKEN = LAYERS * KVH * BASE_B_PER_TOKEN_HEAD

# ---------------------------------------------------------------------------- PRF calibration
PRF = [  # B, L, fp16, h54 host, h54 gpu, h33 host, h33 gpu  (ms per decode step, attention only)
    (1, 4096, 0.97, 2.90, 1.85, 2.79, 1.86),
    (1, 8192, 1.46, 3.49, 2.21, 3.37, 2.21),
    (1, 32768, 5.13, 5.39, 3.39, 5.31, 3.39),
    (8, 4096, 5.17, 13.26, 3.76, 12.05, 4.08),
    (8, 8192, 10.15, 15.75, 4.38, 14.83, 4.40),
    (8, 32768, float("nan"), 29.04, 13.30, 28.60, 13.20),
]


def fit_attention() -> np.ndarray:
    x = np.array([[1.0, b, b * l] for b, l, *_ in PRF])
    y = np.array([r[4] for r in PRF]) * 1e-3
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


ATT = fit_attention()


def t_attn(b: int, l: int) -> float:
    return float(ATT[0] + ATT[1] * b + ATT[2] * b * l)


def fit_row_cost(col_host: int, col_gpu: int) -> tuple[float, float, list]:
    pts = []
    for r in PRF:
        b, l = r[0], r[1]
        extra = (r[col_host] - r[col_gpu]) * 1e-3
        pts.append((b, l, extra / (b * ROWS_PER_SEQ_STEP)))
    x = np.array([[1.0, math.log2(l / 4096)] for _, l, _ in pts])
    y = np.array([p[2] for p in pts])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(coef[0]), float(coef[1]), pts


ROW54 = fit_row_cost(3, 4)
ROW33 = fit_row_cost(5, 6)


def t_row(codec: str, l: int) -> float:
    """Per-row zero-copy cost; interpolated linearly in row bytes between the measured h33 and h54 fits."""
    a54, s54, _ = ROW54
    a33, s33, _ = ROW33
    c54 = a54 + s54 * math.log2(l / 4096)
    c33 = a33 + s33 * math.log2(l / 4096)
    kb, vb = CODECS[codec]
    byt = (M_KEY * row_bytes(kb) + K_VAL * row_bytes(vb)) / (M_KEY + K_VAL)
    b54 = (M_KEY * 84 + K_VAL * 68) / 192
    b33 = (M_KEY * 52 + K_VAL * 52) / 192
    return c33 + (c54 - c33) * (byt - b33) / (b54 - b33)


@dataclass
class Option:
    name: str
    codec: str = "h54"
    rows_scale: float = 1.0          # fraction of rows that cross PCIe (GPU hot cache: 1 - hit)
    fixed_host_s: float = 0.0        # per-sequence per-step maintenance (cache lookup / copy-back)
    pipelined: bool = False
    kappa: float = 0.0
    cpu_correct: bool = False
    attn_overlap: bool = False       # fetch of one (sequence, head) overlaps base scoring of others; GEMMs not split
    host_frac: float = 1.0           # host memory multiplier (eviction / demotion / prefix sharing)
    weight_bytes: float = WEIGHT_BYTES_FP16


def step_time(o: Option, b: int, l: int) -> dict:
    tw = weight_time(wbytes=o.weight_bytes)
    ta = t_attn(b, l)
    if o.cpu_correct:
        th = 2 * LAYERS * RTT + b * 2 * MAC_PER_SEQ_STEP / CPU_GFLOPS
    else:
        th = b * (ROWS_PER_SEQ_STEP * o.rows_scale * t_row(o.codec, l) + o.fixed_host_s)
    gpu = tw + ta
    if o.attn_overlap:
        tot = tw + max(ta, th) + o.kappa * min(ta, th)
    elif o.pipelined:
        tot = max(gpu, th) + o.kappa * min(gpu, th)
    else:
        tot = gpu + th
    return {"weights": tw, "attn": ta, "host": th, "total": tot, "tok_s": b / tot}


def max_batch_gpu(l: int, weight_bytes: float = WEIGHT_BYTES_FP16, gpu_mem: float = GPU_MEM) -> int:
    free = gpu_mem - weight_bytes - 1.5e9
    return max(0, int(free // (GPU_BYTES_PER_TOKEN * l)))


def host_need(codec: str, l: int, frac: float = 1.0) -> float:
    return host_bytes_per_token(codec) * l * frac


# ---------------------------------------------------------------------------- measured quality (HBF, 15 models)
HBF = {  # (key bits, value bits): (ratio vs R at 4K, at 32K) ; R = K8V8 at 160 rows
    (8, 8): (1.000, 1.000), (6, 8): (1.020, 1.000), (7, 6): (1.003, 1.000), (6, 6): (1.018, 1.002),
    (7, 5): (1.023, 1.001), (6, 5): (1.024, 1.004), (5, 6): (1.073, 1.011), (5, 5): (1.090, 1.012),
    (7, 4): (1.062, 1.028), (6, 4): (1.084, 1.027), (5, 4): (1.122, 1.028), (8, 4): (1.075, 1.026),
    (4, 8): (1.232, 1.042),
}
NC0 = (57.9, 25.8)   # VFL: no host rows (2-bit base only), same panel family


def bits_elem(kb: int, vb: int) -> float:
    return ((kb + 0.25) + (vb + 0.25)) / 2


def pareto(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Lower-left frontier: no other point has <= x and <= y with one strict."""
    out = []
    for x, y, n in points:
        if not any((x2 <= x and y2 <= y) and (x2 < x or y2 < y) for x2, y2, _ in points):
            out.append((x, y, n))
    return sorted(out)


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"assumptions": ASSUMPTIONS}
    # calibration report
    res["attention_fit"] = {"a_s": ATT[0], "b_s_per_seq": ATT[1], "c_s_per_seq_token": ATT[2],
                            "residuals_ms": [round((t_attn(r[0], r[1]) - r[4] * 1e-3) * 1e3, 3) for r in PRF]}
    res["row_cost_fit"] = {nm: {"ns_at_4K": f[0] * 1e9, "ns_per_doubling": f[1] * 1e9,
                                "points_ns": [(b, l, round(t * 1e9, 1)) for b, l, t in f[2]]}
                           for nm, f in (("h54", ROW54), ("h33", ROW33))}
    # accounting
    res["accounting"] = {c: {"host_bytes_per_token": host_bytes_per_token(c),
                             "host_bits_per_element": host_bits_per_element(c),
                             "fetch_MiB_per_step": fetch_bytes_per_step(c) / 2 ** 20,
                             "host_over_gpu_per_token": host_bytes_per_token(c) / GPU_BYTES_PER_TOKEN}
                         for c in CODECS}
    res["accounting"]["gpu_bytes_per_token"] = GPU_BYTES_PER_TOKEN
    res["accounting"]["fp16_kv_bytes_per_token"] = LAYERS * KVH * D * 2 * 2
    # roofline points (operational intensity in FLOP per byte of the tier that carries them)
    roof = {
        "ceilings": {"fp16_tensor_flops": FP16_TENSOR, "fp32_simt_flops": FP32_SIMT, "hbm_spec_Bps": HBM_SPEC,
                     "hbm_base_achieved_Bps": HBM_BASE_ACHIEVED, "pcie_dma_Bps": PCIE_DMA,
                     "zero_copy_row_Bps_4K": 84 / (ROW54[0]),
                     "zero_copy_row_Bps_32K": 84 / (ROW54[0] + 3 * ROW54[1])},
        "points": {
            "weights GEMV/GEMM, B=1": (1.0, "hbm"), "weights, B=8": (8.0, "hbm"), "weights, B=32": (32.0, "hbm"),
            "base attention (2-bit), any B": (2 * 2 * D * G / BASE_B_PER_TOKEN_HEAD, "hbm"),
            "fp16 attention, any B": (2 * 2 * D * G / (2 * 2 * D), "hbm"),
            "key-row rescore (h54)": ((2 * D * G + 3 * D) / 84, "host"),
            "value-row correction (h54)": ((2 * D * G + 3 * D) / 68, "host"),
        },
    }
    res["roofline"] = roof
    # step-time grid
    opts = [
        Option("deployed: serial zero-copy row gather (calibrated on PRF)"),
        Option("host codec h33 (fewer bytes, same rows)", codec="h33"),
        Option("GPU hot-row cache, hit 0.85 (optimistic; TRC live cache: no gain at 32K)", rows_scale=0.15,
               fixed_host_s=LAYERS * 9.5e-6),
        Option("ideal layer-ahead prefetch: max(GPU, host) (upper bound)", pipelined=True, kappa=0.0),
        Option("attention-phase overlap across heads/sequences, kappa 0.5 (model)", attn_overlap=True, kappa=0.5),
        Option("near-data CPU correction (assumed 640 GFLOP/s, 15 us RTT)", cpu_correct=True),
        Option("INT4 weights + deployed host path", weight_bytes=0.5 * PARAMS * 1.07),
        Option("INT4 weights + near-data CPU correction", cpu_correct=True, weight_bytes=0.5 * PARAMS * 1.07),
    ]
    grid = {}
    for o in opts:
        for l in (4096, 32768, 131072):
            for b in (1, 2, 4, 8, 16, 32):
                bmax = max_batch_gpu(l, o.weight_bytes)
                st = step_time(o, b, l)
                grid[f"{o.name}|{l}|{b}"] = {**{k: round(v * 1e3, 3) if k != "tok_s" else round(v, 1)
                                                for k, v in st.items()},
                                             "fits_gpu": b <= bmax, "max_batch_gpu": bmax}
    res["step_grid"] = grid
    res["weight_eff_sensitivity_ms"] = {str(e): round(weight_time(e) * 1e3, 2) for e in (0.70, 0.85, 0.95)}
    # crossover batch where host fetch equals weight time (serial, deployed)
    res["crossover_batch"] = {str(l): weight_time() / (ROWS_PER_SEQ_STEP * t_row("h54", l)) for l in
                              (4096, 32768, 131072)}
    # capacity: when does host DRAM bind?
    cap = {}
    for gpus, gmem, dram, wb, tag in ((2, 24e9, HOST_DRAM_STAR, WEIGHT_BYTES_FP16, "STAR 2x4090, fp16 weights"),
                                      (2, 24e9, HOST_DRAM_STAR, 0.5 * PARAMS * 1.07, "STAR 2x4090, INT4 weights"),
                                      (8, 80e9, 1.0e12, WEIGHT_BYTES_FP16, "8x80GB node, 1 TB DRAM"),
                                      (8, 80e9, 1.0e12, 0.5 * PARAMS * 1.07, "8x80GB node INT4, 1 TB DRAM")):
        for l in (131072, 1048576):
            nseq = gpus * max(0, int((gmem - wb - 1.5e9) // (GPU_BYTES_PER_TOKEN * l)))
            row = {"seqs_gpu_bound": nseq}
            for c in ("h54", "h33", "h32"):
                need = nseq * host_need(c, l)
                row[f"host_GB_{c}"] = round(need / 1e9, 1)
                row[f"host_binds_{c}"] = bool(need > dram - 30e9)
            cap[f"{tag}|{l}"] = row
    res["capacity"] = cap
    # host-memory policies (bytes per token over 32 layers x 8 KV heads); quality status per policy
    per_lh = {"h54": 152, "h43": 120, "h33": 104, "h32": 88, "key-only h5": 84}
    pol = {
        "h54 (deployed)": (per_lh["h54"], "measured (HBF: 1.12x R at 4K, 1.03x at 32K)"),
        "h43 uniform": (per_lh["h43"], "pending (OPX)"),
        "h33 uniform": (per_lh["h33"], "pending (OPX); two-model 32K data only"),
        "h32 uniform": (per_lh["h32"], "pending (OPX)"),
        "key-only h5 (drop value residuals)": (per_lh["key-only h5"], "pending (NAB2 R3 K-only arm)"),
        "demote 90% cold to key-only, 10% h54": (0.1 * 152 + 0.9 * 84, "unmeasured (needs selection traces)"),
        "demote 90% cold to h32, 10% h54": (0.1 * 152 + 0.9 * 88, "unmeasured"),
        "evict 50% cold residuals (base only)": (0.5 * 152, "unmeasured; with NO host rows at all KL is 26-58x R"),
        "h54 + 50%-shared prefix over 16 requests": (152 * (0.5 + 0.5 / 16), "exact if residuals are bit-identical"),
    }
    res["host_policies"] = {k: {"bytes_per_token": v[0] * LAYERS * KVH, "GB_per_128K_seq": v[0] * LAYERS * KVH * 131072 / 1e9,
                                "host_over_gpu": v[0] * LAYERS * KVH / GPU_BYTES_PER_TOKEN, "quality": v[1]}
                            for k, v in pol.items()}
    # measured Pareto (host bits per element vs KL ratio), 4K and 32K
    par = {}
    for i, tag in enumerate(("4K", "32K")):
        pts = [(bits_elem(kb, vb), v[i], f"K{kb}V{vb}") for (kb, vb), v in HBF.items()]
        pts.append((0.0, NC0[i], "no host rows"))
        par[tag] = {"points": pts, "frontier": pareto(pts)}
    res["pareto_measured"] = par
    (out / "rfp_results.json").write_text(json.dumps(res, indent=1, default=float))
    make_figures(out, res, opts)
    print(json.dumps({"attention_fit": res["attention_fit"], "row_cost_fit": res["row_cost_fit"],
                      "crossover_batch": res["crossover_batch"]}, indent=1, default=float))
    return 0


def make_figures(out: Path, res: dict, opts: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

    # 1. hierarchical roofline
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.logspace(-1, 3, 200)
    c = res["roofline"]["ceilings"]
    for bw, lab, ls in ((c["hbm_spec_Bps"], "HBM spec 1008 GB/s", "-"),
                        (c["hbm_base_achieved_Bps"], "HBM achieved by base scan 553 GB/s", "--"),
                        (c["pcie_dma_Bps"], "PCIe pinned DMA 22.5 GB/s", "-"),
                        (c["zero_copy_row_Bps_4K"], "zero-copy 84 B rows, 4K (%.1f GB/s)" % (c["zero_copy_row_Bps_4K"] / 1e9), ":"),
                        (c["zero_copy_row_Bps_32K"], "zero-copy 84 B rows, 32K (%.1f GB/s)" % (c["zero_copy_row_Bps_32K"] / 1e9), ":")):
        ax.loglog(x, np.minimum(c["fp16_tensor_flops"], bw * x), ls, lw=1.2, label=lab)
    ax.axhline(c["fp32_simt_flops"], color="grey", lw=0.6, ls="-.")
    offsets = {"weights, B=32": (-60, 6), "base attention (2-bit), any B": (4, -10),
               "key-row rescore (h54)": (-75, 6), "value-row correction (h54)": (4, -10)}
    for name, (oi, tier) in res["roofline"]["points"].items():
        bw = c["hbm_spec_Bps"] if tier == "hbm" else c["zero_copy_row_Bps_4K"]
        y = min(c["fp16_tensor_flops"], bw * oi)
        ax.plot(oi, y, "o" if tier == "hbm" else "s", ms=5, color="C3" if tier == "host" else "C0")
        ax.annotate(name, (oi, y), fontsize=6.5, xytext=offsets.get(name, (3, 3)), textcoords="offset points")
    ax.set_xlabel("operational intensity (FLOP per byte of the tier that carries the data)")
    ax.set_ylabel("attainable FLOP/s")
    ax.set_title("RTX 4090 (STAR): HBM vs host-row tiers, Llama-3.1-8B decode")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(True, which="both", lw=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_roofline.png", dpi=200)
    plt.close(fig)

    # 2. throughput vs batch (time roofline)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=False)
    for ax, l in zip(axs, (4096, 32768, 131072)):
        for o in opts:
            bs = [b for b in (1, 2, 4, 8, 16, 32)]
            ys = [step_time(o, b, l)["tok_s"] for b in bs]
            ok = [b <= max_batch_gpu(l, o.weight_bytes) for b in bs]
            ax.plot(bs, ys, "-", lw=1, label=o.name)
            ax.plot([b for b, k in zip(bs, ok) if not k], [y for y, k in zip(ys, ok) if not k], "x", color="k", ms=4)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{l // 1024}K context (x: not in 24 GB)")
        ax.set_xlabel("batch")
        ax.grid(True, lw=0.3)
    axs[0].set_ylabel("decode tokens/s per GPU (model)")
    axs[0].legend(fontsize=5.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig2_throughput_vs_batch.png", dpi=200)
    plt.close(fig)

    # 3. measured Pareto: host bits vs KL ratio
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, tag in zip(axs, ("4K", "32K")):
        pts = res["pareto_measured"][tag]["points"]
        fr = res["pareto_measured"][tag]["frontier"]
        for xx, yy, n in pts:
            if n == "no host rows":
                continue
            ax.plot(xx, yy, "o", color="C0", ms=4)
            ax.annotate(n, (xx, yy), fontsize=6.5, xytext=(2, 2), textcoords="offset points")
        frx = [p[0] for p in fr if p[2] != "no host rows"]
        fry = [p[1] for p in fr if p[2] != "no host rows"]
        ax.plot(frx, fry, "-", color="C3", lw=1, label="measured frontier (HBF, 15 models, median)")
        ax.axvspan(2.0, 4.75, color="orange", alpha=0.12, label="h32-h43 band: OPX pending; ladders/eviction: unmeasured")
        ax.set_xlabel("host bits per element (K and V averaged)")
        ax.set_ylabel("decode KL / KL(K8V8 reference R)")
        ax.set_title(f"{tag}: no host rows = {NC0[0 if tag == '4K' else 1]} x R (off scale)")
        ax.grid(True, lw=0.3)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig3_pareto_hostbits_quality.png", dpi=200)
    plt.close(fig)

    # 4. host-memory policies: bytes per token, coloured by quality status
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    items = sorted(res["host_policies"].items(), key=lambda kv: kv[1]["bytes_per_token"])
    cols = {"measured": "C2", "pending": "C1", "unmeasured": "C3", "exact": "C0"}
    for i, (k, v) in enumerate(items):
        q = v["quality"].split(" ")[0].rstrip(";")
        ax.barh(i, v["bytes_per_token"] / 1024, color=cols.get(q, "grey"))
        ax.text(v["bytes_per_token"] / 1024 + 0.3, i, f"{v['GB_per_128K_seq']:.2f} GB/128K, {v['quality']}",
                va="center", fontsize=6)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([k for k, _ in items], fontsize=7)
    ax.set_xlabel("host KiB per token (Llama-3.1-8B; GPU base = 17.5 KiB)")
    ax.set_xlim(0, 75)
    ax.set_title("Host-memory policies: green measured / orange pending / red unmeasured / blue exact", fontsize=8)
    ax.grid(True, axis="x", lw=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig4_host_policies.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
