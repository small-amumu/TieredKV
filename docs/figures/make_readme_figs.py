# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Render the README result figures from readme_figs_data.json.

Usage: python -I docs/figures/make_readme_figs.py
Reads only readme_figs_data.json (next to this file) and writes fig_*.png next to it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA: dict[str, Any] = json.loads((HERE / "readme_figs_data.json").read_text(encoding="utf8"))

GRAY, ORANGE, BLUE, GREEN = "#7f7f7f", "#E69F00", "#0072B2", "#009E73"
SKY, VERM, PURPLE = "#56B4E9", "#D55E00", "#CC79A7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "hatch.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "savefig.dpi": 200,
    "figure.dpi": 200,
})


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(HERE / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", name)


def _plain_log(ax: plt.Axes, axis: str = "y", fmt: str = "{x:g}") -> None:
    """Use plain decimal tick labels on a log axis."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(x=v)))
    a.set_minor_formatter(NullFormatter())


def _bar(ax: plt.Axes, x: Any, h: Any, color: str, hatch: str, width: float, **kw: Any) -> Any:
    return ax.bar(x, h, width, color=color, edgecolor="black", linewidth=0.9, hatch=hatch, **kw)


# --------------------------------------------------------------------------------------
def fig_fidelity() -> None:
    models = DATA["fig1"]["models"]
    keys = ["qwen3_0p6b", "qwen3_1p7b", "qwen3_4b"]
    labels = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]
    arms = [("kivi2", "KIVI-2", GRAY, "//"), ("h2o", "H2O", ORANGE, "xx"),
            ("tiered", "TieredKV", GREEN, "\\\\")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for ax, field, title in [(axes[0], "decode_kl_median", "Decode span"),
                             (axes[1], "answer_kl_median", "Answer span")]:
        w = 0.26
        for j, (a, name, col, hat) in enumerate(arms):
            vals = [models[k][a][field] for k in keys]
            _bar(ax, np.arange(3) + (j - 1) * w, vals, col, hat, w)
        ax.set_yscale("log")
        _plain_log(ax)
        ax.set_xticks(range(3), labels)
        ax.set_title(title)
        ax.set_ylabel("KL to 16-bit (median)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(5e-4, 0.4)
    axes[1].set_ylim(5e-5, 0.3)
    handles = [Patch(facecolor=c, edgecolor="black", hatch=h, label=n) for _, n, c, h in arms]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, "fig_fidelity_kl_bars.png")


# --------------------------------------------------------------------------------------
def fig_ladder() -> None:
    models = DATA["fig2"]["models"]
    order = sorted(models, key=lambda m: models[m]["kl_full_ladder"] / models[m]["kl_specache_style"])
    series = [("kl_residual_rows", "Residual rows", ORANGE, "o"),
              ("kl_two_stage", "+ two-stage fetch", SKY, "s"),
              ("kl_full_ladder", "+ logit correction", PURPLE, "^"),
              ("kl_h54", "TieredKV (h54, about half the bytes)", GREEN, "D")]
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    x = np.arange(len(order))
    ax.axhline(1.0, color=GRAY, lw=2, ls="--", zorder=1)
    for key, name, col, mk in series:
        y = [models[m][key] / models[m]["kl_specache_style"] for m in order]
        big = key == "kl_h54"
        ax.plot(x, y, mk, color=col, ms=6.5 if big else 5.5, mec="black", mew=0.7,
                label=name, zorder=4 if big else 3)
    ax.set_yscale("log")
    ax.set_ylim(0.06, 1.6)
    ax.set_yticks([0.1, 0.2, 0.5, 1.0])
    _plain_log(ax)
    ax.set_xticks(x, order, rotation=40, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("KL ratio to SpeCache-style")
    ax.grid(axis="x", visible=False)
    ax.text(len(order) - 0.45, 1.07, "SpeCache-style (exact FP16 rows) = 1", ha="right", va="bottom",
            color=GRAY, fontsize=8, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, columnspacing=1.5)
    fig.tight_layout()
    _save(fig, "fig_ladder_models.png")


# --------------------------------------------------------------------------------------
def fig_same_gpu() -> None:
    arms = DATA["fig3"]["arms"]
    groups = [("TieredKV", GREEN, "\\\\", [("h54", "h54")]),
              ("RetroInfer", BLUE, "xx", [("ret009", "0.009"), ("ret018", "0.018"), ("ret036", "0.036")]),
              ("ShadowKV", ORANGE, "//", [("skv1024", "1024"), ("skv2048", "2048"), ("skv4096", "4096")])]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw={"width_ratios": [1, 1.2]})
    for pi, ax in enumerate(axes):
        pos, ticks, tlabels, spans = 0.0, [], [], []
        glist = ([("Full\nattention", GRAY, "", [("fp", "")])] if pi == 1 else []) + groups
        for name, col, hat, members in glist:
            start = pos
            for key, lab in members:
                a = arms[key]
                if pi == 0:
                    v = a["pcie_mb_per_step"]
                    ax.bar(pos, v["median"], 0.8, color=col, edgecolor="black", linewidth=0.9,
                           hatch=hat, yerr=[[v["median"] - v["q1"]], [v["q3"] - v["median"]]],
                           error_kw={"elinewidth": 1.0, "capsize": 2.5})
                    ax.text(pos, v["q3"] * 1.12, f"{v['median']:.1f}", ha="center", va="bottom",
                            fontsize=7)
                else:
                    s = a["score"]
                    ax.bar(pos, s["mean"], 0.8, color=col, edgecolor="black", linewidth=0.9,
                           hatch=hat, yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],
                           error_kw={"elinewidth": 1.0, "capsize": 2.5})
                    ax.text(pos, s["hi"] + 0.8, f"{s['mean']:.1f}", ha="center", va="bottom",
                            fontsize=7)
                ticks.append(pos)
                tlabels.append(lab)
                pos += 1.0
            spans.append((name, (start + pos - 1) / 2))
            pos += 0.7
        ax.set_xticks(ticks, tlabels)
        ax.tick_params(axis="x", labelsize=7)
        for name, xc in spans:
            ax.text(xc, -0.12, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, fontweight="bold")
        ax.grid(axis="x", visible=False)
    axes[0].set_yscale("log")
    axes[0].set_ylim(3, 400)
    axes[0].set_yticks([5, 10, 20, 50, 100, 200])
    _plain_log(axes[0])
    axes[0].set_ylabel("PCIe traffic (MB per step)")
    axes[0].set_title("Traffic, median and IQR")
    axes[1].set_ylim(0, 68)
    axes[1].set_ylabel("RULER score")
    axes[1].set_title("Accuracy, mean and 95% CI")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    _save(fig, "fig_same_gpu_offload.png")


# --------------------------------------------------------------------------------------
def fig_frontier() -> None:
    models = DATA["fig4"]["models"]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), sharey=True)
    uni = ["h32", "h33", "h43", "h44", "h54"]
    for ax, (mname, d) in zip(axes, models.items()):
        xs = [d[a]["bytes_per_record"] for a in uni]
        ys = [d[a]["ratio_vs_h54"] for a in uni]
        ax.errorbar(xs, ys, yerr=[[d[a]["ratio_vs_h54"] - d[a]["lo"] for a in uni],
                                  [d[a]["hi"] - d[a]["ratio_vs_h54"] for a in uni]],
                    color=GRAY, marker="o", lw=2, ms=5, capsize=2, zorder=2)
        for a in uni:
            ax.annotate(a, (d[a]["bytes_per_record"], d[a]["ratio_vs_h54"]), textcoords="offset points",
                        xytext=(11, 4), ha="center", fontsize=7, color="#444444")
        for a, col, mk in [("rnd_G1024_h32_75", ORANGE, "s"), ("sc_G1024_h32_75", GREEN, "o")]:
            e = d[a]
            ax.errorbar([e["bytes_per_record"]], [e["ratio_vs_h54"]],
                        yerr=[[e["ratio_vs_h54"] - e["lo"]], [e["hi"] - e["ratio_vs_h54"]]],
                        color=col, marker=mk, ms=8, mec="black", mew=0.8, capsize=2, ls="none", zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(0.9, 4.6)
        ax.set_yticks([1, 2, 3, 4])
        _plain_log(ax)
        ax.set_xlim(80, 158)
        ax.set_xticks([88, 104, 120, 136, 152])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(mname)
        ax.set_xlabel("Host bytes per record")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("KL ratio to h54")
    handles = [Line2D([], [], color=GRAY, marker="o", lw=2, ms=5, label="Uniform bit width"),
               Line2D([], [], color=ORANGE, marker="s", ms=7, mec="black", ls="none",
                      label="Random demotion"),
               Line2D([], [], color=GREEN, marker="o", ms=7, mec="black", ls="none",
                      label="Score-based demotion")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, "fig_host_memory_frontier.png")


# --------------------------------------------------------------------------------------
def fig_residual() -> None:
    models = DATA["fig5"]["models"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    split = {"n54": "5+4", "n44": "4+4", "n43": "4+3", "n33": "3+3"}
    ref_bytes, ref_kl = 152.0, 1.0  # h54 by definition
    for ax, tag, (mname, d) in zip(axes, "ab", models.items()):
        ks = ["n33", "n43", "n44", "n54"]
        xs = np.array([d[k]["bytes_per_record"] for k in ks])
        ys = np.array([d[k]["ratio_vs_h54"] for k in ks])
        lo = np.array([d[k]["lo"] for k in ks])
        hi = np.array([d[k]["hi"] for k in ks])
        # better / worse than h54 regions
        ax.axhspan(0.80, ref_kl, color=GREEN, alpha=0.07, lw=0, zorder=0)
        ax.axhline(ref_kl, color="#333333", lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(91.5, 0.875, "better than h54", fontsize=8, color=GREEN, style="italic", va="center")
        ax.text(91.5, 1.035, "worse than h54", fontsize=8, color="#777777", style="italic", va="center")
        # measured curve with its 95% CI band
        ax.fill_between(xs, lo, hi, color=GREEN, alpha=0.22, lw=0, zorder=2)
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], color=GREEN, lw=2.4, marker="o", ms=8,
                    mfc=GREEN, mec="white", mew=1.4, ecolor="#00664a", elinewidth=1.1, capsize=3,
                    zorder=4)
        for k, x, y in zip(ks, xs, ys):
            ax.annotate(split[k], (x, y), textcoords="offset points",
                        xytext={"n33": (16, 4), "n44": (-15, -14)}.get(k, (0, 11)),
                        ha="center", fontsize=8.5,
                        fontweight="bold", color="#1a1a1a",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
                        zorder=6)
        # previous coding at the same bit split, and the step-normalized replacement
        o = d["h44"]
        ax.errorbar([o["bytes_per_record"]], [o["ratio_vs_h54"]],
                    yerr=[[o["ratio_vs_h54"] - o["lo"]], [o["hi"] - o["ratio_vs_h54"]]],
                    color=ORANGE, marker="s", ms=8, mec="black", mew=0.9, ecolor="#a86f00",
                    capsize=3, ls="none", zorder=5)
        n44 = d["n44"]
        ax.annotate("", xy=(n44["bytes_per_record"] + 1.0, n44["ratio_vs_h54"] + 0.018),
                    xytext=(o["bytes_per_record"] - 0.6, o["ratio_vs_h54"] - 0.03),
                    arrowprops=dict(arrowstyle="-|>", color="#a86f00", lw=1.3,
                                    connectionstyle="arc3,rad=0.25", mutation_scale=11), zorder=5)
        ax.text(o["bytes_per_record"] + 2.0, o["ratio_vs_h54"] + 0.035, "previous 4+4",
                fontsize=8, color="#a86f00", ha="left", fontweight="bold")
        # bytes saved at equal KL: 4+4 vs h54
        ax.plot([ref_bytes], [ref_kl], marker="*", color=GRAY, ms=15, mec="black", mew=0.8,
                ls="none", zorder=6)
        yb = 0.84
        ax.annotate("", xy=(n44["bytes_per_record"], yb), xytext=(ref_bytes, yb),
                    arrowprops=dict(arrowstyle="<|-|>", color="#333333", lw=1.1,
                                    mutation_scale=9))
        pct = 100 * (n44["bytes_per_record"] - ref_bytes) / ref_bytes
        ax.text((n44["bytes_per_record"] + ref_bytes) / 2, yb + 0.018, f"{pct:.0f}% bytes",
                ha="center", fontsize=8.5, fontweight="bold", color="#333333")
        ax.set_title(f"({tag}) {mname}", fontsize=11)
        ax.set_xlabel("Host bytes per record", fontsize=10)
        ax.set_xticks([96, 112, 128, 136, 144, 152])
        ax.tick_params(labelsize=9)
        ax.set_xlim(90, 158)
        ax.set_ylim(0.80, 1.47)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("KL ratio to h54", fontsize=10)
    handles = [Line2D([], [], color=GREEN, marker="o", lw=2.4, ms=8, mec="white", mew=1.4,
                      label="Step-normalized coding (key+value bits), 95% CI"),
               Line2D([], [], color=ORANGE, marker="s", ms=8, mec="black", ls="none",
                      label="Previous coding, 4+4"),
               Line2D([], [], color=GRAY, marker="*", ms=13, mec="black", ls="none",
                      label="h54 (reference, 152 B)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.2, handletextpad=0.4, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "fig_residual_coding.png")


# --------------------------------------------------------------------------------------
def _heat(fig: plt.Figure, ax: plt.Axes, cax: plt.Axes, rows: list[str], cols: list[str],
          mat: np.ndarray, avg: np.ndarray, vmin: float, title: str, ref_row: str,
          bold_row: str = "TieredKV") -> None:
    """Score heatmap: per-task cells, a separated Avg column with the difference to ref_row,
    the TieredKV row framed, value labels with automatic contrast, and a colorbar."""
    # warm sequential map, cut before its darkest end so the black frame and text stay visible
    cmap = LinearSegmentedColormap.from_list(
        "warm", plt.get_cmap("YlOrRd")(np.linspace(0.02, 0.80, 256)))
    vmax = 100.0
    full = np.hstack([mat, np.full((len(rows), 1), np.nan), avg[:, None]])
    nc = full.shape[1]
    mesh = ax.pcolormesh(np.arange(nc + 1), np.arange(len(rows) + 1), np.ma.masked_invalid(full),
                         cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=1.6)
    ref = avg[rows.index(ref_row)]
    for i, r in enumerate(rows):
        for j in range(nc):
            v = full[i, j]
            if np.isnan(v):
                continue
            dark = (v - vmin) / (vmax - vmin) > 0.72
            color = "white" if dark else "#1a1a1a"
            txt = f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
            if j == nc - 1:
                ax.text(j + 0.5, i + 0.38, f"{v:.1f}", ha="center", va="center", fontsize=11.5,
                        fontweight="bold", color=color)
                d = v - ref
                sub = "ref." if r == ref_row else f"{d:+.1f}"
                ax.text(j + 0.5, i + 0.76, sub, ha="center", va="center", fontsize=8.5,
                        color=color, style="italic")
            else:
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=11, color=color)
    i_b = rows.index(bold_row)
    for x0, x1 in ((0, nc - 2), (nc - 1, nc)):
        ax.add_patch(plt.Rectangle((x0 + 0.03, i_b + 0.03), x1 - x0 - 0.06, 0.94, fill=False,
                                   edgecolor="black", linewidth=2.0, zorder=5))
    ax.set_xticks(np.arange(nc) + 0.5, cols)
    ax.set_yticks(np.arange(len(rows)) + 0.5, rows)
    for lab in ax.get_yticklabels():
        if lab.get_text() == bold_row:
            lab.set_fontweight("bold")
    for lab in ax.get_xticklabels():
        if lab.get_text() == "Avg":
            lab.set_fontweight("bold")
    ax.set_xlim(0, nc)
    ax.set_ylim(len(rows), 0)
    ax.grid(False)
    ax.tick_params(length=0, labelsize=10.5)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, pad=20, loc="left", fontsize=12.5)
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(labelsize=9, width=0.8, length=2.5)
    cb.set_label("Score", fontsize=10, fontweight="bold")


def fig_heatmap() -> None:
    pa, pb = DATA["fig6"]["panel_a"], DATA["fig6"]["panel_b"]
    short = {"niah_single_1": "S1", "niah_single_2": "S2", "niah_multikey_1": "MK1",
             "niah_multikey_2": "MK2", "niah_multivalue": "MV", "niah_multiquery": "MQ",
             "fwe": "FWE", "qa_1": "QA1", "qa_2": "QA2", "vt": "VT", "cwe": "CWE"}
    rows_a = list(pa["rows"])
    mat_a = np.array([pa["rows"][r] for r in rows_a], dtype=float)
    avg_a = np.array([pa["avg"][r] for r in rows_a], dtype=float)
    rows_b = list(pb["rows"])
    mat_b = np.array([pb["rows"][r]["scores"] for r in rows_b], dtype=float)
    avg_b = np.array([pb["rows"][r]["avg"] for r in rows_b], dtype=float)
    fig = plt.figure(figsize=(8.2, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 4], width_ratios=[40, 1], hspace=0.42, wspace=0.04)
    ref_a = next(r for r in rows_a if r.startswith("Full attention"))
    ref_b = next(r for r in rows_b if r.startswith("Full attention"))
    _heat(fig, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), rows_a,
          [short[t] for t in pa["tasks"]] + ["", "Avg"], mat_a, avg_a, 40,
          "(a) RULER 128K, 10 tasks x 10 items", ref_a)
    # budgets of the plotted arms (source lines ret018 and skv2048, see readme_figs_data.json)
    show_b = {"RetroInfer": "RetroInfer 0.018", "ShadowKV": "ShadowKV 2048"}
    rows_b = [show_b.get(r, r) for r in rows_b]
    _heat(fig, fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), rows_b,
          [short[t] for t in pb["tasks"]] + ["", "Avg"], mat_b, avg_b, 0,
          "(b) RULER 130K, 5 tasks x 50 items, one RTX 4090", ref_b)
    fig.text(0.5, 0.035, f"Avg column: mean score and its difference to {ref_a} in (a) and to "
             f"{ref_b} in (b). TieredKV row framed.", ha="center", va="top", fontsize=8.5,
             color="#444444")
    fig.text(0.5, 0.01, "S: NIAH-single, MK: NIAH-multikey, MV: NIAH-multivalue, MQ: NIAH-multiquery, "
             "FWE: frequent words, CWE: common words, QA: question answering, VT: variable tracking",
             ha="center", va="top", fontsize=8.5, color="#444444")
    _save(fig, "fig_ruler_heatmap.png")


# --------------------------------------------------------------------------------------
def _diff_bars(ax: plt.Axes, ypos: float, v: list[float], h: float, hatch: str, alpha: float,
               fmt: str = "{:+.2f}") -> None:
    d, lo, hi = v
    ax.barh(ypos, d, h, color=GREEN, edgecolor="black", linewidth=0.9, hatch=hatch, alpha=alpha,
            xerr=[[d - lo], [hi - d]], error_kw={"elinewidth": 1.0, "capsize": 2.5, "alpha": 1.0})
    ax.text(max(hi, d, 0) + 0.25, ypos, fmt.format(d), va="center", ha="left", fontsize=7)


def fig_diff() -> None:
    f7 = DATA["fig7"]
    vs_full, vs_comp = f7["vs_full"], f7["vs_competitors"]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.6),
                             gridspec_kw={"height_ratios": [len(vs_full), len(vs_comp) * 0.8]})
    ax = axes[0]
    for i, e in enumerate(vs_full):
        if "run2" in e:
            _diff_bars(ax, i - 0.19, e["run1"], 0.34, "", 1.0)
            _diff_bars(ax, i + 0.19, e["run2"], 0.34, "//", 0.55)
        else:
            _diff_bars(ax, i, e["run1"], 0.5, "", 1.0)
    ax.set_yticks(range(len(vs_full)), [e["label"] for e in vs_full])
    ax.set_ylim(len(vs_full) - 0.5, -0.5)
    ax.set_xlim(-5.5, 6.2)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("TieredKV minus full attention (score points, paired 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=GREEN, edgecolor="black", label="Full attention, run 1 or only run"),
                       Patch(facecolor=GREEN, edgecolor="black", hatch="//", alpha=0.55,
                             label="Full attention, run 2")],
              loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    ax = axes[1]
    for i, e in enumerate(vs_comp):
        _diff_bars(ax, i, e["v"], 0.5, "", 1.0, "{:+.1f}")
    ax.set_yticks(range(len(vs_comp)), [e["label"] for e in vs_comp])
    ax.set_ylim(len(vs_comp) - 0.5, -0.5)
    ax.set_xlim(-5.5, 15.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("TieredKV minus competitor (score points, paired 95% CI)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(h_pad=2.0)
    _save(fig, "fig_ruler_vs_baselines.png")


if __name__ == "__main__":
    fig_fidelity()
    fig_ladder()
    fig_same_gpu()
    fig_frontier()
    fig_residual()
    fig_heatmap()
    fig_diff()
