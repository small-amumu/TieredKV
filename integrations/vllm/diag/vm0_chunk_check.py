# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""VM0 step 1a (CPU): is the research attend() row-independent of how the keys arrive?

M0 (VM0) called sel29_q.attend once per vLLM prefill chunk: the chunk's query rows against the keys
up to the chunk's end. The research harness calls it once with every row and every key. If a
method's output for a row depends on keys after that row (e.g. a base quantised over all keys at
once), the two calls differ and VM0's gap is a protocol / wiring effect, not input amplification.

On the CPU, float32: Hugging Face model with the research attention, calibration exactly as
sel29_lab (CQ.CAL, Q.calib, Q.calib_all), then one evaluation window (CQ.EVAL[0], start 4096,
4096 tokens) captured with exact attention. For a few layers, rows >= F through attend() once
(keys 0..L-1) vs in chunks of C rows (keys 0..chunk end), per method. Reports the max abs and
relative difference of the outputs and the share of rows that differ.
Usage: python vm0_chunk_check.py --model DIR --research ROOT --out F.json [--layers 0,8,15]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

RESEARCH_SUBDIRS = ("code", "rootcause", "rootcause/acs", "rootcause/spd", "rootcause/vfl")
METHODS = (
    None,
    "kivi128",
    "kivi128k80_x2s160_h88_r0",
    "stkm8v8128",
    "stkm8v8128k64_x2s128_p50_h54",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--research", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--module", default="sel29_q", help="research entry module (sel29_q / sel14_q)")
    ap.add_argument("--methods", default="", help="comma list (default: the built-in set)")
    ap.add_argument("--save", default="", help="save the one-call outputs per (layer, method) here")
    a = ap.parse_args()
    root = Path(a.research)
    for sub in reversed(RESEARCH_SUBDIRS):
        sys.path.insert(0, str(root / sub))
    import bl_q as BLQ
    import cq_lab3 as CQ
    import importlib

    Q = importlib.import_module(a.module)
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(16)
    dev = "cpu"
    Q.CHUNK["rows"], BLQ.BL["chunk"] = a.chunk, a.chunk
    Q.PREFILL["on"], Q.PREFILL["from"] = False, a.rows
    X: dict = {"cap": False, "kv": {}}

    def cap_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):  # type: ignore[no-untyped-def]  # noqa: E501
        li = int(module.layer_idx)
        q, k, v = query[0].float(), key[0].float(), value[0].float()
        scale = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
        if X["cap"]:
            X["kv"][li] = (k.clone(), v.clone(), q.clone(), scale, dict(kw))
        out = Q.attend(q, k, v, li, None, scale, kw)
        return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None

    AttentionInterface.register("vm0_cap", cap_attn)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, attn_implementation="vm0_cap"
    ).eval()
    re_ = model.model.rotary_emb
    BLQ.BL["inv_freq"] = re_.inv_freq.detach().float()
    BLQ.BL["asc"] = float(getattr(re_, "attention_scaling", 1.0))
    Q.RC.R["inv_freq"], Q.RC.R["asc"] = BLQ.BL["inv_freq"], BLQ.BL["asc"]
    t0 = time.time()
    X["cap"] = True
    with torch.no_grad():
        model(input_ids=CQ.text_ids(tok, CQ.CAL[0], CQ.CAL[1], CQ.CAL[2], dev), use_cache=False)
    X["cap"] = False
    for li, t in X["kv"].items():
        Q.calib(li, t[0], t[1], t[2])
    alist = sorted(X["kv"])
    Q.calib_all(model, {li: t[:3] for li, t in X["kv"].items()}, alist)
    X["kv"] = {}
    ids = CQ.text_ids(tok, CQ.EVAL[0][1], 4096, a.seq, dev)
    X["cap"] = True
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    X["cap"] = False
    t_cap = time.time() - t0
    layers = [int(x) for x in a.layers.split(",")] if a.layers else [0, len(alist) // 2, alist[-1]]
    F, C, L = a.rows, a.chunk, a.seq
    res: dict = {
        "model": a.model, "seq": L, "rows": F, "chunk": C, "secs_capture": round(t_cap, 1),
        "md5_sel29_q": __import__("hashlib").md5(Path(Q.__file__).read_bytes()).hexdigest(),
        "cells": [],
    }  # fmt: skip
    methods = [m or None for m in a.methods.split(",")] if a.methods else list(METHODS)
    saved: dict = {}
    for li in layers:
        k, v, q, scale, kw = X["kv"][li]
        for m in methods:
            with torch.no_grad():
                full = Q.attend(q[:, F:], k, v, li, m, scale, kw)
                parts = [
                    Q.attend(q[:, c : c + C], k[:, : c + C], v[:, : c + C], li, m, scale, kw)
                    for c in range(F, L, C)
                ]
            ch = torch.cat(parts, dim=1)
            saved[f"{li}:{m or 'exact'}"] = full.clone()
            d = (full - ch).abs()
            denom = full.abs().amax().clamp_min(1e-30)
            row_diff = (d.amax(dim=(0, 2)) > 1e-6 * denom).float().mean()
            cell = {
                "layer": li,
                "method": m or "exact",
                "max_abs": float(d.max()),
                "max_rel": float(d.max() / denom),
                "rows_differing": float(row_diff),
                "mean_rel": float(d.mean() / full.abs().mean().clamp_min(1e-30)),
            }
            res["cells"].append(cell)
            print("CHK", json.dumps(cell), flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.save:
        torch.save(saved, a.save)
    res["module"] = a.module
    res["module_md5"] = __import__("hashlib").md5(Path(Q.__file__).read_bytes()).hexdigest()
    res["secs"] = round(time.time() - t0, 1)
    res["done"] = True
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
