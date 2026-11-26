# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""sel14_lab (rootcause fork; round D2): sel13_lab with sel14_q (adds the _h<rk><rv> host-residual store); records host bits
per run and the number of _hff equality checks against sel4_q.

sel13_lab (rootcause fork; diagnosis D1): sel12_lab with sel13_q (adds the _vx value-exact control and the _p<L>
positive-correction names). Per segment it also stores, for every _vx run, the number of base calls whose values were
replaced (must equal the number of attention layers, checked).

sel12_lab (rootcause fork; round S1): sel11_lab with sel12_q (8-bit-metadata stack names). sel10_lab (rootcause fork; combination round): sel6_lab with sel10_q (stk128 names: the main line's strongest base
wfc_ck_w128_rc under sel4_q selection / two-stage / debias) and the main line's full calibration (pkv8 / 9 / 10 / 11).
sel6_lab (rootcause fork; configuration policy step 1): sel4_lab with sel6_q, i.e. every sel4_q method plus the
per-token kernel base (tok128[rc] names), and the queries of the calibration text captured as well (channel balance
and rotary mean of rc_q / pkv7_q). Original sel4_lab docstring follows.
sel4_lab: round S4 of the layered-KV selection line (rootcause fork; registered in reg_pkv.md). Every method of sel4_q
(ours, two-stage controls, mass-matched shift) and of bl_q (Quest / ShadowKV / FlexiCache reproductions) on the
evaluation windows of S1-S3, scored by next-token KL to the exact model on rows >= --rows. Rows < --rows use exact
attention for every method (dense prefill). Before the evaluation windows, one exact pass on the calibration text (CQ.CAL file,
disjoint from the evaluation texts) sets the KIVI calibration (CQ.CAL window, exactly as S1-S3) and, for every flexi_ run,
profiles FlexiCache's unstable heads on 3 --seq windows (offsets 40000 / 46000 / 52000; teacher-forced rows stand in for
the official generated decode steps) (bl_q.flexi_profile_layer / flexi_pick_unstable, U% of all
(layer, KV head) pairs). For --seq > 4096 the logits are formed from the last hidden state per 512-row chunk (rows >=
--rows only), as the main line's P11, so 16K windows fit.
Usage: python sel4_lab.py <model_dir> <out_json> --runs a,b [--seq 4096] [--rows 2048] [--chunk 512] [--dtype float32]
       [--starts 4096,36864] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/user/kvt_bench/hc_mac/code")
sys.path.insert(0, "/home/user/kvt_bench/hc_mac/rootcause")
import bl_q as BLQ  # noqa: E402
import cq_lab3 as CQ  # noqa: E402
import sel14_q as Q  # noqa: E402

CAL_STARTS = (40000, 46000, 52000)          # FlexiCache profiling windows (as S3's calibration windows)
X: dict = {"run": None, "cap": False, "prof_on": False, "kv": {}, "shape": None, "prof": {}, "prof_B": ()}


def sel_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    li = int(module.layer_idx)
    q, k, v = query[0].float(), key[0].float(), value[0].float()
    scale = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
    if X["cap"]:
        X["kv"][li] = (k.clone(), v.clone(), q.clone())                    # KIVI + per-token base calibration
        X["shape"] = (q.shape[0], k.shape[0], q.shape[-1])
    if X["prof_on"]:
        for B in X["prof_B"]:                                              # FlexiCache profiling, no storage of q
            X["prof"].setdefault(B, {}).setdefault(li, []).append(BLQ.flexi_profile_layer(q, k, Q.PREFILL["from"], B).cpu())
    out = Q.attend(q, k, v, li, X["run"], scale, kw)
    return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None


AttentionInterface.register("sel_attn", sel_attn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--starts", default="4096,36864")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke != a.out.endswith(".smoke.json"):
        raise SystemExit("smoke output must end with .smoke.json (and only smoke)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = a.device
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    Q.CHUNK["rows"], BLQ.BL["chunk"] = a.chunk, a.chunk
    Q.PREFILL["on"], Q.PREFILL["from"] = True, a.rows
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="sel_attn").to(dev).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    re_ = model.model.rotary_emb
    BLQ.BL["inv_freq"], BLQ.BL["asc"] = re_.inv_freq.detach().float(), float(getattr(re_, "attention_scaling", 1.0))
    Q.RC.R["inv_freq"], Q.RC.R["asc"] = BLQ.BL["inv_freq"], BLQ.BL["asc"]
    starts = tuple(int(x) for x in a.starts.split(","))
    runs = tuple(r for r in a.runs.split(",") if r)
    for r in runs:
        (BLQ.parse if BLQ.is_bl(r) else Q.parse)(r)
    flexi = sorted({(BLQ.parse(r)["B"], BLQ.parse(r)["U"]) for r in runs if r.startswith("flexi_")})
    X["prof_B"] = tuple(sorted({B for B, _ in flexi}))
    long_ = a.seq > 4096
    fcap = getattr(model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config,
                   "final_logit_softcapping", None)

    def ids_at(si, path, st, n):
        if a.smoke:
            return torch.randint(100, tok.vocab_size - 100, (1, n), generator=torch.Generator().manual_seed(st + 5 + 1000 * si)).to(dev)
        w = CQ.text_ids(tok, path, st, n, dev)
        assert w.shape[1] == n, f"{path} from {st}: only {w.shape[1]} of {n} tokens"
        return w

    def forward(ids):
        with torch.no_grad():
            if long_:
                return model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            return model(input_ids=ids, use_cache=False).logits[0].float()

    def head(h):
        z = model.lm_head(h).float()
        if fcap:
            z = torch.tanh(z / fcap) * fcap
        return z

    def kl(z0, z, r_from):
        tot, n = 0.0, 0
        for r0 in range(r_from, z0.shape[0], 512):
            a0, a1 = (head(z0[r0:r0 + 512]), head(z[r0:r0 + 512])) if long_ else (z0[r0:r0 + 512], z[r0:r0 + 512])
            lr, ln = torch.log_softmax(a0, -1), torch.log_softmax(a1, -1)
            tot += float((lr.exp() * (lr - ln)).sum())
            n += lr.shape[0]
        return tot / n

    assert all(CQ.CAL[0] != p for _, p in CQ.EVAL), "calibration text must differ from the evaluation texts"
    X["cap"] = True                                                        # KIVI calibration exactly as S1-S3
    try:
        with torch.no_grad():
            model(input_ids=ids_at(0, CQ.CAL[0], CQ.CAL[1], CQ.CAL[2]), use_cache=False)
    finally:
        X["cap"] = False
    if X["prof_B"]:
        X["prof_on"] = True
        try:
            for cst in CAL_STARTS:
                forward(ids_at(0, CQ.CAL[0], cst, a.seq))
        finally:
            X["prof_on"] = False
    X["prof"] = {B: {li: torch.cat(v, 0) for li, v in d.items()} for B, d in X["prof"].items()}
    for li, t in X["kv"].items():
        Q.calib(li, *t)
    alist = sorted(X["kv"])
    nope = Q.calib_all(model, X["kv"], alist)
    X["kv"] = {}
    KVH = X["shape"][1]
    for B, U in flexi:
        M = max(1, round(U / 100 * len(alist) * KVH)) if U else 0
        BLQ.BL["flexi_unstable"][(B, U)] = BLQ.flexi_pick_unstable(X["prof"][B], M) if M else set()
    H_, KVH, d = X["shape"]
    res = {"model": a.model, "dtype": a.dtype, "seq": a.seq, "rows": a.rows, "starts": list(starts), "smoke": a.smoke,
           "runs": list(runs), "chunk": a.chunk, "prefill_exact": True, "heads": H_, "kv_heads": KVH, "head_dim": d,
           "attn_layers": alist,
           "flexi_unstable": {f"{B}_{U}": sorted(map(list, BLQ.BL["flexi_unstable"][(B, U)])) for B, U in flexi},
           "bits": {r: Q.eff_bits(r, a.seq, d, H_ // KVH) for r in runs if not BLQ.is_bl(r)},
           "md5": {n: hashlib.md5(Path(p).read_bytes()).hexdigest() for n, p in
                   (("sel14_lab.py", __file__), ("sel14_q.py", Q.__file__), ("sel13_q.py", Q.Q13.__file__), ("sel12_q.py", Q.S12.__file__), ("sel11_q.py", Q.S0.Q.__file__), ("sel10_q.py", Q.S10.__file__), ("s0_cov.py", Q.S0.__file__), ("sel4_q.py", Q.S4.__file__), ("rc_q.py", Q.RC.__file__),
                    ("pkv7_q.py", Q.Q7.__file__), ("pkv11_q.py", Q.P11.__file__), ("pkv10_q.py", Q.P10.__file__),
                    ("pkv9_q.py", Q.P9.__file__), ("pkv8_q.py", Q.P8.__file__), ("bl_q.py", BLQ.__file__),
                    ("pkv5_q.py", Q.Q5.__file__), ("cq_lab3.py", CQ.__file__))},
           "nope_layers": nope,
           "seg": []}
    t0 = time.time()
    for si, (name, path) in enumerate(CQ.EVAL):
        for st in starts[:1] if a.smoke else starts:
            ids = ids_at(si, path, st, a.seq)
            z0 = forward(ids)
            rec = {"doc": name, "start": st, "D": {}, "vx_calls": {}}
            for run in runs:
                X["run"] = run
                n0 = Q.VX["n"]
                try:
                    z = forward(ids)
                finally:
                    X["run"] = None
                if Q.Q13.split(Q.split(run)[0] if Q.split(run) else run):
                    rec["vx_calls"][run] = Q.VX["n"] - n0
                    if rec["vx_calls"][run] != len(alist):
                        raise RuntimeError(f"{run}: {rec['vx_calls'][run]} value replacements for {len(alist)} layers")
                rec["D"][run] = kl(z0, z, a.rows)
                del z
            res["seg"].append(rec)
            print(f"{name} {st} ({time.time() - t0:.0f}s) " + " ".join(f"{k} {v:.4g}" for k, v in rec["D"].items()), flush=True)
            Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    res["seconds"] = time.time() - t0
    res["host_bits"] = {r: Q.host_bits(r, d) for r in runs}
    res["hff_checks"] = Q.HCHECK["n"]
    res["m8_zsat"] = dict(Q.CLIP)          # m8 tokens coded and candidate zero points saturating e4m3
    print(f"m8 zero saturation counts {Q.CLIP}", flush=True)
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
