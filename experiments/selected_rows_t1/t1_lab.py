# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""T1 (REG_T1.md): per-step selection statistics of the deployed configuration on NAB1's traces.

Derived from nab1_lab.py (md5 031f432d): same attention hook, KIVI calibration pass, Q.calib / Q.calib_all and trace
handling. Changes: only the deployed run kivi128k64_x2s128_j50_h54; sel14_q._attend_h is replaced by t1_q.t1_attend_h
(a verbatim copy plus read-only statistics); no exact forward, no KL; Q0a (smoke) checks that the final hidden state
equals the original function's bitwise; Q0e checks the traces md5 against NAB1's lab record.

usage: python -I t1_lab.py <model_dir> <traces_jsonl> <out_json> --stack <dir holding hc_mac/> --nab1_lab <lab json>
       [--items 0,1,...] [--max_len 8800] [--chunk 256] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AttentionInterface, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("traces")
ap.add_argument("out")
ap.add_argument("--stack", required=True)
ap.add_argument("--nab1_lab", required=True)
ap.add_argument("--items", default="")
ap.add_argument("--max_len", type=int, default=8800)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--smoke", action="store_true")
A = ap.parse_args()

ST = Path(A.stack)
for sub in ("hc_mac/rootcause/vfb", "hc_mac/rootcause/vfl", "hc_mac/rootcause/spd", "hc_mac/rootcause/acs",
            "hc_mac/rootcause", "hc_mac/code"):
    sys.path.insert(0, str(ST / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v8g  # noqa: E402

v8g.install()
import bl_q as BLQ  # noqa: E402
import cq_lab3 as CQ  # noqa: E402
import sel14_q as Q14  # noqa: E402
import sel30_q as Q  # noqa: E402
import t1_q as T1  # noqa: E402

RUN = "kivi128k64_x2s128_j50_h54"
SEL14_MD5 = "bee1a8504bfc159b1b8bff1878d58932"
CQ.CAL = (str(ST / "data" / "diag_long_text.txt"), CQ.CAL[1], CQ.CAL[2])
X: dict = {"run": None, "cap": False, "kv": {}, "shape": None}


def sel_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    li = int(module.layer_idx)
    q, k, v = query[0].float(), key[0].float(), value[0].float()
    scale = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
    if X["cap"]:
        X["kv"][li] = (k.clone(), v.clone(), q.clone())
        X["shape"] = (q.shape[0], k.shape[0], q.shape[-1])
    if kw.get("softcap") or kw.get("sliding_window"):
        raise RuntimeError("T1: softcap / sliding window not supported")
    out = Q.attend(q, k, v, li, X["run"], scale, kw)
    return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None


AttentionInterface.register("sel_attn", sel_attn)


def md5_file(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main() -> None:
    if A.smoke != A.out.endswith(".smoke.json"):
        raise SystemExit("smoke output must end with .smoke.json (and only smoke)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = "cuda"
    Q.CHUNK["rows"], BLQ.BL["chunk"] = A.chunk, A.chunk
    Q.PREFILL["on"] = True
    if md5_file(Q14.__file__) != SEL14_MD5:
        raise SystemExit(f"sel14_q md5 {md5_file(Q14.__file__)} != {SEL14_MD5}")
    if T1.S4.CHUNK["rows"] != A.chunk:
        raise SystemExit(f"chunk setting does not reach sel14_q: {T1.S4.CHUNK['rows']} != {A.chunk}")
    # Q0e: the traces are NAB1's
    nab1 = json.loads(Path(A.nab1_lab).read_text(encoding="utf-8"))
    tr_md5 = md5_file(A.traces)
    if nab1["md5"].get(Path(A.traces).name) != tr_md5:
        raise SystemExit(f"Q0e: traces md5 {tr_md5} != NAB1 record {nab1['md5'].get(Path(A.traces).name)}")
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.float32, attn_implementation="sel_attn",
                                                 device_map=dev).eval()
    re_ = model.model.rotary_emb
    BLQ.BL["inv_freq"], BLQ.BL["asc"] = re_.inv_freq.detach().float(), float(getattr(re_, "attention_scaling", 1.0))
    Q.RC.R["inv_freq"], Q.RC.R["asc"] = BLQ.BL["inv_freq"], BLQ.BL["asc"]
    Q.parse(RUN)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.model)
    traces = [json.loads(line) for line in Path(A.traces).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids_seen = [t["i"] for t in traces]
    if len(set(ids_seen)) != len(ids_seen):
        raise SystemExit("duplicate trace ids")
    if A.items:
        want = [int(x) for x in A.items.split(",")]
        by = {t["i"]: t for t in traces}
        traces = [by[i] for i in want]
    elif A.smoke:
        traces = traces[:1]

    def forward(ids):
        with torch.no_grad():
            return model.model(input_ids=ids, use_cache=False).last_hidden_state[0]

    X["cap"] = True
    try:
        with torch.no_grad():
            cal = CQ.text_ids(tok, CQ.CAL[0], CQ.CAL[1], CQ.CAL[2], dev)
            if cal.shape[1] != CQ.CAL[2]:
                raise SystemExit(f"calibration text: {cal.shape[1]} of {CQ.CAL[2]} tokens")
            model(input_ids=cal, use_cache=False)
    finally:
        X["cap"] = False
    for li, t in X["kv"].items():
        Q.calib(li, *t)
    alist = sorted(X["kv"])
    nope = Q.calib_all(model, X["kv"], alist)
    X["kv"] = {}
    H_, KVH, d = X["shape"]
    md5 = {Path(p).name: md5_file(p) for p in (__file__, T1.__file__, Q14.__file__, Q.__file__, v8g.__file__,
                                                BLQ.__file__, CQ.__file__, A.traces, CQ.CAL[0])}
    st_root = ST.resolve()

    def cur_mods() -> dict:
        out_ = {}
        for nm, mo in sorted(sys.modules.items()):
            f = getattr(mo, "__file__", None)
            if not isinstance(f, str) or not f or not os.path.isabs(f):        # torch.ops / torch.classes carry bare names like "_ops.py"
                continue
            rp = Path(f).resolve()
            if rp.is_relative_to(st_root):
                out_[nm] = md5_file(rp)
        return out_
    mods = cur_mods()
    need = {"sel14_q", "sel30_q", "bl_q", "cq_lab3", "v8g", "t1_q"}
    if not need <= set(mods):
        raise SystemExit(f"stack module check sees {sorted(mods)}; missing {sorted(need - set(mods))}")
    res = {"model": A.model, "traces": A.traces, "run": RUN, "chunk": A.chunk, "max_len": A.max_len, "heads": H_,
           "kv_heads": KVH, "head_dim": d, "attn_layers": alist, "nope_layers": nope, "md5": md5, "smoke": A.smoke,
           "policies": [T1.pol_name(p) for p in T1.POLICIES], "pages": list(T1.PAGES), "probation": list(T1.PROB),
           "buckets": [list(b) for b in T1.BUCKETS], "stack_modules_md5": mods,
           "script_md5": md5[Path(__file__).name],
           "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                        "gpu": torch.cuda.get_device_name(0)}, "items": []}
    out = Path(A.out)
    done = {}
    if out.exists() and not A.smoke:
        old = json.loads(out.read_text(encoding="utf-8"))
        if any(old.get(k) != res[k] for k in ("model", "traces", "run", "chunk", "max_len", "md5", "versions",
                                              "script_md5", "stack_modules_md5", "policies", "probation",
                                              "pages", "buckets")):
            raise SystemExit(f"{out} was written by another configuration; refusing to resume")
        done = {it["i"]: it for it in old["items"]}
    res["items"] = list(done.values())

    def save() -> None:
        if cur_mods() != mods:                       # a module imported lazily or changed on disk mid-run
            raise SystemExit("stack modules changed since the start of the run (REG section 5)")
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(res))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(out)

    Q14._attend_h = T1.t1_attend_h
    t0 = time.time()
    for tr in traces:
        if tr["i"] in done:
            continue
        ids_l = tr["prompt_ids"] + tr["gen_ids"]
        L = min(len(ids_l), A.max_len)
        p_len = len(tr["prompt_ids"])
        ids = torch.tensor([ids_l[:L]], device=dev)
        Q.PREFILL["from"] = p_len
        X["run"] = RUN
        rec = {"i": tr["i"], "L": L, "prompt": p_len, "truncated_by_max_len": len(ids_l) > A.max_len}
        if p_len >= L:
            rec["skipped"] = "no decode rows"
            res["items"].append(rec)
            save()
            continue
        torch.cuda.reset_peak_memory_stats()
        if A.smoke:                                          # Q0a: original vs copy, bitwise
            Q14._attend_h = T1.ORIG
            T1.CFG["on"] = False
            h_orig = forward(ids)
            Q14._attend_h = T1.t1_attend_h
            T1.reset_item(tr["i"], p_len)
            T1.CFG["on"] = True
            h_t1 = forward(ids)
            T1.CFG["on"] = False
            rec["q0a_equal"] = bool(torch.equal(h_orig, h_t1))
            if not rec["q0a_equal"]:
                raise SystemExit(f"Q0a FAILED: max |diff| {float((h_orig - h_t1).abs().max())}")
            del h_orig, h_t1
        else:
            T1.reset_item(tr["i"], p_len)
            T1.CFG["on"] = True
            t_run = time.time()
            try:
                _ = forward(ids)
            finally:
                T1.CFG["on"] = False
                X["run"] = None
            torch.cuda.synchronize()
            rec["sec"] = round(time.time() - t_run, 1)
        missing = [li for li in alist if li not in T1.STATS]
        if missing:
            raise SystemExit(f"item {tr['i']}: no statistics for layers {missing}")
        empty = [li for li in alist if float(T1.STATS[li]["cand"][2]) <= 0]
        if empty:
            raise SystemExit(f"item {tr['i']}: no decode rows recorded for layers {empty}")
        bad_calls = {li: T1.CALLS.get(li, 0) for li in alist if T1.CALLS.get(li, 0) != 1}
        if bad_calls:
            raise SystemExit(f"item {tr['i']}: the T1 copy was not called exactly once per layer: {bad_calls}")
        rec["layers"] = {str(li): T1.stats_to_json(T1.STATS[li], T1.STATE[li]) for li in alist}
        rec["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        rec["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
        X["run"] = None
        res["items"].append(rec)
        save()
        print(f"item {tr['i']} L={L} ({time.time() - t0:.0f}s)", flush=True)
    res["seconds"] = time.time() - t0
    save()
    print("DONE t1_lab", A.model, len(res["items"]), flush=True)


if __name__ == "__main__":
    main()
