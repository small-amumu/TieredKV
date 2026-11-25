# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""NAB1 step 2: fixed-trace replay of 16-bit reasoning traces under compressed KV caches (REG_NAB1.md).

Derived from sel30_lab.py (round VFB): same attention hook, the same KIVI calibration pass on CQ.CAL, the same
Q.calib / Q.calib_all, the same chunked logit head and next-token KL to the exact model. Changes:
- input: the traces written by nab1_gen.py instead of text windows; per trace, rows < len(prompt) use exact attention
  (dense prefill, Q.PREFILL["from"] = prompt length) and every later row is a decode row;
- an eviction arm "h2o" (heavy-hitter eviction, nab1_evict.evict_attend) whose per-trace token budget matches the GPU
  peak bytes of the KIVI-128 base: B = round(Q.eff_bits("kivi128", L, d, g) / 16 * L + 55) FP16 tokens (Not All
  Bits, App. B; the + 55 turns the mean-occupancy window charge into the full 128-token window, REG A3);
- readouts per trace and run: KL(exact || run) by decode position bucket, KL over the answer span (rows
  [think_end - 1, L): from the "</think>" row, which predicts the first answer token), and the share of answer rows
  whose argmax equals the exact model's argmax.
No refcheck, no FlexiCache profiling, no timing claims.

usage: python nab1_lab.py <model_dir> <traces_jsonl> <out_json> --stack <dir holding hc_mac/> --runs a,b
       [--items 0,1,...] [--max_len 8800] [--chunk 512] [--evict_chunk 128] [--dtype float32] [--smoke]
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
ap.add_argument("--runs", required=True)
ap.add_argument("--items", default="")
ap.add_argument("--max_len", type=int, default=8800)
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--evict_chunk", type=int, default=128)
ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
ap.add_argument("--smoke", action="store_true")
A = ap.parse_args()

ST = Path(A.stack)
for sub in ("hc_mac/rootcause/vfb", "hc_mac/rootcause/vfl", "hc_mac/rootcause/spd", "hc_mac/rootcause/acs",
            "hc_mac/rootcause", "hc_mac/code"):
    sys.path.insert(0, str(ST / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v8g  # noqa: E402

v8g.install()                               # before any other import parses a name (as sel30_lab)
import bl_q as BLQ  # noqa: E402
import cq_lab3 as CQ  # noqa: E402
import nab1_evict as EV  # noqa: E402
import sel30_q as Q  # noqa: E402

CQ.CAL = (str(ST / "data" / "diag_long_text.txt"), CQ.CAL[1], CQ.CAL[2])   # same text, this host's path
BUCKETS = ((0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 1 << 30))   # decode position (from prompt end)
EV_SINKS, EV_WINDOW = 4, 128                            # nab1_evict.evict_attend defaults
KIVI_PEAK_EXTRA = 64 * (16 - 2.25) / 16                 # FP16 tokens: KIVI window peak (128) minus mean (64)
X: dict = {"run": None, "cap": False, "kv": {}, "shape": None, "budget": None, "prefill": 0, "ev_calls": 0}


def sel_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    li = int(module.layer_idx)
    q, k, v = query[0].float(), key[0].float(), value[0].float()
    scale = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
    if X["cap"]:
        X["kv"][li] = (k.clone(), v.clone(), q.clone())
        X["shape"] = (q.shape[0], k.shape[0], q.shape[-1])
    if kw.get("softcap") or kw.get("sliding_window"):
        raise RuntimeError("NAB1: softcap / sliding window not supported")
    if X["run"] == "h2o":
        # peak retained keys = alive keys before the chunk (<= budget - evict_chunk) + the chunk's own keys
        out = EV.evict_attend(q, k, v, scale, max(0, X["budget"] - A.evict_chunk), start=X["prefill"],
                              chunk=A.evict_chunk, sinks=EV_SINKS, window=EV_WINDOW)
        X["ev_calls"] += 1
    else:
        out = Q.attend(q, k, v, li, X["run"], scale, kw)
    return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None


AttentionInterface.register("sel_attn", sel_attn)


def main() -> None:
    if A.smoke != A.out.endswith(".smoke.json"):
        raise SystemExit("smoke output must end with .smoke.json (and only smoke)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = "cuda"
    dt = torch.float32 if A.dtype == "float32" else torch.bfloat16
    Q.CHUNK["rows"], BLQ.BL["chunk"] = A.chunk, A.chunk
    Q.PREFILL["on"] = True
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=dt, attn_implementation="sel_attn").to(dev).eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.model)
    re_ = model.model.rotary_emb
    BLQ.BL["inv_freq"], BLQ.BL["asc"] = re_.inv_freq.detach().float(), float(getattr(re_, "attention_scaling", 1.0))
    Q.RC.R["inv_freq"], Q.RC.R["asc"] = BLQ.BL["inv_freq"], BLQ.BL["asc"]
    runs = tuple(r for r in A.runs.split(",") if r)
    for r in runs:
        if r != "h2o":
            Q.parse(r)
    traces = [json.loads(line) for line in Path(A.traces).read_text(encoding="utf-8").splitlines() if line.strip()]
    if A.items:
        want = [int(x) for x in A.items.split(",")]
        by = {t["i"]: t for t in traces}
        missing = [i for i in want if i not in by]
        if missing:
            raise SystemExit(f"--items not in {A.traces}: {missing}")
        traces = [by[i] for i in want]
    if A.smoke:
        traces = traces[:1]

    def forward(ids):
        with torch.no_grad():
            return model.model(input_ids=ids, use_cache=False).last_hidden_state[0]

    def head(h):
        return model.lm_head(h).float()

    @torch.no_grad()
    def readouts(z0, z, p_len, a0, L):
        """KL per decode bucket, KL over the answer span, argmax agreement on the answer span."""
        sums = {b: [0.0, 0] for b in BUCKETS}
        ans = [0.0, 0, 0]                                   # KL sum, rows, argmax agreements
        dec = [0.0, 0]
        for r0 in range(p_len, L, 512):
            r1 = min(L, r0 + 512)
            l0, l1 = torch.log_softmax(head(z0[r0:r1]), -1), torch.log_softmax(head(z[r0:r1]), -1)
            klr = (l0.exp() * (l0 - l1)).sum(-1).tolist()   # per row
            agree = (l0.argmax(-1) == l1.argmax(-1)).tolist()
            for j in range(r1 - r0):
                row = r0 + j
                kv_ = klr[j]
                if a0 is not None and row >= a0 - 1:        # rows that predict answer tokens (and the end token)
                    ans[0] += kv_; ans[1] += 1; ans[2] += int(agree[j])
                    continue
                pos = row - p_len
                for b in BUCKETS:
                    if b[0] <= pos < b[1]:
                        sums[b][0] += kv_; sums[b][1] += 1
                dec[0] += kv_; dec[1] += 1
        return ({f"{b[0]}-{b[1]}": (s / n if n else None, n) for b, (s, n) in sums.items()},
                (dec[0] / dec[1] if dec[1] else None, dec[1]),
                (ans[0] / ans[1] if ans[1] else None, ans[1], ans[2] / ans[1] if ans[1] else None))

    X["cap"] = True                                         # KIVI calibration exactly as sel30_lab (CQ.CAL)
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
    g = H_ // KVH
    md5 = {Path(p).name: hashlib.md5(Path(p).read_bytes()).hexdigest() for p in
           (__file__, EV.__file__, Q.__file__, v8g.__file__, BLQ.__file__, CQ.__file__, A.traces)}
    res = {"model": A.model, "traces": A.traces, "dtype": A.dtype, "runs": list(runs), "chunk": A.chunk,
           "evict_chunk": A.evict_chunk, "max_len": A.max_len, "heads": H_, "kv_heads": KVH, "head_dim": d,
           "attn_layers": alist, "nope_layers": nope, "md5": md5, "smoke": A.smoke,
           "host_bits": {r: Q.host_bits(r, d) for r in runs if r != "h2o"},
           "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                        "gpu": torch.cuda.get_device_name(0)},
           "script_md5": md5[Path(__file__).name], "items": []}
    out = Path(A.out)
    done = {}
    if out.exists() and not A.smoke:
        old = json.loads(out.read_text(encoding="utf-8"))
        same = all(old.get(k) == res[k] for k in ("model", "traces", "dtype", "runs", "chunk", "evict_chunk", "max_len",
                                                 "md5", "versions"))
        if not same:
            raise SystemExit(f"{out} was written by another configuration; refusing to resume")
        done = {it["i"]: it for it in old["items"]}
    res["items"] = list(done.values())                      # keep every item already on disk
    print(f"resume: {len(res['items'])} items already done", flush=True)
    think_id = tok.convert_tokens_to_ids("</think>")

    def save() -> None:
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(res))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(out)                                     # atomic on POSIX

    t0 = time.time()
    for tr in traces:
        if tr["i"] in done:
            continue
        ids_l = tr["prompt_ids"] + tr["gen_ids"]
        L = min(len(ids_l), A.max_len)
        p_len = len(tr["prompt_ids"])
        a0 = tr["think_end"] if tr["think_end"] is not None and tr["think_end"] < L else None
        if a0 is not None and ids_l[a0 - 1] != think_id:
            raise SystemExit(f"item {tr['i']}: token at think_end - 1 is {ids_l[a0 - 1]}, not </think> {think_id}")
        ids = torch.tensor([ids_l[:L]], device=dev)
        Q.PREFILL["from"] = p_len
        X["prefill"] = p_len
        eff = Q.eff_bits("kivi128", L, d, g)[0]          # (gpu, read) bits per element, K/V mean, incl. window
        # eff charges the FP16 residual window at its mean occupancy (R / 2 = 64 tokens, pkv5_q.eff_bits); the
        # eviction budget is a peak, so it gets the base's peak: the full R = 128 window, i.e. 64 more tokens at
        # 16 - 2.25 bits for K and V (A3): + 64 * 13.75 / 16 = 55 FP16 tokens
        X["budget"] = max(1, round(eff / 16 * L + KIVI_PEAK_EXTRA))
        X["run"] = None
        z0 = forward(ids)
        rec = {"i": tr["i"], "L": L, "prompt": p_len, "think_end": a0, "ok16": tr["ok"], "ended": tr["ended"],
               "truncated_by_max_len": len(ids_l) > A.max_len, "kivi_eff_bits": eff, "h2o_budget": X["budget"],
               "h2o_budget_met": X["budget"] - A.evict_chunk >= EV_SINKS + EV_WINDOW,
               "eff_bits": {}, "bucket": {}, "decode": {}, "answer": {}, "sec": {}}
        for run in runs:
            X["run"] = run
            X["ev_calls"] = 0
            t_run = time.time()
            try:
                z = forward(ids)
            finally:
                X["run"] = None
            torch.cuda.synchronize()
            rec["sec"][run] = round(time.time() - t_run, 1)
            if run == "h2o" and X["ev_calls"] != len(alist):
                raise RuntimeError(f"h2o: {X['ev_calls']} evicting calls for {len(alist)} layers")
            rec["eff_bits"][run] = X["budget"] * 16 / L if run == "h2o" else Q.eff_bits(run, L, d, g)[0]
            rec.setdefault("read_bits", {})[run] = None if run == "h2o" else Q.eff_bits(run, L, d, g)[1]
            rec["bucket"][run], rec["decode"][run], rec["answer"][run] = readouts(z0, z, p_len, a0, L)
            del z
        res["items"].append(rec)
        save()
        print(f"item {tr['i']} L={L} ({time.time() - t0:.0f}s) "
              + " ".join(f"{r} dec {rec['decode'][r][0]} ans {rec['answer'][r][0]}" for r in runs), flush=True)
    res["seconds"] = time.time() - t0
    save()
    print("DONE nab1_lab", A.model, len(res["items"]), flush=True)


if __name__ == "__main__":
    main()
