# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""NAB2 (REG_NAB2.md): NAB1's fixed-trace replay (nab1_lab.py 031f432d) with the ablation arms of nab2_q.py.

Changes from nab1_lab.py, nothing else: the h2o arm is removed; runs named in nab2_q.ARMS go through nab2_q.attend
(call-scoped patches, then sel30_q.attend on the routing name); their bits come from nab2_q.bits; per run the number
of patched calls is recorded (nab2_q.attend itself raises unless every routed call fires its patch exactly once).
The md5 record also covers sel14_q.py and pkv5_q.py, the frozen modules the patches wrap. The h2o budget fields
still written per item are nab1_lab's bookkeeping and unused here. NAB2 never resumes (an existing output file is
refused, so one file is one process); every item records its arm order and the process's run id; --perm_seed runs
the arms of each item in a seeded random order (Q0-d); versions also record the CUDA version.
usage: python nab2_lab.py <model_dir> <traces_jsonl> <out_json> --stack <dir holding hc_mac/> --runs a,b [--items ..]
       [--perm_seed N]
The text below is nab1_lab.py's.

NAB1 step 2: fixed-trace replay of 16-bit reasoning traces under compressed KV caches (REG_NAB1.md).

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
import random
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
# Q0-d (REG_NAB2.md): >= 0 runs the arms of each item in a seeded random order (seed * 1000 + item);
# -1 keeps the --runs order. Every item records its order and this process's run id.
ap.add_argument("--perm_seed", type=int, default=-1)
# the per-model control replay may also run t_variant_off (patches present, every ablation off), so the
# patched function is checked against the deployed arm on every model (REG_NAB2.md, Q0-e)
ap.add_argument("--ctl", action="store_true")
A = ap.parse_args()
RUN_ID = __import__("uuid").uuid4().hex
if A.perm_seed < -1:
    raise SystemExit("--perm_seed must be -1 (off) or >= 0")
if Path(A.out).exists():
    # NAB2: no resume, so one output file is always one process (one run id); delete it to rerun
    raise SystemExit(f"{A.out} exists; NAB2 does not resume (one file = one process)")

ST = Path(A.stack)
for sub in ("hc_mac/rootcause/vfb", "hc_mac/rootcause/vfl", "hc_mac/rootcause/spd", "hc_mac/rootcause/acs",
            "hc_mac/rootcause", "hc_mac/code"):
    sys.path.insert(0, str(ST / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v8g  # noqa: E402

v8g.install()                               # before any other import parses a name (as sel30_lab)
import bl_q as BLQ  # noqa: E402
import cq_lab3 as CQ  # noqa: E402
import sel30_q as Q  # noqa: E402
import nab2_q as N2  # noqa: E402

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
    if N2.is_arm(X["run"]):
        out = N2.attend(q, k, v, li, X["run"], scale, kw)
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
        if r == "h2o":
            raise SystemExit("NAB2: no h2o arm")
        if r in N2.TEST_ARMS and not A.smoke and not (A.ctl and r == "t_variant_off"):
            raise SystemExit(f"NAB2: self-test arm {r} only in --smoke runs (t_variant_off also with --ctl)")
        Q.parse(N2.ARMS[r]["route"] if N2.is_arm(r) else r)   # routing names (native arms included) must parse
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
           (__file__, N2.__file__, N2.Q14.__file__, N2.Q5.__file__, Q.__file__, v8g.__file__, BLQ.__file__, CQ.__file__,
            A.traces, CQ.CAL[0])}
    st_root = ST.resolve()
    for m in list(sys.modules.values()):                   # every stack module the replay loaded (resume guard)
        f = getattr(m, "__file__", None)
        # absolute paths only: torch.ops / torch.classes carry a bare relative __file__ ('_ops.py'),
        # which would resolve into the cwd (inside the stack root) and name a file that does not exist
        if f and os.path.isabs(f) and f.endswith(".py") and Path(f).resolve().is_relative_to(st_root):
            rel = Path(f).resolve().relative_to(st_root).as_posix()
            md5[rel] = hashlib.md5(Path(f).read_bytes()).hexdigest()
    res = {"model": A.model, "traces": A.traces, "dtype": A.dtype, "runs": list(runs), "chunk": A.chunk,
           "evict_chunk": A.evict_chunk, "max_len": A.max_len, "heads": H_, "kv_heads": KVH, "head_dim": d,
           "attn_layers": alist, "nope_layers": nope, "md5": md5, "smoke": A.smoke,
           "host_bits": {r: Q.host_bits(r, d) for r in runs if not N2.is_arm(r)},
           "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda},
           "perm_seed": A.perm_seed, "run_id": RUN_ID,
           "script_md5": md5[Path(__file__).name], "items": []}
    out = Path(A.out)              # existence was refused before the model loaded (no resume)
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
        order = list(runs)
        if A.perm_seed >= 0:                                 # Q0-d: a seeded per-item arm order
            random.Random(A.perm_seed * 1000 + tr["i"]).shuffle(order)
        rec["order"], rec["run_id"] = order, RUN_ID
        for run in order:
            X["run"] = run
            c0 = (N2.STATE["calls_h"], N2.STATE["calls_base"])
            f0 = N2.STATE["calls_frozen"]                       # calls of the frozen sel14 _attend_h
            t_run = time.time()
            try:
                z = forward(ids)
            finally:
                X["run"] = None
            torch.cuda.synchronize()
            rec["sec"][run] = round(time.time() - t_run, 1)
            calls = (N2.STATE["calls_h"] - c0[0], N2.STATE["calls_base"] - c0[1])
            rec.setdefault("patched_calls", {})[run] = calls
            rec.setdefault("frozen_calls", {})[run] = N2.STATE["calls_frozen"] - f0
            if N2.is_arm(run):
                a_ = N2.ARMS[run]
                if a_["route"] == run:                                   # native arm: no patch may fire
                    want = (0, 0)
                else:                                                     # one patched call per layer
                    want = (0, len(alist)) if a_["kivi_bits"] is not None else (len(alist), 0)
                if calls != want:
                    raise RuntimeError(f"{run}: patched calls {calls}, expected {want}")
                b_ = N2.bits(run, L, d, g)
                rec["eff_bits"][run], rec.setdefault("read_bits", {})[run] = b_["gpu"], b_["read"]
                rec.setdefault("host_bits", {})[run] = b_["host"]
            else:
                if calls != (0, 0):
                    raise RuntimeError(f"{run}: a nab2_q patch fired on a stack arm ({calls})")
                rec["eff_bits"][run], rec.setdefault("read_bits", {})[run] = Q.eff_bits(run, L, d, g)[:2]
                rec.setdefault("host_bits", {})[run] = Q.host_bits(run, d)
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
