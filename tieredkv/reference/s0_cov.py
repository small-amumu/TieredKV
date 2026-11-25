# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""s0_cov (rootcause fork; round S0, forward-only diagnostics, registered in reg_pkv.md before any output).
One exact forward per sample (prefix + teacher-forced answer, lit protocol: the cache of the prefix is what is quantized);
inside the attention of every layer, for the last R <= 64 rows (the answer rows) and the prefix keys outside the 4 sinks
and the 128-token window (positions [4, T0 - 128), T0 = prefix length), the module computes, per query head:
  coverage of a base:  id@M  = |exact top-64 inside the base's top-M| / 64,  mass@M = exact softmax mass (over the
                       region) inside the base's top-M, M in {64, 128, 256}; GQA also with the candidate set shared per
                       KV head (base score = max over the group), reported as group mean and group worst.
  bases (keys only; values are not part of the ranking):
    stk     the main line's stack base (sel10_q.stack_base, wfc_ck_w128_rc): 3-bit hi half (min-max + clip search),
            1-bit lo half (mean +- MAD), fp16 metadata. Recomputed here and asserted equal to stack_base.
    stk_m8  (c) hi half step in float8 e4m3, zero in e4m3 step units, lo half (mean, MAD) in e4m3 (metadata 8 bit)
    stk_sym8 (c) hi half symmetric (no zero point, scale max|y| / 3.5 in e4m3), lo half as stk_m8
    stk_u4s (b) hi half codebook: uniform over mean +- 4 sigma (8 levels), sigma per token, fp16 (mean, sigma)
    stk_lm  (b) hi half codebook: Lloyd-Max levels of a unit Gaussian times sigma, fp16 (mean, sigma)
    stk2    hi half at 2 bits (min-max + clip search), lo half as stk: keys ~1.6 bits incl. metadata (with 2-bit values
            ~1.95 bits mean over K and V, the W-0008 'B' allocation); not a 2-bit key
  (e) step overlap: |top-128(row r) n top-128(row r+1)| / 128 of the stk base, shared per KV head
  (f) sigma_j variation: coefficient of variation over the region of sel10_q's per-token score-error sigma_j (stack_sigma2)
  (g) host residual: residual r = k - k_stk coded at 4 bits per token (min-max, fp16 scale / zero) -> k4; inside the stk
      top-128, the top-64 by k4 logits vs by exact logits: overlap / 64 and exact mass of the chosen 64 / of the exact top-64
  (d) value tail: with exact attention weights, output error ||o - o_exact|| / ||o_exact|| when the stk top-64 rows use
      exact values and the rest use (i) the stack's 2-bit values or (ii) the mean of exact values over 256-token blocks
Calibration exactly as sel11_lab (CQ.CAL text). Inputs: --jsonl samples ({"input": ..., "answer": ...}) and/or --text
file windows (start offsets; the last 64 tokens act as the answer). Output json: per sample, per layer means.
Usage: python s0_cov.py <model_dir> <out_json> [--jsonl f] [--text f --starts a,b --seq n] [--dtype float16] [--smoke]"""
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
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

_KR = os.environ.get("KVT_ROOT", "/home/user/kvt_bench")          # codec_fix_v3: overridable (was hard-coded)
sys.path.insert(0, _KR + "/hc_mac/code")
sys.path.insert(0, _KR + "/hc_mac/rootcause")
import cq_lab3 as CQ  # noqa: E402
import sel11_q as Q  # noqa: E402

S10 = Q.S10
P10, P9, P11, Q7, Q5 = S10.P10, S10.P9, S10.P11, S10.Q7, S10.Q5
SINKS, WIN, KTOP, MS = 4, 128, 64, (64, 128, 256)
E4MIN = 2.0 ** -9                                                       # smallest float8 e4m3 subnormal
LM3 = torch.tensor([-2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520])   # Lloyd-Max, N(0,1), 8 levels
X: dict = {"cap": False, "eval": False, "kv": {}, "shape": None, "R": 0, "T0": 0, "acc": {}, "li": -1, "skip": []}


def e4(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def hi_quant(y, kind):
    """hi-half codes of y [..., h] -> reconstruction, per token. kind: clip3 / clip2 (sel10_q._step_clip), m8 (clip3 with
    float8 step and zero), u4s (uniform mean +- 4 sigma), lm (Lloyd-Max)."""
    if kind in ("clip3", "clip2"):
        return S10._step_clip(y, 2 if kind == "clip2" else 3)[0]
    if kind == "m8":            # the clip search with the step in float8 e4m3 and the zero in float8 e4m3 step units
        lv = 7
        mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
        c, h = (mn + mx) / 2, (mx - mn) / 2
        best, berr = None, None
        for gm in Q.Q7.Q6.GAMMAS:
            raw = 2 * h * gm / lv
            sc = torch.where(raw > 0, e4(raw).clamp_min(E4MIN), 0.0)     # no flush to 0 -> exact (review)
            safe = torch.where(sc > 0, sc, torch.ones_like(sc))
            lo = e4((c - h * gm) / safe) * sc
            xh = torch.where(sc > 0, torch.clamp(torch.round((y - lo) / safe), 0, lv) * sc + lo, y)
            err = ((xh - y) ** 2).sum(-1, keepdim=True)
            best, berr = (xh, err) if best is None else (torch.where(err < berr, xh, best), torch.minimum(err, berr))
        return best
    if kind == "sym":           # symmetric 8-level grid, no zero point: levels (i - 3.5) sc, sc = max|y| / 3.5 in e4m3
        raw = y.abs().amax(-1, keepdim=True) / 3.5
        sc = torch.where(raw > 0, e4(raw).clamp_min(E4MIN), 0.0)
        safe = torch.where(sc > 0, sc, torch.ones_like(sc))
        return torch.where(sc > 0, (torch.clamp(torch.round(y / safe + 3.5), 0, 7) - 3.5) * sc, y)
    mu = Q5.f16(y.mean(-1, keepdim=True))
    sd = Q5.f16(y.std(-1, keepdim=True)).clamp_min(1e-12)
    z = (y - mu) / sd
    if kind == "u4s":
        st = 8.0 / 7
        return (torch.clamp(torch.round((z + 4) / st), 0, 7) * st - 4) * sd + mu
    if kind != "lm":
        raise ValueError(kind)
    lv = LM3.to(y.device)
    mids = (lv[1:] + lv[:-1]) / 2                                       # nearest level via bucketize (no x8 broadcast)
    return lv[torch.bucketize(z.contiguous(), mids)] * sd + mu


def lo_quant(y, m8):
    mu = Q5.f16(y.mean(-1, keepdim=True))
    s = Q5.f16((y - mu).abs().mean(-1, keepdim=True))
    if m8:
        mu, s = e4(mu), e4(s).abs()
    return mu + s * torch.where(y >= mu, 1.0, -1.0)


def stack_variant(k, li, hi, m8):
    """the stack's key reconstruction with the hi codebook `hi` and 8-bit metadata `m8` (sel10_q.stack_base lines)."""
    f, n7 = P10.split(S10.STACK)
    rc = bool(f["rc"]) and li not in P11.NOPE
    mt, b = P9.centre(li, k, rc, Q7.parse(n7)["a"])
    x = (k - mt) * b
    I = S10.INFO[li]
    o = I["order"][:, None, :].expand_as(x)
    xs = x.gather(-1, o)
    h = I["h"]
    Hh = Q5.CQ.hadamard(h, k.device)
    yh = hi_quant(xs[..., :h] @ Hh, hi)
    lo = lo_quant(xs[..., h:] @ Hh, m8)
    rec = torch.empty_like(x).scatter_(-1, o, torch.cat([yh @ Hh.T, lo @ Hh.T], -1)) / b + mt
    rec[:, :1] = k[:, :1]
    return rec


def res4(r):
    mn, mx = r.amin(-1, keepdim=True), r.amax(-1, keepdim=True)
    sc = Q5.f16((mx - mn) / 15).clamp_min(1e-12)
    lo = Q5.f16(mn)
    return torch.clamp(torch.round((r - lo) / sc), 0, 15) * sc + lo


def add(key, val):
    val = float(val)
    if not math.isfinite(val):                                           # NaN / inf are counted, never averaged (review)
        X["acc"].setdefault("nonfinite|" + key, [0.0, 0])[1] += 1
        return
    for kk in (key, f"L{X['li']}|{key}"):                                # pooled over layers and per layer (review)
        a = X["acc"].setdefault(kk, [0.0, 0])
        a[0] += float(val)
        a[1] += 1


def metrics(q, k, v, li, scale, kw):
    """q [H, T, d], k, v [KVH, T, d] (post-RoPE, float32); answer rows [T0, T0 + R). With a sliding window sw (e.g.
    Mistral-7B-v0.1, 4096) the region starts at T0 + R - sw, so every compared key is visible to every answer row; the
    4 sinks are then outside the window and simply not part of the region (review)."""
    assert not kw.get("softcap"), "logit softcapping not supported by the S0 metrics"
    X["li"] = li
    H, T, d = q.shape
    KVH = k.shape[0]
    g = H // KVH
    T0, R = X["T0"], X["R"]
    sw = kw.get("sliding_window")
    lo_, hi_ = max(SINKS, T0 + R - sw) if sw else SINKS, T0 - WIN
    if hi_ - lo_ < 4 * max(MS):
        X["skip"].append(li)
        return
    kq_ref, vq = S10.stack_base(k[:, :T0], v[:, :T0], li)[:2]          # sets INFO[li]; prefix only (lit)
    kr = k[:, lo_:hi_]
    qa = q[:, T0:T0 + R]                                                 # [H, R, d]
    rep = lambda t: t.repeat_interleave(g, 0)                           # noqa: E731
    ex = torch.einsum("hrd,hnd->hrn", qa, rep(kr)) * scale              # exact logits [H, R, N]
    pex = torch.softmax(ex, -1)
    top_ex = ex.topk(KTOP, -1).indices
    VARIANTS = (("stk", None, None), ("stk_m8", "m8", True), ("stk_sym8", "sym", True), ("stk_u4s", "u4s", False),
                ("stk_lm", "lm", False), ("stk2", "clip2", False))       # built one at a time and freed (review: memory)
    chk = stack_variant(k[:, :T0], li, "clip3", False)
    if not float((chk[:, 1:] - kq_ref[:, 1:]).abs().max() / k.abs().max().clamp_min(1e-6)) < 1e-4:
        raise RuntimeError("stack_variant does not replicate stack_base")
    del chk
    hit = torch.zeros_like(ex, dtype=torch.bool).scatter_(-1, top_ex, True)
    for nm, hi, m8 in VARIANTS:
        kb = kq_ref if hi is None else stack_variant(k[:, :T0], li, hi, m8)
        bl = torch.einsum("hrd,hnd->hrn", qa, rep(kb[:, lo_:hi_])) * scale
        bs = bl.view(KVH, g, R, -1).amax(1)                              # shared per KV head
        for M in MS:
            ti = bl.topk(M, -1).indices
            add(f"{nm}|id@{M}", hit.gather(-1, ti).float().sum(-1).mean() / KTOP)
            add(f"{nm}|mass@{M}", pex.gather(-1, ti).sum(-1).mean())
            if g > 1:
                ts = rep(bs.topk(M, -1).indices)
                idg = (hit.gather(-1, ts).float().sum(-1) / KTOP).view(KVH, g, R)
                add(f"{nm}|shared_id@{M}_mean", idg.mean())
                add(f"{nm}|shared_id@{M}_worst", idg.amin(1).mean())
                msg = pex.gather(-1, ts).sum(-1).view(KVH, g, R)
                add(f"{nm}|shared_mass@{M}_mean", msg.mean())
                add(f"{nm}|shared_mass@{M}_worst", msg.amin(1).mean())
        if nm == "stk":
            t128 = bs.topk(128, -1).indices                              # (e) step overlap, shared per KV head
            if R > 1:
                a_ = torch.zeros(KVH, R, bs.shape[-1], dtype=torch.bool, device=bs.device).scatter_(-1, t128, True)
                add("stk|step_overlap128", (a_[:, 1:] & a_[:, :-1]).float().sum(-1).mean() / 128)
            stk_top128 = rep(bs.topk(128, -1).indices)                # the method's shared sets (review)
            stk_top64 = rep(bs.topk(KTOP, -1).indices)
        del bl, bs, kb
    # (f) sigma_j coefficient of variation (sel10_q.stack_sigma2 on the region)
    S10.ST["on"], S10.ST["li"] = True, li
    try:
        sg = S10.stack_sigma2(k[:, :T0], qa, scale)[..., lo_:hi_].clamp_min(0).sqrt()
    finally:
        S10.ST["on"] = False
    add("stk|sigma_cv", (sg.std(-1) / sg.mean(-1).clamp_min(1e-30)).mean())
    # (g) 4-bit host residual: rerank inside the stk top-128
    k4 = kq_ref[:, lo_:hi_] + res4(kr - kq_ref[:, lo_:hi_])
    l4 = torch.einsum("hrd,hnd->hrn", qa, rep(k4)) * scale
    c_ex, c_4 = ex.gather(-1, stk_top128), l4.gather(-1, stk_top128)
    pe, p4 = c_ex.topk(KTOP, -1).indices, c_4.topk(KTOP, -1).indices
    he = torch.zeros_like(c_ex, dtype=torch.bool).scatter_(-1, pe, True)
    add("res4|rerank_overlap64", he.gather(-1, p4).float().sum(-1).mean() / KTOP)
    cm = pex.gather(-1, stk_top128)
    add("res4|mass_ratio64", (cm.gather(-1, p4).sum(-1) / cm.gather(-1, pe).sum(-1).clamp_min(1e-30)).mean())
    # (d) value tail with exact weights: stk top-64 exact values, the rest stack 2-bit values or 256-token block means
    sel = stk_top64
    vr, vqr = rep(v[:, lo_:hi_]), rep(vq[:, lo_:hi_])
    o_ex = torch.einsum("hrn,hnd->hrd", pex, vr)
    msk = torch.zeros_like(pex, dtype=torch.bool).scatter_(-1, sel, True)
    N = vr.shape[1]
    nb = N // 256
    vb = vr[:, :nb * 256].reshape(H, nb, 256, d).mean(2).repeat_interleave(256, 1)
    vb = torch.cat([vb, vr[:, nb * 256:].mean(1, keepdim=True).expand(-1, N - nb * 256, -1)], 1)
    for nm, vt in (("v2bit", vqr), ("vblock256", vb)):
        o = torch.einsum("hrn,hnd->hrd", torch.where(msk, pex, 0.0), vr) + torch.einsum("hrn,hnd->hrd", torch.where(msk, 0.0, pex), vt)
        add(f"tail|{nm}_relerr", ((o - o_ex).norm(dim=-1) / o_ex.norm(dim=-1).clamp_min(1e-30)).mean())


def save(path, res):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(res, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)                                                # atomic (review)


def s0_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    li = int(module.layer_idx)
    q, k, v = query[0].float(), key[0].float(), value[0].float()
    scale = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
    if X["cap"]:
        X["kv"][li] = (k.clone(), v.clone(), q.clone())
        X["shape"] = (q.shape[0], k.shape[0], q.shape[-1])
    if X["eval"]:
        with torch.no_grad():
            metrics(q, k, v, li, scale, kw)
    out = Q.attend(q, k, v, li, None, scale, kw)                        # exact attention (Q.CHUNK rows at a time)
    return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None


AttentionInterface.register("s0_attn", s0_attn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--starts", default="4096")
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke != a.out.endswith(".smoke.json"):
        raise SystemExit("smoke output must end with .smoke.json (and only smoke)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    Q.CHUNK["rows"] = a.chunk
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="s0_attn").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    re_ = model.model.rotary_emb
    Q.RC.R["inv_freq"], Q.RC.R["asc"] = re_.inv_freq.detach().float(), float(getattr(re_, "attention_scaling", 1.0))
    if not (all(CQ.CAL[0] != p for _, p in CQ.EVAL) and CQ.CAL[0] != a.text and CQ.CAL[0] != a.jsonl):
        raise SystemExit("calibration text must differ from every input")
    X["cap"] = True
    try:
        with torch.no_grad():
            model(input_ids=CQ.text_ids(tok, CQ.CAL[0], CQ.CAL[1], CQ.CAL[2], "cuda"), use_cache=False)
    finally:
        X["cap"] = False
    for li, t in X["kv"].items():
        Q.calib(li, *t)
    nope = Q.calib_all(model, X["kv"], sorted(X["kv"]))
    X["kv"] = {}
    samples = []
    if a.jsonl:
        for i, line in enumerate(Path(a.jsonl).read_text(encoding="utf-8").splitlines()):
            r = json.loads(line)
            ans = r["answer"] if isinstance(r["answer"], str) else " ".join(map(str, r["answer"]))
            sep = "" if ans.startswith(" ") else " "
            pi = tok(r["input"], return_tensors="pt", add_special_tokens=True).input_ids
            ji = tok(r["input"] + sep + ans, return_tensors="pt", add_special_tokens=True).input_ids   # joint (review)
            if not torch.equal(ji[:, :pi.shape[1]], pi):                  # boundary merge: drop the last prefix token
                pi = pi[:, :-1]
                if not torch.equal(ji[:, :pi.shape[1]], pi):
                    raise RuntimeError(f"sample {i}: prefix tokens change under joint tokenization")
            ids = ji[:, :pi.shape[1] + a.rows]
            if ids.shape[1] <= pi.shape[1]:
                raise RuntimeError(f"sample {i}: empty answer")
            samples.append((r.get("id", f"j{i}"), r.get("task", ""), ids, pi.shape[1]))
    if a.text:
        for st in (int(x) for x in a.starts.split(",")):
            ids = CQ.text_ids(tok, a.text, st, a.seq, "cpu")
            if ids.shape[1] != a.seq:
                raise RuntimeError(f"text window at {st}: {ids.shape[1]} of {a.seq} tokens")
            samples.append((f"text{st}", "text", ids, a.seq - a.rows))
    if a.smoke:
        samples = samples[:2]
    if a.max_samples:
        samples = samples[:a.max_samples]
    H_, KVH, d = X["shape"]
    res = {"model": a.model, "dtype": a.dtype, "rows": a.rows, "smoke": a.smoke, "heads": H_, "kv_heads": KVH,
           "head_dim": d, "nope_layers": nope, "jsonl": a.jsonl,
           "key_bits": {"stk": 2 + 64 / d, "stk_m8": 2 + 32 / d, "stk_sym8": 2 + 24 / d, "stk_u4s": 2 + 64 / d,
                        "stk_lm": 2 + 64 / d, "stk2": 1.5 + 64 / d},     # per key element incl. per-token metadata
           "notes": "mass@M and the value-tail errors are over the region [lo, T0 - 128) only (sinks and window exact)",
           "jsonl_md5": hashlib.md5(Path(a.jsonl).read_bytes()).hexdigest() if a.jsonl else None,
           "md5": {n: hashlib.md5(Path(p).read_bytes()).hexdigest() for n, p in
                   (("s0_cov.py", __file__), ("sel11_q.py", Q.__file__), ("sel10_q.py", S10.__file__))},
           "samples": []}
    t0 = time.time()
    for sid, task, ids, T0 in samples:
        X["acc"], X["T0"], X["R"], X["skip"] = {}, T0, ids.shape[1] - T0, []
        X["eval"] = True
        try:
            with torch.no_grad():
                model.model(input_ids=ids.to("cuda"), use_cache=False)
        finally:
            X["eval"] = False
        rec = {"id": sid, "task": task, "T0": T0, "R": X["R"], "skipped_layers": X["skip"],
               "m": {k: v[0] / max(v[1], 1) for k, v in X["acc"].items()}}
        res["samples"].append(rec)
        print(f"{sid} {task} T0={T0} R={X['R']} ({time.time() - t0:.0f}s) stk id@128 {rec['m'].get('stk|id@128', float('nan')):.3f} "
              f"mass@128 {rec['m'].get('stk|mass@128', float('nan')):.3f}", flush=True)
        save(a.out, res)
    res["seconds"] = time.time() - t0
    save(a.out, res)


if __name__ == "__main__":
    main()
