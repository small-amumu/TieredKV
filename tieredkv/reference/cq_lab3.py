# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""cq_lab: communications-inspired key-cache quantization and an inter-symbol-interference (ISI) view of key error
(predictions registered in ~/kvt_bench/hc_mac/reg_cq.md before non-smoke outputs).

Keys of every full-attention layer are quantized (values exact). Quantizer configs (per token, blocks of 32
consecutive channels of each head, HF channel order):
  q4  : ggml q4_0 rule (signed max / -8, levels -8..7), no rotation
  q4h : q4_0 after a head-wide normalized Hadamard (llama.cpp K rotation)
  a3h : asymmetric min-max INT3 after Hadamard;  a2h : the same at INT2
Arms (applied on top of each config):
  rtn       round to nearest
  p0        rtn with position 0 kept exact
  cen       subtract the static post-RoPE key mean (calibration), quantize, add back; position 0 exact
  eqg, eqs  rtn + receiver-side equalizer: subtract from each logit the Jensen bias of uniform rounding noise,
            Gaussian form sigma^2/2 (eqg) or exact sum_c log(sinh x / x), x = scale |a_c| Delta/2 (eqs);
            a = query in the quantization basis, Delta = per-block step stored with the cache
  cen_eqs   cen + eqs
  clipm     per-block clipping-scale search (gamma in GAMMAS) minimizing block MSE (no side information)
  clipq     the same, minimizing the query-weighted error sum_c w_c e_c^2, w = diag of the calibration query
            second moment in the quantization basis
  cen_clipq cen + clipq
  eden      per-block unbiasing scale x_hat *= ||x||^2 / <x, x_hat> (EDEN), folded into the stored scale
  cen_eden  cen + eden
  bg        Bussgang gain: keys divided by alpha (per layer, KV head) = sum <k, k_hat> / sum ||k||^2 (calibration)
  cp_rtn, cp_cen   "clean prefill" (Tomlinson-Harashima analogue): query rows < P use exact keys, rows >= P
            use the arm's keys; D_late compares to the same arm without cp
Metrics per arm and segment: D (mean KL over positions >= SINK) and D_late (positions >= P).
Diagnostics (one fp forward per segment, local per layer: keys quantized only virtually, fp queries/attention):
  for score error n_tj = scale q_t.(k_hat_j - k_j) - corr_tj (rows t >= SINK): tot = sum p n^2, cm = sum_t nbar_t^2
  (common mode, harmless), isi = tot - cm, out = sum_t ||sum_j p (n - nbar) v_j||^2 (value-projected ISI),
  unw = sum_t mean_{j<=t} n^2 (unweighted score noise), mse = sum ||k_hat - k||^2, z2 = share of key-error energy
  outside the top query subspace (95% of query energy).
Calibration descriptives: pre-/post-RoPE lag-1 correlation of centred keys per channel (DPCM potential),
bits wasted by the offset log2(Delta_raw / Delta_centred), Bussgang alpha.
Usage: python cq_lab.py <model_dir> <out_json> [--dtype ..] [--nseg N] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

DATA = "/home/user/kvt_bench/data/"
CAL = (DATA + "diag_long_text.txt", 40000, 2048)
EVAL = (("qasper", DATA + "qasper_text.txt"), ("mfqa", DATA + "mfqa_text.txt"))
STARTS = (0, 6144, 12288)
SEQ, SINK, P, G = 1024, 4, 512, 32
GAMMAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5)
CFGS = {"q4": ("sym", 4, False), "q4h": ("sym", 4, True), "a3h": ("asym", 3, True), "a2h": ("asym", 2, True)}
ARMS = ("rtn", "p0", "cen", "cen_nop0", "gq", "cen_gq", "cen_gqd", "cen_gq_eden", "pcc", "p0_pcc", "cen_pcc",
        "cen_eqs", "cen_eden", "cen_clipm_eden")
DIAG_ARMS = ("rtn", "cen", "pc", "cen_gq")
S: dict = {"hinv": {}, "hinv_d": {}, "mode": None, "cfg": None, "arm": None, "cs": {}, "m": {}, "sq": {}, "alpha": {}, "w": {}, "U": {},
           "cal": {}, "stats": None, "calls": 0}
HAD: dict = {}


def hadamard(d, dev):
    if d not in HAD:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < d:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        HAD[d] = (h / math.sqrt(d)).to(dev, torch.float32)
    return HAD[d]


def quant_blocks(x, kind, bits, gamma):
    """x [..., nb, G]; gamma broadcastable to [..., nb, 1]. Returns x_hat, step (same leading shape, [...,nb,1])."""
    if kind == "sym":
        lv = 2 ** (bits - 1)
        mx = x.gather(-1, x.abs().argmax(-1, keepdim=True))
        d = gamma * mx / -lv
        safe = torch.where(d != 0, d, torch.ones_like(d))
        xh = torch.where(d != 0, torch.clamp(torch.round(x / safe), -lv, lv - 1) * d, x)
        return xh, d.abs()
    lv = 2 ** bits - 1
    mn, mx = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    c, h = (mn + mx) / 2, (mx - mn) / 2 * gamma
    lo, sc = c - h, 2 * h / lv
    safe = torch.where(sc > 0, sc, torch.ones_like(sc))
    xh = torch.where(sc > 0, torch.clamp(torch.round((x - lo) / safe), 0, lv) * sc + lo, x)
    return xh, sc


def quant_fixed(v, kind, bits, prm):
    """quantize values v [KVH, L] with fixed per-block parameters prm (sym: step d; asym: (lo, sc))."""
    if kind == "sym":
        lv = 2 ** (bits - 1)
        d = prm
        safe = torch.where(d != 0, d, torch.ones_like(d))
        return torch.where(d != 0, torch.clamp(torch.round(v / safe), -lv, lv - 1) * d, v)
    lo, sc = prm
    lv = 2 ** bits - 1
    safe = torch.where(sc > 0, sc, torch.ones_like(sc))
    return torch.where(sc > 0, torch.clamp(torch.round((v - lo) / safe), 0, lv) * sc + lo, v)


def gptq_shape(xb, kind, bits, Hinv):
    """xb [KVH, L, nb, G] in the quantization basis; Hinv [KVH, d, d] upper Cholesky factor of the inverse
    damped query second moment. Sequential channel-wise quantization with error compensation (GPTQ / OBQ)."""
    KVH, L, nb, _ = xb.shape
    d = nb * G
    if kind == "sym":
        lv = 2 ** (bits - 1)
        mx = xb.gather(-1, xb.abs().argmax(-1, keepdim=True))[..., 0]
        prm_all = mx / -lv                                                  # [KVH, L, nb]
        st = prm_all.abs()
    else:
        mn, mx = xb.amin(-1), xb.amax(-1)
        lv = 2 ** bits - 1
        prm_all = (mn, (mx - mn) / lv)
        st = prm_all[1]
    xw = xb.reshape(KVH, L, d).clone()
    out = torch.empty_like(xw)
    for c in range(d):
        b = c // G
        prm = prm_all[:, :, b] if kind == "sym" else (prm_all[0][:, :, b], prm_all[1][:, :, b])
        v = xw[:, :, c]
        qv = quant_fixed(v, kind, bits, prm)
        out[:, :, c] = qv
        if c + 1 < d:
            err = (v - qv) / Hinv[:, c, c][:, None]
            xw[:, :, c + 1:] -= err[..., None] * Hinv[:, c, c + 1:][:, None, :]
    return out.reshape(KVH, L, nb, G), st[..., None]


def key_arm(k, li, cfg, arm):
    """k [KVH, L, d] float32 -> k_hat [KVH, L, d], step [KVH, L, nb] (0 where exact), H (or None)."""
    kind, bits, rot = CFGS[cfg]
    KVH, L, d = k.shape
    base = arm.replace("cp_", "")
    tk = set(base.split("_"))
    cen = "cen" in tk
    sub = S["m"][li][:, None, :] if cen else 0.0
    x = k - sub
    H = hadamard(d, k.device) if rot else None
    if rot:
        x = x @ H
    xb = x.reshape(KVH, L, d // G, G)
    if tk & {"pc", "pcc"}:
        # per-channel grouping along tokens (KIVI-style keys): blocks of G consecutive tokens of one channel
        xt = x.transpose(1, 2).reshape(KVH, d, L // G, G)
        xh, st = quant_blocks(xt, kind, bits, 1.0)
        xh = xh.reshape(KVH, d, L).transpose(1, 2).reshape(KVH, L, d // G, G)
        st = st.reshape(KVH, d, L // G, 1).repeat_interleave(G, 2)[..., 0].transpose(1, 2)  # [KVH, L, d]
        st = st.reshape(KVH, L, d // G, G).amax(-1, keepdim=True)
    elif tk & {"gq", "gqd"}:
        xh, st = gptq_shape(xb, kind, bits, S["hinv_d" if "gqd" in tk else "hinv"][(li, cfg)])
    elif tk & {"clipm", "clipq"}:
        w = torch.ones(KVH, 1, d // G, G, device=k.device) if "clipm" in tk else S["w"][(li, cfg)][:, None]
        best, bstep, berr = None, None, None
        for gm in GAMMAS:
            xh, st = quant_blocks(xb, kind, bits, gm)
            err = (w * (xh - xb) ** 2).sum(-1, keepdim=True)
            if best is None:
                best, bstep, berr = xh, st, err
            else:
                better = err < berr
                best = torch.where(better, xh, best)
                bstep = torch.where(better, st, bstep)
                berr = torch.where(better, err, berr)
        xh, st = best, bstep
    else:
        xh, st = quant_blocks(xb, kind, bits, 1.0)
    if "eden" in tk:
        num, den = (xb * xb).sum(-1, keepdim=True), (xb * xh).sum(-1, keepdim=True)
        sc = torch.where(den > 0, num / den.clamp_min(1e-30), torch.ones_like(den))
        xh, st = xh * sc, st * sc
    xh = xh.reshape(KVH, L, d)
    st = st.reshape(KVH, L, d // G)
    if rot:
        xh = xh @ H.T
    kh = xh + sub
    if "bg" in tk:
        a_ = S["alpha"][(li, cfg)][:, None, None]
        kh = kh / a_
        st = st / a_
    if (cen or "p0" in tk) and "nop0" not in tk:
        kh = kh.clone()
        kh[:, 0] = k[:, 0]
        st = st.clone()
        st[:, 0] = 0
    return kh, st, H


def lsinhc(x):
    small = x < 1e-2
    xs = torch.where(small, torch.ones_like(x), x)
    big = xs + torch.log1p(-torch.exp(-2 * xs)) - math.log(2) - torch.log(xs)
    return torch.where(small, x * x / 6, big)


def eq_corr(q, st, H, scale, g, exact):
    """q [H, L, d] query heads; st [KVH, L, nb]; returns corr [H, Lq, Lk]."""
    a = q @ H if H is not None else q
    stH = st.repeat_interleave(g, 0)                                      # [H, L, nb]
    Hh, L, d = a.shape
    if not exact:
        a2 = (a * a).reshape(Hh, L, d // G, G).sum(-1)                    # [H, L, nb]
        return 0.5 * scale * scale * torch.einsum("htb,hjb->htj", a2, stH * stH) / 12
    ste = stH.repeat_interleave(G, -1)                                    # [H, L, d]
    out = torch.empty(Hh, L, L, device=q.device)
    A = a.abs() * (0.5 * scale)
    for t0 in range(0, L, 16):
        x = A[:, t0:t0 + 16, None, :] * ste[:, None, :, :]
        out[:, t0:t0 + 16] = lsinhc(x).sum(-1)
    return out


def allowed_mask(Lq, Lk, attention_mask, kw, dev):
    qpos = torch.arange(Lk - Lq, Lk, device=dev)[:, None]
    kpos = torch.arange(Lk, device=dev)[None, :]
    al = kpos <= qpos
    if kw.get("sliding_window"):
        al &= kpos > qpos - int(kw["sliding_window"])
    if isinstance(attention_mask, torch.Tensor):
        m = attention_mask[0, 0, -Lq:, -Lk:] if attention_mask.dim() == 4 else attention_mask[-Lq:, -Lk:]
        al &= m if m.dtype == torch.bool else (m > -1e4)
    return al


def scores(q, k, scale, softcap):
    s = torch.einsum("hqd,hkd->hqk", q, k) * scale
    return s


def finish(s, al, softcap):
    if softcap:
        s = torch.tanh(s / softcap) * softcap
    return torch.softmax(s.masked_fill(~al, float("-inf")), -1)


def cq_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    li = int(module.layer_idx)
    q, k, v = query[0].float(), key[0].float(), value[0].float()           # [H,L,d], [KVH,L,d]
    H_, KVH, L, d = q.shape[0], k.shape[0], q.shape[1], q.shape[-1]
    g = H_ // KVH
    scale = scaling if scaling is not None else 1 / math.sqrt(d)
    softcap = kw.get("softcap")
    al = allowed_mask(L, k.shape[1], attention_mask, kw, q.device)
    kx = k.repeat_interleave(g, 0)
    vx = v.repeat_interleave(g, 0)
    mode = S["mode"]
    if mode == "cal":
        S["cal"][li] = (k.clone(), q.clone())
    if mode in ("arm",) and S["arm"] is not None:
        cfg, arm = S["cfg"], S["arm"]
        kh, st, Hm = key_arm(k, li, cfg, arm)
        s = scores(q, kh.repeat_interleave(g, 0), scale, softcap)
        if "eqs" in arm or "eqg" in arm:
            s = s - eq_corr(q, st, Hm, scale, g, exact="eqg" not in arm)
        if "pcc" in arm:
            s0 = scores(q, kx, scale, softcap)
            qp = torch.arange(L, device=q.device)[:, None]
            kp_ = torch.arange(k.shape[1], device=q.device)[None, :]
            own = kp_ >= (qp // G) * G
            s = torch.where(own[None], s0, s)
        if arm.startswith("cp_"):
            s0 = scores(q, kx, scale, softcap)
            rows = torch.arange(L, device=q.device)[None, :, None] < P
            s = torch.where(rows, s0, s)
        p = finish(s, al, softcap)
        S["calls"] += 1
    else:
        p = finish(scores(q, kx, scale, softcap), al, softcap)
        if mode == "diag":
            diag_layer(li, q, k, vx, p, scale, g)
    o = torch.einsum("hqk,hkd->hqd", p, vx)
    return o.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None


AttentionInterface.register("cq_attn", cq_attn)


def diag_layer(li, q, k, vx, p, scale, g):
    pr = p[:, SINK:]                                                       # rows t >= SINK
    cnt = (pr > 0).sum(-1).clamp_min(1)
    for cfg in CFGS:
        for arm in DIAG_ARMS:
            kh, st, Hm = key_arm(k, li, cfg, arm)
            e = kh - k
            n = torch.einsum("hqd,hkd->hqk", q[:, SINK:], e.repeat_interleave(g, 0)) * scale
            if "eqs" in arm or "eqg" in arm:
                n = n - eq_corr(q, st, Hm, scale, g, exact="eqg" not in arm)[:, SINK:]
            n = torch.where(pr > 0, n, torch.zeros_like(n))
            nbar = (pr * n).sum(-1)
            tot = float((pr * n * n).sum())
            cm = float((nbar * nbar).sum())
            dv = torch.einsum("hqk,hkd->hqd", pr * (n - nbar[..., None]), vx)
            U = S["U"][li]                                                 # [KVH, d, r] top query subspace
            ee = float((e * e).sum())
            inside = float((torch.einsum("hld,hdr->hlr", e, U) ** 2).sum())
            rec = S["stats"].setdefault(f"{cfg}|{arm}", {"tot": 0.0, "cm": 0.0, "isi": 0.0, "out": 0.0, "unw": 0.0,
                                                        "mse": 0.0, "z2_num": 0.0, "z2_den": 0.0})
            rec["tot"] += tot
            rec["cm"] += cm
            rec["isi"] += tot - cm
            rec["out"] += float((dv * dv).sum())
            rec["unw"] += float(((n * n).sum(-1) / cnt).sum())
            rec["mse"] += ee
            rec["z2_num"] += ee - inside
            rec["z2_den"] += ee


def cs_hook(mod, args, kwargs):
    pe = kwargs.get("position_embeddings")
    if pe is not None:
        S["cs"][int(mod.layer_idx)] = (pe[0][0].float(), pe[1][0].float())


def rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def text_ids(tok, path, start, n, dev):
    ids = tok(Path(path).read_text(encoding="utf-8", errors="replace"), return_tensors="pt",
              add_special_tokens=False).input_ids[0]
    b = tok.bos_token_id
    bos = b is not None and tok("x", add_special_tokens=True).input_ids[:1] == [b]
    w = ids[start: start + n - (1 if bos else 0)]
    if bos:
        w = torch.cat([torch.tensor([b]), w])
    return w.unsqueeze(0).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--nseg", type=int, default=3)
    ap.add_argument("--starts", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke != a.out.endswith(".smoke.json"):
        raise SystemExit("smoke output must end with .smoke.json (and only smoke)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = "cuda"
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="cq_attn").to(dev).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    for m in model.modules():
        if getattr(m, "layer_idx", None) is not None and m.__class__.__name__.endswith("Attention") \
                and "position_embeddings" in m.forward.__code__.co_varnames:
            m.register_forward_pre_hook(cs_hook, with_kwargs=True)

    def logits(ids):
        with torch.no_grad():
            return model(input_ids=ids, use_cache=False).logits[0].float()

    def rnd(shape, seed):
        return torch.randint(100, tok.vocab_size - 100, shape, generator=torch.Generator().manual_seed(seed)).to(dev)

    # calibration
    cal = text_ids(tok, CAL[0], CAL[1], CAL[2], dev)
    if a.smoke:
        cal = rnd(cal.shape, 7)
    S["mode"], S["cal"] = "cal", {}
    logits(cal)
    S["mode"] = None
    calib = {}
    for li, (k, q) in S["cal"].items():
        KVH, L, d = k.shape
        g = q.shape[0] // KVH
        S["m"][li] = k[:, SINK:].mean(1)
        qg = q.reshape(KVH, g, L, d)
        sq = torch.einsum("hgld,hgle->hde", qg, qg) / (g * L)            # [KVH, d, d] query second moment
        ev, U = torch.linalg.eigh(sq.double())
        r = []
        for h in range(KVH):
            c = torch.cumsum(ev[h].flip(0), 0) / ev[h].sum()
            r.append(int((c < 0.95).sum()) + 1)
        S["U"][li] = torch.stack([U[h][:, -max(r):] for h in range(KVH)]).float()
        rec = {"q_rank95": r}
        for cfg, (kind, bits, rot) in CFGS.items():
            Hm = hadamard(d, dev) if rot else torch.eye(d, device=dev)
            S["w"][(li, cfg)] = torch.diagonal(Hm.T @ sq @ Hm, dim1=-2, dim2=-1).reshape(KVH, d // G, G)
            hq = (Hm.T @ sq @ Hm).double()
            hq = hq + 1e-2 * torch.diagonal(hq, dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(d, device=dev, dtype=torch.float64)
            S["hinv"][(li, cfg)] = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hq)), upper=True).float()
            hd = (Hm.T @ sq @ Hm).double()
            hd = hd + 1.0 * torch.diagonal(hd, dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(d, device=dev, dtype=torch.float64)
            S["hinv_d"][(li, cfg)] = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hd)), upper=True).float()
            kh, st, _ = key_arm(k, li, cfg, "rtn")
            num = (k[:, SINK:] * kh[:, SINK:]).sum((1, 2))
            den = (k[:, SINK:] ** 2).sum((1, 2))
            S["alpha"][(li, cfg)] = (num / den).clamp(0.5, 2.0)
            _, stc, _ = key_arm(k, li, cfg, "cen")
            ratio = (st[:, SINK:].clamp_min(1e-12) / stc[:, SINK:].clamp_min(1e-12)).log2()
            r_ = ratio.flatten()
            e_raw = float((st[:, SINK:] ** 2).mean())
            e_cen = float((stc[:, SINK:] ** 2).mean())
            rec[cfg] = {"alpha": [float(x) for x in S["alpha"][(li, cfg)]], "bits_wasted": float(ratio.mean()),
                        "bw_p50": float(r_.quantile(0.5)), "bw_p90": float(r_.quantile(0.9)),
                        "bw_p99": float(r_.quantile(0.99)),
                        "energy_bits": 0.5 * math.log2(max(e_raw, 1e-30) / max(e_cen, 1e-30))}
        # lag-1 correlation of centred keys, post-RoPE and pre-RoPE
        cos, sin = S["cs"][li]
        rd = cos.shape[-1]
        kp = k.clone()
        kr = kp[..., :rd]
        kp[..., :rd] = (kr * cos - rot_half(kr) * sin) / (cos * cos + sin * sin)
        for tag, kk in (("post", k), ("pre", kp)):
            x = kk[:, SINK:] - kk[:, SINK:].mean(1, keepdim=True)
            num = (x[:, 1:] * x[:, :-1]).sum(1)
            den = (x * x).sum(1).clamp_min(1e-30)
            rho = (num / den).clamp(-0.999, 0.999)                       # [KVH, d]
            en = den / den.sum(-1, keepdim=True)
            gain = -10 * torch.log10(1 - rho.clamp_min(0) ** 2)
            rec[f"rho_{tag}"] = {"energy_weighted_rho": float((en * rho).sum(-1).mean()),
                                 "energy_share_rho_gt_0.9": float((en * (rho > 0.9)).sum(-1).mean()),
                                 "energy_weighted_dpcm_gain_db": float((en * gain).sum(-1).mean())}
        calib[str(li)] = rec
    S["cal"] = {}
    res = {"model": a.model, "dtype": a.dtype, "smoke": a.smoke, "md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
           "calib": calib, "seg": []}
    t0 = time.time()
    starts = tuple(int(x) for x in a.starts.split(",")) if a.starts else STARTS[: a.nseg]
    segs = [(n, st) for st in starts for n, _ in EVAL][: (1 if a.smoke else None)]
    for name, st in segs:
        path = dict(EVAL)[name]
        ids = rnd((1, SEQ), 11 + st) if a.smoke else text_ids(tok, path, st, SEQ, dev)
        z0 = logits(ids)
        lr0 = torch.log_softmax(z0, -1)

        def kl(z, lo):
            ln = torch.log_softmax(z, -1)
            return float((lr0[lo:].exp() * (lr0[lo:] - ln[lo:])).sum(-1).mean())
        rec = {"doc": name, "start": st, "D": {}, "D_late": {}}
        for cfg in CFGS:
            for arm in ARMS:
                S["mode"], S["cfg"], S["arm"], S["calls"] = "arm", cfg, arm, 0
                try:
                    z = logits(ids)
                finally:
                    S["mode"], S["arm"] = None, None
                if S["calls"] == 0:
                    raise RuntimeError("no attention call")
                rec["D"][f"{cfg}|{arm}"] = kl(z, SINK)
                rec["D_late"][f"{cfg}|{arm}"] = kl(z, P)
        S["mode"], S["stats"] = "diag", {}
        try:
            logits(ids)
        finally:
            S["mode"] = None
        rec["diag"] = S["stats"]
        res["seg"].append(rec)
        print(f"{name} {st} ({time.time() - t0:.0f}s) " + " ".join(f"{k} {v:.4g}" for k, v in rec["D"].items()
                                                                  if k.endswith(("|rtn", "|cen", "|cen_gq", "|pcc", "|cen_pcc", "|cen_clipm_eden"))),
              flush=True)
        Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    res["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
