# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""pkv8_q: round P8 of the layered-KV line (registered in reg_pkv.md): reverse water-filling of key bits
over RoPE frequency pairs, with the pair importance taken from calibration data, from the weights alone, or at random,
plus the weight-only statistics used to predict how hard a model's keys are to quantize. Everything else (bases, fetch,
selection, bits) is pkv7_q, used read-only; method names of pkv7_q pass through unchanged.
  wf<src>[rest]  keys: centred and channel-balanced as bal128, then the RoPE pairs (rotate_half pairing (i, i + r/2) inside
                 the rotary dims r, interleaved pairs for GLM, adjacent pairs outside r) are ranked by importance
                 I_p = (sum_{c in p} E q_c^2) (sum_{c in p} Var k_c); the top half of the channels is stored at 3 bits
                 (asym min-max) and the bottom half at 1 bit (mean +- mean absolute deviation; min-max 1-bit levels were
                 found harmful in the smoke), each class per token with its own Hadamard (size d/2) and two FP16 numbers;
                 token 0 exact. Values and everything in [rest] (fetch, sinks, selection) exactly as bal128[rest].
                 src: c = calibration (E q^2 over the query heads of the group, Var k of centred keys)
                      w = weights only (E q_c^2 ~ sum_h ||W_q[h, c] * gamma||^2 (x q_norm^2), Var k_c ~ ||W_k[c] * gamma||^2
                          (x k_norm^2); gamma = the input norm of the layer, 1 + w for zero-centred norms)
                      r = random pair ranking (seeded per layer): the same 3 / 1-bit split without importance
Key bits of wf: 2 + 64 / d (two pairs of FP16 numbers per token); values as bal (2 + 32 / min(128, d)).
"""
from __future__ import annotations

import math

import torch

import pkv5_q as Q5
import pkv7_q as Q7

SINK = Q5.SINK
C8: dict = {"c": {}, "w": {}, "r": {}, "pairs": {}, "wstat": {}}
WF: dict = {"src": None}
NAMES = ("bal128", "bal64", "kivi128_z4", "wfc", "wfw", "wfr", "bal128k112_k8_v8", "bal64k112_k8_v8",
         "kivi128k112_k8_v8_z4", "wfck112_k8_v8", "wfwk112_k8_v8", "wfrk112_k8_v8")


def norm_gain(w):
    """gain vector of an RMSNorm weight: zero-centred norms (Gemma, Qwen3.5 / Qwen3-Next) multiply by 1 + w."""
    w = w.float()
    return 1 + w if w.abs().median() < 0.3 else w


def pairs_of(d, rot, interleaved):
    """list of channel pairs: RoPE pairs inside the rotary dims, adjacent pairs outside."""
    if interleaved:
        P = [(2 * i, 2 * i + 1) for i in range(rot // 2)]
    else:
        P = [(i, i + rot // 2) for i in range(rot // 2)]
    P += [(j, j + 1) for j in range(rot, d - 1, 2)]
    return P


def rope_info(model, d):
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    rp = getattr(cfg, "rope_parameters", None) or {}
    theta = getattr(cfg, "rope_theta", None) or rp.get("rope_theta") or 10000.0
    prf = getattr(cfg, "partial_rotary_factor", None) or rp.get("partial_rotary_factor") or 1.0
    rot = int(d * prf) // 2 * 2
    inter = "glm" in str(getattr(cfg, "model_type", "")).lower()
    return float(theta), rot, inter


def attn_layers(model):
    """yields (layer_idx, attention module, input-norm gain or None) for every full-attention layer."""
    for mod in model.modules():
        a = getattr(mod, "self_attn", None)
        if a is None or not hasattr(a, "layer_idx"):
            continue
        ln = getattr(mod, "input_layernorm", None)
        yield int(a.layer_idx), a, (norm_gain(ln.weight) if ln is not None and getattr(ln, "weight", None) is not None else None)


def qk_weights(a, H, KVH, d):
    """W_q [H, d, D], W_k [KVH, d, D], b_k [KVH, d] or None, q_norm / k_norm gains [d] or None."""
    if hasattr(a, "q_proj"):
        wq, wk = a.q_proj.weight.float(), a.k_proj.weight.float()
        bk = a.k_proj.bias.float() if getattr(a.k_proj, "bias", None) is not None else None
    else:
        w = a.qkv_proj.weight.float()
        wq, wk = w[: H * d], w[H * d: H * d + KVH * d]
        bk = None
        if getattr(a.qkv_proj, "bias", None) is not None:
            bk = a.qkv_proj.bias.float()[H * d: H * d + KVH * d]
    D = wq.shape[1]
    if wq.shape[0] == 2 * H * d:
        # gated attention (Qwen3.5 / Qwen3-Next): q_proj writes [query | gate] per head
        wq = wq.reshape(H, 2 * d, D)[:, :d].reshape(H * d, D)
    gq = norm_gain(a.q_norm.weight) if getattr(a, "q_norm", None) is not None else None
    gk = norm_gain(a.k_norm.weight) if getattr(a, "k_norm", None) is not None else None
    return wq.reshape(H, d, D), wk.reshape(KVH, d, D), (bk.reshape(KVH, d) if bk is not None else None), gq, gk


def calib8(model, kv, li_list):
    """kv[li] = (k [KVH, Lc, d], v, q [H, Lc, d]) from the calibration text; fills C8 for every attention layer."""
    layers = {li: (a, g) for li, a, g in attn_layers(model)}
    for li in li_list:
        k, v, q = kv[li]
        KVH, d = k.shape[0], k.shape[-1]
        H = q.shape[0]
        g = H // KVH
        theta, rot, inter = rope_info(model, d)
        P = pairs_of(d, rot, inter)
        C8["pairs"][li] = P
        kc = (k - Q5.C["m"][li][:, None, :])[:, SINK:]
        qe = (q[:, SINK:] ** 2).mean(1).reshape(KVH, g, d).sum(1)
        idx = torch.tensor(P, device=k.device)

        def rank2(qf, kf):
            pi = (qf[:, idx[:, 0]] + qf[:, idx[:, 1]]) * (kf[:, idx[:, 0]] + kf[:, idx[:, 1]])
            return idx[pi.argsort(-1, descending=True)].reshape(KVH, -1)
        C8["c"][li] = rank2(qe, kc.var(1))
        gen = torch.Generator().manual_seed(1009 * li + 7)
        C8["r"][li] = torch.stack([idx[torch.randperm(len(P), generator=gen)].reshape(-1) for _ in range(KVH)]).to(k.device)
        if li in layers:
            a, gam = layers[li]
            wq, wk, bk, gq, gk = qk_weights(a, H, KVH, d)
            gam = gam.to(wq.device) if gam is not None else torch.ones(wq.shape[-1], device=wq.device)
            qw = ((wq * gam) ** 2).sum(-1).reshape(KVH, g, d).sum(1)
            kw = ((wk * gam) ** 2).sum(-1)
            if gq is not None:
                qw = qw * gq.to(qw.device) ** 2
            if gk is not None:
                kw = kw * gk.to(kw.device) ** 2
            C8["w"][li] = rank2(qw.to(k.device), kw.to(k.device))
        else:
            C8["w"][li] = C8["r"][li]
            C8["wstat"].setdefault("missing", []).append(li)


def sign1(x):
    """1-bit per token over the last dim: mu + s * sign(x - mu), mu the mean and s the mean absolute deviation (both FP16);
    the MSE-optimal 1-bit levels for a symmetric distribution, unlike min-max levels, which push values near the mean
    out to the extremes."""
    mu = Q5.f16(x.mean(-1, keepdim=True))
    s = Q5.f16((x - mu).abs().mean(-1, keepdim=True))
    return mu + s * torch.where(x >= mu, 1.0, -1.0)


def wf_keys(k, li, src, a):
    KVH, L, d = k.shape
    m = Q5.C["m"][li][:, None, :]
    b = (Q7.C7["bal"][li] ** (a / 100))[:, None, :]
    x = (k - m) * b
    order = C8[src][li][:, None, :].expand(KVH, L, d)
    xs = x.gather(-1, order)
    h = d // 2
    Hm = Q5.CQ.hadamard(h, k.device)
    hi = Q5.per_token(xs[..., :h] @ Hm, 3, h) @ Hm.T
    lo = sign1(xs[..., h:] @ Hm) @ Hm.T
    xq = torch.empty_like(x).scatter_(-1, order, torch.cat([hi, lo], -1))
    return xq / b + m


_base7 = Q7.base7


def _base8(p, k, v, li):
    kq, vq, R, z0 = _base7(p, k, v, li)
    if WF["src"] is not None:
        kq = wf_keys(k, li, WF["src"], p["a"])
        kq[:, :1] = k[:, :1]
    return kq, vq, R, z0


Q7.base7 = _base8


def split(name):
    """wf<src><rest> -> (src, pkv7 name with the bal128 base) ; other names -> (None, name)."""
    if name and name.startswith("wf"):
        return name[2], "bal128" + name[3:]
    return None, name


def parse(name):
    src, n7 = split(name)
    if src is not None and src not in "cwr":
        raise ValueError(name)
    return Q7.parse(n7)


def eff_bits(name, L, d, g=1):
    src, n7 = split(name)
    gpu, read = Q7.eff_bits(n7, L, d, g)
    if src is None:
        return gpu, read
    bk_bal = 2 + 32 / min(128, d)
    bk_wf = 2 + 64 / d
    w = min(1.0, (Q7.parse(n7)["R"] / 2 + 1) / L)
    delta = (bk_wf - bk_bal) * (1 - w) / 2
    return gpu + delta, read + delta


def attend(q, k, v, li, name, scale, kw):
    src, n7 = split(name) if name else (None, None)
    WF["src"] = src
    try:
        return Q7.attend(q, k, v, li, n7, scale, kw)
    finally:
        WF["src"] = None


def weight_stats(model, H, KVH, d, L=4096):
    """weight-only statistics per full-attention layer (medians over KV heads), for predicting key quantization
    difficulty: rb = ||b_k|| / ||W_k * gamma||_F (key bias size relative to the key projection), rbf = the same with only
    the RoPE pairs whose wavelength 2 pi / omega is shorter than L (the bias part that rotates within the context and so
    survives static centring), conc = max / median channel norm of W_k * gamma (x k_norm), align = Spearman over channels
    between query energy and key energy."""
    theta, rot, inter = rope_info(model, d)
    P = pairs_of(d, rot, inter)
    om = [theta ** (-2 * i / rot) for i in range(rot // 2)]
    fast =torch.tensor([2 * math.pi / w < L for w in om] + [False] * (len(P) - rot // 2))
    out = []
    for li, a, gam in attn_layers(model):
        wq, wk, bk, gq, gk = qk_weights(a, H, KVH, d)
        gam = gam if gam is not None else torch.ones(wq.shape[-1])
        kn = ((wk * gam) ** 2).sum(-1)
        qn = ((wq * gam) ** 2).sum(-1).reshape(KVH, H // KVH, d).sum(1)
        if gk is not None:
            kn = kn * gk.float() ** 2
        if gq is not None:
            qn = qn * gq.float() ** 2
        scale = kn.sum(-1).sqrt()
        rec = {"layer": li, "conc": float((kn.sqrt().amax(-1) / kn.sqrt().median(-1).values).median())}
        if bk is not None and gk is None:
            idx = torch.tensor(P)
            bp = bk[:, idx[:, 0]] ** 2 + bk[:, idx[:, 1]] ** 2
            rec["rb"] = float((bk.norm(dim=-1) / scale).median())
            rec["rbf"] = float(((bp * fast).sum(-1).sqrt() / scale).median())
        else:
            rec["rb"] = rec["rbf"] = 0.0
        rq, rk = qn.argsort(-1).argsort(-1).float(), kn.argsort(-1).argsort(-1).float()
        rq, rk = rq - rq.mean(-1, keepdim=True), rk - rk.mean(-1, keepdim=True)
        rec["align"] = float(((rq * rk).sum(-1) / (rq.norm(dim=-1) * rk.norm(dim=-1))).median())
        out.append(rec)
    return {"theta": theta, "rot": rot, "interleaved": inter, "layers": out}
calib = Q7.calib
