# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""pkv7_q: round P7 of the layered-KV line (registered in reg_pkv.md): which keys to refine (selection
criteria, keep vs drop regime) and per-token base fixes. Quantizer primitives from pkv5_q, residual quantizer and the
MSE-clip quantizer from pkv6_q (read-only imports). A method name is
  <base><G>[k<K>][_k<kb>][_v<vb>][_s<share>][_z<z>][_o<N>][_c<clip>][_x<crit>][_d<drop>][_a<exp>][_w<R>]
  base   kivi (per-channel keys over G-token groups, per-token values, FP16 residual window R = 128, streaming semantics)
         | ours (centred + Hadamard per-token keys) | bal (ours with static channel balance b = (max|q| / max|k - m|)^(a/100),
           queries divided by b) | ol (ours with N FP16 outlier channels); values of ours / bal / ol: Hadamard per-token INT2;
           ours / bal / ol keep token 0 exact
  K, kb, vb, share, z, N as in pkv6_q; share adds sg2 (query heads split into sub-groups of 2, each sub-group its own top-K
         by its mean base attention: ceil(g/2) K rows per KV head) and sm2 (one shared top-(K ceil(g/2)) by the group mean:
         the same rows read)
  clip   0 (min-max) | 1 (per-group MSE clipping of keys and values, as pkv6_q) | k (MSE clipping of keys only) | w (keys
         only, clipping ratio chosen by the query-energy-weighted error sum_c E[(q_rot)_c^2] e_c^2, energy from calibration)
  crit   selection criterion, all averaged over the query heads that share a set: qs (base attention, default) | or (exact
         attention) | nz (exact scores plus Gaussian noise with the per-row std of base - exact scores) | h2 (accumulated
         exact attention over all earlier query rows, H2O) | zc (that sum divided by the number of rows that attended to the
         key, ZipCache-style normalised attention) | sn (exact attention summed over the last 32 query rows, SnapKV-style
         observation window) | rd (random) | rc (most recent)
  drop   0 (keep: unselected keys stay at the base) | 1 (drop: only selected keys, the exact window and exact sinks attend)
  exp    balance exponent in percent (default 50) | R: FP16 recent window for ours / bal / ol (default 0)
eff_bits(name, L, d, g) returns (GPU bits per element, bits read per element per decode step) with pkv5_q conventions.
"""
from __future__ import annotations

import math
import re

import torch

import pkv5_q as Q5
import pkv6_q as Q6

SINK = Q5.SINK
FB = Q6.FB
C7: dict = {"bal": {}, "qe_bal": {}, "qe_ours": {}}
OPT = re.compile(r"_([kvszocxdaw])([a-z0-9]+)")
CRITS = ("qs", "or", "nz", "h2", "zc", "sn", "rd", "rc")
NAMES = (
    "kivi128", "kivi128k64", "kivi128k112_k8_v8", "kivi128_z4",
    "kivi128k64_z4", "kivi128k64_z4_xor", "kivi128k64_z4_xnz", "kivi128k64_z4_xh2", "kivi128k64_z4_xzc",
    "kivi128k64_z4_xsn", "kivi128k64_z4_xrd", "kivi128k64_z4_xrc",
    "kivi128k64_z4_d1", "kivi128k64_z4_xor_d1", "kivi128k64_z4_xh2_d1", "kivi128k256_z4_d1", "kivi128k256_z4_xor_d1",
    "kivi128k256_z4", "kivi128k256_z4_xor", "kivi128k112_k8_v8_z4", "kivi128k112_k8_v8_z4_xor",
    "kivi128k64_z4_ssg2", "kivi128k64_z4_ssm2",
    "bal128", "bal128k64", "bal128k112_k8_v8", "bal128_w128", "bal128k64_w128", "bal128k112_k8_v8_w128",
    "bal128_ck", "bal128k64_ck", "bal128k112_k8_v8_ck", "bal128_cw", "bal128k64_cw", "bal128_a25", "bal128_a75")


def parse(name):
    m = re.fullmatch(r"(kivi|ours|bal|ol)(\d+)(?:k(\d+))?((?:_[kvszocxdaw][a-z0-9]+)*)", name)
    if not m:
        raise ValueError(name)
    o = {"k": "16", "v": "16", "s": "mean", "z": "0", "o": "4", "c": "0", "x": "qs", "d": "0", "a": "50", "w": "0"}
    for key, val in OPT.findall(m[4]):
        o[key] = val
    kb, vb = int(o["k"]), int(o["v"])
    ok = (kb in (16, 8, 4) and vb in FB and o["s"] in ("mean", "max", "uni", "q", "sg2", "sm2") and o["c"] in ("0", "1", "k", "w")
          and o["x"] in CRITS and o["d"] in ("0", "1") and not (m[1] == "kivi" and (o["c"] != "0" or o["w"] != "0")))
    if not ok:
        raise ValueError(name)
    return dict(base=m[1], G=int(m[2]), K=int(m[3] or 0), kb=kb, vb=vb, share=o["s"], z=int(o["z"]), N=int(o["o"]),
                clip=o["c"], crit=o["x"], drop=o["d"] == "1", a=int(o["a"]), R=int(o["w"]))


def rows_read(p, g):
    """fetched key rows per KV head per decode step."""
    if p["share"] == "q":
        return p["K"] * g
    if p["share"] in ("sg2", "sm2"):
        return p["K"] * math.ceil(g / 2)
    return p["K"]


def eff_bits(name, L, d, g=1):
    p = parse(name)
    if p["base"] == "kivi":
        gpu, _ = Q5.eff_bits(f"kivi{p['G']}", L, d)
    else:
        b = 2 + 32 / min(p["G"], d)
        if p["base"] == "ol":
            b += p["N"] * (16 - 2) / d / 2
        w = min(1.0, (p["R"] / 2 + 1) / L)
        gpu = b * (1 - w) + 16 * w
    gpu += (16 - gpu) * p["z"] / L
    return gpu, gpu + rows_read(p, g) * (FB[p["kb"]] + FB[p["vb"]]) / 2 / L


def calib(li, k, v, q):
    """k, v [KVH, Lc, d], q [H, Lc, d] float32 from the calibration text."""
    Q6.calib(li, k, v, q)
    KVH, d = k.shape[0], k.shape[-1]
    kc = (k - Q5.C["m"][li][:, None, :])[:, SINK:]
    qm = q[:, SINK:].reshape(KVH, -1, d).abs().amax(1)
    C7["bal"][li] = (qm / kc.abs().amax(1).clamp_min(1e-6)).clamp_min(1e-6)  # ratio; exponent applied per method
    H = Q5.CQ.hadamard(d, k.device)
    qq = q[:, SINK:].reshape(KVH, -1, q.shape[1] - SINK, d)
    C7["qe_ours"][li] = ((qq @ H) ** 2).mean((1, 2))
    C7["qe_bal"][li] = {}
    for a in (25, 50, 75):
        b = C7["bal"][li] ** (a / 100)
        C7["qe_bal"][li][a] = (((qq / b[:, None, None, :]) @ H) ** 2).mean((1, 2))


def clip_w(x, bits, G, w):
    """per-token asym INT with the clipping ratio per group chosen by the weighted error sum_c w_c e_c^2; w [KVH, d]."""
    *lead, d = x.shape
    G = min(G, d)
    xb = x.reshape(*lead, d // G, G)
    wb = w.reshape(w.shape[0], 1, d // G, G)
    lv = 2 ** bits - 1
    mn, mx = xb.amin(-1, keepdim=True), xb.amax(-1, keepdim=True)
    c, h = (mn + mx) / 2, (mx - mn) / 2
    best, berr = None, None
    for gm in Q6.GAMMAS:
        lo, sc = Q5.f16(c - h * gm), Q5.f16(2 * h * gm / lv)
        safe = torch.where(sc > 0, sc, torch.ones_like(sc))
        xh = torch.where(sc > 0, torch.clamp(torch.round((xb - lo) / safe), 0, lv) * sc + lo, xb)
        err = (wb * (xh - xb) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, berr = xh, err
        else:
            bt = err < berr
            best, berr = torch.where(bt, xh, best), torch.where(bt, err, berr)
    return best.reshape(x.shape)


def base7(p, k, v, li):
    """-> k_hat, v_hat, residual window R, number of leading tokens exact in the base."""
    G = p["G"]
    if p["base"] == "kivi":
        kq, vq, R, _ = Q5.base(f"kivi{G}", k, v, li)
        return kq, vq, R, 0
    clip = p["clip"]
    m = Q5.C["m"][li][:, None, :]
    H = Q5.CQ.hadamard(k.shape[-1], k.device)
    ptv = Q6.per_token_clip if clip == "1" else Q5.per_token
    vq = ptv(v @ H, 2, G) @ H.T
    if clip in ("1", "k"):
        ptk = Q6.per_token_clip
    elif clip == "w":
        qe = C7["qe_bal"][li][p["a"]] if p["base"] == "bal" else C7["qe_ours"][li]
        ptk = lambda x, bits, g_: clip_w(x, bits, g_, qe)  # noqa: E731
    else:
        ptk = Q5.per_token
    if p["base"] == "ours":
        kq = ptk((k - m) @ H, 2, G) @ H.T + m
    elif p["base"] == "bal":
        b = (C7["bal"][li] ** (p["a"] / 100))[:, None, :]
        kq = (ptk(((k - m) * b) @ H, 2, G) @ H.T) / b + m
    else:
        idx = Q6.C6["ol"][li].topk(p["N"], -1).indices
        om = torch.zeros_like(Q6.C6["ol"][li], dtype=torch.bool).scatter_(-1, idx, True)[:, None, :]
        kq = torch.where(om, k, ptk((k - m).masked_fill(om, 0.0) @ H, 2, G) @ H.T + m)
    kq[:, :1], vq[:, :1] = k[:, :1], v[:, :1]
    return kq, vq, p["R"], 1


def attend(q, k, v, li, name, scale, kw):
    """q [H, Lq, d] (the last Lq positions), k, v [KVH, Lk, d] float32 -> [H, Lq, dv]; name None = exact attention."""
    if name is None:
        return Q5.attend(q, k, v, li, None, scale, kw)
    p = parse(name)
    kq, vq, R, z0 = base7(p, k, v, li)
    z = p["z"]
    if z:
        kq[:, :z], vq[:, :z] = k[:, :z], v[:, :z]
    zz = max(z, z0)
    kb, vb, K = p["kb"], p["vb"], p["K"]
    kr = k if kb == 16 else kq + Q6.res_int(k - kq, kb)
    vr = v if vb == 16 else (vq if vb == 0 else vq + Q6.res_int(v - vq, vb))
    H_, Lq, _ = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    kx, vx, kqx, vqx, krx, vrx = (t.repeat_interleave(g, 0) for t in (k, v, kq, vq, kr, vr))
    out = torch.empty(H_, Lq, v.shape[-1], device=q.device)
    kpos = torch.arange(Lk, device=q.device)[None, :]
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
    crit_kind, share = p["crit"], p["share"]
    gen = torch.Generator(device=q.device).manual_seed(7919 * li + 17)
    acc = torch.zeros(H_, Lk, device=q.device) if crit_kind in ("h2", "zc") else None
    tail = torch.zeros(H_, 0, Lk, device=q.device) if crit_kind == "sn" else None
    sub = torch.arange(g, device=q.device) // 2
    nsub = int(sub.max()) + 1
    for r0 in range(0, Lq, 512):
        r1 = min(Lq, r0 + 512)
        n = r1 - r0
        qpos = torch.arange(Lk - Lq + r0, Lk - Lq + r1, device=q.device)[:, None]
        al = kpos <= qpos
        if win_:
            al &= kpos > qpos - int(win_)

        def fin(s):
            if softcap:
                s = torch.tanh(s / softcap) * softcap
            return torch.softmax(s.masked_fill(~al, float("-inf")), -1)

        qr = q[:, r0:r1]
        s = torch.einsum("hqd,hkd->hqk", qr, kqx) * scale
        se = torch.einsum("hqd,hkd->hqk", qr, kx) * scale
        ex = torch.zeros_like(s, dtype=torch.bool)
        if R:
            ex = ((kpos >= (qpos + 1) // R * R) & al)[None].expand_as(s)
            s = torch.where(ex, se, s)
        if zz:
            ex = ex | (kpos < zz)[None]
        ex = ex & al[None]
        m = torch.zeros_like(ex)
        pe = None
        if crit_kind in ("or", "h2", "zc", "sn") or acc is not None:
            pe = fin(se)
        if K:
            bad = ~al[None] | ex
            if crit_kind == "qs":
                crit = fin(s)
            elif crit_kind == "or":
                crit = pe
            elif crit_kind == "nz":
                e = s - se
                alf = al[None].float()
                cnt = alf.sum(-1, keepdim=True)
                me = (e * alf).sum(-1, keepdim=True) / cnt
                sd = ((((e - me) ** 2) * alf).sum(-1, keepdim=True) / cnt).sqrt()
                crit = fin(se + sd * torch.randn(se.shape, generator=gen, device=q.device))
            elif crit_kind in ("h2", "zc"):
                crit = acc[:, None, :] + pe.cumsum(1)
                if crit_kind == "zc":
                    crit = crit / (qpos - kpos + 1).clamp_min(1)[None].float()
            elif crit_kind == "sn":
                ext = torch.cat([tail, pe], 1)
                c0 = torch.cat([torch.zeros_like(ext[:, :1]), ext.cumsum(1)], 1)
                j = torch.arange(n, device=q.device) + tail.shape[1] + 1
                crit = c0[:, j] - c0[:, (j - 32).clamp_min(0)]
            elif crit_kind == "rd":
                crit = torch.rand(s.shape, generator=gen, device=q.device)
            else:
                crit = kpos.float()[None].expand_as(s).clone()
            crit = crit.masked_fill(bad, -float("inf"))
            cg = crit.reshape(KVH, g, n, Lk)
            if share == "q":
                m = torch.zeros_like(crit, dtype=torch.bool).scatter_(-1, crit.topk(min(K, Lk), -1).indices, True)
            elif share in ("mean", "max", "sm2"):
                kk = min(K * (math.ceil(g / 2) if share == "sm2" else 1), Lk)
                c = cg.amax(1) if share == "max" else cg.mean(1)
                m = torch.zeros_like(c, dtype=torch.bool).scatter_(-1, c.topk(kk, -1).indices, True).repeat_interleave(g, 0)
            elif share == "sg2":
                cs = torch.stack([cg[:, sub == i].mean(1) for i in range(nsub)], 1)
                ms = torch.zeros_like(cs, dtype=torch.bool).scatter_(-1, cs.topk(min(K, Lk), -1).indices, True)
                m = ms[:, sub].reshape(H_, n, Lk)
            else:
                per = min(math.ceil(K / g), Lk)
                u = torch.zeros_like(crit, dtype=torch.bool).scatter_(-1, crit.topk(per, -1).indices, True)
                u = u.reshape(KVH, g, n, Lk).any(1) & ~bad.reshape(KVH, g, n, Lk)[:, 0]
                c = cg.mean(1).masked_fill(u, -float("inf"))
                kk = min(K, Lk)
                fill = torch.arange(kk, device=q.device) < (kk - u.sum(-1, keepdim=True)).clamp_min(0)
                m = (u | torch.zeros_like(u).scatter_(-1, c.topk(kk, -1).indices, fill)).repeat_interleave(g, 0)
            m = m & ~bad
            s = torch.where(m, torch.einsum("hqd,hkd->hqk", qr, krx) * scale, s)
        if acc is not None:
            acc = acc + pe.sum(1)
        if tail is not None:
            tail = torch.cat([tail, pe], 1)[:, -31:]
        if p["drop"]:
            s = s.masked_fill(~(m | ex), float("-inf"))
        pr = fin(s)
        o = torch.einsum("hqk,hkd->hqd", pr, vqx)
        o = o + torch.einsum("hqk,hkd->hqd", pr * ex, vx - vqx)
        if K and vb:
            o = o + torch.einsum("hqk,hkd->hqd", pr * m, vrx - vqx)
        out[:, r0:r1] = o
    return out
