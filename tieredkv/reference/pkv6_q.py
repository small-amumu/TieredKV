# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""pkv6_q: round P6 of the layered-KV line (registered in reg_pkv.md): what to fetch on a KIVI base.
Bases, quantizers, calibration and bit accounting come from pkv5_q (read-only import). A method name is
  <base><G>[k<K>][_k<kb>][_v<vb>][_s<share>][_z<sink>][_o<N>]
  base   kivi (pkv5_q kivi<G>: per-channel keys, per-token values, residual window R = 128, streaming semantics)
         | ours (pkv5_q ours<G>: centred + Hadamard per-token keys, Hadamard per-token values, first token exact)
         | bal (ours with a static per-channel balance b_c = sqrt(max|q_c| / max|k_c - m_c|) from calibration: keys are
           centred, multiplied by b (queries would be divided by b, so q.k is unchanged), Hadamard, per-token INT2, then
           mapped back; b is offline, so the bits equal ours)
         | ol (ours with the N key channels of largest calibration mean |k - m| per KV head stored FP16 and removed
           before the Hadamard; N = _o<N>, default 4; the FP16 channels are counted in the bits)
         _c1 on ours / bal / ol: per-token groups pick their clipping ratio from GAMMAS by group MSE instead of plain
           min-max (keys and values; same stored FP16 min and step, so no extra bits)
  K      keys fetched per KV head per query row (0 = no fetch)
  kb     key refinement: 16 = exact (default), 8 / 4 = base + 8- / 4-bit residual (symmetric INT, blocks of 16, FP16 scale)
  vb     value refinement of the same tokens: 16 (default), 8, 4, or 0 = values stay at the base
  share  mean (default; top-K of the group-mean base attention, one set per KV head) | max (group max) | uni (union of
         each query head's top-ceil(K/g), filled to K by the group mean) | q (each query head its own top-K; reference
         only, it fetches up to g K rows per KV head)
  sink   first z tokens kept exact in the base (default 0)
Fetched bits per key: exact 16, res8 9, res4 5 (per element, K and V counted separately); eff_bits returns
(GPU bits per element, bits read per element per decode step) as in pkv5_q.
"""
from __future__ import annotations

import math
import re

import torch

import pkv5_q as Q5

SINK = Q5.SINK
FB = {16: 16, 8: 9, 4: 5, 0: 0}
C6: dict = {"bal": {}, "ol": {}}
NAMES = ("kivi128", "kivi128_z4", "kivi128k64", "kivi128k64_z4", "kivi128k112_k8_v8", "kivi128k82_v8",
         "kivi128k32", "kivi128k57_k8_v8", "kivi32k64", "kivi32k112_k8_v8", "ours128k64", "ours128k112_k8_v8",
         "kivi128k64_sq", "kivi128k64_smax", "kivi128k64_suni",
         "ours128", "bal128", "ol128_o2", "ol128", "bal128k64", "ol128k64", "bal128k112_k8_v8",
         "ours128_c1", "bal128_c1", "bal128_c1_z4", "bal128k64_c1")


def calib(li, k, v, q=None):
    """k, v [KVH, Lc, d], q [H, Lc, d] float32 from the calibration text."""
    Q5.calib(li, k, v)
    kc = (k - Q5.C["m"][li][:, None, :])[:, SINK:]
    C6["ol"][li] = kc.abs().mean(1)
    if q is not None:
        KVH, d = k.shape[0], k.shape[-1]
        qm = q[:, SINK:].reshape(KVH, -1, d).abs().amax(1)
        C6["bal"][li] = (qm / kc.abs().amax(1).clamp_min(1e-6)).clamp_min(1e-6).sqrt()


GAMMAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def per_token_clip(x, bits, G):
    """per-token asym INT over groups of G with the clipping ratio per group chosen from GAMMAS by group MSE
    (computed online per token, stored as the usual FP16 min and step, so no extra bits)."""
    *lead, d = x.shape
    G = min(G, d)
    xb = x.reshape(*lead, d // G, G)
    lv = 2 ** bits - 1
    mn, mx = xb.amin(-1, keepdim=True), xb.amax(-1, keepdim=True)
    c, h = (mn + mx) / 2, (mx - mn) / 2
    best, berr = None, None
    for gm in GAMMAS:
        lo, sc = Q5.f16(c - h * gm), Q5.f16(2 * h * gm / lv)
        safe = torch.where(sc > 0, sc, torch.ones_like(sc))
        xh = torch.where(sc > 0, torch.clamp(torch.round((xb - lo) / safe), 0, lv) * sc + lo, xb)
        err = ((xh - xb) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, berr = xh, err
        else:
            bt = err < berr
            best, berr = torch.where(bt, xh, best), torch.where(bt, err, berr)
    return best.reshape(x.shape)


def base6(base, G, k, v, li, N, clip):
    """-> k_hat, v_hat, residual window R, number of leading tokens exact in the base."""
    if base == "kivi":
        kq, vq, R, _ = Q5.base(f"kivi{G}", k, v, li)
        return kq, vq, R, 0
    pt = per_token_clip if clip else Q5.per_token
    m = Q5.C["m"][li][:, None, :]
    H = Q5.CQ.hadamard(k.shape[-1], k.device)
    vq = pt(v @ H, 2, G) @ H.T
    if base == "ours":
        kq = pt((k - m) @ H, 2, G) @ H.T + m
    elif base == "bal":
        b = C6["bal"][li][:, None, :]
        kq = (pt(((k - m) * b) @ H, 2, G) @ H.T) / b + m
    else:
        idx = C6["ol"][li].topk(N, -1).indices
        om = torch.zeros_like(C6["ol"][li], dtype=torch.bool).scatter_(-1, idx, True)[:, None, :]
        x = (k - m).masked_fill(om, 0.0)
        kq = torch.where(om, k, pt(x @ H, 2, G) @ H.T + m)
    kq[:, :1], vq[:, :1] = k[:, :1], v[:, :1]
    return kq, vq, 0, 1


def res_int(r, bits, B=16):
    sh = r.shape
    lv = 2 ** (bits - 1)
    rb = r.reshape(*sh[:-1], sh[-1] // B, B)
    mx = Q5.f16(rb.gather(-1, rb.abs().argmax(-1, keepdim=True)))
    d = mx / -lv
    safe = torch.where(d != 0, d, torch.ones_like(d))
    return torch.where(d != 0, torch.clamp(torch.round(rb / safe), -lv, lv - 1) * d, torch.zeros_like(rb)).reshape(sh)


def parse(name):
    m = re.fullmatch(r"(kivi|ours|bal|ol)(\d+)(?:k(\d+))?((?:_[kvszoc][a-z0-9]+)*)", name)
    if not m:
        raise ValueError(name)
    o = {"k": "16", "v": "16", "s": "mean", "z": "0", "o": "4", "c": "0"}
    for t in m[4].split("_")[1:]:
        o[t[0]] = t[1:]
    kb, vb = int(o["k"]), int(o["v"])
    if kb not in (16, 8, 4) or vb not in FB or o["s"] not in ("mean", "max", "uni", "q") or (m[1] == "kivi" and o["c"] != "0"):
        raise ValueError(name)
    return m[1], int(m[2]), int(m[3] or 0), kb, vb, o["s"], int(o["z"]), int(o["o"]), int(o["c"])


def eff_bits(name, L, d):
    base, G, K, kb, vb, share, z, N, _ = parse(name)
    gpu, _ = Q5.eff_bits(f"{'kivi' if base == 'kivi' else 'ours'}{G}", L, d)
    if base == "ol":
        gpu += N * (16 - 2) / d / 2  # FP16 key channels instead of INT2 data (keys only; mean of K and V)
    gpu += (16 - gpu) * z / L
    return gpu, gpu + K * (FB[kb] + FB[vb]) / 2 / L


def attend(q, k, v, li, name, scale, kw):
    """q [H, Lq, d] (the last Lq positions), k, v [KVH, Lk, d] float32 -> [H, Lq, dv]; name None = exact attention."""
    if name is None:
        return Q5.attend(q, k, v, li, None, scale, kw)
    base, G, K, kb, vb, share, z, N, clip = parse(name)
    kq, vq, R, z0 = base6(base, G, k, v, li, N, clip)
    if z:
        kq[:, :z], vq[:, :z] = k[:, :z], v[:, :z]
    zz = max(z, z0)
    kr = k if kb == 16 else kq + res_int(k - kq, kb)
    vr = v if vb == 16 else (vq if vb == 0 else vq + res_int(v - vq, vb))
    H_, Lq, _ = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    kx, vx, kqx, vqx, krx, vrx = (t.repeat_interleave(g, 0) for t in (k, v, kq, vq, kr, vr))
    out = torch.empty(H_, Lq, v.shape[-1], device=q.device)
    kpos = torch.arange(Lk, device=q.device)[None, :]
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
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
        ex = torch.zeros_like(s, dtype=torch.bool)
        if R:
            se = torch.einsum("hqd,hkd->hqk", qr, kx) * scale
            ex = ((kpos >= (qpos + 1) // R * R) & al)[None].expand_as(s)
            s = torch.where(ex, se, s)
        m = torch.zeros_like(ex)
        if K:
            bad = ~al[None] | ex
            if zz:
                # tokens already exact in the base are never fetched
                bad = bad | (kpos < zz)[None]
            crit = fin(s).masked_fill(bad, -float("inf"))
            cg = crit.reshape(KVH, g, n, Lk)
            kk = min(K, Lk)
            if share == "q":
                m = torch.zeros_like(crit, dtype=torch.bool).scatter_(-1, crit.topk(kk, -1).indices, True)
            elif share in ("mean", "max"):
                c = cg.mean(1) if share == "mean" else cg.amax(1)
                m = torch.zeros_like(c, dtype=torch.bool).scatter_(-1, c.topk(kk, -1).indices, True).repeat_interleave(g, 0)
            else:
                per = min(math.ceil(K / g), Lk)
                u = torch.zeros_like(crit, dtype=torch.bool).scatter_(-1, crit.topk(per, -1).indices, True)
                u = u.reshape(KVH, g, n, Lk).any(1) & ~bad.reshape(KVH, g, n, Lk)[:, 0]
                c = cg.mean(1).masked_fill(u, -float("inf"))
                idx = c.topk(kk, -1).indices
                fill = torch.arange(kk, device=q.device) < (kk - u.sum(-1, keepdim=True)).clamp_min(0)
                f = torch.zeros_like(u).scatter_(-1, idx, fill)
                m = (u | f).repeat_interleave(g, 0)
            m = m & ~bad
            s = torch.where(m, torch.einsum("hqd,hkd->hqk", qr, krx) * scale, s)
        p = fin(s)
        o = torch.einsum("hqk,hkd->hqd", p, vqx)
        if R:
            o = o + torch.einsum("hqk,hkd->hqd", p * ex, vx - vqx)
        if K and vb:
            o = o + torch.einsum("hqk,hkd->hqd", p * m, vrx - vqx)
        out[:, r0:r1] = o
    return out
