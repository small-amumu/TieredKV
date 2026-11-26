# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""pkv5_q: quantizers, effective-bit accounting and the shared attention core for round P5 (baselines at matched
effective bits; registered in reg_pkv.md). Every method stores keys AND values as 2-bit data (Kitty: some
key channels 4-bit) with scale / zero rounded to FP16, re-implemented from the papers in one fake-quantization framework:
  kivi<G>       KIVI: keys per channel over groups of G tokens, values per token over groups of G channels, asym INT2;
                tokens not yet quantized stay FP16 (residual R = 128: key j is exact for query row t iff
                j >= floor((t + 1) / R) * R)
  kvq           KVQuant-style nuq2-1% (post-RoPE keys; pre-RoPE quantization and Fisher weighting not reproduced): keys
                per channel with offline inlier ranges (calibration quantiles 0.5% / 99.5%; entries outside kept FP16 as
                sparse outliers), values per token with the ceil(1% d) entries farthest from the token mean kept FP16 and
                the rest normalised by the token min / max (FP16); both snap to a per-layer 4-level codebook from 1-D
                k-means on calibration data; first token FP16
  gear          GEAR: KIVI-style backbone with G = 64; per KV head the 2% largest-magnitude entries kept FP16 (sparse) and
                the quantization residual of the rest approximated at rank 4 (randomised SVD, FP16 factors, computed over
                the whole quantized region of the call); buffer of the 64 most recent tokens FP16 (R = 64)
  rot<G>        QuaRot / RotateKV-style: head-wide Hadamard on keys and values, per-token asym INT2 over groups of G
                channels, first token FP16 (RotateKV's channel reordering and grouped-head rotation not reproduced)
  kitty         Kitty: KIVI with G = 128 where the 12.5% key channels of largest mean magnitude in each 128-token group
                are stored INT4; values INT2 per token (G = 128); first 32 tokens and residual R = 128 FP16
  spc<G>k<K>    SpeCache-style: a kivi<G> copy on GPU; per KV head the top-K keys by the copy's attention (mean over the
                query heads of the group, exact window excluded) are fetched exactly (keys and values) from a CPU FP16 copy
  ours<G>[k<K>] ours: keys centred by the static calibration mean (positions >= 4), head-wide Hadamard, per-token asym
                INT2 over groups of G channels, first token exact; values Hadamard + per-token asym INT2 (G); optional
                top-K per KV head by base attention, fetched exactly as in spc
Effective bits per stored element on GPU, mean of K and V (eff_bits): data bits + 32 bits of FP16 scale / zero per group,
FP16 windows at their mean occupancy (R / 2 tokens, plus sink tokens), sparse outliers 32 bits (FP16 value + 16-bit
index), low-rank factors FP16. Offline per-channel ranges, codebooks and centring means are not counted (O(d) per layer).
"read" adds the exact rows fetched per decode step (16 bits x K tokens) to the bits read per element.
"""
from __future__ import annotations

import math
import re

import torch

import cq_lab3 as CQ

C: dict = {"m": {}, "klo": {}, "khi": {}, "cbk": {}, "cbv": {}}
SINK = 4
NAMES = ("kivi32", "kivi128", "kvq", "gear", "rot32", "rot128", "kitty", "ours32", "ours128",
         "spc32k64", "spc128k64", "spc128k128", "ours32k64", "ours128k32", "ours128k64", "ours128k128")


def f16(x):
    return x.half().float()


def aq(x, bits):
    """asymmetric min-max quantization over the last dim, FP16 min and step."""
    lv = 2 ** bits - 1
    mn, mx = f16(x.amin(-1, keepdim=True)), f16(x.amax(-1, keepdim=True))
    sc = f16((mx - mn) / lv)
    safe = torch.where(sc > 0, sc, torch.ones_like(sc))
    return torch.where(sc > 0, torch.clamp(torch.round((x - mn) / safe), 0, lv) * sc + mn, x)


def per_token(x, bits, G):
    *lead, d = x.shape
    G = min(G, d)
    return aq(x.reshape(*lead, d // G, G), bits).reshape(x.shape)


def _groups(x, G):
    KVH, L, d = x.shape
    Lf = L // G * G
    return Lf, x[:, :Lf].reshape(KVH, Lf // G, G, d).transpose(2, 3)


def _ungroup(out, xt, Lf):
    KVH, _, d, G = xt.shape
    out[:, :Lf] = xt.transpose(2, 3).reshape(KVH, Lf, d)
    return out


def per_channel(x, bits, G):
    """keys [KVH, L, d]: groups of G consecutive tokens per channel; the L % G tail stays exact (inside the window)."""
    Lf, xt = _groups(x, G)
    return _ungroup(x.clone(), aq(xt, bits), Lf) if Lf else x.clone()


def kitty_k(x, G=128, frac=0.125):
    Lf, xt = _groups(x, G)
    if not Lf:
        return x.clone()
    sc = xt.abs().mean(-1)
    idx = sc.topk(max(1, round(frac * x.shape[-1])), -1).indices
    m = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, idx, True)[..., None]
    return _ungroup(x.clone(), torch.where(m, aq(xt, 4), aq(xt, 2)), Lf)


def _nearest(u, cb):
    return cb[(u[..., None] - cb).abs().argmin(-1)]


def _kspan(li):
    lo, hi = C["klo"][li], C["khi"][li]
    return lo, hi, (lo + hi) / 2, ((hi - lo) / 2).clamp_min(1e-8)


def kvq_k(k, li):
    lo, hi, c, h = _kspan(li)
    u = ((k - c) / h).clamp(-1, 1)
    return torch.where((k < lo) | (k > hi), k, c + h * _nearest(u, C["cbk"][li]))


def _vnorm(v):
    n = math.ceil(0.01 * v.shape[-1])
    dev = (v - v.mean(-1, keepdim=True)).abs()
    out = dev >= dev.topk(n, -1).values[..., -1:]
    inf = torch.full_like(v[..., :1], float("inf"))
    mn = f16(torch.where(out, inf, v).amin(-1, keepdim=True))
    mx = f16(torch.where(out, -inf, v).amax(-1, keepdim=True))
    c, h = (mn + mx) / 2, (mx - mn) / 2
    u = torch.where(h > 0, (v - c) / torch.where(h > 0, h, torch.ones_like(h)), torch.zeros_like(v)).clamp(-1, 1)
    return out, c, h, u


def kvq_v(v, li):
    out, c, h, u = _vnorm(v)
    return torch.where(out, v, c + h * _nearest(u, C["cbv"][li]))


def gear(x, kind, G=64, s=0.02, r=4):
    KVH, L, d = x.shape
    Lf = L // G * G
    out = x.clone()
    if not Lf:
        return out
    X = x[:, :Lf]
    n = max(1, int(s * Lf * d))
    thr = X.abs().reshape(KVH, -1).kthvalue(Lf * d - n + 1, -1).values[:, None, None]
    S_ = X.abs() >= thr
    D = torch.where(S_, torch.zeros_like(X), X)
    Q = per_channel(D, 2, G) if kind == "k" else per_token(D, 2, G)
    U, Sv, V = torch.svd_lowrank(D - Q, q=min(r + 2, d), niter=2)
    LR = f16(U[..., :r] * Sv[..., None, :r]) @ f16(V[..., :r]).transpose(-1, -2)
    out[:, :Lf] = torch.where(S_, X, Q + LR)
    return out


def rot(x, G):
    H = CQ.hadamard(x.shape[-1], x.device)
    return per_token(x @ H, 2, G) @ H.T


def parse(name):
    if name in ("kvq", "gear", "kitty"):
        return name, {"kvq": 0, "gear": 64, "kitty": 128}[name], 0
    mm = re.fullmatch(r"(kivi|rot|ours|spc)(\d+)(?:k(\d+))?", name)
    if not mm:
        raise ValueError(name)
    return mm[1], int(mm[2]), int(mm[3] or 0)


def base(name, k, v, li):
    """-> k_hat, v_hat (sinks already exact), residual window R (0 = none), number of fetched keys per KV head."""
    kind, G, sel = parse(name)
    if kind in ("kivi", "spc"):
        kb, vb, R, S = per_channel(k, 2, G), per_token(v, 2, G), 128, 0
    elif kind == "kvq":
        kb, vb, R, S = kvq_k(k, li), kvq_v(v, li), 0, 1
    elif kind == "gear":
        kb, vb, R, S = gear(k, "k"), gear(v, "v"), 64, 0
    elif kind == "kitty":
        kb, vb, R, S = kitty_k(k), per_token(v, 2, 128), 128, 32
    elif kind == "rot":
        kb, vb, R, S = rot(k, G), rot(v, G), 0, 1
    else:
        m = C["m"][li][:, None, :]
        H = CQ.hadamard(k.shape[-1], k.device)
        kb, vb, R, S = per_token((k - m) @ H, 2, G) @ H.T + m, rot(v, G), 0, 1
    if S:
        kb[:, :S], vb[:, :S] = k[:, :S], v[:, :S]
    return kb, vb, R, sel


def eff_bits(name, L, d):
    """(GPU bits per stored element, bits read per element per decode step), mean of K and V, at context length L."""
    kind, G, sel = parse(name)
    Gt = min(G, d) if G else d

    def win(b, R, S):
        w = min(1.0, (R / 2 + S) / L)
        return b * (1 - w) + 16 * w
    if kind in ("kivi", "spc"):
        bk = bv = win(2 + 32 / G, 128, 0)
    elif kind == "kvq":
        bk = win(2 + 0.01 * 32, 0, 1)
        bv = win(2 + 32 / d + math.ceil(0.01 * d) / d * 32, 0, 1)
    elif kind == "gear":
        bk = bv = win(2 + 32 / 64 + 0.02 * 32, 64, 0) + 16 * 4 * (L + d) / (L * d)
    elif kind == "kitty":
        bk = win(2 + 2 * max(1, round(0.125 * d)) / d + 32 / 128, 128, 32)
        bv = win(2 + 32 / min(128, d), 128, 32)
    else:
        bk = bv = win(2 + 32 / Gt, 0, 1)
    gpu = (bk + bv) / 2
    return gpu, gpu + 16 * sel / L


def kmeans1d(x, k=4, it=30):
    x = x.flatten()
    if x.numel() > 2_000_000:
        x = x[torch.randperm(x.numel(), generator=torch.Generator().manual_seed(0))[:2_000_000].to(x.device)]
    c = torch.quantile(x, torch.linspace(0.5 / k, 1 - 0.5 / k, k, device=x.device))
    for _ in range(it):
        a = (x[:, None] - c).abs().argmin(1)
        for j in range(k):
            s = x[a == j]
            if s.numel():
                c[j] = s.mean()
    return c.sort().values


def calib(li, k, v):
    """k, v [KVH, Lc, d] float32 from the calibration text."""
    C["m"][li] = k[:, SINK:].mean(1)
    kk = k[:, 1:]
    C["klo"][li] = torch.quantile(kk, 0.005, dim=1, keepdim=True)
    C["khi"][li] = torch.quantile(kk, 0.995, dim=1, keepdim=True)
    lo, hi, c, h = _kspan(li)
    C["cbk"][li] = kmeans1d(((kk - c) / h)[(kk >= lo) & (kk <= hi)])
    out, _, _, u = _vnorm(v[:, 1:])
    C["cbv"][li] = kmeans1d(u[~out])


def attend(q, k, v, li, name, scale, kw):
    """q [H, Lq, d] (the last Lq positions), k, v [KVH, Lk, d] float32 -> attention output [H, Lq, dv].
    name None = exact attention."""
    H_, Lq, _ = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    kb, vb, R, sel = (k, v, 0, 0) if name is None else base(name, k, v, li)
    kx, vx, kbx, vbx = (t.repeat_interleave(g, 0) for t in (k, v, kb, vb))
    out = torch.empty(H_, Lq, v.shape[-1], device=q.device)
    kpos = torch.arange(Lk, device=q.device)[None, :]
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
    for r0 in range(0, Lq, 512):
        r1 = min(Lq, r0 + 512)
        qpos = torch.arange(Lk - Lq + r0, Lk - Lq + r1, device=q.device)[:, None]
        al = kpos <= qpos
        if win_:
            al &= kpos > qpos - int(win_)

        def fin(s):
            if softcap:
                s = torch.tanh(s / softcap) * softcap
            return torch.softmax(s.masked_fill(~al, float("-inf")), -1)

        qr = q[:, r0:r1]
        s = torch.einsum("hqd,hkd->hqk", qr, kbx) * scale
        ex = None
        if R or sel:
            se = torch.einsum("hqd,hkd->hqk", qr, kx) * scale
        if R:
            ex = ((kpos >= (qpos + 1) // R * R) & al)[None].expand_as(s)
            s = torch.where(ex, se, s)
        if sel:
            bad = ~al[None] if ex is None else (~al[None] | ex)
            crit = fin(s).masked_fill(bad, -float("inf")).reshape(KVH, g, r1 - r0, Lk).mean(1)
            idx = crit.topk(min(sel, Lk), -1).indices
            m = torch.zeros_like(crit, dtype=torch.bool).scatter_(-1, idx, True).repeat_interleave(g, 0) & ~bad
            s = torch.where(m, se, s)
            ex = m if ex is None else (ex | m)
        p = fin(s)
        o = torch.einsum("hqk,hkd->hqd", p, vbx)
        if ex is not None:
            o = o + torch.einsum("hqk,hkd->hqd", p * ex, vx - vbx)
        out[:, r0:r1] = o
    return out
