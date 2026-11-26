# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""pkv9_q: round P9 of the layered-KV line (registered in reg_pkv.md): reverse water-filling of key bits
(pkv8_q) stacked on rotary-aware centring (rootcause/rc_q, round RC1), with a bit-matched uniform control.
Every pkv7_q name passes through; three suffixes are added:
  _rc   keys centred by the calibration pre-RoPE mean rotated to each key's position, m(t) = R(t) m_pre, with the channel
        balance recomputed under that centring (rc_q.calib); storage and bits unchanged.
  _kh   uniform 2-bit keys in per-token groups of d / 2 instead of min(G, d): key bits 2 + 64 / d, exactly the key bits of a
        water-filled base at every head dim (the bit-matched control for wf); values unchanged.
  wf<src>[rest]  pkv8_q water-filling on the bal128[rest] base: RoPE pairs ranked by (sum E q^2)(sum Var k), top half of
        the channels 3-bit (clipped as [rest] says), bottom half 1-bit (mean +- mean absolute deviation), each half per
        token with its own Hadamard. With _rc the ranking uses the variance of the rotary-centred keys (src c) and the
        keys are centred and balanced as _rc; src r = the pkv8_q random ranking.
Values, windows, sinks, fetch and selection are pkv7_q, used read-only (pkv8_q and rc_q are imported read-only too; their
runtime patches of pkv7_q.base7 are replaced here by base9, which calls the original pkv7_q.base7 for everything it does
not change).
eff_bits: as pkv8_q / pkv7_q, plus (i) _kh adds the same key-bit delta as wf, and (ii) the KIVI value bits use per-token
groups of min(G, d) (pkv5_q.eff_bits counted 32 / G, which undercounts by 0.125 bit per element at d = 64; the old numbers
are kept in the JSON as bits_p8).
"""
from __future__ import annotations

import torch

import pkv5_q as Q5
import pkv8_q as P8
import pkv7_q as Q7
import rc_q as RC

Q6 = Q7.Q6
SINK = Q5.SINK
_orig = P8._base7
C9: dict = {"crc": {}, "overlap": {}}
S9: dict = {"src": None, "kh": False, "rc": False}
NAMES = ("kivi128_z4", "kivi128k112_k8_v8_z4",
         "bal128_ck_w128_rc", "bal128_kh_ck_w128_rc", "wfc_ck_w128_rc", "wfr_ck_w128_rc", "wfc_ck_w128",
         "bal128k112_k8_v8_ck_w128_rc", "bal128k112_k8_v8_kh_ck_w128_rc", "wfck112_k8_v8_ck_w128_rc",
         "wfrk112_k8_v8_ck_w128_rc", "wfc_ck_rc", "wfck112_k8_v8_ck_rc")


def split(name):
    """-> (wf src or None, kh, rc, pkv7 name)."""
    rc = name.endswith("_rc")
    n = name[:-3] if rc else name
    kh = "_kh" in n
    n = n.replace("_kh", "", 1)
    src = None
    if n.startswith("wf"):
        src, n = n[2], "bal128" + n[3:]
        if src not in "cr" or kh:
            raise ValueError(name)
    return src, kh, rc, n


def parse(name):
    src, kh, rc, n7 = split(name)
    p = Q7.parse(n7)
    if (src or kh or rc) and p["base"] != "bal":
        raise ValueError(name)
    if p["clip"] == "w" and (src or kh or rc):
        raise ValueError(name)
    return p


def calib9(kv, li_list):
    """after rc_q.calib (static and rotary means, both balances) and pkv8_q.calib8 (static and random rankings): the
    water-filling ranking under rotary centring, and its top-half overlap with the static ranking (Jaccard, per KV head)."""
    for li in li_list:
        k, v, q = kv[li]
        KVH, Lc, d = k.shape
        g = q.shape[0] // KVH
        cos, sin = RC.cos_sin(Lc, k.device)
        kc = (k - RC.rotate(RC.R["m_pre"][li], cos, sin))[:, SINK:]
        qe = (q[:, SINK:] ** 2).mean(1).reshape(KVH, g, d).sum(1)
        idx = torch.tensor(P8.C8["pairs"][li], device=k.device)
        kf = kc.var(1)
        pi = (qe[:, idx[:, 0]] + qe[:, idx[:, 1]]) * (kf[:, idx[:, 0]] + kf[:, idx[:, 1]])
        C9["crc"][li] = idx[pi.argsort(-1, descending=True)].reshape(KVH, -1)
        h = d // 2
        ov = []
        for j in range(KVH):
            a, b = set(P8.C8["c"][li][j, :h].tolist()), set(C9["crc"][li][j, :h].tolist())
            ov.append(len(a & b) / len(a | b))
        C9["overlap"][li] = sum(ov) / len(ov)


def centre(li, k, rc, a):
    if rc:
        cos, sin = RC.cos_sin(k.shape[1], k.device)
        mt, bal = RC.rotate(RC.R["m_pre"][li], cos, sin), RC.R["bal"][li]
    else:
        mt, bal = Q5.C["m"][li][:, None, :], Q7.C7["bal"][li]
    return mt, (bal ** (a / 100))[:, None, :]


def base9(p, k, v, li):
    kq, vq, R, z0 = _orig(p, k, v, li)
    src, kh, rc = S9["src"], S9["kh"], S9["rc"]
    if p["base"] != "bal" or not (src or kh or rc):
        return kq, vq, R, z0
    d = k.shape[-1]
    mt, b = centre(li, k, rc, p["a"])
    x = (k - mt) * b
    ptk = Q6.per_token_clip if p["clip"] in ("1", "k") else Q5.per_token
    if src:
        order = (C9["crc"][li] if (rc and src == "c") else P8.C8[src][li])[:, None, :].expand_as(x)
        xs = x.gather(-1, order)
        h = d // 2
        Hm = Q5.CQ.hadamard(h, k.device)
        hi = ptk(xs[..., :h] @ Hm, 3, h) @ Hm.T
        lo = P8.sign1(xs[..., h:] @ Hm) @ Hm.T
        xq = torch.empty_like(x).scatter_(-1, order, torch.cat([hi, lo], -1))
    else:
        H = Q5.CQ.hadamard(d, k.device)
        xq = ptk(x @ H, 2, d // 2 if kh else p["G"]) @ H.T
    kq = xq / b + mt
    kq[:, :1] = k[:, :1]
    return kq, vq, R, z0


Q7.base7 = base9


def eff_bits(name, L, d, g=1):
    """-> (gpu, read, gpu_p8, read_p8); the last two are the round-P8 accounting (no KIVI d < G correction)."""
    src, kh, rc, n7 = split(name)
    p = Q7.parse(n7)
    gpu, read = P8.eff_bits(("wf" + src + n7[6:]) if src else n7, L, d, g)
    old = (gpu, read)
    if kh:
        w = min(1.0, (p["R"] / 2 + 1) / L)
        delta = (2 + 64 / d - (2 + 32 / min(128, d))) * (1 - w) / 2
        gpu, read = gpu + delta, read + delta
    if p["base"] == "kivi" and d < p["G"]:
        delta = (32 / d - 32 / p["G"]) * (1 - min(1.0, 64 / L)) / 2 * (1 - p["z"] / L)
        gpu, read = gpu + delta, read + delta
    return gpu, read, old[0] + (gpu - old[0] if kh else 0.0), old[1] + (read - old[1] if kh else 0.0)


def attend(q, k, v, li, name, scale, kw):
    if name is None:
        return Q7.attend(q, k, v, li, None, scale, kw)
    src, kh, rc, n7 = split(name)
    S9.update(src=src, kh=kh, rc=rc)
    try:
        return Q7.attend(q, k, v, li, n7, scale, kw)
    finally:
        S9.update(src=None, kh=False, rc=False)
