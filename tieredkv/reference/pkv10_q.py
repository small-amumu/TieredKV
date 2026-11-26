# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""pkv10_q: round P10 of the layered-KV line (registered in reg_pkv.md): reviewer controls and
extensions of the P9 stack (pkv9_q: water-filling on rotary-aware centring). Every pkv9_q name passes through; added:
  wfs / wfq   water-filling with the static ranking under rotary centring (wfs..._rc) or the rotary ranking under static
              centring (wfq..., no _rc): the 2 x 2 of centring x ranking with wfc..._rc and wfc...
  _q4         water-filled keys with only the top quarter of the channels (by pair importance) at 3 bits and the rest at
              1 bit: key bits 1.5 + 64 / d (a lower operating point)
  _wv / _vh   values: _wv water-fills the value channels (ranked by calibration Var v per KV head; top half 3-bit, bottom
              half 1-bit mean +- mean absolute deviation, each half per token with its own Hadamard); _vh is the
              bit-matched uniform control (2-bit values in per-token groups of d / 2); both 2 + 64 / d value bits
  _r<N>       KIVI only: a residual FP16 window of N tokens instead of 128 (pkv5_q fixes 128); kivi128_z4_r192 matches the
              GPU bits of a water-filled d = 128 stack (2.585 vs 2.591)
Values, fetch, selection, sinks and the window of the per-token bases are pkv7_q (read-only); pkv9_q, pkv8_q and rc_q
are imported read-only and their runtime patches of pkv7_q.base7 are replaced by base10.
"""
from __future__ import annotations

import re

import torch

import pkv5_q as Q5
import pkv7_q as Q7
import pkv8_q as P8
import pkv9_q as P9

Q6 = Q7.Q6
RC = P9.RC
SINK = Q5.SINK
_orig = P8._base7
C10: dict = {"vord": {}}
S10: dict = {"src": None, "kh": False, "rc": False, "q4": False, "wv": False, "vh": False, "r": 0}
FLAGS = re.compile(r"_(kh|wv|vh|q4|r\d+)(?=_|$)")
NAMES = ("kivi128_z4", "kivi128_z4_r192", "kivi32_z4", "kivi128k112_k8_v8_z4",
         "bal128_kh_ck_w128", "bal128_kh_ck_w128_rc", "wfc_ck_w128", "wfq_ck_w128", "wfs_ck_w128_rc", "wfc_ck_w128_rc",
         "wfc_wv_ck_w128_rc", "wfc_vh_ck_w128_rc", "wfck112_k8_v8_ck_w128_rc", "wfck80_k16_v8_ck_w128_rc",
         "wfc_q4_ck_w128_rc", "wfck112_k8_v8_q4_ck_rc")


def split(name):
    """-> (flags dict, pkv7 name)."""
    rc = name.endswith("_rc")
    n = name[:-3] if rc else name
    f = {"src": None, "kh": False, "rc": rc, "q4": False, "wv": False, "vh": False, "r": 0}
    for t in FLAGS.findall(n):
        if t[0] == "r":
            f["r"] = int(t[1:])
        else:
            f[t] = True
    n = FLAGS.sub("", n)
    if n.startswith("wf"):
        f["src"], n = n[2], "bal128" + n[3:]
    return f, n


def parse(name):
    f, n7 = split(name)
    p = Q7.parse(n7)
    src = f["src"]
    ours = src or f["kh"] or f["rc"] or f["q4"] or f["wv"] or f["vh"]
    bad = ((ours and p["base"] != "bal") or (ours and p["clip"] in ("w", "1"))
           or (src is not None and src not in "csqr") or (src == "s" and not f["rc"]) or (src == "q" and f["rc"])
           or (f["kh"] and src) or (f["q4"] and not src) or (f["wv"] and f["vh"])
           or (f["r"] and p["base"] != "kivi") or (f["r"] and f["r"] < 128))
    if bad:
        raise ValueError(name)
    return p


def calib10(kv, li_list):
    """after pkv9_q's calibration: the value-channel ranking (calibration Var v, descending, per KV head)."""
    for li in li_list:
        v = kv[li][1]
        C10["vord"][li] = v[:, SINK:].var(1).argsort(-1, descending=True)


def halves(x, order, h, ptk, bits_hi):
    """water-fill the last dim of x: channels order[:h] at bits_hi (one group), the rest 1-bit (one mean +- mean absolute
    deviation pair); both parts per token, rotated by Hadamard blocks of size h (block-diagonal when d - h > h, so the
    1-bit part keeps a single pair of FP16 numbers)."""
    d = x.shape[-1]
    o = order[:, None, :].expand_as(x)
    xs = x.gather(-1, o)
    Hh = Q5.CQ.hadamard(h, x.device)
    hi = ptk(xs[..., :h] @ Hh, bits_hi, h) @ Hh.T
    if d - h == h:
        # same arithmetic as pkv8_q / pkv9_q (bit-identical results for the half split)
        lo = P8.sign1(xs[..., h:] @ Hh) @ Hh.T
    else:
        lo = xs[..., h:].reshape(*x.shape[:-1], (d - h) // h, h) @ Hh
        lo = (P8.sign1(lo.flatten(-2)).reshape(lo.shape) @ Hh.T).flatten(-2)
    return torch.empty_like(x).scatter_(-1, o, torch.cat([hi, lo], -1))


def base10(p, k, v, li):
    kq, vq, R, z0 = _orig(p, k, v, li)
    f = S10
    if p["base"] == "kivi":
        return kq, vq, (f["r"] or R), z0
    if p["base"] != "bal" or not (f["src"] or f["kh"] or f["rc"] or f["wv"] or f["vh"]):
        return kq, vq, R, z0
    d = k.shape[-1]
    if f["src"] or f["kh"] or f["rc"]:
        mt, b = P9.centre(li, k, f["rc"], p["a"])
        x = (k - mt) * b
        ptk = Q6.per_token_clip if p["clip"] == "k" else Q5.per_token
        if f["src"]:
            key = {"c": "crc" if f["rc"] else "c", "s": "c", "q": "crc", "r": "r"}[f["src"]]
            order = P9.C9["crc"][li] if key == "crc" else P8.C8[key][li]
            xq = halves(x, order, d // 4 if f["q4"] else d // 2, ptk, 3)
        else:
            H = Q5.CQ.hadamard(d, k.device)
            xq = ptk(x @ H, 2, d // 2 if f["kh"] else p["G"]) @ H.T
        kq = xq / b + mt
        kq[:, :1] = k[:, :1]
    if f["wv"]:
        vq = halves(v, C10["vord"][li], d // 2, Q5.per_token, 3)
    elif f["vh"]:
        H = Q5.CQ.hadamard(d, k.device)
        vq = Q5.per_token(v @ H, 2, d // 2) @ H.T
    vq[:, :1] = v[:, :1]
    return kq, vq, R, z0


Q7.base7 = base10


def eff_bits(name, L, d, g=1):
    """-> (gpu, read) per stored element, mean of K and V; KIVI value bits use groups of min(G, d) (pkv9_q fix)."""
    f, n7 = split(name)
    p = Q7.parse(n7)
    if p["base"] == "kivi":
        gpu, read = P9.eff_bits(n7, L, d, g)[:2]
        if f["r"]:
            b = (2 + 32 / p["G"] + 2 + 32 / min(p["G"], d)) / 2
            delta = (16 - b) * (f["r"] - 128) / 2 / L
            gpu, read = gpu + delta, read + delta
        return gpu, read
    src = f["src"]
    if src:
        n9 = "wf" + ("c" if src in "csq" else src) + n7[6:]
    else:
        n9 = n7 + ("_kh" if f["kh"] else "")
    gpu, read = P9.eff_bits(n9, L, d, g)[:2]
    w = min(1.0, (p["R"] / 2 + 1) / L)
    if f["q4"]:
        dk = (1.5 + 64 / d) - (2 + 64 / d)
        gpu, read = gpu + dk * (1 - w) / 2, read + dk * (1 - w) / 2
    if f["wv"] or f["vh"]:
        dv = (2 + 64 / d) - (2 + 32 / min(128, d))
        gpu, read = gpu + dv * (1 - w) / 2, read + dv * (1 - w) / 2
    return gpu, read


def attend(q, k, v, li, name, scale, kw):
    if name is None:
        return Q7.attend(q, k, v, li, None, scale, kw)
    f, n7 = split(name)
    S10.update(f)
    try:
        return Q7.attend(q, k, v, li, n7, scale, kw)
    finally:
        S10.update(src=None, kh=False, rc=False, q4=False, wv=False, vh=False, r=0)
