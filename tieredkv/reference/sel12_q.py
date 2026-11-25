# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""sel12_q (rootcause fork; round S1, capacity route): sel10_q's stack base with 8-bit metadata, so the GPU base drops from
2.375 to 2.25 (m8) or 2.22 (sym8: keys 2.19, values 2.25) bits per element (mean over K and V) asymptotically, end to end through sel4_q's selection / two-stage fetch /
debias. A method name is
  stk(m8|sym8)128<rest>   m8:   hi half step in float8 e4m3 and zero in e4m3 step units (clip search kept), lo half
                                (mean, MAD) in e4m3;
                          sym8: hi half symmetric 8-level grid (scale max|y| / 3.5 in e4m3, no zero point), lo half as m8;
                          m8v8 (S1b): m8 keys and the 2-bit per-token value codes with their step and zero in e4m3
                                (zero in step units), the value quantizer otherwise sel10_q's (Hadamard, per token, min-max);
                          rest as sel10_q (e.g. stk m8 128k64_x2s128_j50);
  any sel10_q name is passed through (stk128..., kivi128..., ...).
The quantizers are s0_cov.hi_quant / lo_quant (S0, reviewed) applied to the stack's centred, balanced, permuted, rotated
coordinates exactly as sel10_q.stack_base; the debias variance uses the stored (8-bit) step and MAD. Row 0 stays exact as
in the stack. Values, window and sinks are sel10_q's."""
from __future__ import annotations

import re

import torch

import s0_cov as S0
import sel10_q as S10

P10, P9, P11, Q7, Q5 = S10.P10, S10.P9, S10.P11, S10.Q7, S10.Q5
NAME = re.compile(r"stk(m8v8|m8|sym8)128(.*)")
MF: dict = {"fmt": None}
_stack_base = S10.stack_base      # = sel11_q.stack_base8 (imported via s0_cov), a pass-through while sel11's MF is None;
                                  # chain at run time: stack_base12 -> stack_base8 -> sel10 original (review)


def _hi_step(y, kind):
    """reconstruction and the stored step (for the debias variance) of the hi half: m8 repeats s0_cov.hi_quant's clip
    search and returns the chosen step; sym8 its scale."""
    if kind == "sym8":
        raw = y.abs().amax(-1, keepdim=True) / 3.5
        sc = torch.where(raw > 0, S0.e4(raw).clamp_min(S0.E4MIN), 0.0)
        return S0.hi_quant(y, "sym"), sc[..., 0]
    lv = 7
    mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    c, h = (mn + mx) / 2, (mx - mn) / 2
    best, berr, bsc = None, None, None
    for gm in S0.Q.Q7.Q6.GAMMAS:
        raw = 2 * h * gm / lv
        sc = torch.where(raw > 0, S0.e4(raw).clamp_min(S0.E4MIN), 0.0)
        safe = torch.where(sc > 0, sc, torch.ones_like(sc))
        zr = (c - h * gm) / safe
        CLIP["zsat"] += int(((zr.abs() > 448) & (sc > 0)).sum())          # zero point saturating e4m3 (review)
        lo = S0.e4(zr) * sc
        xh = torch.where(sc > 0, torch.clamp(torch.round((y - lo) / safe), 0, lv) * sc + lo, y)
        err = ((xh - y) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, berr, bsc = xh, err, sc
        else:
            bt = err < berr
            best, berr, bsc = torch.where(bt, xh, best), torch.where(bt, err, berr), torch.where(bt, sc, bsc)
    CLIP["n"] += int(y[..., 0].numel())
    if not torch.equal(best, S0.hi_quant(y, "m8")):
        raise RuntimeError("sel12 m8 differs from s0_cov.hi_quant")
    return best, bsc[..., 0]


def stack_base12(k, v, li):
    kq, vq, R, z0 = _stack_base(k, v, li)
    fmt = MF["fmt"]
    if fmt is None:
        return kq, vq, R, z0
    f, n7 = P10.split(S10.STACK)
    rc = bool(f["rc"]) and li not in P11.NOPE
    mt, b = P9.centre(li, k, rc, Q7.parse(n7)["a"])
    x = (k - mt) * b
    I = S10.INFO[li]
    o = I["order"][:, None, :].expand_as(x)
    xs = x.gather(-1, o)
    h = I["h"]
    Hh = Q5.CQ.hadamard(h, k.device)
    yh, step = _hi_step(xs[..., :h] @ Hh, "m8" if fmt == "m8v8" else fmt)
    ylo = xs[..., h:] @ Hh
    lo = S0.lo_quant(ylo, True)
    s8 = S0.e4(Q5.f16((ylo - Q5.f16(ylo.mean(-1, keepdim=True))).abs().mean(-1, keepdim=True))).abs()
    rec = torch.empty_like(x).scatter_(-1, o, torch.cat([yh @ Hh.T, lo @ Hh.T], -1)) / b + mt
    kq = kq.clone()
    kq[:, 1:] = rec[:, 1:].to(kq.dtype)
    I["step"], I["s"] = step, s8[..., 0]
    if fmt == "m8v8":
        vq = _v8(v, vq)
    return kq, vq, R, z0


def _v8(v, vq_ref):
    """the stack's 2-bit per-token Hadamard value codes (pkv5_q.per_token over the whole head dim) with an e4m3 step and
    an e4m3 zero in step units; first checks that the fp16-metadata formula reproduces the stack's values (rows 1:)."""
    d = v.shape[-1]
    H = Q5.CQ.hadamard(d, v.device)
    y = v @ H
    if not float((Q5.per_token(y, 2, d) @ H.T - vq_ref)[:, 1:].abs().max() / v.abs().max().clamp_min(1e-6)) < 1e-4:
        raise RuntimeError("value quantizer replication differs from the stack")
    mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    raw = (mx - mn) / 3
    sc = torch.where(raw > 0, S0.e4(raw).clamp_min(S0.E4MIN), 0.0)
    safe = torch.where(sc > 0, sc, torch.ones_like(sc))
    zr = mn / safe
    sat = (zr.abs() > 448) & (sc > 0)
    CLIP["vzsat"] += int(sat.sum())                           # reported; saturated tokens keep an fp16 zero (review)
    lo = torch.where(sat, Q5.f16(mn), S0.e4(zr) * sc)
    q = torch.where(sc > 0, torch.clamp(torch.round((y - lo) / safe), 0, 3) * sc + lo, y)
    out = q @ H.T
    out[:, :1] = vq_ref[:, :1]
    return out.to(vq_ref.dtype)


S10.stack_base = stack_base12


def split(name):
    m = NAME.fullmatch(name)
    return (m[1], "stk128" + m[2]) if m else None


def parse(name):
    s = split(name)
    return S10.parse(s[1] if s else name)


def eff_bits(name, L, d, g=1):
    s = split(name)
    if not s:
        return S10.eff_bits(name, L, d, g)
    gpu, tot = S10.eff_bits(s[1], L, d, g)
    dk = {"m8": 32, "sym8": 40, "m8v8": 32}[s[0]] / d       # key metadata bits saved per element (m8 4 x 8, sym8 4 x 8 + 8)
    dv = (16 if s[0] == "m8v8" else 0) / d                  # value: fp16 step + zero -> 2 x 8
    return gpu - (dk + dv) / 2, tot - (dk + dv) / 2        # mean over K and V


def attend(q, k, v, li, name, scale, kw):
    s = split(name) if name else None
    if not s:
        return S10.attend(q, k, v, li, name, scale, kw)
    MF["fmt"] = s[0]
    try:
        return S10.attend(q, k, v, li, s[1], scale, kw)
    finally:
        MF["fmt"] = None


calib, calib_all = S10.calib, S10.calib_all
RC, Q7, P11, P10, P9, P8 = S10.RC, S10.Q7, S10.P11, S10.P10, S10.P9, S10.P8
S4, Q5 = S10.S4, S10.Q5
PREFILL, CHUNK, ST, INFO = S10.PREFILL, S10.CHUNK, S10.ST, S10.INFO
CLIP = {"n": 0, "zsat": 0, "vzsat": 0, "a": 0, "b": 0}   # n: tokens x layers x KV heads coded by m8; zsat: candidate zero points saturating
