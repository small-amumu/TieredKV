# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""sel11_q (rootcause fork, exploratory): sel10_q with the 1-bit half's per-token (mean, mean absolute deviation) stored
in 8 bits instead of fp16, to see whether the water-filled key record can drop back from 12 to 10 metadata bytes per
token (kernel overhead at long context). A method name is
  stk<F>128<rest>   F = e4 (float8 e4m3) | e5 (float8 e5m2) | rab (the kernel's m8 = 2 form: A = mean - s and B = 2 s
                    as int8 codes in units of the 3-bit half's step / 16, clamped to +-127) | ab (the kernel's form: A = mean - s and B = 2 s each in
                    float8 e4m3) | r8 (int8 codes of mu / step and s / step, step = the
                    3-bit half's stored step, codes clamped to +-127 at 1/16 resolution);
  any sel10_q name is passed through (stk128..., kivi128..., ...).
Only the 1-bit half changes: the 3-bit half, the values, the window and the sinks are sel10_q's. The debias variance uses
the rounded s.
"""
from __future__ import annotations

import re

import torch

import sel10_q as S10

S4, Q5 = S10.S4, S10.Q5
NAME = re.compile(r"stk(e4|e5|r8|ab|rab)128(.*)")
MF: dict = {"fmt": None}
CLIP: dict = {"n": 0, "a": 0, "b": 0}     # rab: tokens, and how many had A or B clipped at 127 / 16 steps
_stack_base = S10.stack_base


def _round8(x, fmt, step):
    if fmt == "e4":                                       # float8 casts do not saturate: clamp to the finite range
        return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype)
    if fmt == "e5":
        return x.clamp(-57344.0, 57344.0).to(torch.float8_e5m2).to(x.dtype)
    u = step.clamp_min(1e-12) / 16
    return torch.clamp(torch.round(x / u), -127, 127) * u


def stack_base8(k, v, li):
    kq, vq, R, z0 = _stack_base(k, v, li)
    fmt = MF["fmt"]
    if fmt is None:
        return kq, vq, R, z0
    I = S10.INFO[li]
    h, order, b, step = I["h"], I["order"], I["b"][:, None, :], I["step"][..., None]
    f, _ = S10.P10.split(S10.STACK)
    rc = bool(f["rc"]) and li not in S10.P11.NOPE
    mt, _ = S10.P9.centre(li, k, rc, S10.Q7.parse(S10.P10.split(S10.STACK)[1])["a"])
    x = (k - mt) * b
    o = order[:, None, :].expand_as(x)
    xs = x.gather(-1, o)
    Hh = Q5.CQ.hadamard(h, k.device)
    yh, _ = S10._step_clip(xs[..., :h] @ Hh, 3)
    ylo = xs[..., h:] @ Hh
    mu = Q5.f16(ylo.mean(-1, keepdim=True))
    s = Q5.f16((ylo - mu).abs().mean(-1, keepdim=True))
    if fmt == "rab":    # the kernel's m8 = 2 form (kbench7 r20): A = mean - s, B = 2 s as int8 codes in step / 16 units
        u = step.clamp_min(1e-12) / 16
        A8 = torch.clamp(torch.round(Q5.f16(mu - s) / u), -127, 127) * u
        B8 = torch.clamp(torch.round(Q5.f16(2 * s) / u), 0, 127) * u
        lo = A8 + B8 * (ylo >= mu).to(ylo.dtype)
        s8 = B8 / 2
        CLIP["n"] += int(mu.numel())
        CLIP["a"] += int(((mu - s).abs() > 127 * u).sum())
        CLIP["b"] += int((2 * s > 127 * u).sum())
    elif fmt == "ab":   # the kernel's stored form: (A, B) = (mean - s, 2 s) each in float8 e4m3, lo = A + B * bit
        A8, B8 = _round8(mu - s, "e4", step), _round8(2 * s, "e4", step).abs()
        lo = A8 + B8 * (ylo >= mu).to(ylo.dtype)
        s8 = B8 / 2
    else:
        mu8, s8 = _round8(mu, fmt, step), _round8(s, fmt, step).abs()
        lo = mu8 + s8 * torch.where(ylo >= mu, 1.0, -1.0)
    rec = torch.empty_like(x).scatter_(-1, o, torch.cat([yh @ Hh.T, lo @ Hh.T], -1)) / b + mt
    kq = kq.clone()
    kq[:, 1:] = rec[:, 1:].to(kq.dtype)                 # row 0 stays as the stack stores it (exact first token)
    I["s"] = s8[..., 0]
    return kq, vq, R, z0


S10.stack_base = stack_base8


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
    dk = 16 / d                                          # two 8-bit fields instead of two fp16: -16 bits per key row
    return gpu - dk / 2, tot - dk / 2                    # mean over K and V


def attend(q, k, v, li, name, scale, kw):
    s = split(name) if name else None
    if not s:
        return S10.attend(q, k, v, li, name, scale, kw)
    MF["fmt"] = s[0]
    try:
        return S10.attend(q, k, v, li, s[1], scale, kw)
    finally:
        MF["fmt"] = None


calib, calib_all, S10_ = S10.calib, S10.calib_all, S10
RC, Q7, P11, P10, P9, P8 = S10.RC, S10.Q7, S10.P11, S10.P10, S10.P9, S10.P8
PREFILL, CHUNK, ST, INFO = S10.PREFILL, S10.CHUNK, S10.ST, S10.INFO
