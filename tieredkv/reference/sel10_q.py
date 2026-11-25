# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""sel10_q (rootcause fork; combination round): sel4_q's selection, two-stage fetch and debias on the main line's strongest
key base (pkv11_q wfc_ck_w128_rc: keys centred by the rotary-aware mean, channel balance, the top half of the RoPE pairs by
calibration importance at 3 bits and the rest at 1 bit (mean +- mean absolute deviation), each half per token after its own
Hadamard, key clipping, values per-token 2-bit, a 128-token FP16 window and 4 exact sinks). Imported read-only: pkv11_q
(and through it pkv10/9/8/7/5_q, rc_q) and sel4_q.
A method name is
  stk128<rest>   = sel4_q's kivi128<rest> (e.g. stk128k64_x2s128_j50) with the base replaced by the stack;
  any sel4_q / bl_q name (KIVI-128 anchors in the same run).
Debias variance for the stack (zero extra storage): the score error of key j is (q / b) . (x_hat - x) in the permuted,
rotated space; the 3-bit half has uniform rounding of the stored step D_j (variance D_j^2 / 12 per element), the 1-bit half
has error variance (pi / 2 - 1) s_j^2 per element for a Gaussian with stored mean absolute deviation s_j, so
sigma_j^2 = scale^2 ( ||((q / b)_P)_hi H||^2 D_j^2 / 12 + ||((q / b)_P)_lo H||^2 (pi / 2 - 1) s_j^2 ).
Known approximation: clipped 3-bit entries have a larger, one-sided error than D^2 / 12 and the 1-bit term assumes a
Gaussian half; the debias fraction absorbs a common scale, and every clause has a no-debias counterpart.
NoPE layers (pkv11_q.NOPE) use static centring and the static ranking, as pkv11_q.
"""
from __future__ import annotations

import math
import re
import types

import torch

import pkv11_q as P11
import sel4_q as S4

P10, P9, P8, RC, Q7 = P11.P10, P11.P9, P11.P8, P11.RC, P11.Q7
Q5, Q6 = S4.Q5, P11.Q7.Q6
STACK = "wfc_ck_w128_rc"
NAME = re.compile(r"stk128(.*)")
ST: dict = {"on": False, "li": None}
INFO: dict = {}
PREFILL, CHUNK = S4.PREFILL, S4.CHUNK


def split(name):
    m = NAME.fullmatch(name)
    return "kivi128" + m[1] if m else None


def _step_clip(y, bits):
    """pkv6_q.per_token_clip over the whole last dim (G = its size), also returning the chosen step."""
    lv = 2 ** bits - 1
    mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    c, h = (mn + mx) / 2, (mx - mn) / 2
    best, berr, bsc = None, None, None
    for gm in Q6.GAMMAS:
        lo, sc = Q5.f16(c - h * gm), Q5.f16(2 * h * gm / lv)
        safe = torch.where(sc > 0, sc, torch.ones_like(sc))
        xh = torch.where(sc > 0, torch.clamp(torch.round((y - lo) / safe), 0, lv) * sc + lo, y)
        err = ((xh - y) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, berr, bsc = xh, err, sc
        else:
            bt = err < berr
            best, berr, bsc = torch.where(bt, xh, best), torch.where(bt, err, berr), torch.where(bt, sc, bsc)
    return best, bsc[..., 0]


def stack_base(k, v, li):
    f, n7 = P10.split(STACK)
    if li in P11.NOPE and f["rc"]:
        f["rc"] = False
    p = Q7.parse(n7)
    P10.S10.update(f)
    try:
        kq, vq, R, z0 = Q7.base7(p, k, v, li)
    finally:
        P10.S10.update(src=None, kh=False, rc=False, q4=False, wv=False, vh=False, r=0)
    d = k.shape[-1]
    h = d // 2
    mt, b = P9.centre(li, k, f["rc"], p["a"])
    x = (k - mt) * b
    order = (P9.C9["crc"][li] if f["rc"] else P8.C8["c"][li])
    o = order[:, None, :].expand_as(x)
    xs = x.gather(-1, o)
    Hh = Q5.CQ.hadamard(h, k.device)
    yh, step = _step_clip(xs[..., :h] @ Hh, 3)
    ylo = xs[..., h:] @ Hh
    mu = Q5.f16(ylo.mean(-1, keepdim=True))
    s = Q5.f16((ylo - mu).abs().mean(-1, keepdim=True))
    lo = mu + s * torch.where(ylo >= mu, 1.0, -1.0)
    rec = torch.empty_like(x).scatter_(-1, o, torch.cat([yh @ Hh.T, lo @ Hh.T], -1)) / b + mt
    err = (rec[:, 1:] - kq[:, 1:]).abs().max() / k.abs().max().clamp_min(1e-6)
    if not float(err) < 1e-4:                  # also catches NaN
        raise RuntimeError(f"layer {li}: stack replication differs from pkv11_q ({float(err):.2e})")
    INFO[li] = dict(order=order, b=b[:, 0, :], step=step, s=s[..., 0], h=h)
    return kq, vq, R, z0


class _Q5Shim(types.ModuleType):
    def __getattr__(self, a):
        return getattr(Q5, a)

    def base(self, name, k, v, li):
        if ST["on"] and name == "kivi128":
            return stack_base(k, v, li)
        return Q5.base(name, k, v, li)


_kivi_sigma2 = S4.kivi_sigma2


def stack_sigma2(k, q, scale, G=128):
    if not ST["on"]:
        return _kivi_sigma2(k, q, scale, G)
    if S4.kivi_sigma2 is not stack_sigma2 or not isinstance(S4.Q5, _Q5Shim):
        raise RuntimeError("sel4_q hooks were replaced by another wrapper module")
    I = INFO[ST["li"]]
    KVH = k.shape[0]
    g = q.shape[0] // KVH
    h = I["h"]
    b = I["b"].repeat_interleave(g, 0)                                     # [H, d]
    order = I["order"].repeat_interleave(g, 0)                             # [H, d]
    qs = (q / b[:, None, :]).gather(-1, order[:, None, :].expand_as(q))
    Hh = Q5.CQ.hadamard(h, q.device)
    ehi = ((qs[..., :h] @ Hh) ** 2).sum(-1)                                # [H, n]
    elo = ((qs[..., h:] @ Hh) ** 2).sum(-1)
    st2 = (I["step"] ** 2).repeat_interleave(g, 0) / 12                    # [H, Lk]
    s2 = (I["s"] ** 2).repeat_interleave(g, 0) * (math.pi / 2 - 1)
    return (ehi[:, :, None] * st2[:, None, :] + elo[:, :, None] * s2[:, None, :]) * (scale * scale)


S4.Q5 = _Q5Shim("pkv5_q_shim10")
S4.kivi_sigma2 = stack_sigma2


_qk_weights = P8.qk_weights


def _qk_weights_safe(a, H, KVH, d):
    """pkv8_q.qk_weights, except that q / k norm gains that are not per-head [d] (OLMo-2 normalises the whole H * d
    projection) are dropped: they only enter the weights-only ranking C8["w"], which the stack (calibration ranking
    "c") never uses; without this calib8 fails on OLMo-2."""
    wq, wk, bk, gq, gk = _qk_weights(a, H, KVH, d)
    if gq is not None and gq.numel() != d:
        gq = None
    if gk is not None and gk.numel() != d:
        gk = None
    return wq, wk, bk, gq, gk


P8.qk_weights = _qk_weights_safe


def calib(li, k, v, q):
    RC.calib(li, k, v, q)                     # pkv7 / pkv6 / pkv5 calibration (incl. KIVI) + rotary means and balances


def calib_all(model, kv, alist):
    """the main line's remaining calibration, in pkv11_lab's order: rankings (pkv8), rotary ranking (pkv9), value ranking
    (pkv10), NoPE layers (pkv11)."""
    P8.calib8(model, kv, alist)
    P9.calib9(kv, alist)
    P10.calib10(kv, alist)
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return P11.set_nope(cfg)


def parse(name):
    s = split(name)
    return S4.parse(s or name)


def eff_bits(name, L, d, g=1):
    s = split(name)
    if not s:
        return S4.eff_bits(name, L, d, g)
    gpu0, tot0 = S4.eff_bits(s, L, d, g)
    gpu = P11.eff_bits(STACK, L, d, g)[0]
    gpu += (16 - gpu) * 3 / L                   # 4 exact sinks here vs the stack's 1 exact token
    return gpu, tot0 - gpu0 + gpu


def attend(q, k, v, li, name, scale, kw):
    s = split(name) if name else None
    if not s:
        return S4.attend(q, k, v, li, name, scale, kw)
    ST["on"], ST["li"] = True, li
    try:
        return S4.attend(q, k, v, li, s, scale, kw)
    finally:
        ST["on"] = False
