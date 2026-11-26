# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""sel17_q (rootcause, round TPS, 2026-11-14): three additions to sel14_q's deployed two-stage fetch, registered in
REG_TPS.md. A name is a sel14_q name followed by optional suffixes, in this order:
  <sel14 name>[_tp<P>][_sp2][_dn|_dx]
  _tp<P>  tail probes: after the two-stage selection, P rows are drawn with replacement from the unselected, non-exact
          keys U of each KV head (probability = group-mean base softmax restricted to U); each probe reads its host key
          and host value (same residual codes as the fetched rows). The softmax denominator and numerator get the
          Hansen-Hurwitz (with-replacement Horvitz-Thompson) estimate of sum_U [exp(l_host) - exp(l_base)] (and the same
          times the values), so the tail uses the base as a control variate. Bytes: the arm's name must already have M and
          K reduced by P (e.g. kivi128k64_x2s144_h88_tp16 reads 160 key and 80 value residual rows, as kivi128k80_x2s160_h88).
  _sp2    log-age (octave) chambers: both selection stages (M candidates, K values) give each non-empty chamber
          floor(age) in [2^j, 2^(j+1)) an equal share of the slots, ranked inside the chamber by the same score; unused
          slots go back to a global top-k. Chamber ratio 2 is fixed (registration), not fitted.
  _dn     subtractive-dither KIVI-128 key base (no overload: range widened by one step; dither regenerated from a fixed
          per-layer seed, never stored); values and everything else as KIVI-128.
  _dx     _dn plus the exact closed-form debias of the unrefined base logits: log E[exp(e)] = sum_c log(sinh(a_c) / a_c),
          a_c = scale * q_c * step_c / 2 (step of the token's 128-token group in channel c). Needs lam = 0.
Also "kivi128_dn": the dithered base alone (no fetch). With no suffix a name is exactly sel14_q's (same function).
Simulation note: the dither is drawn with torch.Generator per layer (the deployed design regenerates it from a
counter-based hash); in this harness both sides see the same tensor, which is what matters for the error model.
"""
from __future__ import annotations

import math
import re

import torch

import sel14_q as Q14

Q13, S12, S10, S4 = Q14.Q13, Q14.S12, Q14.S10, Q14.S4
Q5 = S4.Q5
if getattr(S4, "_sel17_installed", False):
    raise RuntimeError("sel17_q imported twice")
S4._sel17_installed = True
_base0 = Q5.base
DSTEP: dict = {}                    # layer -> group steps [KVH, nb, d] of the last dithered base call
DCALLS = {"n": 0}                   # dithered base calls (the lab checks one per attention layer per forward)
PCALLS = {"n": 0, "rows": 0, "empty": 0}    # probe calls / query rows probed / rows with an empty tail
XCALLS = {"sp": 0, "dx": 0}                 # _attend_x chunks that ran the stratified selection / the exact debias
DEBUG: dict = {"on": False}                 # self-test hook (sel17_selftest.py); never on in the lab
SUF = re.compile(r"_(tp\d+|sp2|dn|dx)$")


def strip(name):
    """-> core sel14 name, P (probes), sp (bool), dith (None / 'dn' / 'dx'); suffix order enforced."""
    P, sp, dith, rest, order = 0, False, None, name, []
    while True:
        m = SUF.search(rest)
        if not m:
            break
        t, rest = m[1], rest[:m.start()]
        order.append(t)
        if t in ("dn", "dx"):
            if dith or sp or P:
                raise ValueError(f"{name}: suffix order is _tp<P>, _sp2, _dn|_dx")
            dith = t
        elif t == "sp2":
            if sp or P:
                raise ValueError(f"{name}: suffix order is _tp<P>, _sp2, _dn|_dx")
            sp = True
        else:
            if P:
                raise ValueError(f"{name}: one _tp only")
            P = int(t[2:])
            if P < 1:
                raise ValueError(f"{name}: _tp needs P >= 1")
    return rest, P, sp, dith


def core(name):
    return strip(name)[0] if name else name


def _check(name):
    c, P, sp, dith = strip(name)
    if not (P or sp or dith):
        return c
    s = Q14.split(c)
    if s is None:
        if c == "kivi128" and dith == "dn" and not P and not sp:
            return c
        raise ValueError(f"{name}: _tp / _sp2 / _dx need a sel14 _h<rk><rv> name; kivi128_dn is the only base-only form")
    if s[3]:
        raise ValueError(f"{name}: no _id / _lk checks with the TPS suffixes")
    p = S4.parse(s[0])
    if dith and p["lam"]:
        raise ValueError(f"{name}: dithered arms need lam = 0 (no Jensen debias)")
    if P and dith == "dx":
        raise ValueError(f"{name}: _tp and _dx are separate arms")
    return c


def parse(name):
    return Q14.parse(_check(name))


def split(name):
    return Q14.split(core(name)) if name else None


def host_bits(name, d):
    return Q14.host_bits(core(name), d)


def eff_bits(name, L, d, g=1):
    c, P, _, _ = strip(name)
    gpu, tot = Q14.eff_bits(c, L, d, g)
    if P:
        s = Q14.split(c)
        tot += P * (Q14._rowbits(s[1], d) + Q14._rowbits(s[2], d)) / (2 * L)
    return gpu, tot


# ---------------------------------------------------------------- dithered KIVI key base
def dither_k(k, bits, G, li):
    """per-channel groups of G tokens (as pkv5_q.per_channel), subtractive dither without overload.
    -> k_hat (tail L % G exact), steps [KVH, nb, d] (0 for zero-range groups, which stay exact)."""
    Lf, xt = Q5._groups(k, G)
    out = k.clone()
    if not Lf:
        return out, torch.zeros(k.shape[0], 0, k.shape[-1], device=k.device)
    lv = 2 ** bits - 1
    mn, mx = xt.amin(-1, keepdim=True), xt.amax(-1, keepdim=True)
    st = Q5.f16((mx - mn) / (lv - 1))                                      # one extra step of range: no overload
    lo = Q5.f16(mn - st / 2)
    gen = torch.Generator(device=k.device).manual_seed(7919 * int(li) + 1)
    u = (torch.rand(xt.shape, generator=gen, device=k.device) - 0.5) * st
    safe = torch.where(st > 0, st, torch.ones_like(st))
    c = torch.clamp(torch.round((xt + u - lo) / safe), 0, lv)
    xh = torch.where(st > 0, c * st + lo - u, xt)
    return Q5._ungroup(out, xh, Lf), st.squeeze(-1)


def _base_dith(name, k, v, li):
    if name != "kivi128":
        raise ValueError(f"dithered base is defined for kivi128 only, got {name}")
    _, vb, R, sel = _base0(name, k, v, li)
    kb, st = dither_k(k, 2, 128, li)
    DSTEP[li] = st
    DCALLS["n"] += 1
    return kb, vb, R, sel


def logsinhc(a):
    """log(sinh(a) / a), even in a, stable for small and large |a|."""
    a = a.abs()
    big = a + torch.log1p(-torch.exp(-2 * a.clamp_min(1e-4))) - torch.log(2 * a.clamp_min(1e-4))
    return torch.where(a < 1e-2, a * a / 6 - a ** 4 / 180, big)


def dither_logmgf(qr, st, scale, Lk, g, G=128):
    """exact log E[exp(e)] of the dithered key base for every (head, row, key) -> [H, n, Lk] (0 for keys outside full groups)."""
    H, n, d = qr.shape
    nb = st.shape[1]
    out = torch.zeros(H, n, Lk, device=qr.device)
    if not nb:
        return out
    stx = st.repeat_interleave(g, 0)                                      # [H, nb, d]
    f = torch.zeros(H, n, nb, device=qr.device)
    for c0 in range(0, d, 32):                                            # chunk channels to bound memory
        a = (scale / 2) * qr[:, :, None, c0:c0 + 32] * stx[:, None, :, c0:c0 + 32]
        f += logsinhc(a).sum(-1)
    out[:, :, :nb * G] = f.repeat_interleave(G, -1)
    return out


# ---------------------------------------------------------------- age-stratified top-k
def strat_topk(score, k, age):
    """score [B, n, Lk] (-inf = not eligible), age [n, Lk] (>= 0) -> bool mask: each non-empty octave chamber
    floor(log2(max(age, 1))) gets floor(k / J) slots (J = non-empty chambers of the row), then a global top-k fills the rest."""
    B, n, Lk = score.shape
    kk = min(k, Lk)
    ok = torch.isfinite(score)
    oc = torch.floor(torch.log2(age.clamp_min(1).double())).long()[None].expand(B, n, Lk)
    nch = int(oc.max()) + 1
    J = torch.zeros(B, n, dtype=torch.long, device=score.device)
    for j in range(nch):
        J += (ok & (oc == j)).any(-1).long()
    share = k // J.clamp_min(1)
    sel = torch.zeros_like(ok)
    rank = torch.arange(kk, device=score.device)
    for j in range(nch):
        v, idx = score.masked_fill(oc != j, float("-inf")).topk(kk, -1)
        keep = (rank < share[..., None]) & torch.isfinite(v)
        sel |= torch.zeros_like(ok).scatter(-1, idx, keep)
    rem = k - sel.sum(-1)
    v, idx = score.masked_fill(sel, float("-inf")).topk(kk, -1)
    keep = (rank < rem[..., None]) & torch.isfinite(v)
    sel |= torch.zeros_like(ok).scatter(-1, idx, keep)
    return sel & ok


# ---------------------------------------------------------------- Hansen-Hurwitz tail estimate
def hh_tail(a_i, b_i, vq_i, vh_i, p_i):
    """with-replacement draws (last dim of a_i = P): a_i / b_i base / host unnormalised weights [H, n, P], values
    [H, n, P, d], draw probabilities p_i [H, n, P] -> (dD [H, n], dN [H, n, d]) unbiased for sum_U (b - a), sum_U (b vh - a vq)."""
    w = 1.0 / p_i
    dD = ((b_i - a_i) * w).mean(-1)
    dN = ((b_i[..., None] * vh_i - a_i[..., None] * vq_i) * w[..., None]).mean(-2)
    return dD, dN


def _gather_rows(x, idx):
    """x [H, Lk, d], idx [H, n, P] -> [H, n, P, d]"""
    H, n, P = idx.shape
    return x.gather(1, idx.reshape(H, n * P, 1).expand(-1, -1, x.shape[-1])).reshape(H, n, P, x.shape[-1])


def _attend_x(q, k, v, li, name, scale, kw, rk, rv, P, sp, dx):
    """sel14_q._attend_h (chk = None) with the TPS additions; with P = 0, sp = dx = False it computes the same output."""
    if S4.DIAG["on"]:
        raise RuntimeError("sel17_q: DIAG is not implemented")
    p = S4.parse(name)
    if p.get("jcal") or p.get("mass") or p["eb"] or p["tw"] or p["crit"][:2] != "2s" or not p["K"]:
        raise ValueError(f"{name}: needs an x2s<M> criterion with K > 0 and a plain / _j / _p correction")
    kq, vq, R, _ = S4.Q5.base("kivi128", k, v, li)
    z = 4
    kq[:, :z], vq[:, :z] = k[:, :z], v[:, :z]
    kh = k if rk >= 16 else kq + Q14._res(k - kq, rk)
    vh = v if rv >= 16 else vq + Q14._res(v - vq, rv)
    k_enc = k
    K, M, lam = p["K"], int(p["crit"][2:]), p["lam"]
    if dx and lam:
        raise ValueError(f"{name}: _dx needs lam = 0")
    H_, Lq, _ = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    kx, vx, kqx, vqx, khx, vhx = (t.repeat_interleave(g, 0) for t in (k, v, kq, vq, kh, vh))
    out = torch.empty(H_, Lq, v.shape[-1], device=q.device)
    kpos = torch.arange(Lk, device=q.device)[None, :]
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
    st = DSTEP.get(li) if dx else None
    if dx and st is None:
        raise RuntimeError(f"{name}: _dx without a dithered base call for layer {li}")

    def cap(s):
        return torch.tanh(s / softcap) * softcap if softcap else s

    for r0 in range(0, Lq, S4.CHUNK["rows"]):
        r1 = min(Lq, r0 + S4.CHUNK["rows"])
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
        sh = torch.einsum("hqd,hkd->hqk", qr, khx) * scale
        ex = ((kpos >= (qpos + 1) // R * R) & al)[None].expand_as(s) | (kpos < z)[None]
        ex = ex & al[None]
        s = torch.where(ex, se, s)
        s2 = S4.kivi_sigma2(k_enc, q[:, r0:r1], scale) if lam else None
        if s2 is not None:
            s2 = s2.masked_fill(ex, 0.0)
        bad = ~al[None] | ex
        age = (qpos - kpos).clamp_min(0)                                   # [n, Lk]
        c1 = fin(s).masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        if sp:
            kc = strat_topk(c1, M, age)
        else:
            kc = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, c1.topk(min(M, Lk), -1).indices, True)
        kc = kc.repeat_interleave(g, 0) & ~bad
        s1 = torch.where(kc, sh, s - (lam / 100) * s2 / 2 if lam else s)
        c2 = fin(s1).masked_fill(~kc, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        if sp:
            m = strat_topk(c2, K, age).repeat_interleave(g, 0)
            XCALLS["sp"] += 1
        else:
            m = torch.zeros_like(c2, dtype=torch.bool).scatter_(-1, c2.topk(min(K, Lk), -1).indices, True).repeat_interleave(g, 0)
        m = m & kc
        sf = torch.where(m | kc, sh, s)
        if lam:
            sf = torch.where(m | ex | kc, sf, sf - (lam / 100) * s2 / 2)
        if dx:
            sf = torch.where(m | ex | kc, sf, sf - dither_logmgf(qr, st, scale, Lk, g))
            XCALLS["dx"] += 1
        if DEBUG["on"]:                                                    # self-test only: masks of the last chunk
            DEBUG.update(kc=kc, m=m, ex=ex, al=al, s=s, sh=sh, se=se, sf=sf, kqx=kqx, vqx=vqx, vhx=vhx, vx=vx)
        if not P:
            pr = fin(sf)
            o = (torch.einsum("hqk,hkd->hqd", pr, vqx) + torch.einsum("hqk,hkd->hqd", pr * ex, vx - vqx)
                 + torch.einsum("hqk,hkd->hqd", pr * m, vhx - vqx))
            out[:, r0:r1] = o
            continue
        # tail probes: U = allowed, not exact, not a candidate (lam = 0 here would keep sf = s on U; with lam the
        # control variate is the corrected base, which is still a valid difference estimator)
        U = al[None] & ~ex & ~kc
        zc = cap(sf).masked_fill(~al[None], float("-inf"))
        mxl = zc.amax(-1, keepdim=True)
        a = torch.exp(zc - mxl)                                            # 0 where not allowed
        N0 = (torch.einsum("hqk,hkd->hqd", a, vqx) + torch.einsum("hqk,hkd->hqd", a * ex, vx - vqx)
              + torch.einsum("hqk,hkd->hqd", a * m, vhx - vqx))
        D0 = a.sum(-1)
        DnU = (a * ~U).sum(-1)
        w = (a / D0[..., None]).masked_fill(~U, 0.0).reshape(KVH, g, n, Lk).mean(1)   # group-mean base softmax on U
        tot = w.sum(-1, keepdim=True)
        valid = tot.squeeze(-1) > 0                                        # [KVH, n]
        pw = torch.where(tot > 0, w / tot.clamp_min(1e-30), torch.full_like(w, 1.0 / Lk))
        gen = torch.Generator(device=q.device).manual_seed(1000003 * int(li) + 7919 * r0 + Lq)
        idx = torch.multinomial(pw.reshape(KVH * n, Lk), P, replacement=True, generator=gen).reshape(KVH, n, P)
        pi = pw.gather(-1, idx)
        idx, pi, valid_h = idx.repeat_interleave(g, 0), pi.repeat_interleave(g, 0), valid.repeat_interleave(g, 0)
        a_i = a.gather(-1, idx)
        b_i = torch.exp(cap(sh).gather(-1, idx) - mxl)
        dD, dN = hh_tail(a_i, b_i, _gather_rows(vqx, idx), _gather_rows(vhx, idx), pi)
        dD, dN = dD * valid_h, dN * valid_h[..., None]
        D = torch.maximum(D0 + dD, DnU)
        out[:, r0:r1] = (N0 + dN) / D[..., None]
        PCALLS["n"] += 1
        PCALLS["rows"] += n
        PCALLS["empty"] += int((~valid).sum())
    return out


def attend(q, k, v, li, name, scale, kw):
    c, P, sp, dith = strip(name) if name else (name, 0, False, None)
    if not (P or sp or dith):
        return Q14.attend(q, k, v, li, name, scale, kw)
    _check(name)
    if dith:
        Q5.base = _base_dith
    try:
        if not (P or sp or dith == "dx"):
            return Q14.attend(q, k, v, li, c, scale, kw)                   # _dn only: sel14 path on the dithered base
        base, rk, rv, _ = Q14.split(c)

        def inner(q_, k_, v_, li_, name_, scale_, kw_):
            return _attend_x(q_, k_, v_, li_, name_, scale_, kw_, rk, rv, P, sp, dith == "dx")

        S4._attend = inner
        try:
            return Q13.attend(q, k, v, li, base, scale, kw)
        finally:
            S4._attend = Q14._s4_attend_inner
    finally:
        Q5.base = _base0


calib, calib_all = Q14.calib, Q14.calib_all
RC, Q7, P11, P10, P9, P8, S0 = Q14.RC, Q14.Q7, Q14.P11, Q14.P10, Q14.P9, Q14.P8, Q14.S0
PREFILL, CHUNK, ST, INFO, CLIP, VX, HCHECK = Q14.PREFILL, Q14.CHUNK, Q14.ST, Q14.INFO, Q14.CLIP, Q14.VX, Q14.HCHECK
