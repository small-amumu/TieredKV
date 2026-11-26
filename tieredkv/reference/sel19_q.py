# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""sel19_q (rootcause, round OTD, 2026-11-15): oracle tail diagnostics on top of sel17_q, registered in REG_OTD.md.
A name is a sel14 x2s _h name with lam = 0 followed by one suffix:
  _otd   R plus statistics; the output is computed exactly as sel14_q._attend_h (must equal R bit for bit)
  _otz   oracle denominator-only tail correction: D += sum_U [exp(l_host) - exp(l_base)], numerator unchanged
  _otzn  oracle full tail correction: numerator and denominator use host logits and host values on U
  _otmu  oracle conditional mean shift: U logits s - mu, mu = sum_U p e / sum_U p per (head, row), e = base - exact logit
U = allowed keys that are neither exact (window, sinks) nor stage-1 candidates. Oracle arms use exact keys and host rows
of every token and are upper bounds, never deployable methods. Any other name is passed to sel17_q unchanged.
Statistics (_otd only) are accumulated in STATS per layer over the query rows >= PREFILL["from"] (the rows that are
scored); the lab resets them before and dumps them after every _otd forward.
"""
from __future__ import annotations

import math
import re

import torch

import sel17_q as Q17

Q14, Q13, S12, S10, S4 = Q17.Q14, Q17.Q13, Q17.S12, Q17.S10, Q17.S4
Q5 = S4.Q5
if getattr(S4, "_sel19_installed", False):
    raise RuntimeError("sel19_q imported twice")
S4._sel19_installed = True
SUF = re.compile(r"(.+)_(otd|otz|otzn|otmu)$")
OCALLS = {"otd": 0, "otz": 0, "otzn": 0, "otmu": 0}     # _attend_o chunks per mode (the lab checks >= layers)
STATS: dict = {}                                         # layer -> dict of float64 tensors (see _new_stats)
VBINS = (-30, 10)                                        # floor(log2 v) bins, clamped
NOCT = 16                                                # age octaves 0..15
DEBUG: dict = {"on": False}


def strip19(name):
    m = SUF.fullmatch(name) if name else None
    return (m[1], m[2]) if m else (name, None)


def core19(name):
    return strip19(name)[0]


def strip(name):
    """sel17-compatible (core, P, sp, dith) of the name without its OTD suffix (the lab uses it)."""
    return Q17.strip(core19(name))


def core(name):
    """sel14 core name (the lab uses it for the VX check)."""
    return Q17.core(core19(name))


def _check(name):
    c, mode = strip19(name)
    if mode is None:
        return c
    s = Q14.split(c)
    if s is None or s[3]:
        raise ValueError(f"{name}: OTD suffixes need a plain sel14 _h<rk><rv> name")
    p = S4.parse(s[0])
    if p["lam"] or p.get("jcal") or p.get("mass") or p["eb"] or p["tw"] or p["crit"][:2] != "2s" or not p["K"]:
        raise ValueError(f"{name}: OTD suffixes need an x2s<M> criterion with K > 0 and lam = 0")
    return c


def parse(name):
    c, mode = strip19(name)
    if mode is None:
        return Q17.parse(name)
    return Q14.parse(_check(name))


def split(name):
    return Q17.split(core19(name)) if name else None


def host_bits(name, d):
    return Q17.host_bits(core19(name), d)


def eff_bits(name, L, d, g=1):
    return Q17.eff_bits(core19(name), L, d, g)


def _new_stats(dev):
    nb = VBINS[1] - VBINS[0] + 1
    z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)  # noqa: E731
    return {"set_p": z(3), "set_pe": z(3), "set_pe2": z(3), "set_n": z(3),       # sets U, C, N
            "v_n": z(nb), "v_e2": z(nb), "v_v": z(nb),
            "dz": z(1), "dz_abs": z(1), "rows": z(1),
            "oct_p": z(NOCT), "oct_pe": z(NOCT), "oct_dz": z(NOCT)}


def reset_stats():
    STATS.clear()


def dump_stats():
    return {str(li): {k: v.tolist() for k, v in d.items()} for li, d in sorted(STATS.items())}


def kivi_key_steps(k, G=128):
    """KIVI-128 key group steps as pkv5_q.per_channel / aq: FP16((FP16(max) - FP16(min)) / 3) -> [KVH, nb, d]."""
    Lf, xt = Q5._groups(k, G)
    if not Lf:
        return torch.zeros(k.shape[0], 0, k.shape[-1], device=k.device)
    mn, mx = Q5.f16(xt.amin(-1)), Q5.f16(xt.amax(-1))
    return Q5.f16((mx - mn) / 3)


def proxy_var(qr, st, scale, Lk, g, G=128):
    """v = scale^2 / 12 * sum_c q_c^2 step_c^2 for every (head, row, key) in a full group; 0 elsewhere -> [H, n, Lk]."""
    H, n, _ = qr.shape
    out = torch.zeros(H, n, Lk, device=qr.device)
    nb = st.shape[1]
    if nb:
        v = torch.einsum("hqd,hbd->hqb", qr * qr, (st * st).repeat_interleave(g, 0)) * (scale * scale / 12)
        out[:, :, :nb * G] = v.repeat_interleave(G, -1)
    return out


def _accumulate(li, rows, U, C, N, p, e, v, age):
    """statistics over the scored rows (rows: bool [n]); p, e, v [H, n, Lk]; U, C, N bool [H, n, Lk]; age [n, Lk]."""
    d = STATS.setdefault(li, _new_stats(p.device))
    r = rows[None, :, None]
    p64, e64 = p.double(), e.double()
    for i, m in enumerate((U & r, C & r, N & r)):
        pm = p64 * m
        d["set_p"][i] += pm.sum()
        d["set_pe"][i] += (pm * e64).sum()
        d["set_pe2"][i] += (pm * e64 * e64).sum()
        d["set_n"][i] += m.sum()
    nm = N & r & (v > 0)
    b = torch.clamp(torch.floor(torch.log2(v.clamp_min(1e-30))), VBINS[0], VBINS[1]).long() - VBINS[0]
    nbins = VBINS[1] - VBINS[0] + 1
    d["v_n"] += torch.bincount(b[nm], minlength=nbins).double()
    d["v_e2"] += torch.bincount(b[nm], weights=(e64 * e64)[nm], minlength=nbins)
    d["v_v"] += torch.bincount(b[nm], weights=v.double()[nm], minlength=nbins)
    um = U & r
    dzr = (p64 * torch.expm1(-e64) * um).sum(-1)                       # per (head, row)
    rr = rows[None].expand_as(dzr)
    d["dz"] += dzr[rr].sum()
    d["dz_abs"] += dzr[rr].abs().sum()
    d["rows"] += rr.sum()
    oc = torch.floor(torch.log2(age.clamp_min(1).double())).long().clamp_max(NOCT - 1)[None].expand_as(p)
    w = p64 * um
    d["oct_p"] += torch.bincount(oc[um], weights=w[um], minlength=NOCT)
    d["oct_pe"] += torch.bincount(oc[um], weights=(w * e64)[um], minlength=NOCT)
    d["oct_dz"] += torch.bincount(oc[um], weights=(w * torch.expm1(-e64))[um], minlength=NOCT)


def _attend_o(q, k, v, li, name, scale, kw, rk, rv, mode):
    """sel14_q._attend_h (chk = None, lam = 0) with the OTD oracle arms / statistics."""
    if S4.DIAG["on"]:
        raise RuntimeError("sel19_q: DIAG is not implemented")
    p_ = S4.parse(name)
    if p_["lam"] or p_["crit"][:2] != "2s" or not p_["K"]:
        raise ValueError(f"{name}: OTD needs x2s<M>, K > 0, lam = 0")
    kq, vq, R, _ = S4.Q5.base("kivi128", k, v, li)
    z = 4
    kq[:, :z], vq[:, :z] = k[:, :z], v[:, :z]
    kh = k if rk >= 16 else kq + Q14._res(k - kq, rk)
    vh = v if rv >= 16 else vq + Q14._res(v - vq, rv)
    K, M = p_["K"], int(p_["crit"][2:])
    H_, Lq, _ = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    kx, vx, kqx, vqx, khx, vhx = (t.repeat_interleave(g, 0) for t in (k, v, kq, vq, kh, vh))
    out = torch.empty(H_, Lq, v.shape[-1], device=q.device)
    kpos = torch.arange(Lk, device=q.device)[None, :]
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
    st = kivi_key_steps(k) if mode == "otd" else None
    pf = S4.PREFILL["from"] if S4.PREFILL["on"] else 0

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
        s_base = torch.einsum("hqd,hkd->hqk", qr, kqx) * scale
        se = torch.einsum("hqd,hkd->hqk", qr, kx) * scale
        sh = torch.einsum("hqd,hkd->hqk", qr, khx) * scale
        ex = ((kpos >= (qpos + 1) // R * R) & al)[None].expand_as(s_base) | (kpos < z)[None]
        ex = ex & al[None]
        s = torch.where(ex, se, s_base)
        bad = ~al[None] | ex
        c1 = fin(s).masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        kc = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, c1.topk(min(M, Lk), -1).indices, True)
        kc = kc.repeat_interleave(g, 0) & ~bad
        s1 = torch.where(kc, sh, s)
        c2 = fin(s1).masked_fill(~kc, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        m = torch.zeros_like(c2, dtype=torch.bool).scatter_(-1, c2.topk(min(K, Lk), -1).indices, True).repeat_interleave(g, 0)
        m = m & kc
        sf = torch.where(m | kc, sh, s)
        U = al[None] & ~ex & ~kc
        OCALLS[mode] += 1
        if mode in ("otd", "otmu"):
            pr = fin(sf)
            if mode == "otmu":
                e = s_base - se
                pu = pr * U
                mu = (pu * e).sum(-1, keepdim=True) / pu.sum(-1, keepdim=True).clamp_min(1e-30)
                pr = fin(torch.where(U, sf - mu, sf))
            o = (torch.einsum("hqk,hkd->hqd", pr, vqx) + torch.einsum("hqk,hkd->hqd", pr * ex, vx - vqx)
                 + torch.einsum("hqk,hkd->hqd", pr * m, vhx - vqx))
            out[:, r0:r1] = o
            if mode == "otd":
                rows = qpos[:, 0] >= pf
                if rows.any():
                    e = s_base - se
                    vp = proxy_var(qr, st, scale, Lk, g)
                    N_ = al[None] & ~ex
                    _accumulate(li, rows, U.expand_as(pr), (kc & N_).expand_as(pr), N_.expand_as(pr), pr, e, vp,
                                (qpos - kpos).clamp_min(0))
            if DEBUG["on"]:
                DEBUG.update(U=U, kc=kc, m=m, ex=ex, al=al, s=s, sh=sh, se=se, sf=sf, vqx=vqx, vhx=vhx, vx=vx)
            continue
        zc = cap(sf).masked_fill(~al[None], float("-inf"))
        mxl = zc.amax(-1, keepdim=True)
        a = torch.exp(zc - mxl)
        b = torch.exp(cap(sh) - mxl).masked_fill(~U, 0.0)
        aU = a * U
        N0 = (torch.einsum("hqk,hkd->hqd", a, vqx) + torch.einsum("hqk,hkd->hqd", a * ex, vx - vqx)
              + torch.einsum("hqk,hkd->hqd", a * m, vhx - vqx))
        D = a.sum(-1) + (b - aU).sum(-1)
        if mode == "otzn":
            N0 = N0 + torch.einsum("hqk,hkd->hqd", b, vhx) - torch.einsum("hqk,hkd->hqd", aU, vqx)
        out[:, r0:r1] = N0 / D[..., None]
        if DEBUG["on"]:
            DEBUG.update(U=U, kc=kc, m=m, ex=ex, al=al, s=s, sh=sh, se=se, sf=sf, vqx=vqx, vhx=vhx, vx=vx)
    return out


def attend(q, k, v, li, name, scale, kw):
    c, mode = strip19(name) if name else (name, None)
    if mode is None:
        return Q17.attend(q, k, v, li, name, scale, kw)
    _check(name)
    base, rk, rv, _ = Q14.split(c)

    def inner(q_, k_, v_, li_, name_, scale_, kw_):
        return _attend_o(q_, k_, v_, li_, name_, scale_, kw_, rk, rv, mode)

    S4._attend = inner
    try:
        return Q13.attend(q, k, v, li, base, scale, kw)
    finally:
        S4._attend = Q14._s4_attend_inner


calib, calib_all = Q17.calib, Q17.calib_all
RC, Q7, P11, P10, P9, P8, S0 = Q17.RC, Q17.Q7, Q17.P11, Q17.P10, Q17.P9, Q17.P8, Q17.S0
PREFILL, CHUNK, ST, INFO, CLIP, VX, HCHECK = Q17.PREFILL, Q17.CHUNK, Q17.ST, Q17.INFO, Q17.CLIP, Q17.VX, Q17.HCHECK
DCALLS, PCALLS, XCALLS = Q17.DCALLS, Q17.PCALLS, Q17.XCALLS
