# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""sel27_q (rootcause, round ACS, 2026-11-18): sel25_q plus ASYNCHRONOUS stage-1 candidates, registered in REG_ACS.md.

Question: if the host fetch is taken off the critical path, the rows that can be corrected at decode step t are the
rows that were requested from what earlier steps saw. These arms emulate that with teacher-forced rows (one query
row = one decode step, consecutive rows = consecutive steps). Only the choice of the stage-1 candidate rows
changes; the base, the residual codec, stage 2 (K value rows among the candidates), the exact window, the sinks
and the tail logits are R's (sel25_q's _r0 path). New suffixes after a plain x2s<M> _h<rk><rv> name:
  _lg<d>   lag: the candidates of row t are the top M rows of the BASE attention weights of row t - d
           (d = 0 is R itself: control, must equal _r0 bit for bit)
  _cw<w>   window: top M of the mean base weights over rows t - w .. t - 1 (lag 1, w >= 2)
  _cb<n>   bigger cache: top M + n of the base weights of row t - 1 (compare with sel25_q's _ba<n>: M + n rows
           chosen by the CURRENT row's base weights)
  _nc0     no candidates at all: nothing is read from the host (the 2-bit base plus the exact window and sinks);
           the reference for "how much of R's benefit an arm loses"
"Base attention weights" are the weights R ranks by: softmax over the allowed keys of the base scores (exact
scores on the exact window and the sinks), averaged over the query heads of a KV head. The history keeps the
weights of every allowed key, including keys that were in the exact window at the earlier row. At row t a key is
a candidate only if it is allowed, not exact at row t, and had a non-zero weight in the history (a key that did not
exist at the earlier rows cannot have been requested). History is carried across the row chunks of one call. The
first T = d + w - 1 computed rows of a call have no full history (the first computed row's own weights stand in):
to keep them out of the scored rows, the last prefill chunk that reaches into the T rows before PREFILL["from"]
is computed as well (its output rows are overwritten by exact attention, as every prefill row), so every scored
row has its full history whenever the call starts at least T rows before PREFILL["from"].
_lg0 takes R's own candidates directly (it does not go through the history code): it checks everything after the
choice of candidates, not the history indexing (that is checked by sel27_selftest.py against a loop).
Statistics per layer over the scored rows (rows >= PREFILL["from"]), STATS27[li] = float64 sums
  [hit, true, mass_async, mass_true, rows, new, rows_prev, cand]: hit = |async candidates and R's candidates|,
  true = |R's candidates|, mass_* = base weight summed over the async / over R's candidates, rows = scored rows x
  KV heads, new = |R's candidates of row t that were not R's candidates of row t - 1| (the rows a cache would have
  to load at this step: churn; the same in every arm), rows_prev = the rows x KV heads that have a previous row in
  the call, cand = |async candidates| (to check the row budget).
Everything else is sel25_q, reached through this module unchanged.
"""
from __future__ import annotations

import re

import torch

import sel25_q as Q25

Q19, Q17, Q14, Q13, S12, S10, S4 = Q25.Q19, Q25.Q17, Q25.Q14, Q25.Q13, Q25.S12, Q25.S10, Q25.S4
Q5 = S4.Q5
if getattr(S4, "_sel27_installed", False):
    raise RuntimeError("sel27_q imported twice")
S4._sel27_installed = True
SUF27 = re.compile(r"(.+)_(lg|cw|cb|nc)(0|[1-9]\d*)$")
KINDS27 = ("lg", "cw", "cb", "nc")
OCALLS = Q25.OCALLS                                     # shared with sel25 / sel19 (the lab checks every key)
OCALLS.update({k_: 0 for k_ in KINDS27})
STATS27: dict = {}                                     # layer -> float64 tensor [5]
DEBUG27: dict = {"on": False}
ZS = Q25.ZS
MAXH = 64                                              # largest lag / window


def strip27(name):
    """(core, kind, N) for a sel27 name, else (name, None, 0)."""
    m = SUF27.fullmatch(name) if name else None
    if not m:
        return name, None, 0
    c, kind, n = m[1], m[2], int(m[3])
    if (kind == "lg" and n > MAXH) or (kind == "cw" and not 2 <= n <= MAXH) or (kind == "cb" and n < 1) \
            or (kind == "nc" and n != 0):
        return name, None, 0
    return c, kind, n


def alias(name):
    """the sel25_q name with the same host bytes: _ba<n> for _cb<n> (M + n key rows), _r0 for the others."""
    c, kind, n = strip27(name)
    if kind is None:
        return name
    return f"{c}_ba{n}" if kind == "cb" else f"{c}_r0"


def strip19(name):
    c, kind, _ = strip27(name)
    return (c, kind) if kind else Q25.strip19(name)


def strip22(name):
    return Q25.strip22(alias(name))


def strip(name):
    return Q25.strip(alias(name))


def core(name):
    return Q25.core(alias(name))


def split(name):
    return Q25.split(alias(name)) if name else None


def parse(name):
    return Q25.parse(alias(name))


def host_bits(name, d):
    return Q25.host_bits(alias(name), d)


def eff_bits(name, L, d, g=1):
    return Q25.eff_bits(alias(name), L, d, g)


def sketch_bits(name, d):
    return Q25.sketch_bits(alias(name), d)


def reset_stats27():
    STATS27.clear()


def dump_stats27():
    return {str(li): STATS27[li].tolist() for li in sorted(STATS27)}


def _attend_a(q, k, v, li, name, scale, kw, rk, rv, kind, N):
    """sel25_q._attend_t's _r0 path with the stage-1 candidates taken from earlier rows (see the module docstring)."""
    if S4.DIAG["on"]:
        raise RuntimeError("sel27_q: DIAG is not implemented")
    p_ = S4.parse(name)
    if p_["lam"] or p_["crit"][:2] != "2s" or not p_["K"]:
        raise ValueError(f"{name}: ACS needs x2s<M>, K > 0, lam = 0")
    kq, vq, R, _ = S4.Q5.base("kivi128", k, v, li)
    z = ZS
    kq[:, :z], vq[:, :z] = k[:, :z], v[:, :z]
    kh = k if rk >= 16 else kq + Q14._res(k - kq, rk)
    vh = v if rv >= 16 else vq + Q14._res(v - vq, rv)
    K, M = p_["K"], int(p_["crit"][2:])
    lag, win, C = {"lg": (N, 1, M), "cw": (1, N, M), "cb": (1, 1, M + N), "nc": (0, 1, 0)}[kind]
    T = lag + win - 1 if lag else 0                    # history rows needed before a row
    prev_true = None                                   # R's candidates of the last row of the previous chunk
    H_, Lq, _ = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    kx, vx, kqx, vqx, khx, vhx = (t.repeat_interleave(g, 0) for t in (k, v, kq, vq, kh, vh))
    out = torch.empty(H_, Lq, v.shape[-1], device=q.device)
    kpos = torch.arange(Lk, device=q.device)[None, :]
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
    pf = S4.PREFILL["from"] if S4.PREFILL["on"] else 0
    carry = None
    st = STATS27.setdefault(li, torch.zeros(8, dtype=torch.float64, device=q.device))

    for r0 in range(0, Lq, S4.CHUNK["rows"]):
        r1 = min(Lq, r0 + S4.CHUNK["rows"])
        n = r1 - r0
        if Lk - Lq + r1 <= pf - T:                       # chunk in the dense prefill and not needed as history
            out[:, r0:r1] = 0
            Q25.SKIPPED["n"] += 1
            continue
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
        w_all = fin(s)
        c1 = w_all.masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        kc_true = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, c1.topk(min(M, Lk), -1).indices, True)
        badk = bad[::g]                                  # [KVH, n, Lk]: the mask does not depend on the head
        if kind == "nc":
            kc = torch.zeros_like(kc_true)
        elif lag == 0:
            kc = kc_true
        else:
            hist = w_all.reshape(KVH, g, n, Lk).mean(1)  # weights of every allowed key, exact window included
            if carry is None:                            # first computed chunk: its first row stands in
                carry = hist[:, :1].expand(-1, T, -1)
            ext = torch.cat([carry, hist], 1)            # [KVH, T + n, Lk]; row t of the chunk is ext[T + t]
            if win == 1:
                prior = ext[:, T - lag:T - lag + n]
            else:
                cs = torch.cat([torch.zeros_like(ext[:, :1], dtype=torch.float64), ext.double().cumsum(1)], 1)
                hi = torch.arange(n, device=q.device) + T - lag + 1
                prior = ((cs[:, hi] - cs[:, hi - win]) / win).float()
            carry = ext[:, ext.shape[1] - T:].clone()
            pc = prior.masked_fill(badk | ~(prior > 0), -float("inf"))
            kc = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, pc.topk(min(C, Lk), -1).indices, True)
            kc = kc & torch.isfinite(pc)                 # fewer than C requestable rows: no filler rows
        rows = (qpos[:, 0] >= pf)[None, :, None]
        c1p = torch.where(torch.isfinite(c1), c1, torch.zeros_like(c1)).double()
        kt, ka = kc_true & ~badk & rows, kc & ~badk & rows
        ktm = kc_true & ~badk                            # R's candidates of every row of the chunk
        before = torch.cat([ktm[:, :1] if prev_true is None else prev_true, ktm[:, :-1]], 1)
        has_prev = torch.ones(n, dtype=torch.bool, device=q.device)
        if prev_true is None:
            has_prev[0] = False                          # the first computed row of the call has no previous row
        rp = rows & has_prev[None, :, None]
        prev_true = ktm[:, -1:].clone()
        st += torch.stack([(kt & ka).sum().double(), kt.sum().double(), (c1p * ka).sum(), (c1p * kt).sum(),
                           (rows.sum() * KVH).double(), (ktm & ~before & rp).sum().double(),
                           (rp.sum() * KVH).double(), ka.sum().double()])
        if DEBUG27["on"]:
            DEBUG27.update(kc=kc.clone(), kc_true=kc_true.clone(), badk=badk.clone(), c1=c1.clone(), qpos=qpos.clone())
        kc = kc.repeat_interleave(g, 0) & ~bad
        s1 = torch.where(kc, sh, s)
        c2 = fin(s1).masked_fill(~kc, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        m = torch.zeros_like(c2, dtype=torch.bool).scatter_(-1, c2.topk(min(K, Lk), -1).indices, True).repeat_interleave(g, 0)
        m = m & kc
        sf = torch.where(m | kc, sh, s)
        OCALLS[kind] += 1
        pr = fin(sf)
        o = (torch.einsum("hqk,hkd->hqd", pr, vqx) + torch.einsum("hqk,hkd->hqd", pr * ex, vx - vqx)
             + torch.einsum("hqk,hkd->hqd", pr * m, vhx - vqx))
        out[:, r0:r1] = o
    return out


def attend(q, k, v, li, name, scale, kw):
    c, kind, N = strip27(name) if name else (name, None, 0)
    if kind is None:
        return Q25.attend(q, k, v, li, name, scale, kw)
    Q25._check(c + "_r0")
    base, rk, rv, _ = Q14.split(c)

    def inner(q_, k_, v_, li_, name_, scale_, kw_):
        return _attend_a(q_, k_, v_, li_, name_, scale_, kw_, rk, rv, kind, N)

    S4._attend = inner
    try:
        return Q13.attend(q, k, v, li, base, scale, kw)
    finally:
        S4._attend = Q14._s4_attend_inner


calib, calib_all = Q25.calib, Q25.calib_all
RC, Q7, P11, P10, P9, P8, S0 = Q25.RC, Q25.Q7, Q25.P11, Q25.P10, Q25.P9, Q25.P8, Q25.S0
PREFILL, CHUNK, ST, INFO, CLIP, VX, HCHECK = Q25.PREFILL, Q25.CHUNK, Q25.ST, Q25.INFO, Q25.CLIP, Q25.VX, Q25.HCHECK
DCALLS, PCALLS, XCALLS = Q25.DCALLS, Q25.PCALLS, Q25.XCALLS
STATS, SKIPPED = Q25.STATS, Q25.SKIPPED
reset_stats, dump_stats, dump_stats22 = Q25.reset_stats, Q25.dump_stats, Q25.dump_stats22
