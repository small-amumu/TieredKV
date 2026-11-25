# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""sel28_q (rootcause, round SPD, 2026-11-18): sel27_q plus stage-1 candidates restricted to a PREDICTED POSITION SET,
registered in REG_SPD.md.

Question (idea: "like speculative decoding, with a small network"): can something cheap tell the large model,
before its own step, which cached positions it will want to refine - so that their residuals can be fetched ahead
of time? The predictor here gives, for every decode step (query row), one ranked list of cached positions shared
by all layers and KV heads. At row t the large model still ranks all rows by its own base weights (that is local
GPU work), but only rows whose position is in the first C entries of the list can be corrected (their residuals
are "in the cache"); everything else is R's _r0 path (base, residual codec, stage 2 among the candidates, exact
window, sinks, tail logits). New suffixes after a plain x2s<M> _h<rk><rv> name:
  _xa<C>   list A at the SAME row: the list written by another (smaller, same-tokenizer) model on the same token
           ids - the draft model of speculative decoding, which runs ahead of the large model
  _xb<C>   list B at the PREVIOUS row (t - 1): the large model's own list of the step before - what it could
           know without any other model (position-level counterpart of sel27's _lg1)
  _xc<C>   list B at the same row: an oracle, the best a position list of size C shared by all layers and heads
           can do (ceiling of _xa)
  _xd<C>   as _xa with a second draft model's list D
  _xe<C>   list E at the same row: the large model's own exact attention summed over its first quarter of
           attention layers only (a draft that needs no second model: the lower layers run before the upper ones;
           here the list is applied to every layer, the lower quarter included, which favours it there)
A list is made by the lab (sel28_lab.py) from exact attention: the probability mass a position receives, summed
over all layers and heads, ranked among the positions that are compressed at that row (not sinks, not in the
exact window, not future). POS holds the lists of the current segment: POS["a"], POS["d"], POS["b"], POS["e"] = int64 [rows, Cmax]
(row r is query position POS["row0"] + r), set by the lab before each forward.
Candidates of row t: the top M by R's base weights among the allowed, non-exact keys whose position is in the list.
Statistics per layer over the scored rows, STATS27 layout (sel27_q): [hit, true, mass, mass_true, rows, 0, 0, cand]
with hit = R's candidates that are in the list (= coverage; they are all corrected), cand = corrected rows.
Everything else is sel27_q / sel25_q, reached through this module unchanged.
"""
from __future__ import annotations

import re

import torch

import sel27_q as Q27

Q25 = Q27.Q25
Q19, Q17, Q14, Q13, S12, S10, S4 = Q25.Q19, Q25.Q17, Q25.Q14, Q25.Q13, Q25.S12, Q25.S10, Q25.S4
Q5 = S4.Q5
if getattr(S4, "_sel28_installed", False):
    raise RuntimeError("sel28_q imported twice")
S4._sel28_installed = True
SUF28 = re.compile(r"(.+)_(xa|xd|xb|xc|xe)([1-9]\d*)$")
KINDS28 = ("xa", "xd", "xb", "xc", "xe")
SRC = {"xa": ("a", 0), "xd": ("d", 0), "xb": ("b", 1), "xc": ("b", 0), "xe": ("e", 0)}   # kind -> (list, lag in rows)
OCALLS = Q25.OCALLS
OCALLS.update({k_: 0 for k_ in KINDS28})
STATS27 = Q27.STATS27
POS: dict = {"a": None, "d": None, "b": None, "e": None, "row0": 0}
DEBUG28: dict = {"on": False}
ZS = Q25.ZS


def strip28(name):
    """(core, kind, C) for a sel28 name, else (name, None, 0)."""
    m = SUF28.fullmatch(name) if name else None
    if not m:
        return name, None, 0
    return m[1], m[2], int(m[3])


def alias(name):
    """the sel25 name with R's host bytes per step (the lists only restrict which rows may be corrected)."""
    c, kind, _ = strip28(name)
    return f"{c}_r0" if kind else Q27.alias(name)


def strip19(name):
    c, kind, _ = strip28(name)
    return (c, kind) if kind else Q27.strip19(name)


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


def _attend_p(q, k, v, li, name, scale, kw, rk, rv, kind, C):
    """sel25_q._attend_t's _r0 path with the stage-1 candidates restricted to the position list (module docstring)."""
    if S4.DIAG["on"]:
        raise RuntimeError("sel28_q: DIAG is not implemented")
    p_ = S4.parse(name)
    if p_["lam"] or p_["crit"][:2] != "2s" or not p_["K"]:
        raise ValueError(f"{name}: SPD needs x2s<M>, K > 0, lam = 0")
    src, lag = SRC[kind]
    sets, row0 = POS[src], POS["row0"]
    if sets is None:
        raise RuntimeError(f"{name}: no position list {src!r} for this forward")
    if C > sets.shape[1]:
        raise ValueError(f"{name}: the list has only {sets.shape[1]} entries per row")
    kq, vq, R, _ = S4.Q5.base("kivi128", k, v, li)
    z = ZS
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
    pf = S4.PREFILL["from"] if S4.PREFILL["on"] else 0
    st = STATS27.setdefault(li, torch.zeros(8, dtype=torch.float64, device=q.device))

    for r0 in range(0, Lq, S4.CHUNK["rows"]):
        r1 = min(Lq, r0 + S4.CHUNK["rows"])
        n = r1 - r0
        if Lk - Lq + r1 <= pf:                           # chunk entirely in the dense prefill: overwritten later
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
        c1 = fin(s).masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        kc_true = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, c1.topk(min(M, Lk), -1).indices, True)
        badk = bad[::g]
        idx = qpos[:, 0] - lag - row0                    # rows of the list used by the rows of this chunk
        scored = qpos[:, 0] >= pf
        if bool(((idx < 0) | (idx >= sets.shape[0]))[scored].any()):
            raise RuntimeError(f"{name}: the position list does not cover the scored rows (row0 {row0}, {sets.shape[0]} rows)")
        S = sets[idx.clamp(0, sets.shape[0] - 1), :C].to(q.device)
        in_s = torch.zeros(n, Lk, dtype=torch.bool, device=q.device).scatter_(1, S, True)
        pc = c1.masked_fill(~in_s[None] | badk, -float("inf"))
        kc = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, pc.topk(min(M, Lk), -1).indices, True)
        kc = kc & torch.isfinite(pc)
        rows = scored[None, :, None]
        c1p = torch.where(torch.isfinite(c1), c1, torch.zeros_like(c1)).double()
        kt, ka = kc_true & ~badk & rows, kc & ~badk & rows
        zero = torch.zeros((), dtype=torch.float64, device=q.device)
        st += torch.stack([(kt & ka).sum().double(), kt.sum().double(), (c1p * ka).sum(), (c1p * kt).sum(),
                           (rows.sum() * KVH).double(), zero, zero, ka.sum().double()])
        if DEBUG28["on"]:
            DEBUG28.update(kc=kc.clone(), kc_true=kc_true.clone(), badk=badk.clone(), in_s=in_s.clone(), c1=c1.clone())
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
    c, kind, C = strip28(name) if name else (name, None, 0)
    if kind is None:
        return Q27.attend(q, k, v, li, name, scale, kw)
    Q25._check(c + "_r0")
    base, rk, rv, _ = Q14.split(c)

    def inner(q_, k_, v_, li_, name_, scale_, kw_):
        return _attend_p(q_, k_, v_, li_, name_, scale_, kw_, rk, rv, kind, C)

    S4._attend = inner
    try:
        return Q13.attend(q, k, v, li, base, scale, kw)
    finally:
        S4._attend = Q14._s4_attend_inner


def position_mass(q, k, scale, kw, row0, accs, rows=256):
    """add to every tensor of accs ([L - row0, L]) the exact attention probability each key position receives from the query rows
    >= row0, summed over the heads of this layer (called by the lab once per attention layer)."""
    H_, L, _ = q.shape
    KVH = k.shape[0]
    kx = k.repeat_interleave(H_ // KVH, 0)
    kpos = torch.arange(L, device=q.device)[None, :]
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
    for a in range(row0, L, rows):
        b = min(L, a + rows)
        qpos = torch.arange(a, b, device=q.device)[:, None]
        al = kpos <= qpos
        if win_:
            al &= kpos > qpos - int(win_)
        s = torch.einsum("hqd,hkd->hqk", q[:, a:b], kx) * scale
        if softcap:
            s = torch.tanh(s / softcap) * softcap
        m = torch.softmax(s.masked_fill(~al, float("-inf")), -1).sum(0)
        for t in accs:
            t[a - row0:b - row0] += m


def position_lists(acc, row0, cmax, group=128):
    """rank, per query row, the positions that are compressed at that row (not sinks, not in the exact window, not
    future) by their mass; int64 [rows, cmax]. Rows with fewer than cmax such positions repeat their first entry
    (a member of the list), so the list never names a position that is not compressed."""
    n, L = acc.shape
    kpos = torch.arange(L, device=acc.device)[None, :]
    qpos = torch.arange(row0, row0 + n, device=acc.device)[:, None]
    ok = (kpos <= qpos) & (kpos >= ZS) & (kpos < (qpos + 1) // group * group)
    a = acc.masked_fill(~ok, -float("inf"))
    val, idx = a.topk(min(cmax, L), -1)
    idx = torch.where(torch.isfinite(val), idx, idx[:, :1].expand_as(idx))
    if not bool(torch.isfinite(val[:, 0]).all()):
        raise RuntimeError("position_lists: a row has no compressed position")
    return idx


calib, calib_all = Q25.calib, Q25.calib_all
RC, Q7, P11, P10, P9, P8, S0 = Q25.RC, Q25.Q7, Q25.P11, Q25.P10, Q25.P9, Q25.P8, Q25.S0
PREFILL, CHUNK, ST, INFO, CLIP, VX, HCHECK = Q25.PREFILL, Q25.CHUNK, Q25.ST, Q25.INFO, Q25.CLIP, Q25.VX, Q25.HCHECK
DCALLS, PCALLS, XCALLS = Q25.DCALLS, Q25.PCALLS, Q25.XCALLS
STATS, SKIPPED = Q25.STATS, Q25.SKIPPED
reset_stats, dump_stats, dump_stats22 = Q25.reset_stats, Q25.dump_stats, Q25.dump_stats22
reset_stats27, dump_stats27 = Q27.reset_stats27, Q27.dump_stats27
