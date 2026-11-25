# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""sel25_q (rootcause, round TS2L, 2026-11-16): sel23_q for long windows, registered in REG_TS2L.md. Row chunks
that end at or before PREFILL["from"] are not computed (zeros; sel4_q.attend overwrites those rows with exact
attention). New suffixes: _r0 = R through this path with nothing replaced (control), _ba<n> = R's M candidates
plus n more by the BASE score (control of _ea<n>: more rows without a better choice). Original docstring follows.

sel23_q (rootcause, round TS2, 2026-11-16): sel22_q plus TWO-SIDED stage-1 selection, registered in REG_TS2.md.
New suffixes (stage 1 takes M/2 candidates by the base score, as R does for all M, and M/2 more by a second score
estimate among the rest; stage 2 and the fetch budget are unchanged):
  _su<N>   second list by the query-subspace sketch score (_sk<N>); tail logits from the base
  _sx<N>   as _su<N>, and the tail logits take the sketch correction
  _ku<b>   second list by b-bit KIVI key scores (equal-byte control of _su16 at b = 3); tail logits from the base
  _kx<b>   as _ku<b>, and the tail logits come from the b-bit keys
  _eu      oracle: second list by the exact scores; tail logits from the base (ceiling of two-sided selection)
  _ex      oracle: second list by the exact scores, tail logits exact
  _ea<n>   oracle coverage: R's M candidates by the base score plus n MORE by the exact scores (n more host key
           rows per KV head and step; stage 2 still takes K value rows); tail logits from the base
  _kv<b>   reference: tail logits from b-bit KIVI keys and tail values from b-bit KIVI values (b - 2 more GPU bits)
Everything else is sel22_q unchanged. Original docstring follows.

sel22_q (rootcause, round TLS, 2026-11-16): tail split oracles and GPU-resident tail logit sketches on top of
sel19_q, registered in REG_TLS.md. A name is a sel14 x2s _h name with lam = 0 followed by one suffix:

Oracles (use exact keys / host rows of every token; upper bounds, never methods). U = allowed keys that are neither
exact (window, sinks) nor stage-1 candidates:
  _tl    U logits from the host rows (h<rk>), U values from the base
  _tv    U values from the host rows, U logits from the base
  _tlv   both (the full host-tail ceiling, computed without the subtraction of sel19 _otzn)
  _te    U logits exact (FP16 keys), U values from the base

Sketches (deployable in principle: a few bytes per token and KV head, stored next to the base on the GPU, no host
read). r = k - k_base is the key residual at encode time; a rank-N code c = E^T r is kept (8 bit per coefficient,
KIVI-style per channel over groups of 128 tokens); the U logits become s_base + scale * (q^T Dm) c. E, Dm [d, N] per KV
head come from the queries of the dense prefill rows (4 <= row < PREFILL["from"]) of the same sequence, never from the
scored rows:
  _sk<N>   E = Dm = top-N eigenvectors of the query second moment S_q (query PCA)
  _sk<N>f  as _sk<N> with unquantised coefficients (cost of the 8-bit coefficient code)
  _sg<N>   residual-whitened: with D = diag(mean r^2 per channel over the prefill keys), V = top-N eigenvectors of
           D^1/2 S_q D^1/2, E = D^-1/2 V, Dm = D^1/2 V (best rank N when the residual covariance is diag(D))
  _sr<N>   control: a random orthonormal basis (seed 1234 + layer)
  _sc<N>   control: the N original channels with the largest S_q[c, c] * D[c] (a channel-selection / bit-allocation code)
  _ss<N>   as _sk<N>, and the sketch also corrects the scores used by the stage-1 selection
N is given for head_dim 128 and scaled to the model: N_eff = max(1, round(N * d / 128)), so a suffix costs the same
extra bits per element in every model.
Value side (U logits unchanged):
  _vb<b>   U values from KIVI-128 values quantised at b bits instead of 2 (GPU bytes spent on finer tail values)
  _vc<n>   coverage: the n U tokens with the largest final probability (mean over the query heads of a KV head, per
           row) take their host values (h<rv>); the stage-2 value rows are unchanged. Costs n more host value rows
           per KV head and step; n is not scaled by d.
Budget control (deployable, no sketch):
  _kb<b>   U logits from KIVI-128 keys quantised at b bits instead of 2 (the same GPU bytes spent on a finer base
           for the logits of the tail; b = 3 costs what _sk16 costs)
Any other name is passed to sel19_q unchanged. Statistics per layer (STATS22, scored rows only): sums over U of
e^2 and (e - e_hat)^2, unweighted and weighted by the final probability, e = base logit - exact logit.
"""
from __future__ import annotations

import re

import torch

import sel19_q as Q19

Q17, Q14, Q13, S12, S10, S4 = Q19.Q17, Q19.Q14, Q19.Q13, Q19.S12, Q19.S10, Q19.S4
Q5 = S4.Q5
if getattr(S4, "_sel22_installed", False) or getattr(S4, "_sel23_installed", False) \
        or getattr(S4, "_sel25_installed", False):
    raise RuntimeError("sel22_q / sel23_q / sel25_q imported twice")
S4._sel25_installed = True
SUF = re.compile(r"(.+)_(r0|ba|tl|tv|tlv|te|sk|sg|sr|sc|ss|kb|vb|vc|su|sx|ku|kx|eu|ex|ea|kv)(\d*)(f?)$")
KINDS = ("r0", "ba", "tl", "tv", "tlv", "te", "sk", "skf", "sg", "sr", "sc", "ss", "kb", "vb", "vc", "su", "sx", "ku", "kx", "eu", "ex", "ea", "kv")
UNION = ("su", "sx", "ku", "kx", "eu", "ex", "ea")
SKETCHES = ("sk", "skf", "sg", "sr", "sc", "ss", "su", "sx")
OCALLS = Q19.OCALLS                                    # shared with sel19 (the lab checks every key)
OCALLS.update({k_: 0 for k_ in KINDS})
STATS22: dict = {}                                     # layer -> float64 tensor [4]: e2, r2, p e2, p r2 over U
INFO22: dict = {}                                      # layer -> captured query energy share of the last sketch call
DEBUG = Q19.DEBUG
ZS = 4                                                 # exact sink rows
SKIPPED = {"n": 0}                                      # prefill chunks not computed
CBITS, CG = 8, 128                                     # coefficient code: bits, token group


def strip22(name):
    """(core, kind, N) or (name, None, 0); kind is one of KINDS."""
    m = SUF.fullmatch(name) if name else None
    if not m:
        return name, None, 0
    core_, kind, n, f = m.groups()
    if kind in ("r0", "tl", "tv", "tlv", "te", "eu", "ex"):
        if n or f:
            return name, None, 0
        return core_, kind, 0
    if not n or int(n) < 1 or (f and kind != "sk") or (kind in ("kb", "vb", "ku", "kx", "kv") and not 3 <= int(n) <= 8):
        return name, None, 0
    return core_, kind + f, int(n)


def strip19(name):
    """(core, mode) for the lab: mode is the OCALLS key of this name's oracle / sketch path, or None."""
    c, kind, _ = strip22(name)
    return (c, kind) if kind else Q19.strip19(name)


def core22(name):
    c, kind, _ = strip22(name)
    return c if kind else name


def strip(name):
    return Q19.strip(core22(name))


def core(name):
    return Q19.core(core22(name))


def _check(name):
    c, kind, _ = strip22(name)
    if kind is None:
        return c
    s = Q14.split(c)
    if s is None or s[3]:
        raise ValueError(f"{name}: TLS suffixes need a plain sel14 _h<rk><rv> name")
    p = S4.parse(s[0])
    if p["lam"] or p.get("jcal") or p.get("mass") or p["eb"] or p["tw"] or p["crit"][:2] != "2s" or not p["K"]:
        raise ValueError(f"{name}: TLS suffixes need an x2s<M> criterion with K > 0 and lam = 0")
    return c


def parse(name):
    c, kind, _ = strip22(name)
    if kind is None:
        return Q19.parse(name)
    return Q14.parse(_check(name))


def split(name):
    return Q19.split(core22(name)) if name else None


def host_bits(name, d):
    return Q19.host_bits(core22(name), d)


def eff_bits(name, L, d, g=1):
    return Q19.eff_bits(core22(name), L, d, g)


def sketch_bits(name, d):
    """extra GPU bits per cached element (keys and values together) of a sketch name; 0 for any other name."""
    _, kind, n = strip22(name)
    if kind == "kv":
        return float(n - 2)                          # b - 2 more bits on keys and values
    if kind in ("kb", "vb", "ku", "kx"):
        return (n - 2) / 2                             # b - 2 more bits on every key (or value) element
    if kind not in ("sk", "sg", "sr", "sc", "ss", "su", "sx"):
        return 0.0
    return n_eff(n, d) * (CBITS + 32 / CG) / (2 * d)   # N_eff coefficients + FP16 min / step per group, per token


def n_eff(N, d):
    return max(1, round(N * d / 128))


def reset_stats():
    STATS22.clear()
    INFO22.clear()
    Q19.reset_stats()


def dump_stats():
    return Q19.dump_stats()


def dump_stats22():
    return {str(li): {"sums": STATS22[li].tolist(), "share": INFO22.get(li)} for li in sorted(STATS22)}


def basis(q, k, kq, kind, N, li, g, pf):
    """encoder E and decoder Dm [KVH, d, N] and the query-energy share they capture (None for sr / sc)."""
    KVH, Lk, d = k.shape
    Lq = q.shape[1]
    lo, hi = ZS, pf - (Lk - Lq)                        # prefill query rows of q
    if kind in ("sr",):
        gen = torch.Generator(device="cpu").manual_seed(1234 + li)
        B = torch.linalg.qr(torch.randn(KVH, d, d, generator=gen, dtype=torch.float64))[0][:, :, :N]
        B = B.to(q.device).float()
        return B, B, None
    if hi - lo < 4 * d:
        raise ValueError(f"layer {li}: {hi - lo} prefill query rows are too few for a d = {d} basis")
    qp = q.reshape(KVH, g, Lq, d)[:, :, lo:hi].reshape(KVH, -1, d).double()
    Sq = qp.transpose(1, 2) @ qp / qp.shape[1]                                   # [KVH, d, d]
    r = (k - kq)[:, ZS:pf].double()
    Dr = (r * r).mean(1).clamp_min(1e-20)                                        # [KVH, d]
    tot = torch.diagonal(Sq, dim1=1, dim2=2).sum(-1)
    if kind == "sc":
        idx = (torch.diagonal(Sq, dim1=1, dim2=2) * Dr).topk(N, -1).indices     # [KVH, N]
        B = torch.zeros(KVH, d, N, dtype=torch.float64, device=q.device)
        B.scatter_(1, idx[:, None, :], 1.0)
        return B.float(), B.float(), None
    if kind == "sg":
        sd = Dr.sqrt()
        w, V = torch.linalg.eigh(sd[:, :, None] * Sq * sd[:, None, :])
        V = V[:, :, -N:]
        E, Dm = V / sd[:, :, None], V * sd[:, :, None]
        share = None
    else:                                                                        # sk, skf, ss
        w, V = torch.linalg.eigh(Sq)
        V = V[:, :, -N:]
        E, Dm = V, V
        share = float((w[:, -N:].sum(-1) / tot).mean())
    return E.float(), Dm.float(), share


def sketch(q, k, kq, kind, N, li, g, pf, scale):
    """(q^T Dm) [H, Lq, N] already times scale, quantised coefficients [H, Lk, N]."""
    E, Dm, share = basis(q, k, kq, kind, n_eff(N, k.shape[-1]), li, g, pf)
    c = torch.einsum("hkd,hdn->hkn", k - kq, E)
    if kind != "skf":
        c = Q5.per_channel(c, CBITS, CG)
    c[:, :ZS] = 0                                                                # sinks are exact; never used on U anyway
    INFO22[li] = share
    qd = torch.einsum("hqd,hdn->hqn", q, Dm.repeat_interleave(g, 0)) * scale
    return qd, c.repeat_interleave(g, 0)


def _acc22(li, rows, U, p, e, eh):
    d = STATS22.setdefault(li, torch.zeros(4, dtype=torch.float64, device=p.device))
    m = (U & rows[None, :, None]).double()
    e64, r64, p64 = e.double(), (e - eh).double(), p.double()
    d += torch.stack([(m * e64 * e64).sum(), (m * r64 * r64).sum(), (m * p64 * e64 * e64).sum(),
                      (m * p64 * r64 * r64).sum()])


def _attend_t(q, k, v, li, name, scale, kw, rk, rv, kind, N):
    """sel14_q._attend_h (chk = None, lam = 0) with the TLS oracle / sketch arms (see the module docstring)."""
    if S4.DIAG["on"]:
        raise RuntimeError("sel22_q: DIAG is not implemented")
    p_ = S4.parse(name)
    if p_["lam"] or p_["crit"][:2] != "2s" or not p_["K"]:
        raise ValueError(f"{name}: TLS needs x2s<M>, K > 0, lam = 0")
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
    is_sk = kind in SKETCHES
    alt = is_sk or kind in ("kb", "ku", "kx", "kv")        # arms with a deployable second logit estimate s_sk
    union = kind in UNION
    if is_sk:
        if not pf:
            raise ValueError(f"{name}: sketches need the dense prefill (PREFILL on) for their basis")
        qd, cx = sketch(q, k, kq, kind, N, li, g, pf, scale)
    if kind in ("kb", "ku", "kx", "kv"):
        kb = Q5.per_channel(k, N, 128)
        kb[:, :z] = k[:, :z]
        kbx = kb.repeat_interleave(g, 0)
    if kind in ("vb", "kv"):
        vb = Q5.per_token(v, N, 128)
        vb[:, :z] = v[:, :z]
        vbx = vb.repeat_interleave(g, 0)

    for r0 in range(0, Lq, S4.CHUNK["rows"]):
        r1 = min(Lq, r0 + S4.CHUNK["rows"])
        n = r1 - r0
        if Lk - Lq + r1 <= pf:                           # chunk entirely in the dense prefill: overwritten later
            out[:, r0:r1] = 0
            SKIPPED["n"] += 1
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
        s_sk = None
        if is_sk:
            s_sk = s_base + torch.einsum("hqn,hkn->hqk", qd[:, r0:r1], cx)
        elif kind in ("kb", "ku", "kx", "kv"):
            s_sk = torch.einsum("hqd,hkd->hqk", qr, kbx) * scale
        s = torch.where(ex, se, s_sk if kind == "ss" else s_base)
        bad = ~al[None] | ex
        c1 = fin(s).masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        M1 = M // 2 if (union and kind != "ea") else M
        if kind == "ba":                                 # control: N more candidates by the base score
            M1 = M + N
        M2 = N if kind == "ea" else M - M1               # ea: N more candidates on top of R's M
        kc = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, c1.topk(min(M1, Lk), -1).indices, True)
        if union:                                      # second list: M2 more by the second estimate, among the rest
            s2 = se if kind in ("eu", "ex", "ea") else s_sk
            c1b = fin(torch.where(ex, se, s2)).masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
            c1b = c1b.masked_fill(kc, -float("inf"))
            kc = kc | torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, c1b.topk(min(M2, Lk), -1).indices, True)
        kc = kc.repeat_interleave(g, 0) & ~bad
        s1 = torch.where(kc, sh, s)
        c2 = fin(s1).masked_fill(~kc, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        m = torch.zeros_like(c2, dtype=torch.bool).scatter_(-1, c2.topk(min(K, Lk), -1).indices, True).repeat_interleave(g, 0)
        m = m & kc
        sf = torch.where(m | kc, sh, s)
        U = (al[None] & ~ex & ~kc).expand_as(sf)
        OCALLS[kind] += 1
        if kind in ("tl", "tlv"):
            sf = torch.where(U, sh, sf)
        elif kind in ("te", "ex"):
            sf = torch.where(U, se, sf)
        elif alt and kind not in ("su", "ku"):
            sf = torch.where(U, s_sk, sf)
        pr = fin(sf)
        o = (torch.einsum("hqk,hkd->hqd", pr, vqx) + torch.einsum("hqk,hkd->hqd", pr * ex, vx - vqx)
             + torch.einsum("hqk,hkd->hqd", pr * m, vhx - vqx))
        if kind in ("tv", "tlv"):
            o = o + torch.einsum("hqk,hkd->hqd", pr * U, vhx - vqx)
        elif kind in ("vb", "kv"):
            o = o + torch.einsum("hqk,hkd->hqd", pr * U, vbx - vqx)
        elif kind == "vc":
            cu = (pr * U).reshape(KVH, g, n, Lk).mean(1)
            mv = torch.zeros_like(cu, dtype=torch.bool).scatter_(-1, cu.topk(min(N, Lk), -1).indices, True)
            mv = mv.repeat_interleave(g, 0) & U
            o = o + torch.einsum("hqk,hkd->hqd", pr * mv, vhx - vqx)
        out[:, r0:r1] = o
        if alt:
            rows = qpos[:, 0] >= pf
            if rows.any():
                _acc22(li, rows, U, pr, s_base - se, (s_base - s_sk) if kind not in ("su", "ku") else torch.zeros_like(se))
        if DEBUG["on"]:
            DEBUG.update(U=U, kc=kc, m=m, ex=ex, al=al, s=s, sh=sh, se=se, sf=sf, vqx=vqx, vhx=vhx, vx=vx,
                         s_base=s_base, s_sk=s_sk, pr=pr, mv=mv if kind == "vc" else None)
    return out


def attend(q, k, v, li, name, scale, kw):
    c, kind, N = strip22(name) if name else (name, None, 0)
    if kind is None:
        return Q19.attend(q, k, v, li, name, scale, kw)
    _check(name)
    base, rk, rv, _ = Q14.split(c)

    def inner(q_, k_, v_, li_, name_, scale_, kw_):
        return _attend_t(q_, k_, v_, li_, name_, scale_, kw_, rk, rv, kind, N)

    S4._attend = inner
    try:
        return Q13.attend(q, k, v, li, base, scale, kw)
    finally:
        S4._attend = Q14._s4_attend_inner


calib, calib_all = Q19.calib, Q19.calib_all
RC, Q7, P11, P10, P9, P8, S0 = Q19.RC, Q19.Q7, Q19.P11, Q19.P10, Q19.P9, Q19.P8, Q19.S0
PREFILL, CHUNK, ST, INFO, CLIP, VX, HCHECK = Q19.PREFILL, Q19.CHUNK, Q19.ST, Q19.INFO, Q19.CLIP, Q19.VX, Q19.HCHECK
DCALLS, PCALLS, XCALLS = Q19.DCALLS, Q19.PCALLS, Q19.XCALLS
STATS = Q19.STATS
