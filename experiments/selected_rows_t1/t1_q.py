# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""T1 (REG_T1.md v4): sel14_q._attend_h with read-only selection statistics.

t1_attend_h equals sel14_q._attend_h (STAR stack, md5 bee1a8504bfc159b1b8bff1878d58932) on the path T1 uses
(kivi128k64_x2s128_j50_h54, chk=None, rk=5, rv=4): the chk ("_id" / "_lk") and HCHECK (rk, rv >= 16) branches are
removed because they are unreachable there; an (M, K, R) check and one _record() call are added. _record only reads.
Q0a in t1_lab.py checks the final hidden state against the original function bitwise.

Demotion schedule (REG v4 section 3.2): tokens leave the FP16 window in blocks of 128. A block is decided G rows after
it left the window ("probation"); the f share of the block with the lowest score is demoted, never promoted. Score:
attention mass received after leaving the window (a_now - a_exit). Controls with the same schedule: a fixed random
order inside the block, and oldest-first inside the block.
"""
from __future__ import annotations

import torch

import sel14_q as Q14

S4 = Q14.S4
ORIG = Q14._attend_h
M_RUN, K_RUN, WIN, SINKS = 128, 64, 128, 4
PROB = (512, 1024, 2048)
FRACS = (0.75, 0.9)
POLICIES = [(kind, g, f) for kind in ("score", "random", "oldest") for g in PROB for f in FRACS]
PAGES = (8, 16, 32, 64)
BUCKETS = ((0, 1024), (1024, 2048), (2048, 4096), (4096, 1 << 30))
SUB = 64
# per policy and bucket, summed: n1, n2, c1cnt, c2cnt, c1 mass on stage-1 in D, c1 mass on stage-1,
# final-weight mass on stage-2 in D, final-weight mass on stage-2, |D|, |E|
NPOL = 10
CFG = {"on": False, "prefill": 0, "item": 0}
STATE: dict = {}
STATS: dict = {}
CALLS: dict = {}


def reset_item(item: int, prefill: int) -> None:
    STATE.clear()
    STATS.clear()
    CALLS.clear()
    CFG.update(item=item, prefill=prefill)


def pol_name(p) -> str:
    kind, g, f = p
    return f"{kind}_G{g}_f{f}"


def _new_stats(dev) -> dict:
    nb = len(BUCKETS)

    def z(*sh):
        return torch.zeros(*sh, dtype=torch.float64, device=dev)
    return {"pol": z(len(POLICIES), nb, NPOL),
            "pages": z(len(PAGES), 2, nb, 4),   # stage x bucket x (rows, pages, random-control rows, pages)
            "churn": z(nb, 2), "union": z(nb, 2), "cand": z(3), "pre_rows": z(1)}


def stats_to_json(sm: dict, st: dict | None = None) -> dict:
    out = {"pol": {}, "pages": {}}
    for pi, p in enumerate(POLICIES):
        out["pol"][pol_name(p)] = sm["pol"][pi].tolist()
    for i, P in enumerate(PAGES):
        out["pages"][str(P)] = {"s1": sm["pages"][i, 0].tolist(), "s2": sm["pages"][i, 1].tolist()}
    for k in ("churn", "union", "cand", "pre_rows"):
        out[k] = sm[k].tolist()
    if st is not None:
        out["zero_score_demoted"] = dict(st["zero"])
        out["demoted_total"] = dict(st["dem"])
    return out


def _bucket(pos: int) -> int:
    for i, (a, b) in enumerate(BUCKETS):
        if a <= pos < b:
            return i
    raise ValueError(pos)


def _window_move(st: dict, li: int, w_new: int, acc_row: torch.Tensor, lk: int, dev) -> None:
    """The window start moved from st['w'] to w_new at this row; acc_row = a before this row (KVH, Lk).
    1) record a_exit for the tokens that just left the window; 2) decide every block whose probation is over."""
    w_old = max(st["w"], 0)
    if w_new > w_old:
        st["a_exit"][:, w_old:w_new] = acc_row[:, w_old:w_new]
        st["exited"][w_old:w_new] = True
    for p in POLICIES:
        nm = pol_name(p)
        g, f = p[1], p[2]
        dm = st["D"][nm]
        while True:
            b0 = st["next"][nm]                                   # next undecided block [b0, b0 + 128)
            b1 = b0 + WIN
            if b1 + g > w_new:                                    # block left the window at w = b1; wait G rows
                break
            lo = max(b0, SINKS)
            if not bool(st["exited"][lo:b1].all()):
                raise RuntimeError(f"T1 Q0c: deciding a block that has not left the window ({nm}, layer {li})")
            nblk = b1 - lo
            need = int(f * nblk)
            if need > 0:
                if p[0] == "score":
                    sc = acc_row[:, lo:b1] - st["a_exit"][:, lo:b1]
                elif p[0] == "random":
                    sc = st["rand"][lo:b1].to(torch.float64).expand(dm.shape[0], -1)
                else:
                    sc = torch.arange(lo, b1, device=dev, dtype=torch.float64).expand(dm.shape[0], -1)
                order = torch.sort(sc, dim=-1, stable=True).indices[:, :need]        # lowest first; earlier on ties
                add = torch.zeros(dm.shape[0], nblk, dtype=torch.bool, device=dev).scatter_(-1, order, True)
                if bool((dm[:, lo:b1] & add).any()):
                    raise RuntimeError(f"T1 Q0c: re-demotion ({nm}, layer {li})")
                dm[:, lo:b1] |= add
                st["dem"][nm] += float(add.sum())
                if p[0] == "score":
                    st["zero"][nm] += float(((sc == 0) & add).sum())
            st["next"][nm] = b1


def _excl_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum along dim 1 without subtraction (no cancellation)."""
    return torch.cumsum(torch.nn.functional.pad(x, (0, 0, 1, 0))[:, :-1], 1)


def _record(li, qpos, cu, c1, kc, m, prK, ex, al, g, KVH, n, Lk) -> None:
    """cu: group-mean base softmax over all allowed keys (window and sinks included, exact scores there), (KVH, n, Lk);
    c1: the selection score (window / sinks / future at -inf); kc, m: (H, n, Lk) masks repeated over the group;
    prK: group-mean final attention weights (KVH, n, Lk); ex (H, n, Lk); al (n, Lk)."""
    dev = c1.device
    if li not in STATE:
        gen = torch.Generator(device="cpu").manual_seed(li * 1000 + CFG["item"])
        STATE[li] = {"acc": torch.zeros(KVH, Lk, dtype=torch.float64, device=dev),
                     "a_exit": torch.zeros(KVH, Lk, dtype=torch.float64, device=dev),
                     "exited": torch.zeros(Lk, dtype=torch.bool, device=dev),
                     "D": {pol_name(p): torch.zeros(KVH, Lk, dtype=torch.bool, device=dev) for p in POLICIES},
                     "next": {pol_name(p): 0 for p in POLICIES},
                     "U": torch.zeros(KVH, Lk, dtype=torch.bool, device=dev), "prev": None, "w": -1,
                     "rand": torch.randperm(Lk, generator=gen).to(dev),
                     "gen": torch.Generator(device=dev).manual_seed(li * 1000 + CFG["item"] + 7),
                     "zero": {pol_name(p): 0.0 for p in POLICIES}, "dem": {pol_name(p): 0.0 for p in POLICIES}}
        STATS[li] = _new_stats(dev)
    st, sm = STATE[li], STATS[li]
    if st["w"] == -1 and int(qpos[0, 0]) != 0:
        raise RuntimeError("T1: the first row this function sees must be position 0 (full teacher forcing)")
    for s0 in range(0, n, SUB):
        s1_ = min(n, s0 + SUB)
        nr = s1_ - s0
        cc = cu[:, s0:s1_].double()
        acc_b = st["acc"][:, None, :] + _excl_cumsum(cc)          # a before each row, every row this function sees
        st["acc"] = st["acc"] + cc.sum(1)
        qp = qpos[s0:s1_, 0]
        w_all = (((qp + 1) // WIN) * WIN).tolist()
        dec = qp >= CFG["prefill"]
        sm["pre_rows"][0] += float((~dec).sum())
        # window moves and decisions on every row (prompt rows included), readouts on decode rows only
        d0 = int(torch.nonzero(dec)[0]) if bool(dec.any()) else nr
        if not bool(dec[d0:].all()):
            raise RuntimeError("T1: decode rows are not contiguous")
        r = 0
        while r < nr:
            w = int(w_all[r])
            r_end = r
            while r_end < nr and w_all[r_end] == w and (r_end < d0) == (r < d0):
                r_end += 1
            if w != st["w"]:
                if w < st["w"]:
                    raise RuntimeError("T1: window start moved backwards")
                _window_move(st, li, w, acc_b[:, r], Lk, dev)
                st["w"] = w
            if r >= d0:
                _readouts(st, sm, li, s0 + r, s0 + r_end, w, c1, kc, m, prK, ex, al, g, KVH, Lk, qp[r:r_end], dev)
            r = r_end


def _readouts(st, sm, li, a, b, w, c1, kc, m, prK, ex, al, g, KVH, Lk, qps, dev) -> None:
    """Readouts for decode rows a..b-1 (chunk-relative) with constant window start w."""
    nd = b - a
    ks, ms, prs = kc[::g, a:b], m[::g, a:b], prK[:, a:b].double()
    if bool(torch.isnan(c1[:, a:b]).any()) or not bool(torch.isfinite(prs).all()):
        raise RuntimeError(f"T1: NaN in the selection score or non-finite final weights (layer {li})")
    c1d = torch.where(torch.isfinite(c1[:, a:b]), c1[:, a:b], torch.zeros((), device=dev)).double()
    el = al[a:b] & ~ex[0, a:b]
    e_cnt = el.sum(-1)
    c1n, c2n = ks.sum(-1), ms.sum(-1)
    if not bool((c1n == torch.clamp(e_cnt, max=M_RUN)[None]).all()):
        raise RuntimeError(f"T1 Q0d: stage-1 count != min(128, |E|) (layer {li})")
    if not bool((c2n == torch.clamp(c1n, max=K_RUN)).all()):
        raise RuntimeError(f"T1 Q0d: stage-2 count != min(64, stage-1) (layer {li})")
    if not bool((e_cnt == max(0, w - SINKS)).all()):
        raise RuntimeError(f"T1: E_t is not [4, w_t) (layer {li})")
    sm["cand"][0] += c1n.sum()
    sm["cand"][1] += c2n.sum()
    sm["cand"][2] += nd * KVH
    bs = torch.tensor([_bucket(int(x) - CFG["prefill"]) for x in qps.tolist()], device=dev)
    prev = st["prev"] if st["prev"] is not None else ks[:, 0]
    prevs = torch.cat([prev[:, None], ks[:, :-1]], 1)
    ch = (ks & ~prevs).sum(-1).double().sum(0)
    has_prev = torch.ones(nd, dtype=torch.float64, device=dev)
    if st["prev"] is None:
        has_prev[0] = 0.0
        ch[0] = 0.0
    sm["churn"][:, 0].index_add_(0, bs, ch)
    sm["churn"][:, 1].index_add_(0, bs, has_prev * KVH)
    st["prev"] = ks[:, -1].clone()
    cum = torch.cummax(ks.to(torch.int8), dim=1).values.bool() | st["U"][:, None, :]
    st["U"] = cum[:, -1].clone()
    e_n = float(max(0, w - SINKS))
    if e_n > 0:
        cov = (cum & el[None]).sum(-1).double().sum(0) / e_n
        sm["union"][:, 0].index_add_(0, bs, cov)
        sm["union"][:, 1].index_add_(0, bs, torch.full((nd,), float(KVH), dtype=torch.float64, device=dev))
    for pi, p in enumerate(POLICIES):
        dm = st["D"][pol_name(p)][:, None, :]
        cols = [(ks & dm).sum(-1).double(), (ms & dm).sum(-1).double(), ks.sum(-1).double(), ms.sum(-1).double(),
                (c1d * (ks & dm)).sum(-1), (c1d * ks).sum(-1), (prs * (ms & dm)).sum(-1), (prs * ms).sum(-1)]
        vals = torch.stack([c.sum(0) for c in cols]
                           + [torch.full((nd,), float(dm.sum()), dtype=torch.float64, device=dev),
                              torch.full((nd,), e_n * KVH, dtype=torch.float64, device=dev)], 1)
        sm["pol"][pi].index_add_(0, bs, vals)
    if e_n > 0:
        els = el[None].expand_as(ks)
        noise = torch.rand(ks.shape, generator=st["gen"], device=dev).masked_fill(~els, -1.0)
        for si, msk in enumerate((ks & els, ms & els)):
            k_ = int(msk[:, 0].sum(-1).max())
            rnd = torch.zeros_like(msk)
            if k_ > 0:
                rnd.scatter_(-1, noise.topk(k_, -1).indices, True)
                rnd &= els
            for pg_i, P in enumerate(PAGES):
                lp = (Lk + P - 1) // P * P
                for col, mm in ((0, msk), (2, rnd)):
                    pad = torch.nn.functional.pad(mm, (0, lp - Lk))
                    pg = pad.view(KVH, mm.shape[1], lp // P, P).any(-1).sum(-1).double().sum(0)
                    cnt = mm.sum(-1).double().sum(0)
                    sm["pages"][pg_i, si, :, col].index_add_(0, bs, cnt)
                    sm["pages"][pg_i, si, :, col + 1].index_add_(0, bs, pg)


def t1_attend_h(q, k, v, li, name, scale, kw, rk, rv, chk=None):
    CALLS[li] = CALLS.get(li, 0) + 1
    if S4.DIAG["on"]:
        raise RuntimeError("sel14_q: DIAG is not implemented for _h runs")
    p = S4.parse(name)
    if p.get("jcal") or p.get("mass") or p["eb"] or p["tw"] or p["crit"][:2] != "2s" or not p["K"]:
        raise ValueError(f"{name}: _h needs an x2s<M> criterion with K > 0 and a plain / _j / _p correction")
    kq, vq, R, _ = S4.Q5.base("kivi128", k, v, li)
    z = 4
    kq[:, :z], vq[:, :z] = k[:, :z], v[:, :z]
    if chk is not None or rk >= 16 or rv >= 16:
        raise ValueError("T1: only chk=None and rk, rv < 16 are used in T1")
    kh = k if rk >= 16 else kq + Q14._res(k - kq, rk)
    vh = v if rv >= 16 else vq + Q14._res(v - vq, rv)
    k_enc = k
    K, M, lam = p["K"], int(p["crit"][2:]), p["lam"]
    if (M, K, R) != (M_RUN, K_RUN, WIN):
        raise ValueError(f"T1 expects M=128, K=64, R=128; got {M}, {K}, {R}")
    H_, Lq, _ = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    kx, vx, kqx, vqx, khx, vhx = (t.repeat_interleave(g, 0) for t in (k, v, kq, vq, kh, vh))
    out = torch.empty(H_, Lq, v.shape[-1], device=q.device)
    kpos = torch.arange(Lk, device=q.device)[None, :]
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
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
        c1 = fin(s).masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        kc = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, c1.topk(min(M, Lk), -1).indices, True)
        kc = kc.repeat_interleave(g, 0) & ~bad
        s1 = torch.where(kc, sh, s - (lam / 100) * s2 / 2 if lam else s)
        c2 = fin(s1).masked_fill(~kc, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        m = torch.zeros_like(c2, dtype=torch.bool).scatter_(-1, c2.topk(min(K, Lk), -1).indices, True).repeat_interleave(g, 0)
        m = m & kc
        sf = torch.where(m | kc, sh, s)
        if lam:
            sf = torch.where(m | ex | kc, sf, sf - (lam / 100) * s2 / 2)
        pr = fin(sf)
        if CFG["on"]:
            with torch.no_grad():
                cu = fin(s).reshape(KVH, g, n, Lk).mean(1)
                _record(li, qpos, cu, c1, kc, m, pr.reshape(KVH, g, n, Lk).mean(1), ex, al, g, KVH, n, Lk)
        o = (torch.einsum("hqk,hkd->hqd", pr, vqx) + torch.einsum("hqk,hkd->hqd", pr * ex, vx - vqx)
             + torch.einsum("hqk,hkd->hqd", pr * m, vhx - vqx))
        out[:, r0:r1] = o
    return out
