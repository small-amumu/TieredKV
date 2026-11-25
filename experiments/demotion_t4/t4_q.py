# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""T4: cold-residual demotion applied inside the forward pass (plan: next-phase plan, line A).

t4_attend_h is t1_q.t1_attend_h (a verbatim copy of sel14_q._attend_h on the kivi128k64_x2s128_j50_h<rk><rv> path)
without T1's read-only statistics, plus an optional demotion policy set in CFG["policy"]. With no policy the
attention path is the original op sequence (bitwise equal to sel14_q._attend_h, checked by the smoke).

Demotion schedule (as T1): tokens leave the 128-token FP16 window in blocks of 128 (block 0 is [4, 128)). A block is
decided G rows after it left the window. Inside the block, tokens are ranked by a policy score (lowest first):
  score   attention mass received after leaving the window (group-mean base softmax, exact on window / sinks),
          ties broken by a fixed random order (not by position);
  random  a fixed random order (seed: layer, item);
  oldest  position.
Tiers (coldest first) give each demoted token an action, applied from the decision row on, never undone:
  h32  the host record is re-encoded from the h54 record (FP16 is gone): r3 = _res(r5, 3), v2 = _res(r4, 2);
  key  the value residual is dropped (key residual kept);
  del  both residuals are dropped: the token's key logit is the base logit with the j correction (as a row that
       is not fetched), and its value is the 2-bit base.
Every quantity the schedule uses comes from rows before the decision row (exclusive prefix sums).
Prompt rows advance the schedule too (their outputs are replaced by exact dense attention in sel4_q, so
demotion never changes them); with prompts of ~70 tokens no block is decided inside the prompt.

Self-test references (smoke only): CFG["ref"] = "h32" | "key" | "del" applies the action to every token at every
row through tensor substitution instead of masks; an all-tokens policy (G = 0, f = 1) must equal it bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

import sel14_q as Q14

S4 = Q14.S4
M_RUN, K_RUN, WIN, SINKS = 128, 64, 128, 4
ACTIONS = ("h32", "key", "del")
PAGES = (8, 16, 32, 64)
BUCKETS = ((0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 1 << 30))
SUB = 64
# per bucket (decode rows; masses are group means over the query heads of a KV head, summed over KV heads):
#  0 rows*KVH, 1 |E|*KVH, 2-4 demoted tokens in E per action (h32, key, del),
#  5 stage-1 rows dropped vs the undemoted run on the same inputs (structurally 0: stage 1 ranks by the base only),
#  6 stage-2 rows dropped vs the undemoted selection,
#  7 final weight on stage-2 rows, 8 of it on demoted tokens,
#  9 undemoted counterfactual (same inputs, no demotion): final weight on its stage-2 rows that are demoted here,
#  10 its final weight on all its stage-2 rows  (9 / 10 is T1's w2 measured inside this run),
#  11 stage-1 rows reading an h32 key, 12 stage-2 rows reading an h32 value,
#  13 stage-2 rows whose value residual is gone (key / del: the slot is not refilled, K = 64 stays the budget),
#  14 stage-1 rows whose key residual is gone (del: base key with the j correction)
NREC = 15
# page statistics per view: logical selections (s1, s2) and the records that are actually readable, by pool
# (K54: stage-1 keys with an h54 key record, i.e. kept or key-only; K32: h32 keys; V54: stage-2 values with an
# h54 value record; V32: h32 values)
VIEWS = ("s1", "s2", "K54", "K32", "V54", "V32")


@dataclass(frozen=True)
class Policy:
    kind: str                                   # score | random | oldest
    g: int                                      # probation rows
    tiers: tuple[tuple[float, str], ...]        # (fraction of the block, action), coldest first
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("score", "random", "oldest"):
            raise ValueError(self.kind)
        if self.g < 0 or self.g % WIN or not self.tiers:   # the window moves in steps of 128 rows
            raise ValueError(f"{self}: G must be a non-negative multiple of {WIN}")
        tot = 0.0
        for f, a in self.tiers:
            if a not in ACTIONS or not 0.0 < f <= 1.0:
                raise ValueError(self)
            tot += f
        if tot > 1.0 + 1e-12:
            raise ValueError(f"{self}: tiers add up to {tot}")


@dataclass
class Cfg:
    policy: Policy | None = None
    ref: str | None = None                      # smoke-only uniform reference action
    prefill: int = 0
    item: int = 0
    pages: bool = False                         # page histograms / reuse (deployed and one demotion arm)
    cap: int = 0                                # state capacity in tokens for incremental decoding (0: the
                                                # first call's key length, i.e. teacher forcing as in A2)
    calls: dict = field(default_factory=dict)


CFG = Cfg()
STATE: dict = {}
STATS: dict = {}


def reset_item(item: int, prefill: int, policy: Policy | None, ref: str | None = None, pages: bool = False,
               cap: int = 0) -> None:
    if policy is not None and ref is not None:
        raise ValueError("a policy and a uniform reference are exclusive")
    if ref is not None and ref not in ACTIONS:
        raise ValueError(ref)
    if not 0 <= item < 1000:                      # seeds are layer * 1000 + item
        raise ValueError(f"item id {item} outside [0, 1000)")
    if cap < 0:
        raise ValueError(cap)
    if cap and pages:                             # page reuse keeps the previous row's page map (fixed width)
        raise ValueError("T4: page statistics are not supported with incremental decoding (cap > 0)")
    # contract: the random order (random policy, and the tie-break of the score policy) is a permutation of the
    # allocated length, i.e. of cap when cap > 0, else of the first call's key length. Runs are comparable (paired)
    # only when they use the same cap: every arm of a generated problem uses cap = prompt + max_new; replays use 0.
    STATE.clear()
    STATS.clear()
    CFG.calls.clear()
    CFG.policy, CFG.ref, CFG.prefill, CFG.item, CFG.pages, CFG.cap = policy, ref, prefill, item, pages, cap


def _bucket(pos: int) -> int:
    for i, (a, b) in enumerate(BUCKETS):
        if a <= pos < b:
            return i
    raise ValueError(pos)


def _excl_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum along dim 1 without subtraction (no cancellation)."""
    return torch.cumsum(torch.nn.functional.pad(x, (0, 0, 1, 0))[:, :-1], 1)


def _state(li: int, KVH: int, Lk: int, dev) -> dict:
    if li in STATE and STATE[li]["acc"].shape[1] < Lk:
        raise RuntimeError(f"T4: key length {Lk} exceeds the state capacity {STATE[li]['acc'].shape[1]}")
    if li not in STATE:
        gen = torch.Generator(device="cpu").manual_seed(li * 1000 + CFG.item)
        Lk = max(Lk, CFG.cap)                     # allocate once; every use below slices to the current length
        STATE[li] = {"acc": torch.zeros(KVH, Lk, dtype=torch.float64, device=dev),
                     "a_exit": torch.zeros(KVH, Lk, dtype=torch.float64, device=dev),
                     "exited": torch.zeros(Lk, dtype=torch.bool, device=dev),
                     # tier action index + 1 (0 = kept) and the first row position the action applies to
                     "act": torch.zeros(KVH, Lk, dtype=torch.int8, device=dev),
                     "dpos": torch.full((KVH, Lk), 1 << 40, dtype=torch.int64, device=dev),
                     "next": 0, "w": -1, "rand": torch.randperm(Lk, generator=gen).to(dev),
                     "prev_tp": {}, "dem": [0.0] * len(ACTIONS), "zero": 0.0}
        STATS[li] = {"rec": torch.zeros(len(BUCKETS), NREC, dtype=torch.float64, device=dev),
                     # page size x view x bucket x hits (index 0 unused, 1..P)
                     "hist": {P: torch.zeros(len(VIEWS), len(BUCKETS), P + 1, dtype=torch.float64, device=dev)
                              for P in PAGES},
                     # page size x view x bucket x (pages touched at a row that had a previous row, of them reused)
                     "reuse": torch.zeros(len(PAGES), len(VIEWS), len(BUCKETS), 2, dtype=torch.float64,
                                          device=dev)}
    return STATE[li]


def _decide(st: dict, li: int, w_new: int, acc_row: torch.Tensor, row_pos: int, dev) -> None:
    """The window start moved to w_new at the row at position row_pos; acc_row = attention mass before that row.
    Record a_exit for tokens that just left the window and decide every block whose probation is over."""
    pol = CFG.policy
    w_old = max(st["w"], 0)
    if w_new > w_old:
        st["a_exit"][:, w_old:w_new] = acc_row[:, w_old:w_new]
        st["exited"][w_old:w_new] = True
    KVH = acc_row.shape[0]
    while True:
        b0 = st["next"]
        b1 = b0 + WIN
        if b1 + pol.g > w_new:
            break
        lo = max(b0, SINKS)
        if not bool(st["exited"][lo:b1].all()):
            raise RuntimeError(f"T4: deciding a block that has not left the window (layer {li})")
        nblk = b1 - lo
        rnd = st["rand"][lo:b1].to(torch.float64).expand(KVH, -1)
        if pol.kind == "score":
            sc = acc_row[:, lo:b1] - st["a_exit"][:, lo:b1]
            o1 = torch.sort(rnd, dim=-1, stable=True).indices                 # random order first ...
            o2 = torch.sort(torch.gather(sc, 1, o1), dim=-1, stable=True).indices
            order = torch.gather(o1, 1, o2)                                     # ... then by score: random tie-break
        elif pol.kind == "random":
            order = torch.sort(rnd, dim=-1, stable=True).indices
        else:
            order = torch.arange(nblk, device=dev).expand(KVH, -1)
        cum = 0.0
        start = 0
        for f, a in pol.tiers:
            cum += f
            end = min(nblk, int(cum * nblk + 1e-9))
            if end > start:
                idx = order[:, start:end] + lo
                if bool((st["act"].gather(1, idx) != 0).any()):
                    raise RuntimeError(f"T4: re-demotion (layer {li})")
                ai = ACTIONS.index(a)
                st["act"].scatter_(1, idx, ai + 1)
                st["dpos"].scatter_(1, idx, row_pos)
                st["dem"][ai] += float(idx.numel())
                if pol.kind == "score":
                    st["zero"] += float((torch.gather(sc, 1, idx - lo) == 0).sum())
            start = end
        st["next"] = b1


def _schedule(li: int, cu: torch.Tensor, qpos: torch.Tensor, KVH: int, Lk: int, dev) -> None:
    """Advance the demotion schedule over the rows of one chunk; cu is the group-mean base softmax (KVH, n, Lk)."""
    st = _state(li, KVH, Lk, dev)
    if st["w"] == -1 and int(qpos[0, 0]) != 0:
        raise RuntimeError("T4: the first row this function sees must be position 0 (full teacher forcing)")
    n = cu.shape[1]
    for s0 in range(0, n, SUB):
        s1_ = min(n, s0 + SUB)
        cc = cu[:, s0:s1_].double()
        acc_b = st["acc"][:, None, :Lk] + _excl_cumsum(cc)
        st["acc"][:, :Lk] += cc.sum(1)
        qp = qpos[s0:s1_, 0].tolist()
        for r, pos in enumerate(qp):
            w = (pos + 1) // WIN * WIN
            if w != st["w"]:
                if w < st["w"]:
                    raise RuntimeError("T4: window start moved backwards")
                _decide(st, li, w, acc_b[:, r], pos, dev)
                st["w"] = w


def _masks(li: int, qpos: torch.Tensor, n: int, KVH: int, Lk: int, dev) -> list[torch.Tensor]:
    """Per action, (KVH, n, Lk) bool: token demoted with that action at this row."""
    st = STATE[li]
    live = qpos.view(1, n, 1) >= st["dpos"][:, None, :Lk]
    return [live & (st["act"][:, :Lk] == ai + 1)[:, None, :] for ai in range(len(ACTIONS))]


def _record(li, qpos, kc, m, kc0, m0, pr, Dany, ex, al, g, KVH, n, Lk, dev, pr0=None) -> None:
    """Decode rows only. kc/m: selection under demotion, kc0/m0: the undemoted selection on the same inputs
    (H, n, Lk), pr: final weights (H, n, Lk), pr0: the undemoted final weights (defaults to pr), Dany: (KVH, n, Lk)."""
    st, sm = STATE[li], STATS[li]
    qp = qpos[:, 0]
    dec = qp >= CFG.prefill
    if not bool(dec.any()):
        return
    d0 = int(torch.nonzero(dec)[0])
    if not bool(dec[d0:].all()):
        raise RuntimeError("T4: decode rows are not contiguous")
    sl = slice(d0, n)
    nd = n - d0
    ks, ms, ks0, ms0 = kc[::g, sl], m[::g, sl], kc0[::g, sl], m0[::g, sl]
    prs = pr[:, sl].reshape(KVH, g, nd, Lk).mean(1).double()
    D = Dany[:, sl]
    el = al[sl] & ~ex[0, sl]
    e_cnt = el.sum(-1).double()
    bs = torch.tensor([_bucket(int(x) - CFG.prefill) for x in qp[sl].tolist()], device=dev)
    act = STATE[li]["act"][:, :Lk]
    live = qp[sl].view(1, nd, 1) >= st["dpos"][:, None, :Lk]
    cols = [torch.full((nd,), float(KVH), dtype=torch.float64, device=dev), e_cnt * KVH]
    for ai in range(len(ACTIONS)):
        cols.append(((act == ai + 1)[:, None, :] & live & el[None]).sum(-1).double().sum(0))
    Dh, Dk, Dd = (((act == ai + 1)[:, None, :] & live) for ai in range(len(ACTIONS)))
    cols += [(ks0 & ~ks).sum(-1).double().sum(0), (ms0 & ~ms).sum(-1).double().sum(0),
             (prs * ms).sum(-1).sum(0), (prs * (ms & D)).sum(-1).sum(0)]
    prs0 = prs if pr0 is None else pr0[:, sl].reshape(KVH, g, nd, Lk).mean(1).double()
    cols += [(prs0 * (ms0 & D)).sum(-1).sum(0), (prs0 * ms0).sum(-1).sum(0)]
    cols += [(ks & Dh).sum(-1).double().sum(0), (ms & Dh).sum(-1).double().sum(0),
             (ms & (Dk | Dd)).sum(-1).double().sum(0), (ks & Dd).sum(-1).double().sum(0)]
    sm["rec"].index_add_(0, bs, torch.stack(cols, 1))
    if not CFG.pages:
        return
    kse, mse = ks & el[None], ms & el[None]
    views = (kse, mse, kse & ~Dh & ~Dd, kse & Dh, mse & ~(Dh | Dk | Dd), mse & Dh)
    for si, msk in enumerate(views):
        for pi, P in enumerate(PAGES):
            lp = (Lk + P - 1) // P * P
            hits = torch.nn.functional.pad(msk, (0, lp - Lk)).view(KVH, nd, lp // P, P).sum(-1)
            for b in range(len(BUCKETS)):
                rows = bs == b
                if bool(rows.any()):
                    hb = hits[:, rows].flatten()
                    hb = hb[hb > 0]                      # touched pages only (bin 0 stays empty)
                    sm["hist"][P][si, b] += torch.bincount(hb, minlength=P + 1)[:P + 1].double()
            tp = hits > 0
            prev = st["prev_tp"].get((si, P))
            prevs = torch.cat([(prev if prev is not None else torch.zeros_like(tp[:, :1]))[:, -1:], tp[:, :-1]], 1)
            has = torch.ones(nd, dtype=torch.bool, device=dev)
            if prev is None:
                has[0] = False
            tot = (tp.sum(-1).double().sum(0)) * has
            reu = ((tp & prevs).sum(-1).double().sum(0)) * has
            sm["reuse"][pi, si, :, 0].index_add_(0, bs, tot)
            sm["reuse"][pi, si, :, 1].index_add_(0, bs, reu)
            st["prev_tp"][(si, P)] = tp[:, -1:].clone()


def stats_to_json(li: int) -> dict:
    st, sm = STATE[li], STATS[li]
    # end of item: demoted tokens per action (all KV heads) and |E| * KVH at the last row (host residual records)
    out = {"rec": sm["rec"].tolist(), "dem": dict(zip(ACTIONS, st["dem"])), "zero_score_demoted": st["zero"],
           "E_end": float(max(0, st["w"] - SINKS) * st["act"].shape[0])}
    if CFG.pages:
        out["hist"] = {str(P): sm["hist"][P].tolist() for P in PAGES}
        out["reuse"] = {str(P): sm["reuse"][i].tolist() for i, P in enumerate(PAGES)}
    return out


def t4_attend_h(q, k, v, li, name, scale, kw, rk, rv, chk=None):
    CFG.calls[li] = CFG.calls.get(li, 0) + 1
    if S4.DIAG["on"]:
        raise RuntimeError("sel14_q: DIAG is not implemented for _h runs")
    p = S4.parse(name)
    if p.get("jcal") or p.get("mass") or p["eb"] or p["tw"] or p["crit"][:2] != "2s" or not p["K"]:
        raise ValueError(f"{name}: _h needs an x2s<M> criterion with K > 0 and a plain / _j / _p correction")
    kq, vq, R, _ = S4.Q5.base("kivi128", k, v, li)
    z = 4
    kq[:, :z], vq[:, :z] = k[:, :z], v[:, :z]
    if chk is not None or rk >= 16 or rv >= 16:
        raise ValueError("T4: only chk=None and rk, rv < 16 are used")
    kh = k if rk >= 16 else kq + Q14._res(k - kq, rk)
    vh = v if rv >= 16 else vq + Q14._res(v - vq, rv)
    pol, ref = CFG.policy, CFG.ref
    need = set() if pol is None else {a for _, a in pol.tiers}
    if ref is not None:
        need = {ref}
    if (need or ref) and (rk, rv) != (5, 4):
        raise ValueError("T4: demotion is defined from the h54 record")
    if "h32" in need:
        # re-encode from the h54 record (the host has no FP16 copy): same _res calls as the stack's h54 path
        k32 = kq + Q14._res(Q14._res(k - kq, rk), 3)
        v32 = vq + Q14._res(Q14._res(v - vq, rv), 2)
    k_enc = k
    K, M, lam = p["K"], int(p["crit"][2:]), p["lam"]
    if (M, K, R) != (M_RUN, K_RUN, WIN):
        raise ValueError(f"T4 expects M=128, K=64, R=128; got {M}, {K}, {R}")
    H_, Lq, _ = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    if ref == "h32":                                   # uniform reference: substitute the records themselves
        kh, vh = k32, v32
    kx, vx, kqx, vqx, khx, vhx = (t.repeat_interleave(g, 0) for t in (k, v, kq, vq, kh, vh))
    if "h32" in need and ref is None:
        k32x, v32x = k32.repeat_interleave(g, 0), v32.repeat_interleave(g, 0)
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
        if pol is None and ref is None:                 # original path (bitwise sel14_q._attend_h)
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
            o = (torch.einsum("hqk,hkd->hqd", pr, vqx) + torch.einsum("hqk,hkd->hqd", pr * ex, vx - vqx)
                 + torch.einsum("hqk,hkd->hqd", pr * m, vhx - vqx))
            if CFG.pages:
                _state(li, KVH, Lk, q.device)
                _record(li, qpos, kc, m, kc, m, pr, torch.zeros(KVH, n, Lk, dtype=torch.bool, device=q.device),
                        ex, al, g, KVH, n, Lk, q.device)
            out[:, r0:r1] = o
            continue
        s_corr = s - (lam / 100) * s2 / 2 if lam else s
        c1 = fin(s).masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        kc = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, c1.topk(min(M, Lk), -1).indices, True)
        kc = kc.repeat_interleave(g, 0) & ~bad
        if ref is not None:                             # uniform reference through substitution
            if ref == "del":
                sh = s_corr
            Dh32 = Dkey = Ddel = None
            keepv = None if ref == "h32" else torch.zeros_like(kc)
        else:
            cu = fin(s).reshape(KVH, g, n, Lk).mean(1)
            _schedule(li, cu, qpos, KVH, Lk, q.device)
            Dh32, Dkey, Ddel = (d.repeat_interleave(g, 0) for d in _masks(li, qpos, n, KVH, Lk, q.device))
            sh0 = sh
            if "h32" in need:
                sh = torch.where(Dh32, torch.einsum("hqd,hkd->hqk", qr, k32x) * scale, sh)
            sh = torch.where(Ddel, s_corr, sh)
            keepv = ~(Dh32 | Dkey | Ddel)
        s1 = torch.where(kc, sh, s_corr)
        c2 = fin(s1).masked_fill(~kc, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
        m = torch.zeros_like(c2, dtype=torch.bool).scatter_(-1, c2.topk(min(K, Lk), -1).indices, True).repeat_interleave(g, 0)
        m = m & kc
        sf = torch.where(m | kc, sh, s)
        if lam:
            sf = torch.where(m | ex | kc, sf, sf - (lam / 100) * s2 / 2)
        pr = fin(sf)
        o = torch.einsum("hqk,hkd->hqd", pr, vqx) + torch.einsum("hqk,hkd->hqd", pr * ex, vx - vqx)
        if keepv is None:
            o = o + torch.einsum("hqk,hkd->hqd", pr * m, vhx - vqx)
        else:
            o = o + torch.einsum("hqk,hkd->hqd", pr * (m & keepv), vhx - vqx)
            if Dh32 is not None and "h32" in need:
                o = o + torch.einsum("hqk,hkd->hqd", pr * (m & Dh32), v32x - vqx)
        if ref is None:
            with torch.no_grad():                       # the undemoted selection on the same inputs (counterfactual)
                s10 = torch.where(kc, sh0, s_corr)
                c20 = fin(s10).masked_fill(~kc, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
                m0 = torch.zeros_like(c20, dtype=torch.bool).scatter_(-1, c20.topk(min(K, Lk), -1).indices,
                                                                      True).repeat_interleave(g, 0) & kc
                sf0 = torch.where(m0 | kc, sh0, s)
                if lam:
                    sf0 = torch.where(m0 | ex | kc, sf0, sf0 - (lam / 100) * s2 / 2)
                pr0 = fin(sf0)
                Dany = (~keepv)[::g]
                _record(li, qpos, kc, m, kc, m0, pr, Dany, ex, al, g, KVH, n, Lk, q.device, pr0=pr0)
                del sf0, pr0
        out[:, r0:r1] = o
    return out
