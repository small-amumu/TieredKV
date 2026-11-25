# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""kbench5: fused two-stage merge for the layered-KV decode step (rootcause fork, exploratory; FA3-style migration item 1).
Builds on kbench4 (same base kernel, tile selection, phase-B merge, timing helpers) and adds:
  one<K>   one-stage reference path at any K (tile stage-1 keeps next_pow2(K) per tile, stage 2 the global top-K); the
           K selected tokens get exact keys and values (kbench4 fixA / fixB)
  two<M>k<K>  two-stage path: stage 1-2 select M candidates by base attention; then two kernels replace the earlier
           gather / einsum / topk / gather chain (several launches with host-visible dependencies):
             fix2A  grid (batch * KV head, M / KCA): zero-copy exact keys of the candidates, their exact scores se and
                    base scores sb (packed codes), written to a small GPU scratch [BK, 2, GP, M]
             fix2B  grid (batch * KV head, M / CC): every program re-ranks all M candidates from the scratch (the
                    attention of each candidate under the scores with the M candidates exact, group mean over the g query
                    heads, exactly the S3 / S4 x2s criterion without the debias), finds the top-K by bisection on the
                    probability bits (identical in every program: same inputs, same reductions; a
                    program takes its own lanes out of the full vector exactly), then for its own
                    CC candidates reads the exact values of the selected ones only (masked zero-copy loads: an unselected
                    lane issues no PCIe request) and writes the correction relative to the base in fixA's format: every
                    candidate gets its exact score, the selected ones also the exact value (dN = ee_sel . v_exact +
                    (ee_unsel - eb) . v_base, dZ = sum ee - sum eb, Mc)
             fixB   unchanged (kbench4)
           so the whole step is base, sel1, sel2, fix2A, fix2B, fixB in one CUDA graph; the two PCIe round trips (keys,
           then values) remain dependent, everything in between stays on the GPU.
The debias (_j) term is not in the kernels yet (j0 semantics). Correctness: python kbench5.py --check (fp32 torch
reference of the two-stage semantics, and of one<K> via kbench4.reference). Timing: whole step and selection + merge,
cold cache (layer copies streaming > 2x L2), interleaved rounds, bootstrap interval of the per-round ratio.
Usage: python kbench5.py [--check] [--shapes 1x8192,1x32768,8x8192,8x32768,32x8192] [--out kbench5.json]"""
import argparse
import json
import random
import statistics as S
import time

import torch
import triton
import triton.language as tl

import kbench4 as K4
from kbench4 import D, G, GP, KVH, P, _kth_bits, _planes, fixA_kernel, fixB_kernel, sel_stage1, sel_stage2


@triton.jit
def fix2A_kernel(q_ptr, qs_ptr, idx_ptr, kex_ptr, kp_ptr, meta_ptr, bias_ptr, sc_out, L, sm_scale, GH: tl.constexpr,
                 GP: tl.constexpr, P: tl.constexpr, M: tl.constexpr, KC: tl.constexpr, RC: tl.constexpr):
    """exact and base scores of a chunk of KC candidates (zero-copy exact keys); scratch layout [BK, 2, GP, M]."""
    bk = tl.program_id(0)
    c = tl.program_id(1)
    og = tl.arange(0, GP)
    op = tl.arange(0, P)
    gm = og < GH
    D4: tl.constexpr = 4 * P
    od = tl.arange(0, D4)
    ok = c * KC + tl.arange(0, KC)
    tok = tl.load(idx_ptr + bk * M + ok)
    row = bk * L + tok
    qf = tl.load(q_ptr + (bk * GH + og[:, None]) * D4 + od[None, :], mask=gm[:, None], other=0.0)
    kx = tl.load(kex_ptr + row[:, None] * D4 + od[None, :])
    se = tl.dot(qf, tl.trans(kx)) * sm_scale                              # [GP, KC]
    mt = tl.load(meta_ptr + row[:, None] * 4 + tl.arange(0, 4)[None, :]).to(tl.float32)
    sc_, zp_ = tl.split(tl.reshape(mt, [KC, 2, 2]))
    ksr, vsc = tl.split(sc_)
    kzr, vzp = tl.split(zp_)
    qb = q_ptr + (bk * GH + og[:, None]) * D4 + op[None, :]
    kb = tl.load(kp_ptr + row[:, None] * P + op[None, :])
    raw = (tl.dot(tl.load(qb, mask=gm[:, None], other=0.0), tl.trans((kb & 3).to(tl.float16)))
           + tl.dot(tl.load(qb + P, mask=gm[:, None], other=0.0), tl.trans(((kb >> 2) & 3).to(tl.float16)))
           + tl.dot(tl.load(qb + 2 * P, mask=gm[:, None], other=0.0), tl.trans(((kb >> 4) & 3).to(tl.float16)))
           + tl.dot(tl.load(qb + 3 * P, mask=gm[:, None], other=0.0), tl.trans(((kb >> 6) & 3).to(tl.float16))))
    qsum = tl.load(qs_ptr + bk * GH + og, mask=gm, other=0.0)
    sb = (raw * ksr[None, :] + qsum[:, None] * kzr[None, :]) * sm_scale
    if RC == 2:
        sb += tl.load(bias_ptr + (bk * GH + og[:, None]) * L + tok[None, :], mask=gm[:, None], other=0.0)
    so = sc_out + (bk * 2 * GP + og[:, None]) * M + ok[None, :]
    tl.store(so, se)
    tl.store(so + GP * M, sb)


@triton.jit
def _crit(sc_in, bk, lanes, Mx, Z1, og, GH: tl.constexpr, GP: tl.constexpr, M: tl.constexpr):
    # lanes >= M (padding up to a power of two) give probability 0 and are excluded by the caller
    """group-mean attention of the given candidate lanes under the scores with every candidate exact."""
    crit = tl.zeros(lanes.shape, tl.float32)
    for g in tl.static_range(GH):
        mx = tl.sum(tl.where(og == g, Mx, 0.0), 0)
        z1 = tl.sum(tl.where(og == g, Z1, 0.0), 0)
        se = tl.load(sc_in + (bk * 2 * GP + g) * M + lanes, mask=lanes < M, other=-float("inf"))
        crit += tl.exp(se - mx) / z1
    crit = crit / GH
    return tl.where((crit == crit) & (crit < 3.0e38), crit, 0.0)    # NaN / inf never outrank a finite candidate


@triton.jit
def fix2B_kernel(m_in, l_in, sc_in, idx_ptr, vex_ptr, vp_ptr, meta_ptr, pn_out, pz_out, pm_out, selm_out, L, nsplit,
                 kap_out, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
                 CC: tl.constexpr, NS: tl.constexpr, MASKV: tl.constexpr, USESEL: tl.constexpr = False,
                 MP: tl.constexpr = 0, RADIX: tl.constexpr = False):
    bk = tl.program_id(0)
    c = tl.program_id(1)
    og = tl.arange(0, GP)
    op = tl.arange(0, P)
    gm = og < GH
    D4: tl.constexpr = 4 * P
    os_ = tl.arange(0, NS)
    sm_ = os_[:, None] < nsplit
    mm = tl.load(m_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_, other=-float("inf"))
    ll = tl.load(l_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_, other=0.0)
    MV = tl.max(mm, 0)
    MV = tl.where(gm, MV, 0.0)
    ZV = tl.sum(ll * tl.exp(mm - MV[None, :]), 0)
    # first-stage denominators with every candidate exact: Z1 = Z_base - sum_cand exp(sb) + sum_cand exp(se)
    MW: tl.constexpr = M if MP == 0 else MP                                  # lane width (power of two >= M)
    oj = tl.arange(0, MW)
    jm = oj < M
    sa = sc_in + (bk * 2 * GP + og[:, None]) * M + oj[None, :]
    se_all = tl.load(sa, mask=gm[:, None] & jm[None, :], other=-float("inf"))
    sb_all = tl.load(sa + GP * M, mask=gm[:, None] & jm[None, :], other=-float("inf"))
    Mx = tl.where(gm, tl.maximum(MV, tl.max(se_all, 1)), 0.0)
    ze = tl.sum(tl.exp(se_all - Mx[:, None]), 1)
    Z1 = ZV * tl.exp(MV - Mx) - tl.sum(tl.exp(sb_all - Mx[:, None]), 1) + ze
    # the subtraction can cancel (a dominant base candidate whose exact score is far lower): Z1 is at least the exact
    # candidate mass, so floor it there (keeps the probabilities finite and non-negative, the bit order valid)
    # (and strictly positive: ze itself underflows when every exact candidate score is ~87 nats below the base max)
    Z1 = tl.where(gm, tl.maximum(Z1, tl.maximum(ze, 1e-30)), 1.0)
    if c == 0:   # kappa = removed base candidate mass / first-stage denominator (cancellation amplification)
        tl.store(kap_out + bk * GP + og, tl.sum(tl.exp(sb_all - Mx[:, None]), 1) / Z1, mask=gm)
    oc = c * CC + tl.arange(0, CC)
    if USESEL:   # unfused control: the selection was made outside (torch); no ranking work here
        sel = tl.load(selm_out + bk * M + oc) != 0
    else:
        # top-K of the M candidates: bisection on the probability bits, ties by candidate order (as sel_stage2)
        b32 = _crit(sc_in, bk, oj, Mx, Z1, og, GH, GP, M).to(tl.int32, bitcast=True)
        b32 = tl.where(jm, b32, -1)                                            # padding lanes never count, never tie
        if RADIX:
            lo = _kth_bits(b32, K)
        else:
            lo = 0
            up = 0x7F800000
            for _ in range(31):
                mid = lo + (up - lo + 1) // 2
                ok_ = tl.sum((b32 >= mid).to(tl.int32), 0) >= K
                lo = tl.where(ok_, mid, lo)
                up = tl.where(ok_, up, mid - 1)
        n_gt = tl.sum((b32 > lo).to(tl.int32), 0)
        tie_before = tl.sum(((b32 == lo) & (oj < c * CC)).to(tl.int32), 0)
        # this program's CC candidates
        bc = tl.sum(tl.where(oj[:, None] == oc[None, :], b32[:, None], 0), 0)    # the chunk's lanes of the full vector, exactly
        tie = bc == lo
        sel = (bc > lo) | (tie & (n_gt + tie_before + tl.cumsum(tie.to(tl.int32), 0) - 1 < K))
        tl.store(selm_out + bk * M + oc, sel.to(tl.int8))
    tok = tl.load(idx_ptr + bk * M + oc)
    row = bk * L + tok
    sp = sc_in + (bk * 2 * GP + og[:, None]) * M + oc[None, :]
    se = tl.load(sp, mask=gm[:, None], other=-float("inf"))
    sb = tl.load(sp + GP * M, mask=gm[:, None], other=-float("inf"))
    Mc = tl.maximum(MV, tl.max(se, 1))
    Mc = tl.where(gm, Mc, 0.0)
    ee = tl.exp(se - Mc[:, None])
    eb = tl.exp(sb - Mc[:, None])
    dz = tl.sum(ee, 1) - tl.sum(eb, 1)
    ees = tl.where(sel[None, :], ee, 0.0)
    wb = eb - tl.where(sel[None, :], 0.0, ee)                            # base-value weight to subtract
    if MASKV:   # exact values of the selected candidates only
        vx = tl.load(vex_ptr + row[:, None] * D4 + tl.arange(0, D4)[None, :], mask=sel[:, None], other=0.0)
    else:       # timing control: read all CC values (ees is zero for the unselected, so the result is the same)
        vx = tl.load(vex_ptr + row[:, None] * D4 + tl.arange(0, D4)[None, :])
    mt = tl.load(meta_ptr + row[:, None] * 4 + tl.arange(0, 4)[None, :]).to(tl.float32)
    sc_, zp_ = tl.split(tl.reshape(mt, [CC, 2, 2]))
    ksr, vsc = tl.split(sc_)
    kzr, vzp = tl.split(zp_)
    # every fp16 operand is split into hi + lo halves (about 22 bits of mantissa in total, no pre-scale, so no overflow
    # for value scales above 16); the four tensor-core dots per plane recover the fp32 path to ~1e-6
    wv = wb * vsc[None, :]
    wbh = wv.to(tl.float16)
    wbl = ((wv - wbh.to(tl.float32)) * 2048.0).to(tl.float16)       # lo scaled by 2^11: stays out of the subnormal range
    wbz = tl.sum(wb * vzp[None, :], 1)[:, None]
    eh = ees.to(tl.float16)
    el = ((ees - eh.to(tl.float32)) * 2048.0).to(tl.float16)
    e0, e1, e2, e3 = _planes(tl.dot(eh, vx) + tl.dot(el, vx) * (1.0 / 2048.0), GP, P)
    vb = tl.load(vp_ptr + row[:, None] * P + op[None, :])
    o = (bk * tl.num_programs(1) + c) * GP + og
    pb = pn_out + o[:, None] * D4 + op[None, :]
    c0 = (vb & 3).to(tl.float16)
    c1_ = ((vb >> 2) & 3).to(tl.float16)
    c2 = ((vb >> 4) & 3).to(tl.float16)
    c3 = ((vb >> 6) & 3).to(tl.float16)
    tl.store(pb, e0 - tl.dot(wbh, c0) - tl.dot(wbl, c0) * (1.0 / 2048.0) - wbz, mask=gm[:, None])
    tl.store(pb + P, e1 - tl.dot(wbh, c1_) - tl.dot(wbl, c1_) * (1.0 / 2048.0) - wbz, mask=gm[:, None])
    tl.store(pb + 2 * P, e2 - tl.dot(wbh, c2) - tl.dot(wbl, c2) * (1.0 / 2048.0) - wbz, mask=gm[:, None])
    tl.store(pb + 3 * P, e3 - tl.dot(wbh, c3) - tl.dot(wbl, c3) * (1.0 / 2048.0) - wbz, mask=gm[:, None])
    tl.store(pz_out + o, dz, mask=gm)
    tl.store(pm_out + o, Mc, mask=gm)


class Step5(K4.Step):
    """mode 'one': K exact rows (keys and values); mode 'two': M candidates with exact keys, top-K of them with exact
    values. Production path of kbench4 (packed metadata, wide rows, two-phase merge, separate selection kernels)."""

    def __init__(self, B, L, K, rc, mode="one", M=128, kca=16, cc=32, maskv=True, **kw):
        self.mode, self.Kv, self.maskv = mode, K, maskv
        kw = dict(kw, fixp=True, fuse=False, kc=16)
        # stage-2 capacity: (L / T) tiles x next_pow2(selected per tile) candidates must fit one program (16384);
        # widen the stage-1 tile when it would not (e.g. 224 candidates at L = 70001)
        kt = triton.next_power_of_2(M if mode == "two" else K)
        T = kw.get("T", 1024)
        while kt * triton.cdiv(L, T) > 16384:
            T *= 2
        kw["T"] = T
        if mode == "two":
            assert M % cc == 0 and M % kca == 0 and K <= M and cc >= 16 and cc & (cc - 1) == 0
            super().__init__(B, L, M, rc, **kw)                              # selection buffers sized for M
            self.NC2, self.kca, self.cc = M, kca, cc
            self.sc = torch.empty(self.BK * 2 * GP * M, device="cuda")
            self.selm = torch.zeros(self.BK, M, device="cuda", dtype=torch.int8)
            self.kap = torch.zeros(self.BK * GP, device="cuda")
            self.nch2 = M // cc
            assert self.nch2 <= self.nch
        else:
            assert K % 16 == 0
            super().__init__(B, L, K, rc, **kw)
        self.KT = triton.next_power_of_2(self.K)                             # per-tile candidates (power of two)
        self.cand = torch.empty(self.BK, self.ntile * self.KT, device="cuda", dtype=torch.int64)

    def select(self):
        # FUSE=True: per-head max / sum reduced from the split partials inside stage 1 (the M_in / Z_in pointers are unused)
        sel_stage1[(self.BK, self.ntile)](self.s, self.Z, self.Z, self.cand, self.L, self.ntile, self.m, self.l, self.nsplit,
                                          GH=G, GP=GP, T=self.T, K=self.KT, BISECT=self.bisect, FUSE=True, NS=self.NS,
                                          RADIX=self.radix, num_warps=4)
        C = self.ntile * self.KT
        assert C <= 16384, "stage-2 candidates exceed one program"
        sel_stage2[(self.BK,)](self.cand, self.idx, C, K=self.K, CP=triton.next_power_of_2(C), RADIX=self.radix, num_warps=8)

    def fix(self):
        if self.mode == "one":
            return super().fix()
        fix2A_kernel[(self.BK, self.NC2 // self.kca)](self.q, self.qs, self.idx, self.kz_, self.kp, self.mt, self.bias, self.sc,
                                                     self.L, self.scale, GH=G, GP=GP, P=P, M=self.NC2, KC=self.kca, RC=self.rc,
                                                     num_warps=4)
        fix2B_kernel[(self.BK, self.nch2)](self.m, self.l, self.sc, self.idx, self.vz_, self.vp, self.mt, self.pn, self.pz, self.pm,
                                           self.selm, self.L, self.nsplit, self.kap, GH=G, GP=GP, P=P, M=self.NC2, K=self.Kv, CC=self.cc,
                                           NS=self.NS, MASKV=self.maskv, MP=triton.next_power_of_2(self.NC2), RADIX=self.radix, num_warps=4)
        return fixB_kernel[(self.BK, 4)](self.m, self.l, self.acc, self.pn, self.pz, self.pm, self.out, self.nsplit, self.nch2,
                                         GH=G, GP=GP, P=P, NS=self.NS, NC=triton.next_power_of_2(self.nch2), UNROLL=self.unroll,
                                         num_warps=4)

    def fix_unfused(self):
        """same work as fix() with the second-stage ranking done by separate torch ops between the kernels (the
        unfused chain: scores kernel, torch softmax / group mean / top-K / mask, correction kernel, merge kernel)."""
        fix2A_kernel[(self.BK, self.NC2 // self.kca)](self.q, self.qs, self.idx, self.kz_, self.kp, self.mt, self.bias, self.sc,
                                                     self.L, self.scale, GH=G, GP=GP, P=P, M=self.NC2, KC=self.kca, RC=self.rc,
                                                     num_warps=4)
        sc = self.sc.view(self.BK, 2, GP, self.NC2)[:, :, :G]
        ms = self.m.view(self.BK, self.nsplit, GP)[..., :G]
        ls = self.l.view(self.BK, self.nsplit, GP)[..., :G]
        MV = ms.amax(1)
        ZV = (ls * torch.exp(ms - MV[:, None])).sum(1)
        se, sb = sc[:, 0], sc[:, 1]
        Mx = torch.maximum(MV, se.amax(-1))
        ze = torch.exp(se - Mx[..., None]).sum(-1)
        Z1 = torch.maximum(ZV * torch.exp(MV - Mx) - torch.exp(sb - Mx[..., None]).sum(-1) + ze, ze.clamp_min(1e-30))
        crit = torch.nan_to_num((torch.exp(se - Mx[..., None]) / Z1[..., None]).mean(1), nan=0.0, posinf=0.0)
        self.selm.zero_().scatter_(1, crit.topk(self.Kv, -1).indices, 1)
        fix2B_kernel[(self.BK, self.nch2)](self.m, self.l, self.sc, self.idx, self.vz_, self.vp, self.mt, self.pn, self.pz, self.pm,
                                           self.selm, self.L, self.nsplit, self.kap, GH=G, GP=GP, P=P, M=self.NC2, K=self.Kv,
                                           CC=self.cc, NS=self.NS, MASKV=self.maskv, USESEL=True, MP=triton.next_power_of_2(self.NC2), RADIX=self.radix, num_warps=4)
        return fixB_kernel[(self.BK, 4)](self.m, self.l, self.acc, self.pn, self.pz, self.pm, self.out, self.nsplit, self.nch2,
                                         GH=G, GP=GP, P=P, NS=self.NS, NC=triton.next_power_of_2(self.nch2), UNROLL=self.unroll,
                                         num_warps=4)

    def phase(self, name):
        """one phase of the step alone (buffers from a previous full step), for the per-phase timing."""
        if name == "base":
            return self.base()
        if name == "select":
            return self.select()
        if self.mode == "one":
            if name == "fixA":
                return fixA_kernel[(self.BK, self.nch)](self.q, self.qs, self.m, self.idx, self.kz_, self.vz_, self.kp, self.vp, self.mt,
                                                        self.bias, self.pn, self.pz, self.pm, self.prev, self.ck, self.cv, self.nk, self.nv,
                                                        self.L, self.nsplit, self.scale, GH=G, GP=GP, P=P, K=self.K, KC=self.kc,
                                                        NS=self.NS, RC=self.rc, CACHE=self.cache, HILO=self.hilo, num_warps=4)
            nc = self.nch
        else:
            if name == "fix2A":
                return fix2A_kernel[(self.BK, self.NC2 // self.kca)](self.q, self.qs, self.idx, self.kz_, self.kp, self.mt, self.bias,
                                                                    self.sc, self.L, self.scale, GH=G, GP=GP, P=P, M=self.NC2,
                                                                    KC=self.kca, RC=self.rc, num_warps=4)
            if name == "fix2B":
                return fix2B_kernel[(self.BK, self.nch2)](self.m, self.l, self.sc, self.idx, self.vz_, self.vp, self.mt, self.pn, self.pz,
                                                          self.pm, self.selm, self.L, self.nsplit, self.kap, GH=G, GP=GP, P=P, M=self.NC2,
                                                          K=self.Kv, CC=self.cc, NS=self.NS, MASKV=self.maskv, MP=triton.next_power_of_2(self.NC2), RADIX=self.radix, num_warps=4)
            nc = self.nch2
        assert name == "fixB", name
        return fixB_kernel[(self.BK, 4)](self.m, self.l, self.acc, self.pn, self.pz, self.pm, self.out, self.nsplit, nc,
                                         GH=G, GP=GP, P=P, NS=self.NS, NC=triton.next_power_of_2(nc), UNROLL=self.unroll, num_warps=4)

    def sf(self):
        self.select()
        self.fix()


def _deq(pk, sc, zp):
    planes = [(pk >> (2 * i)) & 3 for i in range(4)]
    c = torch.cat(planes, -1).float()                                    # channel = plane * P + j
    return c * sc.float()[..., None] + zp.float()[..., None]


def _realistic(st, seed=1):
    """overwrite the pinned exact K / V in place with the dequantised base plus uniform rounding noise inside the 2-bit
    bin (exact = scale * (code + u) + zero, u ~ U(-1/2, 1/2)), as for real quantised data."""
    g = torch.Generator().manual_seed(seed)
    for host, (pk, sc, zp) in ((st.kh, (st.kp, st.ks, st.kz)), (st.vh, (st.vp, st.vs, st.vz))):
        d = _deq(pk, sc, zp).cpu()
        u = torch.rand(d.shape, generator=g) - 0.5
        d = d + u * sc.float().cpu()[..., None]
        if st.rc == 2 and host is st.kh:   # stored keys are centred; the exact key is centred + R(t) m (bias = q . R(t) m)
            t = torch.arange(st.L, dtype=torch.float64)[:, None] * st.invf64.cpu()[None]
            cos, sin = t.cos().float(), t.sin().float()
            m = st.mpre.cpu().repeat(st.B, 1)                                # [BK, D], KV head of each row
            ma, mb = m[:, None, :D // 2], m[:, None, D // 2:]
            d = d + torch.cat([ma * cos - mb * sin, mb * cos + ma * sin], -1)
        host.copy_(d.half())


def _scores(st):
    """fp32 base scores (dequantised planar codes, rotary bias as kbench4.reference), exact scores, base and exact values."""
    BK, L = st.BK, st.L

    def deq(pk, sc, zp):
        planes = [(pk >> (2 * i)) & 3 for i in range(4)]
        c = torch.cat(planes, -1).float()
        return c * sc.float()[..., None] + zp.float()[..., None]
    kd, vd = deq(st.kp, st.ks, st.kz), deq(st.vp, st.vs, st.vz)
    q = st.q.float().view(BK, G, D)
    s = torch.einsum("bgd,bld->bgl", q, kd) * st.scale
    if st.rc == 2:
        t = torch.arange(L, device=q.device, dtype=torch.float64)[:, None] * st.invf64[None]
        cos, sin = t.cos().float(), t.sin().float()
        ca, cb = st.ca.view(BK, G, D // 2), st.cb.view(BK, G, D // 2)
        s = s + (torch.einsum("bgi,li->bgl", ca, cos) + torch.einsum("bgi,li->bgl", cb, sin)) * st.scale
    kx, vx = st.kh.to(q.device).float(), st.vh.to(q.device).float()
    sx = torch.einsum("bgd,bld->bgl", q, kx) * st.scale
    return s, sx, vd, vx


def ref_two(st, cand=None, mut=None, return_crit=False):
    """fp32 torch reference of the two-stage step (x2s semantics, no debias): candidates = top-M of the base group-mean
    attention (or the given cand), their scores exact; among them the top-K by the group-mean attention under those
    scores get exact values. Returns (out, candidates, selected tokens). mut: near-miss mutants the check must reject:
    'base_rank' (rank candidates by base attention), 'z_base' (rank with the base denominators, no candidate
    correction), 'base_unsel' (unselected candidates keep base scores)."""
    s, sx, vd, vx = _scores(st)
    BK, M, K = st.BK, st.NC2, st.Kv
    if cand is None:
        cand = torch.softmax(s, -1).mean(1).topk(M, -1).indices
    ce = cand[:, None, :].expand(BK, G, M)
    s1 = s.clone()
    s1.scatter_(2, ce, torch.gather(sx, 2, ce))
    if mut == "base_rank":
        c2 = torch.gather(torch.softmax(s, -1).mean(1), 1, cand)
    elif mut == "z_base":
        mx = s1.amax(-1, keepdim=True)
        c2 = torch.gather((torch.exp(s1 - mx) / torch.exp(s - mx).sum(-1, keepdim=True)).mean(1), 1, cand)
    else:
        c2 = torch.gather(torch.softmax(s1, -1).mean(1), 1, cand)
    sel = torch.gather(cand, 1, c2.topk(K, -1).indices)
    vfin = vd.clone()
    vfin.scatter_(1, sel[..., None].expand(BK, K, D), torch.gather(vx, 1, sel[..., None].expand(BK, K, D)))
    sf = s1
    if mut == "base_unsel":
        sf = s.clone()
        se_ = sel[:, None, :].expand(BK, G, K)
        sf.scatter_(2, se_, torch.gather(sx, 2, se_))
    out = torch.einsum("bgl,bld->bgd", torch.softmax(sf, -1), vfin)
    if return_crit:   # second-stage criterion of every candidate (for tie-aware comparisons)
        return out, cand, sel, c2
    return out, cand, sel


def _near_tie(st, ksel, sel_ref, rtol=1e-4):
    """True when every selection difference between the kernel and the fp32 reference is a near tie: per row, the
    smallest reference criterion among the kernel-only picks is within rtol of the largest among the reference-only picks."""
    _, cand, _, c2 = ref_two(st, st.idx, return_crit=True)
    for r in range(st.BK):
        cr = dict(zip(cand[r].tolist(), c2[r].tolist()))
        ko = set(ksel[r].tolist()) - set(sel_ref[r].tolist())
        ro = set(sel_ref[r].tolist()) - set(ksel[r].tolist())
        if ko or ro:
            a, b = min(cr[t] for t in ko), max(cr[t] for t in ro)
            if abs(a - b) > rtol * max(abs(a), abs(b), 1e-30):
                return False
    return True


def _kernel_sel(st):
    """the kernel's selected tokens per row (asserts exactly K per row and valid indices first)."""
    assert bool(((st.selm == 0) | (st.selm == 1)).all()), "selection mask has unwritten lanes (sentinel left)"
    cnt = st.selm.long().sum(1)
    assert bool((cnt == st.Kv).all()), f"selected per row {cnt.tolist()}"
    assert bool(((st.idx >= 0) & (st.idx < st.L)).all()), "candidate index out of range"
    return st.idx[st.selm.bool()].view(st.BK, st.Kv)


def _mismatch(a, b):
    return sum(len(set(y.tolist()) - set(x.tolist())) for x, y in zip(a.cpu(), b.cpu()))


def _overlap(a, b):
    return sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a.cpu(), b.cpu())) / b.numel()


TOL = 2e-4   # observed kernel-vs-reference error is <= 6e-5 (fp16 rotary bias and fp16 value dot)


def check():
    ok_all = True
    caught_any = {}   # a mutant must be rejected in at least one configuration (in some it is genuinely equivalent:
                      # at long L the candidate correction of the denominators can leave the top-K unchanged)
    for rc in (0, 2):
        for L in (5000, 70001):
            for qs in (1.0, 4.0):
                for M, K, cc in ((128, 64, 32), (128, 64, 16), (128, 32, 32), (128, 32, 16), (224, 96, 32), (144, 48, 16)):
                        st = Step5(2, L, K, rc, mode="two", M=M, cc=cc, qscale=qs)
                        st.selm.fill_(7)                                   # sentinel: every lane must be written
                        st.step()
                        torch.cuda.synchronize()
                        o_ref, cand_ref, _ = ref_two(st)
                        o_own, _, sel_own = ref_two(st, st.idx)
                        got = st.out.view(st.BK, G, D).clone()
                        ksel = _kernel_sel(st)
                        dup = int(sum(len(r) - len(set(r)) for r in st.idx.cpu().tolist()))
                        err = float((got - o_own).norm() / o_own.norm())
                        mis = _mismatch(ksel, sel_own)
                        co = _overlap(st.idx, cand_ref)
                        st.maskv = False                                   # unmasked value loads: same result
                        st.fix()
                        torch.cuda.synchronize()
                        dmask = float((st.out.view(st.BK, G, D) - got).norm() / got.norm())
                        st.maskv = True
                        tie_ok = mis == 0 or _near_tie(st, ksel, sel_own)          # fp16 bias vs fp32 reference: near ties may flip
                        good = err < TOL and dup == 0 and tie_ok and co >= 0.99 and dmask < 1e-6
                        ok_all &= good
                        print(f"check two rc={rc} L={L} qscale={qs} M={M} K={K} cc={cc}: candidates vs fp32 reference {co:.4f}; "
                              f"selected mismatches (same candidates) {mis}{' (near ties)' if mis and tie_ok else ''}; rel err vs fp32 reference {float((got - o_ref).norm() / o_ref.norm()):.3g}, "
                              f"same candidates {err:.3g}; unmasked vs masked {dmask:.2g} {'OK' if good else 'FAIL'}", flush=True)
                        if rc == 2 and qs == 1.0 and cc == 32:   # near-miss mutants must be rejected by the same test
                            for mut in ("base_rank", "z_base", "base_unsel"):
                                om, _, sm = ref_two(st, st.idx, mut=mut)
                                em, mm = float((got - om).norm() / om.norm()), _mismatch(ksel, sm)
                                caught = em >= TOL or mm > 0
                                caught_any[mut] = caught_any.get(mut, False) or caught
                                print(f"    mutant {mut}: rel err {em:.3g}, selected mismatches {mm} -> {'caught' if caught else 'NOT CAUGHT'}", flush=True)
                        del st
                        torch.cuda.empty_cache()
            for K in (64, 96):
                st = Step5(2, L, K, rc, mode="one", qscale=4.0)
                st.step()
                torch.cuda.synchronize()
                o_ref, _ = K4.reference(st, st.idx)
                err = float((st.out.view(st.BK, G, D) - o_ref).norm() / o_ref.norm())
                ok_all &= err < TOL
                print(f"check one rc={rc} L={L} K={K}: rel err vs reference (same selection) {err:.3g} {'OK' if err < TOL else 'FAIL'}", flush=True)
                del st
                torch.cuda.empty_cache()
    # all ties: zero queries give equal attention everywhere, so the selection must be the first K candidates in lane
    # order (exercises the tie rule, tie_before and the chunk offsets of every program)
    for K, cc in ((64, 32), (64, 16), (32, 32), (32, 16), (40, 32), (40, 16)):   # K = 40: the cut falls inside a chunk
        st = Step5(2, 5000, K, 0, mode="two", M=128, cc=cc)
        st.q.zero_()
        st.qs.zero_()
        st.step()
        torch.cuda.synchronize()
        ksel = _kernel_sel(st)
        mis = _mismatch(ksel, st.idx[:, :K])
        ok_all &= mis == 0
        print(f"check ties K={K} cc={cc}: selection vs first K candidates, mismatches {mis} {'OK' if mis == 0 else 'FAIL'}", flush=True)
        del st
        torch.cuda.empty_cache()
    # realistic 2-bit noise: exact = dequantised base + uniform rounding error inside the bin, so exact and base scores
    # are close (cancellation in the corrections is real) and kappa spans flat to peaked attention
    for hilo in (False, True):
        for rc in (0, 2):
            for L in (5000, 70001):
                for qs in (1.0, 4.0, 8.0):
                    st = Step5(2, L, 64, rc, mode="two", M=128, cc=32, qscale=qs, hilo=hilo)
                    _realistic(st)
                    st.step()
                    torch.cuda.synchronize()
                    o_own, _, sel_own = ref_two(st, st.idx)
                    got = st.out.view(st.BK, G, D)
                    err = float((got - o_own).norm() / o_own.norm())
                    ksel = _kernel_sel(st)
                    mis = _mismatch(ksel, sel_own)
                    kap = st.kap.view(st.BK, GP)[:, :G].flatten().cpu()
                    written = bool((kap > 0).all())
                    # hilo off is the old base kernel: reported only (its error grows with kappa); hilo on is gated
                    good = (err < TOL and mis == 0 and written) if hilo else (mis == 0 and written)
                    ok_all &= good
                    print(f"check realistic hilo={int(hilo)} rc={rc} L={L} qscale={qs}: kappa median {float(kap.median()):.3g} "
                          f"max {float(kap.max()):.3g}; selected mismatches {mis}; rel err {err:.3g} "
                          f"{'OK' if good else 'FAIL'}{'' if hilo else ' (reported)'}", flush=True)
                    if hilo and L == 5000 and qs == 4.0:   # near-miss mutants must be rejected on realistic data too
                        for mut in ("base_rank", "z_base", "base_unsel"):
                            om, _, sm = ref_two(st, st.idx, mut=mut)
                            em, mm = float((got - om).norm() / om.norm()), _mismatch(ksel, sm)
                            caught_any[mut] = caught_any.get(mut, False) or em >= TOL or mm > 0
                            print(f"    realistic mutant {mut}: rel err {em:.3g}, selected mismatches {mm}", flush=True)
                    del st
                    torch.cuda.empty_cache()
    # one-stage path on realistic data: hi / lo on both the base kernel and fixA (the subtracted base term must match)
    for rc in (0, 2):
        for qs in (1.0, 8.0):
            for hilo in (False, True):
                st = Step5(2, 5000, 96, rc, mode="one", qscale=qs, hilo=hilo)
                _realistic(st)
                st.step()
                torch.cuda.synchronize()
                o_ref, _ = K4.reference(st, st.idx)
                err = float((st.out.view(st.BK, G, D) - o_ref).norm() / o_ref.norm())
                good = err < TOL if hilo else True
                ok_all &= good
                print(f"check realistic one-stage K=96 hilo={int(hilo)} rc={rc} qscale={qs}: rel err {err:.3g} "
                      f"{'OK' if good else 'FAIL'}{'' if hilo else ' (reported)'}", flush=True)
                del st
                torch.cuda.empty_cache()
    # graph replay must read the CURRENT host rows (zero copy, nothing cached): rewrite the pinned K / V in place with
    # fresh rounding noise; the selection (base only) must not change, the output must follow the new exact rows
    for rc in (0, 2):
        st = Step5(2, 5000, 64, rc, mode="two", M=128, cc=32, qscale=4.0, hilo=True)
        _realistic(st, seed=1)
        replay = K4.graph_of(st.step)
        replay()
        torch.cuda.synchronize()
        o1, idx1 = st.out.clone(), st.idx.clone()
        _realistic(st, seed=2)
        replay()
        torch.cuda.synchronize()
        same_idx = bool(torch.equal(idx1, st.idx))
        o_new, _, _ = ref_two(st, st.idx)
        o_old_ref = o1.view(st.BK, G, D)
        err = float((st.out.view(st.BK, G, D) - o_new).norm() / o_new.norm())
        moved = float((st.out - o1).norm() / o1.norm())
        good = err < TOL and same_idx and moved > 10 * err
        ok_all &= good
        print(f"check replay after host rewrite rc={rc}: same candidates {same_idx}; rel err vs new reference {err:.3g}, "
              f"change vs old output {moved:.3g} {'OK' if good else 'FAIL'}", flush=True)
        del st, o_old_ref
        torch.cuda.empty_cache()
    # radix-select top-K threshold must give bitwise the same selection and output as the 31-pass bisection
    for mode, M, K, L in (("one", 128, 64, 5000), ("one", 128, 96, 70001), ("two", 128, 64, 5000), ("two", 224, 96, 70001),
                          ("two", 144, 48, 131072)):
        for rc in (0, 2):
            gh = torch.Generator().manual_seed(11)
            host = (torch.randn(2 * KVH, L, D, generator=gh).half().pin_memory(), torch.randn(2 * KVH, L, D, generator=gh).half().pin_memory())
            kw = dict(M=M, cc=16) if mode == "two" else {}
            a = Step5(2, L, K, rc, mode=mode, qscale=4.0, host=host, **kw)
            b = Step5(2, L, K, rc, mode=mode, qscale=4.0, host=host, radix=True, **kw)
            a.step()
            b.step()
            torch.cuda.synchronize()
            same = bool(torch.equal(a.out, b.out)) and all(set(x.tolist()) == set(y.tolist()) for x, y in zip(a.idx.cpu(), b.idx.cpu()))
            ok_all &= same
            print(f"check radix {mode} M={M} K={K} L={L} rc={rc}: bitwise same selection and output as bisection {same} "
                  f"{'OK' if same else 'FAIL'}", flush=True)
            del a, b
            torch.cuda.empty_cache()
    print(f"mutants rejected in at least one configuration: {caught_any}", flush=True)
    ok_all &= all(caught_any.values())
    ok_all &= check_b16()
    print("ALL PASS" if ok_all else "SOME FAIL", flush=True)


def check_b16():
    """bias16 (rotary bias stored in fp16): on realistic data, error against the fp32 reference with the kernel's own
    selection, next to the same error of the fp32-bias step on identical data. fp16 rounding of a bias of size |b| moves a
    logit by about |b| 2^-11, so the bound is looser than TOL: B16_TOL."""
    ok = True
    for mode, K, kw in (("one", 96, {}), ("two", 64, dict(M=128, cc=32)), ("two", 96, dict(M=224, cc=32))):
        for qs in (1.0, 4.0):
            errs = {}
            for b16 in (False, True):
                st = Step5(2, 5000, K, 2, mode=mode, qscale=qs, hilo=True, bias16=b16, seed=3, **kw)
                _realistic(st)
                st.step()
                torch.cuda.synchronize()
                o_ref = K4.reference(st, st.idx)[0] if mode == "one" else ref_two(st, st.idx)[0]
                errs[b16] = float((st.out.view(st.BK, G, D) - o_ref).norm() / o_ref.norm())
                del st
                torch.cuda.empty_cache()
            good = errs[True] < B16_TOL
            ok &= good
            print(f"check bias16 {mode} K={K} {kw} qscale={qs}: rel err fp16 bias {errs[True]:.3g}, fp32 bias {errs[False]:.3g} "
                  f"{'OK' if good else 'FAIL'}", flush=True)
    return ok


B16_TOL = 2e-3


VARIANTS = {"one96": dict(mode="one", K=96), "one96_aa": dict(mode="one", K=96), "one64": dict(mode="one", K=64),
            "one128": dict(mode="one", K=128), "one144": dict(mode="one", K=144), "one160": dict(mode="one", K=160),
            "two128k64_cc32": dict(mode="two", K=64, M=128, cc=32), "two144k48_cc16": dict(mode="two", K=48, M=144, cc=16),
            "two224k96_cc32": dict(mode="two", K=96, M=224, cc=32),
            "one96_rx": dict(mode="one", K=96, radix=True), "two128k64_rx": dict(mode="two", K=64, M=128, cc=32, radix=True),
            "two224k96_rx": dict(mode="two", K=96, M=224, cc=32, radix=True)}
GATE = 0.10   # two-stage passes when the upper end of the 95% interval of the per-round relative extra (cold whole step,
              # against one96 at matched transfer) is <= 10%


def _boot(xs, rnd):
    bs = sorted(S.median(rnd.choices(xs, k=len(xs))) for _ in range(5000))
    return [round(bs[124], 4), round(bs[4874], 4)]


def run(B, L, rounds):
    """one shape. Every variant has its own pinned exact K / V (no host-side reuse across variants) and its own n layer
    copies; whole step (base + select + merge) is the cold measurement (each replay streams the packed caches of n
    copies, > 2x L2). 'sf' (select + merge only) streams little and is warm in a large L2: reported, not gated.
    one96_aa is an A/A control (same configuration as one96, different data): its relative difference to one96 must be
    near zero, else the timing is not trustworthy at this shape."""
    BK = B * KVH
    n = K4._copies(2 * BK * L * (P + 4))
    fns, keep = {}, []
    for vi, (name, v) in enumerate(VARIANTS.items()):
        kw = dict(v)
        mode, K = kw.pop("mode"), kw.pop("K")
        host = (torch.randn(BK, L, D).half().pin_memory(), torch.randn(BK, L, D).half().pin_memory())
        steps = [Step5(B, L, K, 2, mode=mode, seed=1000 * vi + i, host=host, **kw) for i in range(n)]
        for st in steps:
            st.step()
        torch.cuda.synchronize()
        keep += steps + [host]
        fns[name + "_step"] = K4.graph_of(lambda ss=steps: [x.step() for x in ss])
        fns[name + "_sf"] = K4.graph_of(lambda ss=steps: [x.sf() for x in ss])
        if name == "one96":
            fns["one96_base"] = K4.graph_of(lambda ss=steps: [x.base() for x in ss])
    fns["empty_graph"] = K4.graph_of(lambda: None)
    t = K4.time_group(fns, rounds=rounds)
    import hashlib
    res = {"B": B, "L": L, "copies": n, "rc": 2, "l2_bytes": K4.L2_BYTES,
           "md5": {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in ("kbench5.py", "kbench4.py")}}
    raw = {}
    for nm, v in t.items():
        div = 1 if nm == "empty_graph" else n
        res[nm + "_us"] = round(v["ms"] * 1000 / div, 2)
        raw[nm] = [x * 1000 / div for x in v["rounds"]]
    cl = [c for c in t["empty_graph"]["clocks"] if c]
    res["clock_mhz_min_max"] = [min(cl), max(cl)] if cl else None
    res["raw_rounds_us"] = {k: [round(x, 3) for x in v] for k, v in raw.items()}   # every per-round value, for re-analysis
    rnd = random.Random(0)
    for other in ("one96_aa", "two128k64_cc32", "two144k48_cc16", "two224k96_cc32", "one128", "one144", "one160"):
        for part in ("step", "sf"):
            rel = [b / a - 1 for a, b in zip(raw[f"one96_{part}"], raw[f"{other}_{part}"])]
            ext = [b - a for a, b in zip(raw[f"one96_{part}"], raw[f"{other}_{part}"])]
            a_, b_ = raw[f"one96_{part}"], raw[f"{other}_{part}"]
            dmed = S.median(b_) - S.median(a_)
            res[f"{other}_vs_one96_{part}"] = dict(extra_us=round(S.median(ext), 2), extra_ci=_boot(ext, rnd),
                                                   rel=round(S.median(rel), 4), rel_ci=_boot(rel, rnd),
                                                   # the paired median and the difference of medians must agree; a large
                                                   # gap means unstable rounds and the paired figure alone is not reported
                                                   diff_of_medians_us=round(dmed, 2),
                                                   consistent=abs(dmed - S.median(ext)) <= max(2.0, 0.25 * abs(S.median(ext))),
                                                   spread_one96=round(max(a_) / min(a_), 3), spread_other=round(max(b_) / min(b_), 3))
        # extra of the cold whole step relative to the part one96 spends outside the base kernel (select + merge)
        sfc = [a - b for a, b in zip(raw["one96_step"], raw["one96_base"])]
        relsf = [(b - a) / max(c, 1e-6) for a, b, c in zip(raw["one96_step"], raw[f"{other}_step"], sfc)]
        res[f"{other}_vs_one96_selmerge_cold"] = dict(one96_selmerge_us=round(S.median(sfc), 2), rel=round(S.median(relsf), 4),
                                                       rel_ci=_boot(relsf, rnd))
    for a, b in (("one96", "one96_rx"), ("two128k64_cc32", "two128k64_rx"), ("two224k96_cc32", "two224k96_rx")):
        d = [y - x for x, y in zip(raw[f"{a}_step"], raw[f"{b}_step"])]
        res[f"{b}_minus_{a}_step"] = dict(us=round(S.median(d), 2), ci=_boot(d, rnd))
    aa = res["one96_aa_vs_one96_step"]["rel_ci"]
    res["aa_ok"] = abs(aa[0]) <= 0.03 and abs(aa[1]) <= 0.03
    for two in ("two128k64_cc32", "two144k48_cc16", "two224k96_cc32"):
        r_ = res[f"{two}_vs_one96_step"]
        res[f"gate_{two}"] = ("UNSTABLE" if not (r_["consistent"] and res["aa_ok"]) else
                              "PASS" if r_["rel_ci"][1] <= GATE else "FAIL")
        # latency-matched view: is the two-stage step cheaper than single-stage with all candidates' keys and values?
        res[f"{two}_minus_one128_step_us"] = round(S.median([b - a for a, b in zip(raw["one128_step"], raw[f"{two}_step"])]), 2)
    keep = fns = None
    torch.cuda.empty_cache()
    return res


RC_SET = ("one96", "one128", "one160", "two128k64_cc32", "two224k96_cc32")


def run_rc(B, L, rounds):
    """one shape, the policy's five configurations in three base variants, interleaved in one timing group: rotary
    centring with the fp32 bias pre-pass (rc = 2, as run()), without centring (rc = 0, name suffix _rc0), and with the
    bias stored in fp16 (suffix _b16); plus the one96 A/A control. Paired per-round differences rc2 - rc0 (the cost of
    centring) and b16 - rc2 (what the fp16 bias saves) per configuration."""
    BK = B * KVH
    n = K4._copies(2 * BK * L * (P + 4))
    items = [(nm, VARIANTS[nm], 2, False) for nm in RC_SET] + [("one96_aa", VARIANTS["one96_aa"], 2, False)]
    items += [(nm + "_rc0", VARIANTS[nm], 0, False) for nm in RC_SET] + [(nm + "_b16", VARIANTS[nm], 2, True) for nm in RC_SET]
    fns, keep = {}, []
    for vi, (name, v, rc, b16) in enumerate(items):
        kw = dict(v)
        mode, K = kw.pop("mode"), kw.pop("K")
        host = (torch.randn(BK, L, D).half().pin_memory(), torch.randn(BK, L, D).half().pin_memory())
        steps = [Step5(B, L, K, rc, mode=mode, seed=1000 * vi + i, host=host, bias16=b16, **kw) for i in range(n)]
        for st in steps:
            st.step()
        torch.cuda.synchronize()
        keep += steps + [host]
        fns[name + "_step"] = K4.graph_of(lambda ss=steps: [x.step() for x in ss])
    fns["empty_graph"] = K4.graph_of(lambda: None)
    t = K4.time_group(fns, rounds=rounds)
    import hashlib
    res = {"B": B, "L": L, "copies": n, "rc": "0,2,2+bias16", "l2_bytes": K4.L2_BYTES,
           "md5": {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in ("kbench5.py", "kbench4.py")}}
    raw = {}
    for nm, v in t.items():
        div = 1 if nm == "empty_graph" else n
        res[nm + "_us"] = round(v["ms"] * 1000 / div, 2)
        raw[nm] = [x * 1000 / div for x in v["rounds"]]
    res["raw_rounds_us"] = {k: [round(x, 3) for x in v] for k, v in raw.items()}
    rnd = random.Random(0)

    def pdiff(a, b):
        d = [y - x for x, y in zip(raw[a], raw[b])]
        dm = S.median(raw[b]) - S.median(raw[a])
        return dict(us=round(S.median(d), 2), ci=_boot(d, rnd), rel=round(S.median([y / x - 1 for x, y in zip(raw[a], raw[b])]), 4),
                    diff_of_medians_us=round(dm, 2), consistent=abs(dm - S.median(d)) <= max(2.0, 0.25 * abs(S.median(d))))
    for nm in RC_SET:
        res[f"{nm}_centring_cost"] = pdiff(f"{nm}_rc0_step", f"{nm}_step")
        res[f"{nm}_b16_minus_rc2"] = pdiff(f"{nm}_step", f"{nm}_b16_step")
    aa = pdiff("one96_step", "one96_aa_step")
    res["one96_aa_vs_one96"] = aa
    res["aa_ok"] = abs(aa["rel"]) <= 0.03 and aa["consistent"]
    keep = fns = None
    torch.cuda.empty_cache()
    return res


PHASES = {"one96": (dict(mode="one", K=96), ("base", "select", "fixA", "fixB")),
          "two128k64": (dict(mode="two", K=64, M=128, cc=32), ("base", "select", "fix2A", "fix2B", "fixB"))}


def phases(B, L, rounds):
    """per-phase cost by cumulative prefixes: graphs of base, base + select, base + select + fix..., each starting from
    the cold base kernel on n layer copies (as the full step), so a phase's cost is the paired difference of two
    consecutive prefixes (no phase runs warm on its own). Also the unfused second stage against the fused one, and the
    base / fixA hi-lo split on and off (own host buffers, so no shared host-cache warmth), with an A/A control."""
    BK = B * KVH
    n = K4._copies(2 * BK * L * (P + 4))
    fns, keep, same = {}, [], None
    groups = {}
    for vi, (name, (v, parts)) in enumerate(PHASES.items()):
        kw = dict(v)
        mode, K = kw.pop("mode"), kw.pop("K")
        for tag, hilo in (("", False), ("_hilo", True)):
            host = (torch.randn(BK, L, D).half().pin_memory(), torch.randn(BK, L, D).half().pin_memory())
            steps = [Step5(B, L, K, 2, mode=mode, seed=1000 * vi + i, host=host, hilo=hilo, **kw) for i in range(n)]
            for st in steps:
                st.step()
            torch.cuda.synchronize()
            keep += steps + [host]
            pre = []
            for k_ in range(1, len(parts) + 1):
                ps = parts[:k_]
                nm = f"{name}{tag}_pre{k_}"
                fns[nm] = K4.graph_of(lambda ss=steps, ps=ps: [[x.phase(p_) for p_ in ps] for x in ss])
                pre.append(nm)
            groups[name + tag] = (parts, pre)
            fns[f"{name}{tag}_step"] = K4.graph_of(lambda ss=steps: [x.step() for x in ss])
            if mode == "two" and not hilo:
                out_f = [x.out.clone() for x in steps]
                for x in steps:
                    x.fix_unfused()
                torch.cuda.synchronize()
                same = all(torch.equal(a_, x.out) for a_, x in zip(out_f, steps))
                fns[f"{name}_step_unfused"] = K4.graph_of(lambda ss=steps: [(x.base(), x.select(), x.fix_unfused()) for x in ss])
            if mode == "one" and not hilo:   # A/A: the same step graph timed twice under two names
                fns[f"{name}_step_aa"] = K4.graph_of(lambda ss=steps: [x.step() for x in ss])
    fns["empty_graph"] = K4.graph_of(lambda: None)
    t = K4.time_group(fns, rounds=rounds)
    res = {"B": B, "L": L, "copies": n, "unfused_bitwise_same_output": same}
    raw = {}
    for nm, v in t.items():
        div = 1 if nm == "empty_graph" else n
        res[nm + "_us"] = round(v["ms"] * 1000 / div, 2)
        raw[nm] = [x * 1000 / div for x in v["rounds"]]
    res["raw_rounds_us"] = {k: [round(x, 3) for x in v] for k, v in raw.items()}
    rnd = random.Random(0)

    def diff(a, b):
        d = [y - x for x, y in zip(raw[a], raw[b])]
        return dict(us=round(S.median(d), 2), ci=_boot(d, rnd), diff_of_medians=round(S.median(raw[b]) - S.median(raw[a]), 2))
    for g, (parts, pre) in groups.items():   # phase k = prefix k - prefix k-1 (phase 1 = prefix 1)
        res[f"{g}_phase_{parts[0]}"] = dict(us=round(S.median(raw[pre[0]]), 2))
        for k_ in range(1, len(parts)):
            res[f"{g}_phase_{parts[k_]}"] = diff(pre[k_ - 1], pre[k_])
    for a, b in (("one96_step", "one96_step_aa"), ("one96_step", "two128k64_step"), ("two128k64_step", "two128k64_step_unfused"),
                 ("two128k64_step", "two128k64_hilo_step"), ("one96_step", "one96_hilo_step"),
                 ("one96_hilo_step", "two128k64_hilo_step")):
        res[f"{b}_minus_{a}"] = diff(a, b)
    aa = res["one96_step_aa_minus_one96_step"]
    res["aa_ok"] = abs(aa["us"]) <= max(0.5, 0.01 * res["one96_step_us"])
    keep = fns = None
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--rcmode", action="store_true", help="run_rc: centring on / off / fp16 bias for the policy configurations")
    ap.add_argument("--shapes", default="1x8192,1x32768,8x8192,8x32768,32x8192")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--out", default="kbench5.json")
    a = ap.parse_args()
    if a.check:
        check()
        return
    K4.L2_BYTES = torch.cuda.get_device_properties(0).L2_cache_size   # cold-cache copies sized for this device
    out = []
    for sh in a.shapes.split(","):
        B, L = (int(x) for x in sh.split("x"))
        t0 = time.time()
        r = phases(B, L, a.rounds) if a.phases else run_rc(B, L, a.rounds) if a.rcmode else run(B, L, a.rounds)
        out.append(r)
        print(json.dumps(r), f"({time.time() - t0:.0f}s)", flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
