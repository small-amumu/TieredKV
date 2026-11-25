# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""kbench3: third-iteration layered-KV decode step (rootcause fork, exploratory). Same shapes and timing as kbench.py; three
kernels plus one small torch.topk, optionally captured in a CUDA graph:
  base   split-L decode kernel on a planar-packed 2-bit KV cache: byte j of a token holds the codes of channels
         j, j+32, j+64, j+96 (2 bits each), so one 32-byte load gives four 32-channel planes; scores use the integer codes
         directly, s = scale * (q . codes) + zero * sum(q), and values likewise (p * vscale) @ codes + (p . vzero);
         optional rotary-aware centring add-back q . R(t) m via the angle-addition identity (D/2 cos / sin per tile and
         a constant per-tile cos / sin table, two small dots), not per element
  sel    stage 1: per tile of T tokens, group-mean softmax probability over the g query heads of a KV head (with the
         per-head global max and sum from the reduction) and the tile's top-K by tl.sort on packed (prob bits, index)
         keys; stage 2: torch.topk over the (L / T) * K candidates. Exact: the global top-K lies in the union of tile top-Ks
  fix    one program per (batch, KV head): reduces the split partials, gathers the K selected exact rows straight from
         pinned host memory (zero copy) and the K base rows from the packed cache, and replaces the base contribution by
         the exact one in fp32 (max-rescaled numerator and denominator); writes the attention output
Correctness is checked against a torch reference (python kbench3.py --check).
v3 changes: the rotary add-back is one dot of [A | B] against a stacked [cos; sin] table at a selectable tensor-core
precision (tf32 / tf32x3 / ieee), optionally reloading the table per tile instead of pinning it in registers; codes
can be widened to fp16 by the exponent trick (0x6400 | c) - 1024 instead of an int-to-float conversion; the per-tile
top-K is a bisection on the probability bits plus a prefix-sum compaction instead of a full bitonic sort; the number
of splits grows at small batch so every SM gets work; --tune sweeps block size, warps and stages per shape.
Usage: python kbench3.py [--B ..] [--L ..] [--K 64] [--reps 20] [--graph] [--tune] [--out kbench3.json]"""
import argparse
import json
import math
import time

import torch
import triton
import triton.language as tl

H, KVH, D = 32, 8, 128
G = H // KVH
GP = 16
P = D // 4  # plane width


@triton.jit
def _widen(c, MAGIC: tl.constexpr):
    """2-bit codes (uint8 0..3) to fp16: an int-to-float conversion, or the exponent trick (0x6400 | c) - 1024, exact."""
    if MAGIC:
        return (c.to(tl.int16) | 0x6400).to(tl.float16, bitcast=True) - 1024.0
    else:
        return c.to(tl.float16)


@triton.jit
def base_kernel(q_ptr, qs_ptr, kp_ptr, ks_ptr, kz_ptr, vp_ptr, vs_ptr, vz_ptr, u_ptr, w_ptr, invf2_ptr, cs_ptr,
                m_out, l_out, acc_out, s_out, L, split_len, sm_scale,
                GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, RC: tl.constexpr,
                RCP: tl.constexpr, RELOAD: tl.constexpr, MAGIC: tl.constexpr):
    bk = tl.program_id(0)
    sp = tl.program_id(1)
    nsplit = tl.num_programs(1)
    og = tl.arange(0, GP)
    op = tl.arange(0, P)
    gm = og < GH
    qrow = (bk * GH + og[:, None]) * (4 * P)
    q0 = tl.load(q_ptr + qrow + op[None, :], mask=gm[:, None], other=0.0)
    q1 = tl.load(q_ptr + qrow + P + op[None, :], mask=gm[:, None], other=0.0)
    q2 = tl.load(q_ptr + qrow + 2 * P + op[None, :], mask=gm[:, None], other=0.0)
    q3 = tl.load(q_ptr + qrow + 3 * P + op[None, :], mask=gm[:, None], other=0.0)
    qsum = tl.load(qs_ptr + bk * GH + og, mask=gm, other=0.0)
    m_i = tl.full([GP], -float("inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    a0 = tl.zeros([GP, P], tl.float32)
    a1 = tl.zeros([GP, P], tl.float32)
    a2 = tl.zeros([GP, P], tl.float32)
    a3 = tl.zeros([GP, P], tl.float32)
    az = tl.zeros([GP], tl.float32)
    start = sp * split_len
    end = tl.minimum(start + split_len, L)
    on = tl.arange(0, BLOCK_N)
    if RC:
        od = tl.arange(0, 4 * P)                       # [A | B] over the D/2 frequency pairs, stacked
        u = tl.load(u_ptr + (bk * GH + og[:, None]) * (4 * P) + od[None, :], mask=gm[:, None], other=0.0)
        w = tl.load(w_ptr + (bk * GH + og[:, None]) * (4 * P) + od[None, :], mask=gm[:, None], other=0.0)
        invf2 = tl.load(invf2_ptr + od)
        if not RELOAD:
            cs = tl.load(cs_ptr + od[:, None] * BLOCK_N + on[None, :])   # [cos(n f); sin(n f)], [D, BLOCK_N]
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + on
        nm = offs_n < end
        base = bk * L + offs_n
        kb = tl.load(kp_ptr + base[:, None] * P + op[None, :], mask=nm[:, None], other=0)     # [BLOCK_N, P] uint8
        raw = (tl.dot(q0, tl.trans(_widen(kb & 3, MAGIC))) + tl.dot(q1, tl.trans(_widen((kb >> 2) & 3, MAGIC)))
               + tl.dot(q2, tl.trans(_widen((kb >> 4) & 3, MAGIC))) + tl.dot(q3, tl.trans(_widen((kb >> 6) & 3, MAGIC))))
        ksc = tl.load(ks_ptr + base, mask=nm, other=0.0).to(tl.float32)
        kzp = tl.load(kz_ptr + base, mask=nm, other=0.0).to(tl.float32)
        s = (raw * ksc[None, :] + qsum[:, None] * kzp[None, :]) * sm_scale
        if RC:
            t0 = n0.to(tl.float32) * invf2
            ab = u * tl.cos(t0)[None, :] + w * tl.sin(t0)[None, :]
            if RELOAD:
                cs = tl.load(cs_ptr + od[:, None] * BLOCK_N + on[None, :])
            s += tl.dot(ab, cs, input_precision=RCP) * sm_scale
        s = tl.where(nm[None, :], s, -float("inf"))
        tl.store(s_out + (bk * GH + og[:, None]) * L + offs_n[None, :], s.to(tl.float16), mask=gm[:, None] & nm[None, :])
        mn = tl.maximum(m_i, tl.max(s, 1))
        al = tl.exp(m_i - mn)
        p = tl.exp(s - mn[:, None])
        l_i = l_i * al + tl.sum(p, 1)
        vb = tl.load(vp_ptr + base[:, None] * P + op[None, :], mask=nm[:, None], other=0)
        vsc = tl.load(vs_ptr + base, mask=nm, other=0.0).to(tl.float32)
        vzp = tl.load(vz_ptr + base, mask=nm, other=0.0).to(tl.float32)
        pv = (p * vsc[None, :] * 4096.0).to(tl.float16)   # keep small weights out of the fp16 subnormal range
        a0 = a0 * al[:, None] + tl.dot(pv, _widen(vb & 3, MAGIC))
        a1 = a1 * al[:, None] + tl.dot(pv, _widen((vb >> 2) & 3, MAGIC))
        a2 = a2 * al[:, None] + tl.dot(pv, _widen((vb >> 4) & 3, MAGIC))
        a3 = a3 * al[:, None] + tl.dot(pv, _widen((vb >> 6) & 3, MAGIC))
        az = az * al + tl.sum(p * vzp[None, :], 1)
        m_i = mn
    o = (bk * nsplit + sp) * GP + og
    tl.store(m_out + o, m_i)
    tl.store(l_out + o, l_i)
    ao = acc_out + o[:, None] * (4 * P) + op[None, :]
    tl.store(ao, a0 / 4096.0 + az[:, None])
    tl.store(ao + P, a1 / 4096.0 + az[:, None])
    tl.store(ao + 2 * P, a2 / 4096.0 + az[:, None])
    tl.store(ao + 3 * P, a3 / 4096.0 + az[:, None])


@triton.jit
def reduce_stats_kernel(m_in, l_in, M_out, Z_out, nsplit, GH: tl.constexpr, GP: tl.constexpr, NS: tl.constexpr):
    """per (batch * KV head): global max M and sum Z over splits, for the selection criterion."""
    bk = tl.program_id(0)
    os_ = tl.arange(0, NS)
    og = tl.arange(0, GP)
    msk = os_[:, None] < nsplit
    m = tl.load(m_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=msk, other=-float("inf"))
    l = tl.load(l_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=msk, other=0.0)
    M = tl.max(m, 0)
    Z = tl.sum(l * tl.exp(m - M[None, :]), 0)
    tl.store(M_out + bk * GP + og, M)
    tl.store(Z_out + bk * GP + og, Z)


@triton.jit
def sel_stage1(s_in, M_in, Z_in, cand_out, L, ntile, GH: tl.constexpr, GP: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
               BISECT: tl.constexpr):
    bk = tl.program_id(0)
    tt = tl.program_id(1)
    on = tt * T + tl.arange(0, T)
    nm = on < L
    crit = tl.zeros([T], tl.float32)
    for g in tl.static_range(GH):
        M = tl.load(M_in + bk * GP + g)
        Z = tl.load(Z_in + bk * GP + g)
        s = tl.load(s_in + (bk * GH + g) * L + on, mask=nm, other=-float("inf")).to(tl.float32)
        crit += tl.exp(s - M) / Z
    crit = crit / GH
    bits = crit.to(tl.int32, bitcast=True).to(tl.int64)          # non-negative fp32: bit order = value order
    key = (bits << 32) | on.to(tl.int64)
    key = tl.where(nm, key, 0)
    if BISECT:
        # K-th largest probability bits by bisection (31 counting passes), then prefix-sum compaction: every key above
        # the threshold plus the first ties up to K; same set as the sort up to the order of ties
        b32 = tl.where(nm, crit.to(tl.int32, bitcast=True), 0)
        lo = 0
        hi = 0x7F800000
        for _ in range(31):
            mid = lo + (hi - lo + 1) // 2
            ok_ = tl.sum((b32 >= mid).to(tl.int32), 0) >= K
            lo = tl.where(ok_, mid, lo)
            hi = tl.where(ok_, hi, mid - 1)
        gt = b32 > lo
        tie = b32 == lo
        c1 = tl.sum(gt.to(tl.int32), 0)
        pg = tl.cumsum(gt.to(tl.int32), 0) - 1
        pt = c1 + tl.cumsum(tie.to(tl.int32), 0) - 1
        pos = tl.where(gt, pg, tl.where(tie & (pt < K), pt, -1))
        tl.store(cand_out + (bk * ntile + tt) * K + pos, key, mask=pos >= 0)
    else:
        key = tl.sort(key, descending=True)
        ok = tl.arange(0, T)
        tl.store(cand_out + (bk * ntile + tt) * K + ok, key, mask=ok < K)


@triton.jit
def fix_kernel(q_ptr, qs_ptr, m_in, l_in, acc_in, idx_ptr, kex_ptr, vex_ptr, kp_ptr, ks_ptr, kz_ptr, vp_ptr, vs_ptr, vz_ptr,
               u_ptr, invf2_ptr, cs_ptr,
               out_ptr, L, nsplit, sm_scale, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr, K: tl.constexpr, NS: tl.constexpr,
               RC: tl.constexpr, BLOCK_N: tl.constexpr, RCP: tl.constexpr):
    """one program per (batch, KV head): reduce the split partials, then replace the base contribution of the K selected
    keys by their exact contribution (exact rows read from pinned host memory), plane by plane (channel = plane * P + j)."""
    bk = tl.program_id(0)
    og = tl.arange(0, GP)
    op = tl.arange(0, P)
    gm = og < GH
    D4: tl.constexpr = 4 * P
    os_ = tl.arange(0, NS)
    sm_ = os_ < nsplit
    m = tl.load(m_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_[:, None], other=-float("inf"))
    l = tl.load(l_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_[:, None], other=0.0)
    M = tl.max(m, 0)
    Z = tl.sum(l * tl.exp(m - M[None, :]), 0)
    A0 = tl.zeros([GP, P], tl.float32)
    A1 = tl.zeros([GP, P], tl.float32)
    A2 = tl.zeros([GP, P], tl.float32)
    A3 = tl.zeros([GP, P], tl.float32)
    for s_ in range(0, nsplit):
        w = tl.exp(tl.load(m_in + (bk * nsplit + s_) * GP + og) - M)[:, None]
        ab = acc_in + ((bk * nsplit + s_) * GP + og[:, None]) * D4 + op[None, :]
        A0 += tl.load(ab) * w
        A1 += tl.load(ab + P) * w
        A2 += tl.load(ab + 2 * P) * w
        A3 += tl.load(ab + 3 * P) * w
    ok = tl.arange(0, K)
    tok = tl.load(idx_ptr + bk * K + ok)
    row = bk * L + tok
    qb = q_ptr + (bk * GH + og[:, None]) * D4 + op[None, :]
    q0 = tl.load(qb, mask=gm[:, None], other=0.0)
    q1 = tl.load(qb + P, mask=gm[:, None], other=0.0)
    q2 = tl.load(qb + 2 * P, mask=gm[:, None], other=0.0)
    q3 = tl.load(qb + 3 * P, mask=gm[:, None], other=0.0)
    kxb = kex_ptr + row[:, None] * D4 + op[None, :]                     # exact rows, zero copy from host
    se = (tl.dot(q0, tl.trans(tl.load(kxb))) + tl.dot(q1, tl.trans(tl.load(kxb + P))) + tl.dot(q2, tl.trans(tl.load(kxb + 2 * P)))
          + tl.dot(q3, tl.trans(tl.load(kxb + 3 * P)))) * sm_scale
    kb = tl.load(kp_ptr + row[:, None] * P + op[None, :])
    raw = (tl.dot(q0, tl.trans((kb & 3).to(tl.float16))) + tl.dot(q1, tl.trans(((kb >> 2) & 3).to(tl.float16)))
           + tl.dot(q2, tl.trans(((kb >> 4) & 3).to(tl.float16))) + tl.dot(q3, tl.trans(((kb >> 6) & 3).to(tl.float16))))
    qsum = tl.load(qs_ptr + bk * GH + og, mask=gm, other=0.0)
    sb = (raw * tl.load(ks_ptr + row).to(tl.float32)[None, :] + qsum[:, None] * tl.load(kz_ptr + row).to(tl.float32)[None, :]) * sm_scale
    if RC:
        # same angle decomposition as the base kernel: tile start t0 = tok // BLOCK_N * BLOCK_N, offset n = tok % BLOCK_N
        oh = tl.arange(0, 2 * P)
        invf = tl.load(invf2_ptr + oh)
        t0 = (tok // BLOCK_N * BLOCK_N).to(tl.float32)
        nn = tok % BLOCK_N
        a0 = t0[:, None] * invf[None, :]                                      # [K, D/2]
        cn = tl.load(cs_ptr + oh[None, :] * BLOCK_N + nn[:, None])
        sn = tl.load(cs_ptr + (2 * P + oh[None, :]) * BLOCK_N + nn[:, None])
        cosv = tl.cos(a0) * cn - tl.sin(a0) * sn
        sinv = tl.sin(a0) * cn + tl.cos(a0) * sn
        ca = tl.load(u_ptr + (bk * GH + og[:, None]) * D4 + oh[None, :], mask=gm[:, None], other=0.0)
        cb = tl.load(u_ptr + (bk * GH + og[:, None]) * D4 + 2 * P + oh[None, :], mask=gm[:, None], other=0.0)
        rb = tl.dot(ca, tl.trans(cosv), input_precision=RCP) + tl.dot(cb, tl.trans(sinv), input_precision=RCP)
        sb += rb * sm_scale
    M2 = tl.maximum(M, tl.max(se, 1))  # exact keys already contain the offset: no rotary term on se
    r = tl.exp(M - M2)
    eb = tl.exp(sb - M2[:, None])
    ee = tl.exp(se - M2[:, None])
    Z2 = Z * r - tl.sum(eb, 1) + tl.sum(ee, 1)
    vb = tl.load(vp_ptr + row[:, None] * P + op[None, :])
    vsc = tl.load(vs_ptr + row).to(tl.float32)
    vzp = tl.load(vz_ptr + row).to(tl.float32)
    ebs = (eb * vsc[None, :] * 4096.0).to(tl.float16)
    ebz = tl.sum(eb * vzp[None, :], 1)[:, None]
    ee16 = (ee * 4096.0).to(tl.float16)
    vxb = vex_ptr + row[:, None] * D4 + op[None, :]
    ob = out_ptr + (bk * GH + og[:, None]) * D4 + op[None, :]
    inv = 1.0 / Z2[:, None]
    o0 = (A0 * r[:, None] - (tl.dot(ebs, (vb & 3).to(tl.float16)) - tl.dot(ee16, tl.load(vxb))) / 4096.0 - ebz) * inv
    o1 = (A1 * r[:, None] - (tl.dot(ebs, ((vb >> 2) & 3).to(tl.float16)) - tl.dot(ee16, tl.load(vxb + P))) / 4096.0 - ebz) * inv
    o2 = (A2 * r[:, None] - (tl.dot(ebs, ((vb >> 4) & 3).to(tl.float16)) - tl.dot(ee16, tl.load(vxb + 2 * P))) / 4096.0 - ebz) * inv
    o3 = (A3 * r[:, None] - (tl.dot(ebs, ((vb >> 6) & 3).to(tl.float16)) - tl.dot(ee16, tl.load(vxb + 3 * P))) / 4096.0 - ebz) * inv
    tl.store(ob, o0, mask=gm[:, None])
    tl.store(ob + P, o1, mask=gm[:, None])
    tl.store(ob + 2 * P, o2, mask=gm[:, None])
    tl.store(ob + 3 * P, o3, mask=gm[:, None])


class _CAI:
    TYPES = {torch.float16: "<f2", torch.int8: "|i1"}

    def __init__(self, t):
        self.t = t
        self.__cuda_array_interface__ = {"shape": tuple(t.shape), "typestr": self.TYPES[t.dtype], "data": (t.data_ptr(), False),
                                         "version": 3, "strides": None}


def host_as_cuda(t):
    return torch.as_tensor(_CAI(t), device="cuda")


class Step:
    """one layer's decode step with fixed buffers (graph-capturable)."""

    def __init__(self, B, L, K, rc, dev="cuda", BLOCK_N=64, T=1024, seed=0, warps=4, stages=2, rcp="tf32", reload=False,
                 magic=True, bisect=True, nprog=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.B, self.L, self.K, self.rc, self.BLOCK_N, self.T = B, L, K, rc, BLOCK_N, T
        self.warps, self.stages, self.rcp, self.reload, self.magic, self.bisect = warps, stages, rcp, reload, magic, bisect
        BK = self.BK = B * KVH
        self.q = torch.randn(BK * G, D, generator=g).half().to(dev)
        self.qs = self.q.float().sum(-1).contiguous()
        self.kp = torch.randint(0, 256, (BK, L, P), generator=g, dtype=torch.uint8).to(dev)
        self.vp = torch.randint(0, 256, (BK, L, P), generator=g, dtype=torch.uint8).to(dev)
        self.ks = (torch.rand(BK, L, generator=g) * 0.1).half().to(dev)
        self.kz = (-torch.rand(BK, L, generator=g) * 0.15).half().to(dev)
        self.vs, self.vz = self.ks.clone(), self.kz.clone()
        self.kh = torch.randn(BK, L, D, generator=g).half().pin_memory()
        self.vh = torch.randn(BK, L, D, generator=g).half().pin_memory()
        self.kz_, self.vz_ = host_as_cuda(self.kh), host_as_cuda(self.vh)
        invf = 1.0 / (500000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        self.invf = invf.to(dev)
        on = torch.arange(BLOCK_N, dtype=torch.float32)
        self.ctab = torch.cos(invf[:, None] * on[None]).contiguous().to(dev)
        self.stab = torch.sin(invf[:, None] * on[None]).contiguous().to(dev)
        self.invf2 = torch.cat([invf, invf]).to(dev)
        self.cs = torch.cat([self.ctab, self.stab], 0).contiguous()
        mpre = torch.randn(KVH, D, generator=g).to(dev)
        qf = self.q.float().view(B, KVH, G, D)
        ma, mb = mpre[:, :D // 2], mpre[:, D // 2:]
        qa, qb = qf[..., :D // 2], qf[..., D // 2:]
        self.ca = (qa * ma[None, :, None] + qb * mb[None, :, None]).reshape(BK * G, D // 2).contiguous()
        self.cb = (qb * ma[None, :, None] - qa * mb[None, :, None]).reshape(BK * G, D // 2).contiguous()
        self.u = torch.cat([self.ca, self.cb], 1).contiguous()
        self.w = torch.cat([self.cb, -self.ca], 1).contiguous()
        self.mpre = mpre
        # enough splits that every SM gets several programs at small batch (nprog: target number of programs)
        nprog = nprog or 4 * torch.cuda.get_device_properties(dev).multi_processor_count
        ns = max(1, min(256, max((L + 2047) // 2048, -(-nprog // BK)), L // (2 * BLOCK_N)))
        self.split_len = triton.cdiv(triton.cdiv(L, ns), BLOCK_N) * BLOCK_N
        self.nsplit = triton.cdiv(L, self.split_len)
        self.NS = triton.next_power_of_2(self.nsplit)
        self.m = torch.empty(BK * self.nsplit * GP, device=dev)
        self.l = torch.empty_like(self.m)
        self.acc = torch.empty(BK * self.nsplit * GP, D, device=dev)
        self.s = torch.empty(BK * G, L, device=dev, dtype=torch.float16)
        self.M = torch.empty(BK * GP, device=dev)
        self.Z = torch.empty_like(self.M)
        self.ntile = triton.cdiv(L, T)
        self.cand = torch.empty(BK, self.ntile * K, device=dev, dtype=torch.int64)
        self.idx = torch.empty(BK, K, device=dev, dtype=torch.int64)
        self.out = torch.empty(BK * G, D, device=dev)
        self.scale = 1 / math.sqrt(D)

    def base(self):
        base_kernel[(self.BK, self.nsplit)](self.q, self.qs, self.kp, self.ks, self.kz, self.vp, self.vs, self.vz, self.u, self.w,
                                            self.invf2, self.cs, self.m, self.l, self.acc, self.s, self.L, self.split_len,
                                            self.scale, GH=G, GP=GP, P=P, BLOCK_N=self.BLOCK_N, RC=self.rc, RCP=self.rcp,
                                            RELOAD=self.reload, MAGIC=self.magic, num_warps=self.warps, num_stages=self.stages)

    def select(self):
        reduce_stats_kernel[(self.BK,)](self.m, self.l, self.M, self.Z, self.nsplit, GH=G, GP=GP, NS=self.NS)
        sel_stage1[(self.BK, self.ntile)](self.s, self.M, self.Z, self.cand, self.L, self.ntile, GH=G, GP=GP, T=self.T, K=self.K,
                                          BISECT=self.bisect, num_warps=4)
        top = (self.cand >> 32).topk(self.K, -1).indices
        self.idx.copy_(torch.gather(self.cand, 1, top) & 0xFFFFFFFF)

    def fix(self):
        fix_kernel[(self.BK,)](self.q, self.qs, self.m, self.l, self.acc, self.idx, self.kz_, self.vz_, self.kp, self.ks, self.kz,
                               self.vp, self.vs, self.vz, self.u, self.invf2, self.cs, self.out, self.L,
                               self.nsplit, self.scale, GH=G, GP=GP, P=P, K=self.K, NS=self.NS, RC=self.rc, BLOCK_N=self.BLOCK_N,
                               RCP=self.rcp, num_warps=4)

    def step(self):
        self.base()
        self.select()
        self.fix()


def reference(st, idx=None):
    """torch reference of the whole step on dequantised planar codes (fp32); idx overrides the selection."""
    BK, L, K = st.BK, st.L, st.K

    def deq(pk, sc, zp):
        planes = [(pk >> (2 * i)) & 3 for i in range(4)]
        c = torch.cat(planes, -1).float()                                # channel = plane * P + j
        return c * sc.float()[..., None] + zp.float()[..., None]
    kd, vd = deq(st.kp, st.ks, st.kz), deq(st.vp, st.vs, st.vz)
    q = st.q.float().view(BK, G, D)
    s = torch.einsum("bgd,bld->bgl", q, kd) * st.scale
    if st.rc:
        t = torch.arange(L, device=q.device, dtype=torch.float32)[:, None] * st.invf[None]
        cos, sin = t.cos(), t.sin()
        ca, cb = st.ca.view(BK, G, D // 2), st.cb.view(BK, G, D // 2)
        s = s + (torch.einsum("bgi,li->bgl", ca, cos) + torch.einsum("bgi,li->bgl", cb, sin)) * st.scale
    p = torch.softmax(s, -1)
    crit = p.mean(1)
    if idx is None:
        idx = crit.topk(K, -1).indices
    kx, vx = st.kh.to(q.device).float(), st.vh.to(q.device).float()
    se = s.clone()
    ie = idx[:, None, :].expand(BK, G, K)
    sx = torch.einsum("bgd,bkd->bgk", q, torch.gather(kx, 1, idx[..., None].expand(BK, K, D))) * st.scale
    se.scatter_(2, ie, sx)
    pf = torch.softmax(se, -1)
    vfin = vd.clone()
    vfin.scatter_(1, idx[..., None].expand(BK, K, D), torch.gather(vx, 1, idx[..., None].expand(BK, K, D)))
    o = torch.einsum("bgl,bld->bgd", pf, vfin)
    return o, idx


def check(**cfg):
    for rc in (False, True):
        for L in (5000, 70000):
            st = Step(2, L, 64, rc, **cfg)
            st.step()
            torch.cuda.synchronize()
            o_ref, idx_ref = reference(st)
            got = st.out.view(st.BK, G, D)
            same = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(st.idx.cpu(), idx_ref.cpu())) / idx_ref.numel()
            o_same, _ = reference(st, st.idx)
            print(f"check {cfg} rc={rc} L={L} nsplit={st.nsplit}: selected-set overlap with reference {same:.4f}; output rel err "
                  f"{float((got - o_ref).norm() / o_ref.norm()):.3g}; with the kernel's own selection: rel err "
                  f"{float((got - o_same).norm() / o_same.norm()):.3g}", flush=True)
            del st
            torch.cuda.empty_cache()


def timed(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


CFG_GRID = [dict(BLOCK_N=bn, warps=w, stages=st) for bn in (32, 64, 128) for w in (2, 4, 8) for st in (1, 2, 3, 4)]


def tune_base(B, L, K, rc, reps, extra):
    """best base-kernel config for one shape (median time over reps)."""
    best = None
    for c in CFG_GRID:
        try:
            st = Step(B, L, K, rc, **c, **extra)
            t = timed(st.base, reps)
        except Exception as e:  # out of shared memory / registers for this config
            t = float("inf")
        if best is None or t < best[0]:
            best = (t, c)
        try:
            del st
        except NameError:
            pass
        torch.cuda.empty_cache()
    return best


def run(B, L, K, reps, graph, tune, extra):
    res = {"B": B, "L": L, "K": K, "extra": extra}
    for rc in (False, True):
        tag = "rc_" if rc else ""
        cfg = dict(BLOCK_N=64, warps=4, stages=2)
        if tune:
            t, cfg = tune_base(B, L, K, rc, max(5, reps // 2), extra)
            res[tag + "cfg"] = cfg
        st = Step(B, L, K, rc, **cfg, **extra)
        res[tag + "nsplit"] = st.nsplit
        res[tag + "base"] = timed(st.base, reps)
        res[tag + "select"] = timed(st.select, reps)
        res[tag + "fix"] = timed(st.fix, reps)
        res[tag + "step"] = timed(st.step, reps)
        if graph:
            st.step()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                st.step()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            with torch.cuda.graph(gr):
                st.step()
            res[tag + "step_graph"] = timed(gr.replay, reps)
        del st
        torch.cuda.empty_cache()
    if B * L * KVH * D * 4 < 12e9:
        k16 = torch.randn(B, KVH, L, D, device="cuda", dtype=torch.float16)
        v16 = torch.randn(B, KVH, L, D, device="cuda", dtype=torch.float16)
        q16 = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
        res["fp16"] = timed(lambda: torch.nn.functional.scaled_dot_product_attention(q16, k16, v16, enable_gqa=True), reps)
        del k16, v16
    res["bytes_2bit_kv"] = 2 * B * KVH * L * (P + 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--B", default="1,8,32")
    ap.add_argument("--L", default="8192,32768,131072")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--rcp", default="tf32")
    ap.add_argument("--reload", type=int, default=0)
    ap.add_argument("--magic", type=int, default=1)
    ap.add_argument("--bisect", type=int, default=1)
    ap.add_argument("--nprog", type=int, default=0)
    ap.add_argument("--out", default="kbench3.json")
    a = ap.parse_args()
    extra = dict(rcp=a.rcp, reload=bool(a.reload), magic=bool(a.magic), bisect=bool(a.bisect), nprog=a.nprog)
    if a.check:
        check(**extra)
        return
    out = []
    for B in (int(x) for x in a.B.split(",")):
        for L in (int(x) for x in a.L.split(",")):
            if B * L > 32 * 131072 // 2:
                continue
            t0 = time.time()
            r = run(B, L, a.K, a.reps, a.graph, a.tune, extra)
            out.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), f"({time.time() - t0:.0f}s)", flush=True)
            with open(a.out, "w") as f:
                json.dump(out, f)


if __name__ == "__main__":
    main()
