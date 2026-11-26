# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""kbench4: fourth-iteration layered-KV decode step (rootcause fork, exploratory). Same shapes and timing as kbench.py; three
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
Correctness is checked against a torch reference (python kbench4.py --check).
v3 changes: the rotary add-back is one dot of [A | B] against a stacked [cos; sin] table at a selectable tensor-core
precision (tf32 / tf32x3 / ieee), optionally reloading the table per tile instead of pinning it in registers; codes
can be widened to fp16 by the exponent trick (0x6400 | c) - 1024 instead of an int-to-float conversion; the per-tile
top-K is a bisection on the probability bits plus a prefix-sum compaction instead of a full bitonic sort; the number
of splits grows at small batch so every SM gets work; --tune sweeps block size, warps and stages per shape.
v4 changes (from the Xavier Volta study: occupancy and load width dominate): the rotary table and its dot can run in fp16
(RCP='fp16': half the shared-memory staging of the tf32 path, so more blocks per SM); the four per-token fp16 scale / zero
values of K and V are packed into one 8-byte record per token (META=1: one wide load instead of four 2-byte loads);
timing: see time_group (15 interleaved rounds, raw per-round times kept) and run (cold-cache layer rotation).
Rotary modes: 0 none; 1 add-back inside the base kernel (v3 style; --rcp tf32|fp16, --reload); 2 a per-step pre-pass
(bias_kernel) writes the bias once and both the base and the merge kernel read it, so they agree exactly.
Usage: python kbench4.py [--check] [--B ..] [--L ..] [--K 64] [--modes 0,2] [--tune] [--rcp fp16] [--reload 1] [--meta 1]
       [--wide 1] [--out kbench4.json]"""
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
def base_kernel(q_ptr, qs_ptr, kp_ptr, ks_ptr, kz_ptr, vp_ptr, vs_ptr, vz_ptr, meta_ptr, u_ptr, w_ptr, invf2_ptr, cs_ptr, bias_ptr,
                m_out, l_out, acc_out, s_out, L, split_len, sm_scale,
                GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, RC: tl.constexpr,
                RCP: tl.constexpr, RELOAD: tl.constexpr, MAGIC: tl.constexpr, META: tl.constexpr, HILO: tl.constexpr = False):
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
    # RC: 0 none; 1 rotary add-back computed in this kernel; 2 add a precomputed per-token bias (bias_kernel)
    if RC == 1:
        od = tl.arange(0, 4 * P)                       # [A | B] over the D/2 frequency pairs, stacked
        invf2 = tl.load(invf2_ptr + od)
        if not RELOAD:   # loop invariants held in registers (u, w, table); RELOAD=1 re-reads them per tile from L1
            u = tl.load(u_ptr + (bk * GH + og[:, None]) * (4 * P) + od[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
            w = tl.load(w_ptr + (bk * GH + og[:, None]) * (4 * P) + od[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
            cs = tl.load(cs_ptr + od[:, None] * BLOCK_N + on[None, :])   # [cos(n f); sin(n f)], [D, BLOCK_N]
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + on
        nm = offs_n < end
        base = bk * L + offs_n
        kb = tl.load(kp_ptr + base[:, None] * P + op[None, :], mask=nm[:, None], other=0)     # [BLOCK_N, P] uint8
        raw = (tl.dot(q0, tl.trans(_widen(kb & 3, MAGIC))) + tl.dot(q1, tl.trans(_widen((kb >> 2) & 3, MAGIC)))
               + tl.dot(q2, tl.trans(_widen((kb >> 4) & 3, MAGIC))) + tl.dot(q3, tl.trans(_widen((kb >> 6) & 3, MAGIC))))
        if META:
            # one 8-byte record per token: (k scale, k zero, v scale, v zero)
            mt = tl.load(meta_ptr + base[:, None] * 4 + tl.arange(0, 4)[None, :], mask=nm[:, None], other=0.0).to(tl.float32)
            # reshape [n, 4] -> [n, 2, 2] is row-major: x[n, a, b] = mt[n, 2a + b]; tl.split splits the last dim
            sc, zp = tl.split(tl.reshape(mt, [BLOCK_N, 2, 2]))   # sc = (k scale, v scale), zp = (k zero, v zero)
            ksc, vsc = tl.split(sc)
            kzp, vzp = tl.split(zp)
        else:
            ksc = tl.load(ks_ptr + base, mask=nm, other=0.0).to(tl.float32)
            kzp = tl.load(kz_ptr + base, mask=nm, other=0.0).to(tl.float32)
        s = (raw * ksc[None, :] + qsum[:, None] * kzp[None, :]) * sm_scale
        if RC == 1:
            t0 = n0.to(tl.float32) * invf2
            if RELOAD:
                u = tl.load(u_ptr + (bk * GH + og[:, None]) * (4 * P) + od[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
                w = tl.load(w_ptr + (bk * GH + og[:, None]) * (4 * P) + od[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
                cs = tl.load(cs_ptr + od[:, None] * BLOCK_N + on[None, :])
            ab = u * tl.cos(t0)[None, :] + w * tl.sin(t0)[None, :]
            if RCP == "fp16":
                s += tl.dot(ab.to(tl.float16), cs.to(tl.float16)) * sm_scale
            else:
                s += tl.dot(ab, cs, input_precision=RCP) * sm_scale
        if RC == 2:   # precomputed rotary bias, already scaled by sm_scale
            s += tl.load(bias_ptr + (bk * GH + og[:, None]) * L + offs_n[None, :], mask=gm[:, None] & nm[None, :], other=0.0)
        s = tl.where(nm[None, :], s, -float("inf"))
        tl.store(s_out + (bk * GH + og[:, None]) * L + offs_n[None, :], s.to(tl.float16), mask=gm[:, None] & nm[None, :])
        mn = tl.maximum(m_i, tl.max(s, 1))
        al = tl.exp(m_i - mn)
        p = tl.exp(s - mn[:, None])
        l_i = l_i * al + tl.sum(p, 1)
        vb = tl.load(vp_ptr + base[:, None] * P + op[None, :], mask=nm[:, None], other=0)
        if not META:
            vsc = tl.load(vs_ptr + base, mask=nm, other=0.0).to(tl.float32)
            vzp = tl.load(vz_ptr + base, mask=nm, other=0.0).to(tl.float32)
        if HILO:   # p * vscale split into fp16 hi + lo (about 22 mantissa bits): the merge subtracts this base contribution
            # of the selected rows again, so its rounding is amplified by kappa (removed base mass / final denominator)
            pw = p * vsc[None, :] * 4096.0
            pv = pw.to(tl.float16)
            pl = ((pw - pv.to(tl.float32)) * 2048.0).to(tl.float16)   # lo scaled by 2^11 (|lo| <= ulp(hi) / 2)
            c0, c1, c2, c3 = _widen(vb & 3, MAGIC), _widen((vb >> 2) & 3, MAGIC), _widen((vb >> 4) & 3, MAGIC), _widen((vb >> 6) & 3, MAGIC)
            a0 = a0 * al[:, None] + tl.dot(pv, c0) + tl.dot(pl, c0) * (1.0 / 2048.0)
            a1 = a1 * al[:, None] + tl.dot(pv, c1) + tl.dot(pl, c1) * (1.0 / 2048.0)
            a2 = a2 * al[:, None] + tl.dot(pv, c2) + tl.dot(pl, c2) * (1.0 / 2048.0)
            a3 = a3 * al[:, None] + tl.dot(pv, c3) + tl.dot(pl, c3) * (1.0 / 2048.0)
        else:
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
    tl.store(ao, a0 / 4096.0 + az[:, None], mask=gm[:, None])
    tl.store(ao + P, a1 / 4096.0 + az[:, None], mask=gm[:, None])
    tl.store(ao + 2 * P, a2 / 4096.0 + az[:, None], mask=gm[:, None])
    tl.store(ao + 3 * P, a3 / 4096.0 + az[:, None], mask=gm[:, None])


@triton.jit
def bias_kernel(u_ptr, w_ptr, c0_ptr, s0_ptr, cs_ptr, bias_out, L, ntile, sm_scale, GH: tl.constexpr, GP: tl.constexpr,
                D: tl.constexpr, BB: tl.constexpr, MODE: tl.constexpr):
    """rotary-aware centring add-back q . R(t) m_pre for every key position, once per decode step and layer, written as
    a per-(query head, token) fp32 bias already scaled by sm_scale. Each program walks several tiles of BB tokens and loads
    the constant [cos(n f); sin(n f)] table [D, BB] once. Per tile starting at t0: [A | B] = u cos(t0 f) + w sin(t0 f)
    (cos / sin of the tile start precomputed in float64 on the host), then bias = [A | B] @ table:
      MODE 'fp16'  one fp16 tensor-core dot with fp32 accumulation (rows padded to GP)
      MODE 'tf32x3' three-pass tf32 (near fp32), slower"""
    bk = tl.program_id(0)
    pid = tl.program_id(1)
    nprog = tl.num_programs(1)
    og = tl.arange(0, GP)
    od = tl.arange(0, D)
    on = tl.arange(0, BB)
    gm = og < GH
    u = tl.load(u_ptr + (bk * GH + og[:, None]) * D + od[None, :], mask=gm[:, None], other=0.0)
    w = tl.load(w_ptr + (bk * GH + og[:, None]) * D + od[None, :], mask=gm[:, None], other=0.0)
    cs = tl.load(cs_ptr + od[:, None] * BB + on[None, :])
    if MODE == "fp16":
        cs16 = cs.to(tl.float16)
    for tt in range(pid, ntile, nprog):
        c0 = tl.load(c0_ptr + tt * D + od)
        s0 = tl.load(s0_ptr + tt * D + od)
        ab = u * c0[None, :] + w * s0[None, :]
        if MODE == "fp16":
            b = tl.dot(ab.to(tl.float16), cs16)
        else:
            b = tl.dot(ab, cs, input_precision="tf32x3")
        offs = tt * BB + on
        tl.store(bias_out + (bk * GH + og[:, None]) * L + offs[None, :], b * sm_scale, mask=gm[:, None] & (offs < L)[None, :])


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
def _kth_bits(b, K):
    """K-th largest value of b (int32; valid entries >= 0, invalid -1), the same threshold the 31-pass bisection finds
    (0 when fewer than K entries are valid): radix select over four 8-bit digits, one 256-bin histogram and one prefix
    sum per digit instead of 31 full counting reductions."""
    ob = tl.arange(0, 512)
    vb = ob < 256
    pre = 0
    k = K
    nvalid = tl.sum((b >= 0).to(tl.int32), 0)
    for p in tl.static_range(4):
        sh = 24 - 8 * p                                                    # compile-time int (p is static)
        if p == 0:
            match = b >= 0
        else:
            match = (b >= 0) & ((b >> (sh + 8)) == pre)
        dig = tl.where(match, (b >> sh) & 0xFF, 256)                      # non-matching lanes go to bin 256 (ignored)
        h = tl.where(vb, tl.histogram(dig, 512), 0)
        suf = tl.sum(h, 0) - tl.cumsum(h, 0) + h                          # entries with digit >= d
        d = tl.max(tl.where(vb & (suf >= k), ob, 0), 0)
        k = k - tl.sum(tl.where(vb & (ob > d), h, 0), 0)                  # minus entries strictly above d
        pre = (pre << 8) | d
    return tl.where(nvalid >= K, pre, 0)


@triton.jit
def sel_stage1(s_in, M_in, Z_in, cand_out, L, ntile, m_in, l_in, nsplit, GH: tl.constexpr, GP: tl.constexpr, T: tl.constexpr,
               K: tl.constexpr, BISECT: tl.constexpr, FUSE: tl.constexpr, NS: tl.constexpr, RADIX: tl.constexpr = False):
    bk = tl.program_id(0)
    tt = tl.program_id(1)
    on = tt * T + tl.arange(0, T)
    nm = on < L
    crit = tl.zeros([T], tl.float32)
    if FUSE:   # per-head global max / sum computed here from the split partials (no separate reduce_stats launch)
        og = tl.arange(0, GP)
        os_ = tl.arange(0, NS)
        sm_ = os_[:, None] < nsplit
        mm = tl.load(m_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_, other=-float("inf"))
        ll = tl.load(l_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_, other=0.0)
        MV = tl.max(mm, 0)
        ZV = tl.sum(ll * tl.exp(mm - MV[None, :]), 0)
    for g in tl.static_range(GH):
        if FUSE:
            M = tl.sum(tl.where(og == g, MV, 0.0), 0)
            Z = tl.sum(tl.where(og == g, ZV, 0.0), 0)
        else:
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
        b32 = tl.where(nm, crit.to(tl.int32, bitcast=True), -1)   # invalid lanes: never counted, never tied
        if RADIX:
            lo = _kth_bits(b32, K)
        else:
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
        ok = tl.arange(0, K)
        tl.store(cand_out + (bk * ntile + tt) * K + ok, tl.full([K], -1, tl.int64))   # slots left unfilled stay -1
        tl.debug_barrier()
        tl.store(cand_out + (bk * ntile + tt) * K + pos, key, mask=pos >= 0)
    else:
        key = tl.where(nm, key, -1)
        key = tl.sort(key, descending=True)
        ok = tl.arange(0, T)
        tl.store(cand_out + (bk * ntile + tt) * K + ok, key, mask=ok < K)


@triton.jit
def _planes(x, R: tl.constexpr, P: tl.constexpr):
    """[R, 4P] in natural channel order -> four [R, P] planes, plane i = channels i*P .. i*P+P-1 (channel = (2a+b)*P + j)."""
    y = tl.permute(tl.reshape(x, [R, 2, 2, P]), (0, 3, 1, 2))     # [R, P, a, b]
    yb0, yb1 = tl.split(y)                                        # b = 0 / 1 -> [R, P, a]
    p0, p2 = tl.split(yb0)
    p1, p3 = tl.split(yb1)
    return p0, p1, p2, p3


@triton.jit
def sel_stage2(cand_ptr, idx_out, C, K: tl.constexpr, CP: tl.constexpr, RADIX: tl.constexpr = False):
    """global top-K of one (batch, KV head) from the per-tile candidates (keys = probability bits << 32 | token, -1 =
    empty): bisection on the probability bits plus a prefix-sum compaction, writing the K token indices. Replaces the
    torch topk + gather + mask + copy of stage 2 (several launches) with one kernel."""
    bk = tl.program_id(0)
    o = tl.arange(0, CP)
    msk = o < C
    hi = (tl.load(cand_ptr + bk * C + o, mask=msk, other=-1) >> 32).to(tl.int32)   # -1 for empty slots
    if RADIX:
        lo = _kth_bits(hi, K)
    else:
        lo = 0
        up = 0x7F800000
        for _ in range(31):
            mid = lo + (up - lo + 1) // 2
            ok_ = tl.sum((hi >= mid).to(tl.int32), 0) >= K
            lo = tl.where(ok_, mid, lo)
            up = tl.where(ok_, up, mid - 1)
    gt = hi > lo
    tie = (hi == lo) & (hi >= 0)
    c1 = tl.sum(gt.to(tl.int32), 0)
    pg = tl.cumsum(gt.to(tl.int32), 0) - 1
    pt = c1 + tl.cumsum(tie.to(tl.int32), 0) - 1
    pos = tl.where(gt, pg, tl.where(tie & (pt < K), pt, -1))
    key = tl.load(cand_ptr + bk * C + o, mask=msk, other=-1)          # re-read: not kept live across the bisection
    tl.store(idx_out + bk * K + pos, key & 0x7FFFFFFF, mask=pos >= 0)


@triton.jit
def fix_kernel(q_ptr, qs_ptr, m_in, l_in, acc_in, idx_ptr, kex_ptr, vex_ptr, kp_ptr, ks_ptr, kz_ptr, vp_ptr, vs_ptr, vz_ptr,
               u_ptr, invf2_ptr, cs_ptr, bias_ptr, meta_ptr, cand_ptr, C,
               out_ptr, L, nsplit, sm_scale, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr, K: tl.constexpr, NS: tl.constexpr,
               RC: tl.constexpr, BLOCK_N: tl.constexpr, RCP: tl.constexpr, WIDE: tl.constexpr, META: tl.constexpr,
               SEL: tl.constexpr, CP: tl.constexpr, UNROLL: tl.constexpr):
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
    ok = tl.arange(0, K)
    if SEL:   # stage-2 top-K fused here (same bisection + compaction as sel_stage2); idx written, then read back
        oc = tl.arange(0, CP)
        cm = oc < C
        hi = (tl.load(cand_ptr + bk * C + oc, mask=cm, other=-1) >> 32).to(tl.int32)
        lo = 0
        up = 0x7F800000
        for _ in range(31):
            mid = lo + (up - lo + 1) // 2
            ok_ = tl.sum((hi >= mid).to(tl.int32), 0) >= K
            lo = tl.where(ok_, mid, lo)
            up = tl.where(ok_, up, mid - 1)
        gt = hi > lo
        tie = (hi == lo) & (hi >= 0)
        c1 = tl.sum(gt.to(tl.int32), 0)
        pos = tl.where(gt, tl.cumsum(gt.to(tl.int32), 0) - 1,
                       tl.where(tie & (c1 + tl.cumsum(tie.to(tl.int32), 0) - 1 < K), c1 + tl.cumsum(tie.to(tl.int32), 0) - 1, -1))
        key = tl.load(cand_ptr + bk * C + oc, mask=cm, other=-1)
        tl.store(idx_ptr + bk * K + pos, key & 0x7FFFFFFF, mask=pos >= 0)
        tl.debug_barrier()
    tok = tl.load(idx_ptr + bk * K + ok)
    row = bk * L + tok
    qb = q_ptr + (bk * GH + og[:, None]) * D4 + op[None, :]
    q0 = tl.load(qb, mask=gm[:, None], other=0.0)
    q1 = tl.load(qb + P, mask=gm[:, None], other=0.0)
    q2 = tl.load(qb + 2 * P, mask=gm[:, None], other=0.0)
    q3 = tl.load(qb + 3 * P, mask=gm[:, None], other=0.0)
    od = tl.arange(0, D4)
    if WIDE:   # each exact row as one 256-byte read over PCIe (zero copy from pinned host memory)
        qf = tl.load(q_ptr + (bk * GH + og[:, None]) * D4 + od[None, :], mask=gm[:, None], other=0.0)
        se = tl.dot(qf, tl.trans(tl.load(kex_ptr + row[:, None] * D4 + od[None, :]))) * sm_scale
        vx = tl.load(vex_ptr + row[:, None] * D4 + od[None, :])
    else:      # four 64-byte reads per row
        kxb = kex_ptr + row[:, None] * D4 + op[None, :]
        se = (tl.dot(q0, tl.trans(tl.load(kxb))) + tl.dot(q1, tl.trans(tl.load(kxb + P))) + tl.dot(q2, tl.trans(tl.load(kxb + 2 * P)))
              + tl.dot(q3, tl.trans(tl.load(kxb + 3 * P)))) * sm_scale
    if META:   # one 8-byte record per selected token: (k scale, k zero, v scale, v zero)
        mt = tl.load(meta_ptr + row[:, None] * 4 + tl.arange(0, 4)[None, :]).to(tl.float32)
        sc_, zp_ = tl.split(tl.reshape(mt, [K, 2, 2]))
        ksr, vsc = tl.split(sc_)
        kzr, vzp = tl.split(zp_)
    else:
        ksr = tl.load(ks_ptr + row).to(tl.float32)
        kzr = tl.load(kz_ptr + row).to(tl.float32)
        vsc = tl.load(vs_ptr + row).to(tl.float32)
        vzp = tl.load(vz_ptr + row).to(tl.float32)
    kb = tl.load(kp_ptr + row[:, None] * P + op[None, :])
    raw = (tl.dot(q0, tl.trans((kb & 3).to(tl.float16))) + tl.dot(q1, tl.trans(((kb >> 2) & 3).to(tl.float16)))
           + tl.dot(q2, tl.trans(((kb >> 4) & 3).to(tl.float16))) + tl.dot(q3, tl.trans(((kb >> 6) & 3).to(tl.float16))))
    qsum = tl.load(qs_ptr + bk * GH + og, mask=gm, other=0.0)
    sb = (raw * ksr[None, :] + qsum[:, None] * kzr[None, :]) * sm_scale
    if RC == 2:   # the same precomputed bias the base kernel added, so the removed base contribution matches exactly
        sb += tl.load(bias_ptr + (bk * GH + og[:, None]) * L + tok[None, :], mask=gm[:, None], other=0.0)
    if RC == 1:
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
        if RCP == "fp16":
            rb = (tl.dot(ca.to(tl.float16), tl.trans(cosv.to(tl.float16))) + tl.dot(cb.to(tl.float16), tl.trans(sinv.to(tl.float16)))).to(tl.float32)
        else:
            rb = tl.dot(ca, tl.trans(cosv), input_precision=RCP) + tl.dot(cb, tl.trans(sinv), input_precision=RCP)
        sb += rb * sm_scale
    # split reduction placed after the exact-row reads are issued, so the PCIe round trips overlap it; only the GH real
    # query rows of the partials are stored by the base kernel and read here
    A0 = tl.zeros([GP, P], tl.float32)
    A1 = tl.zeros([GP, P], tl.float32)
    A2 = tl.zeros([GP, P], tl.float32)
    A3 = tl.zeros([GP, P], tl.float32)
    for s0 in range(0, nsplit, UNROLL):   # UNROLL splits per iteration: more partial loads in flight
        for u_ in tl.static_range(UNROLL):
            s_ = s0 + u_
            vm = gm & (s_ < nsplit)
            w = tl.exp(tl.load(m_in + (bk * nsplit + s_) * GP + og, mask=vm, other=-float("inf")) - M)[:, None]
            ab = acc_in + ((bk * nsplit + s_) * GP + og[:, None]) * D4 + op[None, :]
            A0 += tl.load(ab, mask=vm[:, None], other=0.0) * w
            A1 += tl.load(ab + P, mask=vm[:, None], other=0.0) * w
            A2 += tl.load(ab + 2 * P, mask=vm[:, None], other=0.0) * w
            A3 += tl.load(ab + 3 * P, mask=vm[:, None], other=0.0) * w
    M2 = tl.maximum(M, tl.max(se, 1))  # exact keys already contain the offset: no rotary term on se
    r = tl.exp(M - M2)
    eb = tl.exp(sb - M2[:, None])
    ee = tl.exp(se - M2[:, None])
    Z2 = Z * r - tl.sum(eb, 1) + tl.sum(ee, 1)
    vb = tl.load(vp_ptr + row[:, None] * P + op[None, :])
    ebs = (eb * vsc[None, :] * 4096.0).to(tl.float16)
    ebz = tl.sum(eb * vzp[None, :], 1)[:, None]
    ee16 = (ee * 4096.0).to(tl.float16)
    vxb = vex_ptr + row[:, None] * D4 + op[None, :]
    ob = out_ptr + (bk * GH + og[:, None]) * D4 + op[None, :]
    inv = 1.0 / Z2[:, None]
    if WIDE:
        e0, e1, e2, e3 = _planes(tl.dot(ee16, vx), GP, P)
    else:
        e0 = tl.dot(ee16, tl.load(vxb))
        e1 = tl.dot(ee16, tl.load(vxb + P))
        e2 = tl.dot(ee16, tl.load(vxb + 2 * P))
        e3 = tl.dot(ee16, tl.load(vxb + 3 * P))
    o0 = (A0 * r[:, None] - (tl.dot(ebs, (vb & 3).to(tl.float16)) - e0) / 4096.0 - ebz) * inv
    o1 = (A1 * r[:, None] - (tl.dot(ebs, ((vb >> 2) & 3).to(tl.float16)) - e1) / 4096.0 - ebz) * inv
    o2 = (A2 * r[:, None] - (tl.dot(ebs, ((vb >> 4) & 3).to(tl.float16)) - e2) / 4096.0 - ebz) * inv
    o3 = (A3 * r[:, None] - (tl.dot(ebs, ((vb >> 6) & 3).to(tl.float16)) - e3) / 4096.0 - ebz) * inv
    tl.store(ob, o0, mask=gm[:, None])
    tl.store(ob + P, o1, mask=gm[:, None])
    tl.store(ob + 2 * P, o2, mask=gm[:, None])
    tl.store(ob + 3 * P, o3, mask=gm[:, None])


@triton.jit
def fixA_kernel(q_ptr, qs_ptr, m_in, idx_ptr, kex_ptr, vex_ptr, kp_ptr, vp_ptr, meta_ptr, bias_ptr,
                pn_out, pz_out, pm_out, prev_ptr, ck_ptr, cv_ptr, nk_ptr, nv_ptr, L, nsplit, sm_scale, GH: tl.constexpr,
                GP: tl.constexpr, P: tl.constexpr, K: tl.constexpr, KC: tl.constexpr, NS: tl.constexpr, RC: tl.constexpr,
                CACHE: tl.constexpr, HILO: tl.constexpr = False):
    """phase A of the two-phase merge: one program per (batch * KV head, chunk of KC selected keys). Reads the chunk's
    exact rows (zero copy, issued by NCH programs in parallel instead of one) and its base rows, and writes the chunk's
    contribution relative to its own reference max Mc = max(M, max exact score): dN = exact - base numerator (natural
    channel order), dZ = sum exp(se - Mc) - sum exp(sb - Mc), and Mc. Production path only: packed metadata, wide rows,
    rotary mode 0 or 2."""
    bk = tl.program_id(0)
    c = tl.program_id(1)
    nch = tl.num_programs(1)
    og = tl.arange(0, GP)
    op = tl.arange(0, P)
    gm = og < GH
    D4: tl.constexpr = 4 * P
    od = tl.arange(0, D4)
    os_ = tl.arange(0, NS)
    mm = tl.load(m_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=os_[:, None] < nsplit, other=-float("inf"))
    M = tl.max(mm, 0)
    ok = c * KC + tl.arange(0, KC)
    tok = tl.load(idx_ptr + bk * K + ok)
    row = bk * L + tok
    qf = tl.load(q_ptr + (bk * GH + og[:, None]) * D4 + od[None, :], mask=gm[:, None], other=0.0)
    if CACHE:
        # cross-step cache: the previous step's selected exact rows sit on the GPU (ck / cv, slot j holds token prev[j]);
        # a hit reads the GPU copy, a miss reads pinned host memory. The two loads use complementary masks, and a
        # predicated-off lane issues no PCIe request, so only misses cross the bus. Rows used now are written to the
        # next cache (nk / nv) in selection order, so the next step's prev list is this step's idx.
        pk = tl.arange(0, K)
        prev = tl.load(prev_ptr + bk * K + pk)
        match = tok[:, None] == prev[None, :]                                  # [KC, K]
        hit = tl.max(match.to(tl.int32), 1) > 0
        slot = tl.min(tl.where(match, pk[None, :], K), 1)                  # first matching slot (K when none)
        slot = tl.where(hit, slot, 0)
        crow = bk * K + slot
        kx = tl.where(hit[:, None], tl.load(ck_ptr + crow[:, None] * D4 + od[None, :], mask=hit[:, None], other=0.0),
                      tl.load(kex_ptr + row[:, None] * D4 + od[None, :], mask=~hit[:, None], other=0.0))
        vx = tl.where(hit[:, None], tl.load(cv_ptr + crow[:, None] * D4 + od[None, :], mask=hit[:, None], other=0.0),
                      tl.load(vex_ptr + row[:, None] * D4 + od[None, :], mask=~hit[:, None], other=0.0))
        nrow = bk * K + ok
        tl.store(nk_ptr + nrow[:, None] * D4 + od[None, :], kx)
        tl.store(nv_ptr + nrow[:, None] * D4 + od[None, :], vx)
    else:
        kx = tl.load(kex_ptr + row[:, None] * D4 + od[None, :])
        vx = tl.load(vex_ptr + row[:, None] * D4 + od[None, :])
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
    Mc = tl.maximum(M, tl.max(se, 1))
    ee = tl.exp(se - Mc[:, None])
    eb = tl.exp(sb - Mc[:, None])
    dz = tl.sum(ee, 1) - tl.sum(eb, 1)
    ebz = tl.sum(eb * vzp[None, :], 1)[:, None]
    vb = tl.load(vp_ptr + row[:, None] * P + op[None, :])
    o = (bk * nch + c) * GP + og
    pb = pn_out + o[:, None] * D4 + op[None, :]
    if HILO:   # fp16 hi + lo (lo scaled by 2^11) for both weights: the subtracted base term must match the base kernel's
        ew = eb * vsc[None, :] * 4096.0
        ebs = ew.to(tl.float16)
        ebl = ((ew - ebs.to(tl.float32)) * 2048.0).to(tl.float16)
        xw = ee * 4096.0
        xh = xw.to(tl.float16)
        xl = ((xw - xh.to(tl.float32)) * 2048.0).to(tl.float16)
        e0, e1, e2, e3 = _planes(tl.dot(xh, vx) + tl.dot(xl, vx) * (1.0 / 2048.0), GP, P)
        q0, q1, q2, q3 = (vb & 3).to(tl.float16), ((vb >> 2) & 3).to(tl.float16), ((vb >> 4) & 3).to(tl.float16), ((vb >> 6) & 3).to(tl.float16)
        tl.store(pb, (e0 - tl.dot(ebs, q0) - tl.dot(ebl, q0) * (1.0 / 2048.0)) / 4096.0 - ebz, mask=gm[:, None])
        tl.store(pb + P, (e1 - tl.dot(ebs, q1) - tl.dot(ebl, q1) * (1.0 / 2048.0)) / 4096.0 - ebz, mask=gm[:, None])
        tl.store(pb + 2 * P, (e2 - tl.dot(ebs, q2) - tl.dot(ebl, q2) * (1.0 / 2048.0)) / 4096.0 - ebz, mask=gm[:, None])
        tl.store(pb + 3 * P, (e3 - tl.dot(ebs, q3) - tl.dot(ebl, q3) * (1.0 / 2048.0)) / 4096.0 - ebz, mask=gm[:, None])
    else:
        ebs = (eb * vsc[None, :] * 4096.0).to(tl.float16)
        e0, e1, e2, e3 = _planes(tl.dot((ee * 4096.0).to(tl.float16), vx), GP, P)
        tl.store(pb, (e0 - tl.dot(ebs, (vb & 3).to(tl.float16))) / 4096.0 - ebz, mask=gm[:, None])
        tl.store(pb + P, (e1 - tl.dot(ebs, ((vb >> 2) & 3).to(tl.float16))) / 4096.0 - ebz, mask=gm[:, None])
        tl.store(pb + 2 * P, (e2 - tl.dot(ebs, ((vb >> 4) & 3).to(tl.float16))) / 4096.0 - ebz, mask=gm[:, None])
        tl.store(pb + 3 * P, (e3 - tl.dot(ebs, ((vb >> 6) & 3).to(tl.float16))) / 4096.0 - ebz, mask=gm[:, None])
    tl.store(pz_out + o, dz, mask=gm)
    tl.store(pm_out + o, Mc, mask=gm)


@triton.jit
def fixB_kernel(m_in, l_in, acc_in, pn_in, pz_in, pm_in, out_ptr, nsplit, nch, GH: tl.constexpr, GP: tl.constexpr,
                P: tl.constexpr, NS: tl.constexpr, NC: tl.constexpr, UNROLL: tl.constexpr):
    """phase B: one program per (batch * KV head, channel plane). Reduces the split partials of that plane and merges
    the NCH chunk contributions by log-sum-exp rescaling: M2 = max(M, max_c Mc), out = (A r + sum_c dN_c exp(Mc - M2))
    / (Z r + sum_c dZ_c exp(Mc - M2)), r = exp(M - M2)."""
    bk = tl.program_id(0)
    pl = tl.program_id(1)
    og = tl.arange(0, GP)
    op = tl.arange(0, P)
    gm = og < GH
    D4: tl.constexpr = 4 * P
    os_ = tl.arange(0, NS)
    sm_ = os_[:, None] < nsplit
    mm = tl.load(m_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_, other=-float("inf"))
    ll = tl.load(l_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_, other=0.0)
    M = tl.max(mm, 0)
    Z = tl.sum(ll * tl.exp(mm - M[None, :]), 0)
    A = tl.zeros([GP, P], tl.float32)
    for s0 in range(0, nsplit, UNROLL):
        for u_ in tl.static_range(UNROLL):
            s_ = s0 + u_
            vm = gm & (s_ < nsplit)
            w = tl.exp(tl.load(m_in + (bk * nsplit + s_) * GP + og, mask=vm, other=-float("inf")) - M)
            w = tl.where(vm, w, 0.0)
            A += tl.load(acc_in + ((bk * nsplit + s_) * GP + og[:, None]) * D4 + pl * P + op[None, :], mask=vm[:, None], other=0.0) * w[:, None]
    oc = tl.arange(0, NC)
    cm = (oc[:, None] < nch) & gm[None, :]
    pm = tl.load(pm_in + (bk * nch + oc[:, None]) * GP + og[None, :], mask=cm, other=-float("inf"))
    M2 = tl.maximum(M, tl.max(pm, 0))
    r = tl.exp(M - M2)
    wc = tl.where(cm, tl.exp(pm - M2[None, :]), 0.0)                               # [NC, GP]
    pz = tl.load(pz_in + (bk * nch + oc[:, None]) * GP + og[None, :], mask=cm, other=0.0)
    Z2 = Z * r + tl.sum(pz * wc, 0)
    N = A * r[:, None]
    for c in range(0, nch):
        wv = tl.sum(tl.where(oc[:, None] == c, wc, 0.0), 0)                       # [GP]
        N += tl.load(pn_in + ((bk * nch + c) * GP + og[:, None]) * D4 + pl * P + op[None, :], mask=gm[:, None], other=0.0) * wv[:, None]
    tl.store(out_ptr + (bk * GH + og[:, None]) * D4 + pl * P + op[None, :], N / Z2[:, None], mask=gm[:, None])


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
                 magic=True, bisect=True, nprog=0, meta=True, wide=True, BB=64, host=None, qscale=1.0, bias_mode="fp16",
                 ns_fixed=0, sel2=True, fuse=True, unroll=4, fixp=False, kc=16, cache=False, hilo=False, radix=False,
                 bias16=False):
        g = torch.Generator(device="cpu").manual_seed(seed)
        rc = int(rc)   # 0 none, 1 in-kernel rotary add-back, 2 precomputed bias (bias_kernel)
        self.bias_mode, self.sel2, self.fuse, self.unroll, self.fixp, self.kc = bias_mode, sel2, fuse and not fixp, unroll, fixp, kc
        self.cache = cache
        self.hilo = hilo     # base kernel: fp16 hi + lo split of p * vscale (merge cancellation stays at fp32 level)
        self.radix = radix   # top-K threshold by radix select (4 histogram passes) instead of 31-pass bisection
        if cache:   # cross-step exact-row cache lives in the two-phase merge
            fixp = self.fixp = True
            self.fuse = False
        if fixp:   # two-phase merge: production path only
            assert int(rc) in (0, 2) and meta and wide and K % kc == 0 and kc >= 16 and kc & (kc - 1) == 0
        assert L >= K, "context shorter than the number of fetched keys"
        self.B, self.L, self.K, self.rc, self.BLOCK_N, self.T, self.wide, self.BB = B, L, K, rc, BLOCK_N, T, wide, BB
        self.warps, self.stages, self.rcp, self.reload, self.magic, self.bisect, self.meta = warps, stages, rcp, reload, magic, bisect, meta
        BK = self.BK = B * KVH
        self.q = (torch.randn(BK * G, D, generator=g) * qscale).half().to(dev)   # qscale > 1: sharper attention
        self.qs = self.q.float().sum(-1).contiguous()
        self.kp = torch.randint(0, 256, (BK, L, P), generator=g, dtype=torch.uint8).to(dev)
        self.vp = torch.randint(0, 256, (BK, L, P), generator=g, dtype=torch.uint8).to(dev)
        self.ks = (torch.rand(BK, L, generator=g) * 0.1).half().to(dev)
        self.kz = (-torch.rand(BK, L, generator=g) * 0.15).half().to(dev)
        self.vs = (torch.rand(BK, L, generator=g) * 0.1 + 0.02).half().to(dev)      # independent of K, so a K/V mix-up shows
        self.vz = (-torch.rand(BK, L, generator=g) * 0.15 - 0.01).half().to(dev)
        self.mt = torch.stack([self.ks, self.kz, self.vs, self.vz], -1).contiguous()   # [BK, L, 4] fp16
        if host is None:   # exact fp16 K / V in pinned host memory (shared between Step objects of one shape when given)
            host = (torch.randn(BK, L, D, generator=g).half().pin_memory(), torch.randn(BK, L, D, generator=g).half().pin_memory())
        self.kh, self.vh = host
        # exact rows: zero-copy views of pinned host memory, or device tensors (ablation: no PCIe in the merge)
        self.kz_, self.vz_ = (self.kh, self.vh) if self.kh.is_cuda else (host_as_cuda(self.kh), host_as_cuda(self.vh))
        invf64 = 1.0 / (500000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
        invf = invf64.float()
        self.invf, self.invf64 = invf.to(dev), invf64.to(dev)
        # bias pre-pass tables: cos / sin of every tile start t0 = i * BB (float64, then fp32) and of the in-tile offsets
        t0 = torch.arange(triton.cdiv(L, BB), dtype=torch.float64)[:, None] * BB * invf64[None]
        self.c0tab = torch.cat([t0.cos(), t0.cos()], 1).float().contiguous().to(dev)
        self.s0tab = torch.cat([t0.sin(), t0.sin()], 1).float().contiguous().to(dev)
        nb = torch.arange(BB, dtype=torch.float64)
        self.csB = torch.cat([torch.cos(invf64[:, None] * nb[None]), torch.sin(invf64[:, None] * nb[None])], 0).float().contiguous().to(dev)
        # bias16: the RC = 2 bias stored in fp16 (half its write + read traffic; base and merge read the same rounded value)
        self.bias = torch.empty(BK * G, L, device=dev, dtype=torch.float16 if bias16 else torch.float32)
        on = torch.arange(BLOCK_N, dtype=torch.float64)
        self.ctab = torch.cos(invf64[:, None] * on[None]).float().contiguous().to(dev)
        self.stab = torch.sin(invf64[:, None] * on[None]).float().contiguous().to(dev)
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
        # operands of the in-kernel path (RC = 1): stored in fp16 when the rotary dot runs in fp16 (half the staging)
        h = (lambda x: x.half().contiguous()) if rcp == "fp16" else (lambda x: x)
        self.u_k, self.w_k, self.cs_k = h(self.u), h(self.w), h(self.cs)
        self.mpre = mpre
        # enough splits that every SM gets several programs at small batch (nprog: target number of programs)
        self.nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        nprog = nprog or 4 * self.nsm
        ns = max(1, min(256, max((L + 2047) // 2048, -(-nprog // BK)), L // (2 * BLOCK_N)))
        if ns_fixed:   # fixed number of splits (time-vs-L fits need the same reduction length at every L)
            ns = ns_fixed
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
        self.nch = K // kc
        self.pn = torch.empty(BK * self.nch * GP, D, device=dev)
        self.pz = torch.empty(BK * self.nch * GP, device=dev)
        self.pm = torch.empty(BK * self.nch * GP, device=dev)
        # cross-step cache (double buffer): previous selection, its exact rows, and the rows written this step
        self.prev = torch.full((BK, K), -1, device=dev, dtype=torch.int64)   # -1: empty cache, never matches a token
        self.ck, self.cv = torch.zeros(BK * K, D, device=dev, dtype=torch.float16), torch.zeros(BK * K, D, device=dev, dtype=torch.float16)
        self.nk, self.nv = torch.empty_like(self.ck), torch.empty_like(self.cv)
        self.scale = 1 / math.sqrt(D)

    def prep(self):
        """per-step rotary bias pre-pass (RC = 2): bias[head, token] = sm_scale * q . R(t) m_pre."""
        ntile = triton.cdiv(self.L, self.BB)
        npb = max(1, min(ntile, -(-2 * self.nsm // self.BK)))   # about two programs per SM in total
        bias_kernel[(self.BK, npb)](self.u, self.w, self.c0tab, self.s0tab, self.csB, self.bias, self.L, ntile, self.scale,
                                    GH=G, GP=GP, D=D, BB=self.BB, MODE=self.bias_mode, num_warps=4)

    def base(self):
        if self.rc == 2:
            self.prep()
        return base_kernel[(self.BK, self.nsplit)](self.q, self.qs, self.kp, self.ks, self.kz, self.vp, self.vs, self.vz, self.mt, self.u_k,
                                            self.w_k, self.invf2, self.cs_k, self.bias, self.m, self.l, self.acc, self.s, self.L,
                                            self.split_len,
                                            self.scale, GH=G, GP=GP, P=P, BLOCK_N=self.BLOCK_N, RC=self.rc, RCP=self.rcp,
                                            RELOAD=self.reload, MAGIC=self.magic, META=self.meta, HILO=self.hilo,
                                            num_warps=self.warps,
                                            num_stages=self.stages)

    def select(self):
        if not self.fuse:
            reduce_stats_kernel[(self.BK,)](self.m, self.l, self.M, self.Z, self.nsplit, GH=G, GP=GP, NS=self.NS)
        sel_stage1[(self.BK, self.ntile)](self.s, self.M, self.Z, self.cand, self.L, self.ntile, self.m, self.l, self.nsplit, GH=G,
                                          GP=GP, T=self.T, K=self.K, BISECT=self.bisect, FUSE=self.fuse, NS=self.NS,
                                          RADIX=self.radix, num_warps=4)
        if self.fuse:   # stage 2 runs inside the merge kernel
            return
        if self.sel2:
            C = self.ntile * self.K
            assert C <= 16384, "stage-2 candidates exceed one program; use the torch path"
            sel_stage2[(self.BK,)](self.cand, self.idx, C, K=self.K, CP=triton.next_power_of_2(C), RADIX=self.radix, num_warps=8)
        else:
            top = (self.cand >> 32).topk(self.K, -1).indices
            self.idx.copy_(torch.gather(self.cand, 1, top) & 0xFFFFFFFF)

    def fix(self):
        if self.fixp:
            fixA_kernel[(self.BK, self.nch)](self.q, self.qs, self.m, self.idx, self.kz_, self.vz_, self.kp, self.vp, self.mt, self.bias,
                                             self.pn, self.pz, self.pm, self.prev, self.ck, self.cv, self.nk, self.nv, self.L,
                                             self.nsplit, self.scale, GH=G, GP=GP, P=P, K=self.K, KC=self.kc, NS=self.NS, RC=self.rc,
                                             CACHE=self.cache, HILO=self.hilo, num_warps=4)
            return fixB_kernel[(self.BK, 4)](self.m, self.l, self.acc, self.pn, self.pz, self.pm, self.out, self.nsplit, self.nch,
                                             GH=G, GP=GP, P=P, NS=self.NS, NC=triton.next_power_of_2(self.nch), UNROLL=self.unroll,
                                             num_warps=4)
        return fix_kernel[(self.BK,)](self.q, self.qs, self.m, self.l, self.acc, self.idx, self.kz_, self.vz_, self.kp, self.ks, self.kz,
                               self.vp, self.vs, self.vz, self.u, self.invf2, self.cs, self.bias, self.mt, self.cand,
                               self.ntile * self.K, self.out, self.L,
                               self.nsplit, self.scale, GH=G, GP=GP, P=P, K=self.K, NS=self.NS, RC=self.rc, BLOCK_N=self.BLOCK_N,
                               RCP="tf32" if self.rcp == "fp16" else self.rcp, WIDE=self.wide, META=self.meta,
                               SEL=self.fuse, CP=triton.next_power_of_2(self.ntile * self.K), UNROLL=self.unroll, num_warps=4)

    def seed_cache(self, hit_rate, seed=0):
        """benchmark helper: fill prev / ck / cv so that a fraction hit_rate of the current selection (self.idx) is cached
        (the rest of prev are tokens not selected). Random data selects the same keys on every replay, so a real
        previous-step list cannot be used; this fixes the hit rate at the measured cross-step overlap instead. Not for use
        after swap() (it reads idx as the current selection)."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = self.idx.cpu()
        prev = torch.empty_like(idx)
        nh = int(round(hit_rate * self.K))
        for b in range(self.BK):
            sel = idx[b].tolist()
            sset = set(sel)
            others = [t for t in torch.randperm(self.L, generator=g).tolist() if t not in sset][: self.K - nh]
            row = sel[:nh] + others
            prev[b] = torch.tensor([row[j] for j in torch.randperm(self.K, generator=g).tolist()])
        self.prev.copy_(prev)
        rows = (torch.arange(self.BK)[:, None] * self.L + prev).reshape(-1)
        self.ck.copy_(self.kh.reshape(-1, D)[rows].to(self.ck.device))
        self.cv.copy_(self.vh.reshape(-1, D)[rows].to(self.cv.device))

    def swap(self):
        """ping-pong the cross-step cache by swapping buffer references: this step's selection (idx) and exact rows (nk / nv)
        become the next step's prev / ck / cv, and the old cache buffers receive the next step's writes. Capturing one CUDA
        graph before and one after a swap gives the two parities of a real decode loop, replayed alternately with no copy."""
        self.prev, self.idx = self.idx, self.prev
        self.ck, self.nk = self.nk, self.ck
        self.cv, self.nv = self.nv, self.cv

    def reset_slot(self, b):
        """a new request in batch slot b: empty its cache in both parities (prev and idx swap roles), so no stale token
        can hit; issued on the current stream before the slot's next step."""
        self.prev[b * KVH:(b + 1) * KVH].fill_(-1)
        self.idx[b * KVH:(b + 1) * KVH].fill_(-1)

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
        t = torch.arange(L, device=q.device, dtype=torch.float64)[:, None] * st.invf64[None]   # float64 angles
        cos, sin = t.cos().float(), t.sin().float()
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


def _check1(tag, rc, L, qscale=1.0, **cfg):
    st = Step(2, L, 64, rc, qscale=qscale, **cfg)
    st.step()
    torch.cuda.synchronize()
    o_ref, idx_ref = reference(st)
    got = st.out.view(st.BK, G, D)
    same = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(st.idx.cpu(), idx_ref.cpu())) / idx_ref.numel()
    o_same, _ = reference(st, st.idx)
    p = torch.softmax(torch.einsum("bgd,bld->bgl", st.q.float().view(st.BK, G, D), st.kh.cuda().float()) * st.scale, -1)
    mass = float(p.topk(64, -1).values.sum(-1).mean())
    dup = int(sum(len(r) - len(set(r)) for r in st.idx.cpu().tolist()))
    err = float((got - o_same).norm() / o_same.norm())
    print(f"check {tag} rc={rc} L={L} qscale={qscale} nsplit={st.nsplit}: top-64 exact mass {mass:.2f}; overlap with reference "
          f"{same:.4f}; duplicate picks {dup}; rel err vs reference {float((got - o_ref).norm() / o_ref.norm()):.3g}, with the "
          f"kernel's own selection {err:.3g} {'OK' if err < 1e-3 and dup == 0 else 'FAIL'}", flush=True)
    del st
    torch.cuda.empty_cache()


def check(**cfg):
    """kernel vs fp32 torch reference (float64 rotary angles). Production modes 0 and 2 at an unaligned and a long L,
    with flat (qscale 1) and sharp (qscale 4) attention; the remaining branches (WIDE=0, META=0, bisect off, int-to-float
    widening, other tile configs) once each; then bitwise checks that pure re-layouts change nothing (META on/off) and
    that a rerun is identical. Mode 1 (in-kernel add-back) is an occupancy probe only and is not checked here."""
    for rc in (0, 2):
        for L in (5000, 70001):
            for qs in (1.0, 4.0):
                _check1("prod", rc, L, qs, **cfg)
    for tag, kw in (("wide0", dict(wide=False)), ("meta0", dict(meta=False)), ("sort", dict(bisect=False)),
                    ("int2f", dict(magic=False)), ("bn32w2s1", dict(BLOCK_N=32, warps=2, stages=1)),
                    ("bn128w8s3", dict(BLOCK_N=128, warps=8, stages=3))):
        _check1(tag, 2, 5000, 4.0, **dict(cfg, **kw))
    for rc in (0, 2):   # two-phase merge and cross-step cache must reproduce the one-phase merge
        for L in (5000, 70001):
            st = Step(2, L, 64, rc, qscale=4.0, fuse=False, **cfg)
            st.step()
            o1 = st.out.clone()
            st.fixp = True
            st.fix()
            o2 = st.out.clone()
            st.cache = True
            st.seed_cache(0.6)
            st.fix()
            torch.cuda.synchronize()
            d2 = float((o2 - o1).norm() / o1.norm())
            d3 = float((st.out - o1).norm() / o1.norm())
            nk_ok = bool(torch.equal(st.nk.view(st.BK, st.K, D).cpu(), torch.gather(st.kh, 1, st.idx.cpu()[..., None].expand(-1, -1, D))))
            # the hit branch must really be used: poisoning the cache changes the output at hit rate 1, not at hit rate 0
            st.seed_cache(1.0)
            st.ck.mul_(2.0)
            st.fix()
            d_poison1 = float((st.out - o1).norm() / o1.norm())
            st.seed_cache(0.0)
            st.ck.mul_(2.0)
            st.fix()
            d_poison0 = float((st.out - o1).norm() / o1.norm())
            st.prev.fill_(-1)                       # empty cache (initial state): everything from host
            st.fix()
            d_empty = float((st.out - o1).norm() / o1.norm())
            nk_ok = nk_ok and d_poison1 > 1e-2 and d_poison0 < 1e-3 and d_empty < 1e-3
            print(f"  cache branch: poisoned at hit rate 1 -> {d_poison1:.2e} (must be large), at hit rate 0 -> {d_poison0:.2e}, "
                  f"empty cache -> {d_empty:.2e}", flush=True)
            print(f"check two-phase rc={rc} L={L}: vs one-phase {d2:.2e}; cached (60% hits) vs one-phase {d3:.2e}; next-cache rows "
                  f"{'exact' if nk_ok else 'WRONG'} {'OK' if max(d2, d3) < 1e-3 and nk_ok else 'FAIL'}", flush=True)
            del st
            torch.cuda.empty_cache()
    for rc in (0, 2):
        outs = {}
        for tag, kw in (("meta1", dict(cfg, meta=True)), ("meta0", dict(cfg, meta=False)), ("meta1_again", dict(cfg, meta=True))):
            st = Step(2, 5000, 64, rc, qscale=4.0, **kw)
            st.step()
            torch.cuda.synchronize()
            outs[tag] = (st.out.clone(), st.idx.clone(), st.s.clone())
            del st
        for tag in ("meta0", "meta1_again"):
            eq = all(torch.equal(x, y) for x, y in zip(outs["meta1"], outs[tag]))
            print(f"bitwise rc={rc}: meta1 vs {tag}: {'IDENTICAL' if eq else 'DIFFERENT'}", flush=True)


def _clock():
    try:
        return torch.cuda.clock_rate()   # MHz via NVML, may be unavailable in containers
    except Exception:
        return None


def time_group(fns, rounds=15, min_ms=50.0, seed=0, flush=None):
    """interleaved timing of several callables: warm each for 300 ms; then per round, in a freshly shuffled order, one
    event pair around n back-to-back calls (n sized so a round lasts >= min_ms, which amortises host launch latency);
    per-call time = elapsed / n. With flush (a callable that evicts L2) every call is preceded by it and the flush-only
    time is subtracted. Returns {name: dict(ms=median over rounds, spread=max/min over rounds, clocks=[MHz per round])}."""
    import random
    rng = random.Random(seed)
    names = list(fns)
    for nm in names:
        t0 = time.time()
        while time.time() - t0 < 0.3:
            fns[nm]()
        torch.cuda.synchronize()
    nrep = {}
    for nm in names:   # size n from a short probe
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            fns[nm]()
        b.record()
        torch.cuda.synchronize()
        nrep[nm] = max(10, int(min_ms / max(a.elapsed_time(b) / 10, 1e-3)))
    if flush is not None:
        fns = dict(fns, _flush=lambda: None)
        nrep["_flush"] = max(nrep.values())
        names = names + ["_flush"]
    out = {nm: [] for nm in names}
    clocks = []
    for _ in range(rounds):
        order = names[:]
        rng.shuffle(order)
        for nm in order:
            fn, n = fns[nm], nrep[nm]
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(n):
                if flush is not None:
                    flush()
                fn()
            b.record()
            torch.cuda.synchronize()
            out[nm].append(a.elapsed_time(b) / n)
        clocks.append(_clock())
    res = {}
    fl = out.pop("_flush") if flush is not None else None    # per round, same order as every other method's list
    for nm, ts in out.items():
        ts = sorted(ts) if fl is None else sorted(t - f for t, f in zip(ts, fl))
        raw = out[nm] if fl is None else [t - f for t, f in zip(out[nm], fl)]
        res[nm] = dict(ms=ts[len(ts) // 2], spread=ts[-1] / max(ts[0], 1e-9), clocks=clocks, rounds=raw)
    return res


def graph_of(fn):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    return gr.replay


CFG_GRID = [dict(BLOCK_N=bn, warps=w, stages=st) for bn in (32, 64, 128) for w in (2, 4, 8) for st in (1, 2, 3)]
L2_BYTES = 64 << 20


def _tune(make, B, L, K):
    """best base config for one shape; each config is built, timed and freed in turn (on a tuning seed, not the reported
    one); configs that fail to compile or run out of resources are logged, not silently dropped."""
    t, skipped = {}, []
    for i, c in enumerate(CFG_GRID):
        try:
            st = make(c)
            st.base()
            torch.cuda.synchronize()
            t[i] = time_group({i: graph_of(st.base)}, rounds=3, min_ms=20.0)[i]["ms"]
        except Exception as e:   # OutOfResources / compilation failure for this config
            skipped.append((c, type(e).__name__))
        st = None
        torch.cuda.empty_cache()
    if not t:
        raise RuntimeError(f"no config ran for B={B} L={L}: {skipped}")
    return CFG_GRID[min(t, key=t.get)], skipped


def _copies(per_layer_bytes):
    """number of layer copies to rotate so that one timed call streams > 2x L2 (cold-cache timing); at most 32."""
    return int(min(32, max(1, -(-2 * L2_BYTES // max(1, per_layer_bytes)))))


def _sdpa_graph(q16, k16, v16):
    """SDPA pinned to the flash backend (falls back to efficient; the backend used is returned)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel(be):
                f = lambda: torch.nn.functional.scaled_dot_product_attention(q16, k16, v16, enable_gqa=True)  # noqa: E731
                return graph_of(f), str(be)
        except Exception:
            continue
    raise RuntimeError("no fused SDPA backend")


def run(B, L, K, tune, extra, modes, cold=True):
    """one shape. Methods (all CUDA graphs, timed interleaved by time_group): v4 base and whole step per rotary mode,
    the bias pre-pass alone, v3 whole step re-timed in the same process (tuned on the same grid when --tune), PyTorch
    SDPA fp16 (flash backend), and an empty graph as the replay floor. With cold=True each timed call runs n copies of the
    layer (n chosen so the bytes streamed exceed 2x L2) and the time is divided by n."""
    import kbench3 as V3
    res = {"B": B, "L": L, "K": K, "extra": extra}
    BK = B * KVH
    kv2 = 2 * BK * L * (P + 4)
    n = _copies(kv2) if cold else 1
    res["layer_copies"] = n
    host = (torch.randn(BK, L, D).half().pin_memory(), torch.randn(BK, L, D).half().pin_memory())
    fns, keep = {}, []

    def rot(steps, fn):
        return graph_of(lambda: [fn(x) for x in steps])

    for rc in modes:
        cfg = dict(BLOCK_N=64, warps=4, stages=2)
        if tune:
            cfg, sk = _tune(lambda c: Step(B, L, K, rc, seed=12345, host=host, **c, **extra), B, L, K)
            res[f"v4_rc{rc}_skipped"] = sk
        res[f"v4_rc{rc}_cfg"] = cfg
        steps = [Step(B, L, K, rc, seed=i, host=host, **cfg, **extra) for i in range(n)]
        for st in steps:
            st.step()
        torch.cuda.synchronize()
        res[f"v4_rc{rc}_nsplit"] = steps[0].nsplit
        keep += steps
        fns[f"v4_rc{rc}_base"] = rot(steps, lambda x: x.base())
        fns[f"v4_rc{rc}_step"] = rot(steps, lambda x: x.step())
        if rc == 2:
            fns["v4_rc2_prep"] = rot(steps, lambda x: x.prep())
    for rc in (False, True):
        cfg = dict(BLOCK_N=64, warps=4, stages=2)
        if tune:
            cfg, sk = _tune(lambda c: V3.Step(B, L, K, rc, seed=12345, **c), B, L, K)
            res[f"v3_rc{int(rc)}_skipped"] = sk
        res[f"v3_rc{int(rc)}_cfg"] = cfg
        steps = [V3.Step(B, L, K, rc, seed=i, **cfg) for i in range(min(n, 4))]   # v3 allocates its own host copies
        for st in steps:
            st.step()
        torch.cuda.synchronize()
        keep += steps
        fns[f"v3_rc{int(rc)}_step"] = rot(steps, lambda x: x.step())
        res[f"v3_rc{int(rc)}_copies"] = len(steps)
    nf = _copies(2 * BK * L * D * 2) if cold else 1
    if B * L * KVH * D * 4 * nf < 12e9:
        q16 = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
        kvs = [(torch.randn(B, KVH, L, D, device="cuda", dtype=torch.float16),
                torch.randn(B, KVH, L, D, device="cuda", dtype=torch.float16)) for _ in range(nf)]
        keep += [q16, kvs]
        from torch.nn.attention import SDPBackend, sdpa_kernel
        g, be = _sdpa_graph(q16, *kvs[0])
        res["sdpa_backend"] = be
        bk_ = SDPBackend.FLASH_ATTENTION if "FLASH" in be else SDPBackend.EFFICIENT_ATTENTION

        def sd():
            with sdpa_kernel(bk_):
                for k_, v_ in kvs:
                    torch.nn.functional.scaled_dot_product_attention(q16, k_, v_, enable_gqa=True)
        fns["fp16_sdpa"] = graph_of(sd)
        res["fp16_copies"] = nf
    else:
        res["fp16_sdpa"] = "skipped (memory)"
    fns["empty_graph"] = graph_of(lambda: None)
    t = time_group(fns)
    div = {k: (n if k.startswith("v4") else res.get("v3_rc0_copies", 1) if k.startswith("v3") else res.get("fp16_copies", 1)
               if k.startswith("fp16") else 1) for k in t}
    for nm, v in t.items():
        res[nm] = v["ms"] / div[nm]
        res[nm + "_spread"] = v["spread"]
        res[nm + "_rounds"] = [x / div[nm] for x in v["rounds"]]
    res["clocks"] = t["empty_graph"]["clocks"]
    res["bytes_per_tok_kvhead"] = {"kv_2bit": 2 * (P + 4), "s_roundtrip": 2 * G * 2, "bias_roundtrip_rc2": 2 * G * 4,
                                   "pcie_exact_per_step_per_kvhead": 2 * K * D * 2}
    keep = fns = None
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--B", default="1,8,32")
    ap.add_argument("--L", default="8192,32768,131072")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--modes", default="0,2")
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--rcp", default="tf32")
    ap.add_argument("--reload", type=int, default=0)
    ap.add_argument("--magic", type=int, default=1)
    ap.add_argument("--bisect", type=int, default=1)
    ap.add_argument("--nprog", type=int, default=0)
    ap.add_argument("--meta", type=int, default=1)
    ap.add_argument("--wide", type=int, default=1)
    ap.add_argument("--bias", default="fp16")
    ap.add_argument("--out", default="kbench4.json")
    a = ap.parse_args()
    extra = dict(rcp=a.rcp, reload=bool(a.reload), magic=bool(a.magic), bisect=bool(a.bisect), nprog=a.nprog, meta=bool(a.meta), wide=bool(a.wide), bias_mode=a.bias)
    if a.check:
        check(**extra)
        return
    out = []
    for B in (int(x) for x in a.B.split(",")):
        for L in (int(x) for x in a.L.split(",")):
            if B * L > 32 * 131072 // 2:
                print(json.dumps({"B": B, "L": L, "skipped": "host memory (pinned exact K/V > cgroup limit)"}), flush=True)
                continue
            t0 = time.time()
            r = run(B, L, a.K, a.tune, extra, [int(x) for x in a.modes.split(",")], cold=not a.hot)
            out.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith("_rounds")}),
                  f"({time.time() - t0:.0f}s)", flush=True)
            with open(a.out, "w") as f:
                json.dump(out, f)


if __name__ == "__main__":
    main()
