# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""kbench7 (rootcause fork): the kbench5 decode step with the main line's water-filled key base and an FP16 window, so the
kernel implements the base whose accuracy rounds SC / P9-P12 measure (wfc_ck_w128_rc). kbench4 / kbench5 are imported
read-only; new are the kernels that decode keys and window_kernel (exact FP16 window).
  key layout   the per-token 32-byte key record keeps its size: after the channel permutation (top half of the RoPE pairs by
               calibration importance first) and one 64-point Hadamard per half, byte j holds the 3-bit code of hi channel j
               (bits 0-2), of hi channel j+32 (bits 3-5) and the signs of lo channels j and j+32 (bits 6, 7): 64 channels at
               3 bits and 64 at 1 bit (contiguous codes: decode is kb & 7, (kb >> 3) & 7)
  metadata     the 8-byte record (hi scale, hi zero, v scale, v zero) as before, plus a 4-byte lo record pre-merged at cache
               write, (A, B) = (mean - s, 2 s) with s the mean absolute deviation (lo value = A + B * bit): +4 bytes per token
  queries      the base score uses q_w = ((q / b)_perm) with one Hadamard per half, rebuilt every step (qprep, inside the
               timed step) by one kernel launch (qprep_kernel, structure borrowed from an earlier fork):
               qw = fp16(q * (1 / b_h)) @ PH[h], PH[h] = Perm_h . blockdiag(H64, H64) (entries 0 / +-1/8, exact in fp16,
               one tensor-core dot), stored fp16, and qs2 = (sum of the hi half, sum of the lo half); exact scores and
               the rotary bias use q itself.
               The timing steps use b = 1 and a random permutation (only shapes matter there); the round-trip check
               encodes real keys with a random balance b and permutation and compares with q . k_hat in the key space
  base score   s = (sc_hi (q_hi . c_hi) + z_hi sum(q_hi) + s_lo (2 (q_lo . bit) - sum(q_lo)) + mu_lo sum(q_lo)) * scale
               (+ bias), the same expression in the base kernel and in both merge kernels (the removed base term must
               match exactly)
  window       with win > 0, W (<= 128) further tokens are FP16 on the GPU (the context is L quantized tokens + W window
               tokens; callers pass the quantized length as L), attended exactly by window_kernel, whose partial (max,
               sum, accumulator) is one more split of the existing reduction; the candidates for selection are quantized tokens
               only, but their probabilities are normalised over quantized + window tokens (the window split enters the
               global max / sum)
Correctness: python kbench7.py --check (torch fp32 reference with the kernel's own selection). Timing: --shapes as kbench5
with the five policy configurations at rc2 (the rc0 cost of centring is measured in TB), plain base vs water-filled
base vs water-filled + window.
Usage: python kbench7.py --check | [--shapes 1x8192,...] [--rounds 15] [--win 128] [--out kbench7.json]"""
import argparse
import hashlib
import json
import math
import random
import statistics as S
import time

import torch
import triton
import triton.language as tl

import kbench4 as K4
import kbench5 as K5
from kbench4 import _planes, _widen

G, GP, P, D, KVH = K4.G, K4.GP, K4.P, K4.D, K4.KVH
WB = 128                                     # window capacity
MD5 = {n: hashlib.md5(open(f, "rb").read()).hexdigest() for n, f in
       (("kbench7.py", __file__), ("kbench5.py", K5.__file__), ("kbench4.py", K4.__file__))}


@triton.jit
def _lo_pair(ptr, idx, mask, M8: tl.constexpr):
    """the per-token lo record (A, B) of the tokens idx (1-D), fp32. M8 = 0: fp16 pairs read as a [N, 2] tile; 1: two
    float8 e4m3 bytes; 2: two int8 codes in units of the 3-bit half's step / 16 (the caller multiplies by ksc / 16);
    r21: 2 reads each token's two int8 bytes as ONE int16 (a coalesced 1-D load, split by shifts; the [N, 2] byte tile of
    r20 cost +26% at 1 x 32K) and 3 reads the fp16 pair as one int32 (same values as 0)."""
    if M8 == 2:
        x = tl.load(ptr.to(tl.pointer_type(tl.int16)) + idx, mask=mask, other=0).to(tl.int32)
        return ((x << 24) >> 24).to(tl.float32), (x >> 8).to(tl.float32)        # little endian: A low byte, B high
    elif M8 == 3:
        x = tl.load(ptr.to(tl.pointer_type(tl.int32)) + idx, mask=mask, other=0)
        a = (x & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        b = ((x >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        return a, b
    elif M8 == 1:
        m2 = tl.load(ptr + idx[:, None] * 2 + tl.arange(0, 2)[None, :], mask=mask[:, None], other=0)
        return tl.split(m2.to(tl.float8e4nv, bitcast=True).to(tl.float32))
    else:
        m2 = tl.load(ptr + idx[:, None] * 2 + tl.arange(0, 2)[None, :], mask=mask[:, None], other=0.0)
        return tl.split(m2.to(tl.float32))


@triton.jit
def _wf_q(qb, gm, P: tl.constexpr):
    """the four water-filled query planes (hi j, hi j + 32, lo j, lo j + 32) of a program's rows, loaded once."""
    qh0 = tl.load(qb, mask=gm[:, None], other=0.0)
    qh1 = tl.load(qb + P, mask=gm[:, None], other=0.0)
    ql0 = tl.load(qb + 2 * P, mask=gm[:, None], other=0.0)
    ql1 = tl.load(qb + 3 * P, mask=gm[:, None], other=0.0)
    return qh0, qh1, ql0, ql1


@triton.jit
def _wf_scores(qh0, qh1, ql0, ql1, kb, ksc, kzp, mu, slo, qsh, qsl, MAGIC: tl.constexpr):
    """water-filled base logits (before the scale and the bias) of the rows whose key bytes are kb [N, P]; codes are
    widened to fp16 by kbench4._widen (exact for small integers)."""
    c0 = _widen(kb & 7, MAGIC)                          # bits 0-2: 3-bit code of hi channel j
    c1 = _widen((kb >> 3) & 7, MAGIC)                   # bits 3-5: 3-bit code of hi channel j + 32
    b0 = _widen((kb >> 6) & 1, MAGIC)
    b1 = _widen((kb >> 7) & 1, MAGIC)
    rh = tl.dot(qh0, tl.trans(c0)) + tl.dot(qh1, tl.trans(c1))
    rl = tl.dot(ql0, tl.trans(b0)) + tl.dot(ql1, tl.trans(b1))
    # lo record pre-merged at cache write: mu = A = mean - s, slo = B = 2 s, so lo value = A + B * bit and the lo term is
    # B (q_lo . bit) + A sum(q_lo): two multiply-adds instead of four operations
    return (rh * ksc[None, :] + qsh[:, None] * kzp[None, :] + slo[None, :] * rl + mu[None, :] * qsl[:, None])


@triton.jit
def _qprep_body(bk, q_ptr, ib_ptr, t_ptr, perm_ptr, qt_out, qts_out, GH: tl.constexpr, GP: tl.constexpr, D4: tl.constexpr,
                KV: tl.constexpr, CB: tl.constexpr):
    """qprep_kernel's work for one (batch, KV head) bk. r29: PH[h] = Perm_h . blockdiag(H64, H64) is applied as what it is
    - a gather of q in permuted order (perm_ptr, int32 [KV, D4]; 1 / b gathered alike, ib_ptr = 1 / b permuted) and one
    64 x 64 Hadamard dot per half (t_ptr = H64, fp16, entries +-1/8, shared by all heads) - instead of a dense 128 x 128
    dot with PH[h]: a quarter of the tensor-core work and of the matrix bytes. Same roundings (q / b to fp16, fp16 qw,
    fp32 half sums of the fp16 qw); the fp32 accumulation runs over the 64 nonzero terms only."""
    h = bk % KV
    og = tl.arange(0, GP)
    gm = og < GH
    oj = tl.arange(0, D4 // 2)
    hm = tl.load(t_ptr + oj[:, None] * (D4 // 2) + oj[None, :])                     # H64, fp16, exact
    for half in tl.static_range(2):
        pj = tl.load(perm_ptr + h * D4 + half * (D4 // 2) + oj)
        q = tl.load(q_ptr + (bk * GH + og[:, None]) * D4 + pj[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
        ib = tl.load(ib_ptr + h * D4 + half * (D4 // 2) + oj)
        qb = (q * ib[None, :]).to(tl.float16)
        qt = tl.dot(qb, hm).to(tl.float16)
        tl.store(qt_out + (bk * GH + og[:, None]) * D4 + half * (D4 // 2) + oj[None, :], qt, mask=gm[:, None])
        tl.store(qts_out + (bk * GH + og) * 2 + half, tl.sum(qt.to(tl.float32), 1), mask=gm)


@triton.jit
def _qprep_half(bk, q_ptr, ib_ptr, t_ptr, qt_out, qts_out, GH: tl.constexpr, GP: tl.constexpr, D4: tl.constexpr,
                KV: tl.constexpr, CB: tl.constexpr, HALF: tl.constexpr):
    """r26: one half of _qprep_body (HALF 0: the hi columns 0 .. D4/2 - 1 and their sum; HALF 1: the lo columns and theirs),
    with the same blocks, dots, roundings and summation order, so two programs reproduce _qprep_body bitwise with half
    the serial chain each."""
    h = bk % KV
    og = tl.arange(0, GP)
    gm = og < GH
    od = tl.arange(0, D4)
    q = tl.load(q_ptr + (bk * GH + og[:, None]) * D4 + od[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
    ib = tl.load(ib_ptr + h * D4 + od)
    qb = (q * ib[None, :]).to(tl.float16)
    acc = tl.zeros([GP], tl.float32)
    for cb in tl.static_range(HALF * (D4 // (2 * CB)), (HALF + 1) * (D4 // (2 * CB))):
        oc = cb * CB + tl.arange(0, CB)
        t = tl.load(t_ptr + h * D4 * D4 + od[:, None] * D4 + oc[None, :])
        qt = tl.dot(qb, t).to(tl.float16)
        tl.store(qt_out + (bk * GH + og[:, None]) * D4 + oc[None, :], qt, mask=gm[:, None])
        acc += tl.sum(qt.to(tl.float32), 1)
    tl.store(qts_out + (bk * GH + og) * 2 + HALF, acc, mask=gm)


@triton.jit
def qprep_kernel(q_ptr, ib_ptr, t_ptr, perm_ptr, qt_out, qts_out, GH: tl.constexpr, GP: tl.constexpr, D4: tl.constexpr,
                 KV: tl.constexpr, CB: tl.constexpr):
    """(structure borrowed from an earlier fork's kbench7 qprep_kernel, md5 761db3ac; precision split changed here) one
    program per (batch, KV head): qw = fp16(q / b_h) @ PH[h] with PH[h] = Perm_h . blockdiag(H64, H64), whose entries
    are 0 / +-1/8 and exact in fp16, so the product is one fp16 tensor-core dot with fp32 accumulation; roundings:
    q / b to fp16, and the fp16 store of qw; qw stored fp16; the half sums of the fp16-rounded qw in fp32."""
    _qprep_body(tl.program_id(0), q_ptr, ib_ptr, t_ptr, perm_ptr, qt_out, qts_out, GH, GP, D4, KV, CB)


@triton.jit
def _uw_body(bk, q_ptr, mp_ptr, u_out, w_out, GH: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, KV: tl.constexpr):
    """r28: the per-step query-side preprocessing every rotary-bias (RC 2) step needs and kbench4 does once on the host
    (Step.__init__): for the rows of (batch, KV head) bk, ca = qa ma + qb mb, cb = qb ma - qa mb (q split in halves,
    m_pre of KV head bk % KV), u = [ca | cb], w = [cb | -ca], fp32."""
    h = bk % KV
    og = tl.arange(0, GP)
    gm = og < GH
    oh = tl.arange(0, D // 2)
    qa = tl.load(q_ptr + (bk * GH + og[:, None]) * D + oh[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
    qb = tl.load(q_ptr + (bk * GH + og[:, None]) * D + D // 2 + oh[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
    ma = tl.load(mp_ptr + h * D + oh)[None, :]
    mb = tl.load(mp_ptr + h * D + D // 2 + oh)[None, :]
    ca = qa * ma + qb * mb
    cb = qb * ma - qa * mb
    r = (bk * GH + og[:, None]) * D
    tl.store(u_out + r + oh[None, :], ca, mask=gm[:, None])
    tl.store(u_out + r + D // 2 + oh[None, :], cb, mask=gm[:, None])
    tl.store(w_out + r + oh[None, :], cb, mask=gm[:, None])
    tl.store(w_out + r + D // 2 + oh[None, :], -ca, mask=gm[:, None])


@triton.jit
def qside_kernel(q_ptr, mp_ptr, u_out, w_out, ib_ptr, t_ptr, perm_ptr, qt_out, qts_out, GH: tl.constexpr, GP: tl.constexpr,
                 D: tl.constexpr, KV: tl.constexpr, CB: tl.constexpr, WF: tl.constexpr):
    """r28: one launch for the per-step query-side preprocessing of both steps. Program (bk, 0): u, w (_uw_body), which
    the plain and the water-filled steps both need; WF = 1 adds program (bk, 1) with the water-filled query transform
    (_qprep_body), so the transform costs no launch of its own (grid (BK, 1) for plain, (BK, 2) for water-filled)."""
    bk = tl.program_id(0)
    if tl.program_id(1) == 0:
        _uw_body(bk, q_ptr, mp_ptr, u_out, w_out, GH, GP, D, KV)
    else:
        if WF:
            _qprep_body(bk, q_ptr, ib_ptr, t_ptr, perm_ptr, qt_out, qts_out, GH, GP, D, KV, CB)


@triton.jit
def bias_q_kernel(u_ptr, w_ptr, c0_ptr, s0_ptr, cs_ptr, bias_out, L, ntile, npb, sm_scale, q_ptr, ib_ptr, t_ptr, perm_ptr, qt_out,
                  qts_out, GH: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BB: tl.constexpr, KV: tl.constexpr,
                  CB: tl.constexpr):
    """r23: kbench4.bias_kernel (MODE fp16) and qprep_kernel in ONE launch: per (batch, KV head) row, programs
    1 .. npb walk the bias tiles exactly as bias_kernel does with npb programs (as pid - 1), and program 0 does qprep_kernel's
    query transform (same arithmetic, so qw / qs2 / bias are bitwise those of the two separate launches). Saves the
    query transform's own launch (3.2 us of the 4.5 us water-filling overhead at 1 x 32K on the RTX PRO 5000)."""
    bk = tl.program_id(0)
    pid = tl.program_id(1)
    # r24: the query-transform program is grid row 0 (dispatched first, so its ~3 us overlaps the bias programs instead
    # of trailing them: r23 put it in the last row and the launch took 15.4 us against bias_kernel's 13.4)
    # r26: the transform is split over grid rows 0 and 1 (hi / lo half), bias programs are rows 2 .. npb + 1
    if pid == 0:
        _qprep_body(bk, q_ptr, ib_ptr, t_ptr, perm_ptr, qt_out, qts_out, GH, GP, D, KV, CB)
    else:
        pid = pid - 1
        og = tl.arange(0, GP)
        od = tl.arange(0, D)
        on = tl.arange(0, BB)
        gm = og < GH
        u = tl.load(u_ptr + (bk * GH + og[:, None]) * D + od[None, :], mask=gm[:, None], other=0.0)
        w = tl.load(w_ptr + (bk * GH + og[:, None]) * D + od[None, :], mask=gm[:, None], other=0.0)
        cs16 = tl.load(cs_ptr + od[:, None] * BB + on[None, :]).to(tl.float16)
        for tt in range(pid, ntile, npb):
            c0 = tl.load(c0_ptr + tt * D + od)
            s0 = tl.load(s0_ptr + tt * D + od)
            ab = u * c0[None, :] + w * s0[None, :]
            b = tl.dot(ab.to(tl.float16), cs16)
            offs = tt * BB + on
            tl.store(bias_out + (bk * GH + og[:, None]) * L + offs[None, :], b * sm_scale,
                     mask=gm[:, None] & (offs < L)[None, :])


@triton.jit
def _rcbias(u_ptr, w_ptr, c0_ptr, s0_ptr, csb_ptr, bk, tok, og, gm, GH: tl.constexpr, GP: tl.constexpr, NT: tl.constexpr,
            D4: tl.constexpr, BB: tl.constexpr, DC: tl.constexpr):
    """(borrowed from an earlier fork's kbench7, md5 761db3ac, reviewed here) unscaled rotary bias of NT tokens at any
    positions with the operands of kbench4.bias_kernel MODE fp16: b = sum_d fp16(u_d c0[t // BB, d] + w_d s0[t // BB, d])
    fp16(csB[d, t % BB]), in fp32 over chunks of DC channels (fp16 x fp16 products are exact in fp32, so this differs from
    the tensor-core dot of the base kernel only in summation order)."""
    tt = tok // BB
    nn = tok % BB
    b = tl.zeros([GP, NT], tl.float32)
    for dc in tl.static_range(D4 // DC):
        od = dc * DC + tl.arange(0, DC)
        u = tl.load(u_ptr + (bk * GH + og[:, None]) * D4 + od[None, :], mask=gm[:, None], other=0.0)
        w = tl.load(w_ptr + (bk * GH + og[:, None]) * D4 + od[None, :], mask=gm[:, None], other=0.0)
        c0 = tl.load(c0_ptr + tt[:, None] * D4 + od[None, :])
        s0 = tl.load(s0_ptr + tt[:, None] * D4 + od[None, :])
        cs = tl.load(csb_ptr + od[None, :] * BB + nn[:, None]).to(tl.float16).to(tl.float32)
        ab = u[:, None, :] * c0[None, :, :] + w[:, None, :] * s0[None, :, :]
        b += tl.sum(ab.to(tl.float16).to(tl.float32) * cs[None, :, :], 2)
    return b


@triton.jit
def _base_body(bk, sp, og, op, gm, qh0, qh1, ql0, ql1, qsh, qsl, kp_ptr, vp_ptr, meta_ptr, meta2_ptr, bias_ptr, u_ptr, w_ptr,
               c0_ptr, s0_ptr, csb_ptr, m_out, l_out, acc_out, s_out, L, split_len, nsp, sm_scale, GH: tl.constexpr,
               GP: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, RC: tl.constexpr, RELOAD: tl.constexpr,
               MAGIC: tl.constexpr, M8: tl.constexpr = False):
    """the quantized-split work of base_kernel_wf (one split sp of one batch * KV head), given the query planes."""
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
    if RC == 3:
        od3 = tl.arange(0, 4 * P)
        if not RELOAD:
            u = tl.load(u_ptr + (bk * GH + og[:, None]) * (4 * P) + od3[None, :], mask=gm[:, None], other=0.0)
            w = tl.load(w_ptr + (bk * GH + og[:, None]) * (4 * P) + od3[None, :], mask=gm[:, None], other=0.0)
            cs = tl.load(csb_ptr + od3[:, None] * BLOCK_N + on[None, :]).to(tl.float16)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + on
        nm = offs_n < end
        base = bk * L + offs_n
        kb = tl.load(kp_ptr + base[:, None] * P + op[None, :], mask=nm[:, None], other=0)
        mt = tl.load(meta_ptr + base[:, None] * 4 + tl.arange(0, 4)[None, :], mask=nm[:, None], other=0.0).to(tl.float32)
        sc, zp = tl.split(tl.reshape(mt, [BLOCK_N, 2, 2]))
        ksc, vsc = tl.split(sc)
        kzp, vzp = tl.split(zp)
        mu, slo = _lo_pair(meta2_ptr, base, nm, M8)
        if M8 == 2:
            u = ksc * 0.0625
            mu, slo = mu * u, slo * u
        s = _wf_scores(qh0, qh1, ql0, ql1, kb, ksc, kzp, mu, slo, qsh, qsl, MAGIC) * sm_scale
        if RC == 2:
            s += tl.load(bias_ptr + (bk * GH + og[:, None]) * L + offs_n[None, :], mask=gm[:, None] & nm[None, :], other=0.0)
        if RC == 3:
            if RELOAD:
                u = tl.load(u_ptr + (bk * GH + og[:, None]) * (4 * P) + od3[None, :], mask=gm[:, None], other=0.0)
                w = tl.load(w_ptr + (bk * GH + og[:, None]) * (4 * P) + od3[None, :], mask=gm[:, None], other=0.0)
                cs = tl.load(csb_ptr + od3[:, None] * BLOCK_N + on[None, :]).to(tl.float16)
            tt = n0 // BLOCK_N
            c0 = tl.load(c0_ptr + tt * (4 * P) + od3)
            s0 = tl.load(s0_ptr + tt * (4 * P) + od3)
            ab = u * c0[None, :] + w * s0[None, :]
            s += tl.dot(ab.to(tl.float16), cs) * sm_scale
        s = tl.where(nm[None, :], s, -float("inf"))
        tl.store(s_out + (bk * GH + og[:, None]) * L + offs_n[None, :], s.to(tl.float16), mask=gm[:, None] & nm[None, :])
        mn = tl.maximum(m_i, tl.max(s, 1))
        al = tl.exp(m_i - mn)
        p = tl.exp(s - mn[:, None])
        l_i = l_i * al + tl.sum(p, 1)
        vb = tl.load(vp_ptr + base[:, None] * P + op[None, :], mask=nm[:, None], other=0)
        pv = (p * vsc[None, :] * 4096.0).to(tl.float16)
        a0 = a0 * al[:, None] + tl.dot(pv, _widen(vb & 3, MAGIC))
        a1 = a1 * al[:, None] + tl.dot(pv, _widen((vb >> 2) & 3, MAGIC))
        a2 = a2 * al[:, None] + tl.dot(pv, _widen((vb >> 4) & 3, MAGIC))
        a3 = a3 * al[:, None] + tl.dot(pv, _widen((vb >> 6) & 3, MAGIC))
        az = az * al + tl.sum(p * vzp[None, :], 1)
        m_i = mn
    o = (bk * nsp + sp) * GP + og
    tl.store(m_out + o, m_i)
    tl.store(l_out + o, l_i)
    ao = acc_out + o[:, None] * (4 * P) + op[None, :]
    tl.store(ao, a0 / 4096.0 + az[:, None], mask=gm[:, None])
    tl.store(ao + P, a1 / 4096.0 + az[:, None], mask=gm[:, None])
    tl.store(ao + 2 * P, a2 / 4096.0 + az[:, None], mask=gm[:, None])
    tl.store(ao + 3 * P, a3 / 4096.0 + az[:, None], mask=gm[:, None])




@triton.jit
def base_kernel_wf(qw_ptr, qs2_ptr, q_ptr, ib_ptr, t_ptr, perm_ptr, kw_ptr, vw_ptr, W, nsb, kp_ptr, vp_ptr, meta_ptr, meta2_ptr,
                   bias_ptr, u_ptr, w_ptr, c0_ptr, s0_ptr, csb_ptr, m_out, l_out, acc_out, s_out, L, split_len, nsp,
                   sm_scale, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, RC: tl.constexpr,
                   RELOAD: tl.constexpr, MAGIC: tl.constexpr, FQ: tl.constexpr, WIN: tl.constexpr, KV: tl.constexpr,
                   WB: tl.constexpr, M8: tl.constexpr = False, WF1: tl.constexpr = False):
    """kbench4.base_kernel (META path, no hi / lo) with water-filled key decoding; partials are written with a split stride
    nsp. Rotary modes: 0 none; 2 read the pre-pass bias; 3 compute the bias in the tile exactly as kbench4.bias_kernel
    MODE fp16 does (borrowed from an earlier fork; needs BLOCK_N == BB and split_len a multiple of BLOCK_N, so tile
    n0 // BLOCK_N is the bias table row; RELOAD re-reads u, w and the [cos; sin] table per tile).
    Fusion (one launch per layer step): FQ = 1 computes the query planes in every program with exactly qprep_kernel's
    arithmetic (fp16(q / b) @ PH[h], fp16 store, fp32 half sums; program sp == 0 also writes qw / qs2 for the merge
    kernels); WIN = 1 adds one program per batch * KV head (sp == nsb, or sp == 0 with WF1) that does window_kernel's
    exact window split."""
    bk = tl.program_id(0)
    sp = tl.program_id(1)
    og = tl.arange(0, GP)
    op = tl.arange(0, P)
    gm = og < GH
    if WIN:      # compile-time: without a window the window code is not compiled into the kernel
        # WF1: the window program is launched first (grid row 0, quantized split sp - 1 in row sp) so that its work is not
        # left to the tail of the last wave; its partial keeps slot nsb either way
        wrow = 0 if WF1 else nsb
        if sp == wrow:
            _window_body(bk, q_ptr, kw_ptr, vw_ptr, m_out, l_out, acc_out, W, nsb, nsp, sm_scale, GH, GP, 4 * P, WB)
        else:
            if WF1:
                sp = sp - 1
            _base_prog(bk, sp, og, op, gm, qw_ptr, qs2_ptr, q_ptr, ib_ptr, t_ptr, perm_ptr, kp_ptr, vp_ptr, meta_ptr, meta2_ptr,
                       bias_ptr, u_ptr, w_ptr, c0_ptr, s0_ptr, csb_ptr, m_out, l_out, acc_out, s_out, L, split_len, nsp,
                       sm_scale, GH, GP, P, BLOCK_N, RC, RELOAD, MAGIC, FQ, KV, M8)
    else:
        _base_prog(bk, sp, og, op, gm, qw_ptr, qs2_ptr, q_ptr, ib_ptr, t_ptr, perm_ptr, kp_ptr, vp_ptr, meta_ptr, meta2_ptr, bias_ptr,
                   u_ptr, w_ptr, c0_ptr, s0_ptr, csb_ptr, m_out, l_out, acc_out, s_out, L, split_len, nsp, sm_scale, GH,
                   GP, P, BLOCK_N, RC, RELOAD, MAGIC, FQ, KV, M8)


@triton.jit
def _base_prog(bk, sp, og, op, gm, qw_ptr, qs2_ptr, q_ptr, ib_ptr, t_ptr, perm_ptr, kp_ptr, vp_ptr, meta_ptr, meta2_ptr, bias_ptr, u_ptr,
               w_ptr, c0_ptr, s0_ptr, csb_ptr, m_out, l_out, acc_out, s_out, L, split_len, nsp, sm_scale, GH: tl.constexpr,
               GP: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, RC: tl.constexpr, RELOAD: tl.constexpr,
               MAGIC: tl.constexpr, FQ: tl.constexpr, KV: tl.constexpr, M8: tl.constexpr = False):
    """one quantized split: the query planes (computed when FQ, else loaded) and _base_body. r30: FQ == 2 computes them
    in the r29 gather form of _qprep_body (q gathered in permuted order times permuted 1 / b, fp16, one 64 x 64 Hadamard per
    half split into its two 32-column planes; ib_ptr = permuted 1 / b, t_ptr = H64 fp16, perm_ptr = int32 permutation), so
    the transform costs each program two small dots instead of a separate launch or the dense 128 x 128 product."""
    if FQ == 2:
        h = bk % KV
        oj = tl.arange(0, 2 * P)
        hb = t_ptr + oj[:, None] * (2 * P)
        hm0 = tl.load(hb + op[None, :])                    # H64 columns 0 .. P - 1
        hm1 = tl.load(hb + P + op[None, :])                # H64 columns P .. 2P - 1
        pj = tl.load(perm_ptr + h * (4 * P) + oj)
        qg = tl.load(q_ptr + (bk * GH + og[:, None]) * (4 * P) + pj[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
        ibv = tl.load(ib_ptr + h * (4 * P) + oj)
        qbh = (qg * ibv[None, :]).to(tl.float16)
        qh0 = tl.dot(qbh, hm0).to(tl.float16)
        qh1 = tl.dot(qbh, hm1).to(tl.float16)
        pj = tl.load(perm_ptr + h * (4 * P) + 2 * P + oj)
        qg = tl.load(q_ptr + (bk * GH + og[:, None]) * (4 * P) + pj[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
        ibv = tl.load(ib_ptr + h * (4 * P) + 2 * P + oj)
        qbl = (qg * ibv[None, :]).to(tl.float16)
        ql0 = tl.dot(qbl, hm0).to(tl.float16)
        ql1 = tl.dot(qbl, hm1).to(tl.float16)
        qsh = tl.sum(qh0.to(tl.float32), 1) + tl.sum(qh1.to(tl.float32), 1)
        qsl = tl.sum(ql0.to(tl.float32), 1) + tl.sum(ql1.to(tl.float32), 1)
        if sp == 0:
            qo = qw_ptr + (bk * GH + og[:, None]) * (4 * P) + op[None, :]
            tl.store(qo, qh0, mask=gm[:, None])
            tl.store(qo + P, qh1, mask=gm[:, None])
            tl.store(qo + 2 * P, ql0, mask=gm[:, None])
            tl.store(qo + 3 * P, ql1, mask=gm[:, None])
            tl.store(qs2_ptr + (bk * GH + og) * 2, qsh, mask=gm)
            tl.store(qs2_ptr + (bk * GH + og) * 2 + 1, qsl, mask=gm)
    elif FQ:
        od4 = tl.arange(0, 4 * P)
        h = bk % KV
        qr = tl.load(q_ptr + (bk * GH + og[:, None]) * (4 * P) + od4[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
        ib = tl.load(ib_ptr + h * (4 * P) + od4)
        qb16 = (qr * ib[None, :]).to(tl.float16)
        tb = t_ptr + h * (4 * P) * (4 * P) + od4[:, None] * (4 * P)
        qh0 = tl.dot(qb16, tl.load(tb + op[None, :])).to(tl.float16)
        qh1 = tl.dot(qb16, tl.load(tb + P + op[None, :])).to(tl.float16)
        ql0 = tl.dot(qb16, tl.load(tb + 2 * P + op[None, :])).to(tl.float16)
        ql1 = tl.dot(qb16, tl.load(tb + 3 * P + op[None, :])).to(tl.float16)
        qsh = tl.sum(qh0.to(tl.float32), 1) + tl.sum(qh1.to(tl.float32), 1)
        qsl = tl.sum(ql0.to(tl.float32), 1) + tl.sum(ql1.to(tl.float32), 1)
        if sp == 0:
            qo = qw_ptr + (bk * GH + og[:, None]) * (4 * P) + op[None, :]
            tl.store(qo, qh0, mask=gm[:, None])
            tl.store(qo + P, qh1, mask=gm[:, None])
            tl.store(qo + 2 * P, ql0, mask=gm[:, None])
            tl.store(qo + 3 * P, ql1, mask=gm[:, None])
            tl.store(qs2_ptr + (bk * GH + og) * 2, qsh, mask=gm)
            tl.store(qs2_ptr + (bk * GH + og) * 2 + 1, qsl, mask=gm)
    else:
        qb = qw_ptr + (bk * GH + og[:, None]) * (4 * P) + op[None, :]
        qh0, qh1, ql0, ql1 = _wf_q(qb, gm, P)
        qsh = tl.load(qs2_ptr + (bk * GH + og) * 2, mask=gm, other=0.0)
        qsl = tl.load(qs2_ptr + (bk * GH + og) * 2 + 1, mask=gm, other=0.0)
    _base_body(bk, sp, og, op, gm, qh0, qh1, ql0, ql1, qsh, qsl, kp_ptr, vp_ptr, meta_ptr, meta2_ptr, bias_ptr, u_ptr,
               w_ptr, c0_ptr, s0_ptr, csb_ptr, m_out, l_out, acc_out, s_out, L, split_len, nsp, sm_scale, GH, GP, P,
               BLOCK_N, RC, RELOAD, MAGIC, M8)


@triton.jit
def _window_body(bk, q_ptr, kw_ptr, vw_ptr, m_out, l_out, acc_out, W, wsp, nsp, sm_scale, GH: tl.constexpr,
                 GP: tl.constexpr, D: tl.constexpr, WB: tl.constexpr, WC: tl.constexpr = 32):
    """exact attention over the W (<= WB) FP16 window tokens of one batch * KV head; written as split wsp of nsp.
    r18: the window is streamed in WC-token chunks with an online softmax, so its key / value tiles are WC x D instead
    of WB x D; when the window is fused into base_kernel_wf this keeps the kernel's shared memory (the maximum over both
    branches) at the quantized body's size and lets more programs reside per SM (the WB x D tiles took 36 KiB and
    limited the fused kernel to 2 programs per SM)."""
    og = tl.arange(0, GP)
    od = tl.arange(0, D)
    oc = tl.arange(0, WC)
    gm = og < GH
    q = tl.load(q_ptr + (bk * GH + og[:, None]) * D + od[None, :], mask=gm[:, None], other=0.0)
    m_i = tl.full([GP], -float("inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    for w0 in tl.range(0, W, WC, num_stages=1):   # r19: not multi-buffered (at most WB / WC = 4 trips)
        ow = w0 + oc
        wm = ow < W
        k = tl.load(kw_ptr + (bk * WB + ow[:, None]) * D + od[None, :], mask=wm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(wm[None, :], s, -float("inf"))
        mn = tl.maximum(m_i, tl.max(s, 1))
        al = tl.exp(m_i - mn)
        p = tl.where(wm[None, :], tl.exp(s - mn[:, None]), 0.0)
        l_i = l_i * al + tl.sum(p, 1)
        v = tl.load(vw_ptr + (bk * WB + ow[:, None]) * D + od[None, :], mask=wm[:, None], other=0.0)
        acc = acc * al[:, None] + tl.dot((p * 4096.0).to(tl.float16), v)   # same fp16 subnormal guard as the base
        m_i = mn
    o = (bk * nsp + wsp) * GP + og
    tl.store(m_out + o, m_i)                                       # W == 0: an empty split (m = -inf, l = 0, acc = 0)
    tl.store(l_out + o, l_i)
    tl.store(acc_out + o[:, None] * D + od[None, :], acc / 4096.0, mask=gm[:, None])



@triton.jit
def window_kernel(q_ptr, kw_ptr, vw_ptr, m_out, l_out, acc_out, W, wsp, nsp, sm_scale, GH: tl.constexpr, GP: tl.constexpr,
                  D: tl.constexpr, WB: tl.constexpr):
    """the window as its own launch (unfused path)."""
    _window_body(tl.program_id(0), q_ptr, kw_ptr, vw_ptr, m_out, l_out, acc_out, W, wsp, nsp, sm_scale, GH, GP, D, WB)


@triton.jit
def fixA_kernel_wf(q_ptr, qw_ptr, qs2_ptr, m_in, idx_ptr, kex_ptr, vex_ptr, kp_ptr, vp_ptr, meta_ptr, meta2_ptr, bias_ptr,
                   u_ptr, w_ptr, c0_ptr, s0_ptr, csb_ptr, pn_out, pz_out, pm_out, L, nsplit, sm_scale, GH: tl.constexpr,
                   GP: tl.constexpr, P: tl.constexpr, K: tl.constexpr, KC: tl.constexpr, NS: tl.constexpr, RC: tl.constexpr,
                   BB: tl.constexpr, M8: tl.constexpr = False):
    """kbench4.fixA_kernel (no cross-step cache, no hi / lo) with the water-filled base scores."""
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
    kx = tl.load(kex_ptr + row[:, None] * D4 + od[None, :])
    vx = tl.load(vex_ptr + row[:, None] * D4 + od[None, :])
    se = tl.dot(qf, tl.trans(kx)) * sm_scale
    mt = tl.load(meta_ptr + row[:, None] * 4 + tl.arange(0, 4)[None, :]).to(tl.float32)
    sc_, zp_ = tl.split(tl.reshape(mt, [KC, 2, 2]))
    ksr, vsc = tl.split(sc_)
    kzr, vzp = tl.split(zp_)
    mu, slo = _lo_pair(meta2_ptr, row, row >= 0, M8)
    if M8 == 2:
        mu, slo = mu * (ksr * 0.0625), slo * (ksr * 0.0625)
    qb = qw_ptr + (bk * GH + og[:, None]) * D4 + op[None, :]
    qsh = tl.load(qs2_ptr + (bk * GH + og) * 2, mask=gm, other=0.0)
    qsl = tl.load(qs2_ptr + (bk * GH + og) * 2 + 1, mask=gm, other=0.0)
    kb = tl.load(kp_ptr + row[:, None] * P + op[None, :])
    qh0, qh1, ql0, ql1 = _wf_q(qb, gm, P)
    sb = _wf_scores(qh0, qh1, ql0, ql1, kb, ksr, kzr, mu, slo, qsh, qsl, True) * sm_scale
    if RC == 2:
        sb += tl.load(bias_ptr + (bk * GH + og[:, None]) * L + tok[None, :], mask=gm[:, None], other=0.0)
    if RC == 3:     # the same bias per selected token (matches the base kernel up to fp32 summation order)
        sb += _rcbias(u_ptr, w_ptr, c0_ptr, s0_ptr, csb_ptr, bk, tok, og, gm, GH, GP, KC, D4, BB, 32) * sm_scale
    Mc = tl.maximum(M, tl.max(se, 1))
    ee = tl.exp(se - Mc[:, None])
    eb = tl.exp(sb - Mc[:, None])
    dz = tl.sum(ee, 1) - tl.sum(eb, 1)
    ebz = tl.sum(eb * vzp[None, :], 1)[:, None]
    vb = tl.load(vp_ptr + row[:, None] * P + op[None, :])
    o = (bk * nch + c) * GP + og
    pb = pn_out + o[:, None] * D4 + op[None, :]
    ebs = (eb * vsc[None, :] * 4096.0).to(tl.float16)
    e0, e1, e2, e3 = _planes(tl.dot((ee * 4096.0).to(tl.float16), vx), GP, P)
    tl.store(pb, (e0 - tl.dot(ebs, (vb & 3).to(tl.float16))) / 4096.0 - ebz, mask=gm[:, None])
    tl.store(pb + P, (e1 - tl.dot(ebs, ((vb >> 2) & 3).to(tl.float16))) / 4096.0 - ebz, mask=gm[:, None])
    tl.store(pb + 2 * P, (e2 - tl.dot(ebs, ((vb >> 4) & 3).to(tl.float16))) / 4096.0 - ebz, mask=gm[:, None])
    tl.store(pb + 3 * P, (e3 - tl.dot(ebs, ((vb >> 6) & 3).to(tl.float16))) / 4096.0 - ebz, mask=gm[:, None])
    tl.store(pz_out + o, dz, mask=gm)
    tl.store(pm_out + o, Mc, mask=gm)


@triton.jit
def fix2A_kernel_wf(q_ptr, qw_ptr, qs2_ptr, idx_ptr, kex_ptr, kp_ptr, meta_ptr, meta2_ptr, bias_ptr, u_ptr, w_ptr, c0_ptr,
                    s0_ptr, csb_ptr, sc_out, L, sm_scale, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr, M: tl.constexpr,
                    KC: tl.constexpr, RC: tl.constexpr, BB: tl.constexpr, M8: tl.constexpr = False):
    """kbench5.fix2A_kernel with the water-filled base scores."""
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
    se = tl.dot(qf, tl.trans(kx)) * sm_scale
    mt = tl.load(meta_ptr + row[:, None] * 4 + tl.arange(0, 4)[None, :]).to(tl.float32)
    sc_, zp_ = tl.split(tl.reshape(mt, [KC, 2, 2]))
    ksr, vsc = tl.split(sc_)
    kzr, vzp = tl.split(zp_)
    mu, slo = _lo_pair(meta2_ptr, row, row >= 0, M8)
    if M8 == 2:
        mu, slo = mu * (ksr * 0.0625), slo * (ksr * 0.0625)
    qb = qw_ptr + (bk * GH + og[:, None]) * D4 + op[None, :]
    qsh = tl.load(qs2_ptr + (bk * GH + og) * 2, mask=gm, other=0.0)
    qsl = tl.load(qs2_ptr + (bk * GH + og) * 2 + 1, mask=gm, other=0.0)
    kb = tl.load(kp_ptr + row[:, None] * P + op[None, :])
    qh0, qh1, ql0, ql1 = _wf_q(qb, gm, P)
    sb = _wf_scores(qh0, qh1, ql0, ql1, kb, ksr, kzr, mu, slo, qsh, qsl, True) * sm_scale
    if RC == 2:
        sb += tl.load(bias_ptr + (bk * GH + og[:, None]) * L + tok[None, :], mask=gm[:, None], other=0.0)
    if RC == 3:     # the same bias per selected token (matches the base kernel up to fp32 summation order)
        sb += _rcbias(u_ptr, w_ptr, c0_ptr, s0_ptr, csb_ptr, bk, tok, og, gm, GH, GP, KC, D4, BB, 32) * sm_scale
    so = sc_out + (bk * 2 * GP + og[:, None]) * M + ok[None, :]
    tl.store(so, se)
    tl.store(so + GP * M, sb)


def hadamard(n, dev):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / n ** 0.5).float().to(dev)


class Step7(K5.Step5):
    """Step5 with the water-filled key base (wf=True) and an optional FP16 window of win tokens (0 = none). rc = 3 is
    centring with the bias computed inside the base and merge kernels (no pre-pass, no bias tensor); the data, tables and
    references are those of rc = 2 (self.rc stays 2, the kernels get self.rck). reload3: the RELOAD variant of rc 3."""

    def __init__(self, B, L, K, rc, wf=True, win=0, reload3=False, fuse=False, fuse_q=None, fuse_w=None, m8=False, wfirst=False, maxnreg=None, qbias=False,
                 bppsm=2, qside=False, **kw):
        assert kw.get("cache", False) is False and kw.get("hilo", False) is False, "production path only"
        assert rc in (0, 2, 3) and (rc != 3 or wf), "rc 3 exists only on the water-filled path"
        assert wf or not m8, "the float8 lo record exists only on the water-filled path"
        super().__init__(B, L, K, 2 if rc == 3 else rc, **kw)
        self.rck, self.reload3 = rc, bool(reload3)
        assert rc != 3 or (self.BLOCK_N == self.BB and self.split_len % self.BLOCK_N == 0), "rc 3 needs tiles = bias tiles"
        self.wf, self.win = wf, int(win)
        # fuse sets both; fuse_q / fuse_w choose the query transform / the window separately
        self.fuse_q = bool(fuse) if fuse_q is None else (2 if fuse_q == "g" else bool(fuse_q))   # r30: "g" gather form
        self.fuse_w = bool(fuse) if fuse_w is None else bool(fuse_w)
        self.wfirst = bool(wfirst)          # fused window program in grid row 0 instead of the last row
        # r22: cap on registers per thread of the water-filled base kernel (Triton maxnreg). The kernel sits at ~166
        # registers, just under the 170 that 3 resident 4-warp programs per SM allow (65536 / (3 * 128)); the 2-byte lo
        # record's decode pushes it to ~180 and occupancy to 2 (+26% at 1 x 32K). None leaves ptxas free.
        self.maxnreg = maxnreg
        self.qbias = bool(qbias)            # r23: query transform folded into the rotary bias pre-pass (one launch)
        self.bppsm = int(bppsm)             # r27: bias pre-pass programs per SM (kbench4.Step.prep fixes 2), both paths
        # r28: per-step query-side preprocessing on the GPU for BOTH paths (u, w from q, which kbench4 precomputes on the
        # host; the water-filled step's query transform rides in the same launch)
        self.qside = bool(qside)
        assert not self.qside or (rc == 2 and not self.fuse_q and not self.qbias)
        assert not self.wfirst or (self.fuse_w and self.win > 0), "wfirst needs the window fused (fuse_w, win > 0)"
        self.kinfo = {}                     # compiled kernels of the last step (registers / spills for the check)
        assert 0 <= self.win <= WB and D == 4 * P
        g = torch.Generator(device="cpu").manual_seed(kw.get("seed", 0) + 777)
        BK = self.BK
        # water-filled query (per KV head: a channel permutation, then a 64-point Hadamard per half) and its half sums
        self.perm = torch.stack([torch.randperm(D, generator=g) for _ in range(KVH)]).cuda()
        self.qbal = torch.ones(KVH, D, device="cuda")                          # balance b (1 for timing; 1 / b applied in qprep)
        self.H64 = hadamard(D // 2, "cuda")
        self.qw = torch.empty(BK * G, D, dtype=torch.float16, device="cuda")
        self.qs2 = torch.empty(BK * G, 2, device="cuda")
        self.Tq = torch.empty(KVH, D, D, device="cuda", dtype=torch.float16)   # Perm . blockdiag(H64, H64), exact
        self.ib = torch.empty(KVH, D, device="cuda")                              # 1 / b per KV head, fp32
        self.build_T()
        self.qprep()
        self.mt2 = torch.stack([(torch.randn(BK, L, generator=g) * 0.02).half(),
                                (torch.rand(BK, L, generator=g) * 0.05 + 0.01).half()], -1).contiguous().cuda()
        # lo record (A, B) in 2 B per token instead of 4: m8 = 1 two float8 e4m3 bytes; m8 = 2 (r20) two int8 codes in
        # units of the 3-bit half's step / 16 (sel11_q format r8, clamped to +-127)
        self.m8 = int(m8)
        assert self.m8 in (0, 1, 2, 3)          # 3 (r21): the fp16 record read as one int32 per token
        self.mt2b = torch.empty(BK, L, 2, dtype=torch.uint8, device="cuda")
        self.sync_lo()
        self.nsb = self.nsplit                                                   # splits of the quantized tokens
        if self.win:
            self.kw = (torch.randn(BK, WB, D, generator=g)).half().cuda()
            self.vw = (torch.randn(BK, WB, D, generator=g)).half().cuda()
            self.nsplit = self.nsb + 1
            self.NS = triton.next_power_of_2(self.nsplit)
            self.m = torch.empty(BK * self.nsplit * GP, device="cuda")
            self.l = torch.empty_like(self.m)
            self.acc = torch.empty(BK * self.nsplit * GP, D, device="cuda")

    def prep(self):
        """kbench4.Step.prep (rotary bias pre-pass, RC = 2) with the grid's programs per SM as a parameter (bppsm; 2 is
        kbench4's choice and gives the identical launch), so the plain and the water-filled steps can be tuned alike."""
        if getattr(self, "qside", False) and not self.wf:
            self.run_qside()
        ntile = triton.cdiv(self.L, self.BB)
        npb = max(1, min(ntile, -(-getattr(self, "bppsm", 2) * self.nsm // self.BK)))
        K4.bias_kernel[(self.BK, npb)](self.u, self.w, self.c0tab, self.s0tab, self.csB, self.bias, self.L, ntile, self.scale,
                                       GH=G, GP=GP, D=D, BB=self.BB, MODE=self.bias_mode, num_warps=4)

    def run_qside(self):
        """r28: u, w (and for the water-filled step qw, qs2) from the current q in one launch."""
        qside_kernel[(self.BK, 2 if self.wf else 1)](self.q, self.mpre, self.u, self.w, self.ibp, self.H64h, self.perm32, self.qw, self.qs2,
                                                    GH=G, GP=GP, D=D, KV=KVH, CB=32, WF=self.wf, num_warps=4)

    def build_T(self):
        """the per-KV-head query transform, split for precision: 1 / b_h (fp32, applied before the fp16 rounding) and
        PH[h] = Perm_h . blockdiag(H64, H64) (Perm_h[c, j] = 1 iff perm_h[j] = c; entries 0 / +-1/8, exact in fp16), so
        (q / b_h) @ PH[h] = qprep_torch; rebuilt whenever perm / qbal change (calibration constants, outside the step)."""
        bd = torch.block_diag(self.H64, self.H64)
        eye = torch.eye(D, device=bd.device)
        for h in range(KVH):
            self.Tq[h].copy_((eye[:, self.perm[h]] @ bd).half())
            self.ib[h].copy_(1.0 / self.qbal[h].float())
        # r29 gather form of the same transform: perm (int32), 1 / b in permuted order, H64 in fp16 (exact)
        self.perm32 = self.perm.to(torch.int32).contiguous()
        self.ibp = torch.gather(self.ib, 1, self.perm.long()).contiguous()
        self.H64h = self.H64.half().contiguous()

    def sync_lo(self):
        """write the 2-byte copy of the lo record from mt2 (the kernels read it when m8)."""
        if self.m8 == 2:
            u = (self.mt[..., 0:1].float() / 16).clamp_min(1e-30)   # a zero fp16 step: codes 0 / +-127, no NaN
            self.mt2b.copy_(torch.clamp(torch.round(self.mt2.float() / u), -127, 127).to(torch.int8).view(torch.uint8))
        else:
            self.mt2b.copy_(self.mt2.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8))

    def lo_eff(self):
        """the lo record as the kernels read it (fp16 values, or the 2-byte rounded ones when m8), fp32."""
        if self.m8 == 2:
            return self.mt2b.view(torch.int8).float() * (self.mt[..., 0:1].float() / 16)
        return self.mt2b.view(torch.float8_e4m3fn).float() if self.m8 == 1 else self.mt2.float()

    def lo_ptr(self):
        return self.mt2b if self.m8 in (1, 2) else self.mt2

    def qprep(self):
        """q_w and its half sums with one kernel launch (part of every timed wf step)."""
        qprep_kernel[(self.BK,)](self.q, self.ibp, self.H64h, self.perm32, self.qw, self.qs2, GH=G, GP=GP, D4=D, KV=KVH, CB=32,
                                 num_warps=4)

    def qprep_torch(self):
        """reference: q_w = ((q / b) permuted) with one Hadamard per half, and its half sums, in torch ops."""
        h = D // 2
        qf = self.q.float().view(self.B, KVH, G, D) / self.qbal[None, :, None, :]
        qp = torch.gather(qf, 3, self.perm[None, :, None, :].expand_as(qf))
        self.qw.copy_(torch.cat([qp[..., :h] @ self.H64, qp[..., h:] @ self.H64], -1).reshape(-1, D))
        self.qs2.copy_(torch.stack([self.qw[:, :h].float().sum(-1), self.qw[:, h:].float().sum(-1)], -1))

    def base(self):
        if not self.wf:
            assert not self.win
            return super().base()
        if self.qbias and self.rck == 2 and not self.fuse_q:
            assert self.bias_mode == "fp16"
            ntile = triton.cdiv(self.L, self.BB)
            npb = max(1, min(ntile, -(-self.bppsm * self.nsm // self.BK)))   # bias programs as Step7.prep
            bias_q_kernel[(self.BK, npb + 1)](self.u, self.w, self.c0tab, self.s0tab, self.csB, self.bias, self.L, ntile, npb,
                                              self.scale, self.q, self.ibp, self.H64h, self.perm32, self.qw, self.qs2, GH=G, GP=GP, D=D,
                                              BB=self.BB, KV=KVH, CB=32, num_warps=4)
        elif self.qside:
            self.run_qside()
            self.qside, _q = False, True          # prep() must not launch it again
            try:
                self.prep()
            finally:
                self.qside = _q
        else:
            if not self.fuse_q:
                self.qprep()
            if self.rck == 2:
                self.prep()
        win_f = self.fuse_w and self.win > 0
        kw_, vw_ = (self.kw, self.vw) if self.win else (self.qw, self.qw)          # dummies when there is no window
        self.kinfo["base"] = base_kernel_wf[(self.BK, self.nsb + (1 if win_f else 0))](
            self.qw, self.qs2, self.q, *((self.ibp, self.H64h, self.perm32) if self.fuse_q == 2 else (self.ib, self.Tq, self.perm32)), kw_, vw_, self.win, self.nsb, self.kp, self.vp, self.mt, self.lo_ptr(),
            self.bias, *self._rt(), self.m, self.l, self.acc, self.s, self.L, self.split_len, self.nsplit, self.scale,
            GH=G, GP=GP, P=P, BLOCK_N=self.BLOCK_N, RC=self.rck, RELOAD=self.reload3, MAGIC=self.magic, FQ=self.fuse_q,
            WIN=win_f, KV=KVH, WB=WB, M8=self.m8, WF1=self.wfirst, num_warps=self.warps, num_stages=self.stages,
            **({"maxnreg": self.maxnreg} if self.maxnreg else {}))
        if self.win and not win_f:
            window_kernel[(self.BK,)](self.q, self.kw, self.vw, self.m, self.l, self.acc, self.win, self.nsb, self.nsplit,
                                      self.scale, GH=G, GP=GP, D=D, WB=WB, num_warps=4)

    def _rt(self):
        """the rotary operands of kbench4.bias_kernel (u, w, tile-start cos / sin tables, in-tile [cos; sin] table)."""
        return self.u, self.w, self.c0tab, self.s0tab, self.csB

    def phase(self, name):
        if self.wf:
            raise NotImplementedError("per-phase timing is not implemented for the water-filled step")
        return super().phase(name)

    def fix_unfused(self):
        if self.wf:
            raise NotImplementedError("the unfused control is not implemented for the water-filled step")
        return super().fix_unfused()

    def fix(self):
        if not self.wf:
            return super().fix()
        if self.mode == "one":
            self.kinfo["fixA"] = fixA_kernel_wf[(self.BK, self.nch)](self.q, self.qw, self.qs2, self.m, self.idx, self.kz_, self.vz_, self.kp, self.vp,
                                                self.mt, self.lo_ptr(), self.bias, *self._rt(), self.pn, self.pz, self.pm, self.L,
                                                self.nsplit, self.scale, GH=G, GP=GP, P=P, K=self.K, KC=self.kc, NS=self.NS,
                                                RC=self.rck, BB=self.BB, M8=self.m8, num_warps=4)
            nc = self.nch
        else:
            self.kinfo["fix2A"] = fix2A_kernel_wf[(self.BK, self.NC2 // self.kca)](self.q, self.qw, self.qs2, self.idx, self.kz_, self.kp, self.mt,
                                                             self.lo_ptr(), self.bias, *self._rt(), self.sc, self.L, self.scale, GH=G,
                                                             GP=GP, P=P, M=self.NC2, KC=self.kca, RC=self.rck, BB=self.BB, M8=self.m8,
                                                             num_warps=4)
            K5.fix2B_kernel[(self.BK, self.nch2)](self.m, self.l, self.sc, self.idx, self.vz_, self.vp, self.mt, self.pn,
                                                  self.pz, self.pm, self.selm, self.L, self.nsplit, self.kap, GH=G, GP=GP, P=P,
                                                  M=self.NC2, K=self.Kv, CC=self.cc, NS=self.NS, MASKV=self.maskv,
                                                  MP=triton.next_power_of_2(self.NC2), RADIX=self.radix, num_warps=4)
            nc = self.nch2
        return K4.fixB_kernel[(self.BK, 4)](self.m, self.l, self.acc, self.pn, self.pz, self.pm, self.out, self.nsplit, nc,
                                            GH=G, GP=GP, P=P, NS=self.NS, NC=triton.next_power_of_2(nc), UNROLL=self.unroll,
                                            num_warps=4)


def wf_scores_ref(st):
    """fp32 torch reference pieces: water-filled base scores [BK, G, L] (+ rotary bias), exact scores, base and exact
    values, and the window (scores [BK, G, W], values [BK, W, D])."""
    BK, L = st.BK, st.L
    kp = st.kp.long()
    c0 = kp & 7
    c1 = (kp >> 3) & 7
    b0, b1 = (kp >> 6) & 1, (kp >> 7) & 1
    mt, mt2 = st.mt.float(), st.lo_eff()
    hi = torch.cat([c0, c1], -1).float() * mt[..., 0:1] + mt[..., 1:2]                # [BK, L, 64]
    lo = mt2[..., 0:1] + mt2[..., 1:2] * torch.cat([b0, b1], -1).float()              # [BK, L, 64]; record (mean - s, 2 s)
    kd = torch.cat([hi, lo], -1)                                                       # water-filled basis
    qw = st.qw.float().view(BK, G, D)
    s = torch.einsum("bgd,bld->bgl", qw, kd) * st.scale
    if st.rc == 2:
        t = torch.arange(L, device=qw.device, dtype=torch.float64)[:, None] * st.invf64[None]
        cos, sin = t.cos().float(), t.sin().float()
        ca, cb = st.ca.view(BK, G, D // 2), st.cb.view(BK, G, D // 2)
        s = s + (torch.einsum("bgi,li->bgl", ca, cos) + torch.einsum("bgi,li->bgl", cb, sin)) * st.scale
    vd = K5._deq(st.vp, st.vs, st.vz)
    q = st.q.float().view(BK, G, D)
    kx, vx = st.kh.to(q.device).float(), st.vh.to(q.device).float()
    sx = torch.einsum("bgd,bld->bgl", q, kx) * st.scale
    if st.win:
        sw = torch.einsum("bgd,bwd->bgw", q, st.kw[:, :st.win].float()) * st.scale
        vw = st.vw[:, :st.win].float()
    else:
        sw, vw = s[..., :0], vd[:, :0]
    return s, sx, vd, vx, sw, vw


def ref_one(st, idx):
    s, sx, vd, vx, sw, vw = wf_scores_ref(st)
    BK, K = st.BK, st.K
    ie = idx[:, None, :].expand(BK, G, K)
    se = s.clone()
    se.scatter_(2, ie, torch.gather(sx, 2, ie))
    vf = vd.clone()
    vf.scatter_(1, idx[..., None].expand(BK, K, D), torch.gather(vx, 1, idx[..., None].expand(BK, K, D)))
    p = torch.softmax(torch.cat([se, sw], -1), -1)
    return torch.einsum("bgl,bld->bgd", p, torch.cat([vf, vw], 1))


def ref_two(st, cand, sel=None):
    """two-stage reference on the kernel's candidates: stage-1 corrected scores, second-stage criterion (group-mean
    probability over [quantized | window] tokens), top-K selection; sel (per row) overrides the selection, so the output
    can be checked with the kernel's own pick. Returns (out, reference selection, criterion per row {token: value})."""
    s, sx, vd, vx, sw, vw = wf_scores_ref(st)
    BK, M, K = st.BK, st.NC2, st.Kv
    ce = cand[:, None, :].expand(BK, G, M)
    s1 = s.clone()
    s1.scatter_(2, ce, torch.gather(sx, 2, ce))
    full = torch.softmax(torch.cat([s1, sw], -1), -1)[..., :st.L]
    c2 = torch.gather(full.mean(1), 1, cand)
    sel_ref = torch.gather(cand, 1, c2.topk(K, -1).indices)
    use = sel_ref if sel is None else sel
    vf = vd.clone()
    vf.scatter_(1, use[..., None].expand(BK, K, D), torch.gather(vx, 1, use[..., None].expand(BK, K, D)))
    out = torch.einsum("bgl,bld->bgd", torch.softmax(torch.cat([s1, sw], -1), -1), torch.cat([vf, vw], 1))
    crit = [dict(zip(cand[r].tolist(), c2[r].tolist())) for r in range(BK)]
    return out, sel_ref, crit


def _ties_ok(picked, ref, crit, rtol, slack=None):
    """every difference between two selections is a near tie of the criterion (per row, pairwise, relative): for every
    kernel-only pick a and reference-only pick b, crit[b] - crit[a] <= (rtol + slack[a] + slack[b]) * max(|crit|). The
    slack (a relative bound on crit) is valid while the softmax denominator stays fp32 and only the logits are fp16."""
    for r in range(len(ref)):
        ko, ro = picked[r] - ref[r], ref[r] - picked[r]
        for ta in ko:
            for tb in ro:
                a, b = crit[r][ta], crit[r][tb]
                if not (math.isfinite(a) and math.isfinite(b)):
                    return False
                tol = rtol + (slack[r][ta] + slack[r][tb] if slack else 0.0)
                if b - a > tol * max(abs(a), abs(b), 1e-30):
                    return False
    return True


def encode_wf(st, seed=5):
    """round trip: real (Gaussian, offset) keys -> balance b, permutation, one 64-point Hadamard per half -> 3-bit min-max
    hi half / 1-bit mean +- MAD lo half -> the packed bytes and metadata the kernels read. Writes them, the matching qw / qs2
    and the exact keys (= the keys) into st, and returns k_hat in the ORIGINAL key space [BK, L, D]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    BK, L, h = st.BK, st.L, D // 2
    k = (torch.randn(BK, L, D, generator=g) + torch.randn(BK, 1, D, generator=g) * 2.0).cuda()
    b = (torch.rand(KVH, D, generator=g) * 1.5 + 0.5).cuda()
    perm = torch.stack([torch.randperm(D, generator=g) for _ in range(KVH)]).cuda()
    H = hadamard(h, "cuda")
    bk_b = b.repeat(st.B, 1)[:, None, :]                                     # [BK, 1, D] (row bk -> KV head bk % KVH)
    bk_p = perm.repeat(st.B, 1)[:, None, :].expand(BK, L, D)
    xs = torch.gather(k * bk_b, 2, bk_p)
    yh, yl = xs[..., :h] @ H, xs[..., h:] @ H
    mn = yh.amin(-1, keepdim=True).half().float()
    sc = ((yh.amax(-1, keepdim=True) - mn) / 7).half().float().clamp_min(1e-8)
    code = torch.clamp(torch.round((yh - mn) / sc), 0, 7).long()                # [BK, L, 64]
    mu = yl.mean(-1, keepdim=True).half().float()
    sd = (yl - mu).abs().mean(-1, keepdim=True).half().float()
    bit = (yl >= mu).long()
    c0, c1, b0, b1 = code[..., :32], code[..., 32:], bit[..., :32], bit[..., 32:]
    byte = c0 | (c1 << 3) | (b0 << 6) | (b1 << 7)          # contiguous 3-bit codes (bits 0-2, 3-5), signs 6, 7
    st.kp.copy_(byte.to(torch.uint8))
    st.mt[..., 0], st.mt[..., 1] = sc[..., 0].half(), mn[..., 0].half()
    st.mt2.copy_(torch.cat([mu - sd, 2 * sd], -1).half())                    # pre-merged lo record (A, B)
    st.sync_lo()
    lo8 = st.lo_eff()                                                          # the stored (pre-merged) lo record
    A, Bv = lo8[..., 0:1], lo8[..., 1:2]
    yq = torch.cat([(code.float() * sc + mn) @ H.T, (A + Bv * bit.float()) @ H.T], -1)
    kh = torch.empty_like(yq).scatter_(2, bk_p, yq) / bk_b                    # undo the permutation, then the balance
    st.perm.copy_(perm)                                                        # in place: qprep (every step) reads these
    st.qbal.copy_(b)
    st.build_T()
    st.qprep()
    st.kh.copy_(k.half().cpu())
    return kh


def check():
    """(1) kernels vs the fp32 reference with the kernel's own selection, over rotary 0 / 2, windows 0 / 64 / 128, one- and
    two-stage, L = 5000 / 70001; selections compared with the reference up to near ties; exact K, no duplicates, in range.
    (2) round trip: real keys encoded by encode_wf; the kernel's stored base scores vs q . k_hat in the key space.
    (3) mutants the check must reject: the lo deviation scaled by 1.05, the window dropped, hi / lo halves swapped.
    (4) wf=False equals kbench5.Step5 bitwise."""
    TOL = 2e-4
    # stored fp16 logits vs the fp32 reference (same qw), normalised by the row-global max: 4.0e-4 - 5.5e-4 measured on
    # box2 (check2.log, all rc 0 / 2 configurations); the lo x1.05 mutant gives 2.0e-3. Gate 1e-3 set from that measurement
    SDEV = 1e-3
    ok = True
    for rc in (0, 2, 3):
        for win in (0, 1, 33, 64, 100, 128):   # 1, 33, 100: partial 32-token window chunks (review r18)
            for mode, K, kw in (("one", 96, {}), ("two", 64, dict(M=128, cc=32)), ("two", 96, dict(M=224, cc=32))):
                for L in (5000, 70001):
                    st = Step7(2, L, K, rc, wf=True, win=win, mode=mode, qscale=4.0, seed=1, **kw)
                    st.step()
                    torch.cuda.synchronize()
                    s, *_, sw, _ = wf_scores_ref(st)
                    sk = st.s.view(st.BK, G, -1)[..., :L].float()
                    smax = float(s.abs().max())
                    sdev = float((sk - s).abs().max()) / smax          # stored fp16 stage-1 logits vs the fp32 reference
                    crit1 = torch.softmax(torch.cat([s, sw], -1), -1)[..., :L].mean(1)
                    nsel = st.K if mode == "one" else st.NC2
                    idx = st.idx.cpu()
                    sane = bool((idx >= 0).all() and (idx < L).all()) and all(len(set(r.tolist())) == nsel for r in idx)
                    top = crit1.topk(nsel, -1).indices.cpu()
                    c1 = [dict(enumerate(crit1[r].tolist())) for r in range(st.BK)]
                    # stage-1 scores are stored in fp16: the logit of token t carries up to 2^-11 |s_t| rounding (max over
                    # the heads of its group; the row max is fp32), so each token gets that slack in the relative tie test
                    sl = (2.0 ** -11 * st.s.view(st.BK, G, -1)[..., :L].float().abs().amax(1)).cpu()
                    sl1 = [dict(enumerate(sl[r].tolist())) for r in range(st.BK)]
                    sel1 = [set(r.tolist()) for r in idx]
                    ties1 = _ties_ok(sel1, [set(r.tolist()) for r in top], c1, 1e-4, sl1)
                    # control: swapping the strongest selected token for the weakest token must fail the tie test
                    best = max(sel1[0], key=lambda t: c1[0][t])
                    worst = min((t for t in range(L) if t not in sel1[0]), key=lambda t: c1[0][t])
                    ctrl = not _ties_ok([(sel1[0] - {best}) | {worst}], [set(top[0].tolist())], c1[:1], 1e-4, sl1[:1])
                    ties1 = ties1 and ctrl
                    got = st.out.view(st.BK, G, D)
                    if mode == "one":
                        ref = ref_one(st, st.idx)
                        extra = ""
                        ties2 = True
                    else:
                        kpos = [sorted(torch.nonzero(st.selm[r]).flatten().tolist()) for r in range(st.BK)]
                        ksel = torch.stack([st.idx[r, kpos[r]] for r in range(st.BK)])
                        ref, sel_ref, crit = ref_two(st, st.idx, sel=ksel)
                        ties2 = _ties_ok([set(r.tolist()) for r in ksel.cpu()], [set(r.tolist()) for r in sel_ref.cpu()],
                                         crit, 1e-4) and all(len(x) == st.Kv for x in kpos)   # fp32 scores
                        extra = f"; second stage exact K and near-tie agreement {ties2}"
                    err = float((got - ref).norm() / ref.norm())
                    good = err < TOL and sdev < SDEV and sane and ties1 and ties2
                    ok &= good
                    print(f"check wf rc={rc} win={win} {mode} K={K} {kw} L={L}: rel err vs fp32 reference with the kernel's "
                          f"selection {err:.3g}; stored logits max rel dev {sdev:.3g}; stage-1 selection in range / exact / near-tie (wrong-selection control rejected) "
                          f"{sane}/{ties1} ({ctrl}){extra} "
                          f"{'OK' if good else 'FAIL'}", flush=True)
                    if rc == 2 and L == 5000 and mode == "one":
                        # mutants (the reference is perturbed, the kernel output is not): each must exceed TOL
                        m2, qw = st.mt2.clone(), st.qw.clone()
                        e2 = None
                        try:
                            st.mt2[..., 1] *= 1.05
                            e1 = float((got - ref_one(st, st.idx)).norm() / ref.norm())
                            d1 = float((sk - wf_scores_ref(st)[0]).abs().max()) / smax
                            st.mt2.copy_(m2)
                            st.qw.copy_(torch.cat([qw[:, D // 2:], qw[:, :D // 2]], -1))
                            e3 = float((got - ref_one(st, st.idx)).norm() / ref.norm())
                            d3 = float((sk - wf_scores_ref(st)[0]).abs().max()) / smax
                            st.qw.copy_(qw)
                            if win >= 64:                     # the window mutant only exists when there is a window
                                st.win = 0
                                e2 = float((got - ref_one(st, st.idx)).norm() / ref.norm())
                        finally:
                            st.mt2.copy_(m2)
                            st.qw.copy_(qw)
                            st.win = win
                        # a mutant is caught by the output test or by the stored-logit test
                        caught = (e1 > TOL or d1 > SDEV) and (e3 > TOL or d3 > SDEV) and (e2 is None or e2 > TOL)
                        ok &= caught
                        e2s = "n/a" if e2 is None else f"{e2:.3g}"
                        print(f"  mutants rejected (output / stored logits: lo x1.05 {e1:.3g} / {d1:.3g}, window dropped {e2s}, hi/lo swapped "
                              f"{e3:.3g} / {d3:.3g}; "
                              f"smallest margin over TOL x{min([e1, e3] + ([e2] if e2 is not None else [])) / TOL:.1f}) "
                              f"{'OK' if caught else 'FAIL'}", flush=True)
                    del st
                    torch.cuda.empty_cache()
    # the one-launch query transform (qprep_kernel) vs the torch construction, with b = 1 and with a real balance b
    for real in (False, True):
        st = Step7(2, 5000, 96, 2, wf=True, mode="one", qscale=4.0, seed=6)
        if real:
            encode_wf(st)
        st.qprep()
        torch.cuda.synchronize()
        qk, sk2 = st.qw.float().clone(), st.qs2.clone()
        st.qprep_torch()
        torch.cuda.synchronize()
        dq = float((qk - st.qw.float()).abs().max() / st.qw.float().abs().max())
        ds = float((sk2 - st.qs2).abs().max() / st.qs2.abs().max())
        good = dq < 2e-3 and ds < 2e-3        # two fp16 roundings (q / b, qw) vs torch's one (qw); bound ~ 2^-11 x a few
        ok &= good
        print(f"check qprep kernel vs torch ({'real b' if real else 'b = 1'}): qw max rel dev {dq:.3g}, half sums {ds:.3g} "
              f"{'OK' if good else 'FAIL'}", flush=True)
        del st
        torch.cuda.empty_cache()
    # round trip on encoded real keys
    for rc in (0, 2, 3):
        for mode, K, kw in (("one", 96, {}), ("two", 64, dict(M=128, cc=32))):
            st = Step7(2, 5000, K, rc, wf=True, win=64, mode=mode, qscale=1.0, seed=2, **kw)
            kh = encode_wf(st)
            st.step()
            torch.cuda.synchronize()
            sk = torch.einsum("bgd,bld->bgl", st.q.float().view(st.BK, G, D), kh) * st.scale
            if rc >= 2:
                t = torch.arange(st.L, device=kh.device, dtype=torch.float64)[:, None] * st.invf64[None]
                cos, sin = t.cos().float(), t.sin().float()
                ca, cb = st.ca.view(st.BK, G, D // 2), st.cb.view(st.BK, G, D // 2)
                sk = sk + (torch.einsum("bgi,li->bgl", ca, cos) + torch.einsum("bgi,li->bgl", cb, sin)) * st.scale
            sker = st.s.view(st.BK, G, st.L).float()
            dev = float((sker - sk).abs().max() / sk.abs().max())
            ref = ref_one(st, st.idx) if mode == "one" else ref_two(st, st.idx, sel=torch.stack(
                [st.idx[r, torch.nonzero(st.selm[r]).flatten()] for r in range(st.BK)]))[0]
            err = float((st.out.view(st.BK, G, D) - ref).norm() / ref.norm())
            good = dev < 2e-3 and err < TOL
            ok &= good
            print(f"check round trip rc={rc} {mode}: kernel base scores vs q . k_hat in the key space, max rel dev {dev:.3g}; "
                  f"output rel err {err:.3g} {'OK' if good else 'FAIL'}", flush=True)
            del st
            torch.cuda.empty_cache()
    # in-kernel bias (rc 3) vs the pre-pass (rc 2) on the same data: stored logits within fp16 rounding (summation order
    # only), output within the tolerance; also the RELOAD variant
    for mode, K, kw in (("one", 96, {}), ("two", 64, dict(M=128, cc=32))):
        for L in (5000, 70001):
            outs, logs = [], []
            for rc, rl in ((2, False), (3, False), (3, True)):
                st = Step7(2, L, K, rc, wf=True, win=64, reload3=rl, mode=mode, qscale=4.0, seed=4, **kw)
                st.step()
                torch.cuda.synchronize()
                outs.append(st.out.float().clone())
                print(f"  compiled rc={rc} reload={rl} {mode}: " + ", ".join(
                    f"{k} regs {v.n_regs} spills {v.n_spills}" for k, v in st.kinfo.items()), flush=True)
                logs.append(st.s.view(st.BK, G, -1)[..., :L].float().clone())
                del st
                torch.cuda.empty_cache()
            ls = float(logs[0].abs().max())
            dl = [float((x - logs[0]).abs().max()) / ls for x in logs[1:]]
            do = [float((x - outs[0]).norm() / outs[0].norm()) for x in outs[1:]]
            good = max(dl) < 2e-3 and max(do) < TOL
            ok &= good
            print(f"check rc3 vs rc2 {mode} L={L}: stored logits max rel dev (rc3, rc3 reload) {dl[0]:.3g}, {dl[1]:.3g}; "
                  f"output rel diff {do[0]:.3g}, {do[1]:.3g} {'OK' if good else 'FAIL'}", flush=True)
    # fused launch (query transform + window inside the base kernel) vs the separate launches: same arithmetic, so the
    # outputs and stored logits must agree to fp32 noise (bitwise expected); also against the fp32 reference
    for L in (5000, 70001):
        for win in (0, 1, 100, 64):
            for mode, K, kw in (("one", 96, {}), ("two", 64, dict(M=128, cc=32))):
                outs = []
                # w1 / a1: window first (wfirst); qb / qbw (r23): query transform inside the bias pre-pass launch
                fzs = (False, "w", True, "qb", "qbw") + (("w1", "a1") if win else ())
                for fz in fzs:
                    fzkw = {"w": dict(fuse_q=False, fuse_w=True), "w1": dict(fuse_q=False, fuse_w=True, wfirst=True),
                            "a1": dict(fuse=True, wfirst=True), "qb": dict(qbias=True),
                            "qbw": dict(qbias=True, fuse_w=True)}.get(fz, dict(fuse=fz))
                    st = Step7(2, L, K, 2, wf=True, win=win, mode=mode, qscale=4.0, seed=9, **fzkw, **kw, **WF_CFG(L))
                    st.step()
                    torch.cuda.synchronize()
                    outs.append((st.out.float().clone(), st.s.view(st.BK, G, -1)[..., :L].float().clone(),
                                 st.qw.float().clone()))
                    if fz is True:
                        got = st.out.view(st.BK, G, D)
                        if mode == "one":
                            ref = ref_one(st, st.idx)
                        else:
                            ksel = torch.stack([st.idx[r, torch.nonzero(st.selm[r]).flatten()] for r in range(st.BK)])
                            ref = ref_two(st, st.idx, sel=ksel)[0]
                        err = float((got - ref).norm() / ref.norm())
                    del st
                    torch.cuda.empty_cache()
                (o0, s0_, q0) = outs[0]
                o1 = outs[2][0]
                # r29: the separate / bias-folded transform is the gather + H64 form, the in-kernel one (FQ: True, a1)
                # still the dense PH dot: same roundings, different fp32 summation (zero terms), so FQ variants are
                # compared within tolerance (qw 2e-3 relative, outputs TOL) and all others bitwise
                fqs = [fz_ in (True, "a1") for fz_ in fzs]
                same = all(torch.equal(o0, o_) and torch.equal(s0_, s_) and torch.equal(q0, q_)
                           for (o_, s_, q_), f in zip(outs[1:], fqs[1:]) if not f)
                near = all(float((o_ - o0).norm() / o0.norm()) < TOL and float((q_ - q0).abs().max() / q0.abs().max()) < 2e-3
                           for (o_, s_, q_), f in zip(outs[1:], fqs[1:]) if f)
                same = same and near
                do = float((o1 - o0).norm() / o0.norm())
                good = same and err < TOL
                ok &= good
                print(f"check fused vs separate launches win={win} {mode} L={L}: bitwise (FQ within tol) {same}, out rel diff {do:.3g}; "
                      f"fused vs fp32 reference {err:.3g} {'OK' if good else 'FAIL'}", flush=True)
    # r28 query-side kernel: u / w from q on the GPU must match the host values kbench4 computes (fp32 rounding only),
    # qw / qs2 must equal qprep_kernel's bitwise, and the step output must match the host-u / w step and the reference
    for L in (5000, 70001):
        for wf_ in (False, True):
            for mode, K, kw in (("one", 96, {}), ("two", 64, dict(M=128, cc=32))):
                outs = []
                for qs in (False, True):
                    st = Step7(2, L, K, 2, wf=wf_, mode=mode, qscale=4.0, seed=13, qside=qs, **kw,
                               **(WF_CFG(L) if wf_ else {}))
                    u0, w0 = st.u.clone(), st.w.clone()
                    st.u.zero_(), st.w.zero_()
                    if qs:                                       # u / w now come only from the kernel
                        st.step()
                    else:
                        st.u.copy_(u0), st.w.copy_(w0)
                        st.step()
                    torch.cuda.synchronize()
                    du = float(max((st.u - u0).abs().max() / u0.abs().max(), (st.w - w0).abs().max() / w0.abs().max()))
                    outs.append((st.out.float().clone(), st.qw.float().clone(), st.qs2.clone(), du))
                    del st
                    torch.cuda.empty_cache()
                do = float((outs[1][0] - outs[0][0]).norm() / outs[0][0].norm())
                qeq = (not wf_) or (torch.equal(outs[0][1], outs[1][1]) and torch.equal(outs[0][2], outs[1][2]))
                good = outs[1][3] < 1e-6 and do < TOL and bool(qeq)
                ok &= good
                print(f"check qside wf={wf_} {mode} L={L}: u/w vs host rel {outs[1][3]:.3g}, out rel diff {do:.3g}, "
                      f"qw/qs2 bitwise {qeq} {'OK' if good else 'FAIL'}", flush=True)
    # window first in the grid (wfirst) at the timed shapes and split counts (ns_fixed), window 128 fused, rc 2, both
    # one- and two-stage: bitwise equal to the window-last launch and within TOL of the fp32 reference (review r17)
    for B_, L_, ns_ in ((1, 32768, 12), (1, 32768, 19), (1, 32768, 56), (1, 8192, 8), (1, 8192, 32), (2, 5000, 7)):
        for mode, K, kw in (("one", 128, {}), ("two", 64, dict(M=128, cc=32))):
            res = []
            for w1 in (False, True):
                st = Step7(B_, L_, K, 2, wf=True, win=128, fuse_w=True, wfirst=w1, ns_fixed=ns_, mode=mode, qscale=4.0,
                           seed=12, **kw, **WF_CFG(L_, 2, B_))
                st.step()
                torch.cuda.synchronize()
                got = st.out.view(st.BK, G, D).float()
                if mode == "one":
                    ref = ref_one(st, st.idx)
                else:
                    ksel = torch.stack([st.idx[r, torch.nonzero(st.selm[r]).flatten()] for r in range(st.BK)])
                    ref = ref_two(st, st.idx, sel=ksel)[0]
                res.append((got.clone(), st.s.view(st.BK, G, -1)[..., :L_].float().clone(),
                            float((got - ref).norm() / ref.norm()), st.nsb))
                del st
                torch.cuda.empty_cache()
            same = torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
            good = bool(same) and max(res[0][2], res[1][2]) < TOL
            ok &= good
            print(f"check wfirst B={B_} L={L_} ns={ns_} (splits {res[1][3]}) {mode}: bitwise {same}, fp32 ref err "
                  f"{res[0][2]:.3g} / {res[1][2]:.3g} {'OK' if good else 'FAIL'}", flush=True)
    # float8 lo record (m8): kernels vs the fp32 reference on the float8-rounded record, one- / two-stage, window
    # fused, rc 2, both lengths; and the round trip with real keys (encode_wf) under m8
    for m8v in (1, 2, 3):   # 1 float8 e4m3, 2 int8 codes in step / 16 units (r20), 3 fp16 read as int32 (r21)
        for L in (5000, 70001):
            for mode, K, kw in (("one", 96, {}), ("two", 64, dict(M=128, cc=32))):
                st = Step7(2, L, K, 2, wf=True, win=64, fuse_w=True, m8=m8v, mode=mode, qscale=4.0, seed=11, **kw, **WF_CFG(L))
                st.step()
                torch.cuda.synchronize()
                got = st.out.view(st.BK, G, D)
                if mode == "one":
                    ref = ref_one(st, st.idx)
                else:
                    ksel = torch.stack([st.idx[r, torch.nonzero(st.selm[r]).flatten()] for r in range(st.BK)])
                    ref = ref_two(st, st.idx, sel=ksel)[0]
                err = float((got - ref).norm() / ref.norm())
                sk = st.s.view(st.BK, G, -1)[..., :L].float()
                s_ref = wf_scores_ref(st)[0]
                sdev = float((sk - s_ref).abs().max() / s_ref.abs().max())
                good = err < TOL and sdev < SDEV
                ok &= good
                print(f"check m8={m8v} lo record {mode} L={L}: rel err {err:.3g}, stored logits dev {sdev:.3g} {'OK' if good else 'FAIL'}",
                      flush=True)
                del st
                torch.cuda.empty_cache()
        for mode, K, kw in (("one", 96, {}), ("two", 64, dict(M=128, cc=32))):
            st = Step7(2, 5000, K, 2, wf=True, win=64, m8=m8v, mode=mode, qscale=1.0, seed=12, **kw)
            kh = encode_wf(st)
            st.step()
            torch.cuda.synchronize()
            sk = torch.einsum("bgd,bld->bgl", st.q.float().view(st.BK, G, D), kh) * st.scale
            t = torch.arange(st.L, device=kh.device, dtype=torch.float64)[:, None] * st.invf64[None]
            ca, cb = st.ca.view(st.BK, G, D // 2), st.cb.view(st.BK, G, D // 2)
            sk = sk + (torch.einsum("bgi,li->bgl", ca, t.cos().float()) + torch.einsum("bgi,li->bgl", cb, t.sin().float())) * st.scale
            dev = float((st.s.view(st.BK, G, st.L).float() - sk).abs().max() / sk.abs().max())
            good = dev < 2e-3
            ok &= good
            print(f"check m8={m8v} round trip {mode}: kernel base scores vs q . k_hat (k_hat from the 2-byte record) {dev:.3g} "
                  f"{'OK' if good else 'FAIL'}", flush=True)
            del st
            torch.cuda.empty_cache()
    # every swept (B, L) launch configuration of the timing (CFG), fused with a window, against the fp32 reference
    for (B_, L_) in CFG["wf"]:
        st = Step7(B_, L_, 96, 2, wf=True, win=64, fuse=True, mode="one", qscale=4.0, seed=10, **WF_CFG(L_, 2, B_))
        st.step()
        torch.cuda.synchronize()
        ref = ref_one(st, st.idx)
        err = float((st.out.view(st.BK, G, D) - ref).norm() / ref.norm())
        good = err < TOL
        ok &= good
        print(f"check swept config B={B_} L={L_} {WF_CFG(L_, 2, B_)} fused + window: rel err {err:.3g} "
              f"{'OK' if good else 'FAIL'}", flush=True)
        del st, ref
        torch.cuda.empty_cache()
    # the tuned launch configurations of the timing (WF_CFG) against the reference, rc 2 and 3, one- and two-stage
    for L in (5000, 70001):
        for rc in (2, 3):
            for mode, K, kw in (("one", 96, {}), ("two", 64, dict(M=128, cc=32))):
                st = Step7(2, L, K, rc, wf=True, win=64, mode=mode, qscale=4.0, seed=8, **kw, **WF_CFG(L, rc))
                st.step()
                torch.cuda.synchronize()
                got = st.out.view(st.BK, G, D)
                if mode == "one":
                    ref = ref_one(st, st.idx)
                else:
                    ksel = torch.stack([st.idx[r, torch.nonzero(st.selm[r]).flatten()] for r in range(st.BK)])
                    ref = ref_two(st, st.idx, sel=ksel)[0]
                err = float((got - ref).norm() / ref.norm())
                good = err < TOL
                ok &= good
                print(f"check tuned config {WF_CFG(L, rc)} rc={rc} {mode} L={L}: rel err {err:.3g} "
                      f"{'OK' if good else 'FAIL'}", flush=True)
                del st
                torch.cuda.empty_cache()
    a = Step7(2, 5000, 96, 2, wf=False, mode="one", qscale=4.0, seed=3)
    b = K5.Step5(2, 5000, 96, 2, mode="one", qscale=4.0, seed=3)
    a.step()
    b.step()
    torch.cuda.synchronize()
    same = bool(torch.equal(a.out, b.out))
    ok &= same
    print(f"check wf=False equals kbench5 Step5 bitwise: {same}", flush=True)
    print("ALL PASS" if ok else "SOME FAIL", flush=True)
    return ok


VARIANTS = {"one96": dict(mode="one", K=96), "one128": dict(mode="one", K=128), "one160": dict(mode="one", K=160),
            "two128k64": dict(mode="two", K=64, M=128, cc=32), "two224k96": dict(mode="two", K=96, M=224, cc=32)}


# best base-kernel launch configuration (BLOCK_N, stages; 4 warps) per shape for each kernel, from kb7_sweep.py on the
# RTX 4080 SUPER (box2, sweep_all.log, both kernels swept at every shape): each kernel is timed at its own best
CFG = {"plain": {(1, 8192): (128, 1), (1, 32768): (64, 2), (8, 8192): (128, 1), (8, 32768): (128, 2), (32, 8192): (128, 3)},
       "wf": {(1, 8192): (32, 1), (1, 32768): (128, 2), (8, 8192): (128, 2), (8, 32768): (128, 2), (32, 8192): (128, 2)}}


CFGJ: dict = {}      # r24: per-device swept (BLOCK_N, stages, ns) per kind | B | L, loaded from --cfg


def WF_CFG(L, rc=2, B=None, kind="wf"):
    """launch configuration of a base kernel at shape (B, L): the swept best when the shape was swept, else short
    contexts 32-token tiles without pipelining and long contexts 128-token tiles; with the in-kernel bias (rc 3) the
    rotary tile table follows the tile (BB = BLOCK_N); with the pre-pass (rc 2) bias_kernel keeps its 64-token tiles."""
    bn, stg = CFG[kind].get((B, L), (32, 1) if L <= 16384 else (128, 2))
    cfg = dict(BLOCK_N=bn, stages=stg)
    if rc == 3:
        cfg["BB"] = bn
    return cfg


def run(B, L, rounds, win):
    """one shape, interleaved: each configuration on the plain per-token base (kbench5, rc2), the water-filled base (rc2)
    and the water-filled base with a win-token window (rc2), plus an A/A copy (the rc0 cost of centring is in TB).
    The water-filled steps include the query transform (qprep, one kernel launch), which the plain step does not need:
    the wf - plain differences contain it."""
    BK = B * KVH
    # bytes streamed per step and layer copy, used only to size the cold-cache rotation: key 32 + value 32 + meta 8 + lo
    # (mean, MAD) 4 per token and KV head, plus the FP16 window (charged to every variant; the rc2 scratch s / bias is not
    # counted, so plain is slightly over- and wf slightly under-charged, both well past 2 x L2 at the timed shapes)
    per = BK * L * (2 * (P + 4) + 4) + 2 * BK * WB * D * 2
    # kbench4._copies caps at 32 copies, which misses 2 x L2 at 1 x 8192 on a 96 MiB-L2 Blackwell; allow up to 64
    n = min(64, max(K4._copies(per), math.ceil(2 * K4.L2_BYTES / per)))
    items = []
    for nm, v in VARIANTS.items():
        # r24 formal set: plain; _wf the previous water-filled step (query transform its own launch); _wfQ the
        # current one (query transform inside the bias pre-pass, registers capped at 168), no window; _wfwQ the same
        # with the window fused (the float8 / int8 lo records and the unfused window are dropped: slower, see r20-r22)
        items += [(nm, v, False, 0, 2), (nm + "_wf", v, True, 0, 2), (nm + "_wfQ", v, True, 0, 2),
                  (nm + "_wfwQ", v, True, win, 2)]
    items.append(("one96_aa", VARIANTS["one96"], False, 0, 2))
    fns, keep = {}, []
    # one pinned host copy of the exact rows for every variant of this shape (kbench4.Step accepts a shared host):
    # identical exact data across the paired variants and 1 GiB instead of one copy per variant
    host = (torch.randn(BK, L, D).half().pin_memory(), torch.randn(BK, L, D).half().pin_memory())
    for vi, (name, v, wf, w, rc) in enumerate(items):
        kw = dict(v)
        q = name.endswith("Q")
        kw["fuse_w"] = name.endswith("wfwQ")
        if q:
            kw.update(qbias=True, maxnreg=168)
        mode, K = kw.pop("mode"), kw.pop("K")
        kind = ("wfwQ" if kw["fuse_w"] else "wfQ") if q else ("wf" if wf else "plain")
        cj = CFGJ.get(f"{kind}|{B}|{L}")                 # swept best on THIS device (--cfg), else the 4080 table
        assert cj or not CFGJ, f"--cfg has no entry {kind}|{B}|{L}"   # no silent fallback (review r25)
        if cj:
            kw.update(BLOCK_N=cj[0], stages=cj[1], ns_fixed=cj[2])
            if len(cj) > 3:
                kw["bppsm"] = cj[3]                       # r29: swept bias pre-pass programs per SM
        else:
            kw.update(WF_CFG(L, rc, B, "wf" if wf else "plain"))
        steps = [Step7(B, L, K, rc, wf=wf, win=w, mode=mode, seed=1000 * vi + i, host=host, **kw) for i in range(n)]
        for st in steps:
            st.step()
        torch.cuda.synchronize()
        keep += steps + [host]
        fns[name + "_step"] = K4.graph_of(lambda ss=steps: [x.step() for x in ss])
    fns["empty_graph"] = K4.graph_of(lambda: None)
    t = K4.time_group(fns, rounds=rounds)
    res = {"B": B, "L": L, "copies": n, "win": win, "l2_bytes": K4.L2_BYTES, "cfgj": dict(CFGJ),
           "md5": MD5, "cold": n * per >= 2 * K4.L2_BYTES}
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
        return dict(us=round(S.median(d), 2), ci=K5._boot(d, rnd), rel=round(S.median([y / x - 1 for x, y in zip(raw[a], raw[b])]), 4),
                    diff_of_medians_us=round(dm, 2), consistent=abs(dm - S.median(d)) <= max(2.0, 0.25 * abs(S.median(d))))
    for nm in VARIANTS:
        res[f"{nm}_wf_minus_plain"] = pdiff(f"{nm}_step", f"{nm}_wf_step")
        res[f"{nm}_wfQ_minus_plain"] = pdiff(f"{nm}_step", f"{nm}_wfQ_step")          # the base's own cost (target)
        res[f"{nm}_wfwQ_minus_plain"] = pdiff(f"{nm}_step", f"{nm}_wfwQ_step")        # + the fused 128-token window
        res[f"{nm}_wfQ_minus_wf"] = pdiff(f"{nm}_wf_step", f"{nm}_wfQ_step")          # what r23 saved
    aa = pdiff("one96_step", "one96_aa_step")
    res["one96_aa_vs_one96"] = aa
    res["aa_ok"] = abs(aa["rel"]) <= 0.03 and aa["consistent"]
    keep = fns = None
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--shapes", default="1x8192,1x32768,8x8192,8x32768,32x8192")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--out", default="kbench7.json")
    ap.add_argument("--cfg", default="", help="json {kind|B|L: [BLOCK_N, stages, ns]} swept on this device (kb7_cfg.py)")
    a = ap.parse_args()
    if a.check:
        check()
        return
    K4.L2_BYTES = torch.cuda.get_device_properties(0).L2_cache_size
    if a.cfg:
        CFGJ.update(json.load(open(a.cfg)))
    out = []
    for sh in a.shapes.split(","):
        B, L = (int(x) for x in sh.split("x"))
        t0 = time.time()
        r = run(B, L, a.rounds, a.win)
        out.append(r)
        print(json.dumps(r), f"({time.time() - t0:.0f}s)", flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
