# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""kalign_fr = kalign.py f96b21f5 + fused host residual records (make_kalign_fr.py; RHR, 2026-11-19).
kalign (TieredKV, kernel path aligned with the accuracy reference; 2026-11-07, v2: e4m3 bytes read directly, aligned
encoder aenc_kernel; v1 passed kalign_check on Llama-3.2-3B, 28 layers x 3 lengths, median rel 3.9e-4, max 2.5e-3).
The reference accuracy numbers come from cap_attn.attend over cap_codec bytes (arm stkm8v8128k64_x2s128_p50_h<kb><vb>).
The timing kernels (kbench7.Step7 / kres.Step7R) implemented a simplified variant. StepA runs the SAME algorithm as
cap_attn.attend on the SAME stored bytes (one layer, one query row per sequence, B = 1 here):
  coded rows    [Z, W0) of a cap_attn Store (Z = 4 sinks, W0 = floor(Lk / 128) * 128): key / value code bytes as stored
                (cap_codec's key record is kbench7's; its value record is kbench7's planar layout) and the e4m3 metadata
                bytes themselves (key: hi step, hi zero in steps, lo mean, lo MAD; value: step, zero in steps), decoded in
                the kernels (70 B per token and KV head on the GPU, as cap_codec.RECORD_GPU)
  exact rows    the 4 FP16 sinks and the FP16 residual window rows [W0, Lk) (incl. the current token): one exact split
                (window kernel over W = Z + Lk - W0 <= 132 rows)
  calibration   the stack's channel order, balance b = bal^(a / 100), rotary mean m_pre (times the model's attention
                scaling) and the model's own inv_freq, with the bias tables built at the coded rows' absolute positions
  stage 1       top-M of the group-mean softmax of the RAW base scores, normalised over coded + exact rows (l1)
  stage 2 / out logit correction of cap_attn (x = s - (lam / 100) s2 / 2, s2 = (|qh|^2 step^2 / 12 + |ql|^2 MAD^2 (pi/2 - 1))
                scale^2) on every coded non-candidate row: the base kernel accumulates the normaliser l and the value sum
                with the corrected logits; candidates get exact-key scores se = s_raw + scale (H q) . r_k (host residual)
                and fix2B removes their corrected base mass; top-K by the group-mean final probability
  values        base values, candidate base values and value residuals all stay in Hadamard coordinates (as stored); the
                exact split's accumulator is rotated INTO Hadamard coordinates in the window kernel; the output is rotated
                back once (out @ H, H symmetric orthonormal)
Encoder (aenc_kernel): rows of a cache segment (prefill, or the 128 window rows that leave the window when Lk reaches a
multiple of 128) -> the same bytes as cap_attn.encode_rows (clip search over the stack's GAMMAS with e4m3 step / zero,
lo mean / MAD with cap_codec's fp16-then-e4m3 rounding, value Hadamard + e4m3 step / zero, residual records of
(k - k_hat) H and (v - v_hat) H); the rotary mean uses per-position cos / sin tables built exactly as cap_attn.cos_sin_at.
Float sums run in a different order than torch, so a code can differ at an exact rounding tie: kalign_check --enc reports
per-field mismatch rates.
Check: kalign_check.py (real model, IncStore bytes, per layer vs cap_attn.attend)."""
import math
import os
import sys

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kbench4 as K4  # noqa: E402
import kbench5 as K5  # noqa: E402
import kbench7 as K7  # noqa: E402
import kres as KR  # noqa: E402
from kbench4 import _planes, _kth_bits, _widen  # noqa: E402
from kbench5 import _crit  # noqa: E402
from kbench7 import _lo_pair, _wf_q, _wf_scores  # noqa: E402
from kres import _res_rows, _res_rows_w  # noqa: E402

D, P, KVH, GP = K7.D, K7.P, K7.KVH, K7.GP
Z_SINK, R_WIN = 4, 128


def configure(kvh, gh):
    """set the model geometry (KV heads, query heads per KV head) for every module of the kernel chain; call once per
    process before building StepA objects (the host code reads these module globals; the kernels take them as arguments
    and pad the query group to GP = 16 lanes). d = 128 is fixed (cap_codec layout)."""
    assert kvh >= 1 and 1 <= gh <= GP, (kvh, gh)
    me = sys.modules[__name__]
    for m in (K4, K5, K7, KR, me):
        for name, val in (("KVH", kvh), ("G", gh), ("H", kvh * gh)):
            if name in vars(m) and isinstance(vars(m)[name], int):
                setattr(m, name, val)
    return kvh, gh
WBA = 136                                  # exact-row capacity: 4 sinks + up to 128 window rows (+ pad)
PI2M1 = math.pi / 2 - 1


@triton.jit
def _e4f(x):
    """uint8 e4m3 bytes -> fp32."""
    return x.to(tl.float8e4nv, bitcast=True).to(tl.float32)


@triton.jit
def _meta(km_ptr, vm_ptr, row, mask, N: tl.constexpr):
    """cap_codec metadata of rows [N] -> (k step, k zero (absolute), lo A = mean - MAD, lo B = 2 MAD, lo MAD, v step,
    v zero (absolute)), fp32."""
    k4 = _e4f(tl.load(km_ptr + row[:, None] * 4 + tl.arange(0, 4)[None, :], mask=mask[:, None], other=0))
    a_, b_ = tl.split(tl.reshape(k4, [N, 2, 2]))            # split takes the last axis: a_ = bytes (0, 2), b_ = (1, 3)
    ksc, mu = tl.split(a_)
    kzr, sl = tl.split(b_)
    v2 = _e4f(tl.load(vm_ptr + row[:, None] * 2 + tl.arange(0, 2)[None, :], mask=mask[:, None], other=0))
    vsc, vzr = tl.split(v2)
    return ksc, kzr * ksc, mu - sl, 2.0 * sl, sl, vsc, vzr * vsc


@triton.jit
def _qenergy(qh0, qh1, ql0, ql1):
    a = qh0.to(tl.float32)
    b = qh1.to(tl.float32)
    c = ql0.to(tl.float32)
    d = ql1.to(tl.float32)
    return tl.sum(a * a, 1) + tl.sum(b * b, 1), tl.sum(c * c, 1) + tl.sum(d * d, 1)


@triton.jit
def abase_kernel(qw_ptr, qs2_ptr, kp_ptr, vp_ptr, km_ptr, vm_ptr, bias_ptr, m_out, l_out, l1_out, acc_out, s_out,
                 L, split_len, nsp, sm_scale, cpk, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr,
                 BLOCK_N: tl.constexpr, MAGIC: tl.constexpr, LAM: tl.constexpr):
    _abase_body(tl.program_id(0), tl.program_id(1), qw_ptr, qs2_ptr, kp_ptr, vp_ptr, km_ptr, vm_ptr, bias_ptr, m_out, l_out,
                l1_out, acc_out, s_out, L, split_len, nsp, sm_scale, cpk, GH, GP, P, BLOCK_N, MAGIC, LAM)


@triton.jit
def _abase_body(bk, sp, qw_ptr, qs2_ptr, kp_ptr, vp_ptr, km_ptr, vm_ptr, bias_ptr, m_out, l_out, l1_out, acc_out, s_out,
                L, split_len, nsp, sm_scale, cpk, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr,
                BLOCK_N: tl.constexpr, MAGIC: tl.constexpr, LAM: tl.constexpr):
    """one quantized split: raw base scores s (stored fp16 for stage 1, l1 = their normaliser), and with LAM the corrected
    logits sc = s + cpk (|qh|^2 step^2 / 12 + |ql|^2 MAD^2 (pi/2 - 1)) (cpk = -(lam / 200) scale^2) for l and the value
    sum (values in Hadamard coordinates, as stored)."""
    og = tl.arange(0, GP)
    op = tl.arange(0, P)
    gm = og < GH
    qb = qw_ptr + (bk * GH + og[:, None]) * (4 * P) + op[None, :]
    qh0, qh1, ql0, ql1 = _wf_q(qb, gm, P)
    qsh = tl.load(qs2_ptr + (bk * GH + og) * 2, mask=gm, other=0.0)
    qsl = tl.load(qs2_ptr + (bk * GH + og) * 2 + 1, mask=gm, other=0.0)
    ehi, elo = _qenergy(qh0, qh1, ql0, ql1)
    cH = ehi * (cpk / 12.0)
    cL = elo * (cpk * 0.5707963267948966)
    m_i = tl.full([GP], -float("inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    l1 = tl.zeros([GP], tl.float32)
    a0 = tl.zeros([GP, P], tl.float32)
    a1 = tl.zeros([GP, P], tl.float32)
    a2 = tl.zeros([GP, P], tl.float32)
    a3 = tl.zeros([GP, P], tl.float32)
    az = tl.zeros([GP], tl.float32)
    start = sp * split_len
    end = tl.minimum(start + split_len, L)
    on = tl.arange(0, BLOCK_N)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + on
        nm = offs_n < end
        base = bk * L + offs_n
        kb = tl.load(kp_ptr + base[:, None] * P + op[None, :], mask=nm[:, None], other=0)
        ksc, kzp, mu, slo, mad, vsc, vzp = _meta(km_ptr, vm_ptr, base, nm, BLOCK_N)
        s = _wf_scores(qh0, qh1, ql0, ql1, kb, ksc, kzp, mu, slo, qsh, qsl, MAGIC) * sm_scale
        s += tl.load(bias_ptr + (bk * GH + og[:, None]) * L + offs_n[None, :], mask=gm[:, None] & nm[None, :], other=0.0)
        s = tl.where(nm[None, :], s, -float("inf"))
        tl.store(s_out + (bk * GH + og[:, None]) * L + offs_n[None, :], s.to(tl.float16), mask=gm[:, None] & nm[None, :])
        if LAM:
            sc = s + cH[:, None] * (ksc * ksc)[None, :] + cL[:, None] * (mad * mad)[None, :]
        else:
            sc = s
        mn = tl.maximum(m_i, tl.maximum(tl.max(s, 1), tl.max(sc, 1)))
        al = tl.exp(m_i - mn)
        p1 = tl.exp(s - mn[:, None])
        p = tl.exp(sc - mn[:, None])
        l1 = l1 * al + tl.sum(p1, 1)
        l_i = l_i * al + tl.sum(p, 1)
        vb = tl.load(vp_ptr + base[:, None] * P + op[None, :], mask=nm[:, None], other=0)
        # p <= 1 and vsc is an e4m3 step (<= 448): the fp16 operand p vsc S stays below 65504 only if S max(vsc) does, so S
        # is 4096 (as before) while max(vsc) < 15.8 and otherwise the largest power of two keeping it below 60000; the
        # dot is brought back to the 4096 scale in fp32 (exact power-of-two factor; 1.0 whenever S = 4096)
        vmx = tl.max(vsc)
        sv = tl.where(vmx < 15.8, 4096.0, tl.exp2(tl.floor(tl.log2(60000.0 / tl.maximum(vmx, 1e-30)))))
        rs = 4096.0 / sv
        pv = (p * vsc[None, :] * sv).to(tl.float16)
        a0 = a0 * al[:, None] + tl.dot(pv, _widen(vb & 3, MAGIC)) * rs
        a1 = a1 * al[:, None] + tl.dot(pv, _widen((vb >> 2) & 3, MAGIC)) * rs
        a2 = a2 * al[:, None] + tl.dot(pv, _widen((vb >> 4) & 3, MAGIC)) * rs
        a3 = a3 * al[:, None] + tl.dot(pv, _widen((vb >> 6) & 3, MAGIC)) * rs
        az = az * al + tl.sum(p * vzp[None, :], 1)
        m_i = mn
    o = (bk * nsp + sp) * GP + og
    tl.store(m_out + o, m_i)
    tl.store(l_out + o, l_i)
    tl.store(l1_out + o, l1)
    ao = acc_out + o[:, None] * (4 * P) + op[None, :]
    tl.store(ao, a0 / 4096.0 + az[:, None], mask=gm[:, None])
    tl.store(ao + P, a1 / 4096.0 + az[:, None], mask=gm[:, None])
    tl.store(ao + 2 * P, a2 / 4096.0 + az[:, None], mask=gm[:, None])
    tl.store(ao + 3 * P, a3 / 4096.0 + az[:, None], mask=gm[:, None])


@triton.jit
def awin_kernel(q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, wsp, nsp, sm_scale, GH: tl.constexpr,
                GP: tl.constexpr, D: tl.constexpr, WB: tl.constexpr, WC: tl.constexpr = 32):
    _awin_body(tl.program_id(0), q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, wsp, nsp, sm_scale, GH, GP,
               D, WB, WC)


@triton.jit
def abasew_kernel(qw_ptr, qs2_ptr, kp_ptr, vp_ptr, km_ptr, vm_ptr, bias_ptr, m_out, l_out, l1_out, acc_out, s_out,
                  L, split_len, nsp, sm_scale, cpk, q_ptr, kw_ptr, vw_ptr, hpm_ptr, W, nsb, GH: tl.constexpr,
                  GP: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, MAGIC: tl.constexpr, LAM: tl.constexpr,
                  WB: tl.constexpr):
    """one launch per layer step: grid row 0 is the exact split (sinks + window, partial slot nsb), rows 1 .. nsb the
    quantized splits 0 .. nsb - 1 (the exact split first, so its work does not trail the last wave)."""
    bk = tl.program_id(0)
    sp = tl.program_id(1)
    if sp == 0:
        _awin_body(bk, q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, nsb, nsp, sm_scale, GH, GP, 4 * P, WB, 32)
    else:
        _abase_body(bk, sp - 1, qw_ptr, qs2_ptr, kp_ptr, vp_ptr, km_ptr, vm_ptr, bias_ptr, m_out, l_out, l1_out, acc_out, s_out,
                    L, split_len, nsp, sm_scale, cpk, GH, GP, P, BLOCK_N, MAGIC, LAM)


@triton.jit
def _awin_body(bk, q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, wsp, nsp, sm_scale, GH: tl.constexpr,
               GP: tl.constexpr, D: tl.constexpr, WB: tl.constexpr, WC: tl.constexpr):
    """exact attention over the W (<= WB) FP16 rows (sinks + residual window) of one batch * KV head, as split wsp; the
    accumulator is stored in Hadamard coordinates (acc @ H, fp16 hi / lo split against the exact +-1 matrix)."""
    og = tl.arange(0, GP)
    od = tl.arange(0, D)
    oc = tl.arange(0, WC)
    gm = og < GH
    q = tl.load(q_ptr + (bk * GH + og[:, None]) * D + od[None, :], mask=gm[:, None], other=0.0)
    m_i = tl.full([GP], -float("inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    for w0 in tl.range(0, W, WC, num_stages=1):
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
        acc = acc * al[:, None] + tl.dot((p * 4096.0).to(tl.float16), v)
        m_i = mn
    acc = acc / 4096.0
    hpm = tl.load(hpm_ptr + od[:, None] * D + od[None, :])
    ah = acc.to(tl.float16)
    alo = ((acc - ah.to(tl.float32)) * 2048.0).to(tl.float16)
    acch = (tl.dot(ah, hpm) + tl.dot(alo, hpm) * (1.0 / 2048.0)) * 0.08838834764831845
    o = (bk * nsp + wsp) * GP + og
    tl.store(m_out + o, m_i)
    tl.store(l_out + o, l_i)
    tl.store(l1_out + o, l_i)
    tl.store(acc_out + o[:, None] * D + od[None, :], acch, mask=gm[:, None])


@triton.jit
def _rw_load(rp_ptr, row, mask, NW: tl.constexpr):
    """raw residual record words [R, 32] (int32, lanes >= NW zero) of rows `row` (int64) where mask."""
    ow = tl.arange(0, 32)
    rp32 = rp_ptr.to(tl.pointer_type(tl.int32))
    return tl.load(rp32 + row[:, None] * NW + ow[None, :], mask=mask[:, None] & (ow[None, :] < NW), other=0)


@triton.jit
def _rw_decode(words, mn, st, mask, BITS: tl.constexpr):
    """kres._res_rows_w's decode from words already in registers: code * step + min, rows outside mask 0."""
    c = tl.arange(0, 128)
    R: tl.constexpr = words.shape[0]
    if BITS == 4:
        j = c % 64
        w = tl.gather(words, tl.broadcast_to((j // 4)[None, :], (R, 128)), axis=1)
        code = (w >> (8 * (j % 4) + 4 * (c // 64))[None, :]) & 15
    else:
        p = c * BITS
        w0 = tl.gather(words, tl.broadcast_to((p // 32)[None, :], (R, 128)), axis=1).to(tl.uint32).to(tl.int64)
        w1 = tl.gather(words, tl.broadcast_to(tl.minimum(p // 32 + 1, 31)[None, :], (R, 128)), axis=1).to(tl.uint32).to(tl.int64)
        code = (((w0 | (w1 << 32)) >> (p % 32)[None, :]) & ((1 << BITS) - 1)).to(tl.int32)
    return tl.where(mask[:, None], code.to(tl.float32) * st[:, None] + mn[:, None], 0.0)


@triton.jit
def _res_rows_wf(rp_ptr, row, mask, BITS: tl.constexpr, NBY: tl.constexpr, RSW: tl.constexpr):
    """fused record: NW = NBY / 4 data words, then fp16 min, fp16 step, record stride RSW words. v2 (11-19 14:0xZ):
    the SAME instruction sequence as kres._res_rows_w (one int32 load of the NW data words, two fp16 loads of min / step,
    the same decode), only the addresses differ: min / step sit in the row's own record, right after its data words.
    v1 bitcast min / step out of one extra int32 word: bytes and selection were equal, but the fp32 results differed in
    the last bit from the split path (a different compiled float sequence), failing the bitwise gate."""
    NW: tl.constexpr = NBY // 4
    NWP: tl.constexpr = 32
    c = tl.arange(0, 128)
    ow = tl.arange(0, NWP)
    rp32 = rp_ptr.to(tl.pointer_type(tl.int32))
    rp16 = rp_ptr.to(tl.pointer_type(tl.float16))
    words = tl.load(rp32 + row[:, None].to(tl.int64) * RSW + ow[None, :], mask=mask[:, None] & (ow[None, :] < NW), other=0)
    R: tl.constexpr = words.shape[0]
    if BITS == 4:
        j = c % 64
        w = tl.gather(words, tl.broadcast_to((j // 4)[None, :], (R, 128)), axis=1)
        code = (w >> (8 * (j % 4) + 4 * (c // 64))[None, :]) & 15
    else:
        p = c * BITS
        w0 = tl.gather(words, tl.broadcast_to((p // 32)[None, :], (R, 128)), axis=1).to(tl.uint32).to(tl.int64)
        w1 = tl.gather(words, tl.broadcast_to(tl.minimum(p // 32 + 1, NWP - 1)[None, :], (R, 128)), axis=1).to(tl.uint32).to(tl.int64)
        code = (((w0 | (w1 << 32)) >> (p % 32)[None, :]) & ((1 << BITS) - 1)).to(tl.int32)
    mn = tl.load(rp16 + row.to(tl.int64) * (2 * RSW) + 2 * NW, mask=mask, other=0.0).to(tl.float32)
    st = tl.load(rp16 + row.to(tl.int64) * (2 * RSW) + 2 * NW + 1, mask=mask, other=0.0).to(tl.float32)
    return tl.where(mask[:, None], code.to(tl.float32) * st[:, None] + mn[:, None], 0.0)


@triton.jit
def _lookup(cidx_ptr, bk, tok, M: tl.constexpr):
    """slot of each token of `tok` [R] in the previous step's cache index row bk ([M] int64, -1 = empty), -1 if absent."""
    om = tl.arange(0, M)
    prev = tl.load(cidx_ptr + bk * M + om)
    hit = tok[:, None] == prev[None, :]
    return tl.max(tl.where(hit, om[None, :], -1), 1)


@triton.jit
def _rec_fetch(rp_ptr, rm_ptr, row, crp_ptr, crm_ptr, crow, use, hit, NW: tl.constexpr):
    """record words + (min, step) of rows wanted (`use`): from the GPU cache rows `crow` where hit, else from host rows
    `row` (masked loads: a hit issues no host request)."""
    fh = use & (hit == 0)
    fc = use & (hit != 0)
    words = _rw_load(rp_ptr, row, fh, NW) | _rw_load(crp_ptr, crow, fc, NW)
    mn = tl.where(fc, tl.load(crm_ptr + crow * 2, mask=fc, other=0.0).to(tl.float32),
                  tl.load(rm_ptr + row * 2, mask=fh, other=0.0).to(tl.float32))
    st = tl.where(fc, tl.load(crm_ptr + crow * 2 + 1, mask=fc, other=0.0).to(tl.float32),
                  tl.load(rm_ptr + row * 2 + 1, mask=fh, other=0.0).to(tl.float32))
    return words, mn, st


@triton.jit
def _rec_store(words, mn, st, nrp_ptr, nrm_ptr, nidx_ptr, nrow, tokv, NW: tl.constexpr):
    """write records (words, min, step) and their token ids to the next step's cache rows nrow."""
    ow = tl.arange(0, 32)
    n32 = nrp_ptr.to(tl.pointer_type(tl.int32))
    tl.store(n32 + nrow[:, None] * NW + ow[None, :], words, mask=(ow[None, :] < NW))
    tl.store(nrm_ptr + nrow * 2, mn.to(tl.float16))
    tl.store(nrm_ptr + nrow * 2 + 1, st.to(tl.float16))
    tl.store(nidx_ptr + nrow, tokv)


@triton.jit
def afix2A(qh_ptr, qw_ptr, qs2_ptr, idx_ptr, krp_ptr, krm_ptr, kp_ptr, km_ptr, vm_ptr, bias_ptr, sc_out, L, sm_scale,
           cpk, ckp, ckm, ckidx, nkp, nkm, nkidx, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr, M: tl.constexpr,
           KC: tl.constexpr, KB: tl.constexpr, WORDS: tl.constexpr, LAM: tl.constexpr, CACHE: tl.constexpr = False,
           FUSED: tl.constexpr = False, RSW: tl.constexpr = 0):
    """kres.fix2A_res (RC 2) with the logit correction: writes se = s_raw + scale (H q) . r_k (exact-key score of the
    candidate) and sb = the corrected base logit the base kernel accumulated (removed by fix2B)."""
    bk = tl.program_id(0)
    c = tl.program_id(1)
    og = tl.arange(0, GP)
    op = tl.arange(0, P)
    gm = og < GH
    D4: tl.constexpr = 4 * P
    od = tl.arange(0, D4)
    ok = c * KC + tl.arange(0, KC)
    tok = tl.load(idx_ptr + bk * M + ok)
    row = bk.to(tl.int64) * L + tok
    qh = tl.load(qh_ptr + (bk * GH + og[:, None]) * D4 + od[None, :], mask=gm[:, None], other=0.0)
    ksr, kzr, mu, slo, mad, vsc, vzp = _meta(km_ptr, vm_ptr, row, row >= 0, KC)
    qb = qw_ptr + (bk * GH + og[:, None]) * D4 + op[None, :]
    qsh = tl.load(qs2_ptr + (bk * GH + og) * 2, mask=gm, other=0.0)
    qsl = tl.load(qs2_ptr + (bk * GH + og) * 2 + 1, mask=gm, other=0.0)
    kb = tl.load(kp_ptr + row[:, None] * P + op[None, :])
    qh0, qh1, ql0, ql1 = _wf_q(qb, gm, P)
    sb = _wf_scores(qh0, qh1, ql0, ql1, kb, ksr, kzr, mu, slo, qsh, qsl, True) * sm_scale
    sb += tl.load(bias_ptr + (bk * GH + og[:, None]) * L + tok[None, :], mask=gm[:, None], other=0.0)
    tl.static_assert(not (CACHE and FUSED), "S5 cache and fused records are exclusive")
    if CACHE:            # S5: candidates fetched last step come from the GPU cache, the rest from host memory
        NWK: tl.constexpr = 4 * KB
        slot = _lookup(ckidx, bk, tok, M)
        hitk = (slot >= 0).to(tl.int32)
        crow = bk.to(tl.int64) * M + tl.maximum(slot, 0)
        allr = tok >= 0
        words, mnk, stk = _rec_fetch(krp_ptr, krm_ptr, row, ckp, ckm, crow, allr, hitk, NWK)
        rk = _rw_decode(words, mnk, stk, allr, KB)
        _rec_store(words, mnk, stk, nkp, nkm, nkidx, bk.to(tl.int64) * M + ok, tok, NWK)
    elif FUSED:
        rk = _res_rows_wf(krp_ptr, row, row >= 0, KB, 16 * KB, RSW)
    elif WORDS:
        rk = _res_rows_w(krp_ptr, krm_ptr, row, row >= 0, KB, 16 * KB)
    else:
        rk = _res_rows(krp_ptr, krm_ptr, row, row >= 0, KB, 16 * KB)
    se = sb + tl.dot(qh, tl.trans(rk), input_precision="tf32x3") * sm_scale
    if LAM:
        ehi, elo = _qenergy(qh0, qh1, ql0, ql1)
        sb = sb + (ehi * (cpk / 12.0))[:, None] * (ksr * ksr)[None, :] + (elo * (cpk * 0.5707963267948966))[:, None] * (mad * mad)[None, :]
    so = sc_out + (bk * 2 * GP + og[:, None]) * M + ok[None, :]
    tl.store(so, se)
    tl.store(so + GP * M, sb)


@triton.jit
def afix2B(m_in, l_in, sc_in, idx_ptr, vrp_ptr, vrm_ptr, vp_ptr, km_ptr, vm_ptr, pn_out, pz_out, pm_out, selm_out, L, nsplit,
           kap_out, cvp, cvm, cvidx, nvp, nvm, nvidx, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr, M: tl.constexpr,
           K: tl.constexpr, CC: tl.constexpr, NS: tl.constexpr, MP: tl.constexpr, RADIX: tl.constexpr, VB: tl.constexpr,
           WORDS: tl.constexpr, CACHE: tl.constexpr = False, FUSED: tl.constexpr = False, RSW: tl.constexpr = 0):
    """kres.fix2B_res with every value term left in Hadamard coordinates (no rotation here; the step rotates the output
    once)."""
    bk = tl.program_id(0)
    c = tl.program_id(1)
    og = tl.arange(0, GP)
    op = tl.arange(0, P)
    gm = og < GH
    os_ = tl.arange(0, NS)
    sm_ = os_[:, None] < nsplit
    mm = tl.load(m_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_, other=-float("inf"))
    ll = tl.load(l_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_, other=0.0)
    MV = tl.max(mm, 0)
    MV = tl.where(gm, MV, 0.0)
    ZV = tl.sum(ll * tl.exp(mm - MV[None, :]), 0)
    MW: tl.constexpr = M if MP == 0 else MP
    oj = tl.arange(0, MW)
    jm = oj < M
    sa = sc_in + (bk * 2 * GP + og[:, None]) * M + oj[None, :]
    se_all = tl.load(sa, mask=gm[:, None] & jm[None, :], other=-float("inf"))
    sb_all = tl.load(sa + GP * M, mask=gm[:, None] & jm[None, :], other=-float("inf"))
    Mx = tl.where(gm, tl.maximum(MV, tl.max(se_all, 1)), 0.0)
    ze = tl.sum(tl.exp(se_all - Mx[:, None]), 1)
    Z1 = ZV * tl.exp(MV - Mx) - tl.sum(tl.exp(sb_all - Mx[:, None]), 1) + ze
    Z1 = tl.where(gm, tl.maximum(Z1, tl.maximum(ze, 1e-30)), 1.0)
    if c == 0:
        tl.store(kap_out + bk * GP + og, tl.sum(tl.exp(sb_all - Mx[:, None]), 1) / Z1, mask=gm)
    oc = c * CC + tl.arange(0, CC)
    b32 = _crit(sc_in, bk, oj, Mx, Z1, og, GH, GP, M).to(tl.int32, bitcast=True)
    b32 = tl.where(jm, b32, -1)
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
    bc = tl.sum(tl.where(oj[:, None] == oc[None, :], b32[:, None], 0), 0)
    tie = bc == lo
    sel = (bc > lo) | (tie & (n_gt + tie_before + tl.cumsum(tie.to(tl.int32), 0) - 1 < K))
    tl.store(selm_out + bk * M + oc, sel.to(tl.int8))
    tok = tl.load(idx_ptr + bk * M + oc)
    row = bk.to(tl.int64) * L + tok
    sp = sc_in + (bk * 2 * GP + og[:, None]) * M + oc[None, :]
    se = tl.load(sp, mask=gm[:, None], other=-float("inf"))
    sb = tl.load(sp + GP * M, mask=gm[:, None], other=-float("inf"))
    Mc = tl.maximum(MV, tl.max(se, 1))
    Mc = tl.where(gm, Mc, 0.0)
    ee = tl.exp(se - Mc[:, None])
    eb = tl.exp(sb - Mc[:, None])
    dz = tl.sum(ee, 1) - tl.sum(eb, 1)
    ees = tl.where(sel[None, :], ee, 0.0)
    wb = eb - ee
    tl.static_assert(not (CACHE and FUSED), "S5 cache and fused records are exclusive")
    if CACHE:            # S5: selected rows whose value record was fetched last step come from the GPU cache
        NWV: tl.constexpr = 4 * VB
        slot = _lookup(cvidx, bk, tok, M)
        hitv = (slot >= 0).to(tl.int32)
        crow = bk.to(tl.int64) * M + tl.maximum(slot, 0)
        words, mnv, stv = _rec_fetch(vrp_ptr, vrm_ptr, row, cvp, cvm, crow, sel, hitv, NWV)
        rv = _rw_decode(words, mnv, stv, sel, VB)
        _rec_store(words, mnv, stv, nvp, nvm, nvidx, bk.to(tl.int64) * M + oc, tl.where(sel, tok, -1), NWV)
    elif FUSED:
        rv = _res_rows_wf(vrp_ptr, row, sel, VB, 16 * VB, RSW)
    elif WORDS:
        rv = _res_rows_w(vrp_ptr, vrm_ptr, row, sel, VB, 16 * VB)
    else:
        rv = _res_rows(vrp_ptr, vrm_ptr, row, sel, VB, 16 * VB)
    rr = tl.dot(ees, rv, input_precision="tf32x3")                                 # sum ee r_v, Hadamard coordinates
    e0, e1, e2, e3 = _planes(rr, GP, P)
    ksr, kzr, mu, slo, mad, vsc, vzp = _meta(km_ptr, vm_ptr, row, row >= 0, CC)
    wv = wb * vsc[None, :]
    wbh = wv.to(tl.float16)
    wbl = ((wv - wbh.to(tl.float32)) * 2048.0).to(tl.float16)
    wbz = tl.sum(wb * vzp[None, :], 1)[:, None]
    vb = tl.load(vp_ptr + row[:, None] * P + op[None, :])
    o = (bk * tl.num_programs(1) + c) * GP + og
    pb = pn_out + o[:, None] * (4 * P) + op[None, :]
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


@triton.jit
def aprep_kernel(u_ptr, w_ptr, c0_ptr, s0_ptr, cs_ptr, bias_out, L, ntile, npb, sm_scale, q_ptr, ib_ptr, t_ptr, perm_ptr, qt_out,
                 qts_out, hpm_ptr, qh_out, GH: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BB: tl.constexpr,
                 KV: tl.constexpr, CB: tl.constexpr):
    """kbench7.bias_q_kernel (rotary bias tiles + the water-filled query transform in one launch) plus H q for the
    residual scores (fp16 q against the exact +-1 matrix, fp32 accumulation, / sqrt(128)): program 0 of each row does the
    query side, programs 1 .. npb the bias tiles."""
    bk = tl.program_id(0)
    pid = tl.program_id(1)
    if pid == 0:
        K7._qprep_body(bk, q_ptr, ib_ptr, t_ptr, perm_ptr, qt_out, qts_out, GH, GP, D, KV, CB)
        og = tl.arange(0, GP)
        od = tl.arange(0, D)
        gm = og < GH
        qv = tl.load(q_ptr + (bk * GH + og[:, None]) * D + od[None, :], mask=gm[:, None], other=0.0)
        hpm = tl.load(hpm_ptr + od[:, None] * D + od[None, :])
        qhv = tl.dot(qv, hpm) * 0.08838834764831845
        tl.store(qh_out + (bk * GH + og[:, None]) * D + od[None, :], qhv, mask=gm[:, None])
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
            tl.store(bias_out + (bk * GH + og[:, None]) * L + offs[None, :], b * sm_scale, mask=gm[:, None] & (offs < L)[None, :])


@triton.jit
def afixB_kernel(m_in, l_in, acc_in, pn_in, pz_in, pm_in, hpm_ptr, out_ptr, nsplit, nch, GH: tl.constexpr, GP: tl.constexpr,
                 D: tl.constexpr, NS: tl.constexpr, NC: tl.constexpr, UNROLL: tl.constexpr):
    """kbench4.fixB_kernel over all D channels in one program per batch * KV head, followed by the rotation of the output
    out of Hadamard coordinates (out @ H: fp16 hi / lo split against the exact +-1 matrix, / sqrt(128))."""
    bk = tl.program_id(0)
    og = tl.arange(0, GP)
    od = tl.arange(0, D)
    gm = og < GH
    os_ = tl.arange(0, NS)
    sm_ = os_[:, None] < nsplit
    mm = tl.load(m_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_, other=-float("inf"))
    ll = tl.load(l_in + (bk * nsplit + os_[:, None]) * GP + og[None, :], mask=sm_, other=0.0)
    M = tl.max(mm, 0)
    Z = tl.sum(ll * tl.exp(mm - M[None, :]), 0)
    A = tl.zeros([GP, D], tl.float32)
    for s0 in range(0, nsplit, UNROLL):
        for u_ in tl.static_range(UNROLL):
            s_ = s0 + u_
            vm = gm & (s_ < nsplit)
            w = tl.exp(tl.load(m_in + (bk * nsplit + s_) * GP + og, mask=vm, other=-float("inf")) - M)
            w = tl.where(vm, w, 0.0)
            A += tl.load(acc_in + ((bk * nsplit + s_) * GP + og[:, None]) * D + od[None, :], mask=vm[:, None], other=0.0) * w[:, None]
    oc = tl.arange(0, NC)
    cm = (oc[:, None] < nch) & gm[None, :]
    pm = tl.load(pm_in + (bk * nch + oc[:, None]) * GP + og[None, :], mask=cm, other=-float("inf"))
    M2 = tl.maximum(M, tl.max(pm, 0))
    r = tl.exp(M - M2)
    wc = tl.where(cm, tl.exp(pm - M2[None, :]), 0.0)
    pz = tl.load(pz_in + (bk * nch + oc[:, None]) * GP + og[None, :], mask=cm, other=0.0)
    Z2 = Z * r + tl.sum(pz * wc, 0)
    N = A * r[:, None]
    for c in range(0, nch):
        wv = tl.sum(tl.where(oc[:, None] == c, wc, 0.0), 0)
        N += tl.load(pn_in + ((bk * nch + c) * GP + og[:, None]) * D + od[None, :], mask=gm[:, None], other=0.0) * wv[:, None]
    Z2 = tl.maximum(Z2, 1e-30)                       # cancellation guard (fix2B floors Z1 at the exact candidate mass)
    o = N / tl.where(gm, Z2, 1.0)[:, None]
    hpm = tl.load(hpm_ptr + od[:, None] * D + od[None, :])
    oh = o.to(tl.float16)
    ol = ((o - oh.to(tl.float32)) * 2048.0).to(tl.float16)
    orot = (tl.dot(oh, hpm) + tl.dot(ol, hpm) * (1.0 / 2048.0)) * 0.08838834764831845
    tl.store(out_ptr + (bk * GH + og[:, None]) * D + od[None, :], orot, mask=gm[:, None])


@triton.jit
def appw_kernel(newk_ptr, newv_ptr, kw_ptr, vw_ptr, cnt_ptr, WB: tl.constexpr, ZS: tl.constexpr):
    """per-step append: the new token's K / V row into exact-row slot ZS + (step % 128) of each batch * KV head (the
    window rows follow the sinks; every 128 steps the 128 window rows are encoded into the base, StepA.encode)."""
    bk = tl.program_id(0)
    c = tl.arange(0, 128)
    n = tl.load(cnt_ptr) % 128
    row = (bk * WB + ZS + n).to(tl.int64)
    tl.store(kw_ptr + row * 128 + c, tl.load(newk_ptr + bk * 128 + c))
    tl.store(vw_ptr + row * 128 + c, tl.load(newv_ptr + bk * 128 + c))


# ======================================================================== encoder (cap_attn.encode_rows in one kernel)
E4MIN = 2.0 ** -9


@triton.jit
def _e4r(x):
    """fp32 -> nearest e4m3 value as fp32 (cap_codec.e4: clamp to +-448, round to nearest even), computed exactly in fp32:
    Triton's fp32 -> fp8e4nv cast goes through fp16 (double rounding: ~1/128 of the values land on the wrong neighbour)."""
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    e = ((tl.abs(x).to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
    e = tl.maximum(e, -6)                                     # subnormals: fixed step 2^-9
    step = ((e - 3 + 127) << 23).to(tl.float32, bitcast=True)   # 3 mantissa bits
    return libdevice.rint(x / step) * step                    # x / step is exact (power of two)


@triton.jit
def _e4b(x):
    """an e4m3 value (fp32) -> its byte."""
    return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)


@triton.jit
def _f16(x):
    return x.to(tl.float16).to(tl.float32)


@triton.jit
def _enc_res(r, rp_ptr, rm_ptr, row, rmask, BITS: tl.constexpr, TB: tl.constexpr):
    """cap_codec.encode_res of r [TB, 128] (Hadamard coordinates): fp16 min / step, codes packed as cap_codec._pack_bits."""
    lv: tl.constexpr = (1 << BITS) - 1
    NBY: tl.constexpr = 16 * BITS
    mn = _f16(tl.min(r, 1))
    mx = _f16(tl.max(r, 1))
    st = _f16(tl.math.div_rn(mx - mn, lv * 1.0))
    safe = tl.where(st > 0, st, 1.0)
    code = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(r - mn[:, None], safe[:, None])), 0.0), lv * 1.0).to(tl.int32)
    if BITS == 4:      # nibbles: byte j = code j | code j + 64 << 4
        oj = tl.arange(0, 64)
        lo = tl.gather(code, tl.broadcast_to(oj[None, :], (TB, 64)), axis=1)
        hi = tl.gather(code, tl.broadcast_to((oj + 64)[None, :], (TB, 64)), axis=1)
        tl.store(rp_ptr + row[:, None] * NBY + oj[None, :], (lo | (hi << 4)).to(tl.int8), mask=rmask[:, None])
    else:              # little-endian bit stream: byte j holds stream bits 8j .. 8j+7, i.e. codes c0 = 8j // BITS onwards
        ob = tl.arange(0, 128)                               # output bytes (NBY = 16 * BITS <= 128)
        c0 = (ob * 8) // BITS
        o0 = ob * 8 - c0 * BITS                              # bit offset of byte j inside code c0
        NT: tl.constexpr = (8 + BITS - 1) // BITS + 1        # codes that can touch one byte
        byte = tl.zeros([TB, 128], tl.int32)
        for t in tl.static_range(NT):
            ci = tl.minimum(c0 + t, 127)
            g = tl.gather(code, tl.broadcast_to(ci[None, :], (TB, 128)), axis=1)
            byte = byte | (((g << (BITS * t)) >> o0[None, :]) & 255)
        tl.store(rp_ptr + row[:, None] * NBY + ob[None, :], byte.to(tl.int8), mask=rmask[:, None] & (ob < NBY)[None, :])
    tl.store(rm_ptr + row * 2, mn.to(tl.float16), mask=rmask)
    tl.store(rm_ptr + row * 2 + 1, st.to(tl.float16), mask=rmask)


@triton.jit
def _hcols(x, h_ptr, row_idx, col0, NR: tl.constexpr, NC: tl.constexpr, LD: tl.constexpr, PREC: tl.constexpr):
    """x [TB, NR] @ Hmat[row_idx, col0:col0 + NC] (Hmat row-major with leading dimension LD, fp32)."""
    c = tl.arange(0, NC)
    h = tl.load(h_ptr + row_idx[:, None] * LD + col0 + c[None, :])
    return tl.dot(x, h, input_precision=PREC)


@triton.jit
def _hc(x, hf_ptr, h16_ptr, row_idx, col0, NR: tl.constexpr, NC: tl.constexpr, LD: tl.constexpr, SC: tl.constexpr,
        PREC: tl.constexpr):
    """x [TB, NR] @ Hn[row_idx, col0:col0 + NC] for a normalised Hadamard Hn = Hpm * SC. PREC "f16x2": the exact +-1
    matrix in fp16 and x split into fp16 hi + 2^11-scaled lo (two tensor-core dots, ~22-bit mantissa, fp32 sums);
    otherwise the fp32 matrix with tl.dot(input_precision=PREC)."""
    if PREC == "f16x2":
        c = tl.arange(0, NC)
        h = tl.load(h16_ptr + row_idx[:, None] * LD + col0 + c[None, :])
        xh = x.to(tl.float16)
        xl = ((x - xh.to(tl.float32)) * 2048.0).to(tl.float16)
        return (tl.dot(xh, h) + tl.dot(xl, h) * (1.0 / 2048.0)) * SC
    else:
        return _hcols(x, hf_ptr, row_idx, col0, NR, NC, LD, PREC)


@triton.jit
def aenc_kernel(srck_ptr, srcv_ptr, src_h, src_ts, srow0, pos0, nrows,
                kb_ptr, km_ptr, vb_ptr, vm_ptr, krp_ptr, krm_ptr, vrp_ptr, vrm_ptr, Lc,
                b_ptr, perm_ptr, hh_ptr, h128_ptr, m_ptr, cos_ptr, sin_ptr, gam_ptr, flag_ptr, E4MINv, h64pm_ptr, hpm_ptr,
                TB: tl.constexpr, NG: tl.constexpr, KBITS: tl.constexpr, VBITS: tl.constexpr, PREC: tl.constexpr,
                ZS: tl.constexpr, KV: tl.constexpr = 8, PART: tl.constexpr = 0, RLC=0, ROFF=0):
    """rows srow0 + [0, nrows) of the source cache (KV head bk: srck + bk * src_h + row * src_ts, any float dtype) at
    absolute positions pos0 + [0, nrows) -> cap_codec records at coded index pos - ZS (rows with pos < ZS are sinks:
    skipped). One program = TB rows of one KV head.
    Residual records go to row bk * RL + (pos - ZS - ROFF) of krp / krm / vrp / vrm, RL = RLC if RLC > 0 else Lc
    (defaults: the same rows as the GPU codes). 11-07 staging: StepA passes a GPU staging buffer [BK, ncod, .] with
    RLC = ncod and ROFF = the first coded index, then copies it to pinned host memory in one DMA per buffer (no
    per-row zero-copy PCIe writes, which are ~10x slower on the RTX 5090 of lab05)."""
    bk = tl.program_id(0)
    hk = bk % KV                                             # KV head of this (batch, KV head) row: calibration index
    tb = tl.program_id(1)
    t = tl.arange(0, TB)
    c32 = tl.arange(0, 32)
    c64 = tl.arange(0, 64)
    c128 = tl.arange(0, 128)
    r = tb * TB + t
    rv = r < nrows
    pos = pos0 + r
    ok = rv & (pos >= ZS)
    srow = (srow0 + r).to(tl.int64)
    sk = srck_ptr + bk.to(tl.int64) * src_h + srow[:, None] * src_ts
    sv = srcv_ptr + bk.to(tl.int64) * src_h + srow[:, None] * src_ts
    orow = (bk.to(tl.int64) * Lc + (pos - ZS).to(tl.int64))
    rl = tl.where(RLC > 0, RLC, Lc).to(tl.int64)
    rrow = bk.to(tl.int64) * rl + (pos - ZS - ROFF).to(tl.int64)        # residual record row (== orow by default)
    if PART != 2:
        pa = tl.load(perm_ptr + hk * 128 + c64)
        pb = tl.load(perm_ptr + hk * 128 + 64 + c64)
        kxa = tl.load(sk + pa[None, :], mask=rv[:, None], other=0.0).to(tl.float32)
        kxb = tl.load(sk + pb[None, :], mask=rv[:, None], other=0.0).to(tl.float32)
        # rotary mean at the rows' positions, permuted channels: m_c cos + rot_half(m)_c sin (rc_q.rotate)
        pq = tl.where(pos >= 0, pos, 0).to(tl.int64)
        fa = pa % 64
        fb = pb % 64
        ma = tl.load(m_ptr + hk * 128 + pa)
        mb = tl.load(m_ptr + hk * 128 + pb)
        ra_ = tl.where(pa < 64, -tl.load(m_ptr + hk * 128 + (pa + 64) % 128), tl.load(m_ptr + hk * 128 + (pa + 64) % 128))
        rb_ = tl.where(pb < 64, -tl.load(m_ptr + hk * 128 + (pb + 64) % 128), tl.load(m_ptr + hk * 128 + (pb + 64) % 128))
        rm2 = rv[:, None]
        mta = (ma[None, :] * tl.load(cos_ptr + pq[:, None] * 64 + fa[None, :], mask=rm2, other=0.0)
               + ra_[None, :] * tl.load(sin_ptr + pq[:, None] * 64 + fa[None, :], mask=rm2, other=0.0))
        mtb = (mb[None, :] * tl.load(cos_ptr + pq[:, None] * 64 + fb[None, :], mask=rm2, other=0.0)
               + rb_[None, :] * tl.load(sin_ptr + pq[:, None] * 64 + fb[None, :], mask=rm2, other=0.0))
        ba = tl.load(b_ptr + hk * 128 + pa)
        bb = tl.load(b_ptr + hk * 128 + pb)
        xa = (kxa - mta) * ba[None, :]
        xb = (kxb - mtb) * bb[None, :]
        ha0 = _hc(xa, hh_ptr, h64pm_ptr, c64, 0, 64, 32, 64, 0.125, PREC)
        ha1 = _hc(xa, hh_ptr, h64pm_ptr, c64, 32, 64, 32, 64, 0.125, PREC)
        hb0 = _hc(xb, hh_ptr, h64pm_ptr, c64, 0, 64, 32, 64, 0.125, PREC)
        hb1 = _hc(xb, hh_ptr, h64pm_ptr, c64, 32, 64, 32, 64, 0.125, PREC)
        # ---- hi half: clip search (cap_codec.encode_hi)
        mn = tl.minimum(tl.min(ha0, 1), tl.min(ha1, 1))
        mx = tl.maximum(tl.max(ha0, 1), tl.max(ha1, 1))
        cc = (mn + mx) / 2.0
        hh = (mx - mn) / 2.0
        berr = tl.zeros([TB], tl.float32)
        bq0 = tl.zeros([TB, 32], tl.float32)
        bq1 = tl.zeros([TB, 32], tl.float32)
        bsc = tl.zeros([TB], tl.float32)
        bzr = tl.zeros([TB], tl.float32)
        for gi in tl.static_range(NG):
            gm = tl.load(gam_ptr + gi)
            raw = tl.math.div_rn((2.0 * hh) * gm, 7.0)
            sc = tl.where(raw > 0, tl.maximum(_e4r(raw), E4MINv), 0.0)
            safe = tl.where(sc > 0, sc, 1.0)
            zr = _e4r(tl.math.div_rn(cc - hh * gm, safe))
            lo = zr * sc
            q0 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(ha0 - lo[:, None], safe[:, None])), 0.0), 7.0)
            q1 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(ha1 - lo[:, None], safe[:, None])), 0.0), 7.0)
            x0 = tl.where(sc[:, None] > 0, q0 * sc[:, None] + lo[:, None], ha0)
            x1 = tl.where(sc[:, None] > 0, q1 * sc[:, None] + lo[:, None], ha1)
            err = tl.sum((x0 - ha0) * (x0 - ha0), 1) + tl.sum((x1 - ha1) * (x1 - ha1), 1)
            if gi == 0:
                berr, bq0, bq1, bsc, bzr = err, q0, q1, sc, zr
            else:
                tk = err < berr
                berr = tl.minimum(err, berr)
                bq0 = tl.where(tk[:, None], q0, bq0)
                bq1 = tl.where(tk[:, None], q1, bq1)
                bsc = tl.where(tk, sc, bsc)
                bzr = tl.where(tk, zr, bzr)
        # ---- lo half: mean +- MAD (cap_codec.encode_lo)
        m16 = _f16((tl.sum(hb0, 1) + tl.sum(hb1, 1)) / 64.0)
        mu = _e4r(m16)
        sl = tl.abs(_e4r(_f16((tl.sum(tl.abs(hb0 - m16[:, None]), 1) + tl.sum(tl.abs(hb1 - m16[:, None]), 1)) / 64.0)))
        bt0 = hb0 >= mu[:, None]
        bt1 = hb1 >= mu[:, None]
        kbyte = bq0.to(tl.int32) | (bq1.to(tl.int32) << 3) | (bt0.to(tl.int32) << 6) | (bt1.to(tl.int32) << 7)
        tl.store(kb_ptr + orow[:, None] * 32 + c32[None, :], kbyte.to(tl.uint8), mask=ok[:, None])
        o4 = tl.arange(0, 4)
        kmv = tl.where(o4[None, :] == 0, bsc[:, None], tl.where(o4[None, :] == 1, bzr[:, None], tl.where(o4[None, :] == 2, mu[:, None], sl[:, None])))
        tl.store(km_ptr + orow[:, None] * 4 + o4[None, :], _e4b(kmv), mask=ok[:, None])
        tl.atomic_add(flag_ptr, tl.sum((ok & (bsc == 0)).to(tl.int32), 0), sem="relaxed")
        # ---- key residual: (k - k_hat) H, k_hat = decoded base (original coordinates) + rotary mean
        yh0 = bq0 * bsc[:, None] + (bzr * bsc)[:, None]
        yh1 = bq1 * bsc[:, None] + (bzr * bsc)[:, None]
        yl0 = mu[:, None] + sl[:, None] * tl.where(bt0, 1.0, -1.0)
        yl1 = mu[:, None] + sl[:, None] * tl.where(bt1, 1.0, -1.0)
        xah = _hc(yh0, hh_ptr, h64pm_ptr, c32, 0, 32, 64, 64, 0.125, PREC) + _hc(yh1, hh_ptr, h64pm_ptr, c32 + 32, 0, 32, 64, 64, 0.125, PREC)
        xbh = _hc(yl0, hh_ptr, h64pm_ptr, c32, 0, 32, 64, 64, 0.125, PREC) + _hc(yl1, hh_ptr, h64pm_ptr, c32 + 32, 0, 32, 64, 64, 0.125, PREC)
        rka = kxa - (xah / ba[None, :] + mta)
        rkb = kxb - (xbh / bb[None, :] + mtb)
        rk = (_hc(rka, h128_ptr, hpm_ptr, pa, 0, 64, 128, 128, 0.08838834764831845, PREC)
              + _hc(rkb, h128_ptr, hpm_ptr, pb, 0, 64, 128, 128, 0.08838834764831845, PREC))
        _enc_res(rk, krp_ptr, krm_ptr, rrow, ok, KBITS, TB)
    if PART != 1:
        # ---- values: Hadamard, 2 bits, e4m3 step / zero in steps (cap_codec.encode_v)
        vall = tl.load(sv + c128[None, :], mask=rv[:, None], other=0.0).to(tl.float32)
        yv = _hc(vall, h128_ptr, hpm_ptr, c128, 0, 128, 128, 128, 0.08838834764831845, PREC)       # [TB, 128] = v H
        vmn = tl.min(yv, 1)
        vmx = tl.max(yv, 1)
        vraw = tl.math.div_rn(vmx - vmn, 3.0)
        vsc = tl.where(vraw > 0, tl.maximum(_e4r(vraw), E4MINv), 0.0)
        vsafe = tl.where(vsc > 0, vsc, 1.0)
        vzr = tl.math.div_rn(vmn, vsafe)
        sat = (tl.abs(vzr) > 448.0) & (vsc > 0)
        vze = _e4r(vzr)
        vlo = tl.where(sat, _f16(vmn), vze * vsc)
        w = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(yv - vlo[:, None], vsafe[:, None])), 0.0), 3.0)
        wi = tl.reshape(w.to(tl.int32), [TB, 4, 32])                                           # channel = plane * 32 + j
        vbyte = tl.sum(wi << (2 * tl.arange(0, 4))[None, :, None], axis=1)
        tl.store(vb_ptr + orow[:, None] * 32 + c32[None, :], vbyte.to(tl.uint8), mask=ok[:, None])
        o2 = tl.arange(0, 2)
        vmv = tl.where(o2[None, :] == 0, vsc[:, None], vze[:, None])
        tl.store(vm_ptr + orow[:, None] * 2 + o2[None, :], _e4b(vmv), mask=ok[:, None])
        nbad = tl.sum((ok & (vsc == 0)).to(tl.int32), 0) + tl.sum((ok & sat).to(tl.int32), 0)
        tl.atomic_add(flag_ptr, nbad, sem="relaxed")
        # ---- value residual in Hadamard coordinates: (v - v_hat) H = v H - (code * step + zero * step), H orthogonal
        vdz = vze * vsc                                          # the decoder's zero (decode_v: code * step + zero * step)
        rvh = yv - (w * vsc[:, None] + vdz[:, None])
        _enc_res(rvh, vrp_ptr, vrm_ptr, rrow, ok, VBITS, TB)


def be4(b):
    return b.contiguous().view(torch.float8_e4m3fn).float()


class StepA(KR.Step7R):
    """one layer, B = 1, the reference algorithm on cap_codec bytes (see the module docstring). GH query heads per KV
    head (<= 16; configure(KVH, GH) first; the kernels pad to GP = 16 lanes and mask the rest). Store buffers (coded rows [Z, W0) at index pos - Z):
    kp [KVH, L, 32] key codes, km [KVH, L, 4] key e4m3 metadata, vp [KVH, L, 32] value codes, vm [KVH, L, 2] value e4m3
    metadata (GPU); krp / krm / vrp / vrm residual records (pinned host, or GPU with host=False)."""

    def __init__(self, Lc, GH=4, M=128, K=64, lam=-50, kbits=3, vbits=3, host=True, words=True, seed=0, B=1, rh=None,
                 fused=True):
        assert Lc >= M and 1 <= GH <= GP and GH == K4.G and KVH == K4.KVH and D == 128, (GH, K4.G, KVH, K4.KVH)
        BK = B * KVH
        v = dict(K7.VARIANTS["two128k64"]); mode = v.pop("mode"); v.pop("K")
        v["M"] = M
        kw = dict(v, qbias=False, fuse_w=False, maxnreg=168); kw.update(K7.WF_CFG(Lc, 2, B, "wf"))
        dev = "cuda"
        if rh is not None:       # caller-provided residual buffers (timing: shared by layers within a pinned budget)
            pass
        elif host:
            rh = (torch.zeros(BK, Lc, 16 * kbits, dtype=torch.int8).pin_memory(),
                  torch.zeros(BK, Lc, 2, dtype=torch.float16).pin_memory(),
                  torch.zeros(BK, Lc, 16 * vbits, dtype=torch.int8).pin_memory(),
                  torch.zeros(BK, Lc, 2, dtype=torch.float16).pin_memory())
        else:
            rh = (torch.zeros(BK, Lc, 16 * kbits, dtype=torch.int8, device=dev), torch.zeros(BK, Lc, 2, dtype=torch.float16, device=dev),
                  torch.zeros(BK, Lc, 16 * vbits, dtype=torch.int8, device=dev), torch.zeros(BK, Lc, 2, dtype=torch.float16, device=dev))
        tiny = (torch.zeros(BK, 1, D, dtype=torch.float16).pin_memory(), torch.zeros(BK, 1, D, dtype=torch.float16).pin_memory())
        super().__init__(B, Lc, K, 2, kbits=kbits, vbits=vbits, res_host=rh, words=words, wf=True, win=128, mode=mode,
                         seed=seed, host=tiny, **kw)
        assert self.m8 == 0
        self.GH, self.lam, self.fused = GH, lam, bool(fused)
        self.cpk = -(lam / 200.0) * self.scale * self.scale
        self.km = torch.zeros(BK, Lc, 4, dtype=torch.uint8, device=dev)
        self.vm = torch.zeros(BK, Lc, 2, dtype=torch.uint8, device=dev)
        self.mt = self.mt2 = None                                               # fp16 metadata of the old layout: unused
        self.kw = torch.zeros(BK, WBA, D, dtype=torch.float16, device=dev)
        self.vw = torch.zeros_like(self.kw)
        self.l1 = torch.empty_like(self.l)
        self.W = 0
        self.q.zero_()
        self.Hh = self.H128                                                     # symmetric orthonormal
        self.out2 = torch.empty_like(self.out)
        self.flags = torch.zeros(1, dtype=torch.int32, device=dev)
        # S5 cross-step cache of residual records: current (c*) and next (n*) generation, [BK, M] slots
        self.s5, self.s5_freeze = False, False
        Mc = self.NC2
        mk = lambda w, dt=torch.int8: torch.zeros(BK, Mc, w, dtype=dt, device=dev)  # noqa: E731
        self.ckp, self.ckm, self.nkp, self.nkm = mk(16 * kbits), mk(2, torch.float16), mk(16 * kbits), mk(2, torch.float16)
        self.cvp, self.cvm, self.nvp, self.nvm = mk(16 * vbits), mk(2, torch.float16), mk(16 * vbits), mk(2, torch.float16)
        self.ckidx = torch.full((BK, Mc), -1, dtype=torch.int64, device=dev)
        self.cvidx, self.nkidx, self.nvidx = self.ckidx.clone(), self.ckidx.clone(), self.ckidx.clone()
        self.H64f = self.H64.float().contiguous()
        self.H64pm = (self.H64f * 8).round().half().contiguous()                 # entries exactly +-1
        # scratch lanes og >= GH (GP = 16 padding) and unused split slots are never written by the kernels; start them
        # finite (torch.empty may hand back NaN bit patterns from earlier allocations, and 0 * NaN poisons lane merges)
        for nm_ in ("m", "l", "l1", "acc", "pn", "pz", "pm", "sc", "s", "out", "out2", "qw", "qs2", "qh", "u", "w", "bias"):
            t_ = getattr(self, nm_, None)
            if isinstance(t_, torch.Tensor) and t_.is_floating_point():
                t_.zero_()

    # ---------------------------------------------------------------- data
    def set_calib(self, order, bvec, mpre, asc, inv_freq):
        """order [KVH, D] (xs = x.gather(order)), bvec [KVH, D] balance, mpre [KVH, D] pre-RoPE rotary mean, asc the
        model's attention scaling, inv_freq [D / 2] the model's (as rc_q.R)."""
        self.perm.copy_(order.long())
        self.qbal.copy_(bvec.float())
        self.build_T()
        self.mpre = (mpre.float().cuda() * float(asc)).contiguous()            # decode: q . R(t) (asc m) = bias
        self.m_enc = mpre.float().contiguous().cuda()                           # encode: cos / sin tables carry asc
        self.b_enc = bvec.float().contiguous().cuda()
        self.perm_enc = order.to(torch.int32).contiguous().cuda()
        self.asc, self.invf = float(asc), inv_freq.float().cpu()
        invf64 = inv_freq.double().cpu()
        BB = self.BB
        t0 = (torch.arange(triton.cdiv(self.L, BB), dtype=torch.float64) * BB + Z_SINK)[:, None] * invf64[None]
        self.c0tab.copy_(torch.cat([t0.cos(), t0.cos()], 1).float())
        self.s0tab.copy_(torch.cat([t0.sin(), t0.sin()], 1).float())
        nb = torch.arange(BB, dtype=torch.float64)
        self.csB.copy_(torch.cat([torch.cos(invf64[:, None] * nb[None]), torch.sin(invf64[:, None] * nb[None])], 0).float())
        self.invf64 = invf64.cuda()

    def rope_tables(self, npos):
        """per-position cos / sin [npos, 64] exactly as cap_attn.cos_sin_at (float64 angle, times asc, then fp32)."""
        f = torch.arange(npos, dtype=torch.float64)[:, None] * self.invf64.cpu()[None]
        self.cos_t = (f.cos() * self.asc).float().contiguous().cuda()
        self.sin_t = (f.sin() * self.asc).float().contiguous().cuda()

    def load_store(self, st):
        """copy one layer's cap_attn Store (KVH heads, d = 128, B = 1) into the kernel buffers."""
        Z, W0 = Z_SINK, st.W0
        assert W0 - Z == self.L and self.B == 1, (W0, self.L)
        assert st.kbits == self.kbits and st.vbits == self.vbits
        self.kp.copy_(st.kb[:, Z:W0]); self.vp.copy_(st.vb[:, Z:W0])
        self.km.copy_(st.km[:, Z:W0]); self.vm.copy_(st.vm[:, Z:W0])
        dst = (self.krp, self.krm, self.vrp, self.vrm)
        src = (st.kr[:, Z:W0].view(torch.int8), st.krm[:, Z:W0], st.vr[:, Z:W0].view(torch.int8), st.vrm[:, Z:W0])
        for d_, s_ in zip(dst, src):
            d_.copy_(s_.to(d_.device))
        self.set_exact(st.kx, st.vx)

    def set_exact(self, kx, vx):
        """the exact rows (4 sinks, then the window rows [W0, Lk)) [B * KVH, W, D]."""
        W = kx.shape[1]
        assert W <= WBA
        self.kw.zero_(); self.vw.zero_()
        self.kw[:, :W].copy_(kx.half())
        self.vw[:, :W].copy_(vx.half())
        self.W = W

    enc_warps = 4                    # aenc_prof (RTX 4080): f16x2 TB 16 / 4 warps 0.76 ms per 8K rows and layer (tf32x3 2.2, ieee 2.9)
    # 11-07 encoder write-back mode for host residuals: False = the kernel writes the records straight to pinned host
    # memory (zero-copy, per-row 48 + 4 B PCIe writes: fine on the 4080 / 4090, ~10x slower on the lab05 RTX 5090,
    # aenc_parts2); True = the kernel writes a GPU staging buffer and one D2H DMA per buffer copies it to the pinned
    # store on the *current* stream (ordered after the kernel; overlaps the forward when encode runs on a side stream).
    # Set per device (perception module / caller); bytes in the store are identical either way.
    enc_stage = False
    enc_stage_min = 512              # staged only for >= this many coded rows (prefill); short appends (decode, window
                                     # flushes) keep the zero-copy path: 4 x BK small DMAs per call would cost more
    _STAGE = {}                      # (device, stream, BK, kbits, vbits) -> flat GPU buffers shared by all layers
    n_staged = 0                     # staged launches (instance counter; tests assert the staged path really ran)

    def _stage_views(self, ncod):
        """GPU staging views [BK, ncod, w] (contiguous) for krp / krm / vrp / vrm, from a buffer shared by StepA objects
        with the same geometry on the same stream (stream order makes reuse across layers safe)."""
        st = torch.cuda.current_stream()
        key = (st.device, st.cuda_stream, self.BK, self.kbits, self.vbits)
        ws = (16 * self.kbits, 2, 16 * self.vbits, 2)
        dts = (self.krp.dtype, self.krm.dtype, self.vrp.dtype, self.vrm.dtype)
        buf = StepA._STAGE.get(key)
        if buf is None or buf[0].numel() < self.BK * ncod * ws[0]:
            cap = max(ncod, 1024)
            buf = StepA._STAGE[key] = tuple(torch.empty(self.BK * cap * w, dtype=dt, device=st.device) for w, dt in zip(ws, dts))
        return tuple(b[:self.BK * ncod * w].view(self.BK, ncod, w) for b, w in zip(buf, ws))

    def _launch_enc(self, k, v, src_h, src_ts, srow0, pos0, n, g, prec, TB):
        """aenc_kernel over rows [0, n) at positions pos0 + i; residual records zero-copy or staged (enc_stage)."""
        # never inside CUDA-graph capture (the staging buffer must not be allocated in a graph pool / keyed by the
        # capture stream; e2e_v2 / kalign_time capture encode): captured encodes use the zero-copy path
        stage = self.enc_stage and not self.krp.is_cuda and not torch.cuda.is_current_stream_capturing()
        c0 = max(pos0, Z_SINK) - Z_SINK                  # first coded index written
        ncod = pos0 + n - max(pos0, Z_SINK)              # coded rows written (sinks skipped)
        if ncod < max(1, self.enc_stage_min):
            stage = False
        if ncod > 0:
            assert c0 + ncod <= self.L, (pos0, n, self.L)
        if stage:
            assert self.krp.is_pinned(), "enc_stage needs a pinned host store (async D2H)"
            self.n_staged += 1
            rk, rkm, rv_, rvm = self._stage_views(ncod)
            rlc, roff = ncod, c0
        else:
            rk, rkm, rv_, rvm = self.krp_, self.krm_, self.vrp_, self.vrm_
            rlc, roff = 0, 0
        aenc_kernel[(self.BK, triton.cdiv(n, TB))](
            k, v, src_h, src_ts, srow0, pos0, n, self.kp, self.km, self.vp, self.vm, rk, rkm, rv_,
            rvm, self.L, self.b_enc, self.perm_enc, self.H64f, self.H128, self.m_enc, self.cos_t, self.sin_t, g,
            self.flags, E4MIN, self.H64pm, self.Hpm, TB=TB, NG=g.numel(), KBITS=self.kbits, VBITS=self.vbits, PREC=prec,
            ZS=Z_SINK, KV=KVH, PART=0, RLC=rlc, ROFF=roff, num_warps=self.enc_warps, enable_fp_fusion=False)
        if stage:
            for h, sv in zip((self.krp, self.krm, self.vrp, self.vrm), (rk, rkm, rv_, rvm)):
                if c0 == 0 and ncod == h.shape[1]:
                    h.copy_(sv, non_blocking=True)                  # whole store: one contiguous DMA
                else:
                    for b in range(self.BK):                         # row ranges are contiguous per (batch, KV head)
                        h[b, c0:c0 + ncod].copy_(sv[b], non_blocking=True)

    def encode(self, k, v, r0, r1, gammas, prec="f16x2", TB=16):
        """encode cache rows [r0, r1) of k, v [B * KVH, T, D] (contiguous, any float dtype; row index = position) into the
        store (rows < Z skipped). Needs set_calib, rope_tables(>= r1) and gammas (the stack's clip candidates)."""
        assert k.is_contiguous() and v.is_contiguous() and k.shape == v.shape and r1 - Z_SINK <= self.L
        key = tuple(float(x) for x in gammas)
        if getattr(self, "_gkey", None) != key:          # cached: no host-to-device copy inside a captured graph
            self._gkey, self._gam = key, torch.tensor(list(key), dtype=torch.float32, device="cuda")
        g = self._gam
        n = r1 - r0
        if n <= 0:
            return
        self._launch_enc(k, v, k.shape[1] * D, D, r0, r0, n, g, prec, TB)

    def encode_proj(self, k, v, pos0, gammas, prec="f16x2", TB=16):
        """prefill: encode the rows of k / v in the projection layout [B = 1, n, KVH * D] (token-major, as k_proj / v_proj
        output them) at absolute positions pos0 .. pos0 + n - 1 (rows < Z skipped)."""
        assert self.B == 1 and k.is_contiguous() and v.is_contiguous() and k.shape[-1] == KVH * D
        n = k.shape[1]
        key = tuple(float(x) for x in gammas)
        if getattr(self, "_gkey", None) != key:
            self._gkey, self._gam = key, torch.tensor(list(key), dtype=torch.float32, device="cuda")
        self._launch_enc(k, v, D, KVH * D, 0, pos0, n, self._gam, prec, TB)

    def fuse_records(self, stride_k, stride_v=None):
        """fused copies of the encoded residual records (data, fp16 min, fp16 step, zero padding to the stride in bytes;
        keys stride_k, values stride_v, default stride_k), in the same memory kind as the split records (pinned host, or
        GPU); fix() then reads them (S5 must be off)."""
        assert not self.s5
        stride_v = stride_k if stride_v is None else stride_v
        out = []
        for (rp, rm), stride_bytes in (((self.krp, self.krm), stride_k), ((self.vrp, self.vrm), stride_v)):
            nby = rp.shape[-1]
            assert stride_bytes % 4 == 0 and stride_bytes >= nby + 4
            host = not rp.is_cuda
            f = torch.zeros(rp.shape[0], rp.shape[1], stride_bytes, dtype=torch.int8,
                            **(dict(pin_memory=True) if host else dict(device=rp.device)))
            f[..., :nby].copy_(rp)
            f[..., nby:nby + 4].copy_(rm.contiguous().view(torch.int8).view(rp.shape[0], rp.shape[1], 4))
            out.append((f, KR.K4.host_as_cuda(f) if host else f))
        (self.kf, self.kf_), (self.vf, self.vf_) = out
        self.rec_fused, self.rswk, self.rswv = True, stride_k // 4, stride_v // 4

    def unfuse_records(self):
        self.rec_fused = False

    def s5_reset(self):
        for t in (self.ckidx, self.cvidx, self.nkidx, self.nvidx):
            t.fill_(-1)

    def s5_preload(self, frac=0.7, seed=0):
        """timing at a measured reuse rate: one step fills the cache with this step's candidates / selected rows, then a
        fraction 1 - frac of the slots is emptied (misses) and the cache is frozen (no copy-back), so every replay of the
        same step hits on `frac` of the candidates and of the selected rows."""
        self.s5, self.s5_freeze = True, False
        self.s5_reset()
        self.step(); torch.cuda.synchronize()
        g = torch.Generator(device="cpu").manual_seed(seed)
        drop = (torch.rand(self.ckidx.shape, generator=g) >= frac).to(self.ckidx.device)
        self.ckidx.masked_fill_(drop, -1)
        self.cvidx.masked_fill_(drop, -1)
        self.s5_freeze = True

    def set_q(self, q):
        """q [B * KVH * GH, D] post-RoPE (any float dtype)."""
        GH = self.GH
        qh = q.half()
        self.q.view(-1)[: self.BK * GH * D].copy_(qh.reshape(-1))
        qf = qh.float().view(self.BK, GH, D)
        mp = self.mpre.repeat(self.B, 1)
        ma, mb = mp[:, :D // 2], mp[:, D // 2:]
        qa, qb = qf[..., :D // 2], qf[..., D // 2:]
        ca = (qa * ma[:, None] + qb * mb[:, None]).reshape(-1, D // 2)
        cb = (qb * ma[:, None] - qa * mb[:, None]).reshape(-1, D // 2)
        n = self.BK * GH
        self.u[:n].copy_(torch.cat([ca, cb], 1)); self.w[:n].copy_(torch.cat([cb, -ca], 1))

    # ---------------------------------------------------------------- step
    def base(self):
        if self.fused:
            return self.base_fused()
        GH = self.GH
        K7.qprep_kernel[(self.BK,)](self.q, self.ibp, self.H64h, self.perm32, self.qw, self.qs2, GH=GH, GP=GP, D4=D, KV=KVH,
                                    CB=32, num_warps=4)
        ntile = triton.cdiv(self.L, self.BB)
        npb = max(1, min(ntile, -(-self.bppsm * self.nsm // self.BK)))
        K4.bias_kernel[(self.BK, npb)](self.u, self.w, self.c0tab, self.s0tab, self.csB, self.bias, self.L, ntile, self.scale,
                                       GH=GH, GP=GP, D=D, BB=self.BB, MODE="fp16", num_warps=4)
        self.kinfo["base"] = abase_kernel[(self.BK, self.nsb)](
            self.qw, self.qs2, self.kp, self.vp, self.km, self.vm, self.bias, self.m, self.l, self.l1, self.acc, self.s,
            self.L, self.split_len, self.nsplit, self.scale, self.cpk, GH=GH, GP=GP, P=P, BLOCK_N=self.BLOCK_N,
            MAGIC=self.magic, LAM=bool(self.lam), num_warps=self.warps, num_stages=self.stages,
            **({"maxnreg": self.maxnreg} if self.maxnreg else {}))
        awin_kernel[(self.BK,)](self.q, self.kw, self.vw, self.Hpm, self.m, self.l, self.l1, self.acc, self.W, self.nsb,
                                self.nsplit, self.scale, GH=GH, GP=GP, D=D, WB=WBA, num_warps=4)

    def base_fused(self):
        """two launches per layer step: query side + H q + rotary bias (aprep), exact split + quantized splits (abasew)."""
        GH = self.GH
        ntile = triton.cdiv(self.L, self.BB)
        npb = max(1, min(ntile, -(-self.bppsm * self.nsm // self.BK)))
        aprep_kernel[(self.BK, npb + 1)](self.u, self.w, self.c0tab, self.s0tab, self.csB, self.bias, self.L, ntile, npb, self.scale,
                                         self.q, self.ibp, self.H64h, self.perm32, self.qw, self.qs2, self.Hpm, self.qh, GH=GH, GP=GP,
                                         D=D, BB=self.BB, KV=KVH, CB=32, num_warps=4)
        self.kinfo["base"] = abasew_kernel[(self.BK, self.nsb + 1)](
            self.qw, self.qs2, self.kp, self.vp, self.km, self.vm, self.bias, self.m, self.l, self.l1, self.acc, self.s,
            self.L, self.split_len, self.nsplit, self.scale, self.cpk, self.q, self.kw, self.vw, self.Hpm, self.W, self.nsb,
            GH=GH, GP=GP, P=P, BLOCK_N=self.BLOCK_N, MAGIC=self.magic, LAM=bool(self.lam), WB=WBA, num_warps=self.warps,
            num_stages=self.stages, **({"maxnreg": self.maxnreg} if self.maxnreg else {}))

    def select(self):
        K4.sel_stage1[(self.BK, self.ntile)](self.s, self.Z, self.Z, self.cand, self.L, self.ntile, self.m, self.l1, self.nsplit,
                                             GH=self.GH, GP=GP, T=self.T, K=self.KT, BISECT=self.bisect, FUSE=True, NS=self.NS,
                                             RADIX=self.radix, num_warps=4)
        C = self.ntile * self.KT
        assert C <= 16384
        K4.sel_stage2[(self.BK,)](self.cand, self.idx, C, K=self.K, CP=triton.next_power_of_2(C), RADIX=self.radix, num_warps=8)

    def fix(self):
        GH = self.GH
        if not self.fused:
            torch.matmul(self.q.float(), self.H128, out=self.qh)
        assert not self.s5 or self.words, "the S5 cache path reads records as words"
        fz = getattr(self, "rec_fused", False)
        assert not (fz and self.s5), "fused records and the S5 cache are exclusive"
        afix2A[(self.BK, self.NC2 // self.kca)](self.qh, self.qw, self.qs2, self.idx, self.kf_ if fz else self.krp_, self.krm_,
                                                self.kp, self.km,
                                                self.vm, self.bias, self.sc, self.L, self.scale, self.cpk,
                                                self.ckp, self.ckm, self.ckidx, self.nkp, self.nkm, self.nkidx, GH=GH, GP=GP, P=P,
                                                M=self.NC2, KC=self.kca, KB=self.kbits, WORDS=self.words, LAM=bool(self.lam),
                                                CACHE=self.s5, FUSED=fz, RSW=self.rswk if fz else 0, num_warps=4)
        afix2B[(self.BK, self.nch2)](self.m, self.l, self.sc, self.idx, self.vf_ if fz else self.vrp_, self.vrm_, self.vp,
                                     self.km, self.vm, self.pn,
                                     self.pz, self.pm, self.selm, self.L, self.nsplit, self.kap,
                                     self.cvp, self.cvm, self.cvidx, self.nvp, self.nvm, self.nvidx, GH=GH, GP=GP, P=P, M=self.NC2,
                                     K=self.Kv, CC=self.cc, NS=self.NS, MP=triton.next_power_of_2(self.NC2), RADIX=self.radix,
                                     VB=self.vbits, WORDS=self.words, CACHE=self.s5, FUSED=fz, RSW=self.rswv if fz else 0,
                                     num_warps=4)
        if self.s5 and not self.s5_freeze:          # next generation becomes current (device copies, graph-capturable)
            for c_, n_ in ((self.ckp, self.nkp), (self.ckm, self.nkm), (self.ckidx, self.nkidx),
                           (self.cvp, self.nvp), (self.cvm, self.nvm), (self.cvidx, self.nvidx)):
                c_.copy_(n_)
        if self.fused:
            afixB_kernel[(self.BK,)](self.m, self.l, self.acc, self.pn, self.pz, self.pm, self.Hpm, self.out2, self.nsplit, self.nch2,
                                     GH=GH, GP=GP, D=D, NS=self.NS, NC=triton.next_power_of_2(self.nch2), UNROLL=self.unroll,
                                     num_warps=4)
            return self.out2
        K4.fixB_kernel[(self.BK, 4)](self.m, self.l, self.acc, self.pn, self.pz, self.pm, self.out, self.nsplit, self.nch2,
                                     GH=GH, GP=GP, P=P, NS=self.NS, NC=triton.next_power_of_2(self.nch2), UNROLL=self.unroll,
                                     num_warps=4)
        torch.matmul(self.out, self.Hh, out=self.out2)                           # back from Hadamard coordinates
        return self.out2

    def result(self):
        """[B * KVH * GH, D] fp32 output of the last step."""
        return self.out2.view(-1)[: self.BK * self.GH * D].view(self.BK * self.GH, D)


def calib_of(li, S10, P9, P10, P11, Q7):
    """the stack's static calibration of layer li as StepA.set_calib arguments (rotary-centred layers only)."""
    import cap_attn as CA
    f, n7 = P10.split(S10.STACK)
    rc = bool(f["rc"]) and li not in P11.NOPE
    _, b = CA.centre_at(li, None, KVH, D, "cuda", S10, P9, P10, P11, Q7)
    if not rc:
        # static (non-rotary) mean m: exactly the rotary form with zero frequency and unit scaling (cos 1, sin 0, so
        # R(t) m = m at every position); cap_attn.centre_at returns mt = Q5.C["m"][li] for these layers
        mt, _ = CA.centre_at(li, torch.zeros(1, dtype=torch.long), KVH, D, "cuda", S10, P9, P10, P11, Q7)
        return S10.INFO[li]["order"], b[:, 0, :], mt[:, 0, :].float().cuda(), 1.0, torch.zeros(D // 2)
    R = P9.RC.R
    assert R["inv_freq"].numel() * 2 == D, "partial rotary not implemented"
    return S10.INFO[li]["order"], b[:, 0, :], R["m_pre"][li].float().cuda(), float(R["asc"]), R["inv_freq"].float()
