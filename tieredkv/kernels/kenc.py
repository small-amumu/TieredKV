# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""kenc (TieredKV, S1 fused cache encode; 2026-11-07, v2 after 3 reviews). One Triton kernel writes tokens
into exactly the layered-cache layout the kbench7 decode kernels read (Step7, water-filled key base, m8 = 0, rc 2),
replacing the ~50 torch ops per layer of lat_overhead.Enc.
Per token t (base position) and KV head (D = 128, P = 32):
  centring (rc 2) kc = k - R(t) m, m the pre-RoPE key mean of the KV head (Step.mpre), R(t) the RoPE rotation (rotate_half
          pairs c, c+64, frequency c % 64), cos / sin of t composed from the decode side's own fp64-derived tables
          (c0tab: tile start, csB: in-tile offset, BB tokens per tile), so encoder and decoder use the same angles; the
          decoder adds q . R(t) m back as the rotary bias
  keys    x = (kc * b)[perm] (balance b, channel permutation), hi = x[:64] @ H64, lo = x[64:] @ H64 (entries +-1/8);
          hi: mn = fp16(min), sc = max(fp16((max - mn) / 7), EPS), code = clamp(rint((hi - mn) / sc), 0, 7);
          lo: mu = fp16(mean), s = fp16(mean |lo - mu|), bit = lo >= mu; byte j = c[j] | c[j+32] << 3 | bit[j] << 6 |
          bit[j+32] << 7; mt[..., 0:2] = (sc, mn); mt2 = fp16(mu - s, 2 s)       (kbench7.encode_wf, plus centring)
  values  vz = fp16(min), vs = max(fp16((max - vz) / 3), EPS), code = clamp(rint((v - vz) / vs), 0, 3); byte j = codes of
          channels j, j+32, j+64, j+96 at bits 0, 2, 4, 6; mt[..., 2:4] = (vs, vz)  (kbench5._deq)
  exact   the original fp16 K / V row stored to pinned host memory through a host-mapped view (UVA zero-copy)
EPS = 2^-14 (the smallest normal fp16), so the clamped scale is the stored scale. Divisions are IEEE (div_rn); fp32 sums
are reduced in a different order than torch, so mu / sd may differ from the torch reference by one fp16 ulp at ties: the
check reports exact mismatch counts per field and gates on decode-side compatibility.
Modes (all CUDA-graph capturable; positions come from one device counter per model, advanced once per decode step):
  step    one launch per layer and step: the token leaving the FP16 window (ring slot n % WB, n = tokens evicted so far,
          read from the counter) is encoded at base position n, then the new token is written into that slot (SWAP;
          loads before the barrier, stores after)
  prefill grid (BK, L / TB): tokens straight from the k_proj / v_proj output layout [B, L, KVH * D]; the Hadamard dots
          use tf32x3 (three-pass tf32, ~fp32 accuracy, 28x faster than ieee fp32 here); exact rows either written by the
          kernel through the host-mapped view (host=True) or copied by the DMA engine (host=False + host_dma)
Batch assumption: all sequences of a batch advance in lockstep (one counter); variable lengths need a per-sequence
counter array (not implemented).
Usage: python kenc.py --check | --time [--out f]"""
import argparse
import hashlib
import json
import os
import statistics as S
import subprocess
import sys

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kbench4 as K4  # noqa: E402
import kbench5 as K5  # noqa: E402
import kbench7 as K7  # noqa: E402

D, P, KVH, WB = K7.D, K7.P, K7.KVH, K7.WB
EPS = 2.0 ** -14
assert D == 128 and P == 32


@triton.jit
def _hmul(x, h_ptr, col0, DOT: tl.constexpr, PREC: tl.constexpr = "ieee"):
    """x [TB, 64] fp32 @ H64[:, col0:col0+32] -> [TB, 32] fp32 (ieee)."""
    r = tl.arange(0, 64)
    c = tl.arange(0, 32)
    h = tl.load(h_ptr + r[:, None] * 64 + col0 + c[None, :])
    if DOT:
        return tl.dot(x, h, input_precision=PREC)
    return tl.sum(x[:, :, None] * h[None, :, :], axis=1)


@triton.jit
def _rcentre(kx, p, pos_t, m_ptr, kvh, c0_ptr, s0_ptr, cs_ptr, BB: tl.constexpr):
    """kx [TB, 64] keys of permuted channels p [64] at base positions pos_t [TB] -> kx - R(t) m (rotate_half RoPE)."""
    f = p % 64
    partner = (p + 64) % 128
    sgn = tl.where(p < 64, -1.0, 1.0)
    mp = tl.load(m_ptr + kvh * 128 + p)
    mq = tl.load(m_ptr + kvh * 128 + partner)
    tile = pos_t // BB
    off = pos_t % BB
    c0 = tl.load(c0_ptr + tile[:, None] * 128 + p[None, :])
    s0 = tl.load(s0_ptr + tile[:, None] * 128 + p[None, :])
    cn = tl.load(cs_ptr + f[None, :] * BB + off[:, None])
    sn = tl.load(cs_ptr + (64 + f[None, :]) * BB + off[:, None])
    cos = c0 * cn - s0 * sn
    sin = s0 * cn + c0 * sn
    return kx - (mp[None, :] * cos + sgn[None, :] * mq[None, :] * sin)


@triton.jit
def _hrows(y, h_ptr, row0):
    """y [TB, 32] @ H64[row0:row0+32, :] -> [TB, 64] (rows of the symmetric H64: the inverse transform of one half)."""
    r = tl.arange(0, 32)
    c = tl.arange(0, 64)
    h = tl.load(h_ptr + (row0 + r)[:, None] * 64 + c[None, :])
    return tl.sum(y[:, :, None] * h[None, :, :], axis=1)


@triton.jit
def _res_store(r, BITS: tl.constexpr, rp_ptr, rm_ptr, rowi, scr):
    """one token's residual r [1, 128] (Hadamard coordinates) -> cap_codec.encode_res record: fp16 min / step, b-bit codes
    packed as a little-endian bit stream (nibbles j | j+64 << 4 when b = 4). rowi: scalar int64 row; scr: 128-int scratch."""
    lv = (1 << BITS) - 1
    c128 = tl.arange(0, 128)
    mnr = tl.min(r, 1).to(tl.float16).to(tl.float32)
    mxr = tl.max(r, 1).to(tl.float16).to(tl.float32)
    st = tl.math.div_rn(mxr - mnr, lv * 1.0).to(tl.float16).to(tl.float32)
    safe = tl.where(st > 0, st, 1.0)
    code = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(r - mnr[:, None], safe[:, None])), 0.0), lv * 1.0).to(tl.int32)
    tl.store(scr + c128[None, :], code)
    tl.debug_barrier()
    NBY: tl.constexpr = 16 * BITS
    c64 = tl.arange(0, 64)
    if BITS == 4:
        lo = tl.load(scr + c64)
        hi = tl.load(scr + 64 + c64)
        tl.store(rp_ptr + rowi * NBY + c64, (lo | (hi << 4)).to(tl.int8))
    else:
        p = tl.arange(0, 1024)                                  # up to 128 codes x 8 bits
        pm = p < NBY * 8
        cc = tl.load(scr + p // BITS, mask=pm, other=0)
        bit = tl.where(pm, (cc >> (p % BITS)) & 1, 0)
        byte = tl.sum(tl.reshape(bit, [128, 8]) << tl.arange(0, 8)[None, :], axis=1)
        tl.store(rp_ptr + rowi * NBY + c128, byte.to(tl.int8), mask=c128 < NBY)
    tl.store(rm_ptr + rowi * 2 + c64, tl.where(c64 == 0, tl.sum(mnr, 0), tl.sum(st, 0)).to(tl.float16), mask=c64 < 2)
    tl.debug_barrier()


@triton.jit
def enc_kernel(srck_ptr, srcv_ptr, src_b, src_h, src_ts, newk_ptr, newv_ptr, cnt_ptr, base0,
               kp_ptr, vp_ptr, mt_ptr, mt2_ptr, kh_ptr, vh_ptr, Lcap,
               b_ptr, perm_ptr, h_ptr, m_ptr, c0_ptr, s0_ptr, cs_ptr, EPSv,
               TB: tl.constexpr, DOT: tl.constexpr, SWAP: tl.constexpr, KV: tl.constexpr, RC: tl.constexpr,
               BB: tl.constexpr, WBc: tl.constexpr, HOST: tl.constexpr = True, PREC: tl.constexpr = "ieee",
               krp_ptr=None, krm_ptr=None, vrp_ptr=None, vrm_ptr=None, scr_ptr=None, h128_ptr=None,
               RES: tl.constexpr = False, KBITS: tl.constexpr = 3, VBITS: tl.constexpr = 3):
    tl.static_assert((not SWAP) or TB == 1)
    bk = tl.program_id(0)
    kvh = bk % KV
    tb = tl.program_id(1)
    t = tl.arange(0, TB)
    c32 = tl.arange(0, 32)
    c64 = tl.arange(0, 64)
    c128 = tl.arange(0, 128)
    if SWAP:                                                 # step mode: counter n = tokens evicted so far
        n = tl.load(cnt_ptr)
        pos = n
        slot0 = n % WBc
    else:                                                    # prefill: base positions base0 + row
        pos = base0 + tb * TB
        slot0 = tb * TB
    pos_t = pos + t                                          # [TB] base positions
    srow = (slot0 + t).to(tl.int64)
    sbase = (bk // KV).to(tl.int64) * src_b + kvh * src_h
    sk = srck_ptr + sbase + srow[:, None] * src_ts
    sv = srcv_ptr + sbase + srow[:, None] * src_ts
    pa = tl.load(perm_ptr + kvh * 128 + c64)
    pb = tl.load(perm_ptr + kvh * 128 + 64 + c64)
    ba = tl.load(b_ptr + kvh * 128 + pa)
    bb = tl.load(b_ptr + kvh * 128 + pb)
    kxa = tl.load(sk + pa[None, :]).to(tl.float32)
    kxb = tl.load(sk + pb[None, :]).to(tl.float32)
    k_all = tl.load(sk + c128[None, :])                                         # exact fp16 row (host copy)
    v0 = tl.load(sv + c32[None, :]).to(tl.float32)
    v1 = tl.load(sv + 32 + c32[None, :]).to(tl.float32)
    v2 = tl.load(sv + 64 + c32[None, :]).to(tl.float32)
    v3 = tl.load(sv + 96 + c32[None, :]).to(tl.float32)
    v_all = tl.load(sv + c128[None, :])
    if SWAP:                                                 # the new token takes the evicted token's window slot
        tl.debug_barrier()
        nk = tl.load(newk_ptr + bk * 128 + c128)
        nv = tl.load(newv_ptr + bk * 128 + c128)
        tl.store(srck_ptr + sbase + slot0.to(tl.int64) * src_ts + c128, nk)
        tl.store(srcv_ptr + sbase + slot0.to(tl.int64) * src_ts + c128, nv)
    if RC == 2:
        kxa = _rcentre(kxa, pa, pos_t, m_ptr, kvh, c0_ptr, s0_ptr, cs_ptr, BB)
        kxb = _rcentre(kxb, pb, pos_t, m_ptr, kvh, c0_ptr, s0_ptr, cs_ptr, BB)
    kxa = kxa * ba[None, :]
    kxb = kxb * bb[None, :]
    ha0 = _hmul(kxa, h_ptr, 0, DOT, PREC)                          # hi channels 0..31
    ha1 = _hmul(kxa, h_ptr, 32, DOT, PREC)                         # hi channels 32..63
    hb0 = _hmul(kxb, h_ptr, 0, DOT, PREC)                          # lo channels 0..31
    hb1 = _hmul(kxb, h_ptr, 32, DOT, PREC)
    mn = tl.minimum(tl.min(ha0, 1), tl.min(ha1, 1))
    mx = tl.maximum(tl.max(ha0, 1), tl.max(ha1, 1))
    mn = mn.to(tl.float16).to(tl.float32)
    sc = tl.maximum(tl.math.div_rn(mx - mn, 7.0).to(tl.float16).to(tl.float32), EPSv)
    c0 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(ha0 - mn[:, None], sc[:, None])), 0.0), 7.0).to(tl.int32)
    c1 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(ha1 - mn[:, None], sc[:, None])), 0.0), 7.0).to(tl.int32)
    mu = ((tl.sum(hb0, 1) + tl.sum(hb1, 1)) / 64.0).to(tl.float16).to(tl.float32)
    sd = ((tl.sum(tl.abs(hb0 - mu[:, None]), 1) + tl.sum(tl.abs(hb1 - mu[:, None]), 1)) / 64.0).to(tl.float16).to(tl.float32)
    b0 = (hb0 >= mu[:, None]).to(tl.int32)
    b1 = (hb1 >= mu[:, None]).to(tl.int32)
    byte = c0 | (c1 << 3) | (b0 << 6) | (b1 << 7)
    vmn = tl.minimum(tl.minimum(tl.min(v0, 1), tl.min(v1, 1)), tl.minimum(tl.min(v2, 1), tl.min(v3, 1)))
    vmx = tl.maximum(tl.maximum(tl.max(v0, 1), tl.max(v1, 1)), tl.maximum(tl.max(v2, 1), tl.max(v3, 1)))
    vz = vmn.to(tl.float16).to(tl.float32)
    vs = tl.maximum(tl.math.div_rn(vmx - vz, 3.0).to(tl.float16).to(tl.float32), EPSv)
    q0 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(v0 - vz[:, None], vs[:, None])), 0.0), 3.0).to(tl.int32)
    q1 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(v1 - vz[:, None], vs[:, None])), 0.0), 3.0).to(tl.int32)
    q2 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(v2 - vz[:, None], vs[:, None])), 0.0), 3.0).to(tl.int32)
    q3 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(v3 - vz[:, None], vs[:, None])), 0.0), 3.0).to(tl.int32)
    vbyte = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
    row = bk.to(tl.int64) * Lcap + pos_t.to(tl.int64)                           # [TB]
    tl.store(kp_ptr + row[:, None] * 32 + c32[None, :], byte.to(tl.uint8))
    tl.store(vp_ptr + row[:, None] * 32 + c32[None, :], vbyte.to(tl.uint8))
    tl.store(mt_ptr + row * 4 + 0, sc.to(tl.float16))
    tl.store(mt_ptr + row * 4 + 1, mn.to(tl.float16))
    tl.store(mt_ptr + row * 4 + 2, vs.to(tl.float16))
    tl.store(mt_ptr + row * 4 + 3, vz.to(tl.float16))
    tl.store(mt2_ptr + row * 2 + 0, (mu - sd).to(tl.float16))
    tl.store(mt2_ptr + row * 2 + 1, (2.0 * sd).to(tl.float16))
    if RES:      # residual records (cap_codec layout) instead of exact rows; step mode only
        tl.static_assert(TB == 1)
        A16 = (mu - sd).to(tl.float16).to(tl.float32)
        B16 = (2.0 * sd).to(tl.float16).to(tl.float32)
        yh0 = c0.to(tl.float32) * sc[:, None] + mn[:, None]
        yh1 = c1.to(tl.float32) * sc[:, None] + mn[:, None]
        yl0 = A16[:, None] + B16[:, None] * b0.to(tl.float32)
        yl1 = A16[:, None] + B16[:, None] * b1.to(tl.float32)
        xa_hat = _hrows(yh0, h_ptr, 0) + _hrows(yh1, h_ptr, 32)                 # decoded base, permuted / balanced
        xb_hat = _hrows(yl0, h_ptr, 0) + _hrows(yl1, h_ptr, 32)
        ra = (kxa - xa_hat) / ba[None, :]                                       # centred residual, permuted channels
        rb = (kxb - xb_hat) / bb[None, :]
        hpa = tl.load(h128_ptr + pa[:, None] * 128 + c128[None, :])            # H128 rows of the permuted channels
        hpb = tl.load(h128_ptr + pb[:, None] * 128 + c128[None, :])
        rk = tl.sum(ra[:, :, None] * hpa[None, :, :], axis=1) + tl.sum(rb[:, :, None] * hpb[None, :, :], axis=1)
        h0 = tl.load(h128_ptr + c32[:, None] * 128 + c128[None, :])
        h1 = tl.load(h128_ptr + (32 + c32)[:, None] * 128 + c128[None, :])
        h2 = tl.load(h128_ptr + (64 + c32)[:, None] * 128 + c128[None, :])
        h3 = tl.load(h128_ptr + (96 + c32)[:, None] * 128 + c128[None, :])
        r0 = v0 - (q0.to(tl.float32) * vs[:, None] + vz[:, None])
        r1 = v1 - (q1.to(tl.float32) * vs[:, None] + vz[:, None])
        r2 = v2 - (q2.to(tl.float32) * vs[:, None] + vz[:, None])
        r3 = v3 - (q3.to(tl.float32) * vs[:, None] + vz[:, None])
        rv = (tl.sum(r0[:, :, None] * h0[None, :, :], axis=1) + tl.sum(r1[:, :, None] * h1[None, :, :], axis=1)
              + tl.sum(r2[:, :, None] * h2[None, :, :], axis=1) + tl.sum(r3[:, :, None] * h3[None, :, :], axis=1))
        rowi = tl.sum(row, 0)
        _res_store(rk, KBITS, krp_ptr, krm_ptr, rowi, scr_ptr + bk * 256)
        _res_store(rv, VBITS, vrp_ptr, vrm_ptr, rowi, scr_ptr + bk * 256 + 128)
    if HOST:
        tl.store(kh_ptr + row[:, None] * 128 + c128[None, :], k_all)
        tl.store(vh_ptr + row[:, None] * 128 + c128[None, :], v_all)


@triton.jit
def app16_kernel(newk_ptr, newv_ptr, kc_ptr, vc_ptr, cnt_ptr, Lcap, KV: tl.constexpr):
    """fp16 baseline, one launch per layer: write the new token's K and V rows at position cnt into [BK, Lcap, D]."""
    bk = tl.program_id(0)
    c = tl.arange(0, 128)
    n = tl.load(cnt_ptr).to(tl.int64)
    row = bk.to(tl.int64) * Lcap + n
    tl.store(kc_ptr + row * 128 + c, tl.load(newk_ptr + bk * 128 + c))
    tl.store(vc_ptr + row * 128 + c, tl.load(newv_ptr + bk * 128 + c))


class Rot:
    """the decode side's rotary tables (kbench4.Step: c0tab / s0tab tile starts, csB in-tile offsets), built the same way."""
    def __init__(self, Lcap, BB=64, theta=500000):
        invf64 = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
        t0 = torch.arange(triton.cdiv(Lcap, BB), dtype=torch.float64)[:, None] * BB * invf64[None]
        self.c0tab = torch.cat([t0.cos(), t0.cos()], 1).float().contiguous().cuda()
        self.s0tab = torch.cat([t0.sin(), t0.sin()], 1).float().contiguous().cuda()
        nb = torch.arange(BB, dtype=torch.float64)
        self.csB = torch.cat([torch.cos(invf64[:, None] * nb[None]), torch.sin(invf64[:, None] * nb[None])], 0).float().contiguous().cuda()
        self.BB, self.invf64 = BB, invf64


class Cache:
    """one layer's cache in Step7's layout ([BK, Lcap, ...]), FP16 window ring, calibration constants (balance b,
    permutation, pre-RoPE key mean m). kh / vh: pinned host tensors written through host-mapped views (UVA)."""

    def __init__(self, B, Lcap, rot, cnt, seed=0, host=None, rc=2, mscale=1.0, bufs=None, res=None):
        g = torch.Generator(device="cpu").manual_seed(seed)
        BK = B * KVH
        self.B, self.BK, self.Lcap, self.rot, self.cnt, self.rc = B, BK, Lcap, rot, cnt, rc
        dev = "cuda"
        if bufs is not None:     # write straight into a decode step's buffers (kbench Step: kp / vp / mt / mt2, m8 = 0)
            self.kp, self.vp, self.mt, self.mt2 = bufs["kp"], bufs["vp"], bufs["mt"], bufs["mt2"]
            assert self.kp.shape[:2] == (BK, Lcap) and self.mt2.dtype == torch.float16
        else:
            self.kp = torch.zeros(BK, Lcap, P, dtype=torch.uint8, device=dev)
            self.vp = torch.zeros_like(self.kp)
            self.mt = torch.zeros(BK, Lcap, 4, dtype=torch.float16, device=dev)
            self.mt2 = torch.zeros(BK, Lcap, 2, dtype=torch.float16, device=dev)
        self.host_on = host is not False
        if host is False:        # no exact-row host store (e.g. residual records written elsewhere): dummy, never written
            host = (torch.zeros(1, 1, D, dtype=torch.float16, pin_memory=True), torch.zeros(1, 1, D, dtype=torch.float16, pin_memory=True))
        if host is None:
            host = (torch.zeros(BK, Lcap, D, dtype=torch.float16, pin_memory=True),
                    torch.zeros(BK, Lcap, D, dtype=torch.float16, pin_memory=True))
        self.kh, self.vh = host
        if self.kh.is_cuda:                             # exact rows kept on the device (S3 placement)
            self.khc, self.vhc = self.kh, self.vh
        else:
            assert self.kh.is_pinned() and self.vh.is_pinned()
            self.khc, self.vhc = K4.host_as_cuda(self.kh), K4.host_as_cuda(self.vh)
            assert self.khc.data_ptr() == self.kh.data_ptr(), "host view is not the pinned buffer (UVA required)"
        if bufs is not None and "kw" in bufs:          # the decode step's own FP16 window ring
            self.kw, self.vw = bufs["kw"], bufs["vw"]
            assert self.kw.shape == (BK, WB, D) and self.kw.is_contiguous()
        else:
            self.kw = torch.zeros(BK, WB, D, dtype=torch.float16, device=dev)
            self.vw = torch.zeros_like(self.kw)
        self.b = (torch.rand(KVH, D, generator=g) * 1.5 + 0.5).to(dev)
        self.perm = torch.stack([torch.randperm(D, generator=g) for _ in range(KVH)]).to(torch.int32).to(dev)
        self.m = (torch.randn(KVH, D, generator=g) * mscale).to(dev)
        self.H = K7.hadamard(D // 2, dev).float().contiguous()
        # residual store (krp, krm, vrp, vrm, kbits, vbits): pinned host (or device) tensors in cap_codec layout
        self.res = res
        if res is not None:
            cv = (lambda t: t) if res[0].is_cuda else K4.host_as_cuda
            self.res_v = tuple(cv(t) for t in res[:4])
            self.kbits, self.vbits = res[4], res[5]
            self.scr = torch.zeros(BK, 256, dtype=torch.int32, device=dev)
            self.H128 = K7.hadamard(D, dev).float().contiguous()

    def _args(self):
        r = self.rot
        return (self.kp, self.vp, self.mt, self.mt2, self.khc, self.vhc, self.Lcap, self.b, self.perm, self.H, self.m,
                r.c0tab, r.s0tab, r.csB, EPS)

    def step(self, newk, newv, n_host=None):
        """step mode (counter on the device). n_host: the host's copy of the counter, for the bounds check only."""
        if n_host is not None:
            assert 0 <= n_host < self.Lcap, f"base position {n_host} outside the cache"
        rk = {}
        if self.res is not None:
            rk = dict(krp_ptr=self.res_v[0], krm_ptr=self.res_v[1], vrp_ptr=self.res_v[2], vrm_ptr=self.res_v[3], scr_ptr=self.scr,
                      h128_ptr=self.H128, RES=True, KBITS=self.kbits, VBITS=self.vbits)
        enc_kernel[(self.BK, 1)](self.kw, self.vw, KVH * WB * D, WB * D, D, newk, newv, self.cnt, 0, *self._args(),
                                 TB=1, DOT=False, SWAP=True, KV=KVH, RC=self.rc, BB=self.rot.BB, WBc=WB, HOST=self.host_on,
                                 num_warps=4, **rk)

    def prefill(self, k, v, base0=0, NB=64, host=True, prec="tf32x3"):
        """k / v [B, L, KVH * D] fp16 (projection layout), L a multiple of NB, written at base positions base0 .. base0+L-1."""
        B, L, _ = k.shape
        assert L % NB == 0 and k.is_contiguous() and v.is_contiguous() and base0 + L <= self.Lcap
        enc_kernel[(self.BK, L // NB)](k, v, L * KVH * D, D, KVH * D, k, v, self.cnt, base0, *self._args(),
                                       TB=NB, DOT=True, SWAP=False, KV=KVH, RC=self.rc, BB=self.rot.BB, WBc=WB,
                                       HOST=host, PREC=prec, num_warps=4)

    def host_dma(self, k, v, base0=0):
        """exact rows to pinned host by the copy engine (cudaMemcpy, no SMs): k / v [B, L, KVH * D] -> kh / vh [BK, L, D]
        (the head transpose is a device copy first; the D2H itself is a DMA on the current stream)."""
        B, L, _ = k.shape
        kt = k.view(B, L, KVH, D).permute(0, 2, 1, 3).reshape(B * KVH, L, D)
        vt = v.view(B, L, KVH, D).permute(0, 2, 1, 3).reshape(B * KVH, L, D)
        self.kh[:, base0:base0 + L].copy_(kt, non_blocking=True)
        self.vh[:, base0:base0 + L].copy_(vt, non_blocking=True)


def rot_centre_ref(k, pos, m, rot, kvh):
    """torch reference of _rcentre in the same fp32 table composition. k [BK, T, D] fp32, pos [T] int."""
    BB = rot.BB
    tile, off = (pos // BB).long(), (pos % BB).long()
    c0, s0 = rot.c0tab[tile], rot.s0tab[tile]                                      # [T, 128]
    f = torch.arange(D, device=k.device) % 64
    cn = rot.csB[f][:, off].T                                                      # [T, 128]
    sn = rot.csB[64 + f][:, off].T
    cos, sin = c0 * cn - s0 * sn, s0 * cn + c0 * sn
    mm = m[kvh]                                                                    # [BK, D]
    rh = torch.cat([-mm[:, 64:], mm[:, :64]], -1)
    return k - (mm[:, None, :] * cos[None] + rh[:, None, :] * sin[None])


def ref_encode(k, v, b, perm, H, pos=None, m=None, rot=None):
    """torch reference of the kernel arithmetic. k, v [BK, T, D] fp16; pos [T] base positions (rc 2 when m is given)."""
    BK, T, _ = k.shape
    kvh = torch.arange(BK, device=k.device) % KVH
    kf = k.float()
    if m is not None:
        kf = rot_centre_ref(kf, pos, m, rot, kvh)
    pp = perm[kvh].long()[:, None, :].expand(BK, T, D)
    x = torch.gather(kf, 2, pp) * torch.gather(b[kvh][:, None, :].expand(BK, T, D), 2, pp)
    yh = x[..., :64] @ H
    yl = x[..., 64:] @ H
    mn = yh.amin(-1, keepdim=True).half().float()
    sc = ((yh.amax(-1, keepdim=True) - mn) / 7).half().float().clamp_min(EPS)
    code = torch.clamp(torch.round((yh - mn) / sc), 0, 7).long()
    mu = yl.mean(-1, keepdim=True).half().float()
    sd = (yl - mu).abs().mean(-1, keepdim=True).half().float()
    bit = (yl >= mu).long()
    kp = (code[..., :32] | (code[..., 32:] << 3) | (bit[..., :32] << 6) | (bit[..., 32:] << 7)).to(torch.uint8)
    vf = v.float()
    vz = vf.amin(-1, keepdim=True).half().float()
    vs = ((vf.amax(-1, keepdim=True) - vz) / 3).half().float().clamp_min(EPS)
    vc = torch.clamp(torch.round((vf - vz) / vs), 0, 3).long()
    vp = (vc[..., 0:32] | (vc[..., 32:64] << 2) | (vc[..., 64:96] << 4) | (vc[..., 96:128] << 6)).to(torch.uint8)
    return kp, vp, torch.cat([sc, mn, vs, vz], -1).half(), torch.cat([mu - sd, 2 * sd], -1).half()


def test_kv(BK, T, seed, kind="offset"):
    """keys / values fp16 [BK, T, D]: 'offset' (Gaussian + per-head offset, as encode_wf), 'qwen' (offset ~300 on a few
    channels, spread ~1), 'flat' (some tokens with max == min), 'outlier' (one channel ~300)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(BK, T, D, generator=g) + torch.randn(BK, 1, D, generator=g) * 2.0
    v = torch.randn(BK, T, D, generator=g) * 0.5 + torch.randn(BK, 1, D, generator=g) * 0.3
    if kind == "qwen":
        k[..., :8] += 300.0
    elif kind == "flat":
        k[:, ::7] = 1.5; v[:, ::5] = -0.25
    elif kind == "outlier":
        k[..., 17] = 300.0; v[..., 3] = 40.0
    return k.half().cuda(), v.half().cuda()


def cmp(c, n0, n, k, v, pos, rc):
    """mismatch counts of the cache rows n0 .. n0+n-1 against the torch reference of the same tokens."""
    rkp, rvp, rmt, rmt2 = ref_encode(k, v, c.b, c.perm, c.H, pos, c.m if rc == 2 else None, c.rot)
    sl = slice(n0, n0 + n)
    return dict(kp_mis=int((c.kp[:, sl] != rkp).sum()), vp_mis=int((c.vp[:, sl] != rvp).sum()),
                mt_mis=int((c.mt[:, sl] != rmt).sum()), mt2_mis=int((c.mt2[:, sl] != rmt2).sum()),
                mt2_ulp_max=int((c.mt2[:, sl].view(torch.int16).int() - rmt2.view(torch.int16).int()).abs().max()),
                bytes=int(rkp.numel() + rvp.numel()), host_exact=bool(torch.equal(c.kh[:, sl], k.cpu()) and torch.equal(c.vh[:, sl], v.cpu())))


def decode_compat(B=1, L=2048, mscale=8.0):
    """encode with kenc into a real Step7 and compare its base scores (q . k_hat + rotary bias, kbench7.wf_scores_ref, the
    reference the decode kernels are checked against) with exact q . k; the same with an encoder that skips the centring
    (rc 0), which a correct convention must beat clearly."""
    out = {}
    v = dict(K7.VARIANTS["two128k64"]); mode, K = v.pop("mode"), v.pop("K")
    for rc_enc in (2, 0):
        rot = Rot(L, BB=64)
        cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        st = K7.Step7(B, L, K, 2, wf=True, win=0, mode=mode, seed=3, **v)
        assert st.BB == rot.BB and torch.equal(st.c0tab, rot.c0tab) and torch.equal(st.csB, rot.csB), "rotary tables differ"
        c = Cache(B, L, rot, cnt, seed=4, host=(st.kh, st.vh), rc=rc_enc, mscale=mscale)
        st.perm.copy_(c.perm.long()); st.qbal.copy_(c.b); st.build_T()
        mpre = c.m
        qf = st.q.float().view(B, KVH, K7.G, D)
        ma, mb = mpre[:, :D // 2], mpre[:, D // 2:]
        qa, qb = qf[..., :D // 2], qf[..., D // 2:]
        st.ca.copy_((qa * ma[None, :, None] + qb * mb[None, :, None]).reshape(-1, D // 2))
        st.cb.copy_((qb * ma[None, :, None] - qa * mb[None, :, None]).reshape(-1, D // 2))
        st.u.copy_(torch.cat([st.ca, st.cb], 1)); st.w.copy_(torch.cat([st.cb, -st.ca], 1))
        st.mpre = mpre
        g = torch.Generator(device="cpu").manual_seed(7)
        z = torch.randn(B * KVH, L, D, generator=g).cuda()
        t = torch.arange(L, dtype=torch.float64)[:, None] * rot.invf64[None]
        cos, sin = torch.cat([t.cos(), t.cos()], 1).float().cuda(), torch.cat([t.sin(), t.sin()], 1).float().cuda()
        kvh = torch.arange(B * KVH, device="cuda") % KVH
        pre = z + mpre[kvh][:, None, :]                              # keys = RoPE(t)(z + m): a rotated offset
        rh = torch.cat([-pre[..., 64:], pre[..., :64]], -1)
        k = (pre * cos[None] + rh * sin[None]).half()
        vv = (torch.randn(B * KVH, L, D, generator=g) * 0.5).half().cuda()
        kl = k.view(B, KVH, L, D).permute(0, 2, 1, 3).reshape(B, L, KVH * D).contiguous()
        vl = vv.view(B, KVH, L, D).permute(0, 2, 1, 3).reshape(B, L, KVH * D).contiguous()
        c.prefill(kl, vl)
        torch.cuda.synchronize()
        st.kp.copy_(c.kp); st.vp.copy_(c.vp); st.mt.copy_(c.mt); st.mt2.copy_(c.mt2); st.sync_lo()
        st.ks.copy_(c.mt[..., 0]); st.kz.copy_(c.mt[..., 1]); st.vs.copy_(c.mt[..., 2]); st.vz.copy_(c.mt[..., 3])
        s, sx, vd, vx, sw, vw = K7.wf_scores_ref(st)
        out[f"rc{rc_enc}_score_relerr"] = ((s - sx).norm() / sx.norm()).item()
        out[f"rc{rc_enc}_v_relerr"] = ((vd - vx).norm() / vx.norm()).item()
    out["pass"] = out["rc2_score_relerr"] < 0.5 * out["rc0_score_relerr"] and out["rc2_v_relerr"] < 0.6
    return out


def check():
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {}
    for kind in ("offset", "qwen", "flat", "outlier"):
        for B in (1, 8):
            BK = B * KVH
            Lcap, nstep = 512, 2 * WB + 3                                # step mode over two full window wraps
            rot = Rot(Lcap)
            cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
            c = Cache(B, Lcap, rot, cnt, seed=B)
            k, v = test_kv(BK, WB + nstep, seed=10 + B, kind=kind)
            c.kw.copy_(k[:, :WB]); c.vw.copy_(v[:, :WB])
            for i in range(nstep):
                c.step(k[:, WB + i].contiguous(), v[:, WB + i].contiguous(), n_host=i)
                cnt.add_(1)
            torch.cuda.synchronize()
            r = cmp(c, 0, nstep, k[:, :nstep], v[:, :nstep], torch.arange(nstep, device="cuda"), 2)
            last = {s: max(i for i in range(nstep) if i % WB == s) for s in range(WB)}
            exp = torch.stack([k[:, WB + last[s]] for s in range(WB)], 1)
            r["window_ok"] = bool(torch.equal(c.kw, exp))
            r["untouched_ok"] = bool((c.kp[:, nstep:] == 0).all() and (c.kh[:, nstep:] == 0).all())
            res[f"step_{kind}_B{B}"] = r
            c2 = Cache(B, Lcap, rot, cnt, seed=B)
            L = 256
            kl = k[:, :L].reshape(B, KVH, L, D).permute(0, 2, 1, 3).reshape(B, L, KVH * D).contiguous()
            vl = v[:, :L].reshape(B, KVH, L, D).permute(0, 2, 1, 3).reshape(B, L, KVH * D).contiguous()
            c2.prefill(kl, vl, base0=0)
            torch.cuda.synchronize()
            r2 = cmp(c2, 0, L, k[:, :L], v[:, :L], torch.arange(L, device="cuda"), 2)
            nn = min(L, nstep)                                       # tokens encoded by both modes
            r2["vs_step_kp_mis"] = int((c2.kp[:, :nn] != c.kp[:, :nn]).sum())
            r2["vs_step_mt_mis"] = int((c2.mt[:, :nn] != c.mt[:, :nn]).sum())
            r2["vs_step_mt2_mis"] = int((c2.mt2[:, :nn] != c.mt2[:, :nn]).sum())
            res[f"prefill_{kind}_B{B}"] = r2
    rot = Rot(512); cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    ce = Cache(1, 512, rot, cnt, seed=9); cg = Cache(1, 512, rot, cnt, seed=9)
    k, v = test_kv(KVH, WB + 40, seed=99)
    nk = torch.empty(KVH, D, dtype=torch.float16, device="cuda"); nv = torch.empty_like(nk)
    nk.copy_(k[:, WB]); nv.copy_(v[:, WB])
    cg.kw.copy_(k[:, :WB]); cg.vw.copy_(v[:, :WB])
    cg.step(nk, nv); torch.cuda.synchronize()                      # warm-up compile, then reset state
    cnt.zero_(); cg.kw.copy_(k[:, :WB]); cg.vw.copy_(v[:, :WB]); cg.kp.zero_(); cg.mt.zero_(); cg.mt2.zero_()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        cg.step(nk, nv)
        cnt.add_(1)
    cnt.zero_(); cg.kw.copy_(k[:, :WB]); cg.vw.copy_(v[:, :WB])
    for i in range(40):
        nk.copy_(k[:, WB + i]); nv.copy_(v[:, WB + i]); gr.replay()
    torch.cuda.synchronize()
    cnt.zero_(); ce.kw.copy_(k[:, :WB]); ce.vw.copy_(v[:, :WB])
    for i in range(40):
        ce.step(k[:, WB + i].contiguous(), v[:, WB + i].contiguous()); cnt.add_(1)
    torch.cuda.synchronize()
    res["graph_vs_eager"] = dict(equal=bool(torch.equal(cg.kp[:, :40], ce.kp[:, :40]) and torch.equal(cg.mt[:, :40], ce.mt[:, :40])
                                           and torch.equal(cg.mt2[:, :40], ce.mt2[:, :40]) and torch.equal(cg.kw, ce.kw)))
    res["decode_compat"] = decode_compat()
    for kk, r in res.items():
        print(kk, json.dumps(r), flush=True)
    vals = [r for kk, r in res.items() if kk.startswith(("step_", "prefill_"))]
    # fp32 evaluation order (FMA contraction in the centring, reduction trees) differs from torch, so a value exactly at a
    # rounding boundary can flip: gate on a flip rate <= 1e-4 of the bytes / metadata, metadata within 1 fp16 ulp
    def rate(r, *ks):
        return sum(r[x] for x in ks) / r["bytes"]
    hard = all(rate(r, "kp_mis", "vp_mis") <= 1e-4 and r["mt_mis"] + r["mt2_mis"] <= 1e-4 * r["bytes"] and r["host_exact"]
               and r["mt2_ulp_max"] <= 1 for r in vals) \
        and all(r["window_ok"] and r["untouched_ok"] for kk, r in res.items() if kk.startswith("step_")) \
        and res["graph_vs_eager"]["equal"] and res["decode_compat"]["pass"]
    print("CHECK", "PASS" if hard else "FAIL",
          "| gates: byte / metadata flips vs torch <= 1e-4 (fp32 order at rounding ties), mt2 within 1 fp16 ulp, host rows exact, window ring exact, rows past"
          " the written range untouched, graph replay == eager, decode-side score error with centring < 0.5 x without")
    return res


def res_check():
    """RES mode: the residual records the fused step writes vs cap_codec.encode_res on the residual of the same tokens against
    the base the kernel itself stored (decoded in torch). Gates: code-byte flips <= 1e-3, min / step within 1 fp16 ulp."""
    import cap_codec as C
    torch.backends.cuda.matmul.allow_tf32 = False
    f16 = lambda t: t.half().float()  # noqa: E731
    out = {}
    for kb, vb in ((3, 3), (5, 4)):
        for kind in ("offset", "qwen"):
            B, Lcap, nstep = 2, 512, 2 * WB + 3
            BK = B * KVH
            rot = Rot(Lcap); cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
            res = (torch.zeros(BK, Lcap, 16 * kb, dtype=torch.int8, pin_memory=True), torch.zeros(BK, Lcap, 2, dtype=torch.float16, pin_memory=True),
                   torch.zeros(BK, Lcap, 16 * vb, dtype=torch.int8, pin_memory=True), torch.zeros(BK, Lcap, 2, dtype=torch.float16, pin_memory=True), kb, vb)
            c = Cache(B, Lcap, rot, cnt, seed=3, host=False, res=res)
            k, v = test_kv(BK, WB + nstep, seed=40 + kb, kind=kind)
            c.kw.copy_(k[:, :WB]); c.vw.copy_(v[:, :WB])
            for i in range(nstep):
                c.step(k[:, WB + i].contiguous(), v[:, WB + i].contiguous(), n_host=i)
                cnt.add_(1)
            torch.cuda.synchronize()
            n = nstep
            kt, vt = k[:, :n], v[:, :n]
            kvh = torch.arange(BK, device="cuda") % KVH
            pos = torch.arange(n, device="cuda")
            kc = rot_centre_ref(kt.float(), pos, c.m, rot, kvh)
            pp = c.perm[kvh].long()[:, None, :].expand(BK, n, D)
            bp = torch.gather(c.b[kvh][:, None, :].expand(BK, n, D), 2, pp)
            x = torch.gather(kc, 2, pp) * bp
            kpl = c.kp[:, :n].long()
            code = torch.cat([kpl & 7, (kpl >> 3) & 7], -1).float()
            bit = torch.cat([(kpl >> 6) & 1, (kpl >> 7) & 1], -1).float()
            mt, m2 = c.mt[:, :n].float(), c.mt2[:, :n].float()
            xhat = torch.cat([(code * mt[..., 0:1] + mt[..., 1:2]) @ c.H, (m2[..., 0:1] + m2[..., 1:2] * bit) @ c.H], -1)
            rperm = (x - xhat) / bp
            rorig = torch.empty_like(rperm).scatter_(2, pp, rperm)
            H128 = K7.hadamard(D, "cuda").float()
            pk, mnk, sck, _, _ = C.encode_res(rorig @ H128, f16, kb)
            vhat = K5._deq(c.vp[:, :n], c.mt[:, :n, 2], c.mt[:, :n, 3])
            pv, mnv, scv, _, _ = C.encode_res((vt.float() - vhat) @ H128, f16, vb)
            gk, gv = c.res[0][:, :n].cuda().view(torch.uint8), c.res[2][:, :n].cuda().view(torch.uint8)
            mk, mv = c.res[1][:, :n].cuda(), c.res[3][:, :n].cuda()
            ulp = lambda a_, b_: int((a_.view(torch.int16).int() - b_.view(torch.int16).int()).abs().max())  # noqa: E731
            r = dict(kbyte_flip=(gk != pk).float().mean().item(), vbyte_flip=(gv != pv).float().mean().item(),
                     kmeta_ulp=ulp(mk, torch.cat([mnk, sck], -1)), vmeta_ulp=ulp(mv, torch.cat([mnv, scv], -1)),
                     untouched=bool((c.res[0][:, n:] == 0).all()))
            out[f"h{kb}{vb}_{kind}"] = r
            print(f"h{kb}{vb}_{kind}", json.dumps(r), flush=True)
    ok = all(r["kbyte_flip"] <= 1e-3 and r["vbyte_flip"] <= 1e-3 and r["kmeta_ulp"] <= 1 and r["vmeta_ulp"] <= 1 and r["untouched"]
             for r in out.values())
    print("RESCHECK", "PASS" if ok else "FAIL", "| gates: residual code-byte flips <= 1e-3, min / step within 1 fp16 ulp vs cap_codec on the kernel's own base")
    return out


def gpu_state():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return repr(e)[:80]


def time_graph(fn, warm, reps, inner=10):
    """median and IQR over reps of (inner back-to-back replays) / inner, so per-replay launch gaps are amortised."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(inner):
            g.replay()
        e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1) / inner)
    ts.sort()
    return S.median(ts), ts[len(ts) // 4], ts[3 * len(ts) // 4]


def timing(out, reps, warm, NL=32):
    torch.backends.cuda.matmul.allow_tf32 = False
    here = os.path.dirname(os.path.abspath(__file__))
    md5 = {f: hashlib.md5(open(os.path.join(here, f), "rb").read()).hexdigest()[:8] for f in ("kenc.py", "kbench7.py", "kbench4.py")}
    for B in (1, 8):
        BK = B * KVH
        Lcap = 1 << 14                                              # counter walks forward; wraps at Lcap - 1
        rot = Rot(Lcap)
        cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        caches = [Cache(B, Lcap, rot, cnt, seed=li) for li in range(NL)]
        kc = [torch.zeros(BK, Lcap, D, dtype=torch.float16, device="cuda") for _ in range(NL)]
        vc = [torch.zeros_like(x) for x in kc]
        nk = [torch.randn(BK, D, device="cuda").half() for _ in range(NL)]
        nv = [torch.randn(BK, D, device="cuda").half() for _ in range(NL)]
        Lm = torch.tensor([Lcap - 1], dtype=torch.int32, device="cuda")

        def adv():                                                  # one counter for the model, advanced once per step
            cnt.add_(1); torch.remainder(cnt, Lm, out=cnt)

        def f16_copy():
            for li in range(NL):
                kc[li][:, 5].copy_(nk[li]); vc[li][:, 5].copy_(nv[li])

        def f16_kernel():
            for li in range(NL):
                app16_kernel[(BK,)](nk[li], nv[li], kc[li], vc[li], cnt, Lcap, KV=KVH, num_warps=1)
            adv()

        def fused():
            for li, c in enumerate(caches):
                c.step(nk[li], nv[li])
            adv()

        def fused_nohost():
            for li, c in enumerate(caches):
                c.host_on = False
                c.step(nk[li], nv[li])
                c.host_on = True
            adv()
        rec = dict(B=B, NL=NL, device=torch.cuda.get_device_name(0), md5=md5, torch=torch.__version__, triton=triton.__version__,
                   clock_before=gpu_state())
        for nm, fn in (("fp16_append_copies", f16_copy), ("fp16_append_kernel", f16_kernel), ("ours_fused_step", fused),
                       ("ours_fused_step_nohost", fused_nohost)):
            m, lo, hi = time_graph(fn, warm, reps)
            rec[nm] = dict(ms=m, iqr=[lo, hi])
        rec["clock_after"] = gpu_state()
        rec["note"] = ("per decode step, all NL layers in one CUDA graph, 10 back-to-back replays per sample; fused step = "
                       "encode of the evicted window token (with rc-2 centring) + exact K/V row written to pinned host + new "
                       "token into the window + one device counter add; fp16_append_kernel = one launch per layer writing "
                       "K and V at the counter + the same counter add")
        print(json.dumps(rec), flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        caches = kc = vc = None
        torch.cuda.empty_cache()


def prefill_parts(Ls=(8192, 32768), NL=32, reps=5):
    """time the prefill encode pieces for NL layers (eager, CUDA events): kernel with zero-copy host writes, kernel
    without host writes (ieee / tf32x3 dots), DMA host copy alone."""
    for L in Ls:
        rot = Rot(L); cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        c = Cache(1, L, rot, cnt, seed=1)
        k = torch.randn(1, L, KVH * D, device="cuda").half(); v = torch.randn_like(k)
        res = {}
        for nm, fn in (("kernel_host_ieee", lambda: c.prefill(k, v)), ("kernel_nohost_ieee", lambda: c.prefill(k, v, host=False)),
                       ("kernel_nohost_tf32x3", lambda: c.prefill(k, v, host=False, prec="tf32x3")),
                       ("kernel_nohost_tf32", lambda: c.prefill(k, v, host=False, prec="tf32")),
                       ("dma_host", lambda: c.host_dma(k, v))):
            fn(); torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(NL):
                    fn()
                e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
            res[nm] = S.median(ts)
        print(json.dumps(dict(L=L, NL=NL, ms=res)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true"); ap.add_argument("--time", action="store_true")
    ap.add_argument("--parts", action="store_true"); ap.add_argument("--rescheck", action="store_true")
    ap.add_argument("--out", default=""); ap.add_argument("--reps", type=int, default=50); ap.add_argument("--warm", type=int, default=20)
    a = ap.parse_args()
    if a.check:
        check()
    if a.time:
        timing(a.out, a.reps, a.warm)
    if a.parts:
        prefill_parts()
    if a.rescheck:
        res_check()
