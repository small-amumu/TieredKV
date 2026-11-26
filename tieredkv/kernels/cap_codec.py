# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""cap_codec (capacity round): the byte layout of the deployed GPU base (stkm8v8128, sel12_q m8v8) and of the host
residual store (sel14_q _h44), with encoders that reproduce sel12_q / sel14_q exactly and decoders that read only the bytes.
Per token and KV head (d = 128, h = 64):
  key codes   32 B: byte j (j < 32) = hi code j (bits 0-2) | hi code j+32 << 3 | lo bit j << 6 | lo bit j+32 << 7
              (kbench7's record), in the stack's permuted, balanced, centred, per-half Hadamard coordinates
  key meta     4 B: e4m3 bytes (hi step, hi zero in step units, lo mean, lo mean absolute deviation)
  value codes 32 B: byte j = codes of channels j, j+32, j+64, j+96 (2 bits each, bits 0-1 .. 6-7), Hadamard coordinates
  value meta   2 B: e4m3 bytes (step, zero in step units)
  = 70 B = 2.1875 bits per element (mean over K and V). Sinks and the residual window stay FP16 (not in this record).
  host key / value residual: 4-bit codes of the Hadamard-rotated residual, 2 per byte (64 B), plus FP16 minimum and step
  (4 B): 68 B each = 4.25 bits per element.
Exceptions the byte format cannot express, counted (never silently approximated): a hi half or value row of zero range
(step 0: sel12_q keeps the row exact), and a value zero saturating e4m3 (sel12_q keeps an FP16 zero). Both were 0 in S1 / S1b
apart from row 0, which the stack keeps exact anyway."""
from __future__ import annotations

import torch

E4MIN = 2.0 ** -9


def e4(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def e4b(x):
    """float -> e4m3 byte (uint8); x must already be an e4m3 value (checked)."""
    b = x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if not torch.equal(b.float(), x):
        raise ValueError("e4b: value is not exactly representable in e4m3")
    return b.view(torch.uint8)


def be4(b):
    return b.view(torch.float8_e4m3fn).float()


# ---------------------------------------------------------------- keys (hi half: m8 clip search, lo half: mean +- MAD)
def encode_hi(y, gammas):
    """y [..., 64] -> codes uint8 [..., 64], step byte, zero byte, n_zero_range. Same arithmetic as s0_cov.hi_quant(y, 'm8')."""
    lv = 7
    mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    c, h = (mn + mx) / 2, (mx - mn) / 2
    best = berr = bcode = bsc = bzr = None
    for gm in gammas:
        raw = 2 * h * gm / lv
        sc = torch.where(raw > 0, e4(raw).clamp_min(E4MIN), 0.0)
        safe = torch.where(sc > 0, sc, torch.ones_like(sc))
        zr = e4((c - h * gm) / safe)
        lo = zr * sc
        code = torch.clamp(torch.round((y - lo) / safe), 0, lv)
        xh = torch.where(sc > 0, code * sc + lo, y)
        err = ((xh - y) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, berr, bcode, bsc, bzr = xh, err, code, sc, zr
        else:
            t = err < berr
            best, berr = torch.where(t, xh, best), torch.minimum(err, berr)
            bcode, bsc, bzr = torch.where(t, code, bcode), torch.where(t, sc, bsc), torch.where(t, zr, bzr)
    nzero = int((bsc == 0).sum())
    return bcode.to(torch.uint8), e4b(bsc), e4b(bzr), nzero, best


def decode_hi(code, scb, zrb):
    sc, zr = be4(scb), be4(zrb)
    return code.float() * sc + zr * sc


def encode_lo(y, f16):
    mu = e4(f16(y.mean(-1, keepdim=True)))
    s = e4(f16((y - f16(y.mean(-1, keepdim=True))).abs().mean(-1, keepdim=True))).abs()
    bit = (y >= mu)
    return bit, e4b(mu), e4b(s)


def decode_lo(bit, mub, sb):
    mu, s = be4(mub), be4(sb)
    return mu + s * torch.where(bit, 1.0, -1.0)


def pack_k(code_hi, bit_lo):
    """code_hi uint8 [..., 64] (0..7), bit_lo bool [..., 64] -> uint8 [..., 32]."""
    c0, c1 = code_hi[..., :32].to(torch.int32), code_hi[..., 32:].to(torch.int32)
    b0, b1 = bit_lo[..., :32].to(torch.int32), bit_lo[..., 32:].to(torch.int32)
    return (c0 | (c1 << 3) | (b0 << 6) | (b1 << 7)).to(torch.uint8)


def unpack_k(kb):
    x = kb.to(torch.int32)
    code = torch.cat([x & 7, (x >> 3) & 7], -1).to(torch.uint8)
    bit = torch.cat([(x >> 6) & 1, (x >> 7) & 1], -1).bool()
    return code, bit


# ---------------------------------------------------------------- values (2-bit per token, Hadamard, e4m3 step / zero)
def encode_v(y, f16):
    """y [..., 128] (already Hadamard-rotated) -> codes uint8 [..., 128] (0..3), step byte, zero byte, counts, reconstruction
    (sel12_q._v8 arithmetic)."""
    mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    raw = (mx - mn) / 3
    sc = torch.where(raw > 0, e4(raw).clamp_min(E4MIN), 0.0)
    safe = torch.where(sc > 0, sc, torch.ones_like(sc))
    zr = mn / safe
    sat = (zr.abs() > 448) & (sc > 0)
    zre = e4(zr)
    lo = torch.where(sat, f16(mn), zre * sc)
    code = torch.clamp(torch.round((y - lo) / safe), 0, 3)
    q = torch.where(sc > 0, code * sc + lo, y)
    return code.to(torch.uint8), e4b(sc), e4b(zre), int((sc == 0).sum()), int(sat.sum()), q


def decode_v(code, scb, zrb):
    sc, zr = be4(scb), be4(zrb)
    return code.float() * sc + zr * sc


def pack_v(code):
    c = code.to(torch.int32)
    return (c[..., :32] | (c[..., 32:64] << 2) | (c[..., 64:96] << 4) | (c[..., 96:] << 6)).to(torch.uint8)


def unpack_v(vb):
    x = vb.to(torch.int32)
    return torch.cat([x & 3, (x >> 2) & 3, (x >> 4) & 3, (x >> 6) & 3], -1).to(torch.uint8)


# ---------------------------------------------------------------- host residual store (4 bit, FP16 min and step)
def encode_res4(r_rot, f16):
    """r_rot [..., d] (Hadamard-rotated residual) -> packed uint8 [..., d/2], min fp16, step fp16, reconstruction
    (pkv5_q.aq arithmetic, 4 bits, one group of d)."""
    lv = 15
    mn, mx = f16(r_rot.amin(-1, keepdim=True)), f16(r_rot.amax(-1, keepdim=True))
    sc = f16((mx - mn) / lv)
    safe = torch.where(sc > 0, sc, torch.ones_like(sc))
    code = torch.clamp(torch.round((r_rot - mn) / safe), 0, lv)
    rec = torch.where(sc > 0, code * sc + mn, r_rot)
    c = code.to(torch.int32)
    d2 = r_rot.shape[-1] // 2
    packed = (c[..., :d2] | (c[..., d2:] << 4)).to(torch.uint8)
    return packed, mn.half(), sc.half(), int((sc == 0).sum()), rec


def decode_res4(packed, mn, sc):
    x = packed.to(torch.int32)
    code = torch.cat([x & 15, (x >> 4) & 15], -1).float()
    return code * sc.float() + mn.float()


RECORD_GPU = 32 + 4 + 32 + 2          # bytes per token and KV head on the GPU (stkm8v8)
def _pack_bits(code, bits):
    """code int [..., d] (0 .. 2^bits - 1) -> uint8 [..., d * bits / 8]. Two layouts (both unchanged from b98e8a29):
    bits == 4: half-channel nibbles, byte j = code j (bits 0-3) | code j + d/2 << 4 (j < d/2), as encode_res4;
    other bits: little-endian bit stream per row, bit w of code j is stream bit j * bits + w.
    codec_fix_v3 (stream branch only): output byte b, bit k is stream bit p = 8b + k, i.e. bit p % bits of code p // bits;
    each of the 8 bit positions of every byte is gathered with index_select (deterministic on CUDA, no atomics) from a
    uint8 copy of the codes, so the transient is ~2 bytes per code instead of the old int64 [..., d, bits] expansion
    (~16 * bits bytes per code; OOM at 16K / 32K tokens). Output bytes identical to b98e8a29.
    Behaviour change: d * bits not a multiple of 8 raises ValueError (the old code failed in a reshape)."""
    *lead, d = code.shape
    if d * bits % 8:
        raise ValueError(f"_pack_bits: d * bits = {d * bits} is not a multiple of 8")
    if bits == 4:
        c = code.to(torch.int64)
        return (c[..., : d // 2] | (c[..., d // 2:] << 4)).to(torch.uint8)
    c = code.to(torch.uint8)                                   # codes < 2^bits <= 256
    nb = d * bits // 8
    p = torch.arange(nb * 8, device=c.device).view(nb, 8)     # stream bit index of (byte, bit)
    src, sh = p // bits, (p % bits).to(torch.uint8)
    out = torch.zeros((*lead, nb), dtype=torch.uint8, device=c.device)
    for k in range(8):
        out |= ((c.index_select(-1, src[:, k]) >> sh[:, k]) & 1) << k
    return out


def _unpack_bits(packed, bits, d):
    """inverse of _pack_bits -> int64 [..., d]; stream branch gathers each bit with index_select (deterministic).
    Behaviour change: a byte count that does not hold d codes of `bits` bits raises ValueError."""
    *lead, nb = packed.shape
    if nb * 8 != d * bits:
        raise ValueError(f"_unpack_bits: {nb} bytes do not hold {d} codes of {bits} bits")
    if bits == 4:
        x = packed.to(torch.int64)
        return torch.cat([x & 15, (x >> 4) & 15], -1)
    x = packed.to(torch.int32)
    code = torch.zeros((*lead, d), dtype=torch.int32, device=x.device)
    pos = torch.arange(d, device=x.device) * bits
    for w in range(bits):
        p = pos + w
        code |= ((x.index_select(-1, p // 8) >> (p % 8).to(torch.int32)) & 1) << w
    return code.to(torch.int64)


def encode_res(r_rot, f16, bits):
    """b-bit version of encode_res4 (pkv5_q.aq arithmetic, one group of d): packed uint8 [..., d * bits / 8], min fp16,
    step fp16, count of zero-range rows, reconstruction."""
    lv = 2 ** bits - 1
    mn, mx = f16(r_rot.amin(-1, keepdim=True)), f16(r_rot.amax(-1, keepdim=True))
    sc = f16((mx - mn) / lv)
    safe = torch.where(sc > 0, sc, torch.ones_like(sc))
    code = torch.clamp(torch.round((r_rot - mn) / safe), 0, lv)
    rec = torch.where(sc > 0, code * sc + mn, r_rot)
    return _pack_bits(code, bits), mn.half(), sc.half(), int((sc == 0).sum()), rec


def decode_res(packed, mn, sc, bits, d):
    return _unpack_bits(packed, bits, d).float() * sc.float() + mn.float()


RECORD_HOST = 2 * (64 + 4)            # bytes per token and KV head on the host (h44)
