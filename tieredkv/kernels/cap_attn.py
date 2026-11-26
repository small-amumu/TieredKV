# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""cap_attn (capacity round): one decode step of the deployed system computed from the STORED BYTES only (cap_codec
layout), for the A5 family stkm8v8128k64_x2s128[_j<L>|_p<L>]_h44, one query row per call:
  GPU, per layer and KV head:  key / value code records of rows [0, W0) (W0 = floor(Lk / 128) * 128), the FP16 rows
                               [W0, Lk) (residual window, incl. the current token) and the 4 FP16 sinks;
  host:                        4-bit key / value residual records of rows [0, W0) (read only for the fetched rows);
  static calibration:          the stack's rotary mean, balance, channel order (sel10_q.INFO / pkv9_q.centre inputs).
Algorithm (sel14_q._attend_h with Lq = 1, written over the stored data):
  base scores of all coded rows, chunk by chunk:  s_j = scale (q_hi' . yhat_hi_j + q_lo' . yhat_lo_j + q . m_j), with
    q' = Hadamard of the permuted, balanced query halves and m_j the rotary mean at position j (no key is reconstructed);
  exact scores for sinks and window; stage 1: top-M of the group-mean softmax over coded non-sink rows; the M candidates get
  host keys k_base + res; stage 2 with the logit correction lam on the other coded rows (variance from the stored hi step and
  lo MAD); top-K fetched values v_base + res; output = sum over coded rows of p v_base (decoded in Hadamard coordinates and
  rotated once) + fetched rows (p (v_host - v_base)) + sinks and window exactly.
Streaming: the only per-layer vector that scales with Lk is the base score s [H, W0] fp32 (plus a [KVH, W0] bool
candidate mask); the softmax normalisers of both stages, the stage-1 top-M and the value sums are accumulated chunk by
chunk; the rotary mean is computed only at the chunk's (or the candidates') positions. Nothing scales with d * Lk except
the stored records. cap_check.py compares with sel14_q.attend on real tensors (rel. err. budget 1e-4 with FP32 exact rows)."""
from __future__ import annotations

import math

import torch

import cap_codec as C

R_WIN, Z_SINK = 128, 4
ENC: dict = {"unexpressible": 0}      # encode_rows: zero-range hi / value rows and saturated value zeros (rows >= Z)


class Store:
    """one layer's stored data. Built either from exact k, v (encode(), for checks) or allocated with random bytes
    (alloc(), for the capacity measurement)."""

    def __init__(self):
        self.kb = self.km = self.vb = self.vm = None        # GPU records [KVH, W0, 32 / 4 / 32 / 2] uint8
        self.kr = self.krm = self.vr = self.vrm = None      # host records [KVH, W0, 64] uint8, [KVH, W0, 2] fp16
        self.kx = self.vx = None                            # GPU FP16 exact rows: sinks + window [KVH, Z + w, d]
        self.W0 = 0
        self.kbits = self.vbits = 4                         # host residual widths (set by encode / IncStore)

    def nbytes_gpu(self):
        return sum(t.numel() * t.element_size() for t in (self.kb, self.km, self.vb, self.vm, self.kx, self.vx))

    def nbytes_host(self):
        return sum(t.numel() * t.element_size() for t in (self.kr, self.krm, self.vr, self.vrm))


def encode(k, v, li, S10, Q5, Q6, P9, P10, P11, Q7, pin=False, wdtype=torch.float16, kbits=4, vbits=4):
    """k, v [KVH, Lk, d] fp32 (exact, post-RoPE) -> Store, with the stack's calibration of layer li (sel10_q.INFO set by a
    previous stack_base call on the same layer). Rows [Z, W0) are coded; the code also covers rows < Z (unused)."""
    KVH, Lk, d = k.shape
    h = d // 2
    W0 = Lk // R_WIN * R_WIN
    st = Store()
    st.W0 = W0
    f, n7 = P10.split(S10.STACK)
    rc = bool(f["rc"]) and li not in P11.NOPE
    mt, b = P9.centre(li, k[:, :W0], rc, Q7.parse(n7)["a"])
    I = S10.INFO[li]
    x = (k[:, :W0] - mt) * b
    o = I["order"][:, None, :].expand_as(x)
    xs = x.gather(-1, o)
    Hh = Q5.CQ.hadamard(h, k.device)
    code_hi, scb, zrb, _, _ = C.encode_hi(xs[..., :h] @ Hh, Q6.GAMMAS)
    bit_lo, mub, sb = C.encode_lo(xs[..., h:] @ Hh, Q5.f16)
    st.kb = C.pack_k(code_hi, bit_lo)
    st.km = torch.cat([scb, zrb, mub, sb], -1)
    H = Q5.CQ.hadamard(d, k.device)
    vcode, vscb, vzrb, _, _, _ = C.encode_v(v[:, :W0] @ H, Q5.f16)
    st.vb, st.vm = C.pack_v(vcode), torch.cat([vscb, vzrb], -1)
    code2, bit2 = C.unpack_k(st.kb)
    yh2 = C.decode_hi(code2, st.km[..., 0:1], st.km[..., 1:2])
    yl2 = C.decode_lo(bit2, st.km[..., 2:3], st.km[..., 3:4])
    kq = torch.empty_like(x).scatter_(-1, o, torch.cat([yh2 @ Hh.T, yl2 @ Hh.T], -1)) / b + mt
    vq = C.decode_v(C.unpack_v(st.vb), st.vm[..., 0:1], st.vm[..., 1:2]) @ H.T
    kq[:, :Z_SINK], vq[:, :Z_SINK] = k[:, :Z_SINK], v[:, :Z_SINK]
    st.kr, krm_mn, krm_sc, _, _ = C.encode_res((k[:, :W0] - kq) @ H, Q5.f16, kbits)
    st.kbits = kbits
    st.vr, vrm_mn, vrm_sc, _, _ = C.encode_res((v[:, :W0] - vq) @ H, Q5.f16, vbits)
    st.vbits = vbits
    st.krm, st.vrm = torch.cat([krm_mn, krm_sc], -1), torch.cat([vrm_mn, vrm_sc], -1)
    if pin:
        st.kr, st.krm, st.vr, st.vrm = (t.cpu().pin_memory() for t in (st.kr, st.krm, st.vr, st.vrm))
    zs = min(Z_SINK, W0)                                                  # with W0 = 0 the sinks are inside the window
    st.kx = torch.cat([k[:, :zs], k[:, W0:]], 1).to(wdtype)              # FP16 when deployed (lossless from bf16)
    st.vx = torch.cat([v[:, :zs], v[:, W0:]], 1).to(wdtype)
    return st


def encode_rows(k, v, r0, r1, li, S10, Q5, Q6, P9, P10, P11, Q7, kbits=4, vbits=4):
    """codes of rows [r0, r1) of the cache k, v [KVH, T, d] fp32 (per-token codes; the rotary mean at the rows' own positions,
    so any split of [0, W0) into ranges gives exactly the codes of encode()). Returns the GPU and host records."""
    KVH, _, d = k.shape
    h = d // 2
    kr_, vr_ = k[:, r0:r1], v[:, r0:r1]
    mt, b = centre_at(li, torch.arange(r0, r1, device=k.device), KVH, d, k.device, S10, P9, P10, P11, Q7)
    I = S10.INFO[li]
    x = (kr_ - mt) * b
    o = I["order"][:, None, :].expand_as(x)
    xs = x.gather(-1, o)
    Hh = Q5.CQ.hadamard(h, k.device)
    code_hi, scb, zrb, _, _ = C.encode_hi(xs[..., :h] @ Hh, Q6.GAMMAS)
    lo_row = max(0, Z_SINK - r0)                                           # rows < Z (sinks) are never read from records
    nz_hi = int((C.be4(scb)[:, lo_row:] == 0).sum())
    bit_lo, mub, sb = C.encode_lo(xs[..., h:] @ Hh, Q5.f16)
    kb, km = C.pack_k(code_hi, bit_lo), torch.cat([scb, zrb, mub, sb], -1)
    c2, b2 = C.unpack_k(kb)
    yh2 = C.decode_hi(c2, km[..., 0:1], km[..., 1:2])
    yl2 = C.decode_lo(b2, km[..., 2:3], km[..., 3:4])
    kq = torch.empty_like(x).scatter_(-1, o, torch.cat([yh2 @ Hh.T, yl2 @ Hh.T], -1)) / b + mt
    H = Q5.CQ.hadamard(d, k.device)
    vcode, vscb, vzrb, _, _, _ = C.encode_v(vr_ @ H, Q5.f16)
    yv = vr_ @ H
    vmn, vmx = yv.amin(-1, keepdim=True), yv.amax(-1, keepdim=True)
    vsc = C.be4(vscb)
    nz_v = int((vsc[:, lo_row:] == 0).sum())
    nsat_v = int((((vmn / torch.where(vsc > 0, vsc, torch.ones_like(vsc))).abs() > 448) & (vsc > 0))[:, lo_row:].sum())
    ENC["unexpressible"] += nz_hi + nz_v + nsat_v                          # rows the byte format cannot express
    vb, vm = C.pack_v(vcode), torch.cat([vscb, vzrb], -1)
    vq = C.decode_v(C.unpack_v(vb), vm[..., 0:1], vm[..., 1:2]) @ H.T
    if r0 < Z_SINK:                                                        # sinks: exact (never read from the records)
        z = min(Z_SINK, r1) - r0
        kq[:, :z], vq[:, :z] = kr_[:, :z], vr_[:, :z]
    kr, krmn, krsc, _, _ = C.encode_res((kr_ - kq) @ H, Q5.f16, kbits)
    vr, vrmn, vrsc, _, _ = C.encode_res((vr_ - vq) @ H, Q5.f16, vbits)
    if r1 > Z_SINK:      # residual rows of zero step outside the sinks: decoded as f16(min), not exactly (counted, as GPU rows)
        j0 = max(Z_SINK - r0, 0)
        ENC["res_zero"] = ENC.get("res_zero", 0) + int((krsc[:, j0:] == 0).sum()) + int((vrsc[:, j0:] == 0).sum())
    return kb, km, vb, vm, kr, torch.cat([krmn, krsc], -1), vr, torch.cat([vrmn, vrsc], -1)


class IncStore(Store):
    """a Store grown 128 rows at a time (decode-time incremental encoding; old codes never change)."""

    def __init__(self, KVH, d, dev, kbits=4, vbits=4):
        super().__init__()
        if not all(2 <= b <= 8 and d * b % 8 == 0 for b in (kbits, vbits)):
            raise ValueError(f"cap_attn: host residual widths {kbits}/{vbits} not supported for d={d}")
        e = lambda w, dt=torch.uint8: torch.empty(KVH, 0, w, dtype=dt, device=dev)
        self.kbits, self.vbits = kbits, vbits
        self.kb, self.km, self.vb, self.vm = e(d // 4), e(4), e(d // 4), e(2)
        self.kr, self.krm, self.vr, self.vrm = e(d * kbits // 8), e(2, torch.float16), e(d * vbits // 8), e(2, torch.float16)
        self.W0 = 0

    def grow(self, k, v, W0new, li, S10, Q5, Q6, P9, P10, P11, Q7):
        if W0new <= self.W0:
            return
        recs = encode_rows(k, v, self.W0, W0new, li, S10, Q5, Q6, P9, P10, P11, Q7, self.kbits, self.vbits)
        names = ("kb", "km", "vb", "vm", "kr", "krm", "vr", "vrm")
        for n_, t in zip(names, recs):
            setattr(self, n_, torch.cat([getattr(self, n_), t], 1))
        self.W0 = W0new

    def set_exact(self, k, v, T, wdtype=torch.float16):
        zs = min(Z_SINK, self.W0)
        self.kx = torch.cat([k[:, :zs], k[:, self.W0:T]], 1).to(wdtype)
        self.vx = torch.cat([v[:, :zs], v[:, self.W0:T]], 1).to(wdtype)


def cos_sin_at(pos, RC):
    """rc_q.cos_sin at arbitrary positions (same float64 arithmetic, so equal to slicing rc_q.cos_sin(L))."""
    f = pos.double()[:, None] * RC.R["inv_freq"].to(pos.device).double()[None]
    e = torch.cat([f, f], -1)
    return (e.cos() * RC.R["asc"]).float(), (e.sin() * RC.R["asc"]).float()


def centre_at(li, pos, KVH, d, dev, S10, P9, P10, P11, Q7):
    """pkv9_q.centre at the positions pos (1-D) or, with pos None, only the balance. Returns (mt [KVH, n|1, d], b [KVH, 1, d])."""
    f, n7 = P10.split(S10.STACK)
    rc = bool(f["rc"]) and li not in P11.NOPE
    a = Q7.parse(n7)["a"]
    if rc:
        bal = P9.RC.R["bal"][li]
        if pos is None:
            mt = None
        else:
            cos, sin = cos_sin_at(pos, P9.RC)
            mt = P9.RC.rotate(P9.RC.R["m_pre"][li], cos, sin)
    else:
        mt, bal = P9.Q5.C["m"][li][:, None, :], P9.Q7.C7["bal"][li]
    return mt, (bal ** (a / 100))[:, None, :]


def _take(t, idx):
    return t.gather(1, idx[..., None].expand(*idx.shape, t.shape[-1]))


def _dec_k_full(st, li, S10, Q5, P9, P10, P11, Q7, idx):
    """reconstruct the keys (original coordinates) of the rows idx [KVH, n] from the GPU records."""
    kb, km = _take(st.kb, idx), _take(st.km, idx)
    KVH, n = idx.shape
    d = kb.shape[-1] * 4
    h = d // 2
    code, bit = C.unpack_k(kb)
    yh = C.decode_hi(code, km[..., 0:1], km[..., 1:2])
    yl = C.decode_lo(bit, km[..., 2:3], km[..., 3:4])
    Hh = Q5.CQ.hadamard(h, kb.device)
    o = S10.INFO[li]["order"][:, None, :].expand(KVH, n, d)
    xs = torch.empty(KVH, n, d, device=kb.device).scatter_(-1, o, torch.cat([yh @ Hh.T, yl @ Hh.T], -1))
    mt = torch.empty(KVH, n, d, device=kb.device)
    b = None
    for kv in range(KVH):                                  # rotary mean at each head's own candidate positions
        m_, b = centre_at(li, idx[kv], KVH, d, kb.device, S10, P9, P10, P11, Q7)
        mt[kv] = m_[kv] if m_.shape[1] == n else m_[kv].expand(n, d)
    return xs / b + mt


def attend(q, st, li, scale, M, K, lam, S10, Q5, P9, P10, P11, Q7, chunk=16384, softcap=None, sliding=None):
    """q [H, 1, d] fp32 (post-RoPE, the current row at position Lk - 1, Lk = st.W0 + rows in st.kx - Z_SINK). Returns
    [H, 1, d]."""
    H_, _, d = q.shape
    KVH = st.kx.shape[0]
    g = H_ // KVH
    h = d // 2
    W0 = st.W0
    nw = st.kx.shape[1] - min(Z_SINK, W0)
    Lk = W0 + nw
    dev = q.device
    if sliding and sliding < Lk:
        raise ValueError("cap_attn: sliding-window attention is not implemented")
    tf = (lambda x: torch.tanh(x / softcap) * softcap) if softcap else (lambda x: x)
    _, bvec = centre_at(li, None, KVH, d, dev, S10, P9, P10, P11, Q7)
    bq = bvec[:, 0, :].repeat_interleave(g, 0)
    order = S10.INFO[li]["order"].repeat_interleave(g, 0)
    qs = (q[:, 0] / bq).gather(-1, order)
    Hh = Q5.CQ.hadamard(h, dev)
    H = Q5.CQ.hadamard(d, dev)
    qh, ql = qs[:, :h] @ Hh, qs[:, h:] @ Hh
    ehi, elo = (qh ** 2).sum(-1), (ql ** 2).sum(-1)
    kxg, vxg = st.kx.float().repeat_interleave(g, 0), st.vx.float().repeat_interleave(g, 0)
    sex = tf(torch.einsum("hd,hnd->hn", q[:, 0], kxg) * scale)                  # sinks + window (exact rows)
    lo_c = Z_SINK                                                               # coded rows used: [Z, W0)
    if W0 <= lo_c:                                                              # nothing coded yet: exact attention
        return torch.einsum("hn,hnd->hd", torch.softmax(sex, -1), vxg)[:, None, :]
    s = torch.full((H_, W0), -float("inf"), device=dev)

    def s2_of(km):
        stp = C.be4(km[..., 0]).repeat_interleave(g, 0)
        mad = C.be4(km[..., 3]).abs().repeat_interleave(g, 0)
        return (ehi[:, None] * stp ** 2 / 12 + elo[:, None] * mad ** 2 * (math.pi / 2 - 1)) * scale * scale

    # ---- pass 1: base scores and the stage-1 normaliser
    lse1 = torch.logsumexp(sex, -1)
    for c0 in range(lo_c, W0, chunk):
        c1 = min(W0, c0 + chunk)
        code, bit = C.unpack_k(st.kb[:, c0:c1])
        km = st.km[:, c0:c1]
        yh = C.decode_hi(code, km[..., 0:1], km[..., 1:2])
        yl = C.decode_lo(bit, km[..., 2:3], km[..., 3:4])
        mt, _ = centre_at(li, torch.arange(c0, c1, device=dev), KVH, d, dev, S10, P9, P10, P11, Q7)
        sc = (torch.einsum("hd,hnd->hn", qh, yh.repeat_interleave(g, 0)) + torch.einsum("hd,hnd->hn", ql, yl.repeat_interleave(g, 0))
              + torch.einsum("hd,hnd->hn", q[:, 0], mt.expand(KVH, c1 - c0, d).repeat_interleave(g, 0))) * scale
        s[:, c0:c1] = sc
        lse1 = torch.logaddexp(lse1, torch.logsumexp(tf(sc), -1))
    # ---- stage 1: top-M of the group-mean probability over the coded rows (running top-M)
    Mx = min(M, W0 - lo_c)
    bv = torch.full((KVH, 0), -float("inf"), device=dev)
    bi = torch.zeros((KVH, 0), dtype=torch.long, device=dev)
    for c0 in range(lo_c, W0, chunk):
        c1 = min(W0, c0 + chunk)
        p1 = torch.exp(tf(s[:, c0:c1]) - lse1[:, None]).view(KVH, g, c1 - c0).mean(1)
        vv = torch.cat([bv, p1], -1)
        ii = torch.cat([bi, torch.arange(c0, c1, device=dev).expand(KVH, -1)], -1)
        t = vv.topk(min(Mx, vv.shape[-1]), -1)
        bv, bi = t.values, ii.gather(-1, t.indices)
    cand = bi                                                                   # [KVH, Mx]
    kcmask = torch.zeros(KVH, W0, dtype=torch.bool, device=dev).scatter_(-1, cand, True)
    # ---- host keys of the candidates: base key (decoded) + 4-bit residual (host records)
    kc_base = _dec_k_full(st, li, S10, Q5, P9, P10, P11, Q7, cand)
    ci = cand if st.kr.is_cuda else cand.cpu()
    kr = torch.stack([st.kr[kv][ci[kv]] for kv in range(KVH)]).to(dev, non_blocking=True)
    krm = torch.stack([st.krm[kv][ci[kv]] for kv in range(KVH)]).to(dev, non_blocking=True)
    kh = kc_base + C.decode_res(kr, krm[..., 0:1], krm[..., 1:2], getattr(st, "kbits", 4), d) @ H.T
    shc = tf(torch.einsum("hd,hmd->hm", q[:, 0], kh.repeat_interleave(g, 0)) * scale)   # [H, Mx]
    # ---- stage 2 / final normaliser over: exact rows, candidates (host keys), other coded rows (base - correction)
    lse2 = torch.logaddexp(torch.logsumexp(sex, -1), torch.logsumexp(shc, -1))
    for c0 in range(lo_c, W0, chunk):
        c1 = min(W0, c0 + chunk)
        x = s[:, c0:c1]
        if lam:
            x = x - (lam / 100) * s2_of(st.km[:, c0:c1]) / 2
        x = tf(x).masked_fill(kcmask[:, c0:c1].repeat_interleave(g, 0), -float("inf"))
        lse2 = torch.logaddexp(lse2, torch.logsumexp(x, -1))
    pc = torch.exp(shc - lse2[:, None])                                         # candidates' final probabilities
    c2 = pc.view(KVH, g, Mx).mean(1)
    Kx = min(K, Mx)
    selj = c2.topk(Kx, -1).indices                                              # positions within cand
    sel = cand.gather(-1, selj)                                                 # [KVH, Kx] rows fetched
    # ---- values: coded rows with base values (Hadamard coordinates), candidates with their own probabilities
    acc = torch.zeros(H_, d, device=dev)
    for c0 in range(lo_c, W0, chunk):
        c1 = min(W0, c0 + chunk)
        x = s[:, c0:c1]
        if lam:
            x = x - (lam / 100) * s2_of(st.km[:, c0:c1]) / 2
        p = torch.exp(tf(x) - lse2[:, None]).masked_fill(kcmask[:, c0:c1].repeat_interleave(g, 0), 0.0)
        yv = C.decode_v(C.unpack_v(st.vb[:, c0:c1]), st.vm[:, c0:c1, 0:1], st.vm[:, c0:c1, 1:2])
        acc += torch.einsum("hn,hnd->hd", p, yv.repeat_interleave(g, 0))
    yvc = C.decode_v(C.unpack_v(_take(st.vb, cand)), _take(st.vm, cand)[..., 0:1], _take(st.vm, cand)[..., 1:2])
    acc += torch.einsum("hm,hmd->hd", pc, yvc.repeat_interleave(g, 0))
    o = acc @ H.T
    o += torch.einsum("hn,hnd->hd", torch.exp(sex - lse2[:, None]), vxg)
    si = sel if st.vr.is_cuda else sel.cpu()
    vr = torch.stack([st.vr[kv][si[kv]] for kv in range(KVH)]).to(dev, non_blocking=True)
    vrm = torch.stack([st.vrm[kv][si[kv]] for kv in range(KVH)]).to(dev, non_blocking=True)
    vres = (C.decode_res(vr, vrm[..., 0:1], vrm[..., 1:2], getattr(st, "vbits", 4), d) @ H.T).repeat_interleave(g, 0)
    psel = pc.gather(-1, selj.repeat_interleave(g, 0))
    o += torch.einsum("hk,hkd->hd", psel, vres)
    return o[:, None, :]
