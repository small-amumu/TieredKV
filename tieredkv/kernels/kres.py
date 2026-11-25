# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""kres (TieredKV, S4 residual-format fetch; 2026-11-07). The two-stage fix of kbench7.Step7 (mode 'two') reading
b-bit residual records from pinned host memory instead of exact fp16 rows (the h33 / h54 formats of the accuracy runs:
cap_codec.encode_res, one min / step pair per token and KV head, codes of the Hadamard-rotated residual packed as a
little-endian bit stream, or as nibbles j | j+64 << 4 when b = 4, exactly as cap_codec._pack_bits). Host bytes per token and KV head: 16 * b + 4 per tensor (b = 3: 52 B vs 256 B fp16).
Residual definition (what makes the scores exact up to the residual quantisation):
  key    r_k = k - (k_hat_c + R(t) m)  (k_hat_c the decoded centred base key, R(t) m the rotary centre whose q-product is the
         base score's bias), stored as Hadamard coordinates H r_k (H = normalised 128-point Hadamard, symmetric orthonormal)
  value  r_v = v - v_hat (v_hat the decoded 2-bit base value), stored as H r_v
Kernels (copies of kbench7.fix2A_kernel_wf / kbench5.fix2B_kernel with the exact-row reads replaced):
  fix2A_res  base score sb of each of the M candidates (unchanged), then se = sb + scale * (H q) . r_hat_k: no key is
             decoded; (H q) is one tf32x3 dot per program; r_hat_k = code * step + min
  fix2B_res  the merge weight of the base values becomes (eb - ee) for every candidate and the selected candidates add
             sum_j ee_j r_hat_v,j, accumulated in Hadamard coordinates (tf32x3 dot) and rotated back once per program
             (H = H^-1); the base-value dots are kbench5's (fp16 hi / lo split, ~fp32)
Check (python kres.py --check): Step7 with exact rows set to k_hat_full + H r_hat_k and v_hat + H r_hat_v (the values the
residual path represents) vs Step7R on the same base, q and residual bytes: outputs must agree to ~1e-3 relative (the
only difference is the fp16 bias of the base score vs the fp32 rotary centre in the exact rows), and the selections equal.
Timing (--time): attention only, 32 layers in one CUDA graph, exact fp16 rows vs residual b = 3 / 4,5 (h33, h54) at the
lat_overhead shapes."""
import argparse
import hashlib
import json
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kbench4 as K4  # noqa: E402
import kbench5 as K5  # noqa: E402
import kbench7 as K7  # noqa: E402
import kenc as KE  # noqa: E402
from kbench4 import _planes, _kth_bits  # noqa: E402
from kbench5 import _crit  # noqa: E402
from kbench7 import _lo_pair, _wf_q, _wf_scores, _rcbias  # noqa: E402

D, P, KVH, G, GP = K7.D, K7.P, K7.KVH, K7.G, K7.GP


@triton.jit
def _res_rows(rp_ptr, rm_ptr, row, mask, BITS: tl.constexpr, NBY: tl.constexpr):
    """decoded residual rows [R, 128] fp32 (Hadamard coordinates): code * step + min; rows outside mask give 0."""
    c = tl.arange(0, 128)
    base = rp_ptr + row[:, None].to(tl.int64) * NBY
    if BITS == 4:      # cap_codec packs 4-bit codes as nibbles: byte j = code j | code j+64 << 4
        bj = tl.load(base + (c % 64)[None, :], mask=mask[:, None], other=0).to(tl.int32) & 255
        code = (bj >> (4 * (c // 64))[None, :]) & 15
    else:              # little-endian bit stream: bit w of code j is stream bit j * BITS + w
        p = c * BITS
        i0 = p // 8
        sh = p % 8
        b0 = tl.load(base + i0[None, :], mask=mask[:, None], other=0).to(tl.int32) & 255
        b1 = tl.load(base + i0[None, :] + 1, mask=mask[:, None] & (i0[None, :] + 1 < NBY), other=0).to(tl.int32) & 255
        code = ((b0 | (b1 << 8)) >> sh[None, :]) & ((1 << BITS) - 1)
    mn = tl.load(rm_ptr + row.to(tl.int64) * 2, mask=mask, other=0.0).to(tl.float32)
    st = tl.load(rm_ptr + row.to(tl.int64) * 2 + 1, mask=mask, other=0.0).to(tl.float32)
    return tl.where(mask[:, None], code.to(tl.float32) * st[:, None] + mn[:, None], 0.0)


@triton.jit
def _res_rows_w(rp_ptr, rm_ptr, row, mask, BITS: tl.constexpr, NBY: tl.constexpr):
    """_res_rows with one contiguous int32-word load per row (coalesced, one pass over the host bytes), then the codes are
    taken from registers with tl.gather instead of two byte loads per code from host memory."""
    NW: tl.constexpr = NBY // 4                       # words per row (12 / 16 / 20 for b = 3 / 4 / 5)
    NWP: tl.constexpr = 32                            # padded to a power of two
    c = tl.arange(0, 128)
    ow = tl.arange(0, NWP)
    rp32 = rp_ptr.to(tl.pointer_type(tl.int32))
    words = tl.load(rp32 + row[:, None].to(tl.int64) * NW + ow[None, :], mask=mask[:, None] & (ow[None, :] < NW), other=0)
    R: tl.constexpr = words.shape[0]
    if BITS == 4:      # nibbles: byte j = code j | code j+64 << 4; byte j lives in word j // 4 at bits 8 (j % 4)
        j = c % 64
        w = tl.gather(words, tl.broadcast_to((j // 4)[None, :], (R, 128)), axis=1)
        code = (w >> (8 * (j % 4) + 4 * (c // 64))[None, :]) & 15
    else:
        p = c * BITS
        w0 = tl.gather(words, tl.broadcast_to((p // 32)[None, :], (R, 128)), axis=1).to(tl.uint32).to(tl.int64)
        w1 = tl.gather(words, tl.broadcast_to(tl.minimum(p // 32 + 1, NWP - 1)[None, :], (R, 128)), axis=1).to(tl.uint32).to(tl.int64)
        code = (((w0 | (w1 << 32)) >> (p % 32)[None, :]) & ((1 << BITS) - 1)).to(tl.int32)
    mn = tl.load(rm_ptr + row.to(tl.int64) * 2, mask=mask, other=0.0).to(tl.float32)
    st = tl.load(rm_ptr + row.to(tl.int64) * 2 + 1, mask=mask, other=0.0).to(tl.float32)
    return tl.where(mask[:, None], code.to(tl.float32) * st[:, None] + mn[:, None], 0.0)


@triton.jit
def fix2A_res(qh_ptr, qw_ptr, qs2_ptr, idx_ptr, krp_ptr, krm_ptr, kp_ptr, meta_ptr, meta2_ptr, bias_ptr, u_ptr, w_ptr,
              c0_ptr, s0_ptr, csb_ptr, sc_out, L, sm_scale, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr,
              M: tl.constexpr, KC: tl.constexpr, RC: tl.constexpr, BB: tl.constexpr, M8: tl.constexpr, KB: tl.constexpr,
              WORDS: tl.constexpr = False):
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
    qh = tl.load(qh_ptr + (bk * GH + og[:, None]) * D4 + od[None, :], mask=gm[:, None], other=0.0)   # H q, fp32
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
    if RC == 3:
        sb += _rcbias(u_ptr, w_ptr, c0_ptr, s0_ptr, csb_ptr, bk, tok, og, gm, GH, GP, KC, D4, BB, 32) * sm_scale
    if WORDS:
        rk = _res_rows_w(krp_ptr, krm_ptr, row, row >= 0, KB, 16 * KB)
    else:
        rk = _res_rows(krp_ptr, krm_ptr, row, row >= 0, KB, 16 * KB)              # [KC, 128]
    se = sb + tl.dot(qh, tl.trans(rk), input_precision="tf32x3") * sm_scale
    so = sc_out + (bk * 2 * GP + og[:, None]) * M + ok[None, :]
    tl.store(so, se)
    tl.store(so + GP * M, sb)


@triton.jit
def fix2B_res(m_in, l_in, sc_in, idx_ptr, vrp_ptr, vrm_ptr, hpm_ptr, vp_ptr, meta_ptr, pn_out, pz_out, pm_out, selm_out, L,
              nsplit, kap_out, GH: tl.constexpr, GP: tl.constexpr, P: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
              CC: tl.constexpr, NS: tl.constexpr, MP: tl.constexpr, RADIX: tl.constexpr, VB: tl.constexpr,
              WORDS: tl.constexpr = False):
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
    wb = eb - ee                                     # every candidate: base value weight (eb) minus its exact mass (ee)
    if WORDS:
        rv = _res_rows_w(vrp_ptr, vrm_ptr, row, sel, VB, 16 * VB)
    else:
        rv = _res_rows(vrp_ptr, vrm_ptr, row, sel, VB, 16 * VB)                   # [CC, 128], selected rows only
    od = tl.arange(0, D4)
    hpm = tl.load(hpm_ptr + od[:, None] * D4 + od[None, :])                       # +-1 Sylvester matrix, fp16 (exact)
    rr = tl.dot(ees, rv, input_precision="tf32x3")                                 # sum ee r_v in Hadamard coordinates
    rh_ = rr.to(tl.float16)
    rl_ = ((rr - rh_.to(tl.float32)) * 2048.0).to(tl.float16)                      # hi / lo split: ~22-bit mantissa
    racc = (tl.dot(rh_, hpm) + tl.dot(rl_, hpm) * (1.0 / 2048.0)) * 0.08838834764831845   # x H = x Hpm / sqrt(128)
    e0, e1, e2, e3 = _planes(racc, GP, P)
    mt = tl.load(meta_ptr + row[:, None] * 4 + tl.arange(0, 4)[None, :]).to(tl.float32)
    sc_, zp_ = tl.split(tl.reshape(mt, [CC, 2, 2]))
    ksr, vsc = tl.split(sc_)
    kzr, vzp = tl.split(zp_)
    wv = wb * vsc[None, :]
    wbh = wv.to(tl.float16)
    wbl = ((wv - wbh.to(tl.float32)) * 2048.0).to(tl.float16)
    wbz = tl.sum(wb * vzp[None, :], 1)[:, None]
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


def hadamard128(dev="cuda"):
    return K7.hadamard(128, dev).float().contiguous()


class Step7R(K7.Step7):
    """Step7 (mode 'two') whose fix reads b-bit residual records (kbits, vbits) from pinned host memory."""

    def __init__(self, B, L, K, rc, kbits=3, vbits=3, res_host=None, words=False, **kw):
        super().__init__(B, L, K, rc, **kw)
        assert self.mode == "two" and self.wf, "residual fetch implemented for the water-filled two-stage path"
        assert all(2 <= b <= 8 for b in (kbits, vbits))
        self.kbits, self.vbits, self.words = kbits, vbits, bool(words)
        BK = self.BK
        if res_host is None:
            res_host = (torch.zeros(BK, L, 16 * kbits, dtype=torch.int8, pin_memory=True),
                        torch.zeros(BK, L, 2, dtype=torch.float16, pin_memory=True),
                        torch.zeros(BK, L, 16 * vbits, dtype=torch.int8, pin_memory=True),
                        torch.zeros(BK, L, 2, dtype=torch.float16, pin_memory=True))
        self.krp, self.krm, self.vrp, self.vrm = res_host
        cv = (lambda t: t) if self.krp.is_cuda else K4.host_as_cuda
        self.krp_, self.krm_, self.vrp_, self.vrm_ = (cv(t) for t in res_host)
        self.H128 = hadamard128()
        self.Hpm = (self.H128 * (128 ** 0.5)).round().half().contiguous()        # entries exactly +-1
        assert torch.equal(self.Hpm.float() / 128 ** 0.5, self.H128) or (self.Hpm.float() / 128 ** 0.5 - self.H128).abs().max() < 1e-7
        self.qh = torch.empty(self.BK * G, D, device="cuda")

    def fix(self):
        torch.matmul(self.q.float(), self.H128, out=self.qh)                       # H q once per layer and step
        self.kinfo["fix2A"] = fix2A_res[(self.BK, self.NC2 // self.kca)](
            self.qh, self.qw, self.qs2, self.idx, self.krp_, self.krm_, self.kp, self.mt, self.lo_ptr(), self.bias,
            *self._rt(), self.sc, self.L, self.scale, GH=G, GP=GP, P=P, M=self.NC2, KC=self.kca, RC=self.rck, BB=self.BB,
            M8=self.m8, KB=self.kbits, WORDS=self.words, num_warps=4)
        fix2B_res[(self.BK, self.nch2)](self.m, self.l, self.sc, self.idx, self.vrp_, self.vrm_, self.Hpm, self.vp, self.mt,
                                        self.pn, self.pz, self.pm, self.selm, self.L, self.nsplit, self.kap, GH=G, GP=GP, P=P,
                                        M=self.NC2, K=self.Kv, CC=self.cc, NS=self.NS, MP=triton.next_power_of_2(self.NC2),
                                        RADIX=self.radix, VB=self.vbits, WORDS=self.words, num_warps=4)
        return K4.fixB_kernel[(self.BK, 4)](self.m, self.l, self.acc, self.pn, self.pz, self.pm, self.out, self.nsplit,
                                            self.nch2, GH=G, GP=GP, P=P, NS=self.NS, NC=triton.next_power_of_2(self.nch2),
                                            UNROLL=self.unroll, num_warps=4)


def decoded_base(st):
    """the base keys / values the kernels represent, in plain coordinates [BK, L, D] fp32: centred key k_hat_c (water-filled
    decode, inverse Hadamard per half, inverse permutation, / balance), the rotary centre R(t) m in fp64, and v_hat."""
    BK, L = st.BK, st.L
    kp = st.kp.long()
    code = torch.cat([kp & 7, (kp >> 3) & 7], -1).float()
    bit = torch.cat([(kp >> 6) & 1, (kp >> 7) & 1], -1).float()
    mt, lo = st.mt.float(), st.lo_eff()
    yq = torch.cat([(code * mt[..., 0:1] + mt[..., 1:2]) @ st.H64.T, (lo[..., 0:1] + lo[..., 1:2] * bit) @ st.H64.T], -1)
    kvh = torch.arange(BK, device=kp.device) % KVH
    pp = st.perm[kvh].long()[:, None, :].expand(BK, L, D)
    khc = torch.empty_like(yq).scatter_(2, pp, yq) / st.qbal[kvh][:, None, :].float()
    t = torch.arange(L, dtype=torch.float64, device=kp.device)[:, None] * st.invf64[None]
    cos, sin = torch.cat([t.cos(), t.cos()], 1), torch.cat([t.sin(), t.sin()], 1)
    m = st.mpre[kvh].double()
    rh = torch.cat([-m[:, 64:], m[:, :64]], -1)
    rm = (m[:, None, :] * cos[None] + rh[:, None, :] * sin[None]).float() if st.rck == 2 else 0.0
    vhat = K5._deq(st.vp, st.mt[..., 2], st.mt[..., 3])
    return khc + rm, vhat


def encode_residual(x, xhat, H, bits):
    """residual records of x - xhat in Hadamard coordinates (cap_codec.encode_res arithmetic) and their decoded values."""
    import cap_codec as C
    f16 = lambda t: t.half().float()  # noqa: E731
    packed, mn, sc, nz, rec = C.encode_res((x - xhat) @ H, f16, bits)
    return packed.view(torch.int8), torch.cat([mn, sc], -1), rec


def check(B=1, L=4096):
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {}
    v = dict(K7.VARIANTS["two128k64"]); mode, K = v.pop("mode"), v.pop("K")
    kw = dict(v, qbias=False, fuse_w=False)
    kw.update(K7.WF_CFG(L, 2, B, "wf"))
    for kb, vb, words in ((3, 3, False), (5, 4, False), (3, 3, True), (5, 4, True), (4, 4, True)):
        st = K7.Step7(B, L, K, 2, wf=True, win=0, mode=mode, seed=11, **kw)
        sr = Step7R(B, L, K, 2, kbits=kb, vbits=vb, wf=True, win=0, mode=mode, seed=11, words=words, **kw)
        assert torch.equal(st.kp, sr.kp) and torch.equal(st.q, sr.q), "base / q differ between the two steps"
        gb = torch.Generator(device="cpu").manual_seed(5)
        bal = (torch.rand(KVH, D, generator=gb) * 1.5 + 0.5).cuda()                # non-unit balance (exercises / qbal)
        for x in (st, sr):
            x.qbal.copy_(bal); x.build_T()
        H = hadamard128()
        assert (H @ H - torch.eye(128, device="cuda")).abs().max() < 1e-6, "H is not an involution"
        khat, vhat = decoded_base(st)
        kx, vx = st.kh.cuda().float(), st.vh.cuda().float()                      # true exact rows (random)
        kq, krm, krec = encode_residual(kx, khat, H, kb)
        vq, vrm, vrec = encode_residual(vx, vhat, H, vb)
        sr.krp.copy_(kq.cpu()); sr.krm.copy_(krm.cpu()); sr.vrp.copy_(vq.cpu()); sr.vrm.copy_(vrm.cpu())
        # the reference: exact rows = what the residual records represent
        st.kh.copy_((khat + krec @ H).half().cpu()); st.vh.copy_((vhat + vrec @ H).half().cpu())
        st.step(); sr.step(); torch.cuda.synchronize()
        rel = ((st.out - sr.out).norm() / st.out.norm()).item()
        M = sr.NC2                                                     # score level: residual path vs q . exact rows
        scs = sr.sc.view(sr.BK, 2, GP, M)[:, :, :G]
        idx = sr.idx[:, :M]
        qf = sr.q.float().view(sr.BK, G, D)
        kref = torch.gather(st.kh.cuda().float(), 1, idx[:, :, None].expand(-1, -1, D))
        se_ref = torch.einsum("bgd,bmd->bgm", qf, kref) * sr.scale
        se_err = ((scs[:, 0] - se_ref).abs().max() / se_ref.abs().max()).item()
        same_idx = bool(torch.equal(st.idx, sr.idx))
        sel_eq = bool(torch.equal(st.selm, sr.selm))
        out[f"h{kb}{vb}{'w' if words else ''}"] = dict(out_relerr=rel, cand_equal=same_idx, sel_equal=sel_eq, se_max_rel_err=se_err,
                                 res_bytes_per_token_head=2 * (16 * kb + 4) if kb == vb else (16 * kb + 4) + (16 * vb + 4))
        nm = f"h{kb}{vb}{'w' if words else ''}"
        print(nm, json.dumps(out[nm]), flush=True)
    ok = all(r["out_relerr"] < 2e-3 and r["sel_equal"] and r["se_max_rel_err"] < 2e-3 for r in out.values())
    print("CHECK", "PASS" if ok else "FAIL", "| gates: output and candidate exact-score max rel. err. < 2e-3 vs exact rows = base + decoded residual (fp16), same selection, non-unit balance")
    return out


def timing(out, shapes, reps, warm, win=128):
    torch.backends.cuda.matmul.allow_tf32 = False
    here = os.path.dirname(os.path.abspath(__file__))
    md5 = {}
    for f in ("kres.py", "kbench7.py", "kbench5.py", "kbench4.py", "kenc.py"):
        with open(os.path.join(here, f), "rb") as fh:
            md5[f] = hashlib.md5(fh.read()).hexdigest()[:8]
    NL, H = 32, 32
    for shape in shapes.split(","):
        B, L = map(int, shape.split("x"))
        v = dict(K7.VARIANTS["two128k64"]); mode, K = v.pop("mode"), v.pop("K")
        kw = dict(v, qbias=True, maxnreg=168, fuse_w=win > 0); kw.update(K7.WF_CFG(L, 2, B, "wf"))
        q = torch.randn(B * H, D, device="cuda").half()
        rec = dict(B=B, L=L, win=win, device=torch.cuda.get_device_name(0), md5=md5, clock_before=KE.gpu_state())
        for name, kb, vb in (("fp16_rows", None, None), ("h33", 3, 3), ("h54", 5, 4), ("h33w", 3, 3), ("h54w", 5, 4)):
            sts = hosts = rh = None
            try:
                if kb is None:
                    per = 2 * B * KVH * L * D * 2
                    nc = max(1, min(NL, int(32 * 2 ** 30 // per)))
                    hosts = [(torch.zeros(B * KVH, L, D, dtype=torch.float16, pin_memory=True),
                              torch.zeros(B * KVH, L, D, dtype=torch.float16, pin_memory=True)) for _ in range(nc)]
                    sts = [K7.Step7(B, L, K, 2, wf=True, win=win, mode=mode, seed=li, host=hosts[li % nc], **kw) for li in range(NL)]
                else:
                    per = B * KVH * L * (16 * kb + 16 * vb + 8)
                    nc = max(1, min(NL, int(32 * 2 ** 30 // per)))
                    rh = [(torch.zeros(B * KVH, L, 16 * kb, dtype=torch.int8, pin_memory=True),
                           torch.zeros(B * KVH, L, 2, dtype=torch.float16, pin_memory=True),
                           torch.zeros(B * KVH, L, 16 * vb, dtype=torch.int8, pin_memory=True),
                           torch.zeros(B * KVH, L, 2, dtype=torch.float16, pin_memory=True)) for _ in range(nc)]
                    sts = [Step7R(B, L, K, 2, kbits=kb, vbits=vb, wf=True, win=win, mode=mode, seed=li, res_host=rh[li % nc], words=name.endswith("w"),
                                  host=(torch.zeros(B * KVH, 1, D, dtype=torch.float16, pin_memory=True),) * 2, **kw) for li in range(NL)]
                for st in sts:
                    st.q.copy_(q)
                m, lo, hi = KE.time_graph(lambda: [st.step() for st in sts], warm, reps)
                rec[name] = dict(status="ok", ms=m, iqr10=[lo, hi], copies=nc)
            except torch.cuda.OutOfMemoryError as e:
                rec[name] = dict(status="oom", ms=None, error=repr(e)[:200])
            except Exception as e:  # noqa: BLE001
                rec[name] = dict(status="error", ms=None, error=repr(e)[:300])
            sts = hosts = rh = None
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        rec["clock_after"] = KE.gpu_state()
        print(json.dumps({k: (rec[k]["ms"] if isinstance(rec.get(k), dict) and "ms" in rec[k] else rec.get(k))
                          for k in ("B", "L", "fp16_rows", "h33", "h54", "h33w", "h54w")}), flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true"); ap.add_argument("--time", action="store_true")
    ap.add_argument("--out", default=""); ap.add_argument("--shapes", default="1x8192,1x32768,8x8192,8x32768")
    ap.add_argument("--reps", type=int, default=50); ap.add_argument("--warm", type=int, default=10)
    a = ap.parse_args()
    if a.check:
        check()
    if a.time:
        timing(a.out, a.shapes, a.reps, a.warm)
