# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""sel14_q (rootcause fork; round D2): the deployable host store. sel4_q's two-stage fetch (x2s<M>, K values) reads EXACT
rows for the fetched keys and values, i.e. it assumes an FP16 copy of the whole cache on the host (16 bits per element:
LLaMA-2-7B 512 KiB per token, so 95 GB of host memory holds only ~185K tokens). This module replaces that copy by a store
of residuals over the GPU base, as deployed (LLaMA-2-7B: 262144 elements per token, FP16 copy 512 KiB per token, so
95 GB holds ~181K tokens):
  host key  row j = k_base_j + Qrk(k_j - k_base_j),   host value row j = v_base_j + Qrv(v_j - v_base_j),
Q_b = Hadamard over the head dim, then per token b-bit min-max codes with an FP16 step and minimum (pkv5_q.per_token, one
group of d), i.e. host bits per element (rk + rv) / 2 + 32 / d (4.25 at rk = rv = 4, d = 128: 136 KiB per token for
LLaMA-2-7B, ~481K tokens in 67 GB). pkv5_q.aq keeps rows of zero range exact (no division by a zero step). Everything else is sel4_q's x2s path unchanged: first stage on the base scores, the M
candidates scored with the host keys (stage 2 and the final softmax), the K fetched rows read with the host values, the
logit correction (_j / _p, via sel4_q.parse as patched by sel13_q) on the remaining keys, sinks and the residual window
(sel4_q's rule: rows at or after floor((qpos + 1) / R) * R, plus 4 sinks) exact from the GPU FP16 copy.
A method name is <name>_h<rk><rv> with <name> a sel13_q / sel12_q / sel10_q / sel4_q name whose criterion is x2s<M> (plain,
_j<L> or _p<L> correction; not mass / eb / tw) and rk, rv in {2, ..., 8, f} (f = 16: exact rows, which must reproduce sel4_q:
checked whenever both are f, see HCHECK). Examples: stkm8v8128k64_x2s128_h44, stkm8v8128k64_x2s128_p50_h44.
No DIAG for _h runs (raises if sel4_q.DIAG is on). S3 dev checks (W-0031 / W-0034), as a trailing suffix after _h<rk><rv>:
  _id  identity: the base and the host store are replaced by the exact rows and the logit correction's variance by 0, so the
       method must reproduce exact attention (greedy predictions equal to the fp arm). It checks the plumbing (masks, sinks,
       window, softcap, softmax, output sum) only: with nothing compressed, selection / fetch / host-store errors are invisible;
  _lk  leakage: after the base codes and the host residual codes are formed from the exact cache (encoding), every exact row
       the deployed system does not keep (all but the 4 sinks and the residual window of the earliest query row of the call)
       is replaced by Gaussian noise before attention; the output must equal the plain _h output exactly. Decode calls only
       (one query row: the deployed rule keeps a per-row window), finite caches only, stack names only (their correction
       variance is read from the stored steps, never from k).
Mechanism: for an _h run, sel4_q's module-level _attend (called by sel4_q.attend after the dense-prefill split) is replaced
for the duration of the call by _attend_h; the routing (sel13 -> sel12 -> sel10 -> sel4: stack format, sel10 ST state)
is the normal one, so the base is exactly the one the method name selects."""
from __future__ import annotations

import re

import torch

import sel13_q as Q13

S12, S10, S4 = Q13.S12, Q13.S10, Q13.S4
NAME = re.compile(r"(.+)_h([2-8f])([2-8f])(?:_(id|lk))?")
BITS = {**{str(b): b for b in range(2, 9)}, "f": 16}
HCHECK: dict = {"on": True, "n": 0}        # with rk = rv = 16, compare with sel4_q's own _attend (counted)
if getattr(S4, "_sel14_installed", False):
    raise RuntimeError("sel14_q imported twice")
S4._sel14_installed = True
_s4_attend_inner = S4._attend                 # the original sel4_q._attend (sel10 / sel12 / sel13 do not replace it)


def _res(r, bits):
    if bits >= 16:
        return r
    d = r.shape[-1]
    H = S12.Q5.CQ.hadamard(d, r.device)
    return S12.Q5.per_token(r @ H, bits, d) @ H.T


def _attend_h(q, k, v, li, name, scale, kw, rk, rv, chk=None):
    if S4.DIAG["on"]:
        raise RuntimeError("sel14_q: DIAG is not implemented for _h runs")
    p = S4.parse(name)
    if p.get("jcal") or p.get("mass") or p["eb"] or p["tw"] or p["crit"][:2] != "2s" or not p["K"]:
        raise ValueError(f"{name}: _h needs an x2s<M> criterion with K > 0 and a plain / _j / _p correction")
    kq, vq, R, _ = S4.Q5.base("kivi128", k, v, li)
    z = 4
    kq[:, :z], vq[:, :z] = k[:, :z], v[:, :z]
    if chk == "id":                                                       # identity check: nothing compressed
        kq, vq = k.clone(), v.clone()
    kh = k if rk >= 16 else kq + _res(k - kq, rk)                         # host rows (sinks: exact, never read as host)
    vh = v if rv >= 16 else vq + _res(v - vq, rv)
    k_enc = k                                                             # the cache as encoded (correction variance)
    if chk == "lk":                                                       # leakage check: drop what is not kept
        Lk_, Lq_ = k.shape[1], q.shape[1]
        if Lq_ != 1 or not (S10.ST["on"] and S4.kivi_sigma2 is S10.stack_sigma2):
            raise ValueError(f"{name}_lk: decode calls (one query row) of stack names only")
        if not (torch.isfinite(k).all() and torch.isfinite(v).all()):
            raise ValueError(f"{name}_lk: non-finite cache")
        kp = torch.arange(Lk_, device=k.device)
        keep = (kp < z) | (kp >= (Lk_ - Lq_ + 1) // R * R)
        gen = torch.Generator(device=k.device).manual_seed(104729 + li)
        k = torch.where(keep[None, :, None], k, torch.randn(k.shape, generator=gen, device=k.device, dtype=k.dtype) * k.std())
        v = torch.where(keep[None, :, None], v, torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype) * v.std())
    K, M, lam = p["K"], int(p["crit"][2:]), p["lam"]
    H_, Lq, _ = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    kx, vx, kqx, vqx, khx, vhx = (t.repeat_interleave(g, 0) for t in (k, v, kq, vq, kh, vh))
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
        s2 = S4.kivi_sigma2(k_enc, q[:, r0:r1], scale) if lam else None   # stack: from the stored steps / MADs only
        if s2 is not None and chk == "id":
            s2 = torch.zeros_like(s2)
        if s2 is not None:
            s2 = s2.masked_fill(ex, 0.0)
        bad = ~al[None] | ex
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
        out[:, r0:r1] = o
    if rk >= 16 and rv >= 16 and HCHECK["on"] and chk is None:
        clip0 = dict(S12.CLIP)                                            # the reference call re-runs the base
        ref = _s4_attend_inner(q, k, v, li, name, scale, kw)
        S12.CLIP.update(clip0)
        err = float((ref - out).abs().max() / ref.abs().max().clamp_min(1e-6))
        if not err < 1e-5:
            raise RuntimeError(f"{name}_hff differs from sel4_q by {err:.3g} (layer {li})")
        HCHECK["n"] += 1
    return out


def split(name):
    m = NAME.fullmatch(name) if name else None
    return (m[1], BITS[m[2]], BITS[m[3]], m[4]) if m else None


def parse(name):
    s = split(name)
    return Q13.parse(s[0] if s else name)


def host_bits(name, d):
    """host store bits per element (mean over K and V) of an _h name; None for others (exact host copy = 16)."""
    s = split(name)
    if not s:
        return None
    return (_rowbits(s[1], d) + _rowbits(s[2], d)) / 2


def _rowbits(b, d):
    return b + 32 / d if b < 16 else 16.0


def eff_bits(name, L, d, g=1):
    s = split(name)
    if not s:
        return Q13.eff_bits(name, L, d, g)
    gpu, tot = Q13.eff_bits(s[0], L, d, g)          # sel4_q: tot = gpu + (M + K) * 8 / L, i.e. (M * 16 + K * 16) / (2 L)
    p = Q13.parse(s[0])
    M, K = int(p["crit"][2:]), p["K"]
    if abs((tot - gpu) - (M + K) * 8 / L) > 1e-9:
        raise RuntimeError(f"{name}: unexpected transfer accounting {tot - gpu} for M={M}, K={K}, L={L}")
    return gpu, gpu + (M * _rowbits(s[1], d) + K * _rowbits(s[2], d)) / (2 * L)   # residual rows carry FP16 step + min


def attend(q, k, v, li, name, scale, kw):
    s = split(name) if name else None
    if not s:
        return Q13.attend(q, k, v, li, name, scale, kw)
    base, rk, rv, chk = s

    def inner(q_, k_, v_, li_, name_, scale_, kw_):
        return _attend_h(q_, k_, v_, li_, name_, scale_, kw_, rk, rv, chk)

    S4._attend = inner
    try:
        return Q13.attend(q, k, v, li, base, scale, kw)
    finally:
        S4._attend = _s4_attend_inner


calib, calib_all = Q13.calib, Q13.calib_all
RC, Q7, P11, P10, P9, P8, Q5, S0 = Q13.RC, Q13.Q7, Q13.P11, Q13.P10, Q13.P9, Q13.P8, Q13.Q5, Q13.S0
PREFILL, CHUNK, ST, INFO, CLIP, VX = Q13.PREFILL, Q13.CHUNK, Q13.ST, Q13.INFO, Q13.CLIP, Q13.VX
