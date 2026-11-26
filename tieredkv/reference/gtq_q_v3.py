# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""gtq_q v3 (rootcause fork): Block-GTQ (arXiv 2606.24033, Liang, Zhang, Jia) inside the sel4_q / sel10_q harness, so that
it can be compared with the layered scheme (stk128 + two-stage fetch + debias) on the same windows and models, and combined
with our selection / two-stage fetch / debias. v1-v2 followed the paper's v1 text; v3 is aligned with the official code
(github.com/JIA-Lab-research/blockgtq @ 15269bc0, bench/bench_niah.py settings): norm correction on keys and values, keys
coded pre-RoPE (de-rotate the cache keys, code, rotate back; NoPE layers as stored), all segments rotated
(rotation_threshold = 1 in the official NIAH bench), the official codebook construction. Kept from v2 and recorded as
deviations: per-(layer, head, segment) random rotations instead of the official shared seeds, fp16 segment norms, our
calibration text, the 4 exact sinks of the harness, and protocol "lit" (the prompt is attended exactly; the official bench
also quantises the prefill). Known limitation: norm correction removes only part of the radial shrink at 1-2 bits.
Keys (per layer and KV head):
  score per RoPE 2D block i (channels i, i + d/2): s_i = (E ||q^(i)||^2 + E ||k^(i)||^2) / 2, uncentered second moments
  on the calibration text, averaged over tokens and the query heads of the GQA group (paper Eq. 1, 4);
  widths b_i in {1..8} per coordinate: start at 1, greedy one-bit increments to the block with the largest marginal gain
  3/4 s_i 4^(-b_i), until sum_i b_i = (d/2) * bk (paper Alg. 1; bk = average key bits per coordinate);
  blocks of equal width are concatenated into one segment; each segment of n coordinates is coded by TQ-MSE: its norm
  (fp16), a fixed random orthogonal rotation of the unit vector, and a Gaussian Lloyd-Max codebook of the segment's
  width scaled by 1 / sqrt(n) per rotated coordinate.
Values: uniform TQ-MSE over all d channels at bv bits (one fp16 norm per token).
By default no residual window (the paper's NIAH / LongBench setting); the harness keeps 4 exact sink tokens for every
method. Optional w<R> after the value bits (e.g. gtqk20v20w128k64_x2s128_j50) adds a KIVI-style residual window: keys at
positions >= (qpos + 1) // R * R stay exact, for a window-matched comparison with KIVI-128 and the stack. (Fix: the first version returned R = 0, which made every key exact; no window is now R = 1.)
Debias variance (zero storage): a segment's coordinates have error variance nc_dist(b, n) norm^2 each (the measured
per-coordinate MSE of a unit vector under v3 coding incl. norm correction; about D(b) / n without it), so
sigma_j^2 = scale^2 sum_seg ||q_seg||^2 nc_dist(b_seg, n_seg) norm_seg,j^2 (CPU check: empirical / predicted 1.01 / 1.00
at K2 / K3).
v3 codebooks are the official construction (same algorithm as blockgtq tq.py, 8e-13 in float64 for b = 1..8, float32
rounding 2.4e-7; at b >= 5 the official
Lloyd stops at a worse fixed point than a converged Lloyd-Max, kept for fidelity). Models with QK-norm before RoPE
(OLMo-2, Qwen3, Gemma-3) are out of scope: the official code codes k_proj before k_norm, which the cache cannot recover.
A method name is
  gtqk<K>v<V><rest>   K, V = 10 x average key / value bits per coordinate (gtqk20v20 = the paper's K2V2), rest as in
                      sel4_q kivi128<rest> (e.g. gtqk20v20k64_x2s128_j50: our two-stage fetch and debias on this base);
  any sel10_q name (stk128..., kivi128..., sel4_q / bl_q names) is passed through.
"""
from __future__ import annotations

import hashlib
import math
import re

import torch

import sel10_q as S10

S4, Q5 = S10.S4, S10.Q5
NAME = re.compile(r"gtqk(\d+)v(\d+)(?:w(\d+))?(.*)")
GST: dict = {"on": None, "li": None}       # (bk, bv, R) of the running gtq method
STATS: dict = {}                           # layer -> per-KV-head block scores [KVH, d/2]
ALLOC: dict = {}                           # (layer, bk) -> per-KV-head list of (channel index tensor, width)
INFO: dict = {}                            # layer -> (segments per KV head, key norms [KVH, Lk, nseg_max])
_CB: dict = {}
_DIST: dict = {}
_ROT: dict = {}
_DEV: dict = {}                            # device copies (rotations, codebooks, channel indices)


def _official_unit_centroids(b):
    """the official codebook (blockgtq tq.py _optimal_centroids_torch) for a unit normal; the official one for
    dimension n is this / sqrt(n). b = 1: +-sqrt(2 / pi); b = 2: the rounded constants; b >= 3: Lloyd started at the
    quantile boundaries, one centroid pass and 100 iterations (at b >= 5 this stops at a worse fixed point than a
    converged Lloyd-Max; kept for fidelity, revD)."""
    if b == 1:
        c = math.sqrt(2.0 / math.pi)
        return torch.tensor([-c, c], dtype=torch.float64)
    if b == 2:
        return torch.tensor([-1.51, -0.453, 0.453, 1.51], dtype=torch.float64)
    n = 1 << b
    nd = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    bd = nd.icdf(torch.linspace(0, 1, n + 1, dtype=torch.float64)[1:-1])

    def cents(bd):
        lo = torch.cat([torch.tensor([-math.inf], dtype=torch.float64), bd])
        hi = torch.cat([bd, torch.tensor([math.inf], dtype=torch.float64)])
        pdf = lambda t: torch.where(torch.isfinite(t), torch.exp(-t * t / 2) / math.sqrt(2 * math.pi), torch.zeros_like(t))
        prob = nd.cdf(hi) - nd.cdf(lo)
        return (pdf(lo) - pdf(hi)) / prob.clamp_min(1e-300)
    c = cents(bd)
    for _ in range(100):
        c = cents((c[:-1] + c[1:]) / 2)
    return c.sort().values


def codebook(b):
    """unit-normal codebook (2^b levels; the official construction, v3) and its MSE distortion, computed once."""
    if b not in _CB:
        x = torch.linspace(-8, 8, 400001, dtype=torch.float64)
        w = torch.exp(-x * x / 2)
        c = _official_unit_centroids(b)
        idx = torch.bucketize(x, (c[1:] + c[:-1]) / 2)
        _DIST[b] = float(((x - c[idx]) ** 2 * w).sum() / w.sum())
        _CB[b] = c.float()
    return _CB[b]


_NCD: dict = {}


def nc_dist(b, n):
    """per-coordinate MSE of a unit vector in R^n under v3 coding (Gaussian codebook / sqrt(n), then norm correction),
    measured once on 65536 Haar-uniform unit vectors (fixed seed; relative SE about 0.3%). Replaces D(b) / n of the
    uncorrected codec in the debias variance: with correction the distortion is about 1.11 / 1.02 / 0.98 D(b) at
    b = 1 / 2 / 3 (revE)."""
    if (b, n) not in _NCD:
        g = torch.Generator().manual_seed(1000 * b + n)
        u = torch.randn(65536, n, generator=g, dtype=torch.float64)
        u = u / u.norm(dim=-1, keepdim=True)
        cb = codebook(b).double() / math.sqrt(n)
        uq = cb[torch.bucketize(u, (cb[1:] + cb[:-1]) / 2)]
        uq = uq / uq.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        _NCD[(b, n)] = float(((u - uq) ** 2).sum(-1).mean() / n)
    return _NCD[(b, n)]


def rotation(n, key, dev):
    """fixed random orthogonal n x n matrix per key (layer, KV head, segment / value)."""
    if (key, n) not in _ROT:
        g = torch.Generator().manual_seed(int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16))
        a, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
        _ROT[(key, n)] = a.float()
    kd = ("rot", key, n, str(dev))
    if kd not in _DEV:
        _DEV[kd] = _ROT[(key, n)].to(dev)
    return _DEV[kd]


def tq_mse(x, b, key):
    """TQ-MSE of the rows of x [..., n]: fp16 norm, fixed rotation, Lloyd-Max codebook of width b scaled by 1/sqrt(n),
    then the official norm correction (v3; official tq.py, norm_correction=True by default): the quantised rotated
    unit vector is renormalised to unit length before the inverse rotation."""
    n = x.shape[-1]
    nrm = Q5.f16(x.norm(dim=-1, keepdim=True))
    u = x / nrm.clamp_min(1e-12)
    R = rotation(n, key, x.device)
    y = u @ R
    kd = ("cb", b, n, str(x.device))
    if kd not in _DEV:
        _DEV[kd] = codebook(b).to(x.device) / math.sqrt(n)
    cb = _DEV[kd]
    yq = cb[torch.bucketize(y.contiguous(), (cb[1:] + cb[:-1]) / 2)]
    yn = yq.norm(dim=-1, keepdim=True)
    yq = yq / torch.where(yn > 1e-10, yn, torch.ones_like(yn))
    return (yq @ R.T) * nrm, nrm[..., 0]


def calib(li, k, v, q):
    S10.calib(li, k, v, q)
    KVH, _, d = k.shape
    g = q.shape[0] // KVH
    h = d // 2
    ek = (k.float() ** 2).mean(1)                                                   # [KVH, d]
    eq = (q.float() ** 2).mean(1).view(KVH, g, d).mean(1)
    STATS[li] = ((eq[:, :h] + eq[:, h:]) + (ek[:, :h] + ek[:, h:])).cpu() / 2       # [KVH, d/2]


QKNORM_TYPES = {"olmo2", "olmo3", "qwen3", "qwen3_moe", "gemma3", "gemma3_text"}


def calib_all(model, kv, alist):
    """the main line's calibration, then the v3 scope check: Block-GTQ arms need full calibration (NoPE layers known),
    a static RoPE (default / llama3 rope_type) and no QK-norm before RoPE; otherwise gtq_base refuses to run."""
    out = S10.calib_all(model, kv, alist)
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    rs = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None) or {}
    rt = (rs.get("rope_type") or rs.get("type") or "default") if isinstance(rs, dict) else "default"
    mt = getattr(cfg, "model_type", "")
    bad = []
    if rt not in ("default", "llama3", "linear"):
        bad.append(f"rope_type {rt} changes inv_freq at runtime")
    if mt in QKNORM_TYPES or any(hasattr(m, "k_norm") for m in model.modules()):
        bad.append(f"QK-norm before RoPE ({mt}): the official code codes k_proj before k_norm")
    GST["scope"] = bad
    return out


def allocate(li, bk):
    """greedy integer widths per RoPE block (paper Alg. 1) -> per KV head [(channel indices, width), ...]."""
    if (li, bk) in ALLOC:
        return ALLOC[(li, bk)]
    s = STATS[li]
    KVH, h = s.shape
    budget = round(h * bk)
    out = []
    for kh in range(KVH):
        w = [1] * h
        left = budget - h
        while left > 0:
            gain = [0.75 * float(s[kh, i]) * 4.0 ** (-w[i]) if w[i] < 8 else -1.0 for i in range(h)]
            i = max(range(h), key=gain.__getitem__)
            if gain[i] < 0:
                break
            w[i] += 1
            left -= 1
        segs = []
        for b in sorted(set(w)):
            blk = torch.tensor([i for i in range(h) if w[i] == b])
            segs.append((torch.cat([blk, blk + h]), b))
        out.append(segs)
    ALLOC[(li, bk)] = out
    return out


def _dev_idx(ch, dev):
    kd = ("ch", id(ch), str(dev))
    if kd not in _DEV:
        _DEV[kd] = (ch, ch.to(dev))                          # keep ch alive so id() stays unique
    return _DEV[kd][1]


def gtq_base(k, v, li, bk, bv, R=0):
    if GST.get("scope") is None:
        raise RuntimeError("gtq_q v3: calib_all did not run (NoPE layers and the model scope are unknown)")
    if GST["scope"]:
        raise RuntimeError("gtq_q v3: model out of scope: " + "; ".join(GST["scope"]))
    KVH, Lk, d = k.shape
    segs = allocate(li, bk)
    # v3: keys are coded pre-RoPE, as the official code (it quantises the k_proj output): the cache keys at positions
    # 0..Lk-1 are de-rotated, coded, and rotated back. NoPE layers are coded as stored.
    pre = li not in S10.P11.NOPE
    if pre and RC.R["inv_freq"] is None:
        raise RuntimeError("gtq_q v3: RC.R['inv_freq'] unset; pre-RoPE coding needs the model's rotary frequencies")
    kf = k.float()
    if pre:
        cos, sin = RC.cos_sin(Lk, k.device)
        if cos.shape[-1] != d:
            raise RuntimeError(f"gtq_q v3: partial rotary ({cos.shape[-1]} of {d} channels) is not supported")
        kf = RC.derotate(kf, cos, sin)
    kq = torch.empty_like(kf)
    nmax = max(len(sg) for sg in segs)
    norms = torch.zeros(KVH, Lk, nmax, device=k.device)
    for kh in range(KVH):
        for si, (ch, b) in enumerate(segs[kh]):
            ch = _dev_idx(ch, k.device)
            xr, nr = tq_mse(kf[kh][:, ch], b, ("k", li, kh, si))
            kq[kh][:, ch] = xr
            norms[kh, :, si] = nr
    if pre:
        kq = kq * cos + RC.rot_half(kq) * sin
        norms = norms * RC.R["asc"]                  # segment norms of the stored (post-RoPE) keys, for the debias
    vq = torch.empty_like(v, dtype=torch.float32)
    for kh in range(KVH):
        vq[kh] = tq_mse(v[kh].float(), int(round(bv)), ("v", li, kh))[0]
    INFO[li] = (segs, norms)
    # R = residual window as sel4_q reads it: keys at positions >= (qpos + 1) // R * R stay exact. R = 1 gives no
    # window (the paper's no-buffer setting); R = 0 (the first version) made every key exact (division by zero)
    return kq.to(k.dtype), vq.to(v.dtype), (R if R > 0 else 1), 0


_base_prev = S10._Q5Shim.base


def _shim_base(self, name, k, v, li):
    if GST["on"] is not None and name == "kivi128":
        return gtq_base(k, v, li, *GST["on"])
    return _base_prev(self, name, k, v, li)


S10._Q5Shim.base = _shim_base
_sigma_prev = S10._kivi_sigma2


def gtq_sigma2(k, q, scale, G=128):
    if GST["on"] is None:
        return _sigma_prev(k, q, scale, G)
    segs, norms = INFO[GST["li"]]
    KVH = k.shape[0]
    g = q.shape[0] // KVH
    out = None
    for kh in range(KVH):
        qh = q[kh * g:(kh + 1) * g].float()                                         # [g, n, d]
        acc = 0.0
        for si, (ch, b) in enumerate(segs[kh]):
            n = ch.numel()
            qe = (qh[..., _dev_idx(ch, q.device)] ** 2).sum(-1)                    # [g, n]
            acc = acc + qe[:, :, None] * nc_dist(b, n) * (norms[kh, :, si] ** 2)[None, None, :]
        part = acc * (scale * scale)
        out = part if out is None else torch.cat([out, part], 0)
    return out


S10._kivi_sigma2 = gtq_sigma2


def split(name):
    m = NAME.fullmatch(name)
    return (int(m[1]) / 10, int(m[2]) / 10, "kivi128" + m[4], int(m[3] or 0)) if m else None


def parse(name):
    s = split(name)
    return S4.parse(s[2]) if s else S10.parse(name)


def nseg_mean(bk):
    a = [len(sg) for (li, b), al in ALLOC.items() if b == bk for sg in al]
    return sum(a) / len(a) if a else float("nan")


def eff_bits(name, L, d, g=1):
    s = split(name)
    if not s:
        return S10.eff_bits(name, L, d, g)
    bk, bv, rest, W = s
    gpu0, tot0 = S4.eff_bits(rest, L, d, g)
    for li in STATS:                                                               # every bk (not only the first)
        allocate(li, bk)
    kb = bk + 16 * nseg_mean(bk) / d
    vb = bv + 16 / d
    gpu = (kb + vb) / 2
    # 4 exact sinks + the residual window at its mean size W / 2 over decode rows (as pkv5_q.eff_bits counts KIVI's)
    gpu += (16 - gpu) * min(1.0, (4 + (W / 2 if W > 1 else 0)) / L)
    return gpu, tot0 - gpu0 + gpu


def attend(q, k, v, li, name, scale, kw):
    s = split(name) if name else None
    if not s:
        return S10.attend(q, k, v, li, name, scale, kw)
    GST["on"], GST["li"] = (s[0], s[1], s[3]), li
    try:
        return S4.attend(q, k, v, li, s[2], scale, kw)
    finally:
        GST["on"] = None


RC, Q7, P11, P10, P9, P8 = S10.RC, S10.Q7, S10.P11, S10.P10, S10.P9, S10.P8
PREFILL, CHUNK = S10.PREFILL, S10.CHUNK
