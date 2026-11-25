# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""rc_q: rotary-aware centring for the per-token key base (rootcause line; registered in reg_pkv.md).
A method name is any pkv7_q name, optionally followed by _rc. With _rc on an ours / bal base, keys are centred by the
calibration pre-RoPE mean rotated to each key's position, m(t) = R(t) m_pre, instead of the static post-RoPE mean m_post
of pkv5_q, and the channel balance ratio max|q| / max|k - m| is recomputed with that centring (same exponent rule as
pkv7_q). m_pre is O(d) per layer and the position t is known when a key is written, so storage, streaming and bits are
unchanged (eff_bits = pkv7_q.eff_bits of the name without _rc). Everything else (quantizers, clipping, window, fetch,
selection) is pkv7_q, imported read-only; its base7 is wrapped at runtime, the file is not modified.
"""
from __future__ import annotations

import torch

import pkv5_q as Q5
import pkv7_q as Q7

SINK = Q5.SINK
R: dict = {"m_pre": {}, "bal": {}, "inv_freq": None, "asc": 1.0, "on": False}
_base7 = Q7.base7


def rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


def cos_sin(L, dev):
    f = torch.arange(L, device=dev, dtype=torch.float64)[:, None] * R["inv_freq"].to(dev).double()[None]
    e = torch.cat([f, f], -1)
    return (e.cos() * R["asc"]).float(), (e.sin() * R["asc"]).float()


def derotate(k, cos, sin):
    rd = cos.shape[-1]
    kr = k[..., :rd]
    pre = (kr * cos - rot_half(kr) * sin) / (cos ** 2 + sin ** 2)
    return torch.cat([pre, k[..., rd:]], -1)


def rotate(m, cos, sin):
    """m [KVH, d] pre-RoPE -> [KVH, L, d] at positions 0..L-1."""
    rd = cos.shape[-1]
    mr = m[:, None, :rd].expand(-1, cos.shape[0], -1)
    out = mr * cos + rot_half(mr) * sin
    if rd < m.shape[-1]:
        out = torch.cat([out, m[:, None, rd:].expand(-1, cos.shape[0], -1)], -1)
    return out


def calib(li, k, v, q):
    Q7.calib(li, k, v, q)
    KVH, Lc, d = k.shape
    cos, sin = cos_sin(Lc, k.device)
    R["m_pre"][li] = derotate(k, cos, sin)[:, Q5.SINK:].mean(1)
    kc = (k - rotate(R["m_pre"][li], cos, sin))[:, Q5.SINK:]
    qm = q[:, Q5.SINK:].reshape(KVH, -1, d).abs().amax(1)
    R["bal"][li] = (qm / kc.abs().amax(1).clamp_min(1e-6)).clamp_min(1e-6)


def base7_rc(p, k, v, li):
    if not R["on"] or p["base"] not in ("ours", "bal"):
        return _base7(p, k, v, li)
    cos, sin = cos_sin(k.shape[1], k.device)
    mt = rotate(R["m_pre"][li], cos, sin)
    H = Q5.CQ.hadamard(k.shape[-1], k.device)
    clip = p["clip"]
    ptv = Q7.Q6.per_token_clip if clip == "1" else Q5.per_token
    vq = ptv(v @ H, 2, p["G"]) @ H.T
    ptk = Q7.Q6.per_token_clip if clip in ("1", "k") else Q5.per_token
    if clip == "w":
        raise ValueError("weighted clipping not supported with _rc")
    b = (R["bal"][li] ** (p["a"] / 100))[:, None, :] if p["base"] == "bal" else torch.ones_like(mt[:, :1])
    kq = (ptk(((k - mt) * b) @ H, 2, p["G"]) @ H.T) / b + mt
    kq[:, :1], vq[:, :1] = k[:, :1], v[:, :1]
    return kq, vq, p["R"], 1


Q7.base7 = base7_rc


def split(name):
    return (name[:-3], True) if name.endswith("_rc") else (name, False)


def parse(name):
    return Q7.parse(split(name)[0])


def eff_bits(name, L, d, g=1):
    return Q7.eff_bits(split(name)[0], L, d, g)


def attend(q, k, v, li, name, scale, kw):
    if name is None:
        return Q7.attend(q, k, v, li, None, scale, kw)
    n, rc = split(name)
    R["on"] = rc
    try:
        return Q7.attend(q, k, v, li, n, scale, kw)
    finally:
        R["on"] = False
