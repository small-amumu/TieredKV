# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""rkv_q (round B1c, 2026-11-14): RotateKV (Su et al., github.com/ZunhaiSu/RotateKV @ 3055e2b6) inside the
sel4_q / sel10_q / gtq_q_v3 harness, so it is compared with our stack base and Block-GTQ on the same windows and models.
Every step follows the official code path of the RotateKV2/3/4 setting (utils.py: head_group_num 4, grouped-head key
rotation, outlier-aware key reordering, attention-sink-aware pivots; modeling_llama_RotateKV.py LlamaFlashAttention2;
fuse_weights.py; k_reordering_calibration.py; fake_quant.py):
  keys   (pre-RoPE, as the official k_proj output): grouped-head Walsh-Hadamard over G consecutive KV heads (dimension
         G * d, normalised; fuse_weights rotate_reorder_k_proj), then the calibrated permutation of all KVH * d channels,
         then per-token group-wise asymmetric round-to-nearest, one group per head slot of d channels (official quant_gs = 128 =
         head_dim of its Llama / Mistral models; fake_quant.fake_quant, clip 1.0), then the
         inverse permutation (argsort), the inverse rotation and RoPE;
  values per-head Walsh-Hadamard (fuse_weights rotate_ov_proj), the same per-token group-wise quantiser, inverse rotation;
  pivots the 20 tokens holding the 20 largest |hidden state| entries of the layer input stay exact in keys and values
         (official: topk(k=20) over the flattened hidden states of the forward that fills the cache, i.e. the prefill /
         prompt; at decode the official code quantises every new token). In this harness the prompt is rows <
         PREFILL["from"] (the dense exact prefill) and the scored decode rows read the quantised cache of every earlier
         row, so the lab computes the pivots per layer from the layer input restricted to those prompt rows through
         set_pivots() (one full-window forward per arm; no chunked forward). Duplicates collapse (<= 20 tokens).
Calibration of the permutation (k_reordering_calibration.py): per layer, the signed sum over calibration tokens of the
grouped-rotated pre-RoPE keys, flattened to KVH * d channels, sorted ascending.
Recorded deviations: group = d (= the official 128 when head_dim is 128; the official code fails for other head dims);
G = min(4, KVH) when KVH < 4 or not a multiple of 4 (official: 4); calibration on our calibration text
(official: its own calibration samples); the harness keeps 4 exact sinks and attends the prompt exactly (protocol "lit", as
every harness arm; the official PPL script also quantises the prompt); scales / zeros are kept in float32 as in fake_quant
but counted as 16 bits each in eff_bits. Same scope as gtq_q_v3 (no QK-norm before RoPE, full rotary, static RoPE).
The official fake_quant computes the scale with maxq = bit**2 - 1 (equal to 2**bit - 1 only for 2 and 4 bits); it is kept.
A method name is
  rkv<b>[w<R>]   b = key and value bits (2, 3 or 4); optional w<R> = a KIVI-style FP16 residual window of R tokens (the
                 window-matched control; RotateKV itself has none). No fetch suffix (base only).
  any gtq_q_v3 name (gtqk..., stk128..., kivi128..., sel4_q names) passes through.
"""
from __future__ import annotations

import math
import re

import torch

import gtq_q_v3 as G

S10, S4, RC = G.S10, G.S4, G.RC
NAME = re.compile(r"rkv([234])(?:w(\d+))?")
RST: dict = {"on": None, "li": None, "scope": None}
IDX: dict = {}                     # layer -> permutation of KVH * d channels (LongTensor, cpu)
PIV: dict = {}                     # layer -> exact pivot token indices of the current forward (LongTensor)
_HAD: dict = {}
NPIV = 20


def hadamard(n: int, dev) -> torch.Tensor:
    """normalised Sylvester Walsh-Hadamard matrix (the ordering of fast_hadamard_transform), cached per (n, device)."""
    key = (n, str(dev))
    if key not in _HAD:
        if n & (n - 1):
            raise RuntimeError(f"rkv_q: Hadamard size {n} is not a power of two")
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HAD[key] = (h / math.sqrt(n)).to(dev, torch.float32)
    return _HAD[key]


def group_of(kvh: int) -> int:
    return 4 if kvh % 4 == 0 else (2 if kvh % 2 == 0 else 1)


# --- official fake_quant.py, verbatim semantics (torch ops only) ---------------------------------------------------------
def find_params_per_token_groupwise(x, clip_ratio, bit, quant_gs):
    init_shape = x.shape
    reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // quant_gs, quant_gs)
    xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * clip_ratio
    xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * clip_ratio
    maxq = torch.tensor(bit ** 2 - 1)
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    scale = scale.repeat(1, 1, 1, quant_gs).reshape(init_shape)
    zero = zero.repeat(1, 1, 1, quant_gs).reshape(init_shape)
    return scale, zero


def fake_quant(x, clip_ratio, bit, quant_gs):
    scale, zero = find_params_per_token_groupwise(x, clip_ratio, bit, quant_gs)
    q = torch.clamp(torch.round(x / scale) + zero, 0, torch.tensor(2 ** bit - 1))
    return scale * (q - zero)


# --- transforms ----------------------------------------------------------------------------------------------------------
def k_forward(kpre: torch.Tensor, li: int) -> torch.Tensor:
    """[KVH, L, d] pre-RoPE keys -> [L, KVH * d] in the official cache space (grouped rotation, then reordering)."""
    KVH, L, d = kpre.shape
    g = group_of(KVH)
    x = kpre.permute(1, 0, 2).reshape(L, KVH // g, g * d) @ hadamard(g * d, kpre.device)
    x = x.reshape(L, KVH * d)
    return x[:, IDX[li].to(kpre.device)] if li in IDX else x


def k_inverse(x: torch.Tensor, li: int, KVH: int, d: int) -> torch.Tensor:
    L = x.shape[0]
    g = group_of(KVH)
    if li in IDX:
        x = x[:, IDX[li].to(x.device).argsort()]
    x = x.reshape(L, KVH // g, g * d) @ hadamard(g * d, x.device)      # Walsh-Hadamard is symmetric and involutory
    return x.reshape(L, KVH, d).permute(1, 0, 2)


def calib(li, k, v, q):
    G.calib(li, k, v, q)


def calib_all(model, kv, alist):
    """gtq_q_v3's calibration and scope check, then the RotateKV permutation per layer from the calibration keys."""
    out = G.calib_all(model, kv, alist)
    RST["scope"] = G.GST.get("scope")
    for li in alist:
        k = kv[li][0].float()                                           # post-RoPE calibration keys [KVH, L, d]
        KVH, L, d = k.shape
        if li in S10.P11.NOPE:
            kpre = k
        else:
            cos, sin = RC.cos_sin(L, k.device)
            kpre = RC.derotate(k, cos, sin)
        x = k_forward(kpre, li)                                         # IDX not set yet: grouped rotation only
        _, IDX[li] = torch.sort(x.sum(0).cpu())
    return out


def set_pivots(li: int, hidden: torch.Tensor, rows: int) -> None:
    """official pivot rule on the layer input [L, hidden] restricted to the prompt rows (< rows)."""
    h = hidden[:rows].float()
    _, top = torch.topk(h.abs().flatten(), k=min(NPIV, h.numel()))
    PIV[li] = torch.unique(top // h.shape[-1])


def rkv_base(k, v, li, bits, R):
    if RST["scope"] is None:
        raise RuntimeError("rkv_q: calib_all did not run")
    if RST["scope"]:
        raise RuntimeError("rkv_q: model out of scope: " + "; ".join(RST["scope"]))
    if li not in IDX:
        raise RuntimeError(f"rkv_q: no reordering indices for layer {li}")
    KVH, Lk, d = k.shape
    kf = k.float()
    pre = li not in S10.P11.NOPE
    if pre:
        cos, sin = RC.cos_sin(Lk, k.device)
        if cos.shape[-1] != d:
            raise RuntimeError(f"rkv_q: partial rotary ({cos.shape[-1]} of {d} channels) is not supported")
        kf = RC.derotate(kf, cos, sin)
    x = k_forward(kf, li)                                               # [Lk, KVH * d]
    xq = fake_quant(x[None, None], 1.0, bits, d)[0, 0]          # one group per head slot (official quant_gs = head_dim)
    kq = k_inverse(xq, li, KVH, d)
    if pre:
        kq = kq * cos + RC.rot_half(kq) * sin
    Hd = hadamard(d, k.device)
    vr = v.float() @ Hd                                                 # [KVH, Lk, d]
    vq = fake_quant(vr[None], 1.0, bits, d)[0] @ Hd
    piv = PIV.get(li)
    if piv is not None and piv.numel():
        p = piv.to(k.device)
        p = p[p < Lk]
        kq[:, p] = k.float()[:, p]
        vq[:, p] = v.float()[:, p]
    return kq.to(k.dtype), vq.to(v.dtype), (R if R > 0 else 1), 0


_base_prev = S10._Q5Shim.base


def _shim_base(self, name, k, v, li):
    if RST["on"] is not None and name == "kivi128":
        return rkv_base(k, v, li, *RST["on"])
    return _base_prev(self, name, k, v, li)


S10._Q5Shim.base = _shim_base


def split(name):
    m = NAME.fullmatch(name) if name else None
    return (int(m[1]), int(m[2] or 0)) if m else None


def parse(name):
    return S4.parse("kivi128") if split(name) else G.parse(name)


def eff_bits(name, L, d, g=1):
    s = split(name)
    if not s:
        return G.eff_bits(name, L, d, g)
    bits, W = s
    gpu = bits + 32 / d                                                # FP16 scale and zero per token and group
    exact = 4 + NPIV + (W / 2 if W > 1 else 0)                          # sinks, pivots (upper bound), mean window
    gpu += (16 - gpu) * min(1.0, exact / L)
    return gpu, gpu


def attend(q, k, v, li, name, scale, kw):
    s = split(name)
    if not s:
        return G.attend(q, k, v, li, name, scale, kw)
    RST["on"], RST["li"] = s, li
    try:
        return S4.attend(q, k, v, li, "kivi128", scale, kw)
    finally:
        RST["on"] = None


STATS, GST = G.STATS, G.GST
P11, P10, P9, P8, Q7, Q5 = G.P11, G.P10, G.P9, G.P8, G.Q7, G.Q5
PREFILL, CHUNK = G.PREFILL, G.CHUNK
