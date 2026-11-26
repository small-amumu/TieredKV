# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""v8g (round B1SK, amendment 2, 2026-11-19): the m8v8 value format of sel12_q with the stack's value group size.

sel12_q._v8 (frozen, 6d1a58f0) replicates and re-codes the stack's 2-bit per-token Hadamard values as ONE group over the
whole head dim. The stack itself (pkv7_q.base7: Q5.per_token(v @ H, 2, G), G from sel10_q.STACK = 128) uses groups of
min(G, d). The two agree at d <= 128 and differ at d = 256 (falcon3_3b; issue log 87), where _v8 refuses.

install() replaces, on the imported sel12_q module only (the file stays unchanged):
  - _v8 -> _v8g: groups of Gv = min(G, d), G read from the stack's own parse of sel10_q.STACK; per group an e4m3 step
    and an e4m3 zero in step units, the same saturation rule (fp16 zero when |zero / step| > 448, counted in
    CLIP["vzsat"] per group); the replication check uses the same Gv against the stack's values.
  - eff_bits: value metadata saved per element 16 / Gv (one 16-bit e4m3 pair per group) instead of 16 / d.
At d <= 128, Gv = d and both functions equal the originals bit for bit (v8g_selftest.py checks outputs and counters).
Key side unchanged: the stack codes each key half as one per-token group at any d (sel10_q._step_clip, lo mean / MAD),
so its metadata (4 x 16 bits per token and KV head) and sel12's 32 / d saving hold at d = 256 too."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

import sel12_q as S12

S12_MD5 = "6d1a58f046f59b77795fc5203ff9267b"
S0, S10, Q5, Q7, P10 = S12.S0, S12.S10, S12.Q5, S12.Q7, S12.P10
ORIG = {"v8": S12._v8, "eff_bits": S12.eff_bits}
STATE = {"installed": False, "calls": 0, "gv": {}}


def stack_g() -> int:
    """the value group size the stack uses, from its own parse of sel10_q.STACK (pkv7_q.base7 reads p["G"])."""
    _, n7 = P10.split(S10.STACK)
    p = Q7.parse(n7)
    if p.get("clip") == "1":
        raise RuntimeError("v8g: the stack's values use per_token_clip; _v8g replicates Q5.per_token only")
    return int(p["G"])


def _v8g(v: torch.Tensor, vq_ref: torch.Tensor) -> torch.Tensor:
    d = v.shape[-1]
    gv = min(stack_g(), d)
    if d % gv:
        raise RuntimeError(f"v8g: head dim {d} is not a multiple of the value group {gv}")
    H = Q5.CQ.hadamard(d, v.device)
    y = v @ H
    if not float((Q5.per_token(y, 2, gv) @ H.T - vq_ref)[:, 1:].abs().max() / v.abs().max().clamp_min(1e-6)) < 1e-4:
        raise RuntimeError("value quantizer replication differs from the stack")
    yg = y.reshape(*y.shape[:-1], d // gv, gv)
    mn, mx = yg.amin(-1, keepdim=True), yg.amax(-1, keepdim=True)
    raw = (mx - mn) / 3
    sc = torch.where(raw > 0, S0.e4(raw).clamp_min(S0.E4MIN), 0.0)
    safe = torch.where(sc > 0, sc, torch.ones_like(sc))
    zr = mn / safe
    sat = (zr.abs() > 448) & (sc > 0)
    S12.CLIP["vzsat"] += int(sat.sum())
    lo = torch.where(sat, Q5.f16(mn), S0.e4(zr) * sc)
    q = torch.where(sc > 0, torch.clamp(torch.round((yg - lo) / safe), 0, 3) * sc + lo, yg)
    out = q.reshape(y.shape) @ H.T
    out[:, :1] = vq_ref[:, :1]
    STATE["calls"] += 1
    STATE["gv"][str(d)] = gv
    return out.to(vq_ref.dtype)


def _eff_bits_g(name, L, d, g=1):
    s = S12.split(name)
    if not s or s[0] != "m8v8":
        return ORIG["eff_bits"](name, L, d, g)
    gpu, tot = S12.S10.eff_bits(s[1], L, d, g)
    gv = min(stack_g(), d)
    dk = 32 / d                                   # keys: m8 saves 4 x 8 bits per token and KV head (one group per half)
    dv = 16 / gv                                  # values: fp16 step + zero -> 2 x 8 per value group
    return gpu - (dk + dv) / 2, tot - (dk + dv) / 2


def install() -> None:
    got = hashlib.md5(Path(S12.__file__).read_bytes()).hexdigest()
    if got != S12_MD5:
        raise RuntimeError(f"v8g: sel12_q.py md5 {got} is not the frozen {S12_MD5}")
    if S12._v8 is not ORIG["v8"] or S12.eff_bits is not ORIG["eff_bits"]:
        raise RuntimeError("v8g: sel12_q already patched")
    S12._v8 = _v8g
    S12.eff_bits = _eff_bits_g
    STATE["installed"] = True
