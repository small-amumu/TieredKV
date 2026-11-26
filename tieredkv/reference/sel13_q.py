# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""sel13_q (rootcause fork; diagnosis round D1, exploratory): sel12_q plus two controls.
A method name is
  <name>_vx    the method <name> with every value exact: the base's value codes are replaced by the exact values before
               attention; keys, selection, two-stage fetch and the logit correction are unchanged (stack and kivi128 names
               alike, e.g. stk128k64_x2s128_j50_vx, kivi128k64_x2s128_j50_vx). KL(<name>_vx) keeps only the score (key) error,
               so KL(<name>_vx) / KL(<name>) is the score-only share and 1 minus it the part removed by exact values (KL is
               not additive: the remainder includes the interaction of value and score errors).
  <base>_p<L>  sel4_q's _j<L> with the opposite sign: the unrefined keys' logits get + L / 100 * sigma_j^2 / 2 instead of
               - L / 100 * sigma_j^2 / 2, in both stages of a two-stage name (as _j<L>), e.g. stk128k64_x2s128_p50.
  any other name passes through to sel12_q unchanged.
Mechanisms: (1) sel10_q routes every base through its _Q5Shim.base (the stack while sel10's ST is on, pkv5_q.base for kivi128
otherwise); this module wraps that method once and, while a _vx run is active, returns the exact values in place of the
value codes (asserting float32 and equal shapes; every replacement is counted in VX["n"]). (2) sel4_q's parse is wrapped
once: a name ending in _p<L> is parsed as _j<L> and its "lam" negated; sel4_q's attend reads the module-level parse, so both
the first-stage and the final correction take the negated fraction. Non-control names reach the unwrapped code paths."""
from __future__ import annotations

import re

import torch

import sel12_q as S12

S10 = S12.S10
S4 = S10.S4
BLQ = S4.BLQ
NAME = re.compile(r"(.+)_vx")
PNAME = re.compile(r"(.+)_p(\d+)")
VX: dict = {"on": False, "n": 0}
_shim_base = S10._Q5Shim.base
_s4_parse = S4.parse

if getattr(_shim_base, "_sel13", False) or getattr(_s4_parse, "_sel13", False):
    raise RuntimeError("sel13_q imported twice: hooks already installed")


def _base_vx(self, name, k, v, li):
    out = _shim_base(self, name, k, v, li)
    if not VX["on"]:
        return out
    kq, vq, R, z0 = out
    if not (vq.dtype == v.dtype == torch.float32 and vq.shape == v.shape):
        raise RuntimeError(f"_vx: value codes {vq.dtype} {tuple(vq.shape)} vs exact {v.dtype} {tuple(v.shape)}")
    VX["n"] += 1                                    # base calls whose values were replaced
    return kq, v.clone(), R, z0


def _parse13(name):
    m = PNAME.fullmatch(name)
    if not m:
        return _s4_parse(name)
    p = dict(_s4_parse(f"{m[1]}_j{m[2]}"))
    if p.get("mass") or not p["lam"]:
        raise ValueError(f"{name}: _p needs a plain _j<L> counterpart with L > 0")
    p["lam"] = -p["lam"]
    return p


_base_vx._sel13 = True
_parse13._sel13 = True
S10._Q5Shim.base = _base_vx
S4.parse = _parse13


def split(name):
    m = NAME.fullmatch(name)
    return m[1] if m else None


def parse(name):
    s = split(name)
    return S12.parse(s if s else name)


def eff_bits(name, L, d, g=1):
    s = split(name)
    return S12.eff_bits(s if s else name, L, d, g)     # _vx is a control, not a deployable format: bits of <name>


def attend(q, k, v, li, name, scale, kw):
    s = split(name) if name else None
    if not s:
        return S12.attend(q, k, v, li, name, scale, kw)
    if BLQ.is_bl(s):
        raise ValueError(f"{name}: _vx is defined for stack / kivi128 names only")
    n0 = VX["n"]
    VX["on"] = True
    try:
        out = S12.attend(q, k, v, li, s, scale, kw)
    finally:
        VX["on"] = False
    if VX["n"] == n0:
        raise RuntimeError(f"{name}: no value replacement happened in layer {li}")
    return out


calib, calib_all = S12.calib, S12.calib_all
RC, Q7, P11, P10, P9, P8 = S12.RC, S12.Q7, S12.P11, S12.P10, S12.P9, S12.P8
Q5 = S12.Q5
S0 = S12.S0
PREFILL, CHUNK, ST, INFO, CLIP = S12.PREFILL, S12.CHUNK, S12.ST, S12.INFO, S12.CLIP
