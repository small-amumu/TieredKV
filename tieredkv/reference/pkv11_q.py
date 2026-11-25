# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""pkv11_q: round P11 of the layered-KV line (registered in reg_pkv.md). Two changes over pkv10_q:
  (1) NoPE-aware rotary centring: layers without RoPE (SmolLM3 uses none in every fourth layer) get static centring and the
      static water-filling ranking even when a name asks for _rc (rc_q rotates the mean in every layer, which is wrong
      there; P9 / P10 numbers of SmolLM3 used that version). Models without NoPE layers are unchanged bit for bit.
  (2) the pkv5_q baselines kvq (KVQuant-style), kitty, gear and rot<G> (per-token Hadamard 2-bit, QuaRot-style) run
      through pkv5_q.attend in the same harness, with pkv5_q.eff_bits.
Everything else is pkv10_q (read-only import).
"""
from __future__ import annotations

import pkv10_q as P10

Q5, Q7, P8, P9, RC = P10.Q5, P10.Q7, P10.P8, P10.P9, P10.RC
SINK = P10.SINK
NOPE: set = set()
P5NAMES = ("kvq", "kitty", "gear", "rot128")
NAMES = ("kivi128_z4", "kivi128_z4_r192", "kivi32_z4", "kvq", "kitty", "gear", "rot128",
         "bal128_kh_ck_w128_rc", "wfc_ck_w128_rc", "kivi128k112_k8_v8_z4", "wfck112_k8_v8_ck_w128_rc")
calib10 = P10.calib10


def parse(name):
    if name in P5NAMES:
        return Q5.parse(name)
    return P10.parse(name)


def eff_bits(name, L, d, g=1):
    if name in P5NAMES:
        return tuple(Q5.eff_bits(name, L, d))
    return P10.eff_bits(name, L, d, g)


def attend(q, k, v, li, name, scale, kw):
    if name is None:
        return Q7.attend(q, k, v, li, None, scale, kw)
    if name in P5NAMES:
        return Q5.attend(q, k, v, li, name, scale, kw)
    f, n7 = P10.split(name)
    if li in NOPE and f["rc"]:
        f["rc"] = False
        if f["src"] in ("s", "q"):
            f["src"] = "c"
    P10.S10.update(f)
    try:
        return Q7.attend(q, k, v, li, n7, scale, kw)
    finally:
        P10.S10.update(src=None, kh=False, rc=False, q4=False, wv=False, vh=False, r=0)


def set_nope(cfg):
    """NoPE layers from the config (SmolLM3: no_rope_layers[i] == 0 means layer i has no RoPE)."""
    NOPE.clear()
    flags = getattr(cfg, "no_rope_layers", None)
    if flags:
        NOPE.update(i for i, u in enumerate(flags) if not u)
    return sorted(NOPE)
