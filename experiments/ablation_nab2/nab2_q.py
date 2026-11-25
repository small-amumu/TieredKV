# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""NAB2 (REG_NAB2.md): ablation arms of NAB1's deployed configuration, as call-scoped patches
around the frozen stack.

Arms (decode replay; every arm is routed through sel30_q.attend exactly as in NAB1):
  kivi128k64_x2r128_j50_h54   random candidates: stage 1 ranks keys by sel4_q's x2r criterion
                              (a deterministic hash) instead of the base attention; stage 2 on the
                              host keys, the top K = 64 value rows, the j50 correction and the h54
                              host codec are unchanged (route: NAB1's name).
  kivi128k64_x2s128_h54       lam = 0 (no logit correction): a native stack name, no patch.
  kivi128k64_x2s128_j50_h5n   key-only correction: as NAB1's arm, but the fetched value rows keep
                              their 2-bit base values, so no value residual is stored or read
                              (route: NAB1's name).
  kivi4b128                   KIVI 4-bit ruler: the kivi128 base (per-channel K, per-token V,
                              groups of 128, the 128-token FP16 window and 4 FP16 sinks of the
                              stack) at 4 bits instead of 2; no host store (route: kivi128, with
                              pkv5_q.base patched to 4 bits for the call).
The patched sel14_q._attend_h is built from the frozen function's own source (inspect.getsource)
with three textual edits, each asserted to apply exactly once; with both ablations off it must
equal sel14_q._attend_h bit for bit (nab2_selftest.py). The x2r hash is executed from the lines
of the frozen sel4_q._attend themselves (asserted present there), so it cannot drift from sel4_q.
Every patched call is counted; a call whose patch does not fire exactly once raises.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

import torch

import pkv5_q as Q5
import sel14_q as Q14
import sel30_q as Q

NAB1_OURS = "kivi128k64_x2s128_j50_h54"
LAM0 = "kivi128k64_x2s128_h54"
ARMS: dict[str, dict[str, Any]] = {
    "kivi128k64_x2r128_j50_h54": {"route": NAB1_OURS, "rand": True, "vcorr": True,
                                  "kivi_bits": None},
    LAM0: {"route": LAM0, "rand": False, "vcorr": True, "kivi_bits": None},
    "kivi128k64_x2s128_j50_h5n": {"route": NAB1_OURS, "rand": False, "vcorr": False,
                                  "kivi_bits": None},
    "kivi4b128": {"route": "kivi128", "rand": False, "vcorr": True, "kivi_bits": 4},
    # self-test arms only (nab2_selftest.py): every ablation off, must reproduce their routes
    "t_variant_off": {"route": NAB1_OURS, "rand": False, "vcorr": True, "kivi_bits": None},
    "t_base2": {"route": "kivi128", "rand": False, "vcorr": True, "kivi_bits": 2},
}
TEST_ARMS = ("t_variant_off", "t_base2")
STATE: dict[str, Any] = {"rand": False, "vcorr": True, "calls_h": 0, "calls_base": 0}

# sel4_q's x2r criterion, verbatim from sel4_q._attend (checked at import, then executed as is)
_X2R_LINES = (
    "hh = torch.arange(H_, device=q.device)[:, None, None] * 1.0",
    "rr = qpos[None, :, :].float()",
    "kk = kpos[None, :, :].float()",
    "c1 = torch.frac(torch.sin(li * 3.17 + hh * 12.9898 + rr * 78.233 + kk * 37.719)"
    " * 43758.5453)",
)
_OLD_C1 = 'c1 = fin(s).masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)'
_NEW_C1 = ('c1 = (_nab2_hash(li, H_, q, qpos, kpos) if _NAB2["rand"] else fin(s))'
           '.masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)')
_OLD_V = '+ torch.einsum("hqk,hkd->hqd", pr * m, vhx - vqx))'
_NEW_V = '+ (torch.einsum("hqk,hkd->hqd", pr * m, vhx - vqx) if _NAB2["vcorr"] else 0.0))'
_OLD_HEAD = '    if S4.DIAG["on"]:'
_NEW_HEAD = '    _NAB2["calls_h"] += 1\n    if S4.DIAG["on"]:'


def _build_hash() -> Callable[..., torch.Tensor]:
    """sel4_q's x2r hash of (layer, head, row, key) in (-1, 1) (torch.frac keeps the sign),
    shape [H_, n, Lk]; only its ranking is used."""
    src4 = inspect.getsource(Q14.S4._attend)
    missing = [ln for ln in _X2R_LINES if ln not in src4]
    if missing:
        raise RuntimeError(f"nab2_q: sel4_q's x2r hash changed, missing {missing}")
    # the four lines must be one consecutive block of sel4_q._attend, as they run there
    lines = [ln.strip() for ln in src4.splitlines()]
    starts = [i for i, ln in enumerate(lines) if ln == _X2R_LINES[0]]
    if len(starts) != 1 or lines[starts[0]:starts[0] + len(_X2R_LINES)] != list(_X2R_LINES):
        raise RuntimeError("nab2_q: sel4_q's x2r hash lines are not one consecutive block")
    body = "".join(f"    {ln}\n" for ln in _X2R_LINES)
    src = f"def _nab2_hash(li, H_, q, qpos, kpos):\n{body}    return c1\n"
    ns: dict[str, Any] = {"torch": torch}
    exec(compile(src, "<nab2_q._nab2_hash>", "exec"), ns)  # noqa: S102 (sel4_q's own lines)
    return ns["_nab2_hash"]


def _build_variant(hash_fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    # the source must be the one of the loaded module (not another copy on disk)
    if Path(inspect.getsourcefile(Q14._attend_h)).resolve() != Path(Q14.__file__).resolve():
        raise RuntimeError("nab2_q: sel14_q._attend_h source file is not the loaded module")
    if {"_NAB2", "_nab2_hash", "_nab2_attend_h"} & Q14.__dict__.keys():
        raise RuntimeError("nab2_q: sel14_q already defines a name nab2_q would write")
    src = inspect.getsource(Q14._attend_h)
    for old, new in ((_OLD_C1, _NEW_C1), (_OLD_V, _NEW_V), (_OLD_HEAD, _NEW_HEAD)):
        if src.count(old) != 1:
            raise RuntimeError(f"nab2_q: expected exactly one '{old[:40]}...' in _attend_h")
        src = src.replace(old, new)
    # the variant runs in sel14_q's own (live) module namespace, under its own name, so it sees every
    # module global exactly as the frozen _attend_h does, including any rebinding after import
    head = "def _attend_h("
    if src.count(head) != 1:
        raise RuntimeError("nab2_q: expected exactly one 'def _attend_h(' in its source")
    src = src.replace(head, "def _nab2_attend_h(")
    ns = Q14.__dict__
    ns["_NAB2"], ns["_nab2_hash"] = STATE, hash_fn
    exec(compile(src, "<nab2_q._attend_h>", "exec"), ns)  # noqa: S102 (the stack's own source)
    return ns["_nab2_attend_h"]


def _validate() -> None:
    """import-time check: the lam = 0 arm differs from NAB1's arm only in lam."""
    p_ours, p_l0 = Q.parse(NAB1_OURS), Q.parse(LAM0)
    if not (isinstance(p_ours, dict) and isinstance(p_l0, dict)):
        raise RuntimeError("nab2_q: unexpected parse result")

    def rest(p: dict) -> dict:
        return {k: v for k, v in p.items() if k != "lam"}

    if p_l0.get("lam") != 0 or p_ours.get("lam") != 50 or rest(p_l0) != rest(p_ours):
        raise RuntimeError(f"nab2_q: lam arm parse {p_l0} vs NAB1 arm {p_ours}")
    if Q14.split(LAM0)[1:3] != Q14.split(NAB1_OURS)[1:3]:
        raise RuntimeError("nab2_q: lam arm host codec differs from NAB1's arm")


_NAB2_HASH = _build_hash()
_ATTEND_H = _build_variant(_NAB2_HASH)
# a call counter around the frozen _attend_h (installed after the variant read its source): native
# arms record how often they run it, so lam0 and the deployed arm are shown to use the same function
# the same number of times; the wrapper only counts and returns the frozen function's result
_FROZEN_H = Q14._attend_h


def _counting_attend_h(*args: Any, **kwargs: Any) -> Any:
    STATE["calls_frozen"] += 1
    return _FROZEN_H(*args, **kwargs)


STATE["calls_frozen"] = 0
Q14._attend_h = _counting_attend_h
_BASE0 = Q5.base
_validate()


def _base_bits(bits: int) -> Callable[..., tuple]:
    def base(name: str, k: torch.Tensor, v: torch.Tensor, li: int) -> tuple:
        STATE["calls_base"] += 1
        kind, G, sel = Q5.parse(name)
        if kind != "kivi" or sel:
            raise ValueError(f"nab2_q: the {bits}-bit ruler patches the plain kivi base only, "
                             f"got {name}")
        return Q5.per_channel(k, bits, G), Q5.per_token(v, bits, G), 128, sel
    return base


def is_arm(name: str | None) -> bool:
    return bool(name) and name in ARMS


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, li: int, name: str, scale: float,
           kw: dict) -> torch.Tensor:
    """one attention call of a NAB2 arm: patch, route through sel30_q.attend, restore."""
    a = ARMS[name]
    if a["kivi_bits"] is not None:
        c0 = STATE["calls_base"]
        Q5.base = _base_bits(a["kivi_bits"])
        try:
            out = Q.attend(q, k, v, li, a["route"], scale, kw)
        finally:
            Q5.base = _BASE0
        fired = STATE["calls_base"] - c0
        if fired != 1:                       # every routed call uses the patched base exactly once
            raise RuntimeError(f"{name}: patched base fired {fired} times in one call (layer {li})")
        return out
    if a["route"] == name:                   # native stack name (lam = 0): no patch, no count
        return Q.attend(q, k, v, li, name, scale, kw)
    old = Q14._attend_h
    c0 = STATE["calls_h"]
    STATE["rand"], STATE["vcorr"] = a["rand"], a["vcorr"]
    Q14._attend_h = _ATTEND_H
    try:
        out = Q.attend(q, k, v, li, a["route"], scale, kw)
    finally:
        Q14._attend_h = old
        STATE["rand"], STATE["vcorr"] = False, True
    fired = STATE["calls_h"] - c0
    if fired != 1:                           # every routed call runs the patched _attend_h once
        raise RuntimeError(f"{name}: patched _attend_h fired {fired} times in one call "
                           f"(layer {li})")
    return out


def bits(name: str, L: int, d: int, g: int = 1) -> dict[str, float]:
    """GPU bits per stored element, bits read per element per decode step, host bits per element
    (K/V means)."""
    a = ARMS[name]
    if a["kivi_bits"] is not None:           # the stack's kivi128 accounting at b bits:
        b = a["kivi_bits"] + 32 / Q5.parse(a["route"])[1]   # FP16 step + minimum per group of G
        w = min(1.0, (128 / 2) / L)          # pkv5_q.eff_bits: mean-occupancy FP16 window
        gpu = b * (1 - w) + 16 * w
        gpu += (16 - gpu) * 4 / L            # sel4_q.eff_bits: the 4 FP16 sink rows
        return {"gpu": gpu, "read": gpu, "host": 0.0}
    gpu, read = Q.eff_bits(a["route"], L, d, g)
    host = Q.host_bits(a["route"], d)
    if not a["vcorr"]:                       # value rows are not stored or read on the host
        p = Q.parse(a["route"])
        K = p["K"]
        rv = Q14.split(a["route"])[2]
        read -= K * (rv + 32 / d) / (2 * L)
        host -= (rv + 32 / d) / 2
    return {"gpu": gpu, "read": read, "host": host}
