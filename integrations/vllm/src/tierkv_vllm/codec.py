# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Host residual row codec, CODEC_SPEC v1 (interface I3), reference implementation.

A row holds the residual r = x - x_b of one cached row (one token, layer, KV head) in rk bits (keys)
or rv bits (values): y = r @ H (Sylvester Hadamard, natural order, / sqrt(d)), mn = f16(min y),
mx = f16(max y), step = f16((mx - mn) / (2^b - 1)), codes = clamp(round_half_even((y - mn) /
step), 0, 2^b - 1); step == 0 rows get all-zero codes and decode as mn (the reference keeps them
exact: count them). Decode: r_hat = (codes * step + mn) @ H.
Packed canonical row (little endian, no padding): the d codes as one LSB-first bit stream (code i
at stream bits b*i .. b*i + b - 1, stream bit k = bit k % 8 of byte k // 8), then step (fp16),
then mn (fp16). Sizes at d = 128: h54 key 84 B / value 68 B, h33 52 / 52, h88 132 / 132.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

TIERS: dict[str, tuple[int, int]] = {"h54": (5, 4), "h33": (3, 3), "h88": (8, 8)}
DEFAULT_TIER = "h54"
FORMAT_ID = 1
_H: dict[tuple[int, str], torch.Tensor] = {}


def hadamard(d: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Sylvester Hadamard of order d in natural order, / sqrt(d), float32 (built in float64)."""
    key = (d, str(device))
    if key not in _H:
        if d < 1 or d & (d - 1):
            raise ValueError(f"d = {d} is not a power of two")
        i = torch.arange(d)
        bits = torch.bitwise_and(i[:, None], i[None, :])
        parity = torch.zeros_like(bits)
        while bool((bits > 0).any()):
            parity ^= bits & 1
            bits = bits >> 1
        h = (1.0 - 2.0 * parity.double()) / math.sqrt(d)
        _H[key] = h.to(device=device, dtype=torch.float32)
    return _H[key]


def f16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


@dataclass(frozen=True)
class Encoded:
    codes: torch.Tensor  # [..., d] uint8
    step: torch.Tensor  # [...] float32 (fp16-representable)
    mn: torch.Tensor  # [...] float32 (fp16-representable)
    n_step_zero: int


def encode(r: torch.Tensor, bits: int) -> Encoded:
    """r [..., d] residual rows -> codes / step / mn (all computed in float32)."""
    if not 1 <= bits <= 8:
        raise ValueError("bits must be in 1..8")
    d = r.shape[-1]
    y = r.float() @ hadamard(d, r.device)
    lv = 2**bits - 1
    mn = f16(y.amin(-1))
    mx = f16(y.amax(-1))
    step = f16((mx - mn) / lv)
    safe = torch.where(step > 0, step, torch.ones_like(step))
    codes = torch.clamp(torch.round((y - mn[..., None]) / safe[..., None]), 0, lv)
    codes = torch.where((step > 0)[..., None], codes, torch.zeros_like(codes))
    return Encoded(codes.to(torch.uint8), step, mn, int((step == 0).sum()))


def decode(codes: torch.Tensor, step: torch.Tensor, mn: torch.Tensor) -> torch.Tensor:
    d = codes.shape[-1]
    y = codes.float() * step[..., None] + mn[..., None]
    return y @ hadamard(d, codes.device)


def row_bytes(d: int, bits: int) -> int:
    if (d * bits) % 8:
        raise ValueError(f"d * bits = {d * bits} is not a whole number of bytes")
    return d * bits // 8 + 4


def pack(enc: Encoded, bits: int) -> torch.Tensor:
    """Encoded rows [N, d] -> canonical packed rows uint8 [N, row_bytes(d, bits)]."""
    codes = enc.codes.reshape(-1, enc.codes.shape[-1]).to(torch.int64)
    n, d = codes.shape
    shifts = torch.arange(bits, device=codes.device)
    stream = ((codes[:, :, None] >> shifts) & 1).reshape(n, d * bits)  # LSB of each code first
    byte_w = 1 << torch.arange(8, device=codes.device)
    body = (stream.reshape(n, d * bits // 8, 8) * byte_w).sum(-1).to(torch.uint8)
    step16 = enc.step.reshape(-1).to(torch.float16).view(torch.uint8).reshape(n, 2)
    mn16 = enc.mn.reshape(-1).to(torch.float16).view(torch.uint8).reshape(n, 2)
    return torch.cat([body, step16, mn16], dim=1)


def unpack(rows: torch.Tensor, d: int, bits: int) -> Encoded:
    """Canonical packed rows uint8 [N, row_bytes] -> Encoded (codes [N, d])."""
    nb = d * bits // 8
    if rows.shape[-1] < nb + 4:
        raise ValueError("row shorter than the canonical row")
    body = rows[:, :nb].to(torch.int64)
    n = body.shape[0]
    stream = ((body[:, :, None] >> torch.arange(8, device=rows.device)) & 1).reshape(n, nb * 8)
    w = 1 << torch.arange(bits, device=rows.device)
    codes = (stream.reshape(n, d, bits) * w).sum(-1).to(torch.uint8)
    step = rows[:, nb : nb + 2].contiguous().view(torch.float16).reshape(n).float()
    mn = rows[:, nb + 2 : nb + 4].contiguous().view(torch.float16).reshape(n).float()
    return Encoded(codes, step, mn, int((step == 0).sum()))


@dataclass(frozen=True)
class FormatDescriptor:
    """Per tensor or page group (CODEC_SPEC v1 section 5). key_split is reserved for v2 (I5)."""

    d: int = 128
    rk: int = TIERS[DEFAULT_TIER][0]
    rv: int = TIERS[DEFAULT_TIER][1]
    stride_k: int | None = None  # stored bytes per key row (>= canonical); None = canonical
    stride_v: int | None = None
    format_id: int = FORMAT_ID
    hadamard: str = "sylvester_natural"
    scale_dtype: str = "fp16"
    key_split: dict | None = field(default=None)

    def __post_init__(self) -> None:
        if self.format_id != FORMAT_ID:
            raise ValueError(f"format_id {self.format_id} is not supported (v1 = {FORMAT_ID})")
        if self.key_split is not None:
            raise ValueError("key_split is reserved for CODEC_SPEC v2; a v1 engine rejects it")
        if self.hadamard != "sylvester_natural" or self.scale_dtype != "fp16":
            raise ValueError("v1 needs hadamard = sylvester_natural and scale_dtype = fp16")
        for s, b in ((self.stride_k, self.rk), (self.stride_v, self.rv)):
            if s is not None and s < row_bytes(self.d, b):
                raise ValueError("a stored stride is smaller than the canonical row")

    @classmethod
    def tier(cls, name: str = DEFAULT_TIER, d: int = 128, pad_to: int = 1) -> FormatDescriptor:
        rk, rv = TIERS[name]

        def padded(b: int) -> int:
            return -(-row_bytes(d, b) // pad_to) * pad_to

        return cls(d=d, rk=rk, rv=rv, stride_k=padded(rk), stride_v=padded(rv))

    @property
    def row_bytes_k(self) -> int:
        return row_bytes(self.d, self.rk)

    @property
    def row_bytes_v(self) -> int:
        return row_bytes(self.d, self.rv)

    def bytes_note(self) -> str:
        """What every byte / throughput figure has to state (canonical and stored bytes)."""
        sk = self.stride_k or self.row_bytes_k
        sv = self.stride_v or self.row_bytes_v
        note = f"codec v{self.format_id} K{self.rk}V{self.rv}: canonical {self.row_bytes_k} / "
        note += f"{self.row_bytes_v} B per key / value row"
        if (sk, sv) != (self.row_bytes_k, self.row_bytes_v):
            note += f", stored {sk} / {sv} B"
        return note
