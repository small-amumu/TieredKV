# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""CPU tests of the CODEC_SPEC v1 implementation (the reference bit-identity test against the
research code runs separately: diag/codec_selftest.py, on a machine with the research tree)."""

from __future__ import annotations

import pytest
import torch

from tierkv_vllm import codec


def test_hadamard_orthonormal_and_natural_order() -> None:
    h = codec.hadamard(128)
    assert torch.allclose(h @ h.T, torch.eye(128), atol=1e-6)
    assert torch.equal(h, h.T)
    # natural (Sylvester) order: row 1 alternates sign, row 2 pairs
    s = h[:3, :4] * (128**0.5)
    assert s.round().tolist() == [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1]]


@pytest.mark.parametrize(
    ("tier", "kb", "vb"), [("h54", 84, 68), ("h33", 52, 52), ("h88", 132, 132)]
)
def test_row_sizes(tier: str, kb: int, vb: int) -> None:
    f = codec.FormatDescriptor.tier(tier)
    assert (f.row_bytes_k, f.row_bytes_v) == (kb, vb)


def test_padding_and_descriptor_rules() -> None:
    f = codec.FormatDescriptor.tier("h54", pad_to=16)
    assert (f.stride_k, f.stride_v) == (96, 80)
    assert "stored 96 / 80" in f.bytes_note() and "canonical 84 / 68" in f.bytes_note()
    with pytest.raises(ValueError, match="key_split"):
        codec.FormatDescriptor(key_split={"perm": [0]})
    with pytest.raises(ValueError, match="format_id"):
        codec.FormatDescriptor(format_id=2)


@pytest.mark.parametrize("bits", range(2, 9))
def test_pack_unpack_round_trip(bits: int) -> None:
    g = torch.Generator().manual_seed(bits)
    r = torch.randn(1000, 128, generator=g) * 0.3
    enc = codec.encode(r, bits)
    rows = codec.pack(enc, bits)
    assert rows.shape == (1000, codec.row_bytes(128, bits)) and rows.dtype == torch.uint8
    back = codec.unpack(rows, 128, bits)
    assert torch.equal(back.codes, enc.codes)
    assert torch.equal(back.step, enc.step) and torch.equal(back.mn, enc.mn)


def test_lsb_first_bit_stream() -> None:
    # code 0 = 0b10101 (5 bits), code 1 = 0b00011: stream bits 1,0,1,0,1, 1,1,0,0,0 -> byte0 = 0x75
    codes = torch.zeros(1, 128, dtype=torch.uint8)
    codes[0, 0], codes[0, 1] = 0b10101, 0b00011
    enc = codec.Encoded(codes, torch.tensor([1.0]), torch.tensor([0.0]), 0)
    row = codec.pack(enc, 5)
    assert int(row[0, 0]) == 0b01110101 and int(row[0, 1]) == 0
    assert row[0, 80:82].view(torch.float16).item() == 1.0


def test_step_zero_rows_counted_and_decoded_as_mn() -> None:
    r = torch.zeros(3, 128)
    r[1] = torch.randn(128)
    enc = codec.encode(r, 5)
    assert enc.n_step_zero == 2
    out = codec.decode(enc.codes, enc.step, enc.mn)
    assert torch.allclose(out[0], torch.zeros(128)) and torch.allclose(out[2], torch.zeros(128))


def test_encode_matches_spec_formula() -> None:
    g = torch.Generator().manual_seed(1)
    r = torch.randn(64, 128, generator=g)
    bits = 4
    enc = codec.encode(r, bits)
    y = r @ codec.hadamard(128)
    mn = y.amin(-1).half().float()
    mx = y.amax(-1).half().float()
    step = ((mx - mn) / 15).half().float()
    codes = torch.clamp(torch.round((y - mn[:, None]) / step[:, None]), 0, 15)
    assert torch.equal(enc.codes, codes.to(torch.uint8))
    assert torch.equal(enc.step, step) and torch.equal(enc.mn, mn)
    r_hat = codec.decode(enc.codes, enc.step, enc.mn)
    assert (r_hat - r).abs().max() <= step.max() * 128**0.5  # quantisation error bound
