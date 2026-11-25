# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""VM0 step 1b (CPU): M0's read-back of vLLM's paged cache returns exactly what was written.

Tokens are written through the logical (num_blocks, nkv, block_size, 2 hs) index order that vLLM
documents (block = table[t // bs], offset = t % bs, K then V in the last dim) into both a
contiguous HND tensor and an NHD-physical view; gather_kv must return them byte for byte in
position order, for several requests with shuffled block tables, partial last blocks and the
planted key drop. oracle_rows must select exactly the rows at positions >= first.
"""

from __future__ import annotations

import torch

from tierkv_vllm.m0_paging import gather_kv, oracle_rows, split_kv

NKV, HS, BS = 3, 8, 16


def make_cache(n_blocks: int, nhd: bool) -> torch.Tensor:
    if nhd:  # physical (B, N, H, C), logical view (B, H, N, C), as vLLM's NHD layout
        return torch.zeros(n_blocks, BS, NKV, 2 * HS, dtype=torch.bfloat16).transpose(1, 2)
    return torch.zeros(n_blocks, NKV, BS, 2 * HS, dtype=torch.bfloat16)


def write(cache: torch.Tensor, table: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    for t in range(k.shape[0]):
        b, o = int(table[t // BS]), t % BS
        cache[b, :, o, :HS] = k[t]
        cache[b, :, o, HS:] = v[t]


def test_gather_is_byte_identical() -> None:
    g = torch.Generator().manual_seed(0)
    lens = [1, 15, 16, 17, 300, 1100]
    n_blocks = sum(-(-s // BS) for s in lens) + 5
    for nhd in (False, True):
        cache = make_cache(n_blocks, nhd)
        perm = torch.randperm(n_blocks, generator=g)
        tables, kv, start = [], [], 0
        for s in lens:
            nb = -(-s // BS)
            tables.append(perm[start : start + nb])
            start += nb
            k = torch.randn(s, NKV, HS, generator=g).to(torch.bfloat16)
            v = torch.randn(s, NKV, HS, generator=g).to(torch.bfloat16)
            write(cache, tables[-1], k, v)
            kv.append((k, v))
        kc, vc = split_kv(cache, HS)
        for (k, v), tab, s in zip(kv, tables, lens, strict=True):
            gk, gv = gather_kv(kc, vc, tab, s)
            assert torch.equal(gk, k) and torch.equal(gv, v), (nhd, s)
            if s > 1:
                dk, _ = gather_kv(kc, vc, tab, s, key_drop=1)
                assert torch.equal(dk, k[:-1])


def test_oracle_rows() -> None:
    first = 2048
    # decode below / at / above first; a prefill chunk crossing first; a chunk fully above; empty
    qsl = [0, 1, 2, 3, 515, 1027, 1027]
    seq = [2048, 2049, 5000, 2304, 4096, 3000]
    plan = oracle_rows(qsl, seq, first)
    assert plan == [
        (1, 1, 2, 2049),
        (2, 2, 3, 5000),
        (3, 3 + 256, 515, 2304),  # rows at positions 1792..2303: the last 256 are >= 2048
        (4, 515, 1027, 4096),
    ]
    for _, lo, hi, s in plan:  # selected rows end at position seq_len - 1 and start >= first
        assert s - (hi - lo) >= first
