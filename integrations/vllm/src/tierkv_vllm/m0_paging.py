# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""Paged-cache read-back and row selection of the M0 oracle (pure torch, no vLLM import).

vLLM hands the attention backend a KV cache whose logical shape is (num_blocks, num_kv_heads,
block_size, 2 * head_size): K in the first head_size entries of the last dim, V in the second;
token t of a request with block table B sits in block B[t // block_size] at offset t % block_size.
"""

from __future__ import annotations

import torch


def split_kv(kv_cache: torch.Tensor, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(num_blocks, nkv, block_size, 2 hs) -> K, V views (num_blocks, block_size, nkv, hs)."""
    k, v = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    return k, v


def gather_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    key_drop: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K, V of tokens 0..seq_len-1-key_drop of one request, (n, nkv, hs), in position order."""
    block_size, nkv, hs = key_cache.shape[1], key_cache.shape[2], key_cache.shape[3]
    n_blk = -(-seq_len // block_size)
    blocks = block_table_row[:n_blk].long()
    n_k = seq_len - key_drop
    k = key_cache[blocks].reshape(-1, nkv, hs)[:n_k]
    v = value_cache[blocks].reshape(-1, nkv, hs)[:n_k]
    return k, v


def oracle_rows(
    query_start_loc: list[int], seq_lens: list[int], first: int
) -> list[tuple[int, int, int, int]]:
    """Per request with rows at positions >= first in this step: (request, row_lo, row_hi, seq_len)
    with flat query rows [row_lo, row_hi); the rows are the last ones of the request, ending at
    position seq_len - 1."""
    plan = []
    for i, seq_len in enumerate(seq_lens):
        q0, q1 = query_start_loc[i], query_start_loc[i + 1]
        n_q = q1 - q0
        if n_q <= 0:
            continue
        skip = max(0, first - (seq_len - n_q))
        if skip >= n_q:
            continue
        plan.append((i, q0 + skip, q1, seq_len))
    return plan
