# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""chunked exact causal attention for long prefills (rp_engine E1; no heavy imports, so tests can import it alone)."""
import torch


def chunk_attn(qc, k, v, scale, first):
    """exact causal attention of a query chunk qc [1, H, c, D] whose last row is the last key of k / v [1, KVH, e, D]: the
    first chunk causal (flash or efficient); later chunks with the bottom-right causal mask, efficient backend only (never
    the math fallback, which would materialise c x e scores), GQA expanded. Returns [1, c, H, D]."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right
    if first:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            o = torch.nn.functional.scaled_dot_product_attention(qc, k, v, is_causal=True, scale=scale, enable_gqa=True)
    else:
        g = qc.shape[1] // k.shape[1]
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            o = torch.nn.functional.scaled_dot_product_attention(qc, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1),
                                                                 attn_mask=causal_lower_right(qc.shape[2], k.shape[2]), scale=scale)
    return o.transpose(1, 2)
