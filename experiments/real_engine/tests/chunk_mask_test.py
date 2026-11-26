# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""rp_chunk.chunk_attn (the engine's long-prefill attention, chunk 4096) vs one causal SDPA call, on random tensors with
the engine's shapes (H 32, KVH 8, D 128, bf16); asserts equality and prints the numbers (saved by the caller)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rp_chunk  # noqa: E402

torch.manual_seed(0)
H, KVH, D, L, C = 32, 8, 128, 12000, 4096
q = torch.randn(1, H, L, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, KVH, L, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, KVH, L, D, device="cuda", dtype=torch.bfloat16)
sc = D ** -0.5
full = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sc, enable_gqa=True).transpose(1, 2)
out = torch.cat([rp_chunk.chunk_attn(q[:, :, s0:min(L, s0 + C)], k[:, :, :min(L, s0 + C)], v[:, :, :min(L, s0 + C)], sc,
                                     first=s0 == 0) for s0 in range(0, L, C)], 1)
d = (out.float() - full.float()).abs()
bad = torch.nn.functional.scaled_dot_product_attention(q[:, :, C:2 * C], k[:, :, :2 * C].repeat_interleave(4, 1),
                                                       v[:, :, :2 * C].repeat_interleave(4, 1), is_causal=True, scale=sc)
ctrl = float((bad.transpose(1, 2).float() - full[:, C:2 * C].float()).abs().max())
print(f"CHUNK_TEST max_abs_diff {float(d.max()):.3e} mean {float(d.mean()):.3e} upper_left_control {ctrl:.3e}")
assert float(d.max()) <= 1e-2 and ctrl > 0.1, "chunked attention differs from the causal call (or the control is vacuous)"
print("CHUNK_TEST PASS")
