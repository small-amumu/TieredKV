# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""M0 of the tiered-KV port: an in-engine oracle backend (bf16 KV cache, no new kernels).

Every layer first runs the stock Triton attention for all rows. When MODE["attend"] is set, the rows
of each request whose position is >= MODE["first"] are recomputed by MODE["attend"] (the research
code's attention for method MODE["name"]) on that request's full K / V read back from vLLM's paged
cache, and overwrite the stock output. Rows below MODE["first"] stay exact (the research protocol's
dense prefill). Purpose: check that the engine wiring (paging, positions, GQA layout, layer index,
scale, chunked prefill) reproduces the research numbers before any kernel is written.

Single process only (MODE is a module global set by the driver); eager mode only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from tierkv_vllm.m0_paging import gather_kv, oracle_rows, split_kv

BACKEND_NAME = "TIERKV_M0"
# attend(q [H, n, d] f32, k [KVH, L, d] f32, v [KVH, L, d] f32, layer, name, scale, kw) -> [H, n, d]
AttendFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, str, float, dict], torch.Tensor]
# Planted faults for the parity check: layer_shift (attend gets layer (li + s) % n_layers) and
# key_drop (the newest key_drop keys are cut from the gathered K / V: an off-by-one in seq_len).
MODE: dict[str, Any] = {
    "attend": None,
    "name": None,
    "first": 2048,
    "rows": 0,
    "calls": 0,
    "layer_shift": 0,
    "n_layers": 0,
    "key_drop": 0,
}


class TierKVM0Impl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        if self.attn_type != AttentionType.DECODER:
            raise ValueError("TIERKV_M0: decoder self-attention only")
        if self.kv_cache_dtype not in ("auto", "bfloat16", "float16"):
            raise ValueError("TIERKV_M0 reads exact K / V back from the cache: 16-bit cache only")
        if self.sinks is not None or self.alibi_slopes is not None:
            raise ValueError("TIERKV_M0: sinks / ALiBi are not covered by the research code")
        self._layer: int | None = None

    def forward(  # type: ignore[override]
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        attend: AttendFn | None = MODE["attend"]
        if attend is None or attn_metadata is None:
            return out
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TIERKV_M0 runs eagerly only (enforce_eager=True)")
        if self._layer is None:
            from vllm.model_executor.models.utils import extract_layer_index

            self._layer = extract_layer_index(layer.layer_name)  # type: ignore[attr-defined]
        first = int(MODE["first"])
        li = self._layer
        if MODE["layer_shift"]:
            li = (li + int(MODE["layer_shift"])) % int(MODE["n_layers"])
        key_cache, value_cache = split_kv(kv_cache, self.head_size)
        kw = {
            "softcap": self.logits_soft_cap or None,
            "sliding_window": self.sliding_window[0] + 1 if self.sliding_window[0] >= 0 else None,
        }
        plan = oracle_rows(
            attn_metadata.query_start_loc.tolist(), attn_metadata.seq_lens.tolist(), first
        )
        for i, lo, hi, seq_len in plan:
            k, v = gather_kv(
                key_cache,
                value_cache,
                attn_metadata.block_table[i],
                seq_len,
                int(MODE["key_drop"]),
            )
            rows = slice(lo, hi)
            # [H, n, d]; the rows end at position seq_len - 1
            q = query[rows].transpose(0, 1).float()
            o = attend(
                q,
                k.transpose(0, 1).float(),
                v.transpose(0, 1).float(),
                li,
                MODE["name"],
                self.scale,
                kw,
            )
            output[rows] = o.transpose(0, 1).to(output.dtype).reshape(output[rows].shape)
            MODE["rows"] += hi - lo
        MODE["calls"] += 1
        return output


class TierKVM0Backend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return BACKEND_NAME

    @staticmethod
    def get_impl_cls() -> type[TierKVM0Impl]:
        return TierKVM0Impl
