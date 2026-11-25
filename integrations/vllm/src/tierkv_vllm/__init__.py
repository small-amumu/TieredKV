# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Tiered KV cache (2-bit KIVI-128 GPU base + host residual store) for vLLM.

M0: `TIERKV_M0`, an in-engine oracle backend that routes the scored rows through the research
code's attention (parity check of the engine wiring). Kernels come in M1.
"""

from __future__ import annotations

__all__ = ["register"]


def register() -> None:
    """vLLM general-plugin entry point. Safe to call more than once per process."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    from tierkv_vllm.m0_backend import BACKEND_NAME

    if BACKEND_NAME in AttentionBackendEnum._member_map_:
        return
    register_backend(BACKEND_NAME, "tierkv_vllm.m0_backend.TierKVM0Backend")
