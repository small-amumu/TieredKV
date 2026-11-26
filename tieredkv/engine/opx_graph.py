# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""OPX engine: the decode step in a CUDA graph. Requires kalign_opx with WDEV
(the window row count W read from a device int32; kernel tests opx_kernel_test.py (e)).

What varies per decode step in rp_engine.Engine.step and how it is made static:
  token id, position          -> device buffers TOK [1, 1], POS [1, 1] (int64), filled before each replay
  window write index Z + nw   -> device buffer WIDX [1]; kx / vx rows written with index_copy_ (same fp16 cast as copy_)
  window row count W          -> device int32 WDEV read by abasew_kernel (StepA.Wdev); the whole window buffer is
                                 copied to StepA.kw / vw (rows >= W are never read: masked loads, kernel test (e))
  StepA objects               -> change only at a window flush: one capture per flush epoch
Steps that run eagerly (rp_engine's own code, python W): the step whose token completes a 128-row window (it flushes,
building new StepA objects), and the first decode step while the layer-0 guard is pending. Every other step replays the
captured graph; afterwards the python counters nw / Lk of every layer are advanced exactly as LayerStore.attend does.
The graph contains the same operations in the same order as Engine.step + LayerStore.attend, so the logits must equal
the eager engine bit for bit (checked end to end with the logits digest)."""
from __future__ import annotations

import time
import weakref

import torch

import rp_engine as E

STATS: dict = {"eager": 0, "replay": 0, "captures": 0, "capture_sec": 0.0}


class GraphStep:
    def __init__(self, eng) -> None:
        assert E.KA.__name__ == "kalign_opx", "the graph path needs kalign_opx (WDEV)"
        self.eng = eng
        self.orig_step = type(eng).step
        self.TOK = torch.zeros(1, 1, dtype=torch.int64, device="cuda")
        self.POS = torch.zeros(1, 1, dtype=torch.int64, device="cuda")
        self.WIDX = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.WDEV = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.graph = None
        self.epoch = None
        self.out = None

    # ------------------------------------------------------------------ the captured body (no python side effects)
    def _body(self):
        eng = self.eng
        m = eng.model.model
        h = m.embed_tokens(self.TOK)
        cos, sin = m.rotary_emb(h, self.POS)
        for li, layer in enumerate(m.layers):
            sa = layer.self_attn
            x = layer.input_layernorm(h)
            q = sa.q_proj(x).view(1, 1, eng.H, E.D).transpose(1, 2)
            k = sa.k_proj(x).view(1, 1, eng.KVH, E.D).transpose(1, 2)
            v = sa.v_proj(x).view(1, 1, eng.KVH, E.D).transpose(1, 2)
            q, k = E.apply_rotary_pos_emb(q, k, cos, sin)
            L = eng.layers[li]
            q1, k1, v1 = q[0, :, 0], k[0, :, 0], v[0, :, 0]
            # LayerStore.attend without the python counters, the flush and the guard
            L.inexact += E._fp16_inexact(k1) + E._fp16_inexact(v1)
            L.kx.index_copy_(1, self.WIDX, k1[:, None].to(L.kx.dtype))
            L.vx.index_copy_(1, self.WIDX, v1[:, None].to(L.vx.dtype))
            st = L.st
            st.kw.copy_(L.kx)
            st.vw.copy_(L.vx)
            st.set_q(q1)
            st.step()
            out = st.result()
            o = out.to(h.dtype).view(1, 1, eng.H * E.D)
            h = h + sa.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return eng.model.lm_head(m.norm(h[:, -1]))[0].float()

    def _epoch(self) -> tuple:
        # weak references, compared by identity: id() values of freed StepA objects can be reused by the StepA objects
        # of the next item / width (same allocation order), which would replay a graph over freed buffers (review B1,
        # 2026-11-23). A strong reference would keep the old StepA buffers alive into the next prefill.
        return tuple(weakref.ref(L.st) for L in self.eng.layers)

    def _same_epoch(self) -> bool:
        return self.epoch is not None and len(self.epoch) == len(self.eng.layers) and all(
            r() is not None and r() is L.st for r, L in zip(self.epoch, self.eng.layers))

    def reset(self) -> None:
        """drop the captured graph (called when the engine's layers are released after an item / width)."""
        self.graph, self.epoch, self.out = None, None, None

    def _capture(self) -> None:
        t0 = time.perf_counter()
        L0 = self.eng.layers[0]
        for L in self.eng.layers:
            L.st.Wdev = self.WDEV
        self.graph = None
        # compile / load the WDEV kernel variant outside the capture (module loading is not allowed while capturing):
        # one eager StepA.step per layer on its current buffers (the window of the last step, W = Z + nw). It only
        # rewrites StepA's scratch outputs (no store bytes, no counters), which the next step overwrites.
        self.WDEV.fill_(E.Z + L0.nw)
        for L in self.eng.layers:
            L.st.step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.out = self._body()
        torch.cuda.synchronize()
        self.graph, self.epoch = g, self._epoch()
        STATS["captures"] += 1
        STATS["capture_sec"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ Engine.step replacement
    @torch.no_grad()
    def step(self, tok_id):
        eng = self.eng
        L0 = eng.layers[0]
        eager = E.MUT or L0.guard is None or (L0.Lk + 1) % E.RW == 0
        if eager:
            for L in eng.layers:
                if L.st is not None:
                    L.st.Wdev = None                 # eager attend: python W through set_exact
            STATS["eager"] += 1
            return self.orig_step(eng, tok_id)
        assert all(L.nw == L0.nw and L.Lk == L0.Lk for L in eng.layers), "layers out of step"
        assert L0.nw + 1 <= E.RW and E.Z + L0.nw < E.KA.WBA
        if self.graph is None or not self._same_epoch():
            self._capture()
        self.TOK.fill_(int(tok_id))
        self.POS.fill_(L0.Lk)
        self.WIDX.fill_(E.Z + L0.nw)
        self.WDEV.fill_(E.Z + L0.nw + 1)
        self.graph.replay()
        for L in eng.layers:                         # the counters LayerStore.attend advances
            L.nw += 1
            L.Lk += 1
        STATS["replay"] += 1
        return self.out


def install(eng) -> GraphStep:
    g = GraphStep(eng)
    eng.step = g.step
    return g
