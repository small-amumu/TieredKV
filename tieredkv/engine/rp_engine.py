# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""rp_engine E1 (TieredKV): the real decode engine of the layered KV cache on the deployed kernels (2026-11-09).
See DESIGN_E1.md for the step-by-step correspondence with the accuracy reference (ours_q 92ef53ff / cap_attn 29025867).
- Kernels: kalign f96b21f5 StepA (fused aprep / abasew / sel / afix2A / afix2B / afixB; water-filled stack base; the
  deployed encoder aenc_kernel, f16x2), unchanged.
- Storage per layer and KV head: GPU code records of rows [Z, W0) (70 B per token), host residual records of the same rows
  in pinned memory (zero-copy reads of the fetched rows only), FP16 sinks + residual window [W0, Lk) on the GPU.
- Prefill: exact (HF sdpa_attention_forward, causal, as tb_bench v3.7), layer by layer over the whole prompt; each layer's K / V are encoded right after
  the layer and freed (only sinks + window kept exactly).
- Decode: e2e_v3's token step (HF modules, bf16), eager launches with the true window size W; every 128 steps the window is
  encoded into the base (store grown by 128 rows; old bytes unchanged).
- Llama-family only (q / k / v / o projections, RoPE via model.model.rotary_emb, no q / k norm), B = 1, head_dim 128.
Env: KVT_ROOT (reference tree, for the static calibration), KALIGN_DIR (kernel tree), KALIGN_CAL (calibration text)."""
from __future__ import annotations

import ctypes
import os
import sys

import torch

ROOT = os.environ["KVT_ROOT"]
sys.path[:0] = [f"{ROOT}/hc_mac/rootcause", f"{ROOT}/hc_mac/code"]
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward as _sdpa_fwd  # noqa: E402  (tb_bench v3.7)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cq_lab3 as CQ  # noqa: E402
import cap_attn as CA  # noqa: E402
import sel11_q  # noqa: E402,F401  (import order of the reference harness)
import sel14_q as Q14  # noqa: E402

sys.path.insert(0, os.environ["KALIGN_DIR"])
import kalign as KA  # noqa: E402

CQ.CAL = (os.environ["KALIGN_CAL"],) + tuple(CQ.CAL[1:])
S12, S10 = Q14.S12, Q14.S10
P9, P10, P11, Q7, Q5 = S10.P9, S10.P10, S10.P11, S10.Q7, S10.Q5
Z, RW, D = KA.Z_SINK, CA.R_WIN, 128
M_CAND, K_FETCH, LAM = 128, 64, -50              # stkm8v8128k64_x2s128_p50
MLP_CHUNK = 4096


class EngineFlag(RuntimeError):
    """an item the engine does not decode: rows the byte format cannot express or zero-step residual rows (the reference,
    ours_q, raises on these too), a prompt shorter than 256 tokens (the kernel needs W0 - Z >= M = 128; the reference
    computes these with M = min(M, W0 - Z): a stated deviation), or a failed layer-0 guard. Scored 0 under
    intention-to-treat, as tb_bench scores errors."""


MUT: dict = {}            # mutation tests only (rp_run --mutate): rope_off, shift. Empty in every real run (asserted).


def _cap_capture(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    """attention of the HF forward. Calibration (MODE["cap"]): exactly tb_bench v3.7's capture branch (tb_attn): q / k / v to
    fp32, the post-RoPE K / V / Q of every layer kept on the CPU, the output from the harness's exact attention
    (pkv5_q.attend with name None: fp32 einsum + softmax, causal). Otherwise (only the check mode's HF reference forward):
    SDPA, as tb_bench's prompt / fp-decode path (v3.4)."""
    li = int(module.layer_idx)
    if MODE["cap"]:
        q, k, v = query[0].float(), key[0].float(), value[0].float()
        CAP[li] = (k.cpu(), v.cpu(), q.cpu())
        scale = scaling if scaling is not None else 1 / (q.shape[-1] ** 0.5)
        out = S10.Q5.attend(q, k, v, li, None, scale, kw)
        return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None
    return _sdpa_fwd(module, query, key, value, attention_mask, scaling=scaling, dropout=0.0, is_causal=query.shape[2] > 1)


CAP: dict = {}
MODE = {"cap": False}
CAL_MD5 = "4604ba1d"                                # diag_long_text.txt, the reference calibration text (CQ.CAL[0])
AttentionInterface.register("rp_cap", _cap_capture)


class Arena:
    """one pinned host buffer of fixed capacity holding a [BK, L, w] tensor laid out with row stride L (the layout the
    kernels index: row = bk * L + tok); grow() moves every head's rows to the new stride in place (last head first, so no
    unread byte is overwritten)."""

    def __init__(self, BK, cap, w, dtype):
        self.BK, self.cap, self.w, self.dtype = BK, cap, w, dtype
        self.buf = torch.zeros(BK * cap * w, dtype=dtype, pin_memory=True)    # zeros: an ordering bug fails deterministically
        self.es = self.buf.element_size()
        self.L = 0

    def view(self, L):
        assert L <= self.cap
        return self.buf[: self.BK * L * self.w].view(self.BK, L, self.w)

    def grow(self, L_new):
        L_old = self.L
        assert L_new >= L_old and L_new <= self.cap
        if L_old:
            torch.cuda.synchronize()                              # kernels read these bytes zero-copy: no move while in flight
            base, row = self.buf.data_ptr(), self.w * self.es
            for bk in range(self.BK - 1, 0, -1):                 # bk = 0 stays in place
                ctypes.memmove(base + bk * L_new * row, base + bk * L_old * row, L_old * row)
        self.L = L_new


def _fp16_inexact(x):
    """count of nonzero values outside the range where bf16 -> fp16 is exact (2^-17 <= |x| <= 65504): the reference keeps
    the exact rows in fp16 too (cap_attn wdtype), so this is reported, not corrected."""
    a = x.abs()
    return ((a > 0) & ((a < 2.0 ** -17) | (a > 65504))).sum()


class LayerStore:
    """one layer's deployed storage and its StepA."""

    def __init__(self, li, KVH, GH, kb, vb, cap, prec, calib):
        self.li, self.KVH, self.GH, self.kb, self.vb, self.prec, self.calib = li, KVH, GH, kb, vb, prec, calib
        self.ar = (Arena(KVH, cap, 16 * kb, torch.int8), Arena(KVH, cap, 2, torch.float16),
                   Arena(KVH, cap, 16 * vb, torch.int8), Arena(KVH, cap, 2, torch.float16))
        self.kx = torch.zeros(KVH, KA.WBA, D, dtype=torch.float16, device="cuda")   # sinks [0, Z), window [Z, Z + nw)
        self.vx = torch.zeros_like(self.kx)
        self.st, self.W0, self.Lk, self.nw = None, 0, 0, 0
        self.inexact = torch.zeros((), dtype=torch.int64, device="cuda")   # exact rows whose fp16 copy is not value-exact
        self.guard = None                                                  # layer-0 sanity at the first decode step
        self.gam = torch.tensor([float(x) for x in S10.Q6.GAMMAS], dtype=torch.float32, device="cuda")

    def _new_step(self, W0):
        """StepA over coded rows [Z, W0) whose residual records are the arena views (stride W0 - Z)."""
        L = W0 - Z
        if L < M_CAND:
            raise EngineFlag(f"unsupported by the kernel: {L} coded rows < M = {M_CAND} (W0 - Z >= M needs W0 >= 256)")
        for a in self.ar:
            a.grow(L)
        st = KA.StepA(L, GH=self.GH, M=M_CAND, K=K_FETCH, lam=LAM, kbits=self.kb, vbits=self.vb, host=True, words=True,
                      fused=True, rh=tuple(a.view(L) for a in self.ar))
        assert all(t.data_ptr() == a.buf.data_ptr() for t, a in zip((st.krp, st.krm, st.vrp, st.vrm), self.ar)), \
            "StepA host records do not alias the arenas"
        st.set_calib(*self.calib)
        st.rope_tables(W0)
        return st

    def _check_new_rows(self, st, c0, n, f0):
        """the reference raises on unexpressible rows and on zero-step residual rows; count both for the new rows."""
        torch.cuda.synchronize()
        nflag = int(st.flags.item()) - f0
        krm, vrm = self.ar[1].view(st.L), self.ar[3].view(st.L)
        nzero = int((krm[:, c0:c0 + n, 1] == 0).sum()) + int((vrm[:, c0:c0 + n, 1] == 0).sum())
        if nflag or nzero:
            raise EngineFlag(f"layer {self.li}: {nflag} unexpressible rows, {nzero} zero-step residual rows in [{c0}, {c0 + n})")

    def prefill(self, k, v):
        """k, v [KVH, L0, D] (post-RoPE, exact) of the whole prompt: code rows [Z, W0p), keep sinks + window exactly."""
        KVH, L0, _ = k.shape
        W0 = L0 // RW * RW
        st = self._new_step(W0)
        kt = k[:, :W0].transpose(0, 1).reshape(1, W0, KVH * D).contiguous()     # projection layout, token-major
        vt = v[:, :W0].transpose(0, 1).reshape(1, W0, KVH * D).contiguous()
        f0 = int(st.flags.item())
        st.encode_proj(kt, vt, 0, S10.Q6.GAMMAS, prec=self.prec)
        self._check_new_rows(st, 0, W0 - Z, f0)
        nw = L0 - W0
        self.inexact += _fp16_inexact(k[:, :Z]) + _fp16_inexact(v[:, :Z]) + _fp16_inexact(k[:, W0:L0]) + _fp16_inexact(v[:, W0:L0])
        self.kx.zero_(); self.vx.zero_()
        self.kx[:, :Z].copy_(k[:, :Z]); self.vx[:, :Z].copy_(v[:, :Z])
        self.kx[:, Z:Z + nw].copy_(k[:, W0:L0]); self.vx[:, Z:Z + nw].copy_(v[:, W0:L0])
        self.st, self.W0, self.Lk, self.nw = st, W0, L0, nw
        if self.li == 0:                                # rows of layer 0 kept (bf16, CPU) for the first-step guard
            self.ex_k, self.ex_v = k.to("cpu", torch.bfloat16), v.to("cpu", torch.bfloat16)

    def _flush(self):
        """the window holds rows [W0, W0 + 128): encode them into a store grown by 128 rows (old bytes copied unchanged)."""
        old, W0_old = self.st, self.W0
        W0 = W0_old + RW
        st = self._new_step(W0)                       # arenas re-laid out to stride W0 - Z (old host bytes kept)
        L_old = W0_old - Z
        for f in ("kp", "km", "vp", "vm"):
            getattr(st, f)[:, :L_old].copy_(getattr(old, f)[:, :L_old])
        f0 = int(st.flags.item())
        st._launch_enc(self.kx, self.vx, KA.WBA * D, D, Z, W0_old, RW, self.gam, self.prec, 16)
        if MUT.get("shift"):                          # mutation test: the flushed block's GPU records misaligned by one row
            for f in ("kp", "km", "vp", "vm"):
                t = getattr(st, f)
                t[:, L_old:L_old + RW] = torch.roll(t[:, L_old:L_old + RW].clone(), 1, dims=1)
        try:
            self._check_new_rows(st, L_old, RW, f0)
        except EngineFlag:
            self.st = None                            # the arenas already have the new stride: no use of the old store
            raise
        self.kx[:, Z:].zero_(); self.vx[:, Z:].zero_()
        self.st, self.W0, self.nw = st, W0, 0
        del old

    def attend(self, q, k1, v1):
        """q [H, D], k1 / v1 [KVH, D] (post-RoPE) of the token at position Lk: append, flush at a multiple of 128 (the
        reference's W0 = T // 128 * 128 with T including the current token), then the deployed decode step."""
        assert self.nw < RW
        self.inexact += _fp16_inexact(k1) + _fp16_inexact(v1)
        self.kx[:, Z + self.nw].copy_(k1); self.vx[:, Z + self.nw].copy_(v1)
        self.nw += 1
        self.Lk += 1
        if self.Lk % RW == 0:
            assert self.nw == RW and self.Lk == self.W0 + RW
            self._flush()
        st = self.st
        W = Z + self.nw
        st.set_exact(self.kx[:, :W], self.vx[:, :W])
        st.set_q(q)
        st.step()
        out = st.result()                             # [H, D] fp32
        if self.li == 0 and self.guard is None:       # first decode step: relative difference to exact attention
            kk = torch.cat([self.ex_k.cuda(), k1[:, None].to(torch.bfloat16)], 1)[None]
            vv = torch.cat([self.ex_v.cuda(), v1[:, None].to(torch.bfloat16)], 1)[None]
            ex = torch.nn.functional.scaled_dot_product_attention(q[None, :, None].to(torch.bfloat16), kk, vv, scale=D ** -0.5,
                                                                  enable_gqa=True)[0, :, 0].float()
            rel = float((out - ex).abs().max() / ex.abs().max().clamp_min(1e-12))
            self.guard = dict(rel_vs_exact=rel, finite=bool(torch.isfinite(out).all()))
            self.ex_k = self.ex_v = None
            del kk, vv, ex
            if not (self.guard["finite"] and rel < GUARD_MAX):
                raise EngineFlag(f"layer-0 guard: compressed attention differs from exact by {rel:.3g} (finite {self.guard['finite']})")
        return out


GUARD_MAX = 0.5                                     # layer-0 first-step relative difference to exact attention: a gross-corruption
                                                    # net only (provisional; the check mode records healthy and mutated values)
LONG_PREFILL = 40960                                # longer prompts: chunked exact prefill attention (memory)
PREFILL_Q_CHUNK = 4096


from rp_chunk import chunk_attn as _chunk_attn  # noqa: E402  (tests/chunk_mask_test.py)


class Engine:
    def __init__(self, model_dir, kb, vb, cap, prec="f16x2"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="rp_cap").cuda().eval()
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        assert next(self.model.parameters()).dtype == torch.bfloat16, "model not loaded in bf16"
        cfg = self.model.config
        self.H, self.KVH, self.NL = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        assert hd == D, f"kernel needs head_dim 128, got {hd}"
        assert cfg.model_type == "llama", f"E1 supports Llama-family models only, got {cfg.model_type}"
        self.GH = self.H // self.KVH
        KA.configure(self.KVH, self.GH)
        re_ = self.model.model.rotary_emb
        assert getattr(re_, "rope_type", "default") in ("default", "llama3", "linear"), f"rope type {re_.rope_type} not supported"
        # linear (LLaMA-2-7B-32K, factor 8): HF folds the factor into inv_freq; the kernel and the reference read inv_freq
        S10.RC.R["inv_freq"], S10.RC.R["asc"] = re_.inv_freq.detach().float(), float(getattr(re_, "attention_scaling", 1.0))
        assert P9.RC is S10.RC
        assert abs(self.model.model.layers[0].self_attn.scaling - D ** -0.5) < 1e-12
        # static calibration exactly as the reference (rp0_tf / tb_bench): capture on CQ.CAL, then the stack's INFO per layer
        import hashlib
        cal_md5 = hashlib.md5(open(CQ.CAL[0], "rb").read()).hexdigest()[:8]
        assert cal_md5 == CAL_MD5, f"calibration text md5 {cal_md5} != {CAL_MD5}"
        S10.PREFILL["on"] = False                       # tb_bench v3.7 line 378-379, set before the calibration forward
        S10.CHUNK["rows"] = 512
        cal = CQ.text_ids(self.tok, CQ.CAL[0], CQ.CAL[1], CQ.CAL[2], "cuda")
        assert cal.shape[1] == CQ.CAL[2], f"calibration window {cal.shape[1]} != {CQ.CAL[2]}"
        self.cal_info = dict(cal=CQ.CAL[0], cal_md5=cal_md5, start=CQ.CAL[1], n=CQ.CAL[2])
        CAP.clear()
        MODE["cap"] = True
        try:
            with torch.no_grad():
                self.model(input_ids=cal, use_cache=False, logits_to_keep=1)
        finally:
            MODE["cap"] = False
        alist = sorted(CAP)
        for li in alist:                                # tb_bench v3.7 lines 418-420: CPU captures moved to the device
            Q14.calib(li, *(t.cuda() for t in CAP[li]))
        Q14.calib_all(self.model, {li: tuple(t.cuda() for t in CAP[li]) for li in alist}, alist)
        S10.INFO.clear()
        for li in alist:                                # INFO[li]: the fields the cap path reads (order, b) are calibration
            kk, vv, _ = CAP[li]                         # constants (sel10_q.stack_base; step / s are per-token, unused)
            S12.MF["fmt"] = "m8v8"
            S10.ST["on"], S10.ST["li"] = True, li
            try:
                S10.stack_base(kk.cuda(), vv.cuda(), li)
            finally:
                S12.MF["fmt"] = None
                S10.ST["on"] = False
        assert not (getattr(P11, "NOPE", None) or ()), f"NoPE layers {P11.NOPE}: the engine applies RoPE to every layer"
        self.calib = [KA.calib_of(li, S10, P9, P10, P11, Q7) for li in range(self.NL)]
        CAP.clear()
        torch.cuda.empty_cache()
        self.kb, self.vb, self.cap, self.prec = kb, vb, cap, prec
        self.layers = None

    # ------------------------------------------------------------------ prefill (exact, layer by layer)
    @torch.no_grad()
    def prefill(self, ids, max_new):
        """ids [1, L0] -> logits of the last prompt token [V] fp32; builds every layer's store (host capacity for L0 +
        max_new rows)."""
        m = self.model.model
        L0 = ids.shape[1]
        assert L0 + max_new <= self.cap
        if L0 // RW * RW - Z < M_CAND:
            raise EngineFlag(f"unsupported: prompt of {L0} tokens gives {L0 // RW * RW - Z} coded rows < M = {M_CAND}")
        torch.cuda.synchronize()
        self.layers = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        cap_rows = (L0 + max_new) // RW * RW + RW - Z
        self.layers = [LayerStore(li, self.KVH, self.GH, self.kb, self.vb, cap_rows, self.prec, self.calib[li])
                       for li in range(self.NL)]
        h = m.embed_tokens(ids)
        pos = torch.arange(L0, device="cuda")[None]
        cos, sin = m.rotary_emb(h, pos)
        for li, layer in enumerate(m.layers):
            sa = layer.self_attn
            x = layer.input_layernorm(h)
            k = sa.k_proj(x).view(1, L0, self.KVH, D).transpose(1, 2)
            v = sa.v_proj(x).view(1, L0, self.KVH, D).transpose(1, 2)
            if L0 <= LONG_PREFILL:                      # as tb_bench: one sdpa_attention_forward call over the prompt
                q = sa.q_proj(x).view(1, L0, self.H, D).transpose(1, 2)
                del x
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                o = _sdpa_fwd(sa, q, k, v, None, scaling=sa.scaling, dropout=0.0, is_causal=True)[0]   # tuple[0]: [1, L0, H, D]
                del q
                self.layers[li].prefill(k[0], v[0])
                del k, v
                h = h + sa.o_proj(o.reshape(1, L0, self.H * D))
                del o
            else:                                       # memory only (130K on 24 GB): the same exact causal attention per
                _, k = apply_rotary_pos_emb(k, k, cos, sin)          # query chunk (bottom-right mask; bit-identical on
                for s0 in range(0, L0, PREFILL_Q_CHUNK):             # random tensors, rp0 chunk_mask_test)
                    e = min(L0, s0 + PREFILL_Q_CHUNK)
                    qc = sa.q_proj(x[:, s0:e]).view(1, e - s0, self.H, D).transpose(1, 2)
                    qc, _ = apply_rotary_pos_emb(qc, qc, cos[:, s0:e], sin[:, s0:e])
                    oc = _chunk_attn(qc, k[:, :, :e], v[:, :, :e], sa.scaling, first=s0 == 0)
                    h[:, s0:e] = h[:, s0:e] + sa.o_proj(oc.reshape(1, e - s0, self.H * D))
                    del qc, oc
                del x
                self.layers[li].prefill(k[0], v[0])
                del k, v
            for c0 in range(0, L0, MLP_CHUNK):
                hc = h[:, c0:c0 + MLP_CHUNK]
                h[:, c0:c0 + MLP_CHUNK] = hc + layer.mlp(layer.post_attention_layernorm(hc))
        logits = self.model.lm_head(m.norm(h[:, -1]))
        return logits[0].float()

    # ------------------------------------------------------------------ decode (one token)
    @torch.no_grad()
    def step(self, tok_id):
        """the token at position Lk -> logits of the next token [V] fp32 (e2e_v3 token_step order, eager)."""
        m = self.model.model
        Lk = self.layers[0].Lk
        assert Lk + 1 <= self.cap
        tok = torch.tensor([[tok_id]], device="cuda")
        h = m.embed_tokens(tok)
        cos, sin = m.rotary_emb(h, torch.tensor([[Lk + MUT.get("rope_off", 0)]], device="cuda"))
        for li, layer in enumerate(m.layers):
            sa = layer.self_attn
            x = layer.input_layernorm(h)
            q = sa.q_proj(x).view(1, 1, self.H, D).transpose(1, 2)
            k = sa.k_proj(x).view(1, 1, self.KVH, D).transpose(1, 2)
            v = sa.v_proj(x).view(1, 1, self.KVH, D).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            out = self.layers[li].attend(q[0, :, 0], k[0, :, 0], v[0, :, 0])           # [H, D] fp32
            o = out.to(h.dtype).view(1, 1, self.H * D)
            h = h + sa.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.model.lm_head(m.norm(h[:, -1]))[0].float()

    def generate(self, ids, max_new, eos):
        logits = self.prefill(ids, max_new)
        out = []
        for _ in range(max_new):
            t = int(logits.argmax())
            out.append(t)
            if t in eos:
                break
            logits = self.step(t)
        return out

    def memory(self):
        gpu = sum(sum(getattr(L.st, f).numel() * getattr(L.st, f).element_size() for f in ("kp", "km", "vp", "vm"))
                  + L.kx.numel() * 2 * 2 for L in self.layers if L.st is not None)
        host = sum(sum(a.view(a.L).numel() * a.es for a in L.ar) for L in self.layers)
        return dict(fp16_inexact=int(sum(int(L.inexact) for L in self.layers)), guard=self.layers[0].guard,
                    gpu_store_gib=gpu / 2 ** 30, host_store_gib=host / 2 ** 30,
                    host_arena_gib=sum(sum(a.buf.numel() * a.es for a in L.ar) for L in self.layers) / 2 ** 30)

