# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""rp_run (TieredKV): drive the real engine E1 (rp_engine.py) on RULER items (2026-11-09).
Modes:
  ruler  free greedy decoding exactly as tb_bench v3.7 (md5 ef85729a): prompt = input + answer_prefix (tokenizer BOS, no chat
         template), max new tokens RULER_MAXGEN, stop at EOS (not kept) or the cap, pred = decode(skip_special_tokens),
         score = RULER string_match_all (vt / fwe) or string_match_part (qa_1). Both widths (h33, h54) per item, one prefill
         each (the store depends on the width).
  check  logic check of the engine against the accuracy reference on real decode steps: the exact post-RoPE K / V of every
         layer are kept on the CPU; at the checked steps each layer's engine output is compared with cap_attn.attend on a
         reference IncStore built from those exact rows (same q). Steps: the first decode step and the step of the first window
         flush (decoding continues greedily until a flush has happened). Descriptive (code verification, not a result).
Usage: python -B -I rp_run.py <model_dir> <out.jsonl> --mode ruler|check --data f1,f2 --items 0:10 [--widths 33,54] [--cap 140000]"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics as St
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rp_engine as E  # noqa: E402

RULER_MAXGEN = {"vt": 30, "cwe": 120, "fwe": 50, "qa_1": 32}           # tb_bench v3.7 line 203
RULER_PART = ("qa_1",)
END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>", "<|end_of_text|>")


def ruler_score(pred, refs, kind):
    hit = [1.0 if r.lower() in pred.lower() else 0.0 for r in refs]
    return max(hit) if kind in RULER_PART else sum(hit) / len(refs)


def eos_set(eng):
    eos = set()
    for e in (eng.model.generation_config.eos_token_id, eng.tok.eos_token_id):
        eos |= set(e if isinstance(e, list) else ([e] if e is not None else []))
    vocab = eng.tok.get_vocab()
    return eos | {vocab[t] for t in END_TOKENS if t in vocab}


def decode_tb(eng, ids, mx, eos):
    """tb_bench v3.7 free greedy loop: first token from the prefill logits; EOS stops (not kept); the cap stops without an
    extra forward."""
    t0 = time.time()
    nxt = int(eng.prefill(ids, mx).argmax())
    torch.cuda.synchronize()
    t_pre = time.time() - t0
    toks, stop, t_dec = [], "cap", []
    for step in range(mx):
        if nxt in eos:
            stop = "eos"
            break
        toks.append(nxt)
        if step == mx - 1:
            break
        t1 = time.time()
        nxt = int(eng.step(nxt).argmax())
        t_dec.append(time.time() - t1)
    return toks, stop, t_pre, t_dec


def items_of(data, rng, length):
    """(task tag 'ruler:<kind>@<length>' as tb_bench, index, row, data file md5)."""
    i0, i1 = (int(x) for x in rng.split(":"))
    out = []
    for f in data.split(","):
        kind = Path(f).parent.name
        if kind not in RULER_MAXGEN:
            raise SystemExit(f"{f}: task kind {kind} not supported here (RULER vt / fwe / qa_1 / cwe only)")
        rows = [json.loads(l_) for l_ in open(f, encoding="utf-8")]
        assert len(rows) >= i1, (f, len(rows))
        fmd5 = hashlib.md5(Path(f).read_bytes()).hexdigest()
        out += [(f"ruler:{kind}@{length}", j, rows[j], fmd5) for j in range(i0, i1)]
    return out


ARM = {"33": "stkm8v8128k64_x2s128_p50_h33", "54": "stkm8v8128k64_x2s128_p50_h54"}   # the tb_bench arm each width matches


def write(path, rec):
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":                                 # a line cut by a kill: start a new line
                with open(path, "a", encoding="utf-8") as g:
                    g.write("\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def md5s():
    """md5 of every module that determines the output: this driver, the engine, the kernel tree and every reference-tree
    module loaded from KVT_ROOT (calibration chain sel10-14_q, s0_cov, pkv5-11_q, rc_q, cq_lab3, cap_attn, cap_codec)."""
    here = Path(os.path.dirname(os.path.realpath(__file__)))
    kdir = Path(os.path.realpath(os.path.dirname(E.KA.__file__)))
    fs = {here / "rp_run.py", here / "rp_engine.py", here / "rp_chunk.py"}
    fs |= {kdir / f for f in ("kalign.py", "kbench4.py", "kbench5.py", "kbench7.py", "kres.py", "kenc.py")}
    root = os.path.realpath(E.ROOT)
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and os.path.realpath(f).startswith(root + os.sep) and f.endswith(".py"):
            fs.add(Path(f))
    out = {}
    for p in sorted({Path(os.path.realpath(x)) for x in fs}):
        key = (os.path.relpath(p, os.path.dirname(root)) if str(p).startswith(root + os.sep)    # root/hc_mac/...
               else f"kalign/{p.name}" if p.parent == kdir else f"engine/{p.name}" if p.parent == here else str(p))
        assert key not in out, key
        out[key] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


def read_records(path):
    out = []
    if os.path.exists(path):
        for l_ in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(l_))
            except json.JSONDecodeError:                               # a line cut by a kill: ignored (the item reruns)
                pass
    return out


def done_keys(path):
    return {(r["task"], r["i"], r["width"], r.get("data_md5")) for r in read_records(path)
            if not r.get("meta") and r.get("status") in ("ok", "flag", "skipped_length")}


def run_ruler(eng, a, eos):
    assert not E.MUT, "mutation state in a ruler run"
    done = done_keys(a.out)
    for task, j, r, fmd5 in items_of(a.data, a.items, a.length):
        kind = task[6:].partition("@")[0]
        ids_cpu = eng.tok(r["input"] + r.get("answer_prefix", ""), return_tensors="pt").input_ids
        ids_md5 = hashlib.md5(ids_cpu.numpy().tobytes()).hexdigest()       # as tb_bench's cfg item hash (line 527)
        ids = ids_cpu.cuda()
        mx = RULER_MAXGEN[kind]
        for w in a.widths.split(","):
            if (task, j, f"h{w}", fmd5) in done:
                continue
            eng.kb, eng.vb = int(w[0]), int(w[1])
            rec = dict(task=task, i=j, width=f"h{w}", tb_arm=ARM[w], len=int(ids.shape[1]), L0=int(ids.shape[1]), maxgen=mx,
                       bos=int(ids[0, 0]), ids_md5=ids_md5, data_md5=fmd5,
                       prefill_path="sdpa_attention_forward" if ids.shape[1] <= E.LONG_PREFILL else "chunked_bottom_right")
            if ids.shape[1] + mx > a.length + 1:                          # tb_bench v3.7 length budget (line 491)
                rec.update(status="skipped_length")
                write(a.out, rec)
                continue
            torch.cuda.reset_peak_memory_stats()
            try:
                toks, stop, t_pre, t_dec = decode_tb(eng, ids, mx, eos)
                pred = eng.tok.decode(toks, skip_special_tokens=True)
                rec.update(status="ok", pred=pred, ntok=len(toks), hit_max=len(toks) == mx, stop=stop,
                           score=ruler_score(pred, r["outputs"], kind),
                           prefill_sec=t_pre, decode_ms_median=(St.median(t_dec) * 1e3 if t_dec else None),
                           gpu_peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, **eng.memory())
            except E.EngineFlag as e:                                    # the reference raises here too: error, 0 under ITT
                rec.update(status="flag", score=0.0, error=str(e)[:300])
            except Exception:  # noqa: BLE001  (recorded, not scored; the item can be resumed)
                import traceback
                rec.update(status="error", score=0.0, error=traceback.format_exc()[-1500:])
            torch.cuda.synchronize()
            eng.layers = None
            torch.cuda.empty_cache()
            write(a.out, rec)
            print(json.dumps({k: rec.get(k) for k in ("task", "i", "width", "status", "score", "ntok", "stop", "decode_ms_median")}), flush=True)


class Recorder:
    """keeps the exact post-RoPE K / V of every layer on the CPU (check mode only) and compares at chosen steps."""

    def __init__(self, eng, extra):
        """extra: decode rows to preallocate after the prompt (CPU copies of the exact rows, written in place)."""
        self.eng, self.kv, self.on, self.res, self.exact, self.extra, self.mutate = eng, {}, False, [], False, extra, None
        self.orig = (E.LayerStore.prefill, E.LayerStore.attend, E.LayerStore._flush)
        self.prefill_L, self.nflush = {}, {}
        orig_prefill, orig_attend, orig_flush = self.orig
        rec = self

        def flush(st):                                   # E1CHK3: counts flushes per layer; shift2 corrupts the 2nd only
            n = rec.nflush.get(st.li, 0) + 1
            rec.nflush[st.li] = n
            on = rec.mutate == "shift2" and n == 2 and not rec.exact       # never inside the forward check
            if on:
                E.MUT["shift"] = True
            try:
                return orig_flush(st)
            finally:
                if on:
                    E.MUT.pop("shift", None)

        def prefill(st, k, v):
            KVH, L0, d = k.shape
            rec.prefill_L[st.li] = L0 // E.RW * E.RW - E.Z
            kc = torch.empty(KVH, L0 + rec.extra, d, dtype=torch.bfloat16)
            vc = torch.empty_like(kc)
            kc[:, :L0].copy_(k.to("cpu", torch.bfloat16)); vc[:, :L0].copy_(v.to("cpu", torch.bfloat16))
            rec.kv[st.li] = [kc, vc, L0]
            return orig_prefill(st, k, v)

        def attend(st, q, k1, v1):
            kv = rec.kv[st.li]
            T = kv[2]
            assert T < kv[0].shape[1], "check mode: preallocated rows exhausted"
            kv[0][:, T].copy_(k1.to("cpu", torch.bfloat16)); kv[1][:, T].copy_(v1.to("cpu", torch.bfloat16))
            kv[2] = T + 1
            if rec.mutate == "win" and not rec.exact and 64 <= st.nw <= 67:   # E1CHK3 mutant: window rows 64-67 written
                k1, v1 = torch.roll(k1, 1, dims=0), torch.roll(v1, 1, dims=0)    # with K and V heads rolled by one (the
                                                          # recorder above keeps the true rows for the reference)
            if rec.mutate == "win2" and not rec.exact and 64 <= st.nw <= 67:  # E1CHK4 mutant: window rows 64-67 written
                k1 = kv[0][:, T - 1].to(k1.device, k1.dtype)                # as the previous token's true row (an
                v1 = kv[1][:, T - 1].to(v1.device, v1.dtype)                # off-by-one write; values stay in range)
            out = orig_attend(st, q, k1, v1)
            if rec.mutate == "bypass":                             # mutation test: fp32 exact attention instead of the kernel
                kv = rec.kv[st.li]
                kf, vf = (t[:, :kv[2]].to("cuda", torch.float32) for t in kv[:2])
                g = q.shape[0] // kf.shape[0]
                sc = torch.einsum("hd,hnd->hn", q.float(), kf.repeat_interleave(g, 0)) * E.D ** -0.5
                out = torch.einsum("hn,hnd->hd", torch.softmax(sc, -1), vf.repeat_interleave(g, 0))
                del kf, vf, sc
            if rec.on:
                rec.compare(st, q, out)
            if rec.exact:                                          # forward check: exact attention over the recorded rows
                kv = rec.kv[st.li]
                kk, vv = (t[:, :kv[2]].to("cuda")[None] for t in kv[:2])    # [1, KVH, T, D] bf16, as HF's DynamicCache
                o = torch.nn.functional.scaled_dot_product_attention(q[None, :, None].to(torch.bfloat16), kk, vv,
                                                                     scale=E.D ** -0.5, enable_gqa=True)
                out = o[0, :, 0].float()
            return out

        E.LayerStore.prefill, E.LayerStore.attend, E.LayerStore._flush = prefill, attend, flush

    def restore(self):
        E.LayerStore.prefill, E.LayerStore.attend, E.LayerStore._flush = self.orig

    def compare(self, st, q, out):
        kv = self.kv[st.li]
        kf, vf = (t[:, :kv[2]].to("cuda", torch.float32) for t in kv[:2])
        KVH, T, d = kf.shape
        W0 = T // E.RW * E.RW
        # the reference recomputes the stack's INFO from the item's own cache at every new item (ours_q._inc_attend):
        # do the same and require the fields the decode reads (order, balance b) to equal the engine's calibration
        E.S12.MF["fmt"] = "m8v8"
        E.S10.ST["on"], E.S10.ST["li"] = True, st.li
        try:
            E.S10.stack_base(kf, vf, st.li)
        finally:
            E.S12.MF["fmt"] = None
            E.S10.ST["on"] = False
        c_item = E.KA.calib_of(st.li, E.S10, E.P9, E.P10, E.P11, E.Q7)
        calib_equal = all(torch.equal(torch.as_tensor(x).cpu(), torch.as_tensor(y).cpu()) if torch.is_tensor(x) else x == y
                          for x, y in zip(c_item, st.calib))
        ref = E.CA.IncStore(KVH, d, kf.device, st.kb, st.vb)
        for w_mid in range(4096, W0, 4096):                        # per-token codes: any split gives the same bytes
            ref.grow(kf, vf, w_mid, st.li, E.S10, E.Q5, E.S10.Q6, E.P9, E.P10, E.P11, E.Q7)
        ref.grow(kf, vf, W0, st.li, E.S10, E.Q5, E.S10.Q6, E.P9, E.P10, E.P11, E.Q7)
        ref.set_exact(kf, vf, T)
        o_ref = E.CA.attend(q.float()[:, None, :], ref, st.li, E.D ** -0.5, E.M_CAND, E.K_FETCH, E.LAM, E.S10, E.Q5, E.P9,
                            E.P10, E.P11, E.Q7)[:, 0]
        hn = o_ref.abs().max(-1).values                                            # per query head
        den = torch.maximum(hn, 0.01 * hn.median())                                # tiny heads do not blow up the ratio
        err = (out - o_ref).abs().max(-1).values / den
        # exact attention (fp32) from the same rows: both compressed paths' distance to it
        g = q.shape[0] // KVH
        sc = torch.einsum("hd,hnd->hn", q.float(), kf.repeat_interleave(g, 0)) * E.D ** -0.5
        ex = torch.einsum("hn,hnd->hd", torch.softmax(sc, -1), vf.repeat_interleave(g, 0))
        en = ex.abs().max(-1).values
        nrm = torch.maximum(en, 0.01 * en.median())
        e_eng, e_ref = float(((out - ex).abs().max(-1).values / nrm).median()), float(((o_ref - ex).abs().max(-1).values / nrm).median())
        L = W0 - E.Z
        P = self.prefill_L.get(st.li, L)                                           # coded rows written by the prefill encode

        def diff(a, b):                                                            # per-row mismatch flags [KVH, n]
            return (a.contiguous().view(torch.uint8).cpu() != b.contiguous().view(torch.uint8).cpu()).any(-1)
        gpu = {f: diff(getattr(st.st, g_)[:, :L], getattr(ref, f)[:, E.Z:W0]) for f, g_ in (("kb", "kp"), ("km", "km"), ("vb", "vp"), ("vm", "vm"))}
        host = {f: diff(st.ar[i].view(L), getattr(ref, f)[:, E.Z:W0]) for i, f in enumerate(("kr", "krm", "vr", "vrm"))}
        allf = dict(gpu, **host)
        bytes_eq = {f: float(m[:, :P].float().mean()) for f, m in gpu.items()}             # prefill region, rate
        host_eq = {f: float(m[:, :P].float().mean()) for f, m in host.items()}
        nb = (L - P) // E.RW if L > P else 0                                     # E1CHK3: flushed blocks so far
        per_block = {f: [int(m[:, P + b * E.RW:P + (b + 1) * E.RW].sum()) for b in range(nb)] for f, m in allf.items()}
        flush_rows = {f: max(v) for f, v in per_block.items()} if nb else {}       # max over 128-row blocks, row count
        idx = st.st.idx.reshape(-1)
        self.res.append(dict(layer=st.li, T=T, W0=W0, nw=st.nw, rel_err_max=float(err.max()), rel_err_median=float(err.median()),
                             finite=bool(torch.isfinite(out).all() and torch.isfinite(o_ref).all()), gpu_byte_row_mismatch=bytes_eq,
                             host_byte_row_mismatch=host_eq, flush_block_row_mismatch=flush_rows, flush_blocks=nb, flush_rows_per_block=per_block,
                             calib_item_equal_by_construction=bool(calib_equal),
                             err_vs_exact_engine=e_eng, err_vs_exact_reference=e_ref,
                             idx_in_range=float(((idx >= 0) & (idx < L)).float().mean())))
        del kf, vf, ref
        torch.cuda.empty_cache()


def check_forward(eng, ids, nsteps=6):
    """the engine's manual forward vs the stock HF forward, both with exact attention: prefill logits, then nsteps decode
    steps teacher-forced on the HF greedy tokens (HF: DynamicCache + sdpa_attention_forward, as tb_bench's fp path)."""
    from transformers import DynamicCache
    cache = DynamicCache()
    with torch.no_grad():
        lg = [eng.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1].float()]
        toks = []
        for _ in range(nsteps):
            toks.append(int(lg[-1].argmax()))
            lg.append(eng.model(input_ids=torch.tensor([[toks[-1]]], device="cuda"), past_key_values=cache,
                                use_cache=True).logits[0, -1].float())
    lg = [x.cpu() for x in lg]
    L0 = ids.shape[1]
    hf_k0 = cache.layers[0].keys[0, :, L0:L0 + nsteps].float().cpu()      # HF's post-RoPE layer-0 keys of the decode rows
    cache = None
    torch.cuda.empty_cache()
    R = eng._rec
    R.kv, R.exact, R.on = {}, True, False
    eng.kb, eng.vb = 3, 3                                     # the forward check is width-independent (exact attention output)
    try:
        el = [eng.prefill(ids, nsteps + 1).cpu()]
        for t in toks:
            el.append(eng.step(t).cpu())
    finally:
        R.exact = False
    en_k0 = R.kv[0][0][:, L0:L0 + nsteps].float()                         # the engine's layer-0 keys of the same rows
    key0_rel = float((en_k0 - hf_k0).abs().max() / hf_k0.abs().max().clamp_min(1e-12))
    eng.layers = None
    torch.cuda.empty_cache()
    rows = [dict(step=i, key0_rel=key0_rel, finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
                 max_abs_logit_diff=float((x - y).abs().max()), argmax_equal=int(x.argmax()) == int(y.argmax()),
                 hf_top2_gap=float(x.topk(2).values[0] - x.topk(2).values[1]),
                 kl_hf_engine=float((torch.log_softmax(x, -1).exp() * (torch.log_softmax(x, -1) - torch.log_softmax(y, -1))).sum()))
            for i, (x, y) in enumerate(zip(lg, el))]
    return rows


# pass criteria (code verification; frozen with the engine registration before any check output). The forward check
# replaces the attention OUTPUT by exact attention (the compressed path still runs for bookkeeping and the guard): it tests
# the non-attention forward only. Every mutation run (--mutate) must FAIL at least one criterion.
FWD_KL_MAX = 5e-3            # E1CHK only; since E1CHK2 the KL is recorded but not a criterion (it is flatness-dependent:
                             # E1CHK 3B peaks 2.6e-3 at top-2 gaps 0.75-0.88; at an exact tie a logit difference d gives ~d^2/8)
FWD_LOGIT_MAX = 2.0          # E1CHK2: every compared step, max |logit(HF) - logit(engine)| (E1CHK unmutated per-item max
                             # 0.22-0.31 on 3B, 0.44-0.81 on 8B; RoPE errors are caught by KEY0_MAX)
FWD_TIE = 0.25               # an argmax disagreement is tolerated when HF's top-1 / top-2 gap is <= this (about 2 bf16 ulp)
KEY0_MAX = 1e-2              # layer-0 decode-row keys, engine vs HF cache (same embedding -> same keys; catches RoPE positions)
ATT_MED_MAX = 2e-3           # per checked step: median over layers of the per-layer median relative error (kalign v1: 3.9e-4)
ATT_MAX_OK = 5e-2            # per-layer max relative error <= this in >= 95% of layers (top-M ties may flip a few rows) ...
ATT_MAX_ALL = 0.2            # ... and <= this in every layer
BYTE_MAX = 2e-3              # prefill-region record rows differing, per field, GPU and host. E1CHK2: 1e-3 -> 2e-3 (E1CHK
                             # worst cell 8.31e-4, ratio 0.83; the calib mutant gives 1.0)
FLUSH_ROWS_MAX = 7           # flushed block (128 rows x KV heads): rows differing per field, GPU and host. E1CHK2: 2 -> 7,
                             # from E1CHK's prefill-region rates (Poisson, lambda_hi 0.851 per 1024 rows, 2880 tests:
                             # family-wise false-fail 0.92%); a block-boundary row wrong in all 8 KV heads (8 rows) still fails
EXACT_RATIO = (0.5, 2.0)     # median over layers of (engine error vs exact + 1e-6) / (reference error vs exact + 1e-6)


def verdict_forward(rows, nsteps=6):
    return len(rows) == nsteps + 1 and all(r["finite"] and (r["argmax_equal"] or r["hf_top2_gap"] <= FWD_TIE)
                                           and r["max_abs_logit_diff"] <= FWD_LOGIT_MAX and r["key0_rel"] <= KEY0_MAX for r in rows)


def verdict_attention(checked):
    ok = bool(checked) and any(c["flush"] for c in checked)
    for c in checked:
        L = c["layers"]
        ok &= all(x["finite"] for x in L)
        ok &= St.median(x["rel_err_median"] for x in L) <= ATT_MED_MAX
        ok &= sum(x["rel_err_max"] <= ATT_MAX_OK for x in L) >= 0.95 * len(L)
        ok &= all(x["rel_err_max"] <= ATT_MAX_ALL for x in L)
        ok &= all(max(x["gpu_byte_row_mismatch"].values()) <= BYTE_MAX for x in L)
        ok &= all(max(x["host_byte_row_mismatch"].values()) <= BYTE_MAX for x in L)
        ok &= all(max(x["flush_block_row_mismatch"].values(), default=0) <= FLUSH_ROWS_MAX for x in L)
        r_ = St.median((x["err_vs_exact_engine"] + 1e-6) / (x["err_vs_exact_reference"] + 1e-6) for x in L)
        ok &= EXACT_RATIO[0] <= r_ <= EXACT_RATIO[1]
        ok &= all(x["calib_item_equal_by_construction"] for x in L)
    return bool(ok)


N_FLUSH = 3                       # E1CHK3: flushes compared per (width, phase)
POST = {1: (1, 65, 127), 2: (1,)}  # E1CHK3: decode steps after flush 1 / 2 also compared (window fill 1, 65, 127; 1)
EXPECTED_TAGS = {"first", "flush1", "flush2", "flush3", "f1+1", "f1+65", "f1+127", "f2+1"}


def _phases(ids):
    """E1CHK3: the item as given, and an 'edge' copy whose prompt length is 127 mod 128 (its first decode step flushes and
    the window then starts empty), made by dropping tokens 1..k right after BOS (a code check, not an accuracy input)."""
    k = (ids.shape[1] - (E.RW - 1)) % E.RW
    return [("orig", ids), ("edge", torch.cat([ids[:, :1], ids[:, 1 + k:]], dim=1) if k else ids)]


def run_check(eng, a, eos):
    eng._rec = Recorder(eng, extra=(N_FLUSH + 1) * E.RW + 8)
    eng._rec.mutate = a.mutate if a.mutate != "none" else None
    E.MUT.clear()
    if a.mutate == "rope":
        E.MUT["rope_off"] = 1
    elif a.mutate == "shift":
        E.MUT["shift"] = True
    elif a.mutate == "calib":                                  # engine-only: every layer's channel order rolled by one
        eng.calib = [(torch.roll(c[0], 1, dims=-1),) + tuple(c[1:]) for c in eng.calib]
    try:
        for task, j, r, _ in items_of(a.data, a.items, a.length):
            ids = eng.tok(r["input"] + r.get("answer_prefix", ""), return_tensors="pt").input_ids.cuda()
            try:
                _run_check(eng, a, task, j, ids)
            except Exception:  # noqa: BLE001  (a check that cannot run is a failed check for this item)
                import traceback
                write(a.out, dict(mode="check_error", task=task, i=j, mutate=a.mutate, error=traceback.format_exc()[-2000:]))
                torch.cuda.synchronize()
                eng.layers = None
                torch.cuda.empty_cache()
    finally:
        eng._rec.restore()
        E.MUT.clear()
    recs = [r for r in read_records(a.out) if r.get("mode", "").startswith("check") and r.get("mutate") == a.mutate]
    errs = [r for r in recs if r["mode"] == "check_error"]
    by_item = {}
    for r in recs:
        if r["mode"] in ("check", "check_forward"):
            by_item.setdefault((r["task"], r["i"]), []).append(r["PASS"])
    if a.mutate == "none":     # a real check passes only with no crash and every verdict PASS on every item
        verdict = "PASS" if not errs and by_item and all(all(v) for v in by_item.values()) else "FAIL"
    else:                      # a mutant is caught only by a verdict-level FAIL on every item (a crash does not count)
        verdict = "CAUGHT" if not errs and by_item and all(not all(v) for v in by_item.values()) else "NOT_CAUGHT"
    write(a.out, dict(mode="check_summary", mutate=a.mutate, n_items=len(by_item), n_errors=len(errs), VERDICT=verdict))
    print("CHECK_SUMMARY", a.mutate, verdict, flush=True)


def _run_check(eng, a, task, j, ids):
    try:
        return _run_check_inner(eng, a, task, j, ids)
    finally:
        eng._rec.on = eng._rec.exact = False


def _run_check_inner(eng, a, task, j, ids):
    assert ids.shape[1] <= E.LONG_PREFILL, "check mode keeps HF's full cache: prompts up to 40960 tokens only"
    fw = check_forward(eng, ids)
    write(a.out, dict(mode="check_forward", task=task, i=j, mutate=a.mutate, L0=int(ids.shape[1]), steps=fw, PASS=verdict_forward(fw),
                      limit="engine and reference share the calibration state (S10 / P9 / RC tables): a wrong calibration "
                            "would affect both sides and is not detected here"))
    print(json.dumps(dict(check_forward=fw, PASS=verdict_forward(fw))), flush=True)
    for w in a.widths.split(","):
        for ph, pids in _phases(ids):
            eng.kb, eng.vb = int(w[0]), int(w[1])
            R = eng._rec
            R.kv, R.res, R.nflush = {}, [], {}
            nxt = int(eng.prefill(pids, (N_FLUSH + 1) * E.RW + 4).argmax())
            L0 = pids.shape[1]
            checked, step, nfl, since = [], 0, 0, {}
            while True:
                Lk = eng.layers[0].Lk
                flush_now = (Lk + 1) % E.RW == 0
                tags = (["first"] if step == 0 else []) + ([f"flush{nfl + 1}"] if flush_now else [])
                tags += [f"f{f}+{step - since[f]}" for f, offs in POST.items() if f in since and step - since[f] in offs]
                R.on = bool(tags)
                n0 = len(R.res)
                nxt = int(eng.step(nxt).argmax())
                if flush_now:
                    nfl += 1
                    since[nfl] = step
                if R.on:
                    checked.append(dict(step=step, Lk=Lk, flush=flush_now, tags=tags, layers=R.res[n0:]))
                step += 1
                if nfl >= N_FLUSH or step > (N_FLUSH + 1) * E.RW:
                    break
            R.on = False
            got = {t for c in checked for t in c["tags"]}
            complete = EXPECTED_TAGS <= got
            errs = [x["rel_err_max"] for c in checked for x in c["layers"]]
            rec = dict(mode="check", task=task, i=j, mutate=a.mutate, width=f"h{w}", phase=ph, L0=int(L0), complete=complete,
                       steps=[dict(step=c["step"], Lk=c["Lk"], flush=c["flush"], tags=c["tags"],
                                   rel_err_max=max(x["rel_err_max"] for x in c["layers"]),
                                   rel_err_median=St.median(x["rel_err_median"] for x in c["layers"]),
                                   byte_mismatch_max={f: max(x["gpu_byte_row_mismatch"][f] for x in c["layers"]) for f in ("kb", "km", "vb", "vm")})
                              for c in checked], all_layers=[c["layers"] for c in checked],
                       rel_err_max_all=(max(errs) if errs and all(math.isfinite(x) for x in errs) else float("nan")),
                       PASS=bool(complete and verdict_attention(checked)))
            write(a.out, rec)
            print(json.dumps({k: rec[k] for k in ("width", "phase", "L0", "complete", "rel_err_max_all", "PASS")}), flush=True)
            eng.layers = None
            torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--mode", choices=["ruler", "check"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--items", default="0:1")
    ap.add_argument("--widths", default="33,54")
    ap.add_argument("--cap", type=int, default=140000)
    ap.add_argument("--prec", default="f16x2")
    ap.add_argument("--mutate", default="none", choices=["none", "rope", "shift", "bypass", "calib", "shift2", "win", "win2"],
                    help="check mode only: a deliberately broken engine whose check must FAIL")
    ap.add_argument("--length", type=int, required=True, help="RULER length tag of the data (32768, 130000): task name and budget")
    a = ap.parse_args()
    if a.mode != "check" and a.mutate != "none":
        ap.error("mutations exist only in check mode")
    if os.environ.get("PKV5_ROWS"):
        ap.error("PKV5_ROWS must be unset (it changes pkv5_q)")
    cal_md5 = hashlib.md5(Path(os.environ["KALIGN_CAL"]).read_bytes()).hexdigest()[:8]
    assert cal_md5 == E.CAL_MD5, f"calibration text md5 {cal_md5} != {E.CAL_MD5}"     # before the model load
    m5 = md5s()                                                              # once, before the model load (resume key)
    exp = json.loads((Path(os.path.dirname(os.path.abspath(__file__))) / "EXPECTED_MD5.json").read_text())
    bad = {k: (m5.get(k), v) for k, v in exp.items() if m5.get(k) != v}
    if bad:
        raise SystemExit(f"reference / kernel modules differ from the reviewed versions: {bad}")
    prev = [r for r in read_records(a.out) if r.get("meta") and "md5" in r]
    if prev and prev[0]["md5"] != m5:                                    # resume only with the same code
        raise SystemExit(f"{a.out}: written by a different code version; use a new output file")
    eng = E.Engine(a.model, 3, 3, a.cap, a.prec)
    eos = eos_set(eng)
    write(a.out, dict(meta=True, model=a.model, mode=a.mode, data=a.data, items=a.items, widths=a.widths, cap=a.cap, prec=a.prec,
                      md5=m5, device=torch.cuda.get_device_name(0), torch=torch.__version__,
                      transformers=__import__("transformers").__version__, triton=__import__("triton").__version__, eos=sorted(eos),
                      calibration=eng.cal_info, length=a.length, arms=ARM))
    (run_ruler if a.mode == "ruler" else run_check)(eng, a, eos)
    write(a.out, dict(meta=True, end=True))


if __name__ == "__main__":
    main()
