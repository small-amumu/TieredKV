# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""bl_q: same-harness reproductions of two published sparse-attention KV methods (rootcause fork), for the layered-KV
comparisons of round S4. Both follow the official code read on 2026-11-04:
  Quest     mit-han-lab/Quest 01c1623b, evaluation/quest_attention.py (the accuracy-evaluation path)
  ShadowKV  bytedance/ShadowKV e51904cd, models/kv_cache.py ShadowKVCache + models/base.py layer_compute
  FlexiCache NazmulTakbir/FlexiCache ab0f495b, vllm/attention/ops/flexi_cache_triton_kernels.py (page scores,
            top-k, sparse decode kernel), chunked_prefill_paged_decode.py and v1/worker/gpu_model_runner.py (rerank
            bookkeeping), TopK-Analysis/stability.py + analyze_head_stability.py (unstable-head profiling)
Teacher-forced emulation of decoding: rows < BL["from"] are prefill and use exact attention (both methods prefill densely);
row t >= BL["from"] is a decode step that sees keys 0..t. Inputs are post-RoPE q [H, L, d], k / v [KVH, L, d].

Method names
  quest_b<B>[_k][_all]
                     pages of 16 tokens; per query head, page upper bound sum_c max(q_c kmax_c, q_c kmin_c) over the keys of
                     the page written so far; unselected keys dropped; exact keys and values on the selected pages.
                     Default = evaluation path (evaluation/quest_attention.py): top max(3, min(t+1, B) // 16) pages.
                     _k = deployed kernel path (quest/utils/controller.py, quest/models/QuestAttention.py): the current
                     page always kept plus the top B // 16 - 1 of the earlier pages, no minimum of 3.
                     Layers 0 and 1 dense in both official paths, unless _all. Pages wholly outside a sliding window
                     are excluded from the ranking (the official code has no window; this only helps Quest).
  skv_b<B>_o<N>_r<R>[_g<G>|_fix][_f32]
                     ShadowKV state built from the prefix 0..s-1 at block starts s = from, from + G, ... (G = 128 default):
                     rank-R SVD of the pre-RoPE prefix keys (all KV heads jointly), chunks of 8, landmark = chunk mean of the
                     post-RoPE keys, the N chunks with the lowest min cosine to their landmark kept exact (outliers), the
                     last 4 chunks plus the remainder kept exact (local); per row, softmax(q . landmark * scale) over the
                     other chunks, max over the KV head's query heads, top B // 8 chunks; selected chunks use the low-rank
                     keys (re-rotated) and exact values; tokens s..t (generated in the block) exact; everything else dropped.
                     _fix = official semantics: the state is built once from the prefix 0..from-1 and every later token
                     stays exact (as ShadowKVCache in models/kv_cache.py). The low-rank factors and landmarks are rounded
                     to bf16 as in the official default dtype, unless _f32. B // 8 must not exceed the non-outlier chunks
                     (the official top-k would fail); the efficient ShadowKV_CPU class uses (B // 1024) * 24 outlier
                     chunks, i.e. N = 0 for B < 1024.
  flexi_b<B>_u<U>_r<R>
                     FlexiCache: pages of 16; per KV head, page score = max over its query heads of the Quest upper bound,
                     the current page forced in (sentinel); K = B // 16 pages. U% of all (layer, KV head) pairs are
                     "unstable" (BL["flexi_unstable"][(B, U)], profiled offline by flexi_profile_layer +
                     flexi_pick_unstable on a calibration window, as TopK-Analysis): they re-rank every step. Stable
                     heads rank at decode step 1 (used at once) and at steps R, 2R, ... (used from the next step, as the
                     official post-attention write); between ranks they attend the ranked K pages plus every page created
                     after the ranking. Unselected pages are dropped. Official defaults U = 25, R = 16.
Deviations (documented): ShadowKV's landmark score uses the model's attention scale (the official code divides by
sqrt(128), identical for head_dim 128); ShadowKV only compresses prompts longer than 4096 tokens in the official code, the
emulation applies it at any prefix length; sliding windows and logit softcaps of the model are honoured for both.
"""
from __future__ import annotations

import math
import re

import torch

BL: dict = {"from": 2048, "inv_freq": None, "asc": 1.0, "chunk": 512, "flexi_unstable": {}}


def parse(name):
    m = re.fullmatch(r"quest_b(\d+)(_k)?(_all)?", name)
    if m:
        return {"kind": "quest", "B": int(m[1]), "kernel": bool(m[2]), "dense2": not m[3]}
    m = re.fullmatch(r"flexi_b(\d+)_u(\d+)_r(\d+)", name)
    if m:
        return {"kind": "flexi", "B": int(m[1]), "U": int(m[2]), "R": int(m[3])}
    m = re.fullmatch(r"skv_b(\d+)_o(\d+)_r(\d+)(?:_g(\d+)|(_fix))?(_f32)?", name)
    if m:
        return {"kind": "skv", "B": int(m[1]), "N": int(m[2]), "R": int(m[3]), "G": int(m[4] or 128), "fix": bool(m[5]),
                "bf16": not m[6]}
    raise ValueError(name)


def is_bl(name):
    return name is not None and (name.startswith("quest_") or name.startswith("skv_") or name.startswith("flexi_"))


# rotary helpers (as rc_q.py: rotate-half convention, partial rotary = leading dims)
def _rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


def _cos_sin(L, dev):
    f = torch.arange(L, device=dev, dtype=torch.float64)[:, None] * BL["inv_freq"].to(dev).double()[None]
    e = torch.cat([f, f], -1)
    return (e.cos() * BL["asc"]).float(), (e.sin() * BL["asc"]).float()


def _derotate(k, cos, sin):
    rd = cos.shape[-1]
    kr = k[..., :rd]
    pre = (kr * cos - _rot_half(kr) * sin) / (cos ** 2 + sin ** 2)
    return torch.cat([pre, k[..., rd:]], -1)


def _rotate(k, cos, sin):
    rd = cos.shape[-1]
    kr = k[..., :rd]
    return torch.cat([kr * cos + _rot_half(kr) * sin, k[..., rd:]], -1)


def _probs(s, allow, softcap):
    if softcap:
        s = torch.tanh(s / softcap) * softcap
    return torch.softmax(s.masked_fill(~allow, float("-inf")), -1)


def attend(q, k, v, li, name, scale, kw):
    p = parse(name)
    H_, Lq, d = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    assert Lq == Lk, "teacher-forced emulation expects the full sequence"
    kx, vx = k.repeat_interleave(g, 0), v.repeat_interleave(g, 0)
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
    kpos = torch.arange(Lk, device=q.device)[None, :]
    out = torch.empty(H_, Lq, v.shape[-1], device=q.device)
    F = min(BL["from"], Lq)

    def causal(r0, r1):
        qpos = torch.arange(r0, r1, device=q.device)[:, None]
        al = kpos <= qpos
        if win_:
            al &= kpos > qpos - int(win_)
        return qpos, al

    # prefill rows: exact attention
    for r0 in range(0, F, BL["chunk"]):
        r1 = min(F, r0 + BL["chunk"])
        _, al = causal(r0, r1)
        s = torch.einsum("hqd,hkd->hqk", q[:, r0:r1], kx) * scale
        out[:, r0:r1] = torch.einsum("hqk,hkd->hqd", _probs(s, al[None], softcap), vx)
    if F >= Lq:
        return out
    if p["kind"] == "quest":
        _quest(p, q, k, kx, vx, li, scale, softcap, causal, out, F, win_)
    elif p["kind"] == "flexi":
        _flexi(p, q, k, kx, vx, li, scale, softcap, causal, out, F)
    else:
        _skv(p, q, k, v, kx, vx, scale, softcap, causal, out, F, g)
    return out


def _quest(p, q, k, kx, vx, li, scale, softcap, causal, out, F, win_=None):
    H_, Lq, d = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    PS = 16
    if p["dense2"] and li < 2:
        for r0 in range(F, Lq, BL["chunk"]):
            r1 = min(Lq, r0 + BL["chunk"])
            _, al = causal(r0, r1)
            s = torch.einsum("hqd,hkd->hqk", q[:, r0:r1], kx) * scale
            out[:, r0:r1] = torch.einsum("hqk,hkd->hqd", _probs(s, al[None], softcap), vx)
        return
    P = (Lk + PS - 1) // PS
    pad = P * PS - Lk
    kp = torch.cat([k, k[:, -1:].expand(-1, pad, -1)], 1) if pad else k        # padding never enters a bound (see below)
    kp = kp.reshape(KVH, P, PS, d)
    kmax_pref = kp.cummax(2).values.reshape(KVH, P * PS, d)[:, :Lk]           # max over the page's keys written so far
    kmin_pref = kp.cummin(2).values.reshape(KVH, P * PS, d)[:, :Lk]
    kmax_full, kmin_full = kp.amax(2), kp.amin(2)                             # [KVH, P, d] (full pages only are used)
    kmax_full, kmin_full, kmax_pref, kmin_pref = (t.repeat_interleave(g, 0) for t in (kmax_full, kmin_full, kmax_pref, kmin_pref))
    page_of = torch.arange(Lk, device=q.device) // PS
    for r0 in range(F, Lq, BL["chunk"]):
        r1 = min(Lq, r0 + BL["chunk"])
        qpos, al = causal(r0, r1)
        qr = q[:, r0:r1]
        qp, qn = qr.clamp_min(0), qr.clamp_max(0)
        bound = torch.einsum("hqd,hpd->hqp", qp, kmax_full) + torch.einsum("hqd,hpd->hqp", qn, kmin_full)
        rows = torch.arange(r0, r1, device=q.device)
        own = (qp * kmax_pref[:, rows] + qn * kmin_pref[:, rows]).sum(-1)    # bound of the row's own (partial) page
        prow = rows // PS
        pidx = torch.arange(P, device=q.device)[None, :]
        bound = bound.masked_fill(pidx > prow[:, None], float("-inf"))
        if win_:                                                                # pages wholly outside the window
            bound = bound.masked_fill((pidx + 1) * PS - 1 <= (rows - int(win_))[:, None], float("-inf"))
        bound.scatter_(-1, prow[None, :, None].expand(H_, -1, 1), own[..., None])
        selp = torch.zeros(H_, r1 - r0, P, dtype=torch.bool, device=q.device)
        if p["kernel"]:
            # kernel path: current page always kept, top (B // 16 - 1) of the earlier pages
            nb = max(0, min(p["B"] // PS, int(prow.min()) + 1) - 1)
            earlier = bound.masked_fill(pidx == prow[:, None], float("-inf"))
            npg = torch.clamp(prow, max=nb)                                         # earlier pages available
            kmax_n = int(npg.max())
            if kmax_n:
                top = earlier.topk(kmax_n, -1).indices
                keep = torch.arange(kmax_n, device=q.device)[None, None, :] < npg[None, :, None]
                selp.scatter_(-1, top, keep.expand(H_, -1, -1))
            selp.scatter_(-1, prow[None, :, None].expand(H_, -1, 1), True)
        else:
            budget = torch.clamp(rows + 1, max=p["B"])
            npg = torch.clamp(torch.clamp(budget // PS, min=3), max=prow + 1)   # official: min(max(3, budget // 16), #pages)
            kmax_n = int(npg.max())
            top = bound.topk(kmax_n, -1).indices                                # [H, n, kmax_n]
            keep = torch.arange(kmax_n, device=q.device)[None, None, :] < npg[None, :, None]
            selp.scatter_(-1, top, keep.expand(H_, -1, -1))
        sel = selp.gather(-1, page_of[None, None, :].expand(H_, r1 - r0, -1)) & al[None]
        s = torch.einsum("hqd,hkd->hqk", qr, kx) * scale
        out[:, r0:r1] = torch.einsum("hqk,hkd->hqd", _probs(s, sel, softcap), vx)


def _skv(p, q, k, v, kx, vx, scale, softcap, causal, out, F, g):
    H_, Lq, d = q.shape
    KVH, Lk = k.shape[:2]
    CS, LOCAL = 8, 4
    cos, sin = _cos_sin(Lk, q.device)
    k_pre = _derotate(k, cos, sin)
    step = Lq if p["fix"] else p["G"]
    for b0 in range(F, Lq, step):
        s0 = F if p["fix"] else b0                                                 # prefix the state is built from
        s1 = min(Lq, b0 + step)
        chunks = s0 // CS - LOCAL
        _, al = causal(b0, s1)
        sc_ex = torch.einsum("hqd,hkd->hqk", q[:, b0:s1], kx) * scale
        if chunks <= p["N"]:
            out[:, b0:s1] = torch.einsum("hqk,hkd->hqd", _probs(sc_ex, al[None], softcap), vx)
            continue
        # low-rank keys of the prefix 0..s0-1 (pre-RoPE, all KV heads jointly), re-rotated
        A = k_pre[:, :s0].transpose(0, 1).reshape(s0, KVH * d)
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        r = min(p["R"], S.shape[0])
        Ur, SV = U[:, :r], S[:r, None] * Vh[:r]
        if p["bf16"]:                                                             # official U, SV stored in bf16
            Ur, SV = Ur.bfloat16(), SV.bfloat16()
        k_low = (Ur @ SV).float().reshape(s0, KVH, d).transpose(0, 1)
        k_rec = _rotate(k_low, cos[:s0], sin[:s0])
        # landmarks and outliers from the post-RoPE prefix keys
        kc = k[:, :chunks * CS].reshape(KVH, chunks, CS, d)
        lm = kc.mean(2)
        if p["bf16"]:
            lm = lm.bfloat16().float()
        cosim = torch.nn.functional.cosine_similarity(lm[:, :, None].expand(-1, -1, CS, -1), kc, dim=-1)
        out_idx = cosim.amin(-1).topk(p["N"], -1, largest=False).indices          # [KVH, N]
        is_out = torch.zeros(KVH, chunks, dtype=torch.bool, device=q.device)
        is_out.scatter_(-1, out_idx, True)
        # per row: landmark scores over the non-outlier chunks, max over the query heads of the KV head
        la = torch.einsum("hqd,hcd->hqc", q[:, b0:s1], lm.repeat_interleave(g, 0)) * scale
        la = la.masked_fill(is_out.repeat_interleave(g, 0)[:, None, :], float("-inf"))
        pa = torch.softmax(la, -1).reshape(KVH, g, s1 - b0, chunks).amax(1)        # [KVH, n, chunks]
        nsel = p["B"] // CS
        if nsel > chunks - p["N"]:
            raise ValueError(f"budget {p['B']} needs {nsel} chunks, only {chunks - p['N']} non-outlier chunks at prefix {s0}")
        top = pa.topk(nsel, -1).indices
        selc = torch.zeros(KVH, s1 - b0, chunks, dtype=torch.bool, device=q.device)
        selc.scatter_(-1, top, True)
        tok = torch.zeros(KVH, s1 - b0, Lk, dtype=torch.bool, device=q.device)
        tok[..., :chunks * CS] = selc.repeat_interleave(CS, -1)                   # selected: low-rank keys, exact values
        exact = torch.zeros(KVH, Lk, dtype=torch.bool, device=q.device)
        exact[:, :chunks * CS] = is_out.repeat_interleave(CS, -1)                 # outliers
        exact[:, chunks * CS:] = True                                             # local chunks, remainder, generated
        sel = tok.repeat_interleave(g, 0)
        ex = exact.repeat_interleave(g, 0)[:, None, :]
        sc_rec = torch.einsum("hqd,hkd->hqk", q[:, b0:s1], k_rec.repeat_interleave(g, 0)) * scale
        sc = sc_ex.clone()
        sc[..., :s0] = torch.where(sel[..., :s0], sc_rec, sc_ex[..., :s0])
        allow = (sel | ex) & al[None]
        out[:, b0:s1] = torch.einsum("hqk,hkd->hqd", _probs(sc, allow, softcap), vx)


PS_F = 16


def _kv_bounds(q, k, rows):
    """FlexiCache page scores for decode rows `rows`: [KVH, n, P] = max over the KV head's query heads of the Quest upper
    bound over the keys of each page written so far (row t sees keys 0..t); pages after the row's own page -inf; the
    row's own (last) page +inf (the official sentinel)."""
    H_, Lq, d = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    P = (Lk + PS_F - 1) // PS_F
    pad = P * PS_F - Lk
    kp = (torch.cat([k, k[:, -1:].expand(-1, pad, -1)], 1) if pad else k).reshape(KVH, P, PS_F, d)
    kmx, kmn = kp.amax(2).repeat_interleave(g, 0), kp.amin(2).repeat_interleave(g, 0)
    qr = q[:, rows]
    b = torch.einsum("hqd,hpd->hqp", qr.clamp_min(0), kmx) + torch.einsum("hqd,hpd->hqp", qr.clamp_max(0), kmn)
    b = b.reshape(KVH, g, len(rows), P).amax(1)
    prow = rows // PS_F
    pidx = torch.arange(P, device=q.device)[None, :]
    b = b.masked_fill(pidx[None] > prow[None, :, None], float("-inf"))
    b.scatter_(-1, prow[None, :, None].expand(KVH, -1, 1), float("inf"))
    return b                                                                    # [KVH, n, P]


def _topk_pages(b, K):
    sel = torch.zeros_like(b, dtype=torch.bool)
    sel.scatter_(-1, b.topk(min(K, b.shape[-1]), -1).indices, True)
    return sel & torch.isfinite(b).logical_or(b > 0)                            # never a future page (-inf)


def flexi_profile_layer(q, k, F, B, win=16):
    """instability per (chunk of `win` decode steps, KV head) for one layer, as TopK-Analysis/stability.py:
    1 - mean over the window tail of the random-corrected overlap with the chunk's first step."""
    L = q.shape[1]
    K = B // PS_F
    out = []
    for c0 in range(F, L - win + 1, win):                                       # one chunk of `win` steps at a time
        rows = torch.arange(c0, c0 + win, device=q.device)
        sel = _topk_pages(_kv_bounds(q, k, rows), K)                            # [KVH, win, P]
        n_pages = torch.ceil((rows[1:] + 1).float() / PS_F)
        ov = (sel[:, :1] & sel[:, 1:]).sum(-1).float() / K
        rnd = K / n_pages
        rco = torch.where(rnd < 1, ((ov - rnd) / (1 - rnd).clamp_min(1e-6)).clamp_min(0), torch.zeros_like(ov))
        out.append(1 - rco.mean(-1))
    return torch.stack(out) if out else torch.empty(0, k.shape[0])             # [chunks, KVH]


def flexi_pick_unstable(inst, M, seed=43):
    """inst: {layer: [chunks, KVH]} -> set of (layer, head): per chunk the M most unstable heads over all layers, then the
    M most frequent over a random half of the chunks (analyze_head_stability.py)."""
    import random as _r
    layers = sorted(inst)
    KVH = inst[layers[0]].shape[1]
    allc = torch.cat([inst[l] for l in layers], 1)                              # [chunks, L * KVH]
    sets = [set(allc[i].topk(M).indices.tolist()) for i in range(allc.shape[0])]
    rnd = _r.Random(seed)
    rnd.shuffle(sets)
    train = sets[:len(sets) // 2] or sets
    freq = [0] * allc.shape[1]
    for st in train:
        for h in st:
            freq[h] += 1
    top = sorted(range(len(freq)), key=lambda h: -freq[h])[:M]
    return {(layers[h // KVH], h % KVH) for h in top}


def _flexi(p, q, k, kx, vx, li, scale, softcap, causal, out, F):
    H_, Lq, d = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    K, R = p["B"] // PS_F, p["R"]
    uns = BL["flexi_unstable"].get((p["B"], p["U"]))
    if uns is None:
        raise RuntimeError(f"unstable heads for B={p['B']} U={p['U']} not profiled")
    is_uns = torch.tensor([(li, h) in uns for h in range(KVH)], device=q.device)
    page_of = torch.arange(Lk, device=q.device) // PS_F
    steps = torch.arange(F, Lq, device=q.device)                                # decode step s = t - F + 1
    s_idx = steps - F + 1
    # rank event of each step for stable heads: step 1 (used at once), R, 2R, ... (used from the next step)
    r_ev = torch.where(s_idx <= R, torch.ones_like(s_idx), ((s_idx - 1) // R) * R)
    ev_rows = torch.unique(r_ev) + F - 1                                        # rows where stable heads rank
    ev_sel = _topk_pages(_kv_bounds(q, k, ev_rows), K)                          # [KVH, n_ev, P]
    ev_pos = {int(r): i for i, r in enumerate(ev_rows.tolist())}
    for r0 in range(F, Lq, BL["chunk"]):
        r1 = min(Lq, r0 + BL["chunk"])
        _, al = causal(r0, r1)
        rows = torch.arange(r0, r1, device=q.device)
        now = _topk_pages(_kv_bounds(q, k, rows), K)                            # unstable heads: rank every step
        ei = torch.tensor([ev_pos[int(e) + F - 1] for e in r_ev[r0 - F:r1 - F].tolist()], device=q.device)
        st = ev_sel[:, ei]                                                      # [KVH, n, P] stable selection
        ranked_pages = (torch.ceil((r_ev[r0 - F:r1 - F] + F).float() / PS_F)).long()   # pages at the ranking
        pidx = torch.arange(st.shape[-1], device=q.device)
        st = st | (pidx[None, None, :] >= ranked_pages[None, :, None])          # tail: pages created since
        P_now = now.shape[-1]
        st = st[..., :P_now] if st.shape[-1] >= P_now else torch.cat([st, torch.ones(KVH, r1 - r0, P_now - st.shape[-1], dtype=torch.bool, device=q.device)], -1)
        selp = torch.where(is_uns[:, None, None], now, st)                      # [KVH, n, P]
        tok = selp.gather(-1, page_of[None, None, :].expand(KVH, r1 - r0, -1)).repeat_interleave(g, 0) & al[None]
        s = torch.einsum("hqd,hkd->hqk", q[:, r0:r1], kx) * scale
        out[:, r0:r1] = torch.einsum("hqk,hkd->hqd", _probs(s, tok, softcap), vx)


def selftest(dev="cpu"):
    """budget covering everything must reproduce exact attention (Quest), and full rank + no drops ShadowKV."""
    torch.manual_seed(0)
    H_, KVH, L, d = 4, 2, 300, 16
    q, k, v = torch.randn(H_, L, d, device=dev), torch.randn(KVH, L, d, device=dev), torch.randn(KVH, L, d, device=dev)
    BL["from"], BL["inv_freq"], BL["asc"] = 100, 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d)), 1.0
    sc = d ** -0.5
    kx, vx = k.repeat_interleave(2, 0), v.repeat_interleave(2, 0)
    al = torch.arange(L)[None, :] <= torch.arange(L)[:, None]
    ref = torch.einsum("hqk,hkd->hqd", torch.softmax((torch.einsum("hqd,hkd->hqk", q, kx) * sc).masked_fill(~al, float("-inf")), -1), vx)
    e1 = (attend(q, k, v, 5, "quest_b4096", sc, {}) - ref).abs().max().item()
    e2 = (attend(q, k, v, 5, f"skv_b4096_o0_r{KVH * d}_g16", sc, {}) - ref).abs().max().item()
    e3 = (attend(q, k, v, 5, "quest_b32", sc, {}) - ref).abs().max().item()
    return e1, e2, e3


if __name__ == "__main__":
    print("selftest (quest full budget, skv full rank, quest small budget):", selftest())
