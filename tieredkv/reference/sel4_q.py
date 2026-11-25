# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""sel4_q (round S4, rootcause fork): sel3_q plus
  (1) the error variance computed per row chunk (CHUNK rows; long windows fit), (2) PREFILL["on"]: rows < PREFILL["from"]
      use exact attention for every method (dense prefill, as the published sparse-attention systems), (3) two-stage
      controls: first stage by exact attention (x2o<M>; an ORACLE control: it needs every exact key, so its eff_bits
      understate its real reads and it is never compared on bits) or random candidates (x2r<M>; a deterministic hash of
      (layer, head, row, key), independent of the chunk size), (4) the mass-matched
      constant shift _m<lam>: every unrefined key of a row gets the same shift, the attention-weighted mean (under the
      base scores) of lam/100 * sigma^2/2 over the row's unrefined keys, i.e. the same total shift as _j<lam> without
      the per-key variance structure, (5) the reproduced baselines (bl_q: quest_ / skv_ / flexi_) routed through attend.
sel3_q (round S3, rootcause fork): sel2_q plus
  (1) a two-stage selection x2s<M>: the M keys with the largest base attention (group mean, as xqs) are candidates whose
      exact keys are fetched; their scores become exact; among them the K with the largest attention under those scores
      (other keys debiased by the run's fraction) also get exact values. Transfer per KV head: M keys + K values.
  (2) _jcal resolved per K: CAL["lam"] is a dict {K: fraction} chosen by sel3_lab on several calibration windows.
Round S2 as registered uses _jcal: a per-model debias fraction chosen on the calibration text (CAL['lam'], set by
sel2_lab before the evaluation windows) from {0, 0.1, ..., 1}; _eb and _tw stay available but failed the smoke
(both over-correct) and are not in NAMES.
sel2_q (round S2, rootcause fork): sel_q extended with an empirical-Bayes correction of the unrefined logits
(suffix _eb) and more K / debias-fraction variants. _eb: per query row, over the keys still at the base precision
(causal, not sink / window), prior s ~ N(mu, sigma_s^2) with mu = mean(s_hat), sigma_s^2 = var(s_hat) - mean(sigma_j^2)
(floored), noise e_j ~ N(0, sigma_j^2) from the KIVI group steps; each unrefined key's logit is replaced by
log E[exp(s_j) | s_hat_j] = mu + c_j (s_hat_j - mu) + c_j sigma_j^2 / 2, c_j = sigma_s^2 / (sigma_s^2 + sigma_j^2).
Selection is a function of s_hat only, so conditioning on s_hat accounts for the selection (winner's curse) with no
free parameter. The frequentist _jXX debias subtracts XX% of sigma_j^2 / 2 instead.

sel_q: round S1 of the layered-KV line (rootcause fork; registered in reg_pkv.md): why the noisy base score selects
better keys to refine than exact attention. Base: KIVI-128 (pkv5_q kivi128: per-channel keys over 128-token groups,
per-token values, FP16 residual window of 128 tokens, streaming semantics) with the first 4 tokens exact; keep regime
(unrefined keys stay at the base); K keys per KV head selected by the group mean of the criterion and refined exactly
(keys and values). A method name is
  kivi128 | kivi128k<K>_x<crit>[_j<lam>]
  crit   qs  base attention (the noisy base score)
         or  exact attention
         r<rho> s + rho e + sqrt(1 - rho^2) eps, rho in percent; e = base - exact score, eps an independent Gaussian with
             the per-row std of e (r0 = independent noise, r100 = base score)
         cb  oracle by first-order contribution p_j |exp(e_j) - 1| ||v_j - o|| (exact attention p, exact output o)
  _j<lam> Jensen debias of the unrefined quantized keys: their logits get - lam / 100 * sigma_j^2 / 2, where
         sigma_j^2 = scale^2 sum_c q_c^2 Delta_{c, g(j)}^2 / 12 from the stored per-channel group steps of KIVI
         (zero extra storage; window and sink keys are exact, sigma = 0)
Per-layer diagnostics (rows >= DIAG_FROM of the call, every DIAG_STRIDE-th row, all heads), accumulated in DIAG[layer]:
  [rows, attention KL(p || p_final), ||o_final - o||^2, coherent energy ||(sum_U w_j)(vbar_U - o)||^2,
   incoherent energy ||sum_U w_j (v_j - vbar_U)||^2, net mass sum_U w_j, ||identity - measured output error||^2 (an
   implementation check: with exact values o_final - o = (sum_U w_j v_j - (sum_U w_j) o) / (1 + sum_U w_j) exactly,
   and the coherent / incoherent split of that numerator is exact),
   mean e over refined keys, p-weighted mean e over unrefined keys, mean sigma over attended keys, corr(sigma_j, |e_j|)]
  with w_j = p_j (exp(e_j') - 1) the exact first-order weight change of unrefined key j (e_j' after debias),
  vbar_U = sum |w| v / sum |w|.
"""
from __future__ import annotations

import math
import re

import torch

import bl_q as BLQ
import pkv5_q as Q5

SINK = Q5.SINK
calib = None  # set below
GRID_LAMS = tuple(range(10, 101, 10))   # debias fractions (%) of the calibration grid; 0 = plain kivi128k64_xqs
CAL_KS = (16, 64, 256)                   # K values calibrated separately (sel3_lab)
NAMES = ("kivi128", "kivi128k16_xqs", "kivi128k16_xqs_j50", "kivi128k16_xqs_jcal", "kivi128k64_xqs", "kivi128k64_xqs_j50",
         "kivi128k64_xqs_jcal", "kivi128k256_xqs", "kivi128k256_xqs_j50", "kivi128k256_xqs_jcal", "kivi128k64_x2s128_j50",
         "kivi128k96_xqs_j50", "kivi128k64_x2s256_j50", "kivi128k160_xqs_j50", "kivi128k64_xor_j100")
CAL: dict = {"lam": None}                # {K: fraction in percent} once calibrated (not used in S4: fixed fractions)
CHUNK: dict = {"rows": 512}
PREFILL: dict = {"on": True, "from": 2048}
DIAG: dict = {"on": False, "from": 2048, "stride": 8, "lay": None}


def calib_(li, k, v, q=None):
    Q5.calib(li, k, v)


calib = calib_


def parse(name):
    if name == "kivi128":
        return {"K": 0, "crit": "qs", "lam": 0, "eb": False, "tw": False}
    m = re.fullmatch(r"kivi128k(\d+)_x(qs|or|cb|r\d+|2[sor]\d+)(?:_(j\d+|m\d+|jcal|eb|tw))?", name)
    if not m:
        raise ValueError(name)
    suf = m[3] or ""
    if suf == "jcal":
        if CAL["lam"] is None or int(m[1]) not in CAL["lam"]:
            return {"K": int(m[1]), "crit": m[2], "lam": 0, "eb": False, "tw": False, "jcal": True}
        suf = "j%d" % CAL["lam"][int(m[1])]
    return {"K": int(m[1]), "crit": m[2], "lam": int(suf[1:]) if suf[:1] in ("j", "m") else 0, "mass": suf.startswith("m"),
            "eb": suf == "eb", "tw": suf == "tw"}


def eff_bits(name, L, d, g=1):
    p = parse(name)
    gpu, _ = Q5.eff_bits("kivi128", L, d)
    gpu += (16 - gpu) * 4 / L
    if p["crit"][:2] in ("2s", "2o", "2r"):  # M exact keys + K exact values = (M + K) / 2 full rows per KV head
        return gpu, gpu + (int(p["crit"][2:]) + p["K"]) * 8 / L
    return gpu, gpu + p["K"] * 16 / L


def kivi_sigma2(k, q, scale, G=128):
    """per-key score-error variance under uniform rounding of KIVI per-channel keys: [H, Lq, Lk] (0 outside full groups)."""
    KVH, Lk, d = k.shape
    H_ = q.shape[0]
    g = H_ // KVH
    Lf = Lk // G * G
    out = torch.zeros(H_, q.shape[1], Lk, device=q.device)
    if not Lf:
        return out
    x = k[:, :Lf].reshape(KVH, Lf // G, G, d)
    mn, mx = Q5.f16(x.amin(2)), Q5.f16(x.amax(2))
    st = Q5.f16((mx - mn) / 3)                                      # [KVH, nG, d]
    s2 = (st ** 2).repeat_interleave(g, 0)                          # [H, nG, d]
    v = torch.einsum("hqd,hgd->hqg", q * q, s2) * (scale * scale / 12)
    out[..., :Lf] = v.repeat_interleave(G, -1)
    return out


def attend(q, k, v, li, name, scale, kw):
    if name is None:
        return Q5.attend(q, k, v, li, None, scale, kw)
    if BLQ.is_bl(name):
        BLQ.BL["from"] = PREFILL["from"] if PREFILL["on"] else 0
        return BLQ.attend(q, k, v, li, name, scale, kw)
    if PREFILL["on"] and q.shape[1] <= PREFILL["from"]:
        raise ValueError(f"sequence {q.shape[1]} not longer than the dense prefill {PREFILL['from']}")
    out = _attend(q, k, v, li, name, scale, kw)
    F = min(PREFILL["from"], q.shape[1]) if PREFILL["on"] else 0
    if F:                                                                  # dense prefill rows
        BLQ.BL["from"] = F
        ex = BLQ.attend(q[:, :F], k[:, :F], v[:, :F], li, "quest_b16", scale, kw)   # rows < from are exact in bl_q
        out[:, :F] = ex
    return out


def _attend(q, k, v, li, name, scale, kw):
    p = parse(name)
    if p.get("jcal"):
        raise RuntimeError(f"{name}: calibrated fraction not set before evaluation")
    kq, vq, R, _ = Q5.base("kivi128", k, v, li)
    z = 4
    kq[:, :z], vq[:, :z] = k[:, :z], v[:, :z]
    K, crit_kind, lam = p["K"], p["crit"], p["lam"]
    H_, Lq, _ = q.shape
    KVH, Lk = k.shape[:2]
    g = H_ // KVH
    kx, vx, kqx, vqx = (t.repeat_interleave(g, 0) for t in (k, v, kq, vq))
    out = torch.empty(H_, Lq, v.shape[-1], device=q.device)
    kpos = torch.arange(Lk, device=q.device)[None, :]
    softcap, win_ = kw.get("softcap"), kw.get("sliding_window")
    gen = torch.Generator(device=q.device).manual_seed(7919 * li + 17)
    eb, tw = p["eb"], p["tw"]
    need_s2 = bool(lam or eb or tw or DIAG["on"])
    lay = None
    if DIAG["on"] and DIAG["lay"] is not None:
        lay = DIAG["lay"].setdefault(li, [0.0] * 11)
    for r0 in range(0, Lq, CHUNK["rows"]):
        r1 = min(Lq, r0 + CHUNK["rows"])
        n = r1 - r0
        qpos = torch.arange(Lk - Lq + r0, Lk - Lq + r1, device=q.device)[:, None]
        al = kpos <= qpos
        if win_:
            al &= kpos > qpos - int(win_)

        def fin(s):
            if softcap:
                s = torch.tanh(s / softcap) * softcap
            return torch.softmax(s.masked_fill(~al, float("-inf")), -1)

        qr = q[:, r0:r1]
        s = torch.einsum("hqd,hkd->hqk", qr, kqx) * scale
        se = torch.einsum("hqd,hkd->hqk", qr, kx) * scale
        ex = ((kpos >= (qpos + 1) // R * R) & al)[None].expand_as(s) | (kpos < z)[None]
        ex = ex & al[None]
        s = torch.where(ex, se, s)
        s2 = kivi_sigma2(k, q[:, r0:r1], scale) if need_s2 else None   # per chunk: rows are independent
        if s2 is not None:
            s2 = s2.masked_fill(ex, 0.0)
        m = torch.zeros_like(ex)
        kc = torch.zeros_like(ex)                                           # keys exact, values at the base (x2s)
        pe = fin(se)
        if K and crit_kind[:2] in ("2s", "2o", "2r"):
            bad = ~al[None] | ex
            if crit_kind[1] == "s":
                c1 = fin(s)
            elif crit_kind[1] == "o":
                c1 = pe
            else:                                                          # hash of (layer, head, row, key)
                hh = torch.arange(H_, device=q.device)[:, None, None] * 1.0
                rr = qpos[None, :, :].float()
                kk = kpos[None, :, :].float()
                c1 = torch.frac(torch.sin(li * 3.17 + hh * 12.9898 + rr * 78.233 + kk * 37.719) * 43758.5453)
            c1 = c1.masked_fill(bad, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
            kc = torch.zeros_like(c1, dtype=torch.bool).scatter_(-1, c1.topk(min(int(crit_kind[2:]), Lk), -1).indices, True)
            kc = kc.repeat_interleave(g, 0) & ~bad
            if lam and p["mass"]:                                          # constant shift also in the first stage
                U1 = al[None] & ~(ex | kc)
                w1 = torch.where(U1, fin(s), torch.zeros_like(s))
                sh1 = (w1 * (lam / 100) * s2 / 2).sum(-1, keepdim=True) / w1.sum(-1, keepdim=True).clamp_min(1e-30)
                s1 = torch.where(kc, se, s - sh1)
            else:
                s1 = torch.where(kc, se, s - (lam / 100) * s2 / 2 if lam else s)
            c2 = fin(s1).masked_fill(~kc, -float("inf")).reshape(KVH, g, n, Lk).mean(1)
            m = torch.zeros_like(c2, dtype=torch.bool).scatter_(-1, c2.topk(min(K, Lk), -1).indices, True).repeat_interleave(g, 0)
            m = m & kc
        elif K:
            bad = ~al[None] | ex
            if crit_kind == "qs":
                crit = fin(s)
            elif crit_kind == "or":
                crit = pe
            elif crit_kind == "cb":
                o_ex = torch.einsum("hqk,hkd->hqd", pe, vx)
                # ||v_j - o||^2 = ||v_j||^2 - 2 v_j.o + ||o||^2 without materialising [H, n, Lk, dv]
                dev2 = (vx * vx).sum(-1)[:, None, :] - 2 * torch.einsum("hqd,hkd->hqk", o_ex, vx) + (o_ex * o_ex).sum(-1)[..., None]
                crit = pe * torch.expm1(s - se).abs() * dev2.clamp_min(0).sqrt()
            else:
                rho = int(crit_kind[1:]) / 100
                e = s - se
                alf = (al[None] & ~ex).float()
                cnt = alf.sum(-1, keepdim=True).clamp_min(1)
                me = (e * alf).sum(-1, keepdim=True) / cnt
                sd = ((((e - me) ** 2) * alf).sum(-1, keepdim=True) / cnt).sqrt()
                eps = torch.randn(s.shape, generator=gen, device=q.device) * sd
                crit = fin(se + rho * e + math.sqrt(max(0.0, 1 - rho * rho)) * eps)
            crit = crit.masked_fill(bad, -float("inf"))
            c = crit.reshape(KVH, g, n, Lk).mean(1)
            m = torch.zeros_like(c, dtype=torch.bool).scatter_(-1, c.topk(min(K, Lk), -1).indices, True).repeat_interleave(g, 0)
            m = m & ~bad
        sf = torch.where(m | kc, se, s)
        if lam and not p["mass"]:
            sf = torch.where(m | ex | kc, sf, sf - (lam / 100) * s2 / 2)
        elif lam:                                                          # mass-matched constant shift per row and head
            U = al[None] & ~(m | ex | kc)
            w = torch.where(U, fin(s), torch.zeros_like(s))
            shift = (w * (lam / 100) * s2 / 2).sum(-1, keepdim=True) / w.sum(-1, keepdim=True).clamp_min(1e-30)
            sf = torch.where(U, sf - shift, sf)
        if eb:   # empirical-Bayes posterior log-mean of exp(s) given s_hat, for keys still at the base precision
            E = (al[None] & ~ex).float()
            cnt = E.sum(-1, keepdim=True).clamp_min(1)
            mu = (s * E).sum(-1, keepdim=True) / cnt
            var = (((s - mu) ** 2) * E).sum(-1, keepdim=True) / cnt
            ss2 = (var - (s2 * E).sum(-1, keepdim=True) / cnt).clamp_min(1e-6)
            c_ = ss2 / (ss2 + s2)
            sf = torch.where(m | ex | ~al[None], sf, mu + c_ * (s - mu) + c_ * s2 / 2)
        if tw:   # Tweedie with an exponential upper tail: log E[exp(s) | s_hat] ~ s_hat + sigma^2 (1/2 - beta)
            E = al[None] & ~ex
            n_ = E.sum(-1, keepdim=True)
            srt = s.masked_fill(~E, -float("inf")).sort(-1, descending=True).values
            kq = (n_ // 10).clamp_min(1)                                  # upper decile of the base scores
            thr = srt.gather(-1, kq - 1)
            T = E & (s > thr)
            exc = ((s - thr) * T).sum(-1, keepdim=True) / T.sum(-1, keepdim=True).clamp_min(1)
            beta = 1.0 / exc.clamp_min(1e-3)                              # exponential-tail rate (MLE of the excess)
            sf = torch.where(m | ex | ~al[None], sf, s + s2 * (0.5 - beta))
        pr = fin(sf)
        o = torch.einsum("hqk,hkd->hqd", pr, vqx) + torch.einsum("hqk,hkd->hqd", pr * (ex | m), vx - vqx)
        out[:, r0:r1] = o
        if lay is not None:
            sel = (qpos[:, 0] >= DIAG["from"]) & ((qpos[:, 0] - DIAG["from"]) % DIAG["stride"] == 0)
            if sel.any():
                ii = sel.nonzero()[:, 0]
                pe_, pr_, s_, se_, sf_ = pe[:, ii], pr[:, ii], s[:, ii], se[:, ii], sf[:, ii]
                m_, ex_ = m[:, ii], ex[:, ii]
                al_ = al[None, ii].expand_as(m_)
                o_ex = torch.einsum("hqk,hkd->hqd", pe_, vx)
                o_f = torch.einsum("hqk,hkd->hqd", pr_, vx)          # exact values: isolates the score effect
                U = al_ & ~m_ & ~ex_
                w = torch.where(U, pe_ * torch.expm1(sf_ - se_), torch.zeros_like(pe_))
                tot = w.sum(-1, keepdim=True)
                aw = w.abs()
                vbar = torch.einsum("hqk,hkd->hqd", aw, vx) / aw.sum(-1, keepdim=True).clamp_min(1e-30)
                coh = (tot * (vbar - o_ex)).pow(2).sum(-1)
                inc = (torch.einsum("hqk,hkd->hqd", w, vx) - tot * vbar).pow(2).sum(-1)
                first = (torch.einsum("hqk,hkd->hqd", w, vx) - tot * o_ex) / (1 + tot)
                oerr = (o_f - o_ex).pow(2).sum(-1)
                ferr = (first - (o_f - o_ex)).pow(2).sum(-1)
                lpe, lpr = torch.log(pe_.clamp_min(1e-30)), torch.log(pr_.clamp_min(1e-30))
                kl = torch.where(al_, pe_ * (lpe - lpr), torch.zeros_like(pe_)).sum(-1)
                e_ = s_ - se_
                nsel = m_.sum().clamp_min(1)
                pu = torch.where(U, pe_, torch.zeros_like(pe_))
                s2_ = s2[:, ii] if s2 is not None else torch.zeros_like(pe_)
                att = al_ & ~ex_
                sg = s2_.clamp_min(0).sqrt()
                a_ = sg[att]
                b_ = e_.abs()[att]
                cr = float(torch.corrcoef(torch.stack([a_, b_]))[0, 1]) if a_.numel() > 2 and a_.std() > 0 else 0.0
                nr = float(kl.numel())
                vals = [nr, float(kl.sum()), float(oerr.sum()), float(coh.sum()), float(inc.sum()), float(tot.sum()),
                        float(ferr.sum()), float((e_ * m_).sum() / nsel) * nr, float((pu * e_).sum() / pu.sum().clamp_min(1e-30)) * nr,
                        float(sg[att].mean()) * nr, cr * nr]
                for i_, x_ in enumerate(vals):
                    lay[i_] += x_
    return out
