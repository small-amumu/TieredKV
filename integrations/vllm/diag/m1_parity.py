# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""M1 kernel parity (I4 v1.1 section 2): the deployed kernel kalign.StepA (f96b21f5) vs the byte-level reference
cap_attn.attend (449fdb84, equal to sel14_q._attend_h within 2.1e-6) on the same stored bytes, for B1 =
stkm8v8128k64_x2s128_p50_h54 (M 128, K 64, lam -50 = _p50, host residuals key 5 bit / value 4 bit).

Per attention layer of a real text window, at three lengths crossing 128-row boundaries, the store is built from the
model's own K / V with the stack's calibration (kalign_check protocol: CQ.CAL text, Q14.calib / calib_all), then the last
row's query is attended by
  ref    cap_attn.attend (fp32 torch); its stage-1 candidates C and stage-2 value rows V are recorded from its two
         Tensor.topk calls (the stage-1 vector is checked to be one chunk, so the call sees all coded rows)
  ker    StepA on the same bytes; C = st.idx (coded index + 4), V = st.idx[st.selm]
  enc    StepA's own encoder (aenc_kernel) on the same K / V: byte mismatch rates and the decode on its store
  exact  exact attention (scale of the method's own error)
Near-tie tolerance (fixed before any parity comparison, I4 v1.1 section 2): the harness recomputes the reference's
stage-1 criterion c1 from the stored bytes with the reference's own functions (checked bit-identical to the recorded
values), and in float64 (c1_64) and from fp16-rounded base scores (c1_16, the kernel stores s in fp16 for stage 1).
Over the boundary band (ranks M - 32 .. M + 31 of c1), rel32 = |c1_64 - c1| / c1 and rel16 = |c1_16 - c1| / c1.
Phase A runs the calibration text only and sets tau32 = 4 max rel32, tau16 = 4 max rel16 over all its cells; phase B
(the evaluation windows) uses these frozen values. A mismatched entry j of C is accepted under tau when it lies inside
the band |c1[j] - c1_M| <= tau c1_M (c1_M = the reference's M-th largest value); V entries likewise with the reference's
c2 among its candidates (rows outside the reference's C count as accepted only when they are accepted C swaps).
Usage: python m1_parity.py <model_dir> <out.json> --research ROOT --kalign DIR --data DIR [--seq 8192 --start 4096]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

M_CAND, K_VAL, LAM, KB, VB, Z = 128, 64, -50, 5, 4, 4
BAND = 32
X: dict = {"mode": None, "kv": {}, "rows": [], "steps": {}, "a": None, "tau": None, "taus": []}


def boot(a: argparse.Namespace) -> None:
    root = Path(a.research)
    sys.path[:0] = [str(root / "rootcause"), str(root / "code"), a.kalign]
    global CQ, C, CA, Q14, KA, S12, S10, P9, P10, P11, Q7, Q5  # noqa: PLW0603
    import cq_lab3 as CQ  # noqa: N813
    import cap_codec as C  # noqa: N813
    import cap_attn as CA  # noqa: N813
    import sel11_q  # noqa: F401
    import sel14_q as Q14  # noqa: N813
    import kalign as KA  # noqa: N813

    S12, S10 = Q14.S12, Q14.S10
    P9, P10, P11, Q7, Q5 = S10.P9, S10.P10, S10.P11, S10.Q7, S10.Q5
    d = a.data.rstrip("/") + "/"
    CQ.CAL = (d + Path(CQ.CAL[0]).name,) + tuple(CQ.CAL[1:])
    CQ.EVAL = tuple((n, d + Path(p).name) for n, p in CQ.EVAL)


def stack_info(kk: torch.Tensor, vv: torch.Tensor, li: int) -> None:
    S12.MF["fmt"] = "m8v8"
    S10.ST["on"], S10.ST["li"] = True, li
    try:
        S10.stack_base(kk, vv, li)
    finally:
        S12.MF["fmt"] = None
        S10.ST["on"] = False


class TopkLog:
    """records (values, indices) of every Tensor.topk call inside the block (the reference's two selections)."""

    def __enter__(self) -> list:
        self.calls: list = []
        self._orig = torch.Tensor.topk
        orig, calls = self._orig, self.calls

        def topk(t, *args, **kw):  # type: ignore[no-untyped-def]
            r = orig(t, *args, **kw)
            calls.append((t.detach().clone(), r.values.detach().clone(), r.indices.detach().clone()))
            return r

        torch.Tensor.topk = topk  # type: ignore[method-assign]
        return self.calls

    def __exit__(self, *exc: object) -> None:
        torch.Tensor.topk = self._orig  # type: ignore[method-assign]


def c1_variants(q: torch.Tensor, st, li: int, scale: float, g: int) -> dict:  # type: ignore[no-untyped-def]
    """the reference's stage-1 criterion over coded rows [Z, W0), recomputed from the stored bytes with cap_attn's own
    functions (as its pass 1 with one chunk), in fp32, float64 and from fp16-rounded base scores."""
    H_, _, d = q.shape
    KVH, W0, h, dev = st.kx.shape[0], st.W0, d // 2, q.device
    _, bvec = CA.centre_at(li, None, KVH, d, dev, S10, P9, P10, P11, Q7)
    bq = bvec[:, 0, :].repeat_interleave(g, 0)
    order = S10.INFO[li]["order"].repeat_interleave(g, 0)
    qs = (q[:, 0] / bq).gather(-1, order)
    Hh = Q5.CQ.hadamard(h, dev)
    qh, ql = qs[:, :h] @ Hh, qs[:, h:] @ Hh
    kxg = st.kx.float().repeat_interleave(g, 0)
    code, bit = C.unpack_k(st.kb[:, Z:W0])
    km = st.km[:, Z:W0]
    yh = C.decode_hi(code, km[..., 0:1], km[..., 1:2])
    yl = C.decode_lo(bit, km[..., 2:3], km[..., 3:4])
    mt, _ = CA.centre_at(li, torch.arange(Z, W0, device=dev), KVH, d, dev, S10, P9, P10, P11, Q7)
    out = {}
    for name, dt in (("f32", torch.float32), ("f64", torch.float64)):
        c = lambda x: x.to(dt)  # noqa: E731
        sex = torch.einsum("hd,hnd->hn", c(q[:, 0]), c(kxg)) * scale
        s = (torch.einsum("hd,hnd->hn", c(qh), c(yh).repeat_interleave(g, 0))
             + torch.einsum("hd,hnd->hn", c(ql), c(yl).repeat_interleave(g, 0))
             + torch.einsum("hd,hnd->hn", c(q[:, 0]),
                            c(mt).expand(KVH, W0 - Z, d).repeat_interleave(g, 0))) * scale
        lse1 = torch.logaddexp(torch.logsumexp(sex, -1), torch.logsumexp(s, -1))
        out[name] = torch.exp(s - lse1[:, None]).view(KVH, g, W0 - Z).mean(1).double()
        if name == "f32":
            s16 = s.half().float()
            out["f16"] = torch.exp(s16 - lse1[:, None]).view(KVH, g, W0 - Z).mean(1).double()
    return out


def band_rel(c1: dict) -> tuple[float, float]:
    ref = c1["f32"]
    order = ref.argsort(-1, descending=True)
    lo, hi = max(0, M_CAND - BAND), min(ref.shape[-1], M_CAND + BAND)
    idx = order[:, lo:hi]
    r = ref.gather(-1, idx).clamp_min(1e-300)
    rel32 = ((c1["f64"].gather(-1, idx) - r).abs() / r).max()
    rel16 = ((c1["f16"].gather(-1, idx) - r).abs() / r).max()
    return float(rel32), float(rel16)


def set_compare(ref_set: list[int], ker_set: list[int], val: dict[int, float], kth: float,
                tau: float, extra_ok: set[int] | None = None) -> tuple[int, int, set[int]]:
    """(mismatched entries, unaccepted entries, accepted mismatched rows) for one KV head."""
    rs, ks = set(ref_set), set(ker_set)
    diff = (ks - rs) | (rs - ks)
    bad, ok = 0, set()
    for j in diff:
        v = val.get(j)
        near = v is not None and abs(v - kth) <= tau * kth
        if near or (extra_ok is not None and j in extra_ok):
            ok.add(j)
        else:
            bad += 1
    return len(diff), bad, ok


def check_layer(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, li: int, scale: float, doc: str) -> None:
    a = X["a"]
    KVH, T_all, d = k.shape
    g = q.shape[0] // KVH
    lens = [t for t in (T_all - 130, T_all - 60, T_all) if t > 600]
    st = CA.IncStore(KVH, d, k.device, KB)
    st.vbits = VB  # cap_attn's value residual is fixed at 4 bit (decode_res4); StepA.load_store checks it
    for T in lens:
        kk, vv = k[:, :T], v[:, :T]
        stack_info(kk, vv, li)
        W0 = T // 128 * 128
        st.grow(kk, vv, W0, li, S10, Q5, S10.Q6, P9, P10, P11, Q7)
        st.set_exact(kk, vv, T, torch.float16)
        qr = q[:, T - 1:T]
        c1 = c1_variants(qr, st, li, scale, g)
        rel32, rel16 = band_rel(c1)
        if X["mode"] == "tau":
            X["taus"].append(dict(doc=doc, layer=li, T=T, rel32=rel32, rel16=rel16))
            continue
        with TopkLog() as calls:
            ref = CA.attend(qr, st, li, scale, M_CAND, K_VAL, LAM, S10, Q5, P9, P10, P11, Q7,
                            chunk=1 << 20)[:, 0]
        assert len(calls) == 2, f"expected 2 topk calls, got {len(calls)}"
        (v1in, _, i1), (v2in, _, i2) = calls
        ncod = W0 - Z
        one_chunk = v1in.shape[-1] == ncod
        self_ok = one_chunk and bool(torch.equal(v1in.double(), c1["f32"]))
        cand = (i1 + Z).tolist()                                     # positions
        sel = [[cand[h][j] for j in i2[h].tolist()] for h in range(KVH)]
        c2 = {h: {cand[h][j]: float(v2in[h, j]) for j in range(len(cand[h]))} for h in range(KVH)}
        Lc = W0 - Z
        sa = X["steps"].get((Lc, g))
        if sa is None:
            sa = X["steps"][(Lc, g)] = KA.StepA(Lc, GH=g, kbits=KB, vbits=VB, host=True, words=True)
        sa.set_calib(*KA.calib_of(li, S10, P9, P10, P11, Q7))
        sa.load_store(st)
        sa.set_q(qr[:, 0])
        sa.step()
        torch.cuda.synchronize()
        ker = sa.result().clone()
        kc = (sa.idx[:, :M_CAND] + Z).tolist()
        kv_ = [(sa.idx[h][sa.selm[h].bool()] + Z).tolist() for h in range(KVH)]
        tau32, tau16 = X["tau"]["tau32"], X["tau"]["tau16"]
        sets = {"c_mis": 0, "c_bad32": 0, "c_bad16": 0, "v_mis": 0, "v_bad32": 0, "v_bad16": 0}
        for h in range(KVH):
            vals1 = {j + Z: float(c1["f32"][h, j]) for j in range(ncod)}
            kth1 = sorted(vals1.values(), reverse=True)[min(M_CAND, ncod) - 1]
            kth2 = sorted(c2[h].values(), reverse=True)[min(K_VAL, len(c2[h])) - 1]
            for tag, tau in (("32", tau32), ("16", tau16)):
                n, bad, ok_c = set_compare(cand[h], kc[h], vals1, kth1, tau)
                n2, bad2, _ = set_compare(sel[h], kv_[h], c2[h], kth2, tau, ok_c)
                if tag == "32":
                    sets["c_mis"] += n
                    sets["v_mis"] += n2
                sets[f"c_bad{tag}"] += bad
                sets[f"v_bad{tag}"] += bad2
        # ---- the kernel encoder's own store from the same K / V, decoded by the same kernels
        sa.rope_tables(T)
        sa.encode(kk.contiguous(), vv.contiguous(), 0, W0, S10.Q6.GAMMAS, prec=a.prec)
        mm = {}
        for nm_, mine, ref_ in (("kcode", sa.kp, st.kb), ("kmeta", sa.km, st.km), ("vcode", sa.vp, st.vb),
                                ("vmeta", sa.vm, st.vm), ("kres", sa.krp, st.kr), ("kresm", sa.krm, st.krm),
                                ("vres", sa.vrp, st.vr), ("vresm", sa.vrm, st.vrm)):
            r_ = ref_[:, Z:W0].contiguous().to(mine.device)
            mm[nm_] = float((mine.view(torch.uint8) != r_.view(torch.uint8)).float().mean())
        sa.set_exact(st.kx, st.vx)
        sa.set_q(qr[:, 0])
        sa.step()
        torch.cuda.synchronize()
        kenc = sa.result().clone()
        kx, vx = kk.repeat_interleave(g, 0), vv.repeat_interleave(g, 0)
        ex = torch.einsum("hn,hnd->hd",
                          torch.softmax(torch.einsum("hd,hnd->hn", qr[:, 0], kx) * scale, -1), vx)
        n_ = lambda x: float(x.norm())  # noqa: E731
        r = dict(doc=doc, layer=li, T=T, W0=W0, rel=n_(ker - ref) / n_(ref), err_ref=n_(ref - ex) / n_(ex),
                 err_ker=n_(ker - ex) / n_(ex), rel_enc=n_(kenc - ref) / n_(ref),
                 err_kenc=n_(kenc - ex) / n_(ex), mismatch=mm, enc_flags=int(sa.flags.item()),
                 finite=bool(torch.isfinite(ker).all() and torch.isfinite(kenc).all()),
                 self_check=self_ok, rel32=rel32, rel16=rel16, **sets)
        X["rows"].append(r)
        print("M1P", json.dumps(r), flush=True)


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):  # type: ignore[no-untyped-def]
    li = int(module.layer_idx)
    scale = scaling if scaling is not None else 1 / math.sqrt(query.shape[-1])
    assert abs(scale - 1 / math.sqrt(128)) < 1e-9 and not kw.get("softcap") and not kw.get("sliding_window"), \
        "kalign kernels assume d = 128, scale 1/sqrt(128), no softcap, no sliding window"
    o = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale,
                                                         enable_gqa=True)
    if X["mode"] is not None:
        q, k, v = query[0].float(), key[0].float(), value[0].float()
        if X["mode"] == "cap":
            X["kv"][li] = (k.clone(), v.clone(), q.clone())
        else:
            check_layer(q, k, v, li, scale, X["doc"])
    return o.transpose(1, 2).contiguous(), None


def run(model, ids: torch.Tensor, mode: str, doc: str) -> None:  # type: ignore[no-untyped-def]
    X["mode"], X["doc"] = mode, doc
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        X["mode"] = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--research", required=True)
    ap.add_argument("--kalign", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--start", type=int, default=4096)
    ap.add_argument("--prec", default="f16x2", choices=["ieee", "tf32x3", "f16x2"])  # encode_proj default
    a = ap.parse_args()
    X["a"] = a
    boot(a)
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    AttentionInterface.register("m1_parity", attn)
    H64r, H128r = Q5.CQ.hadamard(64, "cuda").float(), Q5.CQ.hadamard(128, "cuda").float()
    assert (H64r - KA.K7.hadamard(64, "cuda")).abs().max() < 1e-7
    assert (H128r - KA.K7.hadamard(128, "cuda")).abs().max() < 1e-7
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="m1_parity").cuda().eval()
    cfg = model.config
    KVH, H = cfg.num_key_value_heads, cfg.num_attention_heads
    KA.configure(KVH, H // KVH)
    tok = AutoTokenizer.from_pretrained(a.model)
    re_ = model.model.rotary_emb
    S10.RC.R["inv_freq"] = re_.inv_freq.detach().float()
    S10.RC.R["asc"] = float(getattr(re_, "attention_scaling", 1.0))
    run(model, CQ.text_ids(tok, CQ.CAL[0], CQ.CAL[1], CQ.CAL[2], "cuda"), "cap", "cal")
    for li, t in X["kv"].items():
        Q14.calib(li, *t)
    Q14.calib_all(model, X["kv"], sorted(X["kv"]))
    X["kv"] = {}
    # ---- phase A: tau from the calibration text only (frozen before any parity comparison)
    run(model, CQ.text_ids(tok, CQ.CAL[0], CQ.CAL[1], a.seq, "cuda"), "tau", "cal")
    X["tau"] = dict(tau32=4 * max(t["rel32"] for t in X["taus"]), tau16=4 * max(t["rel16"] for t in X["taus"]),
                    cells=len(X["taus"]))
    print("M1P TAU", json.dumps(X["tau"]), flush=True)
    # ---- phase B: the evaluation windows
    for name, path in CQ.EVAL:
        run(model, CQ.text_ids(tok, path, a.start, a.seq, "cuda"), "check", name)
    rows = X["rows"]
    nl = cfg.num_hidden_layers
    rels = sorted(r["rel"] for r in rows)
    rele = sorted(r["rel_enc"] for r in rows)
    agg = dict(
        n=len(rows), full=len(rows) == nl * 3 * len(CQ.EVAL),
        finite=bool(rows) and all(r["finite"] for r in rows),
        self_check=all(r["self_check"] for r in rows),
        median_rel=rels[len(rels) // 2], max_rel=rels[-1], median_rel_enc=rele[len(rele) // 2], max_rel_enc=rele[-1],
        err_consistent=all(abs(r["err_ker"] - r["err_ref"]) <= 0.1 * r["err_ref"] + 1e-3
                           and abs(r["err_kenc"] - r["err_ref"]) <= 0.1 * r["err_ref"] + 1e-3 for r in rows),
        mismatch_max={f: max(r["mismatch"][f] for r in rows) for f in rows[0]["mismatch"]},
        **{k_: sum(r[k_] for r in rows) for k_ in ("c_mis", "c_bad32", "c_bad16", "v_mis", "v_bad32", "v_bad16")},
        cells_with_c_mis=sum(r["c_mis"] > 0 for r in rows), cells_with_v_mis=sum(r["v_mis"] > 0 for r in rows),
    )
    md5 = {Path(p).name: hashlib.md5(Path(p).read_bytes()).hexdigest()[:8] for p in
           (__file__, KA.__file__, CA.__file__, C.__file__, KA.KR.__file__, KA.K7.__file__, KA.K5.__file__,
            KA.K4.__file__, Q14.__file__, S12.__file__, S10.__file__, CQ.__file__)}
    md5.update({Path(p).name: hashlib.md5(Path(p).read_bytes()).hexdigest()[:8]
                for p in (CQ.CAL[0], *(p for _, p in CQ.EVAL))})
    res = dict(model=a.model, start=a.start, seq=a.seq, kb=KB, vb=VB, M=M_CAND, K=K_VAL, lam=LAM, prec=a.prec,
               tau=X["tau"], agg=agg, md5=md5, device=torch.cuda.get_device_name(0),
               script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
               gammas=list(S10.Q6.GAMMAS), taus=X["taus"], rows=rows)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("M1P DONE", json.dumps(agg), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
