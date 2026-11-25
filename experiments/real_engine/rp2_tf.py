# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""rp2_tf (TieredKV, round R2; 2026-11-10): the real decode engine E1 against the accuracy reference and fp on the Table B
protocol (X-0080: tb_bench v3.7b with TB_TF=1 TB_GOLD=1) for Llama-3.2-3B-Instruct, RULER 6 tasks @32768.

One process per task file; one model (the engine's HF model), calibrated once by rp_engine.Engine (the reference's static
calibration). Per item, five arms:
- fp: the HF forward with exact SDPA (tb_bench's fp path) on a DynamicCache. Greedy from the prefill argmax until EOS or the
  task's maxgen gives the fp trajectory T = [t_0 .. t_{n-1}] (EOS not kept). The rows computed while decoding are the fp rows
  of the trajectory pass: row j (after feeding t_j) predicts t_{j+1}, j = 0 .. n-2 (tb_bench: ns_ = len(ref_toks) - 1; the same
  inputs as its separate forced pass). The gold answer G = tokenize(tb_bench v3.7b gold_text: " " + " ".join(x.strip() for x in
  outputs)) on its own (no special tokens) is
  forced from a crop of the same prefill cache: rows j = 0 .. |G| - 2 predict G[j + 1] (the first token comes from the shared
  prefill row and is left out, as tb_bench v3.7).
- ref33 / ref54: the same HF forward, decode attention replaced by cap_attn.attend over a cap_attn.IncStore grown from the cache
  (rp0_tf's reference branch = ours_q 92ef53ff, the accuracy path of stkm8v8128k64_x2s128_p50_h<w>), forced over T and over G,
  each pass from a crop of the prefill cache with a fresh store and the item's own stack info (as rp0_tf, per arm and pass).
- eng33 / eng54: the real engine (rp_engine.Engine.prefill builds its store, Engine.step decodes), forced over T and over G (one
  engine prefill per sequence).
Per sequence s in {tf (= T), gold (= G)} and arm a: s_kl_mean = mean_j KL(fp_j || a_j), s_kl_max, s_top1 = mean_j [argmax a_j ==
target_j], s_nll = mean_j -log a_j(target_j); fp: s_nll and s_mismatch (fp argmax != target, 0 on tf by construction). Engine arms
also: s_kl_ref = mean_j KL(ref_w,j || eng_w,j), s_agree_ref = mean_j [argmax eng == argmax ref]. Per-step KL lists are kept for
analysis. One JSON line per item; no verdict here (r2_verdict.py).
An engine EngineFlag or an exception in an arm is recorded in that arm's status; the item's other arms still run.
Usage: python -B -I rp2_tf.py <model_dir> <out.jsonl> --data <task>/validation.jsonl --items 0:10 [--cap 140000]
Env: as rp_run.py (KVT_ROOT, KALIGN_DIR, KALIGN_CAL)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rp_engine as E  # noqa: E402
import rp_run as RR  # noqa: E402  (helpers only: eos_set, md5s, write)

from transformers import AttentionInterface, DynamicCache  # noqa: E402

WIDTHS = ("33", "54")
GROW_CHUNK = 4096                                   # rows per reference encode call (memory only; multiple of 128)
MAXGEN = {"vt": 30, "cwe": 120, "fwe": 50, "qa_1": 32}  # tb_bench v3.7 RULER_MAXGEN; niah tasks: 128
X: dict = {"arm": "fp", "w": None, "st": {}, "info": set(), "calls": 0, "flags": 0}


def stack_info(kk, vv, li):
    """the item's stack info for the reference (rp0_tf.stack_info: sel10_q.stack_base in the m8v8 format)."""
    E.S12.MF["fmt"] = "m8v8"
    E.S10.ST["on"], E.S10.ST["li"] = True, li
    try:
        E.S10.stack_base(kk, vv, li)
    finally:
        E.S12.MF["fmt"] = None
        E.S10.ST["on"] = False


def rp2_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    """attention of the HF forward after calibration. fp arm and every prompt pass: SDPA as tb_bench (the engine's own
    non-capture branch). ref arms, decode (q_len 1): rp0_tf's reference branch."""
    li = int(module.layer_idx)
    if X["arm"] == "fp" or query.shape[2] != 1:
        return E._sdpa_fwd(module, query, key, value, attention_mask, scaling=scaling, dropout=0.0,
                           is_causal=query.shape[2] > 1)
    scale = scaling if scaling is not None else E.D ** -0.5
    assert abs(scale - E.D ** -0.5) < 1e-9 and not kw.get("softcap") and not kw.get("sliding_window")
    kb, vb = int(X["w"][0]), int(X["w"][1])
    q, k, v = query[0].float(), key[0].float(), value[0].float()
    KVH, T, d = k.shape
    W0 = T // E.RW * E.RW
    assert W0 - E.Z >= E.M_CAND, f"context too short for M = {E.M_CAND} (T = {T})"
    if li not in X["info"]:
        stack_info(k, v, li)
        X["info"].add(li)
    st = X["st"].get(li)
    if st is None:
        st = X["st"][li] = E.CA.IncStore(KVH, d, k.device, kb, vb)
    u0, z0 = E.CA.ENC["unexpressible"], E.CA.ENC.get("res_zero", 0)
    for w_mid in range(st.W0 + GROW_CHUNK, W0, GROW_CHUNK):     # per-token codes: any split gives the same bytes
        st.grow(k, v, w_mid, li, E.S10, E.Q5, E.S10.Q6, E.P9, E.P10, E.P11, E.Q7)
    st.grow(k, v, W0, li, E.S10, E.Q5, E.S10.Q6, E.P9, E.P10, E.P11, E.Q7)
    X["flags"] += E.CA.ENC["unexpressible"] - u0 + E.CA.ENC.get("res_zero", 0) - z0
    st.set_exact(k, v, T)
    out = E.CA.attend(q, st, li, scale, E.M_CAND, E.K_FETCH, E.LAM, E.S10, E.Q5, E.P9, E.P10, E.P11, E.Q7)[:, 0]
    X["calls"] += 1
    return out.view(1, q.shape[0], 1, d).to(query.dtype).transpose(1, 2).contiguous(), None


def crop_to(cache, L0):
    n = cache.get_seq_length() - L0
    assert n >= 0, (n, L0)
    if n:
        cache.crop(-n)                                   # transformers >= 5.20: negative count of rows to remove
    assert cache.get_seq_length() == L0


def fingerprint(cache, nl):
    return [(float(cache.layers[j].keys.float().sum()), float(cache.layers[j].values.float().sum())) for j in range(nl)]


@torch.no_grad()
def hf_forced(model, cache, seq, L0):
    """rows j = 0 .. len(seq) - 2: log-probs after feeding seq[j] (CPU fp32)."""
    rows = []
    for j in range(len(seq) - 1):
        o = model(input_ids=torch.tensor([[seq[j]]], device="cuda"), past_key_values=cache, use_cache=True)
        rows.append(torch.log_softmax(o.logits[0, -1].float(), -1).cpu())
        assert cache.get_seq_length() == L0 + j + 1
        o = None
    return rows


@torch.no_grad()
def eng_forced(eng, ids, seq):
    eng.prefill(ids, len(seq) + 4)
    rows = [torch.log_softmax(eng.step(seq[j]), -1).cpu() for j in range(len(seq) - 1)]
    mem = {k: v for k, v in eng.memory().items() if k in ("fp16_inexact", "guard", "gpu_store_gib", "host_store_gib")}
    return rows, mem


def kl(p, q):
    return float((p.exp() * (p - q)).sum())


def metrics(name, F, A, seq, R=None):
    tg = seq[1:]
    assert len(F) == len(A) == len(tg) and (R is None or len(R) == len(A)), (name, len(F), len(A), len(tg))
    kls = [kl(f, a) for f, a in zip(F, A)]
    out = {f"{name}_steps": len(A), f"{name}_kl_mean": sum(kls) / len(kls), f"{name}_kl_max": max(kls),
           f"{name}_top1": sum(int(a.argmax()) == t for a, t in zip(A, tg)) / len(A),
           f"{name}_nll": sum(float(-a[t]) for a, t in zip(A, tg)) / len(A), f"{name}_kl_steps": kls,
           f"{name}_finite": all(math.isfinite(x) for x in kls) and all(bool(torch.isfinite(a).all()) for a in A)}
    if R is not None:
        krs = [kl(r, a) for r, a in zip(R, A)]
        out.update({f"{name}_kl_ref": sum(krs) / len(krs), f"{name}_kl_ref_max": max(krs),
                    f"{name}_agree_ref": sum(int(a.argmax()) == int(r.argmax()) for a, r in zip(A, R)) / len(A)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--data", required=True, help="one RULER task file .../<task>/validation.jsonl")
    ap.add_argument("--items", default="0:10")
    ap.add_argument("--cap", type=int, default=140000)
    ap.add_argument("--prec", default="f16x2")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: use a new output file")
    if os.environ.get("PKV5_ROWS"):
        raise SystemExit("PKV5_ROWS must be unset (it changes pkv5_q)")
    m5 = RR.md5s()
    m5["engine/rp2_tf.py"] = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    exp = json.loads((Path(os.path.dirname(os.path.abspath(__file__))) / "EXPECTED_MD5.json").read_text())
    bad = {k: (m5.get(k), v) for k, v in exp.items() if m5.get(k) != v}
    if bad:
        raise SystemExit(f"reference / kernel modules differ from the reviewed versions: {bad}")
    assert not E.MUT, "mutation flags must be empty"
    task = Path(a.data).parent.name
    mx = MAXGEN.get(task, 128)
    data_md5 = hashlib.md5(Path(a.data).read_bytes()).hexdigest()
    rows = [json.loads(l_) for l_ in open(a.data, encoding="utf-8")]
    i0, i1 = (int(x) for x in a.items.split(":"))
    assert len(rows) >= i1, (a.data, len(rows))
    eng = E.Engine(a.model, 3, 3, a.cap, a.prec)                 # calibrates with the capture branch ("rp_cap")
    AttentionInterface.register("rp_cap", rp2_attn)              # after calibration: the fp / ref arms' attention
    eos = RR.eos_set(eng)
    nl = eng.NL
    RR.write(a.out, dict(meta=True, model=a.model, data=a.data, data_md5=data_md5, task=task, items=a.items, maxgen=mx,
                         md5=m5, device=torch.cuda.get_device_name(0), torch=torch.__version__,
                         transformers=__import__("transformers").__version__, calibration=eng.cal_info, eos=sorted(eos),
                         cap=a.cap, prec=a.prec))
    tok = eng.tok
    n_err = 0
    for j_item in range(i0, i1):
        r = rows[j_item]
        t0 = time.time()
        rec = dict(task=task, i=j_item, status="ok")
        torch.cuda.reset_peak_memory_stats()
        try:
            ids = tok(r["input"] + r.get("answer_prefix", ""), return_tensors="pt").input_ids.cuda()
            G = tok(" " + " ".join(x.strip() for x in r["outputs"]), add_special_tokens=False).input_ids   # tb_bench gold_text
            L0 = ids.shape[1]
            rec.update(L0=L0, L0_mod128=L0 % 128, gold_len=len(G))
            assert L0 + mx <= int(Path(a.data).parent.parent.name) + 1, "prompt exceeds the length budget (tb_bench rule)"
            # ---- fp: prefill, trajectory, gold
            X.update(arm="fp", w=None, st={}, info=set(), calls=0, flags=0)
            cache = DynamicCache()
            with torch.no_grad():
                o0 = eng.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            first = int(o0.logits[0, -1].argmax())
            o0 = None
            fp0 = fingerprint(cache, nl)
            T, F_T, nxt = [], [], first
            with torch.no_grad():
                for step in range(mx):
                    if nxt in eos:
                        break
                    T.append(nxt)
                    if step == mx - 1:
                        break
                    o = eng.model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
                    lp = torch.log_softmax(o.logits[0, -1].float(), -1)
                    o = None
                    F_T.append(lp.cpu())
                    nxt = int(lp.argmax())
            F_T = F_T[:max(len(T) - 1, 0)]
            seqs = {}
            if len(T) >= 2:
                seqs["tf"] = (T, F_T)
            rec["tf_status"] = "ok" if "tf" in seqs else f"fp trajectory of {len(T)} tokens"
            rec["tf_len"] = len(T)
            if 2 <= len(G) <= mx:
                crop_to(cache, L0)
                assert fingerprint(cache, nl) == fp0, "prefill cache changed (fp gold)"
                seqs["gold"] = (G, hf_forced(eng.model, cache, G, L0))
                rec["gold_status"] = "ok"
            else:
                rec["gold_status"] = f"gold of {len(G)} tokens outside [2, {mx}]"
            rec["fp"] = {}
            for name, (S, F) in seqs.items():
                rec["fp"].update({f"{name}_steps": len(F), f"{name}_nll": sum(float(-f[t]) for f, t in zip(F, S[1:])) / len(F),
                                  f"{name}_mismatch": sum(int(f.argmax()) != t for f, t in zip(F, S[1:]))})
            ref_rows = {}
            # ---- reference arms (HF forward, cap_attn decode attention), one fresh store per pass
            for w in WIDTHS:
                arm = f"ref{w}"
                rec[arm] = {"status": "ok"}
                try:
                    for name, (S, F) in seqs.items():
                        crop_to(cache, L0)
                        assert fingerprint(cache, nl) == fp0, f"prefill cache changed ({arm} {name})"
                        E.S10.INFO.clear()
                        X.update(arm=arm, w=w, st={}, info=set(), calls=0, flags=0)
                        A = hf_forced(eng.model, cache, S, L0)
                        assert X["calls"] == nl * (len(S) - 1), (arm, name, X["calls"], nl, len(S))
                        rec[arm].update(metrics(name, F, A, S))
                        rec[arm][f"{name}_flags"] = X["flags"]
                        ref_rows[(w, name)] = A
                finally:
                    X.update(arm="fp", w=None, st={}, info=set())
            crop_to(cache, L0)
            assert fingerprint(cache, nl) == fp0, "prefill cache changed (after the reference arms)"
            cache = None
            torch.cuda.empty_cache()
            # ---- engine arms
            for w in WIDTHS:
                arm = f"eng{w}"
                rec[arm] = {"status": "ok"}
                eng.kb, eng.vb = int(w[0]), int(w[1])
                try:
                    for name, (S, F) in seqs.items():
                        A, mem = eng_forced(eng, ids, S)
                        rec[arm].update(metrics(name, F, A, S, ref_rows.get((w, name))))
                        rec[arm][f"{name}_mem"] = mem
                        eng.layers = None
                        torch.cuda.empty_cache()
                except E.EngineFlag as e:
                    rec[arm]["status"] = f"flag: {str(e)[:300]}"
                except Exception:  # noqa: BLE001  (recorded; the verdict counts it)
                    rec[arm]["status"] = "error: " + traceback.format_exc()[-1500:]
                finally:
                    eng.layers = None
                    torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001  (item-level failure: recorded, the run continues)
            n_err += 1
            rec["status"] = "error: " + traceback.format_exc()[-2000:]
            X.update(arm="fp", w=None, st={}, info=set())
            eng.layers = None
            torch.cuda.empty_cache()
        E.S10.INFO.clear()                                # no per-item reference state survives into the next item
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        rec["gpu_peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        rec["sec"] = time.time() - t0
        RR.write(a.out, rec)
        print(json.dumps({k: rec.get(k) for k in ("task", "i", "status", "L0", "tf_len", "gold_len", "gpu_peak_gib", "sec")}), flush=True)
    RR.write(a.out, dict(meta=True, end=True, n_errors=n_err))
    print("RP2_TF_DONE" if not n_err else f"RP2_TF_ERRORS {n_err}", flush=True)
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
