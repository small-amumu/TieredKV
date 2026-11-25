# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""T4 lab: fixed-trace replay of NAB1's traces with cold-residual demotion applied in the forward pass.

Derived from nab2_lab.py / t1_lab.py: same attention hook, KIVI calibration pass, Q.calib / Q.calib_all, trace
handling, chunked logit head and next-token KL to the exact model. sel14_q._attend_h is replaced by t4_q.t4_attend_h
for every arm (with no policy it runs the original op sequence). Per item and arm: KL(exact || arm) by decode
position bucket, decode-span KL, answer-span KL and argmax agreement, mean next-token NLL of the trace token, and
t4_q's per-layer records (decided tokens per action, stage-2 changes against the undemoted selection, weights).

--smoke runs the controls on one item: (a) arm h54 through t4 equals the original function bit for bit, and so does
a policy that demotes nothing; (b) a policy that demotes every eligible token with action X equals the uniform
substitution reference for X (h32, key, del), bit for bit.

usage: python -I t4_lab.py <model_dir> <traces_jsonl> <out_json> --stack <dir holding hc_mac/> --runs a,b
       [--items 0,1,..] [--diag8] [--max_len 8800] [--chunk 256] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AttentionInterface, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("traces")
ap.add_argument("out")
ap.add_argument("--stack", required=True)
ap.add_argument("--runs", default="")
ap.add_argument("--items", default="")
ap.add_argument("--diag8", action="store_true", help="8 items at length octiles (sorted by trace length)")
ap.add_argument("--max_len", type=int, default=8800)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--smoke", action="store_true")
A = ap.parse_args()

ST = Path(A.stack).resolve()        # absolute: module __file__ paths must be absolute for the md5 guard
for sub in ("hc_mac/rootcause/vfb", "hc_mac/rootcause/vfl", "hc_mac/rootcause/spd", "hc_mac/rootcause/acs",
            "hc_mac/rootcause", "hc_mac/code"):
    sys.path.insert(0, str(ST / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v8g  # noqa: E402

v8g.install()
import bl_q as BLQ  # noqa: E402
import cq_lab3 as CQ  # noqa: E402
import sel14_q as Q14  # noqa: E402
import sel30_q as Q  # noqa: E402
import t4_q as T4  # noqa: E402

OURS = "kivi128k64_x2s128_j50_h54"
SEL14_MD5 = "bee1a8504bfc159b1b8bff1878d58932"
P = T4.Policy
# arm -> (routing name, policy, uniform reference, page statistics)
ARMS: dict[str, tuple] = {
    "h54": (OURS, None, None, True),
    "h43": ("kivi128k64_x2s128_j50_h43", None, None, False),
    "h33": ("kivi128k64_x2s128_j50_h33", None, None, False),
    "h32": ("kivi128k64_x2s128_j50_h32", None, None, False),          # 3/2-bit encoded from FP16 at write time
    # A2b (11-24): measured uniform points inside the 120-152 B gap (136 B per record: K 4 + V 4, or K 5 + V 3 bits)
    "h44": ("kivi128k64_x2s128_j50_h44", None, None, False),
    "h53": ("kivi128k64_x2s128_j50_h53", None, None, False),
    "h32c": (OURS, None, "h32", False),       # every record re-encoded from h54 (the h32 tier's codec, all tokens)
    "all_del": (OURS, P("oldest", 0, ((1.0, "del"),)), None, False),
    "sc_G512_del75": (OURS, P("score", 512, ((0.75, "del"),)), None, False),
    "sc_G1024_del75": (OURS, P("score", 1024, ((0.75, "del"),)), None, True),
    "sc_G2048_del75": (OURS, P("score", 2048, ((0.75, "del"),)), None, False),
    "sc_G1024_h32_75": (OURS, P("score", 1024, ((0.75, "h32"),)), None, True),
    "sc_G1024_key75": (OURS, P("score", 1024, ((0.75, "key"),)), None, False),
    "sc_G1024_2lvl": (OURS, P("score", 1024, ((0.5, "del"), (0.25, "h32"))), None, False),
    "rnd_G1024_del75": (OURS, P("random", 1024, ((0.75, "del"),)), None, False),
    "old_G1024_del75": (OURS, P("oldest", 1024, ((0.75, "del"),)), None, False),
    "rnd_G1024_h32_75": (OURS, P("random", 1024, ((0.75, "h32"),)), None, False),
}
# smoke-only control arms
TEST: dict[str, tuple] = {
    "t_zero": (OURS, P("score", 1024, ((0.001, "del"),)), None, False),        # int(0.001 * 128) = 0: no demotion
    "t_zero_h32": (OURS, P("score", 1024, ((0.001, "h32"),)), None, False),
    **{f"t_all_{a}": (OURS, P("oldest", 0, ((1.0, a),)), None, False) for a in T4.ACTIONS},
    **{f"t_ref_{a}": (OURS, None, a, False) for a in T4.ACTIONS},
}
CQ.CAL = (str(ST / "data" / "diag_long_text.txt"), CQ.CAL[1], CQ.CAL[2])
BUCKETS = ((0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 1 << 30))
X: dict = {"run": None, "cap": False, "kv": {}, "shape": None}


def sel_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    li = int(module.layer_idx)
    q, k, v = query[0].float(), key[0].float(), value[0].float()
    scale = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
    if X["cap"]:
        X["kv"][li] = (k.clone(), v.clone(), q.clone())
        X["shape"] = (q.shape[0], k.shape[0], q.shape[-1])
    if kw.get("softcap") or kw.get("sliding_window"):
        raise RuntimeError("T4: softcap / sliding window not supported")
    out = Q.attend(q, k, v, li, X["run"], scale, kw)
    return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None


AttentionInterface.register("sel_attn", sel_attn)


def md5_file(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main() -> None:
    if A.smoke != A.out.endswith(".smoke.json"):
        raise SystemExit("smoke output must end with .smoke.json (and only smoke)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = "cuda"
    Q.CHUNK["rows"], BLQ.BL["chunk"] = A.chunk, A.chunk
    Q.PREFILL["on"] = True
    if md5_file(Q14.__file__) != SEL14_MD5:
        raise SystemExit(f"sel14_q md5 {md5_file(Q14.__file__)} != {SEL14_MD5}")
    if T4.S4.CHUNK["rows"] != A.chunk:
        raise SystemExit(f"chunk setting does not reach sel14_q: {T4.S4.CHUNK['rows']} != {A.chunk}")
    runs = tuple(r for r in A.runs.split(",") if r) or (() if A.smoke else tuple(ARMS))
    unknown = [r for r in runs if r not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}")
    if len(set(runs)) != len(runs):
        raise SystemExit("duplicate arms")
    for r in runs:
        Q.parse(ARMS[r][0])
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.float32, attn_implementation="sel_attn",
                                                 device_map=dev).eval()
    re_ = model.model.rotary_emb
    BLQ.BL["inv_freq"], BLQ.BL["asc"] = re_.inv_freq.detach().float(), float(getattr(re_, "attention_scaling", 1.0))
    Q.RC.R["inv_freq"], Q.RC.R["asc"] = BLQ.BL["inv_freq"], BLQ.BL["asc"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.model)
    traces = [json.loads(line) for line in Path(A.traces).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids_seen = [t["i"] for t in traces]
    if len(set(ids_seen)) != len(ids_seen):
        raise SystemExit("duplicate trace ids")
    if A.items and A.diag8:
        raise SystemExit("--items and --diag8 are exclusive")
    if A.items:
        want = [int(x) for x in A.items.split(",")]
        by = {t["i"]: t for t in traces}
        missing = [i for i in want if i not in by]
        if missing:
            raise SystemExit(f"--items not in {A.traces}: {missing}")
        traces = [by[i] for i in want]
    elif A.diag8:
        srt = sorted(traces, key=lambda t: (min(len(t["prompt_ids"]) + len(t["gen_ids"]), A.max_len), t["i"]))
        if len(srt) < 8:
            raise SystemExit("--diag8 needs at least 8 traces")
        traces = [srt[round(j * (len(srt) - 1) / 7)] for j in range(8)]
    elif A.smoke:
        traces = [max(traces, key=lambda t: (min(len(t["prompt_ids"]) + len(t["gen_ids"]), A.max_len), -t["i"]))]

    def forward(ids):
        with torch.no_grad():
            return model.model(input_ids=ids, use_cache=False).last_hidden_state[0]

    def head(h):
        return model.lm_head(h).float()

    @torch.no_grad()
    def readouts(z0, z, ids, p_len, a0, L):
        """KL(exact || arm) per decode bucket, decode-span KL and NLL, answer-span KL and argmax agreement."""
        sums = {b: [0.0, 0] for b in BUCKETS}
        ans = [0.0, 0, 0]
        dec = [0.0, 0, 0.0, 0.0, 0]                          # KL sum, KL rows, NLL arm sum, NLL exact sum, NLL rows
        for r0 in range(p_len, L, 512):
            r1 = min(L, r0 + 512)
            l0, l1 = torch.log_softmax(head(z0[r0:r1]), -1), torch.log_softmax(head(z[r0:r1]), -1)
            klr = (l0.exp() * (l0 - l1)).sum(-1).tolist()
            agree = (l0.argmax(-1) == l1.argmax(-1)).tolist()
            nxt = ids[0, r0 + 1:r1 + 1]
            nr = nxt.shape[0]                                # the last row of the trace has no next token
            nll1 = (-l1[:nr].gather(1, nxt[:, None])[:, 0]).tolist() + [None] * (r1 - r0 - nr)
            nll0 = (-l0[:nr].gather(1, nxt[:, None])[:, 0]).tolist() + [None] * (r1 - r0 - nr)
            for j in range(r1 - r0):
                row = r0 + j
                kv_ = klr[j]
                if a0 is not None and row >= a0 - 1:
                    ans[0] += kv_
                    ans[1] += 1
                    ans[2] += int(agree[j])
                    continue
                pos = row - p_len
                for b in BUCKETS:
                    if b[0] <= pos < b[1]:
                        sums[b][0] += kv_
                        sums[b][1] += 1
                dec[0] += kv_
                dec[1] += 1
                if nll1[j] is not None:                       # the trace's last row has no next token
                    dec[2] += nll1[j]
                    dec[3] += nll0[j]
                    dec[4] += 1
        return ({f"{b[0]}-{b[1]}": (s / n if n else None, n) for b, (s, n) in sums.items()},
                {"kl": dec[0] / dec[1] if dec[1] else None, "rows": dec[1], "nll_rows": dec[4],
                 "nll": dec[2] / dec[4] if dec[4] else None, "nll_exact": dec[3] / dec[4] if dec[4] else None},
                (ans[0] / ans[1] if ans[1] else None, ans[1], ans[2] / ans[1] if ans[1] else None))

    X["cap"] = True
    try:
        with torch.no_grad():
            cal = CQ.text_ids(tok, CQ.CAL[0], CQ.CAL[1], CQ.CAL[2], dev)
            if cal.shape[1] != CQ.CAL[2]:
                raise SystemExit(f"calibration text: {cal.shape[1]} of {CQ.CAL[2]} tokens")
            model(input_ids=cal, use_cache=False)
    finally:
        X["cap"] = False
    for li, t in X["kv"].items():
        Q.calib(li, *t)
    alist = sorted(X["kv"])
    nope = Q.calib_all(model, X["kv"], alist)
    X["kv"] = {}
    H_, KVH, d = X["shape"]
    md5 = {Path(p).name: md5_file(p) for p in (__file__, T4.__file__, Q14.__file__, Q.__file__, v8g.__file__,
                                                BLQ.__file__, CQ.__file__, A.traces, CQ.CAL[0])}
    st_root = ST.resolve()

    def cur_mods() -> dict:
        out_ = {}
        for nm, mo in sorted(sys.modules.items()):
            f = getattr(mo, "__file__", None)
            if not isinstance(f, str) or not f or not os.path.isabs(f):
                continue
            rp = Path(f).resolve()
            if rp.is_relative_to(st_root):
                out_[nm] = md5_file(rp)
        return out_
    mods = cur_mods()
    need = {"sel14_q", "sel30_q", "bl_q", "cq_lab3", "v8g"}
    if not need <= set(mods):
        raise SystemExit(f"stack module check sees {sorted(mods)}; missing {sorted(need - set(mods))}")
    arm_desc = {r: {"route": ARMS[r][0], "policy": None if ARMS[r][1] is None else
                    {"kind": ARMS[r][1].kind, "g": ARMS[r][1].g, "tiers": [list(t) for t in ARMS[r][1].tiers]},
                    "ref": ARMS[r][2], "pages": ARMS[r][3]} for r in runs}
    res = {"model": A.model, "traces": A.traces, "runs": list(runs), "arms": arm_desc, "chunk": A.chunk,
           "max_len": A.max_len, "heads": H_, "kv_heads": KVH, "head_dim": d, "attn_layers": alist,
           "nope_layers": nope, "md5": md5, "smoke": A.smoke, "buckets": [list(b) for b in BUCKETS],
           "t4_buckets": [list(b) for b in T4.BUCKETS], "pages": list(T4.PAGES), "stack_modules_md5": mods,
           "script_md5": md5[Path(__file__).name],
           "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}, "items": []}
    out = Path(A.out)
    done = {}
    if out.exists() and not A.smoke:
        old = json.loads(out.read_text(encoding="utf-8"))
        if any(old.get(k) != res[k] for k in ("model", "traces", "runs", "arms", "chunk", "max_len", "md5", "versions",
                                              "script_md5", "stack_modules_md5", "buckets", "t4_buckets", "pages")):
            raise SystemExit(f"{out} was written by another configuration; refusing to resume")
        done = {it["i"]: it for it in old["items"]}
    res["items"] = list(done.values())

    def save() -> None:
        if cur_mods() != mods:
            raise SystemExit("stack modules changed since the start of the run")
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(res))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(out)

    think_id = tok.convert_tokens_to_ids("</think>")
    ORIG = Q14._attend_h

    def run_arm(arm: str, spec: tuple, ids, tr, p_len) -> torch.Tensor:
        route, pol, ref, pages = spec
        T4.reset_item(tr["i"], p_len, pol, ref, pages)
        X["run"] = route
        try:
            z = forward(ids)
        finally:
            X["run"] = None
        bad = {li: T4.CFG.calls.get(li, 0) for li in alist if T4.CFG.calls.get(li, 0) != 1}
        if bad:
            raise SystemExit(f"{arm}: t4_attend_h not called exactly once per layer: {bad}")
        return z

    Q14._attend_h = T4.t4_attend_h
    t0 = time.time()
    for tr in traces:
        if tr["i"] in done:
            continue
        ids_l = tr["prompt_ids"] + tr["gen_ids"]
        L = min(len(ids_l), A.max_len)
        p_len = len(tr["prompt_ids"])
        if p_len >= L:
            res["items"].append({"i": tr["i"], "L": L, "prompt": p_len, "skipped": "no decode rows"})
            save()
            continue
        a0 = tr["think_end"] if tr["think_end"] is not None and tr["think_end"] < L else None
        if a0 is not None and ids_l[a0 - 1] != think_id:
            raise SystemExit(f"item {tr['i']}: token at think_end - 1 is {ids_l[a0 - 1]}, not </think> {think_id}")
        ids = torch.tensor([ids_l[:L]], device=dev)
        Q.PREFILL["from"] = p_len
        rec = {"i": tr["i"], "L": L, "prompt": p_len, "think_end": a0, "ok16": tr.get("ok"),
               "truncated_by_max_len": len(ids_l) > A.max_len, "bucket": {}, "decode": {}, "answer": {},
               "sec": {}, "layers": {}}
        torch.cuda.reset_peak_memory_stats()
        if A.smoke:
            Q14._attend_h = ORIG
            X["run"] = OURS
            try:
                h_orig = forward(ids)
            finally:
                X["run"] = None
                Q14._attend_h = T4.t4_attend_h
            ctl, bitwise = {}, {}

            def same(a_: torch.Tensor, b_: torch.Tensor, key: str) -> bool:
                # value equality gates (+0.0 == -0.0: adding an all-zero term may flip the sign of a zero);
                # the raw-bit comparison is recorded alongside
                bitwise[key] = bool(torch.equal(a_.view(torch.int32), b_.view(torch.int32)))
                return bool(torch.equal(a_, b_))
            for arm in ("t_zero", "t_zero_h32"):
                ctl[arm] = same(h_orig, run_arm(arm, TEST[arm], ids, tr, p_len), arm)
            ctl["h54"] = same(h_orig, run_arm("h54", ARMS["h54"], ids, tr, p_len), "h54")
            Q14._attend_h = ORIG                                 # the copy on a uniform route (h32) as well
            X["run"] = ARMS["h32"][0]
            try:
                h32_orig = forward(ids)
            finally:
                X["run"] = None
                Q14._attend_h = T4.t4_attend_h
            ctl["h32_route"] = same(h32_orig, run_arm("h32", ARMS["h32"], ids, tr, p_len), "h32_route")
            del h32_orig
            for a in T4.ACTIONS:
                za = run_arm(f"t_all_{a}", TEST[f"t_all_{a}"], ids, tr, p_len)
                zr = run_arm(f"t_ref_{a}", TEST[f"t_ref_{a}"], ids, tr, p_len)
                ctl[f"all_{a}=ref"] = same(za, zr, f"all_{a}=ref")
                ctl[f"all_{a}!=h54"] = not bool(torch.equal(za, h_orig))
                del za, zr
            # (c) a partial dynamic h32 policy must make one forward read both pools: stage-1 rows with h32 keys
            # and with h54 keys, stage-2 rows with h32 values and with h54 values (records summed over layers)
            run_arm("sc_G1024_h32_75", ARMS["sc_G1024_h32_75"], ids, tr, p_len)
            tot = sum(T4.STATS[li]["rec"].sum(0) for li in alist)

            def view_hits(v: str) -> float:          # rows read from one pool (page size 8 histogram, all buckets)
                h = sum(T4.STATS[li]["hist"][8][T4.VIEWS.index(v)].sum(0) for li in alist)
                return float((h * torch.arange(9, device=h.device)).sum())
            mix = {"k32_rows": float(tot[11]), "v32_rows": float(tot[12]), "decided_h32": float(tot[2]),
                   **{f"{v}_rows": view_hits(v) for v in ("K54", "K32", "V54", "V32")}}
            rec["mixed_h32"] = mix
            ctl["mixed_h32_both_pools"] = (mix["K54_rows"] > 0 and mix["K32_rows"] > 0 and mix["V54_rows"] > 0
                                           and mix["V32_rows"] > 0 and mix["K32_rows"] == mix["k32_rows"]
                                           and mix["V32_rows"] == mix["v32_rows"]
                                           and float(tot[13]) == 0.0 and float(tot[14]) == 0.0)
            rec["controls"], rec["controls_rawbits"] = ctl, bitwise
            del h_orig
            if not all(ctl.values()):
                res["items"].append(rec)
                save()
                raise SystemExit(f"smoke controls FAILED: {ctl}")
        X["run"] = None
        z0 = forward(ids)
        for arm in runs:
            t_run = time.time()
            z = run_arm(arm, ARMS[arm], ids, tr, p_len)
            torch.cuda.synchronize()
            rec["sec"][arm] = round(time.time() - t_run, 1)
            rec["bucket"][arm], rec["decode"][arm], rec["answer"][arm] = readouts(z0, z, ids, p_len, a0, L)
            if T4.STATE or T4.STATS:
                missing = [li for li in alist if li not in T4.STATS]
                if missing:
                    raise SystemExit(f"{arm}: no t4 records for layers {missing}")
                rec["layers"][arm] = {str(li): T4.stats_to_json(li) for li in alist}
            del z
        rec["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        rec["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
        res["items"].append(rec)
        save()
        print(f"item {tr['i']} L={L} ({time.time() - t0:.0f}s) "
              + " ".join(f"{r} {rec['decode'][r]['kl']}" for r in runs), flush=True)
    res["seconds"] = time.time() - t0
    save()
    print("DONE t4_lab", A.model, len(res["items"]), flush=True)


if __name__ == "__main__":
    main()
