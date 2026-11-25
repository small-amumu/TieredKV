# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""T4 A3: free generation on held-out MATH-500 problems with cold-residual demotion inside the decode loop.

The model, attention hook, KIVI calibration and stack checks are t4_lab.py's. Decoding is incremental: the prompt is
one prefill call, then one row per call against the cached keys and values (HF DynamicCache, float32), with
t4_q's demotion state sized for the whole sequence (t4_q.reset_item(cap=...)). Sampling follows the trace generator
(nab1_gen.py): Qwen3 chat template with thinking on, the CH4X suffix, temperature 0.6, top-k 20, top-p 0.95, at most
--max_new tokens; one uniform draw per step from a CPU generator seeded by the problem index, so every arm of a
problem sees the same random stream (paired). The answer is scored with the CH4X boxed-answer scorer (copied
unchanged from nab1_gen.py).

--check runs the control instead of generating: one replay trace is fed token by token through the incremental path
(teacher forcing) and compared with the full-sequence forward t4_lab.py uses, per arm: KL(full || incremental) per
row, argmax agreement, max absolute logit difference, and (for demotion arms) whether every layer's demotion
decisions are identical. Output: <out>.check.json.

usage: python -I t4_gen.py <model_dir> <out_jsonl> --stack <dir> --data <math500 test.jsonl> --arms a,b
       [--start 64] [--n 32] [--max_new 8192]
       python -I t4_gen.py <model_dir> <out_jsonl> --stack <dir> --check <traces_jsonl> --arms a,b [--check_len 3000]
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer, DynamicCache

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--stack", required=True)
ap.add_argument("--arms", required=True)
ap.add_argument("--data", default="")
ap.add_argument("--start", type=int, default=64)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--max_new", type=int, default=8192)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--check", default="")
ap.add_argument("--check_len", type=int, default=3000)
ap.add_argument("--exclude_traces", default="", help="replay traces whose items the problems must avoid")
ap.add_argument("--perturb", type=float, default=0.0,
                help="check only: also run each arm's full forward with relative noise of this size on the embeddings "
                     "(the arm's own sensitivity to float32-sized input changes, to compare with the incremental gap)")
A = ap.parse_args()

ST = Path(A.stack).resolve()
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
# same definitions as t4_lab.py's ARMS; "exact" is the model's own attention (no compression)
ARMS: dict[str, tuple] = {
    "exact": (None, None),
    "h54": (OURS, None),
    "h43": ("kivi128k64_x2s128_j50_h43", None),
    "h33": ("kivi128k64_x2s128_j50_h33", None),
    "h32": ("kivi128k64_x2s128_j50_h32", None),
    "sc_G1024_h32_75": (OURS, P("score", 1024, ((0.75, "h32"),))),
    "rnd_G1024_h32_75": (OURS, P("random", 1024, ((0.75, "h32"),))),
    "sc_G1024_2lvl": (OURS, P("score", 1024, ((0.5, "del"), (0.25, "h32")))),
    "sc_G1024_del75": (OURS, P("score", 1024, ((0.75, "del"),))),
}
CQ.CAL = (str(ST / "data" / "diag_long_text.txt"), CQ.CAL[1], CQ.CAL[2])
X: dict = {"run": None, "cap": False, "kv": {}, "shape": None, "inc": False, "sdpa": False}
DATA_MD5 = "d06a3ab6a17029154ffbb29db65c158b"
# --check gates (decided before the check runs): incremental vs full forward of the same arm
GATE = {"argmax_agree_min": 0.999, "kl_mean_max": 1e-6, "decision_agree_min": 0.999, "sdpa_err_max": 1e-4}
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
TEMP, TOP_K, TOP_P = 0.6, 20, 0.95


def last_boxed(s: str) -> str | None:                       # CH4X lc_reason_range2.py, unchanged
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(s) and d:
        d += {"{": 1, "}": -1}.get(s[j], 0); j += 1
    return s[i + 7: j - 1] if d == 0 else None


def norm(x: str) -> str:                                     # CH4X lc_reason_range2.py, unchanged
    x = x.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac")
    x = re.sub(r"\\text\{([^}]*)\}", r"\1", x).replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("$", "")
    x = re.sub(r"\s+", "", x).rstrip(".")
    try:
        return str(float(x.replace(",", ""))).rstrip("0").rstrip(".")
    except ValueError:
        return x


def sel_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    li = int(module.layer_idx)
    q, k, v = query[0].float(), key[0].float(), value[0].float()
    scale = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
    if X["cap"]:
        X["kv"][li] = (k.clone(), v.clone(), q.clone())
        X["shape"] = (q.shape[0], k.shape[0], q.shape[-1])
    if kw.get("softcap") or kw.get("sliding_window"):
        raise RuntimeError("T4: softcap / sliding window not supported")
    if X.get("inc") and X["run"] is not None:
        # sel4_q.attend assumes full teacher forcing (row index = position; it refuses Lq <= PREFILL["from"]).
        # Incremental calls bypass its dense-prefill split: the t4 path runs on every row (prompt rows advance the
        # demotion schedule, as in the full forward), and the prefill call (Lq == Lk: every row a prompt row) then
        # takes sel4_q's own dense rows (bl_q quest_b16 with from = Lq, exact below from).
        lq_, lk_ = q.shape[1], k.shape[1]
        if lq_ == lk_ and lq_ != Q.PREFILL["from"]:
            raise RuntimeError(f"prefill call of {lq_} rows but PREFILL from {Q.PREFILL['from']}")
        if lq_ != lk_ and lq_ != 1:
            raise RuntimeError(f"incremental call with {lq_} query rows")
        prev_on = T4.S4.PREFILL["on"]
        T4.S4.PREFILL["on"] = False
        try:
            out = Q.attend(q, k, v, li, X["run"], scale, kw)
        finally:
            T4.S4.PREFILL["on"] = prev_on
        if lq_ == lk_:                                # equivalence up to float64 rounding: see --check
            BLQ.BL["from"] = lq_
            out = BLQ.attend(q, k, v, li, "quest_b16", scale, kw)
    else:
        out = Q.attend(q, k, v, li, X["run"], scale, kw)
    if X.get("sdpa") and X["run"] is None:           # exact route vs plain causal attention
        g_ = q.shape[0] // k.shape[0]
        lq, lk = q.shape[1], k.shape[1]
        sc_ = torch.einsum("hqd,hkd->hqk", q, k.repeat_interleave(g_, 0)) * scale
        cm = (torch.arange(lk, device=q.device)[None, :]
              > (torch.arange(lq, device=q.device)[:, None] + lk - lq))
        ref = torch.einsum("hqk,hkd->hqd", torch.softmax(sc_.masked_fill(cm[None], float("-inf")), -1),
                           v.repeat_interleave(g_, 0))
        X["sdpa_err"] = max(X.get("sdpa_err", 0.0), float((ref - out).abs().max()))
        X["sdpa_n"] = X.get("sdpa_n", 0) + 1
    return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None


AttentionInterface.register("sel_attn", sel_attn)


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def sample(logits: torch.Tensor, u: float) -> int:
    """Temperature, top-k, top-p, then the inverse CDF at u (one uniform per step: paired across arms)."""
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("non-finite logits")
    lp = torch.log_softmax(logits.double() / TEMP, -1)
    top = torch.topk(lp, TOP_K)
    pr = torch.softmax(top.values, -1)
    cdf = torch.cumsum(pr, -1)
    keep = int((cdf < TOP_P).sum()) + 1                  # smallest prefix with mass >= top_p
    pr = pr[:keep] / pr[:keep].sum()
    cdf = torch.cumsum(pr, -1)
    j = min(int((cdf <= u).sum()), keep - 1)
    return int(top.indices[j])


def lab_arms() -> dict:
    """t4_lab.py's ARMS literal (route, policy) evaluated with the same names, for the parity check."""
    src = (Path(__file__).resolve().parent / "t4_lab.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        tgt = None
        if isinstance(node, ast.AnnAssign):
            tgt = node.target
        elif isinstance(node, ast.Assign):
            tgt = node.targets[0]
        if isinstance(tgt, ast.Name) and tgt.id == "ARMS":
            d = eval(ast.get_source_segment(src, node.value), {"OURS": OURS, "P": P, "T4": T4})  # noqa: S307
            return {k: (v[0], v[1]) for k, v in d.items()}
    raise SystemExit("t4_lab.py has no ARMS table")


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = "cuda"
    arms = [a for a in A.arms.split(",") if a]
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise SystemExit(f"bad --arms {A.arms}")
    if bool(A.check) == bool(A.data):
        raise SystemExit("give exactly one of --data (generation) and --check (control)")
    Q.CHUNK["rows"], BLQ.BL["chunk"] = A.chunk, A.chunk
    Q.PREFILL["on"] = True
    if Q.PREFILL is not T4.S4.PREFILL:
        raise SystemExit("sel30_q.PREFILL is not sel4_q.PREFILL: the incremental bypass would not apply")
    if T4.S4.CHUNK["rows"] != A.chunk:
        raise SystemExit(f"chunk setting does not reach sel14_q: {T4.S4.CHUNK['rows']} != {A.chunk}")
    lab = lab_arms()
    for a in ARMS:
        if a != "exact" and lab.get(a) != ARMS[a]:
            raise SystemExit(f"arm {a} differs from t4_lab.py: {lab.get(a)} vs {ARMS[a]}")
    if md5_file(Q14.__file__) != SEL14_MD5:
        raise SystemExit(f"sel14_q md5 {md5_file(Q14.__file__)} != {SEL14_MD5}")
    for a in arms:
        if ARMS[a][0]:
            Q.parse(ARMS[a][0])
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.float32, attn_implementation="sel_attn",
                                                 device_map=dev).eval()
    re_ = model.model.rotary_emb
    BLQ.BL["inv_freq"], BLQ.BL["asc"] = re_.inv_freq.detach().float(), float(getattr(re_, "attention_scaling", 1.0))
    Q.RC.R["inv_freq"], Q.RC.R["asc"] = BLQ.BL["inv_freq"], BLQ.BL["asc"]
    tok = AutoTokenizer.from_pretrained(A.model)
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
    think_id = tok.convert_tokens_to_ids("</think>")
    if not isinstance(think_id, int) or think_id == tok.unk_token_id:
        raise SystemExit("tokenizer has no </think> token")
    gc = model.generation_config
    if getattr(gc, "repetition_penalty", None) not in (None, 1.0):
        raise SystemExit(f"generation_config repetition_penalty {gc.repetition_penalty} is not applied here")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    ORIG = Q14._attend_h
    Q14._attend_h = T4.t4_attend_h
    try:
        run(model, tok, alist, nope, mods, cur_mods, think_id, eos_ids, gc, arms, dev)
    finally:
        Q14._attend_h = ORIG
        X["run"] = None


def run(model, tok, alist, nope, mods, cur_mods, think_id, eos_ids, gc, arms, dev) -> None:

    def set_arm(arm: str, item: int, p_len: int, cap: int) -> None:
        name, pol = ARMS[arm]
        T4.reset_item(item, p_len, pol, None, False, cap=cap)
        Q.PREFILL["from"] = p_len
        X["run"] = name

    def guard_calls() -> None:
        """t4_attend_h exactly once per layer per forward on compressed routes, never on the exact route."""
        want = 0 if X["run"] is None else 1
        bad = {li: T4.CFG.calls.get(li, 0) for li in alist if T4.CFG.calls.get(li, 0) != want}
        if bad or (want == 0 and T4.CFG.calls):
            raise SystemExit(f"t4_attend_h calls per layer {bad or T4.CFG.calls}, want {want} on {X['run']}")
        T4.CFG.calls.clear()

    @torch.no_grad()
    def fwd(ids: torch.Tensor, cache):
        T4.CFG.calls.clear()
        X["inc"] = cache is not None
        try:
            o = model.model(input_ids=ids, past_key_values=cache, use_cache=cache is not None)
        finally:
            X["inc"] = False
        guard_calls()
        return o.last_hidden_state[0]

    def step(ids: torch.Tensor, cache) -> torch.Tensor:
        return model.lm_head(fwd(ids, cache)[-1:]).float()[0]

    def stats_layers(decode_rows: int) -> dict:
        if set(T4.STATS) != set(alist):
            raise SystemExit(f"t4 records for layers {sorted(T4.STATS)}, want {alist}")
        for li in alist:                              # column 0 is KV heads per recorded decode row
            got, kvh = float(T4.STATS[li]["rec"][:, 0].sum()), T4.STATE[li]["act"].shape[0]
            if got != kvh * decode_rows:
                raise SystemExit(f"layer {li}: records cover {got / kvh} decode rows, want {decode_rows}")
        return {str(li): T4.stats_to_json(li) for li in alist}

    w_files = sorted(Path(A.model).glob("*.safetensors"))
    if not w_files:
        raise SystemExit(f"no safetensors shards in {A.model}")
    weights = {"config_md5": md5_file(Path(A.model) / "config.json"),
               "safetensors": [[f.name, f.stat().st_size, md5_file(f)] for f in w_files]}
    # note: script_md5 and t4_lab_md5 are bound, so any edit to either script refuses to resume (new output file)

    meta = {"model": A.model, "weights": weights, "arms": arms, "max_new": A.max_new, "start": A.start, "n": A.n,
            "chunk": A.chunk, "temperature": TEMP, "top_k": TOP_K, "top_p": TOP_P, "nope_layers": nope,
            "cal_md5": md5_file(CQ.CAL[0]), "t4_lab_md5": md5_file(Path(__file__).resolve().parent / "t4_lab.py"),
            "arm_defs": {a: [ARMS[a][0], None if ARMS[a][1] is None else [ARMS[a][1].kind, ARMS[a][1].g,
                                                                      [list(t) for t in ARMS[a][1].tiers]]]
                         for a in arms}, "pages": False,
            "generation_config": {k: getattr(gc, k, None) for k in ("repetition_penalty", "min_p", "typical_p",
                                                                   "epsilon_cutoff", "no_repeat_ngram_size")},
            "script_md5": md5_file(__file__), "t4_q_md5": md5_file(T4.__file__), "stack_modules_md5": mods,
            "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                         "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}}

    if A.check:
        check(model, alist, arms, dev, meta, set_arm, fwd, stats_layers)
        return

    data_md5 = md5_file(A.data)
    if data_md5 != DATA_MD5:
        raise SystemExit(f"data md5 {data_md5} is not the registered {DATA_MD5}")
    if A.start < 64:
        raise SystemExit("A3 problems must not overlap the replay traces (items 0-63)")
    if not A.exclude_traces:
        raise SystemExit("--exclude_traces is required in generation mode")
    meta["data_md5"] = data_md5
    rows_d = [json.loads(x) for x in Path(A.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    items = list(range(A.start, A.start + A.n))
    if items[-1] >= len(rows_d):
        raise SystemExit("problem range beyond the dataset")
    ex_tr = [json.loads(x) for x in Path(A.exclude_traces).read_text(encoding="utf-8").splitlines() if x.strip()]
    for t in ex_tr:                                    # trace index space == dataset index space
        pr = tok.apply_chat_template([{"role": "user", "content": rows_d[t["i"]]["problem"] + SUFFIX}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if tok(pr, add_special_tokens=False).input_ids != t["prompt_ids"]:
            raise SystemExit(f"replay trace {t['i']} is not dataset item {t['i']} with this prompt")
    used = {t["i"] for t in ex_tr}
    if used & set(items):
        raise SystemExit(f"problems {sorted(used & set(items))} are replay traces")
    meta["exclude_traces_md5"] = md5_file(A.exclude_traces)
    out = Path(A.out)
    lock = open(str(out) + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{out}.lock is held by another run")
    done = set()
    # records are keyed by (problem, arm): extending the range or adding arms resumes; anything that changes a
    # record's content refuses
    bind = ("model", "weights", "max_new", "chunk", "temperature", "top_k", "top_p", "nope_layers", "cal_md5",
            "t4_lab_md5", "generation_config", "script_md5", "t4_q_md5", "stack_modules_md5", "data_md5",
            "versions", "pages")
    torn = 0
    if out.exists():
        raw = out.read_bytes()
        for line in raw.splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                torn += 1
                continue
            cur = meta["arm_defs"].get(r["arm"])
            if any(r["meta"].get(k) != meta[k] for k in bind) or (cur is not None and
                                                                  r["meta"]["arm_defs"][r["arm"]] != cur):
                raise SystemExit(f"{out} holds a record from another configuration; refusing to resume")
            done.add((r["i"], r["arm"]))
        if raw and not raw.endswith(b"\n"):
            with open(out, "a", encoding="utf-8") as f:
                f.write("\n")
    if torn:
        print(f"resume: skipped {torn} unreadable line(s) in {out}", flush=True)
    t_all = time.time()
    for i in items:
        prompt = tok.apply_chat_template([{"role": "user", "content": rows_d[i]["problem"] + SUFFIX}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        p_len = len(p_ids)
        for arm in arms:
            if (i, arm) in done:
                continue
            if cur_mods() != mods:
                raise SystemExit("stack modules changed since the start of the run")
            set_arm(arm, i, p_len, p_len + A.max_new)
            gen_u = torch.Generator(device="cpu").manual_seed(500000 + i)
            cache = DynamicCache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            g: list[int] = []
            logits = step(torch.tensor([p_ids], device=dev), cache)
            ended = False
            n_steps = 0                                  # one-row decode calls
            while True:
                u = float(torch.rand((), generator=gen_u))
                t_ = sample(logits, u)
                g.append(t_)
                if t_ in eos_ids:
                    ended = True
                    break
                if len(g) >= A.max_new:
                    break
                logits = step(torch.tensor([[t_]], device=dev), cache)
                n_steps += 1
            sec = time.time() - t0
            layers = stats_layers(n_steps) if ARMS[arm][1] is not None else {}
            X["run"] = None
            if ended:
                g = g[:-1]
            te = next((j for j, t in enumerate(g) if t == think_id), None)
            ans = tok.decode(g[te + 1:]) if te is not None else ""
            pred = last_boxed(ans) if te is not None else None
            ok = pred is not None and norm(pred) == norm(rows_d[i]["answer"])
            rec = {"i": i, "arm": arm, "prompt_len": p_len, "gen_ids": g, "n_gen": len(g), "ended": ended,
                   "think_end": None if te is None else p_len + te + 1, "pred": pred, "gold": rows_d[i]["answer"],
                   "ok": ok, "sec": round(sec, 1), "ms_per_token": round(1000 * sec / max(1, len(g)), 2),
                   "peak_alloc_mib": torch.cuda.max_memory_allocated() / 2 ** 20, "layers": layers, "meta": meta}
            del cache
            torch.cuda.empty_cache()
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"item {i} {arm} n_gen {len(g)} ended {ended} ok {ok} {sec:.0f}s "
                  f"({rec['ms_per_token']} ms/token) total {time.time() - t_all:.0f}s", flush=True)
    print("DONE t4_gen", A.model, f"s{A.start}n{A.n}", flush=True)


@torch.no_grad()
def check(model, alist, arms, dev, meta, set_arm, fwd, stats_layers) -> None:
    """Teacher-force one replay trace through the incremental path and the full forward, per arm; gate the result.
    Scope: incremental decoding reproduces the same arm's full forward (the path A2 measured), and the exact route
    is plain causal attention. It does not validate a compressed route against anything else."""
    if "exact" not in arms or len(arms) < 2:
        raise SystemExit("--check needs the exact arm and at least one compressed arm")
    outp = Path(A.out + ".check.json")
    if outp.exists():
        raise SystemExit(f"{outp} exists; remove it to rerun the check")
    trs = [json.loads(x) for x in Path(A.check).read_text(encoding="utf-8").splitlines() if x.strip()]
    tr = min(trs, key=lambda t: (abs(len(t["prompt_ids"]) + len(t["gen_ids"]) - A.check_len), t["i"]))
    ids_l = (tr["prompt_ids"] + tr["gen_ids"])[:A.check_len]
    p_len, L = len(tr["prompt_ids"]), len(ids_l)
    if A.check_len > 4096:
        raise SystemExit("--check_len above 4096: the sdpa reference is not chunked")
    if L < p_len + 1500:
        raise SystemExit(f"check trace {tr['i']} has {L - p_len} decode rows; need >= 1500 for demotion decisions")
    ids = torch.tensor([ids_l], device=dev)
    res = {**meta, "check_trace": tr["i"], "L": L, "prompt": p_len, "gate": GATE, "arms_check": {},
           "scope": "incremental vs full forward of the same arm; exact route vs plain causal attention"}
    hid_inc = {}

    def compare(ha: torch.Tensor, hb: torch.Tensor) -> dict:
        """Chunked next-token comparison of two hidden-state row sets (log-probs never held for all rows)."""
        kls, agree, mx = [], 0, 0.0
        for r in range(0, ha.shape[0], 256):
            la = torch.log_softmax(model.lm_head(ha[r:r + 256].to(dev)).float(), -1)
            lb = torch.log_softmax(model.lm_head(hb[r:r + 256].to(dev)).float(), -1)
            kls.append((la.exp() * (la - lb)).sum(-1).cpu())
            agree += int((la.argmax(-1) == lb.argmax(-1)).sum())
            mx = max(mx, float((la - lb).abs().max()))
            del la, lb
        kl = torch.cat(kls)
        return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                "rows_kl_above_1e-4": int((kl > 1e-4).sum()), "argmax_agree": agree / ha.shape[0],
                "max_abs_logprob_diff": mx}

    for arm in arms:
        torch.cuda.reset_peak_memory_stats()
        X["sdpa"], X["sdpa_err"], X["sdpa_n"] = arm == "exact", 0.0, 0
        set_arm(arm, tr["i"], p_len, 0)
        hf = fwd(ids, None)[p_len - 1:]
        full_dec = {li: (T4.STATE[li]["act"].clone(), T4.STATE[li]["dpos"].clone()) for li in T4.STATE}
        pert = None
        if A.perturb > 0:
            gen_p = torch.Generator(device=dev).manual_seed(4242)

            def noise(_m, _i, o):
                return o + A.perturb * o.abs() * torch.randn(o.shape, generator=gen_p, device=o.device, dtype=o.dtype)
            hk = model.model.embed_tokens.register_forward_hook(noise)
            try:
                set_arm(arm, tr["i"], p_len, 0)
                hp = fwd(ids, None)[p_len - 1:]
            finally:
                hk.remove()
            pert = compare(hf, hp)
            if ARMS[arm][1] is not None:
                tot = agree = 0
                for li, (a_f, d_f) in full_dec.items():
                    same = (a_f == T4.STATE[li]["act"]) & (d_f == T4.STATE[li]["dpos"])
                    tot += a_f.numel()
                    agree += int(same.sum())
                pert["decision_agree"] = agree / tot
            del hp
        set_arm(arm, tr["i"], p_len, L)
        cache = DynamicCache()
        rows = [fwd(ids[:, :p_len], cache)[-1]]
        t0 = time.time()
        for t in range(p_len, L):
            rows.append(fwd(ids[:, t:t + 1], cache)[-1])
        torch.cuda.synchronize()
        sec = time.time() - t0
        inc_layers = stats_layers(L - p_len) if ARMS[arm][1] is not None else {}
        hi = torch.stack(rows)
        del rows, cache
        r = compare(hf, hi)
        if pert is not None:
            r["perturbed_full"] = {"rel_noise": A.perturb, **pert}
        r.update({"ms_per_decode_step": 1000 * sec / (L - p_len), "decode_rows": L - p_len,
                  "peak_alloc_mib": torch.cuda.max_memory_allocated() / 2 ** 20})
        if arm == "exact":
            # one reference per layer per call: the full forward plus the prefill call plus every decode row
            r["sdpa_err"], r["sdpa_calls"] = X["sdpa_err"], X["sdpa_n"]
            r["sdpa_calls_want"] = len(alist) * (2 + L - p_len)
        if ARMS[arm][1] is not None:
            tot = agree = 0
            per_layer = {}
            for li, (a_f, d_f) in full_dec.items():
                n_ = a_f.shape[1]
                a_i, d_i = T4.STATE[li]["act"][:, :n_], T4.STATE[li]["dpos"][:, :n_]
                same = (a_f == a_i) & (d_f == d_i)
                tot += a_f.numel()
                agree += int(same.sum())
                per_layer[str(li)] = float(same.double().mean())
            r["decision_agree"] = agree / tot
            r["decision_agree_min_layer"] = min(per_layer.values())
            r["demoted_tokens"] = sum(sum(v["dem"].values()) for v in inc_layers.values())
        hid_inc[arm] = hi.cpu()
        del hf, hi
        res["arms_check"][arm] = r
        X["sdpa"] = False
    for arm, r in res["arms_check"].items():
        if arm != "exact":
            r["kl_vs_exact_mean"] = compare(hid_inc["exact"], hid_inc[arm])["kl_mean"]
        fails = []
        if r["argmax_agree"] < GATE["argmax_agree_min"]:
            fails.append("argmax_agree")
        if r["kl_mean"] > GATE["kl_mean_max"]:
            fails.append("kl_mean")
        if "decision_agree" in r and r["decision_agree"] < GATE["decision_agree_min"]:
            fails.append("decision_agree")
        if "decision_agree" in r and r["demoted_tokens"] <= 0:
            fails.append("no_demotion")
        if arm != "exact" and not r["kl_vs_exact_mean"] > 0:
            fails.append("kl_vs_exact_zero")
        if arm == "exact" and (r["sdpa_calls"] != r["sdpa_calls_want"] or r["sdpa_err"] > GATE["sdpa_err_max"]):
            fails.append("sdpa")
        r["fails"] = fails
        r["pass"] = not fails
        print(arm, json.dumps(r), flush=True)
    res["pass"] = all(r["pass"] for r in res["arms_check"].values())
    outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("CHECK", "PASS" if res["pass"] else "FAIL", flush=True)
    if not res["pass"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
