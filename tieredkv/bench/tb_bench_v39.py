# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""tb_bench v3.9 (2026-11-16, round SKR) = v3.8 with all 13 RULER tasks in the ruler family (adds
niah_single_1, niah_single_2, niah_multikey_1, qa_2). Everything else is v3.8; its text follows.
tb_bench v3.8 (2026-11-16, SeKV comparison) = v3.7b with the LongBench family served by lb16 (16 English
tasks, official prompts and metrics), TB_LB_MINLEN (skip rows with a shorter prompt body; first --n remaining rows) and
no chat template on lb16.NO_CHAT tasks. Everything else is v3.7b; its text follows.
tb_bench v3.7b (2026-11-06, HM1 supplement): v3.7 without the bare-whitespace gate on the first gold token
(Llama-3 splits " 1234567" into " " + digits; the first gold token is predicted by the shared prefill row and left out).
tb_bench v3.7 (2026-11-06) = v3.6 + gold-answer teacher-forced metrics for RULER tasks (TB_GOLD=1): the reference
answer, rendered as " " + " ".join(outputs) (qa_1: " " + outputs[0]), is fed after the prompt; fp keeps its log-prob rows;
every other arm records gold_kl_mean / gold_top1 (vs fp) and gold_nll (mean -log p of the gold tokens, rows 1..n-1; the first
gold token is predicted by the shared prefill row, identical for all arms, and is left out). A continuous, task-anchored
measure with no greedy path dependence. v3.6 added RULER cwe / fwe / qa_1. v3.5 text follows:
tb_bench v3.5 (2026-11-06) = v3.4 + teacher-forced paired metrics on the fp trajectory (tf_kl_mean,
tf_kl_max, tf_top1, tf_nll over the steps whose target is an fp token; EOS row excluded) and per-arm GPU peak memory; free greedy decoding unchanged. v3.4 text follows:
tb_bench v3.4 (R-0038, 2026-11-06) = v3.3 (4fd163e9) + 128K memory changes: prompt attention and the fp arm's
decode through sdpa on the model dtype; the prefill cache stays on the GPU and is cropped back after every arm (write-back
arms restore the prefix from a host copy); a per-layer checksum of the prefix is verified after every arm; the
exact-vs-sdpa check runs on the first 8192 tokens. Everything below is the v3.3 text:
tb_bench v3.3 (rootcause line; S3): task-level benchmark of the layered KV scheme against KIVI, Block-GTQ
and the official KVQuant quantizers (kvq_q: kvq2n / kvq3 / kvq2 and kvq<tag><fetch suffix>; v3.3 adds them and the official
RULER task family ruler:<task>@<len>, read from the S0 shared samples; NIAH / LongBench paths unchanged from v3.2).
Greedy generation; every method is routed through gtq_q.attend, the same code as the KL rounds:
  - KIVI-128 and the sel4_q names;
  - the main line's strongest base stk128 (sel10_q);
  - the Block-GTQ reimplementation gtqk<K>v<V> (gtq_q);
  - any of these with our selection / two-stage fetch / debias suffix.
Task families:
  lb:<task>          LongBench (lb.py prompts, middle truncation to --max_len, metrics): hotpotqa, 2wikimqa, musique,
                     multifieldqa_en, qasper, passage_retrieval_en
  niah:<kind>@<len>  RULER-style needle in a haystack. Haystack: pg19 text. Needles: "One of the special magic numbers for
                     <key> is: <7 digits>." inserted between sentences. Every random draw (keys, values, depths, haystack
                     offset) is a tokenizer-free fraction or choice seeded by (seed, kind, len), so the items match across
                     models up to tokenization. Kinds:
                       single      1 needle, ask its key
                       multikey    4 keys, ask 1
                       multivalue  1 key with 4 values, ask all
                       multiquery  4 keys, ask 2
                     score = fraction of the asked values found as whole 7-digit numbers (RULER string_match_all);
                     also recorded: exact_set (exactly the asked needle values and no other needle value).
Protocol "lit" (KIVI / SpeCache papers):
  - The prompt is attended exactly ONCE per item; the exact prefill cache is kept in host RAM and every arm continues
    from its own GPU copy of it (v3.3; v3.2 deep-copied it on the GPU, which does not fit at 32K on 48 GB).
  - Greedy decode is a manual loop: argmax, no logits processors.
  - Every decode step reads the arm's quantised cache. The whole cache is re-quantised each call (fake quantisation,
    as in the main line's bench), so windows, sinks and selection see the current length; exception: kvq_q's
    no-fetch arms write their quantised rows back into the arm's own cache once (write-back) and quantise only new
    rows at later steps.
  - The cache is always a full DynamicCache: sliding-window layers get every key and apply the window as a mask.
  - Asserted every step: the decode-time key length equals the prompt plus the generated tokens.
Calibration as gtq_lab: one exact pass on CQ.CAL captures q / k / v (stored on CPU), then Q.calib per layer, then
Q.calib_all.
Guards before any item:
  - partial-rotary models are refused;
  - the weights must fit in 80% of the GPU memory;
  - every non-fp arm must actually change the attention (it must not silently equal exact attention);
  - the exact path is compared with the model's sdpa attention on one item (recorded).
Checkpointing: one JSON line per (arm, task, i) in <out>.jsonl (errors recorded, runs resume); summary JSON written
atomically after every item.
Paths: KVT_ROOT (default /home/user/kvt_bench) holds hc_mac/{code,rootcause}, data/ (calibration text),
data/data/*.jsonl (LongBench) and data/r14/pg19_text.txt.
Usage: KVT_ROOT=... python tb_bench.py <model_dir> <out_json> --arms fp,kivi128,... --tasks lb:hotpotqa,niah:single@8192
       [--n 100] [--max_len 8192] [--dtype bfloat16] [--seed 20261105] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(os.environ.get("KVT_ROOT", "/home/user/kvt_bench"))
for p_ in (ROOT, ROOT / "hc_mac" / "code", ROOT / "hc_mac" / "rootcause"):
    sys.path.insert(0, str(p_))
import bl_q as BLQ  # noqa: E402
import cq_lab3 as CQ  # noqa: E402
import importlib  # noqa: E402
Q = importlib.import_module(os.environ.get("TB_Q_MODULE", "kvq_q"))  # v3.3: method module (kvq_q passes every non-kvq name
                                                                       # to gtq_q; a partner module may wrap kvq_q)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb16 as lb  # noqa: E402

CQ.CAL = (str(ROOT / "data" / Path(CQ.CAL[0]).name),) + tuple(CQ.CAL[1:])   # same file, relocatable root
LB_TASKS = lb.TASKS
LB_MINLEN = int(os.environ.get("TB_LB_MINLEN", "0"))
NIAH_KINDS = ("single", "multikey", "multivalue", "multiquery")
NIAH_MAXGEN = {"single": 64, "multikey": 64, "multivalue": 96, "multiquery": 80}
HAYSTACK = ROOT / "data" / "r14" / "pg19_text.txt"
WORDS = ("amber", "basil", "cobalt", "dune", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kestrel", "lilac",
         "meadow", "nectar", "onyx", "pepper", "quartz", "raven", "saffron", "thistle", "umber", "velvet", "willow",
         "yarrow", "zephyr", "alder", "birch", "cedar", "dahlia", "fennel", "ginger", "hazel", "jasper", "laurel",
         "maple", "nutmeg", "olive", "poplar", "quince", "rowan", "sorrel", "tansy", "vervain", "zinnia")
END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>", "<|end_of_text|>")
NUM7 = re.compile(r"(?<![0-9])[0-9]{7}(?![0-9])")
GROUP_RUN = re.compile("(?<![0-9])[0-9]{1,3}(?:[, \t\u00a0\u2009\u202f][0-9]{3})+(?![0-9])")   # a full thousands-grouped run
GROUP_SEP = re.compile("[, \t\u00a0\u2009\u202f]")                    # comma, space, tab, NBSP, thin spaces
X: dict = {"arm": None, "cap": False, "kv": {}, "shape": None, "expect": None, "lens": {}}
from transformers.integrations.sdpa_attention import sdpa_attention_forward as _sdpa_fwd  # noqa: E402
SDPA: dict = {"off": False, "calls": 0, "fallback": 0}                 # v3.4: off = force the v3.3 exact path (check)


def tb_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    li = int(module.layer_idx)
    if not X["cap"] and not SDPA["off"] and (query.shape[2] > 1 or X["arm"] is None):    # v3.4: prompt / fp decode
        if kw.get("sliding_window") or kw.get("softcap"):
            SDPA["fallback"] += 1                                        # windowed / capped layers: v3.3 exact path
        else:
            if X["expect"] is not None and query.shape[2] == 1 and key.shape[2] != X["expect"]:
                raise RuntimeError(f"layer {li}: decode key length {key.shape[2]} != expected {X['expect']} (truncated cache?)")
            SDPA["calls"] += 1
            return _sdpa_fwd(module, query, key, value, attention_mask, scaling=scaling, dropout=0.0,
                             is_causal=query.shape[2] > 1)
    q, k, v = query[0].float(), key[0].float(), value[0].float()
    if X["cap"]:
        X["kv"][li] = (k.cpu(), v.cpu(), q.cpu())
        X["shape"] = (q.shape[0], k.shape[0], q.shape[-1])
    if X["expect"] is not None and q.shape[1] == 1 and k.shape[1] != X["expect"]:
        raise RuntimeError(f"layer {li}: decode key length {k.shape[1]} != expected {X['expect']} (truncated cache?)")
    scale = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
    arm = X["arm"] if q.shape[1] == 1 else None                         # lit: exact prompt attention
    if arm is not None and hasattr(Q, "set_cache_ref"):
        Q.set_cache_ref(li, key[0], value[0])                           # v3.3: the arm's own cache (kvq write-back)
    out = Q.attend(q, k, v, li, arm, scale, kw)
    return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None


AttentionInterface.register("tb_attn", tb_attn)


def niah_item(tok, kind, length, rng, hay_text, hay_sents, over):
    """one RULER-style item: (prompt text, wanted values, all needle values). All draws are tokenizer-free."""
    nk = 1 if kind == "single" else 4
    keys = rng.sample(WORDS, nk) if kind != "multivalue" else [rng.choice(WORDS)] * 4
    vals = []
    while len(vals) < nk:
        v_ = str(rng.randrange(1000000, 10000000))
        if v_ not in hay_text and v_ not in vals:                       # no natural occurrence in the haystack
            vals.append(v_)
    fracs = sorted(rng.random() for _ in range(nk))                     # needle depths
    start = rng.random()                                                # haystack offset (fraction of the sentences)
    pick = rng.sample(range(nk), 2) if kind == "multiquery" else [rng.randrange(nk)]
    if kind == "multivalue":
        ask, want = keys[0], vals
    elif kind == "multiquery":
        ask, want = f"{keys[pick[0]]} and {keys[pick[1]]}", [vals[pick[0]], vals[pick[1]]]
    else:
        ask, want = keys[pick[0]], [vals[pick[0]]]
    head = ("Some special magic numbers are hidden within the following text. Make sure to memorize them. "
            "I will quiz you about the numbers afterwards.\n")
    tail = (f"\nWhat are all the special magic numbers for {ask} mentioned in the provided text? "
            f"The special magic numbers for {ask} mentioned in the provided text are")
    needles = [f"One of the special magic numbers for {k_} is: {v_}." for k_, v_ in zip(keys, vals)]
    budget = length - over - len(tok(head + tail + " ".join(needles), add_special_tokens=False).input_ids)
    if budget < 256:
        raise ValueError(f"length {length} too short")
    i0 = int(start * len(hay_sents))
    body, ntok = [], 0
    j = i0
    while ntok < budget:                                                # whole sentences, wrapping around the book
        s_ = hay_sents[j % len(hay_sents)]
        ntok += len(tok(" " + s_, add_special_tokens=False).input_ids)
        body.append(s_)
        j += 1
    if len(body) > 1:
        body.pop()                                                      # stay at or under the budget
    for f_, nd in sorted(zip(fracs, needles), reverse=True):            # insert deepest first, indices stay valid
        body.insert(int(round(f_ * len(body))), nd)
    return head + " ".join(body) + tail, want, vals


HV = "3.7b"
TF = os.environ.get("TB_TF", "0") == "1"
GOLD = os.environ.get("TB_GOLD", "0") == "1"                            # v3.7: gold-answer teacher forcing (RULER only)


def gold_text(kind, refs):
    """v3.7: one fixed rendering of the reference answer per item (same string for every arm). It starts with a space
    and is tokenized on its own, as the model's own answers start with a space token even after vt's prefix that ends in
    a space (a joint tokenization would merge that space and change the prompt tokens)."""
    return " " + (refs[0].strip() if kind in RULER_PART else " ".join(x.strip() for x in refs))


def seven_digit_numbers(text):
    """whole 7-digit numbers; a grouped run (1,234,567 / 1 234 567) is merged only when it has exactly 7 digits, so
    "1234 567" or "1234567 890" are never read as other numbers (v3.2; v3.1 merged any separator before 3 digits)."""
    def ungroup(m):
        s = GROUP_SEP.sub("", m.group(0))
        return s if len(s) == 7 else m.group(0)
    return set(NUM7.findall(GROUP_RUN.sub(ungroup, text)))


def niah_score(pred, want, allv):
    found = seven_digit_numbers(pred)
    rec = sum(w in found for w in want) / len(want)
    exact = set(want) <= found and not (found & (set(allv) - set(want)))
    return rec, exact


RULER_DIR = Path(os.environ.get("RULER_DIR", str(ROOT / "hc_mac" / "s0_shared" / "ruler" / "e1")))
RULER_TASKS = ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
               "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2")                                    # v3.6: non-retrieval tasks for 128K
RULER_MAXGEN = {"vt": 30, "cwe": 120, "fwe": 50, "qa_1": 32, "qa_2": 32}           # RULER data/synthetic/constants.py; niah 128
RULER_PART = ("qa_1", "qa_2")                                               # RULER eval: qa uses string_match_part


def _cache_slots(cache):
    """(container, index) pairs of every key / value tensor of a DynamicCache (transformers 5.x layers or 4.x lists)."""
    out = []
    if hasattr(cache, "layers"):
        for lay in cache.layers:
            out += [(lay, "keys"), (lay, "values")]
    else:
        out += [(cache.key_cache, i) for i in range(len(cache.key_cache))]
        out += [(cache.value_cache, i) for i in range(len(cache.value_cache))]
    if not out:
        raise RuntimeError("unknown DynamicCache layout")
    return out


def _get(c, a):
    return getattr(c, a) if isinstance(a, str) else c[a]


def _set(c, a, t):
    if isinstance(a, str):
        setattr(c, a, t)
    else:
        c[a] = t


def cache_to_host(cache):
    """move the cache tensors to (pageable) host memory in place (the item's exact reference). Not pinned: pin_memory()
    goes through the caching host allocator, which rounds every block up to a power of two and keeps freed blocks."""
    for c, a in _cache_slots(cache):
        _set(c, a, _get(c, a).to("cpu"))
    return cache


def cache_copy_to(cache_host, dev):
    """a fresh copy of a host-resident cache on the GPU (one per arm)."""
    import copy as _copy
    out = _copy.copy(cache_host)
    if hasattr(out, "layers"):
        out.layers = [_copy.copy(l) for l in cache_host.layers]
    else:
        out.key_cache, out.value_cache = list(cache_host.key_cache), list(cache_host.value_cache)
    for c, a in _cache_slots(out):
        _set(c, a, _get(c, a).to(dev))
    torch.cuda.synchronize()
    return out


def cache_fingerprint(cache, L0):
    """v3.4: per-layer exact integer checksums of the prefix rows [0, L0) of every key / value tensor (bit pattern sum)."""
    out = []
    for c, a in _cache_slots(cache):
        t = _get(c, a)[..., :L0, :]
        out.append(int(t.view(torch.int16).sum(dtype=torch.int64)) if t.element_size() == 2      # v3.5b: no copy
                   else int(t.view(torch.int32).sum(dtype=torch.int64)))
    return out


def cache_host_copy(cache):
    """v3.4: a host copy of the prefill tensors (only for write-back arms); the GPU cache is left in place."""
    return [_get(c, a).to("cpu") for c, a in _cache_slots(cache)]


def cache_restore(cache, L0, host, fpr):
    """v3.4: crop every layer back to the prompt, restore the prefix from host (write-back arms), verify the checksum.
    Returns None when the prefix equals the post-prefill cache, else a short description."""
    cache.crop(L0)
    slots = _cache_slots(cache)
    for j, (c, a) in enumerate(slots):
        t = _get(c, a)
        if t.shape[-2] != L0:
            return f"slot {j}: length {t.shape[-2]} after crop, expected {L0}"
        if host is not None:
            t.copy_(host[j], non_blocking=False)
    now = cache_fingerprint(cache, L0)
    bad = [j for j, (x_, y_) in enumerate(zip(now, fpr)) if x_ != y_]
    return None if not bad else f"prefix checksum differs in {len(bad)} slots (first {bad[:4]})"


def forced_pass(model, cache, toks, L0, dev, nsteps, ref=None):
    """v3.5: feed toks[0..nsteps-1] one at a time on cache (prompt length L0). Row j predicts toks[j + 1].
    ref None: returns the list of float32 log-prob rows (the fp reference). ref given (list of fp rows): returns per-step
    (KL(p_fp || p_arm), arm argmax == toks[j+1], -log p_arm(toks[j+1])) computed row by row (no [steps, V] buffers).
    X["arm"] must be set by the caller."""
    out = []
    for step in range(nsteps):
        X["expect"] = L0 + step + 1
        with torch.no_grad():
            o = model(input_ids=torch.tensor([[toks[step]]], device=dev), past_key_values=cache, use_cache=True)
            lp = torch.log_softmax(o.logits[0, -1].float(), -1)
            o = None
            if ref is None:
                out.append(lp)
            else:
                r_ = ref[step]
                out.append((float((r_.exp() * (r_ - lp)).sum()), int(lp.argmax()) == toks[step + 1], float(-lp[toks[step + 1]])))
            lp = None
    X["expect"] = None
    return out


def writes_back(arm):
    """v3.4: arms that overwrite rows of their own cache (kvq_q no-fetch arms, name kvq<tag> without a fetch suffix)."""
    if arm == "fp" or BLQ.is_bl(arm):
        return False
    sp = getattr(Q, "split", None)
    r = sp(arm) if sp is not None else None
    return bool(r) and r[1] == "kivi128"


def ruler_score(pred, refs, kind=None):
    """RULER scripts/eval/synthetic/constants.py: string_match_all (fraction of the reference strings contained in the
    prediction, case-insensitive) for niah / vt / cwe / fwe; string_match_part (1 if any reference is contained) for qa."""
    hit = [1.0 if r.lower() in pred.lower() else 0.0 for r in refs]
    return max(hit) if kind in RULER_PART else sum(hit) / len(refs)


def atomic_write(path, obj):
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def md5f(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke != a.out.endswith(".smoke.json"):
        raise SystemExit("smoke output must end with .smoke.json (and only smoke)")
    arms = tuple(x for x in a.arms.split(",") if x)
    if arms[0] != "fp":
        raise SystemExit("the first arm must be fp (the exact reference)")
    for x in arms[1:]:
        (BLQ.parse if BLQ.is_bl(x) else Q.parse)(x)
    tasks = tuple(x for x in a.tasks.split(",") if x)
    for t in tasks:
        fam, _, rest = t.partition(":")
        if fam == "lb":
            if rest not in LB_TASKS:
                raise SystemExit(f"unknown task {t}")
        elif fam == "niah":
            kind, _, ln = rest.partition("@")
            if kind not in NIAH_KINDS or not ln.isdigit():
                raise SystemExit(f"unknown task {t}")
        elif fam == "ruler":
            kind, _, ln = rest.partition("@")
            if kind not in RULER_TASKS or not ln.isdigit():
                raise SystemExit(f"unknown task {t}")
            if not os.environ.get("RULER_TAG"):
                raise SystemExit("ruler tasks need RULER_TAG (the sample set of this model, e.g. llama2_7b_32k)")
        else:
            raise SystemExit(f"unknown task {t}")
    # context length per run: LongBench is cut to max_len, NIAH / RULER items are built at the length in the task name
    ctx = max([a.max_len] + [int(t.rpartition("@")[2]) for t in tasks if t.split(":")[0] in ("niah", "ruler")])
    for x in arms[1:]:
        if hasattr(Q, "max_len_check"):
            Q.max_len_check(x, ctx)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    Q.PREFILL["on"] = False
    Q.CHUNK["rows"], BLQ.BL["chunk"] = 512, 512
    dev = a.device
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    for f_ in (CQ.CAL[0], HAYSTACK):
        if not Path(f_).is_file():
            raise SystemExit(f"missing {f_}")
    if md5f(CQ.CAL[0]) == md5f(HAYSTACK):
        raise SystemExit("calibration text equals the haystack")
    try:
        import accelerate  # noqa: F401  (device_map loads straight to the GPU)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="tb_attn", device_map={"": dev}).eval()
    except ImportError:                                                 # same weights, loaded on CPU first
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="tb_attn").to(dev).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    mpe = getattr(model.config, "max_position_embeddings", None)
    if mpe is not None and ctx > mpe * float((getattr(model.config, "rope_scaling", None) or {}).get("factor", 1.0)):
        raise SystemExit(f"task length {ctx} exceeds the model's context ({mpe}, rope_scaling "
                         f"{getattr(model.config, 'rope_scaling', None)})")
    wgb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 30
    tot = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
    if wgb > 0.8 * tot:
        raise SystemExit(f"weights {wgb:.1f} GB > 80% of {tot:.1f} GB")
    re_ = model.model.rotary_emb
    inv = re_.inv_freq.detach().float()
    BLQ.BL["inv_freq"], BLQ.BL["asc"] = inv, float(getattr(re_, "attention_scaling", 1.0))
    Q.RC.R["inv_freq"], Q.RC.R["asc"] = BLQ.BL["inv_freq"], BLQ.BL["asc"]
    cal = CQ.text_ids(tok, CQ.CAL[0], CQ.CAL[1], CQ.CAL[2], dev)
    if cal.shape[1] != CQ.CAL[2]:
        raise SystemExit(f"calibration window {cal.shape[1]} != {CQ.CAL[2]}")
    X["cap"] = True
    try:
        with torch.no_grad():
            model(input_ids=cal, use_cache=False, logits_to_keep=1)
    finally:
        X["cap"] = False
    H_, KVH, d = X["shape"]
    if inv.numel() != d // 2:
        raise SystemExit(f"partial rotary ({inv.numel()} frequencies for head_dim {d}) is not supported")
    for li in sorted(X["kv"]):
        Q.calib(li, *(t.to(dev) for t in X["kv"][li]))
    alist = sorted(X["kv"])
    nope = Q.calib_all(model, {li: tuple(t.to(dev) for t in X["kv"][li]) for li in alist}, alist)
    X["kv"] = {}
    torch.cuda.empty_cache()
    # guard: every non-fp arm must change the attention of a decode row (catches arms that silently equal exact)
    g_ = torch.Generator(device="cpu").manual_seed(1)
    kk = torch.randn(KVH, 3000, d, generator=g_).to(dev) * 2
    vv = torch.randn(KVH, 3000, d, generator=g_).to(dev)
    qq = torch.randn(H_, 1, d, generator=g_).to(dev) * 2
    ex = Q.attend(qq, kk, vv, alist[0], None, 1 / math.sqrt(d), {})
    for arm in arms[1:]:
        if hasattr(Q, "reset"):
            Q.reset()
        if hasattr(Q, "set_cache_ref"):
            Q.set_cache_ref(alist[0], kk.clone(), vv.clone())           # write-back arms mutate their cache
        dd = float((Q.attend(qq, kk, vv, alist[0], arm, 1 / math.sqrt(d), {}) - ex).abs().max())
        if not dd > 1e-6:
            raise SystemExit(f"arm {arm} equals exact attention on random data (diff {dd:.2e}): method not applied")
    del kk, vv, qq, ex
    if hasattr(Q, "reset"):
        Q.reset()
    eos = set()
    for e in (model.generation_config.eos_token_id, tok.eos_token_id):
        eos |= set(e if isinstance(e, list) else ([e] if e is not None else []))
    vocab = tok.get_vocab()
    eos |= {vocab[t] for t in END_TOKENS if t in vocab}

    def wrap(text):
        msgs = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    # base models (LLaMA-2-7B-32K, Mistral-7B-v0.1) have no chat template: only the ruler family runs on them (raw
    # input + answer_prefix, RULER's base template); lb / niah prompts are chat-wrapped and need a template
    if getattr(tok, "chat_template", None) is None:
        if any(t.split(":")[0] in ("lb", "niah") for t in tasks):
            raise SystemExit("this tokenizer has no chat template: lb / niah tasks need one (ruler tasks do not)")
        over = None
    else:
        over = len(tok(wrap(""), add_special_tokens=False).input_ids)
    hay_text = HAYSTACK.read_text(encoding="utf-8", errors="replace")
    hay_sents = [s_ for s_ in re.split(r"(?<=[.!?])\s+", " ".join(hay_text.split())) if len(s_) > 20]
    n = 2 if a.smoke else a.n
    items, counts = [], {}
    for t in tasks:
        fam, _, rest = t.partition(":")
        if fam == "lb":
            rows = [json.loads(l) for l in open(lb.DATA / f"{rest}.jsonl", encoding="utf-8")]
            kept = 0
            for i, r in enumerate(rows):
                if kept >= n:
                    break
                body = tok(lb.PROMPT[rest].format(context=r["context"], input=r["input"]), add_special_tokens=False).input_ids
                if len(body) < LB_MINLEN:                              # v3.8: SeKV keeps examples >= 6.4K tokens
                    continue
                kept += 1
                budget = a.max_len - over
                if len(body) > budget:
                    body = body[: budget // 2] + body[-(budget // 2):]
                if rest in lb.NO_CHAT:                                 # v3.8: official pred.py, no chat template
                    ids = tok(tok.decode(body), add_special_tokens=True, return_tensors="pt").input_ids
                else:
                    ids = tok(wrap(tok.decode(body)), add_special_tokens=False, return_tensors="pt").input_ids
                items.append((t, i, ids, r["answers"], None, lb.MAXGEN[rest]))
            if kept < n:
                print(f"NOTE {t}: only {kept} of {n} rows have >= {LB_MINLEN} prompt tokens", flush=True)
        elif fam == "ruler":
            kind, _, ln = rest.partition("@")
            f_ = RULER_DIR / os.environ["RULER_TAG"] / ln / kind / "validation.jsonl"
            fp_ = f_.parent.parent / "FINGERPRINT"
            if not f_.is_file() or not fp_.is_file():
                raise SystemExit(f"missing RULER samples {f_} (or its FINGERPRINT)")
            rows = [json.loads(l) for l in open(f_, encoding="utf-8")][:n]
            for i, r in enumerate(rows):
                if not (isinstance(r.get("outputs"), list) and r["outputs"] and all(isinstance(x, str) for x in r["outputs"])):
                    raise SystemExit(f"{t} item {i}: outputs must be a non-empty list of strings")      # v3.6
                if kind in ("cwe", "fwe", "qa_1", "qa_2") and "answer_prefix" not in r:
                    raise SystemExit(f"{t} item {i}: answer_prefix missing")                          # v3.6
                # RULER base template: niah/vt store the answer prefix separately; RULER feeds input + answer_prefix; the model's
                # tokenizer adds BOS (as RULER's own pipeline); no chat template
                ids = tok(r["input"] + r.get("answer_prefix", ""), return_tensors="pt").input_ids
                if ids.shape[1] + RULER_MAXGEN.get(kind, 128) > int(ln) + 1:
                    raise SystemExit(f"{t} item {i}: {ids.shape[1]} prompt tokens exceed the length budget")
                gtok = None                                              # v3.7: gold tokens, tokenized on their own
                if GOLD:
                    gtok = tok(gold_text(kind, r["outputs"]), add_special_tokens=False).input_ids
                    if not gtok or not tok.decode(gtok).strip():           # v3.7b: a bare-space first token is fine
                        gtok = None                                      # (it is the shared-prefill row, left out anyway)
                items.append((t, i, ids, r["outputs"], gtok, RULER_MAXGEN.get(kind, 128)))     # allv slot unused by RULER
        else:
            kind, _, ln = rest.partition("@")
            rng = random.Random(f"{a.seed}:{kind}:{ln}")
            for i in range(n):
                text, want, allv = niah_item(tok, kind, int(ln), rng, hay_text, hay_sents, over)
                ids = tok(wrap(text), add_special_tokens=False, return_tensors="pt").input_ids
                items.append((t, i, ids, want, allv, NIAH_MAXGEN[kind]))
        counts[t] = sum(it[0] == t for it in items)
    prov = Q.provenance() if hasattr(Q, "provenance") else {}
    code = {}
    for nm, m in list(sys.modules.items()):                             # every loaded module of the code trees in use
        f_ = getattr(m, "__file__", None)
        if f_ and Path(f_).is_absolute() and Path(f_).is_file() and (
                str(ROOT.resolve()) in str(Path(f_).resolve()) or "kvq_s0" in str(Path(f_).resolve())):
            code[Path(f_).name] = md5f(f_)
    code["tb_bench.py"] = md5f(__file__)
    prov = dict(prov, module=Path(getattr(Q, "__file__", "") or "").name, code=sorted(code.items()),
                cal=md5f(CQ.CAL[0]), ruler_tag=os.environ.get("RULER_TAG", ""), maxgen=[it[5] for it in items])
    wts = sorted((f_.name, f_.stat().st_size) for pat in ("model*.safetensors", "pytorch_model*.bin")
                 for f_ in Path(a.model).glob(pat))
    if not wts:
        raise SystemExit(f"{a.model}: no weight files to fingerprint")
    prov = dict(prov, versions=[torch.__version__, torch.version.cuda, __import__("transformers").__version__],
                weights=[wts, [md5f(Path(a.model) / ix) for ix in ("model.safetensors.index.json",
                                                                    "pytorch_model.bin.index.json")
                               if (Path(a.model) / ix).is_file()]],
                config=md5f(Path(a.model) / "config.json"))
    cfg = hashlib.md5(json.dumps([str(Path(a.model).resolve()), a.max_len, a.seed, n, a.dtype, prov,
                                  [(it[0], it[1], hashlib.md5(it[2].numpy().tobytes()).hexdigest()) for it in items]]
                                 ).encode()).hexdigest()[:12]           # model + settings + exact item token ids
    jl = Path(a.out + "l")                                              # <out>.jsonl
    done = {}
    if jl.exists():
        for l in jl.read_bytes().split(b"\n"):                         # bytes: U+2028 in a pred is not a break
            try:
                r = json.loads(l.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):          # a line cut by a kill
                continue
            if not isinstance(r, dict):
                continue
            if "error" not in r and r.get("cfg") == cfg and r.get("hv") == HV and (not TF or r.get("tf_status") not in (None, "off")) \
                    and (not GOLD or not str(r.get("task", "")).startswith("ruler:") or r.get("gold_status") is not None):
                done[(r["arm"], r["task"], r["i"])] = r                 # NIAH records only from this stop rule
    mods = {}
    for nm, m in list(sys.modules.items()):                             # every loaded module from the code tree
        f_ = getattr(m, "__file__", None)
        if f_ and Path(f_).is_absolute() and Path(f_).is_file() and str(ROOT.resolve()) in str(Path(f_).resolve()):
            mods[nm] = md5f(f_)
    mods["tb_bench.py"] = md5f(__file__)
    mods.update({f"kvq:{k_}": v_ for k_, v_ in prov.items()})
    res = {"model": a.model, "cfg": cfg, "dtype": a.dtype, "proto": "lit", "n": n, "counts": counts, "max_len": a.max_len,
           "seed": a.seed, "smoke": a.smoke, "heads": H_, "kv_heads": KVH, "head_dim": d, "arms": list(arms),
           "tasks": list(tasks), "gold": GOLD, "tf": TF, "nope_layers": nope, "attn_layers": alist, "eos": sorted(eos), "md5": mods,
           "files": {"cal": md5f(CQ.CAL[0]), "haystack": md5f(HAYSTACK),
                     "ruler": {t: md5f(RULER_DIR / os.environ["RULER_TAG"] / t.partition("@")[2] / t[6:].partition("@")[0]
                                       / "validation.jsonl") for t in tasks if t.startswith("ruler:")},
                     "config": md5f(Path(a.model) / "config.json")},
           "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "transformers": __import__("transformers").__version__,
                        "device": torch.cuda.get_device_name(dev)}, "fp_vs_sdpa": None}
    # exact path against the model's own sdpa attention on the first item (prefill logits)
    x0 = items[0][2][:, :8192].to(dev)                                 # v3.4: first 8192 tokens (full item at 128K OOMs)
    with torch.no_grad():
        SDPA["off"] = True                                               # the v3.3 exact float32 path
        try:
            l_tb = model(input_ids=x0, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        finally:
            SDPA["off"] = False
        l_r = model(input_ids=x0, use_cache=False, logits_to_keep=1).logits[0, -1].float()   # v3.4 routed path
        model.set_attn_implementation("sdpa")
        if model.config._attn_implementation != "sdpa":
            raise SystemExit("could not switch to sdpa for the exact-path check")
        try:
            l_sd = model(input_ids=x0, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        finally:
            model.set_attn_implementation("tb_attn")
    res["fp_vs_sdpa"] = {"max_abs_logit_diff": float((l_tb - l_sd).abs().max()), "same_argmax": bool(l_tb.argmax() == l_sd.argmax()),
                         "check_len": int(x0.shape[1]), "routed_vs_sdpa_max_abs": float((l_r - l_sd).abs().max())}
    if float((l_r - l_sd).abs().max()) != 0.0:
        print("WARNING: routed sdpa path differs from the model's sdpa:", res["fp_vs_sdpa"], flush=True)
    # v3.4: one fp decode step, routed sdpa vs the v3.3 exact path, on the same 8192-token prefix cache
    with torch.no_grad():
        c_ = DynamicCache()
        model(input_ids=x0[:, :-1], past_key_values=c_, use_cache=True, logits_to_keep=1)
        X["arm"], X["expect"] = None, None
        l_dr = model(input_ids=x0[:, -1:], past_key_values=c_, use_cache=True).logits[0, -1].float()
        c_.crop(x0.shape[1] - 1)
        SDPA["off"] = True
        try:
            l_de = model(input_ids=x0[:, -1:], past_key_values=c_, use_cache=True).logits[0, -1].float()
        finally:
            SDPA["off"] = False
        c_ = None
    dd_ = float((l_dr - l_de).abs().max())
    res["fp_vs_sdpa"].update(decode_routed_vs_exact_max_abs=dd_, decode_same_argmax=bool(l_dr.argmax() == l_de.argmax()),
                             decode_vs_prefill_logits_max_abs=float((l_de - l_tb).abs().max()))
    print("fp decode step, sdpa vs exact:", dd_, flush=True)
    if not res["fp_vs_sdpa"]["decode_same_argmax"] or dd_ > 2.0:
        raise SystemExit(f"fp decode through sdpa disagrees with the exact path: {res['fp_vs_sdpa']}")
    res["v34"] = {"prefill": "sdpa", "fp_decode": "sdpa", "cache": "gpu, crop per arm, host restore for write-back arms"}
    print("exact path vs sdpa:", res["fp_vs_sdpa"], flush=True)
    t0 = time.time()
    errs, nerr = {}, {}
    if jl.exists() and jl.stat().st_size and not jl.read_bytes().endswith(b"\n"):
        with open(jl, "a", encoding="utf-8") as f_:                     # a line cut by a kill: do not glue onto it
            f_.write("\n")
    fh = open(jl, "a", encoding="utf-8")
    try:
        for t, i, ids, ref, allv, mx in items:
            todo = [arm for arm in arms if (arm, t, i) not in done]
            if not todo:
                continue
            x = ids.to(dev)
            L0 = x.shape[1]

            Q.reset() if hasattr(Q, "reset") else None                  # v3.3: no state from the previous item
            try:
                X["arm"], X["expect"] = None, None
                tp = time.time()
                torch.cuda.reset_peak_memory_stats(dev)                 # v3.5b: prefill peak per item
                cache0 = DynamicCache()
                with torch.no_grad():
                    out0 = model(input_ids=x, past_key_values=cache0, use_cache=True, logits_to_keep=1)
                first = int(out0.logits[0, -1].argmax())
                out0 = None
                prefill_peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30   # v3.5b: before the checksum pass
                cache_gpu, cache0 = cache0, None                         # v3.4: the prefill cache stays on the GPU
                fpr = cache_fingerprint(cache_gpu, L0)
                cache_host = cache_host_copy(cache_gpu) if any(writes_back(a_) for a_ in todo) else None
                prefill_sec = time.time() - tp
                torch.cuda.empty_cache()
            except Exception as e:                                       # noqa: BLE001
                for arm in todo:
                    fh.write(json.dumps({"arm": arm, "task": t, "i": i, "cfg": cfg, "error": f"prefill: {e!r}"}) + "\n")
                    errs[arm] = errs.get(arm, 0) + 1
                    nerr[arm] = nerr.get(arm, 0) + 1
                fh.flush()
                out0 = cache0 = cache_gpu = cache_host = None
                torch.cuda.empty_cache()
                if all(errs.get(a_, 0) >= 3 for a_ in todo):
                    raise SystemExit(f"3 consecutive prefill errors (last: {e!r})")
                continue
            gold_toks = gold_lp = None                                   # v3.7
            if GOLD and t.startswith("ruler:"):                          # v3.7: also when only fp is left (fp gold_nll)
                try:
                    X["arm"], X["expect"] = None, None
                    gold_toks = allv
                    if gold_toks is None or len(gold_toks) < 2 or len(gold_toks) > RULER_MAXGEN.get(t[6:].partition("@")[0], 128):   # within the length budget
                        raise ValueError(f"gold answer of {len(gold_toks)} tokens not usable")
                    gold_lp = forced_pass(model, cache_gpu, gold_toks, L0, dev, len(gold_toks) - 1)
                    if cache_restore(cache_gpu, L0, None, fpr):
                        raise SystemExit(f"item {t}#{i}: fp gold pass changed the prefill cache")
                except SystemExit:
                    raise
                except Exception as e:                                   # noqa: BLE001
                    print(f"fp gold pass failed for {t}#{i}: {e!r}", flush=True)
                    gold_toks = gold_lp = None
                    X["arm"], X["expect"] = None, None
                    if cache_restore(cache_gpu, L0, None, fpr):
                        raise SystemExit(f"item {t}#{i}: cache not restored after a failed fp gold pass")
                torch.cuda.empty_cache()
            ref_toks = ref_lp = None
            ref_mis = None
            if TF and any(a_ != "fp" for a_ in todo):                   # v3.5: the fp trajectory, once per item
                try:
                    X["arm"], X["expect"] = None, None
                    ref_toks, nxt_ = [], first
                    for step in range(mx):
                        if nxt_ in eos:
                            break
                        ref_toks.append(nxt_)
                        if step == mx - 1:
                            break
                        X["expect"] = L0 + step + 1
                        with torch.no_grad():
                            o_ = model(input_ids=torch.tensor([[nxt_]], device=dev), past_key_values=cache_gpu, use_cache=True)
                        nxt_ = int(o_.logits[0, -1].argmax())
                        o_ = None
                    X["expect"] = None
                    bad_r = cache_restore(cache_gpu, L0, None, fpr)
                    if bad_r:
                        raise SystemExit(f"item {t}#{i}: fp reference changed the prefill cache: {bad_r}")
                    ns_ = len(ref_toks) - 1                              # rows with a target inside ref_toks (no EOS row)
                    ref_lp = forced_pass(model, cache_gpu, ref_toks, L0, dev, ns_) if ns_ >= 1 else None
                    bad_r = cache_restore(cache_gpu, L0, None, fpr)
                    if bad_r:
                        raise SystemExit(f"item {t}#{i}: fp forced pass changed the prefill cache: {bad_r}")
                    if ref_lp:
                        ref_mis = sum(int(r_.argmax()) != ref_toks[j + 1] for j, r_ in enumerate(ref_lp))   # recorded, no abort
                    else:
                        ref_lp = None
                except SystemExit:
                    raise
                except Exception as e:                                   # noqa: BLE001
                    print(f"fp reference failed for {t}#{i}: {e!r}", flush=True)
                    ref_toks = ref_lp = None
                    X["arm"], X["expect"] = None, None
                    if cache_restore(cache_gpu, L0, None, fpr):
                        raise SystemExit(f"item {t}#{i}: cache not restored after a failed fp reference")
                torch.cuda.empty_cache()
            for arm in todo:
                ta = time.time()
                torch.cuda.reset_peak_memory_stats(dev)
                rec = {"arm": arm, "task": t, "i": i, "len": L0, "cfg": cfg, "hv": HV}
                try:
                    Q.reset() if hasattr(Q, "reset") else None
                    X["arm"], X["expect"] = None, None
                    cache = cache_gpu                                    # v3.4: decode on the prefill cache, cropped after
                    rec["prefill_sec"] = prefill_sec
                    rec["gpu_peak_prefill_gib"] = prefill_peak
                    X["arm"] = None if arm == "fp" else arm
                    toks, nxt, stop = [], first, "cap"
                    for step in range(mx):
                        if nxt in eos:
                            stop = "eos"
                            break
                        toks.append(nxt)
                        if step == mx - 1:
                            break    # v3.2: no content-based early stop (v3.1 cut answers); EOS or NIAH_MAXGEN only
                        X["expect"] = L0 + step + 1
                        with torch.no_grad():
                            o = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True)
                        nxt = int(o.logits[0, -1].argmax())
                    pred = tok.decode(toks, skip_special_tokens=True)
                    if t.startswith("lb:"):
                        rec["score"] = lb.score(t[3:], pred, ref)
                    elif t.startswith("ruler:"):
                        rec["score"] = ruler_score(pred, ref, t[6:].partition("@")[0])
                    else:
                        rec["score"], rec["exact_set"] = niah_score(pred, ref, allv)
                    rec.update(pred=pred, ntok=len(toks), hit_max=len(toks) == mx, stop=stop, sec=time.time() - ta)
                    if arm != "fp" and not BLQ.is_bl(arm):
                        rec["bits"] = list(Q.eff_bits(arm, L0, d, H_ // KVH))
                    rec["gpu_peak_free_gib"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
                    if not TF or arm == "fp" and ref_lp is None:
                        rec["tf_status"] = "off" if not TF else "no_reference"
                    elif ref_lp is None:
                        rec["tf_status"] = "no_reference"                # empty / 1-token fp trajectory or failed reference
                    elif arm == "fp":
                      try:                                               # v3.5b: guarded as the other arms
                        m_ = len(ref_lp)
                        rec.update(tf_status="ok", tf_steps=m_, tf_ref_mismatch=ref_mis,
                                   tf_nll=sum(float(-r_[ref_toks[j + 1]]) for j, r_ in enumerate(ref_lp)) / m_)
                      except Exception as e_tf:                          # noqa: BLE001
                        rec["tf_status"] = f"error: {e_tf!r}"[:300]
                    else:
                      try:                                               # v3.5b: forced-pass failures stay out of the score
                        X["arm"], X["expect"] = None, None
                        cache = o = None
                        Q.reset() if hasattr(Q, "reset") else None
                        bad_f = cache_restore(cache_gpu, L0, cache_host if writes_back(arm) else None, fpr)
                        if bad_f:
                            raise RuntimeError(f"cache restore before the forced pass: {bad_f}")
                        torch.cuda.reset_peak_memory_stats(dev)
                        X["arm"] = arm
                        st_ = forced_pass(model, cache_gpu, ref_toks, L0, dev, len(ref_lp), ref_lp)
                        X["arm"] = None
                        m_ = len(st_)
                        kl_ = [x_[0] for x_ in st_]
                        nll_ = sum(x_[2] for x_ in st_) / m_
                        fin_ = all(math.isfinite(v_) for v_ in kl_) and math.isfinite(nll_)
                        rec.update(tf_status="ok" if fin_ else "nonfinite", tf_steps=m_, tf_ref_mismatch=ref_mis,
                                   tf_kl_mean=sum(kl_) / m_ if fin_ else None, tf_kl_max=max(kl_) if fin_ else None,
                                   tf_top1=sum(x_[1] for x_ in st_) / m_, tf_nll=nll_ if fin_ else None,
                                   gpu_peak_tf_gib=torch.cuda.max_memory_allocated(dev) / 2 ** 30)
                        st_ = None
                      except Exception as e_tf:                          # noqa: BLE001
                        X["arm"], X["expect"] = None, None
                        st_ = None
                        for k_ in ("tf_kl_mean", "tf_kl_max", "tf_top1", "tf_nll", "tf_steps", "gpu_peak_tf_gib"):
                            rec.pop(k_, None)
                        rec["tf_status"] = f"error: {e_tf!r}"[:300]
                        torch.cuda.empty_cache()
                    if GOLD and t.startswith("ruler:") and gold_lp is None:
                        rec["gold_status"] = "unavailable"               # v3.7: recorded so resume does not loop
                    if GOLD and gold_lp is not None:                     # v3.7: gold-answer pass (failures stay out of the score)
                      try:
                        X["arm"], X["expect"] = None, None
                        cache = o = None
                        if arm == "fp":
                            m_ = len(gold_lp)
                            rec.update(gold_status="ok", gold_steps=m_,
                                       gold_nll=sum(float(-r_[gold_toks[j + 1]]) for j, r_ in enumerate(gold_lp)) / m_)
                        else:
                            Q.reset() if hasattr(Q, "reset") else None
                            bad_g = cache_restore(cache_gpu, L0, cache_host if writes_back(arm) else None, fpr)
                            if bad_g:
                                raise RuntimeError(f"cache restore before the gold pass: {bad_g}")
                            X["arm"] = arm
                            sg_ = forced_pass(model, cache_gpu, gold_toks, L0, dev, len(gold_lp), gold_lp)
                            X["arm"] = None
                            m_ = len(sg_)
                            klg = [x_[0] for x_ in sg_]
                            nllg = sum(x_[2] for x_ in sg_) / m_
                            fin_ = all(math.isfinite(v_) for v_ in klg) and math.isfinite(nllg)
                            rec.update(gold_status="ok" if fin_ else "nonfinite", gold_steps=m_,
                                       gold_kl_mean=sum(klg) / m_ if fin_ else None,
                                       gold_top1=sum(x_[1] for x_ in sg_) / m_, gold_nll=nllg if fin_ else None)
                            sg_ = None
                      except Exception as e_g:                           # noqa: BLE001
                        X["arm"], X["expect"] = None, None
                        for k_ in ("gold_kl_mean", "gold_top1", "gold_nll", "gold_steps"):
                            rec.pop(k_, None)
                        rec["gold_status"] = f"error: {e_g!r}"[:300]
                        torch.cuda.empty_cache()

                except Exception as e:                                   # noqa: BLE001
                    rec["error"] = repr(e)
                    rec["trace"] = traceback.format_exc()[-2000:]
                finally:
                    X["arm"], X["expect"] = None, None
                    cache = o = None
                    Q.reset() if hasattr(Q, "reset") else None
                    torch.cuda.empty_cache()
                bad_ = cache_restore(cache_gpu, L0, cache_host if writes_back(arm) else None, fpr)   # v3.4
                if bad_:
                    rec.pop("score", None)
                    rec["error"] = f"cache restore after arm: {bad_}"
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")   # written before any abort below
                fh.flush()
                if bad_:
                    raise SystemExit(f"arm {arm} item {t}#{i}: {bad_} (prefill cache not restored; run stopped)")
                if "error" in rec:
                    errs[arm] = errs.get(arm, 0) + 1
                    nerr[arm] = nerr.get(arm, 0) + 1
                    if errs[arm] >= 3:
                        raise SystemExit(f"arm {arm}: 3 consecutive item errors (last: {rec['error'][:200]})")
                else:
                    errs[arm] = 0
                if "error" not in rec:
                    done[(arm, t, i)] = rec
            cache_host = cache_gpu = ref_lp = ref_toks = gold_lp = gold_toks = None
            summ = {}
            common = {(t_, i_) for (a_, t_, i_) in done if all((b_, t_, i_) in done for b_ in arms)}
            res["n_common"] = len(common)
            for arm in arms:
                rs = [r for (a_, t_, i_), r in done.items() if a_ == arm and (t_, i_) in common]   # v3.3: paired items
                summ[arm] = {tt: (100 * sum(r["score"] for r in rs if r["task"] == tt) / max(1, sum(r["task"] == tt for r in rs))
                                  if any(r["task"] == tt for r in rs) else None) for tt in tasks}
                summ[arm]["n_done"] = len(rs)
                summ[arm]["n_done_total"] = sum(1 for (a_, _, _) in done if a_ == arm)
                summ[arm]["n_error"] = nerr.get(arm, 0)
                ex_ = [r["exact_set"] for r in rs if "exact_set" in r]
                summ[arm]["niah_exact_set"] = 100 * sum(ex_) / len(ex_) if ex_ else None
            res["summary"] = summ
            res["elapsed"] = time.time() - t0
            atomic_write(a.out, res)
    finally:
        fh.close()
    print(json.dumps(res.get("summary", {}))[:3000], f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
