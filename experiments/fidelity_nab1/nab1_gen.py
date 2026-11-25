# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""NAB1 step 1: 16-bit reasoning traces for fixed-trace replay (REG_NAB1.md).

One model per call. MATH-500 items [start, start + n) in dataset order; Qwen3 chat template with thinking on; the user
message is the problem plus the CH4X suffix. Sampling T 0.6, top-p 0.95, top-k 20 (Qwen3 thinking defaults), at most
--max_new new tokens, batches of --batch items in item order, torch seed 1000 + first item index of the batch.
The answer is scored with the CH4X boxed-answer scorer (lc_reason_range2.py last_boxed / norm, copied unchanged).
Output: one JSON line per item with the token ids, the position right after "</think>" (think_end, None if absent),
the scored answer and the 16-bit correctness. A journal makes the run resumable per batch.

usage: python nab1_gen.py <model_dir> <out_jsonl> --data <math500 test.jsonl> [--start 0] [--n 64] [--batch 8]
       [--max_new 8192] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke != a.out.endswith(".smoke.jsonl"):
        raise SystemExit("smoke output must end with .smoke.jsonl (and only smoke)")
    rows = [json.loads(line) for line in Path(a.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    items = rows[a.start:a.start + a.n]
    if len(items) != a.n:
        raise SystemExit(f"dataset has {len(items)} items in range, want {a.n}")
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    think_id = tok.convert_tokens_to_ids("</think>")
    if not isinstance(think_id, int) or think_id == tok.unk_token_id:
        raise SystemExit("tokenizer has no </think> token")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    data_md5 = hashlib.md5(Path(a.data).read_bytes()).hexdigest()
    if not a.smoke and not data_md5.startswith("d06a3ab6"):
        raise SystemExit(f"data md5 {data_md5} is not the registered d06a3ab6")
    meta = {"model": a.model, "max_new": 16 if a.smoke else a.max_new, "batch": a.batch, "temperature": 0.6,
            "top_p": 0.95, "top_k": 20, "smoke": a.smoke,
            "script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(), "data_md5": data_md5,
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0)}
    bind = ("model", "max_new", "batch", "temperature", "top_p", "top_k", "smoke", "script_md5", "data_md5")
    out = Path(a.out)
    done = set()
    if out.exists():
        raw = out.read_bytes()
        for line in raw.splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:                   # a torn last line from a kill
                continue
            if any(r["meta"].get(k) != meta[k] for k in bind):
                raise SystemExit(f"{out} holds item {r['i']} from another configuration; refusing to resume")
            done.add(r["i"])
        if raw and not raw.endswith(b"\n"):                 # terminate a torn last line before appending
            with open(out, "a", encoding="utf-8") as f:
                f.write("\n")
    t0 = time.time()
    for b0 in range(0, a.n, a.batch):
        idx = [a.start + k for k in range(b0, min(a.n, b0 + a.batch)) if a.start + k not in done]
        if not idx:
            continue
        prompts = [tok.apply_chat_template([{"role": "user", "content": rows[i]["problem"] + SUFFIX}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.manual_seed(1000 + a.start + b0)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                 max_new_tokens=16 if a.smoke else a.max_new, pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]
        with open(out, "a", encoding="utf-8") as f:
            for r, i in enumerate(idx):
                p_ids = enc.input_ids[r][enc.attention_mask[r].bool()].tolist()      # prompt without left padding
                g = gen[r, plen:].tolist()
                cut = next((j for j, t in enumerate(g) if t in eos_ids), None)
                ended = cut is not None
                g = g[:cut] if ended else g
                te = next((j for j, t in enumerate(g) if t == think_id), None)
                think_end = None if te is None else len(p_ids) + te + 1             # first answer position
                ans = tok.decode(g[te + 1:]) if te is not None else ""
                pred = last_boxed(ans) if te is not None else None
                ok = pred is not None and norm(pred) == norm(rows[i]["answer"])
                rec = {"i": i, "prompt_ids": p_ids, "gen_ids": g, "think_end": think_end, "ended": ended,
                       "n_gen": len(g), "pred": pred, "gold": rows[i]["answer"], "ok": ok,
                       "seed": 1000 + a.start + b0, "batch_items": idx, "meta": meta}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
        print(f"batch {b0}: items {idx[0]}-{idx[-1]} ({time.time() - t0:.0f}s)", flush=True)
    print("DONE nab1_gen", a.model, f"s{a.start}n{a.n}", flush=True)


if __name__ == "__main__":
    main()
