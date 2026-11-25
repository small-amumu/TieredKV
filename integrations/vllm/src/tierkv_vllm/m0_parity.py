# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""M0 parity: research attention methods inside vLLM vs. the research harness, same process.

1. Hugging Face model (bf16) with the research attention: KIVI / main-line calibration on the
   calibration text exactly as sel29_lab does (capture q, k, v; Q.calib per layer;
   Q.calib_all).
2. Research logits on each evaluation window (seq tokens, rows >= --rows scored, dense prefill
   below): exact (run None) and every --runs method. Kept on the CPU for the scored rows only.
3. The HF model is freed; vLLM (single process, eager, bf16 cache, chunked prefill --chunk) runs
   the same token ids with backend TIERKV_M0: an exact pass, then one pass per method whose rows
   >= --rows go through the same research attend() on K / V read back from the paged cache
   (calibration state shared), then two planted-fault passes of --fault-run: fault_layer (layer
   index shifted by one: calibration of the wrong layer) and fault_key (the newest key cut from
   the gathered K / V: an off-by-one in the sequence length).
4. Per window and method: D_res = KL(exact_res || m_res), D_vllm = KL(exact_vllm || m_vllm), the
   engine gaps gap_exact = KL(exact_res || exact_vllm) and gap = KL(m_res || m_vllm), top-1
   agreement, and the oracle row count (must be layers x scored rows per pass). Verdicts are
   applied by the registered analyzer, not here.
Default methods: the deployed configuration h54 = stkm8v8128k64_x2s128_p50_h54 (stack base
~2.22 bit, M = 128 key rows, K = 64 value rows, 50 % debias, host residual 5-bit key / 4-bit value)
and its base alone (stkm8v8128). Byte / throughput numbers must name the host codec.
Usage: python -m tierkv_vllm.m0_parity --model DIR --research ROOT --out F.json [--runs A,B]
       [--fault-run A] [--windows 0:4096,1:4096] [--seq 4096] [--rows 2048] [--util 0.7]
--windows: comma list of <eval doc index>:<start> (CQ.EVAL order: 0 qasper, 1 mfqa).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

DEFAULT_METHOD = "stkm8v8128k64_x2s128_p50_h54"  # deployed default (h54)
DEFAULT_BASE = "stkm8v8128"  # its GPU base alone
RESEARCH_SUBDIRS = ("code", "rootcause", "rootcause/acs", "rootcause/spd", "rootcause/vfl")
FAULTS = (("fault_layer", 1, 0), ("fault_key", 0, 1))  # (tag, layer_shift, key_drop)


def kl_rows(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Per-row KL(softmax p || softmax q), computed in 512-row blocks on the CPU in float32."""
    out = []
    for r0 in range(0, p_logits.shape[0], 512):
        lp = torch.log_softmax(p_logits[r0 : r0 + 512].float(), -1)
        lq = torch.log_softmax(q_logits[r0 : r0 + 512].float(), -1)
        out.append((lp.exp() * (lp - lq)).sum(-1))
    return torch.cat(out)


def md5(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tierkv_vllm.m0_parity")
    ap.add_argument("--model", required=True)
    ap.add_argument("--research", required=True, help="root holding code/ and rootcause/")
    ap.add_argument("--runs", default=f"{DEFAULT_METHOD},{DEFAULT_BASE}", help="research names")
    ap.add_argument("--fault-run", default=DEFAULT_METHOD, help="method for the planted faults")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="0:4096,1:4096")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--util", type=float, default=0.7)
    ap.add_argument("--research-module", default="sel29_q", help="research entry module")
    ap.add_argument(
        "--research-float32",
        action="store_true",
        help="also run the research harness with the model in float32 (same calibration): the "
        "research-vs-research gap from input precision alone (VM0 cause split)",
    )
    a = ap.parse_args(argv)
    runs = [r for r in a.runs.split(",") if r]
    if a.fault_run and a.fault_run not in runs:
        raise SystemExit("--fault-run must be one of --runs")

    root = Path(a.research)
    for sub in reversed(RESEARCH_SUBDIRS):
        sys.path.insert(0, str(root / sub))
    import importlib

    import bl_q as BLQ
    import cq_lab3 as CQ

    Q = importlib.import_module(a.research_module)
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = "cuda"
    for r in runs:
        Q.parse(r)
    Q.CHUNK["rows"], BLQ.BL["chunk"] = a.chunk, a.chunk
    Q.PREFILL["on"], Q.PREFILL["from"] = True, a.rows
    X: dict = {"run": None, "cap": False, "kv": {}}

    def research_attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):  # type: ignore[no-untyped-def]  # noqa: E501
        li = int(module.layer_idx)
        q, k, v = query[0].float(), key[0].float(), value[0].float()
        scale = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
        if X["cap"]:
            X["kv"][li] = (k.clone(), v.clone(), q.clone())
        out = Q.attend(q, k, v, li, X["run"], scale, kw)
        return out.unsqueeze(0).transpose(1, 2).contiguous().to(query.dtype), None

    AttentionInterface.register("tierkv_research", research_attn)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, attn_implementation="tierkv_research"
        )
        .to(dev)
        .eval()
    )
    re_ = model.model.rotary_emb
    BLQ.BL["inv_freq"] = re_.inv_freq.detach().float()
    BLQ.BL["asc"] = float(getattr(re_, "attention_scaling", 1.0))
    Q.RC.R["inv_freq"], Q.RC.R["asc"] = BLQ.BL["inv_freq"], BLQ.BL["asc"]

    # 1. calibration, as sel29_lab
    X["cap"] = True
    try:
        with torch.no_grad():
            cal = CQ.text_ids(tok, CQ.CAL[0], CQ.CAL[1], CQ.CAL[2], dev)
            model(input_ids=cal, use_cache=False)
    finally:
        X["cap"] = False
    for li, t in X["kv"].items():
        Q.calib(li, *t)
    alist = sorted(X["kv"])
    nope = Q.calib_all(model, X["kv"], alist)
    X["kv"] = {}
    n_layer = len(alist)

    # 2. research logits
    windows = [tuple(int(x) for x in w.split(":")) for w in a.windows.split(",")]
    ids_w: list[torch.Tensor] = []
    res_logits: list[dict[str, torch.Tensor]] = []
    t0 = time.time()
    for di, st in windows:
        ids = CQ.text_ids(tok, CQ.EVAL[di][1], st, a.seq, dev)
        if ids.shape[1] != a.seq:
            raise SystemExit(f"window {di}:{st} has {ids.shape[1]} tokens")
        ids_w.append(ids[0].cpu())
        z: dict[str, torch.Tensor] = {}
        for run in [None, *runs]:
            X["run"] = run
            try:
                with torch.no_grad():
                    lg = model(input_ids=ids, use_cache=False).logits[0, a.rows :]
                    z[run or "exact"] = lg.float().cpu()
            finally:
                X["run"] = None
        res_logits.append(z)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # 2b. the same research runs with the model in float32 (calibration state unchanged)
    res32: list[dict[str, torch.Tensor]] = []
    if a.research_float32:
        model = (
            AutoModelForCausalLM.from_pretrained(
                a.model, dtype=torch.float32, attn_implementation="tierkv_research"
            )
            .to(dev)
            .eval()
        )
        for ids in ids_w:
            z = {}
            for run in [None, *runs]:
                X["run"] = run
                try:
                    with torch.no_grad():
                        lg = model(input_ids=ids[None].to(dev), use_cache=False).logits[0, a.rows :]
                        z[run or "exact"] = lg.float().cpu()
                finally:
                    X["run"] = None
            res32.append(z)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    t_res = time.time() - t0

    # 3. vLLM with the oracle backend
    import vllm.model_executor.layers.logits_processor as lpm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import tierkv_vllm
    from tierkv_vllm import m0_backend as mb

    tierkv_vllm.register()
    captured: list[torch.Tensor] = []
    orig = lpm.LogitsProcessor.forward

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = orig(self, *args, **kwargs)
        if out is not None and out.dim() == 2 and out.shape[0] > 1:
            captured.append(out.float().cpu())
        return out

    lpm.LogitsProcessor.forward = forward  # type: ignore[method-assign]
    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        attention_backend=mb.BACKEND_NAME,
        enforce_eager=True,
        max_model_len=a.seq + 64,
        max_num_batched_tokens=a.chunk,
        max_num_seqs=1,
        enable_prefix_caching=False,
        gpu_memory_utilization=a.util,
        seed=0,
    )
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
    Q.PREFILL["on"] = False  # the backend keeps rows < first exact itself
    mb.MODE["first"] = a.rows
    mb.MODE["n_layers"] = n_layer

    def vllm_logits(
        ids: torch.Tensor, run: str | None, shift: int = 0, drop: int = 0
    ) -> tuple[torch.Tensor, int]:
        mb.MODE["attend"] = Q.attend if run else None
        mb.MODE["name"] = run
        mb.MODE["layer_shift"], mb.MODE["key_drop"] = shift, drop
        rows0 = mb.MODE["rows"]
        captured.clear()
        try:
            llm.generate([TokensPrompt(prompt_token_ids=ids.tolist())], sp, use_tqdm=False)
        finally:
            mb.MODE["attend"], mb.MODE["layer_shift"], mb.MODE["key_drop"] = None, 0, 0
        z = torch.cat(captured, 0)
        if z.shape[0] not in (a.seq - 1, a.seq):
            raise SystemExit(f"captured {z.shape[0]} rows, expected {a.seq - 1} or {a.seq}")
        return z[a.rows :], mb.MODE["rows"] - rows0

    out: dict = {
        "model": a.model,
        "runs": runs,
        "fault_run": a.fault_run or None,
        "seq": a.seq,
        "rows": a.rows,
        "chunk": a.chunk,
        "dtype": "bfloat16",
        "attn_layers": n_layer,
        "nope_layers": nope,
        "secs_research": round(t_res, 1),
        "md5": {
            Path(p).name: md5(p)
            for p in (
                Q.__file__,
                *(
                    getattr(Q, n).__file__
                    for n in ("Q28", "Q25", "Q14", "S4", "Q5")
                    if hasattr(Q, n)
                ),
                BLQ.__file__,
                CQ.__file__,
                mb.__file__,
                __file__,
            )
        },
        "script_md5": md5(__file__),
        "windows": [],
    }
    t1 = time.time()
    expect = n_layer * (a.seq - a.rows)
    for wi, ((di, st), ids, zr) in enumerate(zip(windows, ids_w, res_logits, strict=True)):
        z32 = res32[wi] if res32 else None
        ve, _ = vllm_logits(ids, None)
        n = min(ve.shape[0], zr["exact"].shape[0])
        ve, re_exact = ve[:n], zr["exact"][:n]
        w: dict = {
            "doc": CQ.EVAL[di][0],
            "start": st,
            "n_rows": n,
            "gap_exact": float(kl_rows(re_exact, ve).mean()),
            "top1_exact": float((re_exact.argmax(-1) == ve.argmax(-1)).float().mean()),
            "methods": {},
        }
        if z32 is not None:
            w["gap_exact_res32"] = float(kl_rows(re_exact, z32["exact"][:n]).mean())
        arms = [(r, r, 0, 0) for r in runs]
        if a.fault_run:
            arms += [(tag, a.fault_run, s, d) for tag, s, d in FAULTS]
        for tag, run, shift, drop in arms:
            vm, oracle_rows = vllm_logits(ids, run, shift, drop)
            vm, rm = vm[:n], zr[run][:n]
            w["methods"][tag] = {
                "run": run,
                "layer_shift": shift,
                "key_drop": drop,
                "D_res": float(kl_rows(re_exact, rm).mean()),
                "D_vllm": float(kl_rows(ve, vm).mean()),
                "gap": float(kl_rows(rm, vm).mean()),
                "top1": float((rm.argmax(-1) == vm.argmax(-1)).float().mean()),
                "nonfinite": int((~torch.isfinite(vm)).sum()),
                "oracle_rows": oracle_rows,
                "oracle_rows_expected": expect,
            }
            if z32 is not None and shift == 0 and drop == 0:
                r32 = z32[run][:n]
                w["methods"][tag]["gap_res32"] = float(kl_rows(rm, r32).mean())
                w["methods"][tag]["D_res32"] = float(kl_rows(z32["exact"][:n], r32).mean())
        out["windows"].append(w)
        print("M0", json.dumps(w), flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    out["secs_vllm"] = round(time.time() - t1, 1)
    out["oracle_calls"] = mb.MODE["calls"]
    out["done"] = True
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
