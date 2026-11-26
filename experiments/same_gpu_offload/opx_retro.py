# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""opx_retro (round OPX, REG_OPX.md; 2026-11-23) = sota_retro (2026-11-08, X-0140) with:
  - qa_2 (RULER HotpotQA, maxgen 32, string_match_part) via opx_common;
  - decode metering: per-forward wall time and NVML PCIe bytes (opx_common.DecodeMeter);
  - host RSS after model load, after prefill + move + graph capture, and at the end of decoding;
  - --timing: EOS ignored, TIMING_STEPS tokens per item, forwards 0-15 not metered, no score;
  - prefill time (prefill_forward + move + capture_cuda_graph, i.e. index build included) recorded;
  - no resume (an existing output file is refused);
  - timing mode also: raw per-forward times, and PROFILE_STEPS more forwards under torch.profiler after
    the timed ones (Chrome trace + operator table in <out>.traces/<task>_<i>.*).
RetroInfer (microsoft/RetrievalAttention) is the official code, unmodified; the only deviation is the
prefill MLP chunk (memory only, row-independent: identical outputs), see patch_mlp_chunk.
The model dir must be named Llama-3.1-8B-Instruct (official config lookup = basename + '.json').
usage: CUDA_VISIBLE_DEVICES=<g> python opx_retro.py <model_dir> <ruler_dir> <out.jsonl> --len 130000
       --tasks vt,cwe,fwe,qa_1,qa_2 --n 50 [--offset 0] [--retrieval_budget 0.018] [--timing]"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RETRO = os.environ.get("RETRO_ROOT", os.path.expanduser("~/work/sota/RetrievalAttention"))
sys.path.insert(0, RETRO)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from opx_common import (ACC_WARM, END_TOKENS, EXTRA_STEPS, PROFILE_STEPS, RULER_MAXGEN, TIMING_STEPS,  # noqa: E402
                        TIMING_WARM, DecodeMeter, host_tensor_census, mem_probe_fields, MEMPROBE, stop_active_meters, profile_forwards, reset_rss_peak,
                        rss_gib, rss_peak_gib, ruler_score, truncate_ids)


def eos_set(model_dir: str, tok) -> set[int]:
    from transformers import GenerationConfig
    e: set = set()
    try:
        g = GenerationConfig.from_pretrained(model_dir).eos_token_id
        e |= set(g if isinstance(g, list) else [g])
    except Exception:                                                 # noqa: BLE001
        pass
    if tok.eos_token_id is not None:
        e.add(tok.eos_token_id)
    v = tok.get_vocab()
    e |= {v[t] for t in END_TOKENS if t in v}
    return {int(x) for x in e if x is not None}


def patch_mlp_chunk(chunk: int) -> None:
    """memory-only deviation (24 GB card): the official layer_prefill runs the prefill MLP in 65536-
    token chunks; the MLP is row-independent, so a smaller chunk gives identical outputs. The method is
    re-created from its own source with only that constant changed; the repository file is untouched."""
    import inspect
    import textwrap
    import model_hub.LLM as M
    src = textwrap.dedent(inspect.getsource(M.LLM.layer_prefill))
    assert src.count("65536") == 2, "unexpected layer_prefill source"
    ns: dict = {}
    exec(compile(src.replace("65536", str(int(chunk))), M.__file__, "exec"), M.__dict__, ns)  # noqa: S102
    M.LLM.layer_prefill = ns["layer_prefill"]


@torch.inference_mode()
def greedy(llm, ids: torch.Tensor, n: int, eos: set[int], attn: str, attn_config: dict,
           force: bool, trace: str | None = None) -> dict:
    peak_reset = reset_rss_peak()
    t0 = time.perf_counter()
    llm.attention_type = attn
    llm.batch_size, llm.input_length = ids.shape
    llm.max_new_length = n + (EXTRA_STEPS if force and trace else 0)
    llm.prefill_bsz, llm.prefill_method = 1, "full"
    llm.init_kv_cache(np.zeros(1, dtype=np.int64), attn_config)
    logits = llm.prefill_forward(inputs_ids=ids)
    t = llm.sampling(logits, do_sample=False)                         # argmax, [1, 1] int64
    llm.move()
    if attn == "RetroInfer":
        llm.kv_cache.capture_cuda_graph()
    torch.cuda.synchronize()
    pre = time.perf_counter() - t0
    rss_pre = rss_gib()
    census = {**host_tensor_census(), **mem_probe_fields()}                                     # after prefill, before decode
    dm = DecodeMeter(TIMING_WARM if force else ACC_WARM)
    out: list[int] = []
    stop, f = "cap", 0
    for step in range(n):
        if not force and int(t[0, 0]) in eos:
            stop = "eos"
            break
        out.append(int(t[0, 0]))
        if step == n - 1:
            break
        dm.before(f)
        t = llm.sampling(llm.decode_forward(inputs_ids=t), do_sample=False)
        int(t[0, 0])                                                  # host sync, as the next check
        dm.after(f)
        f += 1
    res = dict(out=out, stop=stop, prefill_sec=pre, rss_after_prefill_gib=rss_pre,
               rss_end_gib=rss_gib(), forwards=f, rss_peak_gib=rss_peak_gib(),
               rss_peak_reset=peak_reset, **census, **dm.finish())
    if force and trace:                                               # profiled forwards, after the timed ones
        st = {"t": t}

        def fwd() -> None:
            st["t"] = llm.sampling(llm.decode_forward(inputs_ids=st["t"]), do_sample=False)
            int(st["t"][0, 0])
        res.update(profile_forwards(fwd, PROFILE_STEPS, trace))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--len", type=int, required=True)
    ap.add_argument("--tasks", default="vt,cwe,fwe,qa_1,qa_2")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--attn", default="RetroInfer", choices=["RetroInfer", "Full_Flash_Attn"])
    ap.add_argument("--retrieval_budget", type=float, default=0.018)
    ap.add_argument("--estimation_budget", type=float, default=0.232)
    ap.add_argument("--cache_ratio", type=float, default=0.05)
    ap.add_argument("--mlp_chunk", type=int, default=16384)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--trunc", type=int, default=0,
                    help="host-memory probe: BOS + last n-1 prompt tokens, never scored")
    ap.add_argument("--mem_probe", action="store_true", help="host-memory check: RSS split + trimmed RSS")
    a = ap.parse_args()
    MEMPROBE["on"] = a.mem_probe
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists; OPX runners do not resume (one file = one process)")
    model = a.model.rstrip("/")
    if os.path.basename(model) != "Llama-3.1-8B-Instruct":
        raise SystemExit("model dir must be named Llama-3.1-8B-Instruct (official config lookup)")
    from config import generate_config
    from model_hub import load_model, load_tokenizer                 # official RetroInfer
    if a.mlp_chunk != 65536:
        patch_mlp_chunk(a.mlp_chunk)
    tasks = a.tasks.split(",")
    data_md5 = {k: hashlib.md5((Path(a.data) / k / "validation.jsonl").read_bytes()).hexdigest()[:12]
                for k in tasks}
    commit = subprocess.run(["git", "-C", RETRO, "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    import transformers
    rss0 = rss_gib()
    tok = load_tokenizer(model)                                       # official: pad = eos, left padding
    eos = eos_set(model, tok)
    llm = load_model(model, a.len + (TIMING_STEPS + EXTRA_STEPS if a.timing else 0) + 2, torch.bfloat16,
                     "cuda:0", tok)
    tdir = Path(a.out).with_suffix(".traces")                        # timing mode: profiler traces
    if a.timing:
        tdir.mkdir(parents=True, exist_ok=False)
    cfg = dict(attn=a.attn, mlp_chunk=a.mlp_chunk, retrieval_budget=a.retrieval_budget,
               estimation_budget=a.estimation_budget, cache_ratio=a.cache_ratio, timing=a.timing,
               retro_commit=commit, script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:8],
               common_md5=hashlib.md5((Path(HERE) / "opx_common.py").read_bytes()).hexdigest()[:8],
               meter_md5=hashlib.md5((Path(HERE) / "pcie_meter.py").read_bytes()).hexdigest()[:8],
               data_md5=data_md5, transformers=transformers.__version__, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0), rss_before_load_gib=rss0, rss_after_load_gib=rss_gib(),
               eos=sorted(eos))
    print("cfg", json.dumps(cfg), flush=True)
    nerr = 0
    with open(a.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": True, **cfg}) + "\n")
        for kind in tasks:
            rows = [json.loads(x) for x in open(Path(a.data) / kind / "validation.jsonl", encoding="utf-8")]
            rows = rows[a.offset:a.offset + a.n]
            for j, r in enumerate(rows):
                i = a.offset + j
                ids = tok(r["input"] + r.get("answer_prefix", ""), return_tensors="pt").input_ids
                if a.trunc:
                    ids = truncate_ids(ids, a.trunc)
                L0 = int(ids.shape[1])
                if not (4096 < L0 and L0 + RULER_MAXGEN[kind] <= a.len + 1):
                    fh.write(json.dumps({"task": kind, "i": i, "len": L0, "status": "skipped_length",
                                         "error": "prompt tokens outside (4096, len+1-maxgen]"}) + "\n")
                    fh.flush()
                    continue                                  # one item, not the whole arm (review m6)
                attn_config = generate_config(model, a.len, a.attn,   # official RULER scripts: nominal length
                                              a.retrieval_budget, a.estimation_budget, a.cache_ratio,
                                              False, False)
                rec: dict = {"task": kind, "i": i, "len": L0,
                             "ids_md5": hashlib.md5(ids.numpy().tobytes()).hexdigest(),
                             "retrieval_budget": a.retrieval_budget, "attn": a.attn, "timing": a.timing, "trunc": a.trunc,
                             "attn_config": attn_config.get(a.attn)}
                n = TIMING_STEPS if a.timing else RULER_MAXGEN[kind]
                for _attempt in (0, 1):
                    try:
                        llm.kv_cache = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        if hasattr(torch._C, "_host_emptyCache"):
                            torch._C._host_emptyCache()                   # pinned host blocks of the last item
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        g = greedy(llm, ids.to(llm.layers[0].device), n, eos, a.attn, attn_config,
                                   a.timing, str(tdir / f"{kind}_{i}") if a.timing else None)
                        out = g.pop("out")
                        pred = tok.decode(out, skip_special_tokens=True)
                        rec.update(g, pred=pred, ntok=len(out),
                                   score=None if a.timing or a.trunc else ruler_score(pred, r["outputs"], kind),
                                   gpu_peak_alloc_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                                   gpu_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                                   gpu_free_gib_after=torch.cuda.mem_get_info()[0] / 2 ** 30)
                        rec.pop("error", None)
                        rec.pop("tb", None)
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        rec["error"] = "OOM: " + repr(e)[:300]
                        rec["tb"] = traceback.format_exc()[-2000:]
                        stop_active_meters()                  # a decode that raised leaves its PCIe sampler running
                        llm.kv_cache = None
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception as e:                            # noqa: BLE001
                        msg = repr(e)
                        rec["error"] = msg[:500]
                        rec["tb"] = traceback.format_exc()[-2000:]
                        stop_active_meters()                  # a decode that raised leaves its PCIe sampler running
                        if any(s in msg for s in ("CUDA error", "device-side assert", "illegal memory")):
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fh.flush()
                            raise SystemExit(f"sticky CUDA error at {kind} {i}: {msg[:200]}")
                        break
                nerr = nerr + 1 if "error" in rec else 0
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                # no score on stdout: logs are read while monitoring, scores stay blind (issue 172)
                print(kind, i, L0, rec.get("ntok"), rec.get("decode_ms_median"),
                      rec.get("pcie_rx_bytes_per_step"), rec.get("error", ""), flush=True)
                if nerr >= 3:
                    raise SystemExit("3 consecutive errors: abort")
        fh.write(json.dumps({"meta": True, "end": True}) + "\n")


if __name__ == "__main__":
    main()
