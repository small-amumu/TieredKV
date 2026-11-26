# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""opx_skv (round OPX, REG_OPX.md; 2026-11-23) = sota_skv v3 (2026-11-08, X-0140) with:
  - qa_2 (RULER HotpotQA, maxgen 32, string_match_part) via opx_common;
  - decode metering: per-forward wall time and NVML PCIe bytes (opx_common.DecodeMeter);
  - host RSS after model load, after prefill + H2D, and at the end of decoding;
  - --timing: EOS ignored, TIMING_STEPS tokens per item, forwards 0-15 not metered, no score;
  - prefill time (prefill + H2D) recorded separately; no resume (an existing output file is refused);
  - timing mode also: raw per-forward times, and PROFILE_STEPS more forwards under torch.profiler after
    the timed ones (Chrome trace + operator table in <out>.traces/<task>_<i>.*).
ShadowKV is the official code, unmodified unless --fix_v_stride (one kernel argument, see patch_v_stride).
usage: CUDA_VISIBLE_DEVICES=<g> python opx_skv.py <model_dir> <ruler_dir_with_task_subdirs> <out.jsonl> --len 130000
       --tasks vt,cwe,fwe,qa_1,qa_2 --n 50 [--offset 0] [--sparse_budget 2048] [--fix_v_stride] [--timing]"""
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
import types
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SKV = os.environ.get("SKV_ROOT", os.path.expanduser("~/work/sota/ShadowKV"))
sys.path.insert(0, SKV)
if "minference" not in sys.modules:   # imported at module level by ShadowKV, used only with --minference (off)
    _mi = types.ModuleType("minference")

    def _no_minference(*args, **kwargs):
        raise RuntimeError("minference stub called: MInference is disabled in this evaluation")
    _mi.vertical_slash_sparse_attention = _mi.block_sparse_attention = _mi.streaming_forward = _no_minference
    _mi.__path__ = []
    _cfg = types.ModuleType("minference.configs")
    _cfg.__path__ = []
    _m2p = types.ModuleType("minference.configs.model2path")
    _m2p.MODEL2PATH = {}
    sys.modules.update({"minference": _mi, "minference.configs": _cfg,
                        "minference.configs.model2path": _m2p})
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


@torch.inference_mode()
def greedy(llm, ids: torch.Tensor, n: int, eos: set[int], force: bool, trace: str | None = None) -> dict:
    """official calls only: prefill -> H2D -> inference(next, get_ctx(next)); argmax."""
    peak_reset = reset_rss_peak()
    t0 = time.perf_counter()
    logits = llm.prefill(ids)
    t = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    llm.kv_cache.H2D()
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
        logits = llm.inference(input_ids=t, position_ids=llm.get_ctx(t))
        t = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        int(t[0, 0])                                                  # host sync, as the next check
        dm.after(f)
        f += 1
    res = dict(out=out, stop=stop, prefill_sec=pre, rss_after_prefill_gib=rss_pre,
               rss_end_gib=rss_gib(), forwards=f, rss_peak_gib=rss_peak_gib(),
               rss_peak_reset=peak_reset, **census, **dm.finish())
    if force and trace:                                               # profiled forwards, after the timed ones
        st = {"t": t}

        def fwd() -> None:
            lg = llm.inference(input_ids=st["t"], position_ids=llm.get_ctx(st["t"]))
            st["t"] = lg[:, -1, :].argmax(dim=-1, keepdim=True)
            int(st["t"][0, 0])
        res.update(profile_forwards(fwd, PROFILE_STEPS, trace))
    return res


def patch_v_stride() -> None:
    """X-0140 Part C amendment 4: ShadowKVCache_CPU.get_value_cache passes cpu_v_length =
    max_ctx_chunks_len * head_dim as the per-(batch, head) source stride of v_cache_cpu, whose real
    per-head stride is v_cache_cpu[layer].shape[-2] * shape[-1]; with max_length = len + 2048 >
    prompt length, heads >= 1 read V rows from the wrong offset. Only that argument is replaced."""
    import models.kv_cache as KC

    def get_value_cache(self, layer_idx, position_ids):
        vc = self.v_cache_cpu[layer_idx]
        KC.shadowkv.gather_copy_with_offsets(
            vc, self.v_cache_buffer[layer_idx], self.temp, self.offsets, self.cnts, self.signals,
            self.batch_size, self.num_key_value_heads, int(vc.shape[-2] * vc.shape[-1]),
            int(self.sparse_budget * self.head_dim), self.kernel_offset, self.kernel_stride,
            self.select_sets)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len
        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]
    get_value_cache._x0140_fix = True
    KC.ShadowKVCache_CPU.get_value_cache = get_value_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--len", type=int, required=True)
    ap.add_argument("--tasks", default="vt,cwe,fwe,qa_1,qa_2")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sparse_budget", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=160)
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--fix_v_stride", action="store_true")
    ap.add_argument("--method", default="shadowkv_cpu", choices=["shadowkv_cpu", "full"],
                    help="full = the official full-attention path (calibration at 32K only; 130K does not fit a 4090)")
    ap.add_argument("--pad", type=int, default=2048)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--trunc", type=int, default=0,
                    help="host-memory probe: BOS + last n-1 prompt tokens, never scored")
    ap.add_argument("--mem_probe", action="store_true", help="host-memory check: RSS split + trimmed RSS")
    a = ap.parse_args()
    MEMPROBE["on"] = a.mem_probe
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists; OPX runners do not resume (one file = one process)")
    method = a.method                                                 # shadowkv_cpu: V in pinned host memory
    if a.fix_v_stride and method != "shadowkv_cpu":
        raise SystemExit("--fix_v_stride patches ShadowKVCache_CPU only")
    if a.fix_v_stride:
        patch_v_stride()
    max_length = a.len + (TIMING_STEPS + EXTRA_STEPS if a.timing else 0) + a.pad
    tdir = Path(a.out).with_suffix(".traces")                        # timing mode: profiler traces
    if a.timing:
        tdir.mkdir(parents=True, exist_ok=False)
    tasks = a.tasks.split(",")
    data_md5 = {k: hashlib.md5((Path(a.data) / k / "validation.jsonl").read_bytes()).hexdigest()[:12]
                for k in tasks}
    commit = subprocess.run(["git", "-C", SKV, "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    import transformers
    from models import choose_model_class                            # official ShadowKV
    rss0 = rss_gib()
    LLM = choose_model_class(os.path.basename(a.model.rstrip("/")))
    llm = LLM(model_name=a.model, batch_size=1, device="cuda:0", max_length=max_length, attn_mode=method,
              dtype=torch.bfloat16, sparse_budget=a.sparse_budget, rank=a.rank, chunk_size=a.chunk_size,
              minference=False)
    tok = llm.tokenizer
    eos = eos_set(a.model, tok)
    kv = llm.kv_cache
    if a.fix_v_stride and not getattr(type(kv).get_value_cache, "_x0140_fix", False):
        raise SystemExit("fix_v_stride requested but the cache class in use is not patched")
    cfg = dict(method=method, fix_v_stride=a.fix_v_stride, sparse_budget=a.sparse_budget, rank=a.rank,
               chunk_size=a.chunk_size, max_length=max_length, timing=a.timing, skv_commit=commit,
               script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:8],
               common_md5=hashlib.md5((Path(HERE) / "opx_common.py").read_bytes()).hexdigest()[:8],
               meter_md5=hashlib.md5((Path(HERE) / "pcie_meter.py").read_bytes()).hexdigest()[:8],
               data_md5=data_md5, transformers=transformers.__version__, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0), model=os.path.basename(a.model.rstrip("/")),
               rss_before_load_gib=rss0, rss_after_load_gib=rss_gib(), eos=sorted(eos))
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
                rec: dict = {"task": kind, "i": i, "len": L0,
                             "ids_md5": hashlib.md5(ids.numpy().tobytes()).hexdigest(),
                             "method": method, "sparse_budget": a.sparse_budget, "fix_v_stride": a.fix_v_stride,
                             "timing": a.timing, "trunc": a.trunc}
                n = TIMING_STEPS if a.timing else RULER_MAXGEN[kind]
                for _attempt in (0, 1):
                    try:
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        g = greedy(llm, ids.to(llm.device), n, eos, a.timing,
                                   str(tdir / f"{kind}_{i}") if a.timing else None)
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
