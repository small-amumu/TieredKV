# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""T3: near-data CPU-side residual correction microbenchmark (PLAN_RP.md section 4; RFP option 1).

Instead of the GPU reading 192 random residual rows per (layer, sequence, KV head) over PCIe, the CPU next to host
memory decodes them and returns small results, in two round trips per layer (RFP section on near-data CPU correction):
  round 1: GPU -> CPU rotated q (G x D fp16) + 128 key-candidate ids; CPU -> GPU corrected logits q . r_k (G x 128)
  round 2: GPU -> CPU 64 value ids + final probabilities (G x 64);      CPU -> GPU sum_j p_j r_v,j (G x D)
Shapes: Llama-3.1-8B, 32 layers, 8 KV heads, G = 4 query heads per KV head, D = 128; residual codec h54 (keys 5 bits,
values 4 bits per channel) with per-row (min, step) fp16. Synthetic data; the bit packing is a plain little-endian
bit stream (not kalign's word layout): cost and numerics of decode + dot are what is measured.

Modes:
  numeric  (a) CPU fp32 vs GPU results of the same decode + products; noise floor = change of the GPU fp32 result
           under a 1e-7 relative perturbation of q / p (QBANK rule 213: pass line 1.5 x noise floor).
  cpu      (b) per-layer CPU time of both rounds for B sequences (gather ids from a host region of L tokens, decode,
           batched products), torch intra-op threads swept; effective GMAC/s with the RFP's MAC count.
  rtt      (c) GPU <-> CPU round trip through pinned shared memory: a one-program GPU kernel writes the request
           payload and a flag (release, system scope), then spins (acquire, system scope, bounded) for the reply flag;
           a separate responder PROCESS (no GIL contention) polls the request flag, optionally computes, writes the
           reply payload and flag. Eager launches and a CUDA graph of 32 kernels (one decode step's layers).
           The Python responder makes these numbers an upper bound for a C++ worker.
Output: JSON lines (kind = env | numeric | cpu | rtt | error | end) + <out>.summary.json for rq check_done.
usage: python -B rp_cpu.py <out.jsonl> --modes numeric,cpu,rtt [--Bs 1,8,32] [--threads 1,4,16,32,64]
       [--L 32768] [--rtt_n 2000] [--selftest]
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

NL, KVH, G, D = 32, 8, 4, 128
M_K, M_V = 128, 64
BITS_K, BITS_V = 5, 4
SHM_FLAGS = 4096              # bytes reserved for the two flags (each on its own 64-byte line)


# ---------------------------------------------------------------------------------------------- codec
def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """codes uint8 [R, D] (< 2**bits) -> int32 words [R, 4 * bits]: little-endian bit stream, channel c at bits
    [c * bits, (c + 1) * bits)."""
    R = codes.shape[0]
    bitmat = ((codes[:, :, None] >> np.arange(bits, dtype=np.uint8)[None, None, :]) & 1).reshape(R, D * bits)
    by = np.packbits(bitmat.astype(np.uint8), axis=1, bitorder="little")
    return np.ascontiguousarray(by).view(np.int32)


def _tables(bits: int, dev):
    import torch
    c = torch.arange(D, device=dev)
    bi = (c * bits) // 8
    sh = (c * bits) % 8
    return bi, sh, (1 << bits) - 1


def decode(words, meta, bits: int, tab):
    """int32 words [N, 4 bits] + 16-bit (min, step) [N, 2] -> fp32 [N, D] = code * step + min (CPU or GPU)."""
    import torch
    bi, sh, mask = tab
    by = words.contiguous().view(torch.uint8)
    by = torch.cat([by, torch.zeros(by.shape[0], 1, dtype=torch.uint8, device=by.device)], dim=1).to(torch.int32)
    v = ((by[:, bi] | (by[:, bi + 1] << 8)) >> sh) & mask
    m = meta.view(torch.float16).float()
    return v.float() * m[:, 1:2] + m[:, 0:1]


def load_ext():
    """the C++ / OpenMP kernels of rp_cpu_ext.cpp (round1, round2, set_threads), built once and cached."""
    from torch.utils.cpp_extension import load
    src = Path(__file__).resolve().parent / "rp_cpu_ext.cpp"
    # explicit ISA flags (not -march=native): the build cache is keyed on the flag string, not on the host CPU
    return load(name="rp_cpu_ext", sources=[str(src)],
                extra_cflags=["-O3", "-mavx2", "-mfma", "-mf16c", "-fopenmp"], extra_ldflags=["-fopenmp"],
                verbose=False)


def make_rows(n: int, bits: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n residual records: words int32 [n, 4 bits], meta fp16-as-int16 [n, 2] (min ~ N(0, 0.02), step 1e-3-1e-2)."""
    codes = rng.integers(0, 1 << bits, size=(n, D), dtype=np.uint8)
    w = pack(codes, bits)
    meta = np.empty((n, 2), dtype=np.float16)
    meta[:, 0] = rng.normal(0, 0.02, n).astype(np.float16)
    meta[:, 1] = rng.uniform(1e-3, 1e-2, n).astype(np.float16)
    return w, meta.view(np.int16)


# ---------------------------------------------------------------------------------------------- (a) numeric
def run_numeric(fo, args) -> int:
    import torch
    rng = np.random.default_rng(args.seed)
    BH = 8 * KVH
    kw, km = make_rows(BH * M_K, BITS_K, rng)
    vw, vm = make_rows(BH * M_V, BITS_V, rng)
    q = torch.from_numpy(rng.normal(0, 1, (BH, G, D)).astype(np.float16))
    p = torch.softmax(torch.from_numpy(rng.normal(0, 2, (BH, G, M_V)).astype(np.float32)), -1).half()
    scale = 1.0 / math.sqrt(D)

    def correction(dev, qq, pp, tf32: bool):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        rk = decode(torch.from_numpy(kw).to(dev), torch.from_numpy(km).to(dev), BITS_K,
                    _tables(BITS_K, dev)).view(BH, M_K, D)
        rv = decode(torch.from_numpy(vw).to(dev), torch.from_numpy(vm).to(dev), BITS_V,
                    _tables(BITS_V, dev)).view(BH, M_V, D)
        s = torch.bmm(qq.to(dev).float(), rk.transpose(1, 2)) * scale
        o = torch.bmm(pp.to(dev).float(), rv)
        return s.cpu().double(), o.cpu().double()

    def err(a, b):
        d = (a - b).abs()
        rel = d.amax(-1) / b.abs().amax(-1).clamp_min(1e-30)
        return {"max_abs": float(d.max()), "max_rel_rowwise": float(rel.max())}

    s_cpu, o_cpu = correction("cpu", q, p, False)
    s_g, o_g = correction("cuda", q, p, False)
    s_t, o_t = correction("cuda", q, p, True)
    ext = load_ext()
    ext.set_threads(8)
    ids_k = torch.arange(BH * M_K, dtype=torch.int64).view(BH, M_K)
    ids_v = torch.arange(BH * M_V, dtype=torch.int64).view(BH, M_V)
    s_x = ext.round1(torch.from_numpy(kw), torch.from_numpy(km), q.float().contiguous(), ids_k, BITS_K,
                     scale).double()
    o_x = ext.round2(torch.from_numpy(vw), torch.from_numpy(vm), p.float().contiguous(), ids_v, BITS_V).double()
    eps = 1e-7                       # perturbation applied in fp32 (an fp16 round trip would erase it)
    qp = q.float() * (1 + eps * torch.from_numpy(rng.choice([-1.0, 1.0], q.shape).astype(np.float32)))
    pp = p.float() * (1 + eps * torch.from_numpy(rng.choice([-1.0, 1.0], p.shape).astype(np.float32)))
    s_n, o_n = correction("cuda", qp, pp, False)
    s_r, o_r = correction("cuda", q, p, False)                   # plain rerun (GPU non-determinism)
    floor = {"s": max(err(s_n, s_g)["max_rel_rowwise"], err(s_r, s_g)["max_rel_rowwise"]),
             "o": max(err(o_n, o_g)["max_rel_rowwise"], err(o_r, o_g)["max_rel_rowwise"])}
    rec = {"kind": "numeric", "heads": BH,
           "logits_cpu_vs_gpu_fp32": err(s_cpu, s_g), "logits_gpu_tf32_vs_fp32": err(s_t, s_g),
           "values_cpu_vs_gpu_fp32": err(o_cpu, o_g), "values_gpu_tf32_vs_fp32": err(o_t, o_g),
           "logits_cpp_vs_gpu_fp32": err(s_x, s_g), "values_cpp_vs_gpu_fp32": err(o_x, o_g),
           "logits_cpp_vs_torch_cpu": err(s_x, s_cpu), "values_cpp_vs_torch_cpu": err(o_x, o_cpu),
           "fp16_reply_diff_frac_cpp_vs_gpu": {"s": float((s_x.half() != s_g.half()).double().mean()),
                                               "o": float((o_x.half() != o_g.half()).double().mean())},
           "noise_floor_rel": floor,
           "cpp_within_1p5_noise_floor": {"s": err(s_x, s_g)["max_rel_rowwise"] <= 1.5 * floor["s"],
                                          "o": err(o_x, o_g)["max_rel_rowwise"] <= 1.5 * floor["o"]},
           # calibration of the same gate on the torch CPU path (fp32 summation order differs from the GPU's too)
           "torch_cpu_within_1p5_noise_floor": {"s": err(s_cpu, s_g)["max_rel_rowwise"] <= 1.5 * floor["s"],
                                                "o": err(o_cpu, o_g)["max_rel_rowwise"] <= 1.5 * floor["o"]},
           "fp16_reply_rel": {"s": float(((s_cpu.half().double() - s_cpu).abs().amax(-1)
                                           / s_cpu.abs().amax(-1)).max()),
                              "o": float(((o_cpu.half().double() - o_cpu).abs().amax(-1)
                                          / o_cpu.abs().amax(-1)).max())}}
    rec["note"] = ("noise_floor_rel = GPU fp32 change under a 1e-7 relative fp32 perturbation of q / p, or a plain "
                   "rerun (the larger); fp16_reply_rel = the reply's own fp16 rounding (each path rounds its fp32 "
                   "result to fp16 before sending). Read *_cpp_vs_gpu_fp32 against both: a CPU-GPU difference far "
                   "below fp16_reply_rel cannot change what the GPU receives except at rounding boundaries")
    torch.backends.cuda.matmul.allow_tf32 = False
    fo.write(json.dumps(rec) + "\n")
    fo.flush()
    print(f"numeric: logits torch-cpu vs gpu rel {rec['logits_cpu_vs_gpu_fp32']['max_rel_rowwise']:.2e}, cpp vs gpu "
          f"{rec['logits_cpp_vs_gpu_fp32']['max_rel_rowwise']:.2e}; values torch-cpu "
          f"{rec['values_cpu_vs_gpu_fp32']['max_rel_rowwise']:.2e}, cpp {rec['values_cpp_vs_gpu_fp32']['max_rel_rowwise']:.2e}"
          f"; floor {floor}; fp16 reply {rec['fp16_reply_rel']}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- (b) cpu
def _tiled(block: np.ndarray, n: int) -> np.ndarray:
    """n rows made by repeating a random packed block (contents do not affect the cost; memory stays bounded)."""
    reps = -(-n // block.shape[0])
    return np.ascontiguousarray(np.tile(block, (reps, 1))[:n])


def run_cpu(fo, args) -> int:
    """per-layer CPU time of both rounds. Every layer has its own host region (as the engine: 32 layers), and
    every (layer, step) uses fresh random ids, so rows come from DRAM unless the regions fit in the LLC
    (the record carries the footprint and the LLC size)."""
    import torch
    rng = np.random.default_rng(args.seed + 1)
    mac_per_head = M_K * (D + G * D) + M_V * (D + G * D)      # RFP count: decode as D MAC per row + products
    kblk = make_rows(65536, BITS_K, rng)
    vblk = make_rows(65536, BITS_V, rng)
    # main cases, then one cache-resident control (B 8, L 512, one region ~5 MB): if the main cases were not bound by
    # memory, their time would be close to the control's
    cases = [(B, args.L, args.layer_regions, args.threads, False) for B in args.Bs]
    if args.cache_control:
        cases.append((8, 512, 1, [t for t in args.threads if t == 16] or args.threads[:1], True))
    skipped: list[int] = []
    for B, L, nreg, thr, control in cases:
        BH = B * KVH
        need = nreg * BH * L * (16 * BITS_K + 4 + 16 * BITS_V + 4)
        avail = None
        try:
            avail = next(int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
        except (OSError, StopIteration, ValueError):
            pass
        if avail is not None and need > 0.8 * avail:
            fo.write(json.dumps({"kind": "error", "mode": "cpu", "B": B, "trace": f"skipped: regions need {need} B, "
                                 f"MemAvailable {avail} B"}) + "\n")
            fo.flush()
            print(f"cpu B{B}: skipped, regions need {need / 1e9:.1f} GB > 80 % of available", flush=True)
            skipped.append(B)
            continue
        regions = []
        for _ in range(nreg):
            regions.append(tuple(torch.from_numpy(_tiled(x, BH * L)) for x in (*kblk, *vblk)))
        footprint = sum(t.numel() * t.element_size() for r in regions for t in r)
        g = torch.Generator().manual_seed(args.seed + B)
        base = (torch.arange(BH) * L)[:, None]

        def fresh_ids() -> tuple[torch.Tensor, torch.Tensor]:
            ks = torch.randint(0, L, (BH, M_K), generator=g)            # with replacement: cost is the same
            vs = ks.gather(1, torch.randint(0, M_K, (BH, M_V), generator=g))
            return (ks + base).contiguous(), (vs + base).contiguous()
        # distinct decode steps of ids: at least --cpu_steps_max_ids, and enough that the rows touched by one pass
        # over the pool exceed 2 x LLC (issue 224: a small pool kept B=1 rows in the L3); timing never reuses ids
        llc = _llc_bytes() or 256 * 2**20
        bytes_per_layer_set = BH * (M_K * (16 * BITS_K + 4) + M_V * (16 * BITS_V + 4))
        # 2.5 x LLC: value ids are drawn from the key ids, so distinct value rows are ~80 % of the nominal count
        steps_ids = max(args.cpu_steps_max_ids, math.ceil(2.5 * llc / (NL * bytes_per_layer_set)))
        flush = np.ones(int(2.5 * llc) // 8, dtype=np.int64)       # streamed before each timed loop
        n_sets = NL * steps_ids
        id_pool = [fresh_ids() for _ in range(n_sets)]             # created before any timing
        warm_ids = [fresh_ids() for _ in range(NL)]                # warm-up only, never timed
        touched = n_sets * bytes_per_layer_set
        q = torch.randn(BH, G, D).half()
        p = torch.softmax(torch.randn(BH, G, M_V), -1).half()
        tk, tv = _tables(BITS_K, "cpu"), _tables(BITS_V, "cpu")
        scale = 1.0 / math.sqrt(D)
        ext = load_ext()
        for impl, th in [(i, t) for i in args.impls for t in thr]:
            torch.set_num_threads(th)              # also used by the fp16 <-> fp32 conversions of the cpp path
            if impl == "cpp":
                ext.set_threads(th)

            def layer(li: int, si: int) -> tuple[float, float]:
                kw_, km_, vw_, vm_ = regions[li % nreg]
                ki, vi = warm_ids[li] if si < 0 else id_pool[(si * NL + li) % n_sets]
                t0 = time.perf_counter()
                if impl == "torch":
                    kf = ki.reshape(-1)
                    rk = decode(kw_.index_select(0, kf), km_.index_select(0, kf), BITS_K, tk).view(BH, M_K, D)
                    s = torch.bmm(q.float(), rk.transpose(1, 2)).mul_(scale).half()
                else:
                    s = ext.round1(kw_, km_, q.float(), ki, BITS_K, scale).half()
                t1 = time.perf_counter()
                if impl == "torch":
                    vf = vi.reshape(-1)
                    rv = decode(vw_.index_select(0, vf), vm_.index_select(0, vf), BITS_V, tv).view(BH, M_V, D)
                    o = torch.bmm(p.float(), rv).half()
                else:
                    o = ext.round2(vw_, vm_, p.float(), vi, BITS_V).half()
                t2 = time.perf_counter()
                if s.shape != (BH, G, M_K) or o.shape != (BH, G, D):
                    raise RuntimeError(f"shapes {tuple(s.shape)} {tuple(o.shape)}")
                return t1 - t0, t2 - t1

            for i in range(NL):                                # warm-up: one decode step on dedicated ids
                layer(i, -1)
            r1, r2 = [], []
            nstep = max(args.cpu_steps_min, 1)
            int(flush.sum())                    # evict region bytes left in the LLC by the fill and the warm-up
            t_start = time.perf_counter()
            si = 0
            while si < steps_ids and (si < nstep or time.perf_counter() - t_start < args.t_min):
                for li in range(NL):
                    a, b = layer(li, si)
                    r1.append(a)
                    r2.append(b)
                si += 1
            tot = sorted(a + b for a, b in zip(r1, r2))
            r1.sort()
            r2.sort()
            med = lambda x: x[len(x) // 2]  # noqa: E731
            per_layer = med(tot)
            rec = {"kind": "cpu", "impl": impl, "B": B, "L": L, "threads": th, "heads": BH, "layers_timed": len(r1),
                   "cache_control": control,
                   "steps": si, "steps_distinct_ids": steps_ids,     # timing stops at steps_ids: no id reuse
                   "pool_touched_bytes": touched, "timed_touched_bytes": si * NL * bytes_per_layer_set,
                   "llc_flushed_before_timing": True,
                   "t_reached": time.perf_counter() - t_start >= args.t_min,
                   "round1_us_med": med(r1) * 1e6, "round2_us_med": med(r2) * 1e6,
                   "layer_us_med": per_layer * 1e6, "step_ms_est": per_layer * NL * 1e3,
                   "layer_us_q1": tot[len(tot) // 4] * 1e6, "layer_us_q3": tot[3 * len(tot) // 4] * 1e6,
                   "gmac_per_s": BH * mac_per_head / per_layer / 1e9, "loadavg_after": os.getloadavg(),
                   "regions": nreg, "footprint_bytes": footprint, "llc_bytes": _llc_bytes(),
                   "note": "layer_us_med = median over all timed (layer, step) of round1 + round2 (not the sum of "
                           "the two round medians); times include the fp16->fp32 of q / p and the fp16 reply "
                           "conversion, for both impls; rows mostly from DRAM when footprint >> LLC; layers share "
                           "regions when regions < 32"}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(f"cpu {impl} B{B}{' CONTROL(cache)' if control else ''} threads {th}: layer "
                  f"{rec['layer_us_med']:.1f} us (r1 {rec['round1_us_med']:.1f}, "
                  f"r2 {rec['round2_us_med']:.1f}), {rec['gmac_per_s']:.1f} GMAC/s", flush=True)
        del regions, id_pool, flush
    if skipped:                                  # the mode (and the summary) must not read as complete
        raise RuntimeError(f"cpu cases skipped for memory: B {skipped}")
    return 0


def _llc_bytes() -> int | None:
    """total last-level cache size from sysfs (sum over distinct L3 instances), or None."""
    try:
        seen, tot = set(), 0
        for d in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cache/index3"):
            ids = (d / "shared_cpu_list").read_text().strip()
            if ids in seen:
                continue
            seen.add(ids)
            s = (d / "size").read_text().strip()
            mult = {"K": 2**10, "M": 2**20, "G": 2**30}
            if s.isdigit():
                tot += int(s)
            elif s[-1] in mult and s[:-1].isdigit():
                tot += int(s[:-1]) * mult[s[-1]]
            else:
                return None
        return tot or None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------- (c) rtt
def _responder(path: str, nbytes: int, req_off: int, req_n: int, resp_off: int, resp_n: int, compute: str,
               B: int, parent: int) -> None:
    """separate process: poll the request flag, copy the request, optionally compute, write the reply, set the
    reply flag. Writes the ready flag (offset 128) once set up; exits when the request flag is -1 or the parent
    process is gone (checked every ~1e5 polls)."""
    mm = np.memmap(path, dtype=np.uint8, mode="r+", shape=(nbytes,))
    req_flag = mm[0:4].view(np.int32)
    resp_flag = mm[64:68].view(np.int32)
    ready = mm[128:132].view(np.int32)
    req = mm[req_off:req_off + req_n]
    resp = mm[resp_off:resp_off + resp_n]
    work = None
    if compute == "cpu":                                          # round-1 work with the C++ kernel
        import torch
        ext = load_ext()                                          # built by the parent before this process starts
        ext.set_threads(int(os.environ.get("RP_RESP_THREADS", "16")))
        BH = B * KVH
        # key region and id pool both > 2.5 x LLC, ids rotated per request: rows come from DRAM, as in the engine
        llc = _llc_bytes() or 256 * 2**20
        rows_per_head = max(32768, math.ceil(2.5 * llc / (BH * (16 * BITS_K + 4))))
        rng = np.random.default_rng(5)
        kblk = make_rows(65536, BITS_K, rng)
        kw = torch.from_numpy(_tiled(kblk[0], BH * rows_per_head))
        km = torch.from_numpy(_tiled(kblk[1], BH * rows_per_head))
        nsets = max(64, math.ceil(2.5 * llc / (BH * M_K * (16 * BITS_K + 4))))
        g = torch.Generator().manual_seed(5)
        base = (torch.arange(BH) * rows_per_head)[:, None]
        pool = [(torch.randint(0, rows_per_head, (BH, M_K), generator=g) + base).contiguous() for _ in range(nsets)]
        work = [kw, km, ext, BH, pool, 0]
    ready[0] = 1
    last = 0
    polls = 0
    while True:
        s = int(req_flag[0])
        if s == -1:
            break
        if s == last:
            polls += 1
            if polls >= 100_000:
                polls = 0
                if os.getppid() != parent:
                    break
            continue
        last = s
        buf = np.array(req, copy=True)                            # read the request payload
        if work is not None:                                      # round-1 sized work: decode 128 rows per head + dot
            import torch
            kw_, km_, ext, BH, pool, k = work
            ki = pool[k % len(pool)]                              # fresh ids per request (the request's own ids are
            work[5] = k + 1                                       # random bytes in this synthetic payload)
            qq = torch.from_numpy(buf[: BH * G * D * 2].copy()).view(torch.float16).view(BH, G, D).float()
            qq = torch.nan_to_num(qq, nan=0.0, posinf=0.0, neginf=0.0).contiguous()   # request bytes are random
            out = ext.round1(kw_, km_, qq, ki, BITS_K, 1.0).half().contiguous().view(torch.uint8).numpy().reshape(-1)
            n = min(resp_n, out.size)
            resp[:n] = out[:n]
        elif resp_n:
            resp[:min(resp_n, buf.size)] = buf[:min(resp_n, buf.size)]
        resp_flag[0] = s                                          # x86 TSO: payload stores are visible before this


def run_rtt(fo, args) -> int:
    import torch
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["max_it"])
    def k_ping(req_flag, resp_flag, seq_ptr, status_ptr, src, req, resp, dst, max_it, NREQ: tl.constexpr,
               NRESP: tl.constexpr, BLK: tl.constexpr):
        """one program: copy NREQ int32 words device -> host request; CTA barrier; raise the request flag (release,
        system scope); spin on the reply flag (acquire, system scope) for at most max_it polls; CTA barrier; copy
        NRESP words host reply -> device. status gets +1 per timed-out round trip."""
        seq = tl.load(seq_ptr) + 1
        tl.store(seq_ptr, seq)
        for i in range(0, NREQ, BLK):
            o = i + tl.arange(0, BLK)
            tl.store(req + o, tl.load(src + o, mask=o < NREQ, other=0), mask=o < NREQ)
        tl.debug_barrier()                                        # every lane's payload store before the flag
        tl.atomic_xchg(req_flag, seq, sem="release", scope="sys")
        it = 0
        got = tl.atomic_add(resp_flag, 0, sem="acquire", scope="sys")
        while (got != seq) & (it < max_it):
            got = tl.atomic_add(resp_flag, 0, sem="acquire", scope="sys")
            it += 1
        tl.debug_barrier()                                        # every lane reads the reply after the acquire
        tl.atomic_add(status_ptr, (got != seq).to(tl.int32))
        for i in range(0, NRESP, BLK):
            o = i + tl.arange(0, BLK)
            tl.store(dst + o, tl.load(resp + o, mask=o < NRESP, other=0), mask=o < NRESP)

    if any(c == "cpu" for _, c in args.rtt_cases):
        load_ext()                                                # build before a responder process loads it
    rt = None
    for line in Path("/proc/self/maps").read_text().splitlines():  # the libcudart torch loaded
        pth = line.split()[-1]
        if "/libcudart.so" in pth:
            rt = ctypes.CDLL(pth)
            break
    if rt is None:
        rt = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")
    rt.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    rt.cudaHostGetDevicePointer.restype = ctypes.c_int
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    ctx = mp.get_context("spawn")
    failures: list[str] = []
    for B, compute in args.rtt_cases:
        BH = B * KVH
        req_b = BH * (G * D * 2 + M_K * 4)                        # round 1: q fp16 + 128 int32 ids per head
        resp_b = BH * G * M_K * 2                                 # corrected logits fp16
        if compute == "flag":
            req_b = resp_b = 0
        nbytes = SHM_FLAGS + req_b + resp_b + 2 * 4096
        req_off, resp_off = SHM_FLAGS, SHM_FLAGS + req_b + 4096
        path = f"/dev/shm/rp_rtt_{os.getpid()}_{B}_{compute}"
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        addr = None
        registered = False
        proc = None
        try:
            os.ftruncate(fd, nbytes)
            addr = libc.mmap(None, nbytes, 0x3, 0x01, fd, 0)      # PROT_RW, MAP_SHARED
            if addr is None or addr == ctypes.c_void_p(-1).value:
                addr = None
                raise OSError(ctypes.get_errno(), "mmap failed")
            flags = np.ctypeslib.as_array((ctypes.c_uint8 * nbytes).from_address(addr))
            flags[:] = 0
            rc = torch.cuda.cudart().cudaHostRegister(addr, nbytes, 0)
            if int(rc) != 0:
                raise RuntimeError(f"cudaHostRegister rc {rc}")
            registered = True
            dp = ctypes.c_void_p()                                # the kernel uses the host pointer: require UVA
            rc = rt.cudaHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(addr), 0)
            if rc != 0 or dp.value != addr:
                raise RuntimeError(f"device pointer {dp.value} != host pointer {addr} (rc {rc})")

            class _CAI:
                def __init__(self, ptr: int, n: int) -> None:
                    self.__cuda_array_interface__ = {"shape": (n,), "typestr": "<i4", "data": (ptr, False),
                                                     "version": 3, "strides": None}
            hv = lambda off, n: torch.as_tensor(_CAI(addr + off, max(n, 1)), device="cuda")  # noqa: E731
            req_flag, resp_flag = hv(0, 1), hv(64, 1)
            reqv, respv = hv(req_off, req_b // 4), hv(resp_off, resp_b // 4)
            seq = torch.zeros(1, dtype=torch.int32, device="cuda")
            status = torch.zeros(1, dtype=torch.int32, device="cuda")
            src = torch.randint(0, 1 << 30, (max(req_b // 4, 1),), dtype=torch.int32, device="cuda")
            dst = torch.zeros(max(resp_b // 4, 1), dtype=torch.int32, device="cuda")
            proc = ctx.Process(target=_responder, args=(path, nbytes, req_off, req_b, resp_off, resp_b, compute,
                                                        B, os.getpid()), daemon=True)
            proc.start()
            ready = flags[128:132].view(np.int32)
            t0 = time.time()
            while int(ready[0]) != 1:                             # responder start-up (imports, data)
                if not proc.is_alive() or time.time() - t0 > 180:
                    raise RuntimeError("responder did not become ready")
                time.sleep(0.05)
            max_it = args.rtt_max_it

            def ping():
                return k_ping[(1,)](req_flag, resp_flag, seq, status, src, reqv, respv, dst, max_it,
                                    NREQ=req_b // 4, NRESP=resp_b // 4, BLK=256)

            def check(where: str) -> None:
                torch.cuda.synchronize()
                if int(status.item()) or not proc.is_alive():
                    raise RuntimeError(f"{where}: timeouts {int(status.item())}, responder alive {proc.is_alive()}")
            kobj = ping()
            for _ in range(4):
                ping()
            check("warm-up")
            # evidence for the memory-ordering review: atomics, how many are predicated to one lane, barriers
            ptx = (getattr(kobj, "asm", None) or {}).get("ptx", "")
            atom_lines = [l.strip() for l in ptx.splitlines() if "atom." in l]
            ptx_info = {"atom": len(atom_lines), "atom_predicated": sum(l.startswith("@") for l in atom_lines),
                        "bar_sync": sum("bar.sync" in l or "barrier.sync" in l for l in ptx.splitlines()),
                        "atom_samples": atom_lines[:4]}
            # eager: chunks of CH launches between one event pair (launch-bound when a launch costs more than the
            # round trip; the graph below is the round-trip figure)
            CH = 100
            chunks = []
            for _ in range(max(1, args.rtt_n // CH)):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(CH):
                    ping()
                b.record()
                check("eager")
                chunks.append(a.elapsed_time(b) * 1e3 / CH)
            chunks.sort()
            echo_ok = None
            dst.fill_(-1)                                         # device copy and host reply region: sentinel
            if resp_b:
                flags[resp_off:resp_off + resp_b].view(np.int32)[:] = -1
            ping()
            check("readback")
            if compute == "echo":
                n = min(req_b, resp_b) // 4
                echo_ok = bool(torch.equal(dst[:n], src[:n]))
            # CUDA graph: one decode step = 32 round trips back to back
            gph = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.graph(gph, stream=s):
                for _ in range(NL):
                    ping()
            gph.replay()
            check("graph warm-up")
            gt = []
            for _ in range(args.rtt_graph_reps):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                gph.replay()
                b.record()
                check("graph")
                gt.append(a.elapsed_time(b) * 1e3 / NL)
            gt.sort()
            rec = {"kind": "rtt", "B": B, "compute": compute, "req_bytes": req_b, "resp_bytes": resp_b,
                   "eager_us_per_rt_med": chunks[len(chunks) // 2], "eager_us_per_rt_min": chunks[0],
                   "eager_chunks": len(chunks), "eager_chunk": CH,
                   "graph_us_per_rt_med": gt[len(gt) // 2], "graph_us_per_rt_min": gt[0],
                   "graph_us_per_rt_q1": gt[len(gt) // 4], "graph_us_per_rt_q3": gt[3 * len(gt) // 4],
                   "graph_reps": len(gt), "timeouts": int(status.item()), "echo_ok": echo_ok,
                   "max_it": max_it, "ptx": ptx_info,
                   "responder_threads": int(os.environ.get("RP_RESP_THREADS", "16")) if compute == "cpu" else 1}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(f"rtt B{B} {compute}: graph {rec['graph_us_per_rt_med']:.1f} us per round trip (eager "
                  f"{rec['eager_us_per_rt_med']:.1f}), echo_ok {echo_ok}", flush=True)
            if echo_ok is False:
                raise RuntimeError("echo readback mismatch")
        except Exception as e:  # noqa: BLE001  (record, clean up in finally, go on with the next case)
            if args.selftest:
                raise
            failures.append(f"B{B} {compute}: {e!r}"[:200])
            fo.write(json.dumps({"kind": "error", "mode": "rtt", "B": B, "compute": compute,
                                 "trace": traceback.format_exc()[-3000:]}) + "\n")
            fo.flush()
            print(f"ERROR rtt B{B} {compute}: {e!r}", flush=True)
        finally:
            if addr is not None:
                np.ctypeslib.as_array((ctypes.c_uint8 * 4).from_address(addr)).view(np.int32)[0] = -1   # stop
            if proc is not None:
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()
            try:
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001  (keep cleaning up; the original error propagates)
                print(f"cleanup synchronize failed: {e!r}", flush=True)
            if registered:
                torch.cuda.cudart().cudaHostUnregister(addr)
            if addr is not None:
                libc.munmap(addr, nbytes)
            os.close(fd)
            os.unlink(path)
    if failures:
        raise RuntimeError(f"{len(failures)} round-trip case(s) failed: {failures}")
    return 0


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--modes", default="numeric,cpu,rtt")
    ap.add_argument("--Bs", default="1,8,32")
    ap.add_argument("--threads", default="1,4,16,32,64",
                    help="'max' = all CPUs in this process's affinity; use it only on an exclusive host (issue 224)")
    ap.add_argument("--impls", default="torch,cpp", help="torch = tensor ops (reference); cpp = rp_cpu_ext.cpp")
    ap.add_argument("--L", type=int, default=32768)
    ap.add_argument("--layer_regions", type=int, default=32, help="distinct host regions (32 = one per layer)")
    ap.add_argument("--cpu_steps_max_ids", type=int, default=8, help="decode steps with distinct ids in the pool")
    ap.add_argument("--cache_control", type=int, default=1, help="add the cache-resident control case")
    ap.add_argument("--t_min", type=float, default=2.0)
    ap.add_argument("--cpu_steps_min", type=int, default=3)
    ap.add_argument("--rtt_cases", default="1:flag,1:echo,8:echo,1:cpu,8:cpu")
    ap.add_argument("--rtt_n", type=int, default=2000)
    ap.add_argument("--rtt_graph_reps", type=int, default=50)
    ap.add_argument("--rtt_max_it", type=int, default=1_000_000, help="GPU spin bound per round trip (~1 s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        a.Bs, a.threads, a.L, a.layer_regions, a.t_min, a.cpu_steps_min = "1,8", "1,4", 4096, 4, 0.2, 1
        a.cpu_steps_max_ids = 2
        a.rtt_cases, a.rtt_n, a.rtt_graph_reps = "1:flag,1:echo,1:cpu", 200, 3
    a.modes = [m for m in a.modes.split(",") if m]
    a.Bs = [int(x) for x in a.Bs.split(",")]
    ncpu = len(os.sched_getaffinity(0))
    a.threads = sorted({ncpu if x == "max" else min(int(x), ncpu) for x in a.threads.split(",")})
    a.impls = [x for x in a.impls.split(",") if x]
    a.rtt_cases = [(int(x.split(":")[0]), x.split(":")[1]) for x in a.rtt_cases.split(",") if x]
    out = Path(a.out)
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    errors = []
    with out.open("w") as fo:
        import torch
        env = {"kind": "env", "host": socket.gethostname(), "argv": sys.argv, "script_md5": md5,
               "ext_md5": hashlib.md5((Path(__file__).resolve().parent / "rp_cpu_ext.cpp").read_bytes()).hexdigest(),
               "torch": torch.__version__, "cpu_count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
               "loadavg": os.getloadavg(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            env["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                                    if l.startswith("model name"))
        except (OSError, StopIteration):
            env["cpu_model"] = None
        try:
            env["thp_enabled"] = Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text().strip()
            env["numa_nodes"] = len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
        except OSError:
            env["thp_enabled"], env["numa_nodes"] = None, None
        env["llc_bytes"] = _llc_bytes()
        fo.write(json.dumps(env) + "\n")
        fo.flush()
        for mode, fn in (("numeric", run_numeric), ("cpu", run_cpu), ("rtt", run_rtt)):
            if mode not in a.modes:
                continue
            try:
                fn(fo, a)
            except Exception as e:  # noqa: BLE001  (one mode failing must not hide the others)
                errors.append(f"{mode}: {e!r}"[:300])
                fo.write(json.dumps({"kind": "error", "mode": mode, "trace": traceback.format_exc()[-3000:]}) + "\n")
                fo.flush()
                print(f"ERROR {mode}: {e!r}", flush=True)
                if a.selftest:
                    raise
        fo.write(json.dumps({"kind": "end", "complete": not errors, "errors": errors,
                             "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    summ = {"meta": {"script_md5": md5}, "complete": not errors, "modes": a.modes}
    if errors:
        summ["failed"] = "; ".join(errors)
    Path(str(out) + ".summary.json").write_text(json.dumps(summ))
    print(f"T3 {'DONE' if not errors else 'FAILED'}: {errors}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
