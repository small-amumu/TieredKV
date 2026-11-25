# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""OPX diagnostic (root cause of the PCIe excess; not registered, synthetic data, no accuracy):
measures NVML PCIe RX bytes per decode step of the deployed kalign.StepA (M = 128 key rows, K = 64 value
rows, Llama-3.1-8B shapes: 32 layers, 8 KV heads, 4 query heads per KV head, d 128, B = 1), one step of
32 layers captured in a CUDA graph and replayed (no launch / command traffic), for:
  - residual widths kb/vb in --codecs (row bytes 16*kb + 4 for keys, 16*vb + 4 for values);
  - residual records in pinned host memory (zero-copy) vs in GPU memory (same kernels): the host-minus-GPU
    difference is the traffic caused by the residual reads;
  - lengths in --lens (the expected residual traffic does not depend on L).
Expected residual bytes per step: 32 * 8 * (128 * (16 kb + 4) + 64 * (16 vb + 4)).
Output: one JSON line per (L, codec, place) with measured bytes per step, the expected bytes and the ratio,
and per-kernel device time per step from torch.profiler (eager, for reference).
usage: KALIGN_DIR=<kalign tree> python opx_pcie_probe.py <out.jsonl> [--lens 8192,32768] [--codecs 32,33,43,54,34,24]
       [--reps 200]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ["KALIGN_DIR"])
import kalign as KA  # noqa: E402

from pcie_meter import PcieMeter  # noqa: E402

NL, H, KVH, D, G = 32, 32, 8, 128, 4
GAM = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
M_, K_ = 128, 64
dev = "cuda"


def calib(st, g: torch.Generator) -> None:
    order = torch.stack([torch.randperm(D, generator=g) for _ in range(KVH)])
    bvec = torch.rand(KVH, D, generator=g) * 1.5 + 0.5
    mpre = torch.randn(KVH, D, generator=g) * 0.5
    invf = 1.0 / (500000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    st.set_calib(order, bvec, mpre, 1.0, invf.float())


def measure(L: int, kb: int, vb: int, host: bool, reps: int) -> dict:
    g = torch.Generator().manual_seed(0)
    W = KA.Z_SINK + 128
    q = torch.randn(H, D, generator=g).half().to(dev)
    kw = dict(pin_memory=True) if host else dict(device=dev)
    rhs = [(torch.zeros(KVH, L, 16 * kb, dtype=torch.int8, **kw), torch.zeros(KVH, L, 2, dtype=torch.float16, **kw),
            torch.zeros(KVH, L, 16 * vb, dtype=torch.int8, **kw), torch.zeros(KVH, L, 2, dtype=torch.float16, **kw))
           for _ in range(NL)]
    sts = [KA.StepA(L, GH=G, B=1, kbits=kb, vbits=vb, host=host, words=True, seed=li, rh=rhs[li]) for li in range(NL)]
    kr = torch.randn(KVH, L + KA.Z_SINK, D, generator=g).half().to(dev)
    vr = torch.randn(KVH, L + KA.Z_SINK, D, generator=g).half().to(dev)
    for st in sts:
        calib(st, g)
        st.rope_tables(L + KA.Z_SINK)
        st.encode(kr, vr, 0, L + KA.Z_SINK, GAM)
        st.set_exact(kr[:, :W], vr[:, :W])
        st.set_q(q)
    torch.cuda.synchronize()
    del kr, vr

    def step() -> None:
        for st in sts:
            st.step()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        step()
    for _ in range(10):
        gr.replay()
    torch.cuda.synchronize()
    with PcieMeter() as idle:
        time.sleep(1.0)
    t0 = time.perf_counter()
    with PcieMeter() as pm:
        for _ in range(reps):
            gr.replay()
        torch.cuda.synchronize()
        time.sleep(0.05)
    wall = time.perf_counter() - t0
    idle_rate = idle.rx_bytes / max(idle.seconds, 1e-9)
    net = (pm.rx_bytes - idle_rate * pm.seconds) / reps
    expected = NL * KVH * (M_ * (16 * kb + 4) + K_ * (16 * vb + 4))
    # per-kernel device time (eager) for reference
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
    ker: dict = {}
    for e in prof.events():
        if e.device_type == DeviceType.CUDA:
            d = ker.setdefault(e.name[:60], [0.0, 0])
            d[0] += e.time_range.elapsed_us() / 5
            d[1] += 1 / 5
    gr = None
    del sts, rhs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return dict(L=L, kb=kb, vb=vb, place="host" if host else "gpu", reps=reps, graph_ms_per_step=wall / reps * 1e3,
                rx_bytes_per_step_net=net, rx_bytes_per_step_raw=pm.rx_bytes / reps, tx_bytes_per_step=pm.tx_bytes / reps,
                idle_rx_rate=idle_rate, expected_residual_bytes=expected,
                ratio_to_expected=net / expected if host else None,
                kernels={k: dict(us=round(v[0], 1), calls=round(v[1], 2)) for k, v in sorted(ker.items())})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--lens", default="8192,32768")
    ap.add_argument("--codecs", default="32,33,43,54,34,24")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    with open(a.out, "a") as f:
        f.write(json.dumps({"meta": True, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                            "kalign": os.path.join(os.environ["KALIGN_DIR"], "kalign.py")}) + "\n")
    for L in map(int, a.lens.split(",")):
        for cd in a.codecs.split(","):
            for host in (True, False):
                try:
                    r = measure(L, int(cd[0]), int(cd[1]), host, a.reps)
                except Exception as e:  # noqa: BLE001
                    r = dict(L=L, codec=cd, place="host" if host else "gpu", error=repr(e)[:400])
                with open(a.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(json.dumps({k: r.get(k) for k in ("L", "kb", "vb", "place", "graph_ms_per_step",
                                                        "rx_bytes_per_step_net", "expected_residual_bytes",
                                                        "ratio_to_expected", "error")}), flush=True)


if __name__ == "__main__":
    main()
