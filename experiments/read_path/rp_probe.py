# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Read-path microbenchmark RP (PLAN_RP.md, amendment 1): zero-copy residual-row reads from pinned host memory.

T2: per-row cost of kalign-style zero-copy gathers with the host store on 4 KB pages vs 2 MB transparent huge
    pages (madvise), across context lengths (working set); 16-byte vectorized vs scalar loads.
T6 gate 1: for page-clustered selections (h hits in a page of P rows), cost per useful row of
    row       zero-copy gather of the hit rows only (kalign layout and load pattern),
    row_shuf  the same rows in a random order within each group (no help from sorted, adjacent hits),
    pagezc    zero-copy read of every hit page in full (one contiguous run), to a GPU staging buffer,
    dma       copy-engine cudaMemcpyAsync per page (measured per transfer size in dma records; per useful row =
              (words + metadata transfer) / h, computed in the analysis).

Synthetic data only (no model, no accuracy). Layout copies kalign's host records: per array one region of
NL * B * KVH * L rows, row = (layer * B * KVH + b * KVH + h) * L + token; key words [rows, NW_K] int32 (h54: 20 words,
80 B) + key metadata [rows, 2] 16-bit; value words [rows, NW_V] (h54: 16 words, 64 B) + metadata. Each step reads
M = 128 key rows and K = 64 value rows per (layer, sequence, KV head).

Launch modes: "layer" = one launch per layer with the engine's grid (B * KVH, M / 16), the 32 launches captured in
one CUDA graph per index set (as the engine runs); "all" = one launch over all layers (more concurrency; reference).
In the point loops (time_loop) every timed step uses another of --nset index sets and follows an L2 flush (a read-only
reduction over 2 x L2, outside the CUDA events), so a step cannot hit rows cached by an earlier one. DMA replays are
not flushed (copy-engine transfers do not read host data through L2); they are capped at 5000 replays (t_ok tells).

Output: JSON lines, kind = env | baseline | alloc | skip | ptx | point | dma | error | alloc_end | end, plus
<out>.summary.json for rq check_done. "end" is written in a finally (also after an exception) with the failure
counts. A store that raises writes kind=error and the sweep continues. Analysis filters: readback_bad == 0,
pcie_ok, procs_clean, t_ok. Per-step PCIe bytes: use pcie_rx_per_step_rate (samples inside the loop x loop wall
time); pcie_rx_bytes_per_step is PcieMeter's raw integral, kept for reference, biased low.
usage: python -B rp_probe.py <out.jsonl> [--Bs 1,8] [--Ls 4096,8192,32768] [--pms 4k,thp] [--codecs 54,32]
       [--gpu_place 1] [--dma 1] [--nset 16] [--t_min 1.0] [--reps_min 30] [--seed 0]
       python -B rp_probe.py <out.jsonl> --selftest   (small sizes; every path, every readback check made to fail once)
       python -B rp_probe.py <out.jsonl> --preflight  (B 8, L 32768, one page mode, no DMA: memory and time check)
Needs: torch with CUDA, triton, numpy, libnvidia-ml (pcie_meter.py next to this file).
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import math
import os
import re
import resource
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcie_meter import PcieMeter  # noqa: E402

NL, KVH = 32, 8
M_K, M_V, KC = 128, 64, 16
HUGE = 2 << 20
CODECS = {54: (20, 16), 43: (16, 12), 33: (12, 12), 32: (12, 8)}   # (NW_K, NW_V) int32 words: 16 * bits bytes
DEV = "cuda"
PERIOD = 0.01                                                    # PcieMeter sampling period (s)
PCIE_SKIP = 0.1                     # s after the loop start whose NVML samples may still cover pre-loop idle time


# ---------------------------------------------------------------------------------------------- kernels
@triton.jit(do_not_specialize=["g0"])
def k_row(sw, sm, idx, dw, dm, g0, NW: tl.constexpr, KC: tl.constexpr, M: tl.constexpr):
    """kalign _rw_load pattern: KC rows per program, [KC, 32] int32 tile masked to NW words, plus two 16-bit
    metadata loads per row (row * 2, row * 2 + 1). Group g = g0 + program_id(0)."""
    g = g0 + tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    ok = c * KC + tl.arange(0, KC)
    row = tl.load(idx + g * M + ok)
    ow = tl.arange(0, 32)
    lm = ow[None, :] < NW
    w = tl.load(sw + row[:, None] * NW + ow[None, :], mask=lm, other=0)
    m0 = tl.load(sm + row * 2)
    m1 = tl.load(sm + row * 2 + 1)
    o = g * M + ok
    tl.store(dw + o[:, None] * NW + ow[None, :], w, mask=lm)
    tl.store(dm + o * 2, m0)
    tl.store(dm + o * 2 + 1, m1)


@triton.jit(do_not_specialize=["g0", "nw"])
def k_rows(sw, sm, idx, dw, dm, g0, nw, NW: tl.constexpr, KC: tl.constexpr, M: tl.constexpr):
    """k_row with the word count of the LOAD address a run-time value (nw == NW): the compiler cannot prove 16-byte
    alignment, so the word loads stay scalar 32-bit. Stores are the same as k_row (constexpr NW)."""
    g = g0 + tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    ok = c * KC + tl.arange(0, KC)
    row = tl.load(idx + g * M + ok)
    ow = tl.arange(0, 32)
    w = tl.load(sw + row[:, None] * nw + ow[None, :], mask=ow[None, :] < nw, other=0)
    m0 = tl.load(sm + row * 2)
    m1 = tl.load(sm + row * 2 + 1)
    o = g * M + ok
    lm = ow[None, :] < NW
    tl.store(dw + o[:, None] * NW + ow[None, :], w, mask=lm)
    tl.store(dm + o * 2, m0)
    tl.store(dm + o * 2 + 1, m1)


@triton.jit(do_not_specialize=["g0"])
def k_page(sw, sm, pidx, dw, dm, g0, NW: tl.constexpr, P: tl.constexpr, FL: tl.constexpr, NPG: tl.constexpr):
    """one program per hit page: P consecutive rows read as one contiguous run of P * NW words (FL = next pow2)
    and 2 P metadata values, stored to the staging buffer."""
    g = g0 + tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    p0 = tl.load(pidx + g * NPG + j)
    of = tl.arange(0, FL)
    fm = of < P * NW
    w = tl.load(sw + p0 * NW + of, mask=fm, other=0)
    om = tl.arange(0, 2 * P)
    m = tl.load(sm + p0 * 2 + om)
    o = (g * NPG + j) * P
    tl.store(dw + o * NW + of, w, mask=fm)
    tl.store(dm + o * 2 + om, m)


@triton.jit
def k_empty(x):
    """launch-overhead baseline."""
    pass


# ---------------------------------------------------------------------------------------------- CUDA runtime
def _cudart() -> ctypes.CDLL:
    """the libcudart torch already loaded (from /proc/self/maps) if any, else the first one found."""
    major = int(torch.version.cuda.split(".")[0])
    names: list[str] = []
    try:
        for line in Path("/proc/self/maps").read_text().splitlines():
            p = line.split()[-1]
            if "/libcudart.so" in p and p not in names:
                names.append(p)
    except OSError:
        pass
    names += [f"libcudart.so.{major}", "libcudart.so"]
    for d in sys.path:
        names += sorted(glob.glob(os.path.join(d, "nvidia", "*", "lib", f"libcudart.so.{major}*")))
        names += sorted(glob.glob(os.path.join(d, "nvidia", "cuda_runtime", "lib", "libcudart.so*")))
    for nm in names:
        try:
            lib = ctypes.CDLL(nm)
        except OSError:
            continue
        lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                        ctypes.c_void_p]
        lib.cudaMemcpyAsync.restype = ctypes.c_int
        lib.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        lib.cudaHostGetDevicePointer.restype = ctypes.c_int
        lib._rp_path = nm
        return lib
    raise OSError("libcudart not found")


# ---------------------------------------------------------------------------------------------- host memory
class _CAI:
    TYPES = {torch.int32: "<i4", torch.int16: "<i2", torch.uint8: "|u1", torch.int64: "<i8"}

    def __init__(self, t: torch.Tensor) -> None:
        self.__cuda_array_interface__ = {"shape": tuple(t.shape), "typestr": self.TYPES[t.dtype],
                                         "data": (t.data_ptr(), False), "version": 3, "strides": None}


def uva(t: torch.Tensor) -> torch.Tensor:
    """CUDA view of a registered host tensor (zero-copy reads through unified addressing)."""
    return torch.as_tensor(_CAI(t), device=DEV)


_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_LIBC.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PROT_RW, MAP_PRIV_ANON, MADV_HUGEPAGE, MADV_NOHUGEPAGE = 0x3, 0x22, 14, 15


class HostRegion:
    """anonymous mmap (libc, so the mapping can be unmapped while Python objects are alive), 2 MB aligned,
    madvise(HUGEPAGE or NOHUGEPAGE) before the first touch, filled with random bits, then cudaHostRegister'ed with
    flags 0 (as the engine's exact_pin and SGLang's host pool); the device pointer is asserted to equal the host
    pointer. `t` is a flat uint8 tensor over the aligned span; it must not be used after close()."""

    def __init__(self, nbytes: int, pm: str, gen: torch.Generator, rt: ctypes.CDLL) -> None:
        self.n = -(-nbytes // HUGE) * HUGE
        self.registered = False
        self.t = None
        self.base = _LIBC.mmap(None, self.n + HUGE, PROT_RW, MAP_PRIV_ANON, -1, 0)
        if self.base is None or self.base == ctypes.c_void_p(-1).value:
            self.base = None
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            self.addr = self.base + (-self.base) % HUGE
            if _LIBC.madvise(self.addr, self.n, MADV_HUGEPAGE if pm == "thp" else MADV_NOHUGEPAGE) != 0:
                raise OSError(ctypes.get_errno(), "madvise failed")
            self.t = torch.from_numpy(np.ctypeslib.as_array((ctypes.c_uint8 * self.n).from_address(self.addr)))
            if self.t.data_ptr() != self.addr:
                raise RuntimeError("host tensor does not start at the mapped address")
            t0 = time.perf_counter()
            self.t.view(torch.int32).random_(generator=gen)     # first touch: the kernel picks the page size here
            self.fill_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            rc = torch.cuda.cudart().cudaHostRegister(self.addr, self.n, 0)
            if int(rc) != 0:
                raise RuntimeError(f"cudaHostRegister failed: {rc}")
            self.registered = True
            self.reg_s = time.perf_counter() - t0
            dp = ctypes.c_void_p()
            rc = rt.cudaHostGetDevicePointer(ctypes.byref(dp), self.addr, 0)
            if rc != 0 or dp.value != self.addr:
                raise RuntimeError(f"device pointer {dp.value} != host pointer {self.addr} (rc {rc})")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """unregister and unmap even if the synchronize or the unregister raises (then re-raise)."""
        if not self.registered and self.base is None:
            return
        try:
            torch.cuda.synchronize()
        finally:
            try:
                if self.registered:
                    self.registered = False
                    rc = torch.cuda.cudart().cudaHostUnregister(self.addr)
                    if int(rc) != 0:
                        raise RuntimeError(f"cudaHostUnregister failed: {rc}")
            finally:
                self.t = None
                if self.base is not None:
                    base, self.base = self.base, None
                    if _LIBC.munmap(base, self.n + HUGE) != 0:
                        raise OSError(ctypes.get_errno(), "munmap failed")


def smaps_span(addr: int, n: int) -> dict:
    """AnonHugePages / Rss (kB) of the VMAs overlapping [addr, addr + n), and their NUMA placement (numa_maps
    page counts; huge pages may be counted in 4 KB or 2 MB units depending on the kernel, so both totals are given)."""
    huge = rss = vma_bytes = 0
    vma_out = 0                     # overlapping VMAs that extend beyond the region (would mix in neighbours)
    cur = False
    with open("/proc/self/smaps") as f:
        for line in f:
            m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
            if m:
                lo, hi = int(m.group(1), 16), int(m.group(2), 16)
                cur = hi > addr and lo < addr + n
                if cur:
                    vma_bytes += hi - lo
                    vma_out += int(lo < addr or hi > addr + n)
                continue
            if cur and line.startswith("AnonHugePages:"):
                huge += int(line.split()[1])
            elif cur and line.startswith("Rss:"):
                rss += int(line.split()[1])
    numa: dict[str, int] = {}
    with open("/proc/self/numa_maps") as f:
        for line in f:
            lo = int(line.split()[0], 16)
            if addr <= lo < addr + n:
                for tok in line.split():
                    m = re.match(r"^N(\d+)=(\d+)$", tok)
                    if m:
                        numa[m.group(1)] = numa.get(m.group(1), 0) + int(m.group(2))
    return {"anon_huge_kb": huge, "rss_kb": rss, "span_kb": n // 1024, "huge_frac": huge / max(1, n // 1024),
            "vma_exact": bool(vma_bytes == n and vma_out == 0 and huge <= n // 1024),
            "numa_pages": numa, "numa_pages_total": sum(numa.values())}


def proc_status(*keys: str) -> dict:
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            k = line.split(":")[0]
            if k in keys:
                out[k] = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------------------------- monitoring
class NvmlSampler:
    """clocks, clock-event (throttle) reasons, PCIe link, temperature and power, sampled every `period` s."""

    def __init__(self, meter: PcieMeter, period: float = 0.05) -> None:
        self.lib, self.h, self.period = meter._nv.lib, meter._h, period
        self.rows: list[tuple] = []
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None

    def _u(self, fn: str, *a: int) -> int | None:
        v = ctypes.c_uint()
        try:
            rc = getattr(self.lib, fn)(self.h, *a, ctypes.byref(v))
        except AttributeError:
            return None
        return v.value if rc == 0 else None

    def _reasons(self) -> int | None:
        v = ctypes.c_ulonglong()
        for fn in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            try:
                if getattr(self.lib, fn)(self.h, ctypes.byref(v)) == 0:
                    return int(v.value)
            except AttributeError:
                continue
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.rows.append((self._u("nvmlDeviceGetClockInfo", 1), self._u("nvmlDeviceGetClockInfo", 2),
                              self._reasons(), self._u("nvmlDeviceGetCurrPcieLinkGeneration"),
                              self._u("nvmlDeviceGetCurrPcieLinkWidth"), self._u("nvmlDeviceGetTemperature", 0),
                              self._u("nvmlDeviceGetPowerUsage")))
            time.sleep(self.period)

    def __enter__(self) -> "NvmlSampler":
        self.rows = []
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thr is not None:
            self._thr.join()

    def summary(self) -> dict:
        cols = ("sm_mhz", "mem_mhz", "reasons", "pcie_gen", "pcie_width", "temp_c", "power_mw")
        out: dict = {"nvml_samples": len(self.rows)}
        for i, c in enumerate(cols):
            v = [r[i] for r in self.rows if r[i] is not None]
            if c == "reasons":                     # union of the clock-event bit flags seen during the loop
                acc = 0
                for x in v:
                    acc |= x
                out["reasons_or"] = acc if v else None
            else:
                out[f"{c}_min"] = min(v) if v else None
                out[f"{c}_max"] = max(v) if v else None
        return out


def gpu_procs(uuid: str) -> dict:
    """compute processes on this GPU other than ours; unparsed lines and nvidia-smi failures are counted."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return {"pids": [], "unparsed": 0, "smi_ok": False}
    if r.returncode != 0:
        return {"pids": [], "unparsed": 0, "smi_ok": False}
    me, pids, unparsed = os.getpid(), [], 0
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not parts[1].isdigit():
            unparsed += 1
        elif parts[0] == uuid and int(parts[1]) != me:
            pids.append(int(parts[1]))
    return {"pids": pids, "unparsed": unparsed, "smi_ok": True}


# ---------------------------------------------------------------------------------------------- selections
def sel_rows(G: int, L: int, M: int, gen: torch.Generator) -> torch.Tensor:
    """[G, M] absolute rows: M distinct tokens per group, uniform, sorted."""
    tok = torch.rand(G, L, generator=gen, device=DEV).topk(M, dim=1).indices.sort(dim=1).values
    return tok + torch.arange(G, device=DEV)[:, None] * L


def sel_pages(G: int, L: int, M: int, P: int, h: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """M / h distinct pages of P rows per group, h distinct rows in each. Returns (rows [G, M], page starts
    [G, M / h]), both absolute and sorted."""
    npg = M // h
    pg = torch.rand(G, L // P, generator=gen, device=DEV).topk(npg, dim=1).indices.sort(dim=1).values
    off = torch.rand(G, npg, P, generator=gen, device=DEV).topk(h, dim=2).indices.sort(dim=2).values
    base = torch.arange(G, device=DEV)[:, None] * L
    rows = (pg[:, :, None] * P + off).reshape(G, M) + base
    return rows.sort(dim=1).values, pg * P + base


def shuffled(rows: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """the same rows of each group in a random order."""
    perm = torch.rand(rows.shape, generator=gen, device=DEV).argsort(dim=1)
    return rows.gather(1, perm)


# ---------------------------------------------------------------------------------------------- timing
def procs_clean(pre: dict, post: dict) -> bool:
    """no other compute process before or after, nvidia-smi worked both times and every line parsed."""
    return bool(pre["smi_ok"] and post["smi_ok"] and not pre["pids"] and not post["pids"]
                and pre["unparsed"] == 0 and post["unparsed"] == 0)


def pcie_rate_fields(meter: PcieMeter, tw0: float, tw1: float, reps: int) -> dict:
    """per-step PCIe bytes = mean rate of the NVML samples taken inside the timed loop x loop wall time / reps.
    PcieMeter stamps a sample after its RX and TX calls (~50 ms together on lab05), so a sample's NVML window ends
    before its stamp: samples stamped in the first PCIE_SKIP s after tw0 can cover pre-loop idle time and are
    dropped. The loop's traffic is steady, so the mean of the remaining samples is the loop's rate. PcieMeter's own
    integral (pcie_rx_bytes_per_step) also spans the start-up and the post-loop sample; it is kept as recorded."""
    lo, hi = tw0 - meter.t_start + PCIE_SKIP, tw1 - meter.t_start
    s = [(rx, tx) for t, rx, tx in meter.series if lo <= t <= hi]
    if not s:
        return {"pcie_rx_per_step_rate": None, "pcie_tx_per_step_rate": None, "pcie_samples_in_loop": 0}
    rx = sum(a for a, _ in s) / len(s) * 1024.0
    tx = sum(b for _, b in s) / len(s) * 1024.0
    return {"pcie_rx_per_step_rate": rx * (tw1 - tw0) / reps, "pcie_tx_per_step_rate": tx * (tw1 - tw0) / reps,
            "pcie_samples_in_loop": len(s)}


def flush_l2(buf: torch.Tensor) -> None:
    """read-only pass over 2 x L2 (evicts earlier lines without leaving dirty lines of its own)."""
    buf.sum(dtype=torch.int64)


def time_loop(launch, nset: int, flush: torch.Tensor, t_min: float, reps_min: int, reps_max: int,
              meter: PcieMeter, nv: NvmlSampler, uuid: str) -> dict:
    """launch(i) runs one step with index set i. Warm-up; reps >= reps_min, enough for t_min s of step time
    (capped at reps_max); each timed step follows an L2 flush outside the events. PCIe / NVML cover the loop."""
    for i in range(min(nset, 4)):
        launch(i)
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(5)]
    t0 = time.perf_counter()
    for i, (a, b) in enumerate(e):
        flush_l2(flush)
        a.record()
        launch(i % nset)
        b.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / 5
    est = sorted(a.elapsed_time(b) for a, b in e)[2] / 1e3
    reps = int(min(reps_max, max(reps_min, math.ceil(t_min / max(est, 1e-7)))))
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    pre = gpu_procs(uuid)
    with meter, nv:
        tw0 = time.perf_counter()
        for r in range(reps):
            flush_l2(flush)
            ev[r][0].record()
            launch(r % nset)
            ev[r][1].record()
        torch.cuda.synchronize()
        tw1 = time.perf_counter()
    loop_wall = tw1 - tw0
    post = gpu_procs(uuid)
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    q = lambda p: ts[min(len(ts) - 1, int(p * (len(ts) - 1) + 0.5))]   # noqa: E731
    d = {"reps": reps, "ms_med": q(0.5), "ms_q1": q(0.25), "ms_q3": q(0.75), "ms_min": ts[0], "ms_max": ts[-1],
         "kernel_s_total": sum(ts) / 1e3, "wall_per_rep_est_s": wall, "procs_pre": pre, "procs_post": post}
    d.update(meter.as_dict(steps=reps))
    d.update(nv.summary())
    d["loop_wall_s"] = loop_wall
    d.update(pcie_rate_fields(meter, tw0, tw1, reps))
    # one NVML sample (RX + TX calls) takes ~50 ms (measured on lab05), so a >= 1 s loop gives ~20 samples;
    # require a clean sampler and >= 10 samples inside the loop
    d["pcie_ok"] = bool(d.get("pcie_error") is None and d["pcie_samples_in_loop"] >= 10)
    d["procs_clean"] = procs_clean(pre, post)
    d["t_ok"] = bool(d["kernel_s_total"] >= 0.9 * t_min)      # False when reps_max cut the loop short
    return d


# ---------------------------------------------------------------------------------------------- verification
def verify_rows(dw: torch.Tensor, dm: torch.Tensor, rows: torch.Tensor, hw: torch.Tensor, hm: torch.Tensor,
                nw: int) -> int:
    """number of output rows (all of them) whose words or metadata differ from source row rows.reshape(-1)[i]
    (source = host tensors on the CPU, or device tensors)."""
    src = rows.reshape(-1).to(hw.device)
    exp_w = hw.view(-1, nw)[src]
    exp_m = hm.view(-1, 2)[src]
    got_w = dw.view(-1, nw).to(hw.device)
    got_m = dm.view(-1, 2).to(hw.device)
    return int(((got_w != exp_w).any(dim=1) | (got_m != exp_m).any(dim=1)).sum())


def verify_pages(dw: torch.Tensor, dm: torch.Tensor, pages: torch.Tensor, hw: torch.Tensor, hm: torch.Tensor,
                 nw: int, P: int) -> int:
    """number of staged pages (all of them) with any row differing from the source page."""
    p0 = pages.reshape(-1)
    rows = (p0[:, None] + torch.arange(P, device=p0.device)[None, :]).reshape(-1).to(hw.device)
    exp_w = hw.view(-1, nw)[rows]
    exp_m = hm.view(-1, 2)[rows]
    got_w = dw.view(-1, nw).to(hw.device)
    got_m = dm.view(-1, 2).to(hw.device)
    bad_row = (got_w != exp_w).any(dim=1) | (got_m != exp_m).any(dim=1)
    return int(bad_row.view(-1, P).any(dim=1).sum())


# ---------------------------------------------------------------------------------------------- DMA
def dma_capture(rt: ctypes.CDLL, dst0: int, src0: int, offs: list[int], size: int,
                sides: list[torch.cuda.Stream]) -> torch.cuda.CUDAGraph:
    """graph of len(offs) host-to-device copies of `size` bytes. sides = [] puts every copy on the capture stream
    (serial chain); otherwise copies go round-robin to the given streams (created before capture), forked from and
    joined back into the capture stream, so the copy engines may overlap them."""
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cap = torch.cuda.current_stream()
        streams = sides if sides else [cap]
        for s in sides:
            s.wait_stream(cap)
        for i, o in enumerate(offs):
            rc = rt.cudaMemcpyAsync(dst0 + i * size, src0 + o, size, 1, streams[i % len(streams)].cuda_stream)
            if rc != 0:
                raise RuntimeError(f"cudaMemcpyAsync rc {rc}")
        for s in sides:
            cap.wait_stream(s)
    return g


def dma_probe(fo, rt: ctypes.CDLL, regs: dict, G: int, L: int, args, meter: PcieMeter, nv: NvmlSampler,
              uuid: str, base: dict, selftest: bool, counts: dict) -> None:
    """per-transfer cost of copying one page (words: P * row word bytes; metadata: P * 4 bytes) from the real page
    starts of the host regions: ntr transfers per graph, --dma_sets graphs with different random pages replayed in
    rotation, serial and 4-stream. Replays continue until >= t_min s of GPU time (at least --dma_reps). No L2 flush
    here: copy-engine transfers write device memory and do not read host data through the GPU L2."""
    gen = torch.Generator().manual_seed(args.seed + 11)
    sides4 = [torch.cuda.Stream() for _ in range(4)]
    for arr, (wreg, mreg, nw) in regs.items():
        for what, reg, rb in (("words", wreg, 4 * nw), ("meta", mreg, 4)):
            for P in args.Ps:
                size = P * rb
                npages = G * L // P
                if npages * size > reg.n:
                    raise RuntimeError(f"dma pages exceed the region ({npages * size} > {reg.n})")
                pages = [torch.randint(0, npages, (args.dma_ntr,), generator=gen) for _ in range(args.dma_sets)]
                sets = [(p * size).tolist() for p in pages]
                dst = torch.empty(args.dma_ntr * size, dtype=torch.uint8, device=DEV)
                for sides in ([], sides4):
                    ns = max(1, len(sides))
                    graphs = [dma_capture(rt, dst.data_ptr(), reg.addr, offs, size, sides) for offs in sets]
                    for g in graphs:
                        g.replay()
                    torch.cuda.synchronize()
                    a0, b0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a0.record()
                    for g in graphs:
                        g.replay()
                    b0.record()
                    torch.cuda.synchronize()
                    est = a0.elapsed_time(b0) / 1e3 / len(graphs)
                    reps = int(min(5000, max(args.dma_reps, math.ceil(args.t_min / max(est, 1e-7)))))
                    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                          for _ in range(reps)]
                    pre = gpu_procs(uuid)
                    with meter, nv:
                        tw0 = time.perf_counter()
                        for r, (a, b) in enumerate(ev):
                            a.record()
                            graphs[r % len(graphs)].replay()
                            b.record()
                        torch.cuda.synchronize()
                        tw1 = time.perf_counter()
                    loop_wall = tw1 - tw0
                    post = gpu_procs(uuid)
                    ts = sorted(a.elapsed_time(b) for a, b in ev)
                    rec = dict(base, kind="dma", arr=arr, what=what, P=P, size=size, streams=ns, ntr=args.dma_ntr,
                               sets=len(sets), reps=reps, ms_med=ts[len(ts) // 2], ms_min=ts[0], ms_max=ts[-1],
                               ns_per_transfer=ts[len(ts) // 2] * 1e6 / args.dma_ntr, procs_pre=pre, procs_post=post)
                    rec.update(meter.as_dict(steps=reps))
                    rec.update(nv.summary())
                    rec["loop_wall_s"] = loop_wall
                    rec.update(pcie_rate_fields(meter, tw0, tw1, reps))
                    rec["pcie_ok"] = bool(rec.get("pcie_error") is None and rec["pcie_samples_in_loop"] >= 10)
                    rec["procs_clean"] = procs_clean(pre, post)
                    rec["t_ok"] = bool(sum(ts) / 1e3 >= 0.9 * args.t_min)
                    # readback of every transfer of the last replayed set (destination pre-filled with a sentinel)
                    k_last = (reps - 1) % len(graphs)
                    dst.fill_(0xFF)
                    graphs[k_last].replay()
                    torch.cuda.synchronize()
                    got = dst.cpu().view(args.dma_ntr, size)
                    exp = reg.t[: npages * size].view(npages, size)[pages[k_last]]
                    bad = int((got != exp).any(dim=1).sum())
                    if selftest and P == args.Ps[0]:
                        got[args.dma_ntr // 2, 0] ^= 1
                        inj = int((got != exp).any(dim=1).sum())
                        if inj != bad + 1:
                            raise RuntimeError(f"dma readback did not see the injected error ({inj} vs {bad})")
                    rec["readback_bad"] = bad
                    counts["dma_records"] += 1
                    counts["dma_bad"] += int(bad > 0)
                    counts["dma_pcie_bad"] += int(not rec["pcie_ok"])
                    counts["dma_procs_dirty"] += int(not rec["procs_clean"])
                    counts["dma_t_bad"] += int(not rec["t_ok"])
                    if ns == 1 and P == args.Ps[0]:     # host-submitted loop: CPU cost of issuing the copies
                        st = torch.cuda.current_stream().cuda_stream
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for i, o in enumerate(sets[0]):
                            if rt.cudaMemcpyAsync(dst.data_ptr() + i * size, reg.addr + o, size, 1, st) != 0:
                                raise RuntimeError("cudaMemcpyAsync failed")
                        t1 = time.perf_counter()
                        torch.cuda.synchronize()
                        rec["host_loop_cpu_ns_per_transfer"] = (t1 - t0) * 1e9 / args.dma_ntr
                        rec["host_loop_total_ns_per_transfer"] = (time.perf_counter() - t0) * 1e9 / args.dma_ntr
                    fo.write(json.dumps(rec) + "\n")
                    fo.flush()
                    print(f"  dma {arr} {what} P{P} s{ns}: {rec['ns_per_transfer']:.0f} ns/transfer bad {bad} "
                          f"pcie_ok {rec['pcie_ok']}", flush=True)
                    del graphs
                del dst


# ---------------------------------------------------------------------------------------------- env / ptx
def _rd(p: str) -> str | None:
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


def env_record(args: argparse.Namespace, uuid: str) -> dict:
    bus = drv = None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout
        for line in q.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[0] == uuid:
                bus, drv = parts[1], parts[2]
    except (OSError, subprocess.TimeoutExpired):
        pass
    pci = None
    if bus:
        dom, rest = bus.split(":", 1)
        pci = f"{int(dom, 16):04x}:{rest}".lower()
    dev = f"/sys/bus/pci/devices/{pci}" if pci else None
    mem = {}
    for line in (_rd("/proc/meminfo") or "").splitlines():
        if line.split(":")[0] in ("MemTotal", "MemAvailable", "HugePages_Total", "AnonHugePages"):
            mem[line.split(":")[0]] = line.split(":", 1)[1].strip()
    return {"kind": "env", "host": socket.gethostname(), "argv": sys.argv, "args": {k: v for k, v in vars(args).items()},
            "script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
            "pcie_meter_md5": hashlib.md5((Path(__file__).parent / "pcie_meter.py").read_bytes()).hexdigest(),
            "gpu": torch.cuda.get_device_name(0), "uuid": uuid, "pci": pci, "driver": drv,
            "gpu_found_in_smi": bus is not None,
            "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
            "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
            "iommu_classes": sorted(os.listdir("/sys/class/iommu")) if os.path.isdir("/sys/class/iommu") else [],
            "iommu_group_type": _rd(f"{dev}/iommu_group/type") if dev else None,
            "gpu_numa_node": _rd(f"{dev}/numa_node") if dev else None,
            "cmdline": _rd("/proc/cmdline"), "thp_enabled": _rd("/sys/kernel/mm/transparent_hugepage/enabled"),
            "thp_defrag": _rd("/sys/kernel/mm/transparent_hugepage/defrag"),
            "rlimit_memlock": list(resource.getrlimit(resource.RLIMIT_MEMLOCK)), "meminfo": mem,
            "cpu_affinity_n": len(os.sched_getaffinity(0)), "loadavg": _rd("/proc/loadavg"),
            "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def ptx_counts(kernel_obj) -> dict | None:
    asm = getattr(kernel_obj, "asm", None)
    if not asm or "ptx" not in asm:
        return None
    cnt: dict[str, int] = {}
    for m in re.finditer(r"\b(?:ld|st)\.global(?:\.[a-z0-9]+)*", asm["ptx"]):
        cnt[m.group(0)] = cnt.get(m.group(0), 0) + 1
    return cnt


# ---------------------------------------------------------------------------------------------- one store
def run_alloc(fo, args, rt, B: int, L: int, pm: str, codec: int, place: str, meter: PcieMeter, nv: NvmlSampler,
              uuid: str, flush: torch.Tensor, gen_h: torch.Generator, gen_d: torch.Generator, ptx_done: set,
              counts: dict, ladder: bool = False) -> None:
    """one store (B, L, page mode, codec, place): every pattern, path and launch mode for keys and values.
    ladder = True: the context-length ladder (rand pattern only, no DMA), tagged ladder=True in every record."""
    nwk, nwv = CODECS[codec]
    BH = B * KVH
    G = NL * BH
    rows_all = G * L
    arrs = {"k": (nwk, M_K), "v": (nwv, M_V)}
    regions: list[HostRegion] = []
    src: dict[str, tuple] = {}
    base = {"B": B, "L": L, "pm": pm, "codec": codec, "place": place, "ladder": ladder}
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.perf_counter()
        for a, (nw, _) in arrs.items():
            if place == "host":
                rw = HostRegion(rows_all * nw * 4, pm, gen_h, rt)
                regions.append(rw)
                rm = HostRegion(rows_all * 4, pm, gen_h, rt)
                regions.append(rm)
                hw = rw.t[: rows_all * nw * 4].view(torch.int32)
                hm = rm.t[: rows_all * 4].view(torch.int16)
                src[a] = (uva(hw), uva(hm), hw, hm, rw, rm)
            else:
                hw = torch.empty(rows_all * nw, dtype=torch.int32, device=DEV).random_(generator=gen_d)
                hm = torch.empty(rows_all * 2, dtype=torch.int16, device=DEV).random_(generator=gen_d)
                src[a] = (hw, hm, hw, hm, None, None)
        alloc = dict(base, kind="alloc", setup_s=time.perf_counter() - t0)
        if place == "host":
            alloc["regions"] = {f"{a}_{w}": dict(smaps_span(r.addr, r.n), fill_s=r.fill_s, reg_s=r.reg_s)
                                for a in arrs for w, r in (("words", src[a][4]), ("meta", src[a][5]))}
            alloc["is_pinned"] = {a: bool(src[a][2].is_pinned()) for a in arrs}
            fr = min(v["huge_frac"] for v in alloc["regions"].values())
            frmax = max(v["huge_frac"] for v in alloc["regions"].values())
            if (pm == "thp" and fr < 0.9) or (pm == "4k" and frmax > 0.0):
                print(f"WARNING huge-page coverage {fr:.3f}-{frmax:.3f} for page mode {pm}", flush=True)
            # carried into every later record of this store, so the analysis can gate page-mode points on coverage
            base["huge_frac_min"], base["huge_frac_max"] = fr, frmax
            base["vma_exact"] = all(v["vma_exact"] for v in alloc["regions"].values())
        alloc.update(proc_status("VmHWM", "VmRSS", "VmLck", "VmPin"))
        fo.write(json.dumps(alloc) + "\n")
        fo.flush()

        pats: list[tuple] = [("rand", 0, 0)]
        if codec == 54 and not ladder:
            for P in args.Ps:
                h = 1
                while h <= P:
                    pats.append(("page", P, h))
                    h *= 2
        for a, (nw, M) in arrs.items():
            sw, sm, hw, hm = src[a][:4]
            for kind, P, h in pats:
                if place == "host":
                    paths = [("row", "layer"), ("row_shuf", "layer")]
                    if kind == "rand":
                        paths += [("rows", "layer"), ("row", "all")]
                    else:
                        paths += [("pagezc", "layer")]
                else:
                    paths = [("row", "layer")] + ([("row", "all")] if kind == "rand" else [])
                if kind == "page" and (M % h or M // h > L // P):
                    fo.write(json.dumps(dict(base, kind="skip", arr=a, pat=kind, P=P, h=h,
                                             paths=[f"{p}/{m}" for p, m in paths], why="pages")) + "\n")
                    counts["skips"] += 1
                    continue
                sets, psets = [], []
                for _ in range(args.nset):
                    if kind == "rand":
                        sets.append(sel_rows(G, L, M, gen_d))
                    else:
                        r_, p_ = sel_pages(G, L, M, P, h, gen_d)
                        sets.append(r_)
                        psets.append(p_)
                shuf = [shuffled(s, gen_d) for s in sets]
                for path, mode in paths:
                    pnpg = M // h if path == "pagezc" else 0
                    if path == "pagezc":
                        dw = torch.empty(G * pnpg * P * nw, dtype=torch.int32, device=DEV)
                        dm = torch.empty(G * pnpg * P * 2, dtype=torch.int16, device=DEV)
                        FL = triton.next_power_of_2(P * nw)
                        useful_rows, read_rows = G * M, G * pnpg * P

                        def one(i: int, g0: int, ng: int, dw=dw, dm=dm, FL=FL, npg=pnpg, P=P, sw=sw, sm=sm,
                                nw=nw, ps=psets):
                            return k_page[(ng, npg)](sw, sm, ps[i], dw, dm, g0, NW=nw, P=P, FL=FL, NPG=npg)
                    else:
                        dw = torch.empty(G * M * nw, dtype=torch.int32, device=DEV)
                        dm = torch.empty(G * M * 2, dtype=torch.int16, device=DEV)
                        useful_rows = read_rows = G * M
                        ix = shuf if path == "row_shuf" else sets
                        if path == "rows":
                            def one(i: int, g0: int, ng: int, dw=dw, dm=dm, M=M, ix=ix, sw=sw, sm=sm, nw=nw):
                                return k_rows[(ng, M // KC)](sw, sm, ix[i], dw, dm, g0, nw, NW=nw, KC=KC, M=M)
                        else:
                            def one(i: int, g0: int, ng: int, dw=dw, dm=dm, M=M, ix=ix, sw=sw, sm=sm, nw=nw):
                                return k_row[(ng, M // KC)](sw, sm, ix[i], dw, dm, g0, NW=nw, KC=KC, M=M)
                    kobj = one(0, 0, BH)                   # compile outside any capture
                    torch.cuda.synchronize()
                    graphs = []
                    for i in range(args.nset):            # one graph per index set: 32 per-layer launches or 1
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            if mode == "layer":
                                for li in range(NL):
                                    one(i, li * BH, BH)
                            else:
                                one(i, 0, G)
                        graphs.append(g)

                    def launch(i: int, graphs=graphs) -> None:
                        graphs[i].replay()
                    key = (path, nw, P if path == "pagezc" else 0)
                    if key not in ptx_done:
                        ptx_done.add(key)
                        fo.write(json.dumps({"kind": "ptx", "path": path, "arr": a, "codec": codec, "nw": nw,
                                             "P": P if path == "pagezc" else 0, "insts": ptx_counts(kobj)}) + "\n")
                    d = time_loop(launch, args.nset, flush, args.t_min, args.reps_min, args.reps_max, meter, nv, uuid)
                    # readback: rerun the last index set of the loop into sentinel-filled outputs and compare every
                    # output row with the source. Word sources are non-negative int32, so a row the kernel did not
                    # write fails on its words (16-bit metadata views of the host fill can be negative).
                    i_last = (d["reps"] - 1) % args.nset
                    dw.fill_(-1)
                    dm.fill_(-1)
                    launch(i_last)
                    torch.cuda.synchronize()
                    if path == "pagezc":
                        bad = verify_pages(dw, dm, psets[i_last], hw, hm, nw, P)
                    else:
                        bad = verify_rows(dw, dm, (shuf if path == "row_shuf" else sets)[i_last], hw, hm, nw)
                    if args.selftest:                     # the check must be able to fail: corrupt one row
                        dw.view(-1, nw)[dw.numel() // nw // 2, 0] ^= 1
                        inj = (verify_pages(dw, dm, psets[i_last], hw, hm, nw, P) if path == "pagezc" else
                               verify_rows(dw, dm, (shuf if path == "row_shuf" else sets)[i_last], hw, hm, nw))
                        if inj != bad + 1:
                            raise RuntimeError(f"{path}: readback did not see the injected error ({inj} vs {bad})")
                    counts["points"] += 1
                    counts["readback_bad_points"] += int(bad > 0)
                    counts["pcie_bad_points"] += int(not d["pcie_ok"])
                    counts["procs_dirty_points"] += int(not d["procs_clean"])
                    counts["t_bad_points"] += int(not d["t_ok"])
                    rb = 4 * nw + 4
                    rec = dict(base, kind="point", arr=a, path=path, mode=mode, pat=kind, P=P, h=h, row_bytes=rb,
                               useful_rows=useful_rows, read_rows=read_rows, useful_bytes=useful_rows * rb,
                               read_bytes_logical=read_rows * rb, readback_bad=bad)
                    rec.update(d)
                    rec["ns_per_useful_row"] = d["ms_med"] * 1e6 / useful_rows
                    rec["useful_GBps"] = useful_rows * rb / (d["ms_med"] * 1e-3) / 1e9
                    rx = d.get("pcie_rx_per_step_rate")
                    rec["pcie_rx_over_useful"] = rx / rec["useful_bytes"] if rx is not None else None
                    rec["pcie_rx_over_read"] = rx / rec["read_bytes_logical"] if rx is not None else None
                    fo.write(json.dumps(rec) + "\n")
                    fo.flush()
                    print(f"  {place} {pm} B{B} L{L} c{codec} {a} {path}/{mode} {kind} P{P} h{h}: "
                          f"{rec['ns_per_useful_row']:.2f} ns/row bad {bad} pcie_ok {d['pcie_ok']}", flush=True)
                    del graphs, launch, one, dw, dm
                del sets, psets, shuf
                torch.cuda.empty_cache()
        if place == "host" and args.dma and B == 1 and codec == 54 and not ladder:
            regs = {a: (src[a][4], src[a][5], arrs[a][0]) for a in arrs}
            dma_probe(fo, rt, regs, G, L, args, meter, nv, uuid, base, args.selftest, counts)
        fo.write(json.dumps(dict(base, kind="alloc_end", max_gpu_alloc_bytes=torch.cuda.max_memory_allocated(),
                                 **proc_status("VmHWM", "VmRSS"))) + "\n")
        fo.flush()
    finally:
        src.clear()
        errs = []
        for r in regions:                 # close every region even if one fails, then re-raise
            try:
                r.close()
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))
        torch.cuda.empty_cache()
        if errs:
            raise RuntimeError(f"region close failed: {errs}")


def baseline(fo, flush: torch.Tensor, args, meter: PcieMeter, nv: NvmlSampler, uuid: str) -> None:
    """event-timed replay of a graph of 1 and of NL empty launches: the fixed offset inside a timed step of the
    `all` and the `layer` launch mode respectively (subtract per matching mode)."""
    x = torch.empty(1, device=DEV)
    k_empty[(1,)](x)
    torch.cuda.synchronize()
    for n, mode in ((1, "all"), (NL, "layer")):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                k_empty[(1,)](x)
        d = time_loop(lambda i, g=g: g.replay(), 1, flush, 0.2, args.reps_min, args.reps_max, meter, nv, uuid)
        fo.write(json.dumps(dict(kind="baseline", mode=mode, launches=n, **d)) + "\n")
        fo.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--Bs", default="1,8")
    ap.add_argument("--Ls", default="4096,8192,32768")
    ap.add_argument("--Ps", default="8,16,32,64")
    ap.add_argument("--pms", default="4k,thp")
    ap.add_argument("--ladder_Ls", default="16384,65536,131072",
                    help="extra context lengths for the B=1 rand-only ladder (with --Ls they form 4K..128K)")
    ap.add_argument("--codecs", default="54,32")
    ap.add_argument("--gpu_place", type=int, default=1)
    ap.add_argument("--dma", type=int, default=1)
    ap.add_argument("--dma_ntr", type=int, default=4096)
    ap.add_argument("--dma_sets", type=int, default=4)
    ap.add_argument("--dma_reps", type=int, default=100)
    ap.add_argument("--nset", type=int, default=16)
    ap.add_argument("--t_min", type=float, default=1.0)
    ap.add_argument("--reps_min", type=int, default=30)
    ap.add_argument("--reps_max", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        args.Bs, args.Ls, args.Ps, args.codecs, args.pms = "1", "4096", "8,64", "54,32", "4k,thp"
        args.ladder_Ls = "8192"
        args.nset, args.t_min, args.reps_min, args.dma_ntr, args.dma_sets, args.dma_reps = 4, 0.05, 5, 256, 2, 4
    if args.preflight:
        args.Bs, args.Ls, args.Ps, args.codecs, args.pms, args.dma, args.gpu_place = "8", "32768", "64", "54", "thp", 0, 1
        args.t_min, args.reps_min, args.ladder_Ls = 0.2, 10, "131072"
    args.ladder_Ls = [int(x) for x in args.ladder_Ls.split(",") if x.strip()]
    args.Bs = [int(x) for x in args.Bs.split(",")]
    args.Ls = [int(x) for x in args.Ls.split(",")]
    args.Ps = [int(x) for x in args.Ps.split(",")]
    args.pms = args.pms.split(",")
    args.codecs = [int(x) for x in args.codecs.split(",")]
    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    torch.cuda.init()
    rt = _cudart()
    meter = PcieMeter(period=PERIOD)
    nv = NvmlSampler(meter)
    uuid = meter.uuid
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size // 4, dtype=torch.int32, device=DEV)
    flush.zero_()
    gen_h = torch.Generator().manual_seed(args.seed)
    gen_d = torch.Generator(device=DEV).manual_seed(args.seed + 1)
    ptx_done: set = set()
    counts = {"points": 0, "readback_bad_points": 0, "pcie_bad_points": 0, "procs_dirty_points": 0, "skips": 0,
              "dma_records": 0, "dma_bad": 0, "dma_pcie_bad": 0, "dma_procs_dirty": 0, "alloc_errors": 0,
              "t_bad_points": 0, "dma_t_bad": 0}
    script_md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    t_start = time.time()
    error = None
    with out.open("w") as fo:
        try:
            env = env_record(args, uuid)
            env["cudart_path"] = rt._rp_path
            fo.write(json.dumps(env) + "\n")
            fo.flush()
            baseline(fo, flush, args, meter, nv, uuid)
            for codec in args.codecs:
                for B in args.Bs:
                    for L in args.Ls:
                        if codec != 54 and L != max(args.Ls):
                            continue
                        stores = [(pm, "host") for pm in args.pms]
                        if args.gpu_place and codec == 54:
                            stores.append(("-", "gpu"))
                        for pm, place in stores:
                            try:
                                run_alloc(fo, args, rt, B, L, pm, codec, place, meter, nv, uuid, flush, gen_h,
                                          gen_d, ptx_done, counts)
                            except Exception as e:  # noqa: BLE001  (one failed store must not end the sweep)
                                if args.selftest:
                                    raise
                                counts["alloc_errors"] += 1
                                fo.write(json.dumps({"kind": "error", "B": B, "L": L, "pm": pm, "codec": codec,
                                                     "place": place,
                                                     "trace": traceback.format_exc()[-3000:]}) + "\n")
                                fo.flush()
                                print(f"ERROR store B{B} L{L} {pm} {place} c{codec}: {e!r}", flush=True)
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()     # a sticky CUDA error raises here and ends the run
                        print(f"done codec {codec} B {B} L {L} ({time.time() - t_start:.0f} s)", flush=True)
            # context-length ladder (B 1, rand only): where per-row cost bends with the working set, per page mode
            for L in args.ladder_Ls:
                for pm in args.pms:
                    try:
                        run_alloc(fo, args, rt, 1, L, pm, 54, "host", meter, nv, uuid, flush, gen_h, gen_d,
                                  ptx_done, counts, ladder=True)
                    except Exception as e:  # noqa: BLE001
                        if args.selftest:
                            raise
                        counts["alloc_errors"] += 1
                        fo.write(json.dumps({"kind": "error", "B": 1, "L": L, "pm": pm, "codec": 54, "place": "host",
                                             "ladder": True, "trace": traceback.format_exc()[-3000:]}) + "\n")
                        fo.flush()
                        print(f"ERROR ladder L{L} {pm}: {e!r}", flush=True)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                print(f"done ladder L {L} ({time.time() - t_start:.0f} s)", flush=True)
        except BaseException as e:
            error = repr(e)[:500]
            raise
        finally:
            # "failed" reasons: the sweep raised, a store raised, or any readback (points or DMA) failed; the end
            # line and the JSON summary for rq check_done (it cannot read JSON lines) use the same verdict
            why = []
            if error is not None:
                why.append(error)
            if counts["alloc_errors"]:
                why.append(f"{counts['alloc_errors']} store(s) raised; see kind=error records")
            if counts["readback_bad_points"] or counts["dma_bad"]:
                why.append(f"readback failures: {counts['readback_bad_points']} points, {counts['dma_bad']} dma")
            complete = not why
            fo.write(json.dumps({"kind": "end", "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "elapsed_s": time.time() - t_start, "complete": complete, "error": error,
                                 "why": why, **counts}) + "\n")
            summ = {"meta": {"script_md5": script_md5}, "counts": counts, "complete": complete}
            if why:
                summ["failed"] = "; ".join(why)
            Path(str(out) + ".summary.json").write_text(json.dumps(summ))
    ok = counts["readback_bad_points"] == 0 and counts["dma_bad"] == 0 and counts["alloc_errors"] == 0
    print(f"RP {'DONE' if ok else 'FAILED'}: {counts}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
