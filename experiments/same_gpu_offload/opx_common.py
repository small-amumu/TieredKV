# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""Shared pieces of the three OPX runners (REG_OPX.md): RULER task constants (vt, cwe, fwe, qa_1,
qa_2), scoring as tb_bench v3.7 (qa_2 scored like qa_1, RULER's string_match_part), host RSS, and
the decode meter (per-forward wall time + NVML PCIe bytes over the metered forwards).

Decode loop contract (the same in all three runners):
    dm = DecodeMeter(warm)            # warm = ACC_WARM in accuracy mode, TIMING_WARM in timing mode
    for each decode forward f = 0, 1, ...:
        dm.before(f); <forward, argmax, int() (host sync)>; dm.after(f)
    rec.update(dm.finish())
Forwards with index < warm are neither timed nor metered.
"""
from __future__ import annotations

import gzip
import os
import shutil
import statistics
import time
from typing import Callable

from pcie_meter import PcieMeter

RULER_MAXGEN = {"vt": 30, "cwe": 120, "fwe": 50, "qa_1": 32, "qa_2": 32}   # RULER constants
RULER_PART = ("qa_1", "qa_2")
END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>",
              "<|end_of_text|>")
TIMING_WARM = 16          # timing mode: forwards 0-15 are warm-up (CUDA graphs, caches)
ACC_WARM = 1              # accuracy mode: forward 0 not metered (one-time work, e.g. our engine's layer-0
                          # guard copies the whole exact layer-0 K / V to the GPU at the first step)
TIMING_STEPS = 256        # timing mode: tokens produced per item, EOS ignored
IDLE_SEC = 1.0            # timing mode: idle PCIe background measured before metering, then subtracted
PROFILE_STEPS = 24        # timing mode: extra forwards after the timed ones, recorded by torch.profiler
EXTRA_STEPS = PROFILE_STEPS   # buffer headroom the runners add for the profiled forwards


def ruler_score(pred: str, refs: list[str], kind: str) -> float:
    hit = [1.0 if r.lower() in pred.lower() else 0.0 for r in refs]
    return max(hit) if kind in RULER_PART else sum(hit) / len(refs)


def rss_gib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30


def reset_rss_peak() -> bool:
    """resets VmHWM (Linux clear_refs '5'), so rss_peak_gib() covers one item; False if not supported."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def rss_peak_gib() -> float | None:
    """VmHWM: peak resident set of this process since start or the last reset_rss_peak() (GiB)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 2 ** 20          # kB
    return None


def host_tensor_census(top: int = 8) -> dict:
    """host-memory metric of REG_OPX (primary): bytes of every live CPU torch tensor storage that the
    garbage collector can reach, each storage counted once (views share it), split into pinned and
    pageable; plus the `top` largest storages as [GiB, pinned, shape, dtype] of the first tensor seen on
    each, so the census can be audited. Memory held outside torch tensors (C++ buffers, numpy arrays,
    allocator slack) is not seen here: the length-differenced RSS probe checks for it."""
    import gc
    import warnings

    import torch
    gc.collect()                                  # unreachable cycles holding CPU tensors are not counted (review M1)
    seen: dict[int, tuple[int, bool, list[int], str]] = {}
    with warnings.catch_warnings():                               # isinstance() on deprecated torch objects warns
        warnings.simplefilter("ignore")
        objs = gc.get_objects()
        objs = [o for o in objs if isinstance(o, torch.Tensor)]
    for o in objs:
        try:
            if not isinstance(o, torch.Tensor) or o.device.type != "cpu" or o.is_sparse:
                continue
            st = o.untyped_storage()
            p = st.data_ptr()
            if p == 0 or p in seen:
                continue
            seen[p] = (st.nbytes(), bool(o.is_pinned()), list(o.shape), str(o.dtype))
        except Exception:                                         # noqa: BLE001  (meta / opaque tensors)
            continue
    tot = sum(v[0] for v in seen.values())
    pin = sum(v[0] for v in seen.values() if v[1])
    big = sorted(seen.values(), key=lambda v: -v[0])[:top]
    return {"host_tensor_gib": tot / 2 ** 30, "host_pinned_gib": pin / 2 ** 30,
            "host_tensor_storages": len(seen),
            "host_tensor_top": [[round(v[0] / 2 ** 30, 4), v[1], v[2], v[3]] for v in big]}


MEMPROBE = {"on": False}  # set by the runners' --mem_probe (host-memory check runs only, never in timing runs)


def _rss_parts() -> dict[str, float]:
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            k = line.split(":")[0]
            if k in ("RssAnon", "RssFile", "RssShmem"):
                out[k] = int(line.split()[1]) / 2 ** 20               # kB -> GiB
    return out


def mem_probe_fields() -> dict:
    """--mem_probe only: separates allocator slack from live non-tensor memory. Records the RSS split
    (anon / file / shmem), then releases torch's cached pinned host blocks and glibc's free heap
    (malloc_trim) and records RSS again. Called once, after the census, before the first decode."""
    if not MEMPROBE["on"]:
        return {}
    import ctypes
    import gc

    import torch
    pre = _rss_parts()
    gc.collect()
    host_empty = hasattr(torch._C, "_host_emptyCache")
    if host_empty:
        torch._C._host_emptyCache()
    try:
        trimmed = bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except OSError:
        trimmed = None
    return {"rss_parts_pre": pre, "rss_trim_gib": rss_gib(), "rss_parts_trim": _rss_parts(),
            "host_empty_cache": host_empty, "malloc_trim_released": trimmed}


def truncate_ids(ids, n: int):
    """memory probe (--trunc n, REG_OPX host-memory check): BOS + the last n - 1 prompt tokens, so the
    question and answer prefix stay; the probe is never scored."""
    import torch
    if ids.shape[1] <= n:
        raise SystemExit(f"--trunc {n}: prompt has only {ids.shape[1]} tokens")
    return torch.cat([ids[:, :1], ids[:, -(n - 1):]], dim=1)


ACTIVE_METERS: dict = {}        # id -> PCIe sampler started by DecodeMeter.before and not yet finished
                                # (PcieMeter is a dataclass, so unhashable; the dict holds it alive)


def stop_active_meters() -> int:
    """stops every PCIe sampler left running by a decode that raised (otherwise it keeps polling NVML for the rest of
    the process, next to the meter of the retried item). Called on every error path of the runners."""
    n = 0
    for k, m in list(ACTIVE_METERS.items()):
        m.__exit__()
        ACTIVE_METERS.pop(k, None)
        n += 1
    return n


class DecodeMeter:
    """times and meters decode forwards with index >= warm."""

    def __init__(self, warm: int, idle_sec: float | None = None) -> None:
        self.warm = warm
        # timing mode (warm > 0): before metering, the GPU idles idle_sec seconds (the previous
        # forward ended with a host sync) and the PCIe background rate is measured, to be subtracted
        self.idle_sec = (IDLE_SEC if warm >= TIMING_WARM else 0.0) if idle_sec is None else idle_sec
        self.idle_rx_rate: float | None = None
        self.idle_error: str | None = None
        self.sec: list[float] = []
        self.t_fwd: list[float] = []      # start of each metered forward, seconds since the meter started
        self.meter: PcieMeter | None = None
        self._t = 0.0

    def before(self, f: int) -> None:
        if f == self.warm and self.meter is None:
            if self.idle_sec > 0:
                with PcieMeter() as idle:
                    time.sleep(self.idle_sec)
                self.idle_rx_rate = idle.rx_bytes / max(idle.seconds, 1e-9)
                self.idle_error = idle.error                # a dead idle sampler leaves the background unknown
            self.meter = PcieMeter().__enter__()
            ACTIVE_METERS[id(self.meter)] = self.meter
        self._t = time.perf_counter()

    def after(self, f: int) -> None:
        if f >= self.warm:
            self.sec.append(time.perf_counter() - self._t)
            if self.meter is not None:
                self.t_fwd.append(round(self._t - self.meter.t_start, 4))

    def finish(self) -> dict[str, float | int | str | None]:
        n = len(self.sec)
        out: dict = {
            "metered_forwards": n,
            # raw per-forward wall times (ms, 0.01 ms resolution) for later analysis (timing mode only)
            "decode_ms_list": [round(x * 1e3, 2) for x in self.sec] if self.warm >= TIMING_WARM else None,
            "decode_ms_median": statistics.median(self.sec) * 1e3 if n else None,
            "decode_ms_mean": sum(self.sec) / n * 1e3 if n else None}
        if self.meter is not None:
            self.meter.__exit__()
            ACTIVE_METERS.pop(id(self.meter), None)
            out.update(self.meter.as_dict(n))
            out["pcie_idle_rx_rate"] = self.idle_rx_rate
            if self.idle_error and not out.get("pcie_error"):
                out["pcie_error"] = "idle meter: " + self.idle_error   # the analysis leaves such readings out
            if self.warm >= TIMING_WARM:              # timing mode: the raw PCIe series and forward starts
                out["pcie_series"] = self.meter.series
                out["forward_starts"] = self.t_fwd
            if self.idle_rx_rate is not None and n:
                net = self.meter.rx_bytes - self.idle_rx_rate * self.meter.seconds
                out["pcie_rx_bytes_per_step_net"] = net / n
        else:
            out.update({"pcie_rx_bytes": None, "pcie_rx_bytes_per_step": None})
        return out


def profile_forwards(fwd: Callable[[], None], n: int, path: str) -> dict:
    """runs fwd() n times under torch.profiler (CPU + CUDA activities) after the timed forwards, so
    the profiler overhead never enters the timing; writes <path>.json.gz (Chrome / Perfetto trace) and
    <path>.txt (operator table sorted by GPU time). Returns the file names and the profiled wall time."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(n):
            fwd()
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    raw = path + ".json"
    prof.export_chrome_trace(raw)
    with open(raw, "rb") as f_in, gzip.open(raw + ".gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(raw)
    table = ""
    for key in ("device_time_total", "cuda_time_total"):          # torch >= 2.4 / older
        try:
            table = prof.key_averages().table(sort_by=key, row_limit=60)
            break
        except Exception:                                         # noqa: BLE001
            continue
    with open(path + ".txt", "w", encoding="utf-8") as f:
        f.write(table)
    return {"trace_file": os.path.basename(raw + ".gz"), "trace_table": os.path.basename(path + ".txt"),
            "trace_forwards": n, "trace_wall_ms_per_forward": wall / n * 1e3}
