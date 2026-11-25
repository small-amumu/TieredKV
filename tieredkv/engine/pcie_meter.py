# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""PCIe traffic meter for round OPX (REG_OPX.md section 4): integrates the NVML hardware PCIe
throughput counters of one GPU while a code region runs.

Why NVML and not memcpy counts: our engine and ShadowKV read host rows with zero-copy kernels (no
cudaMemcpy), RetroInfer mixes copies and cache hits; the GPU's own PCIe counters see all of them.

nvmlDeviceGetPcieThroughput returns KB/s averaged over a ~20 ms window (NVML docs). A thread polls
it every `period` seconds and integrates rate x elapsed time. RX = bytes received by the GPU (host to
device, including zero-copy reads), TX = bytes sent by the GPU.

Device: the CUDA device 0 of this process is matched to its NVML handle by UUID
(torch.cuda.get_device_properties(0).uuid), so CUDA_VISIBLE_DEVICES may be anything.

Self-test (run before any formal use, on the card of the run):
  python pcie_meter.py --selftest [--gib 2] [--reps 10]
copies a pinned host tensor to the GPU `reps` times and compares the integrated RX bytes with the
bytes copied; prints the ratio (expected close to 1) and the idle RX rate.
No dependency beyond torch and libnvidia-ml.so.1 (ctypes).
"""
from __future__ import annotations

import argparse
import ctypes
import threading
import time
from dataclasses import dataclass, field

NVML_PCIE_UTIL_TX_BYTES = 0
NVML_PCIE_UTIL_RX_BYTES = 1


class NvmlError(RuntimeError):
    pass


class _Nvml:
    """minimal ctypes binding: init, handle by UUID, PCIe throughput."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            self.lib.nvmlErrorString.restype = ctypes.c_char_p
            msg = self.lib.nvmlErrorString(rc)
            raise NvmlError(f"{what}: NVML error {rc} ({msg.decode() if msg else '?'})")

    def handle_by_uuid(self, uuid: str) -> ctypes.c_void_p:
        h = ctypes.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(h)),
                    f"nvmlDeviceGetHandleByUUID({uuid})")
        return h

    def pcie_kbps(self, h: ctypes.c_void_p, counter: int) -> int:
        v = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetPcieThroughput(h, counter, ctypes.byref(v)),
                    "nvmlDeviceGetPcieThroughput")
        return int(v.value)


def cuda0_uuid() -> str:
    """NVML-style UUID ('GPU-xxxxxxxx-...') of CUDA device 0 of this process."""
    import os
    import torch
    props = torch.cuda.get_device_properties(0)
    if hasattr(props, "uuid"):  # torch >= 2.5
        u = str(props.uuid)
        return u if u.startswith("GPU-") else "GPU-" + u
    # older torch (ShadowKV pins 2.3.1): every OPX launcher sets CUDA_VISIBLE_DEVICES to one full UUID
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if first.startswith("GPU-"):
        return first
    raise RuntimeError("cannot identify CUDA device 0: torch has no props.uuid and "
                       "CUDA_VISIBLE_DEVICES is not a GPU UUID")


@dataclass
class PcieMeter:
    """with PcieMeter() as m: ...; then m.rx_bytes, m.tx_bytes, m.seconds, m.samples."""
    period: float = 0.01
    rx_bytes: float = 0.0
    tx_bytes: float = 0.0
    seconds: float = 0.0
    samples: int = 0
    uuid: str = ""
    error: str | None = None                       # set if the sampler thread died (the reading is then partial)
    series: list = field(default_factory=list)   # [(seconds since start, RX KB/s, TX KB/s)] per sample
    t_start: float = 0.0
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thr: threading.Thread | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._nv = _Nvml()
        self.uuid = cuda0_uuid()
        self._h = self._nv.handle_by_uuid(self.uuid)

    def _run(self) -> None:
        t_prev = time.perf_counter()
        try:
            while not self._stop.is_set():
                time.sleep(self.period)
                rx = self._nv.pcie_kbps(self._h, NVML_PCIE_UTIL_RX_BYTES)
                tx = self._nv.pcie_kbps(self._h, NVML_PCIE_UTIL_TX_BYTES)
                t = time.perf_counter()
                dt = t - t_prev
                t_prev = t
                self.rx_bytes += rx * 1024.0 * dt
                self.tx_bytes += tx * 1024.0 * dt
                self.seconds += dt
                self.samples += 1
                self.series.append((round(t - self.t_start, 4), rx, tx))
        except Exception as e:  # noqa: BLE001  (a dead sampler would undercount silently: record it instead)
            self.error = repr(e)[:300]

    def __enter__(self) -> "PcieMeter":
        self.rx_bytes = self.tx_bytes = self.seconds = 0.0
        self.samples = 0
        self.series = []
        self.error = None
        self.t_start = time.perf_counter()
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thr is not None:
            self._thr.join()

    def as_dict(self, steps: int | None = None) -> dict[str, float | int | str | None]:
        d: dict[str, float | int | str | None] = {
            "pcie_rx_bytes": self.rx_bytes, "pcie_tx_bytes": self.tx_bytes,
            "pcie_seconds": self.seconds, "pcie_samples": self.samples, "pcie_uuid": self.uuid,
            "pcie_error": self.error}
        d["pcie_rx_bytes_per_step"] = self.rx_bytes / steps if steps else None
        return d


def selftest(gib: float, reps: int) -> int:
    import torch
    n = int(gib * 2 ** 30)
    host = torch.empty(n, dtype=torch.uint8).pin_memory()
    dev = torch.empty(n, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    with PcieMeter() as idle:
        time.sleep(2.0)
    with PcieMeter() as m:
        for _ in range(reps):
            dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        time.sleep(0.1)                         # let the last ~20 ms window be sampled
    sent = n * reps
    ratio = m.rx_bytes / sent
    print(f"uuid {m.uuid}; idle RX {idle.rx_bytes / max(idle.seconds, 1e-9) / 2 ** 20:.2f} MiB/s "
          f"over {idle.seconds:.1f} s; copied {sent / 2 ** 30:.2f} GiB in {m.seconds:.2f} s; "
          f"metered RX {m.rx_bytes / 2 ** 30:.3f} GiB; ratio {ratio:.3f}; samples {m.samples}")
    return 0 if 0.8 <= ratio <= 1.25 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gib", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.gib, a.reps)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
