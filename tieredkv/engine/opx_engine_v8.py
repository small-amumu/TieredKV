# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""OPX engine fix, layout part (issue 161). Patches rp_engine (engine7, md5
48545123) running on kalign_opx (make_kalign_opx.py: kalign_fr + fixed residual row stride + fused records written
by the encoder; defaults bit-identical to kalign_fr, kernel tests in opx_kernel_test.py):
  legacy  engine7's layout (split records, row stride = current coded rows: Arena.grow re-lays every head's history
          out at every window flush); kept for the bitwise comparison
  fixed   split records at a fixed row stride = the item's capacity (cap_rows): a flush only appends 128 rows; no
          host memmove
  fused   fixed stride and fused records (data, fp16 min, fp16 step, zero padding to a 32 B multiple) written directly
          by the encoder; the split arenas are not allocated
The algorithm, the encoder arithmetic and every byte value are unchanged; only where the residual records live.
install(layout, fstrides) must run after rp_engine is imported with kalign_opx as its kalign module."""
from __future__ import annotations

import torch

import rp_engine as E

STATE: dict = {"layout": "legacy", "fstrides": None, "exact_pin": False}


def default_fstrides(kb: int, vb: int) -> tuple[int, int]:
    """record bytes (16 * bits + 4) rounded up to a 32 B multiple."""
    up = lambda x: (x + 31) // 32 * 32  # noqa: E731
    return up(16 * kb + 4), up(16 * vb + 4)


# --exact_pin (E4, issue 173): PyTorch's pinned host allocator rounds every allocation up to a power of two (a 0.0775 GiB
# K-residual arena pins 0.125 GiB; ~1.5 GiB extra at 130K). Instead: an anonymous mmap of exactly the needed bytes
# (zero-filled, page-aligned), page-locked with cudaHostRegister (portable | mapped; kernels read it zero-copy through
# the host pointer, as pinned memory from cudaHostAlloc), wrapped as ndarray -> tensor. torch.from_numpy keeps the
# ndarray alive for as long as any tensor or view of it lives; the finalizer on the ndarray unregisters before the
# mapping is released. Byte values and layout are unchanged.
def _unregister(ptr: int) -> None:
    err = torch.cuda.cudart().cudaHostUnregister(ptr)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostUnregister({ptr:#x}): {err}")


def exact_pinned(n: int, dtype: torch.dtype) -> torch.Tensor:
    """zeroed, page-locked host tensor of exactly n elements of dtype (no power-of-two rounding)."""
    import mmap
    import weakref

    import numpy as np
    nbytes = max(n * torch.empty((), dtype=dtype).element_size(), 1)
    # anonymous mapping: page-aligned (as cudaHostAlloc's), zero-filled, its own pages. The whole mapping is
    # registered, so the storage start is inside the registered range: torch's is_pinned() checks the storage
    # start, and a registered sub-range of a larger buffer reads as not pinned (E4 debug, 2026-11-23)
    mm = mmap.mmap(-1, nbytes)
    arr = np.frombuffer(mm, dtype=np.uint8)                 # keeps mm alive; writable
    ptr = int(arr.ctypes.data)
    err = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 3)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostRegister({nbytes} B): {err}")
    weakref.finalize(arr, _unregister, ptr)                 # runs before the ndarray releases mm
    t = torch.from_numpy(arr).view(dtype)
    assert t.data_ptr() == ptr and ptr % 4096 == 0, (t.data_ptr(), ptr)
    assert t.is_pinned(), "registered mapping not reported as pinned"
    return t


def _pinned(*shape: int, dtype=torch.int8) -> torch.Tensor:
    if STATE["exact_pin"]:
        n = 1
        for s in shape:
            n *= s
        return exact_pinned(n, dtype).view(*shape)
    return torch.zeros(*shape, dtype=dtype).pin_memory()


_ORIG_ARENA_INIT = E.Arena.__init__


def _arena_init_v8(self, BK, cap, w, dtype):
    """engine7's Arena.__init__; with --exact_pin the buffer comes from exact_pinned (same zeros, same size)."""
    if not STATE["exact_pin"]:
        _ORIG_ARENA_INIT(self, BK, cap, w, dtype)
        return
    self.BK, self.cap, self.w, self.dtype = BK, cap, w, dtype
    self.buf = exact_pinned(BK * cap * w, dtype)
    self.es = self.buf.element_size()
    self.L = 0


_ORIG_INIT = E.LayerStore.__init__


def _ls_init(self, li, KVH, GH, kb, vb, cap, prec, calib):
    lay = STATE["layout"]
    if lay == "gpures":
        # diagnostic: split records at the fixed stride, but in GPU memory (same kernels, same rows, no host reads);
        # the PCIe of a normal step is then the non-residual floor of the engine
        self.li, self.KVH, self.GH, self.kb, self.vb, self.prec, self.calib = li, KVH, GH, kb, vb, prec, calib
        self.cap_rows = cap
        self.gar = (torch.zeros(KVH, cap, 16 * kb, dtype=torch.int8, device="cuda"),
                    torch.zeros(KVH, cap, 2, dtype=torch.float16, device="cuda"),
                    torch.zeros(KVH, cap, 16 * vb, dtype=torch.int8, device="cuda"),
                    torch.zeros(KVH, cap, 2, dtype=torch.float16, device="cuda"))
        self.ar = ()
        self.kx = torch.zeros(KVH, E.KA.WBA, E.D, dtype=torch.float16, device="cuda")
        self.vx = torch.zeros_like(self.kx)
        self.st, self.W0, self.Lk, self.nw = None, 0, 0, 0
        self.inexact = torch.zeros((), dtype=torch.int64, device="cuda")
        self.guard = None
        self.gam = torch.tensor([float(x) for x in E.S10.Q6.GAMMAS], dtype=torch.float32, device="cuda")
        self.coded = 0
        return
    if lay != "fused":
        _ORIG_INIT(self, li, KVH, GH, kb, vb, cap, prec, calib)
        self.cap_rows = cap
        return
    # fused: same fields as LayerStore.__init__, but fused arenas instead of the four split arenas
    sk, sv = STATE["fstrides"] or default_fstrides(kb, vb)
    self.li, self.KVH, self.GH, self.kb, self.vb, self.prec, self.calib = li, KVH, GH, kb, vb, prec, calib
    self.cap_rows = cap
    self.far = (_pinned(KVH, cap, sk), _pinned(KVH, cap, sv))
    self.tiny = (_pinned(KVH, 1, 16 * kb), _pinned(KVH, 1, 2, dtype=torch.float16),
                 _pinned(KVH, 1, 16 * vb), _pinned(KVH, 1, 2, dtype=torch.float16))
    self.ar = ()
    self.kx = torch.zeros(KVH, E.KA.WBA, E.D, dtype=torch.float16, device="cuda")
    self.vx = torch.zeros_like(self.kx)
    self.st, self.W0, self.Lk, self.nw = None, 0, 0, 0
    self.inexact = torch.zeros((), dtype=torch.int64, device="cuda")
    self.guard = None
    self.gam = torch.tensor([float(x) for x in E.S10.Q6.GAMMAS], dtype=torch.float32, device="cuda")
    self.coded = 0


# --slim (F4: the 130K prefill must fit a 24 GB RTX 4090): every StepA allocates per-layer scratch that each step
# rewrites before reading (bias, s, acc, cand, pn: ~33 MiB per layer at 130K) and benchmark-only arrays that no
# kalign_opx path reads (ks, kz, vs, vz, mt2b: ~10 MiB). Layers run one after another, so one zero-initialised scratch
# buffer per name serves all of them (StepA zeroes acc / pn itself: unused lanes must start finite); the unused arrays
# are dropped. A buffer grows only inside _new_step, i.e. at a flush, after which the CUDA graph is re-captured.
SHARED: dict = {}
SLIM_SHARE = ("bias", "s", "acc", "cand", "pn")
SLIM_DROP = ("ks", "kz", "vs", "vz", "mt2b")


def _shared_view(name: str, t: torch.Tensor) -> torch.Tensor:
    key = (name, t.dtype)
    buf = SHARED.get(key)
    n = t.numel()
    if buf is None or buf.numel() < n:
        buf = torch.zeros(n + n // 8, dtype=t.dtype, device="cuda")     # 1/8 headroom: fewer regrowths
        SHARED[key] = buf
    return buf[:n].view(t.shape)


def _slim(st) -> None:
    for f in SLIM_DROP:
        if hasattr(st, f):
            setattr(st, f, None)
    for f in SLIM_SHARE:
        setattr(st, f, _shared_view(f, getattr(st, f)))


def _new_step_v8(self, W0):
    """StepA over coded rows [Z, W0) on fixed-stride arenas: nothing already written moves."""
    L = W0 - E.Z
    if L < E.M_CAND:
        raise E.EngineFlag(f"unsupported by the kernel: {L} coded rows < M = {E.M_CAND} (W0 - Z >= M needs W0 >= 256)")
    assert L <= self.cap_rows, (L, self.cap_rows)
    kw = dict(GH=self.GH, M=E.M_CAND, K=E.K_FETCH, lam=E.LAM, kbits=self.kb, vbits=self.vb, host=True, words=True,
              fused=True, res_stride=self.cap_rows)
    if STATE["layout"] == "fused":
        st = E.KA.StepA(L, rh=self.tiny, frec=self.far, **kw)
    elif STATE["layout"] == "gpures":
        kw["host"] = False
        st = E.KA.StepA(L, rh=self.gar, **kw)
    else:
        rh = tuple(a.buf.view(a.BK, a.cap, a.w) for a in self.ar)
        st = E.KA.StepA(L, rh=rh, **kw)
        for a in self.ar:
            a.L = L                                    # accounting only (memory()); no re-layout
    self.coded = L
    if STATE.get("slim"):
        _slim(st)
    st.set_calib(*self.calib)
    st.rope_tables(W0)
    return st


def _check_new_rows_v8(self, st, c0, n, f0):
    """as engine7: unexpressible rows (kernel flags) and zero-step residual rows among the new rows [c0, c0 + n)."""
    torch.cuda.synchronize()
    nflag = int(st.flags.item()) - f0
    if STATE["layout"] == "fused":
        nbk, nbv = 16 * self.kb, 16 * self.vb
        kst = self.far[0][:, c0:c0 + n, nbk + 2:nbk + 4].contiguous().view(torch.float16)
        vst = self.far[1][:, c0:c0 + n, nbv + 2:nbv + 4].contiguous().view(torch.float16)
        nzero = int((kst == 0).sum()) + int((vst == 0).sum())
    elif STATE["layout"] == "gpures":
        nzero = int((self.gar[1][:, c0:c0 + n, 1] == 0).sum()) + int((self.gar[3][:, c0:c0 + n, 1] == 0).sum())
    else:
        krm = self.ar[1].buf.view(self.ar[1].BK, self.ar[1].cap, 2)
        vrm = self.ar[3].buf.view(self.ar[3].BK, self.ar[3].cap, 2)
        nzero = int((krm[:, c0:c0 + n, 1] == 0).sum()) + int((vrm[:, c0:c0 + n, 1] == 0).sum())
    if nflag or nzero:
        raise E.EngineFlag(f"layer {self.li}: {nflag} unexpressible rows, {nzero} zero-step residual rows in [{c0}, {c0 + n})")


def _memory_v8(self):
    """engine7's fields; host_store_gib counts the record bytes of the coded rows in the layout in use (fused: the
    padded record), host_arena_gib the allocated pinned bytes."""
    gpu = sum(sum(getattr(L.st, f).numel() * getattr(L.st, f).element_size() for f in ("kp", "km", "vp", "vm"))
              + L.kx.numel() * 2 * 2 for L in self.layers if L.st is not None)
    if STATE["layout"] == "fused":
        host = sum(L.KVH * L.coded * (L.far[0].shape[-1] + L.far[1].shape[-1]) for L in self.layers)
        arena = sum(t.numel() for L in self.layers for t in L.far)
    elif STATE["layout"] == "gpures":
        host = arena = 0
    else:
        host = sum(sum(a.BK * a.L * a.w * a.es for a in L.ar) for L in self.layers)
        arena = sum(sum(a.buf.numel() * a.es for a in L.ar) for L in self.layers)
    return dict(fp16_inexact=int(sum(int(L.inexact) for L in self.layers)), guard=self.layers[0].guard,
                gpu_store_gib=gpu / 2 ** 30, host_store_gib=host / 2 ** 30, host_arena_gib=arena / 2 ** 30,
                layout=STATE["layout"])


# --seal_profile (diagnostic): wall time of each window flush, split into StepA construction (_new_step), the check
# of the new rows (_check_new_rows, includes a device sync), and the rest (GPU code copies + the 128-row encode),
# summed over the layers of one flush event; synchronised at the boundaries, so it slows flushes slightly
SEAL: dict = {"on": False, "events": [], "cur": None}


def _wrap_seal() -> None:
    import time
    orig_flush, orig_new, orig_chk = E.LayerStore._flush, E.LayerStore._new_step, E.LayerStore._check_new_rows

    def new_step(self, W0):
        if SEAL["cur"] is None:
            return orig_new(self, W0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            return orig_new(self, W0)
        finally:
            torch.cuda.synchronize()
            SEAL["cur"]["new_step"] += time.perf_counter() - t

    def check(self, st, c0, n, f0):
        if SEAL["cur"] is None:
            return orig_chk(self, st, c0, n, f0)
        t = time.perf_counter()
        try:
            return orig_chk(self, st, c0, n, f0)
        finally:
            SEAL["cur"]["check"] += time.perf_counter() - t

    def flush(self):
        new_event = self.li == 0
        if new_event:
            SEAL["events"].append(dict(W0=self.W0, total=0.0, new_step=0.0, check=0.0))
        SEAL["cur"] = SEAL["events"][-1]
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            return orig_flush(self)
        finally:
            torch.cuda.synchronize()
            SEAL["cur"]["total"] += time.perf_counter() - t
            SEAL["cur"] = None
    E.LayerStore._flush, E.LayerStore._new_step, E.LayerStore._check_new_rows = flush, new_step, check
    SEAL["on"] = True


def install(layout: str, fstrides: tuple[int, int] | None = None, seal_profile: bool = False,
            slim: bool = False, exact_pin: bool = False) -> None:
    assert layout in ("legacy", "fixed", "fused", "gpures"), layout
    assert E.KA.__name__ == "kalign_opx", f"rp_engine runs on {E.KA.__name__}, not kalign_opx"
    assert not (slim and layout == "legacy"), "--slim needs a fixed-stride layout"
    STATE["layout"], STATE["fstrides"], STATE["slim"] = layout, fstrides, slim
    STATE["exact_pin"] = exact_pin
    E.LayerStore.__init__ = _ls_init
    E.Arena.__init__ = _arena_init_v8
    if layout != "legacy":
        E.LayerStore._new_step = _new_step_v8
        E.LayerStore._check_new_rows = _check_new_rows_v8
        E.Engine.memory = _memory_v8
    if seal_profile:
        _wrap_seal()
