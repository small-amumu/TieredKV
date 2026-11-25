# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""Kernel-level bitwise tests of kalign_opx (make_kalign_opx.py) against kalign_fr (bfc8d073), synthetic data,
one layer at a time is enough (the change is per-layer): B = 1, 8 KV heads, 4 query heads per KV head, d 128.
  a  defaults (RL = L, no frec): residual bytes and step outputs equal kalign_fr
  b  split records at a fixed larger row stride (res_stride = L + 384): outputs equal (a)
  c  fused records written by the encoder (frec, strides from --strides) at a fixed stride: outputs equal (a), and
     the fused bytes equal kalign_fr's fuse_records of (a)'s split records
  d  flush emulation: StepA over L - 128 coded rows, then a new StepA over L rows on the SAME fixed-stride arenas,
     GPU codes copied, only the last 128 rows encoded; outputs equal a single StepA over L rows (split and fused)
Exit 0 only if every check passes.
usage: KALIGN_DIR=<dir with kalign_fr.py, kalign_opx.py and the kernel tree> python opx_kernel_test.py
       [--L 8192] [--codecs 54,43,33,32] [--strides 54:96x96,43:96x64,33:64x64,32:64x64]"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.environ["KALIGN_DIR"])
import kalign_fr as FR  # noqa: E402
import kalign_opx as OX  # noqa: E402

KVH, H, D, G = 8, 32, 128, 4
GAM = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
dev = "cuda"
FAIL: list[str] = []


def check(name: str, ok: bool) -> None:
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    if not ok:
        FAIL.append(name)


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int8), b.contiguous().view(torch.int8))


def calib_tensors(seed: int):
    g = torch.Generator().manual_seed(seed)
    order = torch.stack([torch.randperm(D, generator=g) for _ in range(KVH)])
    bvec = torch.rand(KVH, D, generator=g) * 1.5 + 0.5
    mpre = torch.randn(KVH, D, generator=g) * 0.5
    invf = 1.0 / (500000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    return order, bvec, mpre, 1.0, invf.float()


def pinned(*shape, dtype=torch.int8):
    return torch.zeros(*shape, dtype=dtype).pin_memory()


def split_arenas(rows: int, kb: int, vb: int):
    return (pinned(KVH, rows, 16 * kb), pinned(KVH, rows, 2, dtype=torch.float16),
            pinned(KVH, rows, 16 * vb), pinned(KVH, rows, 2, dtype=torch.float16))


def tiny(kb: int, vb: int):
    return split_arenas(1, kb, vb)


def make(mod, L, kb, vb, cal, rh, kx, vx, q, r0, r1, **kw):
    st = mod.StepA(L, GH=G, B=1, kbits=kb, vbits=vb, host=True, words=True, seed=0, rh=rh, **kw)
    st.set_calib(*cal)
    st.rope_tables(L + mod.Z_SINK)
    if r1 > r0:
        st.encode(kx, vx, r0, r1, GAM)
    return st


def finish(st, kx, vx, q):
    W = st_W(st)
    st.set_exact(kx[:, :W], vx[:, :W])
    st.set_q(q)
    st.step()
    torch.cuda.synchronize()
    return st.result().clone()


def st_W(st) -> int:
    return OX.Z_SINK + 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8192)
    ap.add_argument("--codecs", default="54,43,33,32")
    ap.add_argument("--strides", default="54:96x96,43:96x64,33:64x64,32:64x64")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    strides = {k: tuple(int(x) for x in v.split("x")) for k, v in (t.split(":") for t in a.strides.split(","))}
    L = a.L
    for cd in a.codecs.split(","):
        kb, vb = int(cd[0]), int(cd[1])
        g = torch.Generator().manual_seed(1)
        q = torch.randn(H, D, generator=g).half().to(dev)
        kx = torch.randn(KVH, L + OX.Z_SINK, D, generator=g).half().to(dev)
        vx = torch.randn(KVH, L + OX.Z_SINK, D, generator=g).half().to(dev)
        cal = calib_tensors(7)
        Z = OX.Z_SINK
        # (a) reference kalign_fr vs kalign_opx defaults
        rf = split_arenas(L, kb, vb)
        st_fr = make(FR, L, kb, vb, cal, rf, kx, vx, q, 0, L + Z)
        out_fr = finish(st_fr, kx, vx, q)
        ro = split_arenas(L, kb, vb)
        st_ox = make(OX, L, kb, vb, cal, ro, kx, vx, q, 0, L + Z)
        out_ox = finish(st_ox, kx, vx, q)
        check(f"h{cd} a: residual bytes equal", all(same(x, y) for x, y in zip(rf, ro)))
        check(f"h{cd} a: GPU codes equal", all(same(getattr(st_fr, f), getattr(st_ox, f)) for f in ("kp", "km", "vp", "vm")))
        check(f"h{cd} a: outputs equal", same(out_fr, out_ox))
        # (b) split at a fixed larger stride
        R = L + 384
        rb = split_arenas(R, kb, vb)
        st_b = make(OX, L, kb, vb, cal, rb, kx, vx, q, 0, L + Z, res_stride=R)
        out_b = finish(st_b, kx, vx, q)
        check(f"h{cd} b: stride {R} rows equal (a)'s rows", all(same(x[:, :L], y) for x, y in zip(rb, ro)))
        check(f"h{cd} b: outputs equal", same(out_b, out_fr))
        # (c) fused records written by the encoder at a fixed stride
        sk, sv = strides[cd]
        kf, vf = pinned(KVH, R, sk), pinned(KVH, R, sv)
        st_c = make(OX, L, kb, vb, cal, tiny(kb, vb), kx, vx, q, 0, L + Z, res_stride=R, frec=(kf, vf))
        out_c = finish(st_c, kx, vx, q)
        st_fr.fuse_records(sk, sv)
        check(f"h{cd} c: fused bytes equal fuse_records", same(kf[:, :L], st_fr.kf) and same(vf[:, :L], st_fr.vf))
        check(f"h{cd} c: fused padding zero beyond the coded rows", int(kf[:, L:].abs().sum()) == 0)
        check(f"h{cd} c: outputs equal", same(out_c, out_fr))
        st_fr.unfuse_records()
        # (d) flush emulation on the same fixed-stride arenas (split and fused)
        for tag, kw_rh, kw in (("split", None, {}), ("fused", None, {"frec": None})):
            L1 = L - 128
            if tag == "split":
                arenas = split_arenas(R, kb, vb)
                kw1 = dict(res_stride=R)
                rh = arenas
            else:
                kf2, vf2 = pinned(KVH, R, sk), pinned(KVH, R, sv)
                kw1 = dict(res_stride=R, frec=(kf2, vf2))
                rh = tiny(kb, vb)
            st1 = make(OX, L1, kb, vb, cal, rh, kx, vx, q, 0, L1 + Z, **kw1)
            st2 = make(OX, L, kb, vb, cal, rh, kx, vx, q, 0, 0, **kw1)
            for f in ("kp", "km", "vp", "vm"):
                getattr(st2, f)[:, :L1].copy_(getattr(st1, f)[:, :L1])
            st2.encode(kx, vx, L1 + Z, L + Z, GAM)
            out_d = finish(st2, kx, vx, q)
            check(f"h{cd} d-{tag}: GPU codes equal single build",
                  all(same(getattr(st2, f), getattr(st_ox, f)) for f in ("kp", "km", "vp", "vm")))
            check(f"h{cd} d-{tag}: outputs equal single build", same(out_d, out_fr))
            del st1, st2
        # (e) CUDA graph path: W from a device int32 (WDEV), window buffer copied whole with garbage beyond W,
        #     the step captured once and replayed for several W; outputs equal the python-W set_exact path
        st_e = make(OX, L, kb, vb, cal, split_arenas(R, kb, vb), kx, vx, q, 0, L + Z, res_stride=R)
        wdev = torch.zeros(1, dtype=torch.int32, device=dev)
        kxbuf = torch.randn(KVH, OX.WBA, D, generator=g).half().to(dev)       # garbage beyond W
        vxbuf = torch.randn(KVH, OX.WBA, D, generator=g).half().to(dev)
        qbuf = torch.empty(H, D, dtype=torch.float16, device=dev)
        for Wv in (Z + 1, 64, OX.WBA):
            Wv = min(Wv, OX.WBA)
            kxbuf[:, :Wv].copy_(kx[:, :Wv]); vxbuf[:, :Wv].copy_(vx[:, :Wv])
            # reference: python W through set_exact (zeros beyond W)
            st_e.Wdev = None
            st_e.set_exact(kxbuf[:, :Wv], vxbuf[:, :Wv])
            st_e.set_q(q)
            st_e.step()
            torch.cuda.synchronize()
            ref_e = st_e.result().clone()
            # device W, whole buffer (garbage beyond W), eager
            st_e.Wdev = wdev
            wdev.fill_(Wv)
            st_e.kw.copy_(kxbuf); st_e.vw.copy_(vxbuf)
            st_e.set_q(q)
            st_e.step()
            torch.cuda.synchronize()
            check(f"h{cd} e: W={Wv} device-W eager equals python W", same(st_e.result(), ref_e))
            # captured once per W value here (the engine captures once per flush epoch)
            qbuf.copy_(q)
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                st_e.kw.copy_(kxbuf); st_e.vw.copy_(vxbuf); st_e.set_q(qbuf); st_e.step()
            torch.cuda.current_stream().wait_stream(s_)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                st_e.kw.copy_(kxbuf); st_e.vw.copy_(vxbuf); st_e.set_q(qbuf); st_e.step()
            gr.replay()
            torch.cuda.synchronize()
            check(f"h{cd} e: W={Wv} graph replay equals python W", same(st_e.result(), ref_e))
            del gr
        st_e.Wdev = None
        del st_fr, st_ox, st_b, st_c, st_e
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("KERNEL TEST " + ("PASS" if not FAIL else f"FAIL {FAIL}"), flush=True)
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
