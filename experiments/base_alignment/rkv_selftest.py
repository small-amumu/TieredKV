# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""CPU self-test of rkv_q against the official RotateKV code (fake_quant.py copied verbatim from the repo @ 3055e2b6).
1. fake_quant: rkv_q.fake_quant == official fake_quant (bits 2, 3, 4; shapes of keys [1, 1, L, KVH*d] and values).
2. transforms: k_inverse(k_forward(x)) == x without quantisation; the per-head value Hadamard is an involution.
3. official order: quantising in the rotated + reordered space equals the official sequence written out by hand
   (rotate with a dense Walsh-Hadamard matrix built independently, gather by IDX, fake_quant, gather by argsort, rotate).
4. pivots: set_pivots == official topk(k=20) over the flattened prompt hidden states, token index = flat // hidden.
Usage: python3 -I rkv_selftest.py <path to official fake_quant.py>
"""
from __future__ import annotations

import importlib.util
import math
import sys

import torch

sys.path.insert(0, "/home/user/kvt_bench/hc_mac/code")
sys.path.insert(0, "/home/user/kvt_bench/hc_mac/rootcause")
import rkv_q as R  # noqa: E402


def load_official(path: str):
    spec = importlib.util.spec_from_file_location("off_fq", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sylvester(n: int) -> torch.Tensor:
    h = torch.tensor([[1.0]], dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.kron(torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64), h)
    return (h / math.sqrt(n)).float()


def main() -> None:
    off = load_official(sys.argv[1])
    torch.manual_seed(0)
    fails = []
    for bits in (2, 3, 4):
        for shape, gs in (((1, 1, 300, 1024), 128), ((1, 8, 300, 128), 128), ((1, 1, 300, 256), 64)):
            x = torch.randn(*shape) * 3 + 0.5
            a, b = R.fake_quant(x, 1.0, bits, gs), off.fake_quant(x, 1.0, bits, gs)
            if not torch.equal(a, b):
                fails.append(f"fake_quant bits {bits} shape {shape}: max diff {(a - b).abs().max():.3e}")
    for KVH, d in ((8, 128), (4, 128), (2, 64), (32, 64)):
        li = 1000 + KVH
        L = 50
        k = torch.randn(KVH, L, d)
        R.IDX[li] = torch.randperm(KVH * d)
        back = R.k_inverse(R.k_forward(k, li), li, KVH, d)
        if (back - k).abs().max() > 1e-4:
            fails.append(f"k round trip KVH {KVH} d {d}: {(back - k).abs().max():.3e}")
        g = R.group_of(KVH)
        Hg = sylvester(g * d)
        x = (k.permute(1, 0, 2).reshape(L, KVH // g, g * d) @ Hg.T).reshape(L, KVH * d)[:, R.IDX[li]]
        if (x - R.k_forward(k, li)).abs().max() > 1e-4:
            fails.append(f"k_forward vs independent Sylvester KVH {KVH} d {d}")
        xq = off.fake_quant(x[None, None], 1.0, 2, d)[0, 0]
        ref = (xq[:, R.IDX[li].argsort()].reshape(L, KVH // g, g * d) @ Hg).reshape(L, KVH, d).permute(1, 0, 2)
        ours = R.k_inverse(R.fake_quant(R.k_forward(k, li)[None, None], 1.0, 2, d)[0, 0], li, KVH, d)
        if (ref - ours).abs().max() > 1e-4:
            fails.append(f"official order KVH {KVH} d {d}: {(ref - ours).abs().max():.3e}")
        Hd = R.hadamard(d, torch.device("cpu"))
        if (Hd @ Hd - torch.eye(d)).abs().max() > 1e-5:
            fails.append(f"value Hadamard d {d} not involutory")
    hs = torch.randn(400, 64)
    hs[7, 3] = 90.0
    hs[123, 9] = -80.0
    R.set_pivots(5, hs, 300)
    _, top = torch.topk(hs[:300].abs().flatten(), k=20)
    want = torch.unique(top // 64)
    if not torch.equal(R.PIV[5], want) or 7 not in R.PIV[5].tolist() or 123 not in R.PIV[5].tolist():
        fails.append("pivots differ from the official topk rule")
    # 5. end to end: rkv_base on post-RoPE keys vs an independent implementation (own RoPE, official quantiser).
    import hashlib
    from pathlib import Path
    for KVH, d, L in ((8, 128, 300), (2, 64, 200)):
        li = 2000 + KVH
        inv = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
        R.RC.R["inv_freq"], R.RC.R["asc"] = inv.float(), 1.0
        pos = torch.arange(L, dtype=torch.float64)[:, None] * inv[None]
        cos, sin = torch.cat([pos, pos], -1).cos().float(), torch.cat([pos, pos], -1).sin().float()

        def rope(x):
            a, b = x.chunk(2, -1)
            return x * cos + torch.cat((-b, a), -1) * sin

        kpre = torch.randn(KVH, L, d) * 2 + 0.3
        k, v = rope(kpre), torch.randn(KVH, L, d)
        R.IDX[li] = torch.randperm(KVH * d)
        R.RST["scope"] = []
        R.PIV[li] = torch.tensor([0, 5, 17])
        kq, vq, Rw, _ = R.rkv_base(k, v, li, 2, 0)
        g = R.group_of(KVH)
        Hg, Hd = sylvester(g * d), sylvester(d)
        x = (kpre.permute(1, 0, 2).reshape(L, KVH // g, g * d) @ Hg.T).reshape(L, KVH * d)[:, R.IDX[li]]
        xq = off.fake_quant(x[None, None], 1.0, 2, d)[0, 0]
        ref_pre = (xq[:, R.IDX[li].argsort()].reshape(L, KVH // g, g * d) @ Hg).reshape(L, KVH, d).permute(1, 0, 2)
        ref_k = rope(ref_pre)
        ref_v = off.fake_quant((v @ Hd.T)[None], 1.0, 2, d)[0] @ Hd
        ref_k[:, R.PIV[li]], ref_v[:, R.PIV[li]] = k[:, R.PIV[li]], v[:, R.PIV[li]]
        ek, ev = (kq.float() - ref_k).abs().max(), (vq.float() - ref_v).abs().max()
        if ek > 1e-3 or ev > 1e-3 or Rw != 1:
            fails.append(f"rkv_base end to end KVH {KVH} d {d}: key diff {ek:.3e}, value diff {ev:.3e}, R {Rw}")
    print(f"rkv_q.py md5 {hashlib.md5(Path(R.__file__).read_bytes()).hexdigest()}")
    print("ALL PASS" if not fails else "FAIL\n" + "\n".join(fails))


if __name__ == "__main__":
    main()
