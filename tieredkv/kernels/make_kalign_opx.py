# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Make kalign_opx.py from kalign_fr.py (md5 bfc8d073; RHR part C). Every edit is asserted. Engine fix round
(OPX, issue 161):
  (1) fixed residual row stride: afix2A / afix2B take RL (residual record row stride, rows per (batch, KV head));
      residual records are read at row bk * RL + tok, everything else (GPU codes, bias, masks) still at bk * L + tok.
      StepA(res_stride=R) passes RL = R (default: L, the previous behaviour) and the encoder writes at the same stride
      (aenc_kernel's existing RLC argument), so a store grown by 128 rows keeps every old host byte in place: no
      re-layout (Arena.grow memmove) at window flushes.
  (2) fused records written by the encoder: _enc_res takes RSB (record stride bytes) and FW (min / step inline right
      after the data bytes); aenc_kernel takes KRSB / VRSB (0 = the split layout, as before). The bytes equal
      StepA.fuse_records' layout (data, fp16 min, fp16 step, zero padding). StepA(frec=(kf, vf)) uses caller-provided
      fused arenas [BK, rows, stride bytes] for both encoding and reading.
Defaults (RL = L, KRSB = VRSB = 0, no frec) reproduce kalign_fr bit for bit (test: opx_kernel_test.py)."""
from __future__ import annotations

import hashlib
from pathlib import Path

D = Path(__file__).resolve().parent
src = D / "kalign_fr.py"
assert hashlib.md5(src.read_bytes()).hexdigest().startswith("bfc8d073")
s = src.read_text(encoding="utf-8")


def rep(a: str, b: str, n: int = 1) -> None:
    global s
    k = s.count(a)
    assert k == n, (k, n, a[:90])
    s = s.replace(a, b)


rep('"""kalign_fr = kalign.py f96b21f5 + fused host residual records (make_kalign_fr.py; RHR, 2026-11-19).',
    '"""kalign_opx = kalign_fr.py bfc8d073 + fixed residual row stride (RL / res_stride) + fused records written by the\n'
    'encoder (KRSB / VRSB / frec); make_kalign_opx.py (OPX engine fix, 2026-11-23). Defaults = kalign_fr bit for bit.\n'
    'kalign_fr = kalign.py f96b21f5 + fused host residual records (make_kalign_fr.py; RHR, 2026-11-19).')

# ---- (1) reads at a fixed residual row stride
rep("           FUSED: tl.constexpr = False, RSW: tl.constexpr = 0):\n"
    '    """kres.fix2A_res',
    "           FUSED: tl.constexpr = False, RSW: tl.constexpr = 0, RL=0):\n"
    '    """kres.fix2A_res')
rep("           WORDS: tl.constexpr, CACHE: tl.constexpr = False, FUSED: tl.constexpr = False, RSW: tl.constexpr = 0):",
    "           WORDS: tl.constexpr, CACHE: tl.constexpr = False, FUSED: tl.constexpr = False, RSW: tl.constexpr = 0,\n"
    "           RL=0):")
rep("    row = bk.to(tl.int64) * L + tok\n",
    "    row = bk.to(tl.int64) * L + tok\n"
    "    rrow = bk.to(tl.int64) * RL + tok                     # residual record row (RL = the residual row stride)\n",
    n=2)
rep("_rec_fetch(krp_ptr, krm_ptr, row,", "_rec_fetch(krp_ptr, krm_ptr, rrow,")
rep("_res_rows_wf(krp_ptr, row, row >= 0,", "_res_rows_wf(krp_ptr, rrow, row >= 0,")
rep("_res_rows_w(krp_ptr, krm_ptr, row, row >= 0,", "_res_rows_w(krp_ptr, krm_ptr, rrow, row >= 0,")
rep("_res_rows(krp_ptr, krm_ptr, row, row >= 0,", "_res_rows(krp_ptr, krm_ptr, rrow, row >= 0,")
rep("_rec_fetch(vrp_ptr, vrm_ptr, row,", "_rec_fetch(vrp_ptr, vrm_ptr, rrow,")
rep("_res_rows_wf(vrp_ptr, row, sel,", "_res_rows_wf(vrp_ptr, rrow, sel,")
rep("_res_rows_w(vrp_ptr, vrm_ptr, row, sel,", "_res_rows_w(vrp_ptr, vrm_ptr, rrow, sel,")
rep("_res_rows(vrp_ptr, vrm_ptr, row, sel,", "_res_rows(vrp_ptr, vrm_ptr, rrow, sel,")
rep("CACHE=self.s5, FUSED=fz, RSW=self.rswk if fz else 0, num_warps=4)",
    "CACHE=self.s5, FUSED=fz, RSW=self.rswk if fz else 0,\n"
    "                                                RL=self.res_stride or self.L, num_warps=4)")
rep("VB=self.vbits, WORDS=self.words, CACHE=self.s5, FUSED=fz, RSW=self.rswv if fz else 0,\n"
    "                                     num_warps=4)",
    "VB=self.vbits, WORDS=self.words, CACHE=self.s5, FUSED=fz, RSW=self.rswv if fz else 0,\n"
    "                                     RL=self.res_stride or self.L, num_warps=4)")

# ---- (2) fused records written by the encoder
rep("def _enc_res(r, rp_ptr, rm_ptr, row, rmask, BITS: tl.constexpr, TB: tl.constexpr):",
    "def _enc_res(r, rp_ptr, rm_ptr, row, rmask, BITS: tl.constexpr, TB: tl.constexpr, RSB: tl.constexpr,\n"
    "             FW: tl.constexpr):")
rep("        tl.store(rp_ptr + row[:, None] * NBY + oj[None, :], (lo | (hi << 4)).to(tl.int8), mask=rmask[:, None])",
    "        tl.store(rp_ptr + row[:, None] * RSB + oj[None, :], (lo | (hi << 4)).to(tl.int8), mask=rmask[:, None])")
rep("        tl.store(rp_ptr + row[:, None] * NBY + ob[None, :], byte.to(tl.int8), mask=rmask[:, None] & (ob < NBY)[None, :])",
    "        tl.store(rp_ptr + row[:, None] * RSB + ob[None, :], byte.to(tl.int8), mask=rmask[:, None] & (ob < NBY)[None, :])")
rep("    tl.store(rm_ptr + row * 2, mn.to(tl.float16), mask=rmask)\n"
    "    tl.store(rm_ptr + row * 2 + 1, st.to(tl.float16), mask=rmask)\n",
    "    if FW:             # fused record: fp16 min, fp16 step right after the data bytes (StepA.fuse_records' layout)\n"
    "        rp16 = rp_ptr.to(tl.pointer_type(tl.float16))\n"
    "        tl.store(rp16 + row * (RSB // 2) + NBY // 2, mn.to(tl.float16), mask=rmask)\n"
    "        tl.store(rp16 + row * (RSB // 2) + NBY // 2 + 1, st.to(tl.float16), mask=rmask)\n"
    "    else:\n"
    "        tl.store(rm_ptr + row * 2, mn.to(tl.float16), mask=rmask)\n"
    "        tl.store(rm_ptr + row * 2 + 1, st.to(tl.float16), mask=rmask)\n")
rep("                ZS: tl.constexpr, KV: tl.constexpr = 8, PART: tl.constexpr = 0, RLC=0, ROFF=0):",
    "                ZS: tl.constexpr, KV: tl.constexpr = 8, PART: tl.constexpr = 0, RLC=0, ROFF=0,\n"
    "                KRSB: tl.constexpr = 0, VRSB: tl.constexpr = 0):")
rep("        _enc_res(rk, krp_ptr, krm_ptr, rrow, ok, KBITS, TB)",
    "        _enc_res(rk, krp_ptr, krm_ptr, rrow, ok, KBITS, TB, KRSB if KRSB > 0 else 16 * KBITS, KRSB > 0)")
rep("        _enc_res(rvh, vrp_ptr, vrm_ptr, rrow, ok, VBITS, TB)",
    "        _enc_res(rvh, vrp_ptr, vrm_ptr, rrow, ok, VBITS, TB, VRSB if VRSB > 0 else 16 * VBITS, VRSB > 0)")
# any other _enc_res call site would now miss arguments: there must be none
assert s.count("_enc_res(") == 3, s.count("_enc_res(")

# ---- StepA: res_stride, frec
rep("    def __init__(self, Lc, GH=4, M=128, K=64, lam=-50, kbits=3, vbits=3, host=True, words=True, seed=0, B=1, rh=None,\n"
    "                 fused=True):",
    "    def __init__(self, Lc, GH=4, M=128, K=64, lam=-50, kbits=3, vbits=3, host=True, words=True, seed=0, B=1, rh=None,\n"
    "                 fused=True, res_stride=0, frec=None):")
rep("        assert self.m8 == 0\n        self.GH, self.lam, self.fused = GH, lam, bool(fused)\n",
    "        assert self.m8 == 0\n        self.GH, self.lam, self.fused = GH, lam, bool(fused)\n"
    "        # OPX: residual records at a fixed row stride (rows per (batch, KV head)); 0 = L (kalign_fr behaviour)\n"
    "        self.res_stride = int(res_stride)\n"
    "        assert self.res_stride == 0 or self.res_stride >= Lc, (self.res_stride, Lc)\n"
    "        self.frec_on = frec is not None\n"
    "        if self.frec_on:             # caller-provided fused arenas [BK, rows, stride bytes] (encode and read)\n"
    "            kf, vf = frec\n"
    "            assert kf.shape[-1] % 32 == 0 and vf.shape[-1] % 32 == 0, (kf.shape, vf.shape)\n"
    "            assert kf.shape[-1] >= 16 * kbits + 4 and vf.shape[-1] >= 16 * vbits + 4\n"
    "            assert kf.shape[1] >= (self.res_stride or Lc) and vf.shape[1] >= (self.res_stride or Lc)\n"
    "            cv = (lambda t: t) if kf.is_cuda else KR.K4.host_as_cuda\n"
    "            self.kf, self.kf_, self.vf, self.vf_ = kf, cv(kf), vf, cv(vf)\n"
    "            self.rec_fused, self.rswk, self.rswv = True, kf.shape[-1] // 4, vf.shape[-1] // 4\n")
rep("            rk, rkm, rv_, rvm = self.krp_, self.krm_, self.vrp_, self.vrm_\n"
    "            rlc, roff = 0, 0\n",
    "            if getattr(self, \"frec_on\", False):\n"
    "                rk, rkm, rv_, rvm = self.kf_, self.kf_, self.vf_, self.vf_\n"
    "                krsb, vrsb = self.kf.shape[-1], self.vf.shape[-1]\n"
    "            else:\n"
    "                rk, rkm, rv_, rvm = self.krp_, self.krm_, self.vrp_, self.vrm_\n"
    "            rlc, roff = getattr(self, \"res_stride\", 0), 0\n")
rep("        if stage:\n            assert self.krp.is_pinned(), \"enc_stage needs a pinned host store (async D2H)\"\n",
    "        krsb = vrsb = 0\n"
    "        if stage:\n            assert self.krp.is_pinned(), \"enc_stage needs a pinned host store (async D2H)\"\n"
    "            assert not getattr(self, \"frec_on\", False) and not getattr(self, \"res_stride\", 0), \"OPX: staging is split / L-stride only\"\n")
rep("            ZS=Z_SINK, KV=KVH, PART=0, RLC=rlc, ROFF=roff, num_warps=self.enc_warps, enable_fp_fusion=False)",
    "            ZS=Z_SINK, KV=KVH, PART=0, RLC=rlc, ROFF=roff, KRSB=krsb, VRSB=vrsb, num_warps=self.enc_warps,\n"
    "            enable_fp_fusion=False)")
# ---- (3) CUDA graph: the window row count W read from a device int32 when
#      WDEV (constexpr), so one captured decode step serves every window size of a flush epoch; same loop, same masks
rep("def awin_kernel(q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, wsp, nsp, sm_scale, GH: tl.constexpr,\n"
    "                GP: tl.constexpr, D: tl.constexpr, WB: tl.constexpr, WC: tl.constexpr = 32):\n"
    "    _awin_body(tl.program_id(0), q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, wsp, nsp, sm_scale, GH, GP,\n"
    "               D, WB, WC)",
    "def awin_kernel(q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, wsp, nsp, sm_scale, GH: tl.constexpr,\n"
    "                GP: tl.constexpr, D: tl.constexpr, WB: tl.constexpr, WC: tl.constexpr = 32, WDEV: tl.constexpr = False):\n"
    "    _awin_body(tl.program_id(0), q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, wsp, nsp, sm_scale, GH, GP,\n"
    "               D, WB, WC, WDEV)")
rep("                  WB: tl.constexpr):\n"
    '    """one launch per layer step: grid row 0 is the exact split',
    "                  WB: tl.constexpr, WDEV: tl.constexpr = False):\n"
    '    """one launch per layer step: grid row 0 is the exact split')
rep("        _awin_body(bk, q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, nsb, nsp, sm_scale, GH, GP, 4 * P, WB, 32)",
    "        _awin_body(bk, q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, nsb, nsp, sm_scale, GH, GP, 4 * P, WB, 32,\n"
    "                   WDEV)")
rep("def _awin_body(bk, q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, wsp, nsp, sm_scale, GH: tl.constexpr,\n"
    "               GP: tl.constexpr, D: tl.constexpr, WB: tl.constexpr, WC: tl.constexpr):",
    "def _awin_body(bk, q_ptr, kw_ptr, vw_ptr, hpm_ptr, m_out, l_out, l1_out, acc_out, W, wsp, nsp, sm_scale, GH: tl.constexpr,\n"
    "               GP: tl.constexpr, D: tl.constexpr, WB: tl.constexpr, WC: tl.constexpr, WDEV: tl.constexpr = False):")
rep("    for w0 in tl.range(0, W, WC, num_stages=1):\n"
    "        ow = w0 + oc\n"
    "        wm = ow < W\n",
    "    if WDEV:           # OPX CUDA graph: W from a device int32 (same loop, same masks)\n"
    "        Wn = tl.load(W)\n"
    "    else:\n"
    "        Wn = W\n"
    "    for w0 in tl.range(0, Wn, WC, num_stages=1):\n"
    "        ow = w0 + oc\n"
    "        wm = ow < Wn\n")
rep("            self.L, self.split_len, self.nsplit, self.scale, self.cpk, self.q, self.kw, self.vw, self.Hpm, self.W, self.nsb,\n"
    "            GH=GH, GP=GP, P=P, BLOCK_N=self.BLOCK_N, MAGIC=self.magic, LAM=bool(self.lam), WB=WBA, num_warps=self.warps,",
    "            self.L, self.split_len, self.nsplit, self.scale, self.cpk, self.q, self.kw, self.vw, self.Hpm,\n"
    "            self.W if getattr(self, \"Wdev\", None) is None else self.Wdev, self.nsb,\n"
    "            GH=GH, GP=GP, P=P, BLOCK_N=self.BLOCK_N, MAGIC=self.magic, LAM=bool(self.lam), WB=WBA,\n"
    "            WDEV=getattr(self, \"Wdev\", None) is not None, num_warps=self.warps,")
rep("        awin_kernel[(self.BK,)](self.q, self.kw, self.vw, self.Hpm, self.m, self.l, self.l1, self.acc, self.W, self.nsb,\n"
    "                                self.nsplit, self.scale, GH=GH, GP=GP, D=D, WB=WBA, num_warps=4)",
    "        awin_kernel[(self.BK,)](self.q, self.kw, self.vw, self.Hpm, self.m, self.l, self.l1, self.acc,\n"
    "                                self.W if getattr(self, \"Wdev\", None) is None else self.Wdev, self.nsb,\n"
    "                                self.nsplit, self.scale, GH=GH, GP=GP, D=D, WB=WBA,\n"
    "                                WDEV=getattr(self, \"Wdev\", None) is not None, num_warps=4)")
out = D / "kalign_opx.py"
out.write_text(s, encoding="utf-8", newline="\n")
print("ok", hashlib.md5(out.read_bytes()).hexdigest()[:8])
