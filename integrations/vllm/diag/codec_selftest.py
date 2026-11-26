# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""CODEC_SPEC v1 section 6 self-test of tierkv_vllm.codec against the deployed research code.

1. codes / step / mn bit-identical to the reference pkv5_q.aq(r @ cq_lab3.hadamard(d), bits) on
   >= 1e5 random float32 rows (mixed scales) plus real residual rows of one model (K and V of
   Llama-3.2-1B on the calibration text minus the research KIVI-128 base, sinks and window
   excluded); reference codes recovered as round((y_hat - mn) / step); step == 0 rows excluded and
   counted. 2. decode within 1e-6 relative (per row, max abs error / row max abs) of
   sel14_q._res(r, bits). 3. row sizes. 4. pack / unpack round trip, b = 2..8.
Usage: python codec_selftest.py --research ROOT --codec-dir DIR --out F.json [--model DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", required=True)
    ap.add_argument("--codec-dir", required=True, help="directory holding codec.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--n-random", type=int, default=100_000)
    a = ap.parse_args()
    root = Path(a.research)
    for sub in ("code", "rootcause"):
        sys.path.insert(0, str(root / sub))
    sys.path.insert(0, a.codec_dir)
    import codec  # the file under test
    import cq_lab3 as CQ
    import pkv5_q as P5
    import sel14_q as S14

    torch.manual_seed(0)
    d = 128
    rows: dict[str, torch.Tensor] = {}
    scales = torch.tensor([1e-3, 1e-1, 1.0, 30.0])
    r = torch.randn(a.n_random, d) * scales[torch.randint(0, 4, (a.n_random,))][:, None]
    r[::997] = 0.0  # a few all-zero residuals (step == 0 rows)
    rows["random"] = r
    if a.model:
        rows["real"] = real_rows(a.model, root, CQ)
    res: dict = {"codec_file": str(Path(codec.__file__)), "cells": [], "fail": []}
    for name, r in rows.items():
        Hr = CQ.hadamard(r.shape[-1], "cpu")  # real rows carry the model's head_dim
        for bits in (3, 4, 5, 8):
            enc = codec.encode(r, bits)
            y = r @ Hr
            y_hat = P5.aq(y, bits)  # reference values
            mn_ref = P5.f16(y.amin(-1))
            mx_ref = P5.f16(y.amax(-1))
            st_ref = P5.f16((mx_ref - mn_ref) / (2**bits - 1))
            ok_rows = st_ref > 0
            codes_ref = torch.round(
                (y_hat - mn_ref[:, None]) / torch.where(ok_rows, st_ref, 1)[:, None]
            )
            same_codes = bool((enc.codes[ok_rows].float() == codes_ref[ok_rows]).all())
            same_step = bool(torch.equal(enc.step, st_ref)) and bool(torch.equal(enc.mn, mn_ref))
            dec = codec.decode(enc.codes, enc.step, enc.mn)
            ref = S14._res(r, bits)
            denom = ref.abs().amax(-1).clamp_min(1e-30)
            rel = ((dec - ref).abs().amax(-1) / denom)[ok_rows]
            cell = {
                "rows": name,
                "d": int(r.shape[-1]),
                "n": int(r.shape[0]),
                "bits": bits,
                "codes_bit_identical": same_codes,
                "step_mn_bit_identical": same_step,
                "step_zero_rows": int((~ok_rows).sum()),
                "decode_max_rel": float(rel.max()) if rel.numel() else 0.0,
            }
            res["cells"].append(cell)
            print("CODEC", json.dumps(cell), flush=True)
            if not (same_codes and same_step and cell["decode_max_rel"] <= 1e-6):
                res["fail"].append(cell)
    sizes = {t: (codec.row_bytes(d, k), codec.row_bytes(d, v)) for t, (k, v) in codec.TIERS.items()}
    res["row_sizes"] = sizes
    if sizes != {"h54": (84, 68), "h33": (52, 52), "h88": (132, 132)}:
        res["fail"].append({"row_sizes": sizes})
    for bits in range(2, 9):
        e = codec.encode(rows["random"][:5000], bits)
        b = codec.unpack(codec.pack(e, bits), d, bits)
        if not (
            torch.equal(b.codes, e.codes)
            and torch.equal(b.step, e.step)
            and torch.equal(b.mn, e.mn)
        ):
            res["fail"].append({"round_trip_bits": bits})
    res["verdict"] = "PASS" if not res["fail"] else "FAIL"
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("CODEC SELFTEST", res["verdict"], flush=True)
    return 0 if res["verdict"] == "PASS" else 1


def real_rows(model_dir: str, root: Path, CQ) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """K and V residuals vs the research KIVI-128 base on the calibration text (CPU, float32)."""
    for sub in ("rootcause/acs", "rootcause/spd", "rootcause/vfl"):
        sys.path.insert(0, str(root / sub))
    import sel29_q as Q
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    cap: dict = {}

    def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):  # type: ignore[no-untyped-def]  # noqa: E501
        li = int(module.layer_idx)
        q, k, v = query[0].float(), key[0].float(), value[0].float()
        cap[li] = (k.clone(), v.clone(), q.clone())
        s = scaling if scaling is not None else 1 / math.sqrt(q.shape[-1])
        return Q.attend(q, k, v, li, None, s, kw).unsqueeze(0).transpose(1, 2).to(query.dtype), None

    AttentionInterface.register("codec_cap", attn)
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="codec_cap"
    )
    re_ = m.model.rotary_emb
    Q.RC.R["inv_freq"], Q.RC.R["asc"] = (
        re_.inv_freq.float(),
        float(getattr(re_, "attention_scaling", 1.0)),
    )
    with torch.no_grad():
        m(input_ids=CQ.text_ids(tok, CQ.CAL[0], CQ.CAL[1], CQ.CAL[2], "cpu"), use_cache=False)
    for li, t in cap.items():
        Q.calib(li, *t)
    Q.calib_all(m, cap, sorted(cap))
    out = []
    for li in (0, len(cap) // 2, len(cap) - 1):
        k, v, _ = cap[li]
        kq, vq, R, _ = Q.Q5.base("kivi128", k, v, li)
        L = k.shape[1] // R * R  # complete 128-token groups only (the window stays exact)
        out += [
            (k - kq)[:, 4:L].reshape(-1, k.shape[-1]),
            (v - vq)[:, 4:L].reshape(-1, v.shape[-1]),
        ]
    return torch.cat(out).float()


if __name__ == "__main__":
    raise SystemExit(main())
