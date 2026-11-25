# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Reduce a t4_lab output to what t4_analyze.py reads, so outputs from two hosts can be merged locally.

usage: python -I t4_reduce.py <lab_json> <out_json>
Keeps every top-level field except the per-arm layer records, which are replaced by one pseudo-layer whose
record row is the column sum over all layers and rows (t4_analyze only uses column sums of these rows, so
bytes_per_record and the mechanism shares are unchanged). Item fields kept: i, L, think_end, ok16,
truncated_by_max_len, bucket, decode, answer, skipped. Records the input md5.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

KEEP_ITEM = ("i", "L", "think_end", "ok16", "truncated_by_max_len", "bucket", "decode", "answer", "skipped")


def reduce_item(it: dict) -> dict:
    out = {k: it[k] for k in KEEP_ITEM if k in it}
    if "layers" in it:
        lay = {}
        for arm, layers in it["layers"].items():
            rows = [row for v in layers.values() for row in v["rec"]]
            if not rows:
                lay[arm] = {}
                continue
            ncol = len(rows[0])
            if any(len(r) != ncol for r in rows):
                raise SystemExit(f"item {it['i']} arm {arm}: ragged record rows")
            lay[arm] = {"all": {"rec": [[sum(r[j] for r in rows) for j in range(ncol)]]}}
        out["layers"] = lay
    return out


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    raw = src.read_bytes()
    lab = json.loads(raw)
    red = {k: v for k, v in lab.items() if k != "items"}
    red["items"] = [reduce_item(it) for it in lab["items"]]
    red["reduced_from"] = {"path": str(src.resolve()), "md5": hashlib.md5(raw).hexdigest()}
    Path(sys.argv[2]).write_text(json.dumps(red), encoding="utf-8", newline="\n")
    print(f"{src.name}: {len(red['items'])} items, runs {lab['runs']}, md5 {red['reduced_from']['md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
