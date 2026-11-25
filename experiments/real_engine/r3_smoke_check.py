# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
"""R3 smoke pre-commitment, applied mechanically (REGISTRATION_R3.md, "Smoke"): reads only status, GPU peak and timing of the
smoke record (cwe item 0, h54); never the pred or the score. Writes smoke/SMOKE_OK or smoke/SMOKE_FAIL with the reason.
Criteria: exactly one record, status ok; gpu_peak_gib < 46; projected per-task time 40 x (prefill_sec + ntok x median decode
ms / 1000) + 600 s (model load and calibration) < 21600 s; the smoke rc log has "r3_cwe rc=0" and a chunk test PASS.
Usage: python3 -I r3_smoke_check.py <engine7 dir>"""
import json
import os
import sys

d = os.path.join(sys.argv[1], "smoke")
os.makedirs(d, exist_ok=True)
why, rec = [], None
try:
    recs = [json.loads(l) for l in open(os.path.join(d, "out", "r3_cwe.jsonl")) if l.strip()]
    items = [r for r in recs if not r.get("meta")]
    if len(items) != 1:
        why.append(f"{len(items)} smoke records")
    else:
        rec = items[0]
except Exception as e:  # noqa: BLE001  (any failure to read is a failed smoke, written below)
    why.append(f"smoke output unreadable: {e}")
rc = ""
for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
    if f.startswith("rc_") and f.endswith(".log"):
        rc += open(os.path.join(d, f)).read()
if " r3_cwe rc=0 " not in rc + " " or "CHUNK_TEST PASS" not in rc:
    why.append("smoke rc log lacks r3_cwe rc=0 or the chunk test PASS")
proj = None
if rec is not None:
    if rec.get("status") != "ok":
        why.append(f"status {rec.get('status')}")
    peak = rec.get("gpu_peak_gib")
    if not isinstance(peak, (int, float)) or not peak < 46:
        why.append(f"gpu_peak_gib {peak}")
    try:
        per = float(rec["prefill_sec"]) + int(rec["ntok"]) * float(rec.get("decode_ms_median") or 0.0) / 1000.0
        proj = 40 * per + 600
        if not proj < 21600:
            why.append(f"projected per-task time {proj:.0f} s")
    except Exception as e:  # noqa: BLE001
        why.append(f"timing fields missing: {e}")
out = os.path.join(d, "SMOKE_FAIL" if why else "SMOKE_OK")
with open(out, "w") as f:
    f.write(json.dumps(dict(ok=not why, why=why, projected_task_sec=proj,
                            gpu_peak_gib=rec.get("gpu_peak_gib") if rec else None,
                            prefill_sec=rec.get("prefill_sec") if rec else None,
                            decode_ms_median=rec.get("decode_ms_median") if rec else None)) + "\n")
print(os.path.basename(out), why)
sys.exit(1 if why else 0)
