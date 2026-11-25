#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# NAB2 chain (REG_NAB2.md) on STAR: one GPU per call, models in order. Per model: the Q0-a
# control replay (NAB1's two arms and the patched variant with every ablation off, items 0-7),
# its gate (nab2_analyze.py --gate: provenance, code record, Q0-a, Q0-e), then NAB1's two arms and the four ablation arms together on all 64 NAB1
# traces (one run, so every ratio compares arms of the same run and card), then the same six arms
# in a seeded random order per item on eight spread items in a new process (Q0-d). Traces: NAB1's.
# usage (in the nab2 directory): setsid nohup bash nab2_chain.sh <gpu> <model> [<model> ...] \
#   > logs/launch_gpu<gpu>.out 2>&1 < /dev/null &
# The Q0-b marker out/selftest_pass.txt is written only by nab2_selftest.py on a pass.
# models: qwen3_0p6b qwen3_1p7b qwen3_4b. Stop: kill -- -<PGID> (the EXIT trap logs DONE).
set -u
set -o pipefail
W=$HOME/work/nab/nab2; T=$HOME/work/nab/nab1/out; ROOT=$HOME/work/nab
PY=$HOME/work/venv/bin/python; M=$HOME/work/models
GPU=${1:?gpu}; shift
[ $# -ge 1 ] || { echo "no models given"; exit 2; }
CTL=kivi128,kivi128k64_x2s128_j50_h54,t_variant_off   # Q0-e: patched variant, every ablation off
ARMS=kivi128,kivi128k64_x2s128_j50_h54,kivi128k64_x2r128_j50_h54,kivi128k64_x2s128_h54,kivi128k64_x2s128_j50_h5n,kivi4b128
REV_ITEMS=0,9,18,27,36,45,54,63   # Q0-d: eight items spread over the traces, arms in a seeded order
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # as NAB1 (4B fp32 at 8800: ~20 GiB)
LOG=$W/logs/nab2_chain_gpu$GPU.log
mkdir -p "$W/out" "$W/logs" "$HOME/work/nab/nab1/logs"
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
exec 9>>"$HOME/work/nab/nab1/logs/gpu$GPU.lock"
flock -n 9 || { log "gpu$GPU lock held, exit"; exit 3; }
fail=1
trap 'log "DONE nab2 chain gpu$GPU fail=$fail exit=$?"' EXIT
trap 'fail=1; exit 143' TERM INT HUP
cd "$W" || exit 1
# Q0-b gate (REG section 3): the self-test passed on exactly this code and this stack
want=$(md5sum nab2_q.py nab2_lab.py nab2_selftest.py nab2_chain.sh nab2_analyze.py 2>> "$LOG" \
  | awk '{printf " %s", substr($1,1,8); n++} END {if (n != 5) exit 1}') \
  || { log "md5 of the nab2 code failed, exit"; exit 4; }
want="PASS$want"
have=$(head -n 1 out/selftest_pass.txt 2>/dev/null)
if [ "$have" != "$want" ]; then
  log "Q0-b marker missing or stale (have '$have', want '$want')"; exit 4
fi
if ! "$PY" -I - "$ROOT" out/selftest_pass.txt >> "$LOG" 2>&1 <<'EOF'
import hashlib
import json
import sys
from pathlib import Path

root, mk = Path(sys.argv[1]), Path(sys.argv[2])
code = json.loads(mk.read_text(encoding="utf-8").splitlines()[1])
rel = {k: v for k, v in code.items() if "/" in k}
need = ["nab2/nab2_q.py", "nab2/nab2_lab.py"]
need += [s for s in ("/sel14_q.py", "/pkv5_q.py", "/sel30_q.py")
         if sum(k.endswith(s) for k in rel) != 1]
bad = [k for k, v in rel.items()
       if not (root / k).is_file() or hashlib.md5((root / k).read_bytes()).hexdigest() != v]
miss = [k for k in need if k not in rel]
print(f"Q0-b code record: {len(rel)} stack files checked, missing {miss}, mismatched {bad}")
sys.exit(1 if bad or miss else 0)
EOF
then
  log "Q0-b code record does not match the files on disk, exit"; exit 4
fi
log "START gpu$GPU models=[$*] $want traces $(md5sum "$T"/traces_*.jsonl 2>> "$LOG" \
  | awk '{n=split($2,p,"/"); printf "%s:%s ", p[n], substr($1,1,8)}')"
fail=0
for tag in "$@"; do
  case $tag in
    qwen3_0p6b) md=$M/Qwen3-0.6B;;
    qwen3_1p7b) md=$M/Qwen3-1.7B;;
    qwen3_4b)   md=$M/Qwen3-4B;;
    *) log "unknown model $tag"; fail=1; continue;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -I nab2_lab.py "$md" "$T/traces_$tag.jsonl" \
    out/ctl_$tag.json --stack "$ROOT" --runs "$CTL" --items 0,1,2,3,4,5,6,7 --ctl \
    --max_len 8800 --chunk 256 > logs/ctl_$tag.log 2>&1
  rc=$?; log "NAB2 ctl $tag rc=$rc"; [ $rc -ne 0 ] && { fail=1; continue; }
  "$PY" -I nab2_analyze.py --gate $tag "$T" out > logs/gate_$tag.log 2>&1
  rc=$?; log "NAB2 gate $tag rc=$rc $(tail -n 1 logs/gate_$tag.log)"
  [ $rc -ne 0 ] && { fail=1; continue; }
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -I nab2_lab.py "$md" "$T/traces_$tag.jsonl" \
    out/lab2_$tag.json --stack "$ROOT" --runs "$ARMS" --max_len 8800 --chunk 256 \
    > logs/lab2_$tag.log 2>&1
  rc=$?; log "NAB2 lab2 $tag rc=$rc"; [ $rc -ne 0 ] && { fail=1; continue; }
  # Q0-d (order invariance): the six arms again, in a seeded random order per item, on eight items
  # spread over the traces, in a new process; nab2_analyze.py requires every arm to equal lab2
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -I nab2_lab.py "$md" "$T/traces_$tag.jsonl" \
    out/rev_$tag.json --stack "$ROOT" --runs "$ARMS" --items "$REV_ITEMS" --perm_seed 7 \
    --max_len 8800 --chunk 256 > logs/rev_$tag.log 2>&1
  rc=$?; log "NAB2 rev $tag rc=$rc"; [ $rc -ne 0 ] && fail=1
done
exit $fail
